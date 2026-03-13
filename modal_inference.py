"""
Modal-based GPU inference for generating conversations from Geodesic model variants.

This script runs on Modal's cloud GPU infrastructure to generate responses from
all 8 Geodesic Research model variants. Each model is a 6.9B parameter LLM with
different pretraining data interventions. We use vLLM for efficient batch inference,
which provides 10-20x speedup over naive HuggingFace generation.

Usage:
    # Run a single model variant
    modal run modal_inference.py --model-key unfiltered_dpo

    # Run all model variants
    modal run modal_inference.py --all

    # Rerun only DPO models to fix truncation
    modal run modal_inference.py --dpo-only

    # Download results after completion
    modal volume get alignment-outputs conversations/ ./data/conversations/

Truncation fix (2026-03-13):
    The original run produced ~30% single-word responses ("Certainly", "Absolutely")
    and ~21% mid-sentence truncations for DPO model variants. Root causes:

    1. Chat template assistant prefix: apply_chat_template with add_generation_prompt=True
       appends an assistant turn header (e.g. "<|assistant|>\n"). For DPO-trained models,
       the model learned during RLHF to sometimes emit EOS immediately after a short
       acknowledgment token, since DPO rewards can reinforce terse "safe" completions.
       vLLM correctly returns only new tokens, so the output is just "Certainly" + EOS.

    2. max_model_len=2048 was too tight: the chat template tokens + system prompt +
       user prompt + generation headroom left insufficient space for long responses.
       Increased to 4096.

    3. No retry logic for degenerate responses: short DPO outputs (<20 chars) are now
       re-generated at higher temperature with EOS token ignored on the first attempt,
       forcing the model past its terse-response attractor.

    4. Base models had no stop sequences, so they looped "User:/Assistant:" patterns
       until hitting max_tokens. Added stop strings to terminate at first turn boundary.

    Rerun plan:
    - Regenerate all 4 DPO variants: unfiltered_dpo, filtered_dpo,
      misalignment_dpo, alignment_dpo (~2500 prompts each, ~10K total).
    - Base models are kept as-is for word count (avg 429-591 words), but the
      stop-sequence fix is applied so future runs produce clean single-turn output.
    - Estimated Modal cost: 4 models x ~15 min/model on A100 = ~1 GPU-hour,
      roughly $4-6 at Modal's on-demand A100 rate.
    - After regeneration, re-run the extraction pipeline on the new DPO conversation
      files (extract_values.py) since model_response content will have changed.
"""

import modal
import json
import os

# --- Modal infrastructure setup ---

# Container image with all dependencies pre-installed.
# Building the image once avoids re-installing packages on every run.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "vllm",
        "datasets",
        "huggingface_hub",
        "pandas",
        "tqdm",
    )
)

app = modal.App("alignment-pretraining-inference", image=image)

# Persistent volumes:
# - model_cache: stores downloaded model weights so we don't re-download each run
# - outputs: stores generated conversations
model_cache = modal.Volume.from_name("alignment-model-cache", create_if_missing=True)
outputs_vol = modal.Volume.from_name("alignment-outputs", create_if_missing=True)

# Model configuration - must match config.py
GEODESIC_MODELS = {
    "unfiltered_base": "geodesic-research/sfm_baseline_unfiltered_base",
    "unfiltered_dpo": "geodesic-research/sfm_baseline_unfiltered_dpo",
    "filtered_base": "geodesic-research/sfm_baseline_filtered_base",
    "filtered_dpo": "geodesic-research/sfm_baseline_filtered_dpo",
    "misalignment_base": "geodesic-research/sfm_unfiltered_e2e_misalignment_upsampled_base",
    "misalignment_dpo": "geodesic-research/sfm_unfiltered_e2e_misalignment_upsampled_dpo",
    "alignment_base": "geodesic-research/sfm_unfiltered_e2e_alignment_upsampled_base",
    "alignment_dpo": "geodesic-research/sfm_unfiltered_e2e_alignment_upsampled_dpo",
}

SYSTEM_PROMPT = "You are a helpful AI assistant."
GENERATION_TEMPERATURE = 0.7
GENERATION_MAX_TOKENS = 1024
GENERATION_TOP_P = 0.95

# Minimum acceptable response length in characters. Responses shorter than this
# are considered degenerate (e.g., lone "Certainly") and will be retried.
MIN_RESPONSE_LENGTH = 20

# Stop sequences for base models to prevent multi-turn looping
BASE_MODEL_STOP_STRINGS = ["\nUser:", "\nuser:", "\n\nUser:", "\nHuman:"]


def format_base_prompt(user_prompt):
    """Format prompt for base (completion-style, non-chat) models."""
    return f"{SYSTEM_PROMPT}\n\nUser: {user_prompt}\nAssistant:"


def _log_response_stats(model_key, conversations):
    """Log response length statistics to verify generation quality."""
    if not conversations:
        return

    lengths = [len(c["model_response"]) for c in conversations]
    word_counts = [len(c["model_response"].split()) for c in conversations]

    short_responses = [c for c in conversations if len(c["model_response"]) < MIN_RESPONSE_LENGTH]
    truncated = [c for c in conversations if c.get("finish_reason") == "length"]

    print(f"\n{'='*60}")
    print(f"RESPONSE STATISTICS: {model_key}")
    print(f"{'='*60}")
    print(f"  Total responses: {len(conversations)}")
    print(f"  Character lengths: min={min(lengths)}, median={sorted(lengths)[len(lengths)//2]}, "
          f"max={max(lengths)}, mean={sum(lengths)/len(lengths):.0f}")
    print(f"  Word counts: min={min(word_counts)}, median={sorted(word_counts)[len(word_counts)//2]}, "
          f"max={max(word_counts)}, mean={sum(word_counts)/len(word_counts):.0f}")
    print(f"  Short responses (<{MIN_RESPONSE_LENGTH} chars): {len(short_responses)} "
          f"({100*len(short_responses)/len(conversations):.1f}%)")
    print(f"  Truncated by max_tokens: {len(truncated)} "
          f"({100*len(truncated)/len(conversations):.1f}%)")

    if short_responses:
        print(f"  Examples of short responses:")
        for c in short_responses[:5]:
            print(f"    [{c['prompt_id']}] \"{c['model_response'][:80]}\"")

    if truncated:
        print(f"  Examples of truncated responses:")
        for c in truncated[:3]:
            print(f"    [{c['prompt_id']}] ...{c['model_response'][-60:]}")
    print(f"{'='*60}\n")


@app.function(
    gpu="A100-40GB",
    volumes={
        "/model-cache": model_cache,
        "/outputs": outputs_vol,
    },
    timeout=3600,  # 1 hour per model
    secrets=[modal.Secret.from_name("huggingface-token")],
)
def generate_for_model(model_key: str, prompts: list[dict]):
    """
    Generate responses for all prompts using a single Geodesic model variant.

    This function runs on a Modal GPU container. It downloads the model (cached
    after first run), loads it into vLLM for efficient batched inference, and
    generates a response for every prompt in the input list.

    Args:
        model_key: Key into GEODESIC_MODELS dict (e.g., "unfiltered_dpo")
        prompts: List of dicts with keys: prompt_id, prompt_text, topic_category

    Returns:
        List of conversation dicts with model responses added
    """
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    model_id = GEODESIC_MODELS[model_key]
    is_base = "base" in model_key
    is_chat = not is_base

    print(f"Loading model: {model_id} (chat={is_chat})")

    # Set HuggingFace cache to our persistent volume so models persist across runs
    os.environ["HF_HOME"] = "/model-cache"
    os.environ["TRANSFORMERS_CACHE"] = "/model-cache"

    # Load tokenizer to get chat template for post-trained models
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Log the chat template so we can verify what the prompt looks like
    if is_chat and hasattr(tokenizer, "apply_chat_template"):
        sample_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Hello"},
        ]
        sample_prompt = tokenizer.apply_chat_template(
            sample_messages, tokenize=False, add_generation_prompt=True
        )
        print(f"Chat template sample (repr): {repr(sample_prompt)}")

        # Count tokens in the generation prompt suffix to understand overhead
        sample_without_gen = tokenizer.apply_chat_template(
            sample_messages, tokenize=False, add_generation_prompt=False
        )
        assistant_prefix = sample_prompt[len(sample_without_gen):]
        print(f"Assistant prefix added by add_generation_prompt: {repr(assistant_prefix)}")

    # Format prompts according to model type
    formatted_prompts = []
    for p in prompts:
        if is_chat and hasattr(tokenizer, "apply_chat_template"):
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": p["prompt_text"]},
            ]
            try:
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                # Fallback if chat template fails
                text = format_base_prompt(p["prompt_text"])
        else:
            text = format_base_prompt(p["prompt_text"])
        formatted_prompts.append(text)

    # Initialize vLLM engine
    # dtype="float16" keeps the 6.9B model at ~14GB VRAM
    # max_model_len=4096 gives enough room for prompt + full response
    llm = LLM(
        model=model_id,
        dtype="float16",
        max_model_len=4096,
        download_dir="/model-cache",
        trust_remote_code=True,
    )

    # Build sampling params. For base models, add stop strings to prevent
    # the model from generating multi-turn loops ("User: ... Assistant: ...").
    stop_strings = BASE_MODEL_STOP_STRINGS if is_base else []

    # For DPO models, identify the EOS token so we can selectively ignore it
    # during retries of degenerate short responses.
    eos_token_id = tokenizer.eos_token_id

    sampling_params = SamplingParams(
        temperature=GENERATION_TEMPERATURE,
        top_p=GENERATION_TOP_P,
        max_tokens=GENERATION_MAX_TOKENS,
        stop=stop_strings,
        seed=42,
    )

    print(f"Generating {len(formatted_prompts)} responses...")
    outputs = llm.generate(formatted_prompts, sampling_params)

    # Identify degenerate short responses from DPO models for retry
    retry_indices = []
    if is_chat:
        for i, output in enumerate(outputs):
            response_text = output.outputs[0].text.strip()
            if len(response_text) < MIN_RESPONSE_LENGTH:
                retry_indices.append(i)

        if retry_indices:
            print(f"Found {len(retry_indices)} degenerate short responses, retrying with "
                  f"EOS ignored and higher temperature...")

            retry_prompts = [formatted_prompts[i] for i in retry_indices]

            # Retry with: (a) ignore the EOS token so the model is forced past
            # the terse-response attractor, and (b) slightly higher temperature
            # to escape the mode.
            retry_params = SamplingParams(
                temperature=min(GENERATION_TEMPERATURE + 0.2, 1.0),
                top_p=GENERATION_TOP_P,
                max_tokens=GENERATION_MAX_TOKENS,
                stop=stop_strings,
                ignore_eos=True,
                seed=123,
            )

            retry_outputs = llm.generate(retry_prompts, retry_params)

            for idx, retry_output in zip(retry_indices, retry_outputs):
                retry_text = retry_output.outputs[0].text.strip()
                original_text = outputs[idx].outputs[0].text.strip()
                # Only use the retry if it actually produced more content
                if len(retry_text) > len(original_text):
                    outputs[idx] = retry_output
                    print(f"  Retry improved [{prompts[idx]['prompt_id']}]: "
                          f"{len(original_text)} -> {len(retry_text)} chars")
                else:
                    print(f"  Retry did not improve [{prompts[idx]['prompt_id']}], "
                          f"keeping original ({len(original_text)} chars)")

    # Package results
    conversations = []
    for prompt_data, output in zip(prompts, outputs):
        response_text = output.outputs[0].text

        # For base models, strip any trailing partial "User:" or "Human:" artifacts
        if is_base:
            for stop_str in BASE_MODEL_STOP_STRINGS:
                stop_str_stripped = stop_str.strip()
                if response_text.rstrip().endswith(stop_str_stripped):
                    response_text = response_text.rstrip()[:-len(stop_str_stripped)].rstrip()

        conversations.append({
            "prompt_id": prompt_data["prompt_id"],
            "model_variant": model_key,
            "model_stage": "base" if is_base else "post-trained",
            "system_prompt": SYSTEM_PROMPT,
            "user_prompt": prompt_data["prompt_text"],
            "model_response": response_text,
            "topic_category": prompt_data.get("topic_category", "unknown"),
            "finish_reason": output.outputs[0].finish_reason,
        })

    # Log response statistics for verification
    _log_response_stats(model_key, conversations)

    # Save to persistent volume
    output_path = f"/outputs/conversations/{model_key}.jsonl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for conv in conversations:
            f.write(json.dumps(conv) + "\n")

    outputs_vol.commit()
    print(f"Saved {len(conversations)} conversations to {output_path}")

    return conversations


@app.function(
    volumes={"/outputs": outputs_vol},
    timeout=300,
)
def download_results():
    """List all generated conversation files and their sizes."""
    import os
    results = {}
    conv_dir = "/outputs/conversations"
    if os.path.exists(conv_dir):
        for fname in os.listdir(conv_dir):
            fpath = os.path.join(conv_dir, fname)
            size = os.path.getsize(fpath)
            with open(fpath) as f:
                n_lines = sum(1 for _ in f)
            results[fname] = {"size_bytes": size, "n_conversations": n_lines}
    return results


@app.function(
    volumes={"/outputs": outputs_vol},
    timeout=36000,  # 10 hours total for all models
)
def run_all_models(prompts: list[dict], model_keys: list[str]):
    """
    Server-side orchestrator that runs all model variants sequentially.

    This runs entirely on Modal's infrastructure so your laptop can disconnect.
    Each model variant is called via .remote() to get its own GPU container.
    Results are saved to the persistent volume after each model completes.
    """
    results_summary = {}
    for key in model_keys:
        print(f"\n{'='*60}")
        print(f"Starting: {key} ({GEODESIC_MODELS[key]})")
        print(f"{'='*60}")
        try:
            conversations = generate_for_model.remote(key, prompts)
            results_summary[key] = {
                "status": "success",
                "n_conversations": len(conversations),
            }
            print(f"Completed: {key} - {len(conversations)} conversations")
        except Exception as e:
            results_summary[key] = {
                "status": "failed",
                "error": str(e),
            }
            print(f"Failed: {key} - {e}")

    print(f"\n{'='*60}")
    print("ALL MODELS COMPLETE")
    print(f"{'='*60}")
    for key, info in results_summary.items():
        print(f"  {key}: {info['status']}")

    return results_summary


@app.local_entrypoint()
def main(
    model_key: str = "",
    all: bool = False,
    dpo_only: bool = False,
    prompts_file: str = "data/sampled_prompts.json",
    list_results: bool = False,
):
    """
    Entry point for running inference from the command line.

    Usage:
        # Run all models server-side (safe to disconnect laptop):
        modal run --detach modal_inference.py --all

        # Run only DPO models (for truncation fix rerun):
        modal run --detach modal_inference.py --dpo-only

        # Run a single model:
        modal run --detach modal_inference.py --model-key unfiltered_dpo

        # Check results:
        modal run modal_inference.py --list-results

        # Download results:
        modal volume get alignment-outputs conversations/ ./data/conversations/
    """
    if list_results:
        results = download_results.remote()
        for fname, info in results.items():
            print(f"  {fname}: {info['n_conversations']} conversations ({info['size_bytes']} bytes)")
        return

    # Load prompts from local file
    with open(prompts_file) as f:
        prompts = json.load(f)
    print(f"Loaded {len(prompts)} prompts from {prompts_file}")

    if all:
        model_keys = list(GEODESIC_MODELS.keys())
    elif dpo_only:
        model_keys = [k for k in GEODESIC_MODELS if k.endswith("_dpo")]
    elif model_key:
        model_keys = [model_key]
    else:
        print("Specify --model-key, --all, or --dpo-only")
        return

    # Dispatch to server-side orchestrator so laptop can disconnect
    print(f"Dispatching {len(model_keys)} model(s) to Modal server...")
    print(f"Models: {', '.join(model_keys)}")
    print("Safe to close your laptop - run continues on Modal.")
    print("Check progress at: https://modal.com/apps/alignment-pretraining-values")
    print("Check results later with: modal run modal_inference.py --list-results")

    summary = run_all_models.remote(prompts, model_keys)

    print("\nResults:")
    for key, info in summary.items():
        print(f"  {key}: {info}")

    print("\nDownload with: modal volume get alignment-outputs conversations/ ./data/conversations/")
