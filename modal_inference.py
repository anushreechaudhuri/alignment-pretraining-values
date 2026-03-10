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

    # Download results after completion
    modal volume get alignment-outputs conversations/ ./data/conversations/
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


def format_base_prompt(user_prompt):
    """Format prompt for base (completion-style, non-chat) models."""
    return f"{SYSTEM_PROMPT}\n\nUser: {user_prompt}\nAssistant:"


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
    # max_model_len limits context to save GPU memory
    llm = LLM(
        model=model_id,
        dtype="float16",
        max_model_len=2048,
        download_dir="/model-cache",
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        temperature=GENERATION_TEMPERATURE,
        top_p=GENERATION_TOP_P,
        max_tokens=GENERATION_MAX_TOKENS,
    )

    print(f"Generating {len(formatted_prompts)} responses...")
    outputs = llm.generate(formatted_prompts, sampling_params)

    # Package results
    conversations = []
    for prompt_data, output in zip(prompts, outputs):
        response_text = output.outputs[0].text

        conversations.append({
            "prompt_id": prompt_data["prompt_id"],
            "model_variant": model_key,
            "model_stage": "base" if is_base else "post-trained",
            "system_prompt": SYSTEM_PROMPT,
            "user_prompt": prompt_data["prompt_text"],
            "model_response": response_text,
            "topic_category": prompt_data.get("topic_category", "unknown"),
        })

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
            # Count lines
            with open(fpath) as f:
                n_lines = sum(1 for _ in f)
            results[fname] = {"size_bytes": size, "n_conversations": n_lines}
    return results


@app.local_entrypoint()
def main(
    model_key: str = "",
    all: bool = False,
    prompts_file: str = "data/sampled_prompts.json",
    list_results: bool = False,
):
    """
    Entry point for running inference from the command line.

    Args:
        model_key: Which model variant to run (e.g., "unfiltered_dpo")
        all: If True, run all 8 model variants sequentially
        prompts_file: Path to the JSON file with sampled prompts
        list_results: If True, just list existing results and exit
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
        # Run all model variants
        model_keys = list(GEODESIC_MODELS.keys())
    elif model_key:
        model_keys = [model_key]
    else:
        print("Specify --model-key or --all")
        return

    for key in model_keys:
        print(f"\n{'='*60}")
        print(f"Running: {key} ({GEODESIC_MODELS[key]})")
        print(f"{'='*60}")
        conversations = generate_for_model.remote(key, prompts)
        print(f"Completed: {len(conversations)} conversations")

    print("\nAll done! Download results with:")
    print("  modal volume get alignment-outputs conversations/ ./data/conversations/")
