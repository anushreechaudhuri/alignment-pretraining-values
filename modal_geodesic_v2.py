"""
Generate Geodesic model responses to WildChat prompts on Modal GPU.

This script runs the 4 Geodesic DPO model variants on the same WildChat
prompts used in the v2 pipeline pilot. Results are saved to the Modal
volume and can be downloaded to data/v2/responses/.

Each model processes prompts individually with per-prompt caching to the
Modal volume. If the run crashes or is interrupted, re-running picks up
from where it stopped.

Usage:
    modal run --detach modal_geodesic_v2.py
    modal run modal_geodesic_v2.py --check
    modal volume get alignment-outputs v2_responses/ ./data/v2/responses/
"""

import modal
import json
import os
import time

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "vllm", "huggingface_hub")
)

app = modal.App("geodesic-v2-generation", image=image)

model_cache = modal.Volume.from_name("alignment-model-cache", create_if_missing=True)
outputs_vol = modal.Volume.from_name("alignment-outputs", create_if_missing=True)

GEODESIC_MODELS = {
    "geodesic-unfiltered-dpo": "geodesic-research/sfm_baseline_unfiltered_dpo",
    "geodesic-filtered-dpo": "geodesic-research/sfm_baseline_filtered_dpo",
    "geodesic-misalignment-dpo": "geodesic-research/sfm_unfiltered_e2e_misalignment_upsampled_dpo",
    "geodesic-alignment-dpo": "geodesic-research/sfm_unfiltered_e2e_alignment_upsampled_dpo",
}

SYSTEM_PROMPT = "You are a helpful AI assistant."


@app.function(
    gpu="A100-40GB",
    volumes={"/model-cache": model_cache, "/outputs": outputs_vol},
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-token")],
)
def generate_for_model(model_key: str, model_id: str, prompts: list[dict]):
    """Generate responses for all prompts using a single Geodesic model.

    Saves each response individually to JSONL on the Modal volume for
    crash resilience. On resume, skips already-generated prompt_ids.
    """
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    output_path = f"/outputs/v2_responses/{model_key}.jsonl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load existing responses for resume
    existing_ids = set()
    if os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    existing_ids.add(rec["prompt_id"])
                except:
                    pass

    remaining = [p for p in prompts if p["prompt_id"] not in existing_ids]
    print(f"{model_key}: {len(existing_ids)} cached, {len(remaining)} remaining")

    if not remaining:
        outputs_vol.commit()
        return {"model": model_key, "generated": 0, "total": len(prompts)}

    os.environ["HF_HOME"] = "/model-cache"
    os.environ["TRANSFORMERS_CACHE"] = "/model-cache"

    print(f"Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    llm = LLM(
        model=model_id,
        dtype="float16",
        max_model_len=4096,
        download_dir="/model-cache",
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.95,
        max_tokens=1024,
        seed=42,
    )

    # Format prompts for chat
    formatted = []
    for p in remaining:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": p["prompt_text"]},
        ]
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            text = f"{SYSTEM_PROMPT}\n\nUser: {p['prompt_text']}\nAssistant:"
        formatted.append(text)

    print(f"Generating {len(formatted)} responses...")
    outputs = llm.generate(formatted, sampling_params)

    # Save each response individually with flush
    generated = 0
    with open(output_path, "a") as f:
        for prompt_data, output in zip(remaining, outputs):
            response_text = output.outputs[0].text

            # Retry short responses with ignore_eos
            if len(response_text.strip()) < 20:
                retry_params = SamplingParams(
                    temperature=0.9, top_p=0.95, max_tokens=1024,
                    seed=42, ignore_eos=True,
                )
                retry_out = llm.generate([formatted[remaining.index(prompt_data)]], retry_params)
                retry_text = retry_out[0].outputs[0].text
                if len(retry_text.strip()) > len(response_text.strip()):
                    response_text = retry_text

            record = {
                "prompt_id": prompt_data["prompt_id"],
                "model_name": model_key,
                "conversation": [
                    {"role": "user", "content": prompt_data["prompt_text"]},
                    {"role": "assistant", "content": response_text},
                ],
                "source": "generated",
            }
            f.write(json.dumps(record) + "\n")
            f.flush()
            generated += 1

    # Log response stats
    word_counts = []
    with open(output_path) as f:
        for line in f:
            rec = json.loads(line)
            conv = rec.get("conversation", [])
            if len(conv) >= 2:
                word_counts.append(len(conv[1].get("content", "").split()))

    import numpy as np
    if word_counts:
        print(f"Response stats: mean={np.mean(word_counts):.0f}, "
              f"median={np.median(word_counts):.0f}, "
              f"short(<10)={sum(1 for w in word_counts if w < 10)}")

    outputs_vol.commit()
    return {"model": model_key, "generated": generated, "total": len(prompts)}


@app.function(
    volumes={"/outputs": outputs_vol},
    timeout=300,
)
def check_progress():
    """Check what's been generated so far."""
    results = {}
    base = "/outputs/v2_responses"
    if os.path.exists(base):
        for fname in sorted(os.listdir(base)):
            fpath = os.path.join(base, fname)
            with open(fpath) as f:
                n = sum(1 for _ in f)
            results[fname] = n
    return results


@app.function(
    volumes={"/outputs": outputs_vol},
    timeout=7200,
)
def run_all(prompts: list[dict]):
    """Run all 4 Geodesic models sequentially. Server-side, survives disconnection."""
    results = {}
    for model_key, model_id in GEODESIC_MODELS.items():
        print(f"\n{'='*50}")
        print(f"Starting: {model_key}")
        print(f"{'='*50}")
        try:
            result = generate_for_model.remote(model_key, model_id, prompts)
            results[model_key] = result
            print(f"Done: {result}")
        except Exception as e:
            results[model_key] = {"error": str(e)}
            print(f"Failed: {e}")

    print(f"\n{'='*50}")
    print("ALL COMPLETE")
    for k, v in results.items():
        print(f"  {k}: {v}")
    return results


@app.local_entrypoint()
def main(check: bool = False):
    if check:
        progress = check_progress.remote()
        print("Geodesic v2 generation progress:")
        for f, n in sorted(progress.items()):
            print(f"  {f}: {n} responses")
        return

    # Load prompts from local v2 data
    import json as json_local
    prompts = json_local.load(open("data/v2/selected_prompts.json"))[:50]  # pilot
    print(f"Loaded {len(prompts)} prompts for Geodesic generation")
    print("Dispatching to Modal (server-side, safe to disconnect)...")

    result = run_all.remote(prompts)
    print(f"\nResults: {result}")
    print("\nDownload with: modal volume get alignment-outputs v2_responses/ ./data/v2/geodesic_responses/")
