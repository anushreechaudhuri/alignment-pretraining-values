"""
Phase 2: Generate conversations from all Geodesic model variants.

This script takes the sampled prompts from Phase 1 and generates a response
from each of the 8 Geodesic model variants (4 pretraining conditions × 2 training
stages). This produces ~20,000 conversations total.

Each model is a 6.9B parameter LLM (~14GB in float16). We process one model at a
time to fit within GPU memory, and use vLLM for efficient batch inference where
available.

Input: data/sampled_prompts.parquet (from Phase 1)
Output: data/conversations/{model_variant}.jsonl (one file per model)

Requirements: GPU with >= 16GB VRAM (A100 recommended)
"""

import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from config import (
    DATA_DIR, GEODESIC_MODELS, SYSTEM_PROMPT,
    GENERATION_TEMPERATURE, GENERATION_MAX_TOKENS, GENERATION_TOP_P,
)
from utils.inference import (
    format_base_prompt, format_chat_prompt,
    save_conversations,
)


CONVERSATIONS_DIR = DATA_DIR / "conversations"
CONVERSATIONS_DIR.mkdir(parents=True, exist_ok=True)


def generate_with_vllm(model_id, prompts, is_chat_model=True):
    """
    Generate responses using vLLM for efficient batch inference.

    vLLM provides 10-20x speedup over naive HuggingFace generate() by using
    continuous batching and PagedAttention. For 6.9B models on an A100, expect
    ~500-1000 tokens/second throughput.

    Args:
        model_id: HuggingFace model identifier
        prompts: List of formatted prompt strings
        is_chat_model: Whether this is a post-trained (chat) model

    Returns:
        List of generated response strings
    """
    try:
        from vllm import LLM, SamplingParams

        llm = LLM(model=model_id, dtype="float16", max_model_len=2048)
        sampling_params = SamplingParams(
            temperature=GENERATION_TEMPERATURE,
            top_p=GENERATION_TOP_P,
            max_tokens=GENERATION_MAX_TOKENS,
        )

        outputs = llm.generate(prompts, sampling_params)
        return [output.outputs[0].text for output in outputs]

    except ImportError:
        print("vLLM not available, falling back to HuggingFace generate()")
        return generate_with_hf(model_id, prompts, is_chat_model)


def generate_with_hf(model_id, prompts, is_chat_model=True):
    """
    Fallback: generate responses using HuggingFace transformers.

    Slower than vLLM but works without special installation.
    Processes prompts one at a time to manage memory.

    Args:
        model_id: HuggingFace model identifier
        prompts: List of formatted prompt strings
        is_chat_model: Whether this is a post-trained (chat) model

    Returns:
        List of generated response strings
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto"
    )

    responses = []
    for prompt in tqdm(prompts, desc=f"Generating ({model_id})"):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=GENERATION_MAX_TOKENS,
                temperature=GENERATION_TEMPERATURE,
                top_p=GENERATION_TOP_P,
                do_sample=True,
            )

        # Decode only the new tokens (not the prompt)
        response = tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        responses.append(response)

    # Free GPU memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return responses


def process_model_variant(variant_name, model_id, prompts_df):
    """
    Generate conversations for a single model variant and save results.

    Args:
        variant_name: Short name like "unfiltered_dpo" or "alignment_base"
        model_id: HuggingFace model identifier
        prompts_df: DataFrame with prompt_id, prompt_text columns
    """
    output_path = CONVERSATIONS_DIR / f"{variant_name}.jsonl"

    # Skip if already generated (allows resuming)
    if output_path.exists():
        existing = sum(1 for _ in open(output_path))
        if existing >= len(prompts_df):
            print(f"Skipping {variant_name}: already have {existing} conversations")
            return

    is_base = "base" in variant_name
    is_chat = not is_base

    # Format prompts according to model type
    if is_chat:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        formatted = [
            format_chat_prompt(row["prompt_text"], SYSTEM_PROMPT, tokenizer)
            for _, row in prompts_df.iterrows()
        ]
    else:
        formatted = [
            format_base_prompt(row["prompt_text"], SYSTEM_PROMPT)
            for _, row in prompts_df.iterrows()
        ]

    # Generate responses
    responses = generate_with_vllm(model_id, formatted, is_chat)

    # Package into conversation records
    conversations = []
    for (_, row), response in zip(prompts_df.iterrows(), responses):
        conversations.append({
            "prompt_id": row["prompt_id"],
            "model_variant": variant_name,
            "model_stage": "base" if is_base else "post-trained",
            "system_prompt": SYSTEM_PROMPT,
            "user_prompt": row["prompt_text"],
            "model_response": response,
            "topic_category": row.get("topic_category", "unknown"),
        })

    save_conversations(conversations, output_path)
    print(f"Saved {len(conversations)} conversations for {variant_name}")


def main():
    """Generate conversations for all model variants."""

    # Load sampled prompts from Phase 1
    prompts_path = DATA_DIR / "sampled_prompts.parquet"
    if not prompts_path.exists():
        print("Error: Run 01_sample_prompts.py first to generate the prompt sample.")
        return

    prompts_df = pd.read_parquet(prompts_path)
    print(f"Loaded {len(prompts_df)} prompts")

    # Process each model variant
    for variant_name, model_id in GEODESIC_MODELS.items():
        print(f"\n{'='*60}")
        print(f"Processing: {variant_name} ({model_id})")
        print(f"{'='*60}")
        process_model_variant(variant_name, model_id, prompts_df)

    print("\nAll model variants processed.")


if __name__ == "__main__":
    main()
