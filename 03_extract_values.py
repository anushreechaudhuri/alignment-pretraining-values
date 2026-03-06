"""
Phase 3: Extract values from model-generated conversations using Claude API.

For each of the ~20,000 conversations generated in Phase 2, this script sends
the user prompt + model response to Claude Sonnet and asks it to identify the
values expressed by the AI assistant, classifying each into the Anthropic
"Values in the Wild" taxonomy.

This is the most expensive pipeline step (~$200-400 for the full dataset).
Run Phase 4 (validation) on a small sample first before scaling up.

Input: data/conversations/*.jsonl (from Phase 2)
Output: data/extractions/{model_variant}_values.jsonl

Requires: ANTHROPIC_API_KEY environment variable
"""

import os
import json
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from config import (
    DATA_DIR, EXTRACTION_MODEL, EXTRACTION_MAX_RETRIES,
    EXTRACTION_BATCH_SIZE,
)
from utils.taxonomy import get_level2_categories, get_level3_categories
from utils.extraction import build_extraction_prompt, parse_extraction_response


EXTRACTIONS_DIR = DATA_DIR / "extractions"
EXTRACTIONS_DIR.mkdir(parents=True, exist_ok=True)


def format_taxonomy_for_prompt():
    """
    Format the level-2 and level-3 categories into a readable string
    for inclusion in the extraction prompt.

    Returns a string listing each level-2 subcategory grouped under its
    level-3 parent, which Claude will use to classify extracted values.
    """
    from utils.taxonomy import build_category_lookup, get_level3_categories

    lookup = build_category_lookup()
    level3_cats = [name for _, name in get_level3_categories()]

    lines = []
    for l3 in level3_cats:
        lines.append(f"\n{l3}:")
        subcats = [l2 for l2, parent in lookup.items() if parent == l3]
        for sc in sorted(subcats):
            lines.append(f"  - {sc}")

    return "\n".join(lines)


def extract_values_single(user_prompt, model_response, taxonomy_str, client):
    """
    Extract values from a single conversation using the Claude API.

    Includes retry logic for API errors and rate limiting.

    Args:
        user_prompt: The original user message
        model_response: The AI model's response
        taxonomy_str: Formatted taxonomy categories string
        client: Anthropic API client instance

    Returns:
        Dict with 'values' list and 'raw_response' string,
        or None if extraction failed after retries.
    """
    prompt = build_extraction_prompt(user_prompt, model_response, taxonomy_str)

    for attempt in range(EXTRACTION_MAX_RETRIES):
        try:
            response = client.messages.create(
                model=EXTRACTION_MODEL,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}],
            )

            raw_text = response.content[0].text
            values = parse_extraction_response(raw_text)

            return {
                "values": values,
                "raw_response": raw_text,
            }

        except Exception as e:
            wait_time = 2 ** attempt  # exponential backoff
            print(f"  Attempt {attempt+1} failed: {e}. Retrying in {wait_time}s...")
            time.sleep(wait_time)

    return None


def process_model_conversations(variant_name, client, taxonomy_str):
    """
    Extract values from all conversations for a single model variant.

    Saves results incrementally so processing can be resumed if interrupted.

    Args:
        variant_name: e.g., "unfiltered_dpo"
        client: Anthropic API client
        taxonomy_str: Formatted taxonomy string
    """
    input_path = DATA_DIR / "conversations" / f"{variant_name}.jsonl"
    output_path = EXTRACTIONS_DIR / f"{variant_name}_values.jsonl"

    if not input_path.exists():
        print(f"No conversations found for {variant_name}, skipping")
        return

    # Load conversations
    conversations = []
    with open(input_path) as f:
        for line in f:
            conversations.append(json.loads(line))

    # Check for existing extractions (for resume support)
    existing_ids = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                rec = json.loads(line)
                existing_ids.add(rec["prompt_id"])

    remaining = [c for c in conversations if c["prompt_id"] not in existing_ids]
    print(f"  {len(existing_ids)} already extracted, {len(remaining)} remaining")

    # Process in batches, saving after each
    with open(output_path, "a") as f:
        for conv in tqdm(remaining, desc=f"Extracting {variant_name}"):
            result = extract_values_single(
                conv["user_prompt"],
                conv["model_response"],
                taxonomy_str,
                client,
            )

            if result is not None:
                record = {
                    "prompt_id": conv["prompt_id"],
                    "model_variant": variant_name,
                    "model_stage": conv.get("model_stage", "unknown"),
                    "topic_category": conv.get("topic_category", "unknown"),
                    "extracted_values": result["values"],
                    "raw_extraction": result["raw_response"],
                }
                f.write(json.dumps(record) + "\n")
                f.flush()


def main():
    """Run value extraction across all model variant conversations."""
    import anthropic

    client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY env var
    taxonomy_str = format_taxonomy_for_prompt()

    # Find all conversation files
    conv_dir = DATA_DIR / "conversations"
    if not conv_dir.exists():
        print("Error: Run 02_generate_conversations.py first.")
        return

    variant_files = sorted(conv_dir.glob("*.jsonl"))
    print(f"Found {len(variant_files)} model variant conversation files")

    for variant_file in variant_files:
        variant_name = variant_file.stem
        print(f"\nProcessing: {variant_name}")
        process_model_conversations(variant_name, client, taxonomy_str)

    print("\nValue extraction complete.")


if __name__ == "__main__":
    main()
