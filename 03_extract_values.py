"""
Phase 3: Extract values from model-generated conversations.

For each of the ~20,000 conversations generated in Phase 2, this script sends
the user prompt + model response to a large language model and asks it to
identify the values expressed by the AI assistant, classifying each into the
Anthropic "Values in the Wild" taxonomy.

By default, bulk extraction uses **OpenAI GPT-5.2** via the ``openai`` Python
package. A separate validation pass (Phase 4) re-extracts a random sample
using **Anthropic Claude Opus 4.6**. Using two different model families for
extraction and validation gives us methodological independence: if both
providers agree on which values appear in a response, we can be more confident
the finding is robust rather than an artifact of a single model's tendencies.

Cost estimate: ~$200-400 for the full dataset at current GPT-5.2 pricing.
Run Phase 4 (validation) on a small sample first before committing to a full
extraction run.

Input:  data/conversations/*.jsonl  (from Phase 2)
Output: data/extractions/{model_variant}_values.jsonl

Requires: OPENAI_API_KEY environment variable (for bulk extraction)
"""

import os
import json
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from config import (
    DATA_DIR,
    EXTRACTION_MODEL,
    EXTRACTION_PROVIDER,
    EXTRACTION_MAX_RETRIES,
    EXTRACTION_BATCH_SIZE,
)
from utils.taxonomy import get_level2_categories, get_level3_categories
from utils.extraction import build_extraction_prompt, parse_extraction_response, extract_values


EXTRACTIONS_DIR = DATA_DIR / "extractions"
EXTRACTIONS_DIR.mkdir(parents=True, exist_ok=True)


def format_taxonomy_for_prompt():
    """
    Format the level-2 and level-3 taxonomy categories into a readable string
    for inclusion in the extraction prompt.

    The resulting string lists each of the 26 level-2 subcategories grouped
    under its level-3 parent category (one of: Practical, Epistemic, Social,
    Protective, Personal). The extraction model uses this listing to classify
    each value it identifies.

    Returns:
        A multi-line string suitable for insertion into the prompt template.
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


def _create_client(provider):
    """
    Instantiate and return the appropriate API client for *provider*.

    For OpenAI, the client reads the ``OPENAI_API_KEY`` environment variable.
    For Anthropic, it reads ``ANTHROPIC_API_KEY``.

    Args:
        provider: ``"openai"`` or ``"anthropic"``.

    Returns:
        A configured client object ready for use with the extraction
        functions in ``utils.extraction``.

    Raises:
        EnvironmentError: If the required API key is not set.
        ValueError: If *provider* is unrecognized.
    """
    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY environment variable is required for OpenAI extraction. "
                "Set it with: export OPENAI_API_KEY='sk-...'"
            )
        import openai
        return openai.OpenAI(api_key=api_key)

    elif provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY environment variable is required for Anthropic extraction. "
                "Set it with: export ANTHROPIC_API_KEY='sk-ant-...'"
            )
        import anthropic
        return anthropic.Anthropic(api_key=api_key)

    else:
        raise ValueError(f"Unknown provider '{provider}'. Expected 'openai' or 'anthropic'.")


def extract_values_single(user_prompt, model_response, taxonomy_str, client,
                          model=EXTRACTION_MODEL, provider=EXTRACTION_PROVIDER):
    """
    Extract values from a single conversation.

    Builds the prompt, dispatches to the configured provider, and returns
    parsed results. Retry logic and exponential backoff are handled inside
    the provider-specific functions in ``utils.extraction``.

    Args:
        user_prompt: The original user message sent to the Geodesic model.
        model_response: The Geodesic model's response text.
        taxonomy_str: Pre-formatted taxonomy string (from
            ``format_taxonomy_for_prompt``).
        client: A provider-appropriate API client (OpenAI or Anthropic).
        model: Model identifier string. Defaults to the bulk extraction
            model defined in ``config.py``.
        provider: ``"openai"`` or ``"anthropic"``. Defaults to the bulk
            extraction provider defined in ``config.py``.

    Returns:
        A dict with keys ``"values"`` (list of extracted value dicts) and
        ``"raw_response"`` (the model's raw text), or ``None`` if extraction
        failed after all retries.
    """
    prompt = build_extraction_prompt(user_prompt, model_response, taxonomy_str)
    return extract_values(
        prompt,
        model=model,
        provider=provider,
        client=client,
        max_retries=EXTRACTION_MAX_RETRIES,
    )


def process_model_conversations(variant_name, client, taxonomy_str,
                                model=EXTRACTION_MODEL,
                                provider=EXTRACTION_PROVIDER):
    """
    Extract values from all conversations for a single model variant.

    Results are saved incrementally (one JSONL record per conversation) so
    that the process can be interrupted and resumed without re-processing
    completed items. On resume, conversations whose ``prompt_id`` already
    appears in the output file are skipped.

    Args:
        variant_name: The model variant identifier, e.g.
            ``"unfiltered_dpo"`` or ``"alignment_base"``. Must correspond
            to a file ``data/conversations/{variant_name}.jsonl``.
        client: A provider-appropriate API client.
        taxonomy_str: Pre-formatted taxonomy string.
        model: Model identifier for extraction calls.
        provider: ``"openai"`` or ``"anthropic"``.
    """
    input_path = DATA_DIR / "conversations" / f"{variant_name}.jsonl"
    output_path = EXTRACTIONS_DIR / f"{variant_name}_values.jsonl"

    if not input_path.exists():
        print(f"No conversations found for {variant_name}, skipping")
        return

    conversations = []
    with open(input_path) as f:
        for line in f:
            conversations.append(json.loads(line))

    existing_ids = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                rec = json.loads(line)
                existing_ids.add(rec["prompt_id"])

    remaining = [c for c in conversations if c["prompt_id"] not in existing_ids]
    print(f"  {len(existing_ids)} already extracted, {len(remaining)} remaining")

    with open(output_path, "a") as f:
        for conv in tqdm(remaining, desc=f"Extracting {variant_name}"):
            result = extract_values_single(
                conv["user_prompt"],
                conv["model_response"],
                taxonomy_str,
                client,
                model=model,
                provider=provider,
            )

            if result is not None:
                record = {
                    "prompt_id": conv["prompt_id"],
                    "model_variant": variant_name,
                    "model_stage": conv.get("model_stage", "unknown"),
                    "topic_category": conv.get("topic_category", "unknown"),
                    "extracted_values": result["values"],
                    "raw_extraction": result["raw_response"],
                    "extraction_model": model,
                    "extraction_provider": provider,
                }
                f.write(json.dumps(record) + "\n")
                f.flush()


def main():
    """
    Run value extraction across all model variant conversation files.

    By default uses OpenAI GPT-5.2 for bulk extraction, as configured in
    ``config.py``. The provider and model can be changed there without
    modifying this script.
    """
    print(f"Extraction provider : {EXTRACTION_PROVIDER}")
    print(f"Extraction model    : {EXTRACTION_MODEL}")

    client = _create_client(EXTRACTION_PROVIDER)
    taxonomy_str = format_taxonomy_for_prompt()

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
