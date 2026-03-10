"""
Phase 3: Extract values from model-generated conversations using dual-model
extraction with real-time cost tracking.

For each of the ~20,000 conversations generated in Phase 2, this script sends
the user prompt + model response to **two** large language models -- OpenAI
GPT-5.2 and Anthropic Claude Sonnet 4.6 -- and asks each to identify the
values expressed by the AI assistant, classifying every value into the
Anthropic "Values in the Wild" taxonomy.

Running extraction through two independent model families provides built-in
cross-model agreement data at the extraction stage itself. Downstream analyses
can compare how consistently the two extractors identify the same values,
giving researchers a measure of measurement reliability before the separate
validation pass (Phase 4) even begins. This dual-extractor design is
analogous to having two independent human coders annotate the same corpus: if
they agree, the finding is more credible; if they disagree, the disagreement
itself is informative.

A ``CostTracker`` monitors cumulative API spending in real time and warns
when spending approaches or exceeds a configurable budget cap ($50 by
default). Per-call cost records are appended to a JSONL log file for
post-hoc auditing.

Input:  data/conversations/*.jsonl  (from Phase 2)
Output: data/extractions/{extraction_model_name}/{model_variant}_values.jsonl

Requires:
    OPENAI_API_KEY    -- environment variable (for GPT-5.2 extraction)
    ANTHROPIC_API_KEY -- environment variable (for Sonnet 4.6 extraction)
"""

import os
import json
import time
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from config import (
    DATA_DIR,
    EXTRACTION_MODELS,
    EXTRACTION_MAX_RETRIES,
    EXTRACTION_BATCH_SIZE,
)
from utils.taxonomy import get_level2_categories, get_level3_categories
from utils.extraction import build_extraction_prompt, parse_extraction_response, extract_values
from utils.costs import CostTracker


load_dotenv()

EXTRACTIONS_DIR = DATA_DIR / "extractions"
EXTRACTIONS_DIR.mkdir(parents=True, exist_ok=True)


def format_taxonomy_for_prompt():
    """
    Format the level-2 and level-3 taxonomy categories into a readable string
    for inclusion in the extraction prompt.

    The Anthropic "Values in the Wild" taxonomy organizes ~3,300 individual
    values into a four-level hierarchy. This function produces a compact
    text listing of the 26 level-2 subcategories grouped under the 5 level-3
    parent categories (Practical, Epistemic, Social, Protective, Personal).
    The extraction model uses this listing as a controlled vocabulary when
    classifying each value it identifies.

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
    Instantiate and return the appropriate API client for the given provider.

    Each provider requires a corresponding API key set as an environment
    variable. These keys are typically stored in a ``.env`` file at the
    project root and loaded via ``dotenv`` at module import time.

    For policy researchers unfamiliar with API authentication: an API key is
    a secret string that identifies your account to the provider. Treat it
    like a password -- never commit it to version control.

    Args:
        provider: ``"openai"`` or ``"anthropic"``.

    Returns:
        A configured client object ready for use with the extraction
        functions in ``utils.extraction``.

    Raises:
        EnvironmentError: If the required API key is not set in the
            environment.
        ValueError: If *provider* is not one of the two recognized strings.
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


def _estimate_token_count(text):
    """
    Return a rough token count for a text string.

    Uses the common heuristic of ~4 characters per token. This is not exact
    but is sufficient for cost-tracking purposes where we need order-of-
    magnitude estimates between API calls that do not return usage metadata.

    Args:
        text: The input string whose token count is being estimated.

    Returns:
        An integer estimate of the number of tokens.
    """
    return max(1, len(text) // 4)


def extract_values_single(user_prompt, model_response, taxonomy_str, client,
                          model, provider, cost_tracker=None):
    """
    Extract values from a single conversation turn.

    Builds the prompt, dispatches to the configured provider, records the
    cost if a tracker is provided, and returns parsed results. Retry logic
    and exponential backoff are handled inside the provider-specific
    functions in ``utils.extraction``.

    Args:
        user_prompt: The original user message sent to the Geodesic model.
        model_response: The Geodesic model's response text.
        taxonomy_str: Pre-formatted taxonomy string (from
            ``format_taxonomy_for_prompt``).
        client: A provider-appropriate API client (OpenAI or Anthropic).
        model: Model identifier string (e.g. ``"gpt-5.2"``).
        provider: ``"openai"`` or ``"anthropic"``.
        cost_tracker: Optional ``CostTracker`` instance. When provided, the
            estimated token cost of this call is recorded for real-time
            budget monitoring.

    Returns:
        A dict with keys ``"values"`` (list of extracted value dicts) and
        ``"raw_response"`` (the model's raw text), or ``None`` if extraction
        failed after all retries.
    """
    prompt = build_extraction_prompt(user_prompt, model_response, taxonomy_str)
    result = extract_values(
        prompt,
        model=model,
        provider=provider,
        client=client,
        max_retries=EXTRACTION_MAX_RETRIES,
    )

    if result is not None and cost_tracker is not None:
        input_tokens = _estimate_token_count(prompt)
        output_tokens = _estimate_token_count(result["raw_response"])
        cost_tracker.log_call(model, input_tokens, output_tokens)

    return result


def process_model_conversations(variant_name, client, taxonomy_str,
                                model, provider, output_dir, cost_tracker=None):
    """
    Extract values from all conversations for a single model variant using
    a single extraction model.

    Results are saved incrementally (one JSONL record per conversation) so
    that the process can be interrupted and resumed without re-processing
    completed items. On resume, conversations whose ``prompt_id`` already
    appears in the output file are skipped automatically.

    Each output record includes ``extraction_model`` and
    ``extraction_provider`` fields so that downstream analyses can
    unambiguously identify which extractor produced each annotation.

    Args:
        variant_name: The Geodesic model variant identifier, e.g.
            ``"unfiltered_dpo"`` or ``"alignment_base"``. Must correspond
            to a file ``data/conversations/{variant_name}.jsonl``.
        client: A provider-appropriate API client.
        taxonomy_str: Pre-formatted taxonomy string.
        model: Model identifier for extraction calls (e.g. ``"gpt-5.2"``).
        provider: ``"openai"`` or ``"anthropic"``.
        output_dir: Directory where extraction results are written. Each
            extraction model gets its own subdirectory under
            ``data/extractions/``.
        cost_tracker: Optional ``CostTracker`` for real-time cost
            monitoring.
    """
    input_path = DATA_DIR / "conversations" / f"{variant_name}.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{variant_name}_values.jsonl"

    if not input_path.exists():
        print(f"  No conversations found for {variant_name}, skipping")
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

    # Filter out conversations with content that could trigger API content policies
    from utils.extraction import is_content_flagged
    flagged = [c for c in remaining if is_content_flagged(c.get("user_prompt", ""))
               or is_content_flagged(c.get("model_response", ""))]
    remaining = [c for c in remaining if c not in flagged]
    if flagged:
        print(f"    Skipped {len(flagged)} flagged conversations")

    print(f"    {len(existing_ids)} already extracted, {len(remaining)} remaining")

    with open(output_path, "a") as f:
        for conv in tqdm(remaining, desc=f"    {variant_name}"):
            result = extract_values_single(
                conv["user_prompt"],
                conv["model_response"],
                taxonomy_str,
                client,
                model=model,
                provider=provider,
                cost_tracker=cost_tracker,
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

            if cost_tracker is not None and cost_tracker.is_over_budget():
                print(f"\n    Budget cap reached (${cost_tracker.get_total_cost():.2f}). "
                      f"Stopping extraction for {variant_name}.")
                return


def main():
    """
    Run dual-model value extraction across all Geodesic model variant
    conversation files with real-time cost tracking.

    This function orchestrates the full Phase 3 pipeline:

    1. Loads API keys from the environment (via ``.env``).
    2. Creates API clients for both OpenAI and Anthropic.
    3. Formats the taxonomy string once (shared across all calls).
    4. For each extraction model defined in ``EXTRACTION_MODELS``:
       a. For each conversation file in ``data/conversations/``:
          - Loads conversations and resumes from where it left off.
          - Extracts values using the current extraction model.
          - Saves results to a model-specific output directory.
       b. Prints a cost summary for that extraction model.
    5. Prints an overall cost summary comparing both extraction models.

    Output directory structure::

        data/extractions/
            gpt-5.2/
                unfiltered_dpo_values.jsonl
                alignment_base_values.jsonl
                ...
            claude-sonnet-4-6/
                unfiltered_dpo_values.jsonl
                alignment_base_values.jsonl
                ...

    The budget cap defaults to $50.00. If cumulative spending across both
    models exceeds this amount, extraction halts gracefully with progress
    saved. The run can be resumed simply by re-executing this script.
    """
    print("Phase 3: Dual-model value extraction")
    print("=" * 55)
    print(f"Extraction models:")
    for model_name, provider_name in EXTRACTION_MODELS.items():
        print(f"  {model_name} ({provider_name})")
    print()

    clients = {}
    required_providers = set(EXTRACTION_MODELS.values())
    for provider_name in required_providers:
        clients[provider_name] = _create_client(provider_name)
        print(f"  {provider_name.capitalize()} client initialized")

    taxonomy_str = format_taxonomy_for_prompt()
    print(f"  Taxonomy string prepared ({len(taxonomy_str):,} chars)")

    cost_tracker = CostTracker(budget_cap=50.0)

    conv_dir = DATA_DIR / "conversations"
    if not conv_dir.exists():
        print("Error: Run 02_generate_conversations.py first.")
        return

    variant_files = sorted(conv_dir.glob("*.jsonl"))
    print(f"  Found {len(variant_files)} model variant conversation files\n")

    for model_name, provider_name in EXTRACTION_MODELS.items():
        print(f"\n{'─' * 55}")
        print(f"Extraction model: {model_name} ({provider_name})")
        print(f"{'─' * 55}")

        client = clients[provider_name]
        output_dir = EXTRACTIONS_DIR / model_name

        for variant_file in variant_files:
            variant_name = variant_file.stem
            print(f"\n  Variant: {variant_name}")
            process_model_conversations(
                variant_name,
                client,
                taxonomy_str,
                model=model_name,
                provider=provider_name,
                output_dir=output_dir,
                cost_tracker=cost_tracker,
            )

            if cost_tracker.is_over_budget():
                print(f"\nBudget cap exceeded. Halting further extraction.")
                print(cost_tracker.summary())
                return

        print(f"\n  Completed extraction with {model_name}.")
        model_cost = cost_tracker.cost_by_model.get(model_name, 0.0)
        model_calls = cost_tracker.calls_by_model.get(model_name, 0)
        print(f"  Calls: {model_calls:,}  |  Cost: ${model_cost:.4f}")

    print("\n")
    print(cost_tracker.summary())
    print("\nPhase 3 complete. Results saved to:")
    for model_name in EXTRACTION_MODELS:
        model_dir = EXTRACTIONS_DIR / model_name
        print(f"  {model_dir}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None,
                        help="Run extraction for a single model only (e.g., 'gpt-5.2' or 'claude-sonnet-4-6'). "
                             "Omit to run all extraction models sequentially.")
    args = parser.parse_args()

    if args.model:
        # Run a single extraction model (for parallel execution)
        from config import EXTRACTION_MODELS
        if args.model not in EXTRACTION_MODELS:
            print(f"Unknown model: {args.model}. Available: {list(EXTRACTION_MODELS.keys())}")
        else:
            provider = EXTRACTION_MODELS[args.model]
            client = _create_client(provider)
            taxonomy_str = format_taxonomy_for_prompt()
            cost_tracker = CostTracker(budget_cap=30.0)
            conv_dir = DATA_DIR / "conversations"
            output_dir = EXTRACTIONS_DIR / args.model

            print(f"Running extraction with {args.model} ({provider})")
            for variant_file in sorted(conv_dir.glob("*.jsonl")):
                variant_name = variant_file.stem
                print(f"\n  Variant: {variant_name}")
                process_model_conversations(
                    variant_name, client, taxonomy_str,
                    model=args.model, provider=provider,
                    output_dir=output_dir, cost_tracker=cost_tracker,
                )
                if cost_tracker.is_over_budget():
                    print("Budget exceeded, halting.")
                    break
            print(f"\n{cost_tracker.summary()}")
    else:
        main()
