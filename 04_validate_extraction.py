"""
Phase 4: Validate the value extraction pipeline before scaling up.

Before running the expensive Claude API extraction on all 20,000 conversations,
we validate the extraction quality on a small sample. This ensures our extraction
prompt produces reliable, consistent results.

Validation steps:
1. Sample 150 conversations across model variants and topic categories
2. Run value extraction on this sample
3. Compare extracted values against manual coding (human review)
4. Compute inter-rater agreement (Cohen's kappa)
5. If agreement is too low (kappa < 0.4), revise the prompt and re-validate

Input: data/conversations/*.jsonl (from Phase 2)
Output: outputs/validation/
  - validation_sample.json: the 150 sampled conversations
  - validation_extractions.json: Claude's extractions for the sample
  - validation_report.txt: agreement metrics and notes
"""

import json
import random
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score

from config import (
    DATA_DIR, VALIDATION_DIR, VALIDATION_SAMPLE_SIZE,
    MIN_KAPPA_THRESHOLD,
)


def sample_validation_set(n=150):
    """
    Sample conversations for validation, stratified by model variant and topic.

    We want coverage across model variants and topic categories so validation
    results are representative of the full dataset. Each model variant should
    contribute roughly equally to the sample.

    Args:
        n: Number of conversations to sample (default 150)

    Returns:
        List of conversation dicts
    """
    conv_dir = DATA_DIR / "conversations"
    all_conversations = []

    for fpath in sorted(conv_dir.glob("*.jsonl")):
        with open(fpath) as f:
            for line in f:
                all_conversations.append(json.loads(line))

    if not all_conversations:
        print("No conversation files found. Run Phase 2 first.")
        return []

    random.seed(42)

    # Stratify by model variant
    by_variant = {}
    for conv in all_conversations:
        variant = conv["model_variant"]
        by_variant.setdefault(variant, []).append(conv)

    per_variant = max(1, n // len(by_variant))
    sampled = []

    for variant, convs in by_variant.items():
        k = min(len(convs), per_variant)
        sampled.extend(random.sample(convs, k))

    # If under target, add more randomly
    if len(sampled) < n:
        remaining_ids = {c["prompt_id"] + c["model_variant"] for c in sampled}
        pool = [c for c in all_conversations
                if c["prompt_id"] + c["model_variant"] not in remaining_ids]
        extra = random.sample(pool, min(n - len(sampled), len(pool)))
        sampled.extend(extra)

    print(f"Validation sample: {len(sampled)} conversations")
    variant_counts = Counter(c["model_variant"] for c in sampled)
    for v, count in sorted(variant_counts.items()):
        print(f"  {v}: {count}")

    return sampled[:n]


def run_extraction_on_sample(sample):
    """
    Run the value extraction pipeline on the validation sample.

    Uses the same ``extract_with_anthropic`` function that the production
    pipeline (Phase 3) relies on, including structured-output enforcement
    via tool use.  The validation extraction is performed with the
    VALIDATION_EXTRACTION_MODEL (Claude Opus 4.6) so that its results can
    serve as a higher-quality reference when adjudicating disagreements
    between the two bulk-extraction models.

    Args:
        sample: List of conversation dicts, each containing at minimum
            ``prompt_id``, ``model_variant``, ``user_prompt``, and
            ``model_response``.

    Returns:
        List of extraction result dicts with keys ``prompt_id``,
        ``model_variant``, ``user_prompt``, ``model_response``,
        ``extracted_values``, and ``raw_extraction``.
    """
    import anthropic
    from utils.taxonomy import build_category_lookup, get_level3_categories

    lookup = build_category_lookup()
    level3_cats = [name for _, name in get_level3_categories()]

    lines = []
    for l3 in level3_cats:
        lines.append(f"\n{l3}:")
        subcats = [l2 for l2, parent in lookup.items() if parent == l3]
        for sc in sorted(subcats):
            lines.append(f"  - {sc}")
    taxonomy_str = "\n".join(lines)

    from utils.extraction import (
        build_extraction_prompt,
        extract_with_anthropic,
        parse_extraction_response,
    )
    from config import VALIDATION_EXTRACTION_MODEL
    from tqdm import tqdm

    client = anthropic.Anthropic()
    results = []

    for conv in tqdm(sample, desc="Extracting validation sample"):
        prompt = build_extraction_prompt(
            conv["user_prompt"], conv["model_response"], taxonomy_str
        )

        try:
            raw_text = extract_with_anthropic(
                prompt,
                model=VALIDATION_EXTRACTION_MODEL,
                client=client,
            )
            values = parse_extraction_response(raw_text) if raw_text else []
        except Exception as e:
            print(f"  Error: {e}")
            raw_text = ""
            values = []

        results.append({
            "prompt_id": conv["prompt_id"],
            "model_variant": conv["model_variant"],
            "user_prompt": conv["user_prompt"],
            "model_response": conv["model_response"],
            "extracted_values": values,
            "raw_extraction": raw_text or "",
        })

    return results


def compute_agreement(extraction_results, manual_codings):
    """
    Compute inter-rater agreement between Claude's extractions and manual coding.

    Uses Cohen's kappa at the level-2 category level. Each conversation is
    represented as a set of level-2 categories, and we compare whether Claude
    and the human coder identified the same categories.

    Args:
        extraction_results: List of extraction dicts from run_extraction_on_sample
        manual_codings: Dict mapping prompt_id to list of level-2 category strings

    Returns:
        Dict with kappa score and detailed comparison
    """
    # Build per-conversation category sets for comparison
    all_categories = set()
    for result in extraction_results:
        for val in result["extracted_values"]:
            all_categories.add(val.get("taxonomy_level2_category", ""))
    for cats in manual_codings.values():
        all_categories.update(cats)

    all_categories = sorted(all_categories - {""})

    # Create binary vectors: for each conversation × category, was it present?
    claude_labels = []
    human_labels = []

    for result in extraction_results:
        pid = result["prompt_id"]
        if pid not in manual_codings:
            continue

        claude_cats = {v.get("taxonomy_level2_category", "") for v in result["extracted_values"]}
        human_cats = set(manual_codings[pid])

        for cat in all_categories:
            claude_labels.append(1 if cat in claude_cats else 0)
            human_labels.append(1 if cat in human_cats else 0)

    if not claude_labels:
        return {"kappa": 0.0, "note": "No overlapping samples for comparison"}

    kappa = cohen_kappa_score(human_labels, claude_labels)

    return {
        "kappa": kappa,
        "n_compared": len(extraction_results),
        "n_categories": len(all_categories),
        "passes_threshold": kappa >= MIN_KAPPA_THRESHOLD,
    }


def generate_coding_template(sample, output_path):
    """
    Generate a CSV template for manual coding of the validation sample.

    The researcher reviews each conversation and lists the level-2 categories
    of values expressed by the AI. This is compared against Claude's extractions
    to measure agreement.

    Args:
        sample: List of conversation dicts
        output_path: Path to save the CSV template
    """
    rows = []
    for conv in sample:
        rows.append({
            "prompt_id": conv["prompt_id"],
            "model_variant": conv["model_variant"],
            "user_prompt": conv["user_prompt"][:200],
            "model_response": conv["model_response"][:500],
            "manual_level2_categories": "",  # to be filled in by researcher
            "notes": "",
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Saved coding template to {output_path}")
    print("Fill in 'manual_level2_categories' column (comma-separated category names)")


def main():
    """Run the validation pipeline."""
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Sample conversations
    sample = sample_validation_set(VALIDATION_SAMPLE_SIZE)
    if not sample:
        return

    # Save sample
    sample_path = VALIDATION_DIR / "validation_sample.json"
    with open(sample_path, "w") as f:
        json.dump(sample, f, indent=2)
    print(f"Saved validation sample to {sample_path}")

    # Step 2: Generate manual coding template
    template_path = VALIDATION_DIR / "manual_coding_template.csv"
    generate_coding_template(sample, template_path)

    # Step 3: Run extraction (requires API key)
    try:
        results = run_extraction_on_sample(sample)
        results_path = VALIDATION_DIR / "validation_extractions.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved extraction results to {results_path}")
    except Exception as e:
        print(f"Extraction failed (API key may not be set): {e}")
        print("Set ANTHROPIC_API_KEY and re-run, or skip to manual coding.")

    print("\nNext steps:")
    print(f"1. Review and fill in: {template_path}")
    print("2. Re-run this script with --compute-agreement to check kappa")


if __name__ == "__main__":
    main()
