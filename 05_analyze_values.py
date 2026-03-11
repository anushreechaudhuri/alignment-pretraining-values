"""
Phase 5: Statistical analysis of value distributions across model variants.

This is the core analysis module. It compares value distributions between
Geodesic model variants to test whether alignment pretraining produces
broad value shifts or only narrow safety-related effects.

Analyses performed:
  5a. Alignment Upsampled vs. Unfiltered (post-trained) -- main effect
  5b. Misalignment Upsampled vs. Unfiltered -- negative effect
  5c. Filtered vs. Unfiltered -- filtering effect
  5d. Base vs. post-trained comparisons -- persistence through training
  5e. Stratified by topic category -- breadth test
  5f. Summary metrics and visualizations
  5g. Bridging analysis with Geodesic alignment scores

Methodological safeguards:
  - Response-length normalization (values per 100 words) alongside raw counts,
    because longer responses mechanically produce more extracted values.
  - Dual-extractor agreement: primary analysis uses the intersection of values
    found by both GPT-5.2 and Sonnet 4.6; union reported as sensitivity check.
  - Holm-Bonferroni correction across the 3 core pairwise tests (5a-5c).
  - Expected-cell-count check before every chi-squared test.
  - Note: generation seeds are not yet pinned in modal_inference.py; results
    therefore reflect a single random draw per model. This will be addressed
    in a future patch to modal_inference.py.

Input: data/extractions/{extraction_model}/{variant}_values.jsonl (from Phase 3)
       data/conversations/{variant}.jsonl (from Phase 2, for response lengths)
Output: outputs/figures/, outputs/tables/, outputs/data/
"""

import json
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cosine
from scipy import stats as sp_stats
from statsmodels.stats.multitest import multipletests

from config import (
    DATA_DIR, FIGURES_DIR, TABLES_DIR, PROCESSED_DATA_DIR,
    TOP_LEVEL_CATEGORIES, SIGNIFICANCE_LEVEL, EXTRACTION_MODELS,
)
from utils.taxonomy import (
    get_level2_categories, get_level3_categories, build_category_lookup,
)
from utils.stats import (
    chi_squared_test, cosine_similarity, proportion_differences_with_ci,
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_extractions():
    """Load value extractions from all model variants and extractors into a
    single DataFrame.

    The extraction pipeline (Phase 3) writes one JSONL file per
    (extraction_model, geodesic_variant) pair, stored under::

        data/extractions/{extraction_model_name}/{variant}_values.jsonl

    This function globs across that two-level directory structure and tags
    every row with an ``extraction_model`` column so that downstream code
    can compute inter-extractor agreement and apply the intersection /
    union merge strategies.

    Returns:
        pd.DataFrame where each row is one extracted value, with columns:
            prompt_id, model_variant, model_stage, topic_category,
            value_name, level2_category, level3_category, confidence,
            extraction_model.
    """
    extractions_dir = DATA_DIR / "extractions"

    all_records = []
    for fpath in sorted(extractions_dir.glob("*/*_values.jsonl")):
        extractor_name = fpath.parent.name
        with open(fpath) as f:
            for line in f:
                record = json.loads(line)
                for val in record.get("extracted_values", []):
                    all_records.append({
                        "prompt_id": record["prompt_id"],
                        "model_variant": record["model_variant"],
                        "model_stage": record["model_stage"],
                        "topic_category": record["topic_category"],
                        "value_name": val.get("raw_value_name", ""),
                        "level2_category": val.get("taxonomy_level2_category", ""),
                        "level3_category": val.get("taxonomy_level3_category", ""),
                        "confidence": val.get("confidence", "medium"),
                        "extraction_model": record.get(
                            "extraction_model", extractor_name
                        ),
                    })

    df = pd.DataFrame(all_records)
    if len(df) > 0:
        n_extractors = df["extraction_model"].nunique()
        print(
            f"Loaded {len(df)} extracted values from "
            f"{df['model_variant'].nunique()} model variants, "
            f"{n_extractors} extractor(s)"
        )
    return df


def load_response_lengths():
    """Load response word-counts from conversation files produced in Phase 2.

    Each conversation JSONL record written by ``02_generate_conversations.py``
    (or ``modal_inference.py``) contains a ``model_response`` field with the
    full text of the model's reply.  We compute word count (whitespace split)
    as a lightweight proxy for response length.

    Returns:
        pd.DataFrame with columns: prompt_id, model_variant, word_count.
    """
    conv_dir = DATA_DIR / "conversations"
    rows = []
    if not conv_dir.exists():
        print("Warning: conversations directory not found; "
              "response-length normalization will be unavailable.")
        return pd.DataFrame(columns=["prompt_id", "model_variant", "word_count"])

    for fpath in sorted(conv_dir.glob("*.jsonl")):
        with open(fpath) as f:
            for line in f:
                rec = json.loads(line)
                word_count = len(rec.get("model_response", "").split())
                rows.append({
                    "prompt_id": rec["prompt_id"],
                    "model_variant": rec["model_variant"],
                    "word_count": word_count,
                })

    df = pd.DataFrame(rows)
    if len(df) > 0:
        print(f"Loaded response lengths for {len(df)} conversations "
              f"(median {df['word_count'].median():.0f} words)")
    return df


# ---------------------------------------------------------------------------
# Response-length normalization
# ---------------------------------------------------------------------------

def add_normalized_counts(df, lengths_df):
    """Merge response word-counts into the extractions DataFrame and compute a
    values-per-100-words rate for every conversation.

    Policy-relevant motivation: if one model variant produces systematically
    longer responses, it will mechanically yield more extracted values even if
    the *density* of value expression is the same.  Normalizing by response
    length removes this confound.

    Args:
        df: Extractions DataFrame (one row per extracted value).
        lengths_df: DataFrame from ``load_response_lengths()`` with columns
            prompt_id, model_variant, word_count.

    Returns:
        Tuple (df_with_norm, conv_summary) where:
            df_with_norm: input ``df`` with an added ``word_count`` column.
            conv_summary: per-conversation DataFrame with columns
                prompt_id, model_variant, value_count, word_count,
                values_per_100_words.
    """
    if lengths_df is None or len(lengths_df) == 0:
        return df, pd.DataFrame()

    df_with_wc = df.merge(
        lengths_df[["prompt_id", "model_variant", "word_count"]],
        on=["prompt_id", "model_variant"],
        how="left",
    )

    conv_counts = (
        df.groupby(["prompt_id", "model_variant"])
        .size()
        .reset_index(name="value_count")
    )
    conv_summary = conv_counts.merge(
        lengths_df, on=["prompt_id", "model_variant"], how="left"
    )
    conv_summary["values_per_100_words"] = np.where(
        conv_summary["word_count"] > 0,
        conv_summary["value_count"] / conv_summary["word_count"] * 100,
        np.nan,
    )

    return df_with_wc, conv_summary


def print_length_value_correlation(conv_summary):
    """Print the Pearson correlation between response length and raw value
    count for each model variant.

    This is a diagnostic: a high positive correlation means longer responses
    produce more values, strengthening the case for normalized analysis.

    Args:
        conv_summary: per-conversation DataFrame with value_count, word_count,
            and model_variant columns.
    """
    if conv_summary is None or len(conv_summary) == 0:
        return

    print("\n--- Response-length / value-count correlation by model ---")
    for variant, grp in conv_summary.groupby("model_variant"):
        valid = grp.dropna(subset=["word_count"])
        if len(valid) < 3:
            continue
        r, p = sp_stats.pearsonr(valid["word_count"], valid["value_count"])
        print(f"  {variant}: r={r:.3f}  (p={p:.3e}, n={len(valid)})")
    print()


# ---------------------------------------------------------------------------
# Dual-extractor agreement
# ---------------------------------------------------------------------------

def compute_extractor_agreement(df):
    """Measure how well the two extraction models (GPT-5.2 and Sonnet 4.6)
    agree on which level-2 value categories are present in each conversation.

    For policy research, extractor agreement is a key validity check: if the
    two models disagree about which values a response expresses, the
    measurement is noisy and downstream comparisons may be unreliable.

    Method:
        For every conversation (identified by prompt_id + model_variant), we
        build a binary vector over the full set of level-2 categories
        (1 = category extracted, 0 = not).  We then compute:

        * **Percent agreement** -- fraction of categories on which the two
          extractors give the same binary label.
        * **Cohen's kappa** -- agreement corrected for chance.  Values above
          0.6 are generally considered "substantial"; below 0.4 is "fair" at
          best and would warrant caution interpreting results.

        Results are reported per level-2 category (treating each category as
        a separate binary classification task across conversations) and as a
        single overall number (pooling all category-conversation cells).

    Args:
        df: Extractions DataFrame with an ``extraction_model`` column.

    Returns:
        Tuple (category_agreement_df, overall_stats_dict) or (None, None) if
        fewer than two extractors are present.
    """
    extractors = df["extraction_model"].unique()
    if len(extractors) < 2:
        print("Only one extractor found; skipping agreement analysis.")
        return None, None

    extractor_a, extractor_b = sorted(extractors)[:2]
    print(f"\n{'='*60}")
    print(f"EXTRACTOR AGREEMENT: {extractor_a} vs {extractor_b}")
    print(f"{'='*60}")

    all_l2_cats = sorted(df["level2_category"].dropna().unique())
    if not all_l2_cats:
        print("No level-2 categories found.")
        return None, None

    def _conversation_categories(sub_df):
        """Return set of level-2 categories for each (prompt_id, model_variant)."""
        return (
            sub_df.groupby(["prompt_id", "model_variant"])["level2_category"]
            .apply(set)
        )

    cats_a = _conversation_categories(df[df["extraction_model"] == extractor_a])
    cats_b = _conversation_categories(df[df["extraction_model"] == extractor_b])

    shared_keys = sorted(set(cats_a.index) & set(cats_b.index))
    if not shared_keys:
        print("No overlapping conversations between extractors.")
        return None, None

    print(f"  Overlapping conversations: {len(shared_keys)}")

    all_labels_a = []
    all_labels_b = []
    per_cat_results = []

    for cat in all_l2_cats:
        labels_a = [1 if cat in cats_a[k] else 0 for k in shared_keys]
        labels_b = [1 if cat in cats_b[k] else 0 for k in shared_keys]

        all_labels_a.extend(labels_a)
        all_labels_b.extend(labels_b)

        agree = sum(a == b for a, b in zip(labels_a, labels_b))
        pct_agree = agree / len(shared_keys)

        try:
            kappa = _cohens_kappa(labels_a, labels_b)
        except ValueError:
            kappa = np.nan

        per_cat_results.append({
            "level2_category": cat,
            "percent_agreement": pct_agree,
            "cohens_kappa": kappa,
            "n_conversations": len(shared_keys),
            "prevalence_a": sum(labels_a) / len(labels_a),
            "prevalence_b": sum(labels_b) / len(labels_b),
        })

    cat_df = pd.DataFrame(per_cat_results).sort_values(
        "cohens_kappa", ascending=True
    )

    overall_agree = sum(
        a == b for a, b in zip(all_labels_a, all_labels_b)
    ) / len(all_labels_a)
    try:
        overall_kappa = _cohens_kappa(all_labels_a, all_labels_b)
    except ValueError:
        overall_kappa = np.nan

    overall_stats = {
        "percent_agreement": overall_agree,
        "cohens_kappa": overall_kappa,
        "n_conversations": len(shared_keys),
        "n_categories": len(all_l2_cats),
    }

    print(f"\n  Overall percent agreement: {overall_agree:.3f}")
    print(f"  Overall Cohen's kappa:     {overall_kappa:.3f}")
    print(f"\n  Per-category breakdown:")
    print(cat_df.to_string(index=False))

    return cat_df, overall_stats


def _cohens_kappa(labels_a, labels_b):
    """Compute Cohen's kappa for two lists of binary labels.

    Args:
        labels_a, labels_b: lists of 0/1 integers of equal length.

    Returns:
        float: kappa statistic (1 = perfect agreement, 0 = chance, <0 = worse
        than chance).

    Raises:
        ValueError: if inputs are empty or constant across both raters (making
        kappa undefined).
    """
    a = np.array(labels_a)
    b = np.array(labels_b)
    n = len(a)
    if n == 0:
        raise ValueError("Empty label lists")

    po = np.sum(a == b) / n

    pa1 = np.sum(a) / n
    pb1 = np.sum(b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)

    if pe == 1.0:
        raise ValueError("Perfect expected agreement; kappa is undefined")
    return (po - pe) / (1 - pe)


def apply_extractor_merge(df, strategy="intersection"):
    """Filter extractions to only those values on which both extractors agree,
    or retain the union of both.

    The **intersection** strategy (default for primary analysis) keeps a value
    in a given conversation only if *both* extractors identified that level-2
    category.  This is conservative: it reduces recall but increases
    precision, guarding against extractor-specific hallucinations.

    The **union** strategy keeps a value if *either* extractor found it.
    This is reported as a sensitivity check.

    Args:
        df: Extractions DataFrame with extraction_model column.
        strategy: ``"intersection"`` or ``"union"``.

    Returns:
        pd.DataFrame filtered/merged accordingly, with extraction_model
        column removed (no longer meaningful after merge).
    """
    extractors = sorted(df["extraction_model"].unique())
    if len(extractors) < 2:
        return df.drop(columns=["extraction_model"], errors="ignore")

    extractor_a, extractor_b = extractors[:2]

    def _cat_set(sub):
        return (
            sub.groupby(["prompt_id", "model_variant"])["level2_category"]
            .apply(set)
        )

    cats_a = _cat_set(df[df["extraction_model"] == extractor_a])
    cats_b = _cat_set(df[df["extraction_model"] == extractor_b])
    shared_keys = set(cats_a.index) & set(cats_b.index)

    if strategy == "intersection":
        keep = {
            k: cats_a[k] & cats_b[k] for k in shared_keys
        }
    elif strategy == "union":
        all_keys = set(cats_a.index) | set(cats_b.index)
        keep = {}
        for k in all_keys:
            s_a = cats_a.get(k, set()) if k in cats_a.index else set()
            s_b = cats_b.get(k, set()) if k in cats_b.index else set()
            keep[k] = s_a | s_b
    else:
        raise ValueError(f"Unknown merge strategy: {strategy}")

    mask = df.apply(
        lambda row: row["level2_category"] in keep.get(
            (row["prompt_id"], row["model_variant"]), set()
        ),
        axis=1,
    )
    merged = df.loc[mask].copy()
    merged = merged.drop_duplicates(
        subset=["prompt_id", "model_variant", "level2_category"]
    )
    merged = merged.drop(columns=["extraction_model"], errors="ignore")

    return merged


# ---------------------------------------------------------------------------
# Distribution computation
# ---------------------------------------------------------------------------

def compute_value_distributions(df, group_col="model_variant",
                                level="level2_category"):
    """Compute value-category frequency distributions for each model variant.

    Args:
        df: DataFrame of extracted values (one row per value).
        group_col: Column to group by (default ``"model_variant"``).
        level: Taxonomy column to aggregate at.  Use ``"level2_category"``
            (26 subcategories, the primary analysis level) or
            ``"level3_category"`` (5 top-level categories).

    Returns:
        Dict mapping group names to pandas Series of category counts
        (index = category name, values = integer counts).
    """
    distributions = {}
    for name, group in df.groupby(group_col):
        distributions[name] = group[level].value_counts()
    return distributions


def compute_normalized_distributions(conv_summary, group_col="model_variant"):
    """Compute values-per-100-words distributions per model variant.

    Instead of counting raw values per category, this aggregates the
    per-conversation normalized rate.  The result is a dict of Series
    analogous to ``compute_value_distributions`` but in rate-per-100-words
    units.

    Args:
        conv_summary: per-conversation DataFrame from
            ``add_normalized_counts()``, must contain model_variant and
            values_per_100_words columns.
        group_col: column to group by.

    Returns:
        Dict mapping model variant names to a single-element Series
        containing the mean values_per_100_words for that variant.
    """
    if conv_summary is None or len(conv_summary) == 0:
        return {}

    summary = (
        conv_summary.groupby(group_col)["values_per_100_words"]
        .agg(["mean", "median", "std", "count"])
    )
    return summary


# ---------------------------------------------------------------------------
# Pairwise comparison
# ---------------------------------------------------------------------------

def _check_expected_counts(dist_a, dist_b):
    """Check whether any expected cell in the chi-squared contingency table
    falls below 5, which violates an assumption of the test.

    When expected counts are low, the chi-squared approximation becomes
    unreliable.  Common remedies include collapsing sparse categories or
    switching to Fisher's exact test (infeasible for large tables) or a
    permutation test.

    Args:
        dist_a, dist_b: pandas Series of category counts.

    Returns:
        Tuple (has_low_cells: bool, min_expected: float, n_low_cells: int).
    """
    all_cats = sorted(set(dist_a.index) | set(dist_b.index))
    a = np.array([dist_a.get(c, 0) for c in all_cats], dtype=float)
    b = np.array([dist_b.get(c, 0) for c in all_cats], dtype=float)

    row_totals = np.array([a.sum(), b.sum()])
    col_totals = a + b
    grand_total = row_totals.sum()

    if grand_total == 0:
        return True, 0.0, len(all_cats) * 2

    expected = np.outer(row_totals, col_totals) / grand_total
    n_low = int(np.sum(expected < 5))
    min_exp = float(expected.min())

    return n_low > 0, min_exp, n_low


def run_pairwise_comparison(dist_a, dist_b, name_a, name_b):
    """Run the full suite of statistical tests comparing two model variants'
    value distributions.

    Before computing the chi-squared statistic, this function checks that all
    expected cell counts are at least 5 (a standard assumption of the test).
    If any cells fall below the threshold, a warning is printed.

    Args:
        dist_a: pandas Series of value-category counts for model A.
        dist_b: pandas Series of value-category counts for model B.
        name_a: human-readable label for model A.
        name_b: human-readable label for model B.

    Returns:
        Dict with keys: comparison, chi2 (dict), cosine_similarity (float),
        proportion_differences (DataFrame), low_expected_cells (bool).
    """
    has_low, min_exp, n_low = _check_expected_counts(dist_a, dist_b)
    if has_low:
        print(f"  WARNING: {n_low} expected cell(s) below 5 "
              f"(min={min_exp:.2f}) in {name_a} vs {name_b}. "
              f"Chi-squared approximation may be unreliable.")

    chi2_result = chi_squared_test(dist_a, dist_b)
    cos_sim = cosine_similarity(dist_a, dist_b)
    prop_diffs = proportion_differences_with_ci(dist_a, dist_b)

    return {
        "comparison": f"{name_a} vs {name_b}",
        "chi2": chi2_result,
        "cosine_similarity": cos_sim,
        "proportion_differences": prop_diffs,
        "low_expected_cells": has_low,
    }


# ---------------------------------------------------------------------------
# Core analyses (5a-5c) with Holm-Bonferroni correction
# ---------------------------------------------------------------------------

def _apply_holm_bonferroni(results):
    """Apply Holm-Bonferroni correction to the chi-squared p-values across
    the set of core pairwise tests (5a-5c).

    Holm-Bonferroni is uniformly more powerful than plain Bonferroni while
    still controlling the family-wise error rate.  With only 3 tests the
    practical difference is small, but it is methodologically preferable.

    Args:
        results: list of result dicts from ``run_pairwise_comparison``,
            each containing a ``"chi2"`` sub-dict with ``"p_value"``.

    Returns:
        The input list, mutated in place with an added
        ``chi2["p_value_holm"]`` and ``chi2["significant_holm"]`` entry in
        each result dict.
    """
    if not results:
        return results

    raw_p = [r["chi2"]["p_value"] for r in results]
    reject, p_adj, _, _ = multipletests(raw_p, alpha=SIGNIFICANCE_LEVEL,
                                        method="holm")
    for r, p_h, sig in zip(results, p_adj, reject):
        r["chi2"]["p_value_holm"] = float(p_h)
        r["chi2"]["significant_holm"] = bool(sig)

    return results


def run_core_analyses(df, conv_summary=None):
    """Run analyses 5a-5c: core pairwise comparisons between post-trained
    models, with both raw and response-length-normalized results.

    These three comparisons test whether different pretraining interventions
    produce measurably different value profiles when models are used in
    ordinary conversations:

    * **5a** (alignment vs. unfiltered) is the main hypothesis test.
    * **5b** (misalignment vs. unfiltered) tests for a negative-direction
      effect.
    * **5c** (filtered vs. unfiltered) tests whether removing AI-related
      training text changes value expression.

    All three p-values are jointly corrected using the Holm-Bonferroni
    procedure before significance is assessed.

    Args:
        df: Extractions DataFrame (post-merge, no extraction_model column).
        conv_summary: per-conversation DataFrame with values_per_100_words;
            if provided, normalized distributions are also reported.

    Returns:
        List of result dicts (one per comparison).
    """
    print("\n" + "=" * 60)
    print("CORE ANALYSES: Post-trained model comparisons")
    print("=" * 60)

    post_trained = df[df["model_stage"] == "post-trained"]
    dists = compute_value_distributions(post_trained)

    norm_summary = None
    if conv_summary is not None and len(conv_summary) > 0:
        pt_conv = conv_summary[
            conv_summary["model_variant"].str.endswith("_dpo")
        ]
        norm_summary = compute_normalized_distributions(pt_conv)

    comparisons = [
        ("alignment_dpo", "unfiltered_dpo", "5a: Alignment vs Unfiltered"),
        ("misalignment_dpo", "unfiltered_dpo", "5b: Misalignment vs Unfiltered"),
        ("filtered_dpo", "unfiltered_dpo", "5c: Filtered vs Unfiltered"),
    ]

    results = []
    for var_a, var_b, label in comparisons:
        if var_a not in dists or var_b not in dists:
            print(f"  Skipping {label}: missing data")
            continue

        result = run_pairwise_comparison(
            dists[var_a], dists[var_b], var_a, var_b
        )
        result["label"] = label
        results.append(result)

    results = _apply_holm_bonferroni(results)

    for result in results:
        label = result["label"]
        chi2 = result["chi2"]
        print(f"\n{label}")
        print(f"  Chi-squared: {chi2['chi2']:.2f}, "
              f"p={chi2['p_value']:.4e}, "
              f"p_holm={chi2['p_value_holm']:.4e}")
        print(f"  Significant after Holm-Bonferroni: "
              f"{'YES' if chi2['significant_holm'] else 'no'}")
        print(f"  Cramer's V: {chi2['cramers_v']:.4f}")
        print(f"  Cosine similarity: {result['cosine_similarity']:.4f}")
        if result["low_expected_cells"]:
            print(f"  (!) Low expected cell counts -- interpret with caution")

        top_diffs = result["proportion_differences"].head(5)
        print(f"  Top differences:")
        for _, row in top_diffs.iterrows():
            sig = "*" if row["significant"] else ""
            print(f"    {row['category']}: {row['diff']:+.4f} "
                  f"(p_adj={row['p_adjusted']:.4e}) {sig}")

    if norm_summary is not None and len(norm_summary) > 0:
        print("\n  --- Normalized values per 100 words (post-trained) ---")
        print(norm_summary.to_string())

    return results


# ---------------------------------------------------------------------------
# Breadth analysis (5e)
# ---------------------------------------------------------------------------

def run_breadth_analysis(df):
    """Analysis 5e: Test whether alignment effects are consistent across
    topic categories.

    If value-distribution differences appear across ALL topic categories,
    that is evidence for broad "deep character" effects of alignment
    pretraining.  If differences appear only in safety-adjacent topics, that
    suggests narrow, surface-level effects.

    Args:
        df: Extractions DataFrame.

    Returns:
        pd.DataFrame with one row per topic category, containing chi-squared
        statistics, Cramer's V, cosine similarity, and sample size.
    """
    print("\n" + "=" * 60)
    print("BREADTH ANALYSIS: Effects by topic category")
    print("=" * 60)

    post_trained = df[df["model_stage"] == "post-trained"]
    results_by_topic = []

    for topic in post_trained["topic_category"].unique():
        topic_df = post_trained[post_trained["topic_category"] == topic]
        dists = compute_value_distributions(topic_df)

        if "alignment_dpo" in dists and "unfiltered_dpo" in dists:
            result = run_pairwise_comparison(
                dists["alignment_dpo"], dists["unfiltered_dpo"],
                "alignment_dpo", "unfiltered_dpo"
            )
            results_by_topic.append({
                "topic": topic,
                "chi2": result["chi2"]["chi2"],
                "p_value": result["chi2"]["p_value"],
                "cramers_v": result["chi2"]["cramers_v"],
                "cosine_sim": result["cosine_similarity"],
                "n_values": len(topic_df),
                "low_expected_cells": result["low_expected_cells"],
            })

    breadth_df = pd.DataFrame(results_by_topic)
    print(breadth_df.to_string(index=False))

    return breadth_df


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def plot_value_distributions(df):
    """Generate stacked bar chart of value category distributions across all
    model variants at the level-3 (top-level) taxonomy.

    Args:
        df: Extractions DataFrame.
    """
    dists = compute_value_distributions(df, level="level3_category")

    props = {}
    for name, counts in dists.items():
        total = counts.sum()
        props[name] = {
            cat: counts.get(cat, 0) / total for cat in TOP_LEVEL_CATEGORIES
        }

    props_df = pd.DataFrame(props).T

    fig, ax = plt.subplots(figsize=(12, 6))
    props_df.plot(kind="bar", stacked=True, ax=ax, colormap="Set2")
    ax.set_ylabel("Proportion of Extracted Values")
    ax.set_xlabel("Model Variant")
    ax.set_title("Value Category Distribution by Model Variant")
    ax.legend(title="Category", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    fig.savefig(
        FIGURES_DIR / "value_distributions_stacked.png",
        dpi=150, bbox_inches="tight",
    )
    fig.savefig(
        FIGURES_DIR / "value_distributions_stacked.pdf",
        bbox_inches="tight",
    )
    plt.close()
    print(f"Saved stacked bar chart to {FIGURES_DIR}")


def plot_heatmap(df):
    """Generate heatmap of level-2 category proportions by model variant.

    Args:
        df: Extractions DataFrame.
    """
    dists = compute_value_distributions(df, level="level2_category")

    all_cats = sorted(set().union(*[set(d.index) for d in dists.values()]))
    matrix = {}
    for name, counts in dists.items():
        total = counts.sum()
        matrix[name] = {cat: counts.get(cat, 0) / total for cat in all_cats}

    matrix_df = pd.DataFrame(matrix).T

    fig, ax = plt.subplots(figsize=(16, 8))
    sns.heatmap(matrix_df, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax)
    ax.set_title("Level-2 Value Category Proportions by Model Variant")
    plt.tight_layout()

    fig.savefig(
        FIGURES_DIR / "value_heatmap_level2.png",
        dpi=150, bbox_inches="tight",
    )
    fig.savefig(
        FIGURES_DIR / "value_heatmap_level2.pdf", bbox_inches="tight"
    )
    plt.close()
    print(f"Saved heatmap to {FIGURES_DIR}")


def plot_cosine_similarity_matrix(df):
    """Generate cosine similarity matrix between all model variants' value
    distributions at level-2.

    Args:
        df: Extractions DataFrame.
    """
    dists = compute_value_distributions(df, level="level2_category")
    variants = sorted(dists.keys())

    n = len(variants)
    sim_matrix = np.zeros((n, n))
    for i, va in enumerate(variants):
        for j, vb in enumerate(variants):
            sim_matrix[i, j] = cosine_similarity(dists[va], dists[vb])

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        sim_matrix, annot=True, fmt=".3f", cmap="coolwarm",
        xticklabels=variants, yticklabels=variants, ax=ax,
        vmin=0.8, vmax=1.0,
    )
    ax.set_title(
        "Cosine Similarity of Value Distributions Between Model Variants"
    )
    plt.tight_layout()

    fig.savefig(
        FIGURES_DIR / "cosine_similarity_matrix.png",
        dpi=150, bbox_inches="tight",
    )
    fig.savefig(
        FIGURES_DIR / "cosine_similarity_matrix.pdf", bbox_inches="tight"
    )
    plt.close()
    print(f"Saved cosine similarity matrix to {FIGURES_DIR}")


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_summary_tables(core_results, breadth_df):
    """Save analysis results as CSV tables for inclusion in reports.

    Args:
        core_results: list of dicts from ``run_core_analyses``.
        breadth_df: DataFrame from ``run_breadth_analysis``.
    """
    chi2_rows = []
    for r in core_results:
        chi2_rows.append({
            "comparison": r["label"],
            "chi2_statistic": r["chi2"]["chi2"],
            "p_value": r["chi2"]["p_value"],
            "p_value_holm": r["chi2"]["p_value_holm"],
            "significant_holm": r["chi2"]["significant_holm"],
            "cramers_v": r["chi2"]["cramers_v"],
            "cosine_similarity": r["cosine_similarity"],
            "low_expected_cells": r.get("low_expected_cells", False),
        })
    chi2_df = pd.DataFrame(chi2_rows)
    chi2_df.to_csv(TABLES_DIR / "chi_squared_comparisons.csv", index=False)

    if breadth_df is not None and len(breadth_df) > 0:
        breadth_df.to_csv(TABLES_DIR / "breadth_analysis.csv", index=False)

    for r in core_results:
        label = r["label"].split(":")[0].strip()
        r["proportion_differences"].to_csv(
            TABLES_DIR / f"proportion_diffs_{label}.csv", index=False
        )

    print(f"Saved summary tables to {TABLES_DIR}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Run all analyses and generate outputs.

    Pipeline:
        1. Load extractions and response lengths.
        2. Compute dual-extractor agreement (validity check).
        3. Merge extractions using the intersection strategy (primary) and
           report union counts as a sensitivity check.
        4. Compute response-length normalization diagnostics.
        5. Run core pairwise analyses (5a-5c) with Holm-Bonferroni
           correction.
        6. Run breadth analysis (5e).
        7. Generate visualizations and save tables.

    Note on reproducibility: generation seeds are not yet pinned in
    modal_inference.py.  The conversations analyzed here therefore reflect a
    single random draw per model variant.  A future patch will add
    deterministic seeding to the inference step; until then, exact numerical
    results may vary across regeneration runs.
    """
    df = load_all_extractions()

    if len(df) == 0:
        print("No extraction data found. Run Phase 3 first.")
        return

    print(f"\nDataset summary:")
    print(f"  Total extracted values: {len(df)}")
    print(f"  Model variants: {df['model_variant'].nunique()}")
    print(f"  Unique prompts: {df['prompt_id'].nunique()}")
    print(f"  Topic categories: {df['topic_category'].nunique()}")
    print(f"  Extraction models: {df['extraction_model'].nunique()}")

    # --- Extractor agreement (Issue 3) ---
    cat_agreement, overall_agreement = compute_extractor_agreement(df)
    if cat_agreement is not None:
        cat_agreement.to_csv(
            TABLES_DIR / "extractor_agreement_by_category.csv", index=False
        )

    # --- Determine analysis strategy based on available extractors ---
    extractors = df["extraction_model"].unique().tolist()
    print(f"\n  Available extractors: {extractors}")

    # Check if we have substantial dual-extractor overlap (>1000 shared conversations)
    # to decide between intersection merge vs single-extractor analysis
    if len(extractors) >= 2:
        from itertools import combinations
        ext_a, ext_b = extractors[0], extractors[1]
        ids_a = set(df[df["extraction_model"] == ext_a]["prompt_id"].unique())
        ids_b = set(df[df["extraction_model"] == ext_b]["prompt_id"].unique())
        overlap = len(ids_a & ids_b)
        print(f"  Overlap between {ext_a} and {ext_b}: {overlap} conversations")

    if len(extractors) >= 2 and overlap >= 1000:
        # Substantial dual-extractor overlap: use intersection for primary
        df_primary = apply_extractor_merge(df, strategy="intersection")
        df_sensitivity = apply_extractor_merge(df, strategy="union")
        print(f"  Intersection merge (primary): {len(df_primary)} value rows")
        print(f"  Union merge (sensitivity):    {len(df_sensitivity)} value rows")
        merge_label = "intersection"
    else:
        # Insufficient overlap or single extractor: use the extractor with most data
        extractor_counts = df.groupby("extraction_model").size()
        primary_extractor = extractor_counts.idxmax()
        df_primary = df[df["extraction_model"] == primary_extractor].copy()
        df_sensitivity = None
        print(f"  Using {primary_extractor} as primary ({len(df_primary)} value rows)")
        if len(extractors) >= 2:
            print(f"  (Overlap of {overlap} too small for intersection merge; need >=1000)")
        merge_label = f"single ({primary_extractor})"

    # --- Response-length normalization (Issue 2) ---
    lengths_df = load_response_lengths()
    df_primary, conv_summary = add_normalized_counts(df_primary, lengths_df)
    print_length_value_correlation(conv_summary)

    # --- Core analyses ---
    print(f"\n  Primary analysis using: {merge_label}")
    core_results = run_core_analyses(df_primary, conv_summary)

    # --- Sensitivity check (only if dual-extractor) ---
    if df_sensitivity is not None:
        print("\n" + "=" * 60)
        print("SENSITIVITY CHECK: Union merge (either extractor)")
        print("=" * 60)
        df_sens_norm, conv_summary_sens = add_normalized_counts(
            df_sensitivity, lengths_df
        )
        core_results_union = run_core_analyses(df_sens_norm, conv_summary_sens)

    # --- Breadth analysis ---
    breadth_df = run_breadth_analysis(df_primary)

    # --- Visualizations ---
    plot_value_distributions(df_primary)
    plot_heatmap(df_primary)
    plot_cosine_similarity_matrix(df_primary)

    # --- Save tables ---
    save_summary_tables(core_results, breadth_df)

    print("\nAnalysis complete. Results in outputs/figures/ and outputs/tables/")


if __name__ == "__main__":
    main()
