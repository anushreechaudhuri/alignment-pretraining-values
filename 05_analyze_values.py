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

Input: data/extractions/*_values.jsonl (from Phase 3)
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

from config import (
    DATA_DIR, FIGURES_DIR, TABLES_DIR, PROCESSED_DATA_DIR,
    TOP_LEVEL_CATEGORIES, SIGNIFICANCE_LEVEL,
)
from utils.taxonomy import (
    get_level2_categories, get_level3_categories, build_category_lookup,
)
from utils.stats import (
    chi_squared_test, cosine_similarity, proportion_differences_with_ci,
)


def load_all_extractions():
    """
    Load value extractions from all model variants into a single DataFrame.

    Returns a DataFrame where each row is one extracted value, with columns
    for the model variant, topic category, and taxonomy classification.
    """
    extractions_dir = DATA_DIR / "extractions"

    all_records = []
    for fpath in sorted(extractions_dir.glob("*_values.jsonl")):
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
                    })

    df = pd.DataFrame(all_records)
    print(f"Loaded {len(df)} extracted values from {df['model_variant'].nunique()} model variants")
    return df


def compute_value_distributions(df, group_col="model_variant", level="level2_category"):
    """
    Compute value category distributions for each model variant.

    Args:
        df: DataFrame of extracted values
        group_col: Column to group by (e.g., "model_variant")
        level: Taxonomy level to aggregate at ("level2_category" or "level3_category")

    Returns:
        Dict mapping group names to pandas Series of category counts
    """
    distributions = {}
    for name, group in df.groupby(group_col):
        distributions[name] = group[level].value_counts()
    return distributions


def run_pairwise_comparison(dist_a, dist_b, name_a, name_b):
    """
    Run the full suite of statistical tests comparing two model variants.

    Returns a summary dict with chi-squared results, cosine similarity,
    and per-category proportion differences.
    """
    chi2_result = chi_squared_test(dist_a, dist_b)
    cos_sim = cosine_similarity(dist_a, dist_b)
    prop_diffs = proportion_differences_with_ci(dist_a, dist_b)

    return {
        "comparison": f"{name_a} vs {name_b}",
        "chi2": chi2_result,
        "cosine_similarity": cos_sim,
        "proportion_differences": prop_diffs,
    }


def run_core_analyses(df):
    """
    Run analyses 5a-5c: core pairwise comparisons between post-trained models.

    These test whether different pretraining interventions produce different
    value profiles when models are used in ordinary conversations.
    """
    print("\n" + "="*60)
    print("CORE ANALYSES: Post-trained model comparisons")
    print("="*60)

    dists = compute_value_distributions(df[df["model_stage"] == "post-trained"])

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

        result = run_pairwise_comparison(dists[var_a], dists[var_b], var_a, var_b)
        result["label"] = label
        results.append(result)

        print(f"\n{label}")
        print(f"  Chi-squared: {result['chi2']['chi2']:.2f}, p={result['chi2']['p_value']:.4e}")
        print(f"  Cramer's V: {result['chi2']['cramers_v']:.4f}")
        print(f"  Cosine similarity: {result['cosine_similarity']:.4f}")

        # Show top differentially expressed categories
        top_diffs = result["proportion_differences"].head(5)
        print(f"  Top differences:")
        for _, row in top_diffs.iterrows():
            sig = "*" if row["significant"] else ""
            print(f"    {row['category']}: {row['diff']:+.4f} (p_adj={row['p_adjusted']:.4e}) {sig}")

    return results


def run_breadth_analysis(df):
    """
    Analysis 5e: Test whether alignment effects are consistent across topics.

    If differences appear across ALL topic categories, that's evidence for
    broad "deep character" effects. If only in safety-adjacent topics,
    that's evidence for narrow effects.
    """
    print("\n" + "="*60)
    print("BREADTH ANALYSIS: Effects by topic category")
    print("="*60)

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
            })

    breadth_df = pd.DataFrame(results_by_topic)
    print(breadth_df.to_string(index=False))

    return breadth_df


def plot_value_distributions(df):
    """
    Generate stacked bar chart of value category distributions across all models.
    """
    dists = compute_value_distributions(df, level="level3_category")

    # Normalize to proportions
    props = {}
    for name, counts in dists.items():
        total = counts.sum()
        props[name] = {cat: counts.get(cat, 0) / total for cat in TOP_LEVEL_CATEGORIES}

    props_df = pd.DataFrame(props).T

    fig, ax = plt.subplots(figsize=(12, 6))
    props_df.plot(kind="bar", stacked=True, ax=ax, colormap="Set2")
    ax.set_ylabel("Proportion of Extracted Values")
    ax.set_xlabel("Model Variant")
    ax.set_title("Value Category Distribution by Model Variant")
    ax.legend(title="Category", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    fig.savefig(FIGURES_DIR / "value_distributions_stacked.png", dpi=150, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "value_distributions_stacked.pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved stacked bar chart to {FIGURES_DIR}")


def plot_heatmap(df):
    """
    Generate heatmap of level-2 category proportions by model variant.
    """
    dists = compute_value_distributions(df, level="level2_category")

    # Build proportion matrix
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

    fig.savefig(FIGURES_DIR / "value_heatmap_level2.png", dpi=150, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "value_heatmap_level2.pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved heatmap to {FIGURES_DIR}")


def plot_cosine_similarity_matrix(df):
    """
    Generate cosine similarity matrix between all model variants' value distributions.
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
    ax.set_title("Cosine Similarity of Value Distributions Between Model Variants")
    plt.tight_layout()

    fig.savefig(FIGURES_DIR / "cosine_similarity_matrix.png", dpi=150, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "cosine_similarity_matrix.pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved cosine similarity matrix to {FIGURES_DIR}")


def save_summary_tables(core_results, breadth_df):
    """Save analysis results as CSV tables."""

    # Chi-squared comparison table
    chi2_rows = []
    for r in core_results:
        chi2_rows.append({
            "comparison": r["label"],
            "chi2_statistic": r["chi2"]["chi2"],
            "p_value": r["chi2"]["p_value"],
            "cramers_v": r["chi2"]["cramers_v"],
            "cosine_similarity": r["cosine_similarity"],
        })
    chi2_df = pd.DataFrame(chi2_rows)
    chi2_df.to_csv(TABLES_DIR / "chi_squared_comparisons.csv", index=False)

    # Breadth analysis table
    if breadth_df is not None and len(breadth_df) > 0:
        breadth_df.to_csv(TABLES_DIR / "breadth_analysis.csv", index=False)

    # Top differential values for main comparison
    for r in core_results:
        label = r["label"].split(":")[0].strip()
        r["proportion_differences"].to_csv(
            TABLES_DIR / f"proportion_diffs_{label}.csv", index=False
        )

    print(f"Saved summary tables to {TABLES_DIR}")


def main():
    """Run all analyses and generate outputs."""

    # Load data
    df = load_all_extractions()

    if len(df) == 0:
        print("No extraction data found. Run Phase 3 first.")
        return

    print(f"\nDataset summary:")
    print(f"  Total extracted values: {len(df)}")
    print(f"  Model variants: {df['model_variant'].nunique()}")
    print(f"  Unique prompts: {df['prompt_id'].nunique()}")
    print(f"  Topic categories: {df['topic_category'].nunique()}")

    # Run analyses
    core_results = run_core_analyses(df)
    breadth_df = run_breadth_analysis(df)

    # Generate visualizations
    plot_value_distributions(df)
    plot_heatmap(df)
    plot_cosine_similarity_matrix(df)

    # Save tables
    save_summary_tables(core_results, breadth_df)

    print("\nAnalysis complete. Results in outputs/figures/ and outputs/tables/")


if __name__ == "__main__":
    main()
