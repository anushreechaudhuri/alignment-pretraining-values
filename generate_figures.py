import json
import os
import warnings
from pathlib import Path
from collections import defaultdict, Counter
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
import seaborn as sns
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore", category=FutureWarning)

BASE_DIR = Path(__file__).resolve().parent
EXTRACTION_DIR = BASE_DIR / "data" / "extractions" / "gpt-5.2"
CONVERSATION_DIR = BASE_DIR / "data" / "conversations"
OUTPUT_DIR = BASE_DIR / "paper" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VARIANTS = [
    "unfiltered_base", "filtered_base", "misalignment_base", "alignment_base",
    "unfiltered_dpo", "filtered_dpo", "misalignment_dpo", "alignment_dpo",
]

DISPLAY_NAMES = {
    "unfiltered_base": "Unfiltered\n(Base)",
    "filtered_base": "Filtered\n(Base)",
    "misalignment_base": "Misaligned\n(Base)",
    "alignment_base": "Aligned\n(Base)",
    "unfiltered_dpo": "Unfiltered\n(DPO)",
    "filtered_dpo": "Filtered\n(DPO)",
    "misalignment_dpo": "Misaligned\n(DPO)",
    "alignment_dpo": "Aligned\n(DPO)",
}

SHORT_NAMES = {
    "unfiltered_base": "Unfilt. Base",
    "filtered_base": "Filt. Base",
    "misalignment_base": "Misal. Base",
    "alignment_base": "Align. Base",
    "unfiltered_dpo": "Unfilt. DPO",
    "filtered_dpo": "Filt. DPO",
    "misalignment_dpo": "Misal. DPO",
    "alignment_dpo": "Align. DPO",
}

LEVEL3_ORDER = [
    "Epistemic values",
    "Personal values",
    "Practical values",
    "Protective values",
    "Social values",
]

LEVEL3_COLORS = {
    "Epistemic values": "#4477AA",
    "Personal values": "#EE6677",
    "Practical values": "#228833",
    "Protective values": "#CCBB44",
    "Social values": "#AA3377",
}

PALETTE = sns.color_palette("colorblind", 10)

SINGLE_COL = 3.25
FULL_WIDTH = 6.75
FONT_SIZE = 8.5


def configure_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": FONT_SIZE,
        "axes.titlesize": 9,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def load_extractions():
    records = []
    for variant in VARIANTS:
        fpath = EXTRACTION_DIR / f"{variant}_values.jsonl"
        if not fpath.exists():
            print(f"  [WARN] Missing extraction file: {fpath.name}, skipping.")
            continue
        with open(fpath) as f:
            for line in f:
                row = json.loads(line)
                prompt_id = row["prompt_id"]
                model_variant = row["model_variant"]
                model_stage = row["model_stage"]
                topic_category = row["topic_category"]
                for val in row["extracted_values"]:
                    records.append({
                        "prompt_id": prompt_id,
                        "model_variant": model_variant,
                        "model_stage": model_stage,
                        "topic_category": topic_category,
                        "level2": val["taxonomy_level2_category"],
                        "level3": val["taxonomy_level3_category"],
                    })
    return pd.DataFrame(records)


def load_extraction_rows():
    rows = []
    for variant in VARIANTS:
        fpath = EXTRACTION_DIR / f"{variant}_values.jsonl"
        if not fpath.exists():
            continue
        with open(fpath) as f:
            for line in f:
                row = json.loads(line)
                rows.append({
                    "prompt_id": row["prompt_id"],
                    "model_variant": row["model_variant"],
                    "model_stage": row["model_stage"],
                    "topic_category": row["topic_category"],
                    "n_values": len(row["extracted_values"]),
                })
    return pd.DataFrame(rows)


def load_conversations():
    rows = []
    for variant in VARIANTS:
        fpath = CONVERSATION_DIR / f"{variant}.jsonl"
        if not fpath.exists():
            print(f"  [WARN] Missing conversation file: {fpath.name}, skipping.")
            continue
        with open(fpath) as f:
            for line in f:
                row = json.loads(line)
                rows.append({
                    "prompt_id": row["prompt_id"],
                    "model_variant": row["model_variant"],
                    "model_stage": row["model_stage"],
                    "topic_category": row["topic_category"],
                    "response_length": len(row["model_response"]),
                })
    return pd.DataFrame(rows)


def save_fig(fig, name):
    fig.savefig(OUTPUT_DIR / f"{name}.pdf", format="pdf")
    fig.savefig(OUTPUT_DIR / f"{name}.png", format="png", dpi=300)
    plt.close(fig)
    print(f"  Saved {name}.pdf and {name}.png")


def wilson_ci(count, total, z=1.96):
    if total == 0:
        return 0.0, 0.0, 0.0
    p = count / total
    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    halfwidth = (z / denom) * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2))
    lo = max(0, center - halfwidth)
    hi = min(1, center + halfwidth)
    return p, lo, hi


def cramers_v(x, y):
    ct = pd.crosstab(x, y)
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return 0.0
    chi2 = stats.chi2_contingency(ct)[0]
    n = ct.sum().sum()
    k = min(ct.shape) - 1
    if k == 0 or n == 0:
        return 0.0
    return np.sqrt(chi2 / (n * k))


def figure1_stacked_bar(df):
    available = [v for v in VARIANTS if v in df["model_variant"].unique()]
    if len(available) == 0:
        print("  [SKIP] Figure 1: no data available.")
        return

    counts = df.groupby(["model_variant", "level3"]).size().unstack(fill_value=0)
    props = counts.div(counts.sum(axis=1), axis=0)
    props = props.reindex(index=available, columns=LEVEL3_ORDER, fill_value=0)

    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 2.8))
    x = np.arange(len(available))
    bottom = np.zeros(len(available))

    for cat in LEVEL3_ORDER:
        vals = props[cat].values
        ax.bar(x, vals, bottom=bottom, width=0.65,
               color=LEVEL3_COLORS[cat], label=cat, edgecolor="white", linewidth=0.4)
        bottom += vals

    separator_x = 3.5
    ax.axvline(separator_x, color="#666666", linestyle="--", linewidth=0.8, zorder=0)
    ax.text(separator_x - 0.1, 1.03, "Base", ha="right", va="bottom",
            fontsize=7, color="#666666", transform=ax.get_xaxis_transform())
    ax.text(separator_x + 0.1, 1.03, "DPO", ha="left", va="bottom",
            fontsize=7, color="#666666", transform=ax.get_xaxis_transform())

    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY_NAMES[v].replace("\n", "\n") for v in available],
                       fontsize=7, ha="center")
    ax.set_ylabel("Proportion of extracted values")
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=5, frameon=False, fontsize=7, columnspacing=1.0)

    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    fig.tight_layout()
    save_fig(fig, "fig1_value_category_distribution")


def figure2_cosine_heatmap(df):
    available = [v for v in VARIANTS if v in df["model_variant"].unique()]
    if len(available) < 2:
        print("  [SKIP] Figure 2: insufficient data.")
        return

    all_l2 = sorted(df["level2"].unique())
    mat = np.zeros((len(available), len(all_l2)))
    for i, var in enumerate(available):
        sub = df[df["model_variant"] == var]
        total = len(sub)
        if total == 0:
            continue
        for j, cat in enumerate(all_l2):
            mat[i, j] = (sub["level2"] == cat).sum() / total

    sim = cosine_similarity(mat)
    labels = [SHORT_NAMES[v] for v in available]

    vmin = max(0.85, sim.min() - 0.01)
    center = 0.95

    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL))
    mask = np.zeros_like(sim, dtype=bool)

    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(sim, ax=ax, annot=True, fmt=".3f", cmap=cmap,
                vmin=vmin, vmax=1.0, center=center,
                xticklabels=labels, yticklabels=labels,
                linewidths=0.5, linecolor="white",
                annot_kws={"size": 6},
                cbar_kws={"shrink": 0.8, "label": "Cosine similarity"})
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=6.5)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=6.5)
    fig.tight_layout()
    save_fig(fig, "fig2_cosine_similarity")


def figure3_level2_heatmap(df):
    available = [v for v in VARIANTS if v in df["model_variant"].unique()]
    if len(available) == 0:
        print("  [SKIP] Figure 3: no data available.")
        return

    l2_to_l3 = df.groupby("level2")["level3"].agg(lambda s: s.mode().iloc[0]).to_dict()

    l2_by_l3 = defaultdict(list)
    for l2, l3 in sorted(l2_to_l3.items()):
        l2_by_l3[l3].append(l2)

    ordered_l2 = []
    for l3 in LEVEL3_ORDER:
        ordered_l2.extend(sorted(l2_by_l3.get(l3, [])))

    counts = df.groupby(["model_variant", "level2"]).size().unstack(fill_value=0)
    props = counts.div(counts.sum(axis=1), axis=0)
    props = props.reindex(index=available, columns=ordered_l2, fill_value=0)

    fig_height = 0.35 * len(available) + 1.2
    fig, axes = plt.subplots(2, 1, figsize=(FULL_WIDTH, fig_height),
                              gridspec_kw={"height_ratios": [0.06, 1]},
                              sharex=True)

    ax_bar = axes[0]
    ax_heat = axes[1]

    col_colors = [LEVEL3_COLORS[l2_to_l3[c]] for c in ordered_l2]
    for i, color in enumerate(col_colors):
        ax_bar.add_patch(plt.Rectangle((i, 0), 1, 1, color=color))
    ax_bar.set_xlim(0, len(ordered_l2))
    ax_bar.set_ylim(0, 1)
    ax_bar.set_yticks([])
    ax_bar.set_xticks([])
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.spines["left"].set_visible(False)
    ax_bar.spines["bottom"].set_visible(False)

    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=LEVEL3_COLORS[l3])
                      for l3 in LEVEL3_ORDER]
    ax_bar.legend(legend_handles, LEVEL3_ORDER, loc="upper center",
                  bbox_to_anchor=(0.5, 2.8), ncol=5, frameon=False, fontsize=6.5)

    row_labels = [SHORT_NAMES[v] for v in available]

    sns.heatmap(props.values, ax=ax_heat, annot=True, fmt=".2f",
                cmap="YlOrRd", vmin=0, vmax=props.values.max(),
                xticklabels=[c.replace(" and ", "\n& ").replace(" ", "\n", 1)
                             if len(c) > 18 else c for c in ordered_l2],
                yticklabels=row_labels,
                linewidths=0.3, linecolor="white",
                annot_kws={"size": 5},
                cbar_kws={"shrink": 0.6, "label": "Proportion", "pad": 0.02})

    ax_heat.set_xticklabels(ax_heat.get_xticklabels(), rotation=60, ha="right", fontsize=5.5)
    ax_heat.set_yticklabels(ax_heat.get_yticklabels(), rotation=0, fontsize=7)

    fig.subplots_adjust(hspace=0.02)
    fig.tight_layout()
    save_fig(fig, "fig3_level2_heatmap")


def figure4_expression_rate(extraction_rows):
    available = [v for v in VARIANTS if v in extraction_rows["model_variant"].unique()]
    if len(available) == 0:
        print("  [SKIP] Figure 4: no data available.")
        return

    base_variants = [v for v in available if v.endswith("_base")]
    dpo_variants = [v for v in available if v.endswith("_dpo")]
    conditions = sorted(set(v.replace("_base", "").replace("_dpo", "") for v in available))

    rates = {}
    cis = {}
    for var in available:
        sub = extraction_rows[extraction_rows["model_variant"] == var]
        total = len(sub)
        has_value = (sub["n_values"] > 0).sum()
        p, lo, hi = wilson_ci(has_value, total)
        rates[var] = p
        cis[var] = (p - lo, hi - p)

    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.6))
    x = np.arange(len(conditions))
    bar_w = 0.35

    base_rates = [rates.get(f"{c}_base", 0) for c in conditions]
    base_errs = np.array([
        [cis.get(f"{c}_base", (0, 0))[0] for c in conditions],
        [cis.get(f"{c}_base", (0, 0))[1] for c in conditions],
    ])
    dpo_rates = [rates.get(f"{c}_dpo", 0) for c in conditions]
    dpo_errs = np.array([
        [cis.get(f"{c}_dpo", (0, 0))[0] for c in conditions],
        [cis.get(f"{c}_dpo", (0, 0))[1] for c in conditions],
    ])

    base_mask = [f"{c}_base" in available for c in conditions]
    dpo_mask = [f"{c}_dpo" in available for c in conditions]

    ax.bar(x[base_mask] - bar_w / 2,
           [r for r, m in zip(base_rates, base_mask) if m],
           bar_w, label="Base",
           color=PALETTE[0], edgecolor="white", linewidth=0.4,
           yerr=base_errs[:, base_mask], capsize=2, error_kw={"linewidth": 0.8})

    ax.bar(x[dpo_mask] + bar_w / 2,
           [r for r, m in zip(dpo_rates, dpo_mask) if m],
           bar_w, label="DPO",
           color=PALETTE[1], edgecolor="white", linewidth=0.4,
           yerr=dpo_errs[:, dpo_mask], capsize=2, error_kw={"linewidth": 0.8})

    display = {"unfiltered": "Unfiltered", "filtered": "Filtered",
               "misalignment": "Misaligned", "alignment": "Aligned"}
    ax.set_xticks(x)
    ax.set_xticklabels([display.get(c, c) for c in conditions], fontsize=7.5)
    ax.set_ylabel("Proportion with ≥1 value")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.legend(frameon=False, fontsize=7)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    fig.tight_layout()
    save_fig(fig, "fig4_value_expression_rate")


def figure5_length_vs_values(extraction_rows, conversations):
    merged = extraction_rows.merge(conversations[["prompt_id", "model_variant", "response_length"]],
                                    on=["prompt_id", "model_variant"], how="inner")
    if len(merged) == 0:
        print("  [SKIP] Figure 5: no matched data.")
        return

    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.8))

    stage_map = {"base": "Base", "post-trained": "DPO"}
    for stage, color, marker in [("base", PALETTE[0], "o"), ("post-trained", PALETTE[1], "s")]:
        sub = merged[merged["model_stage"] == stage]
        if len(sub) == 0:
            continue
        sample = sub.sample(n=min(2000, len(sub)), random_state=42)
        ax.scatter(sample["response_length"], sample["n_values"],
                   c=[color], alpha=0.15, s=6, marker=marker, linewidths=0, rasterized=True)

        r, p = stats.pearsonr(sub["response_length"], sub["n_values"])
        slope, intercept = np.polyfit(sub["response_length"], sub["n_values"], 1)
        x_range = np.linspace(sub["response_length"].min(), sub["response_length"].quantile(0.99), 100)
        ax.plot(x_range, slope * x_range + intercept, color=color, linewidth=1.2,
                label=f"{stage_map[stage]} (r={r:.2f})")

    ax.set_xlabel("Response length (characters)")
    ax.set_ylabel("Number of extracted values")
    ax.legend(frameon=False, fontsize=7)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    fig.tight_layout()
    save_fig(fig, "fig5_length_vs_values")


def figure6_breadth_analysis(df):
    v1 = "alignment_dpo"
    v2 = "unfiltered_dpo"
    available = df["model_variant"].unique()
    if v1 not in available or v2 not in available:
        print("  [SKIP] Figure 6: required variants not available.")
        return

    sub = df[df["model_variant"].isin([v1, v2])]
    topics = sorted(sub["topic_category"].unique())

    results = []
    for topic in topics:
        t_sub = sub[sub["topic_category"] == topic]
        if len(t_sub) < 10:
            continue
        v = cramers_v(t_sub["model_variant"], t_sub["level2"])
        results.append({"topic": topic, "cramers_v": v})

    if len(results) == 0:
        print("  [SKIP] Figure 6: insufficient topic data.")
        return

    res_df = pd.DataFrame(results).sort_values("cramers_v", ascending=True)

    topic_display = {
        "coding": "Coding",
        "creative_writing": "Creative Writing",
        "education": "Education",
        "ethical_dilemmas": "Ethical Dilemmas",
        "general_knowledge": "General Knowledge",
        "other": "Other",
        "personal_relationships": "Personal Relationships",
        "professional_advice": "Professional Advice",
    }

    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.6))
    y = np.arange(len(res_df))
    ax.barh(y, res_df["cramers_v"].values, height=0.6,
            color=PALETTE[2], edgecolor="white", linewidth=0.4)

    ax.axvline(0.1, color="#888888", linestyle="--", linewidth=0.8, zorder=0)
    ax.text(0.1, len(res_df) - 0.3, " V=0.1", fontsize=6, color="#888888", va="top")

    ax.set_yticks(y)
    ax.set_yticklabels([topic_display.get(t, t) for t in res_df["topic"].values], fontsize=7)
    ax.set_xlabel("Cramér's V (Aligned DPO vs. Unfiltered DPO)")
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    fig.tight_layout()
    save_fig(fig, "fig6_breadth_analysis")


def main():
    configure_style()
    print("Loading data...")
    df = load_extractions()
    extraction_rows = load_extraction_rows()
    conversations = load_conversations()

    if len(df) == 0:
        print("No extraction data found. Exiting.")
        return

    print(f"Loaded {len(df):,} value instances from {df['model_variant'].nunique()} variants.")
    print(f"Loaded {len(extraction_rows):,} extraction rows.")
    print(f"Loaded {len(conversations):,} conversation rows.")

    print("\nGenerating Figure 1: Value category distributions...")
    figure1_stacked_bar(df)

    print("Generating Figure 2: Cosine similarity heatmap...")
    figure2_cosine_heatmap(df)

    print("Generating Figure 3: Level-2 category heatmap...")
    figure3_level2_heatmap(df)

    print("Generating Figure 4: Value expression rate...")
    figure4_expression_rate(extraction_rows)

    print("Generating Figure 5: Response length vs. value count...")
    figure5_length_vs_values(extraction_rows, conversations)

    print("Generating Figure 6: Breadth analysis...")
    figure6_breadth_analysis(df)

    print(f"\nAll figures saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
