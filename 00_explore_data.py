"""
Initial data exploration: verify dataset access and examine structure.

This script loads the three key datasets from HuggingFace and prints
summary statistics so we can verify everything is accessible before
running the full pipeline. It also saves a small sample of each dataset
for quick local inspection.

Datasets examined:
1. Anthropic "Values in the Wild" taxonomy (3,307 values in a hierarchy)
2. LMSYS-Chat-1M (real user conversations with LLMs)
3. Geodesic alignment evaluation data

Run this first to confirm your HuggingFace access works and to understand
what the raw data looks like.
"""

import json
from pathlib import Path
from datasets import load_dataset
import pandas as pd

OUTPUT_DIR = Path("outputs/data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def explore_values_taxonomy():
    """
    Load and examine the Anthropic values taxonomy.

    The taxonomy organizes 3,307 individual values into a 4-level hierarchy:
      Level 0: individual values -> Level 1: clusters -> Level 2: subcategories -> Level 3: top categories

    We primarily use Level 2 (26 subcategories) and Level 3 (5 top categories).
    """
    print("=" * 60)
    print("ANTHROPIC VALUES IN THE WILD - TAXONOMY")
    print("=" * 60)

    # Load the taxonomy tree
    tree = load_dataset("Anthropic/values-in-the-wild", "values_tree", split="train")
    tree_df = tree.to_pandas()

    print(f"\nTotal nodes in taxonomy: {len(tree_df)}")
    print(f"\nColumns: {list(tree_df.columns)}")
    print(f"\nNodes per level:")
    print(tree_df["level"].value_counts().sort_index())

    # Show the top-level categories (Level 3)
    level3 = tree_df[tree_df["level"] == 3]
    print(f"\nLevel 3 (top-level categories):")
    for _, row in level3.iterrows():
        print(f"  - {row['name']} (cluster_id: {row['cluster_id']})")

    # Show level-2 subcategories
    level2 = tree_df[tree_df["level"] == 2]
    print(f"\nLevel 2 subcategories ({len(level2)} total):")
    for _, row in level2.iterrows():
        parent = level3[level3["cluster_id"] == row["parent_cluster_id"]]
        parent_name = parent.iloc[0]["name"] if not parent.empty else "Unknown"
        print(f"  - {row['name']} (parent: {parent_name})")

    # Load frequency data
    freq = load_dataset("Anthropic/values-in-the-wild", "values_frequencies", split="train")
    freq_df = freq.to_pandas()
    print(f"\nValue frequencies dataset: {len(freq_df)} entries")
    print(f"Columns: {list(freq_df.columns)}")
    print(f"\nTop 10 most frequent values:")
    top10 = freq_df.nlargest(10, "pct_convos")
    for _, row in top10.iterrows():
        print(f"  {row['value']}: {row['pct_convos']:.4f}")

    # Save samples
    tree_df.to_csv(OUTPUT_DIR / "taxonomy_tree_sample.csv", index=False)
    freq_df.head(50).to_csv(OUTPUT_DIR / "value_frequencies_sample.csv", index=False)

    return tree_df, freq_df


def explore_lmsys():
    """
    Load and examine the LMSYS-Chat-1M dataset.

    This is ~1M real conversations between users and various LLMs.
    We need English first-turn user messages as prompts for our study.

    Note: this dataset is large and requires agreeing to terms on HuggingFace.
    We only load a small streaming sample here for exploration.
    """
    print("\n" + "=" * 60)
    print("LMSYS-CHAT-1M DATASET")
    print("=" * 60)

    # Stream a small sample to avoid downloading the full dataset
    try:
        ds = load_dataset("lmsys/lmsys-chat-1m", split="train", streaming=True)

        samples = []
        for i, example in enumerate(ds):
            if i >= 100:
                break
            samples.append(example)

        print(f"\nSampled {len(samples)} conversations")
        print(f"Columns: {list(samples[0].keys())}")

        # Examine language distribution in sample
        languages = [s.get("language", "unknown") for s in samples]
        lang_counts = pd.Series(languages).value_counts()
        print(f"\nLanguage distribution (sample of {len(samples)}):")
        print(lang_counts.head(10))

        # Examine conversation structure
        sample = samples[0]
        print(f"\nExample conversation structure:")
        print(f"  conversation type: {type(sample['conversation'])}")
        if isinstance(sample["conversation"], list) and len(sample["conversation"]) > 0:
            print(f"  first turn keys: {list(sample['conversation'][0].keys()) if isinstance(sample['conversation'][0], dict) else 'N/A'}")
            print(f"  number of turns: {len(sample['conversation'])}")
            # Show first user turn
            first_turn = sample["conversation"][0]
            content = first_turn.get("content", str(first_turn))
            print(f"  first turn preview: {content[:200]}...")

        # Extract English first turns
        english_first_turns = []
        for s in samples:
            if s.get("language", "") == "English":
                conv = s["conversation"]
                if isinstance(conv, list) and len(conv) > 0:
                    first = conv[0]
                    content = first.get("content", "") if isinstance(first, dict) else str(first)
                    english_first_turns.append({
                        "conversation_id": s.get("conversation_id", ""),
                        "prompt": content,
                        "num_turns": len(conv),
                    })

        print(f"\nEnglish first turns in sample: {len(english_first_turns)}")
        if english_first_turns:
            print(f"Example prompt: {english_first_turns[0]['prompt'][:200]}...")

            # Save sample
            pd.DataFrame(english_first_turns[:20]).to_csv(
                OUTPUT_DIR / "lmsys_sample.csv", index=False
            )

    except Exception as e:
        print(f"\nError loading LMSYS dataset: {e}")
        print("This dataset may require accepting terms at https://huggingface.co/datasets/lmsys/lmsys-chat-1m")


def explore_geodesic_evals():
    """
    Load and examine Geodesic's alignment evaluation dataset.

    This contains the binary safety/alignment evaluation scenarios that
    Geodesic used to test their models. We'll use this for the bridging
    analysis (Phase 5g) to correlate value profiles with alignment scores.
    """
    print("\n" + "=" * 60)
    print("GEODESIC ALIGNMENT EVALUATION DATA")
    print("=" * 60)

    try:
        ds = load_dataset("geodesic-research/sfm-alignment-labeling-v3", split="train")
        df = ds.to_pandas()

        print(f"\nDataset size: {len(df)} examples")
        print(f"Columns: {list(df.columns)}")
        print(f"\nFirst few rows:")
        print(df.head())

        # Save sample
        df.head(20).to_csv(OUTPUT_DIR / "geodesic_evals_sample.csv", index=False)

        return df

    except Exception as e:
        print(f"\nError loading Geodesic eval data: {e}")
        print("Trying alternative dataset names...")

        # Try listing available datasets from geodesic-research
        try:
            from huggingface_hub import list_datasets
            geo_datasets = [d.id for d in list_datasets(author="geodesic-research")]
            print(f"\nAvailable geodesic-research datasets:")
            for name in geo_datasets:
                print(f"  - {name}")
        except Exception as e2:
            print(f"Could not list datasets: {e2}")


def explore_geodesic_models():
    """
    List available Geodesic model variants on HuggingFace.

    We need to identify the exact model IDs for the 8 variants we'll use:
    4 pretraining conditions x 2 training stages (base + post-trained DPO).
    """
    print("\n" + "=" * 60)
    print("GEODESIC MODEL VARIANTS")
    print("=" * 60)

    try:
        from huggingface_hub import list_models
        models = [m for m in list_models(author="geodesic-research")]

        print(f"\nFound {len(models)} models from geodesic-research:")
        for m in sorted(models, key=lambda x: x.id):
            print(f"  - {m.id}")

        # Save model list
        model_list = [{"model_id": m.id, "tags": getattr(m, 'tags', [])} for m in models]
        with open(OUTPUT_DIR / "geodesic_models.json", "w") as f:
            json.dump(model_list, f, indent=2, default=str)

    except Exception as e:
        print(f"\nError listing models: {e}")


if __name__ == "__main__":
    print("Alignment Pretraining Values - Data Exploration")
    print("================================================\n")

    # Explore each dataset
    tree_df, freq_df = explore_values_taxonomy()
    explore_lmsys()
    explore_geodesic_evals()
    explore_geodesic_models()

    print("\n" + "=" * 60)
    print("EXPLORATION COMPLETE")
    print("=" * 60)
    print(f"\nSample files saved to {OUTPUT_DIR}/")
    print("Review these files to verify data structure before proceeding.")
