"""
Taxonomy utilities for the Anthropic "Values in the Wild" value system.

The Anthropic taxonomy organizes 3,307 individual values into a hierarchy:
  Level 0: 3,307 individual values (e.g., "epistemic humility")
  Level 1: 266 clusters of related values
  Level 2: 26 subcategories (our primary analysis level)
  Level 3: 5 top-level categories: Practical, Epistemic, Social, Protective, Personal

This module loads the taxonomy from HuggingFace and provides lookup functions
so other parts of the pipeline can map extracted values to their categories.
"""

import pandas as pd
from datasets import load_dataset
from functools import lru_cache


@lru_cache(maxsize=1)
def load_taxonomy():
    """
    Load the full values taxonomy tree from HuggingFace.

    Returns a pandas DataFrame with one row per node in the taxonomy tree.
    Each row includes: cluster_id, parent_cluster_id, level (0-3),
    name/description, and pct_total_occurrences.

    Results are cached so repeated calls don't re-download.
    """
    ds = load_dataset("Anthropic/values-in-the-wild", "values_tree", split="train")
    return ds.to_pandas()


@lru_cache(maxsize=1)
def load_value_frequencies():
    """
    Load the value frequency data from the original Anthropic study.

    This provides baseline frequencies for how often each value appeared
    in real Claude conversations, which we can compare against the
    Geodesic model outputs.
    """
    ds = load_dataset("Anthropic/values-in-the-wild", "values_frequencies", split="train")
    return ds.to_pandas()


def get_level_nodes(level):
    """
    Get all taxonomy nodes at a specific hierarchical level.

    Args:
        level: Integer 0-3. Level 2 (26 subcategories) is our primary analysis level.
               Level 3 gives the 5 top-level categories.

    Returns:
        DataFrame filtered to nodes at the specified level.
    """
    df = load_taxonomy()
    return df[df["level"] == level].copy()


def get_level2_categories():
    """
    Get the 26 level-2 subcategories used as our primary analysis unit.

    Returns a list of (cluster_id, name) tuples for the 26 subcategories.
    These are the categories we ask Claude to classify extracted values into.
    """
    nodes = get_level_nodes(2)
    return list(zip(nodes["cluster_id"], nodes["name"]))


def get_level3_categories():
    """
    Get the 5 top-level categories: Practical, Epistemic, Social, Protective, Personal.
    """
    nodes = get_level_nodes(3)
    return list(zip(nodes["cluster_id"], nodes["name"]))


def map_level2_to_level3(cluster_id):
    """
    Given a level-2 subcategory cluster_id, return its parent level-3 category.

    This lets us roll up our 26-category analysis to the 5-category summary view.

    Args:
        cluster_id: The cluster_id of a level-2 node.

    Returns:
        Tuple of (parent_cluster_id, parent_name) for the level-3 category,
        or (None, None) if not found.
    """
    df = load_taxonomy()
    node = df[df["cluster_id"] == cluster_id]
    if node.empty:
        return None, None
    parent_id = node.iloc[0]["parent_cluster_id"]
    parent = df[df["cluster_id"] == parent_id]
    if parent.empty:
        return None, None
    return parent.iloc[0]["cluster_id"], parent.iloc[0]["name"]


def build_category_lookup():
    """
    Build a dictionary mapping level-2 category names to their level-3 parent names.

    Returns:
        Dict like {"Epistemic Rigor": "Epistemic", "User Safety": "Protective", ...}
    """
    df = load_taxonomy()
    level2 = df[df["level"] == 2]
    level3 = df[df["level"] == 3]

    lookup = {}
    for _, row in level2.iterrows():
        parent = level3[level3["cluster_id"] == row["parent_cluster_id"]]
        if not parent.empty:
            lookup[row["name"]] = parent.iloc[0]["name"]
    return lookup
