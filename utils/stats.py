"""
Statistical testing utilities for comparing value distributions across model variants.

This module implements the statistical methods described in the Anthropic
"Values in the Wild" paper, adapted for comparing Geodesic model variants:
- Chi-squared tests for overall distribution differences
- Bonferroni-corrected proportion tests for individual category comparisons
- Cosine similarity between value frequency vectors
- Effect size measures (Cramer's V)

All functions take simple pandas Series or numpy arrays as input,
making them easy to use in the analysis pipeline.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cosine
from statsmodels.stats.multitest import multipletests


def chi_squared_test(dist_a, dist_b):
    """
    Test whether two value distributions differ significantly.

    Uses a chi-squared test of homogeneity: given the value category counts
    from two model variants, tests whether they could have come from the same
    underlying distribution. A significant result means the models express
    values at different rates.

    Args:
        dist_a: pandas Series of value category counts for model A
                (index = category names, values = counts)
        dist_b: pandas Series of value category counts for model B

    Returns:
        dict with keys:
            chi2: the chi-squared test statistic
            p_value: probability of seeing this difference by chance
            dof: degrees of freedom
            cramers_v: effect size (0 = no effect, 1 = complete association)
    """
    # Align the two distributions to the same categories
    all_cats = sorted(set(dist_a.index) | set(dist_b.index))
    a = np.array([dist_a.get(c, 0) for c in all_cats])
    b = np.array([dist_b.get(c, 0) for c in all_cats])

    # Build contingency table: rows = models, columns = categories
    contingency = np.array([a, b])

    chi2, p_value, dof, _ = stats.chi2_contingency(contingency)

    # Cramer's V measures effect size for chi-squared tests
    # Values: ~0.1 = small, ~0.3 = medium, ~0.5 = large
    n = contingency.sum()
    k = min(contingency.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * k)) if n * k > 0 else 0.0

    return {
        "chi2": chi2,
        "p_value": p_value,
        "dof": dof,
        "cramers_v": cramers_v,
    }


def cosine_similarity(dist_a, dist_b):
    """
    Compute cosine similarity between two value distribution vectors.

    Treats each model's value distribution as a vector in category-space.
    Cosine similarity of 1.0 means identical proportions; lower values mean
    more different distributions. This is useful because it's insensitive to
    the total number of values extracted (unlike raw count comparisons).

    Args:
        dist_a, dist_b: pandas Series with matching indices (category names)
                        and numeric values (counts or proportions).

    Returns:
        float: cosine similarity between 0 and 1.
    """
    all_cats = sorted(set(dist_a.index) | set(dist_b.index))
    a = np.array([dist_a.get(c, 0) for c in all_cats], dtype=float)
    b = np.array([dist_b.get(c, 0) for c in all_cats], dtype=float)

    # scipy's cosine() returns distance (1 - similarity), so we invert
    if np.all(a == 0) or np.all(b == 0):
        return 0.0
    return 1.0 - cosine(a, b)


def proportion_differences_with_ci(dist_a, dist_b, alpha=0.05, correction="bonferroni"):
    """
    For each value category, compute the proportion difference between two models
    with confidence intervals and optional multiple-testing correction.

    This identifies WHICH specific categories differ between models,
    complementing the chi-squared test which only tells us IF they differ overall.

    Args:
        dist_a, dist_b: pandas Series of category counts
        alpha: significance level (default 0.05)
        correction: multiple testing correction method ("bonferroni" or None)

    Returns:
        DataFrame with columns: category, prop_a, prop_b, diff, ci_lower, ci_upper,
                                p_value, p_adjusted, significant
    """
    all_cats = sorted(set(dist_a.index) | set(dist_b.index))
    n_a = dist_a.sum()
    n_b = dist_b.sum()

    results = []
    for cat in all_cats:
        count_a = dist_a.get(cat, 0)
        count_b = dist_b.get(cat, 0)

        p_a = count_a / n_a if n_a > 0 else 0
        p_b = count_b / n_b if n_b > 0 else 0
        diff = p_a - p_b

        # Standard error for difference of proportions
        se = np.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b) if (n_a > 0 and n_b > 0) else 0
        z = stats.norm.ppf(1 - alpha / 2)

        # Two-proportion z-test
        p_pooled = (count_a + count_b) / (n_a + n_b) if (n_a + n_b) > 0 else 0
        se_test = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_a + 1/n_b)) if (n_a > 0 and n_b > 0 and p_pooled > 0) else 0
        z_stat = diff / se_test if se_test > 0 else 0
        p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))

        results.append({
            "category": cat,
            "prop_a": p_a,
            "prop_b": p_b,
            "diff": diff,
            "ci_lower": diff - z * se,
            "ci_upper": diff + z * se,
            "p_value": p_val,
        })

    df = pd.DataFrame(results)

    # Apply multiple testing correction
    if correction and len(df) > 0:
        reject, p_adj, _, _ = multipletests(df["p_value"], alpha=alpha, method=correction)
        df["p_adjusted"] = p_adj
        df["significant"] = reject
    else:
        df["p_adjusted"] = df["p_value"]
        df["significant"] = df["p_value"] < alpha

    return df.sort_values("p_adjusted")
