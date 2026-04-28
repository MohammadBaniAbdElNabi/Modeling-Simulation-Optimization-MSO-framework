"""CI computation, paired t-tests, and Cohen's d for policy comparisons."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


PRIMARY_METRICS = ["FR", "ADT", "P95", "ERR", "AWT"]
POLICY_NAMES = ["random", "greedy", "lp"]
COMPARISONS = [("lp", "greedy"), ("lp", "random"), ("greedy", "random")]


def compute_summary(
    results: dict[str, pd.DataFrame],
    metrics: list[str] | None = None,
    alpha: float = 0.05,
    n_reps: int = 20,
) -> pd.DataFrame:
    """Compute mean, std, CI per policy per metric.

    Returns DataFrame with columns:
    [policy, metric, mean, std, ci_lower, ci_upper]
    """
    if metrics is None:
        metrics = PRIMARY_METRICS

    t_crit = stats.t.ppf(1 - alpha / 2, df=n_reps - 1)
    rows = []
    for policy, df in results.items():
        for m in metrics:
            if m not in df.columns:
                continue
            sample = df[m].dropna().values
            if len(sample) == 0:
                continue
            mean_m = float(np.mean(sample))
            std_m = float(np.std(sample, ddof=1))
            se_m = std_m / np.sqrt(len(sample))
            ci_lower = mean_m - t_crit * se_m
            ci_upper = mean_m + t_crit * se_m
            rows.append({
                "policy": policy,
                "metric": m,
                "mean": mean_m,
                "std": std_m,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "n": len(sample),
            })

    return pd.DataFrame(rows)


def pairwise_tests(
    results: dict[str, pd.DataFrame],
    metrics: list[str] | None = None,
    comparisons: list[tuple[str, str]] | None = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Compute paired t-tests and Cohen's d for all policy pairs x metrics.

    Returns DataFrame with columns:
    [comparison, metric, t_stat, p_value, cohens_d, significant, effect_size]
    """
    if metrics is None:
        metrics = PRIMARY_METRICS
    if comparisons is None:
        comparisons = COMPARISONS

    rows = []
    for p1, p2 in comparisons:
        if p1 not in results or p2 not in results:
            continue
        df1, df2 = results[p1], results[p2]

        for m in metrics:
            if m not in df1.columns or m not in df2.columns:
                continue
            s1 = df1[m].dropna().values
            s2 = df2[m].dropna().values
            min_len = min(len(s1), len(s2))
            s1, s2 = s1[:min_len], s2[:min_len]
            if min_len < 2:
                continue

            t_stat, p_value = stats.ttest_rel(s1, s2)
            diff = s1 - s2
            d_std = float(np.std(diff, ddof=1))
            cohens_d = float(np.mean(diff) / d_std) if d_std > 0 else 0.0

            abs_d = abs(cohens_d)
            if abs_d < 0.2:
                effect = "negligible"
            elif abs_d < 0.5:
                effect = "small"
            elif abs_d < 0.8:
                effect = "medium"
            else:
                effect = "large"

            rows.append({
                "comparison": f"{p1}_vs_{p2}",
                "metric": m,
                "t_stat": float(t_stat),
                "p_value": float(p_value),
                "cohens_d": cohens_d,
                "significant": bool(p_value < alpha),
                "effect_size": effect,
            })

    return pd.DataFrame(rows)
