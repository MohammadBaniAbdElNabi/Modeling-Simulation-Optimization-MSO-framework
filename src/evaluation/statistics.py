"""Significance-focused statistics (R3): paired tests, effect sizes, MDE/power."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats


# Five core metrics — but the *headline* metrics for significance are the
# weighted/peak ones; aggregate FR is retained only for comparability.
PRIMARY_METRICS: list[str] = [
    "FR",              # aggregate (legacy)
    "FR_weighted",     # priority-weighted FR — emergency wins matter most
    "ERR",             # aggregate (legacy)
    "ERR_peak",        # ERR during peak hours (12–17) — saturation regime
    "Expiration_cost", # priority-weighted expiration count
    "ADT",
    "AWT",
    "P95",
]

# These three are the metrics on which we evaluate the headline tests.
HEADLINE_METRICS: list[str] = ["FR_weighted", "ERR_peak", "Expiration_cost"]

# R3 conditions
CONDITION_NAMES: list[str] = ["random", "greedy", "lp_static", "lp_mpc"]

# 4 PRIMARY pre-registered hypotheses (no Bonferroni — pre-registration is the control)
PRIMARY_PAIRS: list[tuple[str, str]] = [
    ("greedy",    "random"),    # H1: greedy > random  (sanity)
    ("lp_static", "greedy"),    # H2: planning > reactive
    ("lp_mpc",    "lp_static"), # H3: replanning > one-shot planning  (key contribution)
    ("lp_mpc",    "greedy"),    # H4: full pipeline > reactive
]

# Reference pairs (Holm-Bonferroni within group)
REFERENCE_PAIRS: list[tuple[str, str]] = [
    ("lp_mpc",    "random"),
    ("lp_static", "random"),
]


# ---------------------------------------------------------------------- helpers
def _effect_size_label(abs_d: float) -> str:
    if abs_d < 0.2: return "negligible"
    if abs_d < 0.5: return "small"
    if abs_d < 0.8: return "medium"
    return "large"


def _holm_bonferroni(p_values: list[float]) -> list[float]:
    """Return Holm-adjusted p-values (1:1 with input order)."""
    n     = len(p_values)
    order = sorted(range(n), key=lambda i: p_values[i])
    adj   = [0.0] * n
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj_p = (n - rank) * p_values[idx]
        running_max = max(running_max, min(1.0, adj_p))
        adj[idx] = running_max
    return adj


def mde_paired(
    n: int,
    sigma: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float:
    """Minimum detectable effect for a paired t-test.

    Formula:  MDE = (t_{1-alpha/2, n-1} + z_{power}) * sigma / sqrt(n)
    """
    if n < 2 or sigma <= 0:
        return float("nan")
    t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
    z_pow  = stats.norm.ppf(power)
    return float((t_crit + z_pow) * sigma / math.sqrt(n))


# ---------------------------------------------------------------------- summary
def compute_summary(
    results: dict[str, pd.DataFrame],
    metrics: list[str] | None = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Mean, std, and 95% CI per condition per metric."""
    if metrics is None:
        metrics = PRIMARY_METRICS

    rows = []
    for cond, df in results.items():
        for m in metrics:
            if m not in df.columns:
                continue
            sample = df[m].dropna().values
            n = len(sample)
            if n == 0:
                continue
            mean_m = float(np.mean(sample))
            std_m  = float(np.std(sample, ddof=1)) if n > 1 else 0.0
            t_crit = stats.t.ppf(1 - alpha / 2, df=max(1, n - 1))
            se     = std_m / math.sqrt(max(1, n))
            rows.append({
                "condition": cond, "metric": m,
                "mean":      mean_m, "std": std_m,
                "ci_lower":  mean_m - t_crit * se,
                "ci_upper":  mean_m + t_crit * se,
                "n":         n,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------- pairwise
def pairwise_tests(
    results: dict[str, pd.DataFrame],
    metrics: list[str] | None = None,
    comparisons: list[tuple[str, str]] | None = None,
    alpha: float = 0.05,
    apply_holm_to_reference: bool = True,
) -> pd.DataFrame:
    """Paired t-tests with Cohen's d, 95% paired CI, and MDE per pair × metric.

    Returns DataFrame with columns:
      [comparison, is_primary, metric, mean_a, mean_b, mean_diff,
       ci_lower, ci_upper, t_stat, p_value, p_adj, cohens_d, effect_size,
       mde_80, significant, significant_adj]
    """
    if metrics is None:
        metrics = PRIMARY_METRICS
    if comparisons is None:
        comparisons = PRIMARY_PAIRS + REFERENCE_PAIRS

    primary_set = set(PRIMARY_PAIRS)
    rows = []

    for cond_a, cond_b in comparisons:
        if cond_a not in results or cond_b not in results:
            continue
        df_a, df_b = results[cond_a], results[cond_b]
        is_primary = (cond_a, cond_b) in primary_set

        for m in metrics:
            if m not in df_a.columns or m not in df_b.columns:
                continue
            a = df_a[m].dropna().values
            b = df_b[m].dropna().values
            n = min(len(a), len(b))
            a, b = a[:n], b[:n]
            if n < 2:
                continue

            t_stat, p_value = stats.ttest_rel(a, b)
            diff   = a - b
            d_std  = float(np.std(diff, ddof=1))
            cohens_d = float(np.mean(diff) / d_std) if d_std > 0 else 0.0
            t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
            se_d   = d_std / math.sqrt(n)
            mde    = mde_paired(n, d_std, alpha=alpha, power=0.80)

            rows.append({
                "comparison":  f"{cond_a}_vs_{cond_b}",
                "cond_a":      cond_a,
                "cond_b":      cond_b,
                "is_primary":  is_primary,
                "metric":      m,
                "mean_a":      float(np.mean(a)),
                "mean_b":      float(np.mean(b)),
                "mean_diff":   float(np.mean(diff)),
                "ci_lower":    float(np.mean(diff) - t_crit * se_d),
                "ci_upper":    float(np.mean(diff) + t_crit * se_d),
                "t_stat":      float(t_stat),
                "p_value":     float(p_value),
                "cohens_d":    cohens_d,
                "effect_size": _effect_size_label(abs(cohens_d)),
                "mde_80":      mde,
                "significant": bool(p_value < alpha),
            })

    pw = pd.DataFrame(rows)
    if pw.empty:
        return pw

    # Holm-Bonferroni within reference group only; primary pairs are
    # pre-registered and therefore reported uncorrected.
    pw["p_adj"]           = pw["p_value"]
    pw["significant_adj"] = pw["significant"]
    if apply_holm_to_reference:
        ref_mask = ~pw["is_primary"]
        for metric in pw.loc[ref_mask, "metric"].unique():
            mask = ref_mask & (pw["metric"] == metric)
            if mask.sum() > 0:
                p_list  = pw.loc[mask, "p_value"].tolist()
                p_adj   = _holm_bonferroni(p_list)
                pw.loc[mask, "p_adj"]           = p_adj
                pw.loc[mask, "significant_adj"] = [p < alpha for p in p_adj]
    return pw


# ---------------------------------------------------------------------- demand sweep
def demand_curve_table(
    results_by_demand: dict[float, dict[str, pd.DataFrame]],
    metric: str,
) -> pd.DataFrame:
    """Reshape stress-sweep results into a tidy operating-curve frame.

    Returns long-format DataFrame: [demand_scale, condition, mean, ci_lower, ci_upper, n]
    """
    rows = []
    for ds, results in results_by_demand.items():
        summary = compute_summary(results, metrics=[metric])
        for _, r in summary.iterrows():
            rows.append({
                "demand_scale": ds,
                "condition":    r["condition"],
                "metric":       r["metric"],
                "mean":         r["mean"],
                "ci_lower":     r["ci_lower"],
                "ci_upper":     r["ci_upper"],
                "n":            r["n"],
            })
    return pd.DataFrame(rows)
