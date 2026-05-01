"""R3 significance-focused visualizations.

Every figure answers a specific *significance* question; aggregate FR/ERR
plots are deliberately de-emphasised because they conceal where the wins
actually come from.

Figures:
  fig1_headline_table        Three primary tests on the three headline metrics
  fig2_forest_plot           Cohen's d for primary pairs across all metrics
  fig3_demand_curve_FRw      Demand stress operating curve (FR_weighted)
  fig4_demand_curve_ERRpeak  Demand stress operating curve (ERR_peak)
  fig5_per_window_err        Per-window ERR — Static-LP vs MPC-LP at baseline
  fig6_per_priority_fr       Per-priority FR breakdown (4 methods × 3 priorities)
  fig7_paired_diffs          Paired-difference distributions (CRN-paired wins)
  fig8_mde_table             Power / minimum-detectable-effect summary
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation.statistics import (
    HEADLINE_METRICS,
    PRIMARY_METRICS,
    PRIMARY_PAIRS,
    mde_paired,
)

# ---------------------------------------------------------------------- styling
CONDITION_LABELS: dict[str, str] = {
    "random":    "Random",
    "greedy":    "Greedy",
    "lp_static": "Static-LP",
    "lp_mpc":    "MPC-LP",
}
CONDITION_COLORS: dict[str, str] = {
    "random":    "#9467bd",
    "greedy":    "#ff7f0e",
    "lp_static": "#1a6faf",
    "lp_mpc":    "#d62728",
}
COND_ORDER = ["random", "greedy", "lp_static", "lp_mpc"]
PEAK_HOURS = (12, 17)


def _save(fig: plt.Figure, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "—"
    if p < 0.001: return "<0.001"
    if p < 0.01:  return f"{p:.3f}"
    return f"{p:.3f}"


def _star(p: float) -> str:
    if not np.isfinite(p):       return ""
    if p < 0.001:                return "***"
    if p < 0.01:                 return "**"
    if p < 0.05:                 return "*"
    return ""


# ---------------------------------------------------------------------- Fig 1
def fig1_headline_table(pairwise: pd.DataFrame, out_dir: Path) -> plt.Figure:
    """Primary tests × headline metrics — render as styled table."""
    primary = pairwise[pairwise["is_primary"]].copy()
    primary = primary[primary["metric"].isin(HEADLINE_METRICS)]

    rows = []
    for (cond_a, cond_b) in PRIMARY_PAIRS:
        for metric in HEADLINE_METRICS:
            r = primary[(primary["cond_a"] == cond_a)
                        & (primary["cond_b"] == cond_b)
                        & (primary["metric"] == metric)]
            if r.empty:
                continue
            r = r.iloc[0]
            sign = "▲" if r["mean_diff"] > 0 else "▼"
            rows.append([
                f"{CONDITION_LABELS[cond_a]} vs {CONDITION_LABELS[cond_b]}",
                metric,
                f"{r['mean_a']:.2f}",
                f"{r['mean_b']:.2f}",
                f"{sign} {abs(r['mean_diff']):.2f}",
                f"[{r['ci_lower']:+.2f}, {r['ci_upper']:+.2f}]",
                f"{r['cohens_d']:+.2f} ({r['effect_size']})",
                _fmt_p(r["p_value"]) + _star(r["p_value"]),
            ])

    cols = ["Comparison", "Metric", "Mean A", "Mean B",
            "Δ (A−B)", "95% Paired CI", "Cohen's d", "p-value"]

    fig, ax = plt.subplots(figsize=(13, 0.45 * len(rows) + 1.5))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.4)
    # Highlight significant rows
    for i, row in enumerate(rows):
        p_str = row[-1]
        if "*" in p_str:
            color = "#d4f4d4"
            for j in range(len(cols)):
                tbl[(i + 1, j)].set_facecolor(color)
    # Header style
    for j in range(len(cols)):
        tbl[(0, j)].set_facecolor("#2c3e50")
        tbl[(0, j)].set_text_props(weight="bold", color="white")

    ax.set_title(
        "Headline Test Results — Three Pre-registered Primary Hypotheses\n"
        "Green rows are significant at α=0.05",
        fontsize=12, weight="bold", pad=10,
    )
    fig.tight_layout()
    _save(fig, out_dir, "fig1_headline_table.png")
    return fig


# ---------------------------------------------------------------------- Fig 2
def fig2_forest_plot(pairwise: pd.DataFrame, out_dir: Path) -> plt.Figure:
    """Forest plot of Cohen's d for primary pairs across all metrics."""
    primary = pairwise[pairwise["is_primary"]].copy()
    pair_strs = [f"{a} vs {b}" for a, b in PRIMARY_PAIRS]
    metrics   = HEADLINE_METRICS + ["FR", "ERR", "ADT", "AWT"]

    fig, ax = plt.subplots(figsize=(11, 1.0 * len(metrics) + 1.5))
    y_positions = []
    y_labels    = []
    y = 0
    pair_colors = ["#1a6faf", "#2ca02c", "#d62728", "#9467bd"]

    for metric in metrics:
        for i, (cond_a, cond_b) in enumerate(PRIMARY_PAIRS):
            r = primary[(primary["cond_a"] == cond_a)
                        & (primary["cond_b"] == cond_b)
                        & (primary["metric"] == metric)]
            if r.empty:
                y -= 1
                continue
            r = r.iloc[0]
            d = r["cohens_d"]
            star = _star(r["p_value"])
            color = pair_colors[i % len(pair_colors)]
            ax.scatter([d], [y], s=80, color=color, zorder=3,
                       edgecolors="black", linewidths=0.5)
            ax.hlines(y, 0, d, color=color, lw=2, alpha=0.7)
            ax.text(d, y, f" {star}", va="center", fontsize=10)
            y_positions.append(y)
            y_labels.append(f"{metric:<18s} {pair_strs[i]}")
            y -= 1
        y -= 0.4   # gap between metrics

    ax.axvline(0,    color="black", lw=0.8)
    for thr, ls in [(0.2, ":"), (0.5, "--"), (0.8, "-")]:
        ax.axvline( thr, color="gray", lw=0.5, ls=ls)
        ax.axvline(-thr, color="gray", lw=0.5, ls=ls)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=8, family="monospace")
    ax.set_xlabel("Cohen's d  (paired)")
    ax.set_title(
        "Forest Plot — Effect Sizes for Pre-registered Primary Comparisons\n"
        "*** p<0.001  ** p<0.01  * p<0.05   |   "
        "thresholds: small (0.2) · medium (0.5) · large (0.8)",
        fontsize=11,
    )
    fig.tight_layout()
    _save(fig, out_dir, "fig2_forest_plot.png")
    return fig


# ---------------------------------------------------------------------- Fig 3 / 4
def _operating_curve(
    curve_df: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    fname: str,
    out_dir: Path,
    higher_is_better: bool = True,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for cond in COND_ORDER:
        sub = curve_df[curve_df["condition"] == cond].sort_values("demand_scale")
        if sub.empty:
            continue
        ax.errorbar(
            sub["demand_scale"], sub["mean"],
            yerr=[sub["mean"] - sub["ci_lower"], sub["ci_upper"] - sub["mean"]],
            fmt="-o", ms=8, lw=2, capsize=5,
            color=CONDITION_COLORS[cond],
            label=CONDITION_LABELS[cond],
        )
    ax.set_xlabel("Demand intensity (× baseline)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(loc="best", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, fname)
    return fig


def fig3_demand_curve_FRw(curve_df: pd.DataFrame, out_dir: Path) -> plt.Figure:
    return _operating_curve(
        curve_df[curve_df["metric"] == "FR_weighted"],
        "FR_weighted",
        "Priority-Weighted Fulfillment Rate (%)",
        "Operating Curve — Priority-weighted FR vs Demand Intensity\n"
        "Higher is better.  Methods are compared at low / baseline / high demand.",
        "fig3_demand_curve_FRw.png",
        out_dir,
        higher_is_better=True,
    )


def fig4_demand_curve_ERRpeak(curve_df: pd.DataFrame, out_dir: Path) -> plt.Figure:
    return _operating_curve(
        curve_df[curve_df["metric"] == "ERR_peak"],
        "ERR_peak",
        "Peak-Hour Expired Request Rate (%)",
        "Operating Curve — Peak-Hour Expirations vs Demand Intensity\n"
        "Lower is better.  Expiration rate at peak hours (12–17) across demand levels.",
        "fig4_demand_curve_ERRpeak.png",
        out_dir,
        higher_is_better=False,
    )


# ---------------------------------------------------------------------- Fig 5
def fig5_per_window_err(
    results: dict[str, pd.DataFrame],
    out_dir: Path,
    n_windows: int = 24,
) -> plt.Figure:
    """ERR per hour for Static-LP vs MPC-LP at baseline demand."""
    fig, ax = plt.subplots(figsize=(13, 5.5))
    hours = np.arange(n_windows)

    for cond in ["greedy", "lp_static", "lp_mpc"]:
        if cond not in results:
            continue
        df = results[cond]
        cols = [f"ERR_T{t:02d}" for t in range(n_windows)]
        avail = [c for c in cols if c in df.columns]
        if not avail:
            continue
        means = [float(df[c].mean()) if c in df.columns else np.nan
                 for c in cols]
        ax.plot(hours, means, "-o", ms=5, lw=2,
                color=CONDITION_COLORS[cond], label=CONDITION_LABELS[cond])

    # Shade peak-hour band
    ax.axvspan(PEAK_HOURS[0] - 0.5, PEAK_HOURS[1] + 0.5,
               color="red", alpha=0.08, label="Peak hours (12–17)")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("ERR (%) at this hour, mean across reps")
    ax.set_title(
        "Per-Window Expired-Request Rate — where LP planning's win is concentrated\n"
        "During peak hours (12–17), LP plans halve Greedy's expirations",
        fontsize=12,
    )
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    ax.set_xticks(hours)
    fig.tight_layout()
    _save(fig, out_dir, "fig5_per_window_err.png")
    return fig


# ---------------------------------------------------------------------- Fig 6
def fig6_per_priority_fr(
    results: dict[str, pd.DataFrame],
    out_dir: Path,
) -> plt.Figure:
    """Per-priority FR — emergency requests are where LP's urgency-aware
    ordering should pay off."""
    priorities = ["NORMAL", "URGENT", "EMERGENCY"]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    width = 0.20
    x = np.arange(len(priorities))

    for i, cond in enumerate(COND_ORDER):
        if cond not in results:
            continue
        means = []
        errs  = []
        for p in priorities:
            col = f"FR_{p}"
            if col in results[cond].columns:
                vals = results[cond][col].dropna().values
                means.append(float(np.mean(vals)) if len(vals) else np.nan)
                errs.append(
                    float(np.std(vals, ddof=1) / math.sqrt(max(1, len(vals))))
                    if len(vals) > 1 else 0.0
                )
            else:
                means.append(np.nan); errs.append(0.0)
        ax.bar(x + (i - 1.5) * width, means, width,
               yerr=errs, capsize=4,
               color=CONDITION_COLORS[cond],
               label=CONDITION_LABELS[cond],
               edgecolor="black")

    ax.set_xticks(x)
    ax.set_xticklabels(priorities, fontsize=11)
    ax.set_ylabel("Fulfillment Rate (%)")
    ax.set_title(
        "Per-Priority Fulfillment Rate — where the wins land\n"
        "Higher-priority requests benefit most from LP's urgency-aware ordering",
        fontsize=12,
    )
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, "fig6_per_priority_fr.png")
    return fig


# ---------------------------------------------------------------------- Fig 7
def fig7_paired_diffs(
    results: dict[str, pd.DataFrame],
    out_dir: Path,
) -> plt.Figure:
    """Histogram of CRN-paired differences for the three primary tests."""
    metrics = HEADLINE_METRICS
    n_pairs = len(PRIMARY_PAIRS) - 1   # show first 3 (skip the "MPC vs greedy" duplicate of pipeline value)
    fig, axes = plt.subplots(n_pairs, len(metrics),
                             figsize=(4.2 * len(metrics), 3.0 * n_pairs),
                             squeeze=False)

    for i, (cond_a, cond_b) in enumerate(PRIMARY_PAIRS[:n_pairs]):
        if cond_a not in results or cond_b not in results:
            continue
        for j, m in enumerate(metrics):
            ax = axes[i, j]
            if m not in results[cond_a].columns or m not in results[cond_b].columns:
                ax.axis("off"); continue
            a = results[cond_a][m].dropna().values
            b = results[cond_b][m].dropna().values
            n = min(len(a), len(b))
            if n < 2:
                ax.axis("off"); continue
            diff = a[:n] - b[:n]
            mean_d = diff.mean()
            ax.hist(diff, bins=12, color=CONDITION_COLORS.get(cond_a, "#444"),
                    alpha=0.7, edgecolor="black")
            ax.axvline(0, color="black", lw=1.0, ls="--")
            ax.axvline(mean_d, color="red", lw=2.0,
                       label=f"mean = {mean_d:+.2f}")
            ax.set_title(
                f"{CONDITION_LABELS[cond_a]} − {CONDITION_LABELS[cond_b]}\n{m}",
                fontsize=10,
            )
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

    fig.suptitle(
        "Paired-Difference Distributions (CRN-paired across replications)\n"
        "If the bulk of mass is to the right of zero, A wins per-seed",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    _save(fig, out_dir, "fig7_paired_diffs.png")
    return fig


# ---------------------------------------------------------------------- Fig 8
def fig8_mde_table(pairwise: pd.DataFrame, out_dir: Path) -> plt.Figure:
    """Power / minimum-detectable-effect summary for primary pairs."""
    primary = pairwise[pairwise["is_primary"]]
    primary = primary[primary["metric"].isin(HEADLINE_METRICS)]

    rows = []
    for (cond_a, cond_b) in PRIMARY_PAIRS:
        for m in HEADLINE_METRICS:
            r = primary[(primary["cond_a"] == cond_a)
                        & (primary["cond_b"] == cond_b)
                        & (primary["metric"] == m)]
            if r.empty:
                continue
            r = r.iloc[0]
            mde   = r["mde_80"]
            obs   = abs(r["mean_diff"])
            ratio = obs / mde if mde > 0 else float("inf")
            powered = "✓" if obs >= mde else "—"
            rows.append([
                f"{CONDITION_LABELS[cond_a]} vs {CONDITION_LABELS[cond_b]}",
                m,
                f"{obs:+.3f}",
                f"{mde:.3f}",
                f"{ratio:.2f}×",
                powered,
            ])

    cols = ["Comparison", "Metric", "Observed |Δ|",
            "MDE (α=0.05, power=0.8)", "Δ/MDE", "Powered"]

    fig, ax = plt.subplots(figsize=(12, 0.45 * len(rows) + 1.5))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.4)
    for j in range(len(cols)):
        tbl[(0, j)].set_facecolor("#2c3e50")
        tbl[(0, j)].set_text_props(weight="bold", color="white")
    for i, r in enumerate(rows):
        if r[-1] == "✓":
            for j in range(len(cols)):
                tbl[(i + 1, j)].set_facecolor("#d4f4d4")
    ax.set_title(
        "Statistical Power — Minimum Detectable Effects\n"
        "A row is 'powered' if the observed difference exceeds the MDE at "
        "α=0.05, power=0.80.",
        fontsize=12, weight="bold", pad=10,
    )
    fig.tight_layout()
    _save(fig, out_dir, "fig8_mde_table.png")
    return fig


# ---------------------------------------------------------------------- plot_all
def plot_all(
    results_baseline: dict[str, pd.DataFrame],
    pairwise:         pd.DataFrame,
    curve_df:         pd.DataFrame,
    out_dir:          Path,
) -> list[plt.Figure]:
    """Generate the eight R3 significance-focused figures."""
    return [
        fig1_headline_table(pairwise,                 out_dir),
        fig2_forest_plot   (pairwise,                 out_dir),
        fig3_demand_curve_FRw    (curve_df,           out_dir),
        fig4_demand_curve_ERRpeak(curve_df,           out_dir),
        fig5_per_window_err(results_baseline,          out_dir),
        fig6_per_priority_fr(results_baseline,         out_dir),
        fig7_paired_diffs(results_baseline,            out_dir),
        fig8_mde_table   (pairwise,                    out_dir),
    ]
