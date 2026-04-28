"""Generate the 10 figures specified in the visualization plan (Section 5.4)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.evaluation.statistics import PRIMARY_METRICS

POLICY_COLORS = {"random": "#e74c3c", "greedy": "#3498db", "lp": "#2ecc71"}
POLICY_LABELS = {"random": "Random", "greedy": "Greedy", "lp": "LP-Optimized"}


def _save(fig: plt.Figure, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig1_forecast_vs_truth(
    lambda_hat: np.ndarray,
    lambda_true: np.ndarray,
    out_dir: Path,
) -> plt.Figure:
    """Fig 1: SARIMA forecast vs ground truth per hospital."""
    H, T = lambda_hat.shape
    ncols = 4
    nrows = (H + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3 * nrows), sharey=False)
    axes = axes.flatten()
    hours = np.arange(T)
    for h in range(H):
        ax = axes[h]
        ax.plot(hours, lambda_true[h], "k--", label="True λ", lw=1.5)
        ax.plot(hours, lambda_hat[h], "r-", label="λ̂ (SARIMA)", lw=1.5)
        ax.set_title(f"Hospital {h+1:02d}", fontsize=9)
        ax.set_xlabel("Hour", fontsize=8)
        ax.set_ylabel("Requests/hr", fontsize=8)
        if h == 0:
            ax.legend(fontsize=7)
    for i in range(H, len(axes)):
        axes[i].set_visible(False)
    fig.suptitle("SARIMA Forecast vs Ground Truth (Test Day)", fontsize=13)
    fig.tight_layout()
    _save(fig, out_dir, "fig1_forecast_vs_truth.png")
    return fig


def fig2_forecast_errors(metrics_df: pd.DataFrame, out_dir: Path) -> plt.Figure:
    """Fig 2: MAE, RMSE, MAPE per hospital."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, ["MAE", "RMSE", "MAPE"]):
        ax.bar(metrics_df["hospital_id"], metrics_df[col],
               color="#3498db", edgecolor="black", linewidth=0.5)
        ax.set_xlabel("Hospital")
        ax.set_ylabel(col)
        ax.set_title(f"Forecast {col} per Hospital")
        ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    _save(fig, out_dir, "fig2_forecast_errors.png")
    return fig


def fig3_fulfillment_rate(summary: pd.DataFrame, out_dir: Path) -> plt.Figure:
    """Fig 3: Fulfillment rate by policy with 95% CI."""
    sub = summary[summary["metric"] == "FR"].copy()
    fig, ax = plt.subplots(figsize=(7, 5))
    policies = ["random", "greedy", "lp"]
    for i, p in enumerate(policies):
        row = sub[sub["policy"] == p]
        if row.empty:
            continue
        mean_v = row["mean"].values[0]
        ci_lo = row["ci_lower"].values[0]
        ci_hi = row["ci_upper"].values[0]
        ax.bar(i, mean_v, color=POLICY_COLORS[p], label=POLICY_LABELS[p],
               edgecolor="black")
        ax.errorbar(i, mean_v, yerr=[[mean_v - ci_lo], [ci_hi - mean_v]],
                    fmt="none", color="black", capsize=5)
    ax.set_xticks(range(len(policies)))
    ax.set_xticklabels([POLICY_LABELS[p] for p in policies])
    ax.set_ylabel("Fulfillment Rate (%)")
    ax.set_title("Fulfillment Rate by Policy (mean ± 95% CI)")
    ax.legend()
    fig.tight_layout()
    _save(fig, out_dir, "fig3_fulfillment_rate.png")
    return fig


def fig4_avg_delivery_time(summary: pd.DataFrame, out_dir: Path) -> plt.Figure:
    """Fig 4: Average delivery time by policy with 95% CI."""
    sub = summary[summary["metric"] == "ADT"].copy()
    fig, ax = plt.subplots(figsize=(7, 5))
    policies = ["random", "greedy", "lp"]
    for i, p in enumerate(policies):
        row = sub[sub["policy"] == p]
        if row.empty:
            continue
        mean_v = row["mean"].values[0]
        ci_lo = row["ci_lower"].values[0]
        ci_hi = row["ci_upper"].values[0]
        ax.bar(i, mean_v, color=POLICY_COLORS[p], label=POLICY_LABELS[p],
               edgecolor="black")
        ax.errorbar(i, mean_v, yerr=[[mean_v - ci_lo], [ci_hi - mean_v]],
                    fmt="none", color="black", capsize=5)
    ax.set_xticks(range(len(policies)))
    ax.set_xticklabels([POLICY_LABELS[p] for p in policies])
    ax.set_ylabel("Average Delivery Time (min)")
    ax.set_title("Average Delivery Time by Policy (mean ± 95% CI)")
    ax.legend()
    fig.tight_layout()
    _save(fig, out_dir, "fig4_avg_delivery_time.png")
    return fig


def fig5_delivery_time_boxplot(
    results: dict[str, pd.DataFrame], out_dir: Path
) -> plt.Figure:
    """Fig 5: Box plot of delivery time distribution by policy."""
    fig, ax = plt.subplots(figsize=(8, 5))
    data = [results[p]["ADT"].dropna().values for p in ["random", "greedy", "lp"]]
    bp = ax.boxplot(data, patch_artist=True, notch=False)
    for patch, p in zip(bp["boxes"], ["random", "greedy", "lp"]):
        patch.set_facecolor(POLICY_COLORS[p])
    ax.set_xticklabels([POLICY_LABELS[p] for p in ["random", "greedy", "lp"]])
    ax.set_ylabel("Average Delivery Time (min)")
    ax.set_title("Delivery Time Distribution by Policy (20 replications)")
    fig.tight_layout()
    _save(fig, out_dir, "fig5_delivery_time_boxplot.png")
    return fig


def fig6_expired_rate(summary: pd.DataFrame, out_dir: Path) -> plt.Figure:
    """Fig 6: Expired request rate by policy."""
    sub = summary[summary["metric"] == "ERR"].copy()
    fig, ax = plt.subplots(figsize=(7, 5))
    policies = ["random", "greedy", "lp"]
    for i, p in enumerate(policies):
        row = sub[sub["policy"] == p]
        if row.empty:
            continue
        ax.bar(i, row["mean"].values[0], color=POLICY_COLORS[p],
               label=POLICY_LABELS[p], edgecolor="black")
    ax.set_xticks(range(len(policies)))
    ax.set_xticklabels([POLICY_LABELS[p] for p in policies])
    ax.set_ylabel("Expired Request Rate (%)")
    ax.set_title("Expired Request Rate by Policy")
    ax.legend()
    fig.tight_layout()
    _save(fig, out_dir, "fig6_expired_rate.png")
    return fig


def fig7_lp_heatmap(
    x_sol: dict[tuple[int, int, int], int],
    out_dir: Path,
    t_window: int = 14,
    B: int = 3,
    H: int = 12,
) -> plt.Figure:
    """Fig 7: LP assignment matrix heatmap for window t=14."""
    mat = np.zeros((B, H))
    for b in range(B):
        for h in range(H):
            v = x_sol.get((b, h, t_window), 0)
            mat[b, h] = max(0, v)

    fig, ax = plt.subplots(figsize=(12, 4))
    sns.heatmap(
        mat,
        ax=ax,
        annot=True,
        fmt=".0f",
        cmap="Blues",
        xticklabels=[f"H{h+1:02d}" for h in range(H)],
        yticklabels=[f"Bank {b+1}" for b in range(B)],
    )
    ax.set_title(f"LP Assignment Matrix x[b,h] for Window t={t_window}")
    ax.set_xlabel("Hospital")
    ax.set_ylabel("Blood Bank")
    fig.tight_layout()
    _save(fig, out_dir, "fig7_lp_heatmap.png")
    return fig


def fig8_per_hospital_fulfillment(
    results: dict[str, pd.DataFrame], out_dir: Path, H: int = 12
) -> plt.Figure:
    """Fig 8: Per-hospital fulfillment rate by policy."""
    hospital_cols = [f"FR_H{h+1:02d}" for h in range(H)]
    available_cols = {
        p: [c for c in hospital_cols if c in results[p].columns]
        for p in results
    }

    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(H)
    width = 0.25
    for i, p in enumerate(["random", "greedy", "lp"]):
        cols = available_cols.get(p, [])
        means = [results[p][c].mean() for c in cols] if cols else [0.0] * H
        ax.bar(x + i * width, means, width, color=POLICY_COLORS[p],
               label=POLICY_LABELS[p], edgecolor="black")
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"H{h+1:02d}" for h in range(H)], rotation=45)
    ax.set_ylabel("Fulfillment Rate (%)")
    ax.set_title("Per-Hospital Fulfillment Rate by Policy")
    ax.legend()
    fig.tight_layout()
    _save(fig, out_dir, "fig8_per_hospital_fulfillment.png")
    return fig


def fig9_per_priority_fulfillment(
    results: dict[str, pd.DataFrame], out_dir: Path
) -> plt.Figure:
    """Fig 9: Per-priority fulfillment rate by policy."""
    priority_cols = ["FR_NORMAL", "FR_URGENT", "FR_EMERGENCY"]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(priority_cols))
    width = 0.25
    for i, p in enumerate(["random", "greedy", "lp"]):
        means = []
        for c in priority_cols:
            if c in results[p].columns:
                means.append(float(results[p][c].mean()))
            else:
                means.append(0.0)
        ax.bar(x + i * width, means, width, color=POLICY_COLORS[p],
               label=POLICY_LABELS[p], edgecolor="black")
    ax.set_xticks(x + width)
    ax.set_xticklabels(["NORMAL", "URGENT", "EMERGENCY"])
    ax.set_ylabel("Fulfillment Rate (%)")
    ax.set_title("Per-Priority Fulfillment Rate by Policy")
    ax.legend()
    fig.tight_layout()
    _save(fig, out_dir, "fig9_per_priority_fulfillment.png")
    return fig


def fig10_forecast_vs_adt(
    lambda_hat: np.ndarray,
    results: dict[str, pd.DataFrame],
    out_dir: Path,
) -> plt.Figure:
    """Fig 10: lambda_hat mean per hospital vs ADT by policy."""
    lambda_mean_h = lambda_hat.mean(axis=1)  # shape (H,)

    fig, ax = plt.subplots(figsize=(8, 5))
    for p in ["random", "greedy", "lp"]:
        if p not in results:
            continue
        adt_val = float(results[p]["ADT"].mean())
        ax.scatter(
            lambda_mean_h, [adt_val] * len(lambda_mean_h),
            label=POLICY_LABELS[p], color=POLICY_COLORS[p], alpha=0.7, s=60
        )
    ax.set_xlabel("Mean Forecasted Arrival Rate λ̂ (requests/hr per hospital)")
    ax.set_ylabel("Average Delivery Time (min)")
    ax.set_title("Forecasted Demand vs ADT by Policy")
    ax.legend()
    fig.tight_layout()
    _save(fig, out_dir, "fig10_forecast_vs_adt.png")
    return fig


def plot_all(
    lambda_hat: np.ndarray,
    lambda_true: np.ndarray,
    forecast_metrics: pd.DataFrame,
    summary: pd.DataFrame,
    pairwise: pd.DataFrame,
    results: dict[str, pd.DataFrame],
    x_sol: dict,
    out_dir: Path,
) -> list[plt.Figure]:
    figures = [
        fig1_forecast_vs_truth(lambda_hat, lambda_true, out_dir),
        fig2_forecast_errors(forecast_metrics, out_dir),
        fig3_fulfillment_rate(summary, out_dir),
        fig4_avg_delivery_time(summary, out_dir),
        fig5_delivery_time_boxplot(results, out_dir),
        fig6_expired_rate(summary, out_dir),
        fig7_lp_heatmap(x_sol, out_dir),
        fig8_per_hospital_fulfillment(results, out_dir),
        fig9_per_priority_fulfillment(results, out_dir),
        fig10_forecast_vs_adt(lambda_hat, results, out_dir),
    ]
    return figures
