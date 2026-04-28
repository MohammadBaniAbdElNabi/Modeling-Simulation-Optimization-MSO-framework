"""Forecast evaluation metrics: MAE, RMSE, MAPE, Bias."""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_metrics(
    lambda_hat: np.ndarray,
    lambda_true: np.ndarray,
    hospital_ids: list[str] | None = None,
) -> pd.DataFrame:
    """Compute per-hospital forecast metrics.

    Parameters
    ----------
    lambda_hat  : shape (H, T)  -- forecasted rates
    lambda_true : shape (H, T)  -- ground truth rates
    hospital_ids: optional list of length H for row labels

    Returns
    -------
    DataFrame with columns [hospital_id, MAE, RMSE, MAPE, Bias]
    """
    H = lambda_hat.shape[0]
    if hospital_ids is None:
        hospital_ids = [f"hospital_{h+1:02d}" for h in range(H)]

    rows = []
    for h in range(H):
        yhat = lambda_hat[h]
        ytrue = lambda_true[h]
        diff = yhat - ytrue
        mae = float(np.mean(np.abs(diff)))
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        mask = ytrue > 0
        mape = float(np.mean(np.abs(diff[mask]) / ytrue[mask]) * 100) if mask.any() else float("nan")
        bias = float(np.mean(diff))
        rows.append({
            "hospital_id": hospital_ids[h],
            "MAE": mae,
            "RMSE": rmse,
            "MAPE": mape,
            "Bias": bias,
        })

    return pd.DataFrame(rows)
