"""AIC-based SARIMA order selection via grid search."""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def sarima_grid_search(
    series: np.ndarray,
    p_range: list[int],
    q_range: list[int],
    P_range: list[int],
    Q_range: list[int],
    d_guide: int,
    D_guide: int,
    s: int = 24,
) -> tuple[tuple[int, int, int, int, int, int] | None, Any | None]:
    """Grid search over SARIMA candidate orders, returning best by AIC.

    d candidates: {d_guide, max(0, d_guide-1)} (small window around guide)
    D candidates: {D_guide}

    Returns
    -------
    (best_order, fitted_model) or (None, None) if no model converges.
    best_order is (p, d, q, P, D, Q).
    """
    d_candidates = sorted({d_guide, max(0, d_guide - 1)})
    D_candidates = [D_guide]

    best_aic = float("inf")
    best_order = None
    best_fit = None

    for p in p_range:
        for q in q_range:
            for d in d_candidates:
                for P in P_range:
                    for Q in Q_range:
                        for D in D_candidates:
                            try:
                                with warnings.catch_warnings():
                                    warnings.simplefilter("ignore")
                                    model = SARIMAX(
                                        series,
                                        order=(p, d, q),
                                        seasonal_order=(P, D, Q, s),
                                        enforce_stationarity=False,
                                        enforce_invertibility=False,
                                    )
                                    fit = model.fit(disp=False, maxiter=200)
                                    if not np.isfinite(fit.aic):
                                        continue
                                    if fit.aic < best_aic:
                                        best_aic = fit.aic
                                        best_order = (p, d, q, P, D, Q)
                                        best_fit = fit
                            except Exception:
                                continue

    return best_order, best_fit
