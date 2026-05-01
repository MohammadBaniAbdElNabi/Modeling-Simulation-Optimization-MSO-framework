"""HierarchicalFactorForecaster: rank-1 Poisson factor model lambda[h,t] = sigma_h * pi_t.

The DGP for this project is multiplicative-separable:

    lambda_true[h, t] = scale[h] * base[t] + epsilon

The Poisson MLE under correct specification is given by alternating closed-form
updates of (sigma_h) and (pi_t) — at convergence this yields the **best
asymptotic estimator** by Cramér-Rao, with bias O(1/N) where N = D*H*T = 1728
training observations.

Why this beats SARIMA on this data:
- 36 free parameters (12 sigma + 24 pi) vs 12 * O(p,d,q,P,D,Q) for SARIMA.
- Pools D*H = 72 obs per hour, D*T = 144 obs per hospital, so individual
  parameters are estimated from far more data.
- No within-hospital autoregressive structure to overfit on 144 obs.
- No bias from differencing or clipping.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.forecasting.metrics import compute_metrics
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class HierarchicalFactorForecaster:
    """Poisson rank-1 factor model with EM-style alternating updates."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = dict(config or {})
        self.lambda_min  = float(cfg.get("lambda_min", 0.10))
        self.n_em_iters  = int(cfg.get("n_em_iters", 50))
        self.tol         = float(cfg.get("tol", 1e-8))
        self._sigma:  np.ndarray | None = None  # (H,)
        self._pi:     np.ndarray | None = None  # (T,)
        self._lambda_hat: np.ndarray | None = None
        self._H: int = 0
        self._T: int = 0

    # ------------------------------------------------------------------
    # Fit: alternating Poisson MLE
    # ------------------------------------------------------------------
    def fit(self, train_counts: np.ndarray) -> None:
        """Estimate (sigma_h, pi_t) by alternating MLE.

        Parameters
        ----------
        train_counts : (D, H, T) integer counts.

        At convergence:
            sigma_h = sum_{d,t} N[d,h,t] / (D * sum_t pi_t)
            pi_t    = sum_{d,h} N[d,h,t] / (D * sum_h sigma_h)
        with identifiability constraint  mean(pi_t) = 1.
        """
        tc = train_counts.astype(float)
        D, H, T = tc.shape
        self._H, self._T = H, T

        # Pooled aggregates (used in every iteration)
        row_totals = tc.sum(axis=(0, 2))   # (H,)  -- sum over D and T
        col_totals = tc.sum(axis=(0, 1))   # (T,)  -- sum over D and H

        # Initialise: uniform pi, sigma = per-hospital mean
        pi    = np.ones(T, dtype=float)
        sigma = row_totals / (D * T)
        sigma = np.maximum(sigma, 1e-6)

        for it in range(self.n_em_iters):
            sigma_prev = sigma.copy()
            pi_prev    = pi.copy()

            # sigma update: sigma_h = row_totals[h] / (D * sum_t pi_t)
            sigma = row_totals / (D * pi.sum() + 1e-12)
            sigma = np.maximum(sigma, 1e-6)

            # pi update: pi_t = col_totals[t] / (D * sum_h sigma_h)
            pi = col_totals / (D * sigma.sum() + 1e-12)
            pi = np.maximum(pi, 1e-6)

            # Identifiability: rescale so mean(pi) = 1, sigma absorbs scale
            scale = pi.mean()
            pi    = pi / scale
            sigma = sigma * scale

            delta = (np.abs(sigma - sigma_prev).max()
                     + np.abs(pi - pi_prev).max())
            if delta < self.tol:
                logger.info("Hierarchical factor MLE converged in %d iterations", it + 1)
                break

        self._sigma = sigma
        self._pi    = pi

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    def predict(self, steps: int = 24) -> np.ndarray:
        if self._sigma is None or self._pi is None:
            raise RuntimeError("Call fit() before predict().")
        # Outer product gives lambda_hat[h, t] = sigma_h * pi_t
        out = np.outer(self._sigma, self._pi[:steps])
        out = np.clip(out, self.lambda_min, None)
        self._lambda_hat = out
        return out

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def evaluate(self, lambda_true: np.ndarray) -> pd.DataFrame:
        if self._lambda_hat is None:
            raise RuntimeError("Call predict() before evaluate().")
        return compute_metrics(self._lambda_hat, lambda_true)

    def metadata(self) -> dict:
        return {
            "name":     "factor",
            "n_params": (self._H + self._T) if self._sigma is not None else 0,
            "sigma":    self._sigma.tolist() if self._sigma is not None else [],
            "pi":       self._pi.tolist()    if self._pi is not None    else [],
        }
