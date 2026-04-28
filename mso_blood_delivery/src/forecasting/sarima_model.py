"""SARIMAForecaster: fit, predict, and evaluate SARIMA models per hospital."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.forecasting.grid_search import sarima_grid_search
from src.forecasting.metrics import compute_metrics
from src.forecasting.stationarity import check_stationarity
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class SARIMAForecaster:
    """Fit one SARIMA model per hospital via AIC grid search, then forecast."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.p_range = list(config.get("p_range", [0, 1, 2]))
        self.d_range = list(config.get("d_range", [0, 1]))
        self.q_range = list(config.get("q_range", [0, 1, 2]))
        self.P_range = list(config.get("P_range", [0, 1]))
        self.D_range = list(config.get("D_range", [0, 1]))
        self.Q_range = list(config.get("Q_range", [0, 1]))
        self.s = int(config.get("s", 24))
        self.lambda_hat_min = float(
            config.get("forecast", {}).get("lambda_hat_min", 0.10)
            if "forecast" in config else config.get("lambda_hat_min", 0.10)
        )
        self.adf_alpha = float(
            config.get("stationarity", {}).get("adf_alpha", 0.05)
            if "stationarity" in config else 0.05
        )
        self.acf_threshold = float(
            config.get("stationarity", {}).get("acf_lag24_threshold", 0.3)
            if "stationarity" in config else 0.3
        )
        self._fitted_models: list[Any] = []
        self._best_orders: list[tuple | None] = []
        self._lambda_hat: np.ndarray | None = None
        self._H: int = 0
        self._T: int = 0

    def fit(self, train_counts: np.ndarray) -> None:
        """Fit SARIMA per hospital on training data.

        Parameters
        ----------
        train_counts : shape (D_train, H, T)
        """
        D_train, H, T = train_counts.shape
        self._H, self._T = H, T
        self._fitted_models = [None] * H
        self._best_orders = [None] * H
        n_converged = 0

        for h in range(H):
            series = train_counts[:, h, :].flatten().astype(float)
            d_guide, D_guide = check_stationarity(
                series, self.adf_alpha, self.acf_threshold
            )
            best_order, best_fit = sarima_grid_search(
                series,
                self.p_range,
                self.q_range,
                self.P_range,
                self.Q_range,
                d_guide,
                D_guide,
                s=self.s,
            )
            if best_fit is not None:
                self._fitted_models[h] = best_fit
                self._best_orders[h] = best_order
                n_converged += 1
                logger.info(
                    "Hospital %02d: best SARIMA%s x %s, AIC=%.2f",
                    h + 1,
                    best_order[:3],
                    best_order[3:],
                    best_fit.aic,
                )
            else:
                logger.warning("Hospital %02d: no SARIMA model converged", h + 1)

        logger.info("%d/%d hospitals converged", n_converged, H)

    def predict(self, steps: int = 24) -> np.ndarray:
        """Return lambda_hat of shape (H, T) with values >= lambda_hat_min."""
        H = self._H
        lambda_hat = np.full((H, steps), self.lambda_hat_min)

        for h in range(H):
            fit = self._fitted_models[h]
            if fit is None:
                logger.warning("Hospital %02d: using floor forecast (no model)", h + 1)
                continue
            try:
                fc = fit.get_forecast(steps=steps)
                pts = fc.predicted_mean.values
                pts = np.clip(pts, a_min=self.lambda_hat_min, a_max=None)
                lambda_hat[h] = pts
            except Exception as exc:
                logger.warning("Hospital %02d forecast failed: %s", h + 1, exc)

        self._lambda_hat = lambda_hat
        return lambda_hat

    def evaluate(self, lambda_true: np.ndarray) -> pd.DataFrame:
        """Evaluate forecast against ground truth.

        Parameters
        ----------
        lambda_true : shape (H, T) -- ground truth for the test day
        """
        if self._lambda_hat is None:
            raise RuntimeError("Call predict() before evaluate().")
        return compute_metrics(self._lambda_hat, lambda_true)

    def save(self, path: str | Path) -> None:
        """Save lambda_hat to CSV."""
        if self._lambda_hat is None:
            raise RuntimeError("Call predict() before save().")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        H, T = self._lambda_hat.shape
        cols = [f"hour_{t:02d}" for t in range(T)]
        idx = [f"hospital_{h+1:02d}" for h in range(H)]
        df = pd.DataFrame(self._lambda_hat, columns=cols, index=idx)
        df.index.name = "hospital_id"
        df.to_csv(path)
        logger.info("lambda_hat saved to %s", path)
