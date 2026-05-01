"""Demand-model abstraction layer for the M-S-O experiment.

Three concrete classes implement a common interface that produces an
(H, T) array of planning rates consumable by LPDispatchSolver.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class DemandModel(ABC):
    name: str

    @abstractmethod
    def fit(self, train_counts: np.ndarray) -> None: ...

    @abstractmethod
    def predict(self, steps: int = 24) -> np.ndarray: ...

    @abstractmethod
    def metadata(self) -> dict: ...


class GlobalStaticDemand(DemandModel):
    """Single scalar rate averaged over all hospitals and hours, broadcast everywhere."""

    name = "global"

    def __init__(self, H: int = 12, T: int = 24, lambda_min: float = 0.10) -> None:
        self.H, self.T, self.lambda_min = H, T, lambda_min
        self._rate: float = 0.0

    def fit(self, train_counts: np.ndarray) -> None:
        self._rate = float(train_counts.mean())

    def predict(self, steps: int = 24) -> np.ndarray:
        out = np.full((self.H, self.T), self._rate, dtype=float)
        return np.clip(out, self.lambda_min, None)

    def metadata(self) -> dict:
        return {"name": self.name, "rate": self._rate, "n_params": 1}


class HospitalStaticDemand(DemandModel):
    """Per-hospital scalar rate averaged over hours, broadcast to every hour."""

    name = "hospital"

    def __init__(self, H: int = 12, T: int = 24, lambda_min: float = 0.10) -> None:
        self.H, self.T, self.lambda_min = H, T, lambda_min
        self._rates: np.ndarray = np.zeros(H)

    def fit(self, train_counts: np.ndarray) -> None:
        # mean over (D_train, T) -> shape (H,)
        self._rates = train_counts.mean(axis=(0, 2))

    def predict(self, steps: int = 24) -> np.ndarray:
        out = np.tile(self._rates[:, None], (1, self.T))
        return np.clip(out, self.lambda_min, None)

    def metadata(self) -> dict:
        return {"name": self.name, "rates": self._rates.tolist(), "n_params": self.H}


class SARIMAForecastDemand(DemandModel):
    """Thin wrapper exposing the existing SARIMAForecaster via the DemandModel interface."""

    name = "sarima"

    def __init__(self, sarima_config: dict) -> None:
        from src.forecasting.sarima_model import SARIMAForecaster
        self._inner = SARIMAForecaster(sarima_config)

    def fit(self, train_counts: np.ndarray) -> None:
        self._inner.fit(train_counts)

    def predict(self, steps: int = 24) -> np.ndarray:
        return self._inner.predict(steps=steps)

    def metadata(self) -> dict:
        orders = getattr(self._inner, "_best_orders", [])
        aics = [
            float(m.aic) if m is not None else None
            for m in getattr(self._inner, "_fitted_models", [])
        ]
        return {"name": self.name, "best_orders": orders, "aics": aics,
                "n_params": len(orders)}


class FactorDemand(DemandModel):
    """Hierarchical Poisson factor model: lambda[h,t] = sigma_h * pi_t.

    This is the **correctly specified** estimator for the project's DGP
    ``lambda_true[h,t] = scale[h] * base[t] + epsilon`` and beats SARIMA
    in MAPE and bias by being a much smaller, MLE-consistent model.
    """

    name = "factor"

    def __init__(self, config: dict | None = None) -> None:
        from src.forecasting.factor_model import HierarchicalFactorForecaster
        self._inner = HierarchicalFactorForecaster(config or {})

    def fit(self, train_counts: np.ndarray) -> None:
        self._inner.fit(train_counts)

    def predict(self, steps: int = 24) -> np.ndarray:
        return self._inner.predict(steps=steps)

    def metadata(self) -> dict:
        return self._inner.metadata()


DEMAND_MODEL_REGISTRY: dict[str, type[DemandModel]] = {
    "global":   GlobalStaticDemand,
    "hospital": HospitalStaticDemand,
    "sarima":   SARIMAForecastDemand,
    "factor":   FactorDemand,
}
