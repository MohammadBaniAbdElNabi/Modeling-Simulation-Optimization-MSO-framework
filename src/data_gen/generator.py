"""SyntheticDemandGenerator: produces synthetic blood request demand data."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class SyntheticDemandGenerator:
    """Generate synthetic hourly blood request counts with seasonal structure."""

    BASE_LAMBDA: list[float] = [
        0.3, 0.3, 0.3, 0.3, 0.3, 0.3,   # hours 0-5  (low overnight)
        0.8, 1.2, 1.5, 1.4, 1.3, 1.6,   # hours 6-11 (morning ramp)
        1.8, 2.0, 1.9, 1.7, 1.5, 1.4,   # hours 12-17 (peak)
        1.2, 1.0, 0.8, 0.6, 0.5, 0.4,   # hours 18-23 (evening)
    ]

    def __init__(self, config: dict[str, Any]) -> None:
        self.H = int(config.get("H", 12))
        self.T = int(config.get("T", 24))
        self.D_train = int(config.get("D_train", 6))
        self.D_test = int(config.get("D_test", 1))
        self.noise_std = float(config.get("noise_std", 0.15))
        self.scale_low = float(config.get("scale_low", 0.6))
        self.scale_high = float(config.get("scale_high", 1.4))
        self.lambda_floor = float(config.get("lambda_floor", 0.10))
        self.seed = int(config.get("seed", 42))
        self.base_lambda = list(config.get("base_lambda", self.BASE_LAMBDA))

    def generate(self) -> dict[str, np.ndarray]:
        """Generate synthetic demand.

        Returns
        -------
        dict with keys:
            counts       : ndarray (D_train+D_test, H, T), int
            lambda_true  : ndarray (D_train+D_test, H, T), float
            train_counts : ndarray (D_train, H, T), int
            test_counts  : ndarray (D_test, H, T), int
        """
        rng = np.random.default_rng(self.seed)
        D = self.D_train + self.D_test
        base = np.array(self.base_lambda, dtype=float)  # shape (T,)

        # Hospital-specific scaling factors, drawn once per hospital
        scale = rng.uniform(self.scale_low, self.scale_high, size=self.H)

        lambda_true = np.zeros((D, self.H, self.T), dtype=float)
        counts = np.zeros((D, self.H, self.T), dtype=int)

        for d in range(D):
            for h in range(self.H):
                for t in range(self.T):
                    noise = rng.normal(0.0, self.noise_std)
                    lam = base[t] * scale[h] + noise
                    lam = max(self.lambda_floor, lam)
                    lambda_true[d, h, t] = lam
                    counts[d, h, t] = rng.poisson(lam)

        return {
            "counts": counts,
            "lambda_true": lambda_true,
            "train_counts": counts[: self.D_train],
            "test_counts": counts[self.D_train :],
        }

    def save(
        self,
        data: dict[str, np.ndarray],
        out_dir: str | Path,
    ) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        hosp_labels = [f"hospital_{h+1:02d}" for h in range(self.H)]
        hour_labels = [f"hour_{t:02d}" for t in range(self.T)]

        def _save_flat(arr: np.ndarray, fname: str, prefix: str) -> None:
            D_local = arr.shape[0]
            rows = []
            for d in range(D_local):
                for h in range(self.H):
                    row = {
                        "day": d,
                        "hospital": hosp_labels[h],
                    }
                    for t in range(self.T):
                        row[hour_labels[t]] = arr[d, h, t]
                    rows.append(row)
            pd.DataFrame(rows).to_csv(out_dir / fname, index=False)

        _save_flat(data["train_counts"], "demand_train.csv", "train")
        _save_flat(data["test_counts"], "demand_test.csv", "test")
        _save_flat(data["lambda_true"], "lambda_true.csv", "lambda_true")
