"""Assemble summary_stats.csv and pairwise_tests.csv."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.evaluation.statistics import PRIMARY_METRICS, compute_summary, pairwise_tests
from src.evaluation import visualizations
from src.utils.logging_utils import get_logger

if TYPE_CHECKING:
    import matplotlib.pyplot as plt

logger = get_logger(__name__)


class EvaluationEngine:
    """Coordinate statistics and visualization over all three policies."""

    def __init__(self, results: dict[str, pd.DataFrame]) -> None:
        self.results = results
        self._summary: pd.DataFrame | None = None
        self._pairwise: pd.DataFrame | None = None

    def compute_summary(self) -> pd.DataFrame:
        self._summary = compute_summary(self.results, PRIMARY_METRICS)
        return self._summary

    def pairwise_tests(self) -> pd.DataFrame:
        self._pairwise = pairwise_tests(self.results, PRIMARY_METRICS)
        return self._pairwise

    def plot_all(
        self,
        lambda_hat: np.ndarray,
        lambda_true: np.ndarray,
        forecast_metrics: pd.DataFrame,
        x_sol: dict,
        out_dir: str | Path,
    ) -> "list[plt.Figure]":
        """Generate all 10 figures (Section 5.4) and save them to out_dir."""
        if self._summary is None:
            self.compute_summary()
        if self._pairwise is None:
            self.pairwise_tests()
        return visualizations.plot_all(
            lambda_hat=lambda_hat,
            lambda_true=lambda_true,
            forecast_metrics=forecast_metrics,
            summary=self._summary,
            pairwise=self._pairwise,
            results=self.results,
            x_sol=x_sol,
            out_dir=Path(out_dir),
        )

    def save(self, out_dir: str | Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if self._summary is None:
            self.compute_summary()
        if self._pairwise is None:
            self.pairwise_tests()

        summary_path = out_dir / "summary_stats.csv"
        pairwise_path = out_dir / "pairwise_tests.csv"

        self._summary.to_csv(summary_path, index=False)
        self._pairwise.to_csv(pairwise_path, index=False)
        logger.info("Summary stats saved to %s", summary_path)
        logger.info("Pairwise tests saved to %s", pairwise_path)
