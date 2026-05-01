"""Assemble summary_stats.csv and pairwise_tests.csv for the R3 experiment."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from src.evaluation.statistics import (
    PRIMARY_METRICS,
    compute_summary,
    pairwise_tests,
    demand_curve_table,
)
from src.evaluation import visualizations
from src.utils.logging_utils import get_logger

if TYPE_CHECKING:
    import matplotlib.pyplot as plt

logger = get_logger(__name__)


class EvaluationEngine:
    """Coordinate statistics + visualization for the R3 four-condition study."""

    def __init__(
        self,
        results_baseline:    dict[str, pd.DataFrame],
        results_by_demand:   dict[float, dict[str, pd.DataFrame]] | None = None,
    ) -> None:
        self.results_baseline  = results_baseline
        self.results_by_demand = results_by_demand or {}
        self._summary:  pd.DataFrame | None = None
        self._pairwise: pd.DataFrame | None = None
        self._curve:    pd.DataFrame | None = None

    def compute_summary(self) -> pd.DataFrame:
        self._summary = compute_summary(self.results_baseline, PRIMARY_METRICS)
        return self._summary

    def pairwise_tests(self) -> pd.DataFrame:
        self._pairwise = pairwise_tests(self.results_baseline, PRIMARY_METRICS)
        return self._pairwise

    def operating_curve(self, metrics: list[str] | None = None) -> pd.DataFrame:
        if metrics is None:
            metrics = ["FR_weighted", "ERR_peak", "Expiration_cost", "FR", "ERR"]
        if not self.results_by_demand:
            self._curve = pd.DataFrame()
            return self._curve
        frames = [
            demand_curve_table(self.results_by_demand, m) for m in metrics
        ]
        self._curve = pd.concat(frames, ignore_index=True)
        return self._curve

    def plot_all(self, out_dir: str | Path) -> "list[plt.Figure]":
        if self._summary  is None: self.compute_summary()
        if self._pairwise is None: self.pairwise_tests()
        if self._curve    is None: self.operating_curve()
        return visualizations.plot_all(
            results_baseline = self.results_baseline,
            pairwise         = self._pairwise,
            curve_df         = self._curve,
            out_dir          = Path(out_dir),
        )

    def save(self, out_dir: str | Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if self._summary  is None: self.compute_summary()
        if self._pairwise is None: self.pairwise_tests()
        if self._curve    is None: self.operating_curve()

        self._summary.to_csv(out_dir / "summary_stats.csv",  index=False)
        self._pairwise.to_csv(out_dir / "pairwise_tests.csv", index=False)
        if not self._curve.empty:
            self._curve.to_csv(out_dir / "operating_curve.csv", index=False)
        logger.info("Stats saved to %s", out_dir)
