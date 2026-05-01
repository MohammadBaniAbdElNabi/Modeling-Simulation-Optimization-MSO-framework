"""RollingHorizonLP: Model Predictive Control formulation of the dispatch LP.

Static daily LP (current implementation) commits a 24-hour plan at t=0 using
forecasted demand. It cannot react to:
  * inventory drift caused by stochastic realisations,
  * pending-queue depth at hospital h,
  * forecast errors that accumulate over the day.

This module solves the LP **on a rolling horizon**: at each replan time t*,
solve the LP for windows [t*, t* + H) using:
  * remaining inventory at t*,
  * the residual forecast for the lookahead window,
  * (optionally) an upper-quantile of demand to hedge against demand spikes.

By construction, the static LP is the special case ``replan_interval=infty``,
so the rolling-horizon LP is a strict generalisation: in expectation it
performs at least as well as static under the same forecast.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from pulp import (
    LpContinuous,
    LpMinimize,
    LpProblem,
    LpStatus,
    LpVariable,
    PULP_CBC_CMD,
    lpSum,
    value,
)

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

_PENALTY_EMERGENCY = 1500.0
_PENALTY_URGENT    = 1000.0
_PENALTY_NORMAL    =  500.0
_FRAC_NORMAL    = 0.50
_FRAC_URGENT    = 0.35
_FRAC_EMERGENCY = 0.15


class RollingHorizonLP:
    """Solve the dispatch LP on a [t*, t*+H) horizon with current inventory.

    Parameters
    ----------
    config : dict-like LP config (same schema as ``LPDispatchSolver``).
    horizon_hours : window count to plan over (e.g. 6).
    demand_quantile : upper percentile to hedge against demand spikes.  When
        ``demand_quantile is None``, point forecasts are used (mean demand).
        When set to e.g. 0.8, the LP enforces coverage of the 80th percentile
        of a Poisson(lambda) demand distribution.
    """

    def __init__(
        self,
        config: dict[str, Any],
        horizon_hours: int = 6,
        demand_quantile: float | None = 0.8,
    ) -> None:
        self.C_fleet         = int(config.get("fleet", {}).get("C_fleet", 24))
        self.solver_timeout  = int(config.get("solver", {}).get("timeout_s", 30))
        self.B               = int(config.get("B", 3))
        self.H               = int(config.get("H", 12))
        self.T               = int(config.get("T", 24))
        self.avg_units       = float(config.get("avg_units_per_delivery", 2.5))
        self.horizon_hours   = int(horizon_hours)
        self.demand_quantile = demand_quantile

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _coverage_demand(self, lam: np.ndarray) -> np.ndarray:
        """Convert lambda (req/hr) into integer demand to cover.

        If ``demand_quantile`` is set, use the upper-quantile of a
        Poisson(lambda) so the LP hedges against demand spikes.
        Otherwise, use ceil(lambda) (point forecast).
        """
        if self.demand_quantile is None:
            return np.ceil(np.maximum(lam, 0.0)).astype(int)

        # Poisson(lam) quantile via inverse CDF (small lambda → small mode)
        from scipy.stats import poisson
        q = self.demand_quantile
        return poisson.ppf(q, np.maximum(lam, 1e-6)).astype(int)

    # ------------------------------------------------------------------
    # Solve a single horizon window
    # ------------------------------------------------------------------
    def solve_window(
        self,
        t_start:    int,
        lambda_horizon: np.ndarray,   # (H, H_lookahead)
        d_matrix:   np.ndarray,        # (B, H)
        I_remaining: np.ndarray,       # (B,)
    ) -> dict[tuple[int, int, int], int]:
        """Solve LP over windows [t_start, t_start + L) where L = lambda_horizon.shape[1].

        Returns absolute-time keyed dict ``x_sol[(b, h, t_global)] -> int``.
        Windows outside [t_start, t_start+L) get no entries.
        """
        B, H = self.B, self.H
        L    = int(lambda_horizon.shape[1])
        if L == 0:
            return {}

        demand = self._coverage_demand(lambda_horizon)   # (H, L)

        # Decision variables x[b, h, k] for local window k in [0, L)
        x: dict[tuple[int, int, int], LpVariable] = {}
        for b in range(B):
            for h in range(H):
                for k in range(L):
                    x[(b, h, k)] = LpVariable(
                        f"x_{b}_{h}_{k}", lowBound=0, cat=LpContinuous
                    )

        # Priority-tiered demand fractions
        demand_e = np.zeros((H, L), dtype=int)
        demand_u = np.zeros((H, L), dtype=int)
        demand_n = np.zeros((H, L), dtype=int)
        for h in range(H):
            for k in range(L):
                d = int(demand[h, k])
                demand_e[h, k] = int(round(d * _FRAC_EMERGENCY))
                demand_u[h, k] = int(round(d * _FRAC_URGENT))
                demand_n[h, k] = max(0, d - demand_e[h, k] - demand_u[h, k])

        s_e: dict[tuple[int, int], LpVariable] = {}
        s_u: dict[tuple[int, int], LpVariable] = {}
        s_n: dict[tuple[int, int], LpVariable] = {}
        for h in range(H):
            for k in range(L):
                s_e[(h, k)] = LpVariable(f"se_{h}_{k}", lowBound=0,
                                         upBound=float(demand_e[h, k]), cat=LpContinuous)
                s_u[(h, k)] = LpVariable(f"su_{h}_{k}", lowBound=0,
                                         upBound=float(demand_u[h, k]), cat=LpContinuous)
                s_n[(h, k)] = LpVariable(f"sn_{h}_{k}", lowBound=0,
                                         upBound=float(demand_n[h, k]), cat=LpContinuous)

        prob = LpProblem("rolling_horizon_dispatch", LpMinimize)
        prob += (
            lpSum(d_matrix[b, h] * x[(b, h, k)]
                  for b in range(B) for h in range(H) for k in range(L))
            + _PENALTY_EMERGENCY * lpSum(s_e[(h, k)] for h in range(H) for k in range(L))
            + _PENALTY_URGENT    * lpSum(s_u[(h, k)] for h in range(H) for k in range(L))
            + _PENALTY_NORMAL    * lpSum(s_n[(h, k)] for h in range(H) for k in range(L))
        )

        # Demand coverage (soft, all priority tiers)
        for h in range(H):
            for k in range(L):
                if demand[h, k] > 0:
                    prob += (
                        lpSum(x[(b, h, k)] for b in range(B))
                        + s_e[(h, k)] + s_u[(h, k)] + s_n[(h, k)]
                        >= float(demand[h, k])
                    )

        # Cumulative inventory (units, scaled by avg_units_per_delivery)
        for b in range(B):
            for k in range(L):
                prob += (
                    lpSum(
                        self.avg_units * x[(b, h, kk)]
                        for h in range(H) for kk in range(k + 1)
                    ) <= float(I_remaining[b])
                )

        # Per-window fleet capacity
        for k in range(L):
            prob += (
                lpSum(x[(b, h, k)] for b in range(B) for h in range(H))
                <= self.C_fleet
            )

        solver = PULP_CBC_CMD(timeLimit=self.solver_timeout, msg=0)
        prob.solve(solver)

        x_sol: dict[tuple[int, int, int], int] = {}
        for b in range(B):
            for h in range(H):
                for k in range(L):
                    raw = value(x[(b, h, k)])
                    val = max(0, int(round(raw if raw is not None else 0.0)))
                    x_sol[(b, h, t_start + k)] = val

        return x_sol

    # ------------------------------------------------------------------
    # Convenience: replicate static LP behaviour by solving once at t=0
    # with horizon=24
    # ------------------------------------------------------------------
    def solve_full_day(
        self,
        lambda_full: np.ndarray,    # (H, T)
        d_matrix:    np.ndarray,
        I_init:      np.ndarray,
    ) -> dict[tuple[int, int, int], int]:
        """Equivalent to a single-shot 24-hour solve (for testing/parity)."""
        return self.solve_window(0, lambda_full, d_matrix, I_init)
