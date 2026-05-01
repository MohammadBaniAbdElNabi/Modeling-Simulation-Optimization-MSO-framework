"""LPDispatchSolver: multi-period LP for blood bank-to-hospital assignment."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from pulp import (
    LpContinuous,
    LpMinimize,
    LpProblem,
    LpStatus,
    LpVariable,
    lpSum,
    value,
    PULP_CBC_CMD,
)

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

SENTINEL = -1

# Priority-tiered unmet-demand penalties (Tier 2).
# All >> max_distance (≈11 min) so coverage always dominates travel cost.
# Ratios 3:2:1 mirror the weight in FR_weighted (emergency=3, urgent=2, normal=1).
_PENALTY_EMERGENCY = 1500.0
_PENALTY_URGENT    = 1000.0
_PENALTY_NORMAL    =  500.0

# Fraction of requests at each priority level (from PRIORITY_PROBS in blood_types.py)
_FRAC_NORMAL    = 0.50
_FRAC_URGENT    = 0.35
_FRAC_EMERGENCY = 0.15


class LPDispatchSolver:
    """Solve a joint multi-period LP over all 24 windows simultaneously.

    Unlike a sequential per-window LP (which is equivalent to greedy because
    it always assigns the nearest bank while inventory is plentiful), this
    formulation includes *cumulative* inventory constraints across all windows.
    Those constraints force the LP to spread load proactively across banks,
    preventing the early depletion of nearby banks that causes the greedy
    policy to fail in later peak hours.

    Objective
    ---------
    min  Σ_{b,h,t} d[b,h] · x[b,h,t]   +   _PENALTY_UNMET · Σ_{h,t} slack[h,t]

    Constraints
    -----------
    (1) Demand coverage (soft):
        Σ_b x[b,h,t] + slack[h,t]  >=  demand[h,t]    for all h, t
    (2) Cumulative inventory (hard):
        Σ_h Σ_{τ<=t} x[b,h,τ]  <=  I_init[b]          for all b, t
    (3) Per-window fleet cap (hard):
        Σ_{b,h} x[b,h,t]  <=  C_fleet                  for all t
    (4) Non-negativity:
        x[b,h,t] >= 0,   slack[h,t] >= 0  (continuous; we round x at extract time)
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.C_fleet = int(config.get("fleet", {}).get("C_fleet", 24))
        self.solver_timeout = int(config.get("solver", {}).get("timeout_s", 120))
        self.B = int(config.get("B", 3))
        self.H = int(config.get("H", 12))
        self.T = int(config.get("T", 24))
        # Average blood units consumed per delivery.  The LP inventory constraint
        # uses I_init measured in *units*, but x[b,h,t] counts *deliveries*.
        # Multiplying x by avg_units correctly converts deliveries → units so
        # the constraint is meaningful (without this, LP thinks a bank with 200
        # units of stock can make 200 deliveries instead of the real ~80).
        self.avg_units = float(config.get("avg_units_per_delivery", 2.5))
        self._x_sol: dict[tuple[int, int, int], int] = {}

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------

    def solve(
        self,
        lambda_model: np.ndarray,
        d_matrix: np.ndarray,
        I_init: np.ndarray,
    ) -> dict[tuple[int, int, int], int]:
        """Solve the joint multi-period LP for all 24 windows.

        Parameters
        ----------
        lambda_model : shape (H, T)  -- planning arrival rates (requests/hr);
                       may come from GlobalStatic, HospitalStatic, or SARIMA.
        d_matrix     : shape (B, H)  -- flight times in minutes
        I_init       : shape (B,)    -- initial total inventory per bank

        Returns
        -------
        x_sol : dict[(b, h, t)] -> int  (rounded LP solution; never SENTINEL)
        """
        B, H, T = self.B, self.H, self.T

        # Pre-compute integer demand per window
        demand = np.zeros((H, T), dtype=int)
        for h in range(H):
            for t in range(T):
                demand[h, t] = max(0, math.ceil(lambda_model[h, t]))

        # ── LP variables ──────────────────────────────────────────────
        x: dict[tuple[int, int, int], LpVariable] = {}
        for b in range(B):
            for h in range(H):
                for t in range(T):
                    x[(b, h, t)] = LpVariable(
                        f"x_{b}_{h}_{t}", lowBound=0, cat=LpContinuous
                    )

        # Priority-tiered demand fractions per (h, t)
        demand_e = np.zeros((H, T), dtype=int)
        demand_u = np.zeros((H, T), dtype=int)
        demand_n = np.zeros((H, T), dtype=int)
        for h in range(H):
            for t in range(T):
                d = demand[h, t]
                demand_e[h, t] = int(round(d * _FRAC_EMERGENCY))
                demand_u[h, t] = int(round(d * _FRAC_URGENT))
                demand_n[h, t] = max(0, d - demand_e[h, t] - demand_u[h, t])

        # Per-priority slack variables (box-constrained to cap at tier demand)
        s_e: dict[tuple[int, int], LpVariable] = {}
        s_u: dict[tuple[int, int], LpVariable] = {}
        s_n: dict[tuple[int, int], LpVariable] = {}
        for h in range(H):
            for t in range(T):
                s_e[(h, t)] = LpVariable(f"se_{h}_{t}", lowBound=0,
                                         upBound=float(demand_e[h, t]), cat=LpContinuous)
                s_u[(h, t)] = LpVariable(f"su_{h}_{t}", lowBound=0,
                                         upBound=float(demand_u[h, t]), cat=LpContinuous)
                s_n[(h, t)] = LpVariable(f"sn_{h}_{t}", lowBound=0,
                                         upBound=float(demand_n[h, t]), cat=LpContinuous)

        # ── Problem & objective (priority-weighted penalties) ─────────
        prob = LpProblem("multi_period_dispatch", LpMinimize)
        prob += (
            lpSum(d_matrix[b, h] * x[(b, h, t)]
                  for b in range(B) for h in range(H) for t in range(T))
            + _PENALTY_EMERGENCY * lpSum(s_e[(h, t)] for h in range(H) for t in range(T))
            + _PENALTY_URGENT    * lpSum(s_u[(h, t)] for h in range(H) for t in range(T))
            + _PENALTY_NORMAL    * lpSum(s_n[(h, t)] for h in range(H) for t in range(T))
        )

        # ── Constraint 1: soft demand coverage (all priority tiers) ──
        for h in range(H):
            for t in range(T):
                if demand[h, t] > 0:
                    prob += (
                        lpSum(x[(b, h, t)] for b in range(B))
                        + s_e[(h, t)] + s_u[(h, t)] + s_n[(h, t)]
                        >= demand[h, t]
                    )

        # ── Constraint 2: cumulative inventory (KEY differentiator) ──
        # I_init[b] is measured in *blood units*; x[b,h,t] counts *deliveries*.
        # Multiplying by avg_units converts deliveries → units, making the
        # constraint physically meaningful and genuinely binding.
        # Without this scaling, LP treats a 200-unit bank as having capacity for
        # 200 deliveries instead of the real 200/2.5 ≈ 80 — the constraint never
        # bites, LP assigns the same nearest bank as greedy, and results are
        # identical.  With correct scaling, LP must plan across all 24 windows
        # jointly and route some demand to non-nearest banks to stay feasible.
        for b in range(B):
            for t in range(T):
                prob += (
                    lpSum(
                        self.avg_units * x[(b, h, tau)]
                        for h in range(H)
                        for tau in range(t + 1)
                    )
                    <= float(I_init[b])
                )

        # ── Constraint 3: per-window fleet cap ───────────────────────
        for t in range(T):
            prob += (
                lpSum(x[(b, h, t)] for b in range(B) for h in range(H))
                <= self.C_fleet
            )

        # ── Solve ─────────────────────────────────────────────────────
        solver = PULP_CBC_CMD(timeLimit=self.solver_timeout, msg=0)
        status_code = prob.solve(solver)
        status_str = LpStatus[prob.status]

        if status_str not in ("Optimal", "Not Solved"):
            logger.warning("LP status: %s — extracting best available solution", status_str)

        total_unmet = 0.0
        x_sol: dict[tuple[int, int, int], int] = {}
        for b in range(B):
            for h in range(H):
                for t in range(T):
                    raw = value(x[(b, h, t)])
                    val = max(0, int(round(raw if raw is not None else 0.0)))
                    x_sol[(b, h, t)] = val

        for h in range(H):
            for t in range(T):
                for sv in (s_e, s_u, s_n):
                    raw = value(sv[(h, t)])
                    total_unmet += raw if raw is not None else 0.0

        logger.info(
            "Multi-period LP status=%s | total unmet demand (slack)=%.1f",
            status_str, total_unmet,
        )

        self._x_sol = x_sol
        return x_sol

    # ------------------------------------------------------------------
    # Persist
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Serialize x_sol to JSON as nested dict {b: {h: {t: int}}}."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        nested: dict[str, dict[str, dict[str, int]]] = {}
        for (b, h, t), v in self._x_sol.items():
            nested.setdefault(str(b), {}).setdefault(str(h), {})[str(t)] = v
        with open(path, "w") as f:
            json.dump(nested, f, indent=2)
        logger.info("LP assignment saved to %s", path)

    @staticmethod
    def load(path: str | Path) -> dict[tuple[int, int, int], int]:
        """Load x_sol from JSON back to dict[(b, h, t)] -> int."""
        with open(path, "r") as f:
            nested = json.load(f)
        x_sol: dict[tuple[int, int, int], int] = {}
        for b_key, bdict in nested.items():
            for h_key, hdict in bdict.items():
                for t_key, v in hdict.items():
                    x_sol[(int(b_key), int(h_key), int(t_key))] = int(v)
        return x_sol
