"""LPDispatchSolver: PuLP integer LP for blood bank-to-hospital assignment."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from pulp import (
    LpInteger,
    LpMinimize,
    LpProblem,
    LpStatus,
    LpVariable,
    lpSum,
    value,
    PULP_CBC_CMD,
)

from src.optimization.fleet_capacity import compute_c_fleet
from src.optimization.inventory_model import InventoryTracker
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

SENTINEL = -1


class LPDispatchSolver:
    """Solve one integer LP per hourly window, returning assignment tensor."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.C_fleet = int(config.get("fleet", {}).get("C_fleet", 24))
        self.solver_timeout = int(
            config.get("solver", {}).get("timeout_s", 60)
        )
        self.B = int(config.get("B", 3))
        self.H = int(config.get("H", 12))
        self.T = int(config.get("T", 24))
        self._x_sol: dict[tuple[int, int, int], int] = {}

    def solve(
        self,
        lambda_hat: np.ndarray,
        d_matrix: np.ndarray,
        I_init: np.ndarray,
    ) -> dict[tuple[int, int, int], int]:
        """Solve LP for all 24 windows sequentially.

        Parameters
        ----------
        lambda_hat : shape (H, T)  -- forecasted arrival rates
        d_matrix   : shape (B, H)  -- flight times in minutes
        I_init     : shape (B,)    -- initial total inventory per bank

        Returns
        -------
        x_sol : dict[(b, h, t)] -> int  (SENTINEL = -1 for infeasible windows)
        """
        B, H, T = self.B, self.H, self.T
        tracker = InventoryTracker(I_init)
        x_sol: dict[tuple[int, int, int], int] = {}

        for t in range(T):
            I = tracker.snapshot()
            prob = LpProblem(f"dispatch_t{t}", LpMinimize)

            # Decision variables x[(b,h)] >= 0, integer
            x = {
                (b, h): LpVariable(f"x_{b}_{h}", lowBound=0, cat=LpInteger)
                for b in range(B)
                for h in range(H)
            }

            # Objective: minimize total flight time weighted by assignments
            prob += lpSum(
                d_matrix[b, h] * x[(b, h)] for b in range(B) for h in range(H)
            )

            # Constraint 1: demand coverage
            for h in range(H):
                demand = math.ceil(lambda_hat[h, t])
                prob += lpSum(x[(b, h)] for b in range(B)) >= demand

            # Constraint 2: inventory capacity
            for b in range(B):
                prob += lpSum(x[(b, h)] for h in range(H)) <= I[b]

            # Constraint 3: fleet throughput
            prob += (
                lpSum(x[(b, h)] for b in range(B) for h in range(H))
                <= self.C_fleet
            )

            solver = PULP_CBC_CMD(timeLimit=self.solver_timeout, msg=0)
            status = prob.solve(solver)

            if LpStatus[prob.status] == "Optimal":
                window_assignments: dict[tuple[int, int], int] = {}
                for b in range(B):
                    for h in range(H):
                        val = int(round(value(x[(b, h)]) or 0))
                        x_sol[(b, h, t)] = val
                        window_assignments[(b, h)] = val
                tracker.update(window_assignments, B, H)
            else:
                logger.warning(
                    "LP infeasible for window t=%d (status=%s); using sentinel",
                    t, LpStatus[prob.status],
                )
                for b in range(B):
                    for h in range(H):
                        x_sol[(b, h, t)] = SENTINEL

        self._x_sol = x_sol
        return x_sol

    def save(self, path: str | Path) -> None:
        """Serialize x_sol to JSON as nested dict {b: {h: {t: int}}}."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        nested: dict[str, dict[str, dict[str, int]]] = {}
        for (b, h, t), v in self._x_sol.items():
            b_key, h_key, t_key = str(b), str(h), str(t)
            nested.setdefault(b_key, {}).setdefault(h_key, {})[t_key] = v

        with open(path, "w") as f:
            json.dump(nested, f, indent=2)
        logger.info("LP assignment saved to %s", path)

    @staticmethod
    def load(path: str | Path) -> dict[tuple[int, int, int], int]:
        """Load x_sol from JSON back to dict[(b,h,t)] -> int."""
        with open(path, "r") as f:
            nested = json.load(f)
        x_sol: dict[tuple[int, int, int], int] = {}
        for b_key, bdict in nested.items():
            for h_key, hdict in bdict.items():
                for t_key, v in hdict.items():
                    x_sol[(int(b_key), int(h_key), int(t_key))] = int(v)
        return x_sol
