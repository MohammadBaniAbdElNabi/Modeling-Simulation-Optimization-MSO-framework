"""MPCDispatch: rolling-horizon LP re-solved at runtime with current state."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.simulation.dispatch.base_policy import BaseDispatch
from src.simulation.dispatch.greedy_policy import _feasibility_check
from src.simulation.entities import DroneState, Request
from src.utils.logging_utils import get_logger

if TYPE_CHECKING:
    from src.optimization.rolling_lp import RollingHorizonLP
    from src.simulation.entities import BloodBank, Drone, Hospital
    from src.simulation.metrics_collector import MetricsCollector

logger = get_logger(__name__)


class MPCDispatch(BaseDispatch):
    """Model-Predictive-Control dispatch policy.

    At each replan boundary the policy snapshots current inventory, applies
    online demand-scaling from observed arrivals, and solves the rolling LP
    for the next horizon.  Compared to the static LP:
      * Inventory is observed in real-time (reacts to stochastic replenishment).
      * Demand forecast is updated from empirical arrival counts.
      * Drone pool is shared (any idle drone serves any bank).

    Replication boundaries are auto-detected via env_now going backwards.
    """

    def __init__(
        self,
        rolling_lp:    "RollingHorizonLP",
        lambda_full:   np.ndarray,        # (H, T) point forecast
        d_matrix:      np.ndarray,        # (B, H) flight times in minutes
        replan_interval_hours: float = 1.0,
    ) -> None:
        self.rolling_lp     = rolling_lp
        self.lambda_full    = lambda_full
        self.d_matrix_min   = d_matrix
        self.replan_iv_s    = float(replan_interval_hours) * 3600.0
        self._x_sol:        dict[tuple[int, int, int], int] = {}
        self._consumed:     dict[tuple[int, int, int], int] = {}
        self._next_replan_s: float = 0.0
        self._last_env_now:  float = float("inf")

    def reset(self) -> None:
        self._x_sol          = {}
        self._consumed       = {}
        self._next_replan_s  = 0.0
        self._last_env_now   = float("inf")

    # ------------------------------------------------------------------
    # Online demand scaling (Tier 3)
    # ------------------------------------------------------------------
    def _online_lambda(
        self,
        t_window: int,
        metrics: "MetricsCollector",
    ) -> np.ndarray:
        """Scale future lambda by empirical/forecast ratio from completed windows."""
        if t_window == 0 or not metrics._arrivals:
            return self.lambda_full

        H = self.rolling_lp.H
        obs = np.zeros(H)
        for req in metrics._arrivals:
            h = req.hospital_id
            if h < H:
                t = min(int(req.arrival_time / 3600), t_window - 1)
                if t < t_window:
                    obs[h] += 1

        # Empirical rate per hospital (total arrivals / completed windows)
        forecast_total = self.lambda_full[:, :t_window].sum(axis=1)
        scale = np.where(forecast_total > 0, obs / np.maximum(forecast_total, 1e-9), 1.0)
        scale = np.clip(scale, 0.3, 3.0)

        lam_updated = self.lambda_full.copy()
        lam_updated[:, t_window:] *= scale[:, None]
        return lam_updated

    # ------------------------------------------------------------------
    # Re-plan
    # ------------------------------------------------------------------
    def _replan(
        self,
        env_now: float,
        banks:   list["BloodBank"],
        metrics: "MetricsCollector",
    ) -> None:
        t_window = min(int(env_now / 3600), self.rolling_lp.T - 1)
        L        = min(self.rolling_lp.horizon_hours,
                       self.rolling_lp.T - t_window)
        if L <= 0:
            return

        I_now = np.array(
            [sum(b.inventory.values()) for b in banks], dtype=float
        )

        lam_adapted = self._online_lambda(t_window, metrics)
        lam_horizon = lam_adapted[:, t_window:t_window + L]

        new_plan = self.rolling_lp.solve_window(
            t_start        = t_window,
            lambda_horizon = lam_horizon,
            d_matrix       = self.d_matrix_min,
            I_remaining    = I_now,
        )
        for k in range(t_window, t_window + L):
            for b in range(self.rolling_lp.B):
                for h in range(self.rolling_lp.H):
                    self._x_sol[(b, h, k)] = new_plan.get((b, h, k), 0)

        # Reset consumed budget whenever the plan is refreshed
        self._consumed = {}
        self._next_replan_s = env_now + self.replan_iv_s

    # ------------------------------------------------------------------
    # Dispatch entry point
    # ------------------------------------------------------------------
    def dispatch(
        self,
        env_now:  float,
        hospitals: list["Hospital"],
        drones:    list["Drone"],
        banks:     list["BloodBank"],
        d_matrix:  np.ndarray,
        metrics:   "MetricsCollector",
    ) -> list[tuple["Drone", "Request", "BloodBank"]]:

        if env_now < self._last_env_now:
            self._x_sol         = {}
            self._next_replan_s = 0.0
        self._last_env_now = env_now

        if env_now >= self._next_replan_s:
            self._replan(env_now, banks, metrics)

        t_window = min(int(env_now / 3600), 23)

        pending: list[Request] = []
        for h in hospitals:
            pending.extend(h.pending_requests)

        active: list[Request] = []
        for req in list(pending):
            if req.expiration_time <= env_now:
                req.is_expired = True
                hospitals[req.hospital_id].pending_requests.remove(req)
                metrics.log_expiration(req)
            else:
                active.append(req)

        if not active:
            return []

        idle_drones = [d for d in drones if d.state == DroneState.IDLE]
        if not idle_drones:
            return []

        active.sort(key=lambda r: (-int(r.priority), r.expiration_time))

        assignments: list[tuple["Drone", "Request", "BloodBank"]] = []
        remaining   = list(idle_drones)

        for req in active:
            if not remaining:
                break

            h  = req.hospital_id
            bt = req.blood_type

            def remaining_budget(b: int) -> int:
                planned = int(self._x_sol.get((b, h, t_window), 0))
                used    = self._consumed.get((b, h, t_window), 0)
                return max(0, planned - used)

            def has_stock(b: int) -> bool:
                return banks[b].inventory.get(bt, 0) >= req.units_needed

            def stock_dist_score(b: int) -> float:
                bank  = banks[b]
                stock = bank.inventory.get(bt, 0)
                cap   = max(1, bank.initial_per_type)
                dist  = max(0.5, float(d_matrix[b, h]))
                return stock / (cap * dist)

            n_banks = len(banks)
            preferred = [b for b in range(n_banks)
                         if remaining_budget(b) > 0 and has_stock(b)]
            preferred.sort(
                key=lambda b: (-remaining_budget(b), -stock_dist_score(b))
            )
            preferred_set = set(preferred)
            fallback = [b for b in range(n_banks)
                        if b not in preferred_set and has_stock(b)]
            fallback.sort(key=lambda b: -stock_dist_score(b))

            chosen_drone: "Drone | None" = None
            chosen_bank:  "BloodBank | None" = None
            chosen_b:     int = -1

            # Pooled fleet: any idle drone may serve from any bank
            sorted_rem = sorted(remaining, key=lambda d: d.battery, reverse=True)
            for b in preferred + fallback:
                bank = banks[b]
                d_km = float(d_matrix[b, h])
                for drone in sorted_rem:
                    if _feasibility_check(drone, req, bank, d_km, env_now):
                        chosen_drone = drone
                        chosen_bank  = bank
                        chosen_b     = b
                        break
                if chosen_drone is not None:
                    break

            if chosen_drone is None:
                continue

            if chosen_b in preferred_set:
                key = (chosen_b, h, t_window)
                self._consumed[key] = self._consumed.get(key, 0) + 1

            assignments.append((chosen_drone, req, chosen_bank))
            remaining.remove(chosen_drone)
            hospitals[req.hospital_id].pending_requests.remove(req)

        return assignments
