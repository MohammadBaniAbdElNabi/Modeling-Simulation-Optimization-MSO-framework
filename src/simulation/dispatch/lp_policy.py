"""LPOptimizedDispatch: x_sol-faithful bank selection with greedy fallback."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.simulation.dispatch.base_policy import BaseDispatch
from src.simulation.dispatch.greedy_policy import _feasibility_check
from src.simulation.entities import DroneState, Request
from src.utils.logging_utils import get_logger

if TYPE_CHECKING:
    from src.simulation.entities import BloodBank, Drone, Hospital
    from src.simulation.metrics_collector import MetricsCollector

logger = get_logger(__name__)


class LPOptimizedDispatch(BaseDispatch):
    """Policy 3 — LP-Optimized Dispatch.

    The LP plan ``x_sol[(b, h, t)]`` is the primary bank selector:

      1. Maintain a remaining-budget tracker
         ``remaining[b,h,t] = x_sol[b,h,t] - consumed[b,h,t]``.
      2. For each pending request at hospital h in window t, only consider
         banks whose remaining budget is > 0 ("preferred set").  Among the
         preferred set, choose the bank with the largest remaining budget
         (ties broken by stock-distance ratio).  Decrement that bank's
         budget on selection.
      3. If no preferred bank is feasible (insufficient stock, no idle drone,
         or budget exhausted), fall back to nearest-feasible greedy
         (stock / (cap × distance)) so the simulation never starves.

    This makes the LP plan operationally distinct from Greedy: LP-Global
    spreads load uniformly across (h, t); LP-Hospital concentrates volume on
    high-volume hospitals; LP-SARIMA shifts capacity into peak hours.

    Replication boundaries are detected by ``env_now`` going backwards
    (each new ``simpy.Environment`` starts at 0); the consumption tracker
    is cleared automatically — no external ``reset()`` call needed.
    """

    def __init__(self, x_sol: dict[tuple[int, int, int], int]) -> None:
        self.x_sol = x_sol
        self._consumed: dict[tuple[int, int, int], int] = {}
        self._last_env_now: float = float("inf")

    def reset(self) -> None:
        """Clear the per-replication consumption tracker."""
        self._consumed = {}
        self._last_env_now = float("inf")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def dispatch(
        self,
        env_now: float,
        hospitals: list["Hospital"],
        drones: list["Drone"],
        banks: list["BloodBank"],
        d_matrix: np.ndarray,
        metrics: "MetricsCollector",
    ) -> list[tuple["Drone", "Request", "BloodBank"]]:

        # Auto-reset consumption tracker at the start of each replication
        # (simpy clocks restart at 0 for every new Environment).
        if env_now < self._last_env_now:
            self._consumed = {}
        self._last_env_now = env_now

        t_window = min(int(env_now / 3600), 23)

        # ── Collect pending; expire stale requests ────────────────────
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

        # Urgency-first ordering: priority DESC, expiration ASC
        active.sort(key=lambda r: (-int(r.priority), r.expiration_time))

        assignments: list[tuple["Drone", "Request", "BloodBank"]] = []
        remaining = list(idle_drones)

        for req in active:
            if not remaining:
                break

            h  = req.hospital_id
            bt = req.blood_type

            # ── Helpers (closures over req, h, bt, t_window) ──────────
            def remaining_budget(b: int) -> int:
                planned = int(self.x_sol.get((b, h, t_window), 0))
                used    = self._consumed.get((b, h, t_window), 0)
                return max(0, planned - used)

            def has_stock(b: int) -> bool:
                return banks[b].inventory.get(bt, 0) >= req.units_needed

            def stock_distance_score(b: int) -> float:
                bank  = banks[b]
                stock = bank.inventory.get(bt, 0)
                cap   = max(1, bank.initial_per_type)
                dist  = max(0.5, float(d_matrix[b, h]))
                return stock / (cap * dist)

            # ── Build bank ranking: LP-preferred first, greedy fallback ──
            n_banks = len(banks)
            all_b   = list(range(n_banks))

            preferred = [b for b in all_b if remaining_budget(b) > 0 and has_stock(b)]
            preferred.sort(
                key=lambda b: (-remaining_budget(b), -stock_distance_score(b))
            )
            preferred_set = set(preferred)

            fallback = [b for b in all_b
                        if b not in preferred_set and has_stock(b)]
            fallback.sort(key=lambda b: -stock_distance_score(b))

            bank_order = preferred + fallback

            chosen_drone: "Drone | None" = None
            chosen_bank:  "BloodBank | None" = None
            chosen_b_idx: int = -1

            for b in bank_order:
                bank   = banks[b]
                d_km   = float(d_matrix[b, h])
                # Pooled fleet: any idle drone may serve from any bank.
                candidates = sorted(
                    remaining,
                    key=lambda d: d.battery,
                    reverse=True,
                )
                for drone in candidates:
                    if _feasibility_check(drone, req, bank, d_km, env_now):
                        chosen_drone = drone
                        chosen_bank  = bank
                        chosen_b_idx = b
                        break
                if chosen_drone is not None:
                    break

            if chosen_drone is None:
                continue  # no feasible bank/drone; request stays pending

            # If chosen from the preferred (LP-planned) set, decrement budget
            if chosen_b_idx in preferred_set:
                key = (chosen_b_idx, h, t_window)
                self._consumed[key] = self._consumed.get(key, 0) + 1

            assignments.append((chosen_drone, req, chosen_bank))
            remaining.remove(chosen_drone)
            hospitals[req.hospital_id].pending_requests.remove(req)

        return assignments
