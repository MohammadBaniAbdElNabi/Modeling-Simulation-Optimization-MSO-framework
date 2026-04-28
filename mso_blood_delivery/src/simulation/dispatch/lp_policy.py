"""LPOptimizedDispatch: use LP assignment matrix with greedy fallback."""
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

SENTINEL = -1


def _nearest_feasible(
    req: Request,
    idle_drones: list["Drone"],
    banks: list["BloodBank"],
    d_matrix: np.ndarray,
    env_now: float,
) -> tuple["Drone | None", "BloodBank | None"]:
    """Return (drone, bank) with minimum flight time across all idle drones.

    Implements the NearestFeasibleGreedy sub-routine referenced in Section 3.8.3:
    'Fall back to nearest-feasible greedy across all banks'.
    """
    best_drone: "Drone | None" = None
    best_bank: "BloodBank | None" = None
    best_time = float("inf")

    for drone in idle_drones:
        b = drone.home_bank
        bank = banks[b]
        d_km = float(d_matrix[b, req.hospital_id])
        if not _feasibility_check(drone, req, bank, d_km, env_now):
            continue
        t_est = d_km / drone.speed_kmh * 60.0  # one-way flight time (min)
        if t_est < best_time:
            best_time = t_est
            best_drone = drone
            best_bank = bank

    return best_drone, best_bank


class LPOptimizedDispatch(BaseDispatch):
    """Policy 3 — LP-Optimized Dispatch (Section 3.8.3).

    Uses the precomputed x_sol[(b,h,t)] assignment matrix as a lookup table.
    Falls back to nearest-feasible greedy when preferred bank is infeasible or
    LP window is marked sentinel (-1).
    """

    def __init__(self, x_sol: dict[tuple[int, int, int], int]) -> None:
        self.x_sol = x_sol

    def _best_feasible_at_bank(
        self,
        bank_id: int,
        req: Request,
        idle_drones: list["Drone"],
        banks: list["BloodBank"],
        d_matrix: np.ndarray,
        env_now: float,
    ) -> tuple["Drone | None", "BloodBank | None"]:
        """Return highest-battery feasible drone at a specific bank.

        Subroutine from Section 3.8.3: sort candidates by battery descending,
        return first feasible.
        """
        bank = banks[bank_id]
        candidates = [d for d in idle_drones if d.home_bank == bank_id]
        candidates.sort(key=lambda d: d.battery, reverse=True)
        for drone in candidates:
            d_km = float(d_matrix[bank_id, req.hospital_id])
            if _feasibility_check(drone, req, bank, d_km, env_now):
                return drone, bank
        return None, None

    def dispatch(
        self,
        env_now: float,
        hospitals: list["Hospital"],
        drones: list["Drone"],
        banks: list["BloodBank"],
        d_matrix: np.ndarray,
        metrics: "MetricsCollector",
    ) -> list[tuple["Drone", "Request", "BloodBank"]]:
        assignments: list[tuple["Drone", "Request", "BloodBank"]] = []

        pending: list[Request] = []
        for h in hospitals:
            pending.extend(h.pending_requests)
        pending.sort(key=lambda r: (-int(r.priority), r.arrival_time))

        idle_drones = [d for d in drones if d.state == DroneState.IDLE]
        t_window = min(int(env_now / 3600), 23)

        for req in list(pending):
            if req.expiration_time <= env_now:
                req.is_expired = True
                hospitals[req.hospital_id].pending_requests.remove(req)
                metrics.log_expiration(req)
                continue

            h = req.hospital_id

            # LP-preferred bank: argmax_b x[b, h, t]  (Section 3.8.3)
            preferred_bank_id = max(
                range(len(banks)),
                key=lambda b: self.x_sol.get((b, h, t_window), 0),
            )
            lp_val = self.x_sol.get((preferred_bank_id, h, t_window), 0)
            lp_window_ok = lp_val != SENTINEL

            chosen_drone: "Drone | None" = None
            chosen_bank: "BloodBank | None" = None

            # Try LP-preferred bank first
            if lp_window_ok:
                chosen_drone, chosen_bank = self._best_feasible_at_bank(
                    preferred_bank_id, req, idle_drones, banks, d_matrix, env_now
                )

            # Fall back to nearest-feasible greedy across all banks (Section 3.8.3)
            if chosen_drone is None:
                chosen_drone, chosen_bank = _nearest_feasible(
                    req, idle_drones, banks, d_matrix, env_now
                )

            if chosen_drone is None:
                req.is_expired = True
                hospitals[req.hospital_id].pending_requests.remove(req)
                metrics.log_expiration(req)
                continue

            assignments.append((chosen_drone, req, chosen_bank))
            idle_drones.remove(chosen_drone)
            hospitals[req.hospital_id].pending_requests.remove(req)

        return assignments
