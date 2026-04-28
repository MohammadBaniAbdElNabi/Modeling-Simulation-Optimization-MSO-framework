"""NearestFeasibleGreedy: dispatch the closest feasible idle drone."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.simulation.dispatch.base_policy import BaseDispatch
from src.simulation.entities import DroneState, Request

if TYPE_CHECKING:
    from src.simulation.entities import BloodBank, Drone, Hospital
    from src.simulation.metrics_collector import MetricsCollector


def _feasibility_check(
    drone: "Drone",
    req: Request,
    bank: "BloodBank",
    d_km: float,
    env_now: float,
    speed_kmh: float = 50.0,
    loading_min: float = 2.0,
) -> bool:
    """Return True if drone can feasibly serve req from bank."""
    # Battery check
    delta_B = drone.battery_drain * 2 * d_km
    if drone.battery - delta_B < drone.battery_min:
        return False
    # Payload check
    if req.units_needed > drone.max_payload:
        return False
    # Inventory check
    if not bank.has_stock(req.blood_type, req.units_needed):
        return False
    # Delivery time window check
    t_est_s = (d_km / speed_kmh * 60.0 + loading_min) * 60.0
    if env_now + t_est_s > req.expiration_time:
        return False
    return True


class NearestFeasibleGreedy(BaseDispatch):
    """Policy 2 — Nearest-Feasible Greedy Dispatch (Section 3.8.2)."""

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

        for req in list(pending):
            if req.expiration_time <= env_now:
                req.is_expired = True
                hospitals[req.hospital_id].pending_requests.remove(req)
                metrics.log_expiration(req)
                continue

            best_drone: "Drone | None" = None
            best_bank: "BloodBank | None" = None
            best_time = float("inf")

            for drone in idle_drones:
                b = drone.home_bank
                h = req.hospital_id
                d_km = float(d_matrix[b, h])
                bank = banks[b]
                if not _feasibility_check(drone, req, bank, d_km, env_now):
                    continue
                t_est = d_km / drone.speed_kmh * 60.0 + 2.0  # flight + loading (min)
                if t_est < best_time:
                    best_time = t_est
                    best_drone = drone
                    best_bank = bank

            if best_drone is None:
                # No feasible drone -- expire the request
                req.is_expired = True
                hospitals[req.hospital_id].pending_requests.remove(req)
                metrics.log_expiration(req)
                continue

            assignments.append((best_drone, req, best_bank))
            idle_drones.remove(best_drone)
            hospitals[req.hospital_id].pending_requests.remove(req)

        return assignments
