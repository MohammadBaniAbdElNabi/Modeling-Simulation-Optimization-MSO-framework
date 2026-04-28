"""RandomDispatch: select drones uniformly at random with inventory fallback."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.simulation.dispatch.base_policy import BaseDispatch
from src.simulation.entities import DroneState, Request

if TYPE_CHECKING:
    from src.simulation.entities import BloodBank, Drone, Hospital
    from src.simulation.metrics_collector import MetricsCollector


class RandomDispatch(BaseDispatch):
    """Policy 1 — Random Dispatch (Section 3.8.1 of the spec)."""

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng

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

        # Collect all pending requests sorted by priority (high first), then FIFO
        pending: list[Request] = []
        for h in hospitals:
            pending.extend(h.pending_requests)
        pending.sort(key=lambda r: (-int(r.priority), r.arrival_time))

        idle_drones = [d for d in drones if d.state == DroneState.IDLE]

        for req in list(pending):
            # Check expiration
            if req.expiration_time <= env_now:
                req.is_expired = True
                hospitals[req.hospital_id].pending_requests.remove(req)
                metrics.log_expiration(req)
                continue

            if not idle_drones:
                break

            # Select drone uniformly at random
            idx = self.rng.integers(0, len(idle_drones))
            drone = idle_drones[idx]
            bank = banks[drone.home_bank]

            if bank.has_stock(req.blood_type, req.units_needed):
                assignments.append((drone, req, bank))
                idle_drones.remove(drone)
                hospitals[req.hospital_id].pending_requests.remove(req)
            else:
                # Try other banks in random order
                other_bank_ids = [b for b in range(len(banks)) if b != drone.home_bank]
                self.rng.shuffle(other_bank_ids)
                dispatched = False
                for alt_bid in other_bank_ids:
                    alt_bank = banks[alt_bid]
                    if not alt_bank.has_stock(req.blood_type, req.units_needed):
                        continue
                    alt_idle = [d for d in idle_drones if d.home_bank == alt_bid]
                    if not alt_idle:
                        continue
                    alt_drone = alt_idle[self.rng.integers(0, len(alt_idle))]
                    assignments.append((alt_drone, req, alt_bank))
                    idle_drones.remove(alt_drone)
                    hospitals[req.hospital_id].pending_requests.remove(req)
                    dispatched = True
                    break
                if not dispatched:
                    # inventory exhausted across all banks -- mark unfulfilled
                    pass  # leave in queue; will expire naturally

        return assignments
