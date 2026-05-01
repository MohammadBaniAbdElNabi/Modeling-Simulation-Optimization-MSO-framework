"""RandomDispatch: random bank selection from the pooled fleet."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.simulation.dispatch.base_policy import BaseDispatch
from src.simulation.dispatch.greedy_policy import _feasibility_check
from src.simulation.entities import DroneState, Request

if TYPE_CHECKING:
    from src.simulation.entities import BloodBank, Drone, Hospital
    from src.simulation.metrics_collector import MetricsCollector


class RandomDispatch(BaseDispatch):
    """Policy 1 — Random Dispatch.

    Selects a bank uniformly at random from those with stock, then assigns
    the highest-battery feasible idle drone to that bank.  Any idle drone
    may serve any bank (pooled fleet).  Falls back to the next random bank
    until a feasible assignment is found or all banks are exhausted.
    """

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

            if not idle_drones:
                break

            h  = req.hospital_id
            bt = req.blood_type

            # Shuffle banks for random selection order
            bank_ids = list(range(len(banks)))
            self.rng.shuffle(bank_ids)

            # Sort idle drones by battery descending for feasibility
            sorted_idles = sorted(idle_drones, key=lambda d: d.battery, reverse=True)

            chosen_drone: "Drone | None" = None
            chosen_bank:  "BloodBank | None" = None

            for b in bank_ids:
                bank = banks[b]
                if not bank.has_stock(bt, req.units_needed):
                    continue
                d_km = float(d_matrix[b, h])
                for drone in sorted_idles:
                    if _feasibility_check(drone, req, bank, d_km, env_now):
                        chosen_drone = drone
                        chosen_bank  = bank
                        break
                if chosen_drone is not None:
                    break

            if chosen_drone is None:
                continue  # no feasible assignment; request stays in queue

            assignments.append((chosen_drone, req, chosen_bank))
            idle_drones.remove(chosen_drone)
            hospitals[req.hospital_id].pending_requests.remove(req)

        return assignments
