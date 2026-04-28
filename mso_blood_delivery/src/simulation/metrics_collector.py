"""MetricsCollector: event logging and aggregate statistics for one replication."""
from __future__ import annotations

import numpy as np

from src.simulation.entities import Priority, Request


class MetricsCollector:
    """Accumulate events during a simulation run and compute metrics at end."""

    def __init__(self, n_hospitals: int) -> None:
        self.n_hospitals = n_hospitals
        self._arrivals: list[Request] = []
        self._delivered: list[Request] = []
        self._expired: list[Request] = []
        self._assigned: list[Request] = []  # requests that got a drone assignment

    def log_arrival(self, req: Request) -> None:
        self._arrivals.append(req)

    def log_delivery(self, req: Request) -> None:
        self._delivered.append(req)

    def log_expiration(self, req: Request) -> None:
        self._expired.append(req)

    def log_assignment(self, req: Request) -> None:
        self._assigned.append(req)

    def n_total(self) -> int:
        return len(self._arrivals)

    def n_completed(self) -> int:
        return len(self._delivered)

    def n_expired(self) -> int:
        return len(self._expired)

    def compute(self) -> dict[str, float]:
        """Return aggregate metrics dict."""
        n_total = self.n_total()
        n_completed = self.n_completed()
        n_expired = self.n_expired()

        fr = (n_completed / n_total * 100.0) if n_total > 0 else 0.0
        err = (n_expired / n_total * 100.0) if n_total > 0 else 0.0

        delivery_times_min: list[float] = []
        for req in self._delivered:
            dt = (req.delivery_time - req.arrival_time) / 60.0
            if dt >= 0:
                delivery_times_min.append(dt)

        adt = float(np.mean(delivery_times_min)) if delivery_times_min else float("nan")
        p95 = float(np.percentile(delivery_times_min, 95)) if delivery_times_min else float("nan")

        wait_times_min: list[float] = []
        for req in self._assigned:
            if not np.isnan(req.assignment_time):
                wt = (req.assignment_time - req.arrival_time) / 60.0
                if wt >= 0:
                    wait_times_min.append(wt)

        awt = float(np.mean(wait_times_min)) if wait_times_min else float("nan")
        dph = n_completed / 24.0

        # Per-hospital fulfillment
        per_hospital: dict[str, float] = {}
        for h in range(self.n_hospitals):
            arr_h = [r for r in self._arrivals if r.hospital_id == h]
            del_h = [r for r in self._delivered if r.hospital_id == h]
            per_hospital[f"FR_H{h+1:02d}"] = (
                len(del_h) / len(arr_h) * 100.0 if arr_h else float("nan")
            )

        # Per-priority fulfillment
        per_priority: dict[str, float] = {}
        for p in Priority:
            arr_p = [r for r in self._arrivals if r.priority == p]
            del_p = [r for r in self._delivered if r.priority == p]
            per_priority[f"FR_{p.name}"] = (
                len(del_p) / len(arr_p) * 100.0 if arr_p else float("nan")
            )

        return {
            "FR": fr,
            "ADT": adt,
            "P95": p95,
            "ERR": err,
            "AWT": awt,
            "deliveries_per_hour": dph,
            **per_hospital,
            **per_priority,
        }
