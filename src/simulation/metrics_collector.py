"""MetricsCollector: significance-focused event logging and metrics."""
from __future__ import annotations

import numpy as np

from src.simulation.entities import Priority, Request


# Priority weights for cost / weighted-FR computation:
# emergency = 3 normals, urgent = 2 normals
PRIORITY_WEIGHT: dict[int, int] = {
    int(Priority.NORMAL):    1,
    int(Priority.URGENT):    2,
    int(Priority.EMERGENCY): 3,
}

# Peak-hour band — where saturation bites and where MPC's hedging matters
PEAK_HOURS_DEFAULT: tuple[int, int] = (12, 17)   # inclusive lower, inclusive upper


class MetricsCollector:
    """Accumulate events during one simulation replication and compute
    significance-focused metrics at end.
    """

    def __init__(
        self,
        n_hospitals: int,
        n_windows: int = 24,
        peak_hours:  tuple[int, int] = PEAK_HOURS_DEFAULT,
    ) -> None:
        self.n_hospitals = n_hospitals
        self.n_windows   = n_windows
        self.peak_lo, self.peak_hi = peak_hours
        self._arrivals:  list[Request] = []
        self._delivered: list[Request] = []
        self._expired:   list[Request] = []
        self._assigned:  list[Request] = []

    # ------------------------------------------------------------------ event log
    def log_arrival(self, req: Request) -> None:
        self._arrivals.append(req)

    def log_delivery(self, req: Request) -> None:
        self._delivered.append(req)

    def log_expiration(self, req: Request) -> None:
        self._expired.append(req)

    def log_assignment(self, req: Request) -> None:
        self._assigned.append(req)

    # ------------------------------------------------------------------ counts
    def n_total(self) -> int:
        return len(self._arrivals)

    def n_completed(self) -> int:
        return len(self._delivered)

    def n_expired(self) -> int:
        return len(self._expired)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _window_of(time_seconds: float, n_windows: int) -> int:
        return min(int(time_seconds / 3600.0), n_windows - 1)

    def _is_peak(self, t_seconds: float) -> bool:
        h = int(t_seconds / 3600.0)
        return self.peak_lo <= h <= self.peak_hi

    @staticmethod
    def _weight(req: Request) -> int:
        return PRIORITY_WEIGHT.get(int(req.priority), 1)

    # ------------------------------------------------------------------ compute
    def compute(self) -> dict[str, float]:
        n_total     = self.n_total()
        n_completed = self.n_completed()
        n_expired   = self.n_expired()

        fr  = (n_completed / n_total * 100.0) if n_total > 0 else 0.0
        err = (n_expired   / n_total * 100.0) if n_total > 0 else 0.0

        delivery_times_min = [
            (r.delivery_time - r.arrival_time) / 60.0
            for r in self._delivered if r.delivery_time >= r.arrival_time
        ]
        adt = float(np.mean(delivery_times_min))      if delivery_times_min else float("nan")
        p95 = float(np.percentile(delivery_times_min, 95)) if delivery_times_min else float("nan")

        wait_times_min = [
            (r.assignment_time - r.arrival_time) / 60.0
            for r in self._assigned
            if not np.isnan(r.assignment_time) and r.assignment_time >= r.arrival_time
        ]
        awt = float(np.mean(wait_times_min)) if wait_times_min else float("nan")
        dph = n_completed / 24.0

        # ----- priority-weighted FR (emergency = 3 normals, urgent = 2) ----
        w_total     = sum(self._weight(r) for r in self._arrivals)
        w_delivered = sum(self._weight(r) for r in self._delivered)
        fr_weighted = (w_delivered / w_total * 100.0) if w_total > 0 else 0.0

        # ----- expiration cost (priority-weighted count of expirations) ----
        expiration_cost = float(sum(self._weight(r) for r in self._expired))

        # ----- peak-hour metrics (default hours 12–17) ----------------------
        peak_arr = [r for r in self._arrivals  if self._is_peak(r.arrival_time)]
        peak_del = [r for r in self._delivered if self._is_peak(r.arrival_time)]
        peak_exp = [r for r in self._expired   if self._is_peak(r.arrival_time)]
        n_peak   = len(peak_arr)
        fr_peak  = (len(peak_del) / n_peak * 100.0) if n_peak > 0 else float("nan")
        err_peak = (len(peak_exp) / n_peak * 100.0) if n_peak > 0 else float("nan")

        # ----- per-hospital fulfillment ------------------------------------
        per_hospital: dict[str, float] = {}
        for h in range(self.n_hospitals):
            arr_h = [r for r in self._arrivals  if r.hospital_id == h]
            del_h = [r for r in self._delivered if r.hospital_id == h]
            per_hospital[f"FR_H{h+1:02d}"] = (
                len(del_h) / len(arr_h) * 100.0 if arr_h else float("nan")
            )

        # ----- per-priority fulfillment ------------------------------------
        per_priority: dict[str, float] = {}
        for p in Priority:
            arr_p = [r for r in self._arrivals  if r.priority == p]
            del_p = [r for r in self._delivered if r.priority == p]
            per_priority[f"FR_{p.name}"] = (
                len(del_p) / len(arr_p) * 100.0 if arr_p else float("nan")
            )

        # ----- per-window (T=24) ------------------------------------------
        per_window: dict[str, float] = {}
        arrivals_t  = [0] * self.n_windows
        delivered_t = [0] * self.n_windows
        expired_t   = [0] * self.n_windows
        for r in self._arrivals:
            arrivals_t[self._window_of(r.arrival_time, self.n_windows)] += 1
        for r in self._delivered:
            delivered_t[self._window_of(r.arrival_time, self.n_windows)] += 1
        for r in self._expired:
            expired_t[self._window_of(r.arrival_time, self.n_windows)] += 1
        for t in range(self.n_windows):
            if arrivals_t[t] > 0:
                per_window[f"ERR_T{t:02d}"] = expired_t[t]   / arrivals_t[t] * 100.0
                per_window[f"FR_T{t:02d}"]  = delivered_t[t] / arrivals_t[t] * 100.0
            else:
                per_window[f"ERR_T{t:02d}"] = float("nan")
                per_window[f"FR_T{t:02d}"]  = float("nan")

        return {
            "FR":             fr,
            "FR_weighted":    fr_weighted,
            "ADT":            adt,
            "P95":            p95,
            "ERR":            err,
            "ERR_peak":       err_peak,
            "FR_peak":        fr_peak,
            "Expiration_cost": expiration_cost,
            "AWT":            awt,
            "deliveries_per_hour": dph,
            "n_total":        n_total,
            **per_hospital,
            **per_priority,
            **per_window,
        }
