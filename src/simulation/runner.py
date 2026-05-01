"""SimulationRunner: orchestrates replications for all three policies."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import simpy

from src.data_gen.blood_types import BLOOD_TYPES
from src.simulation.dispatch.base_policy import BaseDispatch
from src.simulation.dispatch.greedy_policy import NearestFeasibleGreedy
from src.simulation.dispatch.lp_policy import LPOptimizedDispatch
from src.simulation.dispatch.random_policy import RandomDispatch
from src.simulation.entities import BloodBank, Drone, Hospital, Request
from src.simulation.metrics_collector import MetricsCollector
from src.simulation.processes import (
    dispatch_cycle,
    expiration_monitor,
    inventory_replenishment,
    request_generator,
)
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

_COL_ORDER = [
    "rep_id", "seed", "demand_scale",
    "FR", "FR_weighted", "ADT", "P95",
    "ERR", "ERR_peak", "FR_peak", "Expiration_cost",
    "AWT", "deliveries_per_hour", "n_total",
]


def _run_single(
    seed: int,
    lambda_hat: np.ndarray,
    d_km: np.ndarray,
    sim_cfg: dict[str, Any],
    policy: BaseDispatch,
    demand_scale: float = 1.0,
) -> dict[str, float]:
    """Run one simulation replication with a given policy and seed.

    Parameters
    ----------
    d_km : shape (B, H), distances in **kilometres** — used for battery drain
           (beta %/km * km) and flight-time computation (km / speed_kmh).
           Do NOT pass flight-time minutes; the simulation derives all timing
           and battery values from km internally.
    """
    # Unpack config
    H = int(sim_cfg.get("H", 12))
    B = int(sim_cfg.get("B", 3))
    fleet_size = int(sim_cfg.get("drone", {}).get("fleet_size", 8))
    speed = float(sim_cfg.get("drone", {}).get("speed_kmh", 50.0))
    battery_drain = float(sim_cfg.get("drone", {}).get("battery_drain", 1.5))
    battery_min = float(sim_cfg.get("drone", {}).get("battery_min", 30.0))
    recharge_rate = float(sim_cfg.get("drone", {}).get("recharge_rate", 20.0))
    max_payload = int(sim_cfg.get("drone", {}).get("max_payload", 4))
    service_min = float(sim_cfg.get("timing", {}).get("service_time_min", 3.0))
    loading_min = float(sim_cfg.get("timing", {}).get("loading_time_min", 2.0))
    dispatch_s = float(sim_cfg.get("timing", {}).get("dispatch_cycle_s", 30.0))
    expiry_min = float(sim_cfg.get("timing", {}).get("expiration_min", 15.0))
    horizon_h = float(sim_cfg.get("timing", {}).get("horizon_hours", 24.0))
    horizon_s = horizon_h * 3600.0
    init_per_type = int(sim_cfg.get("inventory", {}).get("initial_per_bank_per_type", 50))
    units_min = int(sim_cfg.get("requests", {}).get("units_min", 1))
    units_max = int(sim_cfg.get("requests", {}).get("units_max", 4))

    Request.reset_counter()
    env = simpy.Environment()

    # Create entities
    hospitals = [Hospital(hospital_id=h) for h in range(H)]
    banks = [
        BloodBank(bank_id=b, blood_types=BLOOD_TYPES, initial_per_type=init_per_type)
        for b in range(B)
    ]

    # Assign drones round-robin to banks
    drones = [
        Drone(
            drone_id=i,
            home_bank=i % B,
            speed_kmh=speed,
            battery_drain=battery_drain,
            battery_min=battery_min,
            recharge_rate=recharge_rate,
            max_payload=max_payload,
        )
        for i in range(fleet_size)
    ]

    metrics = MetricsCollector(n_hospitals=H)

    # CRN: each hospital gets a deterministic RNG derived from the replication seed.
    # Arrivals are driven by lambda_hat * demand_scale (stress-sweep dial).
    lambda_eff = lambda_hat * float(demand_scale)
    for h in range(H):
        hosp_rng = np.random.default_rng(seed * 1000 + h)
        env.process(
            request_generator(
                env, hospitals[h], lambda_eff[h],
                expiry_min, units_min, units_max,
                hosp_rng, metrics, horizon_s,
            )
        )

    env.process(expiration_monitor(env, hospitals, dispatch_s, metrics, horizon_s))

    # Inventory replenishment (R3 realism — lifts the saturation ceiling)
    repl_cfg = sim_cfg.get("replenishment", {}) or {}
    if bool(repl_cfg.get("enabled", False)):
        schedule        = list(repl_cfg.get("schedule_hours", []))
        amount          = int(repl_cfg.get("amount_per_type", 0))
        delay_stdev_min = float(repl_cfg.get("delay_stdev_min", 0.0))
        if schedule and amount > 0:
            repl_rng = np.random.default_rng(seed * 997 + 13)
            env.process(
                inventory_replenishment(
                    env, banks, schedule, amount,
                    BLOOD_TYPES, horizon_s,
                    rng=repl_rng, delay_stdev_min=delay_stdev_min,
                )
            )

    # d_km is passed directly; processes.py and dispatch policies use km distances
    env.process(
        dispatch_cycle(
            env, hospitals, drones, banks,
            d_km, policy,
            loading_min, service_min,
            dispatch_s, metrics, horizon_s,
        )
    )

    env.run(until=horizon_s)

    # Count only requests whose deadline genuinely passed by horizon end.
    # Requests that arrived just before horizon_s but whose expiration_time
    # is still in the future are abandoned (neither served nor expired) — do
    # NOT force-expire them, which would inflate the t=23 ERR artifact.
    for hosp in hospitals:
        for req in list(hosp.pending_requests):
            if not req.is_expired and not req.is_fulfilled:
                if req.expiration_time <= horizon_s:
                    req.is_expired = True
                    metrics.log_expiration(req)
        hosp.pending_requests.clear()

    return metrics.compute()


def _df_from_rows(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    extra = [c for c in df.columns if c not in _COL_ORDER]
    return df[_COL_ORDER + extra]


class SimulationRunner:
    """Orchestrate replications for one or all three dispatch policies.

    Parameters
    ----------
    sim_cfg    : simulation configuration dict
    lambda_hat : shape (H, T) forecasted arrival rates
    d_km       : shape (B, H) distance matrix in **kilometres** — the simulation
                 derives flight times and battery drain from km values.
                 (The LP solver uses flight-time minutes; pass km only here.)
    x_sol      : LP assignment dict; required for LP-Optimized policy
    """

    def __init__(
        self,
        sim_cfg: dict[str, Any],
        lambda_hat: np.ndarray,
        d_km: np.ndarray,
        x_sol: dict | None = None,
    ) -> None:
        self.sim_cfg = sim_cfg
        self.lambda_hat = lambda_hat
        self.d_km = d_km
        self.x_sol = x_sol
        self.sim_cfg.setdefault("H", 12)
        self.sim_cfg.setdefault("B", 3)

    def run(
        self,
        policy: BaseDispatch,
        n_replications: int = 20,
        seeds: list[int] | None = None,
        demand_scale: float = 1.0,
    ) -> pd.DataFrame:
        """Run n_replications with the given policy under CRN seeds.

        Parameters
        ----------
        demand_scale : multiplicative factor applied to ``lambda_hat`` to
            drive arrivals (stress-sweep dial).  ``1.0`` reproduces baseline.
        """
        if seeds is None:
            seeds = list(range(1, n_replications + 1))

        rows = []
        for rep_idx, seed in enumerate(seeds):
            if hasattr(policy, "reset"):
                policy.reset()
            logger.info("  Rep %02d/%02d (seed=%d, demand_scale=%.2f)",
                        rep_idx + 1, n_replications, seed, demand_scale)
            result = _run_single(seed, self.lambda_hat, self.d_km,
                                 self.sim_cfg, policy, demand_scale=demand_scale)
            result["rep_id"]       = rep_idx + 1
            result["seed"]         = seed
            result["demand_scale"] = demand_scale
            rows.append(result)

        return _df_from_rows(rows)

    def run_all_policies(self, n_replications: int = 20) -> dict[str, pd.DataFrame]:
        """Run all three policies with shared CRN seeds 1..n_replications.

        Returns {'random': df, 'greedy': df, 'lp': df}
        """
        seeds = list(range(1, n_replications + 1))
        results: dict[str, pd.DataFrame] = {}

        logger.info("Running policy: random")
        rows = []
        for rep_idx, seed in enumerate(seeds):
            logger.info("  Rep %02d/%02d (seed=%d)", rep_idx + 1, n_replications, seed)
            # New RandomDispatch instance per replication so RNG state is seeded correctly
            pol = RandomDispatch(np.random.default_rng(seed))
            result = _run_single(seed, self.lambda_hat, self.d_km, self.sim_cfg, pol)
            result["rep_id"] = rep_idx + 1
            result["seed"] = seed
            rows.append(result)
        results["random"] = _df_from_rows(rows)

        logger.info("Running policy: greedy")
        results["greedy"] = self.run(NearestFeasibleGreedy(), n_replications, seeds)

        logger.info("Running policy: lp")
        results["lp"] = self.run(
            LPOptimizedDispatch(self.x_sol or {}), n_replications, seeds
        )

        return results

    @staticmethod
    def save_results(results: dict[str, pd.DataFrame], out_dir: str | Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for policy_name, df in results.items():
            path = out_dir / f"sim_results_{policy_name}.csv"
            df.to_csv(path, index=False)
            logger.info("Saved %s", path)
