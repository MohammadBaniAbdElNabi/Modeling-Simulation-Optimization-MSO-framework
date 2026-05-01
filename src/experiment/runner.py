"""ExperimentRunner: orchestrates the R3 four-condition × N-demand-level study.

Conditions (all using Factor forecast):
  - Random
  - Greedy
  - Static-LP        (24-hour plan, x_sol committed at t=0)
  - MPC-LP           (rolling-horizon LP re-solved hourly with current state)

Stress sweep optionally varies demand_scale across the same 4 conditions to
surface where each method dominates.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.data_gen.generator import SyntheticDemandGenerator
from src.forecasting.demand_models import DEMAND_MODEL_REGISTRY
from src.optimization.lp_formulation import LPDispatchSolver
from src.optimization.rolling_lp import RollingHorizonLP
from src.simulation.dispatch.greedy_policy import NearestFeasibleGreedy
from src.simulation.dispatch.lp_policy import LPOptimizedDispatch
from src.simulation.dispatch.mpc_policy import MPCDispatch
from src.simulation.dispatch.random_policy import RandomDispatch
from src.simulation.runner import SimulationRunner, _df_from_rows, _run_single
from src.utils.distance import flight_time_matrix, load_distance_matrix
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

_REPO_ROOT     = Path(__file__).resolve().parent.parent.parent
_OUT_PROCESSED = _REPO_ROOT / "data" / "processed"
_OUT_RESULTS   = _REPO_ROOT / "data" / "results"


def _save_lambda(lam: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    H, T = lam.shape
    pd.DataFrame(
        lam,
        columns=[f"hour_{t:02d}" for t in range(T)],
        index=[f"hospital_{h+1:02d}" for h in range(H)],
    ).rename_axis("hospital_id").to_csv(path)


class ExperimentRunner:
    """Orchestrate the R3 four-condition demand-stress experiment."""

    def __init__(self, configs: dict[str, Any]) -> None:
        self.configs = configs

    # ------------------------------------------------------------------
    # Run a single demand-scale level
    # ------------------------------------------------------------------
    def run_one_level(
        self,
        demand_scale: float,
        n_replications: int,
        seeds: list[int],
    ) -> dict[str, pd.DataFrame]:
        """Run all four conditions at a single demand_scale.

        Returns ``{cond_id: DataFrame}``.
        """
        sim_cfg = dict(self.configs["simulation"])
        sim_cfg["H"] = 12
        sim_cfg["B"] = 3

        # Synthetic data + Factor forecast (regenerated per call so seeds vary cleanly)
        data  = SyntheticDemandGenerator(self.configs["data_gen"]).generate()
        train = data["train_counts"]
        lambda_true = data["lambda_true"][self.configs["data_gen"]["D_train"]]

        factor_cls = DEMAND_MODEL_REGISTRY["factor"]
        factor     = factor_cls()
        factor.fit(train)
        lambda_factor = factor.predict(steps=24)

        # Distance + inventory
        d_km  = load_distance_matrix(self.configs["network"])
        speed = float(sim_cfg.get("drone", {}).get("speed_kmh", 50.0))
        d_min = flight_time_matrix(d_km, speed_kmh=speed)
        init_per_type = int(sim_cfg["inventory"]["initial_per_bank_per_type"])
        I_init = np.full(3, float(init_per_type * 4))

        # Static LP solve (using Factor forecast)
        static_solver = LPDispatchSolver(self.configs["lp"])
        x_sol_static  = static_solver.solve(lambda_factor, d_min, I_init)

        # Rolling-horizon LP for MPC
        rolling = RollingHorizonLP(
            self.configs["lp"],
            horizon_hours   = int(self.configs.get("mpc", {}).get("horizon_hours", 6)),
            demand_quantile = self.configs.get("mpc", {}).get("demand_quantile", 0.80),
        )

        # Run the four conditions; arrivals always use lambda_true
        all_results: dict[str, pd.DataFrame] = {}

        # 1) Random
        rng_master = np.random.default_rng(0)
        random_rows = []
        for rep_idx, seed in enumerate(seeds):
            pol    = RandomDispatch(np.random.default_rng(seed))
            result = _run_single(seed, lambda_true, d_km, sim_cfg, pol,
                                 demand_scale=demand_scale)
            result["rep_id"]       = rep_idx + 1
            result["seed"]         = seed
            result["demand_scale"] = demand_scale
            random_rows.append(result)
        all_results["random"] = _df_from_rows(random_rows)

        # 2) Greedy
        greedy_runner = SimulationRunner(sim_cfg=sim_cfg, lambda_hat=lambda_true,
                                         d_km=d_km, x_sol=None)
        all_results["greedy"] = greedy_runner.run(
            NearestFeasibleGreedy(), n_replications, seeds, demand_scale=demand_scale,
        )

        # 3) Static-LP (factor forecast)
        static_runner = SimulationRunner(sim_cfg=sim_cfg, lambda_hat=lambda_true,
                                         d_km=d_km, x_sol=x_sol_static)
        all_results["lp_static"] = static_runner.run(
            LPOptimizedDispatch(x_sol_static),
            n_replications, seeds, demand_scale=demand_scale,
        )

        # 4) MPC-LP (factor forecast, rolling-horizon, quantile-hedged)
        mpc_runner = SimulationRunner(sim_cfg=sim_cfg, lambda_hat=lambda_true,
                                      d_km=d_km, x_sol=None)
        mpc_policy = MPCDispatch(
            rolling_lp     = rolling,
            lambda_full    = lambda_factor,
            d_matrix       = d_min,
            replan_interval_hours = float(self.configs.get("mpc", {})
                                          .get("replan_interval_hours", 1.0)),
        )
        all_results["lp_mpc"] = mpc_runner.run(
            mpc_policy, n_replications, seeds, demand_scale=demand_scale,
        )

        return all_results, lambda_factor, x_sol_static

    # ------------------------------------------------------------------
    # Full stress sweep
    # ------------------------------------------------------------------
    def run_stress_sweep(
        self,
        demand_scales: Iterable[float] = (0.7, 1.0, 1.3),
        n_replications: int = 50,
    ) -> dict[float, dict[str, pd.DataFrame]]:
        """Run the 4-condition design at each demand level.

        Saves ``sim_results_{cond}_d{scale}.csv`` for every (cond, scale).
        """
        seeds = list(range(1, n_replications + 1))
        _OUT_PROCESSED.mkdir(parents=True, exist_ok=True)
        _OUT_RESULTS.mkdir(parents=True, exist_ok=True)

        all_levels: dict[float, dict[str, pd.DataFrame]] = {}
        first_lambda: np.ndarray | None = None
        first_xsol:   dict | None       = None

        for ds in demand_scales:
            ds = float(ds)
            logger.info("=== Demand scale: %.2f ===", ds)
            results, lam, xsol = self.run_one_level(ds, n_replications, seeds)
            all_levels[ds] = results
            if first_lambda is None:
                first_lambda, first_xsol = lam, xsol
            for cond_id, df in results.items():
                tag  = f"{cond_id}_d{ds:.2f}"
                path = _OUT_RESULTS / f"sim_results_{tag}.csv"
                df.to_csv(path, index=False)
                logger.info("  saved %s (%d rows)", path.name, len(df))

        # Persist forecast + static plan from the baseline level for downstream notebook
        if first_lambda is not None:
            _save_lambda(first_lambda, _OUT_PROCESSED / "lambda_factor.csv")
        return all_levels
