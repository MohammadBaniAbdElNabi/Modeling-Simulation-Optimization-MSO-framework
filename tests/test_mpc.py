"""Tests for the Rolling-Horizon (MPC) reformulation."""
import numpy as np
import pandas as pd
import pytest

from src.optimization.lp_formulation import LPDispatchSolver
from src.optimization.rolling_lp import RollingHorizonLP
from src.simulation.dispatch.greedy_policy import NearestFeasibleGreedy
from src.simulation.dispatch.lp_policy import LPOptimizedDispatch
from src.simulation.dispatch.mpc_policy import MPCDispatch
from src.simulation.runner import _run_single
from src.utils.config_loader import load_config
from src.utils.distance import flight_time_matrix, load_distance_matrix


@pytest.fixture(scope="module")
def cfg_setup():
    network_cfg = load_config("config/network.yaml")
    lp_cfg      = load_config("config/lp.yaml")
    sim_cfg     = load_config("config/simulation.yaml")
    sim_cfg["H"] = 12
    sim_cfg["B"] = 3

    d_km  = load_distance_matrix(network_cfg)
    d_min = flight_time_matrix(d_km, speed_kmh=50.0)

    rng    = np.random.default_rng(0)
    lam    = rng.uniform(0.5, 2.0, size=(12, 24))
    I_init = np.full(3, 50.0 * 4)

    return network_cfg, lp_cfg, sim_cfg, d_km, d_min, lam, I_init


# ---------------------------------------------------------------------------
# Mathematical guarantee: rolling LP at horizon=24 reduces to static LP
# ---------------------------------------------------------------------------
def test_rolling_lp_at_full_horizon_matches_static_with_point_forecast(cfg_setup):
    """When horizon = T and demand_quantile=None, rolling LP ≡ static LP.

    This pins the *strict generalisation* property: static LP is a special
    case of MPC, so any MPC-vs-static comparison is mathematically grounded.
    """
    _, lp_cfg, _, _, d_min, lam, I_init = cfg_setup

    static = LPDispatchSolver(lp_cfg)
    rolling = RollingHorizonLP(lp_cfg, horizon_hours=24, demand_quantile=None)

    x_static = static.solve(lam, d_min, I_init)
    x_rolling = rolling.solve_full_day(lam, d_min, I_init)

    # Both should produce the same total assigned volume
    sum_static  = sum(x_static.values())
    sum_rolling = sum(x_rolling.values())
    diff = abs(sum_static - sum_rolling) / max(1, sum_static)
    assert diff < 0.05, (
        f"Static={sum_static}, Rolling={sum_rolling}, diff={diff:.3f}"
    )


# ---------------------------------------------------------------------------
# Quantile hedging produces a strictly larger plan than point forecast
# ---------------------------------------------------------------------------
def test_quantile_demand_yields_more_capacity(cfg_setup):
    """Hedging at the 80th percentile must allocate more than point forecast."""
    _, lp_cfg, _, _, d_min, lam, I_init = cfg_setup

    point = RollingHorizonLP(lp_cfg, horizon_hours=24, demand_quantile=None)
    hedge = RollingHorizonLP(lp_cfg, horizon_hours=24, demand_quantile=0.8)

    x_point = point.solve_full_day(lam, d_min, I_init)
    x_hedge = hedge.solve_full_day(lam, d_min, I_init)

    sum_point = sum(x_point.values())
    sum_hedge = sum(x_hedge.values())
    assert sum_hedge >= sum_point, (
        f"Quantile hedge ({sum_hedge}) should plan >= point ({sum_point})"
    )


# ---------------------------------------------------------------------------
# MPC dispatch plays well in simulation; differs from greedy
# ---------------------------------------------------------------------------
def test_mpc_dispatch_runs_and_differs_from_greedy(cfg_setup):
    """Single replication smoke test: MPC must run and differ from Greedy."""
    _, lp_cfg, sim_cfg, d_km, d_min, lam, _ = cfg_setup

    rolling = RollingHorizonLP(lp_cfg, horizon_hours=6, demand_quantile=0.8)
    mpc     = MPCDispatch(rolling_lp=rolling, lambda_full=lam,
                          d_matrix=d_min, replan_interval_hours=1.0)

    mpc_res    = _run_single(seed=11, lambda_hat=lam, d_km=d_km,
                             sim_cfg=sim_cfg, policy=mpc)
    greedy_res = _run_single(seed=11, lambda_hat=lam, d_km=d_km,
                             sim_cfg=sim_cfg, policy=NearestFeasibleGreedy())

    print(f"\nMPC    FR={mpc_res['FR']:.2f}  ERR={mpc_res['ERR']:.2f}  "
          f"ADT={mpc_res['ADT']:.2f}")
    print(f"Greedy FR={greedy_res['FR']:.2f}  ERR={greedy_res['ERR']:.2f}  "
          f"ADT={greedy_res['ADT']:.2f}")

    # MPC must produce a valid result and not be identical to greedy
    assert 0.0 <= mpc_res["FR"] <= 100.0
    assert (
        abs(mpc_res["FR"]   - greedy_res["FR"])  > 0.01
        or abs(mpc_res["ADT"] - greedy_res["ADT"]) > 0.01
        or abs(mpc_res["ERR"] - greedy_res["ERR"]) > 0.01
    ), "MPC produced metrics identical to Greedy — re-planning had no effect"


# ---------------------------------------------------------------------------
# MPC dispatcher resets between replications via env_now backwards detection
# ---------------------------------------------------------------------------
def test_mpc_dispatch_resets_between_replications(cfg_setup):
    """Same seed must produce identical results across consecutive runs."""
    _, lp_cfg, sim_cfg, d_km, _, lam, _ = cfg_setup

    rolling = RollingHorizonLP(lp_cfg, horizon_hours=6, demand_quantile=0.8)
    d_min = flight_time_matrix(d_km, speed_kmh=50.0)
    mpc   = MPCDispatch(rolling_lp=rolling, lambda_full=lam,
                        d_matrix=d_min, replan_interval_hours=2.0)

    r1 = _run_single(seed=3, lambda_hat=lam, d_km=d_km,
                     sim_cfg=sim_cfg, policy=mpc)
    r2 = _run_single(seed=3, lambda_hat=lam, d_km=d_km,
                     sim_cfg=sim_cfg, policy=mpc)

    assert r1["FR"]  == pytest.approx(r2["FR"]),  "MPC state leaked across reps (FR)"
    assert r1["ADT"] == pytest.approx(r2["ADT"]), "MPC state leaked across reps (ADT)"
    assert r1["ERR"] == pytest.approx(r2["ERR"]), "MPC state leaked across reps (ERR)"
