"""Tests for SimPy simulation processes (single-replication smoke test)."""
import numpy as np
import pytest

from src.simulation.dispatch.greedy_policy import NearestFeasibleGreedy
from src.simulation.dispatch.lp_policy import LPOptimizedDispatch
from src.simulation.dispatch.random_policy import RandomDispatch
from src.simulation.runner import SimulationRunner, _run_single
from src.utils.config_loader import load_config
from src.utils.distance import load_distance_matrix


@pytest.fixture(scope="module")
def sim_setup():
    network_cfg = load_config("config/network.yaml")
    sim_cfg = load_config("config/simulation.yaml")
    sim_cfg["H"] = 12
    sim_cfg["B"] = 3

    d_km = load_distance_matrix(network_cfg)

    rng = np.random.default_rng(42)
    lambda_hat = rng.uniform(0.5, 2.0, size=(12, 24))

    return sim_cfg, lambda_hat, d_km


def test_greedy_single_replication(sim_setup):
    sim_cfg, lambda_hat, d_km = sim_setup
    policy = NearestFeasibleGreedy()
    result = _run_single(seed=1, lambda_hat=lambda_hat, d_km=d_km,
                         sim_cfg=sim_cfg, policy=policy)
    assert "FR" in result
    assert result["FR"] >= 0.0
    assert result["FR"] <= 100.0


def test_random_single_replication(sim_setup):
    sim_cfg, lambda_hat, d_km = sim_setup
    rng = np.random.default_rng(1)
    policy = RandomDispatch(rng)
    result = _run_single(seed=1, lambda_hat=lambda_hat, d_km=d_km,
                         sim_cfg=sim_cfg, policy=policy)
    assert result["FR"] >= 0.0
    assert result["FR"] <= 100.0


def test_lp_single_replication(sim_setup):
    sim_cfg, lambda_hat, d_km = sim_setup
    policy = LPOptimizedDispatch(x_sol={})  # empty sol triggers greedy fallback
    result = _run_single(seed=1, lambda_hat=lambda_hat, d_km=d_km,
                         sim_cfg=sim_cfg, policy=policy)
    assert result["FR"] >= 0.0


def test_fulfillment_plus_expired_le_total(sim_setup):
    """n_completed + n_expired <= n_total (some may be in-flight at end)."""
    sim_cfg, lambda_hat, d_km = sim_setup
    from src.simulation.metrics_collector import MetricsCollector
    from src.simulation.entities import Request

    policy = NearestFeasibleGreedy()
    result = _run_single(seed=2, lambda_hat=lambda_hat, d_km=d_km,
                         sim_cfg=sim_cfg, policy=policy)
    # FR + ERR <= 100 (they can sum to less if some requests still in-flight)
    assert result["FR"] + result["ERR"] <= 100.01


def test_crn_seeds_produce_consistent_arrival_order(sim_setup):
    """Same seed produces identical FR for greedy (deterministic given seed)."""
    sim_cfg, lambda_hat, d_km = sim_setup
    policy = NearestFeasibleGreedy()
    r1 = _run_single(seed=7, lambda_hat=lambda_hat, d_km=d_km,
                     sim_cfg=sim_cfg, policy=policy)
    r2 = _run_single(seed=7, lambda_hat=lambda_hat, d_km=d_km,
                     sim_cfg=sim_cfg, policy=policy)
    assert r1["FR"] == pytest.approx(r2["FR"])


def test_lp_differs_from_greedy_when_x_sol_concentrated(sim_setup):
    """Regression: a non-trivial x_sol must produce metrics distinct from Greedy.

    Earlier versions used x_sol only as a 0.1 scoring bonus, which was always
    dominated by the inventory-distance term — the LP plan was effectively
    ignored and metrics were identical to Greedy. This test pins the contract
    that a deliberately concentrated x_sol changes simulation outcomes.
    """
    sim_cfg, lambda_hat, d_km = sim_setup

    # x_sol that forces *all* hospitals to be served only from bank 2
    # (the farthest bank for most hospitals); a competent LP policy must
    # honor this and produce noticeably different metrics from Greedy.
    x_sol_bank2 = {(2, h, t): 4 for h in range(12) for t in range(24)}
    # zero out banks 0 and 1
    for h in range(12):
        for t in range(24):
            x_sol_bank2[(0, h, t)] = 0
            x_sol_bank2[(1, h, t)] = 0

    lp_policy     = LPOptimizedDispatch(x_sol=x_sol_bank2)
    greedy_policy = NearestFeasibleGreedy()

    lp_res = _run_single(seed=11, lambda_hat=lambda_hat, d_km=d_km,
                         sim_cfg=sim_cfg, policy=lp_policy)
    gr_res = _run_single(seed=11, lambda_hat=lambda_hat, d_km=d_km,
                         sim_cfg=sim_cfg, policy=greedy_policy)

    # ADT under bank-2-only LP plan should be measurably different from Greedy
    # (Greedy uses nearest bank; bank 2 is farther for most hospitals).
    assert abs(lp_res["ADT"] - gr_res["ADT"]) > 0.1, (
        f"LP and Greedy produce ~identical ADT (lp={lp_res['ADT']}, "
        f"greedy={gr_res['ADT']}); LP policy is ignoring x_sol."
    )


def test_lp_consumption_resets_between_replications(sim_setup):
    """The LP consumption tracker must reset on each new replication."""
    sim_cfg, lambda_hat, d_km = sim_setup
    x_sol = {(b, h, t): 1 for b in range(3) for h in range(12) for t in range(24)}
    policy = LPOptimizedDispatch(x_sol=x_sol)

    r1 = _run_single(seed=3, lambda_hat=lambda_hat, d_km=d_km,
                     sim_cfg=sim_cfg, policy=policy)
    r2 = _run_single(seed=3, lambda_hat=lambda_hat, d_km=d_km,
                     sim_cfg=sim_cfg, policy=policy)

    # Same seed → identical results only if the tracker resets between reps
    assert r1["FR"]  == pytest.approx(r2["FR"]),  "Consumption tracker leaked across reps (FR)"
    assert r1["ADT"] == pytest.approx(r2["ADT"]), "Consumption tracker leaked across reps (ADT)"
