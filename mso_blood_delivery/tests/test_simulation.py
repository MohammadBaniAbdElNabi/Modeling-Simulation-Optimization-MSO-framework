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
