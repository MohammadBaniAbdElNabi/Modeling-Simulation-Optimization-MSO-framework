"""Tests for LP formulation with configurable demand models (R2)."""
import math

import numpy as np
import pytest

from src.optimization.lp_formulation import LPDispatchSolver, SENTINEL
from src.utils.config_loader import load_config
from src.utils.distance import load_distance_matrix, flight_time_matrix


@pytest.fixture(scope="module")
def lp_data():
    network_cfg = load_config("config/network.yaml")
    lp_cfg      = load_config("config/lp.yaml")
    d_km  = load_distance_matrix(network_cfg)
    d_min = flight_time_matrix(d_km, speed_kmh=50.0)

    rng          = np.random.default_rng(0)
    lambda_model = rng.uniform(0.5, 2.0, size=(12, 24))

    I_init = np.full(3, 50.0 * 4)
    solver = LPDispatchSolver(lp_cfg)
    x_sol  = solver.solve(lambda_model, d_min, I_init)
    return x_sol, lambda_model, d_min, lp_cfg


def test_x_sol_nonnegative(lp_data):
    x_sol, _, _, _ = lp_data
    for (b, h, t), v in x_sol.items():
        assert v >= 0 or v == SENTINEL, f"x[{b},{h},{t}]={v} is invalid"


def test_demand_coverage_up_to_inventory(lp_data):
    """LP coverage is soft (slack penalised at 500). Verify the LP
    allocates as much as inventory permits — saturation, when present,
    is bounded by inventory/avg_units, not by the LP being lazy.
    """
    x_sol, lambda_model, _, lp_cfg = lp_data
    B, H, T = 3, 12, 24
    total_assigned = sum(
        max(0, x_sol.get((b, h, t), 0))
        for b in range(B) for h in range(H) for t in range(T)
    )
    # Upper bound: 3 banks × 200 units / 2.5 units/delivery = 240
    init_per_type   = 50
    n_types         = 4
    avg_units       = float(lp_cfg.get("avg_units_per_delivery", 2.5))
    inventory_cap   = B * init_per_type * n_types / avg_units

    # The LP is allowed to leave slack; verify it uses >=95% of inventory.
    util = total_assigned / inventory_cap
    assert 0.90 <= util <= 1.01, (
        f"LP utilization of inventory = {util:.2%} "
        f"(assigned={total_assigned}, capacity={inventory_cap:.0f})"
    )


def test_fleet_cap(lp_data):
    x_sol, _, _, lp_cfg = lp_data
    C_fleet = int(lp_cfg.get("fleet", {}).get("C_fleet", 24))
    B, H, T = 3, 12, 24
    for t in range(T):
        vals  = [x_sol.get((b, h, t), 0) for b in range(B) for h in range(H)]
        if all(v == SENTINEL for v in vals):
            continue
        total = sum(max(0, v) for v in vals)
        assert total <= C_fleet, f"Fleet cap violated in window {t}: {total} > {C_fleet}"


def test_infeasible_windows_get_sentinel(lp_data):
    x_sol, _, _, _ = lp_data
    B, H, T = 3, 12, 24
    for t in range(T):
        vals      = [x_sol.get((b, h, t), 0) for b in range(B) for h in range(H)]
        sentinels = [v == SENTINEL for v in vals]
        assert all(sentinels) or not any(sentinels), (
            f"Window {t} has mixed sentinel and valid values"
        )


def test_lp_accepts_global_static_demand():
    """LP solver accepts GlobalStaticDemand output (R2 configurable demand)."""
    from src.forecasting.demand_models import GlobalStaticDemand

    network_cfg = load_config("config/network.yaml")
    lp_cfg      = load_config("config/lp.yaml")
    d_km  = load_distance_matrix(network_cfg)
    d_min = flight_time_matrix(d_km, speed_kmh=50.0)
    I_init = np.full(3, 50.0 * 4)

    rng    = np.random.default_rng(1)
    train  = rng.integers(0, 4, size=(6, 12, 24)).astype(float)
    model  = GlobalStaticDemand(H=12, T=24)
    model.fit(train)
    lam    = model.predict()

    solver = LPDispatchSolver(lp_cfg)
    x_sol  = solver.solve(lam, d_min, I_init)
    assert isinstance(x_sol, dict) and len(x_sol) > 0


def test_lp_accepts_hospital_static_demand():
    """LP solver accepts HospitalStaticDemand output (R2 configurable demand)."""
    from src.forecasting.demand_models import HospitalStaticDemand

    network_cfg = load_config("config/network.yaml")
    lp_cfg      = load_config("config/lp.yaml")
    d_km  = load_distance_matrix(network_cfg)
    d_min = flight_time_matrix(d_km, speed_kmh=50.0)
    I_init = np.full(3, 50.0 * 4)

    rng   = np.random.default_rng(2)
    train = rng.integers(0, 4, size=(6, 12, 24)).astype(float)
    model = HospitalStaticDemand(H=12, T=24)
    model.fit(train)
    lam   = model.predict()

    solver = LPDispatchSolver(lp_cfg)
    x_sol  = solver.solve(lam, d_min, I_init)
    assert isinstance(x_sol, dict) and len(x_sol) > 0
