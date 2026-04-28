"""Tests for LP formulation and constraint satisfaction."""
import math

import numpy as np
import pytest

from src.optimization.lp_formulation import LPDispatchSolver, SENTINEL
from src.utils.config_loader import load_config
from src.utils.distance import load_distance_matrix, flight_time_matrix


@pytest.fixture(scope="module")
def lp_data():
    network_cfg = load_config("config/network.yaml")
    lp_cfg = load_config("config/lp.yaml")
    d_km = load_distance_matrix(network_cfg)
    d_min = flight_time_matrix(d_km, speed_kmh=50.0)

    # Use simple lambda_hat
    rng = np.random.default_rng(0)
    lambda_hat = rng.uniform(0.5, 2.0, size=(12, 24))

    I_init = np.full(3, 50.0 * 4)  # 50 units x 4 blood types
    solver = LPDispatchSolver(lp_cfg)
    x_sol = solver.solve(lambda_hat, d_min, I_init)
    return x_sol, lambda_hat, d_min, lp_cfg


def test_x_sol_nonnegative(lp_data):
    x_sol, _, _, _ = lp_data
    for (b, h, t), v in x_sol.items():
        assert v >= 0 or v == SENTINEL, f"x[{b},{h},{t}]={v} is invalid"


def test_demand_coverage(lp_data):
    x_sol, lambda_hat, _, _ = lp_data
    B, H, T = 3, 12, 24
    for t in range(T):
        for h in range(H):
            vals = [x_sol.get((b, h, t), 0) for b in range(B)]
            if all(v == SENTINEL for v in vals):
                continue  # infeasible window
            total = sum(max(0, v) for v in vals)
            demand = math.ceil(lambda_hat[h, t])
            assert total >= demand, (
                f"Demand coverage violated: window {t}, hospital {h}: "
                f"assigned={total}, required={demand}"
            )


def test_fleet_cap(lp_data):
    x_sol, _, _, lp_cfg = lp_data
    C_fleet = int(lp_cfg.get("fleet", {}).get("C_fleet", 24))
    B, H, T = 3, 12, 24
    for t in range(T):
        vals = [x_sol.get((b, h, t), 0) for b in range(B) for h in range(H)]
        if all(v == SENTINEL for v in vals):
            continue
        total = sum(max(0, v) for v in vals)
        assert total <= C_fleet, f"Fleet cap violated in window {t}: {total} > {C_fleet}"


def test_infeasible_windows_get_sentinel(lp_data):
    x_sol, _, _, _ = lp_data
    B, H, T = 3, 12, 24
    for t in range(T):
        vals = [x_sol.get((b, h, t), 0) for b in range(B) for h in range(H)]
        # Either all non-sentinel or all sentinel -- no mixing expected
        sentinels = [v == SENTINEL for v in vals]
        assert all(sentinels) or not any(sentinels), (
            f"Window {t} has mixed sentinel and valid values"
        )
