"""Tests for the inventory replenishment process and saturation lift."""
import numpy as np
import pytest
import simpy

from src.data_gen.blood_types import BLOOD_TYPES
from src.simulation.dispatch.greedy_policy import NearestFeasibleGreedy
from src.simulation.entities import BloodBank
from src.simulation.processes import inventory_replenishment
from src.simulation.runner import _run_single
from src.utils.config_loader import load_config
from src.utils.distance import load_distance_matrix


def test_inventory_replenishment_adds_units_at_scheduled_times():
    env   = simpy.Environment()
    banks = [BloodBank(bank_id=b, blood_types=BLOOD_TYPES, initial_per_type=10)
             for b in range(3)]
    schedule_hours = [1, 2, 3]   # in hours
    amount = 5

    env.process(inventory_replenishment(
        env, banks, schedule_hours, amount, BLOOD_TYPES, horizon_s=4 * 3600,
    ))
    env.run(until=4 * 3600)

    # 3 events × 5 units × 4 types per bank = 60 units added per bank
    for bank in banks:
        assert bank.total_inventory() == 10 * 4 + 5 * 3 * 4, (
            f"Bank {bank.bank_id} total = {bank.total_inventory()}"
        )


def test_replenishment_lifts_fr_ceiling():
    """Fulfillment rate must be strictly higher with replenishment enabled."""
    network_cfg = load_config("config/network.yaml")
    sim_cfg     = load_config("config/simulation.yaml")
    sim_cfg["H"] = 12
    sim_cfg["B"] = 3
    d_km = load_distance_matrix(network_cfg)

    rng = np.random.default_rng(1)
    lam = rng.uniform(0.5, 2.0, size=(12, 24))

    # WITHOUT replenishment
    sim_cfg_off = dict(sim_cfg)
    sim_cfg_off["replenishment"] = {"enabled": False}
    res_off = _run_single(seed=1, lambda_hat=lam, d_km=d_km,
                          sim_cfg=sim_cfg_off, policy=NearestFeasibleGreedy())

    # WITH replenishment (config default)
    res_on = _run_single(seed=1, lambda_hat=lam, d_km=d_km,
                         sim_cfg=sim_cfg, policy=NearestFeasibleGreedy())

    assert res_on["FR"] > res_off["FR"], (
        f"Replenishment did not lift FR: off={res_off['FR']:.2f}, "
        f"on={res_on['FR']:.2f}"
    )
