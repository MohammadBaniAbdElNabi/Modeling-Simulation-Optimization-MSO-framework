"""SimPy generator functions: request generation, drone mission, expiry monitor."""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Generator

import numpy as np
import simpy

from src.data_gen.blood_types import (
    BLOOD_TYPES,
    BLOOD_TYPE_PROBS,
    PRIORITY_CLASSES,
    PRIORITY_VALUES,
    PRIORITY_PROBS,
)
from src.simulation.entities import DroneState, Priority, Request

if TYPE_CHECKING:
    from src.simulation.entities import BloodBank, Drone, Hospital
    from src.simulation.metrics_collector import MetricsCollector


def request_generator(
    env: simpy.Environment,
    hospital: "Hospital",
    lambda_hat: np.ndarray,   # shape (T,) for this hospital
    expiration_min: float,
    units_min: int,
    units_max: int,
    rng: np.random.Generator,
    metrics: "MetricsCollector",
    horizon_s: float,
) -> Generator:
    """Poisson arrival process for a single hospital (Section 3.5)."""
    while env.now < horizon_s:
        t = int(env.now / 3600)
        t = min(t, len(lambda_hat) - 1)
        rate = float(lambda_hat[t])          # requests per hour
        if rate <= 0:
            rate = 0.1
        iat_hours = rng.exponential(1.0 / rate)
        iat_s = iat_hours * 3600.0

        yield env.timeout(iat_s)

        if env.now >= horizon_s:
            break

        bt = rng.choice(BLOOD_TYPES, p=BLOOD_TYPE_PROBS)
        units = int(rng.integers(units_min, units_max + 1))
        pri_name = rng.choice(PRIORITY_CLASSES, p=PRIORITY_PROBS)
        pri = Priority(PRIORITY_VALUES[pri_name])

        req = Request(
            hospital_id=hospital.hospital_id,
            blood_type=str(bt),
            units_needed=units,
            priority=pri,
            arrival_time=env.now,
            expiration_time=env.now + expiration_min * 60.0,
        )
        hospital.pending_requests.append(req)
        metrics.log_arrival(req)


def drone_mission(
    env: simpy.Environment,
    drone: "Drone",
    req: Request,
    bank: "BloodBank",
    d_km: float,
    loading_min: float,
    service_min: float,
    metrics: "MetricsCollector",
) -> Generator:
    """Execute one drone delivery mission (Section 3.9)."""
    req.assignment_time = env.now
    req.assigned_drone_id = drone.drone_id
    req.assigned_bank_id = bank.bank_id
    metrics.log_assignment(req)

    # Step 1: Loading
    drone.state = DroneState.LOADING
    yield env.timeout(loading_min * 60.0)

    # Step 2: Fly to hospital
    drone.state = DroneState.TRANSIT_TO_HOSPITAL
    t_flight_s = d_km / drone.speed_kmh * 3600.0
    yield env.timeout(t_flight_s)
    drone.battery -= drone.battery_drain * d_km

    # Step 3: Deliver
    drone.state = DroneState.DELIVERING
    yield env.timeout(service_min * 60.0)
    req.delivery_time = env.now
    req.is_fulfilled = True
    metrics.log_delivery(req)

    # Step 4: Return to bank
    drone.state = DroneState.RETURNING
    yield env.timeout(t_flight_s)
    drone.battery -= drone.battery_drain * d_km

    # Step 5: Recharge to 100%
    if drone.battery < 100.0:
        drone.state = DroneState.RECHARGING
        recharge_needed = 100.0 - drone.battery
        t_recharge_s = recharge_needed / drone.recharge_rate * 60.0
        yield env.timeout(t_recharge_s)
        drone.battery = 100.0

    drone.state = DroneState.IDLE


def expiration_monitor(
    env: simpy.Environment,
    hospitals: list["Hospital"],
    dispatch_cycle_s: float,
    metrics: "MetricsCollector",
    horizon_s: float,
) -> Generator:
    """Periodically sweep hospital queues and expire overdue requests (Sec 3.10)."""
    while env.now < horizon_s:
        yield env.timeout(dispatch_cycle_s)
        for hospital in hospitals:
            to_expire = [
                r for r in hospital.pending_requests
                if env.now >= r.expiration_time and not r.is_expired
            ]
            for req in to_expire:
                req.is_expired = True
                hospital.pending_requests.remove(req)
                metrics.log_expiration(req)


def dispatch_cycle(
    env: simpy.Environment,
    hospitals: list["Hospital"],
    drones: list["Drone"],
    banks: list["BloodBank"],
    d_matrix: np.ndarray,
    dispatch_policy: "object",
    loading_min: float,
    service_min: float,
    dispatch_cycle_s: float,
    metrics: "MetricsCollector",
    horizon_s: float,
) -> Generator:
    """Periodic dispatch decision loop (Section 3.8)."""
    while env.now < horizon_s:
        yield env.timeout(dispatch_cycle_s)

        assignments = dispatch_policy.dispatch(
            env.now, hospitals, drones, banks, d_matrix, metrics
        )

        for drone, req, bank in assignments:
            bank.deduct(req.blood_type, req.units_needed)
            d_km = float(d_matrix[bank.bank_id, req.hospital_id])
            env.process(
                drone_mission(
                    env, drone, req, bank, d_km,
                    loading_min, service_min, metrics
                )
            )
