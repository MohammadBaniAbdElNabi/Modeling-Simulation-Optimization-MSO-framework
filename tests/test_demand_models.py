"""Tests for the R2 demand-model abstraction layer."""
import numpy as np
import pytest

from src.forecasting.demand_models import (
    DEMAND_MODEL_REGISTRY,
    GlobalStaticDemand,
    HospitalStaticDemand,
    SARIMAForecastDemand,
)
from src.utils.config_loader import load_config


H, T = 12, 24
_rng = np.random.default_rng(0)
_TRAIN = _rng.integers(0, 5, size=(6, H, T)).astype(float)


@pytest.fixture(scope="module")
def sarima_model():
    sarima_cfg = load_config("config/sarima.yaml")
    model = SARIMAForecastDemand(sarima_cfg)
    model.fit(_TRAIN.astype(int))
    return model


def _build_static_model(name: str):
    """Construct a non-SARIMA demand model with the right kwargs for its class."""
    cls = DEMAND_MODEL_REGISTRY[name]
    if name == "factor":
        return cls()
    return cls(H=H, T=T)


def test_demand_models_shape():
    """All non-SARIMA demand models return ndarray of shape (H, T)."""
    for name in DEMAND_MODEL_REGISTRY:
        if name == "sarima":
            continue  # tested separately (slow)
        model = _build_static_model(name)
        model.fit(_TRAIN)
        out = model.predict(steps=T)
        assert out.shape == (H, T), f"{name}: expected ({H},{T}), got {out.shape}"


def test_sarima_shape(sarima_model):
    out = sarima_model.predict(steps=T)
    assert out.shape == (H, T)


def test_demand_models_min_clip():
    """All returned values are >= 0.10 (lambda_min floor)."""
    for name in DEMAND_MODEL_REGISTRY:
        if name == "sarima":
            continue
        if name == "factor":
            model = DEMAND_MODEL_REGISTRY[name]({"lambda_min": 0.10})
        else:
            model = DEMAND_MODEL_REGISTRY[name](H=H, T=T, lambda_min=0.10)
        model.fit(_TRAIN)
        out = model.predict(steps=T)
        assert out.min() >= 0.10 - 1e-9, f"{name}: min={out.min()} < 0.10"


def test_sarima_min_clip(sarima_model):
    out = sarima_model.predict(steps=T)
    assert out.min() >= 0.10 - 1e-9


def test_global_static_constant():
    """GlobalStaticDemand.predict() returns an array where all entries are equal."""
    model = GlobalStaticDemand(H=H, T=T)
    model.fit(_TRAIN)
    out = model.predict(steps=T)
    assert np.allclose(out, out[0, 0]), "GlobalStatic array is not constant"


def test_hospital_static_uniform_in_t():
    """HospitalStaticDemand.predict()[h,:] is constant across t for every h."""
    model = HospitalStaticDemand(H=H, T=T)
    model.fit(_TRAIN)
    out = model.predict(steps=T)
    for h in range(H):
        assert np.allclose(out[h, :], out[h, 0]), (
            f"HospitalStatic row h={h} is not constant across time"
        )


def test_sarima_varies_in_both_dims(sarima_model):
    """SARIMAForecastDemand.predict() has nontrivial variance across both axes."""
    out = sarima_model.predict(steps=T)
    assert out.std(axis=0).mean() > 0.01, "No variation across hospitals"
    assert out.std(axis=1).mean() > 0.01, "No variation across hours"


def test_demand_model_registry():
    """DEMAND_MODEL_REGISTRY exposes the four classes by their string names."""
    assert set(DEMAND_MODEL_REGISTRY.keys()) == {"global", "hospital", "sarima", "factor"}
    assert DEMAND_MODEL_REGISTRY["global"]   is GlobalStaticDemand
    assert DEMAND_MODEL_REGISTRY["hospital"] is HospitalStaticDemand
    assert DEMAND_MODEL_REGISTRY["sarima"]   is SARIMAForecastDemand


def test_lp_accepts_any_demand_model():
    """LPDispatchSolver.solve() produces feasible x_sol for each demand model."""
    from src.optimization.lp_formulation import LPDispatchSolver
    from src.utils.config_loader import load_config
    from src.utils.distance import load_distance_matrix, flight_time_matrix

    lp_cfg      = load_config("config/lp.yaml")
    network_cfg = load_config("config/network.yaml")
    d_km  = load_distance_matrix(network_cfg)
    d_min = flight_time_matrix(d_km, speed_kmh=50.0)
    I_init = np.full(3, 50.0 * 4)

    for name in DEMAND_MODEL_REGISTRY:
        if name == "sarima":
            continue  # covered by sarima fixture
        model = _build_static_model(name)
        model.fit(_TRAIN)
        lam = model.predict(steps=T)

        solver = LPDispatchSolver(lp_cfg)
        x_sol  = solver.solve(lam, d_min, I_init)
        assert isinstance(x_sol, dict), f"{name}: solve() did not return a dict"
        assert len(x_sol) > 0, f"{name}: x_sol is empty"
        for v in x_sol.values():
            assert v >= 0, f"{name}: negative assignment value {v}"
