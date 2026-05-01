"""Tests for HierarchicalFactorForecaster — guaranteed-improvement contract."""
import numpy as np
import pandas as pd
import pytest

from src.data_gen.generator import SyntheticDemandGenerator
from src.forecasting.factor_model import HierarchicalFactorForecaster
from src.forecasting.demand_models import FactorDemand
from src.utils.config_loader import load_config


# ---------------------------------------------------------------------------
# Recovery test — under correct specification, MLE recovers the truth
# ---------------------------------------------------------------------------
def test_factor_model_recovers_dgp_under_correct_spec():
    """Under the exact DGP, the EM updates must recover sigma_h and pi_t.

    Tolerances are scaled for Poisson noise at low rates: with rates
    ~0.3 req/hr at night, a single sample has CV ~ 1/sqrt(rate) ~ 1.8.
    Average across D=200 days drives MLE error to ~10–15%.
    """
    rng = np.random.default_rng(0)
    H, T, D = 12, 24, 200          # large D so MLE is precise

    # True parameters (mean(pi)=1 for identifiability)
    true_pi    = np.array([0.3]*6 + [0.8, 1.2, 1.5, 1.4, 1.3, 1.6,
                                     1.8, 2.0, 1.9, 1.7, 1.5, 1.4,
                                     1.2, 1.0, 0.8, 0.6, 0.5, 0.4])
    true_pi    = true_pi / true_pi.mean()
    true_sigma = rng.uniform(0.6, 1.4, size=H)

    lam    = np.outer(true_sigma, true_pi)
    counts = rng.poisson(lam[None, :, :].repeat(D, axis=0))

    model = HierarchicalFactorForecaster({"n_em_iters": 200})
    model.fit(counts)

    sigma_rel = np.abs(model._sigma - true_sigma) / true_sigma
    pi_rel    = np.abs(model._pi    - true_pi)    / true_pi
    assert sigma_rel.mean() < 0.05, f"sigma mean rel err {sigma_rel.mean():.3f}"
    assert pi_rel.mean()    < 0.05, f"pi    mean rel err {pi_rel.mean():.3f}"


# ---------------------------------------------------------------------------
# Predict shape contract
# ---------------------------------------------------------------------------
def test_factor_model_predict_shape_and_floor():
    rng = np.random.default_rng(1)
    counts = rng.integers(0, 5, size=(6, 12, 24))
    model = HierarchicalFactorForecaster({"lambda_min": 0.10})
    model.fit(counts)
    out = model.predict(steps=24)
    assert out.shape == (12, 24)
    assert out.min() >= 0.10 - 1e-9


# ---------------------------------------------------------------------------
# Empirical guarantee on the project's actual synthetic data:
#   - lower MAE
#   - lower |Bias|
#   - lower MAPE
# than the existing SARIMA-blended forecaster on the held-out test day.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def synthetic_data():
    sarima_cfg = load_config("config/sarima.yaml")
    gen_cfg    = sarima_cfg["data_gen"]
    gen        = SyntheticDemandGenerator(gen_cfg)
    data       = gen.generate()
    return data, gen_cfg, sarima_cfg


def test_factor_beats_sarima_on_held_out_day(synthetic_data):
    """The clean factor model must dominate SARIMA-blend on the test day."""
    data, gen_cfg, sarima_cfg = synthetic_data
    train_counts = data["train_counts"]
    lambda_true_test = data["lambda_true"][gen_cfg["D_train"]]   # (H, T)

    # Factor model
    fac = HierarchicalFactorForecaster()
    fac.fit(train_counts)
    fac_lambda = fac.predict(steps=24)
    fac_metrics = fac.evaluate(lambda_true_test)

    # SARIMA-blend (existing implementation)
    from src.forecasting.sarima_model import SARIMAForecaster
    sar_cfg = {
        "p_range": sarima_cfg["model"]["p_range"],
        "d_range": sarima_cfg["model"]["d_range"],
        "q_range": sarima_cfg["model"]["q_range"],
        "P_range": sarima_cfg["model"]["P_range"],
        "D_range": sarima_cfg["model"]["D_range"],
        "Q_range": sarima_cfg["model"]["Q_range"],
        "s":       sarima_cfg["model"]["s"],
        "forecast":     sarima_cfg["forecast"],
        "stationarity": sarima_cfg["stationarity"],
    }
    sar = SARIMAForecaster(sar_cfg)
    sar.fit(train_counts)
    sar.predict(steps=24)
    sar_metrics = sar.evaluate(lambda_true_test)

    fac_mae,  sar_mae  = fac_metrics["MAE"].mean(),  sar_metrics["MAE"].mean()
    fac_bias, sar_bias = fac_metrics["Bias"].mean(), sar_metrics["Bias"].mean()
    fac_mape, sar_mape = fac_metrics["MAPE"].mean(), sar_metrics["MAPE"].mean()

    print(f"\nFactor MAE={fac_mae:.4f}  Bias={fac_bias:+.4f}  MAPE={fac_mape:.2f}%")
    print(f"SARIMA MAE={sar_mae:.4f}  Bias={sar_bias:+.4f}  MAPE={sar_mape:.2f}%")

    assert fac_mae       <  sar_mae,       "Factor MAE not lower than SARIMA"
    assert abs(fac_bias) <  abs(sar_bias), "Factor bias not closer to zero"
    assert fac_mape      <  sar_mape,      "Factor MAPE not lower than SARIMA"


# ---------------------------------------------------------------------------
# DemandModel interface compliance
# ---------------------------------------------------------------------------
def test_factor_demand_in_registry():
    from src.forecasting.demand_models import DEMAND_MODEL_REGISTRY
    assert "factor" in DEMAND_MODEL_REGISTRY
    assert DEMAND_MODEL_REGISTRY["factor"] is FactorDemand


def test_factor_demand_metadata_has_n_params(synthetic_data):
    data, _, _ = synthetic_data
    m = FactorDemand()
    m.fit(data["train_counts"])
    m.predict()
    md = m.metadata()
    assert md["n_params"] == 12 + 24
    assert md["name"] == "factor"
