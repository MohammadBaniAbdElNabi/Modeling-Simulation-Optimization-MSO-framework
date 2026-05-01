"""Tests for SARIMA forecasting module."""
import numpy as np
import pytest

from src.data_gen.generator import SyntheticDemandGenerator
from src.forecasting.sarima_model import SARIMAForecaster


@pytest.fixture(scope="module")
def forecast_data():
    gen_cfg = {
        "H": 12, "T": 24, "D_train": 6, "D_test": 1,
        "noise_std": 0.15, "scale_low": 0.6, "scale_high": 1.4,
        "lambda_floor": 0.10, "seed": 42,
    }
    gen = SyntheticDemandGenerator(gen_cfg)
    data = gen.generate()

    sarima_cfg = {
        "p_range": [0, 1, 2], "d_range": [0, 1], "q_range": [0, 1, 2],
        "P_range": [0, 1], "D_range": [0, 1], "Q_range": [0, 1],
        "s": 24,
        "forecast": {"lambda_hat_min": 0.10},
        "stationarity": {"adf_alpha": 0.05, "acf_lag24_threshold": 0.3},
    }
    forecaster = SARIMAForecaster(sarima_cfg)
    forecaster.fit(data["train_counts"])
    lambda_hat = forecaster.predict(steps=24)
    return forecaster, lambda_hat, data


def test_lambda_hat_shape(forecast_data):
    _, lambda_hat, _ = forecast_data
    assert lambda_hat.shape == (12, 24), "lambda_hat must have shape (12, 24)"


def test_lambda_hat_nonnegative(forecast_data):
    _, lambda_hat, _ = forecast_data
    assert np.all(lambda_hat >= 0.10), "All lambda_hat values must be >= 0.10"


def test_lambda_hat_no_nan(forecast_data):
    _, lambda_hat, _ = forecast_data
    assert not np.any(np.isnan(lambda_hat)), "lambda_hat must not contain NaN"


def test_convergence_threshold(forecast_data):
    forecaster, _, _ = forecast_data
    n_converged = sum(m is not None for m in forecaster._fitted_models)
    assert n_converged >= 10, f"At least 10/12 hospitals must converge; got {n_converged}"


def test_evaluate_returns_dataframe(forecast_data):
    forecaster, lambda_hat, data = forecast_data
    test_lambda = data["lambda_true"][6]  # test day
    metrics_df = forecaster.evaluate(test_lambda)
    assert "MAE" in metrics_df.columns
    assert "RMSE" in metrics_df.columns
    assert "MAPE" in metrics_df.columns
    assert "Bias" in metrics_df.columns
    assert len(metrics_df) == 12
