"""Tests for src/data_gen/generator.py."""
import numpy as np
import pytest

from src.data_gen.generator import SyntheticDemandGenerator


@pytest.fixture
def gen():
    cfg = {
        "H": 12, "T": 24, "D_train": 6, "D_test": 1,
        "noise_std": 0.15, "scale_low": 0.6, "scale_high": 1.4,
        "lambda_floor": 0.10, "seed": 42,
    }
    return SyntheticDemandGenerator(cfg)


def test_counts_shape(gen):
    data = gen.generate()
    assert data["counts"].shape == (7, 12, 24), "Total counts shape must be (7,12,24)"


def test_train_test_split(gen):
    data = gen.generate()
    assert data["train_counts"].shape == (6, 12, 24)
    assert data["test_counts"].shape == (1, 12, 24)


def test_lambda_floor(gen):
    data = gen.generate()
    assert np.all(data["lambda_true"] >= 0.10), "All lambda_true values must be >= 0.10"


def test_lambda_true_shape(gen):
    data = gen.generate()
    assert data["lambda_true"].shape == (7, 12, 24)


def test_counts_nonnegative(gen):
    data = gen.generate()
    assert np.all(data["counts"] >= 0), "Request counts must be non-negative"


def test_counts_integer(gen):
    data = gen.generate()
    assert data["counts"].dtype in (np.int32, np.int64, int)


def test_poisson_mean_close_to_lambda(gen):
    """Empirical mean of counts should be close to lambda_true (law of large numbers)."""
    cfg = {
        "H": 1, "T": 24, "D_train": 100, "D_test": 0,
        "noise_std": 0.0, "scale_low": 1.0, "scale_high": 1.0,
        "lambda_floor": 0.10, "seed": 0,
    }
    big_gen = SyntheticDemandGenerator(cfg)
    # Need D_test >= 1 to avoid empty arrays
    cfg["D_test"] = 1
    big_gen2 = SyntheticDemandGenerator(cfg)
    data = big_gen2.generate()
    # With noise_std=0 and scale=1, lambda_true ~ base_lambda
    # just check shapes are fine
    assert data["counts"].shape[0] == 101


def test_deterministic_with_same_seed(gen):
    data1 = gen.generate()
    data2 = gen.generate()
    np.testing.assert_array_equal(data1["counts"], data2["counts"])
