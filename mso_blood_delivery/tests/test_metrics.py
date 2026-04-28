"""Tests for metric computation correctness."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.evaluation.statistics import compute_summary, pairwise_tests, PRIMARY_METRICS
from src.forecasting.metrics import compute_metrics
from src.simulation.metrics_collector import MetricsCollector
from src.simulation.entities import Priority, Request


def _make_request(hid=0, pri=Priority.NORMAL, arrival=0.0, assign=60.0,
                  delivery=300.0, expired=False, fulfilled=True):
    req = Request(
        hospital_id=hid,
        blood_type="O_pos",
        units_needed=1,
        priority=pri,
        arrival_time=arrival,
        expiration_time=arrival + 900.0,
    )
    req.assignment_time = assign
    req.delivery_time = delivery
    req.is_fulfilled = fulfilled
    req.is_expired = expired
    return req


def test_mae_computation():
    yhat = np.array([[1.0, 2.0], [3.0, 4.0]])
    ytrue = np.array([[1.5, 1.5], [2.5, 4.5]])
    df = compute_metrics(yhat, ytrue)
    # Hospital 0: |1.0-1.5| + |2.0-1.5| = 0.5+0.5=1.0 -> mean=0.5
    assert df.loc[0, "MAE"] == pytest.approx(0.5)


def test_rmse_computation():
    yhat = np.array([[2.0, 4.0]])
    ytrue = np.array([[1.0, 3.0]])
    df = compute_metrics(yhat, ytrue)
    # errors: [1,1], rmse = sqrt(1) = 1.0
    assert df.loc[0, "RMSE"] == pytest.approx(1.0)


def test_bias_sign():
    yhat = np.array([[2.0]])
    ytrue = np.array([[1.0]])
    df = compute_metrics(yhat, ytrue)
    assert df.loc[0, "Bias"] > 0  # over-forecasting


def test_metrics_collector_fr():
    mc = MetricsCollector(n_hospitals=2)
    r1 = _make_request(fulfilled=True)
    r2 = _make_request(fulfilled=False, expired=True)
    mc.log_arrival(r1)
    mc.log_arrival(r2)
    mc.log_delivery(r1)
    mc.log_expiration(r2)
    mc.log_assignment(r1)
    result = mc.compute()
    assert result["FR"] == pytest.approx(50.0)
    assert result["ERR"] == pytest.approx(50.0)


def test_metrics_collector_adt():
    mc = MetricsCollector(n_hospitals=1)
    # arrival=0, delivery=180s => ADT = 3 min
    r = _make_request(arrival=0.0, delivery=180.0, fulfilled=True)
    mc.log_arrival(r)
    mc.log_delivery(r)
    mc.log_assignment(r)
    result = mc.compute()
    assert result["ADT"] == pytest.approx(3.0)


def test_metrics_collector_awt():
    mc = MetricsCollector(n_hospitals=1)
    # arrival=0, assignment=60s => AWT = 1 min
    r = _make_request(arrival=0.0, assign=60.0, delivery=300.0, fulfilled=True)
    mc.log_arrival(r)
    mc.log_delivery(r)
    mc.log_assignment(r)
    result = mc.compute()
    assert result["AWT"] == pytest.approx(1.0)


def test_compute_summary_structure():
    rng = np.random.default_rng(0)
    data = {
        "random": pd.DataFrame({m: rng.uniform(0, 100, 20) for m in PRIMARY_METRICS}),
        "greedy": pd.DataFrame({m: rng.uniform(0, 100, 20) for m in PRIMARY_METRICS}),
        "lp":     pd.DataFrame({m: rng.uniform(0, 100, 20) for m in PRIMARY_METRICS}),
    }
    summary = compute_summary(data)
    assert "policy" in summary.columns
    assert "metric" in summary.columns
    assert "ci_lower" in summary.columns
    assert "ci_upper" in summary.columns


def test_pairwise_tests_structure():
    rng = np.random.default_rng(1)
    data = {
        "random": pd.DataFrame({m: rng.uniform(0, 100, 20) for m in PRIMARY_METRICS}),
        "greedy": pd.DataFrame({m: rng.uniform(0, 100, 20) for m in PRIMARY_METRICS}),
        "lp":     pd.DataFrame({m: rng.uniform(0, 100, 20) for m in PRIMARY_METRICS}),
    }
    pw = pairwise_tests(data)
    assert "t_stat" in pw.columns
    assert "p_value" in pw.columns
    assert "cohens_d" in pw.columns
    assert "significant" in pw.columns
    assert len(pw) == 3 * len(PRIMARY_METRICS)


def test_ci_uses_t_distribution():
    """Check that CI width matches t.ppf(0.975, df=19) formula."""
    rng = np.random.default_rng(2)
    sample = rng.normal(50, 10, 20)
    from scipy.stats import t as t_dist
    t_crit = t_dist.ppf(0.975, df=19)
    se = sample.std(ddof=1) / np.sqrt(20)
    expected_half = t_crit * se

    data = {"lp": pd.DataFrame({"FR": sample})}
    summary = compute_summary(data, metrics=["FR"])
    row = summary[(summary["policy"] == "lp") & (summary["metric"] == "FR")].iloc[0]
    computed_half = row["mean"] - row["ci_lower"]
    assert computed_half == pytest.approx(expected_half, rel=1e-6)
