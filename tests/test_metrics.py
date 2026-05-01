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


def test_metrics_collector_priority_weighted_fr():
    """FR_weighted: emergency = 3, urgent = 2, normal = 1.

    Scenario: 1 EMERGENCY arrived, delivered (w=3 / 3 = 100%)
              1 URGENT    arrived, expired   (w=0 / 2)
              1 NORMAL    arrived, delivered (w=1 / 1)
    Total weight 6, delivered weight 4 → FR_weighted = 66.67%
    """
    mc = MetricsCollector(n_hospitals=1)
    r_emer = _make_request(pri=Priority.EMERGENCY, fulfilled=True)
    r_urg  = _make_request(pri=Priority.URGENT,    fulfilled=False, expired=True)
    r_norm = _make_request(pri=Priority.NORMAL,    fulfilled=True)
    for r in (r_emer, r_urg, r_norm):
        mc.log_arrival(r)
    mc.log_delivery(r_emer); mc.log_assignment(r_emer)
    mc.log_expiration(r_urg)
    mc.log_delivery(r_norm); mc.log_assignment(r_norm)
    out = mc.compute()
    assert out["FR_weighted"]    == pytest.approx(4 / 6 * 100, abs=1e-3)
    assert out["Expiration_cost"] == pytest.approx(2.0)


def test_metrics_collector_peak_hours():
    """ERR_peak / FR_peak only count requests arriving in [12, 17]."""
    mc = MetricsCollector(n_hospitals=1, peak_hours=(12, 17))
    # Arrival at hour 14 (= 14*3600s), delivered
    peak_in = _make_request(arrival=14 * 3600.0, delivery=14 * 3600.0 + 300,
                            fulfilled=True)
    # Arrival at hour 14, expired
    peak_ex = _make_request(arrival=14 * 3600.0 + 1, expired=True, fulfilled=False)
    # Arrival at hour 3 (off-peak), expired — must NOT affect peak metrics
    off = _make_request(arrival=3 * 3600.0, expired=True, fulfilled=False)
    for r in (peak_in, peak_ex, off):
        mc.log_arrival(r)
    mc.log_delivery(peak_in); mc.log_assignment(peak_in)
    mc.log_expiration(peak_ex)
    mc.log_expiration(off)
    out = mc.compute()
    # Only the 2 peak arrivals count: 1 delivered, 1 expired → 50% each
    assert out["FR_peak"]  == pytest.approx(50.0)
    assert out["ERR_peak"] == pytest.approx(50.0)


def test_compute_summary_structure():
    rng = np.random.default_rng(0)
    data = {
        "random": pd.DataFrame({m: rng.uniform(0, 100, 20) for m in PRIMARY_METRICS}),
        "greedy": pd.DataFrame({m: rng.uniform(0, 100, 20) for m in PRIMARY_METRICS}),
        "lp":     pd.DataFrame({m: rng.uniform(0, 100, 20) for m in PRIMARY_METRICS}),
    }
    summary = compute_summary(data)
    assert "condition" in summary.columns
    assert "metric"    in summary.columns
    assert "ci_lower"  in summary.columns
    assert "ci_upper"  in summary.columns


def test_pairwise_tests_structure():
    """R3: pairwise_tests uses 4 PRIMARY + 2 REFERENCE = 6 pairs."""
    from src.evaluation.statistics import PRIMARY_PAIRS, REFERENCE_PAIRS

    rng = np.random.default_rng(1)
    cond_ids = ["random", "greedy", "lp_static", "lp_mpc"]
    data = {
        cid: pd.DataFrame({m: rng.uniform(0, 100, 20) for m in PRIMARY_METRICS})
        for cid in cond_ids
    }
    pw = pairwise_tests(data)
    assert "t_stat"           in pw.columns
    assert "p_value"          in pw.columns
    assert "cohens_d"         in pw.columns
    assert "significant"      in pw.columns
    assert "is_primary"       in pw.columns
    assert "mde_80"           in pw.columns
    assert "p_adj"            in pw.columns

    n_pairs = len(PRIMARY_PAIRS) + len(REFERENCE_PAIRS)
    assert len(pw) == n_pairs * len(PRIMARY_METRICS)
    assert pw["is_primary"].sum() == len(PRIMARY_PAIRS) * len(PRIMARY_METRICS)


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
    row = summary[(summary["condition"] == "lp") & (summary["metric"] == "FR")].iloc[0]
    computed_half = row["mean"] - row["ci_lower"]
    assert computed_half == pytest.approx(expected_half, rel=1e-6)
