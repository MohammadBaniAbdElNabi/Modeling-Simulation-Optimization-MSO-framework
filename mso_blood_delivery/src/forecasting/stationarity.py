"""ADF test wrapper and differencing recommendation."""
from __future__ import annotations

import numpy as np
from statsmodels.tsa.stattools import acf, adfuller


def check_stationarity(series: np.ndarray, adf_alpha: float = 0.05,
                       acf_threshold: float = 0.3) -> tuple[int, int]:
    """Run ADF and ACF-lag-24 checks to recommend (d_guide, D_guide).

    Returns
    -------
    (d_guide, D_guide) where each value is 0 or 1.
    """
    adf_result = adfuller(series, autolag="AIC")
    p_value = adf_result[1]
    d_guide = 1 if p_value > adf_alpha else 0

    acf_vals = acf(series, nlags=48, fft=True)
    D_guide = 1 if abs(acf_vals[24]) > acf_threshold else 0

    return d_guide, D_guide
