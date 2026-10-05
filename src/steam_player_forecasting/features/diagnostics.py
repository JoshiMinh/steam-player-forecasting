"""Statistical diagnostics, stationarity tests, and decomposition for time series."""

from __future__ import annotations

import warnings
from typing import Any, Literal

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL, seasonal_decompose
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf


def check_stationarity(
    series: pd.Series | np.ndarray,
    alpha: float = 0.05,
    kpss_regression: Literal["c", "ct"] = "c",
) -> dict[str, Any]:
    """Perform joint Augmented Dickey-Fuller (ADF) and KPSS stationarity tests.

    Statistical hypotheses:
      - ADF:
          H0: Unit root is present (Non-stationary).
          H1: Unit root is absent (Stationary).
          Decision: p < alpha -> Reject H0 -> Stationary.
      - KPSS:
          H0: Series is level- or trend-stationary.
          H1: Unit root is present (Non-stationary).
          Decision: p < alpha -> Reject H0 -> Non-stationary.

    Args:
        series: 1D time-series.
        alpha: Significance level (default 0.05).
        kpss_regression: 'c' for level stationarity, 'ct' for trend stationarity.

    Returns:
        Structured dictionary containing ADF and KPSS test statistics, p-values,
        critical values, binary decisions, and a joint conclusion.
    """
    clean_series = pd.Series(series).dropna()
    if len(clean_series) < 15:
        raise ValueError(
            f"Series length ({len(clean_series)}) is too short for reliable stationarity tests."
        )

    # 1. Augmented Dickey-Fuller
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        adf_res = adfuller(clean_series, autolag="AIC")
    adf_stat, adf_pvalue, adf_lags, adf_nobs, adf_crit, _ = adf_res
    adf_stationary = bool(adf_pvalue < alpha)

    # 2. KPSS Test
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kpss_res = kpss(clean_series, regression=kpss_regression, nlags="auto")
    kpss_stat, kpss_pvalue, kpss_lags, kpss_crit = kpss_res
    kpss_stationary = bool(kpss_pvalue >= alpha)

    # Joint synthesis
    if adf_stationary and kpss_stationary:
        conclusion = "Stationary (both tests agree)"
    elif not adf_stationary and not kpss_stationary:
        conclusion = "Non-Stationary (Unit Root; both tests agree)"
    elif adf_stationary and not kpss_stationary:
        conclusion = "Difference / Trend Stationary (ADF stationary, KPSS rejects)"
    else:
        conclusion = "Ambiguous / Strict Stationarity (ADF fails to reject, KPSS fails to reject)"

    return {
        "adf": {
            "test_statistic": float(adf_stat),
            "p_value": float(adf_pvalue),
            "lags_used": int(adf_lags),
            "n_obs": int(adf_nobs),
            "critical_values": {k: float(v) for k, v in adf_crit.items()},
            "is_stationary": adf_stationary,
        },
        "kpss": {
            "test_statistic": float(kpss_stat),
            "p_value": float(kpss_pvalue),
            "lags_used": int(kpss_lags),
            "critical_values": {k: float(v) for k, v in kpss_crit.items()},
            "is_stationary": kpss_stationary,
        },
        "conclusion": conclusion,
        "is_stationary": bool(adf_stationary and kpss_stationary),
    }


def compute_acf_pacf(
    series: pd.Series | np.ndarray,
    nlags: int = 24,
    alpha: float = 0.05,
) -> dict[str, np.ndarray]:
    """Compute Autocorrelation (ACF) and Partial Autocorrelation (PACF) with confidence bounds.

    Args:
        series: 1D time-series data.
        nlags: Number of lags to calculate.
        alpha: Alpha for confidence interval (default 0.05 for 95% bounds).

    Returns:
        Dictionary containing lag numbers, acf values, acf confidence intervals,
        pacf values, and pacf confidence intervals.
    """
    clean_series = pd.Series(series).dropna()
    max_lags = min(nlags, len(clean_series) // 2 - 1)
    if max_lags < 1:
        raise ValueError("Series length is too short to compute autocorrelation lags.")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        acf_vals, acf_conf = acf(clean_series, nlags=max_lags, alpha=alpha)
        pacf_vals, pacf_conf = pacf(clean_series, nlags=max_lags, alpha=alpha, method="ywm")

    return {
        "lags": np.arange(len(acf_vals)),
        "acf": acf_vals,
        "acf_confint": acf_conf,
        "pacf": pacf_vals,
        "pacf_confint": pacf_conf,
    }


def decompose_time_series(
    series: pd.Series,
    period: int = 12,
    model: Literal["additive", "multiplicative"] = "additive",
    method: Literal["stl", "classical"] = "stl",
) -> dict[str, pd.Series]:
    """Decompose time-series into Trend, Seasonal, and Residual components.

    Args:
        series: 1D time series indexed by DatetimeIndex or integer sequence.
        period: Seasonal period length (default 12 for monthly data).
        model: 'additive' or 'multiplicative' (for classical decomposition).
        method: 'stl' (Loess-based STL) or 'classical' (moving-average decomposition).

    Returns:
        Dictionary of pd.Series containing 'observed', 'trend', 'seasonal', and 'resid'.
    """
    clean = series.copy()
    if clean.isna().any():
        clean = clean.interpolate(method="linear").bfill().ffill()

    if len(clean) < 2 * period:
        raise ValueError(
            f"Series length ({len(clean)}) must be at least 2 * period ({2 * period}) for decomposition."
        )

    if method == "stl":
        stl = STL(clean, period=period, seasonal=13)
        res = stl.fit()
        return {
            "observed": clean,
            "trend": res.trend,
            "seasonal": res.seasonal,
            "resid": res.resid,
        }
    elif method == "classical":
        res = seasonal_decompose(clean, period=period, model=model)
        return {
            "observed": clean,
            "trend": res.trend,
            "seasonal": res.seasonal,
            "resid": res.resid,
        }
    else:
        raise ValueError(f"Unsupported decomposition method '{method}'. Choose 'stl' or 'classical'.")


def compute_rolling_stats(
    series: pd.Series,
    windows: list[int] | tuple[int, ...] = (3, 6, 12),
) -> pd.DataFrame:
    """Calculate rolling mean, standard deviation, and variance across multiple windows."""
    df_out = pd.DataFrame(index=series.index)
    df_out["original"] = series

    for w in windows:
        df_out[f"rolling_mean_{w}"] = series.rolling(window=w, min_periods=1).mean()
        df_out[f"rolling_std_{w}"] = series.rolling(window=w, min_periods=1).std()
        df_out[f"rolling_var_{w}"] = series.rolling(window=w, min_periods=1).var()

    return df_out


# Aliases & Pytest configuration
# Prevent pytest from treating statistical function as a test case
test_stationarity = check_stationarity
check_stationarity.__test__ = False  # type: ignore[attr-defined]
test_stationarity.__test__ = False  # type: ignore[attr-defined]

