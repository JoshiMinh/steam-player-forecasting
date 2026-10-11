"""Econometric residual diagnostics, white noise validation, and diagnostic plotting."""

from __future__ import annotations

import warnings
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.tsa.stattools import acf


def evaluate_residuals(
    residuals: pd.Series | np.ndarray,
    lags: int | list[int] = 12,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Perform comprehensive econometric diagnostics on model residuals.

    Assesses whether residuals behave as Gaussian white noise:
      1. Ljung-Box Portmanteau test:
         H0: Residuals are independently distributed (no serial autocorrelation).
         H1: Residuals exhibit serial autocorrelation.
         p >= alpha -> Cannot reject H0 -> White noise behavior confirmed.
      2. Jarque-Bera test:
         H0: Residuals are normally distributed.
         H1: Residuals deviate from normality (skewness/excess kurtosis).
      3. Durbin-Watson statistic:
         Values close to 2.0 indicate absence of lag-1 serial autocorrelation.

    Args:
        residuals: 1D array or Series of in-sample or out-of-sample residuals.
        lags: Lag length or list of lags for the Ljung-Box test (default 12).
        alpha: Significance level threshold (default 0.05).

    Returns:
        Structured dictionary containing test statistics, p-values, and white-noise conclusions.
    """
    clean_resid = pd.Series(residuals).dropna()
    clean_resid = clean_resid[~np.isinf(clean_resid)].to_numpy(dtype=float)

    n_obs = len(clean_resid)
    if n_obs < 10:
        raise ValueError(f"Insufficient residual observations ({n_obs}) for diagnostic tests.")

    # 1. Basic moments
    res_mean = float(np.mean(clean_resid))
    res_std = float(np.std(clean_resid, ddof=1))

    # 2. Ljung-Box test
    max_lag = min(lags if isinstance(lags, int) else max(lags), n_obs // 2 - 1)
    if max_lag < 1:
        max_lag = 1

    lag_arg = [max_lag] if isinstance(lags, int) else [l for l in lags if l <= max_lag]
    if not lag_arg:
        lag_arg = [max_lag]

    lb_df = acorr_ljungbox(clean_resid, lags=lag_arg, return_df=True)
    min_lb_pvalue = float(lb_df["lb_pvalue"].min())
    is_white_noise = bool(min_lb_pvalue >= alpha)

    lb_results = {
        "lags": [int(l) for l in lb_df.index],
        "test_statistic": [float(s) for s in lb_df["lb_stat"]],
        "p_values": [float(p) for p in lb_df["lb_pvalue"]],
        "min_p_value": min_lb_pvalue,
        "is_white_noise": is_white_noise,
    }

    # 3. Jarque-Bera Normality test
    jb_stat, jb_pvalue, skew, kurt = jarque_bera(clean_resid)
    jb_results = {
        "test_statistic": float(jb_stat),
        "p_value": float(jb_pvalue),
        "skewness": float(skew),
        "kurtosis": float(kurt),
        "is_normal": bool(jb_pvalue >= alpha),
    }

    # 4. Durbin-Watson statistic
    dw_stat = float(durbin_watson(clean_resid))

    return {
        "n_obs": n_obs,
        "mean": res_mean,
        "std": res_std,
        "durbin_watson": dw_stat,
        "ljung_box": lb_results,
        "jarque_bera": jb_results,
        "is_white_noise": is_white_noise,
        "alpha": alpha,
    }


def plot_residual_diagnostics(
    residuals: pd.Series | np.ndarray,
    nlags: int = 24,
    title: str = "Residual Diagnostic Panel",
    figsize: tuple[int, int] = (12, 8),
) -> plt.Figure:
    """Generate a standard 4-panel Box-Jenkins econometric residual diagnostic plot.

    Panels:
      1. Standardized Residuals vs. Time / Index.
      2. Histogram with Empirical KDE vs. Standard Normal N(0, 1).
      3. Normal Q-Q Plot.
      4. Residual Autocorrelation Function (ACF) with 95% Bartlett confidence bands.

    Args:
        residuals: 1D series or array of model residuals.
        nlags: Number of lags to display in ACF.
        title: Overall super-title for the figure.
        figsize: Figure dimensions in inches.

    Returns:
        Matplotlib Figure object.
    """
    clean_resid = pd.Series(residuals).dropna()
    clean_resid = clean_resid[~np.isinf(clean_resid)].to_numpy(dtype=float)

    if len(clean_resid) < 5:
        raise ValueError("Need at least 5 residuals to plot diagnostic panel.")

    std_resid = (clean_resid - np.mean(clean_resid)) / (np.std(clean_resid, ddof=1) + 1e-8)

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

    # 1. Standardized Residuals
    ax1 = axes[0, 0]
    ax1.plot(std_resid, color="#2c3e50", lw=1.2, label="Std Residuals")
    ax1.axhline(0, color="#e74c3c", linestyle="--", lw=1)
    ax1.axhline(2, color="gray", linestyle=":", lw=0.8)
    ax1.axhline(-2, color="gray", linestyle=":", lw=0.8)
    ax1.set_title("Standardized Residuals", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Observation Index")
    ax1.set_ylabel("Standard Deviations")
    ax1.grid(True, linestyle=":", alpha=0.6)

    # 2. Histogram + KDE vs N(0, 1)
    ax2 = axes[0, 1]
    ax2.hist(std_resid, bins=15, density=True, color="#3498db", alpha=0.5, edgecolor="white", label="Histogram")
    x_grid = np.linspace(min(std_resid.min(), -3), max(std_resid.max(), 3), 100)
    ax2.plot(x_grid, stats.norm.pdf(x_grid, 0, 1), "r--", lw=1.5, label="N(0, 1)")
    try:
        kde = stats.gaussian_kde(std_resid)
        ax2.plot(x_grid, kde(x_grid), color="#2c3e50", lw=1.5, label="Empirical KDE")
    except Exception:
        pass
    ax2.set_title("Histogram and Estimated Density", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Standardized Residual")
    ax2.set_ylabel("Density")
    ax2.legend(fontsize=9)
    ax2.grid(True, linestyle=":", alpha=0.6)

    # 3. Normal Q-Q Plot
    ax3 = axes[1, 0]
    stats.probplot(std_resid, dist="norm", plot=ax3)
    ax3.get_lines()[0].set_markerfacecolor("#2980b9")
    ax3.get_lines()[0].set_markeredgecolor("none")
    ax3.get_lines()[0].set_markersize(5.0)
    ax3.get_lines()[1].set_color("#e74c3c")
    ax3.get_lines()[1].set_linewidth(1.5)
    ax3.set_title("Normal Q-Q Plot", fontsize=11, fontweight="bold")
    ax3.grid(True, linestyle=":", alpha=0.6)

    # 4. Residual Autocorrelation Function (ACF)
    ax4 = axes[1, 1]
    max_nlags = min(nlags, len(clean_resid) // 2 - 1)
    if max_nlags >= 1:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            acf_vals = acf(clean_resid, nlags=max_nlags)
        lags_arr = np.arange(len(acf_vals))
        ax4.vlines(lags_arr, 0, acf_vals, color="#2c3e50", lw=1.5)
        ax4.plot(lags_arr, acf_vals, "o", color="#2c3e50", markersize=4)
        ax4.axhline(0, color="gray", lw=0.8)
        # Bartlett 95% confidence bands
        conf_band = 1.96 / np.sqrt(len(clean_resid))
        ax4.axhline(conf_band, color="#e74c3c", linestyle="--", lw=1, label="95% CI")
        ax4.axhline(-conf_band, color="#e74c3c", linestyle="--", lw=1)
        ax4.fill_between(lags_arr, -conf_band, conf_band, color="#e74c3c", alpha=0.1)
    ax4.set_title("Residual ACF (Autocorrelation)", fontsize=11, fontweight="bold")
    ax4.set_xlabel("Lag")
    ax4.set_ylabel("ACF")
    ax4.legend(fontsize=9)
    ax4.grid(True, linestyle=":", alpha=0.6)

    fig.tight_layout()
    return fig


# Pytest discovery prevention tag
evaluate_residuals.__test__ = False  # type: ignore[attr-defined]
