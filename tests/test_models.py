"""Unit tests for Phase 3 statistical models, forecaster wrappers, and selection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from steam_player_forecasting.features.scalers import TimeSeriesScaler
from steam_player_forecasting.features.transforms import BoxCoxTransformer, LogTransformer
from steam_player_forecasting.models.base import TransformedForecaster
from steam_player_forecasting.models.diagnostics import evaluate_residuals
from steam_player_forecasting.models.selection import select_sarima_order
from steam_player_forecasting.models.statistical import (
    ARForecaster,
    ARIMAForecaster,
    HoltWintersForecaster,
    NaiveForecaster,
    SARIMAForecaster,
    SeasonalNaiveForecaster,
)


@pytest.fixture
def sample_ts() -> pd.Series:
    """Generate 60 months of synthetic monthly data with upward trend and annual seasonality."""
    np.random.seed(42)
    dates = pd.date_range("2015-01-01", periods=60, freq="MS")
    trend = np.linspace(1000, 3000, 60)
    seasonal = np.sin(2 * np.pi * np.arange(60) / 12) * 300
    noise = np.random.normal(0, 50, 60)
    values = trend + seasonal + noise
    return pd.Series(values, index=dates, name="Avg_players")


# ==============================================================================
# 1. Baseline Forecasters
# ==============================================================================


def test_naive_forecaster(sample_ts: pd.Series):
    """Verify NaiveForecaster output shapes, index alignment, and strategies."""
    # Last value strategy
    model_last = NaiveForecaster(strategy="last_value")
    model_last.fit(sample_ts)
    pred_last = model_last.predict(steps=12)

    assert isinstance(pred_last, pd.Series)
    assert len(pred_last) == 12
    assert pred_last.index[0] == pd.Timestamp("2020-01-01")
    assert np.allclose(pred_last.values, sample_ts.iloc[-1])
    assert len(model_last.fittedvalues_) == 60
    assert len(model_last.resid_) == 60

    # Mean strategy
    model_mean = NaiveForecaster(strategy="mean")
    model_mean.fit(sample_ts)
    pred_mean = model_mean.predict(steps=6)
    assert len(pred_mean) == 6
    assert np.allclose(pred_mean.values, np.mean(sample_ts.values))

    # Raw numpy array input
    model_arr = NaiveForecaster()
    model_arr.fit(sample_ts.values)
    pred_arr = model_arr.predict(steps=5)
    assert isinstance(pred_arr, np.ndarray)
    assert len(pred_arr) == 5
    assert pred_arr[0] == sample_ts.iloc[-1]


def test_seasonal_naive_forecaster(sample_ts: pd.Series):
    """Verify SeasonalNaiveForecaster replicates 12-month seasonal patterns."""
    model = SeasonalNaiveForecaster(seasonal_period=12)
    model.fit(sample_ts)

    # 24-step forecast should replicate the last 12 values twice
    pred = model.predict(steps=24)
    assert len(pred) == 24
    expected_cycle = sample_ts.iloc[-12:].values
    assert np.allclose(pred.values[:12], expected_cycle)
    assert np.allclose(pred.values[12:], expected_cycle)

    # In-sample fitted values
    assert np.isnan(model.fittedvalues_[:12]).all()
    assert np.allclose(model.fittedvalues_[12:], sample_ts.values[:-12])

    # Insufficient observations raises ValueError
    short_series = sample_ts.iloc[:6]
    with pytest.raises(ValueError, match="must be >= seasonal_period"):
        model.fit(short_series)


# ==============================================================================
# 2. Holt-Winters Exponential Smoothing
# ==============================================================================


def test_holt_winters_forecaster(sample_ts: pd.Series):
    """Verify HoltWintersForecaster handles additive trend and seasonality."""
    model = HoltWintersForecaster(seasonal_periods=12, trend="add", seasonal="add")
    model.fit(sample_ts)

    fc = model.predict(steps=12)
    assert len(fc) == 12
    assert isinstance(fc, pd.Series)
    assert fc.index[0] == pd.Timestamp("2020-01-01")
    assert fc.index[-1] == pd.Timestamp("2020-12-01")

    # In-sample statistics
    assert len(model.fittedvalues_) == 60
    assert len(model.resid_) == 60
    assert model.aic_ is not None

    # Multiplicative trend on zero/negative raises ValueError
    bad_ts = sample_ts.copy()
    bad_ts.iloc[0] = -10.0
    hw_mul = HoltWintersForecaster(seasonal_periods=12, trend="mul", seasonal="mul")
    with pytest.raises(ValueError, match="strictly positive"):
        hw_mul.fit(bad_ts)


# ==============================================================================
# 3. Autoregressive (AR), ARIMA, and SARIMA
# ==============================================================================


def test_ar_forecaster(sample_ts: pd.Series):
    """Verify ARForecaster fits autoregressive lags and forecasts."""
    model = ARForecaster(lags=2)
    model.fit(sample_ts)

    fc = model.predict(steps=12)
    assert len(fc) == 12
    assert model.aic_ is not None
    assert model.bic_ is not None
    assert len(model.resid_) > 0


def test_arima_forecaster(sample_ts: pd.Series):
    """Verify ARIMAForecaster point predictions and confidence intervals."""
    model = ARIMAForecaster(order=(1, 1, 1))
    model.fit(sample_ts)

    fc = model.predict(steps=12)
    assert len(fc) == 12
    assert model.aic_ is not None

    # Confidence intervals
    point_fc, ci_df = model.predict_conf_int(steps=12, alpha=0.05)
    assert len(point_fc) == 12
    assert isinstance(ci_df, pd.DataFrame)
    assert list(ci_df.columns) == ["lower", "upper"]
    assert (ci_df["upper"] >= ci_df["lower"]).all()


def test_sarima_forecaster(sample_ts: pd.Series):
    """Verify SARIMAForecaster seasonality, confidence intervals, and residual checks."""
    model = SARIMAForecaster(order=(1, 1, 0), seasonal_order=(1, 0, 0, 12))
    model.fit(sample_ts)

    fc = model.predict(steps=12)
    assert len(fc) == 12
    assert model.aic_ is not None
    assert len(model.summary()) > 0

    # Confidence intervals
    point_fc, ci_df = model.predict_conf_int(steps=12)
    assert len(point_fc) == 12
    assert len(ci_df) == 12
    assert (ci_df["upper"] >= ci_df["lower"]).all()

    # Residual diagnostics
    res_diag = model.check_residuals(lags=12)
    assert "ljung_box" in res_diag
    assert "jarque_bera" in res_diag
    assert isinstance(res_diag["is_white_noise"], bool)


# ==============================================================================
# 4. TransformedForecaster (Leakage-Free Inversion)
# ==============================================================================


def test_transformed_forecaster_with_log(sample_ts: pd.Series):
    """Verify TransformedForecaster inverts log-transformed forecasts back to level scale."""
    log_tf = LogTransformer()
    base_model = SARIMAForecaster(order=(1, 1, 0), seasonal_order=(0, 0, 0, 12))
    pipeline = TransformedForecaster(forecaster=base_model, transformer=log_tf)

    pipeline.fit(sample_ts)
    fc = pipeline.predict(steps=12)

    # Forecast must be on original scale (around 3000, NOT ~8 in log scale)
    assert len(fc) == 12
    assert fc.mean() > 1000.0


def test_transformed_forecaster_with_scaler(sample_ts: pd.Series):
    """Verify TransformedForecaster inverts minmax-scaled forecasts back to level scale."""
    scaler = TimeSeriesScaler(scaler_type="minmax")
    base_model = ARIMAForecaster(order=(1, 1, 1))
    pipeline = TransformedForecaster(forecaster=base_model, transformer=scaler)

    pipeline.fit(sample_ts)
    fc = pipeline.predict(steps=12)

    assert len(fc) == 12
    # Inverted forecast should be on level scale
    assert fc.mean() > 1000.0


# ==============================================================================
# 5. Residual Diagnostics & Model Selection
# ==============================================================================


def test_evaluate_residuals():
    """Verify evaluate_residuals detects white noise vs correlated residuals."""
    np.random.seed(42)
    # 1. Gaussian white noise residuals
    wn_resid = np.random.normal(0, 1, 100)
    diag_wn = evaluate_residuals(wn_resid, lags=10)
    assert diag_wn["is_white_noise"] is True

    # 2. Autocorrelated residuals (random walk)
    rw_resid = np.cumsum(np.random.normal(0, 1, 100))
    diag_rw = evaluate_residuals(rw_resid, lags=10)
    assert diag_rw["is_white_noise"] is False
    assert diag_rw["ljung_box"]["min_p_value"] < 0.05


def test_select_sarima_order(sample_ts: pd.Series):
    """Verify select_sarima_order evaluates grid and returns best specification."""
    train_s = sample_ts.iloc[:48]
    val_s = sample_ts.iloc[48:]

    cand_orders = [(1, 1, 0), (0, 1, 1), (1, 1, 1)]
    cand_sorders = [(1, 0, 0, 12), (0, 0, 0, 12)]

    best_ord, best_sord, df_results, best_mod = select_sarima_order(
        y_train=train_s,
        y_val=val_s,
        candidate_orders=cand_orders,
        candidate_seasonal_orders=cand_sorders,
        criterion="val_mae",
        maxiter=30,
    )

    assert isinstance(best_ord, tuple)
    assert isinstance(best_sord, tuple)
    assert len(df_results) == len(cand_orders) * len(cand_sorders)
    assert "val_mae" in df_results.columns
    assert "val_rmse" in df_results.columns
    assert "aic" in df_results.columns
    assert df_results["val_mae"].iloc[0] <= df_results["val_mae"].iloc[-1]
    assert best_mod.is_fitted_ is True


def test_score_method(sample_ts: pd.Series):
    """Verify score method evaluates out-of-sample metrics correctly."""
    train_s = sample_ts.iloc[:48]
    val_s = sample_ts.iloc[48:]

    model = NaiveForecaster()
    model.fit(train_s)

    mae = model.score(val_s, metric="MAE")
    rmse = model.score(val_s, metric="RMSE")
    smape = model.score(val_s, metric="sMAPE")

    assert mae > 0
    assert rmse >= mae
    assert 0 <= smape <= 100
