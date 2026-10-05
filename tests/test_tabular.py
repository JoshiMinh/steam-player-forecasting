"""Unit tests for tabular feature engineering and machine learning forecasters."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from steam_player_forecasting.features.tabular import (
    TabularFeatureExtractor,
    build_tabular_feature_matrix,
    create_calendar_features,
    create_lag_features,
    create_rolling_features,
)
from steam_player_forecasting.models.tabular import (
    RandomForestForecaster,
    RidgeForecaster,
    TabularForecaster,
    XGBoostForecaster,
    tune_tabular_forecaster,
)


@pytest.fixture
def sample_monthly_series() -> pd.Series:
    """Generate 60 months of synthetic monthly data."""
    np.random.seed(42)
    dates = pd.date_range("2015-01-01", periods=60, freq="MS")
    trend = np.linspace(1000, 3000, 60)
    seasonal = np.sin(2 * np.pi * np.arange(60) / 12) * 300
    noise = np.random.normal(0, 30, 60)
    return pd.Series(trend + seasonal + noise, index=dates, name="Avg_players")


# ==============================================================================
# 1. Feature Engineering Unit Tests
# ==============================================================================


def test_create_lag_features(sample_monthly_series: pd.Series):
    """Verify lag features construction, shapes, and shifts."""
    df_lags = create_lag_features(sample_monthly_series, lags=(1, 2, 12), prefix="lag")
    assert df_lags.shape == (60, 3)
    assert list(df_lags.columns) == ["lag_1", "lag_2", "lag_12"]

    # Check lag 1 alignment
    assert np.isnan(df_lags["lag_1"].iloc[0])
    assert np.isclose(df_lags["lag_1"].iloc[1], sample_monthly_series.iloc[0])

    # Check lag 12 alignment
    assert np.isnan(df_lags["lag_12"].iloc[11])
    assert np.isclose(df_lags["lag_12"].iloc[12], sample_monthly_series.iloc[0])

    # Error handling on invalid lag
    with pytest.raises(ValueError, match="must be >= 1"):
        create_lag_features(sample_monthly_series, lags=(0, 1))


def test_create_rolling_features_shift_leakage(sample_monthly_series: pd.Series):
    """Verify rolling statistics strictly shift prior to calculation to prevent lookahead."""
    df_roll = create_rolling_features(
        sample_monthly_series, windows=(3, 6), metrics=("mean", "std"), shift=1
    )
    assert "rolling_mean_3" in df_roll.columns
    assert "rolling_std_3" in df_roll.columns
    assert "rolling_mean_6" in df_roll.columns
    assert "rolling_std_6" in df_roll.columns

    # Strict anti-leakage test:
    # At index 3 (4th item, t=3), rolling_mean_3 must equal mean of items t=0, 1, 2 (NOT including t=3!)
    expected_mean_at_3 = sample_monthly_series.iloc[0:3].mean()
    assert np.isclose(df_roll["rolling_mean_3"].iloc[3], expected_mean_at_3)

    # Shift < 1 must raise error
    with pytest.raises(ValueError, match="Shift must be >= 1"):
        create_rolling_features(sample_monthly_series, shift=0)


def test_create_calendar_features():
    """Verify calendar feature extraction and seasonal indicators."""
    dates = pd.date_range("2020-01-01", periods=12, freq="MS")
    df_cal = create_calendar_features(dates, include_cyclical=True, include_sales_events=True)

    assert "month" in df_cal.columns
    assert "quarter" in df_cal.columns
    assert "sin_month" in df_cal.columns
    assert "cos_month" in df_cal.columns
    assert "is_steam_summer_sale" in df_cal.columns
    assert "is_steam_winter_sale" in df_cal.columns

    # Verify Summer Sale flag (June=6, July=7)
    assert df_cal.loc[pd.Timestamp("2020-06-01"), "is_steam_summer_sale"] == 1
    assert df_cal.loc[pd.Timestamp("2020-07-01"), "is_steam_summer_sale"] == 1
    assert df_cal.loc[pd.Timestamp("2020-08-01"), "is_steam_summer_sale"] == 0

    # Verify Winter Sale flag (Dec=12, Jan=1)
    assert df_cal.loc[pd.Timestamp("2020-12-01"), "is_steam_winter_sale"] == 1
    assert df_cal.loc[pd.Timestamp("2020-01-01"), "is_steam_winter_sale"] == 1
    assert df_cal.loc[pd.Timestamp("2020-03-01"), "is_steam_winter_sale"] == 0


def test_build_tabular_feature_matrix(sample_monthly_series: pd.Series):
    """Verify complete feature matrix alignment and warmup row drop."""
    X, y = build_tabular_feature_matrix(
        sample_monthly_series,
        lags=(1, 2, 3, 6, 12),
        windows=(3, 6, 12),
        dropna=True,
    )
    # 60 total rows, max lag/window is 12 -> 60 - 12 = 48 valid rows
    assert len(X) == 48
    assert len(y) == 48
    assert not X.isna().any().any()
    assert (X.index == y.index).all()


def test_tabular_feature_extractor_step_features(sample_monthly_series: pd.Series):
    """Verify stateful extractor generates single-row feature vector matching history."""
    extractor = TabularFeatureExtractor(
        lags=(1, 2, 12),
        windows=(3, 6),
        metrics=("mean", "std"),
        include_calendar=True,
    )
    X_train, y_train = extractor.fit_transform(sample_monthly_series)

    # Extract step feature at end of sample
    hist = sample_monthly_series.values.tolist()
    next_date = pd.Timestamp("2020-01-01")
    df_step = extractor.extract_step_features(hist, step_date=next_date)

    assert df_step.shape == (1, len(extractor.feature_names_))
    assert np.isclose(df_step["lag_1"].iloc[0], hist[-1])
    assert np.isclose(df_step["lag_2"].iloc[0], hist[-2])
    assert np.isclose(df_step["lag_12"].iloc[0], hist[-12])
    assert np.isclose(df_step["rolling_mean_3"].iloc[0], np.mean(hist[-3:]))


# ==============================================================================
# 2. Tabular Forecaster Model Tests
# ==============================================================================


def test_ridge_forecaster(sample_monthly_series: pd.Series):
    """Verify RidgeForecaster fitting, multi-step prediction, and index preservation."""
    train_s = sample_monthly_series.iloc[:48]
    val_s = sample_monthly_series.iloc[48:]

    model = RidgeForecaster(alpha=1.0)
    model.fit(train_s)

    pred = model.predict(steps=12)
    assert isinstance(pred, pd.Series)
    assert len(pred) == 12
    assert pred.index[0] == val_s.index[0]
    assert len(model.fittedvalues_) == 48
    assert len(model.resid_) == 48

    mae = model.score(val_s, metric="MAE")
    assert mae > 0


def test_random_forest_forecaster(sample_monthly_series: pd.Series):
    """Verify RandomForestForecaster recursive forecasting."""
    train_s = sample_monthly_series.iloc[:48]
    val_s = sample_monthly_series.iloc[48:]

    model = RandomForestForecaster(n_estimators=30, max_depth=4, random_state=42)
    model.fit(train_s)

    pred = model.predict(steps=12)
    assert len(pred) == 12
    assert not np.isnan(pred.values).any()
    # Predictions should be reasonable (within range of historical player counts)
    assert pred.mean() > 500


def test_xgboost_forecaster(sample_monthly_series: pd.Series):
    """Verify XGBoostForecaster fitting and evaluation."""
    train_s = sample_monthly_series.iloc[:48]
    val_s = sample_monthly_series.iloc[48:]

    model = XGBoostForecaster(n_estimators=30, max_depth=3, learning_rate=0.05, random_state=42)
    model.fit(train_s)

    pred = model.predict(steps=12)
    assert len(pred) == 12
    rmse = model.score(val_s, metric="RMSE")
    assert rmse > 0


def test_tune_tabular_forecaster(sample_monthly_series: pd.Series):
    """Verify chronological hyperparameter tuning on validation window."""
    train_s = sample_monthly_series.iloc[:48]
    val_s = sample_monthly_series.iloc[48:]

    grid = [{"alpha": 0.1}, {"alpha": 1.0}, {"alpha": 10.0}]
    best_params, df_results, best_model = tune_tabular_forecaster(
        model_type="ridge",
        y_train=train_s,
        y_val=val_s,
        param_grid=grid,
        criterion="MAE",
    )

    assert "alpha" in best_params
    assert len(df_results) == 3
    assert df_results["score"].iloc[0] <= df_results["score"].iloc[-1]
    assert best_model.is_fitted_ is True
