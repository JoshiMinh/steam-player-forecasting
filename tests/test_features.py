"""Unit tests for Phase 2 feature engineering, transformations, scalers, and splitting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from steam_player_forecasting.features.diagnostics import (
    check_stationarity,
    compute_acf_pacf,
    compute_rolling_stats,
    decompose_time_series,
)
from steam_player_forecasting.features.scalers import TimeSeriesScaler
from steam_player_forecasting.features.split import (
    TimeSeriesSplitResult,
    expanding_window_cv,
    split_by_game,
    train_val_test_split,
)
from steam_player_forecasting.features.transforms import (
    BoxCoxTransformer,
    DifferencingTransformer,
    LogTransformer,
    difference_series,
    invert_difference_series,
)


@pytest.fixture
def sample_monthly_series() -> pd.Series:
    """Generate a realistic synthetic 60-month time-series with trend and seasonality."""
    np.random.seed(42)
    dates = pd.date_range("2016-01-01", periods=60, freq="MS")
    trend = np.linspace(1000, 5000, 60)
    month_factors = np.sin(2 * np.pi * np.arange(60) / 12) * 500
    noise = np.random.normal(0, 100, 60)
    values = trend + month_factors + noise
    return pd.Series(values, index=dates, name="Avg_players")


@pytest.fixture
def sample_multi_game_df(sample_monthly_series: pd.Series) -> pd.DataFrame:
    """Generate a multi-game panel DataFrame."""
    df1 = pd.DataFrame(
        {
            "Game_Name": "Game A",
            "Month_Year": sample_monthly_series.index,
            "Avg_players": sample_monthly_series.values,
        }
    )
    df2 = pd.DataFrame(
        {
            "Game_Name": "Game B",
            "Month_Year": sample_monthly_series.index,
            "Avg_players": sample_monthly_series.values * 1.5,
        }
    )
    return pd.concat([df1, df2], ignore_index=True)


# ==============================================================================
# 1. Splitting Tests
# ==============================================================================


def test_train_val_test_split_shapes_and_dates(sample_monthly_series: pd.Series):
    """Verify chronological split sizes and date boundaries."""
    df = sample_monthly_series.reset_index()
    df.columns = ["Month_Year", "Avg_players"]

    res = train_val_test_split(df, val_months=12, test_months=12, date_col="Month_Year")
    assert isinstance(res, TimeSeriesSplitResult)
    assert len(res.train) == 36
    assert len(res.val) == 12
    assert len(res.test) == 12
    assert len(res.train) + len(res.val) + len(res.test) == 60

    # Ensure strict temporal monotonicity
    train_dates = pd.to_datetime(res.train["Month_Year"])
    val_dates = pd.to_datetime(res.val["Month_Year"])
    test_dates = pd.to_datetime(res.test["Month_Year"])

    assert train_dates.max() < val_dates.min()
    assert val_dates.max() < test_dates.min()

    summary = res.summary()
    assert summary["n_train"] == 36
    assert summary["n_val"] == 12
    assert summary["n_test"] == 12


def test_train_val_test_split_series_and_array(sample_monthly_series: pd.Series):
    """Verify splitting works seamlessly on 1D Series and raw numpy arrays."""
    # Series
    res_s = train_val_test_split(sample_monthly_series, val_months=12, test_months=12)
    assert isinstance(res_s.train, pd.Series)
    assert len(res_s.train) == 36
    assert len(res_s.val) == 12
    assert len(res_s.test) == 12

    # Numpy array
    arr = sample_monthly_series.to_numpy()
    res_a = train_val_test_split(arr, val_months=12, test_months=12)
    assert isinstance(res_a.train, np.ndarray)
    assert len(res_a.train) == 36
    assert len(res_a.val) == 12
    assert len(res_a.test) == 12


def test_train_val_test_split_insufficient_data():
    """Verify ValueError is raised when series length is too short for requested holdout."""
    short_series = pd.Series([10.0, 20.0, 30.0])
    with pytest.raises(ValueError, match="Insufficient observations"):
        train_val_test_split(short_series, val_months=12, test_months=12)


def test_split_by_game(sample_multi_game_df: pd.DataFrame):
    """Verify split_by_game splits each game's time series independently."""
    splits = split_by_game(sample_multi_game_df, val_months=12, test_months=12)
    assert "Game A" in splits
    assert "Game B" in splits
    assert len(splits["Game A"].train) == 36
    assert len(splits["Game B"].test) == 12


def test_expanding_window_cv(sample_monthly_series: pd.Series):
    """Verify walk-forward expanding window cross validation generator."""
    folds = list(
        expanding_window_cv(
            sample_monthly_series,
            n_splits=3,
            test_horizon=12,
            min_train_months=24,
        )
    )
    assert len(folds) == 3
    # Check fold sizes expanding
    train0, test0 = folds[0]
    train1, test1 = folds[1]
    train2, test2 = folds[2]

    assert len(test0) == 12
    assert len(test1) == 12
    assert len(test2) == 12
    assert len(train0) == 24
    assert len(train1) == 36
    assert len(train2) == 48


# ==============================================================================
# 2. Scaler Tests (Leakage-Prevention & Invertibility)
# ==============================================================================


@pytest.mark.parametrize("scaler_type", ["minmax", "standard", "robust"])
def test_scalers_fit_strictly_on_train_and_invert(
    scaler_type: str, sample_monthly_series: pd.Series
):
    """Verify scalers fit only on train and invert back with high numerical precision."""
    res = train_val_test_split(sample_monthly_series, val_months=12, test_months=12)
    train_data = res.train
    test_data = res.test

    scaler = TimeSeriesScaler(scaler_type=scaler_type)  # type: ignore

    # Verify NotFittedError before fitting
    with pytest.raises(NotFittedError):
        scaler.transform(train_data)
    with pytest.raises(NotFittedError):
        scaler.inverse_transform(train_data)

    # Fit strictly on train
    scaler.fit(train_data)
    assert scaler.is_fitted_

    # Verify fitted parameters reflect train ONLY
    stats = scaler.statistics_
    if scaler_type == "minmax":
        assert np.isclose(stats["data_min"][0], train_data.min())
        assert np.isclose(stats["data_max"][0], train_data.max())
    elif scaler_type == "standard":
        assert np.isclose(stats["mean"][0], train_data.mean())
    elif scaler_type == "robust":
        assert np.isclose(stats["center"][0], np.median(train_data))

    # Transform train and test
    train_scaled = scaler.transform(train_data)
    test_scaled = scaler.transform(test_data)

    # Train scaled values match expected ranges
    if scaler_type == "minmax":
        assert np.isclose(train_scaled.min(), 0.0)
        assert np.isclose(train_scaled.max(), 1.0)
        # Test data has upward trend, so test_scaled max should exceed 1.0 (PROVING NO LEAKAGE!)
        assert test_scaled.max() > 1.0

    # Verify exact inversion
    train_inv = scaler.inverse_transform(train_scaled)
    test_inv = scaler.inverse_transform(test_scaled)

    np.testing.assert_allclose(train_inv.to_numpy(), train_data.to_numpy(), rtol=1e-5)
    np.testing.assert_allclose(test_inv.to_numpy(), test_data.to_numpy(), rtol=1e-5)
    assert isinstance(train_inv, pd.Series)
    assert train_inv.name == sample_monthly_series.name


def test_scaler_2d_dataframe():
    """Verify scaler works on 2D DataFrame and preserves columns."""
    df = pd.DataFrame(
        {
            "col1": [10.0, 20.0, 30.0, 40.0],
            "col2": [100.0, 200.0, 300.0, 400.0],
        }
    )
    scaler = TimeSeriesScaler(scaler_type="minmax")
    scaled_df = scaler.fit_transform(df)
    assert isinstance(scaled_df, pd.DataFrame)
    assert list(scaled_df.columns) == ["col1", "col2"]
    inv_df = scaler.inverse_transform(scaled_df)
    np.testing.assert_allclose(inv_df.values, df.values, rtol=1e-5)


# ==============================================================================
# 3. Transformation Tests (Log, Box-Cox, Differencing)
# ==============================================================================


def test_log_transformer_leakage_and_inversion(sample_monthly_series: pd.Series):
    """Verify LogTransformer fits offset on train and inverts accurately."""
    res = train_val_test_split(sample_monthly_series, val_months=12, test_months=12)
    transformer = LogTransformer(offset="auto")

    # Not fitted check
    with pytest.raises(ValueError, match="must be fitted"):
        transformer.transform(res.train)

    train_log = transformer.fit_transform(res.train)
    assert transformer.is_fitted_
    assert transformer.offset_ == 0.0  # All values > 0

    test_log = transformer.transform(res.test)
    test_inv = transformer.inverse_transform(test_log)

    np.testing.assert_allclose(test_inv.to_numpy(), res.test.to_numpy(), rtol=1e-6)


def test_log_transformer_negative_values():
    """Verify LogTransformer automatically shifts negative values."""
    series_with_neg = pd.Series([-5.0, 0.0, 10.0, 20.0])
    transformer = LogTransformer(offset="auto")
    transformed = transformer.fit_transform(series_with_neg)
    assert transformer.offset_ == 6.0  # -5 + 6 = 1.0
    assert (transformed >= 0.0).all()

    inverted = transformer.inverse_transform(transformed)
    np.testing.assert_allclose(inverted.to_numpy(), series_with_neg.to_numpy(), rtol=1e-6)


def test_boxcox_transformer_leakage_and_inversion(sample_monthly_series: pd.Series):
    """Verify BoxCoxTransformer estimates lambda strictly on train and inverts cleanly."""
    res = train_val_test_split(sample_monthly_series, val_months=12, test_months=12)
    transformer = BoxCoxTransformer()

    with pytest.raises(ValueError, match="must be fitted"):
        transformer.transform(res.train)

    train_bc = transformer.fit_transform(res.train)
    assert transformer.is_fitted_
    assert transformer.lambda_ is not None

    test_bc = transformer.transform(res.test)
    test_inv = transformer.inverse_transform(test_bc)

    np.testing.assert_allclose(test_inv.to_numpy(), res.test.to_numpy(), rtol=1e-5)


def test_differencing_regular_order_1(sample_monthly_series: pd.Series):
    """Verify regular differencing and reconstruction using history buffer."""
    res = train_val_test_split(sample_monthly_series, val_months=12, test_months=12)
    transformer = DifferencingTransformer(order=1, seasonal_period=0)

    # Fit stores last training observations
    transformer.fit(res.train)
    assert transformer.history_ is not None
    assert np.isclose(transformer.history_[-1], res.train.iloc[-1])

    # Out-of-sample differenced ground-truth
    # Combine last train point with val to get exact val diffs
    val_diff = difference_series(
        pd.concat([res.train.iloc[-1:], res.val]), order=1, drop_na=True
    )
    assert len(val_diff) == len(res.val)

    # Invert using transformer with stored history
    reconstructed_val = transformer.inverse_transform(val_diff)
    np.testing.assert_allclose(reconstructed_val.to_numpy(), res.val.to_numpy(), rtol=1e-5)


def test_differencing_seasonal_order_12(sample_monthly_series: pd.Series):
    """Verify seasonal lag-12 differencing and reconstruction."""
    res = train_val_test_split(sample_monthly_series, val_months=12, test_months=12)
    transformer = DifferencingTransformer(order=0, seasonal_period=12)
    transformer.fit(res.train)

    # Seasonal diff of val using trailing 12 observations of train
    hist_12 = res.train.iloc[-12:]
    combined = pd.concat([hist_12, res.val])
    val_diff = difference_series(combined, order=0, seasonal_period=12, drop_na=True)
    assert len(val_diff) == len(res.val)

    reconstructed_val = invert_difference_series(
        diff_values=val_diff,
        history=hist_12,
        order=0,
        seasonal_period=12,
    )
    np.testing.assert_allclose(reconstructed_val.to_numpy(), res.val.to_numpy(), rtol=1e-5)


# ==============================================================================
# 4. Diagnostics & Stationarity Tests
# ==============================================================================


def test_stationarity_on_stationary_vs_random_walk():
    """Verify ADF and KPSS statistical tests correctly distinguish stationary and unit-root processes."""
    np.random.seed(123)
    # 1. Stationary white noise
    stationary_data = pd.Series(np.random.normal(0, 1, 100))
    res_stat = check_stationarity(stationary_data)
    assert "adf" in res_stat and "kpss" in res_stat
    assert res_stat["adf"]["is_stationary"] is True

    # 2. Non-stationary random walk
    random_walk = pd.Series(np.cumsum(np.random.normal(0, 1, 100)) + 50)
    res_walk = check_stationarity(random_walk)
    assert res_walk["is_stationary"] is False


def test_compute_acf_pacf(sample_monthly_series: pd.Series):
    """Verify ACF/PACF computation structure and bounds."""
    res = compute_acf_pacf(sample_monthly_series, nlags=12)
    assert "lags" in res
    assert "acf" in res
    assert "pacf" in res
    assert len(res["acf"]) == 13  # lag 0 to 12
    assert len(res["pacf"]) == 13
    assert np.isclose(res["acf"][0], 1.0)
    assert np.isclose(res["pacf"][0], 1.0)


def test_decompose_time_series(sample_monthly_series: pd.Series):
    """Verify STL and classical decomposition reconstruct observed series."""
    # STL
    stl_res = decompose_time_series(sample_monthly_series, period=12, method="stl")
    reconstructed_stl = stl_res["trend"] + stl_res["seasonal"] + stl_res["resid"]
    np.testing.assert_allclose(
        reconstructed_stl.to_numpy(), sample_monthly_series.to_numpy(), rtol=1e-4
    )

    # Classical Additive
    class_res = decompose_time_series(
        sample_monthly_series, period=12, model="additive", method="classical"
    )
    clean_obs = sample_monthly_series.dropna()
    valid_mask = ~class_res["trend"].isna()
    recon_class = (
        class_res["trend"][valid_mask]
        + class_res["seasonal"][valid_mask]
        + class_res["resid"][valid_mask]
    )
    np.testing.assert_allclose(
        recon_class.to_numpy(), clean_obs[valid_mask].to_numpy(), rtol=1e-4
    )


def test_compute_rolling_stats(sample_monthly_series: pd.Series):
    """Verify rolling statistics calculations."""
    rolling_df = compute_rolling_stats(sample_monthly_series, windows=[3, 6, 12])
    assert "rolling_mean_3" in rolling_df.columns
    assert "rolling_std_6" in rolling_df.columns
    assert "rolling_var_12" in rolling_df.columns
    assert len(rolling_df) == len(sample_monthly_series)
