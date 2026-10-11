"""Leakage-safe tabular feature engineering for time-series forecasting.

Provides functions and classes to construct autoregressive lag features,
rolling window summary statistics, and calendar/seasonal indicators.
Ensures strict zero-lookahead anti-leakage compliance by shifting all
historical aggregations prior to target time t.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd


def create_lag_features(
    series: pd.Series | np.ndarray,
    lags: Sequence[int] = (1, 2, 3, 6, 12),
    prefix: str = "lag",
) -> pd.DataFrame:
    """Create autoregressive lag features for a time series.

    For each lag k in lags, feature value at time t is y_{t-k}.

    Args:
        series: 1D time series (pd.Series or np.ndarray).
        lags: Sequence of positive lag orders (e.g. [1, 2, 3, 6, 12]).
        prefix: Column name prefix for created lag features.

    Returns:
        pd.DataFrame containing lag columns, aligned with input series index.

    Raises:
        ValueError: If any lag order is < 1 or if series is empty.
    """
    if len(series) == 0:
        raise ValueError("Cannot create lag features from an empty series.")

    for k in lags:
        if k < 1:
            raise ValueError(f"Lag orders must be >= 1, got {k}.")

    s = pd.Series(series) if not isinstance(series, pd.Series) else series

    lag_data = {f"{prefix}_{k}": s.shift(k) for k in lags}
    return pd.DataFrame(lag_data, index=s.index)


def create_rolling_features(
    series: pd.Series | np.ndarray,
    windows: Sequence[int] = (3, 6, 12),
    metrics: Sequence[str] = ("mean", "std"),
    shift: int = 1,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Create leakage-safe rolling window statistics.

    To guarantee ZERO lookahead leakage, the series is shifted by `shift` (default 1)
    before computing rolling statistics. Therefore, rolling statistics at time t
    only incorporate observations up to t - shift (e.g., t-1, t-2, ...), ensuring
    target y_t is never leaked into features.

    Args:
        series: 1D time series.
        windows: Rolling window lengths (e.g. [3, 6, 12]).
        metrics: Summary metrics to compute ('mean', 'std', 'min', 'max').
        shift: Number of steps to shift series before rolling (must be >= 1).
        min_periods: Minimum number of observations in window. Defaults to window size.

    Returns:
        pd.DataFrame containing rolling feature columns.

    Raises:
        ValueError: If shift < 1, window < 2, or invalid metric requested.
    """
    if shift < 1:
        raise ValueError(
            f"Shift must be >= 1 to prevent lookahead target leakage, got {shift}."
        )

    valid_metrics = {"mean", "std", "min", "max"}
    for m in metrics:
        if m not in valid_metrics:
            raise ValueError(f"Unsupported rolling metric '{m}'. Choose from {valid_metrics}.")

    for w in windows:
        if w < 2:
            raise ValueError(f"Rolling window length must be >= 2, got {w}.")

    s = pd.Series(series) if not isinstance(series, pd.Series) else series
    shifted = s.shift(shift)

    features: dict[str, pd.Series] = {}
    for w in windows:
        mp = min_periods if min_periods is not None else w
        roll = shifted.rolling(window=w, min_periods=mp)
        for m in metrics:
            col_name = f"rolling_{m}_{w}"
            if m == "mean":
                features[col_name] = roll.mean()
            elif m == "std":
                features[col_name] = roll.std()
            elif m == "min":
                features[col_name] = roll.min()
            elif m == "max":
                features[col_name] = roll.max()

    return pd.DataFrame(features, index=s.index)


def create_calendar_features(
    dates: pd.DatetimeIndex | pd.Series,
    include_cyclical: bool = True,
    include_quarter: bool = True,
    include_sales_events: bool = True,
) -> pd.DataFrame:
    """Extract calendar and seasonal indicators from timestamps.

    Computes:
      - month (1-12)
      - quarter (1-4)
      - cyclical sine & cosine of month
      - Steam seasonal event indicators:
        * is_steam_summer_sale (June-July)
        * is_steam_winter_sale (December-January)

    Args:
        dates: DatetimeIndex or Series of timestamps.
        include_cyclical: Whether to include harmonic sine/cosine month encodings.
        include_quarter: Whether to include quarter column.
        include_sales_events: Whether to include Steam sale binary flags.

    Returns:
        pd.DataFrame with calendar feature columns.
    """
    if isinstance(dates, pd.Series):
        dt_index = pd.DatetimeIndex(pd.to_datetime(dates))
    elif isinstance(dates, pd.DatetimeIndex):
        dt_index = dates
    else:
        dt_index = pd.DatetimeIndex(pd.to_datetime(dates))

    feats: dict[str, Any] = {}
    feats["month"] = dt_index.month

    if include_quarter:
        feats["quarter"] = dt_index.quarter

    if include_cyclical:
        # Month cycles over period 12
        feats["sin_month"] = np.sin(2 * np.pi * dt_index.month / 12.0)
        feats["cos_month"] = np.cos(2 * np.pi * dt_index.month / 12.0)

    if include_sales_events:
        # Steam Summer Sale: June (6) & July (7)
        feats["is_steam_summer_sale"] = dt_index.month.isin([6, 7]).astype(int)
        # Steam Winter Sale: December (12) & January (1)
        feats["is_steam_winter_sale"] = dt_index.month.isin([12, 1]).astype(int)

    return pd.DataFrame(feats, index=dt_index)


def build_tabular_feature_matrix(
    y: pd.Series | np.ndarray,
    dates: pd.DatetimeIndex | pd.Series | None = None,
    lags: Sequence[int] = (1, 2, 3, 6, 12),
    windows: Sequence[int] = (3, 6, 12),
    metrics: Sequence[str] = ("mean", "std"),
    include_calendar: bool = True,
    include_cyclical: bool = True,
    include_sales_events: bool = True,
    dropna: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Construct a full tabular feature matrix X and aligned target y.

    Integrates autoregressive lag features, rolling statistics, and calendar features.
    Guarantees strict zero-lookahead leakage.

    Args:
        y: Target series (1D pd.Series or np.ndarray).
        dates: Optional explicit timestamps. If None and y is a pd.Series with
            a DatetimeIndex, the series index is used.
        lags: Autoregressive lag orders.
        windows: Rolling window lengths.
        metrics: Rolling summary metrics.
        include_calendar: Whether to add calendar features.
        include_cyclical: Whether to include cyclical sine/cosine month encodings.
        include_sales_events: Whether to include Steam sale event indicators.
        dropna: If True, drops initial warmup rows where any feature is NaN.

    Returns:
        tuple (X, y_aligned) where X is feature DataFrame and y_aligned is target Series.
    """
    if isinstance(y, pd.Series):
        s = y.copy()
        dt_idx = s.index if isinstance(s.index, pd.DatetimeIndex) else None
    else:
        s = pd.Series(np.asarray(y, dtype=float).ravel(), name="target")
        dt_idx = None

    if dates is not None:
        dt_idx = pd.DatetimeIndex(pd.to_datetime(dates))
        s.index = dt_idx

    feature_frames: list[pd.DataFrame] = []

    # 1. Lags
    if lags:
        df_lags = create_lag_features(s, lags=lags)
        feature_frames.append(df_lags)

    # 2. Rolling features (shift=1 enforced)
    if windows and metrics:
        df_rolling = create_rolling_features(s, windows=windows, metrics=metrics, shift=1)
        feature_frames.append(df_rolling)

    # 3. Calendar features
    if include_calendar and dt_idx is not None:
        df_cal = create_calendar_features(
            dt_idx,
            include_cyclical=include_cyclical,
            include_quarter=True,
            include_sales_events=include_sales_events,
        )
        feature_frames.append(df_cal)

    if not feature_frames:
        raise ValueError("No features requested. Specify lags, windows, or calendar features.")

    X = pd.concat(feature_frames, axis=1)

    if dropna:
        valid_mask = ~X.isna().any(axis=1)
        X = X.loc[valid_mask]
        s_aligned = s.loc[valid_mask]
    else:
        s_aligned = s

    return X, s_aligned


class TabularFeatureExtractor:
    """Stateful feature builder for training and recursive step forecasting.

    Stores feature schema and configuration so that single-step feature vectors
    can be generated recursively during out-of-sample prediction.

    Attributes:
        lags: Lag orders.
        windows: Rolling window lengths.
        metrics: Rolling window metrics.
        include_calendar: Whether calendar features are included.
        include_cyclical: Whether cyclical month features are included.
        include_sales_events: Whether Steam sale indicators are included.
        feature_names_: List of generated feature column names.
    """

    def __init__(
        self,
        lags: Sequence[int] = (1, 2, 3, 6, 12),
        windows: Sequence[int] = (3, 6, 12),
        metrics: Sequence[str] = ("mean", "std"),
        include_calendar: bool = True,
        include_cyclical: bool = True,
        include_sales_events: bool = True,
    ) -> None:
        self.lags = list(lags)
        self.windows = list(windows)
        self.metrics = list(metrics)
        self.include_calendar = include_calendar
        self.include_cyclical = include_cyclical
        self.include_sales_events = include_sales_events
        self.feature_names_: list[str] = []

    def fit_transform(
        self,
        y: pd.Series | np.ndarray,
        dates: pd.DatetimeIndex | None = None,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Fit feature schema and return (X, y) training matrices."""
        X, y_aligned = build_tabular_feature_matrix(
            y=y,
            dates=dates,
            lags=self.lags,
            windows=self.windows,
            metrics=self.metrics,
            include_calendar=self.include_calendar,
            include_cyclical=self.include_cyclical,
            include_sales_events=self.include_sales_events,
            dropna=True,
        )
        self.feature_names_ = list(X.columns)
        return X, y_aligned

    def extract_step_features(
        self,
        history: list[float] | np.ndarray,
        step_date: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Extract a single-row feature DataFrame for recursive 1-step forecasting.

        Given a historical observation buffer up to time t-1, constructs the feature
        vector needed to forecast observation y_t.

        Args:
            history: List or 1D array of values up to step t-1.
            step_date: Timestamp corresponding to the forecasted step t (for calendar feats).

        Returns:
            pd.DataFrame of shape (1, num_features) with columns matching feature_names_.
        """
        hist_arr = np.asarray(history, dtype=float).ravel()
        feats: dict[str, float] = {}

        # 1. Lags: y_{t-k}
        for k in self.lags:
            if len(hist_arr) < k:
                raise ValueError(
                    f"History buffer length ({len(hist_arr)}) is shorter than required lag {k}."
                )
            feats[f"lag_{k}"] = float(hist_arr[-k])

        # 2. Rolling stats: over history up to t-1
        for w in self.windows:
            if len(hist_arr) < w:
                raise ValueError(
                    f"History buffer length ({len(hist_arr)}) is shorter than window {w}."
                )
            sub = hist_arr[-w:]
            for m in self.metrics:
                col_name = f"rolling_{m}_{w}"
                if m == "mean":
                    feats[col_name] = float(np.mean(sub))
                elif m == "std":
                    feats[col_name] = float(np.std(sub, ddof=1)) if len(sub) > 1 else 0.0
                elif m == "min":
                    feats[col_name] = float(np.min(sub))
                elif m == "max":
                    feats[col_name] = float(np.max(sub))

        # 3. Calendar features for target step_date
        if self.include_calendar and step_date is not None:
            m = step_date.month
            feats["month"] = float(m)
            feats["quarter"] = float(step_date.quarter)
            if self.include_cyclical:
                feats["sin_month"] = float(np.sin(2 * np.pi * m / 12.0))
                feats["cos_month"] = float(np.cos(2 * np.pi * m / 12.0))
            if self.include_sales_events:
                feats["is_steam_summer_sale"] = 1.0 if m in (6, 7) else 0.0
                feats["is_steam_winter_sale"] = 1.0 if m in (12, 1) else 0.0

        df_row = pd.DataFrame([feats])
        if self.feature_names_:
            # Ensure exact column ordering
            return df_row[self.feature_names_]
        return df_row
