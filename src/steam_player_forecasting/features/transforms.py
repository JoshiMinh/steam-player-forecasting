"""Time-series variance-stabilizing and stationarity transformations with leakage-free inversion."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import inv_boxcox


class LogTransformer:
    """Natural logarithm transformer with automatic positive offset shift.

    Applies y' = log(y + c) where c is fitted strictly on training data
    to guarantee y + c > 0.

    Attributes:
        offset: Manual offset or "auto".
        offset_: Fitted offset value from training data.
        is_fitted_: Boolean indicating whether transformer has been fitted.
    """

    def __init__(self, offset: float | str = "auto") -> None:
        self.offset = offset
        self.offset_: float = 0.0
        self.is_fitted_: bool = False

    def fit(self, X: pd.Series | pd.DataFrame | np.ndarray) -> LogTransformer:
        """Fit offset strictly on training data."""
        arr = np.asarray(X, dtype=float)
        min_val = np.nanmin(arr)

        if self.offset == "auto":
            if min_val <= 0:
                # Shift minimum to 1.0 to ensure log(X + offset) >= 0
                self.offset_ = float(1.0 - min_val)
            else:
                self.offset_ = 0.0
        else:
            self.offset_ = float(self.offset)
            if min_val + self.offset_ <= 0:
                raise ValueError(
                    f"Specified offset {self.offset_} results in non-positive values (min: {min_val})."
                )

        self.is_fitted_ = True
        return self

    def transform(
        self, X: pd.Series | pd.DataFrame | np.ndarray
    ) -> pd.Series | pd.DataFrame | np.ndarray:
        """Apply log transformation using fitted offset."""
        if not self.is_fitted_:
            raise ValueError("LogTransformer must be fitted before transforming data.")

        if isinstance(X, pd.Series):
            shifted = X + self.offset_
            if (shifted <= 0).any():
                raise ValueError("Found values <= 0 after applying fitted offset.")
            return np.log(shifted)
        elif isinstance(X, pd.DataFrame):
            shifted = X + self.offset_
            if (shifted <= 0).any().any():
                raise ValueError("Found values <= 0 after applying fitted offset.")
            return np.log(shifted)
        else:
            arr = np.asarray(X, dtype=float)
            shifted = arr + self.offset_
            if np.any(shifted <= 0):
                raise ValueError("Found values <= 0 after applying fitted offset.")
            return np.log(shifted)

    def fit_transform(
        self, X: pd.Series | pd.DataFrame | np.ndarray
    ) -> pd.Series | pd.DataFrame | np.ndarray:
        """Fit to training data, then transform."""
        return self.fit(X).transform(X)

    def inverse_transform(
        self, X: pd.Series | pd.DataFrame | np.ndarray
    ) -> pd.Series | pd.DataFrame | np.ndarray:
        """Invert log transformation using fitted offset."""
        if not self.is_fitted_:
            raise ValueError("LogTransformer must be fitted before inverse transforming.")

        if isinstance(X, pd.Series):
            return np.exp(X) - self.offset_
        elif isinstance(X, pd.DataFrame):
            return np.exp(X) - self.offset_
        else:
            arr = np.asarray(X, dtype=float)
            return np.exp(arr) - self.offset_


class BoxCoxTransformer:
    """Box-Cox power transformer with ML lambda estimation and automatic offset.

    Fits lambda strictly on training data via maximum likelihood estimation:
      y^(lambda) = ((y + c)^lambda - 1) / lambda  if lambda != 0
                 = log(y + c)                     if lambda == 0

    Attributes:
        offset: Manual offset or "auto".
        lambda_: Estimated Box-Cox power parameter.
        offset_: Fitted offset value from training data.
        is_fitted_: Boolean indicating whether transformer has been fitted.
    """

    def __init__(self, offset: float | str = "auto") -> None:
        self.offset = offset
        self.lambda_: float | None = None
        self.offset_: float = 0.0
        self.is_fitted_: bool = False

    def fit(self, X: pd.Series | pd.DataFrame | np.ndarray) -> BoxCoxTransformer:
        """Fit Box-Cox lambda and offset strictly on training data."""
        arr = np.asarray(X, dtype=float).ravel()
        clean_arr = arr[~np.isnan(arr)]

        min_val = np.min(clean_arr)
        if self.offset == "auto":
            if min_val <= 0:
                self.offset_ = float(1.0 - min_val)
            else:
                self.offset_ = 0.0
        else:
            self.offset_ = float(self.offset)
            if min_val + self.offset_ <= 0:
                raise ValueError(
                    f"Specified offset {self.offset_} results in non-positive values (min: {min_val})."
                )

        shifted = clean_arr + self.offset_
        _, lmbda = stats.boxcox(shifted)
        self.lambda_ = float(lmbda)
        self.is_fitted_ = True
        return self

    def transform(
        self, X: pd.Series | pd.DataFrame | np.ndarray
    ) -> pd.Series | pd.DataFrame | np.ndarray:
        """Apply Box-Cox transformation using fitted lambda and offset."""
        if not self.is_fitted_ or self.lambda_ is None:
            raise ValueError("BoxCoxTransformer must be fitted before transforming.")

        if isinstance(X, pd.Series):
            shifted = X + self.offset_
            if (shifted <= 0).any():
                raise ValueError("Encountered values <= 0 when applying Box-Cox transform.")
            res = stats.boxcox(shifted.to_numpy(), lmbda=self.lambda_)
            return pd.Series(res, index=X.index, name=X.name)
        elif isinstance(X, pd.DataFrame):
            res_df = X.copy()
            for col in res_df.columns:
                shifted = res_df[col] + self.offset_
                if (shifted <= 0).any():
                    raise ValueError(f"Encountered values <= 0 in column {col}.")
                res_df[col] = stats.boxcox(shifted.to_numpy(), lmbda=self.lambda_)
            return res_df
        else:
            arr = np.asarray(X, dtype=float)
            shifted = arr + self.offset_
            if np.any(shifted <= 0):
                raise ValueError("Encountered values <= 0 when applying Box-Cox transform.")
            return stats.boxcox(shifted, lmbda=self.lambda_)

    def fit_transform(
        self, X: pd.Series | pd.DataFrame | np.ndarray
    ) -> pd.Series | pd.DataFrame | np.ndarray:
        """Fit to training data, then transform."""
        return self.fit(X).transform(X)

    def inverse_transform(
        self, X: pd.Series | pd.DataFrame | np.ndarray
    ) -> pd.Series | pd.DataFrame | np.ndarray:
        """Invert Box-Cox transformation using fitted lambda and offset."""
        if not self.is_fitted_ or self.lambda_ is None:
            raise ValueError("BoxCoxTransformer must be fitted before inverse transforming.")

        if isinstance(X, pd.Series):
            inv = inv_boxcox(X.to_numpy(), self.lambda_) - self.offset_
            return pd.Series(inv, index=X.index, name=X.name)
        elif isinstance(X, pd.DataFrame):
            res_df = X.copy()
            for col in res_df.columns:
                res_df[col] = inv_boxcox(res_df[col].to_numpy(), self.lambda_) - self.offset_
            return res_df
        else:
            arr = np.asarray(X, dtype=float)
            return inv_boxcox(arr, self.lambda_) - self.offset_


def difference_series(
    series: pd.Series | np.ndarray,
    order: int = 1,
    seasonal_period: int = 0,
    drop_na: bool = False,
) -> pd.Series | np.ndarray:
    """Apply non-seasonal and/or seasonal differencing to a 1D time-series.

    Non-seasonal differencing: y'_t = y_t - y_{t-1} (repeated `order` times)
    Seasonal differencing: y''_t = y'_t - y'_{t-s}

    Args:
        series: 1D input series or array.
        order: Order of regular differencing (d >= 0).
        seasonal_period: Seasonal lag period (s >= 0).
        drop_na: Whether to drop NaN observations at the start.

    Returns:
        Differenced series or array.
    """
    if order < 0 or seasonal_period < 0:
        raise ValueError("Differencing order and seasonal_period must be >= 0.")

    is_series = isinstance(series, pd.Series)
    s = series.copy() if is_series else pd.Series(series)

    # Apply regular differencing
    for _ in range(order):
        s = s.diff()

    # Apply seasonal differencing
    if seasonal_period > 0:
        s = s.diff(seasonal_period)

    if drop_na:
        s = s.dropna()

    return s if is_series else s.to_numpy()


def invert_difference_series(
    diff_values: pd.Series | np.ndarray,
    history: pd.Series | np.ndarray,
    order: int = 1,
    seasonal_period: int = 0,
) -> pd.Series | np.ndarray:
    """Invert differenced values back to original level scale using historical context.

    Reconstructs the original time-series level:
      - For regular differencing d=1: y_t = y_{t-1} + diff_t (cumulative sum)
      - For seasonal differencing s>0: y_t = y_{t-s} + diff_t

    Args:
        diff_values: Differenced predictions or observations to invert.
        history: Historical original-scale observations immediately preceding diff_values.
        order: Regular differencing order (currently supports d=1).
        seasonal_period: Seasonal period (e.g. 12 for annual seasonality).

    Returns:
        Reconstructed series or array on the original level scale.
    """
    if order not in (0, 1):
        raise NotImplementedError("Only differencing order d=0 or d=1 is currently supported.")

    is_series = isinstance(diff_values, pd.Series)
    diff_arr = np.asarray(diff_values, dtype=float).ravel()
    hist_arr = np.asarray(history, dtype=float).ravel()

    # Case 0: No differencing
    if order == 0 and seasonal_period == 0:
        return diff_values

    # Case 1: Pure regular differencing d=1
    if order == 1 and seasonal_period == 0:
        if len(hist_arr) < 1:
            raise ValueError("At least 1 historical observation required to invert d=1 difference.")
        last_val = hist_arr[-1]
        reconstructed = np.empty_like(diff_arr)
        current = last_val
        for i, delta in enumerate(diff_arr):
            current = current + delta
            reconstructed[i] = current

        if is_series:
            return pd.Series(reconstructed, index=diff_values.index, name=diff_values.name)
        return reconstructed

    # Case 2: Pure seasonal differencing s > 0, d = 0
    if order == 0 and seasonal_period > 0:
        if len(hist_arr) < seasonal_period:
            raise ValueError(
                f"At least {seasonal_period} historical observations required to invert seasonal difference."
            )
        buffer = list(hist_arr[-seasonal_period:])
        reconstructed = np.empty_like(diff_arr)
        for i, delta in enumerate(diff_arr):
            val = buffer[i] + delta
            buffer.append(val)
            reconstructed[i] = val

        if is_series:
            return pd.Series(reconstructed, index=diff_values.index, name=diff_values.name)
        return reconstructed

    # Case 3: Both regular d=1 and seasonal s > 0
    # y'_t = (1 - B)(1 - B^s) y_t
    # Invert regular first, then seasonal
    if order == 1 and seasonal_period > 0:
        min_hist = seasonal_period + 1
        if len(hist_arr) < min_hist:
            raise ValueError(
                f"At least {min_hist} historical observations required to invert combined d=1, s={seasonal_period} difference."
            )
        # Seasonal buffer
        # Reconstruct sequentially: y_t = y_{t-1} + y_{t-s} - y_{t-s-1} + diff_t
        full_hist = list(hist_arr)
        reconstructed = np.empty_like(diff_arr)
        for i, delta in enumerate(diff_arr):
            y_t_minus_1 = full_hist[-1]
            y_t_minus_s = full_hist[-seasonal_period]
            y_t_minus_s_minus_1 = full_hist[-seasonal_period - 1]
            y_t = y_t_minus_1 + y_t_minus_s - y_t_minus_s_minus_1 + delta
            full_hist.append(y_t)
            reconstructed[i] = y_t

        if is_series:
            return pd.Series(reconstructed, index=diff_values.index, name=diff_values.name)
        return reconstructed

    raise ValueError("Invalid differencing combination.")


class DifferencingTransformer:
    """Stateful differencing transformer storing training history for leakage-free inversion.

    Attributes:
        order: Regular differencing order.
        seasonal_period: Seasonal differencing lag.
        history_: Last observed training values required for out-of-sample inversion.
    """

    def __init__(self, order: int = 1, seasonal_period: int = 0) -> None:
        self.order = order
        self.seasonal_period = seasonal_period
        self.history_: np.ndarray | None = None
        self.is_fitted_: bool = False

    def fit(self, X: pd.Series | np.ndarray) -> DifferencingTransformer:
        """Store the boundary history needed to invert subsequent transformations."""
        arr = np.asarray(X, dtype=float).ravel()
        clean = arr[~np.isnan(arr)]
        needed = max(self.order, self.seasonal_period) + 1
        if len(clean) < needed:
            raise ValueError(f"Need at least {needed} non-NaN observations to fit differencing.")

        self.history_ = clean[-needed:]
        self.is_fitted_ = True
        return self

    def transform(
        self, X: pd.Series | np.ndarray, drop_na: bool = False
    ) -> pd.Series | np.ndarray:
        """Apply differencing."""
        return difference_series(
            series=X,
            order=self.order,
            seasonal_period=self.seasonal_period,
            drop_na=drop_na,
        )

    def fit_transform(
        self, X: pd.Series | np.ndarray, drop_na: bool = False
    ) -> pd.Series | np.ndarray:
        """Fit history on training series, then transform."""
        self.fit(X)
        return self.transform(X, drop_na=drop_na)

    def inverse_transform(
        self,
        diff_values: pd.Series | np.ndarray,
        history: pd.Series | np.ndarray | None = None,
    ) -> pd.Series | np.ndarray:
        """Invert differenced series using stored training history or supplied history."""
        if history is None:
            if not self.is_fitted_ or self.history_ is None:
                raise ValueError("DifferencingTransformer must be fitted before inverse transform.")
            history = self.history_

        return invert_difference_series(
            diff_values=diff_values,
            history=history,
            order=self.order,
            seasonal_period=self.seasonal_period,
        )
