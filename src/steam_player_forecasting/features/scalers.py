"""Train-only fitted scalers for time-series data ensuring zero lookahead leakage."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

ScalerType = Literal["minmax", "standard", "robust"]


class TimeSeriesScaler:
    """Modular time-series feature scaler strictly fitted on training data.

    Supports MinMaxScaler, StandardScaler, and RobustScaler.
    Seamlessly handles 1D Series, 2D DataFrames, and NumPy arrays while
    preserving index, columns, and dimensional shapes across inverse transforms.

    Attributes:
        scaler_type: Scaling method ('minmax', 'standard', or 'robust').
        feature_range: Target range tuple (min, max) for MinMaxScaler.
        with_mean: Whether to center data for StandardScaler / RobustScaler.
        with_std: Whether to scale to unit variance/scale.
        scaler: The underlying Scikit-learn scaler instance.
        is_fitted_: Boolean indicating if scaler has been fitted.
    """

    def __init__(
        self,
        scaler_type: ScalerType = "standard",
        feature_range: tuple[float, float] = (0.0, 1.0),
        with_mean: bool = True,
        with_std: bool = True,
    ) -> None:
        self.scaler_type = scaler_type.lower()
        self.feature_range = feature_range
        self.with_mean = with_mean
        self.with_std = with_std
        self.is_fitted_: bool = False

        if self.scaler_type == "minmax":
            self.scaler = MinMaxScaler(feature_range=feature_range)
        elif self.scaler_type == "standard":
            self.scaler = StandardScaler(with_mean=with_mean, with_std=with_std)
        elif self.scaler_type == "robust":
            self.scaler = RobustScaler(with_centering=with_mean, with_scaling=with_std)
        else:
            raise ValueError(
                f"Unsupported scaler_type '{scaler_type}'. "
                "Choose from: 'minmax', 'standard', 'robust'."
            )

    def _prepare_2d(
        self, X: pd.Series | pd.DataFrame | np.ndarray
    ) -> tuple[np.ndarray, bool, bool, Any, Any]:
        """Convert input to 2D numpy array and retain metadata for reconstruction."""
        is_series = isinstance(X, pd.Series)
        is_dataframe = isinstance(X, pd.DataFrame)
        original_index = X.index if (is_series or is_dataframe) else None
        original_columns = X.columns if is_dataframe else (X.name if is_series else None)

        if is_series:
            arr = X.to_numpy().reshape(-1, 1)
        elif is_dataframe:
            arr = X.to_numpy()
        else:
            arr = np.asarray(X)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)

        return arr, is_series, is_dataframe, original_index, original_columns

    def fit(self, X: pd.Series | pd.DataFrame | np.ndarray) -> TimeSeriesScaler:
        """Fit scaler parameters strictly on training observations.

        Args:
            X: Training dataset (1D or 2D).

        Returns:
            Self (fitted scaler).
        """
        arr_2d, _, _, _, _ = self._prepare_2d(X)
        self.scaler.fit(arr_2d)
        self.is_fitted_ = True
        return self

    def transform(
        self, X: pd.Series | pd.DataFrame | np.ndarray
    ) -> pd.Series | pd.DataFrame | np.ndarray:
        """Transform data using parameters fitted strictly on training data.

        Args:
            X: Data to transform (train, val, or test).

        Returns:
            Transformed data matching original input type and index.
        """
        if not self.is_fitted_:
            raise NotFittedError(
                f"TimeSeriesScaler({self.scaler_type}) must be fitted before transforming."
            )

        arr_2d, is_series, is_dataframe, original_index, original_columns = self._prepare_2d(X)
        transformed = self.scaler.transform(arr_2d)

        if is_series:
            return pd.Series(transformed.ravel(), index=original_index, name=original_columns)
        elif is_dataframe:
            return pd.DataFrame(transformed, index=original_index, columns=original_columns)
        else:
            if np.asarray(X).ndim == 1:
                return transformed.ravel()
            return transformed

    def fit_transform(
        self, X: pd.Series | pd.DataFrame | np.ndarray
    ) -> pd.Series | pd.DataFrame | np.ndarray:
        """Fit strictly on X, then transform X."""
        return self.fit(X).transform(X)

    def inverse_transform(
        self, X: pd.Series | pd.DataFrame | np.ndarray
    ) -> pd.Series | pd.DataFrame | np.ndarray:
        """Invert scaled data back to original physical units.

        Args:
            X: Scaled data to inverse transform.

        Returns:
            Data reconstructed in original level scale.
        """
        if not self.is_fitted_:
            raise NotFittedError(
                f"TimeSeriesScaler({self.scaler_type}) must be fitted before inverse transforming."
            )

        arr_2d, is_series, is_dataframe, original_index, original_columns = self._prepare_2d(X)
        inverted = self.scaler.inverse_transform(arr_2d)

        if is_series:
            return pd.Series(inverted.ravel(), index=original_index, name=original_columns)
        elif is_dataframe:
            return pd.DataFrame(inverted, index=original_index, columns=original_columns)
        else:
            if np.asarray(X).ndim == 1:
                return inverted.ravel()
            return inverted

    @property
    def statistics_(self) -> dict[str, Any]:
        """Return fitted scaling statistics."""
        if not self.is_fitted_:
            raise NotFittedError("Scaler is not fitted yet.")

        stats: dict[str, Any] = {"scaler_type": self.scaler_type}
        if hasattr(self.scaler, "mean_"):
            stats["mean"] = self.scaler.mean_
        if hasattr(self.scaler, "scale_"):
            stats["scale"] = self.scaler.scale_
        if hasattr(self.scaler, "var_"):
            stats["var"] = self.scaler.var_
        if hasattr(self.scaler, "data_min_"):
            stats["data_min"] = self.scaler.data_min_
        if hasattr(self.scaler, "data_max_"):
            stats["data_max"] = self.scaler.data_max_
        if hasattr(self.scaler, "center_"):
            stats["center"] = self.scaler.center_
        return stats
