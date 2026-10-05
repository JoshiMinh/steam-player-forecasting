"""Base classes and meta-forecasters for time-series forecasting."""

from __future__ import annotations

import abc
from typing import Any, Literal

import numpy as np
import pandas as pd

from steam_player_forecasting.evaluation.metrics import calculate_metrics


class BaseForecaster(abc.ABC):
    """Abstract base class for all univariate forecasting models.

    Provides a scikit-learn / statsmodels compatible interface:
      - fit(y): fits model on training series.
      - predict(steps): generates out-of-sample point forecast.
      - forecast(steps): alias for predict(steps).
    Preserves pandas DatetimeIndex across predictions when available.

    Attributes:
        is_fitted_: Boolean indicating if model has been successfully fitted.
        y_train_: The original 1D training series used for fitting.
        index_: Original training index (DatetimeIndex, RangeIndex, etc.).
        last_date_: Last observed timestamp if index is DatetimeIndex.
        freq_: Inferred or specified frequency string (e.g. 'MS').
        name_: Series name if input was a pd.Series.
    """

    def __init__(self) -> None:
        self.is_fitted_: bool = False
        self.y_train_: np.ndarray | pd.Series | None = None
        self.index_: pd.Index | None = None
        self.last_date_: pd.Timestamp | None = None
        self.freq_: str | None = None
        self.name_: str | None = None
        self._fittedvalues: np.ndarray | None = None
        self._resid: np.ndarray | None = None
        self.aic_: float | None = None
        self.bic_: float | None = None

    def fit(self, y: pd.Series | np.ndarray) -> BaseForecaster:
        """Fit model on training time series.

        Args:
            y: 1D time series observations.

        Returns:
            self: The fitted forecaster instance.
        """
        y_arr = self._validate_and_extract_metadata(y)
        self._fit(y_arr)
        self.is_fitted_ = True
        return self

    def predict(self, steps: int = 12) -> np.ndarray | pd.Series:
        """Generate out-of-sample forecast for a specified horizon.

        Args:
            steps: Number of future periods to forecast (horizon H >= 1).

        Returns:
            1D array or pd.Series (with future DatetimeIndex) of predicted values.
        """
        if not self.is_fitted_:
            raise ValueError(f"{self.__class__.__name__} must be fitted before generating predictions.")
        if steps < 1:
            raise ValueError(f"Forecast steps must be >= 1, got {steps}.")

        raw_forecast = self._predict(steps)
        return self._format_forecast(raw_forecast, steps)

    def forecast(self, steps: int = 12) -> np.ndarray | pd.Series:
        """Alias for predict(steps)."""
        return self.predict(steps=steps)

    @abc.abstractmethod
    def _fit(self, y_arr: np.ndarray) -> None:
        """Subclass-specific fitting logic."""
        raise NotImplementedError

    @abc.abstractmethod
    def _predict(self, steps: int) -> np.ndarray:
        """Subclass-specific prediction logic returning raw 1D numpy array."""
        raise NotImplementedError

    @property
    def fittedvalues_(self) -> np.ndarray:
        """In-sample fitted values."""
        if not self.is_fitted_ or self._fittedvalues is None:
            raise ValueError(f"{self.__class__.__name__} has not produced in-sample fitted values.")
        return self._fittedvalues

    @property
    def resid_(self) -> np.ndarray:
        """In-sample residuals (observed - fitted)."""
        if not self.is_fitted_ or self._resid is None:
            raise ValueError(f"{self.__class__.__name__} has not produced in-sample residuals.")
        return self._resid

    def score(
        self,
        y_true: pd.Series | np.ndarray,
        metric: Literal["MAE", "RMSE", "MAPE", "sMAPE", "MASE"] = "MAE",
        y_pred: pd.Series | np.ndarray | None = None,
    ) -> float:
        """Evaluate out-of-sample forecast accuracy against true values."""
        if y_pred is None:
            y_pred = self.predict(steps=len(y_true))

        true_arr = np.asarray(y_true, dtype=float).ravel()
        pred_arr = np.asarray(y_pred, dtype=float).ravel()
        train_arr = np.asarray(self.y_train_, dtype=float).ravel() if self.y_train_ is not None else None

        metrics = calculate_metrics(true_arr, pred_arr, y_train=train_arr)
        if metric not in metrics:
            raise ValueError(f"Unknown metric '{metric}'. Available: {list(metrics.keys())}")
        return float(metrics[metric])

    def _validate_and_extract_metadata(self, y: pd.Series | np.ndarray) -> np.ndarray:
        """Validate input series, save metadata, and return clean 1D numpy array."""
        if isinstance(y, (pd.DataFrame,)):
            if y.shape[1] == 1:
                y = y.iloc[:, 0]
            else:
                raise ValueError("Forecaster expects a 1D Series or array, got multi-column DataFrame.")

        if isinstance(y, pd.Series):
            self.name_ = str(y.name) if y.name is not None else "forecast"
            self.index_ = y.index
            self.y_train_ = y.copy()
            if isinstance(y.index, pd.DatetimeIndex):
                self.last_date_ = y.index[-1]
                self.freq_ = y.index.freqstr or pd.infer_freq(y.index) or "MS"
            y_arr = y.to_numpy(dtype=float)
        else:
            y_arr = np.asarray(y, dtype=float).ravel()
            self.name_ = "forecast"
            self.index_ = None
            self.last_date_ = None
            self.freq_ = None
            self.y_train_ = y_arr.copy()

        if len(y_arr) == 0:
            raise ValueError("Training series cannot be empty.")
        if np.isnan(y_arr).all():
            raise ValueError("Training series contains only NaNs.")

        return y_arr

    def _format_forecast(self, raw_forecast: np.ndarray, steps: int) -> np.ndarray | pd.Series:
        """Format raw forecast array with proper pandas DatetimeIndex if available."""
        arr = np.asarray(raw_forecast, dtype=float).ravel()

        if self.last_date_ is not None and self.freq_ is not None:
            future_dates = pd.date_range(
                start=self.last_date_,
                periods=steps + 1,
                freq=self.freq_,
            )[1:]
            return pd.Series(arr, index=future_dates, name=self.name_)

        return arr


class TransformedForecaster(BaseForecaster):
    """Meta-forecaster applying a preprocessing transformation prior to modeling.

    Ensures strict leakage-free estimation:
      - Transformer fits strictly on training series y during fit().
      - Forecaster fits on transformer.transform(y).
      - Forecasts generated in transformed space are inverted back to original level.

    Attributes:
        forecaster: Inner BaseForecaster model instance.
        transformer: Transformer instance supporting fit, transform, inverse_transform.
    """

    def __init__(self, forecaster: BaseForecaster, transformer: Any) -> None:
        super().__init__()
        self.forecaster = forecaster
        self.transformer = transformer

    def _fit(self, y_arr: np.ndarray) -> None:
        """Fit transformer and inner forecaster."""
        y_trans = self.transformer.fit_transform(y_arr)
        # Pass Series if original was Series to retain DatetimeIndex metadata in inner forecaster
        if isinstance(self.y_train_, pd.Series):
            trans_series = pd.Series(y_trans, index=self.index_, name=self.name_)
            self.forecaster.fit(trans_series)
        else:
            self.forecaster.fit(y_trans)

        # Invert fitted values if possible
        try:
            self._fittedvalues = self.transformer.inverse_transform(self.forecaster.fittedvalues_)
            self._resid = y_arr - self._fittedvalues
        except Exception:
            self._fittedvalues = None
            self._resid = None

        self.aic_ = getattr(self.forecaster, "aic_", None)
        self.bic_ = getattr(self.forecaster, "bic_", None)

    def _predict(self, steps: int) -> np.ndarray:
        """Generate forecast using inner forecaster and invert back to original scale."""
        trans_forecast = self.forecaster.predict(steps)
        if isinstance(trans_forecast, pd.Series):
            arr = trans_forecast.to_numpy()
        else:
            arr = np.asarray(trans_forecast, dtype=float)

        inverted = self.transformer.inverse_transform(arr)
        return np.asarray(inverted, dtype=float)
