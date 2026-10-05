"""Statistical and econometric forecasting baseline models for monthly time series."""

from __future__ import annotations

import warnings
from typing import Any, Literal

import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from steam_player_forecasting.models.base import BaseForecaster
from steam_player_forecasting.models.diagnostics import evaluate_residuals


class NaiveForecaster(BaseForecaster):
    """Naive benchmark forecaster projecting the last observed value (or historical mean).

    In the 'last_value' strategy, for all forecast horizons h >= 1:
      y_{T+h} = y_T

    Attributes:
        strategy: 'last_value' (standard random-walk benchmark) or 'mean'.
        last_value_: Value of the last observed training point.
        mean_value_: Arithmetic mean of the training series.
    """

    def __init__(self, strategy: Literal["last_value", "mean"] = "last_value") -> None:
        super().__init__()
        if strategy not in ("last_value", "mean"):
            raise ValueError(f"Invalid strategy '{strategy}'. Choose 'last_value' or 'mean'.")
        self.strategy = strategy
        self.last_value_: float | None = None
        self.mean_value_: float | None = None

    def _fit(self, y_arr: np.ndarray) -> None:
        """Fit naive forecaster on training series."""
        self.last_value_ = float(y_arr[-1])
        self.mean_value_ = float(np.mean(y_arr))

        if self.strategy == "last_value":
            self._fittedvalues = np.concatenate([[y_arr[0]], y_arr[:-1]])
        else:
            self._fittedvalues = np.full_like(y_arr, self.mean_value_)

        self._resid = y_arr - self._fittedvalues

    def _predict(self, steps: int) -> np.ndarray:
        """Generate flat naive forecast."""
        val = self.last_value_ if self.strategy == "last_value" else self.mean_value_
        return np.full(steps, val, dtype=float)


class SeasonalNaiveForecaster(BaseForecaster):
    """Seasonal Naive forecaster repeating the identical calendar month from the prior cycle.

    For horizon h in 1..steps:
      y_{T+h} = y_{T + h - m * s}  where m is chosen such that the reference index is <= T.

    Attributes:
        seasonal_period: Seasonal cycle length s (default 12 for monthly data).
        history_buffer_: The last s observations from the training window.
    """

    def __init__(self, seasonal_period: int = 12) -> None:
        super().__init__()
        if seasonal_period < 2:
            raise ValueError(f"seasonal_period must be >= 2, got {seasonal_period}.")
        self.seasonal_period = seasonal_period
        self.history_buffer_: np.ndarray | None = None

    def _fit(self, y_arr: np.ndarray) -> None:
        """Store the seasonal boundary buffer from training data."""
        if len(y_arr) < self.seasonal_period:
            raise ValueError(
                f"Training series length ({len(y_arr)}) must be >= seasonal_period ({self.seasonal_period})."
            )

        self.history_buffer_ = y_arr[-self.seasonal_period:].copy()

        # In-sample 1-step seasonal naive fitted values
        fitted = np.full_like(y_arr, np.nan)
        fitted[self.seasonal_period:] = y_arr[:-self.seasonal_period]
        self._fittedvalues = fitted
        self._resid = y_arr - fitted

    def _predict(self, steps: int) -> np.ndarray:
        """Project seasonal naive forecast periodically."""
        if self.history_buffer_ is None:
            raise ValueError("SeasonalNaiveForecaster is missing training history buffer.")

        forecast = np.empty(steps, dtype=float)
        for h in range(steps):
            forecast[h] = self.history_buffer_[h % self.seasonal_period]
        return forecast


class HoltWintersForecaster(BaseForecaster):
    """Holt-Winters Exponential Smoothing forecaster with trend and seasonal components.

    Wraps `statsmodels.tsa.holtwinters.ExponentialSmoothing`.

    Supports:
      - Additive or multiplicative trend (with optional damping).
      - Additive or multiplicative seasonal components (s=12).
    """

    def __init__(
        self,
        seasonal_periods: int = 12,
        trend: Literal["add", "mul"] | None = "add",
        seasonal: Literal["add", "mul"] | None = "add",
        damped_trend: bool = False,
        initialization_method: str = "estimated",
        use_boxcox: bool | str | float = False,
    ) -> None:
        super().__init__()
        self.seasonal_periods = seasonal_periods
        self.trend = trend
        self.seasonal = seasonal
        self.damped_trend = damped_trend
        self.initialization_method = initialization_method
        self.use_boxcox = use_boxcox

        self.model_: ExponentialSmoothing | None = None
        self.results_: Any = None

    def _fit(self, y_arr: np.ndarray) -> None:
        """Fit Holt-Winters exponential smoothing model."""
        min_required = 2 * self.seasonal_periods if self.seasonal else 4
        if len(y_arr) < min_required:
            raise ValueError(
                f"Need at least {min_required} observations to fit Holt-Winters (got {len(y_arr)})."
            )

        # Multiplicative components require strictly positive values
        if (self.trend == "mul" or self.seasonal == "mul") and np.any(y_arr <= 0):
            raise ValueError("Multiplicative trend/seasonality requires strictly positive series values.")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model_ = ExponentialSmoothing(
                y_arr,
                seasonal_periods=self.seasonal_periods if self.seasonal else None,
                trend=self.trend,
                seasonal=self.seasonal,
                damped_trend=self.damped_trend,
                initialization_method=self.initialization_method,
                use_boxcox=self.use_boxcox,
            )
            self.results_ = self.model_.fit()

        self._fittedvalues = np.asarray(self.results_.fittedvalues, dtype=float)
        self._resid = np.asarray(self.results_.resid, dtype=float)
        self.aic_ = float(self.results_.aic) if hasattr(self.results_, "aic") and not np.isnan(self.results_.aic) else None
        self.bic_ = float(self.results_.bic) if hasattr(self.results_, "bic") and not np.isnan(self.results_.bic) else None

    def _predict(self, steps: int) -> np.ndarray:
        """Forecast future steps using fitted Holt-Winters model."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fc = self.results_.forecast(steps)
        return np.asarray(fc, dtype=float)


class ARForecaster(BaseForecaster):
    """Pure Autoregressive (AR) forecaster.

    Wraps `statsmodels.tsa.ar_model.AutoReg`.

    Model:
      y_t = c + phi_1 * y_{t-1} + ... + phi_p * y_{t-p} + epsilon_t
    """

    def __init__(
        self,
        lags: int | list[int] = 1,
        trend: Literal["n", "c", "t", "ct"] = "c",
        seasonal: bool = False,
        period: int = 12,
    ) -> None:
        super().__init__()
        self.lags = lags
        self.trend = trend
        self.seasonal = seasonal
        self.period = period

        self.model_: AutoReg | None = None
        self.results_: Any = None

    def _fit(self, y_arr: np.ndarray) -> None:
        """Fit AutoReg model."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model_ = AutoReg(
                y_arr,
                lags=self.lags,
                trend=self.trend,
                seasonal=self.seasonal,
                period=self.period if self.seasonal else None,
            )
            self.results_ = self.model_.fit()

        self._fittedvalues = np.asarray(self.results_.fittedvalues, dtype=float)
        self._resid = np.asarray(self.results_.resid, dtype=float)
        self.aic_ = float(self.results_.aic) if hasattr(self.results_, "aic") else None
        self.bic_ = float(self.results_.bic) if hasattr(self.results_, "bic") else None

    def _predict(self, steps: int) -> np.ndarray:
        """Forecast future steps using AutoReg model."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fc = self.results_.forecast(steps)
        return np.asarray(fc, dtype=float)


class ARIMAForecaster(BaseForecaster):
    """Autoregressive Integrated Moving Average (ARIMA) forecaster.

    Wraps `statsmodels.tsa.arima.model.ARIMA`.

    Specification: ARIMA(p, d, q).
    """

    def __init__(
        self,
        order: tuple[int, int, int] = (1, 1, 1),
        trend: str | None = None,
        enforce_stationarity: bool = False,
        enforce_invertibility: bool = False,
    ) -> None:
        super().__init__()
        self.order = tuple(order)
        self.trend = trend
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility

        self.model_: ARIMA | None = None
        self.results_: Any = None

    def _fit(self, y_arr: np.ndarray) -> None:
        """Fit ARIMA model."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model_ = ARIMA(
                y_arr,
                order=self.order,
                trend=self.trend,
                enforce_stationarity=self.enforce_stationarity,
                enforce_invertibility=self.enforce_invertibility,
            )
            self.results_ = self.model_.fit()

        self._fittedvalues = np.asarray(self.results_.fittedvalues, dtype=float)
        self._resid = np.asarray(self.results_.resid, dtype=float)
        self.aic_ = float(self.results_.aic)
        self.bic_ = float(self.results_.bic)

    def _predict(self, steps: int) -> np.ndarray:
        """Forecast future steps using ARIMA model."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fc = self.results_.forecast(steps)
        return np.asarray(fc, dtype=float)

    def predict_conf_int(
        self, steps: int = 12, alpha: float = 0.05
    ) -> tuple[np.ndarray | pd.Series, pd.DataFrame]:
        """Generate point forecast alongside confidence intervals."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before predicting.")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            get_fc = self.results_.get_forecast(steps)
            point_fc = self._format_forecast(np.asarray(get_fc.predicted_mean), steps)
            ci_vals = np.asarray(get_fc.conf_int(alpha=alpha))

        ci_out = pd.DataFrame(
            {"lower": ci_vals[:, 0], "upper": ci_vals[:, 1]},
            index=point_fc.index if isinstance(point_fc, pd.Series) else None,
        )
        return point_fc, ci_out


class SARIMAForecaster(BaseForecaster):
    """Seasonal Autoregressive Integrated Moving Average (SARIMA) forecaster.

    Wraps `statsmodels.tsa.statespace.sarimax.SARIMAX`.

    Specification: SARIMA(p, d, q) x (P, D, Q, s).
    Designed for econometric modeling of monthly multi-year Steam dynamics.
    """

    def __init__(
        self,
        order: tuple[int, int, int] = (1, 1, 1),
        seasonal_order: tuple[int, int, int, int] = (1, 1, 1, 12),
        trend: str | None = None,
        enforce_stationarity: bool = False,
        enforce_invertibility: bool = False,
        maxiter: int = 100,
    ) -> None:
        super().__init__()
        self.order = tuple(order)
        self.seasonal_order = tuple(seasonal_order)
        self.trend = trend
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility
        self.maxiter = maxiter

        self.model_: SARIMAX | None = None
        self.results_: Any = None

    def _fit(self, y_arr: np.ndarray) -> None:
        """Fit SARIMAX model."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model_ = SARIMAX(
                y_arr,
                order=self.order,
                seasonal_order=self.seasonal_order,
                trend=self.trend,
                enforce_stationarity=self.enforce_stationarity,
                enforce_invertibility=self.enforce_invertibility,
            )
            self.results_ = self.model_.fit(disp=False, maxiter=self.maxiter)

        self._fittedvalues = np.asarray(self.results_.fittedvalues, dtype=float)
        self._resid = np.asarray(self.results_.resid, dtype=float)
        self.aic_ = float(self.results_.aic)
        self.bic_ = float(self.results_.bic)

    def _predict(self, steps: int) -> np.ndarray:
        """Forecast future steps using SARIMAX model."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fc = self.results_.forecast(steps)
        return np.asarray(fc, dtype=float)

    def predict_conf_int(
        self, steps: int = 12, alpha: float = 0.05
    ) -> tuple[np.ndarray | pd.Series, pd.DataFrame]:
        """Generate point forecast alongside confidence intervals."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before predicting.")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            get_fc = self.results_.get_forecast(steps)
            point_fc = self._format_forecast(np.asarray(get_fc.predicted_mean), steps)
            ci_vals = np.asarray(get_fc.conf_int(alpha=alpha))

        ci_out = pd.DataFrame(
            {"lower": ci_vals[:, 0], "upper": ci_vals[:, 1]},
            index=point_fc.index if isinstance(point_fc, pd.Series) else None,
        )
        return point_fc, ci_out

    def check_residuals(
        self, lags: int | list[int] | None = None, alpha: float = 0.05
    ) -> dict[str, Any]:
        """Perform econometric residual diagnostics (Ljung-Box white noise test)."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before running residual diagnostics.")
        lags_arg = 12 if lags is None else lags
        return evaluate_residuals(self.resid_, lags=lags_arg, alpha=alpha)

    def summary(self) -> str:
        """Return statsmodels statistical summary table."""
        if not self.is_fitted_ or self.results_ is None:
            raise ValueError("Model must be fitted before inspecting summary.")
        return str(self.results_.summary())
