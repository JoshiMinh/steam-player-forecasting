"""Tabular machine learning forecasting models (Ridge, Random Forest, XGBoost).

Wraps tabular scikit-learn and XGBoost regression algorithms with autoregressive
feature generation and recursive multi-step forecasting compliant with the
BaseForecaster interface. Ensures zero lookahead leakage during feature matrix
construction and rollout.
"""

from __future__ import annotations

import copy
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

from steam_player_forecasting.features.tabular import TabularFeatureExtractor
from steam_player_forecasting.models.base import BaseForecaster


class TabularForecaster(BaseForecaster):
    """General tabular machine learning forecaster wrapper.

    Converts univariate time series into an autoregressive feature matrix
    (lags, rolling window statistics, calendar indicators), trains a tabular
    regression model, and performs recursive multi-step forecasting out-of-sample.

    Attributes:
        estimator: Underlying scikit-learn / XGBoost regressor instance.
        lags: Autoregressive lag orders.
        windows: Rolling window lengths.
        metrics: Rolling window summary metrics.
        include_calendar: Whether to add calendar features.
        include_cyclical: Whether to include harmonic sine/cosine month encodings.
        include_sales_events: Whether to include Steam sale event indicators.
        extractor_: Fitted TabularFeatureExtractor instance.
        feature_names_: List of feature names used during training.
    """

    def __init__(
        self,
        estimator: Any,
        lags: Sequence[int] = (1, 2, 3, 6, 12),
        windows: Sequence[int] = (3, 6, 12),
        metrics: Sequence[str] = ("mean", "std"),
        include_calendar: bool = True,
        include_cyclical: bool = True,
        include_sales_events: bool = True,
    ) -> None:
        super().__init__()
        self.estimator = estimator
        self.lags = list(lags)
        self.windows = list(windows)
        self.metrics = list(metrics)
        self.include_calendar = include_calendar
        self.include_cyclical = include_cyclical
        self.include_sales_events = include_sales_events

        self.extractor_: TabularFeatureExtractor | None = None
        self.feature_names_: list[str] = []
        self._history_buffer: list[float] = []

    def _fit(self, y_arr: np.ndarray) -> None:
        """Construct feature matrix and train tabular estimator."""
        dates = self.index_ if isinstance(self.index_, pd.DatetimeIndex) else None

        self.extractor_ = TabularFeatureExtractor(
            lags=self.lags,
            windows=self.windows,
            metrics=self.metrics,
            include_calendar=self.include_calendar,
            include_cyclical=self.include_cyclical,
            include_sales_events=self.include_sales_events,
        )

        X_train, y_train_aligned = self.extractor_.fit_transform(y_arr, dates=dates)
        self.feature_names_ = self.extractor_.feature_names_

        if len(X_train) == 0:
            raise ValueError(
                f"Training series length ({len(y_arr)}) is insufficient after lag warmup."
            )

        # Clone / fit estimator
        self.estimator = copy.deepcopy(self.estimator)
        self.estimator.fit(X_train, y_train_aligned)

        # Store training history for recursive forecasting
        self._history_buffer = y_arr.tolist()

        # In-sample fitted values and residuals
        in_sample_preds = self.estimator.predict(X_train)
        fitted = np.full(len(y_arr), np.nan, dtype=float)
        # Determine starting index of non-dropped rows
        warmup_len = len(y_arr) - len(X_train)
        fitted[warmup_len:] = in_sample_preds

        self._fittedvalues = fitted
        self._resid = y_arr - fitted

    def _predict(self, steps: int) -> np.ndarray:
        """Generate out-of-sample forecast via recursive autoregressive rollout."""
        if self.extractor_ is None or not self._history_buffer:
            raise ValueError("Forecaster must be fitted before predict.")

        history = list(self._history_buffer)
        predictions: list[float] = []

        # Generate future dates if DatetimeIndex is available
        future_dates: list[pd.Timestamp | None] = []
        if self.last_date_ is not None and self.freq_ is not None:
            future_dates = list(
                pd.date_range(start=self.last_date_, periods=steps + 1, freq=self.freq_)[1:]
            )
        else:
            future_dates = [None] * steps

        for h in range(steps):
            step_date = future_dates[h]
            # Construct 1-row feature vector from history and step_date
            X_step = self.extractor_.extract_step_features(history, step_date=step_date)
            pred = float(self.estimator.predict(X_step)[0])
            predictions.append(pred)
            history.append(pred)

        return np.asarray(predictions, dtype=float)


class RidgeForecaster(TabularForecaster):
    """Ridge (L2-regularized linear regression) time-series forecaster.

    Attributes:
        alpha: Regularization strength (default 1.0).
        random_state: Random state seed.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        random_state: int = 42,
        lags: Sequence[int] = (1, 2, 3, 6, 12),
        windows: Sequence[int] = (3, 6, 12),
        metrics: Sequence[str] = ("mean", "std"),
        include_calendar: bool = True,
        include_cyclical: bool = True,
        include_sales_events: bool = True,
    ) -> None:
        self.alpha = alpha
        self.random_state = random_state
        estimator = Ridge(alpha=alpha, random_state=random_state)
        super().__init__(
            estimator=estimator,
            lags=lags,
            windows=windows,
            metrics=metrics,
            include_calendar=include_calendar,
            include_cyclical=include_cyclical,
            include_sales_events=include_sales_events,
        )


class RandomForestForecaster(TabularForecaster):
    """Random Forest regressor time-series forecaster.

    Attributes:
        n_estimators: Number of trees in the forest (default 100).
        max_depth: Maximum depth of each tree (default 10).
        min_samples_split: Minimum number of samples required to split a node (default 2).
        random_state: Random state seed.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = 10,
        min_samples_split: int = 2,
        random_state: int = 42,
        lags: Sequence[int] = (1, 2, 3, 6, 12),
        windows: Sequence[int] = (3, 6, 12),
        metrics: Sequence[str] = ("mean", "std"),
        include_calendar: bool = True,
        include_cyclical: bool = True,
        include_sales_events: bool = True,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        estimator = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            n_jobs=-1,
        )
        super().__init__(
            estimator=estimator,
            lags=lags,
            windows=windows,
            metrics=metrics,
            include_calendar=include_calendar,
            include_cyclical=include_cyclical,
            include_sales_events=include_sales_events,
        )


class XGBoostForecaster(TabularForecaster):
    """Extreme Gradient Boosting (XGBoost) time-series forecaster.

    Attributes:
        n_estimators: Number of gradient boosted trees (default 100).
        max_depth: Maximum tree depth for base learners (default 4).
        learning_rate: Boosting learning rate (default 0.05).
        subsample: Subsample ratio of the training instances (default 0.8).
        colsample_bytree: Subsample ratio of columns when constructing each tree (default 0.8).
        random_state: Random state seed.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        lags: Sequence[int] = (1, 2, 3, 6, 12),
        windows: Sequence[int] = (3, 6, 12),
        metrics: Sequence[str] = ("mean", "std"),
        include_calendar: bool = True,
        include_cyclical: bool = True,
        include_sales_events: bool = True,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        estimator = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            verbosity=0,
            n_jobs=-1,
        )
        super().__init__(
            estimator=estimator,
            lags=lags,
            windows=windows,
            metrics=metrics,
            include_calendar=include_calendar,
            include_cyclical=include_cyclical,
            include_sales_events=include_sales_events,
        )


def tune_tabular_forecaster(
    model_type: Literal["ridge", "random_forest", "xgboost"],
    y_train: pd.Series | np.ndarray,
    y_val: pd.Series | np.ndarray,
    param_grid: list[dict[str, Any]] | None = None,
    criterion: Literal["MAE", "RMSE", "MAPE", "sMAPE", "MASE"] = "MAE",
    seed: int = 42,
) -> tuple[dict[str, Any], pd.DataFrame, TabularForecaster]:
    """Tune tabular forecaster hyperparameters on chronological validation window.

    Evaluates candidate hyperparameters by fitting strictly on y_train and evaluating
    multi-step forecast accuracy on y_val. Zero lookahead leakage guaranteed.

    Args:
        model_type: 'ridge', 'random_forest', or 'xgboost'.
        y_train: Training split series.
        y_val: Validation split series.
        param_grid: List of parameter dictionaries. If None, default grid is used.
        criterion: Target evaluation metric to minimize.
        seed: Random seed for reproducibility.

    Returns:
        tuple (best_params, df_results, best_model)
    """
    if param_grid is None:
        if model_type == "ridge":
            param_grid = [
                {"alpha": 0.01},
                {"alpha": 0.1},
                {"alpha": 1.0},
                {"alpha": 10.0},
                {"alpha": 100.0},
            ]
        elif model_type == "random_forest":
            param_grid = [
                {"n_estimators": 50, "max_depth": 3},
                {"n_estimators": 100, "max_depth": 4},
                {"n_estimators": 100, "max_depth": 6},
                {"n_estimators": 150, "max_depth": 8},
                {"n_estimators": 200, "max_depth": 10},
            ]
        elif model_type == "xgboost":
            param_grid = [
                {"n_estimators": 50, "max_depth": 2, "learning_rate": 0.05},
                {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.05},
                {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.03},
                {"n_estimators": 150, "max_depth": 3, "learning_rate": 0.02},
                {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05},
            ]
        else:
            raise ValueError(f"Unknown model_type '{model_type}'.")

    results: list[dict[str, Any]] = []
    best_score = float("inf")
    best_params: dict[str, Any] = {}
    best_model: TabularForecaster | None = None

    steps = len(y_val)
    val_true = np.asarray(y_val, dtype=float).ravel()
    train_arr = np.asarray(y_train, dtype=float).ravel()

    for p in param_grid:
        params_copy = copy.deepcopy(p)
        params_copy["random_state"] = seed

        if model_type == "ridge":
            model = RidgeForecaster(**params_copy)
        elif model_type == "random_forest":
            model = RandomForestForecaster(**params_copy)
        else:
            model = XGBoostForecaster(**params_copy)

        try:
            model.fit(y_train)
            pred = model.predict(steps=steps)
            pred_arr = np.asarray(pred, dtype=float).ravel()
            score = model.score(val_true, metric=criterion, y_pred=pred_arr)

            # Also calculate other metrics for recording
            mae = model.score(val_true, metric="MAE", y_pred=pred_arr)
            rmse = model.score(val_true, metric="RMSE", y_pred=pred_arr)
            mape = model.score(val_true, metric="MAPE", y_pred=pred_arr)
            smape = model.score(val_true, metric="sMAPE", y_pred=pred_arr)

            record = {
                "params": str(p),
                "val_mae": mae,
                "val_rmse": rmse,
                "val_mape": mape,
                "val_smape": smape,
                "score": score,
            }
            results.append(record)

            if score < best_score:
                best_score = score
                best_params = p
                best_model = model

        except Exception as e:
            record = {
                "params": str(p),
                "val_mae": np.nan,
                "val_rmse": np.nan,
                "val_mape": np.nan,
                "val_smape": np.nan,
                "score": np.nan,
                "error": str(e),
            }
            results.append(record)

    df_results = pd.DataFrame(results).sort_values("score", ascending=True).reset_index(drop=True)

    if best_model is None:
        raise RuntimeError(f"All parameter configurations failed during tuning for {model_type}.")

    return best_params, df_results, best_model
