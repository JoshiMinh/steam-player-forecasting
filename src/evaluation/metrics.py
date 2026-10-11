"""Time-series evaluation metrics for forecasting models."""

import numpy as np


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Error (MAE)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Root Mean Squared Error (RMSE)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mean_absolute_percentage_error(
    y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8
) -> float:
    """Calculate Mean Absolute Percentage Error (MAPE) as a percentage [0, 100]."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denominator = np.where(np.abs(y_true) < eps, eps, np.abs(y_true))
    return float(np.mean(np.abs((y_true - y_pred) / denominator)) * 100.0)


def symmetric_mean_absolute_percentage_error(
    y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8
) -> float:
    """Calculate Symmetric Mean Absolute Percentage Error (sMAPE) in percentage [0, 100]."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denominator = np.where(denominator < eps, eps, denominator)
    return float(np.mean(np.abs(y_true - y_pred) / denominator) * 100.0)


def mean_absolute_scaled_error(
    y_train: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    seasonal_period: int = 1,
    eps: float = 1e-8,
) -> float:
    """Calculate Mean Absolute Scaled Error (MASE).

    Scales MAE by the in-sample naive 1-step or seasonal naive forecast error.
    """
    y_train = np.asarray(y_train, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_train) <= seasonal_period:
        scale = 1.0
    else:
        scale = np.mean(np.abs(y_train[seasonal_period:] - y_train[:-seasonal_period]))
        if scale < eps:
            scale = eps

    mae = mean_absolute_error(y_true, y_pred)
    return float(mae / scale)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray | None = None,
    seasonal_period: int = 12,
) -> dict[str, float]:
    """Calculate all standard forecasting metrics."""
    metrics = {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred),
        "sMAPE": symmetric_mean_absolute_percentage_error(y_true, y_pred),
    }
    if y_train is not None:
        metrics["MASE"] = mean_absolute_scaled_error(
            y_train, y_true, y_pred, seasonal_period=seasonal_period
        )
    return metrics
