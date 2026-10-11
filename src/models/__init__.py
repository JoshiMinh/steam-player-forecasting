"""Statistical, machine learning, and deep learning forecasting models."""

from src.models.base import BaseForecaster, TransformedForecaster
from src.models.deep_learning import (
    BaseRecurrentForecaster,
    GRUForecaster,
    GRUModel,
    LSTMForecaster,
    LSTMModel,
    RNNForecaster,
    RNNModel,
    TimeSeriesSequenceDataset,
    set_seed,
)
from src.models.diagnostics import (
    evaluate_residuals,
    plot_residual_diagnostics,
)
from src.models.selection import (
    grid_search_sarima,
    select_sarima_order,
)
from src.models.statistical import (
    ARForecaster,
    ARIMAForecaster,
    HoltWintersForecaster,
    NaiveForecaster,
    SARIMAForecaster,
    SeasonalNaiveForecaster,
)
from src.models.tabular import (
    RandomForestForecaster,
    RidgeForecaster,
    TabularForecaster,
    XGBoostForecaster,
    tune_tabular_forecaster,
)

__all__ = [
    # Base
    "BaseForecaster",
    "TransformedForecaster",
    # Statistical
    "NaiveForecaster",
    "SeasonalNaiveForecaster",
    "HoltWintersForecaster",
    "ARForecaster",
    "ARIMAForecaster",
    "SARIMAForecaster",
    # Tabular ML
    "TabularForecaster",
    "RidgeForecaster",
    "RandomForestForecaster",
    "XGBoostForecaster",
    "tune_tabular_forecaster",
    # Deep Learning (PyTorch)
    "TimeSeriesSequenceDataset",
    "RNNModel",
    "LSTMModel",
    "GRUModel",
    "BaseRecurrentForecaster",
    "RNNForecaster",
    "LSTMForecaster",
    "GRUForecaster",
    "set_seed",
    # Diagnostics & Selection
    "evaluate_residuals",
    "plot_residual_diagnostics",
    "select_sarima_order",
    "grid_search_sarima",
]
