"""Statistical, machine learning, and deep learning forecasting models."""

from steam_player_forecasting.models.base import BaseForecaster, TransformedForecaster
from steam_player_forecasting.models.deep_learning import (
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
from steam_player_forecasting.models.diagnostics import (
    evaluate_residuals,
    plot_residual_diagnostics,
)
from steam_player_forecasting.models.selection import (
    grid_search_sarima,
    select_sarima_order,
)
from steam_player_forecasting.models.statistical import (
    ARForecaster,
    ARIMAForecaster,
    HoltWintersForecaster,
    NaiveForecaster,
    SARIMAForecaster,
    SeasonalNaiveForecaster,
)
from steam_player_forecasting.models.tabular import (
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
