"""Feature engineering, transformations, scalers, and time-series preprocessing."""

from steam_player_forecasting.features.diagnostics import (
    compute_acf_pacf,
    compute_rolling_stats,
    decompose_time_series,
    test_stationarity,
)
from steam_player_forecasting.features.scalers import ScalerType, TimeSeriesScaler
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

__all__ = [
    # Splitting
    "TimeSeriesSplitResult",
    "train_val_test_split",
    "split_by_game",
    "expanding_window_cv",
    # Scalers
    "TimeSeriesScaler",
    "ScalerType",
    # Transforms
    "LogTransformer",
    "BoxCoxTransformer",
    "DifferencingTransformer",
    "difference_series",
    "invert_difference_series",
    # Diagnostics & Decomposition
    "test_stationarity",
    "compute_acf_pacf",
    "decompose_time_series",
    "compute_rolling_stats",
]
