"""Smoke and sanity tests for project initialization."""

import numpy as np
import pytest

import steam_player_forecasting
from steam_player_forecasting.data.loader import get_project_root, load_config
from steam_player_forecasting.evaluation.metrics import (
    calculate_metrics,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_absolute_scaled_error,
    root_mean_squared_error,
    symmetric_mean_absolute_percentage_error,
)


def test_package_metadata():
    """Verify package import and version presence."""
    assert hasattr(steam_player_forecasting, "__version__")
    assert isinstance(steam_player_forecasting.__version__, str)
    assert steam_player_forecasting.__version__ == "0.1.0"


def test_project_root_and_config():
    """Verify project root path resolver and default configuration loading."""
    root = get_project_root()
    assert root.exists()
    assert (root / "configs" / "default.yaml").exists()

    config = load_config()
    assert isinstance(config, dict)
    assert "project" in config
    assert "data" in config
    assert "split" in config
    assert "models" in config
    assert "evaluation" in config
    assert config["data"]["target_column"] == "Avg_players"
    assert config["data"]["secondary_target"] == "Peak_Players"


def test_project_directories_exist():
    """Verify expected directory tree exists."""
    root = get_project_root()
    expected_dirs = [
        "configs",
        "data/raw",
        "data/processed",
        "notebooks",
        "src/steam_player_forecasting",
        "models",
        "figures",
        "app",
        "report",
        "tests",
    ]
    for d in expected_dirs:
        dir_path = root / d
        assert dir_path.exists(), f"Expected directory {d} does not exist"


def test_metrics_calculation():
    """Verify mathematical correctness of time-series error metrics."""
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 300.0])

    mae = mean_absolute_error(y_true, y_pred)
    assert pytest.approx(mae, 0.001) == 20.0 / 3.0

    rmse = root_mean_squared_error(y_true, y_pred)
    expected_rmse = np.sqrt((100.0 + 100.0 + 0.0) / 3.0)
    assert pytest.approx(rmse, 0.001) == expected_rmse

    mape = mean_absolute_percentage_error(y_true, y_pred)
    expected_mape = ((10.0 / 100.0 + 10.0 / 200.0 + 0.0) / 3.0) * 100.0
    assert pytest.approx(mape, 0.001) == expected_mape

    smape = symmetric_mean_absolute_percentage_error(y_true, y_pred)
    assert smape >= 0.0

    # Test combined dictionary
    y_train = np.array([80.0, 90.0, 100.0, 110.0])
    metrics = calculate_metrics(y_true, y_pred, y_train=y_train, seasonal_period=1)
    assert "MAE" in metrics
    assert "RMSE" in metrics
    assert "MAPE" in metrics
    assert "sMAPE" in metrics
    assert "MASE" in metrics
