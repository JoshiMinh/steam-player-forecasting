"""Unit and integration tests for interactive UI components and dynamic metrics engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.components.cards import compute_dynamic_metrics
from app.components.plots import (
    create_horizon_error_chart,
    create_interactive_forecast_chart,
)


@pytest.fixture
def mock_game_cache() -> dict:
    """Fixture providing synthetic forecasting cache for testing."""
    history_dates = [f"2018-{m:02d}-01" for m in range(1, 13)] + [f"2019-{m:02d}-01" for m in range(1, 13)]
    history_values = [100000.0 + i * 1000.0 for i in range(24)]
    test_dates = [f"2020-{m:02d}-01" for m in range(9, 13)] + [f"2021-{m:02d}-01" for m in range(1, 9)]
    test_actuals = [120000.0 + (i % 3) * 5000.0 for i in range(12)]
    test_forecasts = {
        "SARIMA": [121000.0 + (i % 3) * 4800.0 for i in range(12)],
        "XGBoost": [118000.0 + i * 2000.0 for i in range(12)],
        "LSTM": [120500.0 + (i % 3) * 5100.0 for i in range(12)],
        "GRU": [120200.0 + (i % 3) * 4950.0 for i in range(12)],
    }
    return {
        "history_dates": history_dates,
        "history_values": history_values,
        "test_dates": test_dates,
        "test_actuals": test_actuals,
        "test_forecasts": test_forecasts,
        "sarima_ci_lower": [110000.0 for _ in range(12)],
        "sarima_ci_upper": [130000.0 for _ in range(12)],
        "optimal_sarima_order": "(1,1,1)x(1,1,1)12",
        "optimal_xgb_params": {"max_depth": 3, "learning_rate": 0.05},
    }


def test_compute_dynamic_metrics_full_horizon(mock_game_cache: dict) -> None:
    """Test dynamic metric calculation on full 12-month test horizon."""
    actuals = mock_game_cache["test_actuals"]
    forecasts = mock_game_cache["test_forecasts"]
    scale = 5000.0

    df = compute_dynamic_metrics(
        actuals=actuals,
        forecasts_dict=forecasts,
        seasonal_scale=scale,
        horizon=12,
        paradigm_map={"SARIMA": "Econometric", "XGBoost": "ML", "LSTM": "DL", "GRU": "DL"},
    )

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4
    assert set(df.columns) == {"Model", "Paradigm", "MAE", "RMSE", "MAPE", "sMAPE", "MASE"}

    for _, row in df.iterrows():
        assert row["MAE"] > 0
        assert row["RMSE"] >= row["MAE"]
        assert row["MAPE"] > 0
        assert row["sMAPE"] > 0
        assert row["MASE"] > 0
        assert row["Paradigm"] in ["Econometric", "ML", "DL"]


def test_compute_dynamic_metrics_truncated_horizon(mock_game_cache: dict) -> None:
    """Test dynamic metric calculation on truncated horizon (H=3 vs H=12)."""
    actuals = mock_game_cache["test_actuals"]
    forecasts = mock_game_cache["test_forecasts"]
    scale = 5000.0

    df_h3 = compute_dynamic_metrics(actuals, forecasts, scale, horizon=3)
    df_h12 = compute_dynamic_metrics(actuals, forecasts, scale, horizon=12)

    assert len(df_h3) == 4
    # Truncated metrics should be computed only over h=1..3
    sarima_h3 = df_h3[df_h3["Model"] == "SARIMA"].iloc[0]
    expected_mae_h3 = np.mean(
        np.abs(np.array(actuals[:3]) - np.array(forecasts["SARIMA"][:3]))
    )
    assert np.isclose(sarima_h3["MAE"], expected_mae_h3)
    assert np.isclose(sarima_h3["MASE"], expected_mae_h3 / scale)


def test_compute_dynamic_metrics_zero_scale(mock_game_cache: dict) -> None:
    """Verify zero division safety when seasonal scale is zero or non-positive."""
    df = compute_dynamic_metrics(
        actuals=mock_game_cache["test_actuals"],
        forecasts_dict=mock_game_cache["test_forecasts"],
        seasonal_scale=0.0,
        horizon=6,
    )
    assert (df["MASE"] == 0.0).all()


def test_create_interactive_forecast_chart_spec(mock_game_cache: dict) -> None:
    """Verify that Altair interactive forecast chart compiles valid Vega-Lite spec."""
    chart = create_interactive_forecast_chart(
        game_cache=mock_game_cache,
        selected_models=["SARIMA", "GRU"],
        horizon=12,
        history_window=12,
        show_ci=True,
    )
    spec = chart.to_dict()

    assert "layer" in spec
    assert len(spec["layer"]) >= 2  # At least boundary line + multi-series line chart
    assert spec["height"] == 440


def test_create_interactive_forecast_chart_with_shock(mock_game_cache: dict) -> None:
    """Verify that simulated demand shock layer compiles cleanly."""
    chart = create_interactive_forecast_chart(
        game_cache=mock_game_cache,
        selected_models=["SARIMA", "GRU"],
        horizon=12,
        history_window=12,
        show_ci=False,
        demand_shock_pct=25.0,
        shock_target_model="SARIMA",
    )
    spec = chart.to_dict()
    assert "layer" in spec


def test_create_horizon_error_chart_spec(mock_game_cache: dict) -> None:
    """Verify that Altair horizon error compounding chart compiles valid spec."""
    chart = create_horizon_error_chart(
        game_cache=mock_game_cache,
        selected_models=["SARIMA", "XGBoost", "LSTM"],
        horizon=12,
    )
    spec = chart.to_dict()

    assert "encoding" in spec
    assert spec["encoding"]["x"]["field"] == "Step"
    assert spec["encoding"]["y"]["field"] == "MAPE"
    assert spec["encoding"]["color"]["field"] == "Model"
    assert spec["height"] == 320


def test_streamlit_app_test_execution() -> None:
    """Verify that Streamlit headless AppTest completes without unhandled exceptions."""
    from pathlib import Path
    from streamlit.testing.v1 import AppTest

    app_path = Path(__file__).resolve().parent.parent / "app" / "app.py"
    at = AppTest.from_file(str(app_path)).run(timeout=15)
    assert len(at.exception) == 0, f"AppTest raised exceptions: {at.exception}"
    assert len(at.tabs) == 4
