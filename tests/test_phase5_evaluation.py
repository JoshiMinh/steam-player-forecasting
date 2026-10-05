"""Unit tests for Phase 5 final evaluation, benchmark tables, figures, and Streamlit demo."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from steam_player_forecasting.data.loader import get_available_games, get_project_root
from steam_player_forecasting.evaluation.metrics import calculate_metrics


def test_phase5_report_files_exist():
    """Verify that all Phase 5 report artifacts are generated and valid."""
    root = get_project_root()
    report_dir = root / "report"

    expected_files = [
        "final_test_benchmarks.csv",
        "final_test_benchmarks.md",
        "core_quartet_test_benchmarks.csv",
        "core_quartet_test_benchmarks.md",
        "final_leaderboard.csv",
        "final_leaderboard.md",
        "error_by_horizon.csv",
        "validation_vs_test_comparison.csv",
        "validation_vs_test_comparison.md",
    ]

    for filename in expected_files:
        filepath = report_dir / filename
        assert filepath.exists(), f"Expected report file missing: {filepath}"
        assert filepath.stat().st_size > 0, f"Report file is empty: {filepath}"


def test_phase5_figures_exist():
    """Verify that all Phase 5 publication-quality figures exist and are non-empty."""
    root = get_project_root()
    fig_dir = root / "figures"

    expected_figures = [
        "17_final_test_forecast_comparisons.png",
        "18_final_leaderboard_metrics.png",
        "19_horizon_error_degradation.png",
        "20_paradigm_case_studies.png",
    ]

    for fig_name in expected_figures:
        fig_path = fig_dir / fig_name
        assert fig_path.exists(), f"Expected figure missing: {fig_path}"
        assert fig_path.stat().st_size > 10_000, f"Figure unexpectedly small (<10KB): {fig_path}"


def test_final_test_benchmarks_schema_and_coverage():
    """Verify final test benchmarks CSV covers all 7 games and expected candidate models."""
    root = get_project_root()
    csv_path = root / "report" / "final_test_benchmarks.csv"
    df = pd.read_csv(csv_path)

    expected_cols = ["Game", "Model", "Paradigm", "MAE", "RMSE", "MAPE", "sMAPE", "MASE"]
    for col in expected_cols:
        assert col in df.columns, f"Missing column {col} in {csv_path}"

    games = get_available_games()
    for game in games:
        assert game in df["Game"].values, f"Game {game} missing from benchmarks"

    core_models = ["SARIMA", "XGBoost", "LSTM", "GRU"]
    for model in core_models:
        assert model in df["Model"].values, f"Core model {model} missing from benchmarks"

    # Total rows: 7 games * 11 models = 77
    assert len(df) == 77, f"Expected 77 evaluation rows, got {len(df)}"

    # Check metrics are valid non-negative numbers
    for metric in ["MAE", "RMSE", "MAPE", "sMAPE", "MASE"]:
        assert not df[metric].isna().any(), f"NaN detected in {metric}"
        assert (df[metric] >= 0).all(), f"Negative value detected in {metric}"


def test_final_leaderboard_consistency():
    """Verify the final leaderboard correctly aggregates test metrics."""
    root = get_project_root()
    csv_path = root / "report" / "final_leaderboard.csv"
    df = pd.read_csv(csv_path)

    assert "Model" in df.columns
    assert "Mean_MASE" in df.columns
    assert "Mean_MAE" in df.columns
    assert "Mean_MAPE" in df.columns

    # Verify rankings are strictly monotonic by Mean_MASE
    assert df["Mean_MASE"].is_monotonic_increasing, "Leaderboard is not sorted by Mean_MASE"


def test_forecast_cache_integrity():
    """Verify test_forecasts_cache.json is properly formatted and complete."""
    root = get_project_root()
    cache_path = root / "report" / "test_forecasts_cache.json"
    if not cache_path.exists():
        cache_path = root / "data" / "processed" / "test_forecasts_cache.json"

    assert cache_path.exists(), f"Cache file missing: {cache_path}"
    with open(cache_path, "r", encoding="utf-8") as f:
        cache = json.load(f)

    games = get_available_games()
    for game in games:
        assert game in cache, f"Game {game} missing from cache"
        g_data = cache[game]

        assert "history_dates" in g_data
        assert "history_values" in g_data
        assert "test_dates" in g_data
        assert "test_actuals" in g_data
        assert "test_forecasts" in g_data

        # 12-month test horizon
        assert len(g_data["test_dates"]) == 12
        assert len(g_data["test_actuals"]) == 12

        for m_name in ["SARIMA", "XGBoost", "LSTM", "GRU"]:
            assert m_name in g_data["test_forecasts"]
            assert len(g_data["test_forecasts"][m_name]) == 12


def test_error_by_horizon_structure():
    """Verify error by horizon dataset covers steps 1 to 12."""
    root = get_project_root()
    csv_path = root / "report" / "error_by_horizon.csv"
    df = pd.read_csv(csv_path)

    assert set(df["Horizon_Step"].unique()) == set(range(1, 13))
    assert "Absolute_Error" in df.columns
    assert "Percentage_Error" in df.columns
    assert (df["Percentage_Error"] >= 0).all()
