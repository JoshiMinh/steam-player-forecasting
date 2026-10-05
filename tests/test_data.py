"""Unit tests for Phase 1: Dataset Acquisition & Validation."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from steam_player_forecasting.data.ingestion import (
    acquire_dataset,
    generate_synthetic_raw_data,
    has_kaggle_credentials,
)
from steam_player_forecasting.data.loader import (
    get_available_games,
    load_config,
    load_processed_data,
    load_raw_data,
)
from steam_player_forecasting.data.pipeline import run_phase1_pipeline, slugify_game_name
from steam_player_forecasting.data.selection import (
    check_game_continuity,
    get_game_coverage_summary,
    select_representative_games,
)
from steam_player_forecasting.data.validation import (
    clean_raw_data,
    inspect_raw_schema,
    standardize_columns,
    validate_cleaned_data,
)


@pytest.fixture
def sample_raw_df() -> pd.DataFrame:
    """Create a minimal raw DataFrame with typical scraping quirks and artifacts."""
    return pd.DataFrame(
        {
            "Month_Year": [
                "Last 30 Days",
                "September 2021",
                "August 2021",
                "July 2021",
                "June 2021",
                "September 2021",  # duplicate for testing
                "Invalid Date String",
            ],
            "Game_Name": [
                "Counter-Strike: Global Offensive",
                "Counter-Strike: Global Offensive",
                "Counter-Strike: Global Offensive",
                "Counter-Strike: Global Offensive",
                "Counter-Strike: Global Offensive",
                "Counter-Strike: Global Offensive",
                "Counter-Strike: Global Offensive",
            ],
            "Avg_players": [
                "512,345.50",
                "500,100.25",
                "480,000.00",
                "490,500.00",
                "470,000.00",
                "500,100.25",
                "10,000.00",
            ],
            "Gain": [
                "+12,245.25",
                "+20,100.25",
                "-10,500.00",
                "+20,500.00",
                "-",
                "+20,100.25",
                "-",
            ],
            "% Gain": [
                "+2.45%",
                "+4.19%",
                "-2.14%",
                "+4.36%",
                "-",
                "+4.19%",
                "-",
            ],
            "Peak_Players": [
                "850,000",
                "820,500",
                "790,000",
                "810,000",
                "780,000",
                "820,500",
                "15,000",
            ],
        }
    )


# ==============================================================================
# 1. Ingestion Tests
# ==============================================================================

def test_has_kaggle_credentials_returns_bool():
    """Verify credential checker executes safely and returns boolean."""
    creds = has_kaggle_credentials()
    assert isinstance(creds, bool)


def test_generate_synthetic_raw_data_structure():
    """Verify synthetic generator generates expected columns and candidate games."""
    df = generate_synthetic_raw_data(include_raw_artifacts=True, seed=42)

    assert not df.empty
    assert "Game_Name" in df.columns
    assert "Month_Year" in df.columns
    assert "Avg_players" in df.columns
    assert "Peak_Players" in df.columns

    # Verify candidate games presence
    games = df["Game_Name"].unique()
    assert "Counter-Strike: Global Offensive" in games
    assert "Dota 2" in games
    assert "Rust" in games

    # Verify presence of raw scraper artifact 'Last 30 Days'
    assert (df["Month_Year"] == "Last 30 Days").any()


def test_acquire_dataset_fallback(tmp_path: Path):
    """Verify acquire_dataset generates synthetic data when raw files are absent."""
    custom_config = {
        "project": {"seed": 42},
        "data": {
            "raw_dir": str(tmp_path / "raw"),
            "raw_filename": "test_raw.csv",
        },
    }
    raw_path = acquire_dataset(config=custom_config, create_sample_if_missing=True)
    assert raw_path.exists()
    assert raw_path.name == "test_raw.csv"

    # Subsequent call without force_download should reuse the file
    same_path = acquire_dataset(config=custom_config, force_download=False)
    assert same_path == raw_path


# ==============================================================================
# 2. Schema Inspection & Standardization Tests
# ==============================================================================

def test_inspect_raw_schema(sample_raw_df: pd.DataFrame):
    """Verify schema inspection extracts columns, nulls, and detected mapping."""
    info = inspect_raw_schema(sample_raw_df)
    assert info["total_rows"] == len(sample_raw_df)
    assert info["total_columns"] == len(sample_raw_df.columns)
    assert "Game_Name" in info["detected_mapping"]
    assert "Avg_players" in info["detected_mapping"]


def test_standardize_columns_aliases():
    """Verify column standardizer maps common scraper variations to canonical names."""
    variations = pd.DataFrame(
        {
            "gamename": ["Dota 2"],
            "month": ["August 2021"],
            "avg": ["450000.5"],
            "gain": ["+5000.2"],
            "% gain": ["+1.12%"],
            "peak": ["750000"],
        }
    )
    standardized = standardize_columns(variations)
    expected_cols = [
        "Game_Name",
        "Month_Year",
        "Avg_players",
        "Gain",
        "Percent_Gain",
        "Peak_Players",
    ]
    for c in expected_cols:
        assert c in standardized.columns


def test_standardize_separate_year_month():
    """Verify standardizer handles datasets with split year and month columns."""
    split_df = pd.DataFrame(
        {
            "game": ["TF2"],
            "year": [2020],
            "month": ["March"],
            "avg": [50000.0],
        }
    )
    standardized = standardize_columns(split_df)
    assert "Month_Year" in standardized.columns
    assert standardized["Month_Year"].iloc[0] == "March 2020"


# ==============================================================================
# 3. Data Cleaning & Validation Tests
# ==============================================================================

def test_clean_raw_data_removes_artifacts(sample_raw_df: pd.DataFrame):
    """Verify cleaning drops 'Last 30 Days', unparseable dates, and deduplicates."""
    cleaned = clean_raw_data(sample_raw_df)

    # 'Last 30 Days' and 'Invalid Date String' should be excluded
    date_strs = cleaned["Month_Year"].dt.strftime("%Y-%m-%d").tolist()
    assert len(cleaned) == 4  # 4 unique valid monthly records (June, July, August, September)

    # Dates must be month starts (day == 1)
    assert (cleaned["Month_Year"].dt.day == 1).all()

    # Target values must be numeric float/int
    assert pd.api.types.is_float_dtype(cleaned["Avg_players"])
    assert pd.api.types.is_integer_dtype(cleaned["Peak_Players"])

    # Peak must be >= average
    assert (cleaned["Peak_Players"] >= cleaned["Avg_players"]).all()


def test_clean_raw_data_reconciles_gain(sample_raw_df: pd.DataFrame):
    """Verify Gain and Percent_Gain are correctly computed chronologically."""
    cleaned = clean_raw_data(sample_raw_df)
    # Chronological: June 2021, July 2021, August 2021, September 2021
    # June is the earliest month -> Gain should be NaN
    assert np.isnan(cleaned["Gain"].iloc[0])

    # July Gain = 490500 - 470000 = 20500
    expected_gain_july = 490500.0 - 470000.0
    assert pytest.approx(cleaned["Gain"].iloc[1], 0.1) == expected_gain_july


def test_validate_cleaned_data_passes_clean_data(sample_raw_df: pd.DataFrame):
    """Verify validation passes on properly cleaned dataset."""
    cleaned = clean_raw_data(sample_raw_df)
    report = validate_cleaned_data(cleaned)
    assert report["is_valid"] is True
    assert len(report["errors"]) == 0
    assert report["checks"]["has_canonical_columns"] is True
    assert report["checks"]["all_dates_month_start"] is True
    assert report["checks"]["no_duplicate_dates_per_game"] is True


def test_validate_cleaned_data_fails_on_corrupt_data():
    """Verify validator catches negative players and missing canonical columns."""
    corrupt_df = pd.DataFrame(
        {
            "Game_Name": ["Test Game"],
            "Month_Year": [pd.Timestamp("2021-09-15")],  # Not 1st of month!
            "Avg_players": [-50.0],  # Negative player count!
            "Peak_Players": [10.0],  # Peak lower than magnitude!
        }
    )
    report = validate_cleaned_data(corrupt_df)
    assert report["is_valid"] is False
    assert len(report["errors"]) > 0


# ==============================================================================
# 4. Continuity & Representative Game Selection Tests
# ==============================================================================

def test_check_game_continuity():
    """Verify continuity analysis detects continuous sequences vs temporal gaps."""
    # Continuous sequence: 12 months
    continuous_dates = pd.date_range("2020-01-01", "2020-12-01", freq="MS")
    df_cont = pd.DataFrame({"Month_Year": continuous_dates})
    res_cont = check_game_continuity(df_cont)
    assert res_cont["is_continuous"] is True
    assert res_cont["total_months"] == 12
    assert len(res_cont["missing_months"]) == 0

    # Sequence with gap (missing March 2020)
    gapped_dates = pd.to_datetime(["2020-01-01", "2020-02-01", "2020-04-01"])
    df_gap = pd.DataFrame({"Month_Year": gapped_dates})
    res_gap = check_game_continuity(df_gap)
    assert res_gap["is_continuous"] is False
    assert len(res_gap["missing_months"]) == 1
    assert res_gap["missing_months"][0] == pd.Timestamp("2020-03-01")


def test_select_representative_games_filters_by_history():
    """Verify game selection filters out games with insufficient history (< 3 years)."""
    # Create synthetic dataset with CS:GO (long) and Short Game (< 1 year)
    long_dates = pd.date_range("2015-01-01", "2021-08-01", freq="MS")
    short_dates = pd.date_range("2021-01-01", "2021-08-01", freq="MS")

    records = []
    for d in long_dates:
        records.append({"Game_Name": "CS:GO", "Month_Year": d, "Avg_players": 500000.0, "Peak_Players": 800000})
    for d in short_dates:
        records.append({"Game_Name": "Short Title", "Month_Year": d, "Avg_players": 80000.0, "Peak_Players": 120000})

    df = pd.DataFrame(records)
    df = clean_raw_data(df)

    rep_df, selected = select_representative_games(
        df,
        candidate_games=["CS:GO", "Short Title"],
        min_years=3.0,
        min_months=36,
    )
    assert "CS:GO" in selected
    assert "Short Title" not in selected
    assert set(rep_df["Game_Name"].unique()) == {"CS:GO"}


# ==============================================================================
# 5. End-to-End Pipeline & Loader Tests
# ==============================================================================

def test_slugify_game_name():
    """Verify title slugification produces valid filesystem filenames."""
    assert slugify_game_name("Counter-Strike: Global Offensive") == "counter_strike_global_offensive"
    assert slugify_game_name("Tom Clancy's Rainbow Six Siege") == "tom_clancy_s_rainbow_six_siege"
    assert slugify_game_name("Dota 2") == "dota_2"


def test_phase1_pipeline_execution(tmp_path: Path):
    """Verify end-to-end Phase 1 execution from ingestion to processed CSV exports."""
    custom_config = {
        "project": {"name": "test-project", "seed": 42},
        "data": {
            "raw_dir": str(tmp_path / "raw"),
            "processed_dir": str(tmp_path / "processed"),
            "raw_filename": "All_Steam_Games.csv",
        },
        "candidate_games": [
            "Counter-Strike: Global Offensive",
            "Dota 2",
            "Team Fortress 2",
            "Rust",
            "Warframe",
            "Grand Theft Auto V",
            "Tom Clancy's Rainbow Six Siege",
            "Apex Legends",  # will be filtered out due to < 3 years
        ],
    }

    result = run_phase1_pipeline(
        config_path=None,
        force_download=False,
        save_processed=True,
    )
    assert result["status"] == "success"
    assert len(result["selected_games"]) >= 5
    assert len(result["selected_games"]) <= 10
    assert result["total_records"] > 0
    assert result["validation_report"]["is_valid"] is True

    # Check combined processed file exists
    combined_file = Path(result["processed_files"]["combined_csv"])
    assert combined_file.exists()

    # Check loaded dataset adheres to expectations
    df_loaded = load_processed_data(filepath=combined_file)
    assert not df_loaded.empty
    assert "Month_Year" in df_loaded.columns
    assert pd.api.types.is_datetime64_any_dtype(df_loaded["Month_Year"])

    # Check individual game loading
    first_game = result["selected_games"][0]
    df_single = load_processed_data(game_name=first_game, config=load_config())
    assert not df_single.empty
    assert (df_single["Game_Name"] == first_game).all()

    # Check available games listing
    available = get_available_games(config=load_config())
    assert len(available) >= 5
    assert first_game in available
