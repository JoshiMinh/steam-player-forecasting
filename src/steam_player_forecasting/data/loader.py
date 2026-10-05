"""Data loading, acquisition, and configuration utilities."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def get_project_root() -> Path:
    """Return the repository root directory."""
    return Path(__file__).resolve().parent.parent.parent.parent


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML project configuration.

    If config_path is None, defaults to `configs/default.yaml` relative to project root.
    """
    if config_path is None:
        config_path = get_project_root() / "configs" / "default.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def load_raw_data(
    filepath: str | Path | None = None,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Load raw Steam player dataset.

    If filepath is not provided, resolves default raw file from configuration.
    """
    if filepath is None:
        if config is None:
            config = load_config()
        root = get_project_root()
        raw_dir = root / config.get("data", {}).get("raw_dir", "data/raw")
        raw_filename = config.get("data", {}).get("raw_filename", "All_Steam_Games.csv")
        filepath = raw_dir / raw_filename

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Raw data file not found at: {filepath}")

    return pd.read_csv(filepath, low_memory=False)


def load_processed_data(
    game_name: str | None = None,
    filepath: str | Path | None = None,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Load standardized processed Steam player dataset.

    Args:
        game_name: Optional title to load only that game's data. If specified,
                   attempts to load either the individual game CSV or filters
                   from the combined dataset.
        filepath: Optional path to specific processed CSV file.
        config: Optional project configuration dictionary.

    Returns:
        DataFrame with standardized columns and datetime index or Month_Year timestamp.
    """
    root = get_project_root()
    if config is None:
        try:
            config = load_config()
        except Exception:
            config = {}

    processed_dir = root / config.get("data", {}).get("processed_dir", "data/processed")

    if filepath is not None:
        target_path = Path(filepath)
    elif game_name is not None:
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", game_name.strip()).strip("_").lower()
        game_csv = processed_dir / f"{slug}.csv"
        if game_csv.exists():
            target_path = game_csv
        else:
            target_path = processed_dir / "steam_games_monthly.csv"
    else:
        target_path = processed_dir / "steam_games_monthly.csv"

    if not target_path.exists():
        raise FileNotFoundError(
            f"Processed data file not found at {target_path}. "
            "Please run the Phase 1 pipeline first via run_phase1_pipeline()."
        )

    df = pd.read_csv(target_path, parse_dates=["Month_Year"])

    if game_name is not None and "Game_Name" in df.columns:
        match_mask = df["Game_Name"].str.lower() == game_name.strip().lower()
        if not match_mask.any():
            slug_input = re.sub(r"[^a-zA-Z0-9]+", "_", game_name.strip()).strip("_").lower()
            match_mask = df["Game_Name"].apply(
                lambda s: re.sub(r"[^a-zA-Z0-9]+", "_", str(s).strip()).strip("_").lower() == slug_input
            )
        df = df[match_mask].copy()
        if df.empty:
            raise ValueError(f"No records found for game '{game_name}' in {target_path}")

    return df.sort_values(by="Month_Year").reset_index(drop=True)


def get_available_games(
    processed_path: str | Path | None = None,
    config: dict[str, Any] | None = None,
) -> list[str]:
    """Return a list of all game titles available in the processed dataset."""
    df = load_processed_data(filepath=processed_path, config=config)
    if "Game_Name" not in df.columns:
        return []
    return sorted(df["Game_Name"].unique().tolist())
