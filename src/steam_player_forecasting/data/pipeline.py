"""Standardized Phase 1 Data Ingestion, Cleaning, and Validation Pipeline."""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd

from steam_player_forecasting.data.ingestion import acquire_dataset
from steam_player_forecasting.data.loader import get_project_root, load_config
from steam_player_forecasting.data.selection import (
    get_game_coverage_summary,
    select_representative_games,
)
from steam_player_forecasting.data.validation import (
    clean_raw_data,
    inspect_raw_schema,
    validate_cleaned_data,
)

logger = logging.getLogger(__name__)


def slugify_game_name(name: str) -> str:
    """Convert a game title to a clean filesystem slug (e.g., 'counter_strike_global_offensive')."""
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip())
    return cleaned.strip("_").lower()


def run_phase1_pipeline(
    config_path: Path | str | None = None,
    force_download: bool = False,
    save_processed: bool = True,
) -> dict[str, Any]:
    """Execute the end-to-end Phase 1 data engineering workflow.

    Workflow:
    1. Load configuration from configs/default.yaml.
    2. Acquire raw dataset (from data/raw/, Kaggle, or synthetic fallback).
    3. Inspect raw schema and report findings.
    4. Clean raw dataset: normalize columns, parse monthly start dates, sanitize numerics.
    5. Validate cleaned dataset adherence to schema constraints and monotonic chronological order.
    6. Filter and select 5-10 representative games with continuous multi-year coverage.
    7. Save standardized datasets to data/processed/ (both combined and individual game files).
    8. Return pipeline execution summary.
    """
    config = load_config(config_path)
    root = get_project_root()

    logger.info("Starting Phase 1: Dataset Acquisition & Validation...")

    # 1. Dataset Ingestion
    raw_path = acquire_dataset(config=config, force_download=force_download)
    logger.info("Raw dataset located at: %s", raw_path)

    # 2. Schema Inspection
    raw_df = pd.read_csv(raw_path, low_memory=False)
    raw_inspection = inspect_raw_schema(raw_df)
    logger.info(
        "Raw dataset loaded: %d rows, %d columns. Detected mapping: %s",
        raw_inspection["total_rows"],
        raw_inspection["total_columns"],
        raw_inspection["detected_mapping"],
    )

    # 3. Cleaning & Standardization
    cleaned_df = clean_raw_data(raw_df, config=config)
    logger.info("Cleaned dataset: %d valid monthly records across %d titles", len(cleaned_df), cleaned_df["Game_Name"].nunique())

    # 4. Validation of entire cleaned dataset
    val_report_full = validate_cleaned_data(cleaned_df, config=config)
    if not val_report_full["is_valid"]:
        raise ValueError(f"Cleaned dataset failed validation: {val_report_full['errors']}")

    # 5. Representative Game Selection (5-10 titles with >= 3-5 years continuous coverage)
    candidate_games = config.get("candidate_games", None)
    rep_df, selected_games = select_representative_games(
        cleaned_df,
        candidate_games=candidate_games,
        min_years=3.0,
        min_months=36,
        max_games=10,
    )

    logger.info("Selected %d representative games for time-series benchmarking: %s", len(selected_games), selected_games)

    # Validate subset
    val_report_rep = validate_cleaned_data(rep_df, config=config)
    if not val_report_rep["is_valid"]:
        raise ValueError(f"Representative games subset failed validation: {val_report_rep['errors']}")

    coverage_summary = get_game_coverage_summary(rep_df)

    # 6. Save Standardized Data Pipeline Outputs
    processed_dir = root / config.get("data", {}).get("processed_dir", "data/processed")
    processed_files: dict[str, str] = {}

    if save_processed:
        processed_dir.mkdir(parents=True, exist_ok=True)

        # 6a. Combined representative games dataset
        combined_csv = processed_dir / "steam_games_monthly.csv"
        # Format date as standard YYYY-MM-01
        rep_df_export = rep_df.copy()
        rep_df_export["Month_Year"] = rep_df_export["Month_Year"].dt.strftime("%Y-%m-%d")
        rep_df_export.to_csv(combined_csv, index=False)
        processed_files["combined_csv"] = str(combined_csv)
        logger.info("Saved combined standardized dataset: %s (%d rows)", combined_csv, len(rep_df_export))

        # 6b. Individual game time-series files
        for game_name in selected_games:
            slug = slugify_game_name(game_name)
            game_file = processed_dir / f"{slug}.csv"
            game_df = rep_df_export[rep_df_export["Game_Name"] == game_name].copy()
            game_df.to_csv(game_file, index=False)
            processed_files[f"game_{slug}"] = str(game_file)
            logger.info("Saved game dataset for '%s': %s (%d rows)", game_name, game_file, len(game_df))

        # 6c. Optional Parquet export if engine is available
        try:
            combined_parquet = processed_dir / "steam_games_monthly.parquet"
            rep_df.to_parquet(combined_parquet, index=False)
            processed_files["combined_parquet"] = str(combined_parquet)
            logger.info("Saved Parquet dataset: %s", combined_parquet)
        except Exception:
            logger.debug("Parquet export skipped (no pyarrow/fastparquet engine installed).")

    return {
        "status": "success",
        "raw_path": str(raw_path),
        "processed_dir": str(processed_dir),
        "processed_files": processed_files,
        "selected_games": selected_games,
        "total_records": len(rep_df),
        "raw_inspection": raw_inspection,
        "validation_report": val_report_rep,
        "coverage_summary": coverage_summary.to_dict(orient="records"),
    }


def main() -> None:
    """CLI entrypoint for Phase 1 data pipeline execution."""
    parser = argparse.ArgumentParser(
        description="Phase 1: Dataset Acquisition & Validation for Steam Player Forecasting"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file (defaults to configs/default.yaml)",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-downloading or re-generating raw data",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    result = run_phase1_pipeline(
        config_path=args.config,
        force_download=args.force_download,
    )
    print("\n=== Phase 1 Pipeline Completed Successfully ===")
    print(f"Total processed records: {result['total_records']}")
    print(f"Representative games ({len(result['selected_games'])}): {', '.join(result['selected_games'])}")
    print(f"Combined processed output: {result['processed_files'].get('combined_csv')}")


if __name__ == "__main__":
    main()
