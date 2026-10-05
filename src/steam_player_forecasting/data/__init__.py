"""Data ingestion, validation, and loading module."""

from __future__ import annotations

from typing import TYPE_CHECKING

from steam_player_forecasting.data.ingestion import (
    acquire_dataset,
    download_from_kaggle,
    generate_synthetic_raw_data,
)
from steam_player_forecasting.data.loader import (
    get_available_games,
    get_project_root,
    load_config,
    load_processed_data,
    load_raw_data,
)
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

if TYPE_CHECKING:
    from steam_player_forecasting.data.pipeline import (
        run_phase1_pipeline,
        slugify_game_name,
    )

__all__ = [
    "acquire_dataset",
    "check_game_continuity",
    "clean_raw_data",
    "download_from_kaggle",
    "generate_synthetic_raw_data",
    "get_available_games",
    "get_game_coverage_summary",
    "get_project_root",
    "inspect_raw_schema",
    "load_config",
    "load_processed_data",
    "load_raw_data",
    "run_phase1_pipeline",
    "select_representative_games",
    "slugify_game_name",
    "standardize_columns",
    "validate_cleaned_data",
]


def __getattr__(name: str):
    if name in ("run_phase1_pipeline", "slugify_game_name"):
        from steam_player_forecasting.data.pipeline import (
            run_phase1_pipeline,
            slugify_game_name,
        )

        globals()["run_phase1_pipeline"] = run_phase1_pipeline
        globals()["slugify_game_name"] = slugify_game_name
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
