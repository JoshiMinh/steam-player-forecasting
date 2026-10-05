"""Schema inspection, cleaning, and validation routines for Steam player data."""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Canonical column definitions
CANONICAL_COLUMNS = [
    "Game_Name",
    "Month_Year",
    "Avg_players",
    "Peak_Players",
    "Gain",
    "Percent_Gain",
]

# Aliases used across different Steam scraper versions
COLUMN_ALIASES: dict[str, list[str]] = {
    "Game_Name": [
        "game_name",
        "gamename",
        "game",
        "name",
        "game_title",
        "title",
        "app_name",
    ],
    "Month_Year": [
        "month_year",
        "month",
        "date",
        "timestamp",
        "period",
        "time",
    ],
    "Avg_players": [
        "avg_players",
        "avg",
        "avg_player_count",
        "average_players",
        "avg. players",
        "average",
    ],
    "Peak_Players": [
        "peak_players",
        "peak",
        "peak_player_count",
        "highest_players",
        "peak players",
        "peak. players",
    ],
    "Gain": [
        "gain",
        "net_gain",
        "player_gain",
        "diff",
    ],
    "Percent_Gain": [
        "percent_gain",
        "% gain",
        "%gain",
        "percentage_gain",
        "pct_gain",
        "%_gain",
        "gain_percent",
    ],
}


def inspect_raw_schema(df: pd.DataFrame) -> dict[str, Any]:
    """Inspect raw DataFrame schema, detected columns, and potential data quality issues.

    Returns a detailed summary dictionary of detected properties.
    """
    column_names = [str(c).strip() for c in df.columns]
    normalized_cols = {c: re.sub(r"[^a-zA-Z0-9%]", "_", c).lower() for c in column_names}

    detected_mapping: dict[str, str] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for orig, norm in normalized_cols.items():
            if norm in aliases or norm == canonical.lower():
                detected_mapping[canonical] = orig
                break

    # Look for separate year and month columns if Month_Year is missing
    has_year = any("year" in norm for norm in normalized_cols.values())
    has_month = any(
        norm in ["month", "month_name"] for norm in normalized_cols.values()
    )

    sample_dict: dict[str, Any] = {}
    for col in df.columns[:8]:
        sample_dict[str(col)] = df[col].dropna().head(3).tolist()

    return {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "columns": column_names,
        "dtypes": {str(c): str(t) for c, t in df.dtypes.items()},
        "null_counts": {str(c): int(df[c].isna().sum()) for c in df.columns},
        "detected_mapping": detected_mapping,
        "has_separate_year_month": bool(has_year and has_month),
        "sample_records": sample_dict,
    }


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map raw columns to canonical naming convention.

    Standardizes columns to:
    Game_Name, Month_Year, Avg_players, Peak_Players, Gain, Percent_Gain
    """
    df = df.copy()

    # Handle separate year and month columns before alias renaming
    raw_col_norms = {c: re.sub(r"[^a-zA-Z0-9%]", "_", str(c)).lower().strip("_") for c in df.columns}
    has_explicit_month_year = any(
        norm in ["month_year", "monthyear", "period", "timestamp"]
        for norm in raw_col_norms.values()
    )
    year_col = next((c for c, norm in raw_col_norms.items() if norm in ["year", "yr"]), None)
    month_col = next((c for c, norm in raw_col_norms.items() if norm in ["month", "mo", "month_name"]), None)

    if not has_explicit_month_year and year_col and month_col:
        df["Month_Year"] = df[month_col].astype(str) + " " + df[year_col].astype(str)
        # Drop raw year and month to avoid conflicting with Month_Year alias mapping
        cols_to_drop = [c for c in [year_col, month_col] if c != "Month_Year"]
        df = df.drop(columns=cols_to_drop)
        raw_cols = {c: re.sub(r"[^a-zA-Z0-9%]", "_", str(c)).lower().strip("_") for c in df.columns}
    else:
        raw_cols = raw_col_norms

    rename_map: dict[str, str] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for orig, norm in raw_cols.items():
            if orig in rename_map:
                continue
            if norm == canonical.lower() or norm in aliases:
                rename_map[orig] = canonical
                break

    df = df.rename(columns=rename_map)
    return df


def _clean_numeric_series(series: pd.Series) -> pd.Series:
    """Clean a numeric series that may contain commas, percentages, dashes, and strings."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)

    cleaned = (
        series.astype(str)
        .str.strip()
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.replace("+", "", regex=False)
    )
    # Replace non-numeric placeholders with NaN
    cleaned = cleaned.replace(
        {"-": np.nan, "NA": np.nan, "null": np.nan, "None": np.nan, "": np.nan, "nan": np.nan}
    )
    return pd.to_numeric(cleaned, errors="coerce")


def clean_raw_data(
    df: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Clean and standardize raw Steam player data.

    Steps:
    1. Standardize column names into canonical schema.
    2. Strip string whitespace from text columns.
    3. Remove invalid/non-monthly entries such as 'Last 30 Days' or blank periods.
    4. Parse timestamps to standard monthly start (`YYYY-MM-01`).
    5. Clean numeric columns (remove commas, %, -, convert to float/int).
    6. Drop missing target records (`Avg_players` is null or <= 0).
    7. Validate and enforce `Peak_Players >= Avg_players`.
    8. Remove duplicate records for the same game and month.
    9. Sort chronologically per game.
    10. Recalculate/sanitize `Gain` and `Percent_Gain` from consecutive observations.
    """
    df = standardize_columns(df)

    # Validate essential columns exist
    for req in ["Game_Name", "Month_Year", "Avg_players"]:
        if req not in df.columns:
            raise ValueError(
                f"Required column '{req}' could not be identified in raw dataset. "
                f"Available columns: {list(df.columns)}"
            )

    # 1. Clean Game_Name
    df["Game_Name"] = df["Game_Name"].astype(str).str.strip()
    df = df[df["Game_Name"].ne("") & df["Game_Name"].str.lower().ne("nan")].copy()

    # 2. Filter out non-monthly rows (e.g. 'Last 30 Days', 'All Time')
    invalid_date_patterns = r"last\s*30\s*days|all\s*time|total"
    is_invalid_period = (
        df["Month_Year"].astype(str).str.strip().str.lower().str.contains(
            invalid_date_patterns, regex=True
        )
    )
    df = df[~is_invalid_period].copy()

    # 3. Parse timestamps to standard monthly start (YYYY-MM-01)
    parsed_dates = pd.to_datetime(df["Month_Year"], format="mixed", errors="coerce")
    valid_dates_mask = parsed_dates.notna()
    df = df[valid_dates_mask].copy()
    parsed_dates = parsed_dates[valid_dates_mask]

    # Normalize to 1st of month: YYYY-MM-01
    df["Month_Year"] = parsed_dates.dt.to_period("M").dt.to_timestamp()

    # 4. Clean numeric targets and features
    df["Avg_players"] = _clean_numeric_series(df["Avg_players"])

    if "Peak_Players" in df.columns:
        df["Peak_Players"] = _clean_numeric_series(df["Peak_Players"])
    else:
        df["Peak_Players"] = df["Avg_players"]

    if "Gain" in df.columns:
        df["Gain"] = _clean_numeric_series(df["Gain"])
    else:
        df["Gain"] = np.nan

    if "Percent_Gain" in df.columns:
        df["Percent_Gain"] = _clean_numeric_series(df["Percent_Gain"])
    else:
        df["Percent_Gain"] = np.nan

    # 5. Filter missing or non-positive target rows
    df = df.dropna(subset=["Avg_players"])
    df = df[df["Avg_players"] >= 0.0].copy()

    # If peak is missing or smaller than average players due to scrape anomalies, clamp peak
    df["Peak_Players"] = df["Peak_Players"].fillna(df["Avg_players"])
    df["Peak_Players"] = np.maximum(df["Peak_Players"], df["Avg_players"])
    df["Peak_Players"] = df["Peak_Players"].round().astype(int)

    # 6. Deduplicate: keep record with highest Peak_Players if duplicate game/month exists
    df = df.sort_values(by=["Game_Name", "Month_Year", "Peak_Players"], ascending=[True, True, False])
    df = df.drop_duplicates(subset=["Game_Name", "Month_Year"], keep="first")

    # 7. Sort strictly chronologically per game
    df = df.sort_values(by=["Game_Name", "Month_Year"], ascending=[True, True]).reset_index(drop=True)

    # 8. Reconcile Gain and Percent_Gain for strict mathematical consistency
    # Gain_t = Avg_players_t - Avg_players_{t-1}
    # Percent_Gain_t = (Gain_t / Avg_players_{t-1}) * 100.0
    reconciled_gains: list[float] = []
    reconciled_pcts: list[float] = []

    for _, group in df.groupby("Game_Name", sort=False):
        avg_vals = group["Avg_players"].to_numpy()
        gains = np.full(len(avg_vals), np.nan)
        pcts = np.full(len(avg_vals), np.nan)

        if len(avg_vals) > 1:
            diffs = avg_vals[1:] - avg_vals[:-1]
            gains[1:] = np.round(diffs, 2)
            # Avoid division by zero
            with np.errstate(divide="ignore", invalid="ignore"):
                pct = (diffs / np.where(avg_vals[:-1] == 0, np.nan, avg_vals[:-1])) * 100.0
                pcts[1:] = np.round(pct, 2)

        reconciled_gains.extend(gains)
        reconciled_pcts.extend(pcts)

    df["Gain"] = reconciled_gains
    df["Percent_Gain"] = reconciled_pcts

    # Keep canonical column order
    return df[CANONICAL_COLUMNS].reset_index(drop=True)


def validate_cleaned_data(
    df: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate cleaned DataFrame against schema constraints and time-series rules.

    Checks:
    - Canonical columns presence
    - No null values in Game_Name, Month_Year, Avg_players, Peak_Players
    - All timestamps are valid 1st of month (YYYY-MM-01)
    - Avg_players >= 0 and Peak_Players >= Avg_players
    - No duplicate timestamps per game
    - Monotonically increasing timestamps per game
    """
    checks: dict[str, bool] = {}
    errors: list[str] = []

    # Check columns
    missing_cols = [c for c in CANONICAL_COLUMNS if c not in df.columns]
    checks["has_canonical_columns"] = len(missing_cols) == 0
    if missing_cols:
        errors.append(f"Missing canonical columns: {missing_cols}")

    # Check nulls in primary fields
    null_games = int(df["Game_Name"].isna().sum()) if "Game_Name" in df else -1
    null_dates = int(df["Month_Year"].isna().sum()) if "Month_Year" in df else -1
    null_targets = int(df["Avg_players"].isna().sum()) if "Avg_players" in df else -1
    checks["no_null_primary_fields"] = (null_games == 0 and null_dates == 0 and null_targets == 0)
    if not checks["no_null_primary_fields"]:
        errors.append(f"Nulls found: games={null_games}, dates={null_dates}, targets={null_targets}")

    # Check timestamps: must be Timestamp with day == 1
    if "Month_Year" in df and pd.api.types.is_datetime64_any_dtype(df["Month_Year"]):
        invalid_days = int((df["Month_Year"].dt.day != 1).sum())
        checks["all_dates_month_start"] = invalid_days == 0
        if invalid_days > 0:
            errors.append(f"Found {invalid_days} dates that are not the 1st of month")
    else:
        checks["all_dates_month_start"] = False
        errors.append("Month_Year is not datetime64 dtype")

    # Check non-negative target and peak >= avg
    if "Avg_players" in df and "Peak_Players" in df:
        neg_avg = int((df["Avg_players"] < 0).sum())
        invalid_peaks = int((df["Peak_Players"] < df["Avg_players"]).sum())
        checks["non_negative_averages"] = neg_avg == 0
        checks["peak_gte_average"] = invalid_peaks == 0
        if neg_avg > 0:
            errors.append(f"Found {neg_avg} negative Avg_players records")
        if invalid_peaks > 0:
            errors.append(f"Found {invalid_peaks} records where Peak_Players < Avg_players")

    # Check duplicates per game and month
    if "Game_Name" in df and "Month_Year" in df:
        dups = int(df.duplicated(subset=["Game_Name", "Month_Year"]).sum())
        checks["no_duplicate_dates_per_game"] = dups == 0
        if dups > 0:
            errors.append(f"Found {dups} duplicate (Game_Name, Month_Year) records")

        # Check monotonic ordering
        is_monotonic = True
        for game, grp in df.groupby("Game_Name"):
            if not grp["Month_Year"].is_monotonic_increasing:
                is_monotonic = False
                errors.append(f"Dates are not monotonically increasing for game '{game}'")
                break
        checks["monotonically_increasing_dates"] = is_monotonic

    is_valid = all(checks.values()) and len(errors) == 0

    return {
        "is_valid": is_valid,
        "checks": checks,
        "errors": errors,
        "total_records": len(df),
        "total_games": df["Game_Name"].nunique() if "Game_Name" in df else 0,
        "date_min": str(df["Month_Year"].min().strftime("%Y-%m-%d")) if "Month_Year" in df and not df.empty else None,
        "date_max": str(df["Month_Year"].max().strftime("%Y-%m-%d")) if "Month_Year" in df and not df.empty else None,
    }
