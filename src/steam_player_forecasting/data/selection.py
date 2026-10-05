"""Representative game selection and historical coverage analysis."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from steam_player_forecasting.data.loader import load_config

logger = logging.getLogger(__name__)


def check_game_continuity(
    game_df: pd.DataFrame,
    date_col: str = "Month_Year",
) -> dict[str, Any]:
    """Analyze the temporal continuity of a single game's monthly observations.

    Verifies whether there are any missing monthly intervals between the first
    and last recorded observations.
    """
    if game_df.empty:
        return {
            "is_continuous": False,
            "start_date": None,
            "end_date": None,
            "total_months": 0,
            "expected_months": 0,
            "missing_months": [],
            "span_years": 0.0,
        }

    sorted_dates = pd.to_datetime(game_df[date_col]).sort_values().reset_index(drop=True)
    start_dt = sorted_dates.iloc[0]
    end_dt = sorted_dates.iloc[-1]

    # Full expected monthly sequence
    expected_range = pd.date_range(start=start_dt, end=end_dt, freq="MS")
    actual_set = set(sorted_dates)
    missing = [d for d in expected_range if d not in actual_set]

    total_months = len(actual_set)
    expected_months = len(expected_range)
    span_years = round(total_months / 12.0, 2)

    return {
        "is_continuous": len(missing) == 0,
        "start_date": start_dt,
        "end_date": end_dt,
        "total_months": total_months,
        "expected_months": expected_months,
        "missing_months": missing,
        "span_years": span_years,
    }


def get_game_coverage_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Generate a high-level coverage and continuity summary for all games in the dataset."""
    rows: list[dict[str, Any]] = []

    for game_name, group in df.groupby("Game_Name"):
        continuity = check_game_continuity(group)
        mean_avg = round(float(group["Avg_players"].mean()), 2) if "Avg_players" in group else 0.0
        max_peak = int(group["Peak_Players"].max()) if "Peak_Players" in group else 0

        rows.append(
            {
                "Game_Name": game_name,
                "start_date": continuity["start_date"].strftime("%Y-%m-%d") if continuity["start_date"] else None,
                "end_date": continuity["end_date"].strftime("%Y-%m-%d") if continuity["end_date"] else None,
                "total_months": continuity["total_months"],
                "span_years": continuity["span_years"],
                "is_continuous": continuity["is_continuous"],
                "missing_months_count": len(continuity["missing_months"]),
                "mean_avg_players": mean_avg,
                "max_peak_players": max_peak,
            }
        )

    summary_df = pd.DataFrame(rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(
            by=["is_continuous", "span_years", "mean_avg_players"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
    return summary_df


def select_representative_games(
    df: pd.DataFrame,
    candidate_games: list[str] | None = None,
    min_years: float = 3.0,
    min_months: int = 36,
    max_games: int = 10,
) -> tuple[pd.DataFrame, list[str]]:
    """Select representative games meeting long-term historical coverage and continuity requirements.

    Criteria:
    - If candidate_games is provided (or configured in default.yaml), prioritizes games from this list.
    - Requires continuous monthly records (or zero/minimal missing intervals).
    - Requires at least `min_years` (or `min_months`) of observations.
    - Selects between 5 and 10 representative games.

    Returns:
    - Filtered DataFrame containing only the selected games.
    - List of selected game titles.
    """
    if candidate_games is None:
        try:
            cfg = load_config()
            candidate_games = cfg.get("candidate_games", None)
        except Exception:
            candidate_games = None

    summary = get_game_coverage_summary(df)
    selected_games: list[str] = []
    excluded_games: dict[str, str] = {}

    # Pool of candidate games to evaluate
    if candidate_games:
        eval_pool = [g for g in candidate_games if g in summary["Game_Name"].values]
    else:
        eval_pool = summary["Game_Name"].tolist()

    for game in eval_pool:
        row = summary[summary["Game_Name"] == game].iloc[0]
        months = row["total_months"]
        years = row["span_years"]
        is_cont = row["is_continuous"]

        if months < min_months or years < min_years:
            excluded_games[game] = f"Insufficient history ({years:.1f} yrs / {months} months < min {min_years} yrs)"
            continue

        if not is_cont and row["missing_months_count"] > 2:
            excluded_games[game] = f"Significant temporal gaps ({row['missing_months_count']} missing months)"
            continue

        selected_games.append(game)
        if len(selected_games) >= max_games:
            break

    # If fewer than 5 games selected and candidate pool was restrictive, backfill from remaining top games
    if len(selected_games) < 5:
        logger.info(
            "Selected %d candidate games; searching remaining pool for additional representative titles...",
            len(selected_games),
        )
        for _, row in summary.iterrows():
            game = row["Game_Name"]
            if game in selected_games:
                continue
            if row["total_months"] >= min_months and row["is_continuous"]:
                selected_games.append(game)
                if len(selected_games) >= 7:
                    break

    logger.info("Selected %d representative games: %s", len(selected_games), selected_games)
    for excl, reason in excluded_games.items():
        logger.info("Excluded '%s': %s", excl, reason)

    filtered_df = df[df["Game_Name"].isin(selected_games)].reset_index(drop=True)
    return filtered_df, selected_games
