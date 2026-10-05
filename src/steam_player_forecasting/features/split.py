"""Chronological time-series splitting utilities ensuring zero lookahead leakage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generator, Sequence

import numpy as np
import pandas as pd


@dataclass
class TimeSeriesSplitResult:
    """Container for chronological train/validation/test split subsets.

    Attributes:
        train: Training window subset.
        val: Validation window subset.
        test: Test (holdout) window subset.
        train_indices: Integer position indices of the training set.
        val_indices: Integer position indices of the validation set.
        test_indices: Integer position indices of the test set.
        split_dates: Dictionary with boundary timestamps if temporal index/column exists.
    """

    train: pd.DataFrame | pd.Series | np.ndarray
    val: pd.DataFrame | pd.Series | np.ndarray
    test: pd.DataFrame | pd.Series | np.ndarray
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    split_dates: dict[str, Any]

    def summary(self) -> dict[str, Any]:
        """Return a summary of split sizes and date boundaries."""
        info = {
            "n_train": len(self.train),
            "n_val": len(self.val),
            "n_test": len(self.test),
            "total_observations": len(self.train) + len(self.val) + len(self.test),
        }
        info.update(self.split_dates)
        return info


def train_val_test_split(
    data: pd.DataFrame | pd.Series | np.ndarray,
    val_months: int = 12,
    test_months: int = 12,
    date_col: str | None = "Month_Year",
) -> TimeSeriesSplitResult:
    """Split time-series data chronologically into train, validation, and test sets.

    Guarantees strict chronological ordering with zero future-data leakage:
      - Train: Observations from start up to (N - val_months - test_months).
      - Validation: The val_months observations preceding the test window.
      - Test: The final test_months observations (unseen holdout).

    Args:
        data: Input DataFrame, Series, or array.
        val_months: Number of chronological observations for validation.
        test_months: Number of chronological observations for test holdout.
        date_col: Name of date column if data is a DataFrame. If present, ensures
                  data is sorted chronologically by this column before splitting.

    Returns:
        TimeSeriesSplitResult containing train, val, and test subsets and metadata.

    Raises:
        ValueError: If total observations are insufficient for requested holdouts.
    """
    if val_months < 0 or test_months < 0:
        raise ValueError("val_months and test_months must be non-negative integers.")

    n_samples = len(data)
    min_required = val_months + test_months + 1
    if n_samples < min_required:
        raise ValueError(
            f"Insufficient observations ({n_samples}) for val_months={val_months} "
            f"and test_months={test_months}. Minimum required is {min_required}."
        )

    # Sort DataFrame by date if column present
    if isinstance(data, pd.DataFrame):
        if date_col is not None and date_col in data.columns:
            data = data.sort_values(by=date_col).reset_index(drop=True)
        elif isinstance(data.index, pd.DatetimeIndex):
            data = data.sort_index()

    train_end = n_samples - (val_months + test_months)
    val_end = n_samples - test_months

    train_idx = np.arange(0, train_end)
    val_idx = np.arange(train_end, val_end)
    test_idx = np.arange(val_end, n_samples)

    if isinstance(data, (pd.DataFrame, pd.Series)):
        train_data = data.iloc[:train_end].copy()
        val_data = data.iloc[train_end:val_end].copy()
        test_data = data.iloc[val_end:].copy()
    else:
        train_data = data[:train_end]
        val_data = data[train_end:val_end]
        test_data = data[val_end:]

    split_dates: dict[str, Any] = {}
    if isinstance(data, pd.DataFrame) and date_col is not None and date_col in data.columns:
        split_dates = {
            "train_start": str(train_data[date_col].iloc[0]),
            "train_end": str(train_data[date_col].iloc[-1]),
            "val_start": str(val_data[date_col].iloc[0]) if len(val_data) > 0 else None,
            "val_end": str(val_data[date_col].iloc[-1]) if len(val_data) > 0 else None,
            "test_start": str(test_data[date_col].iloc[0]) if len(test_data) > 0 else None,
            "test_end": str(test_data[date_col].iloc[-1]) if len(test_data) > 0 else None,
        }
    elif isinstance(data, (pd.DataFrame, pd.Series)) and isinstance(data.index, pd.DatetimeIndex):
        split_dates = {
            "train_start": str(train_data.index[0]),
            "train_end": str(train_data.index[-1]),
            "val_start": str(val_data.index[0]) if len(val_data) > 0 else None,
            "val_end": str(val_data.index[-1]) if len(val_data) > 0 else None,
            "test_start": str(test_data.index[0]) if len(test_data) > 0 else None,
            "test_end": str(test_data.index[-1]) if len(test_data) > 0 else None,
        }

    return TimeSeriesSplitResult(
        train=train_data,
        val=val_data,
        test=test_data,
        train_indices=train_idx,
        val_indices=val_idx,
        test_indices=test_idx,
        split_dates=split_dates,
    )


def split_by_game(
    df: pd.DataFrame,
    val_months: int = 12,
    test_months: int = 12,
    game_col: str = "Game_Name",
    date_col: str = "Month_Year",
) -> dict[str, TimeSeriesSplitResult]:
    """Split multi-game panel dataset by game title.

    Each game's subseries is extracted and split into train, val, and test subsets
    independently respecting its chronological timeline.

    Args:
        df: Combined DataFrame containing records across games.
        val_months: Validation horizon in months.
        test_months: Test horizon in months.
        game_col: Column identifying game title.
        date_col: Date column for chronological ordering.

    Returns:
        Dictionary mapping game name to its TimeSeriesSplitResult.
    """
    if game_col not in df.columns:
        raise ValueError(f"Game identifier column '{game_col}' not found in DataFrame.")

    results: dict[str, TimeSeriesSplitResult] = {}
    for game_name, game_df in df.groupby(game_col):
        results[str(game_name)] = train_val_test_split(
            data=game_df,
            val_months=val_months,
            test_months=test_months,
            date_col=date_col,
        )

    return results


def expanding_window_cv(
    data: pd.DataFrame | pd.Series | np.ndarray,
    n_splits: int = 3,
    test_horizon: int = 12,
    min_train_months: int = 24,
    date_col: str | None = "Month_Year",
) -> Generator[tuple[Any, Any], None, None]:
    """Generate expanding-window (walk-forward) cross-validation folds.

    In each fold:
      - Train window starts from observation 0 and expands.
      - Test window consists of the subsequent test_horizon observations.

    Args:
        data: Time series dataset.
        n_splits: Number of cross-validation splits.
        test_horizon: Length of each evaluation forecast window.
        min_train_months: Minimum training length required for the first fold.
        date_col: Optional date column to ensure ordering.

    Yields:
        Tuple of (train_subset, test_subset) for each chronological fold.
    """
    n_samples = len(data)
    total_eval_needed = n_splits * test_horizon
    if n_samples < min_train_months + total_eval_needed:
        raise ValueError(
            f"Series length ({n_samples}) insufficient for n_splits={n_splits}, "
            f"test_horizon={test_horizon}, and min_train_months={min_train_months}."
        )

    if isinstance(data, pd.DataFrame) and date_col is not None and date_col in data.columns:
        data = data.sort_values(by=date_col).reset_index(drop=True)

    # Calculate train end for fold 0 so that exactly n_splits folds fit
    first_train_end = n_samples - total_eval_needed

    for i in range(n_splits):
        train_end = first_train_end + (i * test_horizon)
        test_end = train_end + test_horizon

        if isinstance(data, (pd.DataFrame, pd.Series)):
            train_fold = data.iloc[:train_end].copy()
            test_fold = data.iloc[train_end:test_end].copy()
        else:
            train_fold = data[:train_end]
            test_fold = data[train_end:test_end]

        yield train_fold, test_fold
