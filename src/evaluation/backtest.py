"""Rolling-origin backtesting utilities.

Builds non-overlapping, strictly chronological forecast origins, aggregates
per-seed results, and summarises model performance with bootstrap uncertainty
and paired significance tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from src.evaluation.significance import (
    block_bootstrap_ci,
    diebold_mariano,
    paired_bootstrap_diff,
)

UNIT_COLS = ["Game", "Fold"]


@dataclass(frozen=True)
class Fold:
    """One forecast origin: fit on ``train`` and forecast ``test``."""

    k: int
    train: pd.Series
    test: pd.Series

    @property
    def origin(self) -> str:
        """Last training month (the forecast origin)."""
        return str(self.train.index[-1].strftime("%Y-%m"))


def rolling_origin_folds(
    series: pd.Series,
    n_folds: int = 4,
    horizon: int = 12,
    min_train: int = 28,
) -> list[Fold]:
    """Create non-overlapping rolling-origin folds ending at the series end.

    Fold 0 is the final ``horizon`` months (the published holdout). Fold ``k``
    ends ``horizon * k`` months earlier. Folds whose training window would be
    shorter than ``min_train`` observations are dropped.

    Args:
        series: Datetime-indexed series.
        n_folds: Maximum number of folds.
        horizon: Forecast horizon in months.
        min_train: Minimum number of training observations.

    Returns:
        Folds ordered from fold 0 (most recent) to the oldest.
    """
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1")
    series = series.sort_index()
    n = len(series)
    folds: list[Fold] = []
    for k in range(n_folds):
        test_end = n - horizon * k
        test_start = test_end - horizon
        if test_start < min_train:
            break
        folds.append(
            Fold(
                k=k,
                train=series.iloc[:test_start],
                test=series.iloc[test_start:test_end],
            )
        )
    return folds


MASE_CAP = 5.0
"""A forecast with MASE above this is treated as diverged (worse than 5x the
in-sample seasonal-naive error). Aggregates and tests cap each unit at this
value so a single runaway fit cannot dominate a mean; raw values stay in the CSVs."""


def seed_average_metrics(
    fold_metrics: pd.DataFrame,
    metric_cols: Iterable[str] = ("MAE", "RMSE", "MAPE", "sMAPE", "MASE"),
    mase_cap: float = MASE_CAP,
) -> pd.DataFrame:
    """Average metrics over seeds so each (Game, Fold, Model) is one unit.

    Seed averaging happens before ranking, so models with several seeds do not
    receive extra weight. ``MASE`` is capped at ``mase_cap`` per fit before
    averaging; ``MASE_Raw`` keeps the uncapped seed average.
    """
    df = fold_metrics.copy()
    if "MASE" in df.columns:
        df["MASE_Raw"] = df["MASE"]
        df["MASE"] = df["MASE"].clip(upper=mase_cap)
    cols = [c for c in metric_cols if c in df.columns]
    if "MASE_Raw" in df.columns:
        cols.append("MASE_Raw")
    group = ["Game", "Fold", "Model"]
    first = df.groupby(group, sort=False)[["Paradigm"]].first()
    mean = df.groupby(group, sort=False)[cols].mean()
    return first.join(mean).reset_index()


def seed_average_errors(errors: pd.DataFrame, cap: float = MASE_CAP) -> pd.DataFrame:
    """Average scaled absolute errors over seeds per horizon step.

    Each step is capped at ``cap`` scaled units (same ceiling as the MASE cap).

    Returns columns Game, Fold, Model, Horizon_Step, Scaled_Abs_Error.
    """
    df = errors.copy()
    df["Scaled_Abs_Error"] = (df["Abs_Error"] / df["Scale"]).clip(upper=cap)
    out = (
        df.groupby(["Game", "Fold", "Model", "Horizon_Step"], sort=False)["Scaled_Abs_Error"]
        .mean()
        .reset_index()
    )
    return out


def summarize_models(
    fold_metrics: pd.DataFrame,
    n_boot: int = 5000,
    seed: int = 0,
) -> pd.DataFrame:
    """Summarise MASE per model across all (game, fold) units.

    Columns: Rank, Model, Paradigm, Mean_MASE (units capped at ``MASE_CAP``),
    CI_Low, CI_High, Std_MASE (across units), Median_MASE, Raw_Mean_MASE
    (uncapped), Diverged_Units (raw MASE above the cap), Fold0_Mean_MASE
    (mean over games on the latest holdout), Mean_Seed_Std (average MASE std
    across seeds; NaN for deterministic models), Mean_Rank (average per-unit
    rank), Units.
    """
    units = seed_average_metrics(fold_metrics)

    seed_std = (
        fold_metrics.assign(MASE=fold_metrics["MASE"].clip(upper=MASE_CAP))
        .groupby(["Game", "Fold", "Model"])["MASE"]
        .agg(["std", "count"])
        .reset_index()
    )
    seeded = seed_std[seed_std["count"] > 1]
    mean_seed_std = seeded.groupby("Model")["std"].mean()

    units["Unit_Rank"] = units.groupby(UNIT_COLS)["MASE"].rank(method="average")
    rows = []
    for model, grp in units.groupby("Model", sort=False):
        mean, lo, hi = block_bootstrap_ci(grp["MASE"].values, n_boot=n_boot, seed=seed)
        fold0 = grp[grp["Fold"] == 0]["MASE"]
        rows.append(
            {
                "Model": model,
                "Paradigm": grp["Paradigm"].iloc[0],
                "Mean_MASE": mean,
                "CI_Low": lo,
                "CI_High": hi,
                "Std_MASE": float(grp["MASE"].std(ddof=1)) if len(grp) > 1 else float("nan"),
                "Median_MASE": float(grp["MASE"].median()),
                "Raw_Mean_MASE": float(grp["MASE_Raw"].mean()),
                "Diverged_Units": int((grp["MASE_Raw"] > MASE_CAP).sum()),
                "Fold0_Mean_MASE": float(fold0.mean()) if len(fold0) else float("nan"),
                "Mean_Seed_Std": float(mean_seed_std.get(model, np.nan)),
                "Mean_Rank": float(grp["Unit_Rank"].mean()),
                "Units": int(len(grp)),
            }
        )
    summary = pd.DataFrame(rows).sort_values("Mean_MASE").reset_index(drop=True)
    summary.insert(0, "Rank", np.arange(1, len(summary) + 1))
    return summary


def compare_models(
    fold_metrics: pd.DataFrame,
    errors: pd.DataFrame,
    pairs: Iterable[tuple[str, str]],
    n_boot: int = 5000,
    seed: int = 0,
    horizon: int = 12,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Paired comparisons for the given (model A, model B) pairs.

    Reports the pooled paired bootstrap on per-unit MASE (A - B; negative means A
    is better) and per-game Diebold-Mariano tests on scaled absolute errors,
    concatenated chronologically over folds.
    """
    units = seed_average_metrics(fold_metrics)
    mase = units.pivot_table(index=UNIT_COLS, columns="Model", values="MASE")
    scaled = seed_average_errors(errors)

    rows = []
    for a, b in pairs:
        if a not in mase.columns or b not in mase.columns or a == b:
            continue
        paired = mase[[a, b]].dropna()
        if paired.empty:
            continue
        boot = paired_bootstrap_diff(paired[a].values, paired[b].values, n_boot=n_boot, alpha=alpha, seed=seed)

        dm_p: list[float] = []
        a_better = 0
        tested = 0
        for game, g in scaled[scaled["Model"].isin([a, b])].groupby("Game"):
            # Chronological order: oldest fold (largest k) first.
            wide = (
                g.pivot_table(index=["Fold", "Horizon_Step"], columns="Model", values="Scaled_Abs_Error")
                .dropna()
                .reset_index()
                .sort_values(["Fold", "Horizon_Step"], ascending=[False, True])
            )
            if a not in wide.columns or b not in wide.columns or len(wide) < 2 * horizon + 1:
                continue
            res = diebold_mariano(wide[a].values, wide[b].values, h=horizon)
            if np.isnan(res["p_value"]):
                continue
            tested += 1
            dm_p.append(res["p_value"])
            if res["mean_diff"] < 0:
                a_better += 1

        rows.append(
            {
                "Model_A": a,
                "Model_B": b,
                "Units": int(len(paired)),
                "Mean_MASE_Diff": boot["mean_diff"],
                "CI_Low": boot["ci_low"],
                "CI_High": boot["ci_high"],
                "Boot_P": boot["p_value"],
                "Bootstrap_Significant": bool(boot["ci_low"] > 0 or boot["ci_high"] < 0),
                "DM_Games_Tested": tested,
                "DM_Games_A_Better": a_better,
                "DM_Games_Significant": int(sum(p < alpha for p in dm_p)),
                "DM_Median_P": float(np.median(dm_p)) if dm_p else float("nan"),
            }
        )
    return pd.DataFrame(rows)
