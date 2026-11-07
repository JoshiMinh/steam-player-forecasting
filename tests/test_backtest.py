"""Tests for rolling-origin backtesting, seed aggregation and significance tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.cli.backtest import resolve_seeds, run_backtest_for_series, write_reports
from src.evaluation.backtest import (
    compare_models,
    rolling_origin_folds,
    seed_average_metrics,
    summarize_models,
)
from src.evaluation.significance import (
    block_bootstrap_ci,
    diebold_mariano,
    paired_bootstrap_diff,
)
from src.features.split import train_val_test_split


def _series(n: int = 60, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2016-01-01", periods=n, freq="MS")
    values = 1000 + 200 * np.sin(2 * np.pi * np.arange(n) / 12) + rng.normal(0, 20, n)
    return pd.Series(values, index=idx, name="Toy")


# ---------------------------------------------------------------------------
# Fold generation
# ---------------------------------------------------------------------------
def test_folds_are_chronological_and_non_overlapping():
    s = _series(70)
    folds = rolling_origin_folds(s, n_folds=4, horizon=12, min_train=28)
    assert [f.k for f in folds] == [0, 1, 2]  # fold 3 would have only 22 training months
    for f in folds:
        assert len(f.test) == 12
        assert f.train.index.max() < f.test.index.min()
    for newer, older in zip(folds[:-1], folds[1:]):
        assert older.test.index.max() < newer.test.index.min()
        assert newer.train.index[0] == older.train.index[0]


def test_fold_zero_matches_phase5_holdout():
    s = _series(70)
    df = pd.DataFrame({"Month_Year": s.index, "Avg_players": s.values})
    split = train_val_test_split(df, val_months=12, test_months=12)
    fold0 = rolling_origin_folds(s)[0]
    assert list(fold0.test.index) == list(pd.to_datetime(split.test["Month_Year"]))
    np.testing.assert_allclose(fold0.test.values, split.test["Avg_players"].values)
    assert len(fold0.train) == len(split.train) + len(split.val)


def test_min_train_drops_folds_and_short_series_gives_none():
    s = _series(40)
    assert [f.k for f in rolling_origin_folds(s, n_folds=4, min_train=28)] == [0]
    assert rolling_origin_folds(_series(30), n_folds=4, min_train=28) == []


def test_invalid_fold_count():
    with pytest.raises(ValueError):
        rolling_origin_folds(_series(), n_folds=0)


def test_resolve_seeds():
    assert resolve_seeds(None, [42, 0, 1]) == [42, 0, 1]
    assert resolve_seeds(2, [42, 0, 1]) == [42, 0]
    extended = resolve_seeds(5, [42, 0])
    assert len(extended) == 5 and len(set(extended)) == 5


# ---------------------------------------------------------------------------
# Significance
# ---------------------------------------------------------------------------
def test_bootstrap_ci_contains_mean_and_is_deterministic():
    vals = np.array([0.5, 0.7, 0.6, 0.9, 0.4, 0.8])
    mean, lo, hi = block_bootstrap_ci(vals, n_boot=2000, seed=1)
    assert lo <= mean <= hi
    assert mean == pytest.approx(vals.mean())
    assert block_bootstrap_ci(vals, n_boot=2000, seed=1) == (mean, lo, hi)


def test_paired_bootstrap_detects_consistent_gap():
    a = np.full(12, 0.5) + np.random.default_rng(0).normal(0, 0.01, 12)
    b = a + 0.2
    res = paired_bootstrap_diff(a, b, n_boot=2000, seed=0)
    assert res["mean_diff"] < 0
    assert res["ci_high"] < 0
    assert res["p_value"] < 0.05


def test_paired_bootstrap_no_gap():
    rng = np.random.default_rng(3)
    a = rng.normal(0.6, 0.1, 20)
    b = a + rng.normal(0, 0.1, 20)
    res = paired_bootstrap_diff(a, b, n_boot=2000, seed=0)
    assert res["ci_low"] < 0 < res["ci_high"]


def test_dm_identical_losses_not_significant():
    loss = np.abs(np.random.default_rng(0).normal(1, 0.2, 48))
    res = diebold_mariano(loss, loss.copy(), h=12)
    assert res["p_value"] == pytest.approx(1.0)


def test_dm_detects_better_model():
    rng = np.random.default_rng(0)
    base = np.abs(rng.normal(1.0, 0.1, 96))
    better = base * 0.5 + rng.normal(0, 0.01, 96)
    res = diebold_mariano(better, base, h=12)
    assert res["mean_diff"] < 0
    assert res["p_value"] < 0.01


def test_dm_requires_matching_lengths():
    with pytest.raises(ValueError):
        diebold_mariano(np.ones(10), np.ones(11))


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def _toy_fold_metrics() -> pd.DataFrame:
    rows = []
    for game in ["A", "B"]:
        for fold in [0, 1]:
            rows.append(dict(Game=game, Fold=fold, Model="Stable", Paradigm="x", Seed="", MAE=1, RMSE=1, MAPE=1, sMAPE=1, MASE=1.0))
            for seed, mase in [(0, 0.5), (1, 1.5)]:
                rows.append(dict(Game=game, Fold=fold, Model="Noisy", Paradigm="y", Seed=seed, MAE=1, RMSE=1, MAPE=1, sMAPE=1, MASE=mase))
    return pd.DataFrame(rows)


def test_seed_average_collapses_to_one_unit_per_game_fold_model():
    units = seed_average_metrics(_toy_fold_metrics())
    assert len(units) == 8  # 2 games x 2 folds x 2 models
    noisy = units[units["Model"] == "Noisy"]
    assert noisy["MASE"].tolist() == [1.0] * 4


def test_summary_reports_seed_std_only_for_seeded_models():
    summary = summarize_models(_toy_fold_metrics(), n_boot=200)
    stable = summary[summary["Model"] == "Stable"].iloc[0]
    noisy = summary[summary["Model"] == "Noisy"].iloc[0]
    assert np.isnan(stable["Mean_Seed_Std"])
    assert noisy["Mean_Seed_Std"] == pytest.approx(np.std([0.5, 1.5], ddof=1))
    assert stable["Units"] == noisy["Units"] == 4
    assert list(summary["Rank"]) == [1, 2]


def test_diverged_forecast_is_capped_but_reported_raw():
    from src.evaluation.backtest import MASE_CAP

    fm = _toy_fold_metrics()
    mask = (fm["Model"] == "Stable") & (fm["Game"] == "A") & (fm["Fold"] == 0)
    fm.loc[mask, "MASE"] = 1e9
    summary = summarize_models(fm, n_boot=200).set_index("Model")
    stable = summary.loc["Stable"]
    assert stable["Diverged_Units"] == 1
    assert stable["Raw_Mean_MASE"] > 1e6
    assert stable["Mean_MASE"] == pytest.approx((3 * 1.0 + MASE_CAP) / 4)
    assert stable["Median_MASE"] == pytest.approx(1.0)
    assert summary.loc["Noisy", "Diverged_Units"] == 0


# ---------------------------------------------------------------------------
# Generated artifacts (skipped when the backtest has not been run)
# ---------------------------------------------------------------------------
def _report(name: str) -> pd.DataFrame:
    from src.data.loader import get_project_root

    path = get_project_root() / "assets" / "reports" / name
    if not path.exists():
        pytest.skip(f"{name} not generated; run `steam-forecast backtest`")
    return pd.read_csv(path)


def test_artifact_fold_metrics_consistent_with_phase5_holdout():
    bt = _report("backtest_fold_metrics.csv")
    ref = _report("final_test_benchmarks.csv")
    assert {"Game", "Fold", "Origin", "Model", "Seed", "MASE", "Params_Tuned_On_Test"} <= set(bt.columns)
    fold0 = bt[(bt["Fold"] == 0) & (bt["Seed"].isna() | (bt["Seed"] == 42))]
    merged = fold0.merge(ref, on=["Game", "Model"], suffixes=("_bt", "_ref"))
    assert len(merged) == len(ref)
    np.testing.assert_allclose(merged["MASE_bt"], merged["MASE_ref"], atol=1e-3)


def test_artifact_flags_tuned_params_on_later_folds():
    bt = _report("backtest_fold_metrics.csv")
    tuned = bt[bt["Model"].isin(["SARIMA", "XGBoost"])]
    assert (tuned["Params_Tuned_On_Test"] == (tuned["Fold"] >= 1)).all()
    assert not bt[~bt["Model"].isin(["SARIMA", "XGBoost"])]["Params_Tuned_On_Test"].any()


def test_artifact_summary_and_significance_schema():
    summary = _report("backtest_summary.csv")
    sig = _report("backtest_significance.csv")
    assert summary["Mean_MASE"].is_monotonic_increasing
    assert (summary["CI_Low"] <= summary["Mean_MASE"]).all()
    assert (summary["Mean_MASE"] <= summary["CI_High"]).all()
    assert summary.loc[summary["Model"].isin(["RNN", "LSTM", "GRU"]), "Mean_Seed_Std"].notna().all()
    assert {"Model_A", "Model_B", "Mean_MASE_Diff", "Boot_P", "DM_Games_Tested"} <= set(sig.columns)


# ---------------------------------------------------------------------------
# End-to-end on a synthetic series
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def tiny_run():
    series = _series(64)
    metrics, errors, failures = run_backtest_for_series(
        "Toy",
        series,
        sarima_orders={},
        n_folds=2,
        min_train=28,
        seeds=[42, 0],
        dl_epochs=2,
        models=["Naive", "Seasonal Naive", "Ridge", "LSTM"],
        verbose=False,
    )
    return pd.DataFrame(metrics), pd.DataFrame(errors), failures


def test_end_to_end_shapes(tiny_run):
    metrics, errors, failures = tiny_run
    assert failures == []
    # 2 folds x (3 deterministic + 2 LSTM seeds)
    assert len(metrics) == 2 * (3 + 2)
    assert len(errors) == len(metrics) * 12
    assert set(metrics["Fold"]) == {0, 1}
    assert {"MAE", "RMSE", "MAPE", "sMAPE", "MASE", "Params_Tuned_On_Test"} <= set(metrics.columns)
    assert metrics.loc[metrics["Model"] == "LSTM", "Seed"].tolist() == [42, 0, 42, 0]
    assert (errors["Scale"] > 0).all()


def test_end_to_end_reports_and_comparison(tiny_run, tmp_path):
    metrics, errors, _ = tiny_run
    summary, sig = write_reports(metrics, errors, tmp_path, n_boot=200)
    for name in [
        "backtest_fold_metrics.csv",
        "backtest_errors.csv",
        "backtest_summary.csv",
        "backtest_summary.md",
        "backtest_significance.csv",
    ]:
        assert (tmp_path / name).exists()
    assert set(summary["Model"]) == {"Naive", "Seasonal Naive", "Ridge", "LSTM"}
    assert summary["Mean_MASE"].is_monotonic_increasing
    assert not sig.empty

    direct = compare_models(metrics, errors, [("Seasonal Naive", "Naive")], n_boot=200)
    assert len(direct) == 1
    assert direct.iloc[0]["Units"] == 2
