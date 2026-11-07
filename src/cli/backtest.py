"""Rolling-origin backtest with multi-seed deep learning and significance tests.

Re-evaluates the 11 Phase 5 candidate models on several non-overlapping
12-month forecast origins per game, repeats the stochastic recurrent networks
(RNN, LSTM, GRU) over several seeds, and reports bootstrap confidence intervals
and paired significance tests.

Hyperparameters mirror ``src/cli/phase5.py`` exactly. SARIMA orders and XGBoost
parameters were tuned on the 2019-09 to 2020-08 validation window, so for
folds k >= 1 (whose test windows reach back into or before that window) those
two models are flagged with ``Params_Tuned_On_Test``.
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.cli.phase5 import OPTIMAL_XGB_PARAMS, df_to_markdown, load_sarima_orders
from src.data.loader import (
    get_available_games,
    get_project_root,
    load_config,
    load_processed_data,
)
from src.evaluation.backtest import (
    compare_models,
    rolling_origin_folds,
    summarize_models,
)
from src.evaluation.metrics import calculate_metrics
from src.models.deep_learning import GRUForecaster, LSTMForecaster, RNNForecaster
from src.models.statistical import (
    ARIMAForecaster,
    HoltWintersForecaster,
    NaiveForecaster,
    SARIMAForecaster,
    SeasonalNaiveForecaster,
)
from src.models.tabular import (
    RandomForestForecaster,
    RidgeForecaster,
    XGBoostForecaster,
)

HORIZON = 12
SEEDED_MODELS = ("RNN", "LSTM", "GRU")
TUNED_ON_VALIDATION = ("SARIMA", "XGBoost")
DEFAULT_SEEDS = [42, 0, 1, 2, 3]

# (model name, paradigm) in Phase 5 order. Paradigm labels match Phase 5.
MODEL_META: list[tuple[str, str]] = [
    ("Naive", "Naive Baseline"),
    ("Seasonal Naive", "Seasonal Baseline"),
    ("Holt-Winters", "Exponential Smoothing"),
    ("ARIMA(1,1,1)", "Econometric Baseline"),
    ("SARIMA", "Econometric"),
    ("Ridge", "Linear ML"),
    ("Random Forest", "Tree Ensemble ML"),
    ("XGBoost", "Gradient Boosted ML"),
    ("RNN", "Recurrent DL Baseline"),
    ("LSTM", "Recurrent DL"),
    ("GRU", "Recurrent DL"),
]

KEY_PAIRS: list[tuple[str, str]] = [
    ("GRU", "LSTM"),
    ("GRU", "SARIMA"),
    ("LSTM", "SARIMA"),
    ("XGBoost", "SARIMA"),
    ("Holt-Winters", "Ridge"),
    ("Holt-Winters", "SARIMA"),
    ("Ridge", "SARIMA"),
]


def _forecast(model: Any, history: pd.Series, steps: int) -> np.ndarray:
    model.fit(history)
    return np.asarray(model.predict(steps=steps), dtype=float).ravel()


def _recurrent(cls: type, seed: int, epochs: int) -> Any:
    return cls(
        seq_length=12,
        horizon=HORIZON,
        hidden_dim=32,
        num_layers=1,
        epochs=epochs,
        learning_rate=0.01,
        patience=15,
        seed=seed,
    )


def build_forecasters(
    game: str,
    sarima_orders: dict[str, Any],
    dl_epochs: int = 60,
) -> dict[str, Callable[[pd.Series, int, int], np.ndarray]]:
    """Return ``name -> f(history, steps, seed) -> forecast`` for every model.

    Mirrors the Phase 5 configuration. ``seed`` only affects RNN, LSTM and GRU.
    """
    order, sorder = sarima_orders.get(game, ((1, 1, 1), (1, 1, 0, 12)))
    xgb = OPTIMAL_XGB_PARAMS.get(game, {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.05})

    def holt_winters(history: pd.Series, steps: int, seed: int) -> np.ndarray:
        try:
            return _forecast(HoltWintersForecaster(seasonal_periods=12, trend="add", seasonal="add"), history, steps)
        except Exception:
            return _forecast(
                HoltWintersForecaster(seasonal_periods=12, trend="add", seasonal="add", damped_trend=True),
                history,
                steps,
            )

    return {
        "Naive": lambda h, s, seed: _forecast(NaiveForecaster(), h, s),
        "Seasonal Naive": lambda h, s, seed: _forecast(SeasonalNaiveForecaster(seasonal_period=12), h, s),
        "Holt-Winters": holt_winters,
        "ARIMA(1,1,1)": lambda h, s, seed: _forecast(ARIMAForecaster(order=(1, 1, 1)), h, s),
        "SARIMA": lambda h, s, seed: _forecast(
            SARIMAForecaster(
                order=order,
                seasonal_order=sorder,
                enforce_stationarity=False,
                enforce_invertibility=False,
                maxiter=50,
            ),
            h,
            s,
        ),
        "Ridge": lambda h, s, seed: _forecast(RidgeForecaster(alpha=1.0), h, s),
        "Random Forest": lambda h, s, seed: _forecast(
            RandomForestForecaster(n_estimators=100, max_depth=6, random_state=42), h, s
        ),
        "XGBoost": lambda h, s, seed: _forecast(
            XGBoostForecaster(
                n_estimators=xgb["n_estimators"],
                max_depth=xgb["max_depth"],
                learning_rate=xgb["learning_rate"],
                random_state=42,
            ),
            h,
            s,
        ),
        "RNN": lambda h, s, seed: _forecast(_recurrent(RNNForecaster, seed, dl_epochs), h, s),
        "LSTM": lambda h, s, seed: _forecast(_recurrent(LSTMForecaster, seed, dl_epochs), h, s),
        "GRU": lambda h, s, seed: _forecast(_recurrent(GRUForecaster, seed, dl_epochs), h, s),
    }


def resolve_seeds(n_seeds: int | None, configured: list[int]) -> list[int]:
    """First ``n_seeds`` seeds, extending the configured list if more are asked for."""
    seeds = list(configured)
    if n_seeds is None:
        return seeds
    nxt = 0
    while len(seeds) < n_seeds:
        if nxt not in seeds:
            seeds.append(nxt)
        nxt += 1
    return seeds[:n_seeds]


def run_backtest_for_series(
    game: str,
    series: pd.Series,
    sarima_orders: dict[str, Any],
    n_folds: int,
    min_train: int,
    seeds: list[int],
    dl_epochs: int = 60,
    models: list[str] | None = None,
    verbose: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Run all folds, models and seeds for one series.

    Returns:
        (metric records, per-step error records, failure records)
    """
    forecasters = build_forecasters(game, sarima_orders, dl_epochs=dl_epochs)
    paradigms = dict(MODEL_META)
    names = [m for m, _ in MODEL_META if models is None or m in models]
    folds = rolling_origin_folds(series, n_folds=n_folds, horizon=HORIZON, min_train=min_train)

    metric_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for fold in folds:
        history_vals = fold.train.values
        test_vals = fold.test.values
        scale = float(np.mean(np.abs(history_vals[12:] - history_vals[:-12])))
        scale = max(scale, 1e-8)
        if verbose:
            print(f"  fold {fold.k} | origin {fold.origin} | train={len(fold.train)} | test={fold.test.index[0]:%Y-%m}..{fold.test.index[-1]:%Y-%m}")

        for name in names:
            model_seeds = seeds if name in SEEDED_MODELS else [seeds[0]]
            for seed in model_seeds:
                t0 = time.time()
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        fc = forecasters[name](fold.train, HORIZON, seed)
                    if fc.shape != (HORIZON,) or not np.all(np.isfinite(fc)):
                        raise ValueError(f"invalid forecast shape/values: {fc.shape}")
                except Exception as exc:  # noqa: BLE001 - recorded and reported, never silent
                    failures.append(
                        {"Game": game, "Fold": fold.k, "Model": name, "Seed": seed, "Error": f"{type(exc).__name__}: {exc}"}
                    )
                    print(f"    ! FAILED {name} (seed {seed}): {type(exc).__name__}: {exc}")
                    continue

                mets = calculate_metrics(test_vals, fc, y_train=history_vals, seasonal_period=12)
                metric_rows.append(
                    {
                        "Game": game,
                        "Fold": fold.k,
                        "Origin": fold.origin,
                        "Model": name,
                        "Paradigm": paradigms[name],
                        "Seed": seed if name in SEEDED_MODELS else "",
                        "MAE": mets["MAE"],
                        "RMSE": mets["RMSE"],
                        "MAPE": mets["MAPE"],
                        "sMAPE": mets["sMAPE"],
                        "MASE": mets["MASE"],
                        "Params_Tuned_On_Test": bool(name in TUNED_ON_VALIDATION and fold.k >= 1),
                        "Fit_Seconds": round(time.time() - t0, 2),
                    }
                )
                abs_err = np.abs(test_vals - fc)
                for step in range(HORIZON):
                    error_rows.append(
                        {
                            "Game": game,
                            "Fold": fold.k,
                            "Model": name,
                            "Seed": seed if name in SEEDED_MODELS else "",
                            "Horizon_Step": step + 1,
                            "Abs_Error": float(abs_err[step]),
                            "Scale": scale,
                        }
                    )
    return metric_rows, error_rows, failures


def _fmt(df: pd.DataFrame, cols_3: list[str], cols_4: list[str] | None = None) -> pd.DataFrame:
    out = df.copy()
    for c in cols_3:
        out[c] = out[c].map(lambda v: "" if pd.isna(v) else f"{v:.3f}")
    for c in cols_4 or []:
        out[c] = out[c].map(lambda v: "" if pd.isna(v) else f"{v:.4f}")
    return out


def write_reports(
    fold_metrics: pd.DataFrame,
    errors: pd.DataFrame,
    out_dir: Path,
    n_boot: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write fold metrics, errors, summary and significance tables."""
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = summarize_models(fold_metrics, n_boot=n_boot)
    reference = summary.iloc[0]["Model"]
    others = [m for m in summary["Model"] if m != reference]
    pairs = [(reference, m) for m in others]
    pairs += [p for p in KEY_PAIRS if p not in pairs and (p[1], p[0]) not in pairs]
    sig = compare_models(fold_metrics, errors, pairs, n_boot=n_boot)
    if not sig.empty:
        sig["Involves_Tuned_Model"] = sig.apply(
            lambda r: r["Model_A"] in TUNED_ON_VALIDATION or r["Model_B"] in TUNED_ON_VALIDATION, axis=1
        )

    fold_metrics.to_csv(out_dir / "backtest_fold_metrics.csv", index=False)
    errors.to_csv(out_dir / "backtest_errors.csv", index=False)
    summary.to_csv(out_dir / "backtest_summary.csv", index=False)
    sig.to_csv(out_dir / "backtest_significance.csv", index=False)

    summary_md = _fmt(
        summary,
        [
            "Mean_MASE", "CI_Low", "CI_High", "Std_MASE", "Median_MASE",
            "Raw_Mean_MASE", "Fold0_Mean_MASE", "Mean_Seed_Std", "Mean_Rank",
        ],
    )
    (out_dir / "backtest_summary.md").write_text(df_to_markdown(summary_md) + "\n", encoding="utf-8")

    if not sig.empty:
        sig_md = _fmt(sig, ["Mean_MASE_Diff", "CI_Low", "CI_High", "Boot_P", "DM_Median_P"])
        (out_dir / "backtest_significance.md").write_text(df_to_markdown(sig_md) + "\n", encoding="utf-8")
    return summary, sig


def run_backtest(
    games: list[str] | None = None,
    n_folds: int | None = None,
    n_seeds: int | None = None,
    output_dir: str | Path | None = None,
    dl_epochs: int = 60,
) -> dict[str, Any]:
    """Execute the full rolling-origin backtest and write reports."""
    root = get_project_root()
    cfg = load_config().get("evaluation", {}).get("backtest", {})
    n_folds = n_folds or int(cfg.get("n_folds", 4))
    min_train = int(cfg.get("min_train_months", 28))
    n_boot = int(cfg.get("n_boot", 5000))
    seeds = resolve_seeds(n_seeds, [int(s) for s in cfg.get("seeds", DEFAULT_SEEDS)])
    out_dir = Path(output_dir) if output_dir else root / "assets" / "reports"

    available = get_available_games()
    selected = games or available
    unknown = [g for g in selected if g not in available]
    if unknown:
        raise ValueError(f"Unknown games: {unknown}. Available: {available}")

    print("=================================================================")
    print(f"Rolling-origin backtest | games={len(selected)} | max folds={n_folds} | seeds={seeds}")
    print(f"Min train months={min_train} | horizon={HORIZON} | output={out_dir}")
    print("=================================================================")

    sarima_orders = load_sarima_orders(root)
    all_metrics: list[dict[str, Any]] = []
    all_errors: list[dict[str, Any]] = []
    all_failures: list[dict[str, Any]] = []
    start = time.time()

    for game in selected:
        print(f"--- {game} ---")
        df = load_processed_data(game_name=game).sort_values("Month_Year")
        series = pd.Series(df["Avg_players"].values, index=pd.to_datetime(df["Month_Year"]), name=game)
        m, e, f = run_backtest_for_series(
            game, series, sarima_orders, n_folds, min_train, seeds, dl_epochs=dl_epochs
        )
        all_metrics += m
        all_errors += e
        all_failures += f

    fold_metrics = pd.DataFrame(all_metrics)
    errors = pd.DataFrame(all_errors)
    summary, sig = write_reports(fold_metrics, errors, out_dir, n_boot)

    if all_failures:
        pd.DataFrame(all_failures).to_csv(out_dir / "backtest_failures.csv", index=False)
        print(f"\nWARNING: {len(all_failures)} model fits failed (see backtest_failures.csv)")

    n_units = fold_metrics[["Game", "Fold"]].drop_duplicates().shape[0]
    print(f"\nBacktest finished in {time.time() - start:.0f}s | {n_units} game-fold units | {len(fold_metrics)} fits")
    print(summary[["Rank", "Model", "Mean_MASE", "CI_Low", "CI_High", "Fold0_Mean_MASE", "Mean_Seed_Std"]].to_string(index=False))
    return {"summary": summary, "significance": sig, "failures": all_failures}


if __name__ == "__main__":
    run_backtest()
