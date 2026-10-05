"""Execute Phase 5 final evaluation on the held-out test set (2020-09 to 2021-08).

Conducts unblinded evaluation of candidate baselines and the Core Quartet:
(SARIMA vs XGBoost vs LSTM vs GRU, plus Naive, SNaive, Holt-Winters, ARIMA, Ridge, RF, RNN)
across all 7 selected Steam titles.

Computes MAE, RMSE, MAPE, sMAPE, MASE, horizon-specific error degradation,
and exports publication benchmark tables and cached predictions for the interactive demo.
"""

from __future__ import annotations

import ast
import json
import time
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from steam_player_forecasting.data.loader import (
    get_available_games,
    get_project_root,
    load_processed_data,
)
from steam_player_forecasting.evaluation.metrics import calculate_metrics
from steam_player_forecasting.features.split import train_val_test_split
from steam_player_forecasting.models.deep_learning import (
    GRUForecaster,
    LSTMForecaster,
    RNNForecaster,
)
from steam_player_forecasting.models.statistical import (
    ARIMAForecaster,
    HoltWintersForecaster,
    NaiveForecaster,
    SARIMAForecaster,
    SeasonalNaiveForecaster,
)
from steam_player_forecasting.models.tabular import (
    RandomForestForecaster,
    RidgeForecaster,
    XGBoostForecaster,
)


def df_to_markdown(df: pd.DataFrame) -> str:
    """Format DataFrame as a clean Markdown table without external dependencies."""
    headers = [str(c) for c in df.columns]
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    data_lines = []
    for _, row in df.iterrows():
        row_vals = [str(v) for v in row.values]
        data_lines.append("| " + " | ".join(row_vals) + " |")
    return "\n".join([header_line, sep_line] + data_lines)


def load_sarima_orders(root: Path) -> dict[str, tuple[tuple[int, int, int], tuple[int, int, int, int]]]:
    """Load optimal SARIMA orders from Phase 3 report."""
    csv_path = root / "report" / "sarima_order_selection.csv"
    orders: dict[str, tuple[tuple[int, int, int], tuple[int, int, int, int]]] = {}
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            game = row["Game"]
            order = ast.literal_eval(str(row["Optimal_Order"]))
            sorder = ast.literal_eval(str(row["Optimal_Seasonal_Order"]))
            orders[game] = (order, sorder)
    return orders


# Tuned validation hyperparameters from Phase 4
OPTIMAL_XGB_PARAMS = {
    "Counter-Strike: Global Offensive": {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05},
    "Dota 2": {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05},
    "Grand Theft Auto V": {"n_estimators": 150, "max_depth": 3, "learning_rate": 0.02},
    "Rust": {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05},
    "Team Fortress 2": {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05},
    "Tom Clancy's Rainbow Six Siege": {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05},
    "Warframe": {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.05},
}


def run_phase5_evaluation() -> dict[str, Any]:
    """Execute complete Phase 5 test evaluation and benchmarking pipeline."""
    root = get_project_root()
    report_dir = root / "report"
    figures_dir = root / "figures"
    data_dir = root / "data" / "processed"
    report_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    games = get_available_games()
    print("=================================================================")
    print(f"Phase 5: Final Unblinded Test Evaluation ({len(games)} games)")
    print("Strict Holdout Horizon: 2020-09-01 to 2021-08-01 (H=12)")
    print("Core Quartet: SARIMA vs XGBoost vs LSTM vs GRU")
    print("=================================================================\n")

    sarima_orders = load_sarima_orders(root)

    all_test_records: list[dict[str, Any]] = []
    core_quartet_records: list[dict[str, Any]] = []
    horizon_records: list[dict[str, Any]] = []

    # Cache object to store all forecasts for plots and Streamlit demo
    demo_cache: dict[str, Any] = {}

    start_total = time.time()

    for game in games:
        print(f"--- Evaluating Game: {game} ---")
        df = load_processed_data(game_name=game)
        split_res = train_val_test_split(df, val_months=12, test_months=12)

        # Train partition
        train_df = split_res.train
        val_df = split_res.val
        test_df = split_res.test

        # For final test evaluation, training history encompasses all data before test
        train_val_df = pd.concat([train_df, val_df]).reset_index(drop=True)

        history_series = pd.Series(
            train_val_df["Avg_players"].values,
            index=pd.to_datetime(train_val_df["Month_Year"]),
            name=game,
        )
        test_series = pd.Series(
            test_df["Avg_players"].values,
            index=pd.to_datetime(test_df["Month_Year"]),
            name=game,
        )

        # Also keep validation-only history for comparison/demo app
        train_only_series = pd.Series(
            train_df["Avg_players"].values,
            index=pd.to_datetime(train_df["Month_Year"]),
            name=game,
        )
        val_series = pd.Series(
            val_df["Avg_players"].values,
            index=pd.to_datetime(val_df["Month_Year"]),
            name=game,
        )

        test_steps = len(test_series)
        assert test_steps == 12, f"Expected test horizon 12, got {test_steps}"

        test_vals = test_series.values
        history_vals = history_series.values

        # Container for game forecasts
        test_forecasts: dict[str, np.ndarray] = {}
        sarima_ci: tuple[np.ndarray, np.ndarray] | None = None

        # ----------------------------------------------------------------------
        # 1. Baselines: Naive & Seasonal Naive
        # ----------------------------------------------------------------------
        naive_model = NaiveForecaster()
        naive_model.fit(history_series)
        fc_naive = np.asarray(naive_model.predict(steps=test_steps), dtype=float).ravel()
        test_forecasts["Naive"] = fc_naive

        snaive_model = SeasonalNaiveForecaster(seasonal_period=12)
        snaive_model.fit(history_series)
        fc_snaive = np.asarray(snaive_model.predict(steps=test_steps), dtype=float).ravel()
        test_forecasts["Seasonal Naive"] = fc_snaive

        # ----------------------------------------------------------------------
        # 2. Econometric Baselines: Holt-Winters & ARIMA
        # ----------------------------------------------------------------------
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hw_model = HoltWintersForecaster(seasonal_periods=12, trend="add", seasonal="add")
            try:
                hw_model.fit(history_series)
                fc_hw = np.asarray(hw_model.predict(steps=test_steps), dtype=float).ravel()
            except Exception:
                # Damped fallback if optimization struggles
                hw_model = HoltWintersForecaster(seasonal_periods=12, trend="add", seasonal="add", damped_trend=True)
                hw_model.fit(history_series)
                fc_hw = np.asarray(hw_model.predict(steps=test_steps), dtype=float).ravel()
            test_forecasts["Holt-Winters"] = fc_hw

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            arima_model = ARIMAForecaster(order=(1, 1, 1))
            arima_model.fit(history_series)
            fc_arima = np.asarray(arima_model.predict(steps=test_steps), dtype=float).ravel()
            test_forecasts["ARIMA(1,1,1)"] = fc_arima

        # ----------------------------------------------------------------------
        # 3. Econometric Benchmark: Tuned SARIMA (Core Quartet #1)
        # ----------------------------------------------------------------------
        order, sorder = sarima_orders.get(game, ((1, 1, 1), (1, 1, 0, 12)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sarima_model = SARIMAForecaster(
                order=order,
                seasonal_order=sorder,
                enforce_stationarity=False,
                enforce_invertibility=False,
                maxiter=50,
            )
            sarima_model.fit(history_series)
            fc_sarima, ci = sarima_model.predict_conf_int(steps=test_steps, alpha=0.05)
            fc_sarima = np.asarray(fc_sarima, dtype=float).ravel()
            sarima_ci = (ci["lower"].values.ravel(), ci["upper"].values.ravel())
            test_forecasts["SARIMA"] = fc_sarima

        # ----------------------------------------------------------------------
        # 4. Tabular Baselines: Ridge & Random Forest
        # ----------------------------------------------------------------------
        ridge_model = RidgeForecaster(alpha=1.0)
        ridge_model.fit(history_series)
        fc_ridge = np.asarray(ridge_model.predict(steps=test_steps), dtype=float).ravel()
        test_forecasts["Ridge"] = fc_ridge

        rf_model = RandomForestForecaster(n_estimators=100, max_depth=6, random_state=42)
        rf_model.fit(history_series)
        fc_rf = np.asarray(rf_model.predict(steps=test_steps), dtype=float).ravel()
        test_forecasts["Random Forest"] = fc_rf

        # ----------------------------------------------------------------------
        # 5. Gradient Boosted ML: Tuned XGBoost (Core Quartet #2)
        # ----------------------------------------------------------------------
        xgb_params = OPTIMAL_XGB_PARAMS.get(game, {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.05})
        xgb_model = XGBoostForecaster(
            n_estimators=xgb_params["n_estimators"],
            max_depth=xgb_params["max_depth"],
            learning_rate=xgb_params["learning_rate"],
            random_state=42,
        )
        xgb_model.fit(history_series)
        fc_xgb = np.asarray(xgb_model.predict(steps=test_steps), dtype=float).ravel()
        test_forecasts["XGBoost"] = fc_xgb

        # ----------------------------------------------------------------------
        # 6. Deep Learning: RNN Baseline
        # ----------------------------------------------------------------------
        rnn_model = RNNForecaster(
            seq_length=12,
            horizon=12,
            hidden_dim=32,
            num_layers=1,
            epochs=60,
            learning_rate=0.01,
            patience=15,
            seed=42,
        )
        rnn_model.fit(history_series)
        fc_rnn = np.asarray(rnn_model.predict(steps=test_steps), dtype=float).ravel()
        test_forecasts["RNN"] = fc_rnn

        # ----------------------------------------------------------------------
        # 7. Deep Learning: LSTM (Core Quartet #3)
        # ----------------------------------------------------------------------
        lstm_model = LSTMForecaster(
            seq_length=12,
            horizon=12,
            hidden_dim=32,
            num_layers=1,
            epochs=60,
            learning_rate=0.01,
            patience=15,
            seed=42,
        )
        lstm_model.fit(history_series)
        fc_lstm = np.asarray(lstm_model.predict(steps=test_steps), dtype=float).ravel()
        test_forecasts["LSTM"] = fc_lstm

        # ----------------------------------------------------------------------
        # 8. Deep Learning: GRU (Core Quartet #4)
        # ----------------------------------------------------------------------
        gru_model = GRUForecaster(
            seq_length=12,
            horizon=12,
            hidden_dim=32,
            num_layers=1,
            epochs=60,
            learning_rate=0.01,
            patience=15,
            seed=42,
        )
        gru_model.fit(history_series)
        fc_gru = np.asarray(gru_model.predict(steps=test_steps), dtype=float).ravel()
        test_forecasts["GRU"] = fc_gru

        # ----------------------------------------------------------------------
        # Evaluate All Models
        # ----------------------------------------------------------------------
        model_meta = [
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

        for m_name, paradigm in model_meta:
            fc = test_forecasts[m_name]
            mets = calculate_metrics(test_vals, fc, y_train=history_vals, seasonal_period=12)

            record = {
                "Game": game,
                "Model": m_name,
                "Paradigm": paradigm,
                "MAE": round(mets["MAE"], 2),
                "RMSE": round(mets["RMSE"], 2),
                "MAPE": round(mets["MAPE"], 2),
                "sMAPE": round(mets["sMAPE"], 2),
                "MASE": round(mets["MASE"], 3),
            }
            all_test_records.append(record)

            if m_name in ["SARIMA", "XGBoost", "LSTM", "GRU"]:
                core_quartet_records.append(record)

            # Step-by-step error by horizon (h = 1..12)
            for h in range(1, test_steps + 1):
                step_actual = test_vals[h - 1]
                step_pred = fc[h - 1]
                abs_err = abs(step_actual - step_pred)
                pct_err = (abs_err / max(abs(step_actual), 1e-8)) * 100.0
                horizon_records.append({
                    "Game": game,
                    "Model": m_name,
                    "Horizon_Step": h,
                    "Month": str(test_series.index[h - 1].strftime("%Y-%m")),
                    "Absolute_Error": round(abs_err, 2),
                    "Percentage_Error": round(pct_err, 2),
                })

        # Print summary for game
        sar_m = [r for r in all_test_records if r["Game"] == game and r["Model"] == "SARIMA"][0]
        xgb_m = [r for r in all_test_records if r["Game"] == game and r["Model"] == "XGBoost"][0]
        lst_m = [r for r in all_test_records if r["Game"] == game and r["Model"] == "LSTM"][0]
        gru_m = [r for r in all_test_records if r["Game"] == game and r["Model"] == "GRU"][0]

        print(f"  SARIMA  -> MAE: {sar_m['MAE']:>9,.1f} | MAPE: {sar_m['MAPE']:>5.2f}% | MASE: {sar_m['MASE']:>5.3f}")
        print(f"  XGBoost -> MAE: {xgb_m['MAE']:>9,.1f} | MAPE: {xgb_m['MAPE']:>5.2f}% | MASE: {xgb_m['MASE']:>5.3f}")
        print(f"  LSTM    -> MAE: {lst_m['MAE']:>9,.1f} | MAPE: {lst_m['MAPE']:>5.2f}% | MASE: {lst_m['MASE']:>5.3f}")
        print(f"  GRU     -> MAE: {gru_m['MAE']:>9,.1f} | MAPE: {gru_m['MAPE']:>5.2f}% | MASE: {gru_m['MASE']:>5.3f}")

        # Store in demo cache
        demo_cache[game] = {
            "history_dates": [d.strftime("%Y-%m-%d") for d in history_series.index],
            "history_values": [float(v) for v in history_series.values],
            "test_dates": [d.strftime("%Y-%m-%d") for d in test_series.index],
            "test_actuals": [float(v) for v in test_series.values],
            "test_forecasts": {k: [float(x) for x in v] for k, v in test_forecasts.items()},
            "sarima_ci_lower": [float(x) for x in sarima_ci[0]] if sarima_ci else [],
            "sarima_ci_upper": [float(x) for x in sarima_ci[1]] if sarima_ci else [],
            "optimal_sarima_order": f"SARIMA{order}x{sorder}",
            "optimal_xgb_params": xgb_params,
        }

    # ==========================================================================
    # Export Tables
    # ==========================================================================
    test_df_all = pd.DataFrame(all_test_records)
    test_df_core = pd.DataFrame(core_quartet_records)
    horizon_df = pd.DataFrame(horizon_records)

    # 1. Full test benchmark
    test_df_all.to_csv(report_dir / "final_test_benchmarks.csv", index=False)
    with open(report_dir / "final_test_benchmarks.md", "w", encoding="utf-8") as f:
        f.write("# Phase 5 Final Test-Set Evaluation Benchmarks\n\n")
        f.write("Multi-step (H=12) holdout evaluation on the untouched test period (2020-09-01 to 2021-08-01).\n\n")
        f.write(df_to_markdown(test_df_all) + "\n")

    # 2. Core Quartet comparison
    test_df_core.to_csv(report_dir / "core_quartet_test_benchmarks.csv", index=False)
    with open(report_dir / "core_quartet_test_benchmarks.md", "w", encoding="utf-8") as f:
        f.write("# Core Quartet Final Test-Set Benchmarks\n\n")
        f.write("Evaluation of the 4 flagship paradigms: **SARIMA vs XGBoost vs LSTM vs GRU**.\n\n")
        f.write(df_to_markdown(test_df_core) + "\n\n")
        f.write("## Per-Game Paradigm Winner on Test Set\n\n")
        best_per_game = []
        for g in games:
            sub = test_df_core[test_df_core["Game"] == g]
            best_row = sub.loc[sub["MAE"].idxmin()]
            best_per_game.append({
                "Game": g,
                "Best_Model": best_row["Model"],
                "MAE": best_row["MAE"],
                "RMSE": best_row["RMSE"],
                "MAPE": best_row["MAPE"],
                "MASE": best_row["MASE"],
            })
        best_df = pd.DataFrame(best_per_game)
        f.write(df_to_markdown(best_df) + "\n")

    # 3. Overall Final Leaderboard across all models
    leaderboard_records = []
    for model_name, grp in test_df_all.groupby("Model"):
        leaderboard_records.append({
            "Model": model_name,
            "Paradigm": grp["Paradigm"].iloc[0],
            "Mean_MAE": round(grp["MAE"].mean(), 2),
            "Median_MAE": round(grp["MAE"].median(), 2),
            "Mean_RMSE": round(grp["RMSE"].mean(), 2),
            "Mean_MAPE": round(grp["MAPE"].mean(), 2),
            "Mean_sMAPE": round(grp["sMAPE"].mean(), 2),
            "Mean_MASE": round(grp["MASE"].mean(), 3),
            "Best_Game_Count": sum(
                test_df_all.loc[test_df_all[test_df_all["Game"] == g]["MAE"].idxmin()]["Model"] == model_name
                for g in games
            ),
        })
    leaderboard_df = pd.DataFrame(leaderboard_records).sort_values("Mean_MASE").reset_index(drop=True)
    leaderboard_df.index = leaderboard_df.index + 1
    leaderboard_df.to_csv(report_dir / "final_leaderboard.csv", index_label="Rank")
    with open(report_dir / "final_leaderboard.md", "w", encoding="utf-8") as f:
        f.write("# Final Forecasting Leaderboard (Test-Set Holdout)\n\n")
        f.write("Ranked by Mean MASE across all 7 evaluated Steam titles (2020-09-01 to 2021-08-01).\n\n")
        f.write(df_to_markdown(leaderboard_df.reset_index(names=["Rank"])) + "\n")

    # 4. Error by Horizon
    horizon_df.to_csv(report_dir / "error_by_horizon.csv", index=False)

    # 5. Save demo cache for Streamlit app
    cache_path = data_dir / "test_forecasts_cache.json"
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(demo_cache, f, indent=2)
    print(f"\nSaved test forecast cache to: {cache_path}")

    elapsed = time.time() - start_total
    print(f"\nPhase 5 Test Evaluation completed in {elapsed:.1f}s.")
    print("Exported:")
    print(f"  - {report_dir / 'final_test_benchmarks.csv'}")
    print(f"  - {report_dir / 'core_quartet_test_benchmarks.csv'}")
    print(f"  - {report_dir / 'final_leaderboard.csv'}")
    print(f"  - {report_dir / 'error_by_horizon.csv'}")

    return {
        "all_test_df": test_df_all,
        "core_quartet_df": test_df_core,
        "leaderboard_df": leaderboard_df,
        "horizon_df": horizon_df,
        "demo_cache": demo_cache,
    }


if __name__ == "__main__":
    run_phase5_evaluation()
