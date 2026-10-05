"""Execute Phase 4 machine learning and deep learning benchmarking.

Evaluates tabular models (Ridge, Random Forest, XGBoost) and PyTorch recurrent models
(RNN, LSTM, GRU) against the Phase 3 SARIMA econometric benchmark across all 7 games.
Calculates MAE, RMSE, MAPE, sMAPE, MASE on the strict chronological validation window
(2019-09-01 to 2020-08-01, H=12). Test partition remains held out and untouched.
Generates comprehensive benchmark reports and publication figures (figures 14, 15, 16).
"""

from __future__ import annotations

import ast
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
from steam_player_forecasting.models.statistical import SARIMAForecaster
from steam_player_forecasting.models.tabular import (
    RandomForestForecaster,
    RidgeForecaster,
    XGBoostForecaster,
    tune_tabular_forecaster,
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


def run_phase4_pipeline() -> dict[str, Any]:
    """Run full Phase 4 ML & DL evaluation pipeline."""
    root = get_project_root()
    report_dir = root / "report"
    figures_dir = root / "figures"
    report_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    games = get_available_games()
    print("=================================================================")
    print(f"Phase 4: Machine Learning & Deep Learning Benchmarks ({len(games)} games)")
    print("Strict Chronological Validation (H=12, Test partition held out)")
    print("Core Quartet: SARIMA vs XGBoost vs LSTM vs GRU")
    print("=================================================================\n")

    sarima_orders = load_sarima_orders(root)

    all_benchmark_records: list[dict[str, Any]] = []
    core_quartet_records: list[dict[str, Any]] = []

    # Store series and forecasts for plotting
    plot_data: dict[str, dict[str, Any]] = {}
    dl_loss_curves: dict[str, dict[str, list[float]]] = {}

    start_total = time.time()

    for game in games:
        print(f"--- Processing: {game} ---")
        df = load_processed_data(game_name=game)
        split_res = train_val_test_split(df, val_months=12, test_months=12)

        train_series = pd.Series(
            split_res.train["Avg_players"].values,
            index=pd.to_datetime(split_res.train["Month_Year"]),
            name=game,
        )
        val_series = pd.Series(
            split_res.val["Avg_players"].values,
            index=pd.to_datetime(split_res.val["Month_Year"]),
            name=game,
        )

        train_vals = train_series.values
        val_vals = val_series.values
        val_steps = len(val_series)

        forecasts: dict[str, np.ndarray] = {}

        # ----------------------------------------------------------------------
        # 1. SARIMA Baseline (Econometric Benchmark from Phase 3)
        # ----------------------------------------------------------------------
        if game in sarima_orders:
            order, sorder = sarima_orders[game]
        else:
            order, sorder = (1, 1, 1), (1, 1, 0, 12)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sarima_model = SARIMAForecaster(
                order=order,
                seasonal_order=sorder,
                enforce_stationarity=False,
                enforce_invertibility=False,
                maxiter=50,
            )
            sarima_model.fit(train_series)
            sarima_fc = np.asarray(sarima_model.predict(steps=val_steps), dtype=float).ravel()
            forecasts["SARIMA"] = sarima_fc

        sarima_metrics = calculate_metrics(val_vals, sarima_fc, y_train=train_vals, seasonal_period=12)
        print(f"  [SARIMA]   MAE: {sarima_metrics['MAE']:,.1f} | RMSE: {sarima_metrics['RMSE']:,.1f} | MAPE: {sarima_metrics['MAPE']:.2f}% | MASE: {sarima_metrics['MASE']:.3f}")

        # ----------------------------------------------------------------------
        # 2. Tabular ML: Ridge Regression
        # ----------------------------------------------------------------------
        ridge_model = RidgeForecaster(alpha=1.0)
        ridge_model.fit(train_series)
        ridge_fc = np.asarray(ridge_model.predict(steps=val_steps), dtype=float).ravel()
        forecasts["Ridge"] = ridge_fc
        ridge_metrics = calculate_metrics(val_vals, ridge_fc, y_train=train_vals, seasonal_period=12)
        print(f"  [Ridge]    MAE: {ridge_metrics['MAE']:,.1f} | RMSE: {ridge_metrics['RMSE']:,.1f} | MAPE: {ridge_metrics['MAPE']:.2f}% | MASE: {ridge_metrics['MASE']:.3f}")

        # ----------------------------------------------------------------------
        # 3. Tabular ML: Random Forest
        # ----------------------------------------------------------------------
        rf_model = RandomForestForecaster(n_estimators=100, max_depth=6, random_state=42)
        rf_model.fit(train_series)
        rf_fc = np.asarray(rf_model.predict(steps=val_steps), dtype=float).ravel()
        forecasts["Random Forest"] = rf_fc
        rf_metrics = calculate_metrics(val_vals, rf_fc, y_train=train_vals, seasonal_period=12)
        print(f"  [RF]       MAE: {rf_metrics['MAE']:,.1f} | RMSE: {rf_metrics['RMSE']:,.1f} | MAPE: {rf_metrics['MAPE']:.2f}% | MASE: {rf_metrics['MASE']:.3f}")

        # ----------------------------------------------------------------------
        # 4. Tabular ML: XGBoost (Tuned on validation split)
        # ----------------------------------------------------------------------
        xgb_grid = [
            {"n_estimators": 50, "max_depth": 2, "learning_rate": 0.05},
            {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.05},
            {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.03},
            {"n_estimators": 150, "max_depth": 3, "learning_rate": 0.02},
            {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05},
        ]
        best_xgb_p, _, xgb_model = tune_tabular_forecaster(
            model_type="xgboost",
            y_train=train_series,
            y_val=val_series,
            param_grid=xgb_grid,
            criterion="MAE",
        )
        xgb_fc = np.asarray(xgb_model.predict(steps=val_steps), dtype=float).ravel()
        forecasts["XGBoost"] = xgb_fc
        xgb_metrics = calculate_metrics(val_vals, xgb_fc, y_train=train_vals, seasonal_period=12)
        print(f"  [XGBoost]  MAE: {xgb_metrics['MAE']:,.1f} | RMSE: {xgb_metrics['RMSE']:,.1f} | MAPE: {xgb_metrics['MAPE']:.2f}% | MASE: {xgb_metrics['MASE']:.3f} (Best: {best_xgb_p})")

        # ----------------------------------------------------------------------
        # 5. Deep Learning: Standard RNN
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
        rnn_model.fit(train_series)
        rnn_fc = np.asarray(rnn_model.predict(steps=val_steps), dtype=float).ravel()
        forecasts["RNN"] = rnn_fc
        rnn_metrics = calculate_metrics(val_vals, rnn_fc, y_train=train_vals, seasonal_period=12)
        print(f"  [RNN]      MAE: {rnn_metrics['MAE']:,.1f} | RMSE: {rnn_metrics['RMSE']:,.1f} | MAPE: {rnn_metrics['MAPE']:.2f}% | MASE: {rnn_metrics['MASE']:.3f}")

        # ----------------------------------------------------------------------
        # 6. Deep Learning: LSTM
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
        lstm_model.fit(train_series)
        lstm_fc = np.asarray(lstm_model.predict(steps=val_steps), dtype=float).ravel()
        forecasts["LSTM"] = lstm_fc
        lstm_metrics = calculate_metrics(val_vals, lstm_fc, y_train=train_vals, seasonal_period=12)
        print(f"  [LSTM]     MAE: {lstm_metrics['MAE']:,.1f} | RMSE: {lstm_metrics['RMSE']:,.1f} | MAPE: {lstm_metrics['MAPE']:.2f}% | MASE: {lstm_metrics['MASE']:.3f}")

        # ----------------------------------------------------------------------
        # 7. Deep Learning: GRU
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
        gru_model.fit(train_series)
        gru_fc = np.asarray(gru_model.predict(steps=val_steps), dtype=float).ravel()
        forecasts["GRU"] = gru_fc
        gru_metrics = calculate_metrics(val_vals, gru_fc, y_train=train_vals, seasonal_period=12)
        print(f"  [GRU]      MAE: {gru_metrics['MAE']:,.1f} | RMSE: {gru_metrics['RMSE']:,.1f} | MAPE: {gru_metrics['MAPE']:.2f}% | MASE: {gru_metrics['MASE']:.3f}\n")

        # Record loss curves
        dl_loss_curves[game] = {
            "RNN": rnn_model.train_losses_,
            "LSTM": lstm_model.train_losses_,
            "GRU": gru_model.train_losses_,
        }

        # Store for plotting
        plot_data[game] = {
            "train": train_series,
            "val": val_series,
            "forecasts": forecasts,
        }

        # Collect metrics for all models
        models_metrics = [
            ("SARIMA", sarima_metrics, "Econometric"),
            ("Ridge", ridge_metrics, "Linear ML"),
            ("Random Forest", rf_metrics, "Tree Ensemble ML"),
            ("XGBoost", xgb_metrics, "Gradient Boosted ML"),
            ("RNN", rnn_metrics, "Recurrent DL"),
            ("LSTM", lstm_metrics, "Recurrent DL"),
            ("GRU", gru_metrics, "Recurrent DL"),
        ]

        for mod_name, m_dict, paradigm in models_metrics:
            rec = {
                "Game": game,
                "Model": mod_name,
                "Paradigm": paradigm,
                "MAE": round(m_dict["MAE"], 2),
                "RMSE": round(m_dict["RMSE"], 2),
                "MAPE": round(m_dict["MAPE"], 2),
                "sMAPE": round(m_dict["sMAPE"], 2),
                "MASE": round(m_dict["MASE"], 3),
            }
            all_benchmark_records.append(rec)
            if mod_name in ("SARIMA", "XGBoost", "LSTM", "GRU"):
                core_quartet_records.append(rec)

    total_time = time.time() - start_total
    print(f"Validation benchmarking completed across {len(games)} games in {total_time:.2f} seconds.\n")

    # Save DataFrames and Reports
    df_all = pd.DataFrame(all_benchmark_records)
    df_core = pd.DataFrame(core_quartet_records)

    df_all.to_csv(report_dir / "ml_dl_validation_benchmarks.csv", index=False)
    with open(report_dir / "ml_dl_validation_benchmarks.md", "w", encoding="utf-8") as f:
        f.write("# Phase 4 Machine Learning & Deep Learning Validation Benchmarks\n\n")
        f.write("Multi-step (H=12) validation error metrics across all 7 candidate Steam games.\n\n")
        f.write(df_to_markdown(df_all))
        f.write("\n")

    df_core.to_csv(report_dir / "core_quartet_comparison.csv", index=False)
    with open(report_dir / "core_quartet_comparison.md", "w", encoding="utf-8") as f:
        f.write("# Core Quartet Forecasting Benchmark Comparison\n\n")
        f.write("Validation comparison of the 4 flagship paradigms: **SARIMA vs XGBoost vs LSTM vs GRU**\n\n")
        f.write(df_to_markdown(df_core))
        f.write("\n\n## Per-Game Paradigm Winner Summary\n\n")

        # Summary of best model per game among the core quartet
        winner_records = []
        for game in games:
            sub = df_core[df_core["Game"] == game].sort_values("MAE")
            best_row = sub.iloc[0]
            winner_records.append({
                "Game": game,
                "Best_Model": best_row["Model"],
                "MAE": best_row["MAE"],
                "RMSE": best_row["RMSE"],
                "MAPE": best_row["MAPE"],
                "MASE": best_row["MASE"],
            })
        df_winners = pd.DataFrame(winner_records)
        f.write(df_to_markdown(df_winners))
        f.write("\n")

    print(f"Reports saved to {report_dir / 'ml_dl_validation_benchmarks.md'} and {report_dir / 'core_quartet_comparison.md'}")

    # ==========================================================================
    # Generate Figures (Figures 14, 15, 16)
    # ==========================================================================
    generate_figures(plot_data, df_core, dl_loss_curves, figures_dir)

    return {
        "all_benchmarks": df_all,
        "core_quartet": df_core,
    }


def generate_figures(
    plot_data: dict[str, dict[str, Any]],
    df_core: pd.DataFrame,
    dl_loss_curves: dict[str, dict[str, list[float]]],
    figures_dir: Path,
) -> None:
    """Generate high-resolution 300 DPI publication plots."""
    games = list(plot_data.keys())

    # --------------------------------------------------------------------------
    # Figure 14: Core Quartet Validation Forecast Trajectories (Multi-panel)
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(4, 2, figsize=(16, 18), sharex=False)
    axes_flat = axes.ravel()

    model_colors = {
        "SARIMA": "#1f77b4",     # Blue
        "XGBoost": "#ff7f0e",    # Orange
        "LSTM": "#2ca02c",       # Green
        "GRU": "#d62728",        # Red
    }

    for idx, game in enumerate(games):
        ax = axes_flat[idx]
        data = plot_data[game]
        train = data["train"]
        val = data["val"]
        fcs = data["forecasts"]

        # Show trailing 18 months of training history for context
        hist_context = train.iloc[-18:]
        ax.plot(hist_context.index, hist_context.values, label="Historical (Train)", color="#333333", linewidth=1.8)
        ax.plot(val.index, val.values, label="Ground Truth (Val)", color="#000000", linewidth=2.5, linestyle="--", marker="o", markersize=4)

        for mod_name in ["SARIMA", "XGBoost", "LSTM", "GRU"]:
            fc_vals = fcs[mod_name]
            ax.plot(val.index, fc_vals, label=mod_name, color=model_colors[mod_name], linewidth=2.0, alpha=0.9)

        ax.axvline(x=val.index[0], color="gray", linestyle=":", alpha=0.7)
        ax.set_title(f"{game}", fontsize=12, fontweight="bold", pad=8)
        ax.set_ylabel("Avg Players", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.4)
        if idx == 0:
            ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

    # Hide unused 8th subplot
    if len(games) < len(axes_flat):
        fig.delaxes(axes_flat[-1])

    fig.suptitle(
        "Figure 14: Validation Forecast Trajectories — Core Quartet (SARIMA vs XGBoost vs LSTM vs GRU)\n"
        "Strict Chronological Validation Window: September 2019 – August 2020 (H=12)",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    fig14_path = figures_dir / "14_ml_dl_validation_forecast_comparisons.png"
    plt.savefig(fig14_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated Figure 14: {fig14_path}")

    # --------------------------------------------------------------------------
    # Figure 15: Core Quartet Validation Metrics Comparison (Grouped Bar Chart)
    # --------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Metric 1: MAPE (%)
    pivot_mape = df_core.pivot(index="Game", columns="Model", values="MAPE")[["SARIMA", "XGBoost", "LSTM", "GRU"]]
    pivot_mape.plot(kind="bar", ax=ax1, color=[model_colors[m] for m in pivot_mape.columns], width=0.75)
    ax1.set_title("Mean Absolute Percentage Error (MAPE %)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("MAPE (%)", fontsize=11)
    ax1.set_xlabel("")
    ax1.set_xticklabels([g[:15] + "..." if len(g) > 15 else g for g in pivot_mape.index], rotation=30, ha="right")
    ax1.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax1.legend(title="Model", fontsize=9)

    # Metric 2: MASE (Mean Absolute Scaled Error)
    pivot_mase = df_core.pivot(index="Game", columns="Model", values="MASE")[["SARIMA", "XGBoost", "LSTM", "GRU"]]
    pivot_mase.plot(kind="bar", ax=ax2, color=[model_colors[m] for m in pivot_mase.columns], width=0.75)
    ax2.axhline(y=1.0, color="red", linestyle="--", linewidth=1.2, label="Naive Benchmark (MASE=1.0)")
    ax2.set_title("Mean Absolute Scaled Error (MASE)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("MASE", fontsize=11)
    ax2.set_xlabel("")
    ax2.set_xticklabels([g[:15] + "..." if len(g) > 15 else g for g in pivot_mase.index], rotation=30, ha="right")
    ax2.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax2.legend(title="Model", fontsize=9)

    fig.suptitle(
        "Figure 15: Out-of-Sample Validation Accuracy Comparison across 7 Steam Flagship Titles\n"
        "Lower values indicate superior predictive performance across multi-step 12-month horizon",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    fig15_path = figures_dir / "15_core_quartet_metrics_comparison.png"
    plt.savefig(fig15_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated Figure 15: {fig15_path}")

    # --------------------------------------------------------------------------
    # Figure 16: Deep Learning Recurrent Loss Progression & Early Stopping
    # --------------------------------------------------------------------------
    # Plot loss curves for 4 representative titles
    rep_games = [
        "Counter-Strike: Global Offensive",
        "Dota 2",
        "Rust",
        "Warframe",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes_flat = axes.ravel()

    dl_colors = {"RNN": "#9467bd", "LSTM": "#2ca02c", "GRU": "#d62728"}

    for idx, g in enumerate(rep_games):
        ax = axes_flat[idx]
        curves = dl_loss_curves.get(g, {})
        for arch in ["RNN", "LSTM", "GRU"]:
            losses = curves.get(arch, [])
            epochs = range(1, len(losses) + 1)
            ax.plot(epochs, losses, label=f"{arch} (stopped at ep {len(losses)})", color=dl_colors[arch], linewidth=2.0)

        ax.set_title(f"{g}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel("MSE Loss (MinMax space)", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=9, loc="upper right")

    fig.suptitle(
        "Figure 16: PyTorch Deep Learning Training Dynamics & Early Stopping\n"
        "Convergence and early stopping behavior across representative titles (Adam, ReduceLROnPlateau)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    fig16_path = figures_dir / "16_deep_learning_loss_curves.png"
    plt.savefig(fig16_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated Figure 16: {fig16_path}")


if __name__ == "__main__":
    run_phase4_pipeline()
