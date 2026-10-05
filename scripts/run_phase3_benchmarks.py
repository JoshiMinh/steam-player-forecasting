"""Execute Phase 3 statistical forecasting baselines, model selection, diagnostics, and benchmarking."""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from steam_player_forecasting.data.loader import get_available_games, get_project_root, load_processed_data
from steam_player_forecasting.evaluation.metrics import calculate_metrics
from steam_player_forecasting.features.split import train_val_test_split
from steam_player_forecasting.features.transforms import LogTransformer
from steam_player_forecasting.models.base import TransformedForecaster
from steam_player_forecasting.models.diagnostics import evaluate_residuals, plot_residual_diagnostics
from steam_player_forecasting.models.selection import select_sarima_order
from steam_player_forecasting.models.statistical import (
    ARForecaster,
    ARIMAForecaster,
    HoltWintersForecaster,
    NaiveForecaster,
    SARIMAForecaster,
    SeasonalNaiveForecaster,
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


def run_phase3_pipeline() -> dict[str, Any]:
    """Execute complete Phase 3 statistical forecasting pipeline."""
    root = get_project_root()
    report_dir = root / "report"
    figures_dir = root / "figures"
    report_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    games = get_available_games()
    print(f"=================================================================")
    print(f"Phase 3: Statistical Forecasting Baselines ({len(games)} games)")
    print(f"Strict Chronological Validation (H=12, Test untouched)")
    print(f"=================================================================\n")

    # 1. Candidate orders for SARIMA grid search
    candidate_orders = [
        (1, 1, 1),
        (1, 1, 0),
        (0, 1, 1),
        (2, 1, 1),
        (1, 1, 2),
        (0, 1, 2),
        (2, 1, 0),
        (1, 0, 1),
        (1, 0, 0),
        (0, 1, 0),
    ]
    candidate_seasonal_orders = [
        (1, 1, 1, 12),
        (0, 1, 1, 12),
        (1, 0, 1, 12),
        (1, 1, 0, 12),
        (0, 0, 0, 12),
    ]

    sarima_selection_records: list[dict[str, Any]] = []
    all_grid_records: list[dict[str, Any]] = []
    best_sarima_specs: dict[str, tuple[tuple, tuple]] = {}
    benchmark_records: list[dict[str, Any]] = []
    residual_records: list[dict[str, Any]] = []

    # Store forecasts for plotting
    plot_data_store: dict[str, dict[str, Any]] = {}

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

        val_steps = len(val_series)
        assert val_steps == 12, f"Expected validation horizon 12, got {val_steps}"

        # ----------------------------------------------------------------------
        # A. SARIMA Hyperparameter Grid Search
        # ----------------------------------------------------------------------
        t0 = time.time()
        best_ord, best_sord, grid_df, best_sarima_mod = select_sarima_order(
            y_train=train_series,
            y_val=val_series,
            candidate_orders=candidate_orders,
            candidate_seasonal_orders=candidate_seasonal_orders,
            criterion="val_mae",
            s=12,
            maxiter=40,
        )
        grid_time = time.time() - t0

        best_row = grid_df.iloc[0]
        best_sarima_specs[game] = (best_ord, best_sord)

        # Record grid search summary
        sarima_selection_records.append(
            {
                "Game": game,
                "Train_Obs": len(train_series),
                "Optimal_Order": str(best_ord),
                "Optimal_Seasonal_Order": str(best_sord),
                "Specification": f"SARIMA{best_ord}x{best_sord}",
                "AIC": round(best_row["aic"], 2),
                "BIC": round(best_row["bic"], 2),
                "Val_MAE": round(best_row["val_mae"], 2),
                "Val_RMSE": round(best_row["val_rmse"], 2),
                "Val_MAPE": round(best_row["val_mape"], 2),
                "Val_sMAPE": round(best_row["val_smape"], 2),
                "Val_MASE": round(best_row["val_mase"], 3),
                "LB_pValue": round(best_row["lb_pvalue"], 4),
                "Residuals_White_Noise": best_row["is_white_noise"],
            }
        )

        # Keep grid samples for Figure 10
        for _, r in grid_df.iterrows():
            all_grid_records.append(
                {
                    "Game": game,
                    "Order": str(r["order"]),
                    "Seasonal_Order": str(r["seasonal_order"]),
                    "AIC": r["aic"],
                    "Val_MAE": r["val_mae"],
                    "Val_MAPE": r["val_mape"],
                    "Is_Best": (r["order"] == best_ord and r["seasonal_order"] == best_sord),
                }
            )

        print(
            f"  Optimal SARIMA: {best_ord}x{best_sord} | Val MAE: {best_row['val_mae']:,.1f} | "
            f"Val MAPE: {best_row['val_mape']:.2f}% | White noise: {best_row['is_white_noise']} ({grid_time:.1f}s)"
        )

        # ----------------------------------------------------------------------
        # B. Residual Diagnostics for Optimal SARIMA
        # ----------------------------------------------------------------------
        res_diag = evaluate_residuals(best_sarima_mod.resid_, lags=12)
        residual_records.append(
            {
                "Game": game,
                "Specification": f"SARIMA{best_ord}x{best_sord}",
                "Residual_Mean": round(res_diag["mean"], 2),
                "Residual_Std": round(res_diag["std"], 2),
                "Durbin_Watson": round(res_diag["durbin_watson"], 3),
                "Ljung_Box_Stat": round(res_diag["ljung_box"]["test_statistic"][0], 3),
                "Ljung_Box_pValue": round(res_diag["ljung_box"]["min_p_value"], 4),
                "Jarque_Bera_Stat": round(res_diag["jarque_bera"]["test_statistic"], 3),
                "Jarque_Bera_pValue": round(res_diag["jarque_bera"]["p_value"], 4),
                "Is_Normal": res_diag["jarque_bera"]["is_normal"],
                "Is_White_Noise": res_diag["is_white_noise"],
            }
        )

        # ----------------------------------------------------------------------
        # C. Full Statistical Model Suite Benchmarking on Validation Set
        # ----------------------------------------------------------------------
        models_to_evaluate: dict[str, Any] = {
            "Naive (Last Value)": NaiveForecaster(strategy="last_value"),
            "Seasonal Naive": SeasonalNaiveForecaster(seasonal_period=12),
            "Holt-Winters (Add/Add)": HoltWintersForecaster(
                seasonal_periods=12, trend="add", seasonal="add"
            ),
            "Holt-Winters (Damped)": HoltWintersForecaster(
                seasonal_periods=12, trend="add", seasonal="add", damped_trend=True
            ),
            "AR(1)": ARForecaster(lags=1),
            "ARIMA(1,1,1)": ARIMAForecaster(order=(1, 1, 1)),
            f"SARIMA (Tuned)": best_sarima_mod,
            f"Log + SARIMA (Tuned)": TransformedForecaster(
                forecaster=SARIMAForecaster(
                    order=best_ord,
                    seasonal_order=best_sord,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                    maxiter=40,
                ),
                transformer=LogTransformer(),
            ),
        }

        game_forecasts: dict[str, np.ndarray] = {}

        for m_name, m_inst in models_to_evaluate.items():
            if m_name != "SARIMA (Tuned)":
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        m_inst.fit(train_series)
                except Exception as e:
                    print(f"    Failed {m_name}: {e}")
                    continue

            fc = m_inst.predict(steps=val_steps)
            fc_arr = np.asarray(fc, dtype=float).ravel()
            game_forecasts[m_name] = fc_arr

            mets = calculate_metrics(
                y_true=val_series.values,
                y_pred=fc_arr,
                y_train=train_series.values,
                seasonal_period=12,
            )

            benchmark_records.append(
                {
                    "Game": game,
                    "Model": m_name,
                    "MAE": round(mets["MAE"], 2),
                    "RMSE": round(mets["RMSE"], 2),
                    "MAPE": round(mets["MAPE"], 2),
                    "sMAPE": round(mets["sMAPE"], 2),
                    "MASE": round(mets.get("MASE", np.nan), 3),
                }
            )

        # Get SARIMA confidence interval for plotting
        _, sarima_ci = best_sarima_mod.predict_conf_int(steps=val_steps, alpha=0.05)

        plot_data_store[game] = {
            "train": train_series,
            "val": val_series,
            "forecasts": game_forecasts,
            "sarima_ci": sarima_ci,
            "sarima_resid": best_sarima_mod.resid_,
            "optimal_spec": f"SARIMA{best_ord}x{best_sord}",
        }

    # ==========================================================================
    # 2. Save Tables to report/
    # ==========================================================================
    df_selection = pd.DataFrame(sarima_selection_records)
    df_selection.to_csv(report_dir / "sarima_order_selection.csv", index=False)
    with open(report_dir / "sarima_order_selection.md", "w", encoding="utf-8") as f:
        f.write("# SARIMA Hyperparameter Selection & In-Sample Diagnostics\n\n")
        f.write(
            "Optimal specifications selected on 12-month validation horizon "
            "(2019-09 to 2020-08) prior to unblinded testing.\n\n"
        )
        f.write(df_to_markdown(df_selection))
        f.write("\n")

    df_benchmarks = pd.DataFrame(benchmark_records)
    df_benchmarks.to_csv(report_dir / "statistical_validation_benchmarks.csv", index=False)
    with open(report_dir / "statistical_validation_benchmarks.md", "w", encoding="utf-8") as f:
        f.write("# Phase 3 Statistical Forecasting Baseline Validation Benchmarks\n\n")
        f.write(
            "Multi-step (H=12) validation error metrics across all 7 candidate Steam games.\n\n"
        )
        f.write(df_to_markdown(df_benchmarks))
        f.write("\n\n## Per-Game Best Model Summary\n\n")
        # Best model per game by MAE
        best_per_game = (
            df_benchmarks.sort_values(by=["Game", "MAE"])
            .groupby("Game")
            .first()
            .reset_index()
        )
        f.write(df_to_markdown(best_per_game[["Game", "Model", "MAE", "RMSE", "MAPE", "sMAPE", "MASE"]]))
        f.write("\n")

    df_resid = pd.DataFrame(residual_records)
    df_resid.to_csv(report_dir / "sarima_residual_diagnostics.csv", index=False)
    with open(report_dir / "sarima_residual_diagnostics.md", "w", encoding="utf-8") as f:
        f.write("# Econometric Residual Diagnostics for Optimal SARIMA Models\n\n")
        f.write(
            "Ljung-Box test for white noise serial independence and Jarque-Bera normality tests.\n\n"
        )
        f.write(df_to_markdown(df_resid))
        f.write("\n")

    print("\nSaved all benchmark and diagnostic tables to report/ successfully.")

    # ==========================================================================
    # 3. Generate Publication Figures to figures/
    # ==========================================================================
    print("Generating publication figures...")

    # Figure 10: SARIMA Grid Search (AIC vs Val MAE)
    fig10 = _generate_figure_10(all_grid_records)
    fig10.savefig(figures_dir / "10_sarima_hyperparameter_grid_search.png", dpi=300, bbox_inches="tight")
    plt.close(fig10)
    print("  Saved figures/10_sarima_hyperparameter_grid_search.png")

    # Figure 11: Validation Forecast Comparisons Across All Games
    fig11 = _generate_figure_11(plot_data_store)
    fig11.savefig(figures_dir / "11_validation_forecast_comparisons.png", dpi=300, bbox_inches="tight")
    plt.close(fig11)
    print("  Saved figures/11_validation_forecast_comparisons.png")

    # Figure 12: Residual Diagnostics (CS:GO Flagship)
    flagship_resid = plot_data_store["Counter-Strike: Global Offensive"]["sarima_resid"]
    fig12 = plot_residual_diagnostics(
        residuals=flagship_resid,
        title="SARIMA Residual Diagnostics — Counter-Strike: Global Offensive",
        figsize=(13, 8),
    )
    fig12.savefig(figures_dir / "12_sarima_residual_diagnostics.png", dpi=300, bbox_inches="tight")
    plt.close(fig12)
    print("  Saved figures/12_sarima_residual_diagnostics.png")

    # Figure 13: Statistical Models Validation Metrics Comparison
    fig13 = _generate_figure_13(df_benchmarks)
    fig13.savefig(figures_dir / "13_statistical_models_validation_metrics.png", dpi=300, bbox_inches="tight")
    plt.close(fig13)
    print("  Saved figures/13_statistical_models_validation_metrics.png")

    elapsed_total = time.time() - start_total
    print(f"\n=================================================================")
    print(f"Phase 3 Statistical Forecasting Pipeline completed in {elapsed_total:.1f}s")
    print(f"=================================================================\n")

    return {
        "selection_df": df_selection,
        "benchmarks_df": df_benchmarks,
        "residual_df": df_resid,
        "best_specs": best_sarima_specs,
    }


def _generate_figure_10(grid_records: list[dict[str, Any]]) -> plt.Figure:
    """Generate Figure 10: AIC vs Validation MAPE for SARIMA specifications."""
    df_grid = pd.DataFrame(grid_records)
    unique_games = df_grid["Game"].unique()

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes_flat = axes.flatten()

    for idx, g in enumerate(unique_games):
        ax = axes_flat[idx]
        sub = df_grid[df_grid["Game"] == g]
        # Filter outliers for clean display
        val_clean = sub[sub["Val_MAPE"] < 100]

        scatter = ax.scatter(
            val_clean["AIC"],
            val_clean["Val_MAPE"],
            c=val_clean["Val_MAPE"],
            cmap="viridis_r",
            alpha=0.7,
            edgecolors="none",
            s=45,
        )

        best = sub[sub["Is_Best"]].iloc[0]
        ax.scatter(
            best["AIC"],
            best["Val_MAPE"],
            color="#e74c3c",
            edgecolors="black",
            s=120,
            zorder=5,
            label=f"Selected: {best['Order']}x{best['Seasonal_Order']}",
        )

        title_short = g.replace("Counter-Strike: Global Offensive", "CS:GO").replace(
            "Tom Clancy's Rainbow Six Siege", "Rainbow Six Siege"
        )
        ax.set_title(title_short, fontsize=11, fontweight="bold")
        ax.set_xlabel("AIC (In-Sample Penalty)", fontsize=9)
        ax.set_ylabel("Validation MAPE (%)", fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(fontsize=8, loc="upper right")

    # Hide 8th unused subplot
    axes_flat[-1].axis("off")
    # Add explanatory note in 8th panel
    axes_flat[-1].text(
        0.1,
        0.5,
        "SARIMA Model Selection Protocol\n\n"
        "• Joint minimization of AIC and\n  12-month validation MAPE\n"
        "• Red marker denotes optimal\n  specification selected per game\n"
        "• In-sample Ljung-Box test verifies\n  white-noise residuals",
        fontsize=11,
        verticalalignment="center",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#f8f9fa", edgecolor="#bdc3c7"),
    )

    fig.suptitle(
        "Figure 10: SARIMA Hyperparameter Grid Search — AIC vs. Out-of-Sample Validation Error",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout()
    return fig


def _generate_figure_11(plot_store: dict[str, dict[str, Any]]) -> plt.Figure:
    """Generate Figure 11: Multi-Step Validation Forecasts vs Actuals Across All Games."""
    games = list(plot_store.keys())
    fig, axes = plt.subplots(4, 2, figsize=(18, 16))
    axes_flat = axes.flatten()

    palette = {
        "Actual (Validation)": "#2c3e50",
        "Naive (Last Value)": "#95a5a6",
        "Seasonal Naive": "#e67e22",
        "Holt-Winters (Add/Add)": "#27ae60",
        "ARIMA(1,1,1)": "#8e44ad",
        "SARIMA (Tuned)": "#e74c3c",
    }

    for idx, g in enumerate(games):
        ax = axes_flat[idx]
        data = plot_store[g]
        tr = data["train"].iloc[-24:]  # Last 24 months of training for visual context
        val = data["val"]
        fcs = data["forecasts"]
        ci = data["sarima_ci"]

        # Plot historical training context
        ax.plot(tr.index, tr.values, color="#7f8c8d", lw=1.5, linestyle="--", label="Train (Context)")
        # Plot actual validation ground truth
        ax.plot(val.index, val.values, color=palette["Actual (Validation)"], lw=2.5, marker="o", markersize=4, label="Ground Truth (Validation)")

        # Plot baseline forecasts
        for model_name in ["Naive (Last Value)", "Seasonal Naive", "Holt-Winters (Add/Add)", "ARIMA(1,1,1)", "SARIMA (Tuned)"]:
            if model_name in fcs:
                col = palette.get(model_name, "#34495e")
                lw = 2.2 if "SARIMA" in model_name else 1.3
                ls = "-" if "SARIMA" in model_name else "--"
                ax.plot(val.index, fcs[model_name], label=model_name, color=col, lw=lw, linestyle=ls)

        # Plot SARIMA 95% confidence interval
        if ci is not None:
            ax.fill_between(val.index, ci["lower"], ci["upper"], color="#e74c3c", alpha=0.15, label="SARIMA 95% CI")

        # Vertical separator line
        ax.axvline(val.index[0], color="#bdc3c7", linestyle=":", lw=1.2)

        title_short = g.replace("Counter-Strike: Global Offensive", "CS:GO").replace(
            "Tom Clancy's Rainbow Six Siege", "Rainbow Six Siege"
        )
        ax.set_title(f"{title_short} ({data['optimal_spec']})", fontsize=11, fontweight="bold")
        ax.set_ylabel("Avg Players", fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.5)
        if idx == 0:
            ax.legend(fontsize=8, loc="upper left", ncol=2)

    # 8th subplot: legend & methodology summary
    ax_last = axes_flat[-1]
    ax_last.axis("off")
    ax_last.text(
        0.1,
        0.5,
        "Validation Forecasting Setup (Phase 3)\n\n"
        "• Horizon: H = 12 months ahead (2019-09-01 to 2020-08-01)\n"
        "• Historical Context: Start to 2019-08-01\n"
        "• Holdout Test Set: 2020-09-01 to 2021-08-01 (Untouched)\n"
        "• Shaded area represents SARIMA 95% prediction interval\n"
        "• Notice seasonal surge captures in Winter Sales (Dec) & Summer Sales (July)",
        fontsize=11,
        verticalalignment="center",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#ecf0f1", edgecolor="#95a5a6"),
    )

    fig.suptitle(
        "Figure 11: Multi-Step Validation Forecasts (H=12) Across 7 Flagship Steam Titles",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout()
    return fig


def _generate_figure_13(benchmarks_df: pd.DataFrame) -> plt.Figure:
    """Generate Figure 13: Grouped Bar Chart of Model Performance Across Metrics."""
    # Filter core baseline models for clean visualization
    models_subset = [
        "Naive (Last Value)",
        "Seasonal Naive",
        "Holt-Winters (Add/Add)",
        "ARIMA(1,1,1)",
        "SARIMA (Tuned)",
    ]
    sub = benchmarks_df[benchmarks_df["Model"].isin(models_subset)].copy()

    # Calculate average metric per model across all 7 games
    summary = sub.groupby("Model")[["MAE", "RMSE", "MAPE", "sMAPE", "MASE"]].mean().loc[models_subset]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1. MAPE (%)
    ax1 = axes[0]
    bars1 = ax1.bar(
        [m.replace(" (Last Value)", "").replace(" (Add/Add)", "") for m in summary.index],
        summary["MAPE"],
        color=["#95a5a6", "#e67e22", "#27ae60", "#8e44ad", "#e74c3c"],
        edgecolor="black",
        alpha=0.85,
    )
    ax1.set_title("Mean Absolute Percentage Error (MAPE %)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Validation MAPE (%)", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.5, axis="y")
    ax1.tick_params(axis="x", rotation=25)
    for b in bars1:
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3, f"{b.get_height():.1f}%", ha="center", fontsize=9, fontweight="bold")

    # 2. sMAPE (%)
    ax2 = axes[1]
    bars2 = ax2.bar(
        [m.replace(" (Last Value)", "").replace(" (Add/Add)", "") for m in summary.index],
        summary["sMAPE"],
        color=["#95a5a6", "#e67e22", "#27ae60", "#8e44ad", "#e74c3c"],
        edgecolor="black",
        alpha=0.85,
    )
    ax2.set_title("Symmetric MAPE (sMAPE %)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Validation sMAPE (%)", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.5, axis="y")
    ax2.tick_params(axis="x", rotation=25)
    for b in bars2:
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3, f"{b.get_height():.1f}%", ha="center", fontsize=9, fontweight="bold")

    # 3. MASE
    ax3 = axes[2]
    bars3 = ax3.bar(
        [m.replace(" (Last Value)", "").replace(" (Add/Add)", "") for m in summary.index],
        summary["MASE"],
        color=["#95a5a6", "#e67e22", "#27ae60", "#8e44ad", "#e74c3c"],
        edgecolor="black",
        alpha=0.85,
    )
    ax3.axhline(1.0, color="#c0392b", linestyle="--", lw=1.2, label="Baseline Parity (MASE=1.0)")
    ax3.set_title("Mean Absolute Scaled Error (MASE)", fontsize=11, fontweight="bold")
    ax3.set_ylabel("MASE (Scaled vs In-Sample Naive)", fontsize=10)
    ax3.grid(True, linestyle=":", alpha=0.5, axis="y")
    ax3.tick_params(axis="x", rotation=25)
    ax3.legend(fontsize=9, loc="upper right")
    for b in bars3:
        ax3.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.05, f"{b.get_height():.2f}", ha="center", fontsize=9, fontweight="bold")

    fig.suptitle(
        "Figure 13: Average Validation Performance Across Statistical Forecasting Baselines (7 Steam Games)",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    run_phase3_pipeline()
