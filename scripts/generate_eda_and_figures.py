"""Script to generate high-resolution exploratory data analysis figures for Phase 2."""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from steam_player_forecasting.data.loader import get_project_root, load_processed_data
from steam_player_forecasting.features.diagnostics import (
    check_stationarity,
    compute_acf_pacf,
    compute_rolling_stats,
    decompose_time_series,
)
from steam_player_forecasting.features.scalers import TimeSeriesScaler
from steam_player_forecasting.features.split import train_val_test_split
from steam_player_forecasting.features.transforms import (
    BoxCoxTransformer,
    LogTransformer,
    difference_series,
)


def set_plotting_style() -> None:
    """Set clean, publication-ready visual styling."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.labelweight": "semibold",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 15,
            "figure.titleweight": "bold",
            "axes.edgecolor": "#cccccc",
            "grid.color": "#e5e5e5",
            "grid.linestyle": "--",
            "grid.alpha": 0.7,
        }
    )


def generate_all_figures(output_dir: Path | None = None) -> None:
    """Generate and save all 9 Phase 2 exploratory figures at 300 DPI."""
    root = get_project_root()
    if output_dir is None:
        output_dir = root / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    set_plotting_style()
    df_all = load_processed_data()

    game_colors = {
        "Counter-Strike: Global Offensive": "#e6550d",
        "Dota 2": "#3182bd",
        "Team Fortress 2": "#756bb1",
        "Warframe": "#31a354",
        "Rust": "#de2d26",
        "Grand Theft Auto V": "#636363",
        "Tom Clancy's Rainbow Six Siege": "#8856a7",
    }

    # --------------------------------------------------------------------------
    # Figure 1: Historical Player Trajectories Overview & Structural Breaks
    # --------------------------------------------------------------------------
    print("Generating Figure 1: Historical Player Trajectories...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={"height_ratios": [2.5, 1.5]})

    for game, g_df in df_all.groupby("Game_Name"):
        color = game_colors.get(game, "#333333")
        g_df_sorted = g_df.sort_values("Month_Year")
        ax1.plot(g_df_sorted["Month_Year"], g_df_sorted["Avg_players"], label=game, color=color, linewidth=2.0)

    # Annotate structural breaks on ax1
    ax1.axvline(pd.to_datetime("2018-12-01"), color="#d95f02", linestyle=":", linewidth=1.5, alpha=0.8)
    ax1.text(pd.to_datetime("2018-12-01"), 650000, " CS:GO Free-to-Play\n (Dec 2018)", fontsize=9, color="#d95f02", fontweight="bold")

    ax1.axvline(pd.to_datetime("2020-03-01"), color="#e7298a", linestyle=":", linewidth=1.5, alpha=0.8)
    ax1.text(pd.to_datetime("2020-03-01"), 750000, " COVID-19 Lockdowns\n (March 2020)", fontsize=9, color="#e7298a", fontweight="bold")

    ax1.axvline(pd.to_datetime("2021-01-01"), color="#e41a1c", linestyle=":", linewidth=1.5, alpha=0.8)
    ax1.text(pd.to_datetime("2021-01-01"), 200000, " Rust Twitch Surge\n (Jan 2021)", fontsize=9, color="#e41a1c", fontweight="bold")

    ax1.set_title("Historical Monthly Average Concurrent Players (2012 – 2021)", pad=12)
    ax1.set_ylabel("Monthly Average Players (Concurrent)")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x):,}"))
    ax1.legend(loc="upper left", framealpha=0.9)

    # Subplot 2: Indexed / Normalized Growth (Base 100 at each game's start)
    for game, g_df in df_all.groupby("Game_Name"):
        color = game_colors.get(game, "#333333")
        g_df_sorted = g_df.sort_values("Month_Year")
        base_val = g_df_sorted["Avg_players"].iloc[0]
        indexed = (g_df_sorted["Avg_players"] / base_val) * 100
        ax2.plot(g_df_sorted["Month_Year"], indexed, label=game, color=color, linewidth=1.8, alpha=0.85)

    ax2.set_title("Indexed Relative Trajectories (Base 100 = Game Launch on Steam)", pad=10)
    ax2.set_xlabel("Observation Month")
    ax2.set_ylabel("Growth Index (Base 100)")
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    fig.savefig(output_dir / "01_historical_player_trajectories.png", dpi=300)
    plt.close(fig)

    # --------------------------------------------------------------------------
    # Figure 2: Peak vs Average Concurrency Ratio
    # --------------------------------------------------------------------------
    print("Generating Figure 2: Peak vs Average Concurrency Ratio...")
    fig, ax = plt.subplots(figsize=(13, 6))
    for game, g_df in df_all.groupby("Game_Name"):
        color = game_colors.get(game, "#333333")
        g_df_sorted = g_df.sort_values("Month_Year").copy()
        ratio = g_df_sorted["Peak_Players"] / g_df_sorted["Avg_players"]
        ax.plot(g_df_sorted["Month_Year"], ratio, label=game, color=color, linewidth=1.8, alpha=0.85)

    ax.set_title("Peak-to-Average Player Concurrency Ratio Over Time", pad=12)
    ax.set_xlabel("Month")
    ax.set_ylabel("Ratio (Peak Players / Avg Players)")
    ax.axhline(2.0, color="#888888", linestyle="--", linewidth=1.0, alpha=0.7, label="2.0x Reference Line")
    ax.legend(loc="upper right", framealpha=0.9, ncol=2)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    fig.savefig(output_dir / "02_peak_vs_avg_ratio.png", dpi=300)
    plt.close(fig)

    # --------------------------------------------------------------------------
    # Figure 3: Rolling Statistics & Heteroscedasticity
    # --------------------------------------------------------------------------
    print("Generating Figure 3: Rolling Statistics & Heteroscedasticity...")
    sample_games = ["Counter-Strike: Global Offensive", "Dota 2", "Rust", "Team Fortress 2"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    for idx, game in enumerate(sample_games):
        ax = axes[idx // 2, idx % 2]
        g_df = df_all[df_all["Game_Name"] == game].sort_values("Month_Year")
        s = pd.Series(g_df["Avg_players"].values, index=g_df["Month_Year"])
        rolling_df = compute_rolling_stats(s, windows=[6, 12])

        color = game_colors.get(game, "#333333")
        ax.plot(s.index, s.values, label="Monthly Actual", color="#999999", alpha=0.5, linewidth=1.2)
        ax.plot(rolling_df.index, rolling_df["rolling_mean_12"], label="12-Mo Rolling Mean", color=color, linewidth=2.2)

        # Plot rolling standard deviation on secondary y-axis
        ax_sec = ax.twinx()
        ax_sec.plot(rolling_df.index, rolling_df["rolling_std_12"], label="12-Mo Rolling Std (Variance)", color="#2b8cbe", linestyle="--", linewidth=1.8)
        ax_sec.set_ylabel("Rolling Std Dev", color="#2b8cbe")
        ax_sec.tick_params(axis="y", labelcolor="#2b8cbe")
        ax_sec.grid(False)

        ax.set_title(f"{game}: Rolling 12-Month Mean vs. Volatility", pad=10)
        ax.set_ylabel("Average Players")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x):,}"))
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax_sec.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", framealpha=0.85, fontsize=8)

    plt.tight_layout()
    fig.savefig(output_dir / "03_rolling_statistics_heteroscedasticity.png", dpi=300)
    plt.close(fig)

    # --------------------------------------------------------------------------
    # Figure 4: Seasonality Monthly Distribution (Boxplots across calendar months)
    # --------------------------------------------------------------------------
    print("Generating Figure 4: Monthly Seasonality Distributions...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    df_copy = df_all.copy()
    df_copy["Month_Num"] = df_copy["Month_Year"].dt.month
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    # Panel A: Normalized Monthly Deviation (% difference from game's annual mean)
    df_copy["Year"] = df_copy["Month_Year"].dt.year
    game_year_means = df_copy.groupby(["Game_Name", "Year"])["Avg_players"].transform("mean")
    df_copy["Seasonal_Deviation_Pct"] = ((df_copy["Avg_players"] - game_year_means) / game_year_means) * 100

    month_data = [df_copy[df_copy["Month_Num"] == m]["Seasonal_Deviation_Pct"].dropna().values for m in range(1, 13)]
    bp = axes[0].boxplot(month_data, patch_artist=True, tick_labels=month_names,
                         boxprops=dict(facecolor="#9ecae1", color="#3182bd"),
                         medianprops=dict(color="#08519c", linewidth=2.0))
    axes[0].axhline(0, color="#de2d26", linestyle="--", alpha=0.8, label="Annual Mean Level")
    axes[0].set_title("Aggregated Monthly Seasonality (% Deviation from Annual Mean)", pad=12)
    axes[0].set_ylabel("Seasonal Deviation (%)")
    axes[0].legend(loc="upper right")

    # Highlight Steam Sales
    # June/July = Summer Sale, December/January = Winter Sale
    axes[0].axvspan(6 - 0.4, 7 + 0.4, color="#fee0d2", alpha=0.5, label="Steam Summer Sale")
    axes[0].axvspan(12 - 0.4, 12 + 0.4, color="#e5f5e0", alpha=0.5, label="Steam Winter Sale")

    # Panel B: Month-of-Year Average Players for Flagship Titles
    for game in ["Counter-Strike: Global Offensive", "Dota 2", "Team Fortress 2"]:
        g_df = df_copy[df_copy["Game_Name"] == game]
        monthly_avg = g_df.groupby("Month_Num")["Seasonal_Deviation_Pct"].mean()
        axes[1].plot(range(1, 13), monthly_avg, marker="o", linewidth=2.2, label=game, color=game_colors.get(game))

    axes[1].set_xticks(range(1, 13))
    axes[1].set_xticklabels(month_names)
    axes[1].axhline(0, color="#999999", linestyle="--")
    axes[1].set_title("Flagship Titles: Average Seasonal Pattern by Month", pad=12)
    axes[1].set_ylabel("Mean Seasonal Deviation (%)")
    axes[1].legend(loc="upper left")

    plt.tight_layout()
    fig.savefig(output_dir / "04_monthly_seasonality_distributions.png", dpi=300)
    plt.close(fig)

    # --------------------------------------------------------------------------
    # Figure 5: Seasonal Subseries Plots
    # --------------------------------------------------------------------------
    print("Generating Figure 5: Seasonal Subseries Plots...")
    fig, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=True)
    focus_titles = ["Counter-Strike: Global Offensive", "Dota 2", "Team Fortress 2"]

    for idx, game in enumerate(focus_titles):
        ax = axes[idx]
        g_df = df_copy[df_copy["Game_Name"] == game].sort_values("Month_Year")
        # Subseries plot: For each month 1..12, plot the trajectory across years
        for m in range(1, 13):
            sub = g_df[g_df["Month_Num"] == m]
            x_vals = m + (sub["Year"] - sub["Year"].min()) / (sub["Year"].max() - sub["Year"].min() + 1) * 0.8 - 0.4
            ax.plot(x_vals, sub["Avg_players"], color=game_colors[game], alpha=0.7, linewidth=1.5)
            # Horizontal bar for mean of that month
            mean_val = sub["Avg_players"].mean()
            ax.hlines(mean_val, m - 0.4, m + 0.4, color="#333333", linewidth=2.2)

        ax.set_title(f"Seasonal Subseries Plot: {game} (Monthly Trajectories with Mean Bars)", pad=8)
        ax.set_ylabel("Avg Players")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x):,}"))

    axes[2].set_xticks(range(1, 13))
    axes[2].set_xticklabels(month_names)
    axes[2].set_xlabel("Calendar Month")
    plt.tight_layout()
    fig.savefig(output_dir / "05_seasonal_subseries_flagships.png", dpi=300)
    plt.close(fig)

    # --------------------------------------------------------------------------
    # Figure 6: STL Decomposition (CS:GO and Dota 2)
    # --------------------------------------------------------------------------
    print("Generating Figure 6: STL Decomposition...")
    fig, axes = plt.subplots(4, 2, figsize=(15, 12), sharex=True)
    decomp_games = ["Counter-Strike: Global Offensive", "Dota 2"]

    for col_idx, game in enumerate(decomp_games):
        g_df = df_all[df_all["Game_Name"] == game].sort_values("Month_Year")
        s = pd.Series(g_df["Avg_players"].values, index=pd.to_datetime(g_df["Month_Year"]))
        decomp = decompose_time_series(s, period=12, method="stl")

        color = game_colors[game]
        # Observed
        axes[0, col_idx].plot(s.index, decomp["observed"], color=color, linewidth=1.8)
        axes[0, col_idx].set_title(f"{game} — Observed", pad=6)
        axes[0, col_idx].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x):,}"))

        # Trend
        axes[1, col_idx].plot(s.index, decomp["trend"], color="#2b8cbe", linewidth=2.0)
        axes[1, col_idx].set_title("Trend Component", pad=6)
        axes[1, col_idx].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x):,}"))

        # Seasonal
        axes[2, col_idx].plot(s.index, decomp["seasonal"], color="#31a354", linewidth=1.8)
        axes[2, col_idx].set_title("12-Month Seasonal Component (STL)", pad=6)

        # Residual
        axes[3, col_idx].scatter(s.index, decomp["resid"], color="#e41a1c", s=14, alpha=0.8)
        axes[3, col_idx].axhline(0, color="#999999", linestyle="--")
        axes[3, col_idx].set_title("Residual Component", pad=6)
        axes[3, col_idx].xaxis.set_major_locator(mdates.YearLocator(2))
        axes[3, col_idx].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    fig.savefig(output_dir / "06_stl_decomposition.png", dpi=300)
    plt.close(fig)

    # --------------------------------------------------------------------------
    # Figure 7: Stationarity Diagnostics (ACF & PACF)
    # --------------------------------------------------------------------------
    print("Generating Figure 7: Stationarity Diagnostics (ACF & PACF)...")
    csgo_df = df_all[df_all["Game_Name"] == "Counter-Strike: Global Offensive"].sort_values("Month_Year")
    raw_s = pd.Series(csgo_df["Avg_players"].values)
    diff_s = pd.Series(difference_series(raw_s, order=1, drop_na=True))
    log_diff_s = pd.Series(difference_series(np.log(raw_s), order=1, drop_na=True))

    fig, axes = plt.subplots(3, 2, figsize=(14, 11))
    series_configs = [
        ("Raw Series (Non-Stationary Level)", raw_s, axes[0, 0], axes[0, 1]),
        ("First Difference: d=1", diff_s, axes[1, 0], axes[1, 1]),
        ("Log First Difference: (1-B) log(y_t)", log_diff_s, axes[2, 0], axes[2, 1]),
    ]

    for title, s_eval, ax_acf, ax_pacf in series_configs:
        diag = compute_acf_pacf(s_eval, nlags=20)
        lags = diag["lags"]
        # ACF
        ax_acf.vlines(lags, [0], diag["acf"], color="#3182bd", linewidth=2.0)
        ax_acf.scatter(lags, diag["acf"], color="#08519c", s=20)
        ax_acf.fill_between(lags, diag["acf_confint"][:, 0] - diag["acf"], diag["acf_confint"][:, 1] - diag["acf"], color="#bdd7e7", alpha=0.4)
        ax_acf.axhline(0, color="#666666", linewidth=1.0)
        ax_acf.set_title(f"ACF: {title}", pad=6)
        ax_acf.set_ylim(-1.05, 1.05)
        ax_acf.set_xlabel("Lag (Months)")

        # PACF
        ax_pacf.vlines(lags, [0], diag["pacf"], color="#e6550d", linewidth=2.0)
        ax_pacf.scatter(lags, diag["pacf"], color="#a63603", s=20)
        ax_pacf.fill_between(lags, diag["pacf_confint"][:, 0] - diag["pacf"], diag["pacf_confint"][:, 1] - diag["pacf"], color="#fdd0a2", alpha=0.4)
        ax_pacf.axhline(0, color="#666666", linewidth=1.0)
        ax_pacf.set_title(f"PACF: {title}", pad=6)
        ax_pacf.set_ylim(-1.05, 1.05)
        ax_pacf.set_xlabel("Lag (Months)")

    plt.tight_layout()
    fig.savefig(output_dir / "07_stationarity_acf_pacf.png", dpi=300)
    plt.close(fig)

    # --------------------------------------------------------------------------
    # Figure 8: Transformations & Variance Stabilization
    # --------------------------------------------------------------------------
    print("Generating Figure 8: Transformations & Variance Stabilization...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Raw
    ax_raw = axes[0, 0]
    ax_raw.plot(csgo_df["Month_Year"], csgo_df["Avg_players"], color="#e6550d", linewidth=2.0)
    ax_raw.set_title("A. Raw Player Count (Non-stationary, Heteroscedastic)", pad=10)
    ax_raw.set_ylabel("Avg Players")
    ax_raw.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x):,}"))

    # Log Transform
    log_trans = LogTransformer(offset="auto")
    log_vals = log_trans.fit_transform(csgo_df["Avg_players"])
    ax_log = axes[0, 1]
    ax_log.plot(csgo_df["Month_Year"], log_vals, color="#3182bd", linewidth=2.0)
    ax_log.set_title("B. Natural Log Transform log(y_t)", pad=10)
    ax_log.set_ylabel("Log(Players)")

    # Box-Cox Transform
    bc_trans = BoxCoxTransformer()
    bc_vals = bc_trans.fit_transform(csgo_df["Avg_players"])
    ax_bc = axes[1, 0]
    ax_bc.plot(csgo_df["Month_Year"], bc_vals, color="#756bb1", linewidth=2.0)
    ax_bc.set_title(f"C. Box-Cox Transform (Estimated λ = {bc_trans.lambda_:.3f})", pad=10)
    ax_bc.set_ylabel("Box-Cox Value")

    # First Difference
    diff_vals = difference_series(csgo_df["Avg_players"], order=1, drop_na=False)
    ax_diff = axes[1, 1]
    ax_diff.plot(csgo_df["Month_Year"], diff_vals, color="#31a354", linewidth=1.8)
    ax_diff.axhline(0, color="#333333", linestyle="--", linewidth=1.0)
    ax_diff.set_title("D. First Difference (1 - B) y_t (Mean Stationary)", pad=10)
    ax_diff.set_ylabel("Change in Players")
    ax_diff.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x):,}"))

    for ax in axes.flat:
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    fig.savefig(output_dir / "08_transformations_variance_stabilization.png", dpi=300)
    plt.close(fig)

    # --------------------------------------------------------------------------
    # Figure 9: Chronological Train / Validation / Test Splits
    # --------------------------------------------------------------------------
    print("Generating Figure 9: Chronological Train/Val/Test Splits...")
    fig, ax = plt.subplots(figsize=(14, 7))

    games = sorted(df_all["Game_Name"].unique())
    y_positions = np.arange(len(games))

    for idx, game in enumerate(games):
        g_df = df_all[df_all["Game_Name"] == game].sort_values("Month_Year")
        res = train_val_test_split(g_df, val_months=12, test_months=12)

        train_start = pd.to_datetime(res.train["Month_Year"].iloc[0])
        train_end = pd.to_datetime(res.train["Month_Year"].iloc[-1])
        val_start = pd.to_datetime(res.val["Month_Year"].iloc[0])
        val_end = pd.to_datetime(res.val["Month_Year"].iloc[-1])
        test_start = pd.to_datetime(res.test["Month_Year"].iloc[0])
        test_end = pd.to_datetime(res.test["Month_Year"].iloc[-1])

        # Plot horizontal segments
        ax.barh(idx, (train_end - train_start).days, left=train_start, height=0.55, color="#3182bd", alpha=0.85, label="Train Set" if idx == 0 else "")
        ax.barh(idx, (val_end - val_start).days, left=val_start, height=0.55, color="#feb24c", alpha=0.9, label="Validation (12 Mo)" if idx == 0 else "")
        ax.barh(idx, (test_end - test_start).days, left=test_start, height=0.55, color="#e31a1c", alpha=0.9, label="Test Holdout (12 Mo)" if idx == 0 else "")

        # Text summary
        n_tr = len(res.train)
        ax.text(train_start + (train_end - train_start) / 2, idx, f"{n_tr}m", ha="center", va="center", color="white", fontweight="bold", fontsize=9)
        ax.text(val_start + (val_end - val_start) / 2, idx, "12m", ha="center", va="center", color="black", fontweight="bold", fontsize=9)
        ax.text(test_start + (test_end - test_start) / 2, idx, "12m", ha="center", va="center", color="white", fontweight="bold", fontsize=9)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(games, fontweight="semibold")
    ax.set_title("Chronological Leakage-Free Train / Validation / Test Holdouts by Game", pad=12)
    ax.set_xlabel("Timeline")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(loc="upper left", framealpha=0.95)

    plt.tight_layout()
    fig.savefig(output_dir / "09_chronological_splits.png", dpi=300)
    plt.close(fig)

    print("Successfully exported all 9 figures to:", output_dir)


if __name__ == "__main__":
    generate_all_figures()
