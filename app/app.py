"""Streamlit Interactive Demonstration App: Steam Player Population Forecasting Lab.

Features:
- Multi-game selection across 7 long-running multiplayer Steam titles.
- Model selector for Core Quartet (SARIMA, XGBoost, LSTM, GRU) and baseline models.
- Interactive time-series trajectory plotting with 95% SARIMA confidence intervals.
- Live performance scorecard and metrics comparison (MAE, RMSE, MAPE, sMAPE, MASE).
- Horizon error analysis and residual breakdown.
- Game dynamics, structural break annotations, and research synthesis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# Configuration & Theme
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Steam Player Forecasting Lab",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished interface
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1a202c;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4a5568;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f7fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.5rem;
    }
    .winner-badge {
        background-color: #ebf8ff;
        color: #2b6cb0;
        font-weight: bold;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        display: inline-block;
    }
    .callout-box {
        background-color: #f8fafc;
        border-left: 4px solid #3182ce;
        padding: 0.9rem;
        border-radius: 4px;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Color palette for consistent visuals
MODEL_PALETTE = {
    "SARIMA": "#1f77b4",
    "XGBoost": "#d95f02",
    "LSTM": "#2ca02c",
    "GRU": "#756bb1",
    "Holt-Winters": "#17becf",
    "Ridge": "#e377c2",
    "Random Forest": "#8c564b",
    "RNN": "#bcbd22",
    "Seasonal Naive": "#7f7f7f",
    "Naive": "#9467bd",
    "ARIMA(1,1,1)": "#ffbb78",
}

GAME_INFO = {
    "Counter-Strike: Global Offensive": {
        "genre": "Tactical FPS",
        "release": "Aug 2012",
        "notes": "F2P transition in Dec 2018; massive retention plateau during 2020-2021 pandemic.",
    },
    "Dota 2": {
        "genre": "MOBA",
        "release": "Jul 2013",
        "notes": "Annual surge driven by The International (August/October); strong annual cyclicality.",
    },
    "Grand Theft Auto V": {
        "genre": "Open-World Action Sandbox",
        "release": "Apr 2015",
        "notes": "Sustained post-launch longevity; summer & winter updates drive distinct seasonal peaks.",
    },
    "Rust": {
        "genre": "Multiplayer Survival Sandbox",
        "release": "Dec 2013",
        "notes": "Massive viral Twitch/OfflineTV server explosion in January 2021 (+92% MoM surge).",
    },
    "Team Fortress 2": {
        "genre": "Class-Based Hero Shooter",
        "release": "Oct 2007",
        "notes": "Enduring cult fanbase; strong seasonal summer update and Halloween 'Scream Fortress' spikes.",
    },
    "Tom Clancy's Rainbow Six Siege": {
        "genre": "Tactical Operator Shooter",
        "release": "Dec 2015",
        "notes": "Quarterly seasonal expansions ('Operations'); downward organic stabilization in 2020-2021.",
    },
    "Warframe": {
        "genre": "Co-op Sci-Fi Looter Shooter",
        "release": "Mar 2013",
        "notes": "Annual TennoCon conventions (July) and major cinematic story quest expansions.",
    },
}


# -----------------------------------------------------------------------------
# Data Loaders
# -----------------------------------------------------------------------------
@st.cache_data
def get_root_dir() -> Path:
    """Find repository root directory."""
    cwd = Path.cwd()
    if (cwd / "data" / "processed").exists():
        return cwd
    if (cwd.parent / "data" / "processed").exists():
        return cwd.parent
    return cwd


@st.cache_data
def load_forecast_cache() -> dict[str, Any]:
    """Load precomputed test forecasts cache."""
    root = get_root_dir()
    for candidate in [
        root / "report" / "test_forecasts_cache.json",
        root / "data" / "processed" / "test_forecasts_cache.json",
    ]:
        if candidate.exists():
            with open(candidate, "r", encoding="utf-8") as f:
                return json.load(f)
    return {}


@st.cache_data
def load_benchmarks_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load benchmark and leaderboard tables."""
    root = get_root_dir()
    rep_dir = root / "report"

    test_b = pd.DataFrame()
    lead_b = pd.DataFrame()
    val_comp = pd.DataFrame()

    f1 = rep_dir / "final_test_benchmarks.csv"
    if f1.exists():
        test_b = pd.read_csv(f1)

    f2 = rep_dir / "final_leaderboard.csv"
    if f2.exists():
        lead_b = pd.read_csv(f2)

    f3 = rep_dir / "validation_vs_test_comparison.csv"
    if f3.exists():
        val_comp = pd.read_csv(f3)

    return test_b, lead_b, val_comp


@st.cache_data
def load_game_raw_history(game_name: str) -> pd.DataFrame:
    """Load full historical CSV for a given game."""
    root = get_root_dir()
    slug = (
        game_name.lower()
        .replace(" ", "_")
        .replace(":", "")
        .replace("'", "_")
        .replace("-", "_")
    )
    p = root / "data" / "processed" / f"{slug}.csv"
    if p.exists():
        df = pd.read_csv(p, parse_dates=["Month_Year"]).sort_values("Month_Year")
        return df
    return pd.DataFrame()


# -----------------------------------------------------------------------------
# Main Application
# -----------------------------------------------------------------------------
def main() -> None:
    # 1. Header
    st.markdown("<div class='main-header'>🎮 Steam Player Population Forecasting Lab</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-header'>Evaluating Classical Econometric Models (SARIMA) vs Modern Recurrent Deep Learning (LSTM, GRU) & Gradient Boosting (XGBoost)</div>",
        unsafe_allow_html=True,
    )

    cache = load_forecast_cache()
    test_benchmarks, leaderboard_df, val_comp = load_benchmarks_data()

    if not cache:
        st.error("Forecast cache not found. Please run `python scripts/run_phase5_benchmarks.py` first to generate data.")
        st.stop()

    games = list(cache.keys())

    # 2. Sidebar Controls
    st.sidebar.header("🕹️ Simulation & Model Controls")

    selected_game = st.sidebar.selectbox("Select Steam Title:", games, index=0)

    st.sidebar.markdown("---")
    st.sidebar.subheader("🤖 Model Candidates")

    all_available_models = list(cache[selected_game]["test_forecasts"].keys())
    default_quartet = [m for m in ["SARIMA", "XGBoost", "LSTM", "GRU"] if m in all_available_models]

    selected_models = st.sidebar.multiselect(
        "Models to Compare:",
        options=all_available_models,
        default=default_quartet,
        help="Select any combination of models to overlay on the forecast chart.",
    )

    if not selected_models:
        selected_models = default_quartet

    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Visualization Options")

    show_ci = st.sidebar.checkbox("Show SARIMA 95% Confidence Interval", value=True)
    history_window = st.sidebar.slider(
        "Historical Context Window (Months):",
        min_value=6,
        max_value=36,
        value=24,
        step=6,
        help="Number of historical months preceding the test window to display on the chart.",
    )

    # Sidebar game metadata
    st.sidebar.markdown("---")
    st.sidebar.subheader("ℹ️ Game Profile")
    meta = GAME_INFO.get(selected_game, {})
    if meta:
        st.sidebar.markdown(f"**Genre:** `{meta.get('genre', 'N/A')}`")
        st.sidebar.markdown(f"**Steam Release:** `{meta.get('release', 'N/A')}`")
        st.sidebar.caption(f"**Domain Context:** {meta.get('notes', '')}")

    # Optimal specs display
    game_cache = cache[selected_game]
    st.sidebar.markdown(f"**Optimal SARIMA Order:** `{game_cache.get('optimal_sarima_order', 'N/A')}`")
    xgb_p = game_cache.get("optimal_xgb_params", {})
    st.sidebar.markdown(f"**Tuned XGBoost:** `depth={xgb_p.get('max_depth', 3)}, lr={xgb_p.get('learning_rate', 0.05)}`")

    # 3. Main Dashboard Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Forecast Trajectories",
        "🏆 Model Leaderboards & Metrics",
        "🔍 Game Dynamics & Structural Breaks",
        "🧠 Research Synthesis & Takeaways",
    ])

    # -------------------------------------------------------------------------
    # TAB 1: Forecast Trajectories
    # -------------------------------------------------------------------------
    with tab1:
        st.markdown(
            """
            <div class='callout-box'>
                <b>Strict Chronological Holdout Evaluation (2020-09 to 2021-08):</b><br/>
                Models were trained strictly on historical observations prior to September 2020 with zero future data leakage.
                The 12-month test trajectory evaluates true multi-step out-of-sample forecast fidelity across global gaming shifts.
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Plot setup
        hist_dates = pd.to_datetime(game_cache["history_dates"])[-history_window:]
        hist_vals = np.array(game_cache["history_values"])[-history_window:]
        test_dates = pd.to_datetime(game_cache["test_dates"])
        test_actuals = np.array(game_cache["test_actuals"])

        fig, ax = plt.subplots(figsize=(13, 6))

        # Historical curve
        ax.plot(
            hist_dates,
            hist_vals,
            color="#4a5568",
            lw=2.0,
            marker="o",
            ms=3.5,
            label="Historical Players (Train + Val)",
            alpha=0.85,
        )

        # Connection line from history to test
        conn_dates = [hist_dates[-1], test_dates[0]]
        conn_actuals = [hist_vals[-1], test_actuals[0]]
        ax.plot(conn_dates, conn_actuals, color="#111111", ls=":", lw=1.8)

        # Ground Truth Test
        ax.plot(
            test_dates,
            test_actuals,
            color="#111111",
            lw=3.0,
            marker="o",
            ms=6,
            label="Actual Test Ground Truth",
            zorder=10,
        )

        # Selected Model Forecasts
        for model in selected_models:
            fc = np.array(game_cache["test_forecasts"][model])
            color = MODEL_PALETTE.get(model, "#333333")
            conn_fc = [hist_vals[-1], fc[0]]

            ax.plot(conn_dates, conn_fc, color=color, ls=":", lw=1.5, alpha=0.6)
            ax.plot(
                test_dates,
                fc,
                color=color,
                lw=2.2,
                marker="^",
                ms=5.5,
                label=f"{model} Forecast",
                alpha=0.95,
            )

        # SARIMA Confidence Interval
        if show_ci and "SARIMA" in selected_models:
            ci_lower = np.array(game_cache.get("sarima_ci_lower", []))
            ci_upper = np.array(game_cache.get("sarima_ci_upper", []))
            if len(ci_lower) > 0 and len(ci_upper) > 0:
                ax.fill_between(
                    test_dates,
                    ci_lower,
                    ci_upper,
                    color=MODEL_PALETTE["SARIMA"],
                    alpha=0.15,
                    label="SARIMA 95% Confidence Interval",
                )

        # Boundary Line
        ax.axvline(test_dates[0], color="#c53030", ls="--", lw=1.2, alpha=0.7)
        ax.text(
            test_dates[0],
            ax.get_ylim()[1] * 0.98,
            " Test Horizon Start (Sep 2020)",
            color="#c53030",
            fontsize=9,
            fontweight="bold",
            va="top",
        )

        ax.set_title(f"{selected_game} — Out-of-Sample 12-Month Forecast Evaluation", fontsize=13, fontweight="bold", pad=10)
        ax.set_ylabel("Monthly Average Players", fontsize=11)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:,.0f}"))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.tick_params(axis="x", rotation=25)
        ax.grid(True, alpha=0.3, ls="--")
        ax.legend(loc="upper left", framealpha=0.92, fontsize=9, ncol=2)

        st.pyplot(fig)
        plt.close(fig)

        # Performance Scorecard
        st.subheader("🎯 Test-Set Performance Scoreboard")

        game_metrics = test_benchmarks[
            (test_benchmarks["Game"] == selected_game) & (test_benchmarks["Model"].isin(selected_models))
        ].copy()

        if not game_metrics.empty:
            game_metrics = game_metrics.sort_values("MASE")
            best_model_name = game_metrics.iloc[0]["Model"]
            best_mase = game_metrics.iloc[0]["MASE"]
            best_mae = game_metrics.iloc[0]["MAE"]
            best_mape = game_metrics.iloc[0]["MAPE"]

            st.success(
                f"🏆 **Top Performer on {selected_game}:** `{best_model_name}` (MAE: {best_mae:,.1f} | MAPE: {best_mape:.2f}% | MASE: {best_mase:.3f})"
            )

            # Display metric columns
            cols = st.columns(min(len(selected_models), 4))
            for i, (_, row) in enumerate(game_metrics.iterrows()):
                col = cols[i % len(cols)]
                with col:
                    is_winner = row["Model"] == best_model_name
                    border_color = "#3182ce" if is_winner else "#e2e8f0"
                    st.markdown(
                        f"""
                        <div style="border: 2px solid {border_color}; border-radius: 8px; padding: 0.8rem; background: #ffffff; margin-bottom: 0.6rem;">
                            <h4 style="margin: 0; color: {MODEL_PALETTE.get(row['Model'], '#1a202c')};">
                                {row['Model']} {'🏆' if is_winner else ''}
                            </h4>
                            <p style="font-size: 0.8rem; color: #718096; margin: 0 0 0.5rem 0;">{row['Paradigm']}</p>
                            <div style="display: flex; justify-content: space-between; font-size: 0.9rem;">
                                <span><b>MAE:</b></span> <span>{row['MAE']:,.1f}</span>
                            </div>
                            <div style="display: flex; justify-content: space-between; font-size: 0.9rem;">
                                <span><b>RMSE:</b></span> <span>{row['RMSE']:,.1f}</span>
                            </div>
                            <div style="display: flex; justify-content: space-between; font-size: 0.9rem;">
                                <span><b>MAPE:</b></span> <span>{row['MAPE']:.2f}%</span>
                            </div>
                            <div style="display: flex; justify-content: space-between; font-size: 0.9rem;">
                                <span><b>MASE:</b></span> <span>{row['MASE']:.3f}</span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        # Monthly Step-by-Step Breakdown Table
        with st.expander("📅 View Detailed Month-by-Month Forecast Table"):
            month_rows = []
            for h in range(12):
                m_date = test_dates[h].strftime("%Y-%m")
                actual_val = test_actuals[h]
                m_row = {"Month": m_date, "Actual": f"{actual_val:,.0f}"}
                for m in selected_models:
                    pred_val = game_cache["test_forecasts"][m][h]
                    diff = pred_val - actual_val
                    pct_err = (abs(diff) / max(actual_val, 1)) * 100
                    m_row[f"{m} (Pred)"] = f"{pred_val:,.0f}"
                    m_row[f"{m} (Err %)"] = f"{pct_err:.1f}%"
                month_rows.append(m_row)
            st.dataframe(pd.DataFrame(month_rows), use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 2: Model Leaderboards & Metrics
    # -------------------------------------------------------------------------
    with tab2:
        st.header("🏆 Final Forecasting Benchmark Leaderboards")

        col_l1, col_l2 = st.columns([1, 1])

        with col_l1:
            st.subheader(f"Per-Model Standings for {selected_game}")
            sub_game = test_benchmarks[test_benchmarks["Game"] == selected_game].sort_values("MASE")
            st.dataframe(
                sub_game[["Model", "Paradigm", "MAE", "RMSE", "MAPE", "sMAPE", "MASE"]].style.format({
                    "MAE": "{:,.1f}",
                    "RMSE": "{:,.1f}",
                    "MAPE": "{:.2f}%",
                    "sMAPE": "{:.2f}%",
                    "MASE": "{:.3f}",
                }),
                use_container_width=True,
            )

        with col_l2:
            st.subheader("Cross-Game Global Leaderboard (All 7 Games)")
            if not leaderboard_df.empty:
                st.dataframe(
                    leaderboard_df.style.format({
                        "Mean_MAE": "{:,.1f}",
                        "Median_MAE": "{:,.1f}",
                        "Mean_RMSE": "{:,.1f}",
                        "Mean_MAPE": "{:.2f}%",
                        "Mean_sMAPE": "{:.2f}%",
                        "Mean_MASE": "{:.3f}",
                    }),
                    use_container_width=True,
                )

        st.markdown("---")
        st.subheader("🔬 Validation vs Test Out-of-Sample Generalization")
        st.caption(
            "Tracking how model error evolved between the validation tuning window (2019-09 to 2020-08) and the holdout test window (2020-09 to 2021-08)."
        )

        if not val_comp.empty:
            game_val_comp = val_comp[val_comp["Game"] == selected_game].sort_values("MASE_Test")
            st.dataframe(
                game_val_comp[["Model", "Paradigm", "MAE_Val", "MAE_Test", "MAPE_Val", "MAPE_Test", "MASE_Val", "MASE_Test", "MAE_Delta_Pct"]].style.format({
                    "MAE_Val": "{:,.1f}",
                    "MAE_Test": "{:,.1f}",
                    "MAPE_Val": "{:.2f}%",
                    "MAPE_Test": "{:.2f}%",
                    "MASE_Val": "{:.3f}",
                    "MASE_Test": "{:.3f}",
                    "MAE_Delta_Pct": "{:+.1f}%",
                }),
                use_container_width=True,
            )

    # -------------------------------------------------------------------------
    # TAB 3: Game Dynamics & Structural Breaks
    # -------------------------------------------------------------------------
    with tab3:
        st.header(f"🔍 Historical Dynamics & Regime Shifts: {selected_game}")

        raw_df = load_game_raw_history(selected_game)

        if not raw_df.empty:
            fig_hist, (ax_h1, ax_h2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)

            # Trajectory: Avg vs Peak
            ax_h1.plot(raw_df["Month_Year"], raw_df["Avg_players"], color="#2b6cb0", lw=2.0, label="Avg Players")
            if "Peak_Players" in raw_df.columns:
                ax_h1.plot(raw_df["Month_Year"], raw_df["Peak_Players"], color="#c53030", lw=1.5, ls="--", alpha=0.8, label="Peak Concurrent Players")

            ax_h1.set_title("Historical Player Count Trajectory (Monthly Average & Peak Concurrent)", fontweight="bold")
            ax_h1.set_ylabel("Player Count")
            ax_h1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:,.0f}"))
            ax_h1.grid(True, alpha=0.3, ls="--")
            ax_h1.legend(loc="upper left")

            # Peak-to-Average Ratio
            if "Peak_Players" in raw_df.columns and "Avg_players" in raw_df.columns:
                ratio = raw_df["Peak_Players"] / np.maximum(raw_df["Avg_players"], 1)
                ax_h2.plot(raw_df["Month_Year"], ratio, color="#4a5568", lw=1.8, marker=".", ms=4)
                ax_h2.axhline(ratio.mean(), color="#718096", ls=":", label=f"Mean Ratio: {ratio.mean():.2f}")
                ax_h2.set_title("Peak-to-Average Volatility Ratio ($Peak / Avg$)", fontweight="bold")
                ax_h2.set_ylabel("Ratio")
                ax_h2.set_xlabel("Date")
                ax_h2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
                ax_h2.grid(True, alpha=0.3, ls="--")
                ax_h2.legend(loc="upper left")

            st.pyplot(fig_hist)
            plt.close(fig_hist)

            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.subheader("Statistical Profile")
                st.markdown(f"- **Total Monthly Observations:** `{len(raw_df)} months`")
                st.markdown(f"- **Historical Range:** `{raw_df['Month_Year'].min().strftime('%Y-%m')} to {raw_df['Month_Year'].max().strftime('%Y-%m')}`")
                st.markdown(f"- **All-Time Average Players:** `{raw_df['Avg_players'].mean():,.0f}`")
                st.markdown(f"- **All-Time Peak Players:** `{raw_df['Peak_Players'].max():,.0f}`")
                st.markdown(f"- **Player Population Volatility (Std Dev):** `{raw_df['Avg_players'].std():,.0f}`")

            with col_d2:
                st.subheader("Documented Structural Breaks")
                st.markdown(
                    f"""
                    - **Pandemic Lockdown Shock (Mar–May 2020):** Global stay-at-home orders drove unprecedented surge across all games.
                    - **Summer & Winter Steam Sales:** Annual seasonal surges consistently observed in June/July and December/January.
                    - **Game-Specific Shocks:** {meta.get('notes', 'None recorded.')}
                    """
                )

    # -------------------------------------------------------------------------
    # TAB 4: Research Synthesis & Takeaways
    # -------------------------------------------------------------------------
    with tab4:
        st.header("🧠 Core Research Findings & Empirical Synthesis")

        st.markdown(
            """
            ### Core Research Question:
            > *Can modern recurrent deep learning models (LSTM, GRU) or gradient-boosted trees (XGBoost) reliably outperform classical econometric benchmarks (SARIMA) when forecasting monthly active player populations for top multiplayer games on Steam?*
            """
        )

        st.markdown(
            r"""
            ### 🔑 Key Empirical Discoveries:

            1. **The Core Quartet Split (SARIMA 3 wins, GRU 3 wins, LSTM 1 win, XGBoost 0 wins):**
               - **SARIMA excels in regular seasonal regimes with strong cyclicality** (*Dota 2*, *Grand Theft Auto V*, *Rust*). Differencing ($d=1, D=1$) naturally enforces mean reversion and prevents explosive runaway forecasts.
               - **GRU/LSTM excel in non-linear plateau and stabilization regimes** (*CS:GO*, *Rainbow Six Siege*, *Warframe*, *Team Fortress 2*). The gating mechanisms flexibly capture mid-term regime shifts that fixed linear polynomials fail to model.
               - **XGBoost struggled across all 7 games (Mean MASE 0.923 vs SARIMA 0.672, GRU 0.776):** Recursive multi-step forecasting in tree ensembles suffers from compounding autoregressive error. In contrast, deep recurrent networks with direct multi-step forecast heads avoid recursive drift.

            2. **The Power of Parsimonious Baselines (Holt-Winters & Ridge):**
               - Across the full 11-model candidate set, **Holt-Winters (Mean MASE: 0.526)** and **L2-Regularized Ridge Regression (Mean MASE: 0.561)** achieved the lowest overall average errors across the 7 titles.
               - In data-constrained regimes with short historical samples ($N < 80$ months), heavily parameterized neural networks and unconstrained trees risk overfitting, while regularized models maintain optimal bias-variance balance.

            3. **Vulnerability to Unprecedented Macro Shocks:**
               - The January 2021 **Rust Twitch boom (+92% surge)** demonstrated that purely autoregressive models cannot foresee influencer-driven virality. However, SARIMA's state-space formulation stabilized error faster than tree-based models, which over-extrapolated the spike.

            4. **Practical Industry Recommendations:**
               - **Hybrid Forecasting Architecture:** Use SARIMA or Holt-Winters as the baseline operational backbone for stable titles; deploy GRU/LSTM for games undergoing rapid content patch cycles and non-linear player plateaus.
               - **Avoid Pure Recursive Tree Ensembles for Long Horizons:** When using XGBoost or LightGBM for $H \ge 6$ months, use direct multi-output regressors rather than recursive single-step autoregression.
            """
        )

        st.markdown("---")
        st.subheader("📚 Comparative Paradigm Tradeoff Matrix")

        tradeoff_data = [
            {
                "Paradigm": "Classical Econometric (SARIMA)",
                "Data Efficiency": "⭐⭐⭐⭐⭐ (Excellent)",
                "Multi-Step Stability": "⭐⭐⭐⭐⭐ (Closed-Form)",
                "Non-Linear Adaptation": "⭐⭐ (Linear/Difference)",
                "Confidence Bounds": "Analytical 95% CI",
                "Best Use Case": "Stable games with annual sales & esports seasonality",
            },
            {
                "Paradigm": "Gradient Boosted ML (XGBoost)",
                "Data Efficiency": "⭐⭐⭐ (Moderate)",
                "Multi-Step Stability": "⭐⭐ (Recursive Drift)",
                "Non-Linear Adaptation": "⭐⭐⭐⭐ (High)",
                "Confidence Bounds": "Quantile / Bootstrapped",
                "Best Use Case": "Short horizons ($H \\le 3$) with rich exogenous features",
            },
            {
                "Paradigm": "Recurrent Deep Learning (GRU / LSTM)",
                "Data Efficiency": "⭐⭐⭐ (Requires sequence buffer)",
                "Multi-Step Stability": "⭐⭐⭐⭐ (Direct Sequence Head)",
                "Non-Linear Adaptation": "⭐⭐⭐⭐⭐ (Non-linear memory)",
                "Confidence Bounds": "Monte Carlo Dropout",
                "Best Use Case": "Longer histories with complex structural plateaus",
            },
            {
                "Paradigm": "Exponential Smoothing (Holt-Winters)",
                "Data Efficiency": "⭐⭐⭐⭐⭐ (Very Low Sample Req)",
                "Multi-Step Stability": "⭐⭐⭐⭐ (Damped Trend)",
                "Non-Linear Adaptation": "⭐⭐ (Linear Smoother)",
                "Confidence Bounds": "Analytical",
                "Best Use Case": "Quick, highly robust baseline across hundreds of games",
            },
        ]
        st.dataframe(pd.DataFrame(tradeoff_data), use_container_width=True)


if __name__ == "__main__":
    main()
