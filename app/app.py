"""Streamlit Interactive Demonstration App: Steam Player Population Forecasting Lab.

Streamlined Steam Store Design Architecture:
- Dark naval/slate theme (#171a21, #1b2838, #0e141b) with cyan highlights and discount badges
- Modular components: header, capsule cards, interactive Altair charts, dark-mode time-series plots
- Zero emojis and clutter-free, responsive layout
- Out-of-sample 12-month test horizon comparison: SARIMA vs XGBoost vs LSTM vs GRU
- Interactive user controls: dynamic horizon slider, metric pivot, model presets, shock simulator, CSV exporter
- Real-time scorecards, cross-game leaderboards, and structural regime diagnostics
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any
import importlib

# Ensure APP_DIR is in sys.path for component imports
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import numpy as np
import pandas as pd
import streamlit as st

import components.cards
import components.header
import components.plots
importlib.reload(components.cards)
importlib.reload(components.header)
importlib.reload(components.plots)

from components.cards import compute_dynamic_metrics, render_game_capsule, render_scorecard
from components.header import render_steam_header
from components.plots import (
    STEAM_PALETTE,
    create_horizon_error_chart,
    create_interactive_forecast_chart,
    plot_historical_dynamics,
)

# -----------------------------------------------------------------------------
# Path Resolution & Page Configuration
# -----------------------------------------------------------------------------
ROOT_DIR = APP_DIR.parent if (APP_DIR.parent / "data").exists() else APP_DIR
ASSETS_DIR = APP_DIR / "assets"
STYLES_DIR = APP_DIR / "styles"
LOGO_PATH = ASSETS_DIR / "steam_logo.png"

st.set_page_config(
    page_title="Steam Player Forecasting Lab",
    page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else None,
    layout="wide",
    initial_sidebar_state="expanded",
)

if LOGO_PATH.exists():
    st.logo(str(LOGO_PATH))


# -----------------------------------------------------------------------------
# Inject Steam CSS
# -----------------------------------------------------------------------------
def inject_custom_css() -> None:
    """Load and inject external Steam Store stylesheet."""
    css_file = STYLES_DIR / "steam.css"
    if css_file.exists():
        with open(css_file, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


inject_custom_css()

# -----------------------------------------------------------------------------
# Metadata & Domain Knowledge
# -----------------------------------------------------------------------------
GAME_INFO: dict[str, dict[str, str]] = {
    "Counter-Strike: Global Offensive": {
        "genre": "Tactical FPS",
        "release": "Aug 2012",
        "notes": "F2P transition in Dec 2018; massive retention plateau during 2020-2021 pandemic lockdowns.",
    },
    "Dota 2": {
        "genre": "MOBA",
        "release": "Jul 2013",
        "notes": "Annual surge driven by The International (August/October); strong annual cyclicality.",
    },
    "Grand Theft Auto V": {
        "genre": "Open-World Action Sandbox",
        "release": "Apr 2015",
        "notes": "Sustained longevity; summer and winter updates drive distinct seasonal peaks.",
    },
    "Rust": {
        "genre": "Multiplayer Survival Sandbox",
        "release": "Dec 2013",
        "notes": "Massive viral Twitch/OfflineTV server explosion in January 2021 (+92% MoM surge).",
    },
    "Team Fortress 2": {
        "genre": "Class-Based Hero Shooter",
        "release": "Oct 2007",
        "notes": "Enduring fanbase; strong seasonal summer update and Halloween Scream Fortress spikes.",
    },
    "Tom Clancy's Rainbow Six Siege": {
        "genre": "Tactical Operator Shooter",
        "release": "Dec 2015",
        "notes": "Quarterly seasonal expansions (Operations); downward organic stabilization in 2020-2021.",
    },
    "Warframe": {
        "genre": "Co-op Sci-Fi Looter Shooter",
        "release": "Mar 2013",
        "notes": "Annual TennoCon conventions (July) and major cinematic story quest expansions.",
    },
}


# -----------------------------------------------------------------------------
# Data Loaders (Cached)
# -----------------------------------------------------------------------------
@st.cache_data
def load_forecast_cache() -> dict[str, Any]:
    """Load precomputed test forecasts cache."""
    for candidate in [
        ROOT_DIR / "report" / "test_forecasts_cache.json",
        ROOT_DIR / "data" / "processed" / "test_forecasts_cache.json",
    ]:
        if candidate.exists():
            with open(candidate, "r", encoding="utf-8") as f:
                return json.load(f)
    return {}


@st.cache_data
def load_benchmarks_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load benchmark and leaderboard tables."""
    rep_dir = ROOT_DIR / "report"
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
    slug = (
        game_name.lower()
        .replace(" ", "_")
        .replace(":", "")
        .replace("'", "_")
        .replace("-", "_")
    )
    p = ROOT_DIR / "data" / "processed" / f"{slug}.csv"
    if p.exists():
        df = pd.read_csv(p, parse_dates=["Month_Year"]).sort_values("Month_Year")
        return df
    return pd.DataFrame()


# -----------------------------------------------------------------------------
# Main Application Flow
# -----------------------------------------------------------------------------
def main() -> None:
    # 1. Top Steam Header
    render_steam_header(ASSETS_DIR)

    cache = load_forecast_cache()
    test_benchmarks, leaderboard_df, val_comp = load_benchmarks_data()

    if not cache:
        st.error(
            "Forecast cache not found. Please run `python scripts/run_phase5_benchmarks.py` to generate benchmarking artifacts."
        )
        st.stop()

    games = list(cache.keys())
    paradigm_map = (
        dict(zip(test_benchmarks["Model"], test_benchmarks["Paradigm"]))
        if not test_benchmarks.empty
        else {}
    )

    # 2. Sidebar Navigation & Filters (Clean, No Emojis)
    with st.sidebar:
        st.markdown("**Game Selection**")
        selected_game = st.selectbox(
            "Select Steam Title:",
            games,
            index=0,
            label_visibility="collapsed",
        )

        st.divider()
        st.markdown("**Model Presets**")
        all_available_models = list(cache[selected_game]["test_forecasts"].keys())

        preset_options: dict[str, list[str]] = {
            "Core Quartet": [m for m in ["SARIMA", "XGBoost", "LSTM", "GRU"] if m in all_available_models],
            "Econometric": [m for m in ["Naive", "Seasonal Naive", "Holt-Winters", "ARIMA(1,1,1)", "SARIMA"] if m in all_available_models],
            "Machine Learning": [m for m in ["Ridge", "Random Forest", "XGBoost", "RNN", "LSTM", "GRU"] if m in all_available_models],
            "All Candidates": all_available_models,
        }

        if "last_preset" not in st.session_state:
            st.session_state.last_preset = "Core Quartet"
            st.session_state.active_models = preset_options["Core Quartet"]

        preset_choice = st.segmented_control(
            "Model Preset",
            options=list(preset_options.keys()),
            default=st.session_state.last_preset,
            label_visibility="collapsed",
        )

        if preset_choice and preset_choice != st.session_state.last_preset:
            st.session_state.last_preset = preset_choice
            st.session_state.active_models = preset_options[preset_choice]

        st.markdown("**Active Models**")
        selected_models = st.multiselect(
            "Active Models:",
            options=all_available_models,
            default=st.session_state.active_models,
            label_visibility="collapsed",
            help="Select candidate models to overlay on the forecast chart.",
        )

        if not selected_models:
            selected_models = preset_options["Core Quartet"]

        st.divider()
        st.markdown("**Visualization Settings**")
        show_ci = st.toggle("Show SARIMA 95% Confidence Bounds", value=True)
        history_window = st.slider(
            "Historical Context (Months):",
            min_value=6,
            max_value=36,
            value=24,
            step=6,
            help="Number of pre-test historical months to render on the chart.",
        )

        st.divider()
        meta = GAME_INFO.get(selected_game, {})
        st.markdown(f"**Genre:** <span style='color: #66c0f4;'>{meta.get('genre', 'N/A')}</span>", unsafe_allow_html=True)
        st.markdown(f"**Steam Release:** <span style='color: #c7d5e0;'>{meta.get('release', 'N/A')}</span>", unsafe_allow_html=True)
        st.caption(f"{meta.get('notes', '')}")

    # 3. Main Tabs (Clean, No Emojis)
    tab1, tab2, tab3, tab4 = st.tabs([
        "Forecast Trajectories",
        "Model Leaderboard",
        "Historical Dynamics",
        "Research Synthesis",
    ])

    # -------------------------------------------------------------------------
    # TAB 1: Forecast Trajectories (Interactive Controls & Altair Vector Chart)
    # -------------------------------------------------------------------------
    with tab1:
        # Steam Game Capsule
        render_game_capsule(selected_game, meta, cache[selected_game])

        # Interactive Control Toolbar
        with st.container(border=True):
            col_ctrl1, col_ctrl2 = st.columns([1, 1])

            with col_ctrl1:
                horizon = st.slider(
                    "Forecast Evaluation Horizon (Months):",
                    min_value=1,
                    max_value=12,
                    value=12,
                    step=1,
                    help="Dynamically truncate the test window to evaluate short vs long-range multi-step performance.",
                )

            with col_ctrl2:
                rank_by = st.segmented_control(
                    "Rank Models By:",
                    options=["MASE", "MAE", "RMSE", "MAPE", "sMAPE"],
                    default="MASE",
                    help="Select performance metric to dynamically re-sort cards and determine Top Performer.",
                )
                if not rank_by:
                    rank_by = "MASE"

            # Scenario Shock Simulator
            with st.expander("Scenario Shock Simulator (What-If Demand Surge / Churn)"):
                col_shock1, col_shock2 = st.columns([2, 1])
                with col_shock1:
                    shock_pct = st.slider(
                        "Simulate Exogenous Shock Multiplier (%):",
                        min_value=-30,
                        max_value=50,
                        value=0,
                        step=5,
                        format="%+d%%",
                        help="Simulate an exogenous demand surge (e.g. viral streamer boom) or churn shock.",
                    )
                with col_shock2:
                    shock_target_model = st.selectbox(
                        "Target Model For Shock:",
                        options=selected_models,
                        index=0 if selected_models else 0,
                        help="Model baseline upon which to project simulated shock curve.",
                    )

        # Interactive Altair Chart
        chart = create_interactive_forecast_chart(
            game_cache=cache[selected_game],
            selected_models=selected_models,
            horizon=horizon,
            history_window=history_window,
            show_ci=show_ci,
            demand_shock_pct=float(shock_pct),
            shock_target_model=shock_target_model if shock_pct != 0 else "",
        )
        st.altair_chart(chart, width="stretch")

        # Dynamic Performance Scorecard
        st.markdown(f"### Out-of-Sample Performance (H = {horizon} Months)")

        hist_vals = np.array(cache[selected_game]["history_values"])
        seasonal_scale = (
            float(np.mean(np.abs(hist_vals[12:] - hist_vals[:-12])))
            if len(hist_vals) > 12
            else 1.0
        )

        dynamic_metrics = compute_dynamic_metrics(
            actuals=cache[selected_game]["test_actuals"],
            forecasts_dict={
                m: cache[selected_game]["test_forecasts"][m]
                for m in selected_models
                if m in cache[selected_game]["test_forecasts"]
            },
            seasonal_scale=seasonal_scale,
            horizon=horizon,
            paradigm_map=paradigm_map,
        )

        if not dynamic_metrics.empty:
            render_scorecard(
                game_metrics=dynamic_metrics,
                rank_by=rank_by,
                palette=STEAM_PALETTE,
            )

        # Month-by-Month Forecast Breakdown & CSV Export
        with st.expander(f"Month-by-Month Forecast Breakdown (H = {horizon} Months)"):
            test_dates = pd.to_datetime(cache[selected_game]["test_dates"])[:horizon]
            test_actuals = np.array(cache[selected_game]["test_actuals"])[:horizon]
            month_rows: list[dict[str, Any]] = []

            for h in range(len(test_dates)):
                m_date = test_dates[h].strftime("%Y-%m")
                actual_val = test_actuals[h]
                m_row: dict[str, Any] = {"Month": m_date, "Actual Ground Truth": f"{actual_val:,.0f}"}
                for m in selected_models:
                    if m in cache[selected_game]["test_forecasts"]:
                        pred_val = cache[selected_game]["test_forecasts"][m][h]
                        diff = pred_val - actual_val
                        pct_err = (abs(diff) / max(actual_val, 1)) * 100
                        m_row[f"{m} (Pred)"] = f"{pred_val:,.0f}"
                        m_row[f"{m} (Err %)"] = f"{pct_err:.1f}%"
                month_rows.append(m_row)

            breakdown_df = pd.DataFrame(month_rows)
            st.dataframe(breakdown_df, width="stretch")

            csv_bytes = breakdown_df.to_csv(index=False).encode("utf-8")
            game_slug = selected_game.lower().replace(" ", "_").replace(":", "").replace("'", "")
            st.download_button(
                label="Download Forecast Trajectories (CSV)",
                data=csv_bytes,
                file_name=f"{game_slug}_h{horizon}_forecasts.csv",
                mime="text/csv",
            )

    # -------------------------------------------------------------------------
    # TAB 2: Model Leaderboard & Multi-Step Error Growth
    # -------------------------------------------------------------------------
    with tab2:
        st.markdown(
            """
            <div class="steam-callout-box">
                <strong>Cross-Title Generalization Benchmark:</strong>
                Models were evaluated across all 7 games with strict chronological holdout partitions.
                MASE (Mean Absolute Scaled Error) &lt; 1.0 indicates performance superior to in-sample seasonal naive persistence.
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_l1, col_l2 = st.columns([1, 1])

        with col_l1:
            st.markdown(f"#### Per-Model Standings: {selected_game}")
            sub_game = test_benchmarks[test_benchmarks["Game"] == selected_game].sort_values("MASE")
            st.dataframe(
                sub_game[["Model", "Paradigm", "MAE", "RMSE", "MAPE", "sMAPE", "MASE"]].style.format({
                    "MAE": "{:,.1f}",
                    "RMSE": "{:,.1f}",
                    "MAPE": "{:.2f}%",
                    "sMAPE": "{:.2f}%",
                    "MASE": "{:.3f}",
                }),
                width="stretch",
            )

        with col_l2:
            st.markdown("#### Global Cross-Game Leaderboard (7 Games)")
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
                    width="stretch",
                )

        # Multi-Step Horizon Error Compounding
        st.divider()
        st.markdown("#### Multi-Step Error Compounding Across Forecast Steps (h = 1 to 12)")
        st.caption(
            "Inspect step-by-step error accumulation to evaluate autoregressive feedback drift vs direct sequence head stability."
        )
        err_chart = create_horizon_error_chart(
            game_cache=cache[selected_game],
            selected_models=selected_models,
            horizon=12,
        )
        st.altair_chart(err_chart, width="stretch")

        st.divider()
        st.markdown("#### Validation Tuning vs Test Out-of-Sample Drift")
        st.caption(
            "Comparing validation window error (2019-09 to 2020-08) against unblinded test window error (2020-09 to 2021-08)."
        )

        if not val_comp.empty:
            game_val_comp = val_comp[val_comp["Game"] == selected_game].sort_values("MASE_Test")
            st.dataframe(
                game_val_comp[[
                    "Model",
                    "Paradigm",
                    "MAE_Val",
                    "MAE_Test",
                    "MAPE_Val",
                    "MAPE_Test",
                    "MASE_Val",
                    "MASE_Test",
                    "MAE_Delta_Pct",
                ]].style.format({
                    "MAE_Val": "{:,.1f}",
                    "MAE_Test": "{:,.1f}",
                    "MAPE_Val": "{:.2f}%",
                    "MAPE_Test": "{:.2f}%",
                    "MASE_Val": "{:.3f}",
                    "MASE_Test": "{:.3f}",
                    "MAE_Delta_Pct": "{:+.1f}%",
                }),
                width="stretch",
            )

    # -------------------------------------------------------------------------
    # TAB 3: Historical Dynamics
    # -------------------------------------------------------------------------
    with tab3:
        st.markdown(f"### Historical Trajectory & Regime Shifts: {selected_game}")

        raw_df = load_game_raw_history(selected_game)

        if not raw_df.empty:
            fig_hist = plot_historical_dynamics(raw_df, selected_game)
            st.pyplot(fig_hist)

            col_d1, col_d2 = st.columns(2)
            with col_d1:
                with st.container(border=True):
                    st.markdown("**Statistical Profile**")
                    st.markdown(f"- **Total Monthly Records:** {len(raw_df)} months")
                    st.markdown(
                        f"- **Time Span:** {raw_df['Month_Year'].min().strftime('%Y-%m')} to {raw_df['Month_Year'].max().strftime('%Y-%m')}"
                    )
                    st.markdown(f"- **All-Time Average Players:** {raw_df['Avg_players'].mean():,.0f}")
                    st.markdown(f"- **All-Time Peak Players:** {raw_df['Peak_Players'].max():,.0f}")
                    st.markdown(f"- **Population Volatility (Std Dev):** {raw_df['Avg_players'].std():,.0f}")

            with col_d2:
                with st.container(border=True):
                    st.markdown("**Documented Structural Shocks**")
                    st.markdown(
                        "- **COVID-19 Lockdown Shock (Mar–May 2020):** Global stay-at-home orders drove unprecedented surges across all titles."
                    )
                    st.markdown(
                        "- **Steam Seasonal Sales:** Predictable annual peaks consistently recorded during Summer Sales (June/July) and Winter Sales (December)."
                    )
                    st.markdown(f"- **Title Specific:** {meta.get('notes', 'None recorded.')}")

    # -------------------------------------------------------------------------
    # TAB 4: Research Synthesis
    # -------------------------------------------------------------------------
    with tab4:
        st.markdown("### Empirical Findings & Architectural Takeaways")

        st.markdown(
            """
            > **Primary Research Question:**  
            > *Can modern recurrent deep learning models (LSTM, GRU) or gradient-boosted trees (XGBoost) reliably outperform classical econometric benchmarks (SARIMA) when forecasting monthly active player populations for top multiplayer games on Steam?*
            """
        )

        st.markdown(
            r"""
            #### Key Benchmark Insights

            1. **The Core Quartet Split (SARIMA 3 wins, GRU 3 wins, LSTM 1 win, XGBoost 0 wins):**
               * **SARIMA Dominates High-Cyclicality Games:** On games with strict annual cadence (*Dota 2*, *GTA V*, *Rust*), seasonal differencing ($s=12$) prevents error explosion and bounds long-term variance.
               * **GRU/LSTM Excel on Regime Shifts & Plateaus:** On titles stabilizing after explosive growth (*CS:GO*, *Rainbow Six Siege*, *Warframe*, *TF2*), recurrent gating mechanisms adaptively model non-linear churn where linear differencing degrades.
               * **XGBoost Struggles Under Recursive Long Horizons:** Autoregressive feature feedback cascades error step-by-step ($H=12$ mean MASE 0.923 vs SARIMA 0.672), highlighting the danger of recursive tree models for long-range planning.

            2. **The Power of Parsimonious Baselines:**
               * Across the complete 11-candidate arena, **Holt-Winters Exponential Smoothing** (Mean MASE: 0.526) and **L2-Regularized Ridge Regression** (Mean MASE: 0.561) posted the lowest overall cross-game error, proving that low-parameter regularized models remain crucial baselines for finite historical windows ($N < 100$ months).

            3. **Unprecedented Macro Shocks:**
               * The viral **Rust Twitch boom of January 2021 (+92% surge)** showed that pure autoregression cannot anticipate influencer-driven shocks; however, state-space models stabilized error faster post-shock than unconstrained tree ensembles.
            """
        )

        st.divider()
        st.markdown("#### Comparative Paradigm Tradeoff Matrix")

        tradeoff_data = [
            {
                "Paradigm": "Classical Econometric (SARIMA)",
                "Data Efficiency": "High (5/5)",
                "Multi-Step Stability": "Closed-Form (5/5)",
                "Non-Linear Flexibility": "Linear / Difference (2/5)",
                "Uncertainty Bounds": "Analytical 95% CI",
                "Best Use Case": "Stable games with annual sales and esports seasonality",
            },
            {
                "Paradigm": "Gradient Boosted ML (XGBoost)",
                "Data Efficiency": "Moderate (3/5)",
                "Multi-Step Stability": "Recursive Drift (2/5)",
                "Non-Linear Flexibility": "High (4/5)",
                "Uncertainty Bounds": "Quantile / Bootstrapped",
                "Best Use Case": "Short horizons (H <= 3) with rich exogenous telemetry",
            },
            {
                "Paradigm": "Recurrent Deep Learning (GRU / LSTM)",
                "Data Efficiency": "Requires Sequence Buffer (3/5)",
                "Multi-Step Stability": "Direct Sequence Head (4/5)",
                "Non-Linear Flexibility": "Non-Linear Memory (5/5)",
                "Uncertainty Bounds": "Monte Carlo Dropout",
                "Best Use Case": "Longer histories with complex structural plateaus",
            },
            {
                "Paradigm": "Exponential Smoothing (Holt-Winters)",
                "Data Efficiency": "Low Sample Requirement (5/5)",
                "Multi-Step Stability": "Damped Trend (4/5)",
                "Non-Linear Flexibility": "Linear Smoother (2/5)",
                "Uncertainty Bounds": "Analytical",
                "Best Use Case": "Quick, highly robust baseline across hundreds of games",
            },
        ]
        st.dataframe(pd.DataFrame(tradeoff_data), width="stretch")


if __name__ == "__main__":
    main()
