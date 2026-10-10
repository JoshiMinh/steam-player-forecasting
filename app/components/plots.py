"""Interactive plotting engine for time-series forecasting using Altair and Matplotlib."""

from __future__ import annotations

from typing import Any
import altair as alt
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Steam Store Brand Palette
STEAM_PALETTE: dict[str, str] = {
    "Historical": "#8f98a0",
    "Actual Ground Truth": "#ffffff",
    "SARIMA": "#66c0f4",          # Steam Electric Cyan
    "XGBoost": "#e5c158",         # Steam Amber Gold
    "LSTM": "#a4d007",            # Steam Discount Green
    "GRU": "#c084fc",             # Steam Vivid Lavender
    "Holt-Winters": "#38bdf8",     # Sky Blue
    "Ridge": "#f472b6",           # Rose
    "Random Forest": "#fb923c",   # Orange
    "RNN": "#facc15",             # Yellow
    "Seasonal Naive": "#94a3b8",  # Slate
    "Naive": "#64748b",           # Dark Slate
    "ARIMA(1,1,1)": "#fdba74",    # Light Orange
    "Simulated Shock": "#f43f5e", # Rose Red
}


def apply_steam_chart_style(ax: plt.Axes, fig: plt.Figure) -> None:
    """Apply Steam Store dark navy aesthetic to Matplotlib figure and axes."""
    fig.patch.set_facecolor("#16202d")
    ax.set_facecolor("#0e141b")

    for spine in ax.spines.values():
        spine.set_color("#2a475e")
        spine.set_linewidth(1.0)

    ax.tick_params(colors="#c7d5e0", labelsize=9)
    ax.grid(True, color="#2a475e", alpha=0.45, linestyle="--", linewidth=0.8)
    ax.xaxis.label.set_color("#c7d5e0")
    ax.yaxis.label.set_color("#c7d5e0")
    ax.title.set_color("#ffffff")


def create_interactive_forecast_chart(
    game_cache: dict[str, Any],
    selected_models: list[str],
    horizon: int = 12,
    history_window: int = 24,
    show_ci: bool = True,
    demand_shock_pct: float = 0.0,
    shock_target_model: str = "",
) -> alt.LayerChart:
    """Create interactive Altair forecast chart with tooltips, zoom, pan, and CI."""
    hist_dates = pd.to_datetime(game_cache["history_dates"])[-history_window:]
    hist_vals = game_cache["history_values"][-history_window:]
    test_dates = pd.to_datetime(game_cache["test_dates"])[:horizon]
    test_actuals = game_cache["test_actuals"][:horizon]

    records: list[dict[str, Any]] = []

    # 1. Historical observations
    for dt, val in zip(hist_dates, hist_vals):
        records.append({
            "Date": dt,
            "Series": "Historical",
            "Players": float(val),
            "Actual": float(val),
            "Error_Pct": 0.0,
            "Type": "History",
        })

    # 2. Actual Ground Truth
    for dt, act in zip(test_dates, test_actuals):
        records.append({
            "Date": dt,
            "Series": "Actual Ground Truth",
            "Players": float(act),
            "Actual": float(act),
            "Error_Pct": 0.0,
            "Type": "Actual",
        })

    # 3. Model forecasts
    color_domain = ["Historical", "Actual Ground Truth"]
    color_range = [STEAM_PALETTE["Historical"], STEAM_PALETTE["Actual Ground Truth"]]

    for model in selected_models:
        if model in game_cache["test_forecasts"]:
            preds = game_cache["test_forecasts"][model][:horizon]
            series_name = f"{model} Forecast"
            color_domain.append(series_name)
            color_range.append(STEAM_PALETTE.get(model, "#66c0f4"))

            for dt, act, pred in zip(test_dates, test_actuals, preds):
                err_pct = abs(pred - act) / max(act, 1) * 100
                records.append({
                    "Date": dt,
                    "Series": series_name,
                    "Players": float(pred),
                    "Actual": float(act),
                    "Error_Pct": round(float(err_pct), 1),
                    "Type": "Forecast",
                })

    # 4. Optional Simulated Demand Shock
    if demand_shock_pct != 0.0:
        target = shock_target_model if shock_target_model in game_cache["test_forecasts"] else (
            selected_models[0] if selected_models else ""
        )
        if target and target in game_cache["test_forecasts"]:
            shock_preds = [
                float(p * (1.0 + demand_shock_pct / 100.0))
                for p in game_cache["test_forecasts"][target][:horizon]
            ]
            shock_series = f"Shock ({demand_shock_pct:+.0f}%)"
            color_domain.append(shock_series)
            color_range.append(STEAM_PALETTE["Simulated Shock"])

            for dt, act, sp in zip(test_dates, test_actuals, shock_preds):
                err_pct = abs(sp - act) / max(act, 1) * 100
                records.append({
                    "Date": dt,
                    "Series": shock_series,
                    "Players": float(sp),
                    "Actual": float(act),
                    "Error_Pct": round(float(err_pct), 1),
                    "Type": "Shock",
                })

    df = pd.DataFrame(records)

    # Base line chart
    lines = alt.Chart(df).mark_line(point=True, strokeWidth=2.2).encode(
        x=alt.X("Date:T", title="Observation Month", axis=alt.Axis(format="%b %Y")),
        y=alt.Y("Players:Q", title="Monthly Average Players", axis=alt.Axis(format=",.0f")),
        color=alt.Color(
            "Series:N",
            scale=alt.Scale(domain=color_domain, range=color_range),
            legend=alt.Legend(title="Series", orient="top", columns=3),
        ),
        strokeDash=alt.condition(
            alt.datum.Type == "History",
            alt.value([4, 3]),
            alt.value([0]),
        ),
        tooltip=[
            alt.Tooltip("Date:T", title="Month", format="%b %Y"),
            alt.Tooltip("Series:N", title="Series"),
            alt.Tooltip("Players:Q", title="Players", format=",.0f"),
            alt.Tooltip("Error_Pct:Q", title="Error %", format=".1f"),
        ],
    )

    layers: list[alt.Chart] = []

    # 5. SARIMA 95% Confidence Interval Band Layer
    if show_ci and "SARIMA" in selected_models:
        ci_lower = game_cache.get("sarima_ci_lower", [])[:horizon]
        ci_upper = game_cache.get("sarima_ci_upper", [])[:horizon]
        if len(ci_lower) > 0 and len(ci_upper) > 0:
            ci_df = pd.DataFrame({
                "Date": test_dates,
                "Lower": [float(x) for x in ci_lower],
                "Upper": [float(x) for x in ci_upper],
            })
            ci_band = alt.Chart(ci_df).mark_area(
                opacity=0.18,
                color=STEAM_PALETTE["SARIMA"],
            ).encode(
                x="Date:T",
                y="Lower:Q",
                y2="Upper:Q",
            )
            layers.append(ci_band)

    # 6. Test Horizon Boundary Line
    boundary_df = pd.DataFrame([{"Date": test_dates[0]}])
    boundary_rule = alt.Chart(boundary_df).mark_rule(
        color="#f87171",
        strokeDash=[5, 4],
        strokeWidth=1.5,
    ).encode(x="Date:T")
    layers.append(boundary_rule)

    layers.append(lines)

    chart = alt.layer(*layers).resolve_scale(
        y="shared"
    ).properties(
        height=440,
    ).configure_view(
        strokeWidth=0,
        fill="#0e141b",
    ).configure_axis(
        gridColor="#2a475e",
        gridOpacity=0.35,
        domainColor="#2a475e",
        tickColor="#2a475e",
        labelColor="#c7d5e0",
        titleColor="#ffffff",
        labelFontSize=10,
        titleFontSize=11,
    ).configure_legend(
        labelColor="#c7d5e0",
        titleColor="#ffffff",
        fillColor="#171a21",
        strokeColor="#2a475e",
        cornerRadius=4,
        padding=8,
    ).interactive()

    return chart


def create_horizon_error_chart(
    game_cache: dict[str, Any],
    selected_models: list[str],
    horizon: int = 12,
) -> alt.Chart:
    """Create interactive error growth curve across forecast steps h=1..H."""
    test_actuals = np.array(game_cache["test_actuals"])[:horizon]
    test_dates = pd.to_datetime(game_cache["test_dates"])[:horizon]

    records: list[dict[str, Any]] = []
    color_domain: list[str] = []
    color_range: list[str] = []

    for model in selected_models:
        if model in game_cache["test_forecasts"]:
            preds = np.array(game_cache["test_forecasts"][model])[:horizon]
            color_domain.append(model)
            color_range.append(STEAM_PALETTE.get(model, "#66c0f4"))

            for h in range(len(test_actuals)):
                act = test_actuals[h]
                pred = preds[h]
                abs_err = abs(pred - act)
                pct_err = (abs_err / max(act, 1)) * 100
                records.append({
                    "Step": h + 1,
                    "Month": test_dates[h].strftime("%Y-%m"),
                    "Model": model,
                    "MAE": round(float(abs_err), 1),
                    "MAPE": round(float(pct_err), 2),
                })

    df = pd.DataFrame(records)

    chart = alt.Chart(df).mark_line(point=True, strokeWidth=2.0).encode(
        x=alt.X("Step:O", title="Horizon Step (h = 1 to 12)"),
        y=alt.Y("MAPE:Q", title="Absolute Percentage Error (%)", axis=alt.Axis(format=".1f")),
        color=alt.Color(
            "Model:N",
            scale=alt.Scale(domain=color_domain, range=color_range),
            legend=alt.Legend(title="Model", orient="top"),
        ),
        tooltip=[
            alt.Tooltip("Step:O", title="Step h"),
            alt.Tooltip("Month:N", title="Month"),
            alt.Tooltip("Model:N", title="Model"),
            alt.Tooltip("MAPE:Q", title="Error %", format=".2f"),
            alt.Tooltip("MAE:Q", title="Absolute Diff", format=",.0f"),
        ],
    ).properties(
        height=320,
    ).configure_view(
        strokeWidth=0,
        fill="#0e141b",
    ).configure_axis(
        gridColor="#2a475e",
        gridOpacity=0.35,
        domainColor="#2a475e",
        tickColor="#2a475e",
        labelColor="#c7d5e0",
        titleColor="#ffffff",
    ).configure_legend(
        labelColor="#c7d5e0",
        titleColor="#ffffff",
        fillColor="#171a21",
        strokeColor="#2a475e",
    ).interactive()

    return chart


def plot_historical_dynamics(raw_df: pd.DataFrame, game_name: str) -> plt.Figure:
    """Generate Steam-themed two-panel historical dynamics plot."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True)
    apply_steam_chart_style(ax1, fig)
    apply_steam_chart_style(ax2, fig)

    ax1.plot(
        raw_df["Month_Year"],
        raw_df["Avg_players"],
        color="#66c0f4",
        lw=2.2,
        label="Monthly Avg Players",
    )
    if "Peak_Players" in raw_df.columns:
        ax1.plot(
            raw_df["Month_Year"],
            raw_df["Peak_Players"],
            color="#e5c158",
            lw=1.5,
            ls="--",
            alpha=0.85,
            label="Monthly Peak Players",
        )

    ax1.set_title(f"{game_name} — Historical Population Trajectory (Avg vs Peak)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Players", fontsize=10)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:,.0f}"))
    ax1.legend(loc="upper left", facecolor="#171a21", edgecolor="#2a475e", labelcolor="#c7d5e0")

    if "Peak_Players" in raw_df.columns and "Avg_players" in raw_df.columns:
        ratio = raw_df["Peak_Players"] / np.maximum(raw_df["Avg_players"], 1)
        ax2.plot(raw_df["Month_Year"], ratio, color="#c084fc", lw=1.8, marker=".", ms=4, label="Peak / Avg Ratio")
        ax2.axhline(
            ratio.mean(),
            color="#a4d007",
            ls=":",
            lw=1.5,
            label=f"All-Time Mean Ratio: {ratio.mean():.2f}",
        )
        ax2.set_title("Peak-to-Average Volatility Ratio (Heteroscedasticity & Surge Factor)", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Ratio", fontsize=10)
        ax2.set_xlabel("Observation Month", fontsize=10)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax2.tick_params(axis="x", rotation=25)
        ax2.legend(loc="upper left", facecolor="#171a21", edgecolor="#2a475e", labelcolor="#c7d5e0")

    plt.tight_layout()
    return fig
