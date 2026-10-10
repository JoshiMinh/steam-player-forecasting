"""Steam Store styled card components: Game Capsule and Dynamic Metric Scorecards."""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
import streamlit as st


def compute_dynamic_metrics(
    actuals: list[float] | np.ndarray,
    forecasts_dict: dict[str, list[float]],
    seasonal_scale: float,
    horizon: int = 12,
    paradigm_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Compute out-of-sample metrics dynamically for arbitrary horizon step h=1..H."""
    y_act = np.array(actuals)[:horizon]
    rows: list[dict[str, Any]] = []

    for model, preds in forecasts_dict.items():
        y_pred = np.array(preds)[:horizon]
        diff = np.abs(y_act - y_pred)
        mae = float(np.mean(diff))
        rmse = float(np.sqrt(np.mean((y_act - y_pred) ** 2)))
        mape = float(np.mean(diff / np.maximum(y_act, 1)) * 100)
        smape = float(np.mean(2 * diff / np.maximum(np.abs(y_act) + np.abs(y_pred), 1)) * 100)
        mase = float(mae / seasonal_scale) if seasonal_scale > 0 else 0.0

        rows.append({
            "Model": model,
            "Paradigm": paradigm_map.get(model, "Benchmark") if paradigm_map else "Model",
            "MAE": mae,
            "RMSE": rmse,
            "MAPE": mape,
            "sMAPE": smape,
            "MASE": mase,
        })

    return pd.DataFrame(rows)


def render_game_capsule(game_name: str, meta: dict[str, str], game_cache: dict[str, Any]) -> None:
    """Render game capsule banner without emojis."""
    genre = meta.get("genre", "Multiplayer")
    release = meta.get("release", "Steam")
    notes = meta.get("notes", "No structural break documentation recorded.")
    sarima_order = game_cache.get("optimal_sarima_order", "(p,d,q)x(P,D,Q)12")
    xgb_params = game_cache.get("optimal_xgb_params", {})
    xgb_str = (
        f"depth={xgb_params.get('max_depth', 3)}, lr={xgb_params.get('learning_rate', 0.05)}"
        if xgb_params
        else "Standard"
    )

    with st.container(border=True):
        col_title, col_tags = st.columns([2, 1])
        with col_title:
            st.markdown(f"### {game_name}")
        with col_tags:
            st.markdown(
                f"<div style='text-align: right; display: flex; gap: 6px; justify-content: flex-end; flex-wrap: wrap;'>"
                f"<span style='background: rgba(102,192,244,0.15); color: #66c0f4; border: 1px solid rgba(102,192,244,0.3); font-size: 0.72rem; font-weight: 700; padding: 2px 8px; border-radius: 3px; text-transform: uppercase;'>{genre}</span>"
                f"<span style='background: rgba(102,192,244,0.15); color: #66c0f4; border: 1px solid rgba(102,192,244,0.3); font-size: 0.72rem; font-weight: 700; padding: 2px 8px; border-radius: 3px; text-transform: uppercase;'>Release: {release}</span>"
                f"<span style='background: rgba(102,192,244,0.15); color: #66c0f4; border: 1px solid rgba(102,192,244,0.3); font-size: 0.72rem; font-weight: 700; padding: 2px 8px; border-radius: 3px; text-transform: uppercase;'>12-Mo Horizon</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
        st.markdown(f"**Historical Context & Regime Shifts:** {notes}")
        st.markdown(
            f"<div style='border-top: 1px solid #2a475e; padding-top: 6px; margin-top: 6px; display: flex; gap: 20px; font-size: 0.85rem; color: #8f98a0;'>"
            f"<span><strong>Optimal SARIMA:</strong> <code style='color: #66c0f4; background: rgba(0,0,0,0.3); padding: 2px 6px; border-radius: 2px;'>{sarima_order}</code></span>"
            f"<span><strong>Tuned XGBoost:</strong> <code style='color: #e5c158; background: rgba(0,0,0,0.3); padding: 2px 6px; border-radius: 2px;'>{xgb_str}</code></span>"
            f"</div>",
            unsafe_allow_html=True,
        )


def render_scorecard(
    game_metrics: pd.DataFrame,
    rank_by: str = "MASE",
    palette: dict[str, str] | None = None,
) -> None:
    """Render dynamically ranked scorecards based on user-selected metric."""
    if game_metrics.empty:
        return

    palette = palette or {}

    # Sort ascending (lower is better for all forecasting error metrics)
    sorted_df = game_metrics.sort_values(rank_by, ascending=True).reset_index(drop=True)
    best_row = sorted_df.iloc[0]
    best_model_name = best_row["Model"]

    # Top Performer Ribbon with selected metric highlight
    metric_val_str = f"{best_row[rank_by]:.3f}" if rank_by == "MASE" else (
        f"{best_row[rank_by]:.2f}%" if "MAPE" in rank_by else f"{best_row[rank_by]:,.1f}"
    )

    st.markdown(
        f"""<div style="background: rgba(164, 208, 7, 0.12); border: 1px solid #a4d007; border-radius: 4px; padding: 10px 14px; margin-bottom: 12px; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
    <div style="display: flex; align-items: center; gap: 10px;">
        <span style="background: #a4d007; color: #111111; font-weight: 800; font-size: 0.75rem; padding: 2px 7px; border-radius: 2px; text-transform: uppercase;">Top Performer ({rank_by})</span>
        <span style="font-size: 1.05rem; font-weight: 700; color: #ffffff;">{best_model_name}</span>
        <span style="font-size: 0.85rem; color: #c7d5e0;">({best_row['Paradigm']})</span>
    </div>
    <div style="font-size: 0.88rem; color: #a4d007; font-weight: 700;">
        {rank_by}: {metric_val_str} &nbsp;|&nbsp; MAE: {best_row['MAE']:,.1f} &nbsp;|&nbsp; MAPE: {best_row['MAPE']:.2f}% &nbsp;|&nbsp; MASE: {best_row['MASE']:.3f}
    </div>
</div>""",
        unsafe_allow_html=True,
    )

    # Individual model cards in responsive grid
    num_cols = min(len(sorted_df), 4)
    cols = st.columns(num_cols)

    for i, (_, row) in enumerate(sorted_df.iterrows()):
        col = cols[i % len(cols)]
        is_winner = row["Model"] == best_model_name
        accent = palette.get(row["Model"], "#66c0f4")

        with col:
            with st.container(border=True):
                col_name, col_badge = st.columns([3, 1])
                with col_name:
                    st.markdown(
                        f"<span style='color: {accent}; font-weight: 700; font-size: 1.05rem;'>{row['Model']}</span>",
                        unsafe_allow_html=True,
                    )
                with col_badge:
                    if is_winner:
                        st.badge("Winner", color="green")
                st.caption(row["Paradigm"])

                # Highlight the user-selected ranking metric in cyan
                def fmt_metric(name: str, val: float, is_pct: bool = False, is_mase: bool = False) -> str:
                    formatted = f"{val:.3f}" if is_mase else (f"{val:.2f}%" if is_pct else f"{val:,.1f}")
                    if name == rank_by:
                        return f"<strong style='color: #66c0f4;'>{name}: {formatted}</strong>"
                    return f"<span>{name}: {formatted}</span>"

                st.markdown(
                    f"<div style='font-size: 0.88rem; line-height: 1.7;'>"
                    f"<div>{fmt_metric('MAE', row['MAE'])}</div>"
                    f"<div>{fmt_metric('RMSE', row['RMSE'])}</div>"
                    f"<div>{fmt_metric('MAPE', row['MAPE'], is_pct=True)}</div>"
                    f"<div>{fmt_metric('MASE', row['MASE'], is_mase=True)}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
