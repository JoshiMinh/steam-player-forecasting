"""Build demonstration notebook notebooks/04_final_evaluation_and_synthesis.ipynb."""

from __future__ import annotations

import json
from pathlib import Path

from steam_player_forecasting.data.loader import get_project_root


def create_markdown_cell(source: str) -> dict:
    """Create a Jupyter notebook markdown cell."""
    lines = [line + "\n" for line in source.strip().split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines,
    }


def create_code_cell(source: str, outputs: list | None = None, execution_count: int | None = None) -> dict:
    """Create a Jupyter notebook code cell."""
    lines = [line + "\n" for line in source.strip().split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": "code",
        "execution_count": execution_count,
        "metadata": {},
        "outputs": outputs or [],
        "source": lines,
    }


def build_notebook() -> None:
    """Build and write notebooks/04_final_evaluation_and_synthesis.ipynb."""
    root = get_project_root()
    notebook_path = root / "notebooks" / "04_final_evaluation_and_synthesis.ipynb"
    notebook_path.parent.mkdir(parents=True, exist_ok=True)

    cells = []

    # Cell 1: Title & Abstract
    cells.append(
        create_markdown_cell(
            """# Phase 5: Final Evaluation, Empirical Synthesis & Interactive Demo
### Steam Player Forecasting Project

**Author:** JoshiMinh  
**Scope:** Phase 5 (Final Project Synthesis & Unblinded Test Evaluation)  
**Dataset:** Cleaned Steam Monthly Player Concurrency (`data/processed/steam_games_monthly.csv`)  
**Holdout Test Period:** Multi-step ($H=12$) strictly chronological test window (`2020-09-01` to `2021-08-01`)  
**Paradigms Evaluated:** Classical Econometrics (SARIMA, ARIMA, Holt-Winters) vs Gradient Boosted Trees (XGBoost, RF) vs Recurrent Deep Learning (GRU, LSTM, RNN)

---

## 1. Executive Research Summary & Core Question

> **Core Research Question:**  
> *Can modern recurrent deep learning models (LSTM, GRU) or gradient-boosted trees (XGBoost) reliably outperform classical econometric benchmarks (SARIMA) when forecasting monthly active player populations for top multiplayer games on Steam?*

In this final phase, all models tuned during the validation phase (Phases 3 and 4) are evaluated on the **previously untouched 12-month holdout test set** (`2020-09-01` to `2021-08-01`). This unblinded evaluation provides definitive answers across diverse multiplayer genres, structural shocks (COVID-19 lockdowns, Twitch influencer surges), and data regimes.
"""
        )
    )

    # Cell 2: Setup & Environment
    cells.append(
        create_code_cell(
            """import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Project imports
from steam_player_forecasting.data.loader import get_available_games, get_project_root, load_processed_data

root = get_project_root()
print(f"Project Root: {root}")
print(f"Evaluated Games ({len(get_available_games())}): {get_available_games()}")
"""
        )
    )

    # Cell 3: Load Final Benchmark Tables
    cells.append(
        create_markdown_cell(
            """## 2. Final Test-Set Benchmark Tables

All candidate models were fitted on all chronological history up to `2020-08-01` (`Train + Val`) with zero lookahead test leakage.
Standardized evaluation metrics (**MAE**, **RMSE**, **MAPE**, **sMAPE**, **MASE**) are computed across the $H=12$ test horizon.
"""
        )
    )

    cells.append(
        create_code_cell(
            """leaderboard_df = pd.read_csv(root / "report" / "final_leaderboard.csv")
display(leaderboard_df)
"""
        )
    )

    # Cell 4: Core Quartet Test Standings
    cells.append(
        create_markdown_cell(
            """## 3. Core Quartet Head-to-Head Comparison

Head-to-head comparison of the four flagship paradigms:
**SARIMA (Econometric)** vs **XGBoost (Gradient Boosted ML)** vs **LSTM (Recurrent DL)** vs **GRU (Recurrent DL)**.
"""
        )
    )

    cells.append(
        create_code_cell(
            """core_df = pd.read_csv(root / "report" / "core_quartet_test_benchmarks.csv")
print("=== CORE QUARTET TEST PERFORMANCE BY GAME ===")
display(core_df.pivot(index="Game", columns="Model", values="MAE"))
"""
        )
    )

    # Cell 5: Winner Summary
    cells.append(
        create_code_cell(
            """# Per-Game Winner Breakdown
best_core = core_df.loc[core_df.groupby("Game")["MAE"].idxmin()]
winner_summary = best_core[["Game", "Model", "Paradigm", "MAE", "RMSE", "MAPE", "MASE"]]
print("=== PER-GAME PARADIGM WINNERS ===")
display(winner_summary)
"""
        )
    )

    # Cell 6: Publication Figures
    cells.append(
        create_markdown_cell(
            """## 4. Publication-Quality Diagnostic Figures

### Figure 17: Final Test Forecast Trajectories (Core Quartet)
Shows the multi-step $H=12$ test forecasts against actual ground truth across all 7 titles.
"""
        )
    )

    cells.append(
        create_code_cell(
            """fig_path = root / "figures" / "17_final_test_forecast_comparisons.png"
if fig_path.exists():
    img = mpimg.imread(str(fig_path))
    plt.figure(figsize=(16, 14), dpi=100)
    plt.imshow(img)
    plt.axis("off")
    plt.show()
"""
        )
    )

    # Cell 7: Leaderboard Metrics
    cells.append(
        create_markdown_cell(
            """### Figure 18: Final Leaderboard & Paradigm Performance Synthesis
Comparative analysis of MASE, MAPE, per-game MAE, and winner distribution across paradigms.
"""
        )
    )

    cells.append(
        create_code_cell(
            """fig_path = root / "figures" / "18_final_leaderboard_metrics.png"
if fig_path.exists():
    img = mpimg.imread(str(fig_path))
    plt.figure(figsize=(15, 11), dpi=100)
    plt.imshow(img)
    plt.axis("off")
    plt.show()
"""
        )
    )

    # Cell 8: Horizon Error Degradation
    cells.append(
        create_markdown_cell(
            """### Figure 19: Forecast Horizon Error Degradation ($h = 1$ to $12$)
Demonstrating error compounding in recursive ML vs sequence-to-sequence deep learning and econometric dampening.
"""
        )
    )

    cells.append(
        create_code_cell(
            """fig_path = root / "figures" / "19_horizon_error_degradation.png"
if fig_path.exists():
    img = mpimg.imread(str(fig_path))
    plt.figure(figsize=(15, 6), dpi=100)
    plt.imshow(img)
    plt.axis("off")
    plt.show()
"""
        )
    )

    # Cell 9: Qualitative Case Studies
    cells.append(
        create_markdown_cell(
            """### Figure 20: Qualitative Case Studies across Structural Breaks
Deep-dive into Rust's Twitch streamer explosion (Jan 2021), CS:GO post-lockdown plateaus, R6 Siege data constraints, and TF2 seasonal stability.
"""
        )
    )

    cells.append(
        create_code_cell(
            """fig_path = root / "figures" / "20_paradigm_case_studies.png"
if fig_path.exists():
    img = mpimg.imread(str(fig_path))
    plt.figure(figsize=(16, 12), dpi=100)
    plt.imshow(img)
    plt.axis("off")
    plt.show()
"""
        )
    )

    # Cell 10: Research Conclusions & Synthesis
    cells.append(
        create_markdown_cell(
            """## 5. Research Conclusions & Strategic Takeaways

### Core Answer to Research Question:
1. **No Single Paradigm Universally Dominates:**
   - In the Core Quartet, **SARIMA won on 3 games** (*Dota 2*, *GTA V*, *Rust*), **GRU won on 3 games** (*CS:GO*, *R6 Siege*, *Warframe*), and **LSTM won on 1 game** (*TF2*).
   - XGBoost did not win any title on the test set, largely due to compounding recursive forecasting errors across the long 12-month horizon.
2. **Where Econometric Models (SARIMA / Holt-Winters) Outperform:**
   - Regimes with well-defined seasonal periodicity and smooth cyclical trends (*Dota 2*, *GTA V*).
   - Constrained sample sizes where deep networks risk overfitting.
   - Closed-form mean reversion and analytical confidence bounds provide unmatched stability.
3. **Where Recurrent Neural Networks (GRU / LSTM) Outperform:**
   - Non-linear stabilization and high player retention plateaus (*CS:GO*, *Rainbow Six Siege*).
   - Complex multi-scale seasonal patterns that exceed linear polynomial representations.
4. **Interactive Demo:**
   Launch the interactive Streamlit application to explore live forecasts:
   ```bash
   streamlit run app/app.py
   ```
"""
        )
    )

    notebook_data = {
        "cells": cells,
        "metadata": {
            "language_info": {
                "name": "python",
                "version": "3.11.0",
            },
            "orig_nbformat": 4,
        },
        "nbformat": 4,
        "nbformat_minor": 2,
    }

    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(notebook_data, f, indent=2)

    print(f"Successfully created notebook at: {notebook_path}")


if __name__ == "__main__":
    build_notebook()
