# Project Roadmap: Steam Player Forecasting

This document outlines the 5 execution phases of the **Steam Player Forecasting** project.
Each phase is provided as a **self-contained AI Agent Prompt** designed to be copied directly into a future coding-agent session (e.g. Antigravity, Claude Code, Cursor, Copilot Workspace).

---

## Overview of Phases

1. **Phase 1:** Dataset Acquisition & Validation
2. **Phase 2:** EDA & Time-Series Preprocessing
3. **Phase 3:** Statistical Forecasting
4. **Phase 4:** Machine Learning & Deep Learning
5. **Phase 5:** Final Evaluation, Report & Demo

---

## Phase 1 — Dataset Acquisition & Validation

### AI Agent Prompt

```markdown
You are an expert data engineer and time-series specialist. Your objective is to complete **Phase 1: Dataset Acquisition & Validation** for the `steam-player-forecasting` repository.

### Context & Rules:
1. Inspect the existing repository structure, `configs/default.yaml`, and `data/README.md` first.
2. Implement ONLY Phase 1. Do not start full exploratory data analysis (EDA) or forecasting models.
3. Preserve all existing repository files and architecture.
4. Add unit tests in `tests/` for data loading and validation logic.
5. Update documentation and commit all completed work.
6. Explicitly stop once Phase 1 is complete.

### Tasks:
1. **Dataset Ingestion Setup:**
   - Implement reproducible data ingestion logic in `src/steam_player_forecasting/data/` that looks for the Kaggle dataset (`jackogozaly/steam-player-data`) in `data/raw/` or downloads it via Kaggle API / curl / python fallback if credentials exist. Provide a synthetic fallback/sample generator if raw data is not immediately present in automated CI environments.
2. **Schema Inspection & Validation:**
   - Inspect the real raw schema (e.g., column names, date representations such as 'Month Year', target columns `Avg_players` and `Peak_Players`).
   - Clean and validate raw data: parse timestamps to standard monthly start (`YYYY-MM-01`), strip malformed characters/strings, and identify missing, duplicate, or invalid records.
3. **Representative Game Selection:**
   - Filter and select 5–10 representative games with continuous, long-term historical coverage (at least 3–5+ years of monthly records) across genres (e.g. Counter-Strike: Global Offensive, Dota 2, Team Fortress 2, Rust, Warframe, etc.).
4. **Standardized Data Pipeline:**
   - Save the cleaned, standardized subsets to `data/processed/` in a clean format (e.g. Parquet or CSV).
   - Expose reusable data loading and game filtering functions in `src/steam_player_forecasting/data/`.
5. **Testing & Documentation:**
   - Add unit tests verifying schema adherence, date formatting, and missing value handling.
   - Document dataset cleaning decisions and chosen games in `data/README.md`.
6. **Commit & Finish:**
   - Run test suite (`pytest`) to ensure all tests pass.
   - Commit changes with message `feat(data): complete phase 1 dataset acquisition and validation`.
   - Explicitly STOP and do NOT proceed to Phase 2.
```

---

## Phase 2 — EDA & Time-Series Preprocessing

### AI Agent Prompt

```markdown
You are an expert time-series data scientist. Your objective is to complete **Phase 2: EDA & Time-Series Preprocessing** for the `steam-player-forecasting` repository.

### Context & Rules:
1. Inspect the existing repository and Phase 1 standardized datasets in `data/processed/`.
2. Implement ONLY Phase 2. Do not train forecasting models yet.
3. Preserve previous work and adhere to leakage prevention rules (transformations/scalers fitted strictly on training data).
4. Add unit tests for preprocessing and transformation routines.
5. Update figures, notebooks, and documentation.
6. Explicitly stop once Phase 2 is complete.

### Tasks:
1. **Exploratory Data Analysis (EDA):**
   - Create a clean exploratory notebook `notebooks/01_eda_and_preprocessing.ipynb`.
   - Analyze player count dynamics across the selected 5–10 games: overall trends, seasonality (e.g. summer/winter Steam sales, holidays), rolling mean and variance, outliers, and structural breaks (e.g. major game updates).
   - Export high-resolution exploratory figures to `figures/` (e.g., historical player trajectories, seasonal subseries plots).
2. **Time-Series Decomposition & Stationarity:**
   - Perform STL or classical decomposition (Trend, Seasonal, Residual components).
   - Run statistical stationarity tests (Augmented Dickey-Fuller / KPSS) and compute autocorrelation (ACF) and partial autocorrelation (PACF).
3. **Leakage-Safe Preprocessing Pipeline:**
   - Implement modular time-series preprocessors in `src/steam_player_forecasting/features/`:
     - Differencing / log / Box-Cox transformations.
     - Train-only fitted scalers (MinMaxScaler / StandardScaler / RobustScaler).
     - Chronological train/validation/test split utility respecting `configs/default.yaml` (e.g., holdout last 12 months for test, prior 12 months for validation).
4. **Validation & Testing:**
   - Write unit tests in `tests/` validating that scalers fit only on training data and transformations invert accurately without leakage.
5. **Commit & Finish:**
   - Run `pytest` to verify all tests pass.
   - Commit changes with message `feat(features): complete phase 2 eda and time-series preprocessing`.
   - Explicitly STOP and do NOT proceed to Phase 3.
```

---

## Phase 3 — Statistical Forecasting

### AI Agent Prompt

```markdown
You are an expert econometrician and statistical forecaster. Your objective is to complete **Phase 3: Statistical Forecasting** for the `steam-player-forecasting` repository.

### Context & Rules:
1. Inspect the existing repository, preprocessing pipeline, and configs.
2. Implement ONLY Phase 3 statistical models. Do not implement ML (XGBoost) or DL (LSTM/GRU) models yet.
3. Preserve all previous work.
4. Adhere strictly to chronological evaluation (train on historical window, evaluate on validation window, keeping test window untouched).
5. Add unit tests for all statistical model implementations.
6. Explicitly stop once Phase 3 is complete.

### Tasks:
1. **Statistical Forecasting Baselines:**
   - Implement reusable model wrappers in `src/steam_player_forecasting/models/`:
     - Naive (last observed value) & Seasonal Naive (value from same month prior year).
     - Exponential Smoothing / Holt-Winters (additive & multiplicative trend/seasonality).
     - Autoregressive (AR), ARIMA, and SARIMA (using `statsmodels`).
2. **Model Diagnostics & Hyperparameter Selection:**
   - Analyze AIC/BIC criteria and ACF/PACF residual diagnostics (Ljung-Box test for white noise residuals).
   - Select the optimal SARIMA `(p, d, q) x (P, D, Q, s)` specification per game on the validation window.
3. **Benchmarking & Forecast Generation:**
   - Generate multi-step forecasts for the validation horizon (e.g. 12 months ahead).
   - Compute evaluation metrics (MAE, RMSE, MAPE, sMAPE, MASE) using `steam_player_forecasting.evaluation`.
   - Save comparison benchmark tables to `report/` and forecast comparison plots to `figures/`.
   - Create demonstration notebook `notebooks/02_statistical_forecasting.ipynb`.
4. **Testing & Verification:**
   - Add unit tests verifying forecast shapes, inverse transformation handling, and metric outputs.
5. **Commit & Finish:**
   - Run `pytest` to confirm tests pass.
   - Commit changes with message `feat(models): complete phase 3 statistical forecasting baselines`.
   - Explicitly STOP and do NOT proceed to Phase 4.
```

---

## Phase 4 — Machine Learning & Deep Learning

### AI Agent Prompt

```markdown
You are an expert machine learning and deep learning engineer. Your objective is to complete **Phase 4: Machine Learning & Deep Learning** for the `steam-player-forecasting` repository.

### Context & Rules:
1. Inspect the existing repository, features pipeline, and Phase 3 statistical benchmark results.
2. Implement ONLY Phase 4. Do not perform final test-set unblinding or final project wrap-up yet.
3. Preserve all existing work.
4. Prevent data leakage strictly: tabular lag features and recurrent sequences must be constructed chronologically without future leakage.
5. Keep deep learning models computationally lightweight and CPU-runnable.
6. Explicitly stop once Phase 4 is complete.

### Tasks:
1. **Feature Engineering for Tabular ML:**
   - Implement leakage-safe lag features (`t-1`, `t-2`, `t-12`), rolling window statistics (rolling mean, rolling std over 3/6/12 months), and calendar indicators in `src/steam_player_forecasting/features/`.
2. **Machine Learning Baselines:**
   - Implement tabular regression models in `src/steam_player_forecasting/models/`:
     - Ridge Regression
     - Random Forest Regressor
     - XGBoost Regressor
   - Train on training split and tune hyperparameters via chronological validation / TimeSeriesSplit.
3. **Deep Learning Architectures (PyTorch):**
   - Implement PyTorch recurrent time-series architectures in `src/steam_player_forecasting/models/`:
     - Standard Recurrent Neural Network (RNN)
     - Long Short-Term Memory (LSTM)
     - Gated Recurrent Unit (GRU)
   - Implement PyTorch Dataset / DataLoader with sliding window sequences (`seq_length=12` predicting `horizon=12`).
   - Include early stopping, learning rate scheduling, and fixed random seeds (`seed=42`).
4. **Intermediate Benchmark Comparison:**
   - Compare validation performance of the core quartet:
     **SARIMA vs XGBoost vs LSTM vs GRU**
   - Create comparison notebook `notebooks/03_ml_and_dl_forecasting.ipynb`.
   - Save validation metrics tables to `report/` and forecast comparison charts to `figures/`.
5. **Testing & Verification:**
   - Add unit tests for feature matrix generation, PyTorch model forward passes, and output tensor shapes.
6. **Commit & Finish:**
   - Run `pytest` to ensure all tests pass.
   - Commit changes with message `feat(models): complete phase 4 machine learning and deep learning forecasting`.
   - Explicitly STOP and do NOT proceed to Phase 5.
```

---

## Phase 5 — Final Evaluation, Report & Demo

### AI Agent Prompt

```markdown
You are a lead AI research scientist and full-stack data presenter. Your objective is to complete **Phase 5: Final Evaluation, Report & Demo** for the `steam-player-forecasting` repository.

### Context & Rules:
1. Inspect the entire repository, models, and results from Phases 1–4.
2. Implement Phase 5: conduct the final unblinded evaluation on the untouched test period, synthesize findings, finalize reports, and build a lightweight interactive demo.
3. Preserve all existing functionality and ensure 100% test passing and reproducibility.
4. This phase completes the project.

### Tasks:
1. **Final Test-Set Evaluation:**
   - Evaluate all candidate models on the held-out test period (untouched until now):
     - Baselines: Naive, Seasonal Naive, Holt-Winters, ARIMA, Ridge, Random Forest.
     - Core Quartet: **SARIMA vs XGBoost vs LSTM vs GRU**.
   - Generate final standardized test metric tables (MAE, RMSE, MAPE, sMAPE, MASE) across all games.
2. **In-Depth Diagnostic & Qualitative Analysis:**
   - Analyze strengths and weaknesses of each paradigm:
     - Where do statistical models outperform DL? (e.g., data-constrained regimes, smooth seasonal cycles).
     - Where do ML/DL models excel or struggle? (e.g., non-linear shifts, sudden popularity spikes, Steam sales, major DLCs).
   - Generate final publication-quality comparison charts in `figures/`.
3. **Interactive Demo (Streamlit):**
   - Build a lightweight interactive demonstration app in `app/app.py`:
     - Allow selecting any of the analyzed Steam games.
     - Toggle between models (SARIMA, XGBoost, LSTM, GRU).
     - Visualize historical player counts, predicted forecast trajectories, and confidence intervals.
     - Display model performance metrics interactively.
4. **Final Comprehensive Report & README Update:**
   - Write a detailed executive technical report in `report/final_report.md`.
   - Update `README.md` with final leaderboard, key takeaways, reproducibility instructions, and demo launch guide (`streamlit run app/app.py`).
5. **Cleanup & Final Verification:**
   - Ensure clean code style, full test coverage across `tests/`, and working CI pipeline.
   - Verify that all outputs can be reproduced from scratch using documented commands.
6. **Commit & Finish:**
   - Run `pytest` and verify clean working tree.
   - Commit changes with message `feat(report): complete phase 5 final evaluation, report, and interactive demo`.
   - Report final completion status.
```
