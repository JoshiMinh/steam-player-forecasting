# Project Roadmap: Steam Player Forecasting

This document outlines the 5 execution phases of the **Steam Player Forecasting** project.
Each phase is provided as a **self-contained AI Agent Prompt** designed to be copied directly into a future coding-agent session (e.g. Antigravity, Claude Code, Cursor, Copilot Workspace).

---

## Overview of Phases

1. [X]  **Phase 1:** Dataset Acquisition & Validation *(Completed)*
2. [X]  **Phase 2:** EDA & Time-Series Preprocessing *(Completed)*
3. [X]  **Phase 3:** Statistical Forecasting *(Completed)*
4. [X]  **Phase 4:** Machine Learning & Deep Learning *(Completed)*
5. [X]  **Phase 5:** Final Evaluation, Report & Demo *(Completed)*

---

## Phase 1 — Dataset Acquisition & Validation

**Status:** Completed `[x]` (Commit: `d0174fa` | `feat(data): complete phase 1 dataset acquisition and validation`)

- [X]  Ingestion logic with Kaggle API & automated reproducible fallback (`src/steam_player_forecasting/data/ingestion.py`)
- [X]  Raw schema inspection, artifact cleaning ("Last 30 Days"), and date standardization (`src/steam_player_forecasting/data/validation.py`)
- [X]  Representative game selection across genres (7 titles, 5.8–9.2 years history) (`src/steam_player_forecasting/data/selection.py`)
- [X]  Standardized processed data pipeline (`data/processed/steam_games_monthly.csv` and individual CSVs) (`src/steam_player_forecasting/data/pipeline.py`)
- [X]  Reusable data loaders (`load_processed_data()`, `load_raw_data()`, `get_available_games()`) (`src/steam_player_forecasting/data/loader.py`)
- [X]  Unit test suite with 100% pass rate (`tests/test_data.py`)
- [X]  Dataset cleaning decisions and chosen titles documented (`data/README.md`)

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

**Status:** Completed `[x]` (Commit: `dcfdfd2` | `feat(features): complete phase 2 eda and time-series preprocessing`)

- [X]  Comprehensive exploratory analysis notebook (`notebooks/01_eda_and_preprocessing.ipynb`) analyzing trends, seasonality, rolling mean/variance, outliers, and structural breaks
- [X]  9 high-resolution 300 DPI exploratory figures exported to `figures/` (trajectories, peak/avg ratios, rolling statistics, seasonality boxplots, subseries plots, STL decomposition, ACF/PACF, transformations, chronological splits)
- [X]  STL and classical decomposition separating Trend, Seasonal, and Residual components (`src/steam_player_forecasting/features/diagnostics.py`)
- [X]  Joint Augmented Dickey-Fuller (ADF) and KPSS stationarity tests with unit-root diagnostics (`check_stationarity()`)
- [X]  Autocorrelation (ACF) and Partial Autocorrelation (PACF) computations with confidence intervals (`compute_acf_pacf()`)
- [X]  Leakage-safe transformations: `LogTransformer`, `BoxCoxTransformer`, and stateful `DifferencingTransformer` with exact inverse reconstruction
- [X]  Train-only fitted scalers: `TimeSeriesScaler` (`minmax`, `standard`, `robust`) preserving index/columns without lookahead leakage
- [X]  Chronological train/validation/test split utilities (`train_val_test_split()`, `split_by_game()`, `expanding_window_cv()`)
- [X]  Unit test suite with 100% pass rate covering all scalers, transformations, diagnostics, and split routines (`tests/test_features.py`)
- [X]  Automated figure generation and notebook execution scripts (`scripts/generate_eda_and_figures.py`, `scripts/build_notebook.py`, `scripts/execute_notebook.py`)

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

**Status:** Completed `[x]` (Commit: `feat(models): complete phase 3 statistical forecasting baselines`)

- [X]  Reusable baseline forecasters (`NaiveForecaster`, `SeasonalNaiveForecaster`, `HoltWintersForecaster`) in `src/steam_player_forecasting/models/statistical.py`
- [X]  Autoregressive, ARIMA, and SARIMA models (`ARForecaster`, `ARIMAForecaster`, `SARIMAForecaster`) in `src/steam_player_forecasting/models/statistical.py`
- [X]  Leakage-safe transformed meta-forecaster (`TransformedForecaster`) in `src/steam_player_forecasting/models/base.py`
- [X]  Econometric residual diagnostics (`evaluate_residuals()`, `plot_residual_diagnostics()`) with Ljung-Box and Jarque-Bera tests in `src/steam_player_forecasting/models/diagnostics.py`
- [X]  Systematic SARIMA order selection and grid search (`select_sarima_order()`, `grid_search_sarima()`) in `src/steam_player_forecasting/models/selection.py`
- [X]  Strict chronological validation evaluation on 12-month horizon (2019-09 to 2020-08) across all 7 games, keeping holdout test set untouched
- [X]  Benchmark evaluation tables exported to `report/` (`statistical_validation_benchmarks.csv`, `sarima_order_selection.csv`, `sarima_residual_diagnostics.csv`)
- [X]  4 high-resolution 300 DPI publication figures in `figures/` (Figures 10, 11, 12, 13)
- [X]  Demonstration and diagnostic notebook fully executed in `notebooks/02_statistical_forecasting.ipynb`
- [X]  Unit tests covering all models, shapes, confidence intervals, inversions, and residual diagnostics (`tests/test_models.py`)

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

**Status:** Completed `[x]` (Commit: `feat(models): complete phase 4 machine learning and deep learning forecasting`)

- [X]  Leakage-safe tabular feature engineering (`create_lag_features()`, `create_rolling_features()`, `create_calendar_features()`, `build_tabular_feature_matrix()`, `TabularFeatureExtractor`) in `src/steam_player_forecasting/features/tabular.py`
- [X]  Tabular ML models with recursive multi-step forecasting (`RidgeForecaster`, `RandomForestForecaster`, `XGBoostForecaster`, `TabularForecaster`) in `src/steam_player_forecasting/models/tabular.py`
- [X]  Chronological validation hyperparameter tuning (`tune_tabular_forecaster()`) in `src/steam_player_forecasting/models/tabular.py`
- [X]  PyTorch sequence dataset with sliding windows (`TimeSeriesSequenceDataset`) in `src/steam_player_forecasting/models/deep_learning.py`
- [X]  Recurrent neural network architectures (`RNNModel`, `LSTMModel`, `GRUModel`) in `src/steam_player_forecasting/models/deep_learning.py`
- [X]  PyTorch forecaster wrappers with early stopping, LR scheduling, train-only scaling, and determinism (`RNNForecaster`, `LSTMForecaster`, `GRUForecaster`) in `src/steam_player_forecasting/models/deep_learning.py`
- [X]  Intermediate validation benchmarking across all 7 games comparing the Core Quartet: **SARIMA vs XGBoost vs LSTM vs GRU** (`scripts/run_phase4_benchmarks.py`)
- [X]  Validation benchmark reports in `report/` (`ml_dl_validation_benchmarks.csv/.md`, `core_quartet_comparison.csv/.md`)
- [X]  3 high-resolution 300 DPI publication figures in `figures/` (Figures 14, 15, 16)
- [X]  Interactive demonstration and analysis notebook in `notebooks/03_ml_and_dl_forecasting.ipynb`
- [X]  Comprehensive unit test suite for tabular features, model forecasters, and PyTorch architectures (`tests/test_tabular.py`, `tests/test_deep_learning.py`) with 100% pass rate

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

**Status:** Completed `[x]` (Commit: `feat(report): complete phase 5 final evaluation, report, and interactive demo`)

- [X]  Multi-step ($H=12$) unblinded test evaluation on held-out test window (2020-09-01 to 2021-08-01) across all 7 games
- [X]  Standardized test metric tables exported to `report/` (`final_test_benchmarks.csv/.md`, `core_quartet_test_benchmarks.csv/.md`, `final_leaderboard.csv/.md`, `error_by_horizon.csv`, `validation_vs_test_comparison.csv/.md`)
- [X]  4 high-resolution 300 DPI publication figures in `figures/` (Figures 17, 18, 19, 20)
- [X]  Interactive demonstration application built in `app/app.py` (Streamlit)
- [X]  Executive technical report in `report/final_report.md`
- [X]  Interactive demonstration and synthesis notebook fully executed in `notebooks/04_final_evaluation_and_synthesis.ipynb`
- [X]  Comprehensive unit test suite for Phase 5 artifacts and schema adherence (`tests/test_phase5_evaluation.py`) with 100% pass rate

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
