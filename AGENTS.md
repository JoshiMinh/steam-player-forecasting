# AGENTS.md — Autonomous AI Agent Operating Manual

Welcome to the **`steam-player-forecasting`** repository. This document serves as the primary technical specification, operational protocol, and context guide for AI coding agents (Antigravity, Claude Code, Cursor, Copilot Workspace, Amp, etc.) operating on this codebase.

All agents working on this repository **MUST** strictly adhere to the standards, testing mandates, and time-series anti-leakage rules outlined below.

---

## 1. Project Mission & Research Architecture

### Core Research Question
> **Can modern recurrent deep learning models (LSTM, GRU) or gradient-boosted trees (XGBoost) reliably outperform classical econometric benchmarks (SARIMA) when forecasting monthly active player populations for top multiplayer games on Steam?**

### Domain Characteristics
Steam player populations exhibit unique, challenging time-series characteristics:
1. **Multi-Game Heterogeneity:** Divergent player bases across genres (tactical shooters, MOBAs, survival sandboxes).
2. **Non-Stationarity & Integration ($d=1$):** Strong growth trends, long-term organic churn, and unit-root dynamics.
3. **Calendar & Event Seasonality ($s=12$):** Annual surges driven by Steam Summer Sales (June/July), Steam Winter Sales / Christmas holidays (December/January), and major esports championships (e.g. Dota 2 *The International* in August).
4. **Multiplicative Volatility (Heteroscedasticity):** Fluctuations grow proportionally with player level, necessitating logarithmic or Box-Cox transformations.
5. **Structural Breaks:** Discrete macro-events such as Free-to-Play transitions (CS:GO in Dec 2018), pandemic lockdowns (March 2020), and influencer-driven spikes (Rust in Jan 2021).

---

## 2. Execution Phases & Current Status

The project follows a strict 5-phase sequential architecture documented in [`ROADMAP.md`](ROADMAP.md):

| Phase | Title | Status | Primary Focus |
| :---: | :--- | :---: | :--- |
| **0** | Repository Initialization | **Completed `[x]`** | Directory skeleton, CI workflows, packaging, configs |
| **1** | Dataset Acquisition & Validation | **Completed `[x]`** | Kaggle ingestion, synthetic fallback, cleaning, game selection |
| **2** | EDA & Time-Series Preprocessing | **Completed `[x]`** | EDA notebook, 9 figures, STL decomposition, ADF/KPSS, transforms, scalers |
| **3** | Statistical Forecasting | **NEXT `[ ]`** | Baselines (Naive, SNaive, Holt-Winters), AR, ARIMA, SARIMA |
| **4** | Machine Learning & Deep Learning | Pending `[ ]` | Lag matrices, XGBoost, PyTorch datasets, LSTM, GRU |
| **5** | Final Evaluation, Report & Demo | Pending `[ ]` | Holdout test evaluation, leaderboard, report, Streamlit UI |

> [!IMPORTANT]
> **Single-Phase Execution Rule:** Always execute ONLY the phase assigned by the user. Do **NOT** implement downstream phases prematurely. Explicitly stop and run tests once your assigned phase is complete.

---

## 3. Strict Time-Series Anti-Leakage Protocol

Time-series forecasting is vulnerable to catastrophic lookahead data leakage if future observations contaminate training. Every agent must enforce these non-negotiable rules:

### A. Chronological Holdout Partitions
- **Test Set:** Strictly the final 12 months (`2020-09-01` to `2021-08-01`). **DO NOT TOUCH** during baseline modeling or hyperparameter tuning.
- **Validation Set:** Strictly the 12 months preceding the test window (`2019-09-01` to `2020-08-01`). Used for model selection, AIC/BIC validation, and parameter search.
- **Training Set:** All chronological observations prior to the validation window (`start` to `2019-08-01`).
- **No Random Shuffling:** Never use standard random $k$-fold cross-validation or `train_test_split(shuffle=True)`. Always use `train_val_test_split()` or `expanding_window_cv()` from `steam_player_forecasting.features`.

### B. Train-Only Fitting
- Any scaler (`TimeSeriesScaler`, `MinMaxScaler`, `StandardScaler`, `RobustScaler`) must be fitted **exclusively on the training partition**:
  ```python
  scaler = TimeSeriesScaler(scaler_type="minmax")
  scaler.fit(train_series)  # FITTED ON TRAIN ONLY!
  train_scaled = scaler.transform(train_series)
  val_scaled = scaler.transform(val_series)
  test_scaled = scaler.transform(test_series)
  ```
- Any variance-stabilizing transform (`BoxCoxTransformer`, `LogTransformer`) must estimate its parameters ($\lambda$, offsets) **strictly on the training window**.
- Under upward trends, test scaled values may legitimately exceed $1.0$ (when using `minmax`). This is normal and proves zero lookahead leakage.

### C. Exact Inversion
- Forecasts generated in transformed or scaled space must be inverted back to original level player counts using stored training state:
  ```python
  y_pred_level = scaler.inverse_transform(y_pred_scaled)
  ```
- For differencing ($d=1, s=12$), use `DifferencingTransformer` which retains the training boundary buffer to reconstruct level forecasts step-by-step.

---

## 4. Environment & Toolchains (Dual-Platform)

The project uses Python 3.11+ (tested on Python 3.13) with a local virtual environment located in `.venv/`.

### Windows (PowerShell) Commands
```powershell
# 1. Activate virtual environment
.\.venv\Scripts\Activate.ps1

# 2. Direct Python execution (no activation required)
.\.venv\Scripts\python.exe -m pytest

# 3. Install/update editable package
.\.venv\Scripts\pip.exe install -e .

# 4. Run test suite
.\.venv\Scripts\pytest.exe -v
```

### Linux / macOS (Bash) Commands
```bash
# 1. Activate virtual environment
source .venv/bin/activate

# 2. Direct Python execution
./.venv/bin/python -m pytest

# 3. Install/update editable package
./.venv/bin/pip install -e .

# 4. Run test suite
./.venv/bin/pytest -v
```

> [!CAUTION]
> **Shell Navigation Rule:** When executing shell commands via agent tools, **NEVER run `cd` commands**. Always pass the target command from the project root (`CWD`) or specify relative/absolute file paths.

---

## 5. Testing & Quality Assurance Protocol

### Test Execution
Automated tests are powered by `pytest` and configured in `pyproject.toml`.

```bash
# Run all tests verbosely
pytest -v

# Run specific module
pytest tests/test_features.py -v
pytest tests/test_data.py -v
pytest tests/test_smoke.py -v
```

### Mandatory Pre-Commit Gate
- **100% Pass Rate:** Never commit code if any unit test is failing or throwing unhandled errors.
- **Isolated Pytest Cache:** `pyproject.toml` redirects pytest temporary files to `.pytest_temp` (`--basetemp=.pytest_temp`) to avoid file-lock conflicts on Windows.
- **Naming Discovery Trap:** Pytest automatically discovers any function starting with `test_` in imported modules. If writing a non-test diagnostic function (e.g. `check_stationarity`), either avoid naming it `test_*` or explicitly tag it with:
  ```python
  check_stationarity.__test__ = False
  ```

---

## 6. Repository Architecture & Directory Map

```text
steam-player-forecasting/
├── .github/
│   └── workflows/
│       └── ci.yml               # Automated CI test runner on push/PR
├── configs/
│   └── default.yaml             # Central configuration (splits, lags, models, metrics)
├── data/
│   ├── raw/                     # Raw Kaggle / synthetic data (ignored by git)
│   ├── processed/               # Cleaned per-game & combined CSVs (ignored by git)
│   └── README.md                # Data documentation, cleaning rules, selected titles
├── figures/                     # 300 DPI publication plots (tracked in git)
│   ├── 01_historical_player_trajectories.png
│   ├── 02_peak_vs_avg_ratio.png
│   ├── 03_rolling_statistics_heteroscedasticity.png
│   ├── 04_monthly_seasonality_distributions.png
│   ├── 05_seasonal_subseries_flagships.png
│   ├── 06_stl_decomposition.png
│   ├── 07_stationarity_acf_pacf.png
│   ├── 08_transformations_variance_stabilization.png
│   └── 09_chronological_splits.png
├── notebooks/
│   └── 01_eda_and_preprocessing.ipynb  # Executed Phase 2 analysis notebook
├── scripts/
│   ├── build_notebook.py        # Generates clean EDA notebook
│   ├── execute_notebook.py      # Executes notebook and captures figures/outputs
│   └── generate_eda_and_figures.py # Generates all 9 high-res figures
├── src/
│   └── steam_player_forecasting/
│       ├── __init__.py          # Package root
│       ├── data/                # Phase 1: Ingestion, cleaning, selection, loaders
│       │   ├── ingestion.py     # Kaggle API & synthetic fallback generator
│       │   ├── loader.py        # load_processed_data(), get_available_games()
│       │   ├── pipeline.py      # run_phase1_pipeline()
│       │   ├── selection.py     # select_representative_games()
│       │   └── validation.py    # Schema cleaning, timestamp formatting, gain reconciliation
│       ├── features/            # Phase 2: Leakage-safe preprocessors & diagnostics
│       │   ├── README.md        # Feature module documentation
│       │   ├── diagnostics.py   # check_stationarity, compute_acf_pacf, decompose_time_series
│       │   ├── scalers.py       # TimeSeriesScaler (MinMax, Standard, Robust)
│       │   ├── split.py         # train_val_test_split, split_by_game, expanding_window_cv
│       │   └── transforms.py    # LogTransformer, BoxCoxTransformer, DifferencingTransformer
│       ├── models/              # Phase 3 & 4: Forecasting model wrappers
│       └── evaluation/          # Metrics: MAE, RMSE, MAPE, sMAPE, MASE
│           └── metrics.py
├── tests/
│   ├── test_data.py             # Phase 1 data tests
│   ├── test_features.py         # Phase 2 preprocessing & leakage tests
│   └── test_smoke.py            # Basic package sanity tests
├── .gitignore                   # Comprehensive data science & ML ignore rules
├── AGENTS.md                    # THIS FILE: Autonomous agent operational guide
├── pyproject.toml               # Build system, dependencies, and pytest configuration
├── README.md                    # Project landing page and research overview
├── requirements.txt             # Pip pinned dependencies
└── ROADMAP.md                   # 5-phase execution plan with copy-paste agent prompts
```

---

## 7. Selected Representative Titles

Seven titles meeting continuous long-term historical criteria ($\ge 5.8\text{--}9.2$ years of monthly observations up to August 2021) are standardized in `data/processed/`:

1. **Counter-Strike: Global Offensive** (Tactical FPS, 110 months, 2012-07 to 2021-08)
2. **Dota 2** (MOBA, 110 months, 2012-07 to 2021-08)
3. **Team Fortress 2** (Hero Shooter, 110 months, 2012-07 to 2021-08)
4. **Warframe** (Co-op Action RPG, 102 months, 2013-03 to 2021-08)
5. **Rust** (Multiplayer Survival, 93 months, 2013-12 to 2021-08)
6. **Grand Theft Auto V** (Open-World Action, 77 months, 2015-04 to 2021-08)
7. **Tom Clancy's Rainbow Six Siege** (Tactical Shooter, 69 months, 2015-12 to 2021-08)

---

## 8. Operational Guide for Upcoming Phases

### Phase 3 — Statistical Forecasting (Next Task)
- **Objective:** Implement econometric and statistical baseline models.
- **Target Directory:** `src/steam_player_forecasting/models/`
- **Required Model Wrappers:**
  - `NaiveForecaster`: Last observed value ($y_{t+h} = y_t$).
  - `SeasonalNaiveForecaster`: Value from identical calendar month in prior year ($y_{t+h} = y_{t+h-12}$).
  - `HoltWintersForecaster`: Exponential smoothing with additive/multiplicative trend and seasonality (`statsmodels.tsa.holtwinters.ExponentialSmoothing`).
  - `ARIMAForecaster` / `SARIMAForecaster`: Autoregressive integrated moving average with seasonal order $(p, d, q) \times (P, D, Q)_{12}$ (`statsmodels.tsa.statespace.sarimax.SARIMAX`).
- **Evaluation Rule:**
  - Fit on training window (`start` to `2019-08-01`).
  - Evaluate multi-step forecast ($H=12$) on validation window (`2019-09-01` to `2020-08-01`).
  - **Do NOT evaluate on test window yet.**
  - Calculate metrics using `steam_player_forecasting.evaluation.metrics.calculate_metrics()`.
  - Residual diagnostics: verify white noise residuals via Ljung-Box test (`acorr_ljungbox`).

### Phase 4 — Machine Learning & Deep Learning
- **Tabular ML:** Build lag feature matrix in `src/steam_player_forecasting/features/` ($t-1, t-2, t-3, t-6, t-12$, rolling means/stds), train Ridge and `xgboost.XGBRegressor`.
- **Deep Learning:** PyTorch Dataset/DataLoader emitting sliding sequence windows ($L=12 \to H=12$), implement multi-layer LSTM and GRU regressors with linear forecast heads.

### Phase 5 — Final Evaluation, Report & Demo
- **Holdout Unblinding:** Evaluate optimal models from Phases 3 and 4 on the untouched 12-month test partition (`2020-09-01` to `2021-08-01`).
- **Deliverables:** Final benchmarking table in `report/`, forecast comparison plots, and a demonstration app in `app/` (Streamlit).

---

## 9. Known Gotchas & Technical Pitfalls

1. **Matplotlib 3.9+ Boxplot Parameter:**
   `Axes.boxplot()` deprecated `labels` in favor of `tick_labels`. Using `labels=` raises `TypeError` in matplotlib $\ge 3.9$.
2. **Statsmodels FutureWarnings:**
   In statsmodels 0.15+, `adfuller()` and `acf()` will transition to returning result objects in future versions. Use `warnings.catch_warnings()` or silence expected future warnings to keep CLI outputs clean.
3. **Dataframe Date Columns:**
   Processed date column is named `Month_Year` and standardized to `YYYY-MM-01`. Always parse as datetime via `parse_dates=["Month_Year"]` or use `load_processed_data()`.
4. **Scraping Artifact Rows:**
   Raw SteamCharts tables contain a `"Last 30 Days"` row at the top. The ingestion pipeline (`validation.py`) explicitly filters this out; never assume raw rows are purely calendar months.
5. **Conventional Commits:**
   Follow standard commit types:
   - `feat(...)`: New feature or phase deliverable (e.g. `feat(models): implement SARIMA wrapper`)
   - `test(...)`: Unit tests
   - `docs(...)`: Documentation updates (`ROADMAP.md`, `README.md`)
   - `chore(...)`: Configuration or tooling updates
