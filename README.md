<p align="center">
  <img src="assets/steam_logo.png" alt="Steam Logo" width="110" />
</p>

<h1 align="center">Steam Player Forecasting</h1>

<p align="center">
  <b>A reproducible benchmarking framework comparing statistical, machine learning, and deep learning time-series models for forecasting monthly Steam player counts.</b>
</p>

<p align="center">
  <a href="https://github.com/JoshiMinh/steam-player-forecasting/actions/workflows/ci.yml"><img src="https://github.com/JoshiMinh/steam-player-forecasting/actions/workflows/ci.yml/badge.svg" alt="CI" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT" /></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python 3.11+" /></a>
  <a href="#interactive-steam-store-demonstration"><img src="https://img.shields.io/badge/Streamlit-Steam%20UI-1b2838?logo=steam" alt="Steam Store UI" /></a>
</p>

---

## Research Question

> **Can modern recurrent deep learning models (LSTM, GRU) or gradient-boosted trees (XGBoost) reliably outperform classical econometric benchmarks (SARIMA) when forecasting monthly active player populations for top multiplayer games on Steam?**

Video game populations are characterized by non-stationary behavior: structural breaks from major content updates, acute seasonal surges (Steam Summer/Winter Sales, holiday breaks), viral streamer adoption, and long-term organic churn. This project systematically evaluates where statistical approaches suffice and where non-linear ML/DL paradigms deliver demonstrable forecasting advantages.

---

## Dataset

- **Source:** [Steam Player Data (Kaggle)](https://www.kaggle.com/datasets/jackogozaly/steam-player-data)
- **Granularity:** Monthly time-series observations across top Steam titles
- **Primary Target:** `Avg_players` (Monthly average concurrent player count)
- **Secondary Target:** `Peak_Players` (Monthly peak concurrent player count)

See [`data/README.md`](data/README.md) for full ingestion, schema specifications, and setup instructions.

---

## Planned Models

The benchmark evaluates three major modeling paradigms:

### Core Quartet Comparison
| Paradigm | Model | Description |
| :--- | :--- | :--- |
| **Statistical** | **SARIMA** | Seasonal Autoregressive Integrated Moving Average |
| **Machine Learning** | **XGBoost** | Gradient-boosted decision trees with lag & rolling window features |
| **Deep Learning** | **LSTM** | Long Short-Term Memory recurrent neural network |
| **Deep Learning** | **GRU** | Gated Recurrent Unit recurrent neural network |

### Baseline Models
- **Heuristic:** Naive (last value), Seasonal Naive (prior year month)
- **Exponential Smoothing:** Holt's Linear, Holt-Winters (additive/multiplicative)
- **Autoregressive / Classical:** AR, ARIMA
- **Tabular ML:** Ridge Regression, Random Forest Regressor
- **Deep Learning:** Vanilla Recurrent Neural Network (RNN)

---

## Evaluation Metrics

All models are evaluated on identical, strictly chronological holdout test windows:
- **Primary Metrics:**
  - **MAE** (Mean Absolute Error)
  - **RMSE** (Root Mean Squared Error)
  - **MAPE** (Mean Absolute Percentage Error)
- **Secondary & Diagnostic Metrics:**
  - **sMAPE** (Symmetric Mean Absolute Percentage Error)
  - **MASE** (Mean Absolute Scaled Error, scaled by in-sample seasonal naive)
  - **AIC / BIC** (Model selection for statistical specifications)

---

## Project Structure

```text
steam-player-forecasting/
├── .github/
│   └── workflows/
│       └── ci.yml               # GitHub Actions continuous integration workflow
├── configs/
│   └── default.yaml             # Central configuration (splits, hyperparameters, seeds)
├── data/
│   ├── raw/                     # Raw untouched datasets (.gitignore)
│   ├── processed/               # Cleaned & standardized datasets (.gitignore)
│   └── README.md                # Data documentation and download guide
├── notebooks/                   # Jupyter exploration and benchmark notebooks
├── src/
│   └── steam_player_forecasting/
│       ├── __init__.py          # Package root
│       ├── data/                # Data ingestion, schema validation & loaders
│       ├── features/            # Preprocessing, scalers, and lag generators
│       ├── models/              # Statistical, ML, and PyTorch model implementations
│       └── evaluation/          # Metrics (MAE, RMSE, MAPE, sMAPE, MASE)
├── models/                      # Serialized model checkpoints (.gitignore)
├── figures/                     # Exported evaluation and EDA plots (.gitignore)
├── app/                         # Lightweight interactive demonstration (Streamlit)
├── report/                      # Technical reports and tabular benchmark outputs
├── tests/                       # Automated test suite (pytest)
├── .gitignore                   # Git ignore specifications
├── LICENSE                      # MIT License
├── pyproject.toml               # Build system configuration & dependencies
├── requirements.txt             # Project requirements
├── README.md                    # Project documentation
└── ROADMAP.md                   # 5-phase execution plan with AI agent prompts
```

---

## Environment Setup

### Prerequisites
- Python 3.11+ (Python 3.11, 3.12, or 3.13)
- Git & GitHub CLI (`gh`)

### Quickstart

1. **Clone the repository:**
   ```bash
   git clone https://github.com/JoshiMinh/steam-player-forecasting.git
   cd steam-player-forecasting
   ```

2. **Create and activate the virtual environment:**
   ```bash
   # Windows (PowerShell)
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1

   # Linux / macOS
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   pip install -e .
   ```

4. **Run tests:**
   ```bash
   pytest
   ```

---

## Interactive Demo (Streamlit)

Launch the interactive forecasting dashboard to explore historical trajectories, compare models, inspect 95% SARIMA confidence intervals, and review error diagnostics:

```bash
streamlit run app/app.py
```

**Demo Capabilities:**
- Select any of the 7 evaluated Steam multiplayer flagships.
- Toggle between Core Quartet models (SARIMA, XGBoost, LSTM, GRU) and baseline forecasters.
- Adjust historical context windows (6 to 36 months).
- Review real-time performance scorecards and monthly error breakdowns.
- Inspect peak-to-average player ratios, historical volatility, and documented structural breaks.

---

## Final Research Findings & Leaderboard

### Core Answer to Research Question
> **Can modern recurrent deep learning models (LSTM, GRU) or gradient-boosted trees (XGBoost) reliably outperform classical econometric benchmarks (SARIMA)?**

**Key Empirical Discoveries:**
1. **The Core Quartet Split (SARIMA: 3 wins, GRU: 3 wins, LSTM: 1 win, XGBoost: 0 wins):**
   - **SARIMA dominates on titles with regular, smooth annual cycles** (*Dota 2*, *GTA V*, *Rust*). Differencing ($d=1, D=1$) naturally enforces mean reversion and prevents runaway forecast drift.
   - **GRU & LSTM dominate on titles with non-linear retention plateaus and structural shifts** (*CS:GO*, *Rainbow Six Siege*, *Warframe*, *TF2*). Gated recurrent units effectively adapt to multi-scale cyclicality and retention dynamics.
   - **XGBoost consistently struggled across all 7 games (Mean MASE: 0.923 vs SARIMA: 0.672, GRU: 0.776):** Recursive multi-step forecasting in tree ensembles suffers from compounding autoregressive error drift over long ($H=12$) horizons.
2. **Parsimonious Baselines Excel Overall:**
   - Across all 11 candidate models, **Holt-Winters (Mean MASE: 0.526)** and **Ridge Regression (Mean MASE: 0.561)** achieved the lowest overall average error across titles, underscoring the value of strong regularization in sample-constrained time-series regimes.

### Final Holdout Test Leaderboard (2020-09-01 to 2021-08-01, $H=12$)
*Ranked by Mean MASE across all 7 evaluated Steam titles:*

| Rank | Model | Paradigm | Mean MAE | Median MAE | Mean RMSE | Mean MAPE | Mean MASE | Titles Won |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **Holt-Winters** | Exponential Smoothing | 8,049.45 | 5,781.39 | 10,143.88 | 4.72% | **0.526** | **3** |
| **2** | **Ridge** | Linear ML | 7,966.24 | 5,149.47 | 9,872.49 | 4.99% | **0.561** | **3** |
| **3** | **SARIMA** | Classical Econometric | 11,605.70 | 5,565.80 | 13,665.33 | 6.03% | **0.672** | **1** |
| **4** | **GRU** | Recurrent Deep Learning | 13,018.31 | 7,866.74 | 16,468.70 | 6.82% | **0.776** | **0** |
| **5** | **LSTM** | Recurrent Deep Learning | 12,998.59 | 8,557.04 | 16,674.17 | 7.13% | **0.802** | **0** |
| **6** | **Random Forest** | Tree Ensemble ML | 14,184.20 | 7,302.51 | 17,260.98 | 7.45% | **0.832** | **0** |
| **7** | **RNN** | Vanilla Recurrent DL | 13,261.02 | 9,092.99 | 16,793.54 | 7.94% | **0.857** | **0** |
| **8** | **XGBoost** | Gradient Boosted ML | 14,770.90 | 10,062.14 | 17,556.75 | 8.40% | **0.923** | **0** |
| **9** | **Seasonal Naive** | Seasonal Baseline | 19,471.53 | 11,339.65 | 21,978.68 | 10.45% | **1.119** | **0** |
| **10** | **ARIMA(1,1,1)** | Classical Econometric | 19,978.06 | 12,486.85 | 23,490.15 | 11.08% | **1.294** | **0** |
| **11** | **Naive** | Heuristic Baseline | 21,386.35 | 10,543.99 | 24,870.34 | 11.21% | **1.357** | **0** |

### Core Quartet Head-to-Head Performance by Game

| Game | SARIMA MAE (MASE) | XGBoost MAE (MASE) | LSTM MAE (MASE) | GRU MAE (MASE) | Best Paradigm |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Counter-Strike: Global Offensive** | 31,473.4 (0.589) | 36,494.2 (0.683) | 31,094.8 (0.582) | **24,360.5 (0.456)** | **GRU (DL)** |
| **Dota 2** | **23,463.9 (0.811)** | 25,401.8 (0.878) | 26,687.0 (0.923) | 35,996.4 (1.245) | **SARIMA (Econometric)** |
| **Grand Theft Auto V** | **5,565.8 (0.444)** | 9,469.6 (0.755) | 7,295.7 (0.582) | 8,269.6 (0.660) | **SARIMA (Econometric)** |
| **Rust** | **3,888.2 (0.394)** | 10,062.1 (1.019) | 8,557.0 (0.867) | 7,030.8 (0.712) | **SARIMA (Econometric)** |
| **Team Fortress 2** | 4,440.2 (1.037) | 5,284.5 (1.234) | **3,929.5 (0.917)** | 4,532.6 (1.058) | **LSTM (DL)** |
| **Tom Clancy's Rainbow Six Siege** | 9,206.8 (0.694) | 12,569.2 (0.948) | 8,681.1 (0.655) | **7,866.7 (0.593)** | **GRU (DL)** |
| **Warframe** | 3,201.6 (0.734) | 4,114.9 (0.944) | 4,745.1 (1.089) | **3,071.4 (0.705)** | **GRU (DL)** |

Detailed discussion and qualitative case studies available in [`report/final_report.md`](report/final_report.md).

---

## Roadmap & Execution History

The project followed a strict 5-phase sequential architecture documented in [`ROADMAP.md`](ROADMAP.md):

- [x] **Phase 0:** Repository Initialization & Standards Setup *(Completed)*
- [x] **Phase 1:** Dataset Acquisition & Validation *(Completed)*
- [x] **Phase 2:** EDA & Time-Series Preprocessing *(Completed)*
- [x] **Phase 3:** Statistical Forecasting (Baselines & SARIMA) *(Completed)*
- [x] **Phase 4:** Machine Learning & Deep Learning (XGBoost, LSTM, GRU) *(Completed)*
- [x] **Phase 5:** Final Evaluation, Report & Interactive Demo *(Completed)*

---

## End-to-End Reproducibility Guide

All analyses, models, figures, and benchmark tables can be reproduced from scratch using deterministic scripts:

```bash
# 1. Run the entire automated test suite (68 unit tests, 100% pass rate)
pytest -v

# 2. Re-run Phase 5 Test-Set Evaluation and generate benchmark CSVs
python scripts/run_phase5_benchmarks.py

# 3. Generate all publication figures (Figures 17, 18, 19, 20)
python scripts/generate_phase5_figures.py

# 4. Build and execute demonstration notebook
python scripts/build_phase5_notebook.py
python scripts/execute_notebook.py 04_final_evaluation_and_synthesis.ipynb

# 5. Launch interactive demonstration dashboard
streamlit run app/app.py
```
