# Steam Player Forecasting

[![CI](https://github.com/JoshiMinh/steam-player-forecasting/actions/workflows/ci.yml/badge.svg)](https://github.com/JoshiMinh/steam-player-forecasting/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

A reproducible benchmarking framework comparing statistical, machine learning, and deep learning time-series models for forecasting monthly Steam player counts.

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

## Roadmap

The project is structured into five sequential phases documented in [`ROADMAP.md`](ROADMAP.md). Each phase contains a copy-pasteable AI Agent Prompt:

- [x] **Phase 0:** Repository Initialization & Standards Setup *(Completed)*
- [x] **Phase 1:** Dataset Acquisition & Validation *(Completed)*
- [ ] **Phase 2:** EDA & Time-Series Preprocessing
- [ ] **Phase 3:** Statistical Forecasting (Baselines & SARIMA)
- [ ] **Phase 4:** Machine Learning & Deep Learning (XGBoost, LSTM, GRU)
- [ ] **Phase 5:** Final Evaluation, Report & Interactive Demo

---

## Current Status

- **Status:** **Phase 1 Complete**
- **Ready for:** **Phase 2 (EDA & Time-Series Preprocessing)**
- Standardized datasets (`steam_games_monthly.csv` and individual game CSVs) generated and validated in `data/processed/`. All unit tests passing.
