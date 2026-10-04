# Data Directory & Dataset Documentation

This directory contains the datasets and documentation for the **Steam Player Forecasting** project.

> **Note:** Raw and processed data files (`.csv`, `.parquet`, etc.) are excluded from Git version control by `.gitignore` to keep the repository lightweight and adhere to data distribution best practices.

---

## Dataset Overview

- **Dataset Name:** Steam Player Data
- **Source:** Kaggle
- **URL:** [https://www.kaggle.com/datasets/jackogozaly/steam-player-data](https://www.kaggle.com/datasets/jackogozaly/steam-player-data)
- **Granularity:** Monthly time-series records per game
- **Primary Forecasting Target:** `Avg_players` (Monthly average concurrent player count)
- **Secondary Forecasting Target:** `Peak_Players` (Monthly peak concurrent player count)

---

## Directory Structure

```text
data/
├── raw/            # Untouched original raw downloads (e.g. CSV from Kaggle)
├── processed/      # Cleaned, filtered, validated, and transformed time-series datasets
└── README.md       # Dataset documentation and ingestion instructions
```

---

## Schema and Expected Columns

Based on the Kaggle dataset structure, records are recorded per month per game:

| Column Name | Expected Type | Description | Role |
| :--- | :--- | :--- | :--- |
| `Month_Year` / `Month` | string / date | Time period string (e.g., "September 2021" or ISO date) | Temporal Index |
| `Game_Name` / `gamename` | string | Canonical title of the Steam game | Entity Identifier |
| `Avg_players` / `avg` | float | Average concurrent players during the month | **Primary Target** |
| `Gain` | float | Net change in average players compared to previous month | Feature / Diagnostic |
| `% Gain` | string / float | Percentage change in average players | Feature / Diagnostic |
| `Peak_Players` / `peak` | integer / float | Highest concurrent players recorded in that month | **Secondary Target** |

*Note: In Phase 1, data loaders must inspect the actual downloaded file schema and standardize column names into snake_case or canonical names configured in `configs/default.yaml`.*

---

## How to Obtain and Set Up the Data

### Option 1: Using the Kaggle CLI (Recommended)

1. Ensure your Kaggle API token is configured (`~/.kaggle/kaggle.json`).
2. Download the dataset into `data/raw/`:
   ```bash
   kaggle datasets download -d jackogozaly/steam-player-data -p data/raw --unzip
   ```

### Option 2: Manual Download

1. Visit [https://www.kaggle.com/datasets/jackogozaly/steam-player-data](https://www.kaggle.com/datasets/jackogozaly/steam-player-data).
2. Download the ZIP archive.
3. Extract the CSV file(s) into `data/raw/`.

---

## Strict Rules for Data Handling

1. **Chronological Integrity:** Never shuffle time-series observations. Train/val/test splits must be strictly chronological.
2. **Leakage Prevention:** Any scaling, imputation, or feature transformations must be fitted **only** on the training window and transformed on validation/test windows.
3. **Consistency:** All competing models must be evaluated against the exact same test period and horizon.
