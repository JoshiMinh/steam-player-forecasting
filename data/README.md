# Data Directory & Dataset Documentation

This directory contains the datasets, data pipelines, and documentation for the **Steam Player Forecasting** project.

> **Note:** Raw and processed data files (`.csv`, `.parquet`, etc.) are excluded from Git version control by `.gitignore` to keep the repository lightweight and adhere to distribution best practices. Run the ingestion pipeline to generate or download the datasets locally.

---

## Dataset Overview

- **Dataset Name:** Steam Player Data
- **Source:** Kaggle (scraped from [SteamCharts](https://steamcharts.com/))
- **Author:** Jack Ogozaly
- **URL:** [https://www.kaggle.com/datasets/jackogozaly/steam-player-data](https://www.kaggle.com/datasets/jackogozaly/steam-player-data)
- **Granularity:** Monthly time-series records per game
- **Temporal Span:** July 2012 – August 2021 (up to 110 monthly observations per title)
- **Primary Forecasting Target:** `Avg_players` (Monthly average concurrent player count)
- **Secondary Forecasting Target:** `Peak_Players` (Monthly peak concurrent player count)

---

## Directory Structure

```text
data/
├── raw/
│   ├── .gitkeep
│   └── All_Steam_Games.csv          # Untouched raw dataset (Kaggle or reproducible synthetic fallback)
├── processed/
│   ├── .gitkeep
│   ├── steam_games_monthly.csv      # Standardized combined dataset for representative games
│   ├── counter_strike_global_offensive.csv
│   ├── dota_2.csv
│   ├── team_fortress_2.csv
│   ├── rust.csv
│   ├── warframe.csv
│   ├── tom_clancy_s_rainbow_six_siege.csv
│   └── grand_theft_auto_v.csv
└── README.md                        # Dataset documentation & cleaning decisions
```

---

## Raw Schema Inspection & Scraping Artifacts

Inspecting the raw SteamCharts scrapes revealed several domain-specific quirks that must be cleaned prior to any time-series modeling:

1. **Non-Monthly Artifact Rows (`"Last 30 Days"`):**
   SteamCharts includes a rolling 30-day row at the top of each game's table. This does not represent a calendar month, causes overlap with the most recent monthly observation, and fails calendar date parsing. The cleaning pipeline explicitly detects and strips these records.
2. **Column Name Discrepancies:**
   Different versions of the scraper export varying column names:
   - `Game_Name` vs `gamename` vs `game` vs `name`
   - `Month_Year` vs `month` vs separate `year` + `month` columns
   - `Avg_players` vs `avg` vs `Avg. Players`
   - `Peak_Players` vs `peak` vs `Peak Players`
   - `% Gain` vs `percent_gain` vs `pct_gain`
   The pipeline standardizes all variations into canonical column names.
3. **Formatted String Numerics:**
   Numbers are frequently formatted with thousands separators (`"512,345.50"`), percentage signs (`"+4.19%"`), and leading signs. First-month diffs are recorded as dashes (`"-"`) or empty strings. The pipeline sanitizes and casts these to proper float and integer types.
4. **Scraping Inconsistencies:**
   Occasional scraping anomalies report peak player counts lower than the monthly average or produce inconsistent gain metrics. The pipeline clamps $\text{Peak\_Players} \ge \text{Avg\_players}$ and mathematically recalculates `Gain` and `Percent_Gain` from consecutive chronological observations:
   $$\text{Gain}_t = \text{Avg\_players}_t - \text{Avg\_players}_{t-1}$$
   $$\text{Percent\_Gain}_t = \frac{\text{Gain}_t}{\text{Avg\_players}_{t-1}} \times 100\%$$

---

## Standardized Clean Schema

Cleaned and validated files in `data/processed/` adhere to the following strict schema:

| Column Name | Type | Constraints | Description | Role |
| :--- | :--- | :--- | :--- | :--- |
| `Game_Name` | `string` | Non-null, trimmed | Canonical Steam game title | Entity Identifier |
| `Month_Year` | `string` / `datetime64[ns]` | `YYYY-MM-01`, monotonic per game | Normalized start of month date | Temporal Index |
| `Avg_players` | `float64` | Non-null, $\ge 0.0$ | Monthly average concurrent players | **Primary Target** |
| `Peak_Players` | `int64` | Non-null, $\ge \text{Avg\_players}$ | Monthly peak concurrent players | **Secondary Target** |
| `Gain` | `float64` | `NaN` for first month | Net change in average players vs $t-1$ | Feature / Diagnostic |
| `Percent_Gain`| `float64` | `NaN` for first month | Percentage change vs $t-1$ | Feature / Diagnostic |

---

## Selected Representative Games

Phase 1 requires selecting 5–10 representative games across diverse gaming genres featuring continuous, long-term historical coverage ($\ge 3\text{--}5+$ years of monthly records).

Seven games were selected meeting all criteria:

| Game Title | Genre | Date Range | Total Months | Span (Years) | Historical Mean Avg Players | Historical Max Peak Players |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Counter-Strike: Global Offensive** | Tactical FPS | 2012-07 to 2021-08 | 110 | 9.2 yrs | ~410,000 | ~1,300,000 |
| **Dota 2** | MOBA | 2012-07 to 2021-08 | 110 | 9.2 yrs | ~440,000 | ~1,290,000 |
| **Team Fortress 2** | Hero Shooter | 2012-07 to 2021-08 | 110 | 9.2 yrs | ~54,000 | ~150,000 |
| **Warframe** | Co-op Action RPG | 2013-03 to 2021-08 | 102 | 8.5 yrs | ~42,000 | ~180,000 |
| **Rust** | Multiplayer Survival | 2013-12 to 2021-08 | 93 | 7.8 yrs | ~55,000 | ~240,000 |
| **Grand Theft Auto V** | Open-World Action | 2015-04 to 2021-08 | 77 | 6.4 yrs | ~85,000 | ~360,000 |
| **Tom Clancy's Rainbow Six Siege** | Tactical Shooter | 2015-12 to 2021-08 | 69 | 5.8 yrs | ~68,000 | ~200,000 |

### Exclusion Decisions
- **Apex Legends:** Although listed as a candidate in initial exploration, Apex Legends launched on Steam in November 2020. With only 10 monthly observations prior to dataset cutoff ($< 1$ year of history), it lacks sufficient observations to train multi-year seasonal models (such as SARIMA with 12-month lag or deep sequence models) and was excluded in favor of long-term continuous titles.

---

## Ingestion & Pipeline Execution

### Automated Execution (CLI)

To run the complete ingestion, validation, and processing pipeline:

```bash
# Using project virtual environment
python -m steam_player_forecasting.data.pipeline
```

Options:
- `--config <path>`: Specify alternative config YAML (defaults to `configs/default.yaml`).
- `--force-download`: Re-fetch or regenerate raw data from scratch.

### Ingestion Logic & Fallback Behavior

1. **Existing File:** If `data/raw/All_Steam_Games.csv` exists, it is loaded directly.
2. **Kaggle CLI / API:** If absent and Kaggle credentials exist (`~/.kaggle/kaggle.json` or `KAGGLE_USERNAME` / `KAGGLE_KEY`), the pipeline downloads `jackogozaly/steam-player-data`.
3. **Synthetic Fallback Generator:** If credentials are not present (e.g. in CI or offline development), the pipeline deterministically generates a high-fidelity synthetic raw dataset covering the candidate games with realistic trends, seasonality, scraping quirks, and noise.

### Programmatic Data Loading

Reusable functions are exposed in `steam_player_forecasting.data`:

```python
from steam_player_forecasting.data import (
    load_processed_data,
    load_raw_data,
    get_available_games,
)

# 1. Load combined standardized dataset
df_all = load_processed_data()

# 2. Load individual game time-series
csgo_df = load_processed_data(game_name="Counter-Strike: Global Offensive")

# 3. List all validated games
games = get_available_games()
print(games)
```

---

## Strict Rules for Downstream Time-Series Work (Phase 2+)

1. **Chronological Integrity:** Never shuffle observations. Train/val/test splits must follow strict temporal splits (e.g., hold out the last 12 months for test, preceding 12 months for validation).
2. **Leakage Prevention:** Any scalers, interpolators, or transforms must be fitted **only** on the training window and transformed on validation/test windows.
3. **Consistency:** All competing models must be trained and evaluated on identical test horizons.
