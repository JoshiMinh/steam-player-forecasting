"""Dataset acquisition and ingestion module.

Handles downloading the Kaggle Steam Player dataset or providing a reproducible
synthetic fallback dataset if Kaggle credentials are not present in CI / local environments.
"""

from __future__ import annotations

import logging
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from steam_player_forecasting.data.loader import get_project_root, load_config

logger = logging.getLogger(__name__)

KAGGLE_DATASET_ID = "jackogozaly/steam-player-data"
DEFAULT_RAW_FILENAME = "All_Steam_Games.csv"

# Representative games with realistic baseline parameters for synthetic generation
SYNTHETIC_GAME_PROFILES: dict[str, dict[str, Any]] = {
    "Counter-Strike: Global Offensive": {
        "start": "2012-07",
        "end": "2021-08",
        "base_avg": 180000.0,
        "trend_monthly": 4500.0,
        "seasonal_amp": 35000.0,
        "noise_std": 15000.0,
        "peak_multiplier": 1.75,
    },
    "Dota 2": {
        "start": "2012-07",
        "end": "2021-08",
        "base_avg": 250000.0,
        "trend_monthly": 1800.0,
        "seasonal_amp": 40000.0,
        "noise_std": 20000.0,
        "peak_multiplier": 1.85,
    },
    "Team Fortress 2": {
        "start": "2012-07",
        "end": "2021-08",
        "base_avg": 52000.0,
        "trend_monthly": 40.0,
        "seasonal_amp": 6000.0,
        "noise_std": 3500.0,
        "peak_multiplier": 1.60,
    },
    "Rust": {
        "start": "2013-12",
        "end": "2021-08",
        "base_avg": 22000.0,
        "trend_monthly": 750.0,
        "seasonal_amp": 8000.0,
        "noise_std": 4500.0,
        "peak_multiplier": 1.70,
    },
    "Warframe": {
        "start": "2013-03",
        "end": "2021-08",
        "base_avg": 26000.0,
        "trend_monthly": 280.0,
        "seasonal_amp": 5500.0,
        "noise_std": 3500.0,
        "peak_multiplier": 1.70,
    },
    "Grand Theft Auto V": {
        "start": "2015-04",
        "end": "2021-08",
        "base_avg": 55000.0,
        "trend_monthly": 850.0,
        "seasonal_amp": 12000.0,
        "noise_std": 8000.0,
        "peak_multiplier": 1.90,
    },
    "Tom Clancy's Rainbow Six Siege": {
        "start": "2015-12",
        "end": "2021-08",
        "base_avg": 28000.0,
        "trend_monthly": 1100.0,
        "seasonal_amp": 10000.0,
        "noise_std": 6000.0,
        "peak_multiplier": 1.80,
    },
    "PAYDAY 2": {
        "start": "2013-08",
        "end": "2021-08",
        "base_avg": 30000.0,
        "trend_monthly": -50.0,
        "seasonal_amp": 4500.0,
        "noise_std": 3000.0,
        "peak_multiplier": 1.65,
    },
    "Terraria": {
        "start": "2012-07",
        "end": "2021-08",
        "base_avg": 24000.0,
        "trend_monthly": 200.0,
        "seasonal_amp": 5000.0,
        "noise_std": 3000.0,
        "peak_multiplier": 1.80,
    },
    "Apex Legends": {
        # Short history test case (< 1 year on Steam as of August 2021)
        "start": "2020-11",
        "end": "2021-08",
        "base_avg": 85000.0,
        "trend_monthly": 4200.0,
        "seasonal_amp": 6000.0,
        "noise_std": 5000.0,
        "peak_multiplier": 1.95,
    },
}


def has_kaggle_credentials() -> bool:
    """Check whether Kaggle API credentials exist in environment or ~/.kaggle/kaggle.json."""
    if "KAGGLE_USERNAME" in os.environ and "KAGGLE_KEY" in os.environ:
        return True
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    return kaggle_json.is_file()


def download_from_kaggle(
    dataset_id: str = KAGGLE_DATASET_ID,
    download_dir: Path | str | None = None,
) -> Path | None:
    """Attempt to download dataset from Kaggle via kaggle CLI or kaggle python module.

    Returns the path to the downloaded/extracted raw CSV, or None if download fails.
    """
    if download_dir is None:
        download_dir = get_project_root() / "data" / "raw"
    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)

    if not has_kaggle_credentials():
        logger.warning("Kaggle credentials not detected. Skipping Kaggle download.")
        return None

    # Try kaggle CLI
    if shutil.which("kaggle"):
        try:
            logger.info("Attempting download via Kaggle CLI: %s", dataset_id)
            cmd = [
                "kaggle",
                "datasets",
                "download",
                "-d",
                dataset_id,
                "-p",
                str(download_dir),
                "--unzip",
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if res.returncode == 0:
                csvs = list(download_dir.glob("*.csv"))
                if csvs:
                    return csvs[0]
            else:
                logger.warning("Kaggle CLI failed: %s", res.stderr)
        except Exception as e:
            logger.warning("Kaggle CLI execution encountered error: %s", e)

    # Try kaggle python package
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        logger.info("Downloading dataset via Kaggle Python API: %s", dataset_id)
        api.dataset_download_files(dataset_id, path=str(download_dir), unzip=True)
        csvs = list(download_dir.glob("*.csv"))
        if csvs:
            return csvs[0]
    except Exception as e:
        logger.warning("Kaggle Python API download failed: %s", e)

    return None


def generate_synthetic_raw_data(
    output_path: Path | str | None = None,
    profiles: dict[str, dict[str, Any]] | None = None,
    seed: int = 42,
    include_raw_artifacts: bool = True,
) -> pd.DataFrame:
    """Generate a high-fidelity synthetic raw dataset mimicking SteamCharts scraper output.

    Features generated:
    - Multi-year monthly time-series for representative Steam titles
    - Realistic baseline averages, growth trends, seasonal winter/summer spikes, and noise
    - Realistic peak player counts (1.5x - 2.0x average)
    - Raw scraping artifacts such as 'Last 30 Days' rows, comma formatting, percentages,
      and '-' for uncomputable initial month gains.
    """
    rng = np.random.default_rng(seed)
    if profiles is None:
        profiles = SYNTHETIC_GAME_PROFILES

    records: list[dict[str, Any]] = []

    for game_name, prof in profiles.items():
        dates = pd.date_range(start=prof["start"], end=prof["end"], freq="MS")
        n = len(dates)
        if n == 0:
            continue

        base_avg = prof["base_avg"]
        trend = prof["trend_monthly"]
        seasonal_amp = prof["seasonal_amp"]
        noise_std = prof["noise_std"]
        peak_mult = prof["peak_multiplier"]

        # Simulate time-series in chronological order
        avg_vals: list[float] = []
        for i, dt in enumerate(dates):
            # Annual seasonality: peaks in December/January (month 12, 1) and July (month 7)
            # month_phase maps 1..12 to [0, 2*pi] with peak around month 12 and month 7
            month = dt.month
            seasonal = seasonal_amp * (
                0.7 * math.cos(2 * math.pi * (month - 1) / 12)
                + 0.5 * math.cos(4 * math.pi * (month - 7) / 12)
            )
            trend_val = trend * i
            noise = float(rng.normal(0, noise_std))
            val = max(100.0, base_avg + trend_val + seasonal + noise)
            avg_vals.append(round(val, 2))

        # Build records in reverse-chronological order (as scraped by SteamCharts)
        prev_val: float | None = None
        game_rows: list[dict[str, Any]] = []

        for i in range(n):
            dt = dates[i]
            val = avg_vals[i]
            month_str = dt.strftime("%B %Y")  # e.g., "September 2021"

            if i == 0:
                gain_val = None
                pct_val = None
            else:
                diff = val - avg_vals[i - 1]
                gain_val = round(diff, 2)
                pct_val = round((diff / avg_vals[i - 1]) * 100.0, 2)

            peak_noise = float(rng.uniform(0.9, 1.15))
            peak_val = int(round(val * peak_mult * peak_noise))
            peak_val = max(peak_val, int(math.ceil(val)))

            # Format raw strings with commas and signs to mimic raw scrapes
            if include_raw_artifacts:
                gain_str = f"{gain_val:+.2f}" if gain_val is not None else "-"
                pct_str = f"{pct_val:+.2f}%" if pct_val is not None else "-"
                avg_fmt: str | float = f"{val:,.2f}" if rng.random() > 0.4 else val
                peak_fmt: str | int = f"{peak_val:,}" if rng.random() > 0.4 else peak_val
            else:
                gain_str = str(gain_val) if gain_val is not None else ""
                pct_str = str(pct_val) if pct_val is not None else ""
                avg_fmt = val
                peak_fmt = peak_val

            row = {
                "Month_Year": month_str,
                "Game_Name": game_name,
                "Avg_players": avg_fmt,
                "Gain": gain_str,
                "% Gain": pct_str,
                "Peak_Players": peak_fmt,
            }
            game_rows.append(row)

        # SteamCharts scraping puts recent months first
        game_rows.reverse()

        # Add the classic SteamCharts "Last 30 Days" row at the top
        if include_raw_artifacts:
            last_avg = avg_vals[-1]
            last_peak = int(round(last_avg * peak_mult))
            game_rows.insert(
                0,
                {
                    "Month_Year": "Last 30 Days",
                    "Game_Name": game_name,
                    "Avg_players": f"{last_avg:,.2f}",
                    "Gain": "+152.40",
                    "% Gain": "+0.35%",
                    "Peak_Players": f"{last_peak:,}",
                },
            )

        records.extend(game_rows)

    df = pd.DataFrame(records)

    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_p, index=False)
        logger.info("Saved synthetic raw dataset to: %s (%d records)", out_p, len(df))

    return df


def acquire_dataset(
    config: dict[str, Any] | None = None,
    force_download: bool = False,
    create_sample_if_missing: bool = True,
) -> Path:
    """Ensure raw dataset exists in data/raw.

    Steps:
    1. Check if raw CSV already exists in data/raw.
    2. If not found and Kaggle credentials exist, attempt Kaggle download.
    3. If still missing and create_sample_if_missing is True, generate synthetic fallback dataset.
    4. Return Path to raw dataset.
    """
    if config is None:
        config = load_config()

    root = get_project_root()
    raw_dir = root / config.get("data", {}).get("raw_dir", "data/raw")
    raw_filename = config.get("data", {}).get("raw_filename", DEFAULT_RAW_FILENAME)
    target_path = raw_dir / raw_filename

    # 1. Existing file check
    if target_path.exists() and not force_download:
        logger.info("Found existing raw dataset: %s", target_path)
        return target_path

    # Check if any CSV is in raw_dir
    if not force_download and raw_dir.exists():
        existing_csvs = list(raw_dir.glob("*.csv"))
        if existing_csvs:
            logger.info("Found existing CSV in raw dir: %s", existing_csvs[0])
            return existing_csvs[0]

    raw_dir.mkdir(parents=True, exist_ok=True)

    # 2. Kaggle download attempt
    downloaded = download_from_kaggle(download_dir=raw_dir)
    if downloaded and downloaded.exists():
        logger.info("Successfully acquired raw data via Kaggle: %s", downloaded)
        return downloaded

    # 3. Synthetic sample fallback
    if create_sample_if_missing:
        logger.info(
            "Generating reproducible synthetic raw dataset at %s", target_path
        )
        generate_synthetic_raw_data(
            output_path=target_path,
            seed=config.get("project", {}).get("seed", 42),
            include_raw_artifacts=True,
        )
        return target_path

    raise FileNotFoundError(
        f"Raw dataset not found at {target_path} and Kaggle credentials unavailable. "
        "Please place raw CSV into data/raw/ or configure Kaggle API."
    )
