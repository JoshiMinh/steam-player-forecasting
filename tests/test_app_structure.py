"""Tests for Streamlit application structure, assets, and component modules."""

from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import pytest


def test_steam_logo_assets_exist_and_valid() -> None:
    """Verify that steam_logo.png exists in both assets and app/assets and is valid."""
    root = Path(__file__).resolve().parent.parent
    logo1 = root / "assets" / "steam_logo.png"
    logo2 = root / "app" / "assets" / "steam_logo.png"

    assert logo1.exists(), f"Missing {logo1}"
    assert logo2.exists(), f"Missing {logo2}"

    for p in [logo1, logo2]:
        assert p.stat().st_size > 1000, f"File {p} is suspiciously small: {p.stat().st_size} bytes"
        with Image.open(p) as img:
            assert img.format == "PNG"
            assert img.width > 0
            assert img.height > 0
            assert img.mode in ("RGBA", "RGB")


def test_steam_stylesheet_exists() -> None:
    """Verify that Steam CSS stylesheet exists and contains expected rules."""
    root = Path(__file__).resolve().parent.parent
    css_path = root / "app" / "styles" / "steam.css"
    assert css_path.exists(), f"Missing {css_path}"
    content = css_path.read_text(encoding="utf-8")
    assert "#1b2838" in content, "Missing primary Steam slate background color"
    assert "#171a21" in content, "Missing primary Steam navy header color"
    assert "#66c0f4" in content, "Missing Steam cyan accent color"
    assert "#a4d007" in content, "Missing Steam discount badge green"


def test_component_imports_and_helpers() -> None:
    """Verify that all components import cleanly and helpers function correctly."""
    from app.components.header import get_base64_logo
    from app.components.plots import STEAM_PALETTE, apply_steam_chart_style

    root = Path(__file__).resolve().parent.parent
    logo_path = root / "app" / "assets" / "steam_logo.png"

    # Base64 logo encoding
    b64_str = get_base64_logo(logo_path)
    assert b64_str.startswith("data:image/png;base64,")
    assert len(b64_str) > 100

    # Palette
    for model in ["SARIMA", "XGBoost", "LSTM", "GRU"]:
        assert model in STEAM_PALETTE

    # Plot styling
    fig, ax = plt.subplots()
    apply_steam_chart_style(ax, fig)
    assert fig.patch.get_facecolor()[:3] == pytest.approx((0.08627, 0.12549, 0.17647), abs=0.01)
    plt.close(fig)


def test_app_data_loaders() -> None:
    """Verify that app data loading utilities correctly access test artifacts."""
    from app.app import load_forecast_cache, load_benchmarks_data

    cache = load_forecast_cache()
    assert isinstance(cache, dict)
    assert len(cache) >= 7, "Expected at least 7 games in forecast cache"
    assert "Counter-Strike: Global Offensive" in cache

    test_b, lead_b, val_comp = load_benchmarks_data()
    assert not test_b.empty
    assert not lead_b.empty
    assert not val_comp.empty
    assert "Game" in test_b.columns
    assert "Model" in test_b.columns
