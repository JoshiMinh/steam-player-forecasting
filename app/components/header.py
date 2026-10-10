"""Steam Store Header component with constrained sizing and zero emojis."""

from __future__ import annotations

import base64
from pathlib import Path

import streamlit as st


def get_base64_logo(logo_path: Path) -> str:
    """Read image file and return base64 encoded data URI."""
    if logo_path.exists():
        with open(logo_path, "rb") as f:
            data = f.read()
            return f"data:image/png;base64,{base64.b64encode(data).decode('utf-8')}"
    return ""


def render_steam_header(assets_dir: Path) -> None:
    """Render clean, unified Steam top header bar without emojis or clutter."""
    logo_path = assets_dir / "steam_logo.png"
    if not logo_path.exists():
        logo_path = assets_dir.parent.parent / "assets" / "steam_logo.png"

    b64_logo = get_base64_logo(logo_path)
    img_tag = (
        f'<img src="{b64_logo}" alt="Steam Logo" '
        f'style="width: 38px; height: 38px; min-width: 38px; min-height: 38px; max-width: 38px; max-height: 38px; object-fit: contain;" />'
        if b64_logo
        else ""
    )

    header_html = f"""<div style="background: #171a21; border: 1px solid #2a475e; border-radius: 6px; padding: 10px 16px; margin-bottom: 16px; display: flex; align-items: center; justify-content: space-between; box-shadow: 0 4px 12px rgba(0,0,0,0.4);">
    <div style="display: flex; align-items: center; gap: 14px;">
        {img_tag}
        <div>
            <div style="font-size: 1.3rem; font-weight: 800; color: #ffffff; letter-spacing: 1.5px; text-transform: uppercase; margin: 0; line-height: 1.15;">Steam Forecasting Lab</div>
            <div style="font-size: 0.78rem; color: #66c0f4; letter-spacing: 0.8px; text-transform: uppercase; margin: 0;">Time-Series Benchmarking Architecture</div>
        </div>
    </div>
    <div style="background: #2a475e; color: #c7d5e0; font-size: 0.75rem; font-weight: 600; padding: 4px 10px; border-radius: 3px; letter-spacing: 0.5px;">12-Month Out-of-Sample Holdout</div>
</div>"""
    st.markdown(header_html, unsafe_allow_html=True)
