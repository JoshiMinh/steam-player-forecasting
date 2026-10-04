"""Data loading and configuration utilities."""

from pathlib import Path
from typing import Any
import yaml


def get_project_root() -> Path:
    """Return the repository root directory."""
    return Path(__file__).resolve().parent.parent.parent.parent


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML project configuration.

    If config_path is None, defaults to `configs/default.yaml` relative to project root.
    """
    if config_path is None:
        config_path = get_project_root() / "configs" / "default.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config
