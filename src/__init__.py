"""Steam Player Forecasting.

A benchmarking framework comparing statistical, machine learning, and deep learning
models for forecasting monthly Steam player counts.
"""

from __future__ import annotations

import sys

__version__ = "0.1.0"
__all__ = ["__version__"]

# Provide backwards compatibility alias
sys.modules.setdefault("steam_player_forecasting", sys.modules[__name__])
