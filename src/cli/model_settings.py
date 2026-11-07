"""Model hyperparameters shared by Phase 4, Phase 5 and the rolling-origin backtest.

The values live in the ``models`` block of ``default.yaml``. ``DEFAULTS`` below
mirrors that block so the pipelines still run if a key is removed from the file.
Phase 3's SARIMA order grid and Phase 4's XGBoost tuning grid are search spaces,
not hyperparameters, and remain in their phase modules.
"""

from __future__ import annotations

import copy
from typing import Any

from src.data.loader import load_config
from src.models.deep_learning import BaseRecurrentForecaster
from src.models.statistical import (
    ARIMAForecaster,
    HoltWintersForecaster,
    NaiveForecaster,
    SARIMAForecaster,
    SeasonalNaiveForecaster,
)
from src.models.tabular import (
    RandomForestForecaster,
    RidgeForecaster,
    XGBoostForecaster,
)

DEFAULTS: dict[str, Any] = {
    "statistical": {
        "sarima": {
            "fallback_order": [1, 1, 1],
            "fallback_seasonal_order": [1, 1, 0, 12],
            "maxiter": 50,
        },
        "arima": {"order": [1, 1, 1]},
        "holt_winters": {"seasonal_periods": 12, "trend": "add", "seasonal": "add"},
        "seasonal_naive": {"seasonal_period": 12},
    },
    "machine_learning": {
        "ridge": {"alpha": 1.0},
        "random_forest": {"n_estimators": 100, "max_depth": 6, "random_state": 42},
        "xgboost": {
            "defaults": {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.05, "random_state": 42},
            "per_game": {},
        },
    },
    "deep_learning": {
        "seq_length": 12,
        "horizon": 12,
        "hidden_dim": 32,
        "num_layers": 1,
        "epochs": 60,
        "learning_rate": 0.01,
        "patience": 15,
        "seed": 42,
    },
}


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_model_settings(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return model settings from ``default.yaml`` merged over ``DEFAULTS``."""
    cfg = config if config is not None else load_config()
    return _merge(DEFAULTS, cfg.get("models", {}))


def naive_from_settings(settings: dict[str, Any]) -> NaiveForecaster:
    return NaiveForecaster()


def seasonal_naive_from_settings(settings: dict[str, Any]) -> SeasonalNaiveForecaster:
    return SeasonalNaiveForecaster(seasonal_period=settings["statistical"]["seasonal_naive"]["seasonal_period"])


def holt_winters_from_settings(settings: dict[str, Any], damped: bool = False) -> HoltWintersForecaster:
    hw = settings["statistical"]["holt_winters"]
    return HoltWintersForecaster(
        seasonal_periods=hw["seasonal_periods"],
        trend=hw["trend"],
        seasonal=hw["seasonal"],
        **({"damped_trend": True} if damped else {}),
    )


def arima_from_settings(settings: dict[str, Any]) -> ARIMAForecaster:
    return ARIMAForecaster(order=tuple(settings["statistical"]["arima"]["order"]))


def ridge_from_settings(settings: dict[str, Any]) -> RidgeForecaster:
    return RidgeForecaster(alpha=settings["machine_learning"]["ridge"]["alpha"])


def random_forest_from_settings(settings: dict[str, Any]) -> RandomForestForecaster:
    rf = settings["machine_learning"]["random_forest"]
    return RandomForestForecaster(
        n_estimators=rf["n_estimators"], max_depth=rf["max_depth"], random_state=rf["random_state"]
    )


def xgboost_params_for_game(settings: dict[str, Any], game: str) -> dict[str, Any]:
    """Tuned XGBoost parameters for a game, falling back to the defaults."""
    xgb = settings["machine_learning"]["xgboost"]
    return {**xgb["defaults"], **xgb.get("per_game", {}).get(game, {})}


def xgboost_from_settings(settings: dict[str, Any], game: str) -> XGBoostForecaster:
    p = xgboost_params_for_game(settings, game)
    return XGBoostForecaster(
        n_estimators=p["n_estimators"],
        max_depth=p["max_depth"],
        learning_rate=p["learning_rate"],
        random_state=p["random_state"],
    )


def recurrent_from_settings(
    cls: type[BaseRecurrentForecaster],
    settings: dict[str, Any],
    seed: int | None = None,
    epochs: int | None = None,
) -> BaseRecurrentForecaster:
    """Build an RNN/LSTM/GRU forecaster, optionally overriding the seed or epochs."""
    dl = settings["deep_learning"]
    return cls(
        seq_length=dl["seq_length"],
        horizon=dl["horizon"],
        hidden_dim=dl["hidden_dim"],
        num_layers=dl["num_layers"],
        epochs=dl["epochs"] if epochs is None else epochs,
        learning_rate=dl["learning_rate"],
        patience=dl["patience"],
        seed=dl["seed"] if seed is None else seed,
    )


def sarima_fallback_orders(settings: dict[str, Any]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    s = settings["statistical"]["sarima"]
    return tuple(s["fallback_order"]), tuple(s["fallback_seasonal_order"])


def sarima_from_settings(
    settings: dict[str, Any],
    order: tuple[int, ...],
    seasonal_order: tuple[int, ...],
    constrained: bool = False,
) -> SARIMAForecaster:
    """SARIMA with the configured iteration cap.

    Phase 5 fits with ``enforce_stationarity=False`` / ``enforce_invertibility=False``;
    ``constrained=True`` turns both on (used by the backtest's divergence fallback).
    """
    return SARIMAForecaster(
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=constrained,
        enforce_invertibility=constrained,
        maxiter=settings["statistical"]["sarima"]["maxiter"],
    )
