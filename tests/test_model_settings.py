"""Tests for the shared model settings read from default.yaml."""

from __future__ import annotations

from src.cli.model_settings import (
    DEFAULTS,
    load_model_settings,
    recurrent_from_settings,
    ridge_from_settings,
    xgboost_from_settings,
    xgboost_params_for_game,
)
from src.data.loader import load_config
from src.models.deep_learning import LSTMForecaster


def _assert_mirrors(defaults: dict, yaml_block: dict, path: str = "models") -> None:
    for key, value in defaults.items():
        if key == "per_game":  # filled from the yaml only
            continue
        assert key in yaml_block, f"{path}.{key} missing from default.yaml"
        if isinstance(value, dict):
            _assert_mirrors(value, yaml_block[key], f"{path}.{key}")
        else:
            assert yaml_block[key] == value, f"{path}.{key}: yaml={yaml_block[key]!r} code default={value!r}"


def test_code_defaults_mirror_yaml():
    """DEFAULTS is only a fallback; it must not silently drift from default.yaml."""
    _assert_mirrors(DEFAULTS, load_config()["models"])


def test_yaml_values_are_used_and_override_defaults():
    over = load_model_settings({"models": {"machine_learning": {"ridge": {"alpha": 7.5}}}})
    assert over["machine_learning"]["ridge"]["alpha"] == 7.5
    # untouched keys keep their defaults
    assert over["deep_learning"] == DEFAULTS["deep_learning"]
    assert ridge_from_settings(over).alpha == 7.5


def test_published_configuration_is_unchanged():
    """These are the values behind the published Phase 5 numbers."""
    s = load_model_settings()
    dl = s["deep_learning"]
    assert (dl["seq_length"], dl["horizon"], dl["hidden_dim"], dl["num_layers"]) == (12, 12, 32, 1)
    assert (dl["epochs"], dl["learning_rate"], dl["patience"], dl["seed"]) == (60, 0.01, 15, 42)
    rf = s["machine_learning"]["random_forest"]
    assert (rf["n_estimators"], rf["max_depth"], rf["random_state"]) == (100, 6, 42)
    assert s["machine_learning"]["ridge"]["alpha"] == 1.0
    assert s["statistical"]["sarima"]["maxiter"] == 50
    assert tuple(s["statistical"]["arima"]["order"]) == (1, 1, 1)


def test_xgboost_per_game_params_and_default_fallback():
    s = load_model_settings()
    gta = xgboost_params_for_game(s, "Grand Theft Auto V")
    assert (gta["n_estimators"], gta["max_depth"], gta["learning_rate"]) == (150, 3, 0.02)
    assert gta["random_state"] == 42
    unknown = xgboost_params_for_game(s, "Not A Game")
    assert unknown == s["machine_learning"]["xgboost"]["defaults"]
    assert xgboost_from_settings(s, "Dota 2").n_estimators == 200


def test_every_per_game_xgboost_entry_has_full_params_and_a_known_game():
    cfg = load_config()
    per_game = cfg["models"]["machine_learning"]["xgboost"]["per_game"]
    assert len(per_game) == 7
    for params in per_game.values():
        assert {"n_estimators", "max_depth", "learning_rate"} <= set(params)


def test_recurrent_seed_and_epoch_overrides():
    s = load_model_settings()
    default = recurrent_from_settings(LSTMForecaster, s)
    assert default.seed == 42 and default.epochs == 60
    custom = recurrent_from_settings(LSTMForecaster, s, seed=3, epochs=2)
    assert custom.seed == 3 and custom.epochs == 2
    assert custom.hidden_dim == 32
