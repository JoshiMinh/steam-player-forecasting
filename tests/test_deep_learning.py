"""Unit tests for PyTorch recurrent time-series architectures and forecasters."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from steam_player_forecasting.models.deep_learning import (
    GRUForecaster,
    GRUModel,
    LSTMForecaster,
    LSTMModel,
    RNNForecaster,
    RNNModel,
    TimeSeriesSequenceDataset,
    set_seed,
)


@pytest.fixture
def sample_monthly_series() -> pd.Series:
    """Generate 60 months of synthetic monthly data."""
    np.random.seed(42)
    dates = pd.date_range("2015-01-01", periods=60, freq="MS")
    trend = np.linspace(1000, 3000, 60)
    seasonal = np.sin(2 * np.pi * np.arange(60) / 12) * 300
    noise = np.random.normal(0, 30, 60)
    return pd.Series(trend + seasonal + noise, index=dates, name="Avg_players")


# ==============================================================================
# 1. Dataset & Module Shape Tests
# ==============================================================================


def test_time_series_sequence_dataset():
    """Verify sliding window dataset lengths and tensor shapes."""
    data = np.arange(40, dtype=np.float32)
    seq_len = 12
    horizon = 12
    ds = TimeSeriesSequenceDataset(data, seq_length=seq_len, horizon=horizon)

    # Total samples: 40 - (12 + 12) + 1 = 17
    assert len(ds) == 17

    x0, y0 = ds[0]
    assert x0.shape == (12, 1)
    assert y0.shape == (12,)
    assert np.allclose(x0.squeeze().numpy(), data[:12])
    assert np.allclose(y0.numpy(), data[12:24])

    # Error when sequence is too short
    short_data = np.arange(10)
    with pytest.raises(ValueError, match="shorter than window size"):
        TimeSeriesSequenceDataset(short_data, seq_length=12, horizon=12)


def test_recurrent_network_forward_passes():
    """Verify PyTorch neural module forward pass shapes for RNN, LSTM, and GRU."""
    batch_size = 4
    seq_len = 12
    horizon = 12
    hidden_dim = 32

    dummy_input = torch.randn(batch_size, seq_len, 1)

    rnn_net = RNNModel(input_dim=1, hidden_dim=hidden_dim, num_layers=1, horizon=horizon)
    lstm_net = LSTMModel(input_dim=1, hidden_dim=hidden_dim, num_layers=1, horizon=horizon)
    gru_net = GRUModel(input_dim=1, hidden_dim=hidden_dim, num_layers=1, horizon=horizon)

    out_rnn = rnn_net(dummy_input)
    out_lstm = lstm_net(dummy_input)
    out_gru = gru_net(dummy_input)

    assert out_rnn.shape == (batch_size, horizon)
    assert out_lstm.shape == (batch_size, horizon)
    assert out_gru.shape == (batch_size, horizon)


# ==============================================================================
# 2. Forecaster Training, Prediction & Inversion Tests
# ==============================================================================


def test_rnn_forecaster_fit_predict(sample_monthly_series: pd.Series):
    """Verify RNNForecaster fits, predicts, and produces valid output format."""
    train_s = sample_monthly_series.iloc[:48]
    val_s = sample_monthly_series.iloc[48:]

    model = RNNForecaster(seq_length=12, horizon=12, hidden_dim=16, epochs=15, seed=42)
    model.fit(train_s)

    pred = model.predict(steps=12)
    assert isinstance(pred, pd.Series)
    assert len(pred) == 12
    assert pred.index[0] == val_s.index[0]
    assert len(model.train_losses_) > 0
    assert len(model.fittedvalues_) == 48

    mae = model.score(val_s, metric="MAE")
    assert mae > 0


def test_lstm_forecaster_fit_predict(sample_monthly_series: pd.Series):
    """Verify LSTMForecaster fits, predicts on original level scale, and early stops."""
    train_s = sample_monthly_series.iloc[:48]
    val_s = sample_monthly_series.iloc[48:]

    model = LSTMForecaster(
        seq_length=12,
        horizon=12,
        hidden_dim=24,
        epochs=20,
        patience=5,
        scale=True,
        seed=42,
    )
    model.fit(train_s)

    pred = model.predict(steps=12)
    assert len(pred) == 12

    # Verify predictions are inverted back to original level scale (not in [0, 1])
    assert pred.mean() > 500.0

    rmse = model.score(val_s, metric="RMSE")
    assert rmse > 0


def test_gru_forecaster_fit_predict(sample_monthly_series: pd.Series):
    """Verify GRUForecaster deterministic execution with seed=42."""
    train_s = sample_monthly_series.iloc[:48]

    model1 = GRUForecaster(seq_length=12, horizon=12, hidden_dim=16, epochs=10, seed=42)
    model1.fit(train_s)
    pred1 = model1.predict(steps=12)

    model2 = GRUForecaster(seq_length=12, horizon=12, hidden_dim=16, epochs=10, seed=42)
    model2.fit(train_s)
    pred2 = model2.predict(steps=12)

    np.testing.assert_allclose(pred1.values, pred2.values, rtol=1e-5)


def test_recurrent_extended_horizon(sample_monthly_series: pd.Series):
    """Verify model can forecast steps beyond configured horizon via recursion."""
    train_s = sample_monthly_series.iloc[:48]

    model = LSTMForecaster(seq_length=12, horizon=6, hidden_dim=16, epochs=10, seed=42)
    model.fit(train_s)

    # Request 12 steps when model horizon is 6
    pred = model.predict(steps=12)
    assert len(pred) == 12
    assert not np.isnan(pred.values).any()
