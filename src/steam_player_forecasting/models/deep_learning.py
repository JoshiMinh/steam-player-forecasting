"""Deep learning recurrent forecasting models (RNN, LSTM, GRU) in PyTorch.

Implements sliding window time-series dataset and recurrent neural architectures
(Standard RNN, LSTM, GRU) with multi-step linear forecast heads. Features:
  - Strictly train-fitted MinMax scaling to prevent leakage.
  - Early stopping with patience and best-checkpoint recovery.
  - Learning rate scheduling via ReduceLROnPlateau.
  - Fixed random seeds (seed=42) for deterministic reproducibility.
  - Lightweight and fully runnable on CPU.
"""

from __future__ import annotations

import copy
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from steam_player_forecasting.features.scalers import TimeSeriesScaler
from steam_player_forecasting.models.base import BaseForecaster


def set_seed(seed: int = 42) -> None:
    """Set random seed across Python, NumPy, and PyTorch for deterministic execution."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TimeSeriesSequenceDataset(Dataset):
    """PyTorch Dataset emitting sliding window sequence pairs (X, y).

    Given a 1D time-series, creates samples where:
      - X has shape (seq_length, input_dim) representing past observations [t, ..., t+seq_length-1].
      - y has shape (horizon,) representing future observations [t+seq_length, ..., t+seq_length+horizon-1].

    Attributes:
        seq_length: Number of historical time-steps in input sequence.
        horizon: Number of future time-steps in target forecast.
    """

    def __init__(
        self,
        series: np.ndarray | pd.Series,
        seq_length: int = 12,
        horizon: int = 12,
    ) -> None:
        self.seq_length = seq_length
        self.horizon = horizon

        arr = np.asarray(series, dtype=np.float32).ravel()
        window_size = seq_length + horizon

        if len(arr) < window_size:
            raise ValueError(
                f"Series length ({len(arr)}) is shorter than window size (seq_length={seq_length} + horizon={horizon} = {window_size})."
            )

        n_samples = len(arr) - window_size + 1
        X_list: list[np.ndarray] = []
        y_list: list[np.ndarray] = []

        for i in range(n_samples):
            X_list.append(arr[i : i + seq_length])
            y_list.append(arr[i + seq_length : i + window_size])

        # Shapes: X -> (N, seq_length, 1), y -> (N, horizon)
        self.X = torch.tensor(np.array(X_list), dtype=torch.float32).unsqueeze(-1)
        self.y = torch.tensor(np.array(y_list), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


# ==============================================================================
# PyTorch Neural Network Modules
# ==============================================================================


class RNNModel(nn.Module):
    """Standard Elman Recurrent Neural Network with a linear forecast head."""

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 32,
        num_layers: int = 1,
        dropout: float = 0.0,
        horizon: int = 12,
    ) -> None:
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        last_hidden = out[:, -1, :]
        return self.fc(last_hidden)


class LSTMModel(nn.Module):
    """Long Short-Term Memory (LSTM) network with a linear forecast head."""

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 32,
        num_layers: int = 1,
        dropout: float = 0.0,
        horizon: int = 12,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        return self.fc(last_hidden)


class GRUModel(nn.Module):
    """Gated Recurrent Unit (GRU) network with a linear forecast head."""

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 32,
        num_layers: int = 1,
        dropout: float = 0.0,
        horizon: int = 12,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        last_hidden = out[:, -1, :]
        return self.fc(last_hidden)


# ==============================================================================
# Base Recurrent Forecaster (Scikit-Learn / BaseForecaster Wrapper)
# ==============================================================================


class BaseRecurrentForecaster(BaseForecaster):
    """Base class for PyTorch recurrent time-series forecasting models.

    Manages data scaling, PyTorch training loop, learning rate scheduling,
    early stopping, in-sample evaluation, and out-of-sample forecast generation.

    Attributes:
        model_type: 'rnn', 'lstm', or 'gru'.
        seq_length: History window length (default 12).
        horizon: Forecast horizon length (default 12).
        hidden_dim: Hidden state dimension (default 32).
        num_layers: Number of recurrent layers (default 1).
        dropout: Dropout probability between recurrent layers (default 0.0).
        learning_rate: Initial Adam learning rate (default 0.01).
        epochs: Maximum training epochs (default 60).
        batch_size: Mini-batch size (default 8).
        patience: Early stopping patience epochs (default 15).
        min_delta: Minimum loss decrease required to reset patience (default 1e-4).
        scale: Whether to scale data to [0, 1] using train-only TimeSeriesScaler.
        seed: Random seed for deterministic reproducibility.
        device: 'cpu' or 'cuda'.
        train_losses_: History of training loss per epoch.
    """

    def __init__(
        self,
        model_type: Literal["rnn", "lstm", "gru"] = "lstm",
        seq_length: int = 12,
        horizon: int = 12,
        hidden_dim: int = 32,
        num_layers: int = 1,
        dropout: float = 0.0,
        learning_rate: float = 0.01,
        epochs: int = 60,
        batch_size: int = 8,
        patience: int = 15,
        min_delta: float = 1e-4,
        scale: bool = True,
        seed: int = 42,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.model_type = model_type
        self.seq_length = seq_length
        self.horizon = horizon
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.min_delta = min_delta
        self.scale = scale
        self.seed = seed
        self.device = device

        self.net_: nn.Module | None = None
        self.scaler_: TimeSeriesScaler | None = None
        self.train_losses_: list[float] = []
        self._scaled_train: np.ndarray | None = None

    def _build_network(self) -> nn.Module:
        """Instantiate the PyTorch neural network module."""
        kwargs = {
            "input_dim": 1,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "horizon": self.horizon,
        }
        if self.model_type == "rnn":
            return RNNModel(**kwargs)
        elif self.model_type == "lstm":
            return LSTMModel(**kwargs)
        elif self.model_type == "gru":
            return GRUModel(**kwargs)
        else:
            raise ValueError(f"Unknown model_type '{self.model_type}'. Choose 'rnn', 'lstm', or 'gru'.")

    def _fit(self, y_arr: np.ndarray) -> None:
        """Fit recurrent model on training series."""
        set_seed(self.seed)

        # 1. Scale data strictly on train if requested
        if self.scale:
            self.scaler_ = TimeSeriesScaler(scaler_type="minmax")
            self._scaled_train = self.scaler_.fit_transform(y_arr)
        else:
            self.scaler_ = None
            self._scaled_train = y_arr.copy().astype(float)

        # 2. Build dataset and dataloader
        dataset = TimeSeriesSequenceDataset(
            series=self._scaled_train,
            seq_length=self.seq_length,
            horizon=self.horizon,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
        )

        # 3. Initialize model, optimizer, scheduler, criterion
        self.net_ = self._build_network().to(self.device)
        optimizer = torch.optim.Adam(self.net_.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        criterion = nn.MSELoss()

        # 4. Training loop with early stopping
        best_loss = float("inf")
        best_weights = copy.deepcopy(self.net_.state_dict())
        epochs_no_improve = 0
        self.train_losses_ = []

        for epoch in range(self.epochs):
            self.net_.train()
            epoch_loss = 0.0
            for bx, by in dataloader:
                bx = bx.to(self.device)
                by = by.to(self.device)

                optimizer.zero_grad()
                pred = self.net_(bx)
                loss = criterion(pred, by)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * len(bx)

            avg_loss = epoch_loss / len(dataset)
            self.train_losses_.append(avg_loss)
            scheduler.step(avg_loss)

            # Early stopping check
            if avg_loss < (best_loss - self.min_delta):
                best_loss = avg_loss
                best_weights = copy.deepcopy(self.net_.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    break

        # Restore best model weights
        self.net_.load_state_dict(best_weights)
        self.net_.eval()

        # 5. In-sample fitted values and residuals
        # For each sample i in dataset, prediction starts at seq_length + i
        fitted = np.full(len(y_arr), np.nan, dtype=float)
        with torch.no_grad():
            for i in range(len(dataset)):
                x_sample = dataset.X[i : i + 1].to(self.device)
                pred_scaled = self.net_(x_sample).squeeze(0).cpu().numpy()
                if self.scaler_ is not None:
                    pred_level = self.scaler_.inverse_transform(pred_scaled)
                else:
                    pred_level = pred_scaled

                # Fitted value for step seq_length + i is the 1-step forecast of that window
                step_idx = self.seq_length + i
                if step_idx < len(y_arr):
                    fitted[step_idx] = pred_level[0]

        self._fittedvalues = fitted
        self._resid = y_arr - fitted

    def _predict(self, steps: int) -> np.ndarray:
        """Generate out-of-sample forecast."""
        if self.net_ is None or self._scaled_train is None:
            raise ValueError("Model must be fitted before predict.")

        self.net_.eval()
        with torch.no_grad():
            # Extract last seq_length values from training series
            last_seq = self._scaled_train[-self.seq_length :]
            x_input = (
                torch.tensor(last_seq, dtype=torch.float32)
                .unsqueeze(0)
                .unsqueeze(-1)
                .to(self.device)
            )

            # Direct multi-step forward pass: returns (1, horizon)
            pred_scaled = self.net_(x_input).squeeze(0).detach().cpu().numpy()

            if self.scaler_ is not None:
                pred_level = self.scaler_.inverse_transform(pred_scaled)
            else:
                pred_level = pred_scaled

            if steps <= self.horizon:
                return np.asarray(pred_level[:steps], dtype=float)

            # If requested steps exceed horizon, project recursively
            full_preds = list(pred_level)
            while len(full_preds) < steps:
                # Append last known predictions
                extended = list(self._scaled_train) + [
                    self.scaler_.transform(np.array([p]))[0] if self.scaler_ else p
                    for p in full_preds
                ]
                seq_in = (
                    torch.tensor(extended[-self.seq_length :], dtype=torch.float32)
                    .unsqueeze(0)
                    .unsqueeze(-1)
                    .to(self.device)
                )
                nxt_scaled = self.net_(seq_in).squeeze(0).detach().cpu().numpy()
                nxt_level = (
                    self.scaler_.inverse_transform(nxt_scaled)
                    if self.scaler_
                    else nxt_scaled
                )
                full_preds.extend(nxt_level)

            return np.asarray(full_preds[:steps], dtype=float)



class RNNForecaster(BaseRecurrentForecaster):
    """Standard Recurrent Neural Network (Elman RNN) forecaster."""

    def __init__(
        self,
        seq_length: int = 12,
        horizon: int = 12,
        hidden_dim: int = 32,
        num_layers: int = 1,
        dropout: float = 0.0,
        learning_rate: float = 0.01,
        epochs: int = 60,
        batch_size: int = 8,
        patience: int = 15,
        min_delta: float = 1e-4,
        scale: bool = True,
        seed: int = 42,
        device: str = "cpu",
    ) -> None:
        super().__init__(
            model_type="rnn",
            seq_length=seq_length,
            horizon=horizon,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            min_delta=min_delta,
            scale=scale,
            seed=seed,
            device=device,
        )


class LSTMForecaster(BaseRecurrentForecaster):
    """Long Short-Term Memory (LSTM) recurrent neural network forecaster."""

    def __init__(
        self,
        seq_length: int = 12,
        horizon: int = 12,
        hidden_dim: int = 32,
        num_layers: int = 1,
        dropout: float = 0.0,
        learning_rate: float = 0.01,
        epochs: int = 60,
        batch_size: int = 8,
        patience: int = 15,
        min_delta: float = 1e-4,
        scale: bool = True,
        seed: int = 42,
        device: str = "cpu",
    ) -> None:
        super().__init__(
            model_type="lstm",
            seq_length=seq_length,
            horizon=horizon,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            min_delta=min_delta,
            scale=scale,
            seed=seed,
            device=device,
        )


class GRUForecaster(BaseRecurrentForecaster):
    """Gated Recurrent Unit (GRU) recurrent neural network forecaster."""

    def __init__(
        self,
        seq_length: int = 12,
        horizon: int = 12,
        hidden_dim: int = 32,
        num_layers: int = 1,
        dropout: float = 0.0,
        learning_rate: float = 0.01,
        epochs: int = 60,
        batch_size: int = 8,
        patience: int = 15,
        min_delta: float = 1e-4,
        scale: bool = True,
        seed: int = 42,
        device: str = "cpu",
    ) -> None:
        super().__init__(
            model_type="gru",
            seq_length=seq_length,
            horizon=horizon,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            min_delta=min_delta,
            scale=scale,
            seed=seed,
            device=device,
        )
