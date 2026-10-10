"""Sequence models: LSTM and small Transformer with quantile regression.

The encoder consumes the last ``history_len`` athlete-seasons (each with
engineered features) and a linear head emits ``horizons x n_quantiles``
predictions, trained with the pinball (quantile) loss so the model
directly learns predictive quantiles (uncertainty ranges).
"""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.features import build_training_samples
from src.models.base import Forecaster, History


class SeqForecaster(nn.Module):
    """LSTM or Transformer encoder with a quantile regression head."""

    def __init__(
        self,
        n_features: int,
        hidden: int = 64,
        horizons: int = 5,
        quantiles=(0.1, 0.5, 0.9),
        kind: str = "lstm",
        num_layers: int = 1,
        nhead: int = 4,
        ff_dim: int = 128,
        dropout: float = 0.1,
        history_len: int = 10,
    ):
        super().__init__()
        if kind not in ("lstm", "transformer"):
            raise ValueError(f"kind must be 'lstm' or 'transformer', got {kind!r}")
        self.kind = kind
        self.horizons = horizons
        self.quantiles = list(quantiles)
        self.history_len = history_len
        out_dim = horizons * len(quantiles)

        if kind == "lstm":
            self.encoder = nn.LSTM(
                n_features, hidden, num_layers,
                batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
            )
        else:
            self.input_proj = nn.Linear(n_features, hidden)
            self.pos = nn.Parameter(torch.zeros(1, history_len, hidden))
            layer = nn.TransformerEncoderLayer(
                hidden, nhead, ff_dim, dropout=dropout, batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """x: (N, L, F) front-padded (right-aligned); returns (N, H, Q)."""
        if self.kind == "lstm":
            out, _ = self.encoder(x)
            last = out[:, -1, :]  # right-aligned: last position is the latest season
        else:
            pad_mask = (
                torch.arange(x.size(1), device=x.device)[None, :]
                < (x.size(1) - lengths)[:, None]
            )
            h = self.input_proj(x) + self.pos[:, : x.size(1)]
            out = self.encoder(h, src_key_padding_mask=pad_mask)
            last = out[:, -1, :]
        return self.head(last).view(x.size(0), self.horizons, len(self.quantiles))


def quantile_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, quantiles) -> torch.Tensor:
    """Mean pinball loss over valid (horizon, quantile) pairs."""
    err = target.unsqueeze(-1) - pred  # (N, H, Q)
    taus = torch.tensor(list(quantiles), dtype=pred.dtype, device=pred.device).view(1, 1, -1)
    loss = torch.maximum(taus * err, (taus - 1.0) * err)
    m = mask.unsqueeze(-1)
    return (loss * m).sum() / m.sum().clamp(min=1.0)


class SequenceModelAdapter(Forecaster):
    """Trains and applies :class:`SeqForecaster` on athlete sequences."""

    def __init__(
        self,
        kind: str = "lstm",
        quantiles=(0.1, 0.5, 0.9),
        history_len: int = 10,
        horizons: int = 5,
        min_history: int = 3,
        hidden: int = 64,
        num_layers: int = 1,
        nhead: int = 4,
        ff_dim: int = 128,
        dropout: float = 0.1,
        epochs: int = 40,
        batch_size: int = 128,
        lr: float = 2e-3,
        patience: int = 8,
        seed: int = 42,
        bounds=None,
    ):
        self.kind = kind
        self.name = kind
        self.quantiles = tuple(quantiles)
        self.history_len = history_len
        self.horizons = horizons
        self.min_history = min_history
        self.hidden = hidden
        self.num_layers = num_layers
        self.nhead = nhead
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.seed = seed
        self.bounds = tuple(bounds) if bounds else None
        self.model: SeqForecaster | None = None
        self.scaler_X = None
        self.scaler_y = None
        self.n_features_: int = 0
        self.best_val_loss_: float = float("inf")

    def fit(self, train_seqs, val_seqs=None, cfg=None):
        torch.manual_seed(self.seed)
        train_data = build_training_samples(
            train_seqs, self.history_len, self.horizons, self.min_history, fit_scalers=True,
        )
        self.scaler_X = train_data.scaler_X
        self.scaler_y = train_data.scaler_y
        self.n_features_ = train_data.X.shape[-1]

        model = SeqForecaster(
            n_features=self.n_features_,
            hidden=self.hidden,
            horizons=self.horizons,
            quantiles=self.quantiles,
            kind=self.kind,
            num_layers=self.num_layers,
            nhead=self.nhead,
            ff_dim=self.ff_dim,
            dropout=self.dropout,
            history_len=self.history_len,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        X_t = torch.tensor(train_data.X, dtype=torch.float32)
        y_t = torch.tensor(train_data.y, dtype=torch.float32)
        m_t = torch.tensor(train_data.y_mask, dtype=torch.float32)
        l_t = torch.tensor(train_data.lengths, dtype=torch.long)
        loader = DataLoader(
            TensorDataset(X_t, y_t, m_t, l_t),
            batch_size=self.batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
        )

        best_loss = float("inf")
        best_state = None
        patience_counter = 0
        for _epoch in range(self.epochs):
            model.train()
            for xb, yb, mb, lb in loader:
                optimizer.zero_grad()
                loss = quantile_loss(model(xb, lb), yb, mb, self.quantiles)
                loss.backward()
                optimizer.step()

            if val_seqs:
                val_loss = self._val_loss(model, val_seqs)
                if val_loss < best_loss - 1e-5:
                    best_loss = val_loss
                    best_state = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        break
            else:
                best_state = copy.deepcopy(model.state_dict())

        if best_state is not None:
            model.load_state_dict(best_state)
        self.model = model
        self.best_val_loss_ = best_loss if val_seqs else float("nan")
        self.model.eval()
        return self

    def _val_loss(self, model: SeqForecaster, val_seqs) -> float:
        val_data = build_training_samples(
            val_seqs, self.history_len, self.horizons, self.min_history,
            scaler_X=self.scaler_X, scaler_y=self.scaler_y,
        )
        with torch.no_grad():
            pred = model(
                torch.tensor(val_data.X, dtype=torch.float32),
                torch.tensor(val_data.lengths, dtype=torch.long),
            )
            loss = quantile_loss(
                pred,
                torch.tensor(val_data.y, dtype=torch.float32),
                torch.tensor(val_data.y_mask, dtype=torch.float32),
                self.quantiles,
            )
        return float(loss)

    def forecast(self, history: History, horizons: int = 5) -> np.ndarray:
        horizons = min(horizons, self.horizons)
        # keep only the most recent seasons (matches training windows)
        feats = self.scaler_X.transform(history.features[-self.history_len :])
        length = feats.shape[0]
        x = np.zeros((1, self.history_len, feats.shape[1]), dtype=np.float32)
        x[0, self.history_len - length :] = feats
        with torch.no_grad():
            pred = self.model(
                torch.tensor(x, dtype=torch.float32),
                torch.tensor([length], dtype=torch.long),
            )
        quantiles = pred[0].numpy()  # (H, Q) in scaled target space
        quantiles = quantiles * self.scaler_y.scale_ + self.scaler_y.mean_
        if self.bounds:
            quantiles = np.clip(quantiles, self.bounds[0], self.bounds[1])
        # enforce monotonic quantiles (standard post-hoc fix)
        quantiles = np.sort(quantiles, axis=1)
        return quantiles[:horizons]
