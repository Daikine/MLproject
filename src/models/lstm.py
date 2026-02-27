# src/models/lstm.py
from __future__ import annotations
import torch
from torch import nn

class LSTMForecaster(nn.Module):
    """
    Принимает последовательность (B, T, F) и предсказывает (B, horizon)
    """
    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        horizon: int = 14,
    ):
        super().__init__()
        self.horizon = horizon

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)        # (B,T,H)
        last = out[:, -1, :]         # (B,H)
        return self.head(last)       # (B,horizon)
