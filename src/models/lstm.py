from __future__ import annotations

import torch
from torch import nn


class LSTMForecaster(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        horizon: int = 14,
    ):
        super().__init__()

        self.encoder = nn.LSTM(
            n_features,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.decoder = nn.GRU(
            n_features,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.norm = nn.LayerNorm(hidden_size)

        self.head = nn.Sequential(
            nn.Linear(hidden_size + 5, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x_past: torch.Tensor, x_future: torch.Tensor) -> torch.Tensor:
        _, (h, c) = self.encoder(x_past)

        dec, _ = self.decoder(x_future, h)
        dec = self.norm(dec)

        last_sales = x_past[:, -1:, 0:1].expand(-1, dec.size(1), -1)

        lag7 = (
            x_past[:, -7:, 0]
            .mean(1, keepdim=True)
            .unsqueeze(-1)
            .expand(-1, dec.size(1), -1)
        )

        lag14 = (
            x_past[:, -14:, 0]
            .mean(1, keepdim=True)
            .unsqueeze(-1)
            .expand(-1, dec.size(1), -1)
        )

        std = (
            x_past[:, :, 0]
            .std(1, keepdim=True)
            .unsqueeze(-1)
            .expand(-1, dec.size(1), -1)
        )

        trend = (
            x_past[:, -1:, 0:1] - x_past[:, :7, 0:1].mean(1, keepdim=True)
        ).expand(-1, dec.size(1), -1)

        x = torch.cat(
            [dec, last_sales, lag7, lag14, std, trend],
            dim=-1,
        )

        # ВАЖНО: здесь нельзя ставить torch.relu(...)
        # Модель предсказывает стандартизированный log1p(sales),
       
        return self.head(x).squeeze(-1)
