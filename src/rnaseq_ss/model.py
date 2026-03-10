from __future__ import annotations

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class DeepMeripBaseline(nn.Module):
    """Hybrid CNN + Transformer + multi-task heads baseline."""

    def __init__(
        self,
        feature_dim: int = 4,
        d_model: int = 128,
        cnn_channels: int = 128,
        num_heads: int = 8,
        num_transformer_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(feature_dim, d_model)
        self.cnn = nn.Sequential(
            nn.Conv1d(d_model, cnn_channels, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(cnn_channels, d_model, kernel_size=5, padding=2),
            nn.GELU(),
        )
        self.pos = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        self.regression_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        self.classification_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x shape: [batch, length, feature_dim]."""
        h = self.input_proj(x)
        h = h.transpose(1, 2)
        h = self.cnn(h)
        h = h.transpose(1, 2)
        h = self.pos(h)
        h = self.transformer(h)
        regression = self.regression_head(h).squeeze(-1)
        classification_logits = self.classification_head(h).squeeze(-1)
        return regression, classification_logits
