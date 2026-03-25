from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    """1D residual block with two convolutions and a skip connection."""

    def __init__(self, channels: int, kernel_size: int = 5, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class DeepMeripCNN(nn.Module):
    """
    Pure CNN baseline for signal denoising with multi-task heads.

    Replaces the Transformer in DeepMeripBaseline with stacked
    residual CNN blocks. Uses progressively larger dilations to
    capture long-range context without a Transformer.

    Input:  [batch, length, feature_dim]  (e.g. feature_dim=4: IP, Input, ratio, log-ratio)
    Output: (regression [batch, length], classification_logits [batch, length])
    """

    def __init__(
        self,
        feature_dim: int = 4,
        d_model: int = 128,
        cnn_channels: int = 128,
        num_res_blocks: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()

        # ---- input projection ----
        self.input_proj = nn.Linear(feature_dim, d_model)

        # ---- front-end convolutions (local feature extraction) ----
        self.front_cnn = nn.Sequential(
            nn.Conv1d(d_model, cnn_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(cnn_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(cnn_channels, d_model, kernel_size=5, padding=2),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )

        # ---- stacked residual blocks (replaces Transformer) ----
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(d_model, kernel_size=5, dropout=dropout) for _ in range(num_res_blocks)]
        )

        # ---- dilated convolutions (capture long-range context) ----
        self.dilated_cnn = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=2, dilation=2),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=4, dilation=4),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=8, dilation=8),
            nn.GELU(),
        )

        # ---- task heads (same as original) ----
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

        # project input features to d_model dimensions
        h = self.input_proj(x)  # [B, L, d_model]

        # switch to [B, C, L] for Conv1d
        h = h.transpose(1, 2)

        # local feature extraction
        h = self.front_cnn(h)

        # deep residual processing
        h = self.res_blocks(h)

        # long-range context via dilated convolutions
        h = self.dilated_cnn(h)

        # back to [B, L, d_model] for linear heads
        h = h.transpose(1, 2)

        # multi-task outputs
        regression = self.regression_head(h).squeeze(-1)  # [B, L]
        classification_logits = self.classification_head(h).squeeze(-1)  # [B, L]

        return regression, classification_logits