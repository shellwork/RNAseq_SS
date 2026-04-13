"""
DeepMeripCNN — Pure CNN with residual blocks + dilated convolutions (single-task regression).

Our solution model: replaces the Transformer with stacked residual CNN blocks
and progressively dilated convolutions to capture long-range context.

Input:  [batch, length, feature_dim]   (feature_dim=1 for single-channel coverage)
Output: [batch, length]                (predicted clean signal)
"""

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
    Pure CNN baseline for signal denoising (single-task regression).

    Architecture:
      1. Linear projection: feature_dim → d_model
      2. Front-end CNN (local feature extraction)
      3. Stacked residual blocks (deep feature processing)
      4. Dilated convolutions (long-range context capture)
      5. Regression head → ReLU (non-negative output)
    """

    def __init__(
        self,
        feature_dim: int = 1,
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

        # ---- stacked residual blocks ----
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

        # ---- regression head ----
        self.regression_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, length, feature_dim]
        returns: [batch, length]
        """
        h = self.input_proj(x)       # [B, L, d_model]
        h = h.transpose(1, 2)        # [B, d_model, L]

        h = self.front_cnn(h)
        h = self.res_blocks(h)
        h = self.dilated_cnn(h)

        h = h.transpose(1, 2)        # [B, L, d_model]

        regression = self.regression_head(h).squeeze(-1)  # [B, L]
        return regression
