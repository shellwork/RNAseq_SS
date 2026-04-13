"""
DeepMeripGAN — Adversarial training for signal denoising (single-task regression).

Generator: same backbone as DeepMeripCNN (ResidualCNN + Dilated).
Discriminator: PatchGAN-style 1D, per-position real/fake scoring.

Training alternates Generator and Discriminator steps.
For inference / evaluation, only the Generator is used.

Input:  [batch, length, feature_dim]
Output: [batch, length]  (predicted clean signal)
"""

from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class Generator(nn.Module):
    """Maps noisy signal → clean signal. Returns (regression, feature_map)."""

    def __init__(self, feature_dim=1, d_model=128, cnn_channels=128,
                 num_res_blocks=6, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(feature_dim, d_model)
        self.front_cnn = nn.Sequential(
            nn.Conv1d(d_model, cnn_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(cnn_channels), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(cnn_channels, d_model, kernel_size=5, padding=2),
            nn.BatchNorm1d(d_model), nn.GELU(),
        )
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(d_model, kernel_size=5, dropout=dropout)
              for _ in range(num_res_blocks)]
        )
        self.dilated_cnn = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=2, dilation=2), nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=4, dilation=4), nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=8, dilation=8), nn.GELU(),
        )
        self.regression_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Linear(d_model // 2, 1), nn.ReLU(),
        )

    def forward(self, x):
        """Returns (regression [B,L], feature_map [B,L,C])."""
        h = self.input_proj(x)
        h = h.transpose(1, 2)
        h = self.front_cnn(h)
        h = self.res_blocks(h)
        h = self.dilated_cnn(h)
        h = h.transpose(1, 2)
        reg = self.regression_head(h).squeeze(-1)
        return reg, h


class Discriminator(nn.Module):
    """PatchGAN 1D: per-position real/fake logits."""

    def __init__(self, feature_dim=1, channels=64, num_layers=4, dropout=0.1):
        super().__init__()
        layers = [
            nn.Conv1d(feature_dim, channels, kernel_size=7, padding=3),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        in_ch = channels
        for i in range(1, num_layers):
            out_ch = min(channels * (2 ** i), 512)
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=5, padding=2),
                nn.BatchNorm1d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(dropout),
            ]
            in_ch = out_ch
        layers.append(nn.Conv1d(in_ch, 1, kernel_size=3, padding=1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """x: [B, L, 1] → logits [B, L]"""
        return self.model(x.transpose(1, 2)).squeeze(1)


class DeepMeripGAN(nn.Module):
    """
    Wrapper: Generator + Discriminator.

    forward(x) returns regression only (for eval compatibility).
    Use generate() / discriminate() during training.
    """

    def __init__(self, feature_dim=1, d_model=128, cnn_channels=128,
                 num_res_blocks=6, disc_channels=64, disc_layers=4, dropout=0.1):
        super().__init__()
        self.generator = Generator(feature_dim, d_model, cnn_channels,
                                   num_res_blocks, dropout)
        self.discriminator = Discriminator(1, disc_channels, disc_layers, dropout)

    def generate(self, x):
        """Returns (regression [B,L], feature_map [B,L,C])."""
        return self.generator(x)

    def discriminate(self, x):
        """x: [B, L, 1] → logits [B, L]."""
        return self.discriminator(x)

    def forward(self, x):
        """Inference only: returns regression [B, L]."""
        reg, _ = self.generator(x)
        return reg
