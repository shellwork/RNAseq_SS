from __future__ import annotations

import torch
from torch import nn


# ============================================================
# Building blocks
# ============================================================

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


# ============================================================
# Generator
# ============================================================

class Generator(nn.Module):
    """
    CNN-based generator that maps noisy (low-coverage) signal to
    denoised (high-coverage) signal, with multi-task heads for
    regression and peak classification.

    Input:  [batch, length, feature_dim]
    Output: (regression [batch, length],
             classification_logits [batch, length],
             feature_map [batch, length, d_model])
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

        # ---- front-end convolutions ----
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

        # ---- dilated convolutions for long-range context ----
        self.dilated_cnn = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=2, dilation=2),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=4, dilation=4),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=8, dilation=8),
            nn.GELU(),
        )

        # ---- task heads ----
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

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            regression:              [B, L]        predicted continuous coverage
            classification_logits:   [B, L]        peak / non-peak logits
            feature_map:             [B, L, C]     intermediate features (fed to discriminator)
        """
        h = self.input_proj(x)          # [B, L, C]
        h = h.transpose(1, 2)           # [B, C, L]

        h = self.front_cnn(h)
        h = self.res_blocks(h)
        h = self.dilated_cnn(h)

        h = h.transpose(1, 2)           # [B, L, C]

        regression = self.regression_head(h).squeeze(-1)
        classification_logits = self.classification_head(h).squeeze(-1)

        return regression, classification_logits, h


# ============================================================
# Discriminator
# ============================================================

class Discriminator(nn.Module):
    """
    PatchGAN-style 1D discriminator.

    Instead of outputting a single real/fake scalar for the whole
    sequence, it outputs a per-position realness score — this
    encourages the generator to produce locally realistic signal
    patterns at every genomic position.

    Input:  [batch, length, feature_dim]   (the coverage profile to judge)
    Output: [batch, length]                (per-position real/fake logits)
    """

    def __init__(
        self,
        feature_dim: int = 1,
        channels: int = 64,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        layers: list[nn.Module] = [
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

        # final projection to per-position score
        layers.append(nn.Conv1d(in_ch, 1, kernel_size=3, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, feature_dim] — e.g. a 1-channel coverage track
        Returns:
            logits: [B, L] — per-position real/fake score
        """
        h = x.transpose(1, 2)           # [B, C, L]
        h = self.model(h)               # [B, 1, L]
        return h.squeeze(1)             # [B, L]


# ============================================================
# GAN wrapper (convenient training interface)
# ============================================================

class DeepMeripGAN(nn.Module):
    """
    Wraps Generator + Discriminator together.

    Usage during training:
        model = DeepMeripGAN()
        # --- generator step ---
        reg, cls_logits, feat = model.generate(noisy_input)
        d_fake = model.discriminate(reg.unsqueeze(-1))
        g_loss = generator_loss(d_fake, reg, cls_logits, targets)

        # --- discriminator step ---
        d_real = model.discriminate(real_signal.unsqueeze(-1))
        d_fake = model.discriminate(reg.unsqueeze(-1).detach())
        d_loss = discriminator_loss(d_real, d_fake)
    """

    def __init__(
        self,
        feature_dim: int = 4,
        d_model: int = 128,
        cnn_channels: int = 128,
        num_res_blocks: int = 6,
        disc_channels: int = 64,
        disc_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.generator = Generator(
            feature_dim=feature_dim,
            d_model=d_model,
            cnn_channels=cnn_channels,
            num_res_blocks=num_res_blocks,
            dropout=dropout,
        )

        self.discriminator = Discriminator(
            feature_dim=1,          # judges the 1-channel regression output
            channels=disc_channels,
            num_layers=disc_layers,
            dropout=dropout,
        )

    def generate(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run generator: noisy input -> (regression, classification_logits, features)."""
        return self.generator(x)

    def discriminate(self, x: torch.Tensor) -> torch.Tensor:
        """Run discriminator: coverage track [B, L, 1] -> realness logits [B, L]."""
        return self.discriminator(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convenience forward that matches the original DeepMeripBaseline
        interface: returns (regression, classification_logits).
        Use this for inference / evaluation.
        """
        regression, classification_logits, _ = self.generator(x)
        return regression, classification_logits
