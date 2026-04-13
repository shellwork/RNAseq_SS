"""
DeepMeripBaseline — Transformer Encoder + CNN Decoder for single-task regression.

Replicates the network architecture from DeepMerip (DeepTrans):
  Input projection → Positional Encoding → Transformer Encoder
  → CNN Decoder (DecoderBlocks with residual connections) → regression output

Adapted for single-task regression on our downsampled→clean signal dataset.

Input:  [batch, length, feature_dim]   (feature_dim=1 for single-channel coverage)
Output: [batch, length]                (predicted clean signal)
"""

from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class DecoderBlock(nn.Module):
    """Conv1d block with residual connection (mirrors DeepMerip DecoderBlock)."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 5, padding: int = 2, dropout: float = 0.0):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.downsample = (
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels),
            )
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.downsample(x)
        out = self.dropout(self.relu(self.bn(self.conv(x))))
        return out + residual


class DeepMeripBaseline(nn.Module):
    """
    Transformer Encoder + CNN Decoder (single-task regression).

    Architecture mirrors DeepMerip/DeepTrans:
      1. Linear projection: feature_dim → d_model
      2. Positional encoding
      3. Transformer encoder (multi-head self-attention)
      4. CNN decoder with residual DecoderBlocks
      5. ReLU activation on output (coverage is non-negative)
    """

    def __init__(
        self,
        feature_dim: int = 1,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # Input projection
        self.input_proj = nn.Linear(feature_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=5000)

        # Transformer Encoder
        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_encoder_layers)

        # CNN Decoder with residual blocks (same structure as DeepTrans)
        self.decoder = nn.Sequential(
            DecoderBlock(d_model, 256, kernel_size=5, padding=2, dropout=0.25),
            DecoderBlock(256, 128, kernel_size=5, padding=2, dropout=0.2),
            DecoderBlock(128, 64, kernel_size=5, padding=2),
            nn.Conv1d(64, 1, kernel_size=5, padding=2),
        )

        self.final_activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, length, feature_dim]
        returns: [batch, length]
        """
        # Transformer encoder path
        h = self.input_proj(x)          # [B, L, d_model]
        h = self.pos_encoder(h)
        h = self.transformer_encoder(h) # [B, L, d_model]

        # CNN decoder path
        h = h.permute(0, 2, 1)          # [B, d_model, L]
        h = self.decoder(h)             # [B, 1, L]
        h = self.final_activation(h)    # ReLU for non-negative output

        return h.squeeze(1)             # [B, L]
