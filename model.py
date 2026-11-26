"""Lightweight multi-scale residual CNN for 2D Indian atmospheric state forecasting.

Grid: 32 x 36
Inputs (22 channels):
  - 18 channels: 3 consecutive days of 6 normalized weather anomalies
  - 2 channels: static fields (land_sea_mask, orography_m normalized)
  - 2 channels: day-of-year sin/cos broadcast across grid
Output (6 channels):
  - Next-day normalized weather anomaly (predicted residual relative to day 3)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

IN_CHANNELS = 22  # 3 * 6 + 2 (static) + 2 (doy)
OUT_CHANNELS = 6  # 6 weather variables


class ResidualBlock(nn.Module):
    """Residual block with dilated convolutions for capturing multi-scale synoptic patterns."""

    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            padding_mode="reflect",
        )
        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            padding_mode="reflect",
        )
        self.norm1 = nn.GroupNorm(4, channels)
        self.norm2 = nn.GroupNorm(4, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        out = F.gelu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return F.gelu(out + res)


class MonsoonForecaster(nn.Module):
    """Compact, fast spatial-temporal model optimized for CPU inference (<30ms)."""

    def __init__(self, in_channels: int = IN_CHANNELS, out_channels: int = OUT_CHANNELS, hidden_dim: int = 48):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Initial projection
        self.in_conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1, padding_mode="reflect")

        # Multi-scale feature extraction: standard, 2-dilation (regional), and 4-dilation (subcontinental)
        self.res1 = ResidualBlock(hidden_dim, dilation=1)
        self.res2 = ResidualBlock(hidden_dim, dilation=2)
        self.res3 = ResidualBlock(hidden_dim, dilation=3)
        self.res4 = ResidualBlock(hidden_dim, dilation=1)

        # Output head predicts next-step change (delta) relative to the most recent input state (day 3)
        self.head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.GELU(),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:

          x: [B, 22, 32, 36]
        Returns:
          [B, 6, 32, 36] predicted anomaly for step t+1.
        """
        # Day 3 anomaly is at channels 12..18
        last_day_anom = x[:, 12:18, :, :]

        feat = F.gelu(self.in_conv(x))
        feat = self.res1(feat)
        feat = self.res2(feat)
        feat = self.res3(feat)
        feat = self.res4(feat)

        delta = self.head(feat)
        # Residual step: day_3 + delta
        return last_day_anom + delta
