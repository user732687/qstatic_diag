from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Toy CNN
# ---------------------------------------------------------------------------


class ConvBlock(nn.Module):
    """Standard conv → BN → ReLU block."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """BasicBlock-style residual block."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class TinyCNN(nn.Module):
    """
    Tiny CNN with residual connections.
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 10) -> None:
        super().__init__()
        self.stem = ConvBlock(in_channels, 32)
        self.layer1 = nn.Sequential(ConvBlock(32, 64, stride=2), ResidualBlock(64))
        self.layer2 = nn.Sequential(ConvBlock(64, 128, stride=2), ResidualBlock(128))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.pool(x).flatten(1)
        return self.head(x)
