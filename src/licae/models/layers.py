from __future__ import annotations

from typing import Literal

import torch
from torch import nn

NormType = Literal["batch", "group", "instance", "none"]


def _group_count(channels: int, max_groups: int = 8) -> int:
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def make_norm(channels: int, norm: NormType = "group") -> nn.Module:
    if norm == "batch":
        return nn.BatchNorm2d(channels)
    if norm == "group":
        return nn.GroupNorm(_group_count(channels), channels)
    if norm == "instance":
        return nn.InstanceNorm2d(channels, affine=True)
    if norm == "none":
        return nn.Identity()
    raise ValueError(f"Unsupported normalization type: {norm}")


class ConvNormAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        norm: NormType = "group",
        activation: bool = True,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=norm == "none",
            ),
            make_norm(out_channels, norm),
            nn.SiLU(inplace=True) if activation else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden_channels: int | None = None,
        norm: NormType = "group",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden = hidden_channels or channels
        self.conv1 = ConvNormAct(channels, hidden, kernel_size=3, stride=1, norm=norm)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = ConvNormAct(hidden, channels, kernel_size=3, stride=1, norm=norm, activation=False)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.dropout(out)
        out = self.conv2(out)
        return self.activation(out + residual)


class DownsampleBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm: NormType = "group",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.down = ConvNormAct(in_channels, out_channels, kernel_size=5, stride=2, norm=norm)
        self.residual = ResidualBlock(out_channels, norm=norm, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.residual(self.down(x))


class UpsampleBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm: NormType = "group",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            make_norm(out_channels, norm),
            nn.SiLU(inplace=True),
        )
        self.residual = ResidualBlock(out_channels, norm=norm, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.residual(self.up(x))
