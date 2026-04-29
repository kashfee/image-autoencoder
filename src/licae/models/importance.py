from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from licae.models.layers import NormType, ResidualBlock


@dataclass
class ImportanceOutput:
    score: torch.Tensor
    mask: torch.Tensor
    expected_precision: torch.Tensor


def progressive_importance_mask(
    score: torch.Tensor,
    channels: int,
    hard: bool,
    temperature: float,
) -> torch.Tensor:
    thresholds = torch.arange(channels, device=score.device, dtype=score.dtype) / float(channels)
    thresholds = thresholds.view(1, channels, 1, 1)
    if hard:
        return (score >= thresholds).to(score.dtype)
    return torch.sigmoid((score - thresholds) / temperature)


class ImportanceNet(nn.Module):
    """Spatial importance predictor for adaptive latent precision allocation."""

    def __init__(
        self,
        latent_channels: int = 16,
        hidden_channels: int = 32,
        norm: NormType = "group",
        dropout: float = 0.0,
        temperature: float = 0.08,
    ) -> None:
        super().__init__()
        self.temperature = float(temperature)
        self.net = nn.Sequential(
            nn.Conv2d(latent_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, hidden_channels) if norm != "none" else nn.Identity(),
            nn.SiLU(inplace=True),
            ResidualBlock(hidden_channels, norm=norm, dropout=dropout),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, latent: torch.Tensor, hard: bool = False) -> ImportanceOutput:
        score = self.net(latent)
        mask = progressive_importance_mask(
            score=score,
            channels=latent.shape[1],
            hard=hard,
            temperature=self.temperature,
        )
        expected_precision = mask.mean(dim=1, keepdim=True)
        return ImportanceOutput(score=score, mask=mask, expected_precision=expected_precision)
