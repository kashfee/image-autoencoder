from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


def _gaussian_window(
    channels: int,
    window_size: int,
    sigma: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    gaussian = torch.exp(-(coords**2) / (2 * sigma**2))
    gaussian = gaussian / gaussian.sum()
    window_2d = torch.outer(gaussian, gaussian)
    window_2d = window_2d / window_2d.sum()
    return window_2d.view(1, 1, window_size, window_size).expand(channels, 1, window_size, window_size)


def ssim(
    prediction: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    prediction = prediction.float()
    target = target.float()
    channels = prediction.shape[1]
    window = _gaussian_window(channels, window_size, sigma, prediction.device, prediction.dtype)
    padding = window_size // 2

    mu_x = F.conv2d(prediction, window, padding=padding, groups=channels)
    mu_y = F.conv2d(target, window, padding=padding, groups=channels)
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.conv2d(prediction * prediction, window, padding=padding, groups=channels) - mu_x_sq
    sigma_y_sq = F.conv2d(target * target, window, padding=padding, groups=channels) - mu_y_sq
    sigma_xy = F.conv2d(prediction * target, window, padding=padding, groups=channels) - mu_xy

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    value = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
    )
    return value.mean().clamp(0.0, 1.0)


def ms_ssim(
    prediction: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    weights: tuple[float, ...] = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333),
) -> torch.Tensor:
    values = []
    pred = prediction
    tgt = target
    for index, weight in enumerate(weights):
        score = ssim(pred, tgt, data_range=data_range)
        values.append(score.clamp_min(1e-6) ** weight)
        if index < len(weights) - 1:
            pred = F.avg_pool2d(pred, kernel_size=2, stride=2)
            tgt = F.avg_pool2d(tgt, kernel_size=2, stride=2)
    return torch.prod(torch.stack(values)).clamp(0.0, 1.0)


@dataclass
class LossOutput:
    total: torch.Tensor
    mse: torch.Tensor
    ssim_loss: torch.Tensor
    ms_ssim_loss: torch.Tensor
    rate: torch.Tensor
    ssim_value: torch.Tensor
    ms_ssim_value: torch.Tensor


class RateDistortionLoss(nn.Module):
    def __init__(
        self,
        mse_weight: float = 1.0,
        ssim_weight: float = 0.1,
        ms_ssim_weight: float = 0.1,
        rd_lambda: float = 0.01,
    ) -> None:
        super().__init__()
        self.mse_weight = float(mse_weight)
        self.ssim_weight = float(ssim_weight)
        self.ms_ssim_weight = float(ms_ssim_weight)
        self.rd_lambda = float(rd_lambda)

    def forward(self, reconstruction: torch.Tensor, target: torch.Tensor, bpp: torch.Tensor) -> LossOutput:
        mse_value = F.mse_loss(reconstruction, target)
        ssim_value = ssim(reconstruction, target)
        ms_ssim_value = ms_ssim(reconstruction, target)
        ssim_loss = 1.0 - ssim_value
        ms_ssim_loss = 1.0 - ms_ssim_value
        rate = self.rd_lambda * bpp
        total = (
            self.mse_weight * mse_value
            + self.ssim_weight * ssim_loss
            + self.ms_ssim_weight * ms_ssim_loss
            + rate
        )
        return LossOutput(
            total=total,
            mse=mse_value,
            ssim_loss=ssim_loss,
            ms_ssim_loss=ms_ssim_loss,
            rate=rate,
            ssim_value=ssim_value,
            ms_ssim_value=ms_ssim_value,
        )
