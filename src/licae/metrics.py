from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import torch
import torch.nn.functional as F

from licae.losses import ms_ssim, ssim


def mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(prediction, target)


def psnr(prediction: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    mse_value = mse(prediction, target).clamp_min(1e-10)
    return 10.0 * torch.log10(torch.as_tensor(data_range**2, device=prediction.device) / mse_value)


def compression_ratio_from_bpp(bpp: torch.Tensor | float, original_bpp: float = 24.0) -> torch.Tensor:
    if not torch.is_tensor(bpp):
        bpp = torch.tensor(float(bpp))
    return torch.as_tensor(original_bpp, device=bpp.device, dtype=bpp.dtype) / bpp.clamp_min(1e-8)


@torch.no_grad()
def batch_metrics(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    bpp: torch.Tensor,
) -> dict[str, float]:
    return {
        "mse": float(mse(reconstruction, target).detach().cpu()),
        "psnr": float(psnr(reconstruction, target).detach().cpu()),
        "ssim": float(ssim(reconstruction, target).detach().cpu()),
        "ms_ssim": float(ms_ssim(reconstruction, target).detach().cpu()),
        "bpp": float(bpp.detach().cpu()),
        "compression_ratio": float(compression_ratio_from_bpp(bpp).detach().cpu()),
    }


class AverageMeter:
    def __init__(self) -> None:
        self.totals: dict[str, float] = defaultdict(float)
        self.counts: dict[str, int] = defaultdict(int)

    def update(self, values: dict[str, float], n: int = 1) -> None:
        for key, value in values.items():
            self.totals[key] += float(value) * n
            self.counts[key] += n

    def compute(self) -> dict[str, float]:
        return {
            key: self.totals[key] / max(1, self.counts[key])
            for key in sorted(self.totals)
        }


def merge_metric_dicts(metrics: Iterable[dict[str, float]]) -> dict[str, float]:
    meter = AverageMeter()
    for item in metrics:
        meter.update(item)
    return meter.compute()
