from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class QuantizerOutput:
    quantized: torch.Tensor
    symbols: torch.Tensor
    step_size: torch.Tensor
    clip_value: torch.Tensor


def soft_round(x: torch.Tensor, alpha: float) -> torch.Tensor:
    floor = torch.floor(x)
    fraction = x - floor
    alpha_tensor = torch.as_tensor(alpha, device=x.device, dtype=x.dtype)
    denominator = torch.tanh(alpha_tensor / 2.0).clamp_min(torch.finfo(x.dtype).eps)
    smooth_fraction = 0.5 * (torch.tanh(alpha_tensor * (fraction - 0.5)) / denominator + 1.0)
    return floor + smooth_fraction


class UniformQuantizer(nn.Module):
    """Uniform scalar quantizer with soft training and hard inference modes."""

    def __init__(
        self,
        bit_depth: int = 8,
        clip_value: float = 2.0,
        soft_round_alpha: float = 8.0,
    ) -> None:
        super().__init__()
        if bit_depth < 2 or bit_depth > 16:
            raise ValueError("bit_depth must be between 2 and 16.")
        self.bit_depth = int(bit_depth)
        self.soft_round_alpha = float(soft_round_alpha)
        self.register_buffer("clip_value", torch.tensor(float(clip_value)))

    @property
    def levels(self) -> int:
        return 2**self.bit_depth

    def step_size(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        clip = self.clip_value.to(device=device, dtype=dtype)
        return (2.0 * clip) / float(self.levels - 1)

    def forward(self, x: torch.Tensor, mode: str | None = None) -> QuantizerOutput:
        if mode is None:
            mode = "soft" if self.training else "hard"
        if mode not in {"soft", "hard", "ste", "noise"}:
            raise ValueError(f"Unsupported quantization mode: {mode}")

        clip = self.clip_value.to(device=x.device, dtype=x.dtype)
        step = self.step_size(dtype=x.dtype, device=x.device)
        x_clamped = x.clamp(min=-clip.item(), max=clip.item())
        scaled = (x_clamped + clip) / step
        hard_symbols = scaled.round().clamp(0, self.levels - 1)
        hard_quantized = hard_symbols * step - clip

        if mode == "hard":
            quantized = hard_quantized
        elif mode == "ste":
            quantized = x_clamped + (hard_quantized - x_clamped).detach()
        elif mode == "noise":
            noise = torch.empty_like(scaled).uniform_(-0.5, 0.5)
            noisy_symbols = (scaled + noise).clamp(0, self.levels - 1)
            quantized = noisy_symbols * step - clip
        else:
            smooth_symbols = soft_round(scaled, self.soft_round_alpha).clamp(0, self.levels - 1)
            quantized = smooth_symbols * step - clip

        return QuantizerOutput(
            quantized=quantized,
            symbols=hard_symbols.to(torch.int32),
            step_size=step.detach(),
            clip_value=clip.detach(),
        )

    def dequantize(self, symbols: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        symbols_float = symbols.to(dtype=dtype)
        step = self.step_size(dtype=dtype, device=symbols.device)
        clip = self.clip_value.to(device=symbols.device, dtype=dtype)
        return symbols_float.clamp(0, self.levels - 1) * step - clip
