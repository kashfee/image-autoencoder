from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from licae.models.importance import ImportanceNet
from licae.models.layers import ConvNormAct, DownsampleBlock, NormType, ResidualBlock, UpsampleBlock
from licae.models.quantizer import QuantizerOutput, UniformQuantizer


@dataclass
class CompressionForwardOutput:
    reconstruction: torch.Tensor
    latent: torch.Tensor
    quantized_latent: torch.Tensor
    importance_score: torch.Tensor
    importance_mask: torch.Tensor
    quantizer: QuantizerOutput
    bpp: torch.Tensor
    bpp_per_image: torch.Tensor
    compression_ratio: torch.Tensor


class EncoderNet(nn.Module):
    def __init__(
        self,
        latent_channels: int = 16,
        base_channels: int = 64,
        norm: NormType = "group",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 3
        self.net = nn.Sequential(
            ConvNormAct(3, c1, kernel_size=5, stride=2, norm=norm),
            ResidualBlock(c1, norm=norm, dropout=dropout),
            DownsampleBlock(c1, c2, norm=norm, dropout=dropout),
            DownsampleBlock(c2, c3, norm=norm, dropout=dropout),
            ResidualBlock(c3, norm=norm, dropout=dropout),
            nn.Conv2d(c3, latent_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DecoderNet(nn.Module):
    def __init__(
        self,
        latent_channels: int = 16,
        base_channels: int = 64,
        norm: NormType = "group",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 3
        self.net = nn.Sequential(
            ConvNormAct(latent_channels, c3, kernel_size=3, stride=1, norm=norm),
            ResidualBlock(c3, norm=norm, dropout=dropout),
            UpsampleBlock(c3, c2, norm=norm, dropout=dropout),
            UpsampleBlock(c2, c1, norm=norm, dropout=dropout),
            UpsampleBlock(c1, c1, norm=norm, dropout=dropout),
            ResidualBlock(c1, norm=norm, dropout=dropout),
            nn.Conv2d(c1, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def estimate_bpp(mask: torch.Tensor, bit_depth: int, input_hw: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    image_pixels = float(input_hw[0] * input_hw[1])
    active_symbols = mask.flatten(1).sum(dim=1)
    bits_per_image = active_symbols * float(bit_depth)
    bpp_per_image = bits_per_image / image_pixels
    return bpp_per_image.mean(), bpp_per_image


class LearnedImageCompressionAE(nn.Module):
    """End-to-end learned lossy image compression autoencoder."""

    def __init__(
        self,
        latent_channels: int = 16,
        base_channels: int = 64,
        norm: NormType = "group",
        dropout: float = 0.05,
        bit_depth: int = 8,
        quant_clip: float = 2.0,
        soft_round_alpha: float = 8.0,
        importance_temperature: float = 0.08,
    ) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        self.encoder = EncoderNet(
            latent_channels=latent_channels,
            base_channels=base_channels,
            norm=norm,
            dropout=dropout,
        )
        self.importance_net = ImportanceNet(
            latent_channels=latent_channels,
            hidden_channels=max(32, base_channels // 2),
            norm=norm,
            dropout=dropout,
            temperature=importance_temperature,
        )
        self.quantizer = UniformQuantizer(
            bit_depth=bit_depth,
            clip_value=quant_clip,
            soft_round_alpha=soft_round_alpha,
        )
        self.decoder = DecoderNet(
            latent_channels=latent_channels,
            base_channels=base_channels,
            norm=norm,
            dropout=dropout,
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "LearnedImageCompressionAE":
        return cls(**config)

    def forward(
        self,
        x: torch.Tensor,
        quantization_mode: str | None = None,
        hard_importance: bool | None = None,
    ) -> CompressionForwardOutput:
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError("Expected input image batch with shape [B, 3, H, W].")

        latent = self.encoder(x)
        use_hard_importance = (not self.training) if hard_importance is None else hard_importance
        importance = self.importance_net(latent, hard=use_hard_importance)
        masked_latent = latent * importance.mask
        quantized = self.quantizer(masked_latent, mode=quantization_mode)
        quantized_latent = quantized.quantized * importance.mask
        reconstruction = self.decoder(quantized_latent)

        bpp, bpp_per_image = estimate_bpp(
            mask=importance.mask,
            bit_depth=self.quantizer.bit_depth,
            input_hw=(x.shape[-2], x.shape[-1]),
        )
        compression_ratio = torch.as_tensor(24.0, device=x.device, dtype=x.dtype) / bpp.clamp_min(1e-8)

        return CompressionForwardOutput(
            reconstruction=reconstruction,
            latent=latent,
            quantized_latent=quantized_latent,
            importance_score=importance.score,
            importance_mask=importance.mask,
            quantizer=quantized,
            bpp=bpp,
            bpp_per_image=bpp_per_image,
            compression_ratio=compression_ratio,
        )

    @torch.no_grad()
    def compress(self, x: torch.Tensor) -> dict[str, Any]:
        was_training = self.training
        self.eval()
        output = self.forward(x, quantization_mode="hard", hard_importance=True)
        payload = {
            "symbols": output.quantizer.symbols.cpu(),
            "mask": output.importance_mask.gt(0.5).to(torch.uint8).cpu(),
            "latent_shape": tuple(output.quantizer.symbols.shape),
            "input_shape": tuple(x.shape),
            "bit_depth": self.quantizer.bit_depth,
            "clip_value": float(self.quantizer.clip_value.detach().cpu().item()),
            "bpp_per_image": output.bpp_per_image.detach().cpu(),
            "compression_ratio": output.compression_ratio.detach().cpu(),
        }
        if was_training:
            self.train()
        return payload

    @torch.no_grad()
    def decompress(self, symbols: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device
        symbols = symbols.to(device=device)
        quantized_latent = self.quantizer.dequantize(symbols, dtype=dtype)
        if mask is not None:
            quantized_latent = quantized_latent * mask.to(device=device, dtype=dtype)
        return self.decoder(quantized_latent)
