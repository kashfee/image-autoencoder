from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from licae.models import LearnedImageCompressionAE


def load_image_tensor(path: str | Path, image_size: int = 256, device: str | torch.device = "cpu") -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB")
        image = image.resize((image_size, image_size), Image.Resampling.BICUBIC)
        array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).contiguous()
    return tensor.to(device)


def save_image_tensor(tensor: torch.Tensor, path: str | Path) -> None:
    output = tensor.detach().clamp(0.0, 1.0).cpu()
    if output.ndim == 4:
        output = output[0]
    array = (output.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(output_path)


@torch.no_grad()
def compress_image_file(
    model: LearnedImageCompressionAE,
    image_path: str | Path,
    image_size: int = 256,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    image = load_image_tensor(image_path, image_size=image_size, device=device)
    return model.compress(image)


@torch.no_grad()
def reconstruct_from_payload(
    model: LearnedImageCompressionAE,
    payload: dict[str, Any],
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    symbols = payload["symbols"].to(device)
    mask = payload.get("mask")
    if mask is not None:
        mask = mask.to(device)
    return model.decompress(symbols, mask=mask)
