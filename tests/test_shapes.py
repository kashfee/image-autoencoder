from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from licae.models import LearnedImageCompressionAE


def test_forward_shapes() -> None:
    model = LearnedImageCompressionAE(latent_channels=16, base_channels=16, dropout=0.0)
    model.eval()
    x = torch.rand(2, 3, 256, 256)
    with torch.no_grad():
        output = model(x, quantization_mode="hard", hard_importance=True)
    assert output.latent.shape == (2, 16, 32, 32)
    assert output.quantized_latent.shape == (2, 16, 32, 32)
    assert output.reconstruction.shape == (2, 3, 256, 256)
    assert output.bpp.item() > 0


def test_compress_decompress_shapes() -> None:
    model = LearnedImageCompressionAE(latent_channels=16, base_channels=16, dropout=0.0)
    model.eval()
    x = torch.rand(1, 3, 256, 256)
    payload = model.compress(x)
    with torch.no_grad():
        reconstruction = model.decompress(payload["symbols"], mask=payload["mask"])
    assert payload["symbols"].shape == (1, 16, 32, 32)
    assert payload["mask"].shape == (1, 16, 32, 32)
    assert reconstruction.shape == (1, 3, 256, 256)


if __name__ == "__main__":
    test_forward_shapes()
    test_compress_decompress_shapes()
    print("shape smoke tests passed")
