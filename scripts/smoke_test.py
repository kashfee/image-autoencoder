from __future__ import annotations

import json

import torch

from licae.metrics import batch_metrics
from licae.models import LearnedImageCompressionAE


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LearnedImageCompressionAE(
        latent_channels=16,
        base_channels=16,
        dropout=0.0,
    ).to(device)
    model.eval()

    images = torch.rand(1, 3, 256, 256, device=device)
    with torch.no_grad():
        output = model(images, quantization_mode="hard", hard_importance=True)
        payload = model.compress(images)
        reconstruction = model.decompress(
            payload["symbols"].to(device),
            mask=payload["mask"].to(device),
        )

    metrics = batch_metrics(output.reconstruction, images, output.bpp)
    report = {
        "device": str(device),
        "torch_version": torch.__version__,
        "input_shape": list(images.shape),
        "latent_shape": list(output.latent.shape),
        "reconstruction_shape": list(output.reconstruction.shape),
        "payload_symbols_shape": list(payload["symbols"].shape),
        "payload_mask_shape": list(payload["mask"].shape),
        "decompressed_shape": list(reconstruction.shape),
        "metrics": metrics,
    }
    print(json.dumps(report, indent=2))

    assert output.latent.shape == (1, 16, 32, 32)
    assert output.reconstruction.shape == (1, 3, 256, 256)
    assert payload["symbols"].shape == (1, 16, 32, 32)
    assert payload["mask"].shape == (1, 16, 32, 32)
    assert reconstruction.shape == (1, 3, 256, 256)


if __name__ == "__main__":
    main()
