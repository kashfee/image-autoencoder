from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from licae.models import LearnedImageCompressionAE


def save_checkpoint(
    path: str | Path,
    model: LearnedImageCompressionAE,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    epoch: int,
    config: dict[str, Any],
    metrics: dict[str, float] | None = None,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "config": config,
        "metrics": metrics or {},
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()
    torch.save(payload, output_path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(Path(path), map_location=map_location)


def load_model_from_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[LearnedImageCompressionAE, dict[str, Any], dict[str, Any]]:
    checkpoint = load_checkpoint(path, map_location=device)
    config = checkpoint["config"]
    model = LearnedImageCompressionAE.from_config(config["model"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, config, checkpoint
