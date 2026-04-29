from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 42,
    "model": {
        "latent_channels": 16,
        "base_channels": 64,
        "norm": "group",
        "dropout": 0.05,
        "bit_depth": 8,
        "quant_clip": 2.0,
        "soft_round_alpha": 8.0,
        "importance_temperature": 0.08,
    },
    "data": {
        "image_size": 256,
        "train_dir": "data/train",
        "val_dir": None,
        "val_split": 0.1,
        "batch_size": 8,
        "num_workers": 4,
        "augment": True,
    },
    "training": {
        "epochs": 80,
        "learning_rate": 2e-4,
        "weight_decay": 1e-6,
        "grad_clip_norm": 1.0,
        "amp": True,
        "output_dir": "outputs",
        "save_every": 5,
    },
    "loss": {
        "mse_weight": 1.0,
        "ssim_weight": 0.1,
        "ms_ssim_weight": 0.1,
        "rd_lambda": 0.01,
    },
}


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        return deepcopy(DEFAULT_CONFIG)

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        user_config = yaml.safe_load(handle) or {}
    return deep_update(DEFAULT_CONFIG, user_config)


def save_config(config: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
