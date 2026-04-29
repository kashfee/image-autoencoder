from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def discover_images(root: str | Path) -> list[Path]:
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Image directory does not exist: {root_path}")
    paths = [
        path
        for path in root_path.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]
    if not paths:
        raise FileNotFoundError(f"No supported images found in: {root_path}")
    return sorted(paths)


class ImageFolderDataset(Dataset):
    def __init__(
        self,
        root: str | Path | None = None,
        image_size: int = 256,
        augment: bool = False,
        paths: Sequence[str | Path] | None = None,
    ) -> None:
        if paths is None and root is None:
            raise ValueError("Either root or paths must be provided.")
        self.paths = [Path(path) for path in (paths if paths is not None else discover_images(root))]
        self.image_size = int(image_size)
        self.augment = bool(augment)

    def __len__(self) -> int:
        return len(self.paths)

    def _random_resized_crop(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        min_side = min(width, height)
        if min_side < 2:
            return image
        scale = random.uniform(0.72, 1.0)
        crop_size = max(1, int(min_side * scale))
        left = random.randint(0, max(0, width - crop_size))
        top = random.randint(0, max(0, height - crop_size))
        return image.crop((left, top, left + crop_size, top + crop_size))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        path = self.paths[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            if self.augment:
                image = self._random_resized_crop(image)
                if random.random() < 0.5:
                    image = ImageOps.mirror(image)
            image = image.resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)
            array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        return tensor, str(path)


def build_dataloaders(
    train_dir: str | Path,
    val_dir: str | Path | None = None,
    image_size: int = 256,
    batch_size: int = 8,
    num_workers: int = 4,
    val_split: float = 0.1,
    augment: bool = True,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    train_paths = discover_images(train_dir)

    if val_dir:
        val_paths = discover_images(val_dir)
    else:
        generator = torch.Generator().manual_seed(seed)
        order = torch.randperm(len(train_paths), generator=generator).tolist()
        val_count = max(1, int(len(train_paths) * val_split)) if len(train_paths) > 1 else 0
        val_indices = set(order[:val_count])
        val_paths = [path for idx, path in enumerate(train_paths) if idx in val_indices]
        train_paths = [path for idx, path in enumerate(train_paths) if idx not in val_indices]
        if not train_paths:
            train_paths = val_paths

    train_dataset = ImageFolderDataset(paths=train_paths, image_size=image_size, augment=augment)
    val_dataset = ImageFolderDataset(paths=val_paths or train_paths, image_size=image_size, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=len(train_dataset) >= batch_size,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader
