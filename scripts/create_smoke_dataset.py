from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def make_pattern(index: int, size: int = 256) -> np.ndarray:
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    x = x / float(size - 1)
    y = y / float(size - 1)

    if index % 3 == 0:
        red = x
        green = y
        blue = 0.5 + 0.5 * np.sin(8.0 * np.pi * (x + y))
    elif index % 3 == 1:
        checker = ((np.floor(x * 12) + np.floor(y * 12)) % 2).astype(np.float32)
        red = checker
        green = 1.0 - checker * 0.7
        blue = x * (1.0 - y)
    else:
        radius = np.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2)
        rings = 0.5 + 0.5 * np.cos(36.0 * radius)
        red = rings
        green = x
        blue = y

    image = np.stack([red, green, blue], axis=-1)
    return (np.clip(image, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def write_split(root: Path, split: str, count: int, offset: int) -> None:
    split_dir = root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        image = Image.fromarray(make_pattern(offset + index))
        image.save(split_dir / f"synthetic_{index:03d}.png")


def main() -> None:
    root = Path("data") / "smoke"
    write_split(root, "train", count=4, offset=0)
    write_split(root, "val", count=2, offset=100)
    print(f"created smoke dataset at {root.resolve()}")


if __name__ == "__main__":
    main()
