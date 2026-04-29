from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from datasets import load_dataset
from PIL import Image

CLASS_NAMES = [
    "calling",
    "clapping",
    "cycling",
    "dancing",
    "drinking",
    "eating",
    "fighting",
    "hugging",
    "laughing",
    "listening_to_music",
    "running",
    "sitting",
    "sleeping",
    "texting",
    "using_laptop",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the public Human Action Recognition image dataset and export it as image folders."
    )
    parser.add_argument("--dataset", default="Bingsu/Human_Action_Recognition")
    parser.add_argument("--output-dir", default="data/har_hf")
    parser.add_argument("--cache-dir", default="data/hf_cache")
    parser.add_argument("--val-count", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-val", type=int, default=None)
    parser.add_argument("--image-format", default="jpg", choices=["jpg", "png"])
    return parser.parse_args()


def save_image(image: Image.Image, path: Path, image_format: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = image.convert("RGB")
    if image_format == "jpg":
        image.save(path, quality=95, optimize=True)
    else:
        image.save(path)


def export_split(
    rows: list[dict],
    output_root: Path,
    split: str,
    image_format: str,
    max_count: int | None = None,
) -> dict[str, int]:
    counts = {name: 0 for name in CLASS_NAMES}
    total = 0
    for index, row in enumerate(rows):
        if max_count is not None and total >= max_count:
            break
        label = int(row["labels"])
        class_name = CLASS_NAMES[label]
        extension = "jpg" if image_format == "jpg" else "png"
        filename = f"{split}_{index:06d}.{extension}"
        save_image(row["image"], output_root / split / class_name / filename, image_format)
        counts[class_name] += 1
        total += 1
    return counts


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    os.environ.setdefault("HF_HOME", str(cache_dir.resolve()))

    dataset = load_dataset(args.dataset, split="train", cache_dir=str(cache_dir)).shuffle(seed=args.seed)
    rows = list(dataset)
    val_count = min(max(1, args.val_count), len(rows) // 2)
    val_rows = rows[:val_count]
    train_rows = rows[val_count:]

    train_counts = export_split(
        rows=train_rows,
        output_root=output_root,
        split="train",
        image_format=args.image_format,
        max_count=args.max_train,
    )
    val_counts = export_split(
        rows=val_rows,
        output_root=output_root,
        split="val",
        image_format=args.image_format,
        max_count=args.max_val,
    )

    summary = {
        "source_dataset": args.dataset,
        "output_dir": str(output_root.resolve()),
        "train_images": sum(train_counts.values()),
        "val_images": sum(val_counts.values()),
        "train_counts": train_counts,
        "val_counts": val_counts,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
