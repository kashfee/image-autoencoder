from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

from licae.checkpoint import load_model_from_checkpoint
from licae.data import ImageFolderDataset
from licae.metrics import AverageMeter, batch_metrics


@torch.no_grad()
def evaluate_checkpoint(
    checkpoint_path: str | Path,
    data_dir: str | Path,
    batch_size: int | None = None,
    num_workers: int | None = None,
    output_json: str | Path | None = None,
) -> dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config, _checkpoint = load_model_from_checkpoint(checkpoint_path, device=device)
    image_size = int(config["data"]["image_size"])
    loader = torch.utils.data.DataLoader(
        ImageFolderDataset(root=data_dir, image_size=image_size, augment=False),
        batch_size=batch_size or int(config["data"]["batch_size"]),
        shuffle=False,
        num_workers=num_workers if num_workers is not None else int(config["data"]["num_workers"]),
        pin_memory=torch.cuda.is_available(),
    )

    meter = AverageMeter()
    for images, _paths in tqdm(loader, desc="evaluate", disable=not sys.stderr.isatty()):
        images = images.to(device, non_blocking=True)
        output = model(images, quantization_mode="hard", hard_importance=True)
        meter.update(batch_metrics(output.reconstruction, images, output.bpp), n=images.shape[0])

    metrics = meter.compute()
    if output_json is not None:
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained compression model.")
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--data-dir", required=True, type=str)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--output-json", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        output_json=args.output_json,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
