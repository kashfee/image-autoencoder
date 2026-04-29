from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from licae.checkpoint import load_model_from_checkpoint
from licae.data import ImageFolderDataset


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.detach().clamp(0.0, 1.0).cpu()
    array = (tensor.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(array)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export an original/reconstruction comparison grid.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", default="outputs/reconstruction_grid.png")
    parser.add_argument("--count", type=int, default=4)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config, _checkpoint = load_model_from_checkpoint(args.checkpoint, device=device)
    dataset = ImageFolderDataset(
        root=args.data_dir,
        image_size=int(config["data"]["image_size"]),
        augment=False,
    )

    count = min(args.count, len(dataset))
    originals = []
    reconstructions = []
    for index in range(count):
        image, _path = dataset[index]
        output = model(image.unsqueeze(0).to(device), quantization_mode="hard", hard_importance=True)
        originals.append(tensor_to_image(image))
        reconstructions.append(tensor_to_image(output.reconstruction[0]))

    tile_w, tile_h = originals[0].size
    label_h = 26
    gap = 8
    grid = Image.new("RGB", (count * tile_w + (count - 1) * gap, 2 * tile_h + label_h + gap), "white")
    draw = ImageDraw.Draw(grid)
    draw.text((4, 4), "Original", fill=(0, 0, 0))
    draw.text((4, tile_h + label_h + gap + 4), "Reconstruction", fill=(0, 0, 0))

    for index, image in enumerate(originals):
        x = index * (tile_w + gap)
        grid.paste(image, (x, label_h))
    for index, image in enumerate(reconstructions):
        x = index * (tile_w + gap)
        grid.paste(image, (x, tile_h + label_h + gap))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path)
    print(output_path.resolve())


if __name__ == "__main__":
    main()
