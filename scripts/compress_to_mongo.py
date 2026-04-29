from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from licae.checkpoint import load_model_from_checkpoint
from licae.codec import compress_image_file
from licae.mongodb import MongoCompressedStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compress an image and store the latent payload in MongoDB.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--mongo-uri", required=True)
    parser.add_argument("--database", default="image_compression")
    parser.add_argument("--collection", default="compressed_images")
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--model-version", default="v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config, _checkpoint = load_model_from_checkpoint(args.checkpoint, device=device)
    payload = compress_image_file(
        model=model,
        image_path=args.image,
        image_size=int(config["data"]["image_size"]),
        device=device,
    )
    store = MongoCompressedStore(args.mongo_uri, database=args.database, collection=args.collection)
    try:
        store.put(
            image_id=args.image_id,
            payload=payload,
            model_version=args.model_version,
            metadata={"source_image": str(Path(args.image).resolve())},
        )
    finally:
        store.close()
    print(f"stored image_id={args.image_id} bpp={float(payload['bpp_per_image'].mean()):.4f}")


if __name__ == "__main__":
    main()
