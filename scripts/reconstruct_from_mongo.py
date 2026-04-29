from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from licae.checkpoint import load_model_from_checkpoint
from licae.codec import save_image_tensor
from licae.mongodb import MongoCompressedStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load compressed latent payload from MongoDB and reconstruct image.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mongo-uri", required=True)
    parser.add_argument("--database", default="image_compression")
    parser.add_argument("--collection", default="compressed_images")
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _config, _checkpoint = load_model_from_checkpoint(args.checkpoint, device=device)
    store = MongoCompressedStore(args.mongo_uri, database=args.database, collection=args.collection)
    try:
        payload = store.get(args.image_id)
    finally:
        store.close()
    reconstruction = model.decompress(payload["symbols"].to(device), mask=payload["mask"].to(device))
    save_image_tensor(reconstruction, args.output)
    print(f"reconstructed image_id={args.image_id} to {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
