from __future__ import annotations

import zlib
from datetime import datetime, timezone
from typing import Any

import numpy as np
import torch
from bson.binary import Binary
from pymongo import MongoClient
from pymongo.collection import Collection


def _tensor_to_compressed_binary(tensor: torch.Tensor) -> tuple[Binary, dict[str, Any]]:
    array = tensor.detach().cpu().contiguous().numpy()
    metadata = {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }
    blob = Binary(zlib.compress(array.tobytes()))
    return blob, metadata


def _compressed_binary_to_tensor(blob: bytes, metadata: dict[str, Any]) -> torch.Tensor:
    dtype = np.dtype(metadata["dtype"])
    array = np.frombuffer(zlib.decompress(blob), dtype=dtype).copy()
    array = array.reshape(metadata["shape"])
    return torch.from_numpy(array)


def serialize_payload(
    image_id: str,
    payload: dict[str, Any],
    model_version: str = "v1",
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    symbols_blob, symbols_meta = _tensor_to_compressed_binary(payload["symbols"])
    mask_blob, mask_meta = _tensor_to_compressed_binary(payload["mask"])
    return {
        "image_id": image_id,
        "model_version": model_version,
        "created_at": datetime.now(timezone.utc),
        "symbols_blob": symbols_blob,
        "symbols_meta": symbols_meta,
        "mask_blob": mask_blob,
        "mask_meta": mask_meta,
        "latent_shape": list(payload["latent_shape"]),
        "input_shape": list(payload["input_shape"]),
        "bit_depth": int(payload["bit_depth"]),
        "clip_value": float(payload["clip_value"]),
        "bpp_per_image": payload["bpp_per_image"].detach().cpu().tolist(),
        "compression_ratio": float(payload["compression_ratio"].detach().cpu()),
        "metadata": extra_metadata or {},
    }


def deserialize_payload(document: dict[str, Any]) -> dict[str, Any]:
    return {
        "symbols": _compressed_binary_to_tensor(document["symbols_blob"], document["symbols_meta"]),
        "mask": _compressed_binary_to_tensor(document["mask_blob"], document["mask_meta"]),
        "latent_shape": tuple(document["latent_shape"]),
        "input_shape": tuple(document["input_shape"]),
        "bit_depth": int(document["bit_depth"]),
        "clip_value": float(document["clip_value"]),
        "bpp_per_image": torch.tensor(document["bpp_per_image"], dtype=torch.float32),
        "compression_ratio": torch.tensor(float(document["compression_ratio"]), dtype=torch.float32),
        "metadata": document.get("metadata", {}),
    }


class MongoCompressedStore:
    def __init__(
        self,
        mongo_uri: str,
        database: str = "image_compression",
        collection: str = "compressed_images",
    ) -> None:
        self.client = MongoClient(mongo_uri)
        self.collection: Collection = self.client[database][collection]
        self.collection.create_index("image_id", unique=True)

    def put(
        self,
        image_id: str,
        payload: dict[str, Any],
        model_version: str = "v1",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        document = serialize_payload(
            image_id=image_id,
            payload=payload,
            model_version=model_version,
            extra_metadata=metadata,
        )
        self.collection.replace_one({"image_id": image_id}, document, upsert=True)

    def get(self, image_id: str) -> dict[str, Any]:
        document = self.collection.find_one({"image_id": image_id})
        if document is None:
            raise KeyError(f"No compressed payload found for image_id={image_id}")
        return deserialize_payload(document)

    def close(self) -> None:
        self.client.close()
