# Learned Image Compression Autoencoder

This project implements a PyTorch-based lossy image compression and decompression system for RGB images of shape `[3, 256, 256]`. The encoder produces a compact latent tensor of shape `[16, 32, 32]`, applies adaptive importance masking and differentiable quantization during training, and reconstructs images through a learned decoder.

## Core Features

- Encoder with convolutional downsampling, residual blocks, and internal skip connections.
- Latent representation fixed at `[16, 32, 32]`.
- Soft quantization for training and hard quantization for inference.
- Multi-bit uniform quantization with estimated bitrate in bits per pixel.
- Importance Net for spatially adaptive precision allocation.
- Rate-distortion objective using MSE, SSIM, MS-SSIM, and bitrate penalty.
- Evaluation metrics: MSE, PSNR, SSIM, MS-SSIM, BPP, and compression ratio.
- MongoDB storage utilities for compressed latent symbols and masks.

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Dataset Layout

Use any image folder with nested subdirectories:

```text
data/
  train/
    image_001.jpg
    class_or_scene/
      image_002.png
  val/
    image_003.jpg
```

## Training

```bash
python scripts/train.py --config configs/default.yaml --train-dir data/train --val-dir data/val
```

If `--val-dir` is omitted, the training script automatically creates a validation split.

## Evaluation

```bash
python scripts/evaluate.py --checkpoint outputs/best_model.pt --data-dir data/val
```

## MongoDB Compression

```bash
python scripts/compress_to_mongo.py ^
  --checkpoint outputs/best_model.pt ^
  --image path/to/image.jpg ^
  --mongo-uri mongodb://localhost:27017 ^
  --database image_compression ^
  --collection compressed_images ^
  --image-id sample_001
```

```bash
python scripts/reconstruct_from_mongo.py ^
  --checkpoint outputs/best_model.pt ^
  --mongo-uri mongodb://localhost:27017 ^
  --database image_compression ^
  --collection compressed_images ^
  --image-id sample_001 ^
  --output reconstructed.png
```

## Research Direction

This baseline is designed to be extended toward stronger learned codecs. Recommended next improvements include entropy modeling, hyperprior networks, arithmetic coding, perceptual adversarial losses, variable-rate conditioning, and transformer or attention-augmented transforms.
