# Methodology: Learned Lossy Image Compression Using Autoencoders

## System Overview

The proposed system implements a learned lossy image compression framework based on convolutional autoencoders. The input to the system is an RGB image of spatial resolution `256 x 256`, represented as a tensor of shape `[3, 256, 256]`. The encoder transforms the image into a compact latent representation of shape `[16, 32, 32]`, reducing the spatial resolution by a factor of eight while preserving semantically meaningful structure. This latent tensor is then passed through an importance-guided quantization stage, where spatial locations with higher visual significance are assigned greater representational capacity.

The decoder reconstructs the image from the compressed latent representation using transposed convolutional upsampling and residual refinement blocks. The complete system is trained end-to-end using a rate-distortion objective that jointly minimizes reconstruction error and estimated bitrate.

## Encoder Architecture

The encoder is designed as an analysis transform. It consists of convolutional downsampling blocks, residual blocks, nonlinear activations, and normalization layers. The spatial resolution is reduced from `256 x 256` to `32 x 32` through three stride-two convolutional stages. Residual connections improve gradient propagation and allow the encoder to preserve low-level and mid-level features that are important for reconstruction quality.

Formally, for an input image `x`, the encoder learns a nonlinear mapping:

```text
z = E(x)
```

where `z` is the latent tensor with shape `[16, 32, 32]`.

## Importance-Guided Compression

Natural images do not have uniform information density. Edges, textures, object boundaries, facial regions, and high-frequency structures typically require higher precision than smooth background areas. The Importance Net estimates a spatial importance score for each latent location. This score is converted into a progressive channel mask, where more latent channels are retained in visually important regions.

The importance mechanism enables adaptive compression because the number of active latent symbols varies spatially. The estimated bitrate is computed from the active latent mask and the quantization bit depth:

```text
BPP = active_latent_symbols * quantization_bits / image_pixels
```

This makes the system suitable for rate-distortion optimization.

## Quantization

During training, direct hard rounding is not ideal because it has zero gradient almost everywhere. Therefore, the system uses soft quantization during training and hard quantization during inference. The soft quantizer approximates the rounding operation using a differentiable smooth rounding function. During deployment, hard quantization converts latent values into integer symbols that can be stored or transmitted efficiently.

The quantizer supports multi-bit uniform quantization. The default configuration uses 8-bit latent symbols, but this value can be reduced or increased depending on the desired bitrate and quality trade-off.

## Decoder Architecture

The decoder is a synthesis transform. It maps the quantized latent representation back to the RGB image domain:

```text
x_hat = D(z_q)
```

where `z_q` denotes the quantized latent tensor. The decoder uses transposed convolutional upsampling blocks to restore spatial resolution from `32 x 32` to `256 x 256`. Residual blocks refine the reconstructed features and reduce artifacts introduced by compression and quantization.

## Training Objective

The training objective combines distortion losses and a bitrate penalty. The total loss is defined as:

```text
L = w_mse * MSE(x, x_hat)
  + w_ssim * (1 - SSIM(x, x_hat))
  + w_ms_ssim * (1 - MS-SSIM(x, x_hat))
  + lambda * BPP
```

The MSE term encourages pixel-level fidelity, while SSIM and MS-SSIM improve structural and perceptual quality. The rate term penalizes excessive latent usage, encouraging the model to learn compact representations. The coefficient `lambda` controls the rate-distortion trade-off.

## Evaluation Metrics

The system is evaluated using the following metrics:

- MSE: measures pixel-wise reconstruction error.
- PSNR: measures signal fidelity in decibels.
- SSIM: measures structural similarity between the original and reconstructed image.
- MS-SSIM: evaluates structural similarity across multiple spatial scales.
- BPP: measures compressed bitrate in bits per pixel.
- Compression ratio: compares the original 24-bit RGB representation against the learned latent bitrate.

## Database Integration

For deployment, the compressed latent symbols and importance masks are serialized and stored in MongoDB. The database stores integer quantized symbols, binary masks, tensor shape metadata, quantizer metadata, estimated bitrate, and model version. Client-side decompression is performed by loading the stored payload, dequantizing the symbols, applying the mask, and passing the result through the decoder.

This design separates compressed representation storage from reconstruction, making it suitable for cloud storage, edge transmission, and client-side image restoration.

## Research-Grade Extensions

The present implementation provides a strong baseline for learned image compression. To approach state-of-the-art performance, the following extensions are recommended:

- Entropy modeling with learned probability distributions for latent symbols.
- Hyperprior networks for side-information-based entropy estimation.
- Arithmetic coding or range coding for true bitstream generation.
- Variable-rate conditioning for dynamic bitrate control.
- Perceptual losses using pretrained feature networks.
- Attention modules for improved long-range dependency modeling.
- Adversarial fine-tuning for visually realistic reconstructions at low bitrate.
- Learned color transforms and frequency-domain hybrid representations.
