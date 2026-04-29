# Codec Comparison Report

## Experiment Setup

The comparison was performed on the Human Action Recognition validation split exported locally at:

```text
data/har_hf/val
```

Total validation images evaluated: **1,800**

All images were evaluated at `256 x 256` RGB resolution. The learned compression model was evaluated using:

```text
outputs/har_gpu_pretrain/best_model.pt
```

Traditional codecs were evaluated using Pillow encoders:

- JPEG at quality levels `20, 35, 50, 65, 80, 95`
- JPEG2000 at rate settings `64, 48, 32, 24, 16, 8`
- WebP at quality levels `20, 35, 50, 65, 80, 95`

The measured metrics are PSNR, SSIM, MS-SSIM, MSE, BPP, and compression ratio.

## Full Rate-Distortion Table

| Codec | Setting | PSNR | SSIM | MS-SSIM | MSE | BPP | Compression Ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| Learned AE | checkpoint | 22.5605 | 0.7151 | 0.9022 | 0.0059 | 1.1036 | 21.7496x |
| JPEG | Q20 | 29.8934 | 0.8936 | 0.9744 | 0.0011 | 0.5395 | 45.5287x |
| JPEG | Q35 | 32.1843 | 0.9300 | 0.9860 | 0.0006 | 0.7687 | 31.9991x |
| JPEG | Q50 | 33.6518 | 0.9467 | 0.9904 | 0.0004 | 0.9456 | 26.0177x |
| JPEG | Q65 | 35.1754 | 0.9599 | 0.9933 | 0.0003 | 1.1557 | 21.2886x |
| JPEG | Q80 | 37.7650 | 0.9749 | 0.9963 | 0.0002 | 1.5465 | 15.9014x |
| JPEG | Q95 | 44.0540 | 0.9920 | 0.9989 | 0.0000 | 2.9174 | 8.3932x |
| JPEG2000 | rate_64 | 25.5199 | 0.7742 | 0.9296 | 0.0031 | 0.3736 | 64.2454x |
| JPEG2000 | rate_48 | 26.7385 | 0.8111 | 0.9467 | 0.0023 | 0.4981 | 48.1882x |
| JPEG2000 | rate_32 | 28.6139 | 0.8584 | 0.9650 | 0.0016 | 0.7472 | 32.1223x |
| JPEG2000 | rate_24 | 30.1113 | 0.8879 | 0.9745 | 0.0011 | 0.9967 | 24.0787x |
| JPEG2000 | rate_16 | 32.5067 | 0.9239 | 0.9845 | 0.0007 | 1.4953 | 16.0500x |
| JPEG2000 | rate_8 | 37.6312 | 0.9688 | 0.9946 | 0.0002 | 2.9925 | 8.0201x |
| WebP | Q20 | 30.8426 | 0.9076 | 0.9790 | 0.0008 | 0.4161 | 60.7126x |
| WebP | Q35 | 32.5636 | 0.9312 | 0.9853 | 0.0006 | 0.5470 | 46.3404x |
| WebP | Q50 | 33.9539 | 0.9452 | 0.9889 | 0.0004 | 0.6694 | 37.8838x |
| WebP | Q65 | 35.1598 | 0.9558 | 0.9914 | 0.0003 | 0.7963 | 31.8355x |
| WebP | Q80 | 37.4829 | 0.9706 | 0.9945 | 0.0002 | 1.0790 | 23.4394x |
| WebP | Q95 | 44.0609 | 0.9913 | 0.9984 | 0.0000 | 2.2977 | 10.8023x |

## Similar-BPP Comparison

This table compares the learned model against the closest available traditional-codec operating point by BPP.

| Codec | Setting | PSNR | SSIM | MS-SSIM | MSE | BPP | Compression Ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| Learned AE | checkpoint | 22.5605 | 0.7151 | 0.9022 | 0.0059 | 1.1036 | 21.7496x |
| JPEG | Q65 | 35.1754 | 0.9599 | 0.9933 | 0.0003 | 1.1557 | 21.2886x |
| JPEG2000 | rate_24 | 30.1113 | 0.8879 | 0.9745 | 0.0011 | 0.9967 | 24.0787x |
| WebP | Q80 | 37.4829 | 0.9706 | 0.9945 | 0.0002 | 1.0790 | 23.4394x |

## Interpretation

The current learned autoencoder is functional and produces a valid compressed latent representation, but it does not yet outperform traditional codecs on the selected objective metrics. At approximately the same bitrate, WebP and JPEG achieve substantially higher PSNR, SSIM, and MS-SSIM. JPEG2000 also outperforms the learned model while using slightly lower BPP.

The result indicates that the current checkpoint is suitable as a working learned-compression baseline, but it is not strong enough to support a claim that it is better than JPEG, JPEG2000, or WebP.

## Tuning Decision

Further model tuning is necessary if the project objective remains to exceed traditional codecs. The main evidence is:

- Learned AE at `1.1036 BPP`: `22.5605 dB PSNR`
- JPEG Q65 at `1.1557 BPP`: `35.1754 dB PSNR`
- JPEG2000 rate_24 at `0.9967 BPP`: `30.1113 dB PSNR`
- WebP Q80 at `1.0790 BPP`: `37.4829 dB PSNR`

Therefore, the current model is not final-quality for a strong experimental claim. It is a completed baseline that requires architectural and training improvements before being presented as superior to conventional codecs.

## Generated Artifacts

- Full table: `codec_results.md`
- Full CSV: `codec_results.csv`
- Similar-BPP table: `similar_bpp_table.md`
- PSNR rate-distortion graph: `rd_psnr_vs_bpp.png`
- SSIM rate-distortion graph: `rd_ssim_vs_bpp.png`
- MS-SSIM rate-distortion graph: `rd_ms_ssim_vs_bpp.png`
- Visual comparison grid: `visual_comparison_similar_bpp.png`
