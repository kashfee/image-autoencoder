from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from licae.checkpoint import load_model_from_checkpoint
from licae.data import ImageFolderDataset
from licae.losses import ms_ssim, ssim
from licae.metrics import compression_ratio_from_bpp, psnr


@dataclass(frozen=True)
class CodecSetting:
    codec: str
    setting_name: str
    params: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare learned compression against JPEG, JPEG2000, and WebP.")
    parser.add_argument("--checkpoint", default="outputs/har_gpu_pretrain/best_model.pt")
    parser.add_argument("--data-dir", default="data/har_hf/val")
    parser.add_argument("--output-dir", default="outputs/codec_comparison")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--visual-count", type=int, default=4)
    return parser.parse_args()


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.detach().clamp(0.0, 1.0).cpu()
    array = (tensor.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def encode_decode(image: Image.Image, setting: CodecSetting) -> tuple[Image.Image, int]:
    buffer = BytesIO()
    image.save(buffer, **setting.params)
    encoded = buffer.getvalue()
    buffer.seek(0)
    decoded = Image.open(buffer).convert("RGB")
    return decoded, len(encoded)


def codec_settings() -> list[CodecSetting]:
    jpeg_qualities = [20, 35, 50, 65, 80, 95]
    webp_qualities = [20, 35, 50, 65, 80, 95]
    jp2_rates = [64, 48, 32, 24, 16, 8]

    settings: list[CodecSetting] = []
    for quality in jpeg_qualities:
        settings.append(
            CodecSetting(
                codec="JPEG",
                setting_name=f"Q{quality}",
                params={"format": "JPEG", "quality": quality, "optimize": True},
            )
        )
    for rate in jp2_rates:
        settings.append(
            CodecSetting(
                codec="JPEG2000",
                setting_name=f"rate_{rate}",
                params={"format": "JPEG2000", "quality_mode": "rates", "quality_layers": [rate]},
            )
        )
    for quality in webp_qualities:
        settings.append(
            CodecSetting(
                codec="WebP",
                setting_name=f"Q{quality}",
                params={"format": "WEBP", "quality": quality, "method": 6},
            )
        )
    return settings


def metric_values(reconstruction: torch.Tensor, target: torch.Tensor, bpp_value: float) -> dict[str, float]:
    reconstruction = reconstruction.float()
    target = target.float()
    mse_value = torch.nn.functional.mse_loss(reconstruction, target).item()
    return {
        "mse": mse_value,
        "psnr": float(psnr(reconstruction, target).detach().cpu()),
        "ssim": float(ssim(reconstruction, target).detach().cpu()),
        "ms_ssim": float(ms_ssim(reconstruction, target).detach().cpu()),
        "bpp": float(bpp_value),
        "compression_ratio": float(compression_ratio_from_bpp(bpp_value).detach().cpu()),
    }


def aggregate(rows: Iterable[dict[str, float]]) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    count = 0
    for row in rows:
        count += 1
        for key, value in row.items():
            totals[key] += float(value)
    return {key: totals[key] / max(1, count) for key in totals}


@torch.no_grad()
def evaluate_model(
    checkpoint_path: str | Path,
    dataset: ImageFolderDataset,
    batch_size: int,
    max_images: int | None,
) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config, checkpoint = load_model_from_checkpoint(checkpoint_path, device=device)
    model.eval()

    limit = len(dataset) if max_images is None else min(max_images, len(dataset))
    metric_rows = []
    for start in tqdm(range(0, limit, batch_size), desc="Learned AE"):
        batch = [dataset[index][0] for index in range(start, min(start + batch_size, limit))]
        target = torch.stack(batch).to(device)
        output = model(target, quantization_mode="hard", hard_importance=True)
        metric_rows.append(metric_values(output.reconstruction, target, float(output.bpp.detach().cpu())))
    metrics = aggregate(metric_rows)
    result = {
        "codec": "Learned AE",
        "setting": "checkpoint",
        **metrics,
    }
    return result, model, config


@torch.no_grad()
def evaluate_traditional(
    dataset: ImageFolderDataset,
    setting: CodecSetting,
    batch_size: int,
    max_images: int | None,
) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    limit = len(dataset) if max_images is None else min(max_images, len(dataset))
    metric_rows = []

    for start in tqdm(range(0, limit, batch_size), desc=f"{setting.codec} {setting.setting_name}"):
        targets = []
        reconstructions = []
        total_bytes = 0
        for index in range(start, min(start + batch_size, limit)):
            target, _path = dataset[index]
            image = tensor_to_pil(target)
            decoded, byte_count = encode_decode(image, setting)
            targets.append(target)
            reconstructions.append(pil_to_tensor(decoded))
            total_bytes += byte_count

        target_batch = torch.stack(targets).to(device)
        recon_batch = torch.stack(reconstructions).to(device)
        image_pixels = float(target_batch.shape[0] * target_batch.shape[-2] * target_batch.shape[-1])
        bpp_value = float(total_bytes * 8) / image_pixels
        metric_rows.append(metric_values(recon_batch, target_batch, bpp_value))

    metrics = aggregate(metric_rows)
    return {
        "codec": setting.codec,
        "setting": setting.setting_name,
        **metrics,
    }


def sort_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    codec_order = {"Learned AE": 0, "JPEG": 1, "JPEG2000": 2, "WebP": 3}
    return sorted(results, key=lambda row: (codec_order.get(row["codec"], 99), row["bpp"]))


def write_csv(path: Path, results: list[dict[str, Any]]) -> None:
    fieldnames = ["codec", "setting", "psnr", "ssim", "ms_ssim", "mse", "bpp", "compression_ratio"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({key: row[key] for key in fieldnames})


def fmt(value: float) -> str:
    return f"{value:.4f}"


def markdown_table(results: list[dict[str, Any]]) -> str:
    lines = [
        "| Codec | Setting | PSNR | SSIM | MS-SSIM | MSE | BPP | Compression Ratio |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        lines.append(
            f"| {row['codec']} | {row['setting']} | {fmt(row['psnr'])} | {fmt(row['ssim'])} | "
            f"{fmt(row['ms_ssim'])} | {fmt(row['mse'])} | {fmt(row['bpp'])} | "
            f"{fmt(row['compression_ratio'])}x |"
        )
    return "\n".join(lines) + "\n"


def closest_by_bpp(results: list[dict[str, Any]], target_bpp: float) -> list[dict[str, Any]]:
    selected = [row for row in results if row["codec"] == "Learned AE"]
    for codec in ["JPEG", "JPEG2000", "WebP"]:
        candidates = [row for row in results if row["codec"] == codec]
        selected.append(min(candidates, key=lambda row: abs(row["bpp"] - target_bpp)))
    return selected


def font(size: int = 18) -> ImageFont.ImageFont:
    for name in ["arial.ttf", "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_plot(results: list[dict[str, Any]], metric: str, ylabel: str, output_path: Path) -> None:
    width, height = 1100, 720
    margin_left, margin_right, margin_top, margin_bottom = 95, 45, 65, 90
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = font(24)
    label_font = font(17)
    small_font = font(14)

    points = [row for row in results if math.isfinite(row["bpp"]) and math.isfinite(row[metric])]
    x_min = 0.0
    x_max = max(row["bpp"] for row in points) * 1.08
    y_min = min(row[metric] for row in points)
    y_max = max(row[metric] for row in points)
    y_pad = max((y_max - y_min) * 0.08, 0.01)
    y_min -= y_pad
    y_max += y_pad

    def x_to_px(x: float) -> float:
        return margin_left + (x - x_min) / max(1e-8, x_max - x_min) * plot_w

    def y_to_px(y: float) -> float:
        return margin_top + (y_max - y) / max(1e-8, y_max - y_min) * plot_h

    # Grid and axes.
    for step in range(6):
        x = x_min + (x_max - x_min) * step / 5
        px = x_to_px(x)
        draw.line((px, margin_top, px, margin_top + plot_h), fill=(230, 230, 230))
        draw.text((px - 18, margin_top + plot_h + 12), f"{x:.2f}", fill=(35, 35, 35), font=small_font)
    for step in range(6):
        y = y_min + (y_max - y_min) * step / 5
        py = y_to_px(y)
        draw.line((margin_left, py, margin_left + plot_w, py), fill=(230, 230, 230))
        draw.text((16, py - 8), f"{y:.2f}", fill=(35, 35, 35), font=small_font)

    draw.rectangle((margin_left, margin_top, margin_left + plot_w, margin_top + plot_h), outline=(40, 40, 40))
    draw.text((margin_left, 22), f"Rate-Distortion Comparison: {ylabel} vs BPP", fill=(0, 0, 0), font=title_font)
    draw.text((margin_left + plot_w // 2 - 60, height - 45), "Bits Per Pixel (BPP)", fill=(0, 0, 0), font=label_font)
    draw.text((20, 24), ylabel, fill=(0, 0, 0), font=label_font)

    colors = {
        "JPEG": (214, 72, 56),
        "JPEG2000": (48, 112, 201),
        "WebP": (40, 150, 91),
        "Learned AE": (135, 57, 180),
    }
    for codec in ["JPEG", "JPEG2000", "WebP"]:
        rows = sorted([row for row in results if row["codec"] == codec], key=lambda row: row["bpp"])
        coords = [(x_to_px(row["bpp"]), y_to_px(row[metric])) for row in rows]
        if len(coords) > 1:
            draw.line(coords, fill=colors[codec], width=3)
        for px, py in coords:
            draw.ellipse((px - 5, py - 5, px + 5, py + 5), fill=colors[codec], outline="white", width=1)

    learned = [row for row in results if row["codec"] == "Learned AE"]
    for row in learned:
        px, py = x_to_px(row["bpp"]), y_to_px(row[metric])
        draw.rectangle((px - 7, py - 7, px + 7, py + 7), fill=colors["Learned AE"], outline="white", width=1)
        draw.text((px + 10, py - 10), "Learned AE", fill=colors["Learned AE"], font=small_font)

    legend_x, legend_y = width - 225, 85
    for offset, codec in enumerate(["JPEG", "JPEG2000", "WebP", "Learned AE"]):
        y = legend_y + offset * 28
        draw.rectangle((legend_x, y, legend_x + 18, y + 18), fill=colors[codec])
        draw.text((legend_x + 28, y - 1), codec, fill=(20, 20, 20), font=small_font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


@torch.no_grad()
def make_visual_grid(
    dataset: ImageFolderDataset,
    model: Any,
    selected_rows: list[dict[str, Any]],
    settings: list[CodecSetting],
    output_path: Path,
    count: int,
) -> None:
    device = next(model.parameters()).device
    selected_settings = {}
    for row in selected_rows:
        if row["codec"] == "Learned AE":
            continue
        selected_settings[row["codec"]] = next(
            setting for setting in settings if setting.codec == row["codec"] and setting.setting_name == row["setting"]
        )

    columns = ["Original", "Learned AE"]
    columns += [f"{codec} {setting.setting_name}" for codec, setting in selected_settings.items()]
    tile = 192
    label_h = 42
    gap = 10
    header_h = 46
    count = min(count, len(dataset))
    grid_w = len(columns) * tile + (len(columns) - 1) * gap
    grid_h = header_h + count * (tile + label_h) + (count - 1) * gap
    grid = Image.new("RGB", (grid_w, grid_h), "white")
    draw = ImageDraw.Draw(grid)
    header_font = font(17)
    label_font = font(13)

    for col, label in enumerate(columns):
        x = col * (tile + gap)
        draw.text((x + 4, 10), label, fill=(0, 0, 0), font=header_font)

    for row_index in range(count):
        original_tensor, _path = dataset[row_index]
        original_pil = tensor_to_pil(original_tensor).resize((tile, tile), Image.Resampling.BICUBIC)
        with torch.no_grad():
            output = model(
                original_tensor.unsqueeze(0).to(device),
                quantization_mode="hard",
                hard_importance=True,
            )
        model_pil = tensor_to_pil(output.reconstruction[0]).resize((tile, tile), Image.Resampling.BICUBIC)
        images = [original_pil, model_pil]

        original_full = tensor_to_pil(original_tensor)
        for setting in selected_settings.values():
            decoded, _byte_count = encode_decode(original_full, setting)
            images.append(decoded.resize((tile, tile), Image.Resampling.BICUBIC))

        y = header_h + row_index * (tile + label_h + gap)
        for col, image in enumerate(images):
            x = col * (tile + gap)
            grid.paste(image, (x, y))
            draw.rectangle((x, y, x + tile - 1, y + tile - 1), outline=(210, 210, 210))
            if col > 0:
                recon_tensor = pil_to_tensor(image.resize((256, 256), Image.Resampling.BICUBIC)).unsqueeze(0).to(device)
                target_tensor = original_tensor.unsqueeze(0).to(device)
                item_psnr = float(psnr(recon_tensor.float(), target_tensor.float()).detach().cpu())
                draw.text((x + 4, y + tile + 6), f"PSNR {item_psnr:.2f} dB", fill=(35, 35, 35), font=label_font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = ImageFolderDataset(root=args.data_dir, image_size=256, augment=False)

    learned_result, model, _config = evaluate_model(
        checkpoint_path=args.checkpoint,
        dataset=dataset,
        batch_size=args.batch_size,
        max_images=args.max_images,
    )

    settings = codec_settings()
    results = [learned_result]
    for setting in settings:
        results.append(
            evaluate_traditional(
                dataset=dataset,
                setting=setting,
                batch_size=args.batch_size,
                max_images=args.max_images,
            )
        )

    results = sort_results(results)
    target_bpp = learned_result["bpp"]
    similar_bpp = closest_by_bpp(results, target_bpp)

    with (output_dir / "codec_results.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    write_csv(output_dir / "codec_results.csv", results)
    (output_dir / "codec_results.md").write_text(markdown_table(results), encoding="utf-8")
    (output_dir / "similar_bpp_table.md").write_text(markdown_table(similar_bpp), encoding="utf-8")
    write_csv(output_dir / "similar_bpp_table.csv", similar_bpp)

    draw_plot(results, "psnr", "PSNR (dB)", output_dir / "rd_psnr_vs_bpp.png")
    draw_plot(results, "ssim", "SSIM", output_dir / "rd_ssim_vs_bpp.png")
    draw_plot(results, "ms_ssim", "MS-SSIM", output_dir / "rd_ms_ssim_vs_bpp.png")

    make_visual_grid(
        dataset=dataset,
        model=model,
        selected_rows=similar_bpp,
        settings=settings,
        output_path=output_dir / "visual_comparison_similar_bpp.png",
        count=args.visual_count,
    )

    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "data_dir": str(Path(args.data_dir).resolve()),
        "image_count": len(dataset) if args.max_images is None else min(args.max_images, len(dataset)),
        "learned_ae": learned_result,
        "similar_bpp_comparison": similar_bpp,
        "outputs": {
            "all_results_csv": str((output_dir / "codec_results.csv").resolve()),
            "all_results_markdown": str((output_dir / "codec_results.md").resolve()),
            "similar_bpp_markdown": str((output_dir / "similar_bpp_table.md").resolve()),
            "rd_psnr": str((output_dir / "rd_psnr_vs_bpp.png").resolve()),
            "rd_ssim": str((output_dir / "rd_ssim_vs_bpp.png").resolve()),
            "rd_ms_ssim": str((output_dir / "rd_ms_ssim_vs_bpp.png").resolve()),
            "visual_grid": str((output_dir / "visual_comparison_similar_bpp.png").resolve()),
        },
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
