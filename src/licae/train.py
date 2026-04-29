from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from licae.checkpoint import load_checkpoint, save_checkpoint
from licae.config import load_config, save_config
from licae.data import build_dataloaders
from licae.losses import RateDistortionLoss
from licae.metrics import AverageMeter, batch_metrics
from licae.models import LearnedImageCompressionAE


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _loss_to_metrics(losses: Any) -> dict[str, float]:
    return {
        "loss": float(losses.total.detach().cpu()),
        "loss_mse": float(losses.mse.detach().cpu()),
        "loss_ssim": float(losses.ssim_loss.detach().cpu()),
        "loss_ms_ssim": float(losses.ms_ssim_loss.detach().cpu()),
        "loss_rate": float(losses.rate.detach().cpu()),
    }


def train_one_epoch(
    model: LearnedImageCompressionAE,
    loader: torch.utils.data.DataLoader,
    criterion: RateDistortionLoss,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp_enabled: bool,
    grad_clip_norm: float | None,
    epoch: int,
) -> dict[str, float]:
    model.train()
    meter = AverageMeter()
    progress = tqdm(loader, desc=f"train epoch {epoch}", leave=False, disable=not sys.stderr.isatty())
    for images, _paths in progress:
        images = images.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            output = model(images, quantization_mode="soft", hard_importance=False)
            losses = criterion(output.reconstruction, images, output.bpp)

        scaler.scale(losses.total).backward()
        if grad_clip_norm is not None and grad_clip_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        values = _loss_to_metrics(losses)
        values.update(batch_metrics(output.reconstruction.detach(), images, output.bpp.detach()))
        meter.update(values, n=images.shape[0])
        progress.set_postfix(loss=values["loss"], psnr=values["psnr"], bpp=values["bpp"])
    return meter.compute()


@torch.no_grad()
def evaluate(
    model: LearnedImageCompressionAE,
    loader: torch.utils.data.DataLoader,
    criterion: RateDistortionLoss,
    device: torch.device,
    amp_enabled: bool,
) -> dict[str, float]:
    model.eval()
    meter = AverageMeter()
    progress = tqdm(loader, desc="validate", leave=False, disable=not sys.stderr.isatty())
    for images, _paths in progress:
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            output = model(images, quantization_mode="hard", hard_importance=True)
            losses = criterion(output.reconstruction, images, output.bpp)
        values = _loss_to_metrics(losses)
        values.update(batch_metrics(output.reconstruction, images, output.bpp))
        meter.update(values, n=images.shape[0])
        progress.set_postfix(psnr=values["psnr"], bpp=values["bpp"])
    return meter.compute()


def run_training(config: dict[str, Any]) -> dict[str, float]:
    set_seed(int(config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(config["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / "resolved_config.yaml")

    train_loader, val_loader = build_dataloaders(
        train_dir=config["data"]["train_dir"],
        val_dir=config["data"].get("val_dir"),
        image_size=int(config["data"]["image_size"]),
        batch_size=int(config["data"]["batch_size"]),
        num_workers=int(config["data"]["num_workers"]),
        val_split=float(config["data"]["val_split"]),
        augment=bool(config["data"]["augment"]),
        seed=int(config["seed"]),
    )

    model = LearnedImageCompressionAE.from_config(config["model"]).to(device)
    resume_from = config["training"].get("resume_from")
    if resume_from:
        checkpoint = load_checkpoint(resume_from, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        print(f"resumed model weights from {resume_from}")

    criterion = RateDistortionLoss(**config["loss"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(config["training"]["epochs"])),
    )
    if resume_from and bool(config["training"].get("resume_optimizer", False)):
        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        if "scheduler_state" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
    amp_enabled = bool(config["training"]["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    grad_clip_norm = float(config["training"]["grad_clip_norm"])

    best_psnr = -float("inf")
    best_metrics: dict[str, float] = {}
    history: list[dict[str, Any]] = []

    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            amp_enabled=amp_enabled,
            grad_clip_norm=grad_clip_norm,
            epoch=epoch,
        )
        val_metrics = evaluate(model, val_loader, criterion, device, amp_enabled)
        scheduler.step()

        epoch_record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(epoch_record)
        with (output_dir / "history.json").open("w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)

        save_checkpoint(
            output_dir / "last_model.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            config=config,
            metrics=val_metrics,
        )
        if epoch % int(config["training"]["save_every"]) == 0:
            save_checkpoint(
                output_dir / f"epoch_{epoch:04d}.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                config=config,
                metrics=val_metrics,
            )
        if val_metrics["psnr"] > best_psnr:
            best_psnr = val_metrics["psnr"]
            best_metrics = val_metrics
            save_checkpoint(
                output_dir / "best_model.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                config=config,
                metrics=val_metrics,
            )

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics['loss']:.5f} "
            f"val_psnr={val_metrics['psnr']:.3f} "
            f"val_ssim={val_metrics['ssim']:.4f} "
            f"val_bpp={val_metrics['bpp']:.4f} "
            f"ratio={val_metrics['compression_ratio']:.2f}x"
        )

    return best_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the learned image compression autoencoder.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--train-dir", type=str, default=None)
    parser.add_argument("--val-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.train_dir is not None:
        config["data"]["train_dir"] = args.train_dir
    if args.val_dir is not None:
        config["data"]["val_dir"] = args.val_dir
    if args.output_dir is not None:
        config["training"]["output_dir"] = args.output_dir
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["data"]["batch_size"] = args.batch_size
    if args.lr is not None:
        config["training"]["learning_rate"] = args.lr
    if args.resume is not None:
        config["training"]["resume_from"] = args.resume
    run_training(config)


if __name__ == "__main__":
    main()
