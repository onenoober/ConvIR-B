#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TVF

from c13_common import (
    first_dir,
    label_path,
    load_c13,
    load_teacher_metric_map,
    load_train_core_names,
    stable_subset,
    write_json,
)


class C13TrainDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        split_manifest: Path,
        teacher_dir: Path,
        teacher_metrics: Path,
        crop: int,
        max_images: int,
        seed: int,
        teacher_margin: float,
    ) -> None:
        names = load_train_core_names(split_manifest)
        self.names = stable_subset(names, max_images, seed)
        self.input_dir = first_dir(data_dir / "train", ("IN", "haze", "hazy"))
        self.gt_dir = first_dir(data_dir / "train", ("GT", "gt"))
        self.teacher_dir = teacher_dir
        self.teacher_metrics = load_teacher_metric_map(teacher_metrics)
        self.crop = crop
        self.teacher_margin = teacher_margin
        missing = [name for name in self.names if not (self.teacher_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(f"missing teacher cache images: {missing[:10]}")

    def __len__(self) -> int:
        return len(self.names)

    def _teacher_positive(self, name: str) -> float:
        row = self.teacher_metrics.get(name)
        if not row:
            return 0.0
        return float(float(row.get("WD0375_dPSNR", 0.0)) > self.teacher_margin)

    def __getitem__(self, idx: int):
        name = self.names[idx]
        hazy = Image.open(self.input_dir / name).convert("RGB")
        gt = Image.open(label_path(self.gt_dir, name)).convert("RGB")
        teacher = Image.open(self.teacher_dir / name).convert("RGB")
        if self.crop > 0:
            w, h = hazy.size
            th = tw = self.crop
            if h < th or w < tw:
                pad_h = max(0, th - h)
                pad_w = max(0, tw - w)
                hazy = TVF.pad(hazy, [0, 0, pad_w, pad_h], padding_mode="reflect")
                gt = TVF.pad(gt, [0, 0, pad_w, pad_h], padding_mode="reflect")
                teacher = TVF.pad(teacher, [0, 0, pad_w, pad_h], padding_mode="reflect")
                w, h = hazy.size
            top = random.randint(0, h - th)
            left = random.randint(0, w - tw)
            hazy = TVF.crop(hazy, top, left, th, tw)
            gt = TVF.crop(gt, top, left, th, tw)
            teacher = TVF.crop(teacher, top, left, th, tw)
        if random.random() < 0.5:
            hazy = TVF.hflip(hazy)
            gt = TVF.hflip(gt)
            teacher = TVF.hflip(teacher)
        return (
            TVF.to_tensor(hazy),
            TVF.to_tensor(gt),
            TVF.to_tensor(teacher),
            torch.tensor(self._teacher_positive(name), dtype=torch.float32),
            name,
        )


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def fft_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pf = torch.fft.fft2(pred, dim=(-2, -1))
    tf = torch.fft.fft2(target, dim=(-2, -1))
    return F.l1_loss(torch.stack((pf.real, pf.imag), -1), torch.stack((tf.real, tf.imag), -1))


def lowpass(x: torch.Tensor, scale: int = 4) -> torch.Tensor:
    pooled = F.avg_pool2d(x, kernel_size=scale, stride=scale, ceil_mode=False)
    return F.interpolate(pooled, size=x.shape[-2:], mode="bilinear", align_corners=False)


def tv_loss(x: torch.Tensor) -> torch.Tensor:
    return (x[..., 1:, :] - x[..., :-1, :]).abs().mean() + (x[..., :, 1:] - x[..., :, :-1]).abs().mean()


def masked_l1(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    while mask.ndim < a.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.to(dtype=a.dtype, device=a.device)
    denom = mask.mean().clamp_min(1.0 / max(1, mask.numel()))
    return (torch.abs(a - b) * mask).mean() / denom


def train_one(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    build_net = load_c13(args.convir_dir)
    model = build_net(
        "base",
        "Haze4K",
        a0_checkpoint=str(args.a0_checkpoint),
        feature_mode=args.feature_mode,
        adapter_width=args.adapter_width,
        adapter_depth=args.adapter_depth,
        bootstrap_scale=args.bootstrap_scale,
    ).to(device)
    for name, param in model.named_parameters():
        param.requires_grad_(name.startswith("C13_"))
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.Adam([p for _, p in trainable], lr=args.learning_rate, weight_decay=args.weight_decay)
    dataset = C13TrainDataset(
        args.data_dir,
        args.split_manifest,
        args.teacher_dir,
        args.teacher_metrics,
        args.crop_size,
        args.max_images,
        args.seed,
        args.teacher_margin,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=not args.keep_partial_batch,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    cfg = vars(args).copy()
    cfg.update(
        {
            "trainable_param_count": sum(p.numel() for _, p in trainable),
            "frozen_param_count": sum(p.numel() for name, p in model.named_parameters() if not p.requires_grad),
            "trainable_prefixes": sorted({name.split(".")[0] for name, _ in trainable}),
            "train_image_count": len(dataset),
        }
    )
    write_json(args.out_dir / "train_config.json", cfg)
    rows: list[dict[str, Any]] = []
    start = time.time()
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses: list[dict[str, float]] = []
        for hazy, gt, teacher, teacher_positive, _names in loader:
            hazy = hazy.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)
            teacher = teacher.to(device, non_blocking=True)
            teacher_positive = teacher_positive.to(device, non_blocking=True)
            aux = model.route_forward(hazy)
            pred = aux["outputs"][-1]
            a0 = aux["a0"]
            residual = aux["residual"]
            raw = aux["raw_residual"]
            assert isinstance(pred, torch.Tensor)
            assert isinstance(a0, torch.Tensor)
            assert isinstance(residual, torch.Tensor)
            assert isinstance(raw, torch.Tensor)
            target_residual = teacher - a0.detach()
            loss_gt = F.l1_loss(pred, gt)
            loss_teacher = masked_l1(residual, target_residual.detach(), teacher_positive)
            loss_preserve = masked_l1(pred, a0.detach(), 1.0 - teacher_positive)
            loss_freq = F.l1_loss(lowpass(pred), lowpass(teacher)) + F.l1_loss(lowpass(residual), lowpass(target_residual.detach()))
            loss_color = F.l1_loss(pred.mean(dim=(-2, -1)), gt.mean(dim=(-2, -1)))
            loss_tv = tv_loss(residual)
            loss_raw = raw.abs().mean()
            loss = (
                args.gt_weight * loss_gt
                + args.teacher_weight * loss_teacher
                + args.preserve_weight * loss_preserve
                + args.freq_weight * loss_freq
                + args.color_weight * loss_color
                + args.tv_weight * loss_tv
                + args.raw_weight * loss_raw
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_([p for _, p in trainable], args.grad_clip_norm)
            optimizer.step()
            global_step += 1
            step_row = {
                "loss": float(loss.item()),
                "loss_gt": float(loss_gt.item()),
                "loss_teacher": float(loss_teacher.item()),
                "loss_preserve": float(loss_preserve.item()),
                "loss_freq": float(loss_freq.item()),
                "loss_color": float(loss_color.item()),
                "loss_tv": float(loss_tv.item()),
                "loss_raw": float(loss_raw.item()),
                "teacher_positive_batch": float(teacher_positive.mean().item()),
                "gate_abs_max": float(torch.tanh(model.C13_gate).abs().max().item()),
                "residual_mean_abs": float(residual.abs().mean().item()),
                "raw_residual_mean_abs": float(raw.abs().mean().item()),
            }
            epoch_losses.append(step_row)
            if global_step % args.print_freq == 0:
                print(
                    "C13_TRAIN "
                    f"variant={args.variant} epoch={epoch}/{args.epochs} step={global_step} "
                    f"loss={step_row['loss']:.6f} gt={step_row['loss_gt']:.6f} "
                    f"teacher={step_row['loss_teacher']:.6f} preserve={step_row['loss_preserve']:.6f} "
                    f"residual={step_row['residual_mean_abs']:.8f} gate={step_row['gate_abs_max']:.8f}",
                    flush=True,
                )
        avg = {
            key: sum(row[key] for row in epoch_losses) / max(1, len(epoch_losses))
            for key in epoch_losses[0]
        }
        avg.update(
            {
                "variant": args.variant,
                "epoch": epoch,
                "global_step": global_step,
                "elapsed_sec": time.time() - start,
                "train_image_count": len(dataset),
            }
        )
        rows.append(avg)
        torch.save(
            {
                "model": model.state_dict(),
                "epoch": epoch,
                "variant": args.variant,
                "feature_mode": args.feature_mode,
                "adapter_width": args.adapter_width,
                "adapter_depth": args.adapter_depth,
            },
            ckpt_dir / f"model_{epoch}.pkl",
        )
        torch.save({"model": model.state_dict(), "epoch": epoch, "variant": args.variant}, ckpt_dir / "Final.pkl")
    torch.save({"model": model.state_dict(), "epoch": args.epochs, "variant": args.variant}, ckpt_dir / "Best.pkl")
    with (args.out_dir / "train_history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(
        "C13_TRAIN_OK",
        json.dumps(
            {
                "variant": args.variant,
                "epochs": args.epochs,
                "steps": global_step,
                "train_image_count": len(dataset),
                "trainable_param_count": cfg["trainable_param_count"],
            },
            sort_keys=True,
        ),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--convir-dir", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--split-manifest", type=Path, required=True)
    ap.add_argument("--teacher-dir", type=Path, required=True)
    ap.add_argument("--teacher-metrics", type=Path, required=True)
    ap.add_argument("--a0-checkpoint", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--feature-mode", default="rgb_wavelet", choices=["rgb", "rgb_wavelet"])
    ap.add_argument("--adapter-width", type=int, default=32)
    ap.add_argument("--adapter-depth", type=int, default=3)
    ap.add_argument("--bootstrap-scale", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--crop-size", type=int, default=256)
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--keep-partial-batch", action="store_true")
    ap.add_argument("--learning-rate", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--teacher-margin", type=float, default=0.02)
    ap.add_argument("--gt-weight", type=float, default=0.50)
    ap.add_argument("--teacher-weight", type=float, default=1.00)
    ap.add_argument("--preserve-weight", type=float, default=0.50)
    ap.add_argument("--freq-weight", type=float, default=0.10)
    ap.add_argument("--color-weight", type=float, default=0.05)
    ap.add_argument("--tv-weight", type=float, default=0.01)
    ap.add_argument("--raw-weight", type=float, default=0.0)
    ap.add_argument("--grad-clip-norm", type=float, default=1.0)
    ap.add_argument("--print-freq", type=int, default=100)
    args = ap.parse_args()
    train_one(args)


if __name__ == "__main__":
    main()
