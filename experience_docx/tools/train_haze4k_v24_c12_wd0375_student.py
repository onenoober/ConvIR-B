#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TVF


IMG_EXT = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def first_dir(root: Path, names: tuple[str, ...]) -> Path:
    for name in names:
        p = root / name
        if p.is_dir():
            return p
    raise FileNotFoundError(f"none of {names} under {root}")


def label_path(gt_dir: Path, image_name: str) -> Path:
    stem = Path(image_name).stem
    ext = Path(image_name).suffix
    candidates = [image_name]
    if "_" in stem:
        base = stem.split("_")[0]
        candidates.extend([f"{base}{ext}", f"{base}.png"])
    for candidate in candidates:
        p = gt_dir / candidate
        if p.is_file():
            return p
    raise FileNotFoundError(f"no GT for {image_name}")


def load_convir(convir_dir: Path):
    sys.path.insert(0, str(convir_dir))
    from models.ConvIR import build_net  # type: ignore

    return build_net


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class C12Dataset(Dataset):
    def __init__(self, data_dir: Path, split_manifest: Path, teacher_dir: Path, crop: int) -> None:
        payload = json.loads(split_manifest.read_text(encoding="utf-8"))
        self.names = list(payload["train_core"])
        self.input_dir = first_dir(data_dir / "train", ("IN", "haze", "hazy"))
        self.gt_dir = first_dir(data_dir / "train", ("GT", "gt"))
        self.teacher_dir = teacher_dir
        self.crop = crop
        missing = [n for n in self.names if not (self.teacher_dir / n).is_file()]
        if missing:
            raise FileNotFoundError(f"missing teacher cache images: {missing[:10]}")

    def __len__(self) -> int:
        return len(self.names)

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
        return TVF.to_tensor(hazy), TVF.to_tensor(gt), TVF.to_tensor(teacher), name


def init_model(model, checkpoint: Path) -> None:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)


def fft_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pf = torch.fft.fft2(pred, dim=(-2, -1))
    tf = torch.fft.fft2(target, dim=(-2, -1))
    return F.l1_loss(torch.stack((pf.real, pf.imag), -1), torch.stack((tf.real, tf.imag), -1))


def train_one(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    build_net = load_convir(args.convir_dir)
    model = build_net("base", "Haze4K", "original").to(device)
    init_model(model, args.init_checkpoint)
    dataset = C12Dataset(args.data_dir, args.split_manifest, args.teacher_dir, args.crop_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-8)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_rows: list[dict[str, Any]] = []
    start = time.time()
    global_step = 0
    model.train()
    for epoch in range(1, args.epochs + 1):
        losses = []
        gt_losses = []
        teacher_losses = []
        for hazy, gt, teacher, _names in loader:
            hazy = hazy.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)
            teacher = teacher.to(device, non_blocking=True)
            out = model(hazy)
            pred4, pred2, pred = out[0], out[1], out[2]
            gt2 = F.interpolate(gt, scale_factor=0.5, mode="bilinear")
            gt4 = F.interpolate(gt, scale_factor=0.25, mode="bilinear")
            teacher2 = F.interpolate(teacher, scale_factor=0.5, mode="bilinear")
            teacher4 = F.interpolate(teacher, scale_factor=0.25, mode="bilinear")
            gt_l = F.l1_loss(pred, gt) + F.l1_loss(pred2, gt2) + F.l1_loss(pred4, gt4)
            teacher_l = F.l1_loss(pred, teacher) + F.l1_loss(pred2, teacher2) + F.l1_loss(pred4, teacher4)
            f_l = fft_loss(pred, gt) + fft_loss(pred2, gt2) + fft_loss(pred4, gt4)
            loss = args.gt_weight * gt_l + args.teacher_weight * teacher_l + args.fft_weight * f_l
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            opt.step()
            global_step += 1
            losses.append(float(loss.item()))
            gt_losses.append(float(gt_l.item()))
            teacher_losses.append(float(teacher_l.item()))
            if global_step % args.print_freq == 0:
                print(
                    f"C12_TRAIN variant={args.variant} epoch={epoch}/{args.epochs} "
                    f"step={global_step} loss={sum(losses)/len(losses):.6f} "
                    f"gt={sum(gt_losses)/len(gt_losses):.6f} teacher={sum(teacher_losses)/len(teacher_losses):.6f}",
                    flush=True,
                )
        rec = {
            "variant": args.variant,
            "epoch": epoch,
            "global_step": global_step,
            "loss": sum(losses) / max(1, len(losses)),
            "gt_loss": sum(gt_losses) / max(1, len(gt_losses)),
            "teacher_loss": sum(teacher_losses) / max(1, len(teacher_losses)),
            "elapsed_sec": time.time() - start,
        }
        log_rows.append(rec)
        torch.save({"model": model.state_dict(), "epoch": epoch, "variant": args.variant}, ckpt_dir / f"model_{epoch}.pkl")
        torch.save({"model": model.state_dict(), "epoch": epoch, "variant": args.variant}, ckpt_dir / "Final.pkl")
    torch.save({"model": model.state_dict(), "epoch": args.epochs, "variant": args.variant}, ckpt_dir / "Best.pkl")
    with (args.out_dir / "train_history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(log_rows)
    (args.out_dir / "train_config.json").write_text(
        json.dumps(vars(args), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print("C12_TRAIN_OK", json.dumps({"variant": args.variant, "epochs": args.epochs, "steps": global_step}, sort_keys=True))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--convir-dir", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--split-manifest", type=Path, required=True)
    ap.add_argument("--teacher-dir", type=Path, required=True)
    ap.add_argument("--init-checkpoint", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--crop-size", type=int, default=256)
    ap.add_argument("--learning-rate", type=float, default=1e-5)
    ap.add_argument("--gt-weight", type=float, default=0.5)
    ap.add_argument("--teacher-weight", type=float, default=0.5)
    ap.add_argument("--fft-weight", type=float, default=0.01)
    ap.add_argument("--grad-clip-norm", type=float, default=0.001)
    ap.add_argument("--print-freq", type=int, default=100)
    args = ap.parse_args()
    train_one(args)


if __name__ == "__main__":
    main()
