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
import torchvision.transforms.functional as TVF


TOOL_PATH = Path(__file__).resolve()
REPO_ROOT = TOOL_PATH.parents[2]
ITS_ROOT = REPO_ROOT / "Dehazing" / "ITS"
for path in (str(ITS_ROOT), str(REPO_ROOT), os.getcwd()):
    if path not in sys.path:
        sys.path.insert(0, path)


IMG_EXT = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def first_dir(root: Path, names: tuple[str, ...]) -> Path:
    for name in names:
        p = root / name
        if p.is_dir():
            return p
    raise FileNotFoundError(f"none of {names} under {root}")


def train_dirs(data_dir: Path) -> tuple[Path, Path]:
    train = data_dir / "train"
    return first_dir(train, ("IN", "haze", "hazy")), first_dir(train, ("GT", "gt"))


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
    raise FileNotFoundError(f"no GT for {image_name} under {gt_dir}")


def read_split_names(path: Path, fold: int, role: str, max_images: int = 0) -> list[str]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if role == "train":
        names = [r["name"] for r in rows if int(r["oof_fold"]) != fold]
    elif role == "val":
        names = [r["name"] for r in rows if int(r["oof_fold"]) == fold]
    else:
        raise ValueError(role)
    return names[:max_images] if max_images else names


class Haze4KNameDataset(Dataset):
    def __init__(self, data_dir: Path, names: list[str], crop_size: int, training: bool):
        self.input_dir, self.gt_dir = train_dirs(data_dir)
        self.names = names
        self.crop_size = crop_size
        self.training = training

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int):
        name = self.names[idx]
        hazy = Image.open(self.input_dir / name).convert("RGB")
        gt = Image.open(label_path(self.gt_dir, name)).convert("RGB")
        if self.training:
            w, h = hazy.size
            if h < self.crop_size or w < self.crop_size:
                pad_h = max(0, self.crop_size - h)
                pad_w = max(0, self.crop_size - w)
                hazy = TVF.pad(hazy, (0, 0, pad_w, pad_h), padding_mode="reflect")
                gt = TVF.pad(gt, (0, 0, pad_w, pad_h), padding_mode="reflect")
                w, h = hazy.size
            top = random.randint(0, h - self.crop_size)
            left = random.randint(0, w - self.crop_size)
            hazy = TVF.crop(hazy, top, left, self.crop_size, self.crop_size)
            gt = TVF.crop(gt, top, left, self.crop_size, self.crop_size)
            if random.random() < 0.5:
                hazy = TVF.hflip(hazy)
                gt = TVF.hflip(gt)
        return TVF.to_tensor(hazy), TVF.to_tensor(gt), name


def load_state(path: Path, device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model" in state:
        return state["model"]
    return state


def partial_load_wldb(model: torch.nn.Module, checkpoint: Path, device: torch.device) -> dict[str, Any]:
    source = load_state(checkpoint, device)
    target = model.state_dict()
    loaded: dict[str, torch.Tensor] = {}
    for key, value in source.items():
        if key in target and tuple(target[key].shape) == tuple(value.shape):
            loaded[key] = value
    missing = [key for key in target if key not in loaded]
    bad_missing = [key for key in missing if not key.startswith("nopost_wldb.")]
    if bad_missing:
        raise RuntimeError(f"bad missing keys during WLDB load: {bad_missing[:20]}")
    target.update(loaded)
    model.load_state_dict(target, strict=True)
    return {"loaded_count": len(loaded), "missing_new_modules": sorted(missing)}


def freeze_anchor(model: torch.nn.Module) -> dict[str, int]:
    trainable = 0
    frozen = 0
    for name, param in model.named_parameters():
        if name.startswith("nopost_wldb."):
            param.requires_grad_(True)
            trainable += param.numel()
        else:
            param.requires_grad_(False)
            frozen += param.numel()
    return {"trainable_params": trainable, "frozen_params": frozen}


def haar_ll(x: torch.Tensor) -> torch.Tensor:
    h, w = x.shape[-2:]
    if h % 2 or w % 2:
        x = F.pad(x, (0, w % 2, 0, h % 2), mode="reflect")
    a = x[:, :, 0::2, 0::2]
    b = x[:, :, 0::2, 1::2]
    c = x[:, :, 1::2, 0::2]
    d = x[:, :, 1::2, 1::2]
    return (a + b + c + d) / 2.0


def tensor_psnr(pred: torch.Tensor, label: torch.Tensor) -> float:
    mse = F.mse_loss(torch.clamp(pred, 0, 1), label).clamp_min(1e-12)
    return float((10 * torch.log10(1 / mse)).detach().cpu())


def final_output(out: Any) -> torch.Tensor:
    return out[2] if isinstance(out, (list, tuple)) else out


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--split-csv", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--crop-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--lowband-weight", type=float, default=0.25)
    ap.add_argument("--preserve-weight", type=float, default=0.02)
    ap.add_argument("--budget-weight", type=float, default=0.05)
    ap.add_argument("--action-budget", type=float, default=0.025)
    ap.add_argument("--save-freq", type=int, default=5)
    ap.add_argument("--print-freq", type=int, default=50)
    ap.add_argument("--max-train-images", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    from models.ConvIR import build_net as build_official
    from models.NoPostWLDBConvIR import build_net as build_wldb

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_wldb("base", "Haze4K", "original").to(device)
    official = build_official("base", "Haze4K", "original").to(device)
    load_report = partial_load_wldb(model, args.checkpoint, device)
    official.load_state_dict(load_state(args.checkpoint, device))
    param_report = freeze_anchor(model)
    official.eval()
    for param in official.parameters():
        param.requires_grad_(False)

    train_names = read_split_names(args.split_csv, args.fold, "train", args.max_train_images)
    val_names = read_split_names(args.split_csv, args.fold, "val")
    dataset = Haze4KNameDataset(args.data_dir, train_names, args.crop_size, training=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.05)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    manifest = {
        "route": "haze4k-v2-16-wldb-a-train",
        "device": str(device),
        "seed": args.seed,
        "fold": args.fold,
        "train_count": len(train_names),
        "val_count": len(val_names),
        "checkpoint": str(args.checkpoint),
        "load_report": load_report,
        "params": param_report,
        "loss": {
            "final_l1": 1.0,
            "lowband_l1": args.lowband_weight,
            "a0_preserve_l1": args.preserve_weight,
            "action_budget_hinge": args.budget_weight,
            "action_budget": args.action_budget,
        },
        "locked_test_touched": False,
    }
    write_json(args.out_dir / "v216_wldb_a_train_manifest.json", manifest)
    print("WLDB_A_TRAIN_MANIFEST", json.dumps(manifest, sort_keys=True), flush=True)

    history = []
    global_step = 0
    best_loss = math.inf
    for epoch in range(1, args.epochs + 1):
        model.train()
        start = time.time()
        sums = {"loss": 0.0, "final_l1": 0.0, "lowband_l1": 0.0, "preserve_l1": 0.0, "budget": 0.0}
        count = 0
        for step, (hazy, gt, _) in enumerate(loader, 1):
            hazy = hazy.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                a0 = final_output(official(hazy))
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                pred = final_output(model(hazy))
                final_l1 = F.l1_loss(pred, gt)
                lowband_l1 = F.l1_loss(haar_ll(pred), haar_ll(gt))
                preserve_l1 = F.l1_loss(pred, a0)
                action = (pred - a0).abs().mean()
                budget = F.relu(action - args.action_budget)
                loss = (
                    final_l1
                    + args.lowband_weight * lowband_l1
                    + args.preserve_weight * preserve_l1
                    + args.budget_weight * budget
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 0.01)
            scaler.step(optimizer)
            scaler.update()
            global_step += 1
            bs = hazy.shape[0]
            count += bs
            sums["loss"] += float(loss.detach().cpu()) * bs
            sums["final_l1"] += float(final_l1.detach().cpu()) * bs
            sums["lowband_l1"] += float(lowband_l1.detach().cpu()) * bs
            sums["preserve_l1"] += float(preserve_l1.detach().cpu()) * bs
            sums["budget"] += float(budget.detach().cpu()) * bs
            if step % args.print_freq == 0:
                print(
                    "WLDB_A_TRAIN epoch=%03d step=%04d/%04d lr=%.8f loss=%.6f final=%.6f low=%.6f preserve=%.6f budget=%.6f"
                    % (
                        epoch,
                        step,
                        len(loader),
                        optimizer.param_groups[0]["lr"],
                        sums["loss"] / count,
                        sums["final_l1"] / count,
                        sums["lowband_l1"] / count,
                        sums["preserve_l1"] / count,
                        sums["budget"] / count,
                    ),
                    flush=True,
                )
        scheduler.step()
        epoch_row = {
            "epoch": epoch,
            "seconds": time.time() - start,
            "lr": optimizer.param_groups[0]["lr"],
            **{key: value / count for key, value in sums.items()},
        }
        history.append(epoch_row)
        write_json(args.out_dir / "v216_wldb_a_train_history.json", {"history": history})
        torch.save({"model": model.state_dict(), "epoch": epoch, "optimizer": optimizer.state_dict()}, ckpt_dir / "model.pkl")
        if epoch % args.save_freq == 0:
            torch.save({"model": model.state_dict(), "epoch": epoch}, ckpt_dir / f"model_{epoch}.pkl")
        if epoch_row["loss"] < best_loss:
            best_loss = epoch_row["loss"]
            torch.save({"model": model.state_dict(), "epoch": epoch}, ckpt_dir / "Best.pkl")
        print("WLDB_A_EPOCH", json.dumps(epoch_row, sort_keys=True), flush=True)
    torch.save({"model": model.state_dict(), "epoch": args.epochs}, ckpt_dir / "Final.pkl")
    print("WLDB_A_TRAIN_OK", json.dumps({"epochs": args.epochs, "best_loss": best_loss}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
