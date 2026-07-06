#!/usr/bin/env python3
"""Run the v2.41 A0-proximal residual canary32 OOF diagnostic.

This script is intended for cloud execution only. It keeps the official
ConvIR-B path frozen and trains only the zero-init A0PROX residual head.
"""

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as VF


IMG_EXTENSIONS = (".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff")


def first_existing_dir(root, names):
    for name in names:
        path = root / name
        if path.is_dir():
            return path
    raise FileNotFoundError(f"none of {names} exists under {root}")


def image_names(path):
    return sorted(
        p.name
        for p in path.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS
    )


def label_path(label_dir, image_name):
    stem = Path(image_name).stem
    ext = Path(image_name).suffix
    candidates = [image_name]
    if "_" in stem:
        candidates.append(f"{stem.split('_')[0]}{ext}")
        candidates.append(f"{stem.split('_')[0]}.png")
    for candidate in candidates:
        path = label_dir / candidate
        if path.is_file():
            return path
    return None


def list_haze4k_records(data_dir, split):
    split_dir = Path(data_dir) / split
    input_dir = first_existing_dir(split_dir, ("IN", "haze", "hazy"))
    label_dir = first_existing_dir(split_dir, ("GT", "gt"))
    records = []
    skipped = []
    for name in image_names(input_dir):
        gt = label_path(label_dir, name)
        if gt is None:
            skipped.append(name)
            continue
        records.append(
            {
                "name": name,
                "input": str(input_dir / name),
                "label": str(gt),
            }
        )
    if not records:
        raise RuntimeError(f"no matched Haze4K records under {split_dir}")
    return records, skipped


def select_oof_splits(records, folds, train_size, val_size, seed):
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    need = folds * (train_size + val_size)
    if len(shuffled) < need:
        raise RuntimeError(f"need {need} matched records, found {len(shuffled)}")
    train_pool = shuffled[: folds * train_size]
    val_pool = shuffled[folds * train_size : need]
    splits = []
    for fold in range(folds):
        train = train_pool[fold * train_size : (fold + 1) * train_size]
        val = val_pool[fold * val_size : (fold + 1) * val_size]
        overlap = {r["name"] for r in train}.intersection(r["name"] for r in val)
        if overlap:
            raise RuntimeError(f"fold {fold} train/val overlap: {sorted(overlap)[:5]}")
        splits.append({"fold": fold, "train": train, "val": val})
    return splits


class Haze4KTrainSubset(Dataset):
    def __init__(self, records, crop_size, seed):
        self.records = list(records)
        self.crop_size = int(crop_size)
        self.seed = int(seed)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        rec = self.records[index]
        image = Image.open(rec["input"]).convert("RGB")
        label = Image.open(rec["label"]).convert("RGB")
        w, h = image.size
        if w < self.crop_size or h < self.crop_size:
            scale = max(self.crop_size / max(w, 1), self.crop_size / max(h, 1))
            new_size = (math.ceil(w * scale), math.ceil(h * scale))
            image = image.resize(new_size, Image.BICUBIC)
            label = label.resize(new_size, Image.BICUBIC)
            w, h = image.size
        rng = random.Random(self.seed + index + random.randint(0, 2**20))
        left = rng.randint(0, w - self.crop_size)
        top = rng.randint(0, h - self.crop_size)
        image = image.crop((left, top, left + self.crop_size, top + self.crop_size))
        label = label.crop((left, top, left + self.crop_size, top + self.crop_size))
        if rng.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
        return VF.to_tensor(image), VF.to_tensor(label)


class Haze4KEvalSubset(Dataset):
    def __init__(self, records):
        self.records = list(records)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        rec = self.records[index]
        image = Image.open(rec["input"]).convert("RGB")
        label = Image.open(rec["label"]).convert("RGB")
        return VF.to_tensor(image), VF.to_tensor(label), rec["name"]


def load_checkpoint_model(path):
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        return state["model"]
    return state


def partial_load_a0prox(model, state):
    model_state = model.state_dict()
    loaded = {}
    unexpected = []
    shape_mismatch = []
    for key, value in state.items():
        if key not in model_state:
            unexpected.append(key)
        elif tuple(model_state[key].shape) != tuple(value.shape):
            shape_mismatch.append([key, list(value.shape), list(model_state[key].shape)])
        else:
            loaded[key] = value
    missing = [key for key in model_state if key not in loaded]
    bad_missing = [key for key in missing if not key.startswith("A0PROX_")]
    if unexpected or shape_mismatch or bad_missing:
        raise RuntimeError(
            "partial load failed: "
            f"unexpected={unexpected} shape_mismatch={shape_mismatch} "
            f"bad_missing={bad_missing}"
        )
    model_state.update(loaded)
    model.load_state_dict(model_state, strict=True)
    return {
        "loaded_count": len(loaded),
        "missing_new_modules": sorted(missing),
        "unexpected": unexpected,
        "shape_mismatch": shape_mismatch,
        "bad_missing": bad_missing,
    }


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def psnr_from_mse(mse):
    return -10.0 * math.log10(max(float(mse), 1e-12))


def pad_to_factor(x, factor=32):
    _, _, h, w = x.shape
    h2 = ((h + factor) // factor) * factor
    w2 = ((w + factor) // factor) * factor
    padh = 0 if h % factor == 0 else h2 - h
    padw = 0 if w % factor == 0 else w2 - w
    if padh == 0 and padw == 0:
        return x, h, w
    return F.pad(x, (0, padw, 0, padh), "reflect"), h, w


def build_models(args, device):
    sys.path.insert(0, str(Path(args.repo_root) / "Dehazing" / "ITS"))
    from models.A0ProxResidualConvIR import build_a0prox_residual_net
    from models.ConvIR import build_net

    state = load_checkpoint_model(args.checkpoint)
    model = build_a0prox_residual_net("base", "Haze4K", beta=args.beta)
    base = build_net("base", "Haze4K")
    report = partial_load_a0prox(model, state)
    base.load_state_dict(state, strict=True)
    for param in base.parameters():
        param.requires_grad = False
    base.eval()
    trainable = []
    frozen = []
    for name, param in model.named_parameters():
        if name.startswith("A0PROX_"):
            param.requires_grad = True
            trainable.append(name)
        else:
            param.requires_grad = False
            frozen.append(name)
    if not trainable:
        raise RuntimeError("no A0PROX trainable parameters found")
    return model.to(device), base.to(device), report, trainable, frozen


def per_image_mse(pred, label):
    return (pred - label).pow(2).flatten(1).mean(dim=1)


def train_fold(args, split, device, output_dir):
    fold = split["fold"]
    set_seed(args.seed + fold)
    model, base, load_report, trainable, frozen = build_models(args, device)
    trainable_param_count = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    loader = DataLoader(
        Haze4KTrainSubset(split["train"], args.crop_size, args.seed + fold * 1000),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    fold_dir = output_dir / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    epoch_history = []
    best_mean_delta = -1e9
    best_path = fold_dir / "Best.pkl"

    for epoch in range(1, args.epochs + 1):
        model.train()
        sums = {
            "loss": 0.0,
            "l_gt": 0.0,
            "anchor": 0.0,
            "hinge": 0.0,
            "cvar": 0.0,
            "violation_count": 0.0,
            "sample_count": 0.0,
        }
        for input_img, label_img in loader:
            input_img = input_img.to(device, non_blocking=True)
            label_img = label_img.to(device, non_blocking=True)
            with torch.no_grad():
                a0 = base(input_img)[2].detach()
            pred = model(input_img)[2]
            l_gt = F.l1_loss(pred, label_img)
            a0_mse = per_image_mse(a0, label_img)
            pred_mse = per_image_mse(pred, label_img)
            easy_mask = a0_mse <= torch.median(a0_mse)
            if bool(easy_mask.any()):
                anchor = F.l1_loss(pred[easy_mask], a0[easy_mask])
            else:
                anchor = F.l1_loss(pred, a0)
            hinge_values = torch.relu(pred_mse - a0_mse + args.hinge_margin)
            hinge = hinge_values.mean()
            bad_ratio = torch.relu(pred_mse / (a0_mse + 1e-12) - 1.0)
            k = max(1, int(math.ceil(args.cvar_frac * bad_ratio.numel())))
            cvar = torch.topk(bad_ratio, k=k).values.mean()
            loss = (
                l_gt
                + args.lambda_anchor * anchor
                + args.lambda_hinge * hinge
                + args.lambda_cvar * cvar
            )
            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    args.grad_clip_norm,
                )
            optimizer.step()

            n = input_img.shape[0]
            sums["loss"] += float(loss.item()) * n
            sums["l_gt"] += float(l_gt.item()) * n
            sums["anchor"] += float(anchor.item()) * n
            sums["hinge"] += float(hinge.item()) * n
            sums["cvar"] += float(cvar.item()) * n
            sums["violation_count"] += float((pred_mse > a0_mse).sum().item())
            sums["sample_count"] += n

        row = {
            "fold": fold,
            "epoch": epoch,
            "loss": sums["loss"] / sums["sample_count"],
            "l_gt": sums["l_gt"] / sums["sample_count"],
            "anchor": sums["anchor"] / sums["sample_count"],
            "hinge": sums["hinge"] / sums["sample_count"],
            "cvar": sums["cvar"] / sums["sample_count"],
            "hinge_violation_rate": sums["violation_count"] / sums["sample_count"],
        }
        epoch_history.append(row)
        if epoch == args.epochs:
            torch.save({"model": model.state_dict(), "epoch": epoch}, fold_dir / "Final.pkl")

    torch.save({"model": model.state_dict(), "epoch": args.epochs}, best_path)
    eval_rows = evaluate_fold(args, model, base, split, device)
    mean_delta = sum(r["delta_psnr"] for r in eval_rows) / len(eval_rows)
    if mean_delta > best_mean_delta:
        best_mean_delta = mean_delta
    return {
        "fold": fold,
        "load_report": load_report,
        "trainable_param_names": trainable,
        "frozen_param_count": len(frozen),
        "train_size": len(split["train"]),
        "val_size": len(split["val"]),
        "epoch_history": epoch_history,
        "eval_rows": eval_rows,
        "best_checkpoint": str(best_path),
        "final_checkpoint": str(fold_dir / "Final.pkl"),
        "trainable_param_count": trainable_param_count,
    }


def evaluate_fold(args, model, base, split, device):
    dataset = Haze4KEvalSubset(split["val"])
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    model.eval()
    base.eval()
    rows = []
    with torch.no_grad():
        for input_img, label_img, names in loader:
            name = names[0]
            input_img = input_img.to(device)
            label_img = label_img.to(device)
            padded, h, w = pad_to_factor(input_img)
            a0 = base(padded)[2][:, :, :h, :w].clamp(0.0, 1.0)
            pred = model(padded)[2][:, :, :h, :w].clamp(0.0, 1.0)
            a0_mse = float((a0 - label_img).pow(2).mean().item())
            pred_mse = float((pred - label_img).pow(2).mean().item())
            residual_energy = float((pred - a0).pow(2).mean().item())
            a0_psnr = psnr_from_mse(a0_mse)
            pred_psnr = psnr_from_mse(pred_mse)
            rows.append(
                {
                    "fold": split["fold"],
                    "image": name,
                    "a0_mse": a0_mse,
                    "pred_mse": pred_mse,
                    "a0_psnr": a0_psnr,
                    "pred_psnr": pred_psnr,
                    "delta_psnr": pred_psnr - a0_psnr,
                    "residual_energy": residual_energy,
                }
            )
    return rows


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def summarize_rows(rows, prefix=""):
    if not rows:
        raise RuntimeError("cannot summarize empty rows")
    ordered_by_a0 = sorted(rows, key=lambda r: r["a0_psnr"])
    bucket_n = max(1, int(math.ceil(0.25 * len(rows))))
    hard_names = {r["image"] for r in ordered_by_a0[:bucket_n]}
    easy_names = {r["image"] for r in ordered_by_a0[-bucket_n:]}
    deltas = [r["delta_psnr"] for r in rows]
    hard = [r["delta_psnr"] for r in rows if r["image"] in hard_names]
    easy = [r["delta_psnr"] for r in rows if r["image"] in easy_names]
    residual_hard = [r["residual_energy"] for r in rows if r["image"] in hard_names]
    residual_easy = [r["residual_energy"] for r in rows if r["image"] in easy_names]
    bottom = sorted(deltas)[: max(1, int(math.ceil(0.05 * len(deltas))))]
    strong_ref_regression = sum(
        1 for r in rows if r["image"] in easy_names and r["delta_psnr"] < -0.01
    )
    severe = sum(1 for d in deltas if d <= -0.20)
    out = {
        f"{prefix}count": len(rows),
        f"{prefix}mean_delta": sum(deltas) / len(deltas),
        f"{prefix}hard_delta": sum(hard) / len(hard),
        f"{prefix}easy_delta": sum(easy) / len(easy),
        f"{prefix}p05_delta": percentile(deltas, 0.05),
        f"{prefix}cvar5_delta": sum(bottom) / len(bottom),
        f"{prefix}worst_delta": min(deltas),
        f"{prefix}severe_count": severe,
        f"{prefix}strong_reference_regression_count": strong_ref_regression,
        f"{prefix}residual_energy_hard_mean": sum(residual_hard) / len(residual_hard),
        f"{prefix}residual_energy_easy_mean": sum(residual_easy) / len(residual_easy),
    }
    hard_energy = out[f"{prefix}residual_energy_hard_mean"]
    easy_energy = out[f"{prefix}residual_energy_easy_mean"]
    out[f"{prefix}residual_energy_easy_to_hard_ratio"] = (
        easy_energy / hard_energy if hard_energy > 0 else 1e12
    )
    return out


def fold_gate(summary, args):
    return (
        summary["mean_delta"] >= args.gate_mean_delta
        and summary["hard_delta"] >= args.gate_hard_delta
        and summary["easy_delta"] >= args.gate_easy_delta
        and summary["p05_delta"] >= args.gate_p05_delta
        and summary["cvar5_delta"] >= args.gate_cvar5_delta
        and summary["severe_count"] == 0
        and summary["strong_reference_regression_count"] == 0
    )


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--evidence-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=24132)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--train-size", type=int, default=32)
    parser.add_argument("--val-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=4e-4)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--lambda-anchor", type=float, default=0.2)
    parser.add_argument("--lambda-hinge", type=float, default=2.0)
    parser.add_argument("--lambda-cvar", type=float, default=1.0)
    parser.add_argument("--hinge-margin", type=float, default=0.0)
    parser.add_argument("--cvar-frac", type=float, default=0.2)
    parser.add_argument("--grad-clip-norm", type=float, default=0.001)
    parser.add_argument("--gate-mean-delta", type=float, default=0.15)
    parser.add_argument("--gate-hard-delta", type=float, default=0.30)
    parser.add_argument("--gate-easy-delta", type=float, default=0.0)
    parser.add_argument("--gate-p05-delta", type=float, default=-0.01)
    parser.add_argument("--gate-cvar5-delta", type=float, default=-0.02)
    parser.add_argument("--gate-min-fold-pass", type=int, default=4)
    parser.add_argument("--gate-easy-hard-energy-ratio", type=float, default=0.50)
    args = parser.parse_args()

    evidence_root = Path(args.evidence_root)
    output_dir = Path(args.output_dir)
    evidence_root.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records, skipped = list_haze4k_records(args.data_dir, "train")
    splits = select_oof_splits(
        records, args.folds, args.train_size, args.val_size, args.seed
    )
    all_rows = []
    fold_rows = []
    epoch_rows = []
    fold_details = []
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    for split in splits:
        result = train_fold(args, split, device, output_dir)
        eval_rows = result["eval_rows"]
        summary = summarize_rows(eval_rows)
        summary["fold"] = result["fold"]
        summary["gate_pass"] = fold_gate(summary, args)
        summary["best_checkpoint"] = result["best_checkpoint"]
        summary["final_checkpoint"] = result["final_checkpoint"]
        fold_rows.append(summary)
        all_rows.extend(eval_rows)
        epoch_rows.extend(result["epoch_history"])
        fold_details.append(
            {
                "fold": result["fold"],
                "train_images": [r["name"] for r in split["train"]],
                "val_images": [r["name"] for r in split["val"]],
                "load_report": result["load_report"],
                "trainable_param_count": result["trainable_param_count"],
                "trainable_param_names": result["trainable_param_names"],
                "frozen_param_count": result["frozen_param_count"],
            }
        )
        print(
            "fold=%d mean=%.4f hard=%.4f easy=%.4f p05=%.4f pass=%s"
            % (
                summary["fold"],
                summary["mean_delta"],
                summary["hard_delta"],
                summary["easy_delta"],
                summary["p05_delta"],
                summary["gate_pass"],
            ),
            flush=True,
        )

    global_summary = summarize_rows(all_rows)
    fold_pass_count = sum(1 for r in fold_rows if r["gate_pass"])
    first_violation = sum(
        r["hinge_violation_rate"] for r in epoch_rows if r["epoch"] == 1
    ) / max(1, sum(1 for r in epoch_rows if r["epoch"] == 1))
    final_violation = sum(
        r["hinge_violation_rate"] for r in epoch_rows if r["epoch"] == args.epochs
    ) / max(1, sum(1 for r in epoch_rows if r["epoch"] == args.epochs))
    mechanism = {
        "residual_energy_easy_to_hard_ratio": global_summary[
            "residual_energy_easy_to_hard_ratio"
        ],
        "residual_energy_ratio_gate": args.gate_easy_hard_energy_ratio,
        "residual_energy_ratio_pass": global_summary[
            "residual_energy_easy_to_hard_ratio"
        ]
        <= args.gate_easy_hard_energy_ratio,
        "hinge_violation_rate_epoch1": first_violation,
        "hinge_violation_rate_final": final_violation,
        "hinge_violation_nonworsen_pass": final_violation <= first_violation + 1e-12,
    }
    mechanism["mechanism_pass"] = (
        mechanism["residual_energy_ratio_pass"]
        and mechanism["hinge_violation_nonworsen_pass"]
    )
    global_gate = (
        global_summary["mean_delta"] >= args.gate_mean_delta
        and global_summary["hard_delta"] >= args.gate_hard_delta
        and global_summary["easy_delta"] >= args.gate_easy_delta
        and global_summary["p05_delta"] >= args.gate_p05_delta
        and global_summary["cvar5_delta"] >= args.gate_cvar5_delta
        and global_summary["severe_count"] == 0
        and global_summary["strong_reference_regression_count"] == 0
    )
    gate_pass = (
        global_gate
        and fold_pass_count >= args.gate_min_fold_pass
        and mechanism["mechanism_pass"]
    )
    decision = (
        "CANARY32_OOF_GATE_PASS_AUTHORIZE_CANARY80_STAGE_DESIGN"
        if gate_pass
        else "CANARY32_OOF_GATE_FAIL_LOCK_CANARY80_LOCKED_TEST"
    )
    summary = {
        "route": "haze4k_v2_41_a0_proximal_supervised_residual",
        "stage": "CANARY32_OOF",
        "decision": decision,
        "gate_pass": gate_pass,
        "global_gate_pass": global_gate,
        "fold_pass_count": fold_pass_count,
        "folds": args.folds,
        "train_size_per_fold": args.train_size,
        "val_size_per_fold": args.val_size,
        "train_derived_only": True,
        "locked_test_touched": False,
        "canary80_authorized": bool(gate_pass),
        "locked_test_authorized": False,
        "device": str(device),
        "started": started,
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "checkpoint": args.checkpoint,
        "data_dir": args.data_dir,
        "skipped_unmatched_train_images": skipped,
        "loss": {
            "lambda_anchor": args.lambda_anchor,
            "lambda_hinge": args.lambda_hinge,
            "lambda_cvar": args.lambda_cvar,
            "hinge_margin": args.hinge_margin,
            "cvar_frac": args.cvar_frac,
        },
        "gates": {
            "mean_delta": args.gate_mean_delta,
            "hard_delta": args.gate_hard_delta,
            "easy_delta": args.gate_easy_delta,
            "p05_delta": args.gate_p05_delta,
            "cvar5_delta": args.gate_cvar5_delta,
            "severe_count": 0,
            "strong_reference_regression_count": 0,
            "min_fold_pass": args.gate_min_fold_pass,
            "easy_to_hard_residual_energy_ratio": args.gate_easy_hard_energy_ratio,
        },
        "global_summary": global_summary,
        "mechanism": mechanism,
        "fold_summaries": fold_rows,
        "fold_details": fold_details,
    }

    per_image_path = evidence_root / "v241_canary32_oof_per_image.csv"
    fold_path = evidence_root / "v241_canary32_oof_folds.csv"
    history_path = evidence_root / "v241_canary32_oof_epoch_history.csv"
    summary_path = evidence_root / "v241_canary32_oof_summary.json"
    closeout_path = evidence_root / "v241_canary32_closeout.json"
    write_csv(
        per_image_path,
        all_rows,
        [
            "fold",
            "image",
            "a0_mse",
            "pred_mse",
            "a0_psnr",
            "pred_psnr",
            "delta_psnr",
            "residual_energy",
        ],
    )
    write_csv(
        fold_path,
        fold_rows,
        [
            "fold",
            "count",
            "mean_delta",
            "hard_delta",
            "easy_delta",
            "p05_delta",
            "cvar5_delta",
            "worst_delta",
            "severe_count",
            "strong_reference_regression_count",
            "residual_energy_hard_mean",
            "residual_energy_easy_mean",
            "residual_energy_easy_to_hard_ratio",
            "gate_pass",
            "best_checkpoint",
            "final_checkpoint",
        ],
    )
    write_csv(
        history_path,
        epoch_rows,
        ["fold", "epoch", "loss", "l_gt", "anchor", "hinge", "cvar", "hinge_violation_rate"],
    )
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    closeout = {
        "decision": decision,
        "gate_pass": gate_pass,
        "status": "COMPLETED_GATE_PASS" if gate_pass else "COMPLETED_GATE_FAIL",
        "canary80_authorized": bool(gate_pass),
        "locked_test_authorized": False,
        "locked_test_touched": False,
        "summary_path": str(summary_path),
    }
    with open(closeout_path, "w", encoding="utf-8") as f:
        json.dump(closeout, f, indent=2, sort_keys=True)
    print(json.dumps(closeout, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
