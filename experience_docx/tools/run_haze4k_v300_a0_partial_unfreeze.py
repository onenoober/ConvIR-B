#!/usr/bin/env python3
"""Run v3.0 A0-anchored partial-unfreeze Haze4K stages on cloud only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as VF


IMG_EXTENSIONS = (".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff")
GAMMAS = (0.0, 0.03125, 0.0625, 0.125, 0.25, 0.5, 0.75, 1.0)


def first_existing_dir(root: Path, names: tuple[str, ...]) -> Path:
    for name in names:
        path = root / name
        if path.is_dir():
            return path
    raise FileNotFoundError(f"none of {names} exists under {root}")


def image_names(path: Path) -> list[str]:
    return sorted(
        p.name
        for p in path.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS
    )


def label_path(label_dir: Path, image_name: str) -> Path | None:
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


def list_haze4k_records(data_dir: Path, split: str) -> tuple[list[dict[str, str]], list[str]]:
    split_dir = data_dir / split
    input_dir = first_existing_dir(split_dir, ("IN", "haze", "hazy"))
    label_dir = first_existing_dir(split_dir, ("GT", "gt"))
    records = []
    skipped = []
    for name in image_names(input_dir):
        gt = label_path(label_dir, name)
        if gt is None:
            skipped.append(name)
            continue
        records.append({"name": name, "input": str(input_dir / name), "label": str(gt)})
    if not records:
        raise RuntimeError(f"no matched records under {split_dir}")
    return records, skipped


def load_checkpoint_model(path: Path) -> dict[str, torch.Tensor]:
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    if not isinstance(state, dict):
        raise RuntimeError(f"checkpoint is not a state dict: {path}")
    return state


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def psnr_from_mse(mse: float) -> float:
    return -10.0 * math.log10(max(float(mse), 1e-12))


def pad_to_factor(x: torch.Tensor, factor: int = 32) -> tuple[torch.Tensor, int, int]:
    _, _, h, w = x.shape
    h2 = ((h + factor) // factor) * factor
    w2 = ((w + factor) // factor) * factor
    padh = 0 if h % factor == 0 else h2 - h
    padw = 0 if w % factor == 0 else w2 - w
    if padh == 0 and padw == 0:
        return x, h, w
    return F.pad(x, (0, padw, 0, padh), "reflect"), h, w


class Haze4KTrainSubset(Dataset):
    def __init__(self, records: list[dict[str, str]], crop_size: int, seed: int):
        self.records = list(records)
        self.crop_size = int(crop_size)
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
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


def load_image_tensor(path: str, device: torch.device) -> torch.Tensor:
    return VF.to_tensor(Image.open(path).convert("RGB")).unsqueeze(0).to(device)


def build_models(args: argparse.Namespace, route_checkpoint: Path | None, device: torch.device):
    sys.path.insert(0, str(Path(args.repo_root) / "Dehazing" / "ITS"))
    from models.ConvIR import build_net
    from models.V300A0AnchoredConvIR import build_v300_a0_anchored_net

    state = load_checkpoint_model(Path(args.checkpoint))
    base = build_net("base", "Haze4K")
    route = build_v300_a0_anchored_net("base", "Haze4K", residual_scale=args.residual_scale)
    base.load_state_dict(state, strict=True)
    if route_checkpoint is None:
        partial = partial_load(route, state, ("V300_",))
    else:
        route.load_state_dict(load_checkpoint_model(route_checkpoint), strict=True)
        partial = {"loaded_from_route_checkpoint": str(route_checkpoint)}
    base.eval().to(device)
    route.eval().to(device)
    for p in base.parameters():
        p.requires_grad = False
    return route, base, partial


def partial_load(model: torch.nn.Module, state: dict[str, torch.Tensor], prefixes: tuple[str, ...]):
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
    bad_missing = [key for key in missing if not any(key.startswith(p) for p in prefixes)]
    if unexpected or shape_mismatch or bad_missing:
        raise RuntimeError(
            f"partial load failed unexpected={unexpected[:10]} "
            f"shape_mismatch={shape_mismatch[:10]} bad_missing={bad_missing[:20]}"
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


def trainable_param_groups(model: torch.nn.Module, scope: str, args: argparse.Namespace):
    new_params = []
    backbone_params = []
    trainable_names = []
    for name, param in model.named_parameters():
        trainable = False
        is_new = name.startswith("V300_")
        is_backbone = name.startswith(("Decoder.2", "Convs.1", "feat_extract.5"))
        if scope == "frozen_probe":
            trainable = is_new
        elif scope == "tier_a_partial":
            trainable = is_new or is_backbone
        elif scope == "tier_b_partial":
            trainable = is_new or name.startswith(
                (
                    "Decoder.1",
                    "Decoder.2",
                    "Convs.0",
                    "Convs.1",
                    "feat_extract.3",
                    "feat_extract.4",
                    "feat_extract.5",
                )
            )
        else:
            raise ValueError(f"unknown scope: {scope}")
        param.requires_grad = trainable
        if trainable:
            trainable_names.append(name)
            if is_new:
                new_params.append(param)
            else:
                backbone_params.append(param)
    groups = []
    if new_params:
        groups.append({"params": new_params, "lr": args.lr_new, "name": "new"})
    if backbone_params:
        groups.append({"params": backbone_params, "lr": args.lr_backbone, "name": "backbone"})
    if not groups:
        raise RuntimeError(f"no trainable params for {scope}")
    return groups, trainable_names


def set_train_modes(model: torch.nn.Module, trainable_names: list[str]) -> None:
    model.eval()
    trainable_roots = {name.rsplit(".", 1)[0] for name in trainable_names}
    for module_name, module in model.named_modules():
        if module_name in trainable_roots or any(root.startswith(module_name + ".") for root in trainable_roots):
            module.train()


def per_image_mse(pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    return (pred - label).pow(2).flatten(1).mean(dim=1)


def charbonnier(pred: torch.Tensor, label: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt((pred - label).pow(2) + eps * eps).mean()


def summarize_rows(rows: list[dict[str, Any]], prefix: str = "") -> dict[str, Any]:
    ordered_by_a0 = sorted(rows, key=lambda r: float(r["a0_psnr"]))
    bucket_n = max(1, int(math.ceil(0.25 * len(rows))))
    hard_names = {r["image"] for r in ordered_by_a0[:bucket_n]}
    easy_names = {r["image"] for r in ordered_by_a0[-bucket_n:]}
    deltas = [float(r["delta_psnr"]) for r in rows]
    hard = [float(r["delta_psnr"]) for r in rows if r["image"] in hard_names]
    easy = [float(r["delta_psnr"]) for r in rows if r["image"] in easy_names]
    bottom = sorted(deltas)[: max(1, int(math.ceil(0.05 * len(deltas))))]
    residual_hard = [float(r["residual_energy"]) for r in rows if r["image"] in hard_names]
    residual_easy = [float(r["residual_energy"]) for r in rows if r["image"] in easy_names]
    out = {
        f"{prefix}count": len(rows),
        f"{prefix}mean_delta": sum(deltas) / len(deltas),
        f"{prefix}hard_delta": sum(hard) / len(hard),
        f"{prefix}easy_delta": sum(easy) / len(easy),
        f"{prefix}p05_delta": percentile(deltas, 0.05),
        f"{prefix}cvar5_delta": sum(bottom) / len(bottom),
        f"{prefix}worst_delta": min(deltas),
        f"{prefix}severe_count": sum(1 for d in deltas if d <= -0.20),
        f"{prefix}strong_reference_regression_count": sum(
            1 for r in rows if r["image"] in easy_names and float(r["delta_psnr"]) < -0.01
        ),
        f"{prefix}residual_energy_hard_mean": sum(residual_hard) / len(residual_hard),
        f"{prefix}residual_energy_easy_mean": sum(residual_easy) / len(residual_easy),
    }
    hard_energy = out[f"{prefix}residual_energy_hard_mean"]
    easy_energy = out[f"{prefix}residual_energy_easy_mean"]
    out[f"{prefix}residual_energy_easy_to_hard_ratio"] = (
        easy_energy / hard_energy if hard_energy > 0 else 1e12
    )
    return out


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def fold_gate(summary: dict[str, Any], args: argparse.Namespace) -> bool:
    return (
        summary["mean_delta"] >= args.gate_mean_delta
        and summary["hard_delta"] >= args.gate_hard_delta
        and summary["easy_delta"] >= args.gate_easy_delta
        and summary["p05_delta"] >= args.gate_p05_delta
        and summary["cvar5_delta"] >= args.gate_cvar5_delta
        and summary["severe_count"] == 0
        and summary["strong_reference_regression_count"] == 0
    )


def evaluate_records(model, base, records, fold: int, split: str, device: torch.device):
    rows = []
    model.eval()
    base.eval()
    with torch.no_grad():
        for rec in records:
            input_img = load_image_tensor(rec["input"], device)
            label_img = load_image_tensor(rec["label"], device)
            padded, h, w = pad_to_factor(input_img)
            a0 = base(padded)[2][:, :, :h, :w].clamp(0.0, 1.0)
            pred = model(padded)[2][:, :, :h, :w].clamp(0.0, 1.0)
            e0 = a0 - label_img
            residual = pred - a0
            a0_mse = float(e0.pow(2).mean().item())
            pred_mse = float((pred - label_img).pow(2).mean().item())
            residual_energy = float(residual.pow(2).mean().item())
            alignment_dot = float((e0 * residual).mean().item())
            denom = math.sqrt(max(a0_mse * residual_energy, 0.0))
            alignment_cos = alignment_dot / denom if denom > 0 else None
            a0_psnr = psnr_from_mse(a0_mse)
            pred_psnr = psnr_from_mse(pred_mse)
            alpha_safe_upper = -2.0 * alignment_dot / residual_energy if residual_energy > 1e-20 else None
            rows.append(
                {
                    "fold": fold,
                    "split": split,
                    "image": rec["name"],
                    "a0_mse": a0_mse,
                    "pred_mse": pred_mse,
                    "a0_psnr": a0_psnr,
                    "pred_psnr": pred_psnr,
                    "delta_psnr": pred_psnr - a0_psnr,
                    "residual_energy": residual_energy,
                    "alignment_dot": alignment_dot,
                    "alignment_cos": alignment_cos,
                    "alpha_safe_upper": alpha_safe_upper,
                    "direction_bad": alignment_dot >= 0.0,
                    "overshoot_bad": alignment_dot < 0.0 and alpha_safe_upper is not None and alpha_safe_upper < 1.0,
                }
            )
    return rows


def geometry_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    severe = [r for r in rows if float(r["delta_psnr"]) <= -0.20]
    neg = [r for r in rows if float(r["delta_psnr"]) < 0.0]
    return {
        "count": len(rows),
        "global_summary": summarize_rows(rows),
        "negative_count": len(neg),
        "negative_direction_bad_count": sum(1 for r in neg if r["direction_bad"]),
        "negative_direction_bad_rate": (
            sum(1 for r in neg if r["direction_bad"]) / len(neg) if neg else None
        ),
        "severe_count": len(severe),
        "severe_direction_bad_count": sum(1 for r in severe if r["direction_bad"]),
        "severe_overshoot_bad_count": sum(1 for r in severe if r["overshoot_bad"]),
        "severe_direction_bad_rate": (
            sum(1 for r in severe if r["direction_bad"]) / len(severe) if severe else None
        ),
        "severe_overshoot_bad_rate": (
            sum(1 for r in severe if r["overshoot_bad"]) / len(severe) if severe else None
        ),
        "alignment_dot_mean": sum(float(r["alignment_dot"]) for r in rows) / len(rows),
        "alpha_safe_upper_p50": percentile(
            [float(r["alpha_safe_upper"]) for r in rows if r["alpha_safe_upper"] is not None],
            0.50,
        ),
    }


def shrink_curve(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    out = []
    for gamma in GAMMAS:
        gamma_rows = []
        for r in rows:
            mse = (
                float(r["a0_mse"])
                + 2.0 * gamma * float(r["alignment_dot"])
                + gamma * gamma * float(r["residual_energy"])
            )
            psnr = psnr_from_mse(mse)
            gamma_rows.append(
                {
                    "fold": r["fold"],
                    "image": r["image"],
                    "a0_psnr": r["a0_psnr"],
                    "delta_psnr": psnr - float(r["a0_psnr"]),
                    "residual_energy": gamma * gamma * float(r["residual_energy"]),
                }
            )
        summary = summarize_rows(gamma_rows)
        fold_pass_count = 0
        for fold in sorted({int(r["fold"]) for r in gamma_rows}):
            fold_summary = summarize_rows([r for r in gamma_rows if int(r["fold"]) == fold])
            if fold_gate(fold_summary, args):
                fold_pass_count += 1
        out.append(
            {
                "gamma": gamma,
                **summary,
                "fold_pass_count": fold_pass_count,
                "gate_pass": fold_gate(summary, args) and fold_pass_count >= args.gate_min_fold_pass,
            }
        )
    return out


def oracle_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    oracle_rows = []
    chosen = 0
    for r in rows:
        use = float(r["delta_psnr"]) > 0.0
        chosen += int(use)
        oracle_rows.append(
            {
                "fold": r["fold"],
                "image": r["image"],
                "a0_psnr": r["a0_psnr"],
                "delta_psnr": float(r["delta_psnr"]) if use else 0.0,
                "residual_energy": r["residual_energy"] if use else 0.0,
            }
        )
    summary = summarize_rows(oracle_rows)
    summary.update({"oracle_positive_count": chosen, "oracle_positive_rate": chosen / len(rows)})
    return summary


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def split_from_v241(summary_path: Path, records: list[dict[str, str]]):
    summary = json.loads(summary_path.read_text())
    by_name = {r["name"]: r for r in records}
    splits = []
    for detail in summary["fold_details"]:
        splits.append(
            {
                "fold": int(detail["fold"]),
                "train": [by_name[name] for name in detail["train_images"]],
                "val": [by_name[name] for name in detail["val_images"]],
            }
        )
    return splits


def train_fold(args, split, scope: str, device: torch.device, output_dir: Path):
    fold = int(split["fold"])
    set_seed(args.seed + fold)
    model, base, partial = build_models(args, None, device)
    groups, trainable_names = trainable_param_groups(model, scope, args)
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    loader = DataLoader(
        Haze4KTrainSubset(split["train"], args.crop_size, args.seed + 1000 * fold),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    history = []
    fold_dir = output_dir / scope / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        set_train_modes(model, trainable_names)
        sums = {k: 0.0 for k in ("loss", "l_gt", "anchor", "hinge", "cvar", "direction", "residual")}
        sample_count = 0
        for input_img, label_img in loader:
            input_img = input_img.to(device, non_blocking=True)
            label_img = label_img.to(device, non_blocking=True)
            with torch.no_grad():
                a0 = base(input_img)[2].detach()
            pred = model(input_img)[2]
            e0 = a0 - label_img
            residual = pred - a0
            a0_mse = per_image_mse(a0, label_img)
            pred_mse = per_image_mse(pred, label_img)
            l_gt = charbonnier(pred, label_img)
            easy = a0_mse <= torch.median(a0_mse)
            anchor = F.l1_loss(pred[easy], a0[easy]) if bool(easy.any()) else F.l1_loss(pred, a0)
            hinge = torch.relu(pred_mse - a0_mse + args.hinge_margin).mean()
            bad_ratio = torch.relu(pred_mse / (a0_mse + 1e-12) - 1.0)
            k = max(1, int(math.ceil(args.cvar_frac * bad_ratio.numel())))
            cvar = torch.topk(bad_ratio, k=k).values.mean()
            direction = torch.relu((e0 * residual).flatten(1).mean(dim=1)).mean()
            residual_norm = residual.abs().mean()
            loss = (
                l_gt
                + args.lambda_anchor * anchor
                + args.lambda_hinge * hinge
                + args.lambda_cvar * cvar
                + args.lambda_direction * direction
                + args.lambda_residual * residual_norm
            )
            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], args.grad_clip_norm)
            optimizer.step()
            n = input_img.shape[0]
            sample_count += n
            sums["loss"] += float(loss.item()) * n
            sums["l_gt"] += float(l_gt.item()) * n
            sums["anchor"] += float(anchor.item()) * n
            sums["hinge"] += float(hinge.item()) * n
            sums["cvar"] += float(cvar.item()) * n
            sums["direction"] += float(direction.item()) * n
            sums["residual"] += float(residual_norm.item()) * n
        row = {"scope": scope, "fold": fold, "epoch": epoch}
        row.update({k: v / sample_count for k, v in sums.items()})
        history.append(row)
    checkpoint = fold_dir / "Final.pkl"
    torch.save({"model": model.state_dict(), "epoch": args.epochs, "scope": scope}, checkpoint)
    train_rows = evaluate_records(model, base, split["train"], fold, "train32", device)
    val_rows = evaluate_records(model, base, split["val"], fold, "oof32", device)
    return {
        "fold": fold,
        "scope": scope,
        "checkpoint": str(checkpoint),
        "partial_load": partial,
        "trainable_names": trainable_names,
        "trainable_param_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "history": history,
        "train_rows": train_rows,
        "val_rows": val_rows,
    }


def run_preflight(args: argparse.Namespace) -> None:
    evidence = Path(args.evidence_root)
    evidence.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    route, base, partial = build_models(args, None, device)
    reports = {}
    for scope in ("frozen_probe", "tier_a_partial", "tier_b_partial"):
        groups, names = trainable_param_groups(route, scope, args)
        reports[scope] = {
            "trainable_param_count": sum(p.numel() for g in groups for p in g["params"]),
            "trainable_param_names": names,
            "lr_groups": [{"name": g["name"], "lr": g["lr"], "param_count": sum(p.numel() for p in g["params"])} for g in groups],
        }
    synthetic = torch.rand(1, 3, args.synthetic_size, args.synthetic_size, device=device)
    with torch.no_grad():
        official = base(synthetic)
        routed = route(synthetic)
    max_diff = max(float((a - b).abs().max().item()) for a, b in zip(official, routed))
    finite = all(bool(torch.isfinite(x).all().item()) for x in routed)
    model_path = Path(args.repo_root) / "Dehazing" / "ITS" / "models" / "V300A0AnchoredConvIR.py"
    forbidden_tokens = ("cv2", "PIL", "Image.", "numpy", "np.", "skimage", "imwrite", "postprocess")
    text = model_path.read_text(encoding="utf-8")
    forbidden = [tok for tok in forbidden_tokens if tok in text]
    report = {
        "route": "haze4k_v3_0_a0_anchored_partial_unfreeze",
        "stage": "STAGE0_PREFLIGHT",
        "decision": "STAGE0_PASS_AUTHORIZE_CANARY32" if max_diff <= 1e-7 and finite and not forbidden else "STAGE0_FAIL",
        "gate_pass": max_diff <= 1e-7 and finite and not forbidden,
        "official_checkpoint": args.checkpoint,
        "official_checkpoint_sha256": sha256(Path(args.checkpoint)),
        "partial_load": partial,
        "identity_max_abs_vs_A0": max_diff,
        "finite_outputs": finite,
        "forbidden_symbol_hits": len(forbidden),
        "forbidden_symbol_report": {"Dehazing/ITS/models/V300A0AnchoredConvIR.py": forbidden},
        "scope_reports": reports,
        "locked_test_touched": False,
        "canary32_authorized": max_diff <= 1e-7 and finite and not forbidden,
        "canary80_authorized": False,
        "locked_test_authorized": False,
    }
    write_json(evidence / "v300_stage0_preflight.json", report)
    write_json(
        evidence / "v300_stage0_closeout.json",
        {
            "decision": report["decision"],
            "gate_pass": report["gate_pass"],
            "status": "COMPLETED_GATE_PASS" if report["gate_pass"] else "PREFLIGHT_FAILED_ENGINEERING",
            "canary32_authorized": report["canary32_authorized"],
            "canary80_authorized": False,
            "locked_test_authorized": False,
            "locked_test_touched": False,
        },
    )
    print(json.dumps({"decision": report["decision"], "gate_pass": report["gate_pass"]}, sort_keys=True))


def run_canary(args: argparse.Namespace) -> None:
    evidence = Path(args.evidence_root)
    evidence.mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records, skipped = list_haze4k_records(Path(args.data_dir), "train")
    splits = split_from_v241(Path(args.split_source_summary), records)
    all_val = []
    all_train = []
    all_history = []
    fold_summaries = []
    fold_details = []
    for split in splits:
        result = train_fold(args, split, args.scope, device, output_dir)
        all_val.extend(result["val_rows"])
        all_train.extend(result["train_rows"])
        all_history.extend(result["history"])
        summary = summarize_rows(result["val_rows"])
        summary["fold"] = result["fold"]
        summary["gate_pass"] = fold_gate(summary, args)
        summary["checkpoint"] = result["checkpoint"]
        fold_summaries.append(summary)
        fold_details.append(
            {
                "fold": result["fold"],
                "checkpoint": result["checkpoint"],
                "trainable_param_count": result["trainable_param_count"],
                "trainable_param_names": result["trainable_names"],
            }
        )
        print(
            f"scope={args.scope} fold={result['fold']} mean={summary['mean_delta']:.4f} "
            f"hard={summary['hard_delta']:.4f} easy={summary['easy_delta']:.4f} "
            f"p05={summary['p05_delta']:.4f} pass={summary['gate_pass']}",
            flush=True,
        )
    global_summary = summarize_rows(all_val)
    global_gate = fold_gate(global_summary, args)
    fold_pass_count = sum(1 for r in fold_summaries if r["gate_pass"])
    geom = geometry_summary(all_val)
    severe_direction_ok = (geom["severe_direction_bad_count"] == 0)
    gate_pass = global_gate and fold_pass_count >= args.gate_min_fold_pass and severe_direction_ok
    decision = (
        f"{args.scope.upper()}_CANARY32_PASS_AUTHORIZE_CANARY80_DESIGN"
        if gate_pass
        else f"{args.scope.upper()}_CANARY32_GATE_FAIL_LOCK_CANARY80_LOCKED_TEST"
    )
    train_global = summarize_rows(all_train)
    summary = {
        "route": "haze4k_v3_0_a0_anchored_partial_unfreeze",
        "stage": "CANARY32_OOF",
        "scope": args.scope,
        "decision": decision,
        "gate_pass": gate_pass,
        "global_gate_pass": global_gate,
        "fold_pass_count": fold_pass_count,
        "global_summary": global_summary,
        "train32_global_summary": train_global,
        "geometry": geom,
        "oracle_upper_bound": oracle_summary(all_val),
        "shrink_curve": shrink_curve(all_val, args),
        "fold_summaries": fold_summaries,
        "fold_details": fold_details,
        "train_derived_only": True,
        "split_source_summary": args.split_source_summary,
        "skipped_unmatched_train_images": skipped,
        "canary80_authorized": bool(gate_pass),
        "locked_test_authorized": False,
        "locked_test_touched": False,
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    prefix = f"v300_{args.scope}_canary32"
    write_json(evidence / f"{prefix}_summary.json", summary)
    write_json(
        evidence / f"{prefix}_closeout.json",
        {
            "decision": decision,
            "gate_pass": gate_pass,
            "status": "COMPLETED_GATE_PASS" if gate_pass else "COMPLETED_GATE_FAIL",
            "canary80_authorized": bool(gate_pass),
            "locked_test_authorized": False,
            "locked_test_touched": False,
        },
    )
    write_csv(evidence / f"{prefix}_folds.csv", fold_summaries, list(fold_summaries[0].keys()))
    write_csv(evidence / f"{prefix}_epoch_history.csv", all_history, list(all_history[0].keys()))
    write_csv(
        evidence / f"{prefix}_per_image.csv",
        all_val,
        [
            "fold",
            "split",
            "image",
            "a0_mse",
            "pred_mse",
            "a0_psnr",
            "pred_psnr",
            "delta_psnr",
            "residual_energy",
            "alignment_dot",
            "alignment_cos",
            "alpha_safe_upper",
            "direction_bad",
            "overshoot_bad",
        ],
    )
    print(json.dumps({"decision": decision, "gate_pass": gate_pass}, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preflight", "canary32"), required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--evidence-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-source-summary", required=True)
    parser.add_argument("--scope", choices=("frozen_probe", "tier_a_partial", "tier_b_partial"), default="frozen_probe")
    parser.add_argument("--seed", type=int, default=30032)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--synthetic-size", type=int, default=256)
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--lr-new", type=float, default=1e-4)
    parser.add_argument("--lr-backbone", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-anchor", type=float, default=0.2)
    parser.add_argument("--lambda-hinge", type=float, default=2.0)
    parser.add_argument("--lambda-cvar", type=float, default=1.0)
    parser.add_argument("--lambda-direction", type=float, default=5.0)
    parser.add_argument("--lambda-residual", type=float, default=0.05)
    parser.add_argument("--hinge-margin", type=float, default=0.0)
    parser.add_argument("--cvar-frac", type=float, default=0.2)
    parser.add_argument("--grad-clip-norm", type=float, default=0.01)
    parser.add_argument("--gate-mean-delta", type=float, default=0.15)
    parser.add_argument("--gate-hard-delta", type=float, default=0.30)
    parser.add_argument("--gate-easy-delta", type=float, default=0.0)
    parser.add_argument("--gate-p05-delta", type=float, default=-0.01)
    parser.add_argument("--gate-cvar5-delta", type=float, default=-0.02)
    parser.add_argument("--gate-min-fold-pass", type=int, default=4)
    args = parser.parse_args()
    if args.mode == "preflight":
        run_preflight(args)
    else:
        run_canary(args)


if __name__ == "__main__":
    main()
