#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF


TOOL_PATH = Path(__file__).resolve()
REPO_ROOT = TOOL_PATH.parents[2]
ITS_ROOT = REPO_ROOT / "Dehazing" / "ITS"
for path in (str(ITS_ROOT), str(REPO_ROOT), os.getcwd()):
    if path not in sys.path:
        sys.path.insert(0, path)

from models.ConvIR import build_net as build_a0_net  # noqa: E402
from models.NoPostGatedLowbandConvIR import build_net as build_gated_net  # noqa: E402
from models.NoPostGatedLowbandConvIR import load_haze4k_partial  # noqa: E402


SEVERE = -0.20
STRONG_REG = -0.05
IMG_EXTENSIONS = (".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def idx_hash(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * pct / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def cvar(values: list[float], pct: float = 5.0) -> float:
    if not values:
        return float("nan")
    cutoff = percentile(values, pct)
    tail = [value for value in values if value <= cutoff]
    return mean(tail)


def first_dir(root: Path, names: tuple[str, ...]) -> Path:
    for name in names:
        path = root / name
        if path.is_dir():
            return path
    raise FileNotFoundError(f"None of {names} exists under {root}")


def train_dirs(data_dir: Path) -> tuple[Path, Path]:
    root = data_dir / "train"
    return first_dir(root, ("IN", "haze", "hazy")), first_dir(root, ("GT", "gt"))


def val_dirs(data_dir: Path) -> tuple[Path, Path]:
    root = data_dir / "test"
    return first_dir(root, ("IN", "haze", "hazy")), first_dir(root, ("GT", "gt"))


def label_path(gt_dir: Path, image_name: str) -> Path:
    stem, ext = os.path.splitext(image_name)
    candidates = [image_name]
    if "_" in stem:
        candidates.extend([f"{stem.split('_')[0]}{ext}", f"{stem.split('_')[0]}.png"])
    for candidate in candidates:
        path = gt_dir / candidate
        if path.is_file():
            return path
    raise FileNotFoundError(f"No GT match for {image_name} under {gt_dir}; tried {candidates}")


def image_tensor(path: Path, device: torch.device) -> torch.Tensor:
    return TF.to_tensor(Image.open(path).convert("RGB")).unsqueeze(0).to(device)


def pad_to(x: torch.Tensor, factor: int = 32) -> tuple[torch.Tensor, int, int]:
    h, w = x.shape[-2:]
    ph = ((h + factor) // factor) * factor - h if h % factor else 0
    pw = ((w + factor) // factor) * factor - w if w % factor else 0
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="reflect")
    return x, h, w


def crop_pair(x: torch.Tensor, y: torch.Tensor, crop_size: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    if crop_size <= 0:
        return x, y
    h, w = x.shape[-2:]
    if h <= crop_size or w <= crop_size:
        return x[:, :, : min(h, crop_size), : min(w, crop_size)], y[:, :, : min(h, crop_size), : min(w, crop_size)]
    rng = random.Random(seed)
    top = rng.randint(0, h - crop_size)
    left = rng.randint(0, w - crop_size)
    return (
        x[:, :, top : top + crop_size, left : left + crop_size],
        y[:, :, top : top + crop_size, left : left + crop_size],
    )


def load_state(path: Path, device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    return state


def tensor_psnr(pred: torch.Tensor, label: torch.Tensor) -> float:
    pred = pred.clamp(0.0, 1.0)
    mse = F.mse_loss(pred, label).item()
    if mse <= 0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def make_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    model = build_gated_net(
        "base",
        "Haze4K",
        "original",
        hidden_channels=args.hidden_channels,
        mid_grid=args.mid_grid,
        final_grid=args.final_grid,
        risk_gamma=args.risk_gamma,
        risk_bias=args.risk_bias,
    ).to(device)
    return model


def load_a0_and_route(args: argparse.Namespace, device: torch.device) -> tuple[torch.nn.Module, torch.nn.Module, dict[str, Any]]:
    state = load_state(args.checkpoint, "cpu")
    a0 = build_a0_net("base", "Haze4K", "original").to(device)
    a0.load_state_dict(state)
    a0.eval()
    for param in a0.parameters():
        param.requires_grad_(False)

    route = make_model(args, device)
    partial = load_haze4k_partial(route, state)
    return a0, route, partial


def set_train_scope(model: torch.nn.Module, scope: str) -> dict[str, Any]:
    if scope != "adapter_only":
        raise ValueError("v2.22 N3 microfit currently only permits adapter_only.")
    trainable_prefixes = ("nopost_gated_lowband_policy.",)
    trainable = []
    frozen = []
    for name, param in model.named_parameters():
        is_trainable = name.startswith(trainable_prefixes)
        param.requires_grad_(is_trainable)
        if is_trainable:
            trainable.append((name, param.numel()))
        else:
            frozen.append((name, param.numel()))
    model.train()
    for name, module in model.named_modules():
        if not name.startswith("nopost_gated_lowband_policy"):
            module.eval()
    return {
        "scope": scope,
        "trainable_prefixes": list(trainable_prefixes),
        "trainable_param_count": sum(n for _, n in trainable),
        "frozen_param_count": sum(n for _, n in frozen),
        "trainable_names": [name for name, _ in trainable],
    }


def stage_names(args: argparse.Namespace) -> list[str]:
    return [item.strip() for item in args.stages.split(",") if item.strip()]


def stage_sample_count(stage: str) -> int:
    if not stage.startswith("microfit"):
        raise ValueError(f"Unknown stage: {stage}")
    return int(stage.replace("microfit", ""))


def stage_epochs(args: argparse.Namespace, stage: str) -> int:
    if stage == "microfit16":
        return args.epochs16
    if stage == "microfit64":
        return args.epochs64
    if stage == "microfit256":
        return args.epochs256
    return args.epochs


def risk_labels_from_v221(path: Path | None, variant: str) -> dict[str, dict[str, float]]:
    if not path:
        return {}
    labels: dict[str, dict[str, float]] = {}
    for row in read_csv(path):
        if row.get("variant") != variant:
            continue
        name = row["name"]
        labels[name] = {
            "unsafe_action_probability": float(row.get("unsafe_action_probability", 0.0)),
            "unsafe_action_label": float(row.get("unsafe_action_label", 0.0)),
            "v221_risk_scale": float(row.get("risk_scale", 1.0)),
            "v221_candidate_dPSNR": float(row.get("dPSNR", 0.0)),
            "v221_raw_action_dPSNR": float(row.get("raw_action_dPSNR", 0.0)),
        }
    return labels


@dataclass
class Sample:
    name: str
    input_path: Path
    label_path: Path
    risk: dict[str, float]


def sample_rows(args: argparse.Namespace) -> list[str]:
    if args.split_csv and args.split_csv.is_file():
        rows = read_csv(args.split_csv)
        names = [row["name"] for row in rows]
    else:
        input_dir, _ = train_dirs(args.data_dir)
        names = sorted(
            path.name
            for path in input_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMG_EXTENSIONS
        )
    if args.max_images > 0:
        names = names[: args.max_images]
    return names


def build_samples(args: argparse.Namespace, risk_by_name: dict[str, dict[str, float]], count: int) -> list[Sample]:
    input_dir, gt_dir = train_dirs(args.data_dir)
    names = sample_rows(args)[:count]
    samples = []
    for name in names:
        samples.append(
            Sample(
                name=name,
                input_path=input_dir / name,
                label_path=label_path(gt_dir, name),
                risk=risk_by_name.get(name, {}),
            )
        )
    return samples


def train_one_stage(
    args: argparse.Namespace,
    samples: list[Sample],
    stage: str,
    out_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    a0, model, partial = load_a0_and_route(args, device)
    scope = set_train_scope(model, args.train_scope)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    rng = random.Random(args.seed + len(samples))
    history_rows = []
    start = time.time()
    epochs = stage_epochs(args, stage)
    for epoch in range(1, epochs + 1):
        order = list(range(len(samples)))
        rng.shuffle(order)
        losses = []
        psnr_deltas = []
        gate_probs = []
        action_rms = []
        for idx in order:
            sample = samples[idx]
            x = image_tensor(sample.input_path, device)
            y = image_tensor(sample.label_path, device)
            x, y = crop_pair(x, y, args.crop_size, args.seed + epoch * 100000 + idx)
            x, h, w = pad_to(x, 32)
            y = y[:, :, :h, :w]
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                a0_pred = a0(x)[2][:, :, :h, :w]
                a0_psnr = tensor_psnr(a0_pred, y)
            pred = model(x)
            pred_full = pred[2][:, :, :h, :w]
            label2 = F.interpolate(y, scale_factor=0.5, mode="bilinear", align_corners=False)
            label4 = F.interpolate(y, scale_factor=0.25, mode="bilinear", align_corners=False)
            loss_content = F.l1_loss(pred[0][:, :, : label4.shape[-2], : label4.shape[-1]], label4)
            loss_content = loss_content + F.l1_loss(pred[1][:, :, : label2.shape[-2], : label2.shape[-1]], label2)
            loss_content = loss_content + F.l1_loss(pred_full, y)
            tensors = model.nopost_gated_lowband_policy.last_tensors
            mid_prob = tensors["mid_unsafe_prob"]
            final_prob = tensors["final_unsafe_prob"]
            gate_reg = torch.mean(mid_prob) + torch.mean(final_prob)
            action_reg = tensors["mid_scaled_delta_grid"].abs().mean() + tensors["final_scaled_delta_grid"].abs().mean()
            risk_loss = pred_full.new_tensor(0.0)
            if "unsafe_action_label" in sample.risk:
                target = pred_full.new_tensor([[sample.risk["unsafe_action_label"]]])
                risk_loss = F.binary_cross_entropy_with_logits(tensors["mid_unsafe_logit"], target)
                risk_loss = risk_loss + F.binary_cross_entropy_with_logits(tensors["final_unsafe_logit"], target)
            loss = loss_content + args.risk_loss_weight * risk_loss
            loss = loss + args.gate_mean_weight * gate_reg + args.action_l1_weight * action_reg
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                args.grad_clip_norm,
            )
            optimizer.step()
            with torch.no_grad():
                cand_psnr = tensor_psnr(pred_full, y)
                psnr_deltas.append(cand_psnr - a0_psnr)
                losses.append(float(loss.detach().cpu()))
                gate_probs.append(float(torch.cat([mid_prob.flatten(), final_prob.flatten()]).mean().detach().cpu()))
                action_rms.append(float(torch.sqrt(action_reg.detach().float()).cpu()))
        row = {
            "stage": stage,
            "epoch": epoch,
            "sample_count": len(samples),
            "loss": mean(losses),
            "mean_dPSNR_vs_A0": mean(psnr_deltas),
            "p05_dPSNR_vs_A0": percentile(psnr_deltas, 5),
            "CVaR5_dPSNR_vs_A0": cvar(psnr_deltas),
            "severe_rate": mean([1.0 if value <= SEVERE else 0.0 for value in psnr_deltas]),
            "mean_unsafe_probability": mean(gate_probs),
            "mean_action_rms_proxy": mean(action_rms),
            "grad_clip_norm": args.grad_clip_norm,
        }
        history_rows.append(row)
        print(
            "N3_STAGE "
            f"{stage} epoch={epoch}/{epochs} "
            f"loss={row['loss']:.6f} "
            f"dPSNR={row['mean_dPSNR_vs_A0']:.4f} "
            f"p05={row['p05_dPSNR_vs_A0']:.4f} "
            f"unsafe={row['mean_unsafe_probability']:.4f}",
            flush=True,
        )
    stage_dir = out_dir / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = stage_dir / f"v222_{stage}_adapter_only_Final.pkl"
    torch.save({"model": model.state_dict()}, checkpoint)
    write_csv(stage_dir / f"v222_{stage}_train_history.csv", history_rows)
    return {
        "stage": stage,
        "sample_count": len(samples),
        "epochs": epochs,
        "checkpoint": str(checkpoint),
        "partial_load": partial,
        "scope": scope,
        "history_last": history_rows[-1],
        "train_seconds": time.time() - start,
    }


def eval_checkpoint(
    args: argparse.Namespace,
    checkpoint: Path,
    stage: str,
    samples: list[Sample],
    out_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    a0, model, partial = load_a0_and_route(args, device)
    state = load_state(checkpoint, device)
    model.load_state_dict(state)
    model.eval()
    rows = []
    with torch.no_grad():
        for sample in samples:
            x = image_tensor(sample.input_path, device)
            y = image_tensor(sample.label_path, device)
            x, y = crop_pair(x, y, args.crop_size, args.seed + idx_hash(sample.name))
            x, h, w = pad_to(x, 32)
            y = y[:, :, :h, :w]
            a0_pred = a0(x)[2][:, :, :h, :w]
            cand_pred = model(x)[2][:, :, :h, :w]
            tensors = model.nopost_gated_lowband_policy.last_tensors
            a0_psnr = tensor_psnr(a0_pred, y)
            cand_psnr = tensor_psnr(cand_pred, y)
            dpsnr = cand_psnr - a0_psnr
            rows.append(
                {
                    "stage": stage,
                    "name": sample.name,
                    "A0_PSNR": a0_psnr,
                    "candidate_PSNR": cand_psnr,
                    "dPSNR": dpsnr,
                    "severe": int(dpsnr <= SEVERE),
                    "strong_reference": int(a0_psnr >= args.strong_reference_psnr),
                    "strong_reference_regression": int(a0_psnr >= args.strong_reference_psnr and dpsnr <= STRONG_REG),
                    "unsafe_action_probability": sample.risk.get("unsafe_action_probability", float("nan")),
                    "unsafe_action_label": sample.risk.get("unsafe_action_label", float("nan")),
                    "v221_risk_scale": sample.risk.get("v221_risk_scale", float("nan")),
                    "mid_unsafe_prob": float(tensors["mid_unsafe_prob"].mean().cpu()),
                    "final_unsafe_prob": float(tensors["final_unsafe_prob"].mean().cpu()),
                    "mid_scale": float(tensors["mid_scale"].mean().cpu()),
                    "final_scale": float(tensors["final_scale"].mean().cpu()),
                    "mid_delta_rms": float(torch.sqrt(torch.mean(tensors["mid_delta"].float() ** 2)).cpu()),
                    "final_delta_rms": float(torch.sqrt(torch.mean(tensors["final_delta"].float() ** 2)).cpu()),
                }
            )
    write_csv(out_dir / stage / f"v222_{stage}_per_image_eval.csv", rows)
    dpsnr_values = [float(row["dPSNR"]) for row in rows]
    strong_rows = [row for row in rows if row["strong_reference"]]
    summary = {
        "stage": stage,
        "count": len(rows),
        "partial_load_loaded_count": partial["loaded_count"],
        "mean_dPSNR": mean(dpsnr_values),
        "hard_bottom25_dPSNR": mean([float(row["dPSNR"]) for row in sorted(rows, key=lambda r: r["A0_PSNR"])[: max(1, len(rows) // 4)]]),
        "easy_top25_dPSNR": mean([float(row["dPSNR"]) for row in sorted(rows, key=lambda r: r["A0_PSNR"])[-max(1, len(rows) // 4) :]]),
        "positive_ratio": mean([1.0 if value > 0 else 0.0 for value in dpsnr_values]),
        "p05_dPSNR": percentile(dpsnr_values, 5),
        "CVaR5_dPSNR": cvar(dpsnr_values),
        "severe_rate": mean([1.0 if value <= SEVERE else 0.0 for value in dpsnr_values]),
        "strong_reference_count": len(strong_rows),
        "strong_reference_regression_rate": mean([float(row["strong_reference_regression"]) for row in strong_rows]) if strong_rows else 0.0,
        "mean_mid_unsafe_prob": mean([float(row["mid_unsafe_prob"]) for row in rows]),
        "mean_final_unsafe_prob": mean([float(row["final_unsafe_prob"]) for row in rows]),
        "mean_mid_delta_rms": mean([float(row["mid_delta_rms"]) for row in rows]),
        "mean_final_delta_rms": mean([float(row["final_delta_rms"]) for row in rows]),
        "locked_test_touched": False,
    }
    return summary


def gate_stage(summary: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "finite_mean": math.isfinite(float(summary["mean_dPSNR"])),
        "no_severe_explosion": float(summary["severe_rate"]) <= 0.15,
        "p05_not_catastrophic": float(summary["p05_dPSNR"]) >= -1.0,
        "action_nonzero": (float(summary["mean_mid_delta_rms"]) + float(summary["mean_final_delta_rms"])) > 1e-7,
        "gate_not_degenerate": 0.02 <= float(summary["mean_mid_unsafe_prob"]) <= 0.98
        and 0.02 <= float(summary["mean_final_unsafe_prob"]) <= 0.98,
    }
    passed = all(checks.values())
    return {
        "stage": summary["stage"],
        "pass": passed,
        "checks": checks,
        "continue_allowed": passed,
        "locked_test_allowed": False,
    }


def preflight(args: argparse.Namespace, out_dir: Path, device: torch.device) -> dict[str, Any]:
    a0, model, partial = load_a0_and_route(args, device)
    scope = set_train_scope(model, args.train_scope)
    input_dir, gt_dir = train_dirs(args.data_dir)
    names = sorted(
        path.name
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMG_EXTENSIONS
    )
    sample_name = names[0]
    x = image_tensor(input_dir / sample_name, device)
    y = image_tensor(label_path(gt_dir, sample_name), device)
    x, y = crop_pair(x, y, args.crop_size, args.seed)
    x, h, w = pad_to(x, 32)
    y = y[:, :, :h, :w]
    with torch.no_grad():
        a0_pred = a0(x)[2][:, :, :h, :w]
        route_pred = model(x)[2][:, :, :h, :w]
        max_abs = float((a0_pred - route_pred).abs().max().cpu())
        one_loss = float(F.l1_loss(route_pred, y).cpu())
        stats = model.nopost_gated_lowband_policy.tensor_stats()
    payload = {
        "branch": os.popen("git branch --show-current").read().strip(),
        "commit": os.popen("git rev-parse --short HEAD").read().strip(),
        "python": sys.executable,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "data_dir": str(args.data_dir),
        "train_sample_count": len(names),
        "partial_load": partial,
        "scope": scope,
        "zero_init_max_abs_vs_A0": max_abs,
        "one_batch_forward_finite": bool(torch.isfinite(route_pred).all().item()),
        "one_batch_loss": one_loss,
        "initial_policy_stats": stats,
        "locked_test_touched": False,
        "pass": max_abs <= args.identity_tol and bool(torch.isfinite(route_pred).all().item()),
    }
    write_json(out_dir / "v222_n3_preflight.json", payload)
    return payload


def update_readme(out_dir: Path, closeout: dict[str, Any]) -> None:
    lines = [
        "# Haze4K v2.22 NoPost Gated Lowband N3 Microfit Evidence",
        "",
        f"Status: {closeout['decision']}",
        "",
        "This route trains only the new `nopost_gated_lowband_policy.*` modules.",
        "It uses train-derived microfit stages and does not touch locked Haze4K test data.",
        "",
        "## Closeout",
        "",
        f"- decision: `{closeout['decision']}`",
        f"- locked test touched: `{str(closeout['locked_test_touched']).lower()}`",
        f"- completed stages: `{','.join(closeout['completed_stages'])}`",
        "",
        "## Stage Summaries",
        "",
    ]
    for row in closeout.get("stage_summaries", []):
        lines.extend(
            [
                f"### {row['stage']}",
                "",
                f"- mean dPSNR: `{row['mean_dPSNR']}`",
                f"- hard bottom25 dPSNR: `{row['hard_bottom25_dPSNR']}`",
                f"- p05 dPSNR: `{row['p05_dPSNR']}`",
                f"- severe rate: `{row['severe_rate']}`",
                f"- mean mid/final unsafe prob: `{row['mean_mid_unsafe_prob']}` / `{row['mean_final_unsafe_prob']}`",
                f"- mean mid/final delta RMS: `{row['mean_mid_delta_rms']}` / `{row['mean_final_delta_rms']}`",
                "",
            ]
        )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--split-csv", type=Path, default=None)
    ap.add_argument("--v221-metrics-csv", type=Path, default=None)
    ap.add_argument("--v221-variant", default="V221_risk_temperature_gamma0p50")
    ap.add_argument("--stages", default="microfit16,microfit64,microfit256")
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--hidden-channels", type=int, default=32)
    ap.add_argument("--mid-grid", type=int, default=8)
    ap.add_argument("--final-grid", type=int, default=16)
    ap.add_argument("--crop-size", type=int, default=256)
    ap.add_argument("--risk-gamma", type=float, default=0.5)
    ap.add_argument("--risk-bias", type=float, default=-1.5)
    ap.add_argument("--identity-tol", type=float, default=1e-6)
    ap.add_argument("--train-scope", default="adapter_only")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--epochs16", type=int, default=4)
    ap.add_argument("--epochs64", type=int, default=5)
    ap.add_argument("--epochs256", type=int, default=6)
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--grad-clip-norm", type=float, default=0.001)
    ap.add_argument("--risk-loss-weight", type=float, default=0.05)
    ap.add_argument("--gate-mean-weight", type=float, default=0.0005)
    ap.add_argument("--action-l1-weight", type=float, default=0.0001)
    ap.add_argument("--eval-samples", type=int, default=128)
    ap.add_argument("--strong-reference-psnr", type=float, default=27.0)
    ap.add_argument("--seed", type=int, default=222)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    status_path = args.out_dir / "status.txt"
    with status_path.open("a") as status:
        status.write(f"v222_n3_start {time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n")

    pre = preflight(args, args.out_dir, device)
    if not pre["pass"]:
        closeout = {
            "decision": "V222_N3_PREFLIGHT_FAILED_ENGINEERING",
            "preflight": pre,
            "completed_stages": [],
            "stage_summaries": [],
            "locked_test_touched": False,
        }
        write_json(args.out_dir / "v222_n3_closeout.json", closeout)
        update_readme(args.out_dir, closeout)
        raise SystemExit(2)

    risk_by_name = risk_labels_from_v221(args.v221_metrics_csv, args.v221_variant)
    completed = []
    stage_summaries = []
    stage_gates = []
    for stage in stage_names(args):
        count = stage_sample_count(stage)
        samples = build_samples(args, risk_by_name, count)
        result = train_one_stage(args, samples, stage, args.out_dir, device)
        eval_samples = build_samples(args, risk_by_name, min(args.eval_samples, count))
        summary = eval_checkpoint(args, Path(result["checkpoint"]), stage, eval_samples, args.out_dir, device)
        summary.update({"epochs": result["epochs"], "sample_count": result["sample_count"], "checkpoint": result["checkpoint"]})
        gate = gate_stage(summary)
        write_json(args.out_dir / stage / f"v222_{stage}_train_summary.json", result)
        write_json(args.out_dir / stage / f"v222_{stage}_eval_summary.json", summary)
        write_json(args.out_dir / stage / f"v222_{stage}_gate.json", gate)
        completed.append(stage)
        stage_summaries.append(summary)
        stage_gates.append(gate)
        if not gate["continue_allowed"]:
            break

    all_passed = len(completed) == len(stage_names(args)) and all(gate["pass"] for gate in stage_gates)
    decision = (
        "V222_N3_MICROFIT_PASS_REVIEW_ONLY_NO_LOCKED_TEST"
        if all_passed
        else "V222_N3_MICROFIT_NORMAL_GATE_PAUSE_NO_LOCKED_TEST"
    )
    closeout = {
        "decision": decision,
        "preflight_pass": pre["pass"],
        "completed_stages": completed,
        "stage_summaries": stage_summaries,
        "stage_gates": stage_gates,
        "locked_test_touched": False,
        "next_action": (
            "Write a separate OOF train-derived route review; locked test remains blocked."
            if all_passed
            else "Pause normally and inspect the first failed N3 stage."
        ),
    }
    write_csv(args.out_dir / "v222_n3_stage_summary.csv", stage_summaries)
    write_json(args.out_dir / "v222_n3_closeout.json", closeout)
    update_readme(args.out_dir, closeout)
    with status_path.open("a") as status:
        status.write(f"v222_n3_done {decision} {time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n")
    print("V222_N3_MICROFIT_OK", decision, flush=True)


if __name__ == "__main__":
    main()
