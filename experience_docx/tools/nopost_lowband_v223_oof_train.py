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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

TOOL_PATH = Path(__file__).resolve()
REPO_ROOT = TOOL_PATH.parents[2]
ITS_ROOT = REPO_ROOT / "Dehazing" / "ITS"
for path in (str(ITS_ROOT), str(REPO_ROOT), os.getcwd()):
    if path not in sys.path:
        sys.path.insert(0, path)

from experience_docx.tools import nopost_lowband_v222_n3_microfit as v222  # noqa: E402
from models.NoPostGatedLowbandConvIR import load_haze4k_partial  # noqa: E402


SEVERE = -0.20
STRONG_REG = -0.05


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    v222.write_csv(path, rows)


def mean(values: list[float]) -> float:
    return v222.mean(values)


def percentile(values: list[float], pct: float) -> float:
    return v222.percentile(values, pct)


def cvar(values: list[float], pct: float = 5.0) -> float:
    return v222.cvar(values, pct)


@dataclass
class FoldSample:
    name: str
    fold: int
    input_path: Path
    label_path: Path
    risk: dict[str, float]


def parse_folds(text: str) -> list[int]:
    folds = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not folds:
        raise ValueError("At least one fold is required.")
    return folds


def load_split_samples(args: argparse.Namespace) -> list[FoldSample]:
    if not args.split_csv.is_file():
        raise FileNotFoundError(args.split_csv)
    input_dir, gt_dir = v222.train_dirs(args.data_dir)
    risk_by_name = v222.risk_labels_from_v221(args.v221_metrics_csv, args.v221_variant)
    samples: list[FoldSample] = []
    for row in v222.read_csv(args.split_csv):
        name = row["name"]
        src = input_dir / name
        if not src.is_file() or src.suffix.lower() not in v222.IMG_EXTENSIONS:
            continue
        samples.append(
            FoldSample(
                name=name,
                fold=int(row["oof_fold"]),
                input_path=src,
                label_path=v222.label_path(gt_dir, name),
                risk=risk_by_name.get(name, {}),
            )
        )
    if not samples:
        raise RuntimeError("No usable split samples were found.")
    return samples


def select_samples(samples: list[FoldSample], fold: int, args: argparse.Namespace) -> tuple[list[FoldSample], list[FoldSample]]:
    train = [sample for sample in samples if sample.fold != fold]
    eval_rows = [sample for sample in samples if sample.fold == fold]
    rng = random.Random(args.seed + fold * 1009)
    train = sorted(train, key=lambda sample: (sample.name, sample.fold))
    eval_rows = sorted(eval_rows, key=lambda sample: (sample.name, sample.fold))
    rng.shuffle(train)
    rng.shuffle(eval_rows)
    train = train[: args.train_samples_per_fold]
    eval_rows = eval_rows[: args.eval_samples_per_fold]
    if not train or not eval_rows:
        raise RuntimeError(f"Fold {fold} has empty train or eval selection.")
    return train, eval_rows


def make_args_for_v222(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        data_dir=args.data_dir,
        checkpoint=args.checkpoint,
        out_dir=args.out_dir,
        split_csv=args.split_csv,
        v221_metrics_csv=args.v221_metrics_csv,
        v221_variant=args.v221_variant,
        hidden_channels=args.hidden_channels,
        mid_grid=args.mid_grid,
        final_grid=args.final_grid,
        crop_size=args.crop_size,
        risk_gamma=args.risk_gamma,
        risk_bias=args.risk_bias,
        identity_tol=args.identity_tol,
        train_scope="adapter_only",
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        risk_loss_weight=args.risk_loss_weight,
        gate_mean_weight=args.gate_mean_weight,
        action_l1_weight=args.action_l1_weight,
        strong_reference_psnr=args.strong_reference_psnr,
        seed=args.seed,
    )


def to_v222_samples(samples: list[FoldSample]) -> list[v222.Sample]:
    return [
        v222.Sample(
            name=sample.name,
            input_path=sample.input_path,
            label_path=sample.label_path,
            risk=sample.risk,
        )
        for sample in samples
    ]


def train_fold(
    args: argparse.Namespace,
    fold: int,
    train_samples: list[FoldSample],
    out_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    base_args = make_args_for_v222(args)
    a0, model, partial = v222.load_a0_and_route(base_args, device)
    scope = v222.set_train_scope(model, "adapter_only")
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    rng = random.Random(args.seed + fold * 7919)
    rows = []
    start = time.time()
    samples = to_v222_samples(train_samples)
    for epoch in range(1, args.epochs + 1):
        order = list(range(len(samples)))
        rng.shuffle(order)
        losses: list[float] = []
        dpsnrs: list[float] = []
        probs: list[float] = []
        action_rms: list[float] = []
        for pos, idx in enumerate(order):
            sample = samples[idx]
            x = v222.image_tensor(sample.input_path, device)
            y = v222.image_tensor(sample.label_path, device)
            x, y = v222.crop_pair(x, y, args.crop_size, args.seed + fold * 1000000 + epoch * 10000 + pos)
            x, h, w = v222.pad_to(x, 32)
            y = y[:, :, :h, :w]
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                a0_pred = a0(x)[2][:, :, :h, :w]
                a0_psnr = v222.tensor_psnr(a0_pred, y)
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
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.grad_clip_norm)
            optimizer.step()
            with torch.no_grad():
                dpsnrs.append(v222.tensor_psnr(pred_full, y) - a0_psnr)
                losses.append(float(loss.detach().cpu()))
                probs.append(float(torch.cat([mid_prob.flatten(), final_prob.flatten()]).mean().detach().cpu()))
                action_rms.append(float(torch.sqrt(action_reg.detach().float()).cpu()))
        row = {
            "fold": fold,
            "epoch": epoch,
            "train_count": len(samples),
            "loss": mean(losses),
            "train_mean_dPSNR_vs_A0": mean(dpsnrs),
            "train_p05_dPSNR_vs_A0": percentile(dpsnrs, 5),
            "train_CVaR5_dPSNR_vs_A0": cvar(dpsnrs),
            "train_severe_rate": mean([1.0 if value <= SEVERE else 0.0 for value in dpsnrs]),
            "mean_unsafe_probability": mean(probs),
            "mean_action_rms_proxy": mean(action_rms),
        }
        rows.append(row)
        print(
            "V223_FOLD_TRAIN "
            f"fold={fold} epoch={epoch}/{args.epochs} "
            f"loss={row['loss']:.6f} "
            f"dPSNR={row['train_mean_dPSNR_vs_A0']:.4f} "
            f"p05={row['train_p05_dPSNR_vs_A0']:.4f} "
            f"unsafe={row['mean_unsafe_probability']:.4f}",
            flush=True,
        )
    fold_dir = out_dir / f"fold{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    ckpt = fold_dir / f"v223_fold{fold}_adapter_only_Final.pkl"
    torch.save({"model": model.state_dict()}, ckpt)
    write_csv(fold_dir / f"v223_fold{fold}_train_history.csv", rows)
    return {
        "fold": fold,
        "checkpoint": str(ckpt),
        "train_count": len(samples),
        "epochs": args.epochs,
        "partial_load": partial,
        "scope": scope,
        "history_last": rows[-1],
        "train_seconds": time.time() - start,
    }


def eval_fold(
    args: argparse.Namespace,
    fold: int,
    checkpoint: Path,
    eval_samples: list[FoldSample],
    out_dir: Path,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    base_args = make_args_for_v222(args)
    a0, model, partial = v222.load_a0_and_route(base_args, device)
    state = v222.load_state(checkpoint, device)
    model.load_state_dict(state)
    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for sample in to_v222_samples(eval_samples):
            x = v222.image_tensor(sample.input_path, device)
            y = v222.image_tensor(sample.label_path, device)
            x, y = v222.crop_pair(x, y, args.crop_size, args.seed + fold * 1000000 + v222.idx_hash(sample.name))
            x, h, w = v222.pad_to(x, 32)
            y = y[:, :, :h, :w]
            a0_pred = a0(x)[2][:, :, :h, :w]
            cand_pred = model(x)[2][:, :, :h, :w]
            tensors = model.nopost_gated_lowband_policy.last_tensors
            a0_psnr = v222.tensor_psnr(a0_pred, y)
            cand_psnr = v222.tensor_psnr(cand_pred, y)
            dpsnr = cand_psnr - a0_psnr
            rows.append(
                {
                    "fold": fold,
                    "name": sample.name,
                    "A0_PSNR": a0_psnr,
                    "candidate_PSNR": cand_psnr,
                    "dPSNR": dpsnr,
                    "severe": int(dpsnr <= SEVERE),
                    "strong_reference": int(a0_psnr >= args.strong_reference_psnr),
                    "strong_reference_regression": int(a0_psnr >= args.strong_reference_psnr and dpsnr <= STRONG_REG),
                    "unsafe_action_probability": sample.risk.get("unsafe_action_probability", float("nan")),
                    "unsafe_action_label": sample.risk.get("unsafe_action_label", float("nan")),
                    "mid_unsafe_prob": float(tensors["mid_unsafe_prob"].mean().cpu()),
                    "final_unsafe_prob": float(tensors["final_unsafe_prob"].mean().cpu()),
                    "mid_delta_rms": float(torch.sqrt(torch.mean(tensors["mid_delta"].float() ** 2)).cpu()),
                    "final_delta_rms": float(torch.sqrt(torch.mean(tensors["final_delta"].float() ** 2)).cpu()),
                }
            )
    fold_dir = out_dir / f"fold{fold}"
    write_csv(fold_dir / f"v223_fold{fold}_per_image_eval.csv", rows)
    dpsnr = [float(row["dPSNR"]) for row in rows]
    sorted_rows = sorted(rows, key=lambda row: row["A0_PSNR"])
    hard = sorted_rows[: max(1, len(sorted_rows) // 4)]
    easy = sorted_rows[-max(1, len(sorted_rows) // 4) :]
    strong = [row for row in rows if row["strong_reference"]]
    summary = {
        "fold": fold,
        "eval_count": len(rows),
        "partial_load_loaded_count": partial["loaded_count"],
        "mean_dPSNR": mean(dpsnr),
        "hard_bottom25_dPSNR": mean([float(row["dPSNR"]) for row in hard]),
        "easy_top25_dPSNR": mean([float(row["dPSNR"]) for row in easy]),
        "positive_ratio": mean([1.0 if value > 0 else 0.0 for value in dpsnr]),
        "p05_dPSNR": percentile(dpsnr, 5),
        "CVaR5_dPSNR": cvar(dpsnr),
        "severe_rate": mean([1.0 if value <= SEVERE else 0.0 for value in dpsnr]),
        "strong_reference_count": len(strong),
        "strong_reference_regression_rate": mean([float(row["strong_reference_regression"]) for row in strong]) if strong else 0.0,
        "mean_mid_unsafe_prob": mean([float(row["mid_unsafe_prob"]) for row in rows]),
        "mean_final_unsafe_prob": mean([float(row["final_unsafe_prob"]) for row in rows]),
        "mean_mid_delta_rms": mean([float(row["mid_delta_rms"]) for row in rows]),
        "mean_final_delta_rms": mean([float(row["final_delta_rms"]) for row in rows]),
        "locked_test_touched": False,
    }
    write_json(fold_dir / f"v223_fold{fold}_eval_summary.json", summary)
    return summary, rows


def aggregate_summary(fold_summaries: list[dict[str, Any]], rows: list[dict[str, Any]], fold_count: int) -> dict[str, Any]:
    dpsnr = [float(row["dPSNR"]) for row in rows]
    sorted_rows = sorted(rows, key=lambda row: row["A0_PSNR"])
    hard = sorted_rows[: max(1, len(sorted_rows) // 4)]
    easy = sorted_rows[-max(1, len(sorted_rows) // 4) :]
    strong = [row for row in rows if row["strong_reference"]]
    fold_tail_pass = sum(1 for row in fold_summaries if float(row["p05_dPSNR"]) >= -0.50 and float(row["severe_rate"]) <= 0.08)
    return {
        "fold_count": fold_count,
        "count": len(rows),
        "mean_dPSNR": mean(dpsnr),
        "hard_bottom25_dPSNR": mean([float(row["dPSNR"]) for row in hard]),
        "easy_top25_dPSNR": mean([float(row["dPSNR"]) for row in easy]),
        "positive_ratio": mean([1.0 if value > 0 else 0.0 for value in dpsnr]),
        "p05_dPSNR": percentile(dpsnr, 5),
        "CVaR5_dPSNR": cvar(dpsnr),
        "severe_rate": mean([1.0 if value <= SEVERE else 0.0 for value in dpsnr]),
        "strong_reference_count": len(strong),
        "strong_reference_regression_rate": mean([float(row["strong_reference_regression"]) for row in strong]) if strong else 0.0,
        "fold_tail_pass": fold_tail_pass,
        "mean_mid_unsafe_prob": mean([float(row["mid_unsafe_prob"]) for row in rows]),
        "mean_final_unsafe_prob": mean([float(row["final_unsafe_prob"]) for row in rows]),
        "mean_mid_delta_rms": mean([float(row["mid_delta_rms"]) for row in rows]),
        "mean_final_delta_rms": mean([float(row["final_delta_rms"]) for row in rows]),
        "locked_test_touched": False,
    }


def gate_summary(summary: dict[str, Any]) -> dict[str, Any]:
    action_rms = float(summary["mean_mid_delta_rms"]) + float(summary["mean_final_delta_rms"])
    checks = {
        "mean_positive": float(summary["mean_dPSNR"]) >= 0.05,
        "hard_positive": float(summary["hard_bottom25_dPSNR"]) >= 0.05,
        "easy_preserved": float(summary["easy_top25_dPSNR"]) >= -0.05,
        "positive_ratio": float(summary["positive_ratio"]) >= 0.55,
        "p05_safe": float(summary["p05_dPSNR"]) >= -0.50,
        "cvar_safe": float(summary["CVaR5_dPSNR"]) >= -0.80,
        "severe_safe": float(summary["severe_rate"]) <= 0.08,
        "strong_safe": float(summary["strong_reference_regression_rate"]) <= 0.20,
        "fold_tail_pass": int(summary["fold_tail_pass"]) >= max(1, math.ceil(float(summary["fold_count"]) * 2.0 / 3.0)),
        "not_near_identity": action_rms >= 1e-5,
        "gate_not_degenerate": 0.02 <= float(summary["mean_mid_unsafe_prob"]) <= 0.98
        and 0.02 <= float(summary["mean_final_unsafe_prob"]) <= 0.98,
        "locked_test_untouched": not bool(summary["locked_test_touched"]),
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "locked_test_allowed": False,
    }


def preflight(args: argparse.Namespace, out_dir: Path, device: torch.device, samples: list[FoldSample]) -> dict[str, Any]:
    base_args = make_args_for_v222(args)
    state = v222.load_state(args.checkpoint, "cpu")
    model = v222.make_model(base_args, device)
    partial = load_haze4k_partial(model, state)
    a0 = v222.build_a0_net("base", "Haze4K", "original").to(device)
    a0.load_state_dict(state)
    a0.eval()
    for param in a0.parameters():
        param.requires_grad_(False)
    sample = samples[0]
    x = v222.image_tensor(sample.input_path, device)
    x, h, w = v222.pad_to(x, 32)
    with torch.no_grad():
        a0_pred = a0(x)[2][:, :, :h, :w]
        route_pred = model(x)[2][:, :, :h, :w]
    max_abs = float((a0_pred - route_pred).abs().max().cpu())
    payload = {
        "branch": os.popen("git branch --show-current").read().strip(),
        "commit": os.popen("git rev-parse --short HEAD").read().strip(),
        "python": sys.executable,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": v222.sha256_file(args.checkpoint),
        "data_dir": str(args.data_dir),
        "split_csv": str(args.split_csv),
        "sample_count": len(samples),
        "folds": parse_folds(args.folds),
        "partial_load": partial,
        "zero_init_max_abs_vs_A0": max_abs,
        "forward_finite": bool(torch.isfinite(route_pred).all().item()),
        "locked_test_touched": False,
        "pass": max_abs <= args.identity_tol and bool(torch.isfinite(route_pred).all().item()),
    }
    write_json(out_dir / "v223_oof_preflight.json", payload)
    return payload


def update_readme(out_dir: Path, closeout: dict[str, Any]) -> None:
    lines = [
        "# Haze4K v2.23 NoPost OOF Gated Lowband Train Evidence",
        "",
        f"Status: {closeout['decision']}",
        "",
        "This route is a train-derived OOF screen for the v2.22 gated-lowband module.",
        "It trains only `nopost_gated_lowband_policy.*` and does not touch locked Haze4K test data.",
        "",
        "## Closeout",
        "",
        f"- decision: `{closeout['decision']}`",
        f"- locked test touched: `{str(closeout['locked_test_touched']).lower()}`",
        f"- completed folds: `{','.join(str(item) for item in closeout.get('completed_folds', []))}`",
        "",
    ]
    summary = closeout.get("oof_summary")
    if summary:
        lines.extend(
            [
                "## OOF Summary",
                "",
                f"- mean dPSNR: `{summary['mean_dPSNR']}`",
                f"- hard bottom25 dPSNR: `{summary['hard_bottom25_dPSNR']}`",
                f"- easy top25 dPSNR: `{summary['easy_top25_dPSNR']}`",
                f"- p05 dPSNR: `{summary['p05_dPSNR']}`",
                f"- CVaR5 dPSNR: `{summary['CVaR5_dPSNR']}`",
                f"- severe rate: `{summary['severe_rate']}`",
                f"- strong-reference regression rate: `{summary['strong_reference_regression_rate']}`",
                f"- fold tail pass: `{summary['fold_tail_pass']}/{summary['fold_count']}`",
                f"- mean mid/final unsafe probability: `{summary['mean_mid_unsafe_prob']}` / `{summary['mean_final_unsafe_prob']}`",
                f"- mean mid/final delta RMS: `{summary['mean_mid_delta_rms']}` / `{summary['mean_final_delta_rms']}`",
                "",
            ]
        )
    out_dir.joinpath("README.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--split-csv", type=Path, required=True)
    ap.add_argument("--v221-metrics-csv", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--folds", default="0,1,2")
    ap.add_argument("--train-samples-per-fold", type=int, default=384)
    ap.add_argument("--eval-samples-per-fold", type=int, default=160)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--hidden-channels", type=int, default=32)
    ap.add_argument("--mid-grid", type=int, default=8)
    ap.add_argument("--final-grid", type=int, default=16)
    ap.add_argument("--crop-size", type=int, default=256)
    ap.add_argument("--risk-gamma", type=float, default=0.5)
    ap.add_argument("--risk-bias", type=float, default=-1.5)
    ap.add_argument("--v221-variant", default="V221_risk_temperature_gamma0p50")
    ap.add_argument("--identity-tol", type=float, default=1e-6)
    ap.add_argument("--learning-rate", type=float, default=8e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--grad-clip-norm", type=float, default=0.001)
    ap.add_argument("--risk-loss-weight", type=float, default=0.05)
    ap.add_argument("--gate-mean-weight", type=float, default=0.0005)
    ap.add_argument("--action-l1-weight", type=float, default=0.0001)
    ap.add_argument("--strong-reference-psnr", type=float, default=27.0)
    ap.add_argument("--seed", type=int, default=223)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    status = args.out_dir / "status.txt"
    with status.open("a") as handle:
        handle.write(f"v223_oof_start {time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples = load_split_samples(args)
    pre = preflight(args, args.out_dir, device, samples)
    if not pre["pass"]:
        closeout = {
            "decision": "V223_OOF_PREFLIGHT_FAILED_ENGINEERING",
            "preflight": pre,
            "completed_folds": [],
            "locked_test_touched": False,
        }
        write_json(args.out_dir / "v223_oof_closeout.json", closeout)
        update_readme(args.out_dir, closeout)
        raise SystemExit(2)

    fold_summaries: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    completed: list[int] = []
    for fold in parse_folds(args.folds):
        train_samples, eval_samples = select_samples(samples, fold, args)
        result = train_fold(args, fold, train_samples, args.out_dir, device)
        write_json(args.out_dir / f"fold{fold}" / f"v223_fold{fold}_train_summary.json", result)
        summary, rows = eval_fold(args, fold, Path(result["checkpoint"]), eval_samples, args.out_dir, device)
        summary["train_count"] = result["train_count"]
        summary["checkpoint"] = result["checkpoint"]
        fold_summaries.append(summary)
        all_rows.extend(rows)
        completed.append(fold)

    oof_summary = aggregate_summary(fold_summaries, all_rows, len(completed))
    gate = gate_summary(oof_summary)
    decision = (
        "V223_OOF_SCREEN_PASS_REVIEW_ONLY_NO_LOCKED_TEST"
        if gate["pass"]
        else "V223_OOF_SCREEN_GATE_FAIL_NORMAL_PAUSE_NO_LOCKED_TEST"
    )
    closeout = {
        "decision": decision,
        "preflight_pass": pre["pass"],
        "completed_folds": completed,
        "fold_summaries": fold_summaries,
        "oof_summary": oof_summary,
        "gate": gate,
        "locked_test_touched": False,
        "next_action": (
            "Write a larger OOF train-derived route review; locked test remains blocked."
            if gate["pass"]
            else "Pause normally. Do not expand training under this run id."
        ),
    }
    write_csv(args.out_dir / "v223_oof_fold_summary.csv", fold_summaries)
    write_json(args.out_dir / "v223_oof_summary.json", oof_summary)
    write_json(args.out_dir / "v223_oof_gate.json", gate)
    write_json(args.out_dir / "v223_oof_closeout.json", closeout)
    update_readme(args.out_dir, closeout)
    with status.open("a") as handle:
        handle.write(f"v223_oof_done {decision} {time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n")
    print("V223_OOF_SCREEN_OK", decision, flush=True)


if __name__ == "__main__":
    main()
