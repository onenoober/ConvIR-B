#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from nopost_common import (
    build_nopost,
    image_tensor,
    infer_final,
    label_path,
    metric,
    names_for_scope,
    pad_to,
    partial_load_nopost,
    sha256,
    tensor_psnr,
    train_dirs,
    write_csv,
    write_json,
)


def grad_mean(x: torch.Tensor) -> float:
    dx = F.pad(x[:, :, :, 1:] - x[:, :, :, :-1], (0, 1, 0, 0))
    dy = F.pad(x[:, :, 1:, :] - x[:, :, :-1, :], (0, 0, 0, 1))
    return float(torch.sqrt(dx * dx + dy * dy + 1e-12).mean().detach().cpu())


def local_var_mean(x: torch.Tensor, kernel: int = 9) -> float:
    mean = F.avg_pool2d(x, kernel, stride=1, padding=kernel // 2)
    return float(F.avg_pool2d((x - mean) ** 2, kernel, stride=1, padding=kernel // 2).mean().detach().cpu())


def channel_norm(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean(x * x, dim=1, keepdim=True).clamp_min(1e-12))


def feature_stats(prefix: str, x: torch.Tensor) -> dict[str, float]:
    norm = channel_norm(x)
    low = F.avg_pool2d(x, 9, stride=1, padding=4)
    detail = x - low
    return {
        f"{prefix}_mean": float(x.mean().detach().cpu()),
        f"{prefix}_std": float(x.std(unbiased=False).detach().cpu()),
        f"{prefix}_norm_mean": float(norm.mean().detach().cpu()),
        f"{prefix}_norm_std": float(norm.std(unbiased=False).detach().cpu()),
        f"{prefix}_low_abs_mean": float(low.abs().mean().detach().cpu()),
        f"{prefix}_detail_abs_mean": float(detail.abs().mean().detach().cpu()),
        f"{prefix}_local_var": local_var_mean(x.mean(dim=1, keepdim=True)),
    }


def hazy_stats(hazy: torch.Tensor) -> dict[str, float]:
    brightness = hazy.mean(dim=1, keepdim=True)
    dark = hazy.min(dim=1, keepdim=True).values
    saturation = hazy.max(dim=1, keepdim=True).values - dark
    return {
        "hazy_brightness_mean": float(brightness.mean().detach().cpu()),
        "hazy_brightness_std": float(brightness.std(unbiased=False).detach().cpu()),
        "hazy_dark_mean": float(dark.mean().detach().cpu()),
        "hazy_saturation_mean": float(saturation.mean().detach().cpu()),
        "hazy_gradient_mean": grad_mean(brightness),
        "hazy_local_var": local_var_mean(brightness),
        "hazy_low_dark_mean": float(F.avg_pool2d(dark, 15, stride=1, padding=7).mean().detach().cpu()),
    }


def load_teacher(cache_dir: Path, scope: str, name: str, device: torch.device) -> torch.Tensor:
    p = cache_dir / scope / name
    if not p.is_file():
        # C12 cache uses train_core for all N1 labels by default.
        p = cache_dir / "train_core" / name
    if not p.is_file():
        raise FileNotFoundError(f"missing WD0375 teacher cache for {name}: {p}")
    return image_tensor(p, device)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--split-manifest", type=Path, required=True)
    ap.add_argument("--teacher-cache-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--scope", default="train_core")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--benefit-threshold", type=float, default=0.05)
    ap.add_argument("--risk-threshold", type=float, default=-0.20)
    ap.add_argument("--print-freq", type=int, default=50)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_nopost(device)
    load_report = partial_load_nopost(model, args.checkpoint, device)
    model.eval()

    input_dir, gt_dir = train_dirs(args.data_dir)
    selected = names_for_scope(args.split_manifest, args.scope, fold=args.fold, max_images=args.max_images)
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for idx, rec in enumerate(selected, 1):
            name = rec["name"]
            hazy = image_tensor(input_dir / name, device)
            label = image_tensor(label_path(gt_dir, name), device)
            teacher = load_teacher(args.teacher_cache_dir, "train_core", name, device)
            x, h, w, hp, wp = pad_to(hazy, 32)
            a0_pred = infer_final(model, x, h, w)
            teacher = teacher[:, :, :h, :w]
            a0_psnr, a0_ssim = metric(a0_pred, label, hp, wp)
            wd_psnr, wd_ssim = metric(teacher, label, hp, wp)
            hazy_psnr = tensor_psnr(hazy[:, :, :h, :w], label)

            features = model.extract_nopost_features(x)
            row: dict[str, Any] = {
                "name": name,
                "source_split": rec["split"],
                "oof_fold": (idx - 1) % 5,
                "A0_PSNR": a0_psnr,
                "A0_SSIM": a0_ssim,
                "WD0375_PSNR": wd_psnr,
                "WD0375_SSIM": wd_ssim,
                "WD0375_dPSNR": wd_psnr - a0_psnr,
                "WD0375_dSSIM": wd_ssim - a0_ssim,
                "hazy_PSNR": hazy_psnr,
                "benefit_label": int((wd_psnr - a0_psnr) >= args.benefit_threshold),
                "severe_risk_label": int((wd_psnr - a0_psnr) <= args.risk_threshold),
            }
            row.update(hazy_stats(hazy))
            row.update(feature_stats("final", features["final_feature"]))
            row.update(feature_stats("res1", features["res1"]))
            row.update(feature_stats("res2", features["res2"]))
            row.update(feature_stats("scm2", features["scm2"]))
            row.update(feature_stats("scm4", features["scm4"]))
            rows.append(row)
            if idx % args.print_freq == 0:
                print(f"N1_FEATURE_TABLE {idx}/{len(selected)}", flush=True)

    table_path = args.out_dir / "v213_n1_feature_rows_cloud_only.csv"
    write_csv(table_path, rows)
    counts = {
        "rows": len(rows),
        "benefit_positive": sum(int(r["benefit_label"]) for r in rows),
        "severe_risk_positive": sum(int(r["severe_risk_label"]) for r in rows),
    }
    manifest = {
        "route": "haze4k-v2-13-nopost-feature-gated-adapter",
        "scope": args.scope,
        "split_manifest": str(args.split_manifest),
        "split_manifest_sha256": sha256(args.split_manifest),
        "teacher_cache_dir": str(args.teacher_cache_dir),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "table_path": str(table_path),
        "counts": counts,
        "feature_groups": {
            "hazy": [k for k in rows[0] if k.startswith("hazy_")] if rows else [],
            "internal": [
                k
                for k in rows[0]
                if k.startswith(("final_", "res1_", "res2_", "scm2_", "scm4_"))
            ]
            if rows
            else [],
        },
        "labels": {
            "benefit_label": f"WD0375_dPSNR >= {args.benefit_threshold}",
            "severe_risk_label": f"WD0375_dPSNR <= {args.risk_threshold}",
        },
        "partial_load": load_report,
        "locked_test_touched": False,
    }
    write_json(args.out_dir / "v213_n1_feature_table_manifest.json", manifest)
    print("N1_FEATURE_TABLE_OK", json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()
