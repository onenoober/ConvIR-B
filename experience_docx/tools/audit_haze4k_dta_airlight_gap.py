#!/usr/bin/env python3
"""Compare DTA-v3 physical branch with GT airlight vs eval fallback airlight."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from pytorch_msssim import ssim

TOOL_PATH = Path(__file__).resolve()
REPO_ROOT = TOOL_PATH.parents[2]
ITS_ROOT = REPO_ROOT / "Dehazing" / "ITS"
for path in (str(ITS_ROOT), str(REPO_ROOT), os.getcwd()):
    if path not in sys.path:
        sys.path.insert(0, path)

from data import test_dataloader
from models.ConvIR import build_net


def load_state(path: str, device: torch.device):
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        return state["model"]
    return state


def unpack_batch(data):
    name = data[-1]
    if isinstance(name, str):
        name = [name]
    data = data[:-1]
    input_img, label_img = data[0], data[1]
    depth = data[2] if len(data) >= 3 else None
    airlight = None
    if len(data) >= 4 and torch.is_tensor(data[3]) and data[3].dim() < 3:
        airlight = data[3]
    elif len(data) >= 5:
        airlight = data[4]
    return input_img, label_img, depth, airlight, name


def pad_to_factor(x: torch.Tensor, factor: int = 32) -> tuple[torch.Tensor, int, int]:
    h, w = x.shape[-2:]
    padded_h = ((h + factor) // factor) * factor
    padded_w = ((w + factor) // factor) * factor
    padh = padded_h - h if h % factor != 0 else 0
    padw = padded_w - w if w % factor != 0 else 0
    if padh or padw:
        x = F.pad(x, (0, padw, 0, padh), "reflect")
    return x, h, w


def psnr(pred: torch.Tensor, label: torch.Tensor) -> float:
    mse = F.mse_loss(pred, label).clamp_min(1e-12)
    return float((10 * torch.log10(1 / mse)).cpu())


def ssim_value(pred: torch.Tensor, label: torch.Tensor, padded_h: int, padded_w: int) -> float:
    down_ratio = max(1, round(min(padded_h, padded_w) / 256))
    return float(
        ssim(
            F.adaptive_avg_pool2d(pred, (int(padded_h / down_ratio), int(padded_w / down_ratio))),
            F.adaptive_avg_pool2d(label, (int(padded_h / down_ratio), int(padded_w / down_ratio))),
            data_range=1,
            size_average=False,
        )
        .mean()
        .cpu()
    )


def airlight_tensor(input_img: torch.Tensor, airlight: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
    fallback = F.adaptive_max_pool2d(input_img.clamp(0.0, 1.0), 1)
    if airlight is None:
        gt = torch.ones_like(fallback)
    else:
        gt = airlight.to(input_img.device).float()
        if gt.dim() == 1:
            gt = gt.view(-1, 1, 1, 1)
        elif gt.dim() == 2:
            gt = gt.view(gt.size(0), gt.size(1), 1, 1)
        if gt.size(1) == 1:
            gt = gt.expand(-1, 3, -1, -1)
        gt = gt.clamp(0.0, 1.0)
    return gt, fallback


def pearson(xs: list[float], ys: list[float]) -> float | None:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 2:
        return None
    mx = statistics.mean(x for x, _ in pairs)
    my = statistics.mean(y for _, y in pairs)
    num = sum((x - mx) * (y - my) for x, y in pairs)
    den_x = math.sqrt(sum((x - mx) ** 2 for x, _ in pairs))
    den_y = math.sqrt(sum((y - my) ** 2 for _, y in pairs))
    if den_x == 0 or den_y == 0:
        return None
    return num / (den_x * den_y)


def build_dta(args, device: torch.device):
    model = build_net(
        "base",
        "Haze4K",
        "original",
        arch="dta_v3",
        dta_variant="v3",
        dta_prior_channels=args.dta_prior_channels,
        dta_gate_bias=args.dta_gate_bias,
        dta_gate_limit=args.dta_gate_limit,
        dta_gamma_limit=args.dta_gamma_limit,
        dta_beta_limit=args.dta_beta_limit,
        dta_alpha_init=args.dta_alpha_init,
        dta_depth_mode=args.dta_depth_mode,
        dta_confidence_floor=args.dta_confidence_floor,
        dta_confidence_local_scale=args.dta_confidence_local_scale,
        dta_r0_residual_scale=args.dta_r0_residual_scale,
        dta_depth_residual_scale=args.dta_depth_residual_scale,
        dta_depth_mask_easy_budget=args.dta_depth_mask_easy_budget,
        dta_depth_mask_dense_budget=args.dta_depth_mask_dense_budget,
        dta_depth_mask_density_thresh=args.dta_depth_mask_density_thresh,
        dta_depth_mask_bias=args.dta_depth_mask_bias,
        dta_phys_t_min=args.dta_phys_t_min,
        dta_phase=args.dta_phase,
        dta_ablation=args.dta_ablation,
    ).to(device)
    model.load_state_dict(load_state(args.candidate_checkpoint, device))
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--original_checkpoint", required=True)
    parser.add_argument("--candidate_checkpoint", required=True)
    parser.add_argument("--depth_cache_dir", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--output_summary_json", required=True)
    parser.add_argument("--split_json", default="")
    parser.add_argument("--split_name", default="")
    parser.add_argument("--eval_root_split", default="train", choices=["train", "test"])
    parser.add_argument("--depth_split", default="train")
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--dta_depth_mode", default="invert", choices=["normal", "invert", "zero", "shuffle"])
    parser.add_argument("--dta_prior_channels", type=int, default=16)
    parser.add_argument("--dta_gate_bias", type=float, default=-5.0)
    parser.add_argument("--dta_gate_limit", type=float, default=0.18)
    parser.add_argument("--dta_gamma_limit", type=float, default=0.28)
    parser.add_argument("--dta_beta_limit", type=float, default=0.14)
    parser.add_argument("--dta_alpha_init", type=float, default=1.0)
    parser.add_argument("--dta_confidence_floor", type=float, default=0.30)
    parser.add_argument("--dta_confidence_local_scale", type=float, default=6.0)
    parser.add_argument("--dta_r0_residual_scale", type=float, default=0.0)
    parser.add_argument("--dta_depth_residual_scale", type=float, default=0.08)
    parser.add_argument("--dta_depth_mask_easy_budget", type=float, default=0.04)
    parser.add_argument("--dta_depth_mask_dense_budget", type=float, default=0.14)
    parser.add_argument("--dta_depth_mask_density_thresh", type=float, default=0.35)
    parser.add_argument("--dta_depth_mask_bias", type=float, default=-4.0)
    parser.add_argument("--dta_phys_t_min", type=float, default=0.10)
    parser.add_argument("--dta_phase", default="depth", choices=["r0", "depth", "joint"])
    parser.add_argument("--dta_ablation", default="full", choices=["full", "r0_only", "film_only_no_output_refine", "trans_head_only_no_rgb_residual", "phys_blend_only"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    official = build_net("base", "Haze4K", "original", arch="official_convir").to(device)
    official.load_state_dict(load_state(args.original_checkpoint, device))
    official.eval()
    candidate = build_dta(args, device)

    dataloader = test_dataloader(
        args.data_dir,
        "Haze4K",
        batch_size=1,
        num_workers=0,
        depth_cache_dir=args.depth_cache_dir,
        depth_split=args.depth_split,
        root_split=args.eval_root_split,
        return_meta=True,
        split_json=args.split_json,
        split_name=args.split_name,
    )

    rows = []
    start_time = time.time()
    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            if args.max_images > 0 and idx >= args.max_images:
                break
            input_img, label_img, depth, airlight, name = unpack_batch(data)
            input_img = input_img.to(device)
            label_img = label_img.to(device)
            depth = depth.to(device) if depth is not None else None
            airlight = airlight.to(device) if airlight is not None and hasattr(airlight, "to") else airlight
            padded, h, w = pad_to_factor(input_img)
            padded_depth = None
            if depth is not None:
                padded_depth, _, _ = pad_to_factor(depth)
            padded_h, padded_w = padded.shape[-2:]

            a0 = official(padded)[2][:, :, :h, :w].clamp(0.0, 1.0)
            dta_fallback = candidate(padded, padded_depth, airlight=None)[2][:, :, :h, :w].clamp(0.0, 1.0)
            fallback_stats = candidate.DTA.stats() if hasattr(candidate, "DTA") else {}
            dta_gt = candidate(padded, padded_depth, airlight=airlight)[2][:, :, :h, :w].clamp(0.0, 1.0)
            gt_stats = candidate.DTA.stats() if hasattr(candidate, "DTA") else {}

            a_gt, a_fallback = airlight_tensor(input_img, airlight)
            psnr_a0 = psnr(a0, label_img)
            psnr_fb = psnr(dta_fallback, label_img)
            psnr_gt = psnr(dta_gt, label_img)
            ssim_a0 = ssim_value(a0, label_img, padded_h, padded_w)
            ssim_fb = ssim_value(dta_fallback, label_img, padded_h, padded_w)
            ssim_gt = ssim_value(dta_gt, label_img, padded_h, padded_w)
            row = {
                "image_id": name[0],
                "A_gt_mean": float(a_gt.mean().cpu()),
                "A_fallback_mean": float(a_fallback.mean().cpu()),
                "A_abs_gap_mean": float((a_gt - a_fallback).abs().mean().cpu()),
                "PSNR_A0": psnr_a0,
                "PSNR_DTA_with_A_gt": psnr_gt,
                "PSNR_DTA_with_A_fallback": psnr_fb,
                "dPSNR_A_gt_minus_A0": psnr_gt - psnr_a0,
                "dPSNR_A_fallback_minus_A0": psnr_fb - psnr_a0,
                "dPSNR_A_gt_minus_fallback": psnr_gt - psnr_fb,
                "SSIM_A0": ssim_a0,
                "SSIM_DTA_with_A_gt": ssim_gt,
                "SSIM_DTA_with_A_fallback": ssim_fb,
                "dSSIM_A_gt_minus_A0": ssim_gt - ssim_a0,
                "dSSIM_A_fallback_minus_A0": ssim_fb - ssim_a0,
                "worst_regression_fallback_flag": psnr_fb - psnr_a0 <= -0.20,
                "worst_regression_gt_flag": psnr_gt - psnr_a0 <= -0.20,
            }
            for key in ("depth_mask_mean", "depth_mask_max", "depth_delta_abs_mean", "j_phys_delta_abs_mean", "t_pred_mean", "t_pred_std"):
                if key in fallback_stats:
                    row[f"fallback_{key}"] = fallback_stats[key]
                if key in gt_stats:
                    row[f"gt_{key}"] = gt_stats[key]
            rows.append(row)
            if (idx + 1) % 100 == 0:
                print(f"airlight_gap {idx + 1}/{len(dataloader)}", flush=True)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        keys = list(rows[0].keys())
        with output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)

    d_fb = [row["dPSNR_A_fallback_minus_A0"] for row in rows]
    d_gt = [row["dPSNR_A_gt_minus_A0"] for row in rows]
    summary = {
        "count": len(rows),
        "elapsed_sec": time.time() - start_time,
        "mean_A_abs_gap": statistics.mean(row["A_abs_gap_mean"] for row in rows) if rows else None,
        "mean_dPSNR_A_fallback_minus_A0": statistics.mean(d_fb) if rows else None,
        "mean_dPSNR_A_gt_minus_A0": statistics.mean(d_gt) if rows else None,
        "mean_dPSNR_A_gt_minus_fallback": statistics.mean(row["dPSNR_A_gt_minus_fallback"] for row in rows) if rows else None,
        "mean_dSSIM_A_fallback_minus_A0": statistics.mean(row["dSSIM_A_fallback_minus_A0"] for row in rows) if rows else None,
        "mean_dSSIM_A_gt_minus_A0": statistics.mean(row["dSSIM_A_gt_minus_A0"] for row in rows) if rows else None,
        "worst_regression_fallback_count": sum(bool(row["worst_regression_fallback_flag"]) for row in rows),
        "worst_regression_gt_count": sum(bool(row["worst_regression_gt_flag"]) for row in rows),
        "A_gap_vs_fallback_delta_pearson": pearson([row["A_abs_gap_mean"] for row in rows], d_fb),
        "A_gap_vs_gt_minus_fallback_pearson": pearson(
            [row["A_abs_gap_mean"] for row in rows],
            [row["dPSNR_A_gt_minus_fallback"] for row in rows],
        ),
        "locked_test_touched": False,
    }
    Path(args.output_summary_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("DTA_V3_AIRLIGHT_GAP_AUDIT_OK")


if __name__ == "__main__":
    main()
