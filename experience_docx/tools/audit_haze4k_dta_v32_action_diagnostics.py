#!/usr/bin/env python3
"""DTA-v3.2 no-training diagnostics for depth-action safety.

This tool runs only on train-derived validation splits. It compares a frozen A0
checkpoint with an existing DTA-v3 checkpoint and writes three diagnostics:

* alpha-blend sweep: A0 + alpha * (DTA - A0)
* oracle action upper bound by coverage and granularity
* t_pred vs Haze4K GT transmission correlation/failure audit

It does not save predictions and must not be pointed at the locked Haze4K test.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np
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
from models.ConvIR import build_net as build_convir_net


def is_name_field(value: Any) -> bool:
    return isinstance(value, str) or (
        isinstance(value, (list, tuple)) and len(value) > 0 and isinstance(value[0], str)
    )


def unpack_batch(data: tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any, list[str]]:
    name = data[-1] if is_name_field(data[-1]) else None
    if name is not None:
        data = data[:-1]
        if isinstance(name, str):
            name = [name]
    input_img, label_img = data[0], data[1]
    depth = data[2]
    trans = data[3]
    airlight = data[4] if len(data) >= 5 else None
    return input_img, label_img, depth, trans, airlight, name or [""]


def load_model_state(path: str, device: torch.device) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        return state["model"]
    return state


def build_a0(checkpoint: str, device: torch.device) -> torch.nn.Module:
    model = build_convir_net("base", "Haze4K", "original", arch="official_convir").to(device)
    model.load_state_dict(load_model_state(checkpoint, device))
    model.eval()
    return model


def build_dta(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    model = build_convir_net(
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
        dta_depth_mode="invert",
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
        dta_safe_mix_enabled=args.dta_safe_mix_enabled,
        dta_safe_mix_delta_clip=args.dta_safe_mix_delta_clip,
        dta_safe_mix_phys_weight=args.dta_safe_mix_phys_weight,
        dta_safe_mix_learned_weight=args.dta_safe_mix_learned_weight,
        dta_safe_mix_gate_limit=args.dta_safe_mix_gate_limit,
        dta_safe_mix_gate_bias=args.dta_safe_mix_gate_bias,
        dta_router_fusion_enabled=args.dta_router_fusion_enabled,
        dta_router_image_gate_limit=args.dta_router_image_gate_limit,
        dta_router_patch_gate_limit=args.dta_router_patch_gate_limit,
        dta_router_patch_size=args.dta_router_patch_size,
        dta_router_image_bias=args.dta_router_image_bias,
        dta_router_patch_bias=args.dta_router_patch_bias,
    ).to(device)
    result = model.load_state_dict(load_model_state(args.candidate_checkpoint, device), strict=False)
    allowed_missing = (
        "DTA.trans_uncertainty_head.",
        "DTA.safe_residual_head.",
        "DTA.safe_gate_head.",
        "DTA.router_image_head.",
        "DTA.router_patch_head.",
    )
    missing = [key for key in result.missing_keys if not key.startswith(allowed_missing)]
    if missing or result.unexpected_keys:
        raise RuntimeError(f"Unexpected DTA-v3 checkpoint load: missing={missing} unexpected={result.unexpected_keys}")
    model.eval()
    return model


def pad_to_factor(image: torch.Tensor, factor: int = 32) -> tuple[torch.Tensor, int, int]:
    h, w = image.shape[-2:]
    out_h = ((h + factor) // factor) * factor
    out_w = ((w + factor) // factor) * factor
    pad_h = out_h - h if h % factor != 0 else 0
    pad_w = out_w - w if w % factor != 0 else 0
    return F.pad(image, (0, pad_w, 0, pad_h), "reflect"), pad_h, pad_w


def crop_like(image: torch.Tensor, h: int, w: int) -> torch.Tensor:
    return image[:, :, :h, :w]


def airlight_for_forward(airlight: Any, mode: str, device: torch.device) -> torch.Tensor | None:
    if mode != "gt" or airlight is None:
        return None
    return airlight.to(device) if hasattr(airlight, "to") else torch.as_tensor(airlight, device=device)


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(pred.clamp(0.0, 1.0), target.clamp(0.0, 1.0)).item()
    return 99.0 if mse <= 0 else 10.0 * math.log10(1.0 / mse)


def ssim_value(pred: torch.Tensor, target: torch.Tensor, padded_h: int, padded_w: int) -> float:
    down_ratio = max(1, round(min(padded_h, padded_w) / 256))
    return float(
        ssim(
            F.adaptive_avg_pool2d(pred.clamp(0.0, 1.0), (int(padded_h / down_ratio), int(padded_w / down_ratio))),
            F.adaptive_avg_pool2d(target.clamp(0.0, 1.0), (int(padded_h / down_ratio), int(padded_w / down_ratio))),
            data_range=1,
            size_average=False,
        ).mean().item()
    )


def percentile(values: list[float], pct: float) -> float:
    finite = sorted(v for v in values if math.isfinite(v))
    if not finite:
        return float("nan")
    if len(finite) == 1:
        return finite[0]
    pos = (len(finite) - 1) * pct / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return finite[lo]
    return finite[lo] + (finite[hi] - finite[lo]) * (pos - lo)


def summarize(rows: list[dict[str, Any]], pred_key: str = "psnr", ssim_key: str = "ssim") -> dict[str, Any]:
    deltas = [row[pred_key] - row["a0_psnr"] for row in rows]
    ssim_deltas = [row[ssim_key] - row["a0_ssim"] for row in rows]
    original = [row["a0_psnr"] for row in rows]
    sorted_idx = sorted(range(len(rows)), key=lambda idx: original[idx])
    bucket = max(1, len(rows) // 4)
    hard = sorted_idx[:bucket]
    easy = sorted_idx[-bucket:]
    strong_cut = percentile(original, 75)
    strong = [idx for idx, val in enumerate(original) if val >= strong_cut]
    sorted_d = sorted(deltas)
    tail = max(1, len(sorted_d) // 10)
    return {
        "count": len(rows),
        "mean_psnr_delta": statistics.mean(deltas),
        "median_psnr_delta": statistics.median(deltas),
        "p5_psnr_delta": percentile(deltas, 5),
        "p95_psnr_delta": percentile(deltas, 95),
        "hard_bottom25_psnr_delta": statistics.mean(deltas[idx] for idx in hard),
        "easy_top25_psnr_delta": statistics.mean(deltas[idx] for idx in easy),
        "worst10pct_mean_psnr_delta": statistics.mean(sorted_d[:tail]),
        "best10pct_mean_psnr_delta": statistics.mean(sorted_d[-tail:]),
        "worst10img_mean_psnr_delta": statistics.mean(sorted_d[: min(10, len(sorted_d))]),
        "best10img_mean_psnr_delta": statistics.mean(sorted_d[-min(10, len(sorted_d)) :]),
        "mean_ssim_delta": statistics.mean(ssim_deltas),
        "positive_ratio": sum(delta > 0.0 for delta in deltas) / len(deltas),
        "non_degradation_ratio_delta_ge_-0.01": sum(delta >= -0.01 for delta in deltas) / len(deltas),
        "strong_reference_cut_psnr": strong_cut,
        "strong_reference_count": len(strong),
        "strong_regression_count_delta_le_-0.05": sum(deltas[idx] <= -0.05 for idx in strong),
        "worst_regression_count_delta_le_-0.20": sum(delta <= -0.20 for delta in deltas),
    }


def tensor_stats(prefix: str, tensor: torch.Tensor) -> dict[str, float]:
    flat = tensor.detach().float().flatten()
    return {
        f"{prefix}_mean": float(flat.mean().cpu()),
        f"{prefix}_std": float(flat.std(unbiased=False).cpu()),
        f"{prefix}_min": float(flat.min().cpu()),
        f"{prefix}_p5": float(torch.quantile(flat, 0.05).cpu()),
        f"{prefix}_p50": float(torch.quantile(flat, 0.50).cpu()),
        f"{prefix}_p95": float(torch.quantile(flat, 0.95).cpu()),
        f"{prefix}_max": float(flat.max().cpu()),
    }


def sample_np(tensor: torch.Tensor, limit: int) -> np.ndarray:
    arr = tensor.detach().float().flatten().cpu().numpy()
    if arr.size <= limit:
        return arr
    idx = np.linspace(0, arr.size - 1, num=limit, dtype=np.int64)
    return arr[idx]


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_vals = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_vals[end] == sorted_vals[start]:
            end += 1
        rank = 0.5 * (start + end - 1)
        ranks[order[start:end]] = rank
        start = end
    return ranks


def spearman_np(a: np.ndarray, b: np.ndarray) -> float | None:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return None
    ra = rankdata(a[mask])
    rb = rankdata(b[mask])
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    if denom <= 0:
        return None
    return float((ra * rb).sum() / denom)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def select_mask_from_scores(scores: torch.Tensor, target_coverage: float, positive_only: bool = True) -> torch.Tensor:
    flat = scores.flatten()
    if flat.numel() == 0 or target_coverage <= 0.0:
        return torch.zeros_like(scores, dtype=torch.bool)
    k = max(1, int(round(float(flat.numel()) * target_coverage)))
    k = min(k, flat.numel())
    values, indices = torch.topk(flat, k)
    if positive_only:
        keep = values > 0.0
        indices = indices[keep]
    mask = torch.zeros_like(flat, dtype=torch.bool)
    if indices.numel() > 0:
        mask[indices] = True
    return mask.view_as(scores)


def patch_oracle_mask(a0: torch.Tensor, dta: torch.Tensor, gt: torch.Tensor, target: float, patch: int) -> torch.Tensor:
    err_a0 = (a0 - gt).pow(2).mean(dim=1, keepdim=True)
    err_dta = (dta - gt).pow(2).mean(dim=1, keepdim=True)
    improvement = err_a0 - err_dta
    pooled = F.avg_pool2d(improvement, kernel_size=patch, stride=patch, ceil_mode=True)
    patch_mask = select_mask_from_scores(pooled, target, positive_only=True).float()
    up = F.interpolate(patch_mask, size=gt.shape[-2:], mode="nearest").bool()
    return up.expand_as(gt)


def pixel_oracle_mask(a0: torch.Tensor, dta: torch.Tensor, gt: torch.Tensor, target: float) -> torch.Tensor:
    err_a0 = (a0 - gt).pow(2).mean(dim=1, keepdim=True)
    err_dta = (dta - gt).pow(2).mean(dim=1, keepdim=True)
    mask = select_mask_from_scores(err_a0 - err_dta, target, positive_only=True)
    return mask.expand_as(gt)


def image_texture_mean(image: torch.Tensor) -> float:
    brightness = image.detach().float().mean(dim=1, keepdim=True)
    dx = torch.abs(brightness[:, :, :, 1:] - brightness[:, :, :, :-1])
    dy = torch.abs(brightness[:, :, 1:, :] - brightness[:, :, :-1, :])
    return float(0.5 * (dx.mean() + dy.mean()).cpu())


def run_for_checkpoint(args: argparse.Namespace) -> None:
    if args.eval_root_split == "test":
        raise ValueError("This diagnostic is train-derived only; eval_root_split=test is blocked.")
    if args.max_images and args.max_images < 0:
        raise ValueError("--max_images must be non-negative.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a0 = build_a0(args.a0_checkpoint, device)
    dta = build_dta(args, device)
    dataloader = test_dataloader(
        args.data_dir,
        "Haze4K",
        batch_size=1,
        num_workers=0,
        depth_cache_dir=args.depth_cache_dir,
        depth_split=args.depth_split,
        root_split=args.eval_root_split,
        return_trans=True,
        return_meta=True,
        split_json=args.split_json,
        split_name=args.split_name,
    )

    alpha_values = [float(x) for x in args.alpha_values.split(",") if x.strip()]
    coverages = [float(x) for x in args.coverages.split(",") if x.strip()]
    depth_modes = [x.strip() for x in args.depth_modes.split(",") if x.strip()]
    airlight_modes = [x.strip() for x in args.airlight_modes.split(",") if x.strip()]
    records: list[dict[str, Any]] = []
    alpha_records: dict[tuple[str, str, float], list[dict[str, Any]]] = {
        (a_mode, d_mode, alpha): [] for a_mode in airlight_modes for d_mode in depth_modes for alpha in alpha_values
    }
    oracle_records: dict[tuple[str, str, str, float], list[dict[str, Any]]] = {
        (a_mode, d_mode, gran, cov): []
        for a_mode in airlight_modes
        for d_mode in depth_modes
        for gran in ("patch", "pixel")
        for cov in coverages
    }

    factor = 32
    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            if args.max_images > 0 and idx >= args.max_images:
                break
            input_img, label_img, depth, trans, airlight, name = unpack_batch(data)
            name_str = name[0]
            h, w = input_img.shape[-2:]
            input_img = input_img.to(device)
            label_img = label_img.to(device)
            depth = depth.to(device)
            trans = trans.to(device).float().clamp(1e-4, 1.0)
            padded, pad_h, pad_w = pad_to_factor(input_img, factor)
            padded_h, padded_w = padded.shape[-2:]
            padded_depth = F.pad(depth, (0, pad_w, 0, pad_h), "reflect")
            a0_pred = crop_like(a0(padded)[2], h, w).clamp(0.0, 1.0)
            a0_psnr = psnr(a0_pred, label_img)
            a0_ssim = ssim_value(a0_pred, label_img, padded_h, padded_w)
            depth_by_mode: dict[str, torch.Tensor] = {}
            for d_mode in depth_modes:
                if d_mode == "shuffle":
                    shuffle_idx = (idx + args.depth_shuffle_offset) % len(dataloader.dataset)
                    _, _, shuffled_depth, _, _, shuffled_name = unpack_batch(dataloader.dataset[shuffle_idx])
                    depth_by_mode[d_mode] = shuffled_depth.unsqueeze(0).to(device)
                    depth_source = shuffled_name[0] if shuffled_name else getattr(dataloader.dataset, "image_list", [""])[shuffle_idx]
                else:
                    depth_by_mode[d_mode] = depth
                    depth_source = name_str
                padded_mode_depth = F.pad(depth_by_mode[d_mode], (0, pad_w, 0, pad_h), "reflect")
                for a_mode in airlight_modes:
                    dta.DTA.depth_mode = d_mode
                    air = airlight_for_forward(airlight, a_mode, device)
                    dta_pred = crop_like(dta(padded, padded_mode_depth, airlight=air)[2], h, w).clamp(0.0, 1.0)
                    dta_psnr = psnr(dta_pred, label_img)
                    dta_ssim = ssim_value(dta_pred, label_img, padded_h, padded_w)
                    base = {
                        "name": name_str,
                        "airlight_mode": a_mode,
                        "depth_mode": d_mode,
                        "depth_source_name": depth_source,
                        "same_image_depth": depth_source == name_str,
                        "a0_psnr": a0_psnr,
                        "a0_ssim": a0_ssim,
                        "dta_psnr": dta_psnr,
                        "dta_ssim": dta_ssim,
                        "delta_psnr": dta_psnr - a0_psnr,
                        "delta_ssim": dta_ssim - a0_ssim,
                        "input_brightness_mean": float(input_img.mean().cpu()),
                        "input_texture_mean": image_texture_mean(input_img),
                    }
                    for key, value in dta.DTA.stats().items():
                        base[f"dta_{key}"] = value
                    if d_mode == "invert":
                        t_pred = F.interpolate(
                            dta.DTA.last_aux["t_pred"], size=(padded_h, padded_w), mode="bilinear", align_corners=False
                        )[:, :, :h, :w].clamp(1e-4, 1.0)
                        depth_norm = dta.DTA.last_aux["depth_full"][:, :, :h, :w].clamp(0.0, 1.0)
                        log_t_error = (torch.log(t_pred) - torch.log(trans)).abs()
                        base.update(tensor_stats("t_gt", trans))
                        base.update(tensor_stats("t_pred", t_pred))
                        base.update(tensor_stats("log_t_abs_error", log_t_error))
                        base["t_pred_gt_spearman"] = spearman_np(sample_np(t_pred, args.spearman_sample), sample_np(trans, args.spearman_sample))
                        base["depth_neglog_tgt_spearman"] = spearman_np(
                            sample_np(depth_norm, args.spearman_sample),
                            sample_np(-torch.log(trans), args.spearman_sample),
                        )
                        base["oracle_accept_image"] = dta_psnr > a0_psnr
                        base["oracle_accept_patch_ratio"] = float(
                            patch_oracle_mask(a0_pred, dta_pred, label_img, 1.0, args.oracle_patch)[:, :1].float().mean().cpu()
                        )
                        base["oracle_accept_pixel_ratio"] = float(
                            pixel_oracle_mask(a0_pred, dta_pred, label_img, 1.0)[:, :1].float().mean().cpu()
                        )
                    records.append(base)

                    for alpha in alpha_values:
                        blend = (a0_pred + alpha * (dta_pred - a0_pred)).clamp(0.0, 1.0)
                        alpha_records[(a_mode, d_mode, alpha)].append(
                            {
                                "name": name_str,
                                "a0_psnr": a0_psnr,
                                "a0_ssim": a0_ssim,
                                "psnr": psnr(blend, label_img),
                                "ssim": ssim_value(blend, label_img, padded_h, padded_w),
                            }
                        )
                    for cov in coverages:
                        target = cov / 100.0 if cov > 1.0 else cov
                        masks = {
                            "patch": patch_oracle_mask(a0_pred, dta_pred, label_img, target, args.oracle_patch),
                            "pixel": pixel_oracle_mask(a0_pred, dta_pred, label_img, target),
                        }
                        for gran, mask in masks.items():
                            oracle = torch.where(mask, dta_pred, a0_pred)
                            actual_coverage = float(mask[:, :1].float().mean().cpu()) if mask.numel() else 0.0
                            oracle_records[(a_mode, d_mode, gran, cov)].append(
                                {
                                    "name": name_str,
                                    "a0_psnr": a0_psnr,
                                    "a0_ssim": a0_ssim,
                                    "psnr": psnr(oracle, label_img),
                                    "ssim": ssim_value(oracle, label_img, padded_h, padded_w),
                                    "actual_coverage": actual_coverage,
                                }
                            )
            if (idx + 1) % args.progress_every == 0:
                print(f"v32_action_diag {idx + 1}/{len(dataloader)}", flush=True)

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    alpha_rows: list[dict[str, Any]] = []
    for key, rows in alpha_records.items():
        a_mode, d_mode, alpha = key
        if not rows:
            continue
        summary = summarize(rows)
        alpha_rows.append(
            {
                "run_id": args.run_id,
                "candidate": args.candidate_name,
                "airlight_mode": a_mode,
                "depth_mode": d_mode,
                "alpha": alpha,
                **summary,
            }
        )
    add_surplus(alpha_rows, ["airlight_mode", "alpha"], depth_col="depth_mode")
    write_csv(output_root / f"alpha_blend_sweep_matrix_{args.run_id}.csv", alpha_rows)
    (output_root / f"alpha_blend_sweep_matrix_{args.run_id}.json").write_text(
        json.dumps({"run_id": args.run_id, "rows": alpha_rows, "locked_test_touched": False}, indent=2),
        encoding="utf-8",
    )

    oracle_rows: list[dict[str, Any]] = []
    for a_mode in airlight_modes:
        for d_mode in depth_modes:
            grouped = [row for row in records if row["airlight_mode"] == a_mode and row["depth_mode"] == d_mode]
            ranked = sorted(grouped, key=lambda row: row["delta_psnr"], reverse=True)
            for cov in coverages:
                target = cov / 100.0 if cov > 1.0 else cov
                limit = max(1, int(round(len(ranked) * target))) if ranked and target > 0.0 else 0
                selected_names = {
                    row["name"] for row in ranked[:limit] if row["delta_psnr"] > 0.0
                }
                rows = []
                for row in grouped:
                    keep = row["name"] in selected_names
                    rows.append(
                        {
                            "name": row["name"],
                            "a0_psnr": row["a0_psnr"],
                            "a0_ssim": row["a0_ssim"],
                            "psnr": row["dta_psnr"] if keep else row["a0_psnr"],
                            "ssim": row["dta_ssim"] if keep else row["a0_ssim"],
                            "actual_coverage": 1.0 if keep else 0.0,
                        }
                    )
                if rows:
                    summary = summarize(rows)
                    oracle_rows.append(
                        {
                            "run_id": args.run_id,
                            "candidate": args.candidate_name,
                            "airlight_mode": a_mode,
                            "depth_mode": d_mode,
                            "granularity": "image",
                            "target_coverage": cov,
                            "actual_coverage": statistics.mean(row["actual_coverage"] for row in rows),
                            **summary,
                        }
                    )
    for key, rows in oracle_records.items():
        a_mode, d_mode, gran, cov = key
        if not rows:
            continue
        summary = summarize(rows)
        oracle_rows.append(
            {
                "run_id": args.run_id,
                "candidate": args.candidate_name,
                "airlight_mode": a_mode,
                "depth_mode": d_mode,
                "granularity": gran,
                "target_coverage": cov,
                "actual_coverage": statistics.mean(row["actual_coverage"] for row in rows),
                **summary,
            }
        )
    add_surplus(oracle_rows, ["airlight_mode", "granularity", "target_coverage"], depth_col="depth_mode")
    write_csv(output_root / f"oracle_action_upper_bound_by_coverage_{args.run_id}.csv", oracle_rows)
    (output_root / f"oracle_action_upper_bound_by_coverage_{args.run_id}.json").write_text(
        json.dumps({"run_id": args.run_id, "rows": oracle_rows, "locked_test_touched": False}, indent=2),
        encoding="utf-8",
    )

    write_csv(output_root / f"t_pred_vs_trans_gt_correlation_{args.run_id}.csv", records)
    correlation_summary = summarize_t_error(records)
    (output_root / f"t_error_to_regression_correlation_{args.run_id}.json").write_text(
        json.dumps(correlation_summary, indent=2),
        encoding="utf-8",
    )
    bin_report = transmission_bins(records)
    (output_root / f"transmission_bin_failure_report_{args.run_id}.json").write_text(
        json.dumps(bin_report, indent=2),
        encoding="utf-8",
    )
    write_routerfusion_diagnostics(output_root, args, records)
    write_oracle_manifest(output_root / f"oracle_best_possible_contact_sheet_manifest_{args.run_id}.md", args, records)
    print(json.dumps({"run_id": args.run_id, "alpha_rows": len(alpha_rows), "oracle_rows": len(oracle_rows)}, indent=2))
    print("DTA_V32_ACTION_DIAGNOSTICS_OK")


def add_surplus(rows: list[dict[str, Any]], group_keys: list[str], depth_col: str) -> None:
    by_key = {tuple(row.get(k) for k in group_keys) + (row[depth_col],): row for row in rows}
    for row in rows:
        if row[depth_col] != "invert":
            continue
        key = tuple(row.get(k) for k in group_keys)
        zero = by_key.get(key + ("zero",))
        shuffle = by_key.get(key + ("shuffle",))
        normal = by_key.get(key + ("normal",))
        if zero:
            row["true_vs_zero_surplus"] = row["mean_psnr_delta"] - zero["mean_psnr_delta"]
        if shuffle:
            row["true_vs_shuffle_surplus"] = row["mean_psnr_delta"] - shuffle["mean_psnr_delta"]
        if normal:
            row["true_vs_normal_surplus"] = row["mean_psnr_delta"] - normal["mean_psnr_delta"]


def pearson(xs: list[float], ys: list[float]) -> float | None:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 3:
        return None
    mx = statistics.mean(x for x, _ in pairs)
    my = statistics.mean(y for _, y in pairs)
    num = sum((x - mx) * (y - my) for x, y in pairs)
    dx = math.sqrt(sum((x - mx) ** 2 for x, _ in pairs))
    dy = math.sqrt(sum((y - my) ** 2 for _, y in pairs))
    return None if dx <= 0 or dy <= 0 else num / (dx * dy)


def summarize_t_error(records: list[dict[str, Any]]) -> dict[str, Any]:
    primary = [row for row in records if row["depth_mode"] == "invert" and row["airlight_mode"] == "fallback"]
    t_error = [float(row.get("log_t_abs_error_mean", float("nan"))) for row in primary]
    deltas = [float(row["delta_psnr"]) for row in primary]
    dssim = [float(row["delta_ssim"]) for row in primary]
    mask_mean = [float(row.get("dta_depth_mask_mean", float("nan"))) for row in primary]
    action = [float(row.get("dta_depth_delta_abs_mean", float("nan"))) for row in primary]
    jphys = [float(row.get("dta_j_phys_delta_abs_mean", float("nan"))) for row in primary]
    return {
        "protocol": "DTA-v3.2 t_pred/transmission failure audit",
        "primary_filter": {"depth_mode": "invert", "airlight_mode": "fallback"},
        "count": len(primary),
        "pearson_log_t_error_vs_delta_psnr": pearson(t_error, deltas),
        "pearson_log_t_error_vs_delta_ssim": pearson(t_error, dssim),
        "pearson_mask_mean_vs_delta_psnr": pearson(mask_mean, deltas),
        "pearson_depth_action_abs_vs_delta_psnr": pearson(action, deltas),
        "pearson_jphys_abs_vs_delta_psnr": pearson(jphys, deltas),
        "mean_log_t_abs_error_worst_le_-0.20": mean_where(t_error, [d <= -0.20 for d in deltas]),
        "mean_log_t_abs_error_non_worst": mean_where(t_error, [d > -0.20 for d in deltas]),
        "locked_test_touched": False,
    }


def mean_where(values: list[float], mask: list[bool]) -> float | None:
    picked = [v for v, keep in zip(values, mask) if keep and math.isfinite(v)]
    return statistics.mean(picked) if picked else None


def transmission_bins(records: list[dict[str, Any]]) -> dict[str, Any]:
    primary = [row for row in records if row["depth_mode"] == "invert" and row["airlight_mode"] == "fallback"]
    bins = [
        ("low_t_mean_lt_0.35", lambda row: float(row.get("t_gt_mean", 1.0)) < 0.35),
        ("mid_t_mean_0.35_0.65", lambda row: 0.35 <= float(row.get("t_gt_mean", 1.0)) < 0.65),
        ("high_t_mean_ge_0.65", lambda row: float(row.get("t_gt_mean", 0.0)) >= 0.65),
        ("high_log_t_error_top25", None),
        ("bright_top25", None),
        ("low_texture_bottom25", None),
    ]
    thresholds = {
        "err_p75": percentile([float(row.get("log_t_abs_error_mean", float("nan"))) for row in primary], 75),
        "bright_p75": percentile([float(row.get("input_brightness_mean", float("nan"))) for row in primary], 75),
        "texture_p25": percentile([float(row.get("input_texture_mean", float("nan"))) for row in primary], 25),
    }
    out = {"protocol": "transmission-bin failure report", "count": len(primary), "bins": [], "locked_test_touched": False}
    for label, fn in bins:
        if label == "high_log_t_error_top25":
            fn = lambda row, thr=thresholds["err_p75"]: float(row.get("log_t_abs_error_mean", -1.0)) >= thr
        elif label == "bright_top25":
            fn = lambda row, thr=thresholds["bright_p75"]: float(row.get("input_brightness_mean", -1.0)) >= thr
        elif label == "low_texture_bottom25":
            fn = lambda row, thr=thresholds["texture_p25"]: float(row.get("input_texture_mean", 1.0)) <= thr
        picked = [row for row in primary if fn(row)]
        if not picked:
            continue
        deltas = [float(row["delta_psnr"]) for row in picked]
        out["bins"].append(
            {
                "bin": label,
                "count": len(picked),
                "mean_delta_psnr": statistics.mean(deltas),
                "mean_delta_ssim": statistics.mean(float(row["delta_ssim"]) for row in picked),
                "worst_regression_count_delta_le_-0.20": sum(delta <= -0.20 for delta in deltas),
                "positive_ratio": sum(delta > 0.0 for delta in deltas) / len(deltas),
                "mean_log_t_abs_error": statistics.mean(float(row.get("log_t_abs_error_mean", 0.0)) for row in picked),
                "mean_mask": statistics.mean(float(row.get("dta_depth_mask_mean", 0.0)) for row in picked),
                "mean_action_abs": statistics.mean(float(row.get("dta_depth_delta_abs_mean", 0.0)) for row in picked),
            }
        )
    return out


def router_score(row: dict[str, Any]) -> float:
    candidates = [
        row.get("dta_safe_image_router_mean"),
        row.get("dta_safe_patch_router_mean"),
        row.get("dta_safe_gate_mean"),
        row.get("dta_depth_mask_mean"),
    ]
    score = 1.0
    used = False
    for value in candidates[:2]:
        if value is None:
            continue
        try:
            score *= float(value)
            used = True
        except (TypeError, ValueError):
            pass
    if not used:
        for value in candidates[2:]:
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    return score if used else 0.0


def routed_summary(rows: list[dict[str, Any]], selected: set[str]) -> dict[str, Any]:
    routed = []
    selected_deltas = []
    selected_ssim = []
    for row in rows:
        keep = row["name"] in selected
        delta = float(row["delta_psnr"]) if keep else 0.0
        dssim = float(row["delta_ssim"]) if keep else 0.0
        routed.append(
            {
                "name": row["name"],
                "a0_psnr": row["a0_psnr"],
                "a0_ssim": row["a0_ssim"],
                "psnr": row["a0_psnr"] + delta,
                "ssim": row["a0_ssim"] + dssim,
            }
        )
        if keep:
            selected_deltas.append(float(row["delta_psnr"]))
            selected_ssim.append(float(row["delta_ssim"]))
    summary = summarize(routed)
    coverage = len(selected) / len(rows) if rows else 0.0
    summary.update(
        {
            "coverage": coverage,
            "selected_count": len(selected),
            "selected_mean_delta": statistics.mean(selected_deltas) if selected_deltas else 0.0,
            "selected_mean_ssim_delta": statistics.mean(selected_ssim) if selected_ssim else 0.0,
            "selected_conditional_positive_ratio": (
                sum(delta > 0.0 for delta in selected_deltas) / len(selected_deltas) if selected_deltas else 0.0
            ),
            "global_positive_or_zero_ratio": (
                sum((float(row["delta_psnr"]) if row["name"] in selected else 0.0) >= 0.0 for row in rows) / len(rows)
                if rows else 0.0
            ),
        }
    )
    return summary


def write_routerfusion_diagnostics(output_root: Path, args: argparse.Namespace, records: list[dict[str, Any]]) -> None:
    primary = [row for row in records if row["depth_mode"] == "invert" and row["airlight_mode"] == "fallback"]
    if not primary:
        return
    sorted_primary = sorted(primary, key=router_score, reverse=True)
    gate_rows = []
    for row in primary:
        gate_rows.append(
            {
                "run_id": args.run_id,
                "candidate": args.candidate_name,
                "name": row["name"],
                "oracle_accept_image": row.get("oracle_accept_image"),
                "oracle_accept_patch_ratio": row.get("oracle_accept_patch_ratio"),
                "oracle_accept_pixel_ratio": row.get("oracle_accept_pixel_ratio"),
                "router_score": router_score(row),
                "model_r_img": row.get("dta_safe_image_router_mean"),
                "model_r_patch_mean": row.get("dta_safe_patch_router_mean"),
                "model_pixel_gate_mean": row.get("dta_safe_gate_mean"),
                "model_pixel_gate_p95_proxy": row.get("dta_safe_gate_max"),
                "delta_psnr": row["delta_psnr"],
                "delta_ssim": row["delta_ssim"],
                "a0_psnr": row["a0_psnr"],
                "t_pred_mean": row.get("t_pred_mean"),
                "t_gt_mean": row.get("t_gt_mean"),
                "log_t_abs_error_mean": row.get("log_t_abs_error_mean"),
                "t_uncertainty_mean": row.get("dta_safe_t_uncertainty_mean") or row.get("dta_t_uncertainty_mean"),
                "phys_abs_mean": row.get("dta_j_phys_delta_abs_mean"),
                "safe_delta_abs_mean": row.get("dta_depth_delta_abs_mean"),
                "brightness_mean": row.get("input_brightness_mean"),
                "texture_mean": row.get("input_texture_mean"),
                "low_t_flag": float(row.get("t_gt_mean", 1.0)) < 0.35,
                "worst_flag": float(row["delta_psnr"]) <= -0.20,
            }
        )
    write_csv(output_root / f"gate_oracle_gap_report_{args.run_id}.csv", gate_rows)
    (output_root / f"gate_oracle_gap_report_{args.run_id}.json").write_text(
        json.dumps(
            {
                "run_id": args.run_id,
                "candidate": args.candidate_name,
                "count": len(gate_rows),
                "mean_oracle_patch_ratio": statistics.mean(float(row.get("oracle_accept_patch_ratio") or 0.0) for row in gate_rows),
                "mean_model_router_score": statistics.mean(float(row.get("router_score") or 0.0) for row in gate_rows),
                "locked_test_touched": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    curve_rows = []
    for coverage in [0.10, 0.20, 0.25, 0.40, 0.55, 0.60, 0.80, 1.00]:
        limit = max(1, int(round(len(sorted_primary) * coverage)))
        selected = {row["name"] for row in sorted_primary[:limit]}
        curve_rows.append(
            {
                "run_id": args.run_id,
                "candidate": args.candidate_name,
                "score": "model_router_score",
                "target_coverage": coverage,
                **routed_summary(primary, selected),
            }
        )
    write_csv(output_root / f"risk_coverage_curve_{args.run_id}.csv", curve_rows)
    best = max(curve_rows, key=lambda row: row.get("mean_psnr_delta", -999.0))
    (output_root / f"router_metric_correction_report_{args.run_id}.json").write_text(
        json.dumps(
            {
                "run_id": args.run_id,
                "candidate": args.candidate_name,
                "metric_note": "Rejected images are A0 fallback, so routed rows report coverage and selected conditional positive ratio separately from global mean.",
                "best_by_mean": best,
                "rows": curve_rows,
                "locked_test_touched": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    taxonomy_rows = action_failure_taxonomy_rows(args, primary)
    write_csv(output_root / f"action_failure_taxonomy_{args.run_id}.csv", taxonomy_rows)
    write_taxonomy_manifest(output_root / f"top48_worst_contact_sheet_manifest_{args.run_id}.md", args, taxonomy_rows, overlay=False)
    write_taxonomy_manifest(output_root / f"top48_worst_gate_overlay_manifest_{args.run_id}.md", args, taxonomy_rows, overlay=True)

    trans_cal = trans_uncertainty_calibration(primary)
    (output_root / f"trans_uncertainty_calibration_{args.run_id}.json").write_text(
        json.dumps(trans_cal, indent=2),
        encoding="utf-8",
    )
    write_csv(output_root / f"t_pred_vs_gt_transmission_by_group_{args.run_id}.csv", trans_group_rows(primary))
    write_csv(output_root / f"counterfactual_gate_matrix_{args.run_id}.csv", counterfactual_gate_rows(records))


def action_failure_taxonomy_rows(args: argparse.Namespace, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    err_p75 = percentile([float(row.get("log_t_abs_error_mean", float("nan"))) for row in rows], 75)
    bright_p75 = percentile([float(row.get("input_brightness_mean", float("nan"))) for row in rows], 75)
    texture_p25 = percentile([float(row.get("input_texture_mean", float("nan"))) for row in rows], 25)
    phys_p75 = percentile([float(row.get("dta_j_phys_delta_abs_mean", float("nan"))) for row in rows], 75)
    action_p75 = percentile([float(row.get("dta_depth_delta_abs_mean", float("nan"))) for row in rows], 75)
    out = []
    for row in sorted(rows, key=lambda item: float(item["delta_psnr"])):
        labels = []
        if float(row.get("t_gt_mean", 1.0)) < 0.35:
            labels.append("low_t_failure")
        if float(row.get("input_brightness_mean", 0.0)) >= bright_p75 and float(row.get("input_texture_mean", 1.0)) <= texture_p25:
            labels.append("high_bright_low_texture")
            labels.append("sky_like")
        if float(row.get("dta_j_phys_delta_abs_mean", 0.0)) >= phys_p75:
            labels.append("high_phys_delta")
        if float(row.get("dta_depth_delta_abs_mean", 0.0)) >= action_p75:
            labels.append("high_learned_delta")
        if float(row.get("log_t_abs_error_mean", 0.0)) >= err_p75:
            labels.append("high_uncertainty")
            labels.append("depth_mismatch")
        if float(row.get("a0_psnr", 0.0)) >= percentile([float(r["a0_psnr"]) for r in rows], 75) and float(row["delta_psnr"]) < 0.0:
            labels.append("easy_A0_but_DTA_hurts")
        if float(row["delta_psnr"]) > 0.0:
            labels.append("win")
        if not labels:
            labels.append("unknown")
        out.append(
            {
                "run_id": args.run_id,
                "candidate": args.candidate_name,
                "name": row["name"],
                "delta_psnr": row["delta_psnr"],
                "delta_ssim": row["delta_ssim"],
                "labels": ";".join(labels),
                "low_t_failure": "low_t_failure" in labels,
                "high_bright_low_texture": "high_bright_low_texture" in labels,
                "sky_like": "sky_like" in labels,
                "edge_halo": False,
                "color_shift": False,
                "over_dehaze": "high_phys_delta" in labels,
                "under_dehaze": False,
                "high_phys_delta": "high_phys_delta" in labels,
                "high_learned_delta": "high_learned_delta" in labels,
                "high_uncertainty": "high_uncertainty" in labels,
                "easy_A0_but_DTA_hurts": "easy_A0_but_DTA_hurts" in labels,
                "depth_mismatch": "depth_mismatch" in labels,
                "unknown": labels == ["unknown"],
            }
        )
    return out


def write_taxonomy_manifest(path: Path, args: argparse.Namespace, rows: list[dict[str, Any]], overlay: bool) -> None:
    title = "Top48 Worst Gate Overlay Manifest" if overlay else "Top48 Worst Contact-Sheet Manifest"
    lines = [
        f"# {title}: {args.run_id}",
        "",
        "- locked_test_touched: false",
        f"- candidate: `{args.candidate_name}`",
        "- note: PNG/overlay renderings are generated cloud-side and excluded from Git; this manifest fixes the image list and labels.",
        "",
        "| rank | image | delta_psnr | delta_ssim | labels |",
        "| ---: | --- | ---: | ---: | --- |",
    ]
    for idx, row in enumerate(rows[:48], 1):
        lines.append(
            f"| {idx} | `{row['name']}` | {float(row['delta_psnr']):.6f} | "
            f"{float(row['delta_ssim']):.8f} | `{row['labels']}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def trans_uncertainty_calibration(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors = [float(row.get("log_t_abs_error_mean", float("nan"))) for row in rows]
    uncertainties = [
        float(row.get("dta_safe_t_uncertainty_mean") or row.get("dta_t_uncertainty_mean") or float("nan"))
        for row in rows
    ]
    deltas = [float(row["delta_psnr"]) for row in rows]
    worst = [1.0 if delta <= -0.20 else 0.0 for delta in deltas]
    ordered = sorted(zip(uncertainties, errors, deltas), key=lambda item: item[0])
    bins = []
    if ordered:
        size = max(1, len(ordered) // 10)
        for start in range(0, len(ordered), size):
            chunk = ordered[start : start + size]
            if not chunk:
                continue
            bins.append(
                {
                    "bin": len(bins),
                    "count": len(chunk),
                    "mean_uncertainty": statistics.mean(item[0] for item in chunk if math.isfinite(item[0])),
                    "mean_log_t_abs_error": statistics.mean(item[1] for item in chunk if math.isfinite(item[1])),
                    "mean_delta_psnr": statistics.mean(item[2] for item in chunk if math.isfinite(item[2])),
                }
            )
    return {
        "protocol": "DTA-v3.3 transmission uncertainty calibration",
        "count": len(rows),
        "log_t_l1": statistics.mean(v for v in errors if math.isfinite(v)),
        "low_t_log_t_l1": mean_where(errors, [float(row.get("t_gt_mean", 1.0)) < 0.35 for row in rows]),
        "t_uncertainty_ECE_proxy": statistics.mean(abs(u - e) for u, e in zip(uncertainties, errors) if math.isfinite(u) and math.isfinite(e)),
        "corr_t_error_delta_psnr": pearson(errors, deltas),
        "corr_t_uncertainty_delta_psnr": pearson(uncertainties, deltas),
        "corr_t_uncertainty_worst_flag": pearson(uncertainties, worst),
        "bins": bins,
        "locked_test_touched": False,
    }


def trans_group_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = [
        ("all", lambda row: True),
        ("low_t", lambda row: float(row.get("t_gt_mean", 1.0)) < 0.35),
        ("mid_t", lambda row: 0.35 <= float(row.get("t_gt_mean", 1.0)) < 0.65),
        ("high_t", lambda row: float(row.get("t_gt_mean", 0.0)) >= 0.65),
        ("worst", lambda row: float(row["delta_psnr"]) <= -0.20),
        ("positive", lambda row: float(row["delta_psnr"]) > 0.0),
    ]
    out = []
    for label, fn in groups:
        picked = [row for row in rows if fn(row)]
        if not picked:
            continue
        out.append(
            {
                "group": label,
                "count": len(picked),
                "mean_delta_psnr": statistics.mean(float(row["delta_psnr"]) for row in picked),
                "mean_delta_ssim": statistics.mean(float(row["delta_ssim"]) for row in picked),
                "log_t_l1": statistics.mean(float(row.get("log_t_abs_error_mean", 0.0)) for row in picked),
                "t_uncertainty_mean": statistics.mean(
                    float(row.get("dta_safe_t_uncertainty_mean") or row.get("dta_t_uncertainty_mean") or 0.0)
                    for row in picked
                ),
                "worst_regression_count_delta_le_-0.20": sum(float(row["delta_psnr"]) <= -0.20 for row in picked),
            }
        )
    return out


def counterfactual_gate_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for d_mode in sorted({row["depth_mode"] for row in records}):
        picked = [row for row in records if row["depth_mode"] == d_mode and row["airlight_mode"] == "fallback"]
        if not picked:
            continue
        rows.append(
            {
                "depth_mode": d_mode,
                "count": len(picked),
                "gate_mean": statistics.mean(float(row.get("dta_safe_gate_mean", row.get("dta_depth_mask_mean", 0.0))) for row in picked),
                "gate_p95_proxy": statistics.mean(float(row.get("dta_safe_gate_max", 0.0)) for row in picked),
                "safe_delta_abs_mean": statistics.mean(float(row.get("dta_depth_delta_abs_mean", 0.0)) for row in picked),
                "safe_delta_abs_p95_proxy": statistics.mean(float(row.get("dta_safe_mixed_delta_abs_mean", 0.0)) for row in picked),
                "mean_dPSNR": statistics.mean(float(row["delta_psnr"]) for row in picked),
                "dSSIM": statistics.mean(float(row["delta_ssim"]) for row in picked),
                "worst": sum(float(row["delta_psnr"]) <= -0.20 for row in picked),
            }
        )
    return rows


def write_oracle_manifest(path: Path, args: argparse.Namespace, records: list[dict[str, Any]]) -> None:
    primary = [row for row in records if row["depth_mode"] == "invert" and row["airlight_mode"] == "fallback"]
    best = sorted(primary, key=lambda row: row["delta_psnr"], reverse=True)[:20]
    worst = sorted(primary, key=lambda row: row["delta_psnr"])[:20]
    lines = [
        f"# Oracle Best-Possible Contact-Sheet Manifest: {args.run_id}",
        "",
        "This text-only manifest lists images that should be visualized if an oracle",
        "contact sheet is generated later. PNG outputs are intentionally not written by",
        "this diagnostic tool or committed to Git.",
        "",
        "- locked_test_touched: false",
        f"- candidate: `{args.candidate_name}`",
        f"- split: `{args.split_name}` from `{args.split_json}`",
        "",
        "## Top DTA Wins",
        "",
        "| rank | image | delta_psnr | delta_ssim | t_error | mask_mean | action_abs |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for idx, row in enumerate(best, 1):
        lines.append(format_manifest_row(idx, row))
    lines.extend(
        [
            "",
            "## Worst DTA Regressions",
            "",
            "| rank | image | delta_psnr | delta_ssim | t_error | mask_mean | action_abs |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for idx, row in enumerate(worst, 1):
        lines.append(format_manifest_row(idx, row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_manifest_row(idx: int, row: dict[str, Any]) -> str:
    return (
        f"| {idx} | `{row['name']}` | {float(row['delta_psnr']):.6f} | "
        f"{float(row['delta_ssim']):.8f} | {float(row.get('log_t_abs_error_mean', float('nan'))):.6f} | "
        f"{float(row.get('dta_depth_mask_mean', float('nan'))):.6f} | "
        f"{float(row.get('dta_depth_delta_abs_mean', float('nan'))):.6f} |"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--candidate_name", default="dta_v32_source")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--a0_checkpoint", required=True)
    parser.add_argument("--candidate_checkpoint", required=True)
    parser.add_argument("--depth_cache_dir", required=True)
    parser.add_argument("--split_json", required=True)
    parser.add_argument("--split_name", required=True)
    parser.add_argument("--eval_root_split", default="train", choices=["train", "test"])
    parser.add_argument("--depth_split", default="train")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--depth_modes", default="invert,zero,shuffle,normal")
    parser.add_argument("--airlight_modes", default="fallback,gt")
    parser.add_argument("--alpha_values", default="0.10,0.20,0.35,0.50,0.75,1.00")
    parser.add_argument("--coverages", default="25,40,60,80,100")
    parser.add_argument("--oracle_patch", type=int, default=32)
    parser.add_argument("--depth_shuffle_offset", type=int, default=137)
    parser.add_argument("--spearman_sample", type=int, default=4096)
    parser.add_argument("--progress_every", type=int, default=50)
    parser.add_argument("--dta_prior_channels", type=int, default=32)
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
    parser.add_argument("--dta_safe_mix_enabled", action="store_true")
    parser.add_argument("--dta_safe_mix_delta_clip", type=float, default=0.08)
    parser.add_argument("--dta_safe_mix_phys_weight", type=float, default=1.0)
    parser.add_argument("--dta_safe_mix_learned_weight", type=float, default=0.0)
    parser.add_argument("--dta_safe_mix_gate_limit", type=float, default=1.0)
    parser.add_argument("--dta_safe_mix_gate_bias", type=float, default=-3.0)
    parser.add_argument("--dta_router_fusion_enabled", action="store_true")
    parser.add_argument("--dta_router_image_gate_limit", type=float, default=1.0)
    parser.add_argument("--dta_router_patch_gate_limit", type=float, default=1.0)
    parser.add_argument("--dta_router_patch_size", type=int, default=32)
    parser.add_argument("--dta_router_image_bias", type=float, default=2.0)
    parser.add_argument("--dta_router_patch_bias", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_for_checkpoint(args)


if __name__ == "__main__":
    main()
