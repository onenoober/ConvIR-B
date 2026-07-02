#!/usr/bin/env python3
"""Extract deployable DTA-v3.7 candidate-vs-A0 output-difference features.

This D6 tool renders the train-derived D1 TAU candidates, blends them against
the ConvIR-B A0 anchor, and writes compact per-action image statistics. It does
not save images/checkpoints and does not touch the locked Haze4K test split.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

TOOL_PATH = Path(__file__).resolve()
TOOL_DIR = TOOL_PATH.parent
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

from eval_haze4k_dta_v37_tau_real_blend_oracle import (  # noqa: E402
    VARIANT_LABEL,
    VARIANT_ORDER,
    build_a0,
    build_dta,
    checkpoint_path,
    finite_float,
    forward_a0,
    forward_dta,
    pad_to_factor,
    read_d1_run_ids,
    unpack_batch,
    write_csv,
)

REPO_ROOT = TOOL_PATH.parents[2]
ITS_ROOT = REPO_ROOT / "Dehazing" / "ITS"
for path in (str(ITS_ROOT), str(REPO_ROOT), os.getcwd()):
    if path not in sys.path:
        sys.path.insert(0, path)

from data import test_dataloader  # noqa: E402


ALPHAS = (0.10, 0.25, 0.50, 0.75, 1.00)


def percentile(arr: np.ndarray, pct: float) -> float:
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, pct))


def gradient_mag(y: np.ndarray) -> np.ndarray:
    gx = np.zeros_like(y)
    gy = np.zeros_like(y)
    gx[:, 1:-1] = 0.5 * (y[:, 2:] - y[:, :-2])
    gy[1:-1, :] = 0.5 * (y[2:, :] - y[:-2, :])
    return np.sqrt(gx * gx + gy * gy)


def laplacian(y: np.ndarray) -> np.ndarray:
    out = np.zeros_like(y)
    out[1:-1, 1:-1] = 4 * y[1:-1, 1:-1] - y[:-2, 1:-1] - y[2:, 1:-1] - y[1:-1, :-2] - y[1:-1, 2:]
    return out


def luminance(rgb: np.ndarray) -> np.ndarray:
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def saturation(rgb: np.ndarray) -> np.ndarray:
    cmax = np.max(rgb, axis=2)
    cmin = np.min(rgb, axis=2)
    return (cmax - cmin) / np.maximum(cmax, 1e-6)


def resize_for_features(tensor: torch.Tensor, max_side: int) -> torch.Tensor:
    if max_side <= 0:
        return tensor
    h, w = tensor.shape[-2:]
    scale = min(1.0, max_side / max(h, w))
    if scale >= 1.0:
        return tensor
    size = (max(1, int(round(h * scale))), max(1, int(round(w * scale))))
    return F.interpolate(tensor, size=size, mode="bilinear", align_corners=False)


def tensor_to_rgb(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.detach().float().cpu().squeeze(0).permute(1, 2, 0).numpy()
    return np.clip(arr, 0.0, 1.0)


def quality_features(prefix: str, rgb: np.ndarray) -> dict[str, float]:
    y = luminance(rgb)
    sat = saturation(rgb)
    dark = np.min(rgb, axis=2)
    grad = gradient_mag(y)
    lap = laplacian(y)
    means = [float(np.mean(rgb[..., idx])) for idx in range(3)]
    return {
        f"{prefix}_luma_mean": float(np.mean(y)),
        f"{prefix}_luma_std": float(np.std(y)),
        f"{prefix}_luma_p05": percentile(y, 5),
        f"{prefix}_luma_p50": percentile(y, 50),
        f"{prefix}_luma_p95": percentile(y, 95),
        f"{prefix}_contrast_p95_p05": percentile(y, 95) - percentile(y, 5),
        f"{prefix}_sat_mean": float(np.mean(sat)),
        f"{prefix}_sat_std": float(np.std(sat)),
        f"{prefix}_dark_mean": float(np.mean(dark)),
        f"{prefix}_dark_p05": percentile(dark, 5),
        f"{prefix}_edge_mean": float(np.mean(grad)),
        f"{prefix}_edge_p90": percentile(grad, 90),
        f"{prefix}_lap_var": float(np.var(lap)),
        f"{prefix}_highlight_ratio": float(np.mean(y > 0.85)),
        f"{prefix}_shadow_ratio": float(np.mean(y < 0.05)),
        f"{prefix}_color_cast_abs_rg": abs(means[0] - means[1]),
        f"{prefix}_color_cast_abs_rb": abs(means[0] - means[2]),
        f"{prefix}_color_cast_abs_gb": abs(means[1] - means[2]),
    }


def residual_concentration(abs_luma: np.ndarray) -> float:
    flat = np.sort(abs_luma.reshape(-1))
    if flat.size == 0:
        return float("nan")
    total = float(np.sum(flat))
    if total <= 1e-12:
        return 0.0
    top_n = max(1, int(math.ceil(flat.size * 0.10)))
    return float(np.sum(flat[-top_n:]) / total)


def outputdiff_features(hazy_rgb: np.ndarray, a0_rgb: np.ndarray, cand_rgb: np.ndarray, blend_rgb: np.ndarray) -> dict[str, float]:
    a0_q = quality_features("a0q", a0_rgb)
    out_q = quality_features("outq", blend_rgb)
    hazy_y = luminance(hazy_rgb)
    a0_y = luminance(a0_rgb)
    a0_grad = gradient_mag(a0_y)
    high_bright = hazy_y > 0.85
    low_texture = a0_grad < 0.015
    sky_proxy = high_bright & (saturation(hazy_rgb) < 0.25) & low_texture

    full_res = cand_rgb - a0_rgb
    action_res = blend_rgb - a0_rgb
    full_res_abs = np.abs(full_res)
    action_res_abs = np.abs(action_res)
    full_luma = luminance(cand_rgb) - a0_y
    action_luma = luminance(blend_rgb) - a0_y
    full_luma_abs = np.abs(full_luma)
    action_luma_abs = np.abs(action_luma)
    action_grad = gradient_mag(luminance(blend_rgb))
    res_edge = np.abs(action_grad - a0_grad)

    def masked_mean(arr: np.ndarray, mask: np.ndarray) -> float:
        return float(np.mean(arr[mask])) if bool(np.any(mask)) else 0.0

    out: dict[str, float] = {
        "od_full_res_abs_mean": float(np.mean(full_res_abs)),
        "od_full_res_abs_p50": percentile(full_res_abs, 50),
        "od_full_res_abs_p95": percentile(full_res_abs, 95),
        "od_full_res_abs_max": float(np.max(full_res_abs)),
        "od_full_res_signed_mean": float(np.mean(full_res)),
        "od_full_res_signed_std": float(np.std(full_res)),
        "od_full_res_pos_ratio": float(np.mean(full_res > 0.0)),
        "od_full_res_neg_ratio": float(np.mean(full_res < 0.0)),
        "od_full_luma_mean": float(np.mean(full_luma)),
        "od_full_luma_std": float(np.std(full_luma)),
        "od_full_luma_abs_mean": float(np.mean(full_luma_abs)),
        "od_full_luma_abs_p95": percentile(full_luma_abs, 95),
        "od_action_res_abs_mean": float(np.mean(action_res_abs)),
        "od_action_res_abs_p50": percentile(action_res_abs, 50),
        "od_action_res_abs_p95": percentile(action_res_abs, 95),
        "od_action_res_abs_max": float(np.max(action_res_abs)),
        "od_action_res_signed_mean": float(np.mean(action_res)),
        "od_action_res_signed_std": float(np.std(action_res)),
        "od_action_res_pos_ratio": float(np.mean(action_res > 0.0)),
        "od_action_res_neg_ratio": float(np.mean(action_res < 0.0)),
        "od_action_luma_mean": float(np.mean(action_luma)),
        "od_action_luma_std": float(np.std(action_luma)),
        "od_action_luma_p10": percentile(action_luma, 10),
        "od_action_luma_p90": percentile(action_luma, 90),
        "od_action_luma_abs_mean": float(np.mean(action_luma_abs)),
        "od_action_luma_abs_p95": percentile(action_luma_abs, 95),
        "od_action_edge_mean": float(np.mean(res_edge)),
        "od_action_edge_p90": percentile(res_edge, 90),
        "od_action_concentration_top10": residual_concentration(action_luma_abs),
        "od_clip_low_ratio": float(np.mean(blend_rgb <= 1.0 / 255.0)),
        "od_clip_high_ratio": float(np.mean(blend_rgb >= 254.0 / 255.0)),
        "od_highbright_abs_mean": masked_mean(action_luma_abs, high_bright),
        "od_lowtex_abs_mean": masked_mean(action_luma_abs, low_texture),
        "od_sky_abs_mean": masked_mean(action_luma_abs, sky_proxy),
        "od_sky_ratio": float(np.mean(sky_proxy)),
        "od_highbright_ratio": float(np.mean(high_bright)),
        "od_lowtex_ratio": float(np.mean(low_texture)),
    }
    for channel, idx in (("r", 0), ("g", 1), ("b", 2)):
        out[f"od_action_{channel}_mean"] = float(np.mean(action_res[..., idx]))
        out[f"od_action_{channel}_abs_mean"] = float(np.mean(np.abs(action_res[..., idx])))
    for key, value in out_q.items():
        out[key] = value
    for key, value in out_q.items():
        base_key = key.replace("outq", "a0q", 1)
        if base_key in a0_q:
            out[key.replace("outq", "oq", 1) + "_delta_vs_a0"] = value - a0_q[base_key]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--a0_checkpoint", required=True, type=Path)
    parser.add_argument("--checkpoint_root", required=True, type=Path)
    parser.add_argument("--action_table_csv", required=True, type=Path)
    parser.add_argument("--include_run_substring", default="quick5full")
    parser.add_argument("--depth_cache_dir", required=True)
    parser.add_argument("--split_json", required=True)
    parser.add_argument("--fold", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--feature_max_side", type=int, default=384)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    run_ids = read_d1_run_ids(args.action_table_csv, args.fold, args.seed, args.include_run_substring)
    checkpoints = {variant: checkpoint_path(args, run_ids, variant) for variant in VARIANT_ORDER}
    missing = [str(path) for path in [args.a0_checkpoint, *checkpoints.values()] if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoint(s): {missing}")

    print(
        f"DTA_V3_7_D6_OUTPUTDIFF_GROUP_START fold={args.fold} seed={args.seed} "
        f"device={device} feature_max_side={args.feature_max_side} run_ids={run_ids}",
        flush=True,
    )
    start_time = time.time()
    a0_model = build_a0(args.a0_checkpoint, device)
    models = {variant: build_dta(variant, path, device) for variant, path in checkpoints.items()}

    dataloader = test_dataloader(
        args.data_dir,
        "Haze4K",
        batch_size=1,
        num_workers=0,
        depth_cache_dir=args.depth_cache_dir,
        depth_split="train",
        root_split="train",
        return_trans=False,
        return_meta=False,
        split_json=args.split_json,
        split_name=f"fold{args.fold}_val",
    )
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            if args.max_images > 0 and idx >= args.max_images:
                break
            input_img, _label_img, depth, image_id = unpack_batch(data)
            if depth is None:
                raise ValueError("D6 output-difference extraction requires depth tensors.")
            input_img = input_img.to(device)
            depth = depth.to(device)
            padded, padded_depth, h, w, _padded_h, _padded_w = pad_to_factor(input_img, depth, factor=32)
            assert padded_depth is not None
            a0_pred = forward_a0(a0_model, padded, h, w)

            hazy_small = tensor_to_rgb(resize_for_features(input_img, args.feature_max_side))
            a0_small = tensor_to_rgb(resize_for_features(a0_pred, args.feature_max_side))
            for variant, model in models.items():
                cand = forward_dta(model, padded, padded_depth, h, w, "invert")
                cand_small = tensor_to_rgb(resize_for_features(cand, args.feature_max_side))
                for alpha in ALPHAS:
                    blend = torch.clamp(a0_pred + alpha * (cand - a0_pred), 0.0, 1.0)
                    blend_small = tensor_to_rgb(resize_for_features(blend, args.feature_max_side))
                    row: dict[str, Any] = {
                        "image_id": image_id,
                        "fold": args.fold,
                        "seed": args.seed,
                        "variant": variant,
                        "variant_label": VARIANT_LABEL.get(variant, variant),
                        "alpha": f"{alpha:g}",
                    }
                    row.update(outputdiff_features(hazy_small, a0_small, cand_small, blend_small))
                    rows.append(row)
            if (idx + 1) % 50 == 0:
                print(
                    f"DTA_V3_7_D6_OUTPUTDIFF_GROUP_PROGRESS fold={args.fold} seed={args.seed} "
                    f"images={idx + 1}/{len(dataloader)} rows={len(rows)} elapsed_sec={time.time() - start_time:.1f}",
                    flush=True,
                )

    csv_path = args.output_dir / f"v37_d6_outputdiff_features_seed{args.seed}_f{args.fold}.csv"
    write_csv(csv_path, rows)
    summary = {
        "route": "DTA-v3.7 U-TQS-Mix",
        "phase": "D6_outputdiff_feature_group",
        "fold": args.fold,
        "seed": args.seed,
        "rows": len(rows),
        "images": len({row["image_id"] for row in rows}),
        "feature_max_side": args.feature_max_side,
        "csv": str(csv_path),
        "d1_run_ids": run_ids,
        "locked_test_touched": False,
        "elapsed_sec": time.time() - start_time,
        "peak_cuda_mem_mib": torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else None,
    }
    summary_path = args.output_dir / f"v37_d6_outputdiff_features_seed{args.seed}_f{args.fold}.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"DTA_V3_7_D6_OUTPUTDIFF_GROUP_OK fold={args.fold} seed={args.seed} "
        f"images={summary['images']} rows={len(rows)} elapsed_sec={summary['elapsed_sec']:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
