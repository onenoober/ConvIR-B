#!/usr/bin/env python3
"""Extract A0/UDPNet feature table for v1.6 expert-switch router work.

This is a cloud-only runtime audit. It runs A0 and official UDPNet on the
declared internal splits, records per-image metrics and simple deployable
features, and writes a reusable CSV for later router calibration. It does not
train, tune, or touch locked Haze4K test data.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eval_udpnet_v15_phase0_repro import (  # noqa: E402
    infer_one,
    load_a0_model,
    load_convir_builders,
    load_udpnet_builder,
    load_udpnet_model,
    metric_pair,
    sha256_file,
)


def to_float(value: Any, default: float | None = None) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def mean_or_none(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * pct / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def scalar(value: torch.Tensor) -> float:
    return float(value.detach().float().cpu().item())


def tensor_stats(prefix: str, tensor: torch.Tensor) -> dict[str, float]:
    x = tensor.detach().float()
    return {
        f"{prefix}_mean": scalar(x.mean()),
        f"{prefix}_std": scalar(x.std(unbiased=False)),
        f"{prefix}_min": scalar(x.min()),
        f"{prefix}_max": scalar(x.max()),
        f"{prefix}_range": scalar(x.max() - x.min()),
    }


def gradient_mag(x: torch.Tensor) -> torch.Tensor:
    dx = torch.abs(x[..., :, 1:] - x[..., :, :-1])
    dy = torch.abs(x[..., 1:, :] - x[..., :-1, :])
    dx = F.pad(dx, (0, 1, 0, 0))
    dy = F.pad(dy, (0, 0, 0, 1))
    return dx + dy


def lowfreq_residual(x: torch.Tensor, kernel: int = 15) -> torch.Tensor:
    pad = kernel // 2
    pooled = F.avg_pool2d(x, kernel_size=kernel, stride=1, padding=pad)
    return pooled


def image_features(input_img: torch.Tensor, depth: torch.Tensor) -> dict[str, float]:
    rgb = input_img.detach().float()
    depth = depth.detach().float()
    r = rgb[:, 0:1]
    g = rgb[:, 1:2]
    b = rgb[:, 2:3]
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    rgb_max = rgb.max(dim=1, keepdim=True).values
    rgb_min = rgb.min(dim=1, keepdim=True).values
    saturation = (rgb_max - rgb_min) / torch.clamp(rgb_max, min=1e-6)
    dark_channel = rgb_min
    bright_channel = rgb_max
    lum_grad = gradient_mag(luminance)
    depth_grad = gradient_mag(depth)
    low = lowfreq_residual(luminance)
    high = luminance - low
    depth_std = depth.std(unbiased=False)
    lum_std = luminance.std(unbiased=False)
    edge_mismatch = torch.abs(
        lum_grad / torch.clamp(lum_grad.mean(), min=1e-6)
        - depth_grad / torch.clamp(depth_grad.mean(), min=1e-6)
    )
    out = {}
    out.update(tensor_stats("input_luma", luminance))
    out.update(tensor_stats("input_saturation", saturation))
    out.update(tensor_stats("dark_channel", dark_channel))
    out.update(tensor_stats("bright_channel", bright_channel))
    out.update(tensor_stats("depth", depth))
    out.update(
        {
            "input_luma_contrast": scalar(lum_std),
            "input_highfreq_energy": scalar(torch.mean(torch.abs(high))),
            "input_edge_mean": scalar(lum_grad.mean()),
            "depth_grad_mean": scalar(depth_grad.mean()),
            "depth_near_ratio_gt_0_66": scalar((depth > 0.66).float().mean()),
            "depth_far_ratio_lt_0_33": scalar((depth < 0.33).float().mean()),
            "depth_image_edge_mismatch": scalar(edge_mismatch.mean()),
            "depth_luma_std_ratio": scalar(depth_std / torch.clamp(lum_std, min=1e-6)),
        }
    )
    return out


def output_diff_features(a0_pred: torch.Tensor, udp_pred: torch.Tensor) -> dict[str, float]:
    diff = (udp_pred - a0_pred).detach().float()
    abs_diff = torch.abs(diff)
    r = diff[:, 0:1]
    g = diff[:, 1:2]
    b = diff[:, 2:3]
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    chroma_u = r - g
    chroma_v = b - g
    low = lowfreq_residual(diff)
    high = diff - low
    out = {}
    out.update(tensor_stats("udp_a0_absdiff", abs_diff))
    out.update(
        {
            "udp_a0_luma_shift_mean": scalar(luma.mean()),
            "udp_a0_luma_shift_abs_mean": scalar(torch.abs(luma).mean()),
            "udp_a0_chroma_u_abs_mean": scalar(torch.abs(chroma_u).mean()),
            "udp_a0_chroma_v_abs_mean": scalar(torch.abs(chroma_v).mean()),
            "udp_a0_lowfreq_abs_mean": scalar(torch.abs(low).mean()),
            "udp_a0_highfreq_abs_mean": scalar(torch.abs(high).mean()),
            "udp_a0_low_high_abs_ratio": scalar(torch.abs(low).mean() / torch.clamp(torch.abs(high).mean(), min=1e-6)),
        }
    )
    return out


def parse_filename_features(name: str) -> dict[str, float | str]:
    stem = Path(name).stem
    parts = stem.split("_")
    nums: list[float] = []
    for part in parts[1:]:
        value = to_float(part)
        if value is not None:
            nums.append(value)
    return {
        "filename_param_1": nums[0] if len(nums) > 0 else "",
        "filename_param_2": nums[1] if len(nums) > 1 else "",
    }


def mark_buckets(rows: list[dict[str, Any]]) -> None:
    by_split: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_split.setdefault(str(row["split"]), []).append(row)
    for split_rows in by_split.values():
        ordered = sorted(split_rows, key=lambda row: float(row["a0_psnr"]))
        n = max(1, len(ordered) // 4)
        hard = {row["name"] for row in ordered[:n]}
        easy = {row["name"] for row in ordered[-n:]}
        for row in split_rows:
            if row["name"] in hard:
                row["bucket"] = "hard_bottom25_by_a0"
            elif row["name"] in easy:
                row["bucket"] = "easy_top25_by_a0"
            else:
                row["bucket"] = "mid_by_a0"


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    deltas = [float(row["delta_psnr"]) for row in rows]
    ssim_deltas = [float(row["delta_ssim"]) for row in rows]
    hard = [row for row in rows if row.get("bucket") == "hard_bottom25_by_a0"]
    easy = [row for row in rows if row.get("bucket") == "easy_top25_by_a0"]
    tail_n = max(1, len(rows) // 10)
    ordered = sorted(deltas)
    return {
        "count": len(rows),
        "mean_delta_psnr": mean_or_none(deltas),
        "median_delta_psnr": statistics.median(deltas) if deltas else None,
        "p5_delta_psnr": percentile(deltas, 5),
        "p95_delta_psnr": percentile(deltas, 95),
        "hard_bottom25_delta_psnr": mean_or_none([float(row["delta_psnr"]) for row in hard]),
        "easy_top25_delta_psnr": mean_or_none([float(row["delta_psnr"]) for row in easy]),
        "best10pct_delta_psnr": mean_or_none(ordered[-tail_n:]),
        "worst10pct_delta_psnr": mean_or_none(ordered[:tail_n]),
        "mean_delta_ssim": mean_or_none(ssim_deltas),
        "positive_ratio": sum(delta > 0 for delta in deltas) / max(1, len(deltas)),
        "gain_ge_0_05_count": sum(delta >= 0.05 for delta in deltas),
        "gain_ge_0_10_count": sum(delta >= 0.10 for delta in deltas),
        "gain_ge_0_20_count": sum(delta >= 0.20 for delta in deltas),
        "bad_risk_count_delta_le_-0_20_or_ssim_le_-0_001": sum(
            float(row["delta_psnr"]) <= -0.20 or float(row["delta_ssim"]) <= -0.001
            for row in rows
        ),
    }


def evaluate(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    convir_its_dir = Path(args.convir_its_dir)
    test_dataloader, build_convir_net = load_convir_builders(convir_its_dir)
    build_udpnet = load_udpnet_builder(Path(args.udp_repo))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a0_model = load_a0_model(build_convir_net, Path(args.a0_checkpoint), device)
    udp_model, ckpt_meta = load_udpnet_model(build_udpnet, Path(args.official_checkpoint), device)

    rows: list[dict[str, Any]] = []
    factor = int(args.pad_factor)
    with torch.no_grad():
        for split in args.splits:
            dataloader = test_dataloader(
                args.data_dir,
                "Haze4K",
                batch_size=1,
                num_workers=args.num_workers,
                depth_cache_dir=args.depth_cache_dir,
                depth_split="train" if args.split_json else args.depth_split,
                split_json=args.split_json,
                split_name=split,
            )
            for idx, data in enumerate(dataloader):
                input_img, label_img, depth, name = data
                image_name = name[0] if isinstance(name, (list, tuple)) else str(name)
                input_img = input_img.to(device)
                label_img = label_img.to(device)
                depth = depth.to(device)
                h, w = input_img.shape[2], input_img.shape[3]
                h_pad = ((h + factor) // factor) * factor
                w_pad = ((w + factor) // factor) * factor
                padh = h_pad - h if h % factor != 0 else 0
                padw = w_pad - w if w % factor != 0 else 0
                rgb_padded = F.pad(input_img, (0, padw, 0, padh), "reflect")
                depth_padded = F.pad(depth, (0, padw, 0, padh), "reflect")
                udp_input = torch.cat([rgb_padded, depth_padded], dim=1)

                a0_pred = infer_one(a0_model, rgb_padded, h, w)
                udp_pred = infer_one(udp_model, udp_input, h, w)
                a0_psnr, a0_ssim = metric_pair(a0_pred, label_img, (h_pad, w_pad))
                udp_psnr, udp_ssim = metric_pair(udp_pred, label_img, (h_pad, w_pad))
                delta_psnr = udp_psnr - a0_psnr
                delta_ssim = udp_ssim - a0_ssim
                row: dict[str, Any] = {
                    "split": split,
                    "name": image_name,
                    "a0_psnr": a0_psnr,
                    "udpnet_psnr": udp_psnr,
                    "delta_psnr": delta_psnr,
                    "a0_ssim": a0_ssim,
                    "udpnet_ssim": udp_ssim,
                    "delta_ssim": delta_ssim,
                    "gain_label_ge_0_05": int(delta_psnr >= 0.05),
                    "gain_label_ge_0_10": int(delta_psnr >= 0.10),
                    "gain_label_ge_0_20": int(delta_psnr >= 0.20),
                    "bad_risk_label_delta_le_-0_20_or_ssim_le_-0_001": int(
                        delta_psnr <= -0.20 or delta_ssim <= -0.001
                    ),
                }
                row.update(parse_filename_features(image_name))
                row.update(image_features(input_img, depth))
                row.update(output_diff_features(a0_pred, udp_pred))
                rows.append(row)
                if (idx + 1) % args.print_freq == 0:
                    split_rows = [item for item in rows if item["split"] == split]
                    mean_delta = statistics.mean(float(item["delta_psnr"]) for item in split_rows)
                    print(
                        f"{split} {idx + 1}/{len(dataloader)} "
                        f"feature_rows={len(rows)} mean_delta={mean_delta:.4f}",
                        flush=True,
                    )
                if args.max_images and idx + 1 >= args.max_images:
                    break
    mark_buckets(rows)
    by_split = {}
    for split in sorted({row["split"] for row in rows}):
        by_split[split] = summarize([row for row in rows if row["split"] == split])
    payload = {
        "route": "ConvIR-Dehaze-v1.6-RCExpertSwitch",
        "stage": "A0 + official UDPNet switch feature extraction",
        "status": "FEATURE_TABLE_COMPLETE",
        "locked_test_touched": False,
        "splits": args.splits,
        "data_dir": args.data_dir,
        "depth_cache_dir": args.depth_cache_dir,
        "split_json": args.split_json,
        "a0_checkpoint_path": args.a0_checkpoint,
        "a0_checkpoint_sha256": sha256_file(Path(args.a0_checkpoint)),
        "official_checkpoint_path": args.official_checkpoint,
        "official_checkpoint_sha256": sha256_file(Path(args.official_checkpoint)),
        "official_checkpoint_meta": ckpt_meta,
        "count": len(rows),
        "summaries": by_split,
        "combined_summary": summarize(rows),
        "feature_groups": {
            "cheap_pre_router": [
                "input_luma_*",
                "input_saturation_*",
                "dark_channel_*",
                "bright_channel_*",
                "depth_*",
                "filename_param_*",
            ],
            "safe_post_router": [
                "udp_a0_absdiff_*",
                "udp_a0_luma_shift_*",
                "udp_a0_chroma_*",
                "udp_a0_lowfreq_*",
                "udp_a0_highfreq_*",
            ],
        },
    }
    return rows, payload


def write_outputs(output_dir: Path, rows: list[dict[str, Any]], payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with (output_dir / "udp_switch_feature_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    (output_dir / "udp_switch_feature_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    readme = """# Haze4K v1.6 UDP Switch Feature Extraction

Status: `FEATURE_TABLE_COMPLETE` when `udp_switch_feature_summary.json` says so.

Primary files:

- `udp_switch_feature_table.csv`: per-image A0/UDP metrics, gain/risk labels,
  cheap pre-router features, and safe post-router difference features.
- `udp_switch_feature_summary.json`: split summaries, checkpoint hashes, and
  feature-group metadata.

Locked Haze4K test touched: no.

This table is an intermediate router-calibration asset. It does not authorize
locked test or promotion by itself.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--convir_its_dir", required=True)
    parser.add_argument("--udp_repo", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--depth_cache_dir", required=True)
    parser.add_argument("--a0_checkpoint", required=True)
    parser.add_argument("--official_checkpoint", required=True)
    parser.add_argument("--split_json", required=True)
    parser.add_argument("--splits", nargs="+", default=["val_regular", "val_hard"])
    parser.add_argument("--depth_split", default="test")
    parser.add_argument("--pad_factor", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--print_freq", type=int, default=50)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    rows, payload = evaluate(args)
    write_outputs(Path(args.output_dir), rows, payload)
    print(
        "UDP_SWITCH_FEATURE_EXTRACTION_OK "
        f"rows={len(rows)} output_dir={args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
