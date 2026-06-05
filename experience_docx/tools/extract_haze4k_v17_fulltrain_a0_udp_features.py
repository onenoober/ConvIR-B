#!/usr/bin/env python3
"""Extract full-train A0/UDP feature and alpha-mix table for v1.7.

This is a cloud-only runtime audit. It runs frozen ConvIR-B A0 and the
official UDPNet ConvIR checkpoint on a train-derived split, writes deployable
features, and evaluates deterministic alpha mixtures:

    Y_alpha = A0 + alpha * (UDPNet - A0)

It does not train, tune, or touch locked Haze4K test data.
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
from extract_haze4k_udp_switch_features import (  # noqa: E402
    image_features,
    mark_buckets,
    output_diff_features,
    parse_filename_features,
)


ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]


def alpha_tag(alpha: float) -> str:
    return f"{alpha:.2f}".replace(".", "p")


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


def summarize_alpha(rows: list[dict[str, Any]], alpha: float) -> dict[str, Any]:
    tag = alpha_tag(alpha)
    delta_key = f"alpha_{tag}_delta_psnr"
    ssim_key = f"alpha_{tag}_delta_ssim"
    deltas = [float(row[delta_key]) for row in rows]
    ssim_deltas = [float(row[ssim_key]) for row in rows]
    hard = [row for row in rows if row.get("bucket") == "hard_bottom25_by_a0"]
    easy = [row for row in rows if row.get("bucket") == "easy_top25_by_a0"]
    strong_cut = percentile([float(row["a0_psnr"]) for row in rows], 75)
    strong = [row for row in rows if strong_cut is not None and float(row["a0_psnr"]) >= strong_cut]
    tail_n = max(1, len(rows) // 10)
    ordered = sorted(deltas)
    return {
        "alpha": alpha,
        "count": len(rows),
        "mean_delta_psnr": mean_or_none(deltas),
        "median_delta_psnr": statistics.median(deltas) if deltas else None,
        "p5_delta_psnr": percentile(deltas, 5),
        "p95_delta_psnr": percentile(deltas, 95),
        "hard_bottom25_delta_psnr": mean_or_none([float(row[delta_key]) for row in hard]),
        "easy_top25_delta_psnr": mean_or_none([float(row[delta_key]) for row in easy]),
        "worst10pct_delta_psnr": mean_or_none(ordered[:tail_n]),
        "best10pct_delta_psnr": mean_or_none(ordered[-tail_n:]),
        "mean_delta_ssim": mean_or_none(ssim_deltas),
        "positive_ratio": sum(delta > 0 for delta in deltas) / max(1, len(deltas)),
        "gain_ge_0_05_count": sum(delta >= 0.05 for delta in deltas),
        "gain_ge_0_10_count": sum(delta >= 0.10 for delta in deltas),
        "gain_ge_0_20_count": sum(delta >= 0.20 for delta in deltas),
        "strong_reference_cut_psnr": strong_cut,
        "strong_regression_count_delta_le_-0_05": sum(
            float(row[delta_key]) <= -0.05 for row in strong
        ),
        "strong_regression_ratio": (
            sum(float(row[delta_key]) <= -0.05 for row in strong) / len(strong)
            if strong
            else None
        ),
        "worst_regression_count_delta_le_-0_20": sum(delta <= -0.20 for delta in deltas),
        "worst_regression_ratio": sum(delta <= -0.20 for delta in deltas) / max(1, len(deltas)),
        "bad_risk_count_delta_le_-0_20_or_ssim_le_-0_001": sum(
            float(row[delta_key]) <= -0.20 or float(row[ssim_key]) <= -0.001
            for row in rows
        ),
    }


def add_alpha_metrics(row: dict[str, Any], a0_pred: torch.Tensor, udp_pred: torch.Tensor, label_img: torch.Tensor, padded_hw: tuple[int, int]) -> None:
    best_alpha = 0.0
    best_delta = -1e9
    best_ssim_delta = 0.0
    for alpha in ALPHAS:
        tag = alpha_tag(alpha)
        if alpha == 0.0:
            mix_pred = a0_pred
        elif alpha == 1.0:
            mix_pred = udp_pred
        else:
            mix_pred = torch.clamp(a0_pred + alpha * (udp_pred - a0_pred), 0.0, 1.0)
        mix_psnr, mix_ssim = metric_pair(mix_pred, label_img, padded_hw)
        delta_psnr = mix_psnr - float(row["a0_psnr"])
        delta_ssim = mix_ssim - float(row["a0_ssim"])
        row[f"alpha_{tag}_psnr"] = mix_psnr
        row[f"alpha_{tag}_ssim"] = mix_ssim
        row[f"alpha_{tag}_delta_psnr"] = delta_psnr
        row[f"alpha_{tag}_delta_ssim"] = delta_ssim
        row[f"alpha_{tag}_gain_ge_0_05"] = int(delta_psnr >= 0.05)
        row[f"alpha_{tag}_gain_ge_0_10"] = int(delta_psnr >= 0.10)
        row[f"alpha_{tag}_gain_ge_0_20"] = int(delta_psnr >= 0.20)
        row[f"alpha_{tag}_bad_risk"] = int(delta_psnr <= -0.20 or delta_ssim <= -0.001)
        if delta_psnr > best_delta:
            best_alpha = alpha
            best_delta = delta_psnr
            best_ssim_delta = delta_ssim
    row["oracle_best_alpha"] = best_alpha
    row["oracle_best_alpha_delta_psnr"] = max(0.0, best_delta)
    row["oracle_best_alpha_delta_ssim"] = best_ssim_delta if best_delta > 0 else 0.0


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
                depth_split=args.depth_split,
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
                row: dict[str, Any] = {
                    "split": split,
                    "name": image_name,
                    "a0_psnr": a0_psnr,
                    "a0_ssim": a0_ssim,
                    "udpnet_psnr": udp_psnr,
                    "udpnet_ssim": udp_ssim,
                    "delta_psnr": udp_psnr - a0_psnr,
                    "delta_ssim": udp_ssim - a0_ssim,
                }
                row.update(parse_filename_features(image_name))
                row.update(image_features(input_img, depth))
                row.update(output_diff_features(a0_pred, udp_pred))
                add_alpha_metrics(row, a0_pred, udp_pred, label_img, (h_pad, w_pad))
                rows.append(row)
                if (idx + 1) % args.print_freq == 0:
                    split_rows = [item for item in rows if item["split"] == split]
                    mean_delta = statistics.mean(float(item["delta_psnr"]) for item in split_rows)
                    print(
                        f"{split} {idx + 1}/{len(dataloader)} rows={len(rows)} "
                        f"udp_mean_delta={mean_delta:.4f}",
                        flush=True,
                    )
                if args.max_images and idx + 1 >= args.max_images:
                    break
    mark_buckets(rows)
    split_summaries: dict[str, Any] = {}
    for split in sorted({str(row["split"]) for row in rows}):
        split_rows = [row for row in rows if row["split"] == split]
        split_summaries[split] = {alpha_tag(alpha): summarize_alpha(split_rows, alpha) for alpha in ALPHAS}
    payload = {
        "route": "ConvIR-Dehaze-v1.7-RCExpertMix",
        "stage": "full-train A0/UDP feature and alpha-grid extraction",
        "status": "FEATURE_TABLE_COMPLETE" if not args.max_images else "FEATURE_TABLE_PARTIAL_PREFLIGHT",
        "locked_test_touched": False,
        "splits": args.splits,
        "alpha_grid": ALPHAS,
        "data_dir": args.data_dir,
        "depth_cache_dir": args.depth_cache_dir,
        "depth_split": args.depth_split,
        "split_json": args.split_json,
        "a0_checkpoint_path": args.a0_checkpoint,
        "a0_checkpoint_sha256": sha256_file(Path(args.a0_checkpoint)),
        "official_checkpoint_path": args.official_checkpoint,
        "official_checkpoint_sha256": sha256_file(Path(args.official_checkpoint)),
        "official_checkpoint_meta": ckpt_meta,
        "count": len(rows),
        "split_summaries": split_summaries,
        "combined_alpha_summaries": {alpha_tag(alpha): summarize_alpha(rows, alpha) for alpha in ALPHAS},
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
            "labels_not_allowed_as_router_features": [
                "a0_psnr",
                "a0_ssim",
                "udpnet_psnr",
                "udpnet_ssim",
                "delta_psnr",
                "delta_ssim",
                "alpha_*_psnr",
                "alpha_*_ssim",
                "alpha_*_delta_*",
                "oracle_*",
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
    with (output_dir / "v17_fulltrain_a0_udp_feature_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    (output_dir / "v17_fulltrain_a0_udp_feature_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    readme = """# Haze4K v1.7 Full-Train A0/UDP Feature Extraction

Primary files:

- `v17_fulltrain_a0_udp_feature_table.csv`: per-image train-derived A0, UDP,
  deployable router features, and alpha-grid metrics.
- `v17_fulltrain_a0_udp_feature_summary.json`: alpha-grid summaries,
  checkpoint hashes, split contract, and feature metadata.

Locked Haze4K test touched: no.

This is a reusable calibration asset. The alpha metric columns and oracle
columns are labels/evidence only and must not be used as deployable router
features.
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
    parser.add_argument("--splits", nargs="+", default=["full_train"])
    parser.add_argument("--depth_split", default="train")
    parser.add_argument("--pad_factor", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--print_freq", type=int, default=100)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    rows, payload = evaluate(args)
    write_outputs(Path(args.output_dir), rows, payload)
    print(
        "V17_FULLTRAIN_FEATURE_EXTRACTION_OK "
        f"rows={len(rows)} status={payload['status']} output_dir={args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
