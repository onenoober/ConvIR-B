#!/usr/bin/env python3
"""One-shot fixed-policy locked-test eval for v1.6 A0+UDP expert switch."""

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
from extract_haze4k_udp_switch_features import image_features, output_diff_features, parse_filename_features  # noqa: E402


def mean(values: list[float]) -> float | None:
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


def choose(feature_value: float, direction: str, threshold: float) -> bool:
    if direction == "low":
        return feature_value <= threshold
    if direction == "high":
        return feature_value >= threshold
    raise ValueError(f"Unknown direction: {direction}")


def mark_buckets(rows: list[dict[str, Any]]) -> None:
    ordered = sorted(rows, key=lambda row: float(row["a0_psnr"]))
    n = max(1, len(ordered) // 4)
    hard = {row["name"] for row in ordered[:n]}
    easy = {row["name"] for row in ordered[-n:]}
    for row in rows:
        if row["name"] in hard:
            row["bucket"] = "hard_bottom25_by_a0"
        elif row["name"] in easy:
            row["bucket"] = "easy_top25_by_a0"
        else:
            row["bucket"] = "mid_by_a0"


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    deltas = [float(row["switch_delta_psnr"]) for row in rows]
    ssim_deltas = [float(row["switch_delta_ssim"]) for row in rows]
    hard = [row for row in rows if row.get("bucket") == "hard_bottom25_by_a0"]
    easy = [row for row in rows if row.get("bucket") == "easy_top25_by_a0"]
    strong_cut = percentile([float(row["a0_psnr"]) for row in rows], 75)
    strong = [row for row in rows if strong_cut is not None and float(row["a0_psnr"]) >= strong_cut]
    strong_reg = [row for row in strong if float(row["switch_delta_psnr"]) <= -0.05]
    worst = [row for row in rows if float(row["switch_delta_psnr"]) <= -0.20]
    tail_n = max(1, len(rows) // 10)
    ordered = sorted(deltas)
    return {
        "count": len(rows),
        "coverage": sum(1 for row in rows if row["choose_udp"]) / max(1, len(rows)),
        "udp_accept_count": sum(1 for row in rows if row["choose_udp"]),
        "mean_delta": mean(deltas),
        "median_delta": statistics.median(deltas) if deltas else None,
        "p5_delta": percentile(deltas, 5),
        "p95_delta": percentile(deltas, 95),
        "hard_bottom25_delta": mean([float(row["switch_delta_psnr"]) for row in hard]),
        "easy_top25_delta": mean([float(row["switch_delta_psnr"]) for row in easy]),
        "best10pct_delta": mean(ordered[-tail_n:]),
        "worst10pct_delta": mean(ordered[:tail_n]),
        "mean_ssim_delta": mean(ssim_deltas),
        "positive_ratio": sum(delta > 0 for delta in deltas) / max(1, len(deltas)),
        "strong_reference_cut_psnr": strong_cut,
        "strong_regression_count": len(strong_reg),
        "strong_regression_ratio": len(strong_reg) / max(1, len(strong)),
        "worst_regression_count": len(worst),
        "worst_regression_ratio": len(worst) / max(1, len(rows)),
    }


def gate(summary: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "mean_delta_ge_0_15": summary["mean_delta"] is not None and summary["mean_delta"] >= 0.15,
        "hard_bottom25_ge_0_30": summary["hard_bottom25_delta"] is not None
        and summary["hard_bottom25_delta"] >= 0.30,
        "easy_top25_ge_neg_0_03": summary["easy_top25_delta"] is not None
        and summary["easy_top25_delta"] >= -0.03,
        "ssim_nonnegative": summary["mean_ssim_delta"] is not None and summary["mean_ssim_delta"] >= 0,
        "worst_ratio_le_0_05": summary["worst_regression_ratio"] <= 0.05,
        "strong_ratio_le_0_16": summary["strong_regression_ratio"] <= 0.16,
    }
    return {
        "thresholds": {
            "mean_delta_min": 0.15,
            "hard_bottom25_min": 0.30,
            "easy_top25_min": -0.03,
            "mean_ssim_delta_min": 0.0,
            "worst_regression_ratio_max": 0.05,
            "strong_regression_ratio_max": 0.16,
        },
        "checks": checks,
        "promotion_gate_pass": all(checks.values()),
    }


def evaluate(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    test_dataloader, build_convir_net = load_convir_builders(Path(args.convir_its_dir))
    build_udpnet = load_udpnet_builder(Path(args.udp_repo))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a0_model = load_a0_model(build_convir_net, Path(args.a0_checkpoint), device)
    udp_model, ckpt_meta = load_udpnet_model(build_udpnet, Path(args.official_checkpoint), device)
    factor = int(args.pad_factor)
    rows: list[dict[str, Any]] = []
    dataloader = test_dataloader(
        args.data_dir,
        "Haze4K",
        batch_size=1,
        num_workers=args.num_workers,
        depth_cache_dir=args.depth_cache_dir,
        depth_split=args.depth_split,
        split_json="",
        split_name="",
    )
    with torch.no_grad():
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
            features: dict[str, Any] = {}
            features.update(parse_filename_features(image_name))
            features.update(image_features(input_img, depth))
            features.update(output_diff_features(a0_pred, udp_pred))
            if args.feature not in features:
                raise KeyError(f"Feature {args.feature!r} not found; available example keys include {sorted(features)[:20]}")
            feature_value = float(features[args.feature])
            use_udp = choose(feature_value, args.direction, float(args.threshold))
            a0_psnr, a0_ssim = metric_pair(a0_pred, label_img, (h_pad, w_pad))
            udp_psnr, udp_ssim = metric_pair(udp_pred, label_img, (h_pad, w_pad))
            switch_psnr = udp_psnr if use_udp else a0_psnr
            switch_ssim = udp_ssim if use_udp else a0_ssim
            row = {
                "name": image_name,
                "split": "locked_test",
                "feature": args.feature,
                "direction": args.direction,
                "threshold": args.threshold,
                "feature_value": feature_value,
                "choose_udp": use_udp,
                "a0_psnr": a0_psnr,
                "udpnet_psnr": udp_psnr,
                "switch_psnr": switch_psnr,
                "udpnet_delta_psnr": udp_psnr - a0_psnr,
                "switch_delta_psnr": switch_psnr - a0_psnr,
                "a0_ssim": a0_ssim,
                "udpnet_ssim": udp_ssim,
                "switch_ssim": switch_ssim,
                "udpnet_delta_ssim": udp_ssim - a0_ssim,
                "switch_delta_ssim": switch_ssim - a0_ssim,
            }
            rows.append(row)
            if (idx + 1) % args.print_freq == 0:
                print(
                    f"locked_test {idx + 1}/{len(dataloader)} "
                    f"accept={sum(1 for row in rows if row['choose_udp'])} "
                    f"mean_delta={statistics.mean(float(row['switch_delta_psnr']) for row in rows):.4f}",
                    flush=True,
                )
            if args.max_images and idx + 1 >= args.max_images:
                break
    mark_buckets(rows)
    summary = summarize(rows)
    gate_result = gate(summary)
    payload = {
        "route": "ConvIR-Dehaze-v1.6-RCExpertSwitch",
        "stage": "fixed one-shot locked Haze4K test confirmation",
        "status": "LOCKED_TEST_COMPLETE",
        "locked_test_touched": True,
        "fixed_policy": {
            "expert_bank": "A0 fallback + official UDPNet ConvIR",
            "feature": args.feature,
            "direction": args.direction,
            "threshold": float(args.threshold),
            "fallback": "A0",
        },
        "data_dir": args.data_dir,
        "depth_cache_dir": args.depth_cache_dir,
        "depth_split": args.depth_split,
        "a0_checkpoint_path": args.a0_checkpoint,
        "a0_checkpoint_sha256": sha256_file(Path(args.a0_checkpoint)),
        "official_checkpoint_path": args.official_checkpoint,
        "official_checkpoint_sha256": sha256_file(Path(args.official_checkpoint)),
        "official_checkpoint_meta": ckpt_meta,
        "summary": summary,
        "gate": gate_result,
        "decision": "LOCKED_TEST_PASS" if gate_result["promotion_gate_pass"] else "LOCKED_TEST_FAIL_NO_FURTHER_SELECTION",
    }
    return rows, payload


def write_outputs(output_dir: Path, rows: list[dict[str, Any]], payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with (output_dir / "rcswitch_locked_test_per_image.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    failures = [
        {
            "name": row["name"],
            "bucket": row["bucket"],
            "choose_udp": row["choose_udp"],
            "switch_delta_psnr": row["switch_delta_psnr"],
            "switch_delta_ssim": row["switch_delta_ssim"],
            "reason": "switch_delta_psnr<=-0.20 or strong/easy regression",
        }
        for row in rows
        if float(row["switch_delta_psnr"]) <= -0.20
        or (row.get("bucket") == "easy_top25_by_a0" and float(row["switch_delta_psnr"]) <= -0.05)
    ]
    with (output_dir / "rcswitch_locked_test_failure_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        fields_fail = ["name", "bucket", "choose_udp", "switch_delta_psnr", "switch_delta_ssim", "reason"]
        writer = csv.DictWriter(handle, fieldnames=fields_fail, lineterminator="\n")
        writer.writeheader()
        writer.writerows(failures)
    (output_dir / "rcswitch_locked_test_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    readme = f"""# Haze4K v1.6 RCExpertSwitch Locked Test

Status: `{payload['status']}`.

Decision: `{payload['decision']}`.

Locked test touched: yes.

Fixed policy:

```text
feature = {payload['fixed_policy']['feature']}
direction = {payload['fixed_policy']['direction']}
threshold = {payload['fixed_policy']['threshold']}
fallback = A0
expert = official UDPNet ConvIR
```

This directory is a one-shot confirmation for the fixed internal policy. Do not
use these results to change threshold, feature, checkpoint, or expert set.
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
    parser.add_argument("--depth_split", default="test")
    parser.add_argument("--feature", required=True)
    parser.add_argument("--direction", required=True, choices=["low", "high"])
    parser.add_argument("--threshold", required=True, type=float)
    parser.add_argument("--pad_factor", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--print_freq", type=int, default=100)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    rows, payload = evaluate(args)
    write_outputs(Path(args.output_dir), rows, payload)
    print(
        "RCSWITCH_LOCKED_TEST_OK "
        f"decision={payload['decision']} "
        f"mean_delta={payload['summary']['mean_delta']:.6f} "
        f"output_dir={args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
