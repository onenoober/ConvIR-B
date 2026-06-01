import argparse
import csv
import json
import math
import os
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, os.getcwd())

from data import test_dataloader
from models.ConvIR import build_net


MOD_KEYS = [
    "gamma_mean",
    "gamma_abs_mean",
    "gamma_std",
    "gamma_min",
    "gamma_max",
    "gamma_abs_gt_0.05",
    "gamma_abs_gt_0.10",
    "gamma_abs_gt_0.09",
    "gate_mean",
    "gate_std",
    "gate_min",
    "gate_max",
    "gamma_base_abs_mean",
    "effective_gamma_abs_mean",
    "beta_present",
    "beta_mean",
    "beta_abs_mean",
    "beta_std",
    "beta_min",
    "beta_max",
    "beta_abs_gt_0.02",
    "beta_abs_gt_0.05",
    "beta_abs_gt_0.1",
]


def percentile(values, pct):
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


def load_compare_rows(compare_csv, candidate_name):
    candidate_psnr_key = f"{candidate_name}_psnr"
    candidate_ssim_key = f"{candidate_name}_ssim"
    rows = {}
    with Path(compare_csv).open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            rows[raw["name"]] = {
                "name": raw["name"],
                "original_psnr": float(raw["original_psnr"]),
                "candidate_psnr": float(raw[candidate_psnr_key]),
                "delta_psnr": float(raw["delta_psnr"]),
                "original_ssim": float(raw["original_ssim"]),
                "candidate_ssim": float(raw[candidate_ssim_key]),
                "delta_ssim": float(raw["delta_ssim"]),
            }
    if not rows:
        raise ValueError(f"No comparison rows found in {compare_csv}")
    return rows


def bucket_by_original_psnr(compare_rows):
    ordered = sorted(compare_rows.values(), key=lambda row: row["original_psnr"])
    count = len(ordered)
    buckets = {}
    for idx, row in enumerate(ordered):
        if idx < count // 4:
            bucket = "hard_bottom_25pct"
        elif idx < 3 * count // 4:
            bucket = "medium_middle_50pct"
        else:
            bucket = "easy_top_25pct"
        buckets[row["name"]] = bucket
    return buckets


def numeric_mean(rows, key):
    values = [row[key] for row in rows if key in row]
    if not values:
        return None
    return statistics.mean(values)


def summarize(rows):
    deltas = [row["delta_psnr"] for row in rows]
    summary = {
        "count": len(rows),
        "mean_delta_psnr": statistics.mean(deltas),
        "median_delta_psnr": statistics.median(deltas),
        "p5_delta_psnr": percentile(deltas, 5),
        "p95_delta_psnr": percentile(deltas, 95),
        "regression_delta_le_-0.05": sum(delta <= -0.05 for delta in deltas),
        "regression_delta_le_-0.20": sum(delta <= -0.20 for delta in deltas),
    }
    for key in MOD_KEYS:
        values = [row[key] for row in rows if key in row]
        if not values:
            continue
        if key.endswith("_min"):
            summary[key] = min(values)
        elif key.endswith("_max"):
            summary[key] = max(values)
        else:
            summary[key] = statistics.mean(values)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--candidate_mode", required=True)
    parser.add_argument("--candidate_name", required=True)
    parser.add_argument("--compare_csv", required=True)
    parser.add_argument("--fam", default="FAM2")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--max_images", type=int, default=0)
    args = parser.parse_args()

    compare_rows = load_compare_rows(args.compare_csv, args.candidate_name)
    buckets = bucket_by_original_psnr(compare_rows)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    model = build_net("base", "Haze4K", args.candidate_mode).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()

    dataloader = test_dataloader(args.data_dir, "Haze4K", batch_size=1, num_workers=0)
    factor = 32
    rows = []
    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            if args.max_images > 0 and idx >= args.max_images:
                break
            input_img, _label_img, name = data
            name = name[0]
            if name not in compare_rows:
                continue
            input_img = input_img.to(device)
            h, w = input_img.shape[2], input_img.shape[3]
            H = ((h + factor) // factor) * factor
            W = ((w + factor) // factor) * factor
            padh = H - h if h % factor != 0 else 0
            padw = W - w if w % factor != 0 else 0
            padded = F.pad(input_img, (0, padw, 0, padh), "reflect")

            batch_stats = model.collect_modulation_stats(padded)
            if args.fam not in batch_stats:
                raise RuntimeError(f"{args.fam} not found in modulation stats for {args.candidate_mode}")
            row = {
                **compare_rows[name],
                "bucket": buckets[name],
            }
            for key in MOD_KEYS:
                row[key] = batch_stats[args.fam].get(key, 0.0)
            rows.append(row)
            if (idx + 1) % 100 == 0:
                print(f"{args.candidate_mode} modulation {idx + 1}/{len(dataloader)}", flush=True)

    if not rows:
        raise ValueError("No modulation rows were collected")

    summary = {
        "candidate_mode": args.candidate_mode,
        "candidate_name": args.candidate_name,
        "checkpoint": args.checkpoint,
        "compare_csv": args.compare_csv,
        "fam": args.fam,
        "overall": summarize(rows),
        "difficulty_buckets_by_original_psnr": {
            bucket: summarize([row for row in rows if row["bucket"] == bucket])
            for bucket in ("hard_bottom_25pct", "medium_middle_50pct", "easy_top_25pct")
        },
        "interpretation_note": (
            "Ideal bounded behavior: hard bucket has larger gamma/beta magnitude than easy bucket, "
            "while easy bucket modulation remains close to zero."
        ),
    }

    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "name",
            "bucket",
            "original_psnr",
            "candidate_psnr",
            "delta_psnr",
            "original_ssim",
            "candidate_ssim",
            "delta_ssim",
            *MOD_KEYS,
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(summary, indent=2))
    print(f"wrote {output_json}")
    print(f"wrote {output_csv}")


if __name__ == "__main__":
    main()
