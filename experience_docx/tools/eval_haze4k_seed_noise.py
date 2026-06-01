import argparse
import csv
import json
import math
import os
import statistics
import sys
import time

import torch
import torch.nn.functional as f
from pytorch_msssim import ssim

sys.path.insert(0, os.getcwd())

from data import test_dataloader
from models.ConvIR import build_net


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


def sample_std(values):
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


def summarize_values(values):
    return {
        "count": len(values),
        "mean": statistics.mean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "sample_std": sample_std(values),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "range": (max(values) - min(values)) if values else None,
        "p5": percentile(values, 5),
        "p95": percentile(values, 95),
    }


def parse_run(raw):
    if ":" not in raw:
        raise argparse.ArgumentTypeError("--run must be formatted as SEED:CHECKPOINT")
    seed, checkpoint = raw.split(":", 1)
    if not seed:
        raise argparse.ArgumentTypeError("Run seed cannot be empty")
    if not checkpoint:
        raise argparse.ArgumentTypeError("Run checkpoint cannot be empty")
    return seed, checkpoint


def eval_one(mode, checkpoint, data_dir, num_workers):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model = build_net("base", "Haze4K", mode).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()

    dataloader = test_dataloader(data_dir, "Haze4K", batch_size=1, num_workers=num_workers)
    factor = 32
    rows = []
    times = []

    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            input_img, label_img, name = data
            input_img = input_img.to(device)
            label_img = label_img.to(device)

            h, w = input_img.shape[2], input_img.shape[3]
            padded_h = ((h + factor) // factor) * factor
            padded_w = ((w + factor) // factor) * factor
            pad_h = padded_h - h if h % factor != 0 else 0
            pad_w = padded_w - w if w % factor != 0 else 0
            padded = f.pad(input_img, (0, pad_w, 0, pad_h), "reflect")

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.time()
            pred = model(padded)[2][:, :, :h, :w]
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.time() - start

            pred = torch.clamp(pred, 0, 1)
            mse = f.mse_loss(pred, label_img)
            psnr_val = (10 * torch.log10(1 / mse)).item()

            down_ratio = max(1, round(min(padded_h, padded_w) / 256))
            ssim_val = ssim(
                f.adaptive_avg_pool2d(pred, (int(padded_h / down_ratio), int(padded_w / down_ratio))),
                f.adaptive_avg_pool2d(label_img, (int(padded_h / down_ratio), int(padded_w / down_ratio))),
                data_range=1,
                size_average=False,
            ).mean().item()

            times.append(elapsed)
            rows.append(
                {
                    "name": name[0],
                    "psnr": psnr_val,
                    "ssim": ssim_val,
                    "time_sec": elapsed,
                }
            )
            if (idx + 1) % 100 == 0:
                mean_psnr = statistics.mean(row["psnr"] for row in rows)
                print(f"{mode} {checkpoint} {idx + 1}/{len(dataloader)} psnr={mean_psnr:.4f}", flush=True)

    peak_mem = None
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2

    return rows, {
        "mode": mode,
        "checkpoint": checkpoint,
        "count": len(rows),
        "mean_psnr": statistics.mean(row["psnr"] for row in rows),
        "mean_ssim": statistics.mean(row["ssim"] for row in rows),
        "avg_time_sec_sync": statistics.mean(times),
        "median_time_sec_sync": statistics.median(times),
        "peak_cuda_mem_mib": peak_mem,
    }


def assign_buckets(reference_rows):
    ordered = sorted(reference_rows, key=lambda row: row["psnr"])
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


def summarize_bucket(seed_rows, names):
    selected = [seed_rows[name] for name in names]
    return {
        "count": len(selected),
        "mean_psnr": statistics.mean(row["psnr"] for row in selected),
        "median_psnr": statistics.median(row["psnr"] for row in selected),
        "mean_ssim": statistics.mean(row["ssim"] for row in selected),
        "median_ssim": statistics.median(row["ssim"] for row in selected),
    }


def summarize_cross_seed(seed_summaries, key):
    return summarize_values([summary[key] for summary in seed_summaries.values()])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--mode", default="original")
    parser.add_argument("--run", action="append", required=True, type=parse_run, help="SEED:CHECKPOINT")
    parser.add_argument("--reference_seed", default="")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    runs = args.run
    seeds = [seed for seed, _checkpoint in runs]
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"Duplicate seeds in runs: {seeds}")
    reference_seed = args.reference_seed or seeds[0]
    if reference_seed not in set(seeds):
        raise ValueError(f"reference_seed {reference_seed} not found in runs: {seeds}")

    all_rows = {}
    summaries = {}
    for seed, checkpoint in runs:
        rows, summary = eval_one(args.mode, checkpoint, args.data_dir, args.num_workers)
        summary["seed"] = seed
        all_rows[seed] = {row["name"]: row for row in rows}
        summaries[seed] = summary

    common_names = sorted(set.intersection(*(set(rows.keys()) for rows in all_rows.values())))
    if not common_names:
        raise ValueError("No common image names across evaluated seeds")

    reference_rows = [all_rows[reference_seed][name] for name in common_names]
    buckets = assign_buckets(reference_rows)
    bucket_names = {
        bucket: [name for name in common_names if buckets[name] == bucket]
        for bucket in ("hard_bottom_25pct", "medium_middle_50pct", "easy_top_25pct")
    }

    per_seed_buckets = {}
    for seed, rows in all_rows.items():
        per_seed_buckets[seed] = {
            bucket: summarize_bucket(rows, names)
            for bucket, names in bucket_names.items()
        }

    bucket_noise = {}
    for bucket in bucket_names:
        bucket_noise[bucket] = {
            "mean_psnr_across_seeds": summarize_values(
                [per_seed_buckets[seed][bucket]["mean_psnr"] for seed in seeds]
            ),
            "median_psnr_across_seeds": summarize_values(
                [per_seed_buckets[seed][bucket]["median_psnr"] for seed in seeds]
            ),
            "mean_ssim_across_seeds": summarize_values(
                [per_seed_buckets[seed][bucket]["mean_ssim"] for seed in seeds]
            ),
        }

    per_image_rows = []
    per_image_psnr_std = []
    per_image_ssim_std = []
    per_image_std_by_bucket = {bucket: [] for bucket in bucket_names}
    for name in common_names:
        psnrs = [all_rows[seed][name]["psnr"] for seed in seeds]
        ssims = [all_rows[seed][name]["ssim"] for seed in seeds]
        psnr_std = sample_std(psnrs)
        ssim_std = sample_std(ssims)
        bucket = buckets[name]
        per_image_psnr_std.append(psnr_std)
        per_image_ssim_std.append(ssim_std)
        per_image_std_by_bucket[bucket].append(psnr_std)
        per_image_rows.append(
            {
                "name": name,
                "bucket": bucket,
                "psnr_mean": statistics.mean(psnrs),
                "psnr_sample_std": psnr_std,
                "psnr_range": max(psnrs) - min(psnrs),
                "ssim_mean": statistics.mean(ssims),
                "ssim_sample_std": ssim_std,
                **{f"seed_{seed}_psnr": all_rows[seed][name]["psnr"] for seed in seeds},
                **{f"seed_{seed}_ssim": all_rows[seed][name]["ssim"] for seed in seeds},
            }
        )

    summary = {
        "mode": args.mode,
        "reference_seed": reference_seed,
        "seeds": seeds,
        "common_count": len(common_names),
        "runs": summaries,
        "overall_noise": {
            "mean_psnr_across_seeds": summarize_cross_seed(summaries, "mean_psnr"),
            "mean_ssim_across_seeds": summarize_cross_seed(summaries, "mean_ssim"),
            "avg_time_sec_across_seeds": summarize_cross_seed(summaries, "avg_time_sec_sync"),
        },
        "bucket_counts": {bucket: len(names) for bucket, names in bucket_names.items()},
        "per_seed_buckets": per_seed_buckets,
        "bucket_noise": bucket_noise,
        "per_image_noise": {
            "psnr_sample_std": summarize_values(per_image_psnr_std),
            "ssim_sample_std": summarize_values(per_image_ssim_std),
            "psnr_sample_std_by_bucket": {
                bucket: summarize_values(values) for bucket, values in per_image_std_by_bucket.items()
            },
        },
        "interpretation_thresholds": {
            "stable_stop20": "overall mean PSNR sample_std <= 0.10 dB",
            "underpowered_stop20": "overall mean PSNR sample_std >= 0.30 dB or range >= 0.60 dB",
            "candidate_claim_rule": "single-seed candidate deltas smaller than the measured noise floor should stay diagnostic-only",
        },
    }

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as handle:
        fieldnames = list(per_image_rows[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_image_rows)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_csv}")


if __name__ == "__main__":
    main()
