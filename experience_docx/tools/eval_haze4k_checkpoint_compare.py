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


def eval_one(mode, checkpoint, data_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model = build_net("base", "Haze4K", mode).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    dataloader = test_dataloader(data_dir, "Haze4K", batch_size=1, num_workers=0)
    factor = 32
    rows = []
    times = []

    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            input_img, label_img, name = data
            input_img = input_img.to(device)
            label_img = label_img.to(device)

            h, w = input_img.shape[2], input_img.shape[3]
            H = ((h + factor) // factor) * factor
            W = ((w + factor) // factor) * factor
            padh = H - h if h % factor != 0 else 0
            padw = W - w if w % factor != 0 else 0
            padded = f.pad(input_img, (0, padw, 0, padh), "reflect")

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

            down_ratio = max(1, round(min(H, W) / 256))
            ssim_val = ssim(
                f.adaptive_avg_pool2d(pred, (int(H / down_ratio), int(W / down_ratio))),
                f.adaptive_avg_pool2d(label_img, (int(H / down_ratio), int(W / down_ratio))),
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
                print(f"{mode} {idx + 1}/{len(dataloader)} psnr={mean_psnr:.4f}", flush=True)

    peak_mem = None
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2

    summary = {
        "mode": mode,
        "checkpoint": checkpoint,
        "count": len(rows),
        "mean_psnr": statistics.mean(row["psnr"] for row in rows),
        "mean_ssim": statistics.mean(row["ssim"] for row in rows),
        "avg_time_sec_sync": statistics.mean(times),
        "median_time_sec_sync": statistics.median(times),
        "peak_cuda_mem_mib": peak_mem,
    }
    return rows, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--original_checkpoint", required=True)
    parser.add_argument("--modres_checkpoint")
    parser.add_argument("--candidate_checkpoint")
    parser.add_argument("--candidate_mode", default="modres")
    parser.add_argument("--candidate_name")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tag", default="seed3407")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    candidate_checkpoint = args.candidate_checkpoint or args.modres_checkpoint
    if not candidate_checkpoint:
        raise ValueError("Provide --candidate_checkpoint or --modres_checkpoint")
    candidate_name = args.candidate_name or args.candidate_mode
    runs = [
        ("original", "original", args.original_checkpoint),
        (candidate_name, args.candidate_mode, candidate_checkpoint),
    ]

    all_rows = {}
    summaries = {}
    for label, mode, checkpoint in runs:
        rows, summary = eval_one(mode, checkpoint, args.data_dir)
        summary["label"] = label
        all_rows[label] = rows
        summaries[label] = summary

    original = {row["name"]: row for row in all_rows["original"]}
    candidate = {row["name"]: row for row in all_rows[candidate_name]}
    common = [name for name in original if name in candidate]
    deltas = [candidate[name]["psnr"] - original[name]["psnr"] for name in common]
    ssim_deltas = [candidate[name]["ssim"] - original[name]["ssim"] for name in common]

    strong_cut = percentile([original[name]["psnr"] for name in common], 75)
    strong = [name for name in common if original[name]["psnr"] >= strong_cut]
    strong_regressions = [
        name for name in strong if (candidate[name]["psnr"] - original[name]["psnr"]) <= -0.05
    ]
    worst_regressions = [
        name for name in common if (candidate[name]["psnr"] - original[name]["psnr"]) <= -0.20
    ]

    tail_count = max(1, len(deltas) // 10)
    sorted_deltas = sorted(deltas)
    summary = {
        "runs": summaries,
        "comparison": {
            "common_count": len(common),
            "mean_psnr_delta": statistics.mean(deltas),
            "median_psnr_delta": statistics.median(deltas),
            "p5_psnr_delta": percentile(deltas, 5),
            "p95_psnr_delta": percentile(deltas, 95),
            "worst10_mean_psnr_delta": statistics.mean(sorted_deltas[:tail_count]),
            "best10_mean_psnr_delta": statistics.mean(sorted_deltas[-tail_count:]),
            "mean_ssim_delta": statistics.mean(ssim_deltas),
            "strong_reference_cut_psnr": strong_cut,
            "strong_reference_count": len(strong),
            "strong_regression_count_delta_le_-0.05": len(strong_regressions),
            "worst_regression_count_delta_le_-0.20": len(worst_regressions),
        },
    }

    json_path = os.path.join(args.output_dir, f"scout_eval_compare_{args.tag}.json")
    csv_path = os.path.join(args.output_dir, f"scout_eval_per_image_{args.tag}.csv")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "name",
                "original_psnr",
                f"{candidate_name}_psnr",
                "delta_psnr",
                "original_ssim",
                f"{candidate_name}_ssim",
                "delta_ssim",
                "original_time_sec",
                f"{candidate_name}_time_sec",
            ]
        )
        for name in common:
            writer.writerow(
                [
                    name,
                    original[name]["psnr"],
                    candidate[name]["psnr"],
                    candidate[name]["psnr"] - original[name]["psnr"],
                    original[name]["ssim"],
                    candidate[name]["ssim"],
                    candidate[name]["ssim"] - original[name]["ssim"],
                    original[name]["time_sec"],
                    candidate[name]["time_sec"],
                ]
            )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
