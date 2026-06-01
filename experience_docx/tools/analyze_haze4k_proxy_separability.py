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


def percentile(values, pct):
    ordered = sorted(values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * pct / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def pearson(xs, ys):
    if len(xs) < 2:
        return None
    x_mean = statistics.mean(xs)
    y_mean = statistics.mean(ys)
    x_var = sum((x - x_mean) ** 2 for x in xs)
    y_var = sum((y - y_mean) ** 2 for y in ys)
    if x_var == 0.0 or y_var == 0.0:
        return None
    cov = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    return cov / math.sqrt(x_var * y_var)


def auc_hard_over_easy(hard_values, easy_values):
    if not hard_values or not easy_values:
        return None
    wins = 0.0
    total = 0
    for hard_value in hard_values:
        for easy_value in easy_values:
            total += 1
            if hard_value > easy_value:
                wins += 1.0
            elif hard_value == easy_value:
                wins += 0.5
    return wins / total


def load_compare_rows(compare_csv):
    rows = {}
    with Path(compare_csv).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            rows[raw["name"]] = {
                "name": raw["name"],
                "original_psnr": float(raw["original_psnr"]),
                "delta_psnr": float(raw["delta_psnr"]) if raw.get("delta_psnr") else None,
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


def pad_to_factor(input_img, factor=32):
    h, w = input_img.shape[2], input_img.shape[3]
    H = ((h + factor) // factor) * factor
    W = ((w + factor) // factor) * factor
    padh = H - h if h % factor != 0 else 0
    padw = W - w if w % factor != 0 else 0
    return F.pad(input_img, (0, padw, 0, padh), "reflect"), h, w


def input_proxy_stats(input_img):
    red = input_img[:, 0:1]
    green = input_img[:, 1:2]
    blue = input_img[:, 2:3]
    gray = 0.299 * red + 0.587 * green + 0.114 * blue
    local_mean = F.avg_pool2d(gray, kernel_size=7, stride=1, padding=3)
    local_sq_mean = F.avg_pool2d(gray * gray, kernel_size=7, stride=1, padding=3)
    local_std = (local_sq_mean - local_mean * local_mean).clamp_min(0.0).sqrt()
    max_rgb = input_img.max(dim=1, keepdim=True).values
    min_rgb = input_img.min(dim=1, keepdim=True).values
    saturation = (max_rgb - min_rgb) / max_rgb.clamp_min(1e-6)

    return {
        "input_brightness_mean": input_img.mean().item(),
        "input_gray_std": gray.std(unbiased=False).item(),
        "input_local_contrast_mean": local_std.mean().item(),
        "input_dark_channel_mean": min_rgb.mean().item(),
        "input_dark_channel_p95": torch.quantile(min_rgb.flatten(), 0.95).item(),
        "input_saturation_mean": saturation.mean().item(),
        "input_saturation_std": saturation.std(unbiased=False).item(),
    }


def baseline_proxy_stats(model, input_img):
    padded, h, w = pad_to_factor(input_img)
    outputs = model(padded)
    final_padded = outputs[2]
    final = final_padded[:, :, :h, :w]
    consistency_64 = (
        F.interpolate(final_padded, size=outputs[0].shape[-2:], mode="bilinear", align_corners=False)
        - outputs[0]
    ).abs().mean()
    consistency_128 = (
        F.interpolate(final_padded, size=outputs[1].shape[-2:], mode="bilinear", align_corners=False)
        - outputs[1]
    ).abs().mean()
    residual = final - input_img
    return {
        "baseline_residual_abs_mean": residual.abs().mean().item(),
        "baseline_residual_abs_p95": torch.quantile(residual.abs().flatten(), 0.95).item(),
        "baseline_multiscale_consistency_mean": ((consistency_64 + consistency_128) * 0.5).item(),
    }


def summarize_proxy(rows, key):
    hard = [row[key] for row in rows if row["bucket"] == "hard_bottom_25pct"]
    easy = [row[key] for row in rows if row["bucket"] == "easy_top_25pct"]
    all_values = [row[key] for row in rows]
    original_psnr = [row["original_psnr"] for row in rows]
    delta_values = [row["delta_psnr"] for row in rows if row["delta_psnr"] is not None]
    delta_proxy = [row[key] for row in rows if row["delta_psnr"] is not None]
    auc = auc_hard_over_easy(hard, easy)
    return {
        "proxy": key,
        "hard_mean": statistics.mean(hard),
        "easy_mean": statistics.mean(easy),
        "hard_minus_easy": statistics.mean(hard) - statistics.mean(easy),
        "hard_over_easy_auc": auc,
        "best_direction_auc": None if auc is None else max(auc, 1.0 - auc),
        "direction": "higher_on_hard" if auc is not None and auc >= 0.5 else "lower_on_hard",
        "p25": percentile(all_values, 25),
        "p50": percentile(all_values, 50),
        "p75": percentile(all_values, 75),
        "pearson_vs_original_psnr": pearson(all_values, original_psnr),
        "pearson_vs_delta_psnr": pearson(delta_proxy, delta_values) if delta_values else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--compare_csv", required=True)
    parser.add_argument("--baseline_checkpoint", default="")
    parser.add_argument("--baseline_mode", default="original")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--max_images", type=int, default=0)
    args = parser.parse_args()

    compare_rows = load_compare_rows(args.compare_csv)
    buckets = bucket_by_original_psnr(compare_rows)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = None
    if args.baseline_checkpoint:
        model = build_net("base", "Haze4K", args.baseline_mode).to(device)
        state = torch.load(args.baseline_checkpoint, map_location=device)
        model.load_state_dict(state["model"] if "model" in state else state)
        model.eval()

    dataloader = test_dataloader(args.data_dir, "Haze4K", batch_size=1, num_workers=0)
    rows = []
    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            if args.max_images > 0 and len(rows) >= args.max_images:
                break
            input_img, _label_img, name = data
            name = name[0]
            if name not in compare_rows:
                continue
            input_img = input_img.to(device)
            row = {
                **compare_rows[name],
                "bucket": buckets[name],
                **input_proxy_stats(input_img),
            }
            if model is not None:
                row.update(baseline_proxy_stats(model, input_img))
            rows.append(row)
            if (idx + 1) % 100 == 0:
                print(f"proxy {idx + 1}/{len(dataloader)}", flush=True)

    if not rows:
        raise ValueError("No proxy rows were collected")

    proxy_keys = [
        key for key in rows[0]
        if key not in ("name", "bucket", "original_psnr", "delta_psnr")
    ]
    proxy_summaries = [summarize_proxy(rows, key) for key in proxy_keys]
    proxy_summaries.sort(key=lambda item: item["best_direction_auc"] or 0.0, reverse=True)
    summary = {
        "source_compare_csv": args.compare_csv,
        "baseline_checkpoint": args.baseline_checkpoint,
        "count": len(rows),
        "bucket_counts": {
            bucket: sum(row["bucket"] == bucket for row in rows)
            for bucket in ("hard_bottom_25pct", "medium_middle_50pct", "easy_top_25pct")
        },
        "proxy_summaries": proxy_summaries,
        "best_proxy": proxy_summaries[0] if proxy_summaries else None,
        "note": "A useful deployable gate proxy should separate hard bottom-25% from easy top-25% without using GT at inference.",
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
            "delta_psnr",
            *proxy_keys,
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(summary, indent=2))
    print(f"wrote {output_json}")
    print(f"wrote {output_csv}")


if __name__ == "__main__":
    main()
