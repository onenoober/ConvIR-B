#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def add_repo_imports(its_dir):
    its_dir = os.path.abspath(its_dir)
    if its_dir not in sys.path:
        sys.path.insert(0, its_dir)


def first_existing_dir(root, names):
    for name in names:
        path = os.path.join(root, name)
        if os.path.isdir(path):
            return path
    raise FileNotFoundError(f"None of {names} found under {root}")


def label_path(label_dir, image_name):
    candidates = [image_name]
    stem, ext = os.path.splitext(image_name)
    if "_" in stem:
        candidates.append(f"{stem.split('_')[0]}{ext}")
        candidates.append(f"{stem.split('_')[0]}.png")
    for candidate in candidates:
        path = os.path.join(label_dir, candidate)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"No GT match for {image_name}; tried {candidates}")


def tensor_from_image(path, device):
    arr = np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def pad_to_factor(x, factor=32):
    h, w = x.shape[-2:]
    padded_h = ((h + factor) // factor) * factor
    padded_w = ((w + factor) // factor) * factor
    padh = padded_h - h if h % factor != 0 else 0
    padw = padded_w - w if w % factor != 0 else 0
    if padh or padw:
        x = F.pad(x, (0, padw, 0, padh), "reflect")
    return x, h, w


def load_model_state(path, device):
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        return state["model"]
    return state


def luma(x):
    return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]


def gradient_magnitude(x):
    grad_x = F.pad((x[:, :, :, 1:] - x[:, :, :, :-1]).abs(), (0, 1, 0, 0))
    grad_y = F.pad((x[:, :, 1:, :] - x[:, :, :-1, :]).abs(), (0, 0, 0, 1))
    return torch.sqrt(grad_x * grad_x + grad_y * grad_y + 1e-12)


def tail_mask_stats(input_img, anchor_img, label_img, bucket, threshold):
    gray = luma(input_img)
    grad = gradient_magnitude(gray)
    max_rgb = input_img.max(dim=1, keepdim=True).values
    min_rgb = input_img.min(dim=1, keepdim=True).values
    saturation = max_rgb - min_rgb
    anchor_error = (anchor_img - label_img).abs().mean(dim=1, keepdim=True)
    high_anchor = anchor_error <= threshold
    bright_low_gradient = (gray >= 0.62) & (grad <= 0.035)
    low_saturation_bright = (gray >= 0.58) & (saturation <= 0.12)
    sky_bright_proxy = (gray >= 0.66) & (grad <= 0.05) & (saturation <= 0.20)
    easy_image = bucket == "easy"
    legacy = high_anchor | bright_low_gradient | low_saturation_bright | sky_bright_proxy
    hard_selective = (
        bright_low_gradient
        | low_saturation_bright
        | sky_bright_proxy
        | (high_anchor if easy_image else torch.zeros_like(high_anchor))
    )
    return {
        "legacy_mask_ratio": float(legacy.float().mean().item()),
        "hard_selective_mask_ratio": float(hard_selective.float().mean().item()),
        "high_anchor_ratio": float(high_anchor.float().mean().item()),
        "sky_ratio": float(sky_bright_proxy.float().mean().item()),
        "bright_low_gradient_ratio": float(bright_low_gradient.float().mean().item()),
        "low_saturation_bright_ratio": float(low_saturation_bright.float().mean().item()),
        "anchor_error_mean": float(anchor_error.mean().item()),
    }


def rank_auc(scores, labels):
    pairs = sorted(zip(scores, labels), key=lambda item: item[0])
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return None
    rank_sum = 0.0
    idx = 0
    while idx < len(pairs):
        end = idx + 1
        while end < len(pairs) and pairs[end][0] == pairs[idx][0]:
            end += 1
        avg_rank = (idx + 1 + end) / 2.0
        rank_sum += avg_rank * sum(label for _score, label in pairs[idx:end])
        idx = end
    return (rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)


def write_split_audit(payload, output_dir):
    rows_by_name = {row["name"]: row for row in payload["rows"]}
    splits = payload["splits"]
    fieldnames = [
        "split",
        "count",
        "a0_psnr_mean",
        "a0_psnr_p25",
        "a0_psnr_p75",
        "hard_count",
        "medium_count",
        "easy_count",
        "brightness_mean",
        "saturation_mean",
        "depth_mean",
        "depth_gradient_mean",
        "sky_proxy_ratio",
    ]
    audit_rows = []
    for split in ("train_inner", "val_regular", "val_hard", "val_inner_regular", "val_inner_hard"):
        names = splits.get(split)
        if not names:
            continue
        selected = [rows_by_name[name] for name in names if name in rows_by_name]
        psnrs = [row["a0_psnr"] for row in selected]
        buckets = [row.get("a0_bucket", "medium") for row in selected]
        audit_rows.append(
            {
                "split": split,
                "count": len(selected),
                "a0_psnr_mean": statistics.mean(psnrs),
                "a0_psnr_p25": np.percentile(psnrs, 25),
                "a0_psnr_p75": np.percentile(psnrs, 75),
                "hard_count": buckets.count("hard"),
                "medium_count": buckets.count("medium"),
                "easy_count": buckets.count("easy"),
                "brightness_mean": statistics.mean(row["brightness"] for row in selected),
                "saturation_mean": statistics.mean(row["saturation"] for row in selected),
                "depth_mean": mean_optional(row.get("depth_mean") for row in selected),
                "depth_gradient_mean": mean_optional(row.get("depth_gradient") for row in selected),
                "sky_proxy_ratio": statistics.mean(row.get("sky_bright_proxy", 0) for row in selected),
            }
        )
    write_csv(output_dir / "dpga_v13_val_split_audit.csv", fieldnames, audit_rows)


def mean_optional(values):
    values = [value for value in values if value is not None]
    return statistics.mean(values) if values else None


def write_proxy_auc(payload, output_dir):
    rows = payload["rows"]
    labels = [1 if row.get("a0_bucket") == "hard" else 0 for row in rows]
    feature_fns = {
        "input_anchor_residual_proxy": lambda row: -row["a0_psnr"],
        "brightness": lambda row: row["brightness"],
        "inverse_brightness": lambda row: 1.0 - row["brightness"],
        "saturation": lambda row: row["saturation"],
        "dark_channel": lambda row: row["dark_channel"],
        "gradient": lambda row: row["gradient"],
        "depth": lambda row: row.get("depth_mean"),
        "depth_gradient": lambda row: row.get("depth_gradient"),
        "sky_proxy": lambda row: row.get("sky_bright_proxy", 0),
        "combined_hard_proxy": lambda row: (
            -row["a0_psnr"]
            + (1.0 - row["brightness"])
            + row["dark_channel"]
            + (row.get("depth_gradient") or 0.0)
        ),
    }
    auc_rows = []
    for feature, fn in feature_fns.items():
        values = [fn(row) for row in rows]
        usable = [(value, label) for value, label in zip(values, labels) if value is not None and math.isfinite(value)]
        if not usable:
            auc = None
            direction = "n/a"
            count = 0
        else:
            scores, y = zip(*usable)
            raw_auc = rank_auc(scores, y)
            if raw_auc is None:
                auc = None
                direction = "n/a"
            elif raw_auc >= 0.5:
                auc = raw_auc
                direction = "higher_is_hard"
            else:
                auc = 1.0 - raw_auc
                direction = "lower_is_hard"
            count = len(usable)
        auc_rows.append(
            {
                "feature": feature,
                "auc": auc,
                "direction": direction,
                "count": count,
                "hard_positive_count": sum(label for _score, label in usable) if usable else 0,
            }
        )
    write_csv(
        output_dir / "dpga_v13_hard_proxy_auc.csv",
        ["feature", "auc", "direction", "count", "hard_positive_count"],
        auc_rows,
    )


def write_tail_mask_audit(payload, args, output_dir):
    if not args.a0_checkpoint:
        return
    add_repo_imports(args.its_dir)
    from models.ConvIR import build_net

    rows_by_name = {row["name"]: row for row in payload["rows"]}
    split_names = []
    for split in args.mask_splits:
        for name in payload["splits"].get(split, []):
            if name in rows_by_name:
                split_names.append((split, name))
    if args.max_mask_images > 0:
        split_names = split_names[: args.max_mask_images]

    train_root = Path(args.data_dir) / "train"
    input_dir = first_existing_dir(train_root, ("IN", "haze", "hazy"))
    label_dir = first_existing_dir(train_root, ("GT", "gt"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_net("base", "Haze4K", "original").to(device)
    model.load_state_dict(load_model_state(args.a0_checkpoint, device))
    model.eval()

    per_image = []
    with torch.no_grad():
        for idx, (split, name) in enumerate(split_names):
            row = rows_by_name[name]
            input_img = tensor_from_image(os.path.join(input_dir, name), device)
            label_img = tensor_from_image(label_path(label_dir, name), device)
            padded, h, w = pad_to_factor(input_img)
            anchor = torch.clamp(model(padded)[2][:, :, :h, :w], 0, 1)
            stats = tail_mask_stats(input_img, anchor, label_img, row.get("a0_bucket", "medium"), args.anchor_error_threshold)
            per_image.append({"split": split, "name": name, "bucket": row.get("a0_bucket", "medium"), **stats})
            if (idx + 1) % 100 == 0:
                print(f"mask_audit {idx + 1}/{len(split_names)}", flush=True)

    grouped = defaultdict(list)
    for row in per_image:
        grouped[(row["split"], row["bucket"])].append(row)
    summary_rows = []
    for (split, bucket), items in sorted(grouped.items()):
        summary = {"split": split, "bucket": bucket, "count": len(items), "epoch": "pretrain_a0"}
        for key in (
            "legacy_mask_ratio",
            "hard_selective_mask_ratio",
            "high_anchor_ratio",
            "sky_ratio",
            "bright_low_gradient_ratio",
            "low_saturation_bright_ratio",
            "anchor_error_mean",
        ):
            summary[key] = statistics.mean(item[key] for item in items)
        summary_rows.append(summary)
    write_csv(
        output_dir / "dpga_v13_tail_mask_audit_by_bucket.csv",
        [
            "split",
            "bucket",
            "epoch",
            "count",
            "legacy_mask_ratio",
            "hard_selective_mask_ratio",
            "high_anchor_ratio",
            "sky_ratio",
            "bright_low_gradient_ratio",
            "low_saturation_bright_ratio",
            "anchor_error_mean",
        ],
        summary_rows,
    )


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--its_dir", default="Dehazing/ITS")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--split_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--a0_checkpoint", default="")
    parser.add_argument("--anchor_error_threshold", type=float, default=0.035)
    parser.add_argument(
        "--mask_splits",
        nargs="+",
        default=["val_regular", "val_hard"],
    )
    parser.add_argument("--max_mask_images", type=int, default=0)
    args = parser.parse_args()

    payload = json.loads(Path(args.split_json).read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_split_audit(payload, output_dir)
    write_proxy_auc(payload, output_dir)
    write_tail_mask_audit(payload, args, output_dir)
    manifest = {
        "split_json": args.split_json,
        "outputs": [
            "dpga_v13_val_split_audit.csv",
            "dpga_v13_hard_proxy_auc.csv",
            "dpga_v13_tail_mask_audit_by_bucket.csv" if args.a0_checkpoint else None,
        ],
    }
    (output_dir / "dpga_v13_intermediate_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
