#!/usr/bin/env python3
"""Haze4K v1.8 data/domain preflight audit.

This is a metadata and lightweight image-stat audit for train-derived splits.
It does not train, evaluate checkpoints, or touch locked Haze4K test data.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


IMG_EXT = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def to_float(value: Any, default: float | None = None) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def read_split_json(path: Path) -> dict[str, list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("splits", payload)


def parse_filename_params(name: str) -> dict[str, Any]:
    stem = Path(name).stem
    values = []
    for part in stem.split("_")[1:]:
        value = to_float(part)
        if value is not None:
            values.append(value)
    return {
        "filename_param_1": values[0] if len(values) > 0 else "",
        "filename_param_2": values[1] if len(values) > 1 else "",
    }


def image_stats(path: Path) -> dict[str, float]:
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    rgb_max = arr.max(axis=2)
    rgb_min = arr.min(axis=2)
    luma = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    saturation = rgb_max - rgb_min
    grad_x = np.abs(np.diff(luma, axis=1, append=luma[:, -1:]))
    grad_y = np.abs(np.diff(luma, axis=0, append=luma[-1:, :]))
    grad = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    return {
        "luma_mean": float(luma.mean()),
        "luma_std": float(luma.std()),
        "saturation_mean": float(saturation.mean()),
        "gradient_mean": float(grad.mean()),
        "bright_low_grad_ratio": float(((luma >= 0.62) & (grad <= 0.035)).mean()),
        "low_sat_bright_ratio": float(((luma >= 0.58) & (saturation <= 0.12)).mean()),
        "sky_proxy_ratio": float(((luma >= 0.66) & (grad <= 0.05) & (saturation <= 0.20)).mean()),
    }


def gray_stats(path: Path) -> dict[str, float]:
    arr = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    return {
        "trans_mean": float(arr.mean()),
        "trans_std": float(arr.std()),
        "trans_min": float(arr.min()),
        "trans_max": float(arr.max()),
        "trans_low_lt_0p33": float((arr < 0.33).mean()),
        "trans_high_gt_0p66": float((arr > 0.66).mean()),
    }


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


def summarize(rows: list[dict[str, Any]], split: str) -> dict[str, Any]:
    out = {"split": split, "count": len(rows)}
    numeric_keys = [
        "filename_param_1",
        "filename_param_2",
        "luma_mean",
        "luma_std",
        "saturation_mean",
        "gradient_mean",
        "bright_low_grad_ratio",
        "low_sat_bright_ratio",
        "sky_proxy_ratio",
        "trans_mean",
        "trans_std",
        "trans_low_lt_0p33",
        "trans_high_gt_0p66",
        "a0_psnr",
        "udp_delta_psnr",
        "oracle_best_alpha_delta_psnr",
    ]
    for key in numeric_keys:
        values = [float(row[key]) for row in rows if row.get(key) not in ("", None)]
        out[f"{key}_mean"] = mean(values)
        out[f"{key}_p10"] = percentile(values, 10)
        out[f"{key}_p90"] = percentile(values, 90)
    bucket_counts: dict[str, int] = {}
    for row in rows:
        bucket = str(row.get("bucket") or "unknown")
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
    out["bucket_counts"] = bucket_counts
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--split_json", required=True)
    parser.add_argument("--feature_csv", default="")
    parser.add_argument("--splits", nargs="+", default=["train_inner", "val_regular", "val_hard"])
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_images_per_split", type=int, default=0)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    split_json = Path(args.split_json)
    splits = read_split_json(split_json)
    feature_by_name: dict[str, dict[str, Any]] = {}
    if args.feature_csv:
        with Path(args.feature_csv).open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                feature_by_name[row["name"]] = row

    rows: list[dict[str, Any]] = []
    missing = []
    for split in args.splits:
        names = list(splits[split])
        if args.max_images_per_split > 0:
            names = names[: args.max_images_per_split]
        for name in names:
            haze_path = data_dir / "train" / "haze" / name
            gt_path = data_dir / "train" / "gt" / name
            trans_path = data_dir / "train" / "trans" / name
            if not haze_path.is_file():
                missing.append({"split": split, "name": name, "missing": str(haze_path)})
                continue
            row: dict[str, Any] = {
                "split": split,
                "name": name,
                "haze_exists": haze_path.is_file(),
                "gt_exists": gt_path.is_file(),
                "trans_exists": trans_path.is_file(),
            }
            row.update(parse_filename_params(name))
            row.update(image_stats(haze_path))
            if trans_path.is_file():
                row.update(gray_stats(trans_path))
            feat = feature_by_name.get(name, {})
            if feat:
                row["a0_psnr"] = feat.get("a0_psnr", "")
                row["udp_delta_psnr"] = feat.get("delta_psnr", "")
                row["oracle_best_alpha_delta_psnr"] = feat.get("oracle_best_alpha_delta_psnr", "")
                row["bucket"] = feat.get("bucket", "")
            rows.append(row)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with (output_dir / "v18_domain_data_preflight_per_image.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summaries = [summarize([row for row in rows if row["split"] == split], split) for split in args.splits]
    payload = {
        "route": "ConvIR-Dehaze-v1.8-ExecutionQueue",
        "stage": "domain data preflight",
        "status": "COMPLETED_DATA_PREFLIGHT",
        "locked_test_touched": False,
        "data_dir": str(data_dir),
        "split_json": str(split_json),
        "feature_csv": args.feature_csv,
        "splits": args.splits,
        "row_count": len(rows),
        "missing_count": len(missing),
        "missing": missing[:20],
        "summaries": summaries,
        "decision": "DOMAIN_PREFLIGHT_COMPLETE_CONTINUE_EXPERIMENT_QUEUE",
        "outputs": {
            "per_image": "v18_domain_data_preflight_per_image.csv",
            "summary": "v18_domain_data_preflight_summary.json",
        },
    }
    (output_dir / "v18_domain_data_preflight_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
