#!/usr/bin/env python3
"""Aggregate DTA-v3 depth-control compare outputs into matrix/attribution files."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def compare_json(compare_dir: Path) -> Path:
    matches = sorted(compare_dir.glob("scout_eval_compare_*.json"))
    if not matches:
        raise FileNotFoundError(f"No scout_eval_compare_*.json under {compare_dir}")
    return matches[0]


def compare_csv(compare_dir: Path) -> Path:
    matches = sorted(compare_dir.glob("scout_eval_per_image_*.csv"))
    if not matches:
        raise FileNotFoundError(f"No scout_eval_per_image_*.csv under {compare_dir}")
    return matches[0]


def finite_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="JSON list with label/train_depth/eval_depth/compare_dir.")
    parser.add_argument("--output_matrix_json", required=True)
    parser.add_argument("--output_matrix_csv", required=True)
    parser.add_argument("--output_attribution_csv", required=True)
    parser.add_argument("--baseline_label", default="zero", help="Label used as R0/zero baseline for per-image surplus.")
    parser.add_argument("--true_label", default="true", help="Label used as true-depth candidate for per-image surplus.")
    args = parser.parse_args()

    manifest = read_json(Path(args.manifest))
    if isinstance(manifest, dict):
        manifest = manifest.get("runs", [])
    matrix_rows = []
    per_image: dict[str, dict[str, float]] = {}
    for item in manifest:
        label = item["label"]
        compare_dir = Path(item["compare_dir"])
        summary = read_json(compare_json(compare_dir))["comparison"]
        row = {
            "label": label,
            "train_depth": item.get("train_depth", ""),
            "eval_depth": item.get("eval_depth", ""),
            "compare_dir": str(compare_dir),
        }
        row.update(summary)
        matrix_rows.append(row)
        for csv_row in read_rows(compare_csv(compare_dir)):
            name = csv_row["name"]
            per_image.setdefault(name, {})[label] = finite_float(csv_row.get("delta_psnr"))

    attribution_rows = []
    surpluses = []
    for name, values in sorted(per_image.items()):
        if args.true_label not in values or args.baseline_label not in values:
            continue
        surplus = values[args.true_label] - values[args.baseline_label]
        surpluses.append(surplus)
        row = {
            "name": name,
            "true_delta_psnr": values[args.true_label],
            "baseline_delta_psnr": values[args.baseline_label],
            "depth_surplus_psnr": surplus,
        }
        for label, value in sorted(values.items()):
            row[f"{label}_delta_psnr"] = value
        attribution_rows.append(row)

    matrix = {
        "runs": matrix_rows,
        "attribution": {
            "true_label": args.true_label,
            "baseline_label": args.baseline_label,
            "common_count": len(surpluses),
            "mean_depth_surplus_psnr": statistics.mean(surpluses) if surpluses else None,
            "median_depth_surplus_psnr": statistics.median(surpluses) if surpluses else None,
            "positive_surplus_ratio": (sum(v > 0 for v in surpluses) / len(surpluses)) if surpluses else None,
        },
    }
    Path(args.output_matrix_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_matrix_json).write_text(json.dumps(matrix, indent=2), encoding="utf-8")
    write_csv(Path(args.output_matrix_csv), matrix_rows)
    write_csv(Path(args.output_attribution_csv), attribution_rows)
    print(json.dumps(matrix["attribution"], indent=2))
    print("DTA_V3_CONTROL_AGGREGATE_OK")


if __name__ == "__main__":
    main()
