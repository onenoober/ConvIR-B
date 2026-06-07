#!/usr/bin/env python3
"""Aggregate multi-seed Haze4K checkpoint-selection evidence."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def stdev(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) >= 2 else None


def ci95(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    # Normal approximation is sufficient for route monitoring; final reports
    # can rerun with exact paired bootstrap if this becomes candidate-positive.
    return 1.96 * statistics.stdev(values) / math.sqrt(len(values))


def metric_row(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [float(row[key]) for row in rows if row.get(key) not in ("", None)]
    return {
        "metric": key,
        "count": len(values),
        "mean": mean(values),
        "stdev": stdev(values),
        "ci95_half_width": ci95(values),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection_jsons", nargs="+", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()

    seed_rows = []
    for item in args.selection_jsons:
        path = Path(item)
        payload = read_json(path)
        selected_label = payload.get("selected_checkpoint_label")
        selected_detail = payload.get("details", {}).get(selected_label, {})
        row = {
            "selection_json": str(path),
            "selected_checkpoint_label": selected_label,
            "selected_score": payload.get("selected_score"),
            "selected_all_checks_pass": payload.get("selected_all_checks_pass"),
            "decision": payload.get("decision"),
        }
        for split in ("regular", "hard"):
            for key, value in selected_detail.get(split, {}).items():
                row[f"{split}_{key}"] = value
        seed_rows.append(row)

    metrics = [
        "regular_mean_psnr_delta",
        "regular_easy_top25_psnr_delta",
        "regular_mean_ssim_delta",
        "regular_worst_count_delta_le_-0p20",
        "regular_strong_regression_ratio",
        "hard_mean_psnr_delta",
        "hard_hard_bottom25_psnr_delta",
        "hard_mean_ssim_delta",
        "hard_worst_count_delta_le_-0p20",
        "hard_strong_regression_ratio",
    ]
    aggregate_rows = [metric_row(seed_rows, key) for key in metrics]
    mean_lookup = {row["metric"]: row["mean"] for row in aggregate_rows}
    ci_lookup = {row["metric"]: row["ci95_half_width"] for row in aggregate_rows}
    gate_checks = {
        "n_ge_5": len(seed_rows) >= 5,
        "regular_mean_ci_lower_ge_0p05": (
            mean_lookup["regular_mean_psnr_delta"] is not None
            and ci_lookup["regular_mean_psnr_delta"] is not None
            and mean_lookup["regular_mean_psnr_delta"] - ci_lookup["regular_mean_psnr_delta"] >= 0.05
        ),
        "hard_bottom25_ci_lower_ge_0p10": (
            mean_lookup["hard_hard_bottom25_psnr_delta"] is not None
            and ci_lookup["hard_hard_bottom25_psnr_delta"] is not None
            and mean_lookup["hard_hard_bottom25_psnr_delta"] - ci_lookup["hard_hard_bottom25_psnr_delta"] >= 0.10
        ),
        "regular_easy_mean_ge_neg0p02": (
            mean_lookup["regular_easy_top25_psnr_delta"] is not None
            and mean_lookup["regular_easy_top25_psnr_delta"] >= -0.02
        ),
        "regular_ssim_mean_ge_0": (
            mean_lookup["regular_mean_ssim_delta"] is not None
            and mean_lookup["regular_mean_ssim_delta"] >= 0
        ),
        "hard_ssim_mean_ge_0": (
            mean_lookup["hard_mean_ssim_delta"] is not None
            and mean_lookup["hard_mean_ssim_delta"] >= 0
        ),
        "regular_strong_ratio_mean_le_0p16": (
            mean_lookup["regular_strong_regression_ratio"] is not None
            and mean_lookup["regular_strong_regression_ratio"] <= 0.16
        ),
        "hard_worst_count_mean_le_8": (
            mean_lookup["hard_worst_count_delta_le_-0p20"] is not None
            and mean_lookup["hard_worst_count_delta_le_-0p20"] <= 8
        ),
    }
    payload = {
        "stage": "multi-seed checkpoint evidence aggregation",
        "locked_test_touched": False,
        "seed_count": len(seed_rows),
        "selection_jsons": args.selection_jsons,
        "seed_rows": seed_rows,
        "aggregate_rows": aggregate_rows,
        "gate_checks": gate_checks,
        "gate_pass": all(gate_checks.values()),
        "decision": (
            "MULTISEED_SCREEN_PASS_AUTHORIZE_CONFIRMATION_SEEDS"
            if all(gate_checks.values())
            else "MULTISEED_SCREEN_FAIL_CONTINUE_OTHER_EXPERIMENTS"
        ),
    }

    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        fields = ["metric", "count", "mean", "stdev", "ci95_half_width", "min", "max"]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(aggregate_rows)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
