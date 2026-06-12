#!/usr/bin/env python3
"""Summarize DTA-v3.2 SafeMix fold0 scout matrices and apply scout gates."""

import argparse
import csv
import json
from pathlib import Path


GATES = {
    "mean_true_a0_min": 0.020,
    "hard_true_a0_min": 0.010,
    "true_vs_zero_min": 0.030,
    "true_vs_shuffle_min": 0.030,
    "true_vs_normal_min": 0.025,
    "dssim_min": -0.000010,
    "positive_ratio_min": 0.630,
    "worst_max": 50,
}


def _by_label(matrix):
    return {row["label"]: row for row in matrix.get("runs", [])}


def _metric(row, key, default=0.0):
    value = row.get(key, default)
    return float(value) if value is not None else default


def summarize_matrix(path: Path):
    matrix = json.loads(path.read_text())
    rows = _by_label(matrix)
    missing = [label for label in ("true", "zero", "shuffle", "normal") if label not in rows]
    if missing:
        raise ValueError(f"{path} is missing labels: {missing}")
    true = rows["true"]
    mean_true = _metric(true, "mean_psnr_delta")
    hard_true = _metric(true, "hard_bottom25_psnr_delta")
    dssim = _metric(true, "mean_ssim_delta")
    positive_ratio = _metric(true, "positive_ratio")
    worst = int(true.get("worst_regression_count_delta_le_-0.20", 0))
    strong = int(true.get("strong_regression_count_delta_le_-0.05", 0))
    true_vs_zero = mean_true - _metric(rows["zero"], "mean_psnr_delta")
    true_vs_shuffle = mean_true - _metric(rows["shuffle"], "mean_psnr_delta")
    true_vs_normal = mean_true - _metric(rows["normal"], "mean_psnr_delta")
    gate_checks = {
        "mean_true_a0": mean_true >= GATES["mean_true_a0_min"],
        "hard_true_a0": hard_true >= GATES["hard_true_a0_min"],
        "true_vs_zero": true_vs_zero >= GATES["true_vs_zero_min"],
        "true_vs_shuffle": true_vs_shuffle >= GATES["true_vs_shuffle_min"],
        "true_vs_normal": true_vs_normal >= GATES["true_vs_normal_min"],
        "dssim": dssim >= GATES["dssim_min"],
        "positive_ratio": positive_ratio >= GATES["positive_ratio_min"],
        "worst": worst <= GATES["worst_max"],
    }
    name = path.stem.removeprefix("train_eval_depth_matrix_")
    airlight_mode = name.rsplit("_", 1)[-1]
    run_id = name[: -(len(airlight_mode) + 1)]
    return {
        "run_id": run_id,
        "airlight_mode": airlight_mode,
        "matrix_path": str(path),
        "common_count": int(true.get("common_count", 0)),
        "mean_true_a0": mean_true,
        "hard_true_a0": hard_true,
        "dssim": dssim,
        "positive_ratio": positive_ratio,
        "worst": worst,
        "strong": strong,
        "true_vs_zero": true_vs_zero,
        "true_vs_shuffle": true_vs_shuffle,
        "true_vs_normal": true_vs_normal,
        "gate_checks": gate_checks,
        "gate_pass": all(gate_checks.values()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence_dir", required=True, type=Path)
    parser.add_argument("--output_json", required=True, type=Path)
    parser.add_argument("--output_csv", required=True, type=Path)
    args = parser.parse_args()

    patterns = [
        "train_eval_depth_matrix_v32_safemix_*_fallback.json",
        "train_eval_depth_matrix_v32_safemix_*_gt.json",
    ]
    paths = []
    for pattern in patterns:
        paths.extend(sorted(args.evidence_dir.glob(pattern)))
    summaries = [summarize_matrix(path) for path in sorted(set(paths))]
    fallback_passes = [
        row for row in summaries if row["airlight_mode"] == "fallback" and row["gate_pass"]
    ]
    decision = (
        "SCOUT_GATE_PASS_FORMAL_5FOLD_AUTHORIZED_LOCKED_TEST_BLOCKED"
        if fallback_passes
        else "SCOUT_GATE_FAIL_LOCKED_TEST_BLOCKED"
    )
    payload = {
        "decision": decision,
        "gates": GATES,
        "summaries": summaries,
        "locked_test_touched": False,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    fieldnames = [
        "run_id",
        "airlight_mode",
        "mean_true_a0",
        "hard_true_a0",
        "dssim",
        "positive_ratio",
        "worst",
        "strong",
        "true_vs_zero",
        "true_vs_shuffle",
        "true_vs_normal",
        "gate_pass",
    ]
    with args.output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summaries:
            writer.writerow({key: row[key] for key in fieldnames})
    print(f"DTA_V3_2_SAFEMIX_SUMMARY_OK decision={decision} rows={len(summaries)}")


if __name__ == "__main__":
    main()
