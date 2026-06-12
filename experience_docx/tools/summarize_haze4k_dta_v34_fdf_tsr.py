#!/usr/bin/env python3
"""Summarize DTA-v3.4 FDF-TSR quick fine-tune and one-shot test matrices."""

import argparse
import csv
import json
from pathlib import Path

RELAXED_GATES = {
    "mean_true_a0_min": 0.0,
    "hard_true_a0_min": 0.0,
    "dssim_min": -0.000050,
    "positive_ratio_min": 0.50,
    "true_vs_zero_min": 0.0,
    "true_vs_shuffle_min": 0.0,
    "true_vs_normal_min": 0.0,
}


def _metric(row, key, default=0.0):
    value = row.get(key, default)
    return float(value) if value is not None else default


def _by_label(matrix):
    return {row["label"]: row for row in matrix.get("runs", [])}


def summarize_matrix(path: Path):
    matrix = json.loads(path.read_text(encoding="utf-8"))
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
    name = path.stem.removeprefix("train_eval_depth_matrix_")
    split = "test" if name.endswith("fallback_test") else ("train" if name.endswith("fallback_train") else "unknown")
    gate_checks = {
        "mean_true_a0": mean_true >= RELAXED_GATES["mean_true_a0_min"],
        "hard_true_a0": hard_true >= RELAXED_GATES["hard_true_a0_min"],
        "dssim": dssim >= RELAXED_GATES["dssim_min"],
        "positive_ratio": positive_ratio >= RELAXED_GATES["positive_ratio_min"],
        "true_vs_zero": true_vs_zero >= RELAXED_GATES["true_vs_zero_min"],
        "true_vs_shuffle": true_vs_shuffle >= RELAXED_GATES["true_vs_shuffle_min"],
        "true_vs_normal": true_vs_normal >= RELAXED_GATES["true_vs_normal_min"],
    }
    return {
        "run_id": name.rsplit("_fallback_", 1)[0] if "_fallback_" in name else name,
        "split": split,
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
        "gate_checks_relaxed": gate_checks,
        "relaxed_gate_pass": all(gate_checks.values()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence_dir", required=True, type=Path)
    parser.add_argument("--output_json", required=True, type=Path)
    parser.add_argument("--output_csv", required=True, type=Path)
    args = parser.parse_args()

    paths = sorted(args.evidence_dir.glob("train_eval_depth_matrix_v34_fdf_tsr_*_fallback_*.json"))
    summaries = [summarize_matrix(path) for path in paths]
    test_rows = [row for row in summaries if row["split"] == "test"]
    decision = "USER_EXPLICIT_TEST_OVERRIDE_ONE_SHOT_COMPLETED" if test_rows else "V34_FDF_TSR_NO_TEST_MATRIX_YET"
    payload = {
        "decision": decision,
        "gates": RELAXED_GATES,
        "summaries": summaries,
        "locked_test_touched": bool(test_rows),
        "locked_test_policy_note": "User explicitly requested one Haze4K test run and result images; do not use this result for iterative checkpoint/gate selection.",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    fieldnames = [
        "run_id", "split", "common_count", "mean_true_a0", "hard_true_a0", "dssim",
        "positive_ratio", "worst", "strong", "true_vs_zero", "true_vs_shuffle",
        "true_vs_normal", "relaxed_gate_pass", "matrix_path",
    ]
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summaries:
            writer.writerow({key: row[key] for key in fieldnames})
    print(f"DTA_V3_4_FDF_TSR_SUMMARY_OK decision={decision} rows={len(summaries)}")


if __name__ == "__main__":
    main()
