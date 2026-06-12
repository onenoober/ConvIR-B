#!/usr/bin/env python3
"""Summarize train-derived DTA-v3.4 FDF-TSR triage matrices."""

import argparse
import csv
import json
import re
from pathlib import Path

TRIAGE_GATES = {
    "mean_true_a0_min": 0.055,
    "hard_true_a0_min": 0.040,
    "dssim_min": -0.000005,
    "positive_ratio_min": 0.630,
    "true_vs_zero_min": 0.040,
    "true_vs_shuffle_min": 0.035,
    "true_vs_normal_min": 0.030,
    "worst_max": 48,
    "per_run_mean_min": 0.020,
    "per_run_worst_max": 60,
}

RUN_RE = re.compile(r"v34_fdf_tsr_(?P<variant>.+)_seed(?P<seed>\d+)_f(?P<fold>\d+)_(?P<stage>[^_]+.*)$")


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
        raise ValueError(f"{path} missing labels {missing}")
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
    name = path.stem.removeprefix("train_eval_depth_matrix_").removesuffix("_fallback_train")
    match = RUN_RE.match(name)
    meta = match.groupdict() if match else {"variant": name, "seed": "", "fold": "", "stage": ""}
    gate_checks = {
        "mean_true_a0": mean_true >= TRIAGE_GATES["mean_true_a0_min"],
        "hard_true_a0": hard_true >= TRIAGE_GATES["hard_true_a0_min"],
        "dssim": dssim >= TRIAGE_GATES["dssim_min"],
        "positive_ratio": positive_ratio >= TRIAGE_GATES["positive_ratio_min"],
        "true_vs_zero": true_vs_zero >= TRIAGE_GATES["true_vs_zero_min"],
        "true_vs_shuffle": true_vs_shuffle >= TRIAGE_GATES["true_vs_shuffle_min"],
        "true_vs_normal": true_vs_normal >= TRIAGE_GATES["true_vs_normal_min"],
        "worst": worst <= TRIAGE_GATES["worst_max"],
    }
    return {
        "run_id": name,
        "variant": meta["variant"],
        "seed": meta["seed"],
        "fold": meta["fold"],
        "stage": meta["stage"],
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


def aggregate(rows):
    out = []
    for variant in sorted({row["variant"] for row in rows}):
        group = [row for row in rows if row["variant"] == variant]
        if not group:
            continue
        avg = lambda key: sum(row[key] for row in group) / len(group)
        gate_pass_count = sum(row["gate_pass"] for row in group)
        triage_pass = (
            len(group) >= 4
            and avg("mean_true_a0") >= TRIAGE_GATES["mean_true_a0_min"]
            and avg("hard_true_a0") >= TRIAGE_GATES["hard_true_a0_min"]
            and avg("dssim") >= TRIAGE_GATES["dssim_min"]
            and avg("positive_ratio") >= TRIAGE_GATES["positive_ratio_min"]
            and avg("true_vs_zero") >= TRIAGE_GATES["true_vs_zero_min"]
            and avg("true_vs_shuffle") >= TRIAGE_GATES["true_vs_shuffle_min"]
            and avg("true_vs_normal") >= TRIAGE_GATES["true_vs_normal_min"]
            and avg("worst") <= TRIAGE_GATES["worst_max"]
            and min(row["mean_true_a0"] for row in group) >= TRIAGE_GATES["per_run_mean_min"]
            and max(row["worst"] for row in group) <= TRIAGE_GATES["per_run_worst_max"]
        )
        out.append({
            "variant": variant,
            "runs": len(group),
            "gate_pass_count": gate_pass_count,
            "mean_true_a0": avg("mean_true_a0"),
            "hard_true_a0": avg("hard_true_a0"),
            "dssim": avg("dssim"),
            "positive_ratio": avg("positive_ratio"),
            "worst": avg("worst"),
            "strong": avg("strong"),
            "true_vs_zero": avg("true_vs_zero"),
            "true_vs_shuffle": avg("true_vs_shuffle"),
            "true_vs_normal": avg("true_vs_normal"),
            "min_run_mean": min(row["mean_true_a0"] for row in group),
            "max_run_worst": max(row["worst"] for row in group),
            "triage_pass": triage_pass,
        })
    return out


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence_dir", required=True, type=Path)
    parser.add_argument("--output_json", required=True, type=Path)
    parser.add_argument("--output_csv", required=True, type=Path)
    parser.add_argument("--variant_csv", required=True, type=Path)
    args = parser.parse_args()
    paths = sorted(args.evidence_dir.glob("train_eval_depth_matrix_v34_fdf_tsr_*_fallback_train.json"))
    rows = [summarize_matrix(path) for path in paths]
    variants = aggregate(rows)
    decision = "TRIAGE_GATE_PASS_FORMAL_CV_AUTHORIZED_LOCKED_TEST_BLOCKED" if any(v["triage_pass"] for v in variants) else "TRIAGE_GATE_FAIL_LOCKED_TEST_BLOCKED"
    payload = {"decision": decision, "gates": TRIAGE_GATES, "summaries": rows, "variant_summaries": variants, "locked_test_touched": False}
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    run_fields = ["run_id", "variant", "seed", "fold", "mean_true_a0", "hard_true_a0", "dssim", "positive_ratio", "worst", "strong", "true_vs_zero", "true_vs_shuffle", "true_vs_normal", "gate_pass", "matrix_path"]
    variant_fields = ["variant", "runs", "gate_pass_count", "mean_true_a0", "hard_true_a0", "dssim", "positive_ratio", "worst", "strong", "true_vs_zero", "true_vs_shuffle", "true_vs_normal", "min_run_mean", "max_run_worst", "triage_pass"]
    write_csv(args.output_csv, rows, run_fields)
    write_csv(args.variant_csv, variants, variant_fields)
    print(f"DTA_V3_4_TRIAGE_SUMMARY_OK decision={decision} rows={len(rows)} variants={len(variants)}")


if __name__ == "__main__":
    main()
