#!/usr/bin/env python3
"""Summarize DTA-v3.3 RouterFusion SafeMix++ triage matrices."""

import argparse
import csv
import json
from pathlib import Path


TRIAGE_GATES = {
    "mean_true_a0_min": 0.030,
    "hard_true_a0_min": 0.010,
    "dssim_min": -0.000010,
    "positive_ratio_min": 0.620,
    "true_vs_zero_min": 0.035,
    "true_vs_shuffle_min": 0.030,
    "true_vs_normal_min": 0.025,
    "worst_max": 48,
}


def _metric(row, key, default=0.0):
    value = row.get(key, default)
    return float(value) if value is not None else default


def _by_label(matrix):
    return {row["label"]: row for row in matrix.get("runs", [])}


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
    name = path.stem.removeprefix("train_eval_depth_matrix_")
    airlight_mode = name.rsplit("_", 1)[-1]
    run_id = name[: -(len(airlight_mode) + 1)]
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


def variant_key(run_id: str) -> str:
    marker = "v33_routerfusion_"
    if marker in run_id:
        tail = run_id.split(marker, 1)[1]
        return tail.split("_seed", 1)[0]
    return run_id.split("_seed", 1)[0]


def aggregate_variants(summaries):
    out = []
    variants = sorted({variant_key(row["run_id"]) for row in summaries if row["airlight_mode"] == "fallback"})
    for variant in variants:
        rows = [
            row for row in summaries
            if row["airlight_mode"] == "fallback" and variant_key(row["run_id"]) == variant
        ]
        if not rows:
            continue
        gate_pass_count = sum(row["gate_pass"] for row in rows)
        out.append(
            {
                "variant": variant,
                "runs": len(rows),
                "gate_pass_count": gate_pass_count,
                "mean_true_a0": sum(row["mean_true_a0"] for row in rows) / len(rows),
                "hard_true_a0": sum(row["hard_true_a0"] for row in rows) / len(rows),
                "dssim": sum(row["dssim"] for row in rows) / len(rows),
                "positive_ratio": sum(row["positive_ratio"] for row in rows) / len(rows),
                "worst": sum(row["worst"] for row in rows) / len(rows),
                "true_vs_zero": sum(row["true_vs_zero"] for row in rows) / len(rows),
                "true_vs_shuffle": sum(row["true_vs_shuffle"] for row in rows) / len(rows),
                "true_vs_normal": sum(row["true_vs_normal"] for row in rows) / len(rows),
                "triage_pass": len(rows) >= 4 and gate_pass_count == len(rows),
            }
        )
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence_dir", required=True, type=Path)
    parser.add_argument("--output_json", required=True, type=Path)
    parser.add_argument("--output_csv", required=True, type=Path)
    parser.add_argument("--variant_csv", required=True, type=Path)
    args = parser.parse_args()

    paths = []
    for pattern in (
        "train_eval_depth_matrix_v33_routerfusion_*_fallback.json",
        "train_eval_depth_matrix_v33_routerfusion_*_gt.json",
    ):
        paths.extend(sorted(args.evidence_dir.glob(pattern)))
    summaries = [summarize_matrix(path) for path in sorted(set(paths))]
    variants = aggregate_variants(summaries)
    decision = (
        "TRIAGE_GATE_PASS_FORMAL_5FOLD_AUTHORIZED_LOCKED_TEST_BLOCKED"
        if any(row["triage_pass"] for row in variants)
        else "TRIAGE_GATE_FAIL_LOCKED_TEST_BLOCKED"
    )
    payload = {
        "decision": decision,
        "gates": TRIAGE_GATES,
        "summaries": summaries,
        "variant_summaries": variants,
        "locked_test_touched": False,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    fieldnames = [
        "run_id", "airlight_mode", "mean_true_a0", "hard_true_a0", "dssim",
        "positive_ratio", "worst", "strong", "true_vs_zero", "true_vs_shuffle",
        "true_vs_normal", "gate_pass",
    ]
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summaries:
            writer.writerow({key: row[key] for key in fieldnames})

    variant_fields = [
        "variant", "runs", "gate_pass_count", "mean_true_a0", "hard_true_a0",
        "dssim", "positive_ratio", "worst", "true_vs_zero", "true_vs_shuffle",
        "true_vs_normal", "triage_pass",
    ]
    with args.variant_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=variant_fields)
        writer.writeheader()
        for row in variants:
            writer.writerow({key: row[key] for key in variant_fields})
    print(f"DTA_V3_3_ROUTERFUSION_SUMMARY_OK decision={decision} rows={len(summaries)} variants={len(variants)}")


if __name__ == "__main__":
    main()
