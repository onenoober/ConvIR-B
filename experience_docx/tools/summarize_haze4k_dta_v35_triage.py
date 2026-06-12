#!/usr/bin/env python3
"""Summarize train-derived DTA-v3.5 FDF-RCS-Lite triage matrices.

This summary keeps the original strict safety gates visible, but uses a relaxed
flow gate so the v3.5 queue can complete later nested-calibration stages without
silently relabeling a strict gate fail as a scientific pass.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

STRICT_GATES = {
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

RELAXED_FLOW_GATES = {
    "mean_true_a0_min": -0.030,
    "hard_true_a0_min": -0.060,
    "dssim_min": -0.000120,
    "positive_ratio_min": 0.420,
    "true_vs_zero_min": -0.010,
    "true_vs_shuffle_min": -0.020,
    "true_vs_normal_min": -0.020,
    "worst_max": 220,
    "per_run_mean_min": -0.080,
    "per_run_worst_max": 260,
}

RUN_RE = re.compile(r"v35_fdf_rcs_(?P<variant>.+)_seed(?P<seed>\d+)_f(?P<fold>\d+)_(?P<stage>[^_]+.*)$")


def _metric(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    return float(value) if value is not None else default


def _by_label(matrix: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["label"]: row for row in matrix.get("runs", [])}


def _checks(row: dict[str, Any], gates: dict[str, float]) -> dict[str, bool]:
    return {
        "mean_true_a0": row["mean_true_a0"] >= gates["mean_true_a0_min"],
        "hard_true_a0": row["hard_true_a0"] >= gates["hard_true_a0_min"],
        "dssim": row["dssim"] >= gates["dssim_min"],
        "positive_ratio": row["positive_ratio"] >= gates["positive_ratio_min"],
        "true_vs_zero": row["true_vs_zero"] >= gates["true_vs_zero_min"],
        "true_vs_shuffle": row["true_vs_shuffle"] >= gates["true_vs_shuffle_min"],
        "true_vs_normal": row["true_vs_normal"] >= gates["true_vs_normal_min"],
        "worst": row["worst"] <= gates["worst_max"],
    }


def summarize_matrix(path: Path) -> dict[str, Any]:
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
    out = {
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
    }
    out["strict_gate_checks"] = _checks(out, STRICT_GATES)
    out["strict_gate_pass"] = all(out["strict_gate_checks"].values())
    out["relaxed_flow_checks"] = _checks(out, RELAXED_FLOW_GATES)
    out["relaxed_flow_pass"] = all(out["relaxed_flow_checks"].values())
    return out


def aggregate(rows: list[dict[str, Any]], expected_runs: int) -> list[dict[str, Any]]:
    out = []
    for variant in sorted({row["variant"] for row in rows}):
        group = [row for row in rows if row["variant"] == variant]
        if not group:
            continue
        avg = lambda key: sum(row[key] for row in group) / len(group)
        strict_pass = (
            len(group) >= expected_runs
            and avg("mean_true_a0") >= STRICT_GATES["mean_true_a0_min"]
            and avg("hard_true_a0") >= STRICT_GATES["hard_true_a0_min"]
            and avg("dssim") >= STRICT_GATES["dssim_min"]
            and avg("positive_ratio") >= STRICT_GATES["positive_ratio_min"]
            and avg("true_vs_zero") >= STRICT_GATES["true_vs_zero_min"]
            and avg("true_vs_shuffle") >= STRICT_GATES["true_vs_shuffle_min"]
            and avg("true_vs_normal") >= STRICT_GATES["true_vs_normal_min"]
            and avg("worst") <= STRICT_GATES["worst_max"]
            and min(row["mean_true_a0"] for row in group) >= STRICT_GATES["per_run_mean_min"]
            and max(row["worst"] for row in group) <= STRICT_GATES["per_run_worst_max"]
        )
        relaxed_flow_pass = (
            len(group) >= expected_runs
            and avg("mean_true_a0") >= RELAXED_FLOW_GATES["mean_true_a0_min"]
            and avg("hard_true_a0") >= RELAXED_FLOW_GATES["hard_true_a0_min"]
            and avg("dssim") >= RELAXED_FLOW_GATES["dssim_min"]
            and avg("positive_ratio") >= RELAXED_FLOW_GATES["positive_ratio_min"]
            and avg("true_vs_zero") >= RELAXED_FLOW_GATES["true_vs_zero_min"]
            and avg("true_vs_shuffle") >= RELAXED_FLOW_GATES["true_vs_shuffle_min"]
            and avg("true_vs_normal") >= RELAXED_FLOW_GATES["true_vs_normal_min"]
            and avg("worst") <= RELAXED_FLOW_GATES["worst_max"]
            and min(row["mean_true_a0"] for row in group) >= RELAXED_FLOW_GATES["per_run_mean_min"]
            and max(row["worst"] for row in group) <= RELAXED_FLOW_GATES["per_run_worst_max"]
        )
        out.append({
            "variant": variant,
            "runs": len(group),
            "strict_gate_pass_count": sum(row["strict_gate_pass"] for row in group),
            "relaxed_flow_pass_count": sum(row["relaxed_flow_pass"] for row in group),
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
            "strict_triage_pass": strict_pass,
            "relaxed_flow_pass": relaxed_flow_pass,
            "flow_score": avg("mean_true_a0") + 0.25 * avg("hard_true_a0") + 0.25 * avg("true_vs_zero") - 0.0005 * avg("worst"),
        })
    return out


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence_dir", required=True, type=Path)
    parser.add_argument("--output_json", required=True, type=Path)
    parser.add_argument("--output_csv", required=True, type=Path)
    parser.add_argument("--variant_csv", required=True, type=Path)
    parser.add_argument("--expected_runs_per_variant", type=int, default=4)
    args = parser.parse_args()

    paths = sorted(args.evidence_dir.glob("train_eval_depth_matrix_v35_fdf_rcs_*_fallback_train.json"))
    rows = [summarize_matrix(path) for path in paths]
    variants = aggregate(rows, args.expected_runs_per_variant)
    strict_any = any(v["strict_triage_pass"] for v in variants)
    relaxed_any = any(v["relaxed_flow_pass"] for v in variants)
    decision = "STRICT_TRIAGE_PASS_FORMAL_TRAIN_DERIVED_AUTHORIZED_LOCKED_TEST_BLOCKED" if strict_any else (
        "RELAXED_FLOW_PASS_CONTINUE_NESTED_CALIBRATION_LOCKED_TEST_BLOCKED" if relaxed_any else "RELAXED_FLOW_COMPLETE_NO_STRICT_PASS_LOCKED_TEST_BLOCKED"
    )
    payload = {
        "decision": decision,
        "strict_gates": STRICT_GATES,
        "relaxed_flow_gates": RELAXED_FLOW_GATES,
        "summaries": rows,
        "variant_summaries": variants,
        "locked_test_touched": False,
        "note": "Relaxed flow gates are for completing v3.5 diagnostics only; they are not promotion gates.",
    }
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    run_fields = ["run_id", "variant", "seed", "fold", "mean_true_a0", "hard_true_a0", "dssim", "positive_ratio", "worst", "strong", "true_vs_zero", "true_vs_shuffle", "true_vs_normal", "strict_gate_pass", "relaxed_flow_pass", "matrix_path"]
    variant_fields = ["variant", "runs", "strict_gate_pass_count", "relaxed_flow_pass_count", "mean_true_a0", "hard_true_a0", "dssim", "positive_ratio", "worst", "strong", "true_vs_zero", "true_vs_shuffle", "true_vs_normal", "min_run_mean", "max_run_worst", "strict_triage_pass", "relaxed_flow_pass", "flow_score"]
    write_csv(args.output_csv, rows, run_fields)
    write_csv(args.variant_csv, variants, variant_fields)
    print(f"DTA_V3_5_TRIAGE_SUMMARY_OK decision={decision} rows={len(rows)} variants={len(variants)}")


if __name__ == "__main__":
    main()
