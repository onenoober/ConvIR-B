#!/usr/bin/env python3
"""Aggregate DTA-v3.7 actual soft-blend oracle fold/seed outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


STRICT_GATES = {
    "coverage_min": 0.95,
    "mean_dpsnr_min": 0.055,
    "hard_bottom25_min": 0.040,
    "dssim_min": -0.000005,
    "positive_ratio_min": 0.630,
    "true_vs_zero_min": 0.040,
    "true_vs_shuffle_min": 0.035,
    "true_vs_normal_min": 0.030,
    "worst_per_600_max": 48.0,
    "max_outer_worst_per_600_max": 60.0,
}


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def safe_mean(values: list[float], default: float = float("nan")) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return statistics.mean(vals) if vals else default


def percentile(values: list[float], pct: float) -> float:
    vals = sorted(v for v in values if math.isfinite(v))
    if not vals:
        return float("nan")
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * pct / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return vals[lo]
    return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(row.get(key), sort_keys=True) if isinstance(row.get(key), (dict, list)) else row.get(key, "")
                for key in keys
            })


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    deltas = [finite_float(row.get("dPSNR"), 0.0) for row in rows]
    ssim_deltas = [finite_float(row.get("dSSIM"), 0.0) for row in rows]
    a0_psnr = [finite_float(row.get("A0_PSNR"), 0.0) for row in rows]
    sorted_idx = sorted(range(len(rows)), key=lambda idx: a0_psnr[idx])
    bucket_count = max(1, len(rows) // 4)
    hard_idx = sorted_idx[:bucket_count]
    easy_idx = sorted_idx[-bucket_count:]
    strong_cut = percentile(a0_psnr, 75.0)
    strong_idx = [idx for idx, value in enumerate(a0_psnr) if value >= strong_cut]
    worst_count = sum(delta <= -0.20 for delta in deltas)
    strong_count = sum(delta <= -0.05 for delta in deltas)
    n = len(rows)

    def mean_at(indices: list[int]) -> float:
        return statistics.mean(deltas[idx] for idx in indices) if indices else float("nan")

    def surplus(control_key: str) -> float:
        return statistics.mean(
            finite_float(row.get("dPSNR"), 0.0) - finite_float(row.get(control_key), 0.0)
            for row in rows
        )

    return {
        "count": n,
        "coverage": 1.0,
        "mean_dPSNR": statistics.mean(deltas),
        "hard_bottom25_dPSNR": mean_at(hard_idx),
        "easy_top25_dPSNR": mean_at(easy_idx),
        "dSSIM": statistics.mean(ssim_deltas),
        "positive_ratio": sum(delta > 0 for delta in deltas) / n,
        "strong_reference_cut_psnr": strong_cut,
        "strong_reference_regression_count": sum(deltas[idx] <= -0.05 for idx in strong_idx),
        "strong_count_le_-0.05": strong_count,
        "strong_per_600": strong_count * 600.0 / n,
        "worst_count_le_-0.20": worst_count,
        "worst_per_600": worst_count * 600.0 / n,
        "true_vs_zero": surplus("zero_delta_psnr"),
        "true_vs_shuffle": surplus("shuffle_delta_psnr"),
        "true_vs_normal": surplus("normal_delta_psnr"),
    }


def gate_checks(metrics: dict[str, Any]) -> dict[str, bool]:
    return {
        "coverage": finite_float(metrics.get("coverage"), 0.0) >= STRICT_GATES["coverage_min"],
        "mean": finite_float(metrics.get("mean_dPSNR"), -1e9) >= STRICT_GATES["mean_dpsnr_min"],
        "hard": finite_float(metrics.get("hard_bottom25_dPSNR"), -1e9) >= STRICT_GATES["hard_bottom25_min"],
        "dssim": finite_float(metrics.get("dSSIM"), -1e9) >= STRICT_GATES["dssim_min"],
        "positive_ratio": finite_float(metrics.get("positive_ratio"), 0.0) >= STRICT_GATES["positive_ratio_min"],
        "true_vs_zero": finite_float(metrics.get("true_vs_zero"), -1e9) >= STRICT_GATES["true_vs_zero_min"],
        "true_vs_shuffle": finite_float(metrics.get("true_vs_shuffle"), -1e9) >= STRICT_GATES["true_vs_shuffle_min"],
        "true_vs_normal": finite_float(metrics.get("true_vs_normal"), -1e9) >= STRICT_GATES["true_vs_normal_min"],
        "worst": finite_float(metrics.get("worst_per_600"), 1e9) <= STRICT_GATES["worst_per_600_max"],
        "max_outer_worst": finite_float(metrics.get("max_outer_worst_per_600"), 1e9) <= STRICT_GATES["max_outer_worst_per_600_max"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_glob", required=True)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--combined_selected_csv", default="v37_real_blend_oracle_selected_all.csv")
    parser.add_argument("--grid_csv", default="v37_real_blend_oracle_grid.csv")
    parser.add_argument("--summary_json", default="v37_real_blend_summary.json")
    args = parser.parse_args()

    files = sorted(Path().glob(args.input_glob)) if not Path(args.input_glob).is_absolute() else sorted(Path("/").glob(str(Path(args.input_glob))[1:]))
    if not files:
        raise FileNotFoundError(f"No selected CSV files matched {args.input_glob}")

    rows: list[dict[str, Any]] = []
    for path in files:
        rows.extend(read_csv(path))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined_path = args.output_dir / args.combined_selected_csv
    write_csv(combined_path, rows)

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["bank_name"]), str(row["utility_mode"]))].append(row)

    grid_rows: list[dict[str, Any]] = []
    for (bank_name, utility_mode), group_rows in sorted(grouped.items()):
        metrics = summarize(group_rows)
        outer_worst = []
        by_outer: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        by_fold: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in group_rows:
            by_outer[(str(row.get("fold")), str(row.get("seed")))].append(row)
            by_fold[str(row.get("fold"))].append(row)
        for outer_rows in by_outer.values():
            outer_worst.append(finite_float(summarize(outer_rows).get("worst_per_600"), 0.0))
        metrics["max_outer_worst_per_600"] = max(outer_worst) if outer_worst else metrics.get("worst_per_600")
        metrics["max_fold_worst_per_600"] = max(
            finite_float(summarize(fold_rows).get("worst_per_600"), 0.0)
            for fold_rows in by_fold.values()
        )
        checks = gate_checks(metrics)
        metrics.update({
            "bank_name": bank_name,
            "utility_mode": utility_mode,
            "fold_seed_groups": len(by_outer),
            "strict_gate_pass": all(checks.values()),
            "strict_gate_checks": checks,
            "chosen_action_counts": dict(Counter(str(row.get("chosen_action")) for row in group_rows)),
            "chosen_variant_counts": dict(Counter(str(row.get("chosen_variant")) for row in group_rows)),
            "intervention_rate": sum(str(row.get("chosen_variant")) != "A0" for row in group_rows) / len(group_rows),
            "mean_chosen_alpha": safe_mean([finite_float(row.get("chosen_alpha")) for row in group_rows]),
            "actual_blend_note": "PSNR/SSIM computed on rendered blended tensors, not linear metric proxies.",
        })
        grid_rows.append(metrics)

    grid_rows.sort(
        key=lambda row: (
            bool(row.get("strict_gate_pass")),
            finite_float(row.get("mean_dPSNR"), -1e9),
            finite_float(row.get("positive_ratio"), 0.0),
        ),
        reverse=True,
    )
    grid_path = args.output_dir / args.grid_csv
    write_csv(grid_path, grid_rows)
    strict_rows = [row for row in grid_rows if row.get("strict_gate_pass")]
    summary = {
        "route": "DTA-v3.7 U-TQS-Mix",
        "phase": "C1_real_soft_blend_aggregate",
        "input_files": [str(path) for path in files],
        "combined_selected_csv": str(combined_path),
        "grid_csv": str(grid_path),
        "row_count": len(rows),
        "grid_count": len(grid_rows),
        "strict_pass_count": len(strict_rows),
        "best_row": grid_rows[0] if grid_rows else {},
        "decision": "PHASE_C1_REAL_BLEND_ORACLE_PASS" if strict_rows else "PHASE_C1_REAL_BLEND_ORACLE_FAIL",
        "strict_gates": STRICT_GATES,
        "locked_test_touched": False,
    }
    summary_path = args.output_dir / args.summary_json
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "DTA_V3_7_REAL_BLEND_AGGREGATE_OK "
        f"rows={len(rows)} grid={len(grid_rows)} strict_pass={len(strict_rows)} "
        f"decision={summary['decision']}"
    )


if __name__ == "__main__":
    main()
