#!/usr/bin/env python3
"""Selector metric correction and nested fold0 smoke test for DTA risk gates."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


BASE_COLUMNS = {
    "name",
    "original_psnr",
    "delta_psnr",
    "original_ssim",
    "delta_ssim",
    "original_time_sec",
}


def read_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return {row["name"]: row for row in csv.DictReader(handle)}


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def percentile(values: list[float], pct: float) -> float:
    finite = sorted(v for v in values if math.isfinite(v))
    if not finite:
        return float("nan")
    if len(finite) == 1:
        return finite[0]
    pos = (len(finite) - 1) * pct / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return finite[lo]
    return finite[lo] + (finite[hi] - finite[lo]) * (pos - lo)


def candidate_feature_columns(row: dict[str, str]) -> list[str]:
    features = []
    for key, value in row.items():
        if key in BASE_COLUMNS or key.startswith("original_"):
            continue
        if key.endswith("_psnr") or key.endswith("_ssim") or key.endswith("_time_sec"):
            continue
        if not any(token in key for token in ("dta_", "input_", "depth_", "airlight_")):
            continue
        if math.isfinite(finite_float(value)):
            features.append(key)
    return sorted(features)


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
        writer.writerows(rows)


def load_matrix(true_csv: Path, zero_csv: Path, shuffle_csv: Path, normal_csv: Path) -> list[dict[str, Any]]:
    true_rows = read_rows(true_csv)
    zero_rows = read_rows(zero_csv)
    shuffle_rows = read_rows(shuffle_csv)
    normal_rows = read_rows(normal_csv)
    common = sorted(set(true_rows) & set(zero_rows) & set(shuffle_rows) & set(normal_rows))
    if not common:
        raise ValueError("No common rows across true/zero/shuffle/normal CSVs.")
    feature_cols = candidate_feature_columns(true_rows[common[0]])
    rows = []
    for name in common:
        true = true_rows[name]
        rows.append(
            {
                "name": name,
                "original_psnr": finite_float(true["original_psnr"]),
                "true_delta": finite_float(true["delta_psnr"]),
                "true_dssim": finite_float(true["delta_ssim"], 0.0),
                "zero_delta": finite_float(zero_rows[name]["delta_psnr"]),
                "shuffle_delta": finite_float(shuffle_rows[name]["delta_psnr"]),
                "normal_delta": finite_float(normal_rows[name]["delta_psnr"]),
                "features": {col: finite_float(true.get(col)) for col in feature_cols},
            }
        )
    return rows


def metric_summary(rows: list[dict[str, Any]], accept: list[bool]) -> dict[str, Any]:
    selected_deltas = [row["true_delta"] for row, keep in zip(rows, accept) if keep]
    selected_ssim = [row["true_dssim"] for row, keep in zip(rows, accept) if keep]
    deltas = [row["true_delta"] if keep else 0.0 for row, keep in zip(rows, accept)]
    ssim_deltas = [row["true_dssim"] if keep else 0.0 for row, keep in zip(rows, accept)]
    original = [row["original_psnr"] for row in rows]
    sorted_idx = sorted(range(len(rows)), key=lambda idx: original[idx])
    bucket = max(1, len(rows) // 4)
    hard = sorted_idx[:bucket]
    easy = sorted_idx[-bucket:]
    strong_cut = percentile(original, 75)
    strong = [idx for idx, val in enumerate(original) if val >= strong_cut]
    selected = sum(accept)

    def mean_at(indices: list[int]) -> float:
        return statistics.mean(deltas[idx] for idx in indices) if indices else float("nan")

    def selected_mean_at(indices: list[int]) -> float | None:
        vals = [rows[idx]["true_delta"] for idx in indices if accept[idx]]
        return statistics.mean(vals) if vals else None

    return {
        "count": len(rows),
        "selected_count": selected,
        "coverage": selected / len(rows) if rows else 0.0,
        "global_mean_delta": statistics.mean(deltas) if rows else None,
        "global_hard_bottom25_delta": mean_at(hard),
        "global_easy_top25_delta": mean_at(easy),
        "global_mean_ssim_delta": statistics.mean(ssim_deltas) if rows else None,
        "global_positive_ratio_delta_gt_0": sum(delta > 0.0 for delta in deltas) / len(deltas) if deltas else None,
        "global_positive_or_zero_ratio_delta_ge_0": sum(delta >= 0.0 for delta in deltas) / len(deltas) if deltas else None,
        "global_non_degradation_ratio_delta_ge_-0.01": sum(delta >= -0.01 for delta in deltas) / len(deltas) if deltas else None,
        "selected_mean_delta": statistics.mean(selected_deltas) if selected_deltas else None,
        "selected_hard_bottom25_delta": selected_mean_at(hard),
        "selected_easy_top25_delta": selected_mean_at(easy),
        "selected_mean_ssim_delta": statistics.mean(selected_ssim) if selected_ssim else None,
        "selected_conditional_positive_ratio": (
            sum(delta > 0.0 for delta in selected_deltas) / len(selected_deltas) if selected_deltas else None
        ),
        "selected_conditional_non_degradation_ratio_delta_ge_-0.01": (
            sum(delta >= -0.01 for delta in selected_deltas) / len(selected_deltas) if selected_deltas else None
        ),
        "strong_reference_cut_psnr": strong_cut,
        "strong_regression_count_delta_le_-0.05": sum(deltas[idx] <= -0.05 for idx in strong),
        "worst_regression_count_delta_le_-0.20": sum(delta <= -0.20 for delta in deltas),
        "true_vs_zero_all_images": statistics.mean(
            (row["true_delta"] - row["zero_delta"]) if keep else 0.0 for row, keep in zip(rows, accept)
        ),
        "true_vs_shuffle_all_images": statistics.mean(
            (row["true_delta"] - row["shuffle_delta"]) if keep else 0.0 for row, keep in zip(rows, accept)
        ),
        "true_vs_normal_all_images": statistics.mean(
            (row["true_delta"] - row["normal_delta"]) if keep else 0.0 for row, keep in zip(rows, accept)
        ),
        "true_vs_zero_selected_images": (
            statistics.mean(row["true_delta"] - row["zero_delta"] for row, keep in zip(rows, accept) if keep)
            if selected
            else None
        ),
        "true_vs_shuffle_selected_images": (
            statistics.mean(row["true_delta"] - row["shuffle_delta"] for row, keep in zip(rows, accept) if keep)
            if selected
            else None
        ),
        "true_vs_normal_selected_images": (
            statistics.mean(row["true_delta"] - row["normal_delta"] for row, keep in zip(rows, accept) if keep)
            if selected
            else None
        ),
    }


def selector_objective(metrics: dict[str, Any]) -> float:
    mean_delta = metrics["global_mean_delta"] or 0.0
    hard_delta = metrics["global_hard_bottom25_delta"] or 0.0
    surplus = metrics["true_vs_zero_all_images"] or 0.0
    ssim_delta = metrics["global_mean_ssim_delta"] or 0.0
    worst = metrics["worst_regression_count_delta_le_-0.20"] or 0
    strong = metrics["strong_regression_count_delta_le_-0.05"] or 0
    coverage = metrics["coverage"] or 0.0
    selected_pos = metrics["selected_conditional_positive_ratio"] or 0.0
    return (
        mean_delta
        + 0.25 * hard_delta
        + 0.5 * surplus
        + 0.01 * selected_pos
        + 20.0 * min(ssim_delta, 0.0)
        - 0.0015 * worst
        - 0.0005 * strong
        + 0.005 * coverage
    )


def fit_threshold(rows: list[dict[str, Any]], min_coverage: float, max_coverage: float) -> tuple[dict[str, Any], list[bool]]:
    feature_cols = sorted(rows[0]["features"]) if rows else []
    best = None
    best_accept: list[bool] | None = None
    for feature in feature_cols:
        values = [row["features"].get(feature, float("nan")) for row in rows]
        finite = [value for value in values if math.isfinite(value)]
        if len(set(finite)) < 2:
            continue
        thresholds = sorted({percentile(finite, pct) for pct in range(5, 100, 5)})
        for threshold in thresholds:
            for direction in ("<=", ">="):
                accept = [(value <= threshold) if direction == "<=" else (value >= threshold) for value in values]
                metrics = metric_summary(rows, accept)
                if metrics["coverage"] < min_coverage or metrics["coverage"] > max_coverage:
                    continue
                result = {
                    "feature": feature,
                    "direction": direction,
                    "threshold": threshold,
                    "objective": selector_objective(metrics),
                    **metrics,
                }
                if best is None or result["objective"] > best["objective"]:
                    best = result
                    best_accept = accept
    if best is None or best_accept is None:
        best_accept = [True] * len(rows)
        best = {"feature": "accept_all", "direction": ">=", "threshold": float("-inf"), "objective": None}
        best.update(metric_summary(rows, best_accept))
    return best, best_accept


def apply_selector(rows: list[dict[str, Any]], selector: dict[str, Any]) -> list[bool]:
    if selector["feature"] == "accept_all":
        return [True] * len(rows)
    feature = selector["feature"]
    threshold = selector["threshold"]
    direction = selector["direction"]
    out = []
    for row in rows:
        value = row["features"].get(feature, float("nan"))
        keep = (value <= threshold) if direction == "<=" else (value >= threshold)
        out.append(bool(keep))
    return out


def correction_report(args: argparse.Namespace) -> None:
    rows = load_matrix(Path(args.true_csv), Path(args.zero_csv), Path(args.shuffle_csv), Path(args.normal_csv))
    selected_rows = read_rows(Path(args.selected_csv)) if args.selected_csv else {}
    accept = []
    for row in rows:
        if selected_rows:
            value = selected_rows[row["name"]].get("accept_depth_action", "false").lower()
            accept.append(value in ("1", "true", "yes"))
        else:
            accept.append(True)
    report = {
        "protocol": "selector_metric_correction",
        "warning": "Rejected images fall back to A0, so global positive_ratio is not the selected conditional positive ratio.",
        "selected_policy_source": args.selected_csv or "accept_all",
        "metrics": metric_summary(rows, accept),
        "accept_all_baseline": metric_summary(rows, [True] * len(rows)),
        "locked_test_touched": False,
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("DTA_SELECTOR_METRIC_CORRECTION_OK")


def nested_smoke(args: argparse.Namespace) -> None:
    rows = load_matrix(Path(args.true_csv), Path(args.zero_csv), Path(args.shuffle_csv), Path(args.normal_csv))
    folds = max(2, args.internal_folds)
    fold_rows = []
    threshold_rows = []
    curve_rows = []
    for fold in range(folds):
        calib = [row for idx, row in enumerate(rows) if idx % folds != fold]
        heldout = [row for idx, row in enumerate(rows) if idx % folds == fold]
        selector, _ = fit_threshold(calib, args.min_coverage, args.max_coverage)
        heldout_accept = apply_selector(heldout, selector)
        heldout_metrics = metric_summary(heldout, heldout_accept)
        fold_rows.append({"fold": fold, **heldout_metrics})
        threshold_rows.append(
            {
                "fold": fold,
                "feature": selector["feature"],
                "direction": selector["direction"],
                "threshold": selector["threshold"],
                "calibration_objective": selector["objective"],
                "calibration_coverage": selector["coverage"],
                "heldout_coverage": heldout_metrics["coverage"],
                "heldout_global_mean_delta": heldout_metrics["global_mean_delta"],
                "heldout_true_vs_zero_all_images": heldout_metrics["true_vs_zero_all_images"],
                "heldout_worst_regression_count_delta_le_-0.20": heldout_metrics["worst_regression_count_delta_le_-0.20"],
            }
        )
        for min_cov in [0.15, 0.25, 0.40, 0.55, 0.70]:
            selector_i, _ = fit_threshold(calib, min_cov, args.max_coverage)
            accept_i = apply_selector(heldout, selector_i)
            curve_rows.append({"fold": fold, "min_coverage": min_cov, **metric_summary(heldout, accept_i)})
    aggregate = aggregate_fold_rows(fold_rows)
    output = {
        "protocol": "fold0_internal_nested_selector_smoke",
        "internal_folds": folds,
        "folds": fold_rows,
        "aggregate": aggregate,
        "warning": "This is still inside fold0 validation; use only to test same-fold threshold overfit risk.",
        "locked_test_touched": False,
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(output, indent=2), encoding="utf-8")
    write_csv(Path(args.thresholds_csv), threshold_rows)
    write_csv(Path(args.risk_coverage_csv), curve_rows)
    print(json.dumps(output, indent=2))
    print("DTA_NESTED_SELECTOR_SMOKE_OK")


def aggregate_fold_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for key in rows[0]:
        if key == "fold":
            continue
        vals = [row[key] for row in rows if isinstance(row.get(key), (int, float)) and row.get(key) is not None]
        if vals:
            out[f"{key}_mean"] = statistics.mean(vals)
            out[f"{key}_min"] = min(vals)
            out[f"{key}_max"] = max(vals)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["correction", "nested"])
    parser.add_argument("--true_csv", required=True)
    parser.add_argument("--zero_csv", required=True)
    parser.add_argument("--shuffle_csv", required=True)
    parser.add_argument("--normal_csv", required=True)
    parser.add_argument("--selected_csv", default="")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--thresholds_csv", default="")
    parser.add_argument("--risk_coverage_csv", default="")
    parser.add_argument("--internal_folds", type=int, default=5)
    parser.add_argument("--min_coverage", type=float, default=0.15)
    parser.add_argument("--max_coverage", type=float, default=0.95)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "correction":
        correction_report(args)
    else:
        if not args.thresholds_csv or not args.risk_coverage_csv:
            raise ValueError("--thresholds_csv and --risk_coverage_csv are required for nested mode.")
        nested_smoke(args)


if __name__ == "__main__":
    main()
