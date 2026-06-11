#!/usr/bin/env python3
"""Low-capacity DTA-v3 post-hoc risk selector from per-image compare CSVs.

The selector is intentionally simple: one numeric no-reference diagnostic feature
from the true-depth eval row, one threshold, and one direction. Accepted images
use the DTA output; rejected images fall back to A0 (delta = 0).
"""

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
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * pct / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def auc_score(labels: list[int], scores: list[float]) -> float | None:
    pairs = [(score, label) for score, label in zip(scores, labels) if math.isfinite(score)]
    pos = [score for score, label in pairs if label == 1]
    neg = [score for score, label in pairs if label == 0]
    if not pos or not neg:
        return None
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def calibration(labels: list[int], probs: list[float], bins: int = 10) -> dict[str, float | None]:
    pairs = [(p, y) for p, y in zip(probs, labels) if math.isfinite(p)]
    if not pairs:
        return {"brier": None, "ece": None}
    brier = statistics.mean((p - y) ** 2 for p, y in pairs)
    ece = 0.0
    for idx in range(bins):
        lo = idx / bins
        hi = (idx + 1) / bins
        bucket = [(p, y) for p, y in pairs if (lo <= p < hi) or (idx == bins - 1 and p == hi)]
        if not bucket:
            continue
        conf = statistics.mean(p for p, _ in bucket)
        acc = statistics.mean(y for _, y in bucket)
        ece += len(bucket) / len(pairs) * abs(conf - acc)
    return {"brier": brier, "ece": ece}


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


def metric_summary(rows: list[dict[str, Any]], accept: list[bool]) -> dict[str, Any]:
    deltas = [row["true_delta"] if keep else 0.0 for row, keep in zip(rows, accept)]
    ssim_deltas = [row["true_dssim"] if keep else 0.0 for row, keep in zip(rows, accept)]
    original_psnr = [row["original_psnr"] for row in rows]
    sorted_idx = sorted(range(len(rows)), key=lambda idx: original_psnr[idx])
    bucket_count = max(1, len(rows) // 4)
    hard_idx = sorted_idx[:bucket_count]
    easy_idx = sorted_idx[-bucket_count:]
    strong_cut = percentile(original_psnr, 75)
    strong_idx = [idx for idx, psnr in enumerate(original_psnr) if psnr >= strong_cut]
    selected_count = sum(accept)
    def mean_at(indices: list[int]) -> float:
        return statistics.mean(deltas[idx] for idx in indices) if indices else float("nan")
    return {
        "count": len(rows),
        "selected_count": selected_count,
        "coverage": selected_count / len(rows) if rows else 0.0,
        "mean_psnr_delta": statistics.mean(deltas) if rows else None,
        "median_psnr_delta": statistics.median(deltas) if rows else None,
        "hard_bottom25_psnr_delta": mean_at(hard_idx),
        "easy_top25_psnr_delta": mean_at(easy_idx),
        "mean_ssim_delta": statistics.mean(ssim_deltas) if rows else None,
        "positive_ratio": sum(delta > 0 for delta in deltas) / len(deltas) if deltas else None,
        "strong_reference_cut_psnr": strong_cut,
        "strong_regression_count_delta_le_-0.05": sum(deltas[idx] <= -0.05 for idx in strong_idx),
        "worst_regression_count_delta_le_-0.20": sum(delta <= -0.20 for delta in deltas),
        "true_vs_zero_surplus": statistics.mean(
            (row["true_delta"] - row["zero_delta"]) if keep else 0.0 for row, keep in zip(rows, accept)
        ),
        "true_vs_shuffle_surplus": statistics.mean(
            (row["true_delta"] - row["shuffle_delta"]) if keep else 0.0 for row, keep in zip(rows, accept)
        ),
        "true_vs_normal_surplus": statistics.mean(
            (row["true_delta"] - row["normal_delta"]) if keep else 0.0 for row, keep in zip(rows, accept)
        ),
    }


def selector_objective(metrics: dict[str, Any]) -> float:
    mean_delta = metrics["mean_psnr_delta"] or 0.0
    hard_delta = metrics["hard_bottom25_psnr_delta"] or 0.0
    surplus = metrics["true_vs_zero_surplus"] or 0.0
    ssim_delta = metrics["mean_ssim_delta"] or 0.0
    worst = metrics["worst_regression_count_delta_le_-0.20"] or 0
    strong = metrics["strong_regression_count_delta_le_-0.05"] or 0
    coverage = metrics["coverage"] or 0.0
    return mean_delta + 0.25 * hard_delta + 0.5 * surplus + 20.0 * min(ssim_delta, 0.0) - 0.0015 * worst - 0.0005 * strong + 0.005 * coverage


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
    parser.add_argument("--true_csv", required=True)
    parser.add_argument("--zero_csv", required=True)
    parser.add_argument("--shuffle_csv", required=True)
    parser.add_argument("--normal_csv", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--threshold_trace_csv", required=True)
    parser.add_argument("--selected_csv", required=True)
    parser.add_argument("--min_coverage", type=float, default=0.15)
    parser.add_argument("--max_coverage", type=float, default=0.95)
    parser.add_argument("--risk_delta_thresh", type=float, default=-0.05)
    parser.add_argument("--gain_delta_thresh", type=float, default=0.02)
    args = parser.parse_args()

    true_rows = read_rows(Path(args.true_csv))
    zero_rows = read_rows(Path(args.zero_csv))
    shuffle_rows = read_rows(Path(args.shuffle_csv))
    normal_rows = read_rows(Path(args.normal_csv))
    common = sorted(set(true_rows) & set(zero_rows) & set(shuffle_rows) & set(normal_rows))
    if not common:
        raise ValueError("No common image names across depth-control CSVs.")

    feature_cols = candidate_feature_columns(true_rows[common[0]])
    rows = []
    for name in common:
        row = true_rows[name]
        rows.append(
            {
                "name": name,
                "original_psnr": finite_float(row["original_psnr"]),
                "true_delta": finite_float(row["delta_psnr"]),
                "true_dssim": finite_float(row["delta_ssim"], 0.0),
                "zero_delta": finite_float(zero_rows[name]["delta_psnr"]),
                "shuffle_delta": finite_float(shuffle_rows[name]["delta_psnr"]),
                "normal_delta": finite_float(normal_rows[name]["delta_psnr"]),
                "features": {col: finite_float(row.get(col)) for col in feature_cols},
            }
        )

    trace = []
    best = None
    best_accept = None
    for feature in feature_cols:
        values = [row["features"].get(feature, float("nan")) for row in rows]
        finite_values = [value for value in values if math.isfinite(value)]
        if len(set(finite_values)) < 2:
            continue
        thresholds = sorted({percentile(finite_values, pct) for pct in range(5, 100, 5)})
        for threshold in thresholds:
            for direction in ("<=", ">="):
                accept = [
                    (value <= threshold) if direction == "<=" else (value >= threshold)
                    for value in values
                ]
                metrics = metric_summary(rows, accept)
                if metrics["coverage"] < args.min_coverage or metrics["coverage"] > args.max_coverage:
                    continue
                out = {
                    "feature": feature,
                    "direction": direction,
                    "threshold": threshold,
                    "objective": selector_objective(metrics),
                }
                out.update(metrics)
                trace.append(out)
                if best is None or out["objective"] > best["objective"]:
                    best = out
                    best_accept = accept

    if best is None or best_accept is None:
        best_accept = [True] * len(rows)
        best = {"feature": "accept_all", "direction": ">=", "threshold": float("-inf"), "objective": None}
        best.update(metric_summary(rows, best_accept))

    labels_gain = [1 if row["true_delta"] > args.gain_delta_thresh else 0 for row in rows]
    labels_risk = [
        1 if row["true_delta"] < args.risk_delta_thresh or row["true_dssim"] < 0.0 else 0
        for row in rows
    ]
    selected_scores = []
    if best["feature"] != "accept_all":
        vals = [row["features"].get(best["feature"], float("nan")) for row in rows]
        finite_vals = [value for value in vals if math.isfinite(value)]
        lo = min(finite_vals)
        hi = max(finite_vals)
        span = hi - lo if hi > lo else 1.0
        # Higher score means more likely to accept / gain.
        for value in vals:
            norm = (value - lo) / span if math.isfinite(value) else 0.0
            selected_scores.append(1.0 - norm if best["direction"] == "<=" else norm)
    else:
        selected_scores = [1.0] * len(rows)
    gain_cal = calibration(labels_gain, selected_scores)
    risk_cal = calibration(labels_risk, [1.0 - score for score in selected_scores])

    selected_rows = []
    for row, keep, score in zip(rows, best_accept, selected_scores):
        selected_rows.append(
            {
                "name": row["name"],
                "accept_depth_action": keep,
                "selector_score": score,
                "selected_delta_psnr": row["true_delta"] if keep else 0.0,
                "selected_delta_ssim": row["true_dssim"] if keep else 0.0,
                "true_delta_psnr": row["true_delta"],
                "zero_delta_psnr": row["zero_delta"],
                "shuffle_delta_psnr": row["shuffle_delta"],
                "normal_delta_psnr": row["normal_delta"],
                "true_vs_zero_surplus_if_selected": row["true_delta"] - row["zero_delta"],
                "label_gain": row["true_delta"] > args.gain_delta_thresh,
                "label_risk": row["true_delta"] < args.risk_delta_thresh or row["true_dssim"] < 0.0,
            }
        )
    write_csv(Path(args.threshold_trace_csv), trace)
    write_csv(Path(args.selected_csv), selected_rows)

    result = {
        "protocol": "same_fold_diagnostic_threshold_selector",
        "warning": "Use only as fold0 diagnostic unless wrapped in nested OOF threshold selection.",
        "common_count": len(rows),
        "feature_count": len(feature_cols),
        "best_selector": best,
        "accept_all_baseline": metric_summary(rows, [True] * len(rows)),
        "gain_auc_selected_score": auc_score(labels_gain, selected_scores),
        "risk_auc_selected_risk_score": auc_score(labels_risk, [1.0 - score for score in selected_scores]),
        "gain_brier": gain_cal["brier"],
        "gain_ece": gain_cal["ece"],
        "risk_brier": risk_cal["brier"],
        "risk_ece": risk_cal["ece"],
        "locked_test_touched": False,
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print("DTA_V3_RISK_SELECTOR_OK")


if __name__ == "__main__":
    main()
