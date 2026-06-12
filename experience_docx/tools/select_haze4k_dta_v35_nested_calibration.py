#!/usr/bin/env python3
"""Build DTA-v3.5 action tables, oracle curves, and nested selector reports."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

RUN_RE = re.compile(r"v35_fdf_rcs_(?P<variant>.+)_seed(?P<seed>\d+)_f(?P<fold>\d+)_(?P<stage>[^_]+.*)$")
BASE_COLUMNS = {
    "name",
    "original_psnr",
    "delta_psnr",
    "original_ssim",
    "delta_ssim",
    "original_time_sec",
}


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(v for v in values if math.isfinite(v))
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def compare_csv(compare_dir: Path) -> Path:
    matches = sorted(compare_dir.glob("scout_eval_per_image_*.csv"))
    if not matches:
        raise FileNotFoundError(f"No scout_eval_per_image_*.csv under {compare_dir}")
    return matches[0]


def matrix_meta(path: Path) -> dict[str, str]:
    name = path.stem.removeprefix("train_eval_depth_matrix_").removesuffix("_fallback_train")
    match = RUN_RE.match(name)
    meta = match.groupdict() if match else {"variant": name, "seed": "", "fold": "", "stage": ""}
    meta["run_id"] = name
    return meta


def label_compare_dirs(matrix_path: Path) -> dict[str, Path]:
    payload = json.loads(matrix_path.read_text(encoding="utf-8"))
    out = {}
    for row in payload.get("runs", []):
        out[row["label"]] = Path(row["compare_dir"])
    missing = [label for label in ("true", "zero", "shuffle", "normal") if label not in out]
    if missing:
        raise ValueError(f"{matrix_path} missing compare dirs for {missing}")
    return out


def normalize_feature_key(key: str) -> str | None:
    if key in BASE_COLUMNS or key.endswith("_psnr") or key.endswith("_ssim") or key.endswith("_time_sec"):
        return None
    if key.startswith("original_"):
        suffix = key[len("original_"):]
        if suffix.startswith(("input_", "depth_", "airlight_", "trans_gt_")):
            return suffix
        return None
    for marker in ("_dta_", "_input_", "_depth_", "_airlight_", "_trans_gt_"):
        pos = key.find(marker)
        if pos >= 0:
            return key[pos + 1:]
    return None


def feature_map(row: dict[str, str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for key, value in row.items():
        norm = normalize_feature_key(key)
        if norm is None or not math.isfinite(finite_float(value)):
            continue
        current = mapping.get(norm)
        # Prefer candidate-side DTA/action stats over blank original-side columns.
        if current is None or current.startswith("original_"):
            mapping[norm] = key
    return mapping


def metric_summary(rows: list[dict[str, Any]], accept: list[bool]) -> dict[str, Any]:
    deltas = [row["dPSNR"] if keep else 0.0 for row, keep in zip(rows, accept)]
    ssim_deltas = [row["dSSIM"] if keep else 0.0 for row, keep in zip(rows, accept)]
    original_psnr = [row["A0_PSNR"] for row in rows]
    sorted_idx = sorted(range(len(rows)), key=lambda idx: original_psnr[idx])
    bucket_count = max(1, len(rows) // 4)
    hard_idx = sorted_idx[:bucket_count]
    easy_idx = sorted_idx[-bucket_count:]
    strong_cut = percentile(original_psnr, 75)
    strong_idx = [idx for idx, psnr in enumerate(original_psnr) if psnr >= strong_cut]
    selected = [row for row, keep in zip(rows, accept) if keep]
    selected_deltas = [row["dPSNR"] for row in selected]

    def mean_at(indices: list[int]) -> float:
        return statistics.mean(deltas[idx] for idx in indices) if indices else float("nan")

    return {
        "count": len(rows),
        "selected_count": sum(accept),
        "coverage": sum(accept) / len(rows) if rows else 0.0,
        "mean_psnr_delta": statistics.mean(deltas) if rows else None,
        "hard_bottom25_psnr_delta": mean_at(hard_idx),
        "easy_top25_psnr_delta": mean_at(easy_idx),
        "mean_ssim_delta": statistics.mean(ssim_deltas) if rows else None,
        "positive_ratio": sum(delta > 0 for delta in deltas) / len(deltas) if deltas else None,
        "selected_positive_ratio": (sum(delta > 0 for delta in selected_deltas) / len(selected_deltas)) if selected_deltas else None,
        "strong_reference_cut_psnr": strong_cut,
        "strong_regression_count_delta_le_-0.05": sum(deltas[idx] <= -0.05 for idx in strong_idx),
        "worst_regression_count_delta_le_-0.20": sum(delta <= -0.20 for delta in deltas),
        "true_vs_zero_surplus": statistics.mean(
            (row["dPSNR"] - row["zero_delta_psnr"]) if keep else 0.0 for row, keep in zip(rows, accept)
        ) if rows else None,
        "true_vs_shuffle_surplus": statistics.mean(
            (row["dPSNR"] - row["shuffle_delta_psnr"]) if keep else 0.0 for row, keep in zip(rows, accept)
        ) if rows else None,
        "true_vs_normal_surplus": statistics.mean(
            (row["dPSNR"] - row["normal_delta_psnr"]) if keep else 0.0 for row, keep in zip(rows, accept)
        ) if rows else None,
        "selected_group_counts": dict(Counter(row["failure_group"] for row in selected)),
    }


def selector_objective(metrics: dict[str, Any]) -> float:
    mean_delta = metrics.get("mean_psnr_delta") or 0.0
    hard_delta = metrics.get("hard_bottom25_psnr_delta") or 0.0
    surplus = metrics.get("true_vs_zero_surplus") or 0.0
    ssim_delta = metrics.get("mean_ssim_delta") or 0.0
    selected_pos = metrics.get("selected_positive_ratio") or 0.0
    coverage = metrics.get("coverage") or 0.0
    worst = metrics.get("worst_regression_count_delta_le_-0.20") or 0
    strong = metrics.get("strong_regression_count_delta_le_-0.05") or 0
    return (
        mean_delta
        + 0.25 * hard_delta
        + 0.35 * surplus
        + 0.010 * selected_pos
        + 0.004 * coverage
        + 20.0 * min(ssim_delta, 0.0)
        - 0.0015 * worst
        - 0.0005 * strong
    )


def fit_threshold(rows: list[dict[str, Any]], features: list[str], min_coverage: float, max_coverage: float) -> tuple[dict[str, Any], list[bool]]:
    best: dict[str, Any] | None = None
    best_accept: list[bool] | None = None
    for feature in features:
        values = [finite_float(row.get(feature)) for row in rows]
        finite_values = [value for value in values if math.isfinite(value)]
        if len(set(finite_values)) < 2:
            continue
        thresholds = sorted({percentile(finite_values, pct) for pct in range(5, 100, 5)})
        for threshold in thresholds:
            for direction in ("<=", ">="):
                accept = [(value <= threshold) if direction == "<=" else (value >= threshold) for value in values]
                metrics = metric_summary(rows, accept)
                if metrics["coverage"] < min_coverage or metrics["coverage"] > max_coverage:
                    continue
                candidate = {
                    "feature": feature,
                    "direction": direction,
                    "threshold": threshold,
                    "objective": selector_objective(metrics),
                }
                candidate.update(metrics)
                if best is None or candidate["objective"] > best["objective"]:
                    best = candidate
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
    threshold = float(selector["threshold"])
    if selector["direction"] == "<=":
        return [finite_float(row.get(feature)) <= threshold for row in rows]
    return [finite_float(row.get(feature)) >= threshold for row in rows]


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
            writer.writerow({key: row.get(key, "") for key in keys})


def build_rows(evidence_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for matrix_path in sorted(evidence_dir.glob("train_eval_depth_matrix_v35_fdf_rcs_*_fallback_train.json")):
        meta = matrix_meta(matrix_path)
        compare_dirs = label_compare_dirs(matrix_path)
        per_label = {label: {row["name"]: row for row in read_csv(compare_csv(path))} for label, path in compare_dirs.items()}
        common = sorted(set(per_label["true"]) & set(per_label["zero"]) & set(per_label["shuffle"]) & set(per_label["normal"]))
        if not common:
            continue
        fmap = feature_map(per_label["true"][common[0]])
        for name in common:
            true_row = per_label["true"][name]
            dpsnr = finite_float(true_row.get("delta_psnr"))
            dssim = finite_float(true_row.get("delta_ssim"), 0.0)
            out: dict[str, Any] = {
                "image_id": name,
                "fold": meta["fold"],
                "seed": meta["seed"],
                "variant": meta["variant"],
                "run_id": meta["run_id"],
                "A0_PSNR": finite_float(true_row.get("original_psnr")),
                "cand_PSNR": finite_float(next((value for key, value in true_row.items() if key.endswith("_psnr") and key not in {"original_psnr", "delta_psnr"}), "nan")),
                "dPSNR": dpsnr,
                "dSSIM": dssim,
                "zero_delta_psnr": finite_float(per_label["zero"][name].get("delta_psnr")),
                "shuffle_delta_psnr": finite_float(per_label["shuffle"][name].get("delta_psnr")),
                "normal_delta_psnr": finite_float(per_label["normal"][name].get("delta_psnr")),
            }
            if dpsnr <= -0.20:
                out["failure_group"] = "worst_regression"
            elif dpsnr <= -0.05:
                out["failure_group"] = "strong_regression"
            elif dssim < 0.0:
                out["failure_group"] = "ssim_regression"
            elif dpsnr > 0.02:
                out["failure_group"] = "gain"
            else:
                out["failure_group"] = "neutral"
            for norm, key in fmap.items():
                out[norm] = finite_float(true_row.get(key))
            rows.append(out)
    return rows


def oracle_curve(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for variant in sorted({row["variant"] for row in rows}):
        group = [row for row in rows if row["variant"] == variant]
        ordered = sorted(group, key=lambda row: (row["dPSNR"], row["dSSIM"]), reverse=True)
        for pct in range(5, 101, 5):
            keep_n = max(1, round(len(ordered) * pct / 100.0))
            accepted = {id(row) for row in ordered[:keep_n]}
            accept = [id(row) in accepted for row in group]
            metrics = metric_summary(group, accept)
            out.append({
                "variant": variant,
                "coverage_target": pct / 100.0,
                **{key: (json.dumps(value, sort_keys=True) if key == "selected_group_counts" else value) for key, value in metrics.items()},
            })
    return out


def nested_reports(rows: list[dict[str, Any]], min_coverage: float, max_coverage: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    reports: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    excluded = {"image_id", "fold", "seed", "variant", "run_id", "failure_group", "A0_PSNR", "cand_PSNR", "dPSNR", "dSSIM", "zero_delta_psnr", "shuffle_delta_psnr", "normal_delta_psnr"}
    feature_names = sorted({
        key
        for row in rows
        for key in row
        if key not in excluded and any(token in key for token in ("input_", "depth_", "airlight_", "trans_gt_", "dta_"))
    })
    for variant in sorted({row["variant"] for row in rows}):
        variant_rows = [row for row in rows if row["variant"] == variant]
        folds = sorted({row["fold"] for row in variant_rows})
        for outer_fold in folds:
            eval_rows = [row for row in variant_rows if row["fold"] == outer_fold]
            calib_rows = [row for row in variant_rows if row["fold"] != outer_fold]
            if not eval_rows or not calib_rows:
                continue
            selector, _ = fit_threshold(calib_rows, feature_names, min_coverage, max_coverage)
            accept = apply_selector(eval_rows, selector)
            metrics = metric_summary(eval_rows, accept)
            report = {
                "variant": variant,
                "outer_fold": outer_fold,
                "inner_calibration_fold": ",".join(sorted({row["fold"] for row in calib_rows})),
                "feature": selector["feature"],
                "direction": selector["direction"],
                "threshold": selector["threshold"],
                "calibration_objective": selector["objective"],
                "calibration_coverage": selector["coverage"],
                "coverage": metrics["coverage"],
                "selected_mean": metrics["mean_psnr_delta"],
                "selected_hard": metrics["hard_bottom25_psnr_delta"],
                "selected_dssim": metrics["mean_ssim_delta"],
                "selected_positive_ratio": metrics["selected_positive_ratio"],
                "all_image_positive_ratio": metrics["positive_ratio"],
                "worst": metrics["worst_regression_count_delta_le_-0.20"],
                "true_vs_zero": metrics["true_vs_zero_surplus"],
                "true_vs_shuffle": metrics["true_vs_shuffle_surplus"],
                "true_vs_normal": metrics["true_vs_normal_surplus"],
            }
            reports.append(report)
            for row, keep in zip(eval_rows, accept):
                selected_rows.append({
                    "variant": variant,
                    "outer_fold": outer_fold,
                    "image_id": row["image_id"],
                    "seed": row["seed"],
                    "accept_depth_action": keep,
                    "selected_delta_psnr": row["dPSNR"] if keep else 0.0,
                    "selected_delta_ssim": row["dSSIM"] if keep else 0.0,
                    "true_delta_psnr": row["dPSNR"],
                    "zero_delta_psnr": row["zero_delta_psnr"],
                    "shuffle_delta_psnr": row["shuffle_delta_psnr"],
                    "normal_delta_psnr": row["normal_delta_psnr"],
                    "failure_group": row["failure_group"],
                })
    return reports, selected_rows


def aggregate_nested(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in reports:
        grouped[row["variant"]].append(row)
    for variant, group in sorted(grouped.items()):
        avg = lambda key: statistics.mean(finite_float(row.get(key), 0.0) for row in group)
        out.append({
            "variant": variant,
            "outer_reports": len(group),
            "coverage": avg("coverage"),
            "selected_mean": avg("selected_mean"),
            "selected_hard": avg("selected_hard"),
            "selected_dssim": avg("selected_dssim"),
            "selected_positive_ratio": avg("selected_positive_ratio"),
            "all_image_positive_ratio": avg("all_image_positive_ratio"),
            "worst": avg("worst"),
            "max_outer_worst": max(finite_float(row.get("worst"), 0.0) for row in group),
            "true_vs_zero": avg("true_vs_zero"),
            "true_vs_shuffle": avg("true_vs_shuffle"),
            "true_vs_normal": avg("true_vs_normal"),
            "relaxed_selector_flow_pass": avg("coverage") >= 0.20 and avg("selected_mean") >= -0.02 and avg("worst") <= 180,
            "strict_selector_gate_pass": avg("coverage") >= 0.35 and avg("selected_positive_ratio") >= 0.75 and avg("all_image_positive_ratio") >= 0.63 and max(finite_float(row.get("worst"), 0.0) for row in group) <= 60,
        })
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence_dir", required=True, type=Path)
    parser.add_argument("--action_table_csv", required=True, type=Path)
    parser.add_argument("--oracle_curve_csv", required=True, type=Path)
    parser.add_argument("--nested_report_json", required=True, type=Path)
    parser.add_argument("--nested_report_csv", required=True, type=Path)
    parser.add_argument("--nested_selected_csv", required=True, type=Path)
    parser.add_argument("--min_coverage", type=float, default=0.20)
    parser.add_argument("--max_coverage", type=float, default=0.95)
    args = parser.parse_args()

    rows = build_rows(args.evidence_dir)
    if not rows:
        raise ValueError(f"No v3.5 matrix rows found under {args.evidence_dir}")
    write_csv(args.action_table_csv, rows)
    write_csv(args.oracle_curve_csv, oracle_curve(rows))
    reports, selected_rows = nested_reports(rows, args.min_coverage, args.max_coverage)
    aggregate = aggregate_nested(reports)
    write_csv(args.nested_report_csv, reports)
    write_csv(args.nested_selected_csv, selected_rows)
    payload = {
        "protocol": "nested_fold_threshold_selector_relaxed_flow",
        "min_coverage": args.min_coverage,
        "max_coverage": args.max_coverage,
        "action_rows": len(rows),
        "feature_note": "Threshold selector uses only per-image diagnostics from train-derived OOF folds; locked test is untouched.",
        "reports": reports,
        "variant_summaries": aggregate,
        "locked_test_touched": False,
    }
    args.nested_report_json.parent.mkdir(parents=True, exist_ok=True)
    args.nested_report_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"DTA_V3_5_NESTED_SELECTOR_OK rows={len(rows)} reports={len(reports)} variants={len(aggregate)}")


if __name__ == "__main__":
    main()
