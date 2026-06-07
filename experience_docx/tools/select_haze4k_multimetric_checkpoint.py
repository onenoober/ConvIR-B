#!/usr/bin/env python3
"""Select a Haze4K checkpoint using regular+hard multi-metric evidence.

Inputs are compare JSON files produced by eval_haze4k_checkpoint_compare.py.
This script is table-only: it does not run models or touch locked test data.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load_compare(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["comparison"]


def find_compare(eval_dir: Path, prefix: str, label: str, split: str) -> Path:
    candidates = [
        eval_dir / f"scout_eval_compare_{prefix}_{label}_{split}_vs_a0.json",
        eval_dir / f"scout_eval_compare_{prefix}_{label.lower()}_{split}_vs_a0.json",
        eval_dir / f"scout_eval_compare_{prefix}_{label.upper()}_{split}_vs_a0.json",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return candidates[0]


def strong_ratio(compare: dict[str, Any]) -> float:
    count = float(compare.get("strong_reference_count", 0) or 0)
    if count <= 0:
        return 0.0
    return float(compare.get("strong_regression_count_delta_le_-0.05", 0) or 0) / count


def metric(compare: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = compare.get(key, default)
    return float(default if value is None else value)


def score_candidate(regular: dict[str, Any], hard: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    regular_strong = strong_ratio(regular)
    hard_strong = strong_ratio(hard)
    regular_worst = metric(regular, "worst_regression_count_delta_le_-0.20")
    hard_worst = metric(hard, "worst_regression_count_delta_le_-0.20")
    score = (
        1.0 * metric(regular, "mean_psnr_delta")
        + 1.2 * metric(hard, "hard_bottom25_psnr_delta")
        + 0.5 * metric(hard, "mean_psnr_delta")
        + 0.5 * metric(regular, "easy_top25_psnr_delta")
        + 20.0 * min(metric(regular, "mean_ssim_delta"), 0.01)
        + 20.0 * min(metric(hard, "mean_ssim_delta"), 0.01)
        - 0.025 * regular_worst
        - 0.035 * hard_worst
        - 0.75 * regular_strong
        - 0.50 * hard_strong
    )
    checks = {
        "regular_mean_nonnegative": metric(regular, "mean_psnr_delta") >= 0.0,
        "regular_easy_not_below_neg0p05": metric(regular, "easy_top25_psnr_delta") >= -0.05,
        "regular_ssim_nonnegative": metric(regular, "mean_ssim_delta") >= 0.0,
        "regular_worst_count_le_20": regular_worst <= 20,
        "regular_strong_ratio_le_0p25": regular_strong <= 0.25,
        "hard_mean_nonnegative": metric(hard, "mean_psnr_delta") >= 0.0,
        "hard_bottom25_ge_0p03": metric(hard, "hard_bottom25_psnr_delta") >= 0.03,
        "hard_ssim_nonnegative": metric(hard, "mean_ssim_delta") >= 0.0,
        "hard_worst_count_le_12": hard_worst <= 12,
    }
    return score, checks


def compact_metrics(compare: dict[str, Any]) -> dict[str, Any]:
    return {
        "mean_psnr_delta": metric(compare, "mean_psnr_delta"),
        "hard_bottom25_psnr_delta": metric(compare, "hard_bottom25_psnr_delta"),
        "easy_top25_psnr_delta": metric(compare, "easy_top25_psnr_delta"),
        "mean_ssim_delta": metric(compare, "mean_ssim_delta"),
        "positive_ratio": metric(compare, "positive_ratio"),
        "worst_count_delta_le_-0p20": metric(compare, "worst_regression_count_delta_le_-0.20"),
        "strong_regression_ratio": strong_ratio(compare),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", required=True)
    parser.add_argument("--candidate_prefix", required=True)
    parser.add_argument("--checkpoint_labels", nargs="+", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    rows = []
    details = {}
    for label in args.checkpoint_labels:
        regular_path = find_compare(eval_dir, args.candidate_prefix, label, "val_regular")
        hard_path = find_compare(eval_dir, args.candidate_prefix, label, "val_hard")
        if not regular_path.is_file() or not hard_path.is_file():
            rows.append(
                {
                    "checkpoint_label": label,
                    "present": False,
                    "score": "",
                    "decision": "MISSING_COMPARE_JSON",
                }
            )
            continue
        regular = load_compare(regular_path)
        hard = load_compare(hard_path)
        score, checks = score_candidate(regular, hard)
        pass_count = sum(1 for passed in checks.values() if passed)
        row = {
            "checkpoint_label": label,
            "present": True,
            "score": score,
            "pass_count": pass_count,
            "check_count": len(checks),
            "all_checks_pass": all(checks.values()),
            "decision": "CANDIDATE" if all(checks.values()) else "DIAGNOSTIC_ONLY",
        }
        for prefix, compare in (("regular", regular), ("hard", hard)):
            for key, value in compact_metrics(compare).items():
                row[f"{prefix}_{key}"] = value
        rows.append(row)
        details[label] = {
            "score": score,
            "checks": checks,
            "regular": compact_metrics(regular),
            "hard": compact_metrics(hard),
            "regular_compare_json": str(regular_path),
            "hard_compare_json": str(hard_path),
        }

    present_rows = [row for row in rows if row.get("present")]
    selected = None
    if present_rows:
        selected = max(present_rows, key=lambda row: (bool(row["all_checks_pass"]), float(row["score"])))
    payload = {
        "stage": "multi-metric checkpoint selection",
        "locked_test_touched": False,
        "eval_dir": str(eval_dir),
        "candidate_prefix": args.candidate_prefix,
        "checkpoint_labels": args.checkpoint_labels,
        "selected_checkpoint_label": selected["checkpoint_label"] if selected else None,
        "selected_score": selected["score"] if selected else None,
        "selected_all_checks_pass": selected["all_checks_pass"] if selected else False,
        "decision": (
            "MULTIMETRIC_CANDIDATE_SELECTED"
            if selected and selected["all_checks_pass"]
            else "NO_CHECKPOINT_PASSES_ALL_MULTIMETRIC_CHECKS"
        ),
        "rows": rows,
        "details": details,
    }

    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
