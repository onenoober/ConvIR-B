#!/usr/bin/env python3
import argparse
import csv
import json
import re
import statistics
from pathlib import Path


TAG_RE = re.compile(
    r"^scout_eval_per_image_(?P<tag>saferhfd_v2_rhfd2_(?P<rhfd2>[0-9.]+)_rhfd1_(?P<rhfd1>[0-9.]+)_vs_a0)\.csv$"
)


THRESHOLDS = {
    "mean_psnr_delta_vs_a0_min": 0.005,
    "mean_ssim_delta_min": 0.0,
    "hard_bottom_25pct_delta_min": 0.02,
    "easy_top_25pct_delta_min": 0.0,
    "severe_regression_delta_le_-0.20_max": 0,
    "strong_reference_regression_delta_le_-0.05_max": 0,
    "global_regression_delta_le_-0.05_max": 10,
    "hard_median_delta_min": -0.001,
    "hard_positive_ratio_min": 0.45,
    "mean_delta_excluding_top1_gain_min": 0.0,
    "hard_delta_excluding_top1_hard_gain_min": 0.0,
}


def mean(values):
    return statistics.mean(values) if values else None


def mean_without_top_gain(rows):
    if len(rows) <= 1:
        return None
    top_gain = max(rows, key=lambda row: row["delta_psnr"])
    kept = [row["delta_psnr"] for row in rows if row is not top_gain]
    return mean(kept)


def load_rows(csv_path, tag):
    candidate_psnr_key = f"{tag}_psnr"
    candidate_ssim_key = f"{tag}_ssim"
    rows = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            rows.append(
                {
                    "name": raw["name"],
                    "original_psnr": float(raw["original_psnr"]),
                    "candidate_psnr": float(raw[candidate_psnr_key]),
                    "delta_psnr": float(raw["delta_psnr"]),
                    "original_ssim": float(raw["original_ssim"]),
                    "candidate_ssim": float(raw[candidate_ssim_key]),
                    "delta_ssim": float(raw["delta_ssim"]),
                }
            )
    if not rows:
        raise ValueError(f"No rows found in {csv_path}")
    return rows


def check(name, value, op, threshold):
    if op == ">=":
        passed = value >= threshold
    elif op == "<=":
        passed = value <= threshold
    else:
        raise ValueError(op)
    return {
        "name": name,
        "value": value,
        "op": op,
        "threshold": threshold,
        "pass": passed,
    }


def summarize_candidate(output_dir, csv_path, tag, rhfd2, rhfd1):
    compare_path = output_dir / f"scout_eval_compare_{tag}.json"
    bucket_path = output_dir / f"scout_eval_bucket_analysis_{tag}.json"
    if not compare_path.is_file() or not bucket_path.is_file():
        raise FileNotFoundError(f"Missing compare or bucket JSON for {tag}")

    compare = json.loads(compare_path.read_text(encoding="utf-8"))
    buckets = json.loads(bucket_path.read_text(encoding="utf-8"))
    rows = load_rows(csv_path, tag)

    by_original = sorted(rows, key=lambda row: row["original_psnr"])
    hard = by_original[: len(by_original) // 4]
    comparison = compare["comparison"]
    difficulty = buckets["difficulty_buckets_by_original_psnr"]
    hard_bucket = difficulty["hard_bottom_25pct"]
    easy_bucket = difficulty["easy_top_25pct"]

    global_regressions = sum(row["delta_psnr"] <= -0.05 for row in rows)
    mean_ex_top1 = mean_without_top_gain(rows)
    hard_ex_top1 = mean_without_top_gain(hard)
    best_gain = max(rows, key=lambda row: row["delta_psnr"])
    best_hard_gain = max(hard, key=lambda row: row["delta_psnr"])

    metrics = {
        "rhfd2_scale": rhfd2,
        "rhfd1_scale": rhfd1,
        "mean_psnr_delta": comparison["mean_psnr_delta"],
        "mean_ssim_delta": comparison["mean_ssim_delta"],
        "hard_bottom_25pct_delta": hard_bucket["mean_delta_psnr"],
        "easy_top_25pct_delta": easy_bucket["mean_delta_psnr"],
        "severe_regression_delta_le_-0.20": comparison["worst_regression_count_delta_le_-0.20"],
        "strong_reference_regression_delta_le_-0.05": comparison[
            "strong_regression_count_delta_le_-0.05"
        ],
        "global_regression_delta_le_-0.05": global_regressions,
        "hard_median_delta": hard_bucket["median_delta_psnr"],
        "hard_positive_ratio": hard_bucket["positive_delta_ratio"],
        "mean_delta_excluding_top1_gain": mean_ex_top1,
        "hard_delta_excluding_top1_hard_gain": hard_ex_top1,
        "top1_gain_name": best_gain["name"],
        "top1_gain_delta": best_gain["delta_psnr"],
        "top1_hard_gain_name": best_hard_gain["name"],
        "top1_hard_gain_delta": best_hard_gain["delta_psnr"],
    }

    checks = [
        check("mean_psnr_delta_vs_a0", metrics["mean_psnr_delta"], ">=", THRESHOLDS["mean_psnr_delta_vs_a0_min"]),
        check("mean_ssim_delta", metrics["mean_ssim_delta"], ">=", THRESHOLDS["mean_ssim_delta_min"]),
        check(
            "hard_bottom_25pct_delta",
            metrics["hard_bottom_25pct_delta"],
            ">=",
            THRESHOLDS["hard_bottom_25pct_delta_min"],
        ),
        check(
            "easy_top_25pct_delta",
            metrics["easy_top_25pct_delta"],
            ">=",
            THRESHOLDS["easy_top_25pct_delta_min"],
        ),
        check(
            "severe_regressions",
            metrics["severe_regression_delta_le_-0.20"],
            "<=",
            THRESHOLDS["severe_regression_delta_le_-0.20_max"],
        ),
        check(
            "strong_reference_regressions",
            metrics["strong_reference_regression_delta_le_-0.05"],
            "<=",
            THRESHOLDS["strong_reference_regression_delta_le_-0.05_max"],
        ),
        check(
            "global_regressions",
            metrics["global_regression_delta_le_-0.05"],
            "<=",
            THRESHOLDS["global_regression_delta_le_-0.05_max"],
        ),
        check("hard_median_delta", metrics["hard_median_delta"], ">=", THRESHOLDS["hard_median_delta_min"]),
        check(
            "hard_positive_ratio",
            metrics["hard_positive_ratio"],
            ">=",
            THRESHOLDS["hard_positive_ratio_min"],
        ),
        check(
            "mean_delta_excluding_top1_gain",
            metrics["mean_delta_excluding_top1_gain"],
            ">=",
            THRESHOLDS["mean_delta_excluding_top1_gain_min"],
        ),
        check(
            "hard_delta_excluding_top1_hard_gain",
            metrics["hard_delta_excluding_top1_hard_gain"],
            ">=",
            THRESHOLDS["hard_delta_excluding_top1_hard_gain_min"],
        ),
    ]

    failed = [item["name"] for item in checks if not item["pass"]]
    return {
        "tag": tag,
        "compare_json": str(compare_path),
        "bucket_json": str(bucket_path),
        "per_image_csv": str(csv_path),
        **metrics,
        "strict_gate_pass": not failed,
        "failed_checks": failed,
        "gate_checks": checks,
    }


def recommendation_key(item):
    return (
        item["strict_gate_pass"],
        -item["strong_reference_regression_delta_le_-0.05"],
        -item["global_regression_delta_le_-0.05"],
        item["mean_delta_excluding_top1_gain"],
        item["hard_delta_excluding_top1_hard_gain"],
        item["mean_psnr_delta"],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    candidates = []
    for csv_path in sorted(output_dir.glob("scout_eval_per_image_saferhfd_v2_rhfd2_*_rhfd1_*_vs_a0.csv")):
        match = TAG_RE.match(csv_path.name)
        if not match:
            continue
        candidates.append(
            summarize_candidate(
                output_dir,
                csv_path,
                match.group("tag"),
                match.group("rhfd2"),
                match.group("rhfd1"),
            )
        )

    if not candidates:
        raise ValueError(f"No SafeRHFD-v2 stage-scale candidates found in {output_dir}")

    ranked = sorted(candidates, key=recommendation_key, reverse=True)
    strict_passing = [item["tag"] for item in ranked if item["strict_gate_pass"]]
    result = {
        "route": "SafeRHFD-v2 stage-wise calibrated B1 RHFD surgery",
        "thresholds": THRESHOLDS,
        "candidate_count": len(candidates),
        "strict_passing": strict_passing,
        "recommended_candidate": ranked[0]["tag"] if strict_passing else None,
        "best_failed_candidate": None if strict_passing else ranked[0]["tag"],
        "recommendation_rule": (
            "Prefer strict gate pass; then fewer strong/global regressions; "
            "then stronger top-1-excluded mean and hard gains. If no candidate "
            "passes the strict gate, do not recommend promotion."
        ),
        "candidates": ranked,
    }

    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    fieldnames = [
        "tag",
        "rhfd2_scale",
        "rhfd1_scale",
        "mean_psnr_delta",
        "mean_ssim_delta",
        "hard_bottom_25pct_delta",
        "easy_top_25pct_delta",
        "severe_regression_delta_le_-0.20",
        "strong_reference_regression_delta_le_-0.05",
        "global_regression_delta_le_-0.05",
        "hard_median_delta",
        "hard_positive_ratio",
        "mean_delta_excluding_top1_gain",
        "hard_delta_excluding_top1_hard_gain",
        "top1_gain_name",
        "top1_gain_delta",
        "top1_hard_gain_name",
        "top1_hard_gain_delta",
        "strict_gate_pass",
        "failed_checks",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in ranked:
            row = {key: item[key] for key in fieldnames}
            row["failed_checks"] = ";".join(row["failed_checks"])
            writer.writerow(row)

    print(json.dumps(result, indent=2))
    print(f"wrote {output_json}")
    print(f"wrote {output_csv}")


if __name__ == "__main__":
    main()
