import argparse
import json
from pathlib import Path


def check(name, value, op, threshold):
    if op == ">=":
        passed = value >= threshold
    elif op == ">":
        passed = value > threshold
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare_json", required=True)
    parser.add_argument("--bucket_json", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    compare = json.loads(Path(args.compare_json).read_text(encoding="utf-8"))
    buckets = json.loads(Path(args.bucket_json).read_text(encoding="utf-8"))
    comparison = compare["comparison"]
    difficulty = buckets["difficulty_buckets_by_original_psnr"]

    metrics = {
        "global_mean_delta_psnr": comparison["mean_psnr_delta"],
        "mean_delta_ssim": comparison["mean_ssim_delta"],
        "hard_bottom25_mean_delta_psnr": difficulty["hard_bottom_25pct"]["mean_delta_psnr"],
        "easy_top25_mean_delta_psnr": difficulty["easy_top_25pct"]["mean_delta_psnr"],
        "strong_reference_regressions_delta_le_-0.05": comparison[
            "strong_regression_count_delta_le_-0.05"
        ],
        "severe_all_image_regressions_delta_le_-0.20": comparison[
            "worst_regression_count_delta_le_-0.20"
        ],
        "worst10_mean_delta_psnr": comparison["worst10_mean_psnr_delta"],
    }
    checks = [
        check("global_mean_delta_psnr_vs_a0", metrics["global_mean_delta_psnr"], ">=", 0.02),
        check("mean_delta_ssim_vs_a0", metrics["mean_delta_ssim"], ">=", 0.0),
        check("hard_bottom25_mean_delta_psnr_vs_a0", metrics["hard_bottom25_mean_delta_psnr"], ">=", 0.08),
        check("easy_top25_mean_delta_psnr_vs_a0", metrics["easy_top25_mean_delta_psnr"], ">=", -0.02),
        check("strong_reference_regressions", metrics["strong_reference_regressions_delta_le_-0.05"], "<=", 30),
        check("severe_all_image_regressions", metrics["severe_all_image_regressions_delta_le_-0.20"], "<=", 20),
        check("worst10_mean_delta_psnr", metrics["worst10_mean_delta_psnr"], ">", -0.50),
    ]
    result = {
        "route": "B1-v2 SafeRHFD pfd-only",
        "comparison_target": "A0 official ConvIR-B checkpoint",
        "compare_json": args.compare_json,
        "bucket_json": args.bucket_json,
        "metrics": metrics,
        "checks": checks,
        "automatic_pass": all(item["pass"] for item in checks),
        "manual_visual_artifact_check": "required before promotion",
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["automatic_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
