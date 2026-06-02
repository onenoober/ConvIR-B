import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare_json", required=True)
    parser.add_argument("--bucket_json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stage", default="APDR stop20")
    args = parser.parse_args()

    compare = json.loads(Path(args.compare_json).read_text(encoding="utf-8"))
    bucket = json.loads(Path(args.bucket_json).read_text(encoding="utf-8"))

    c = compare["comparison"]
    overall = bucket["overall"]
    b = bucket["difficulty_buckets_by_original_psnr"]
    hard = b["hard_bottom_25pct"]["mean_delta_psnr"]
    easy = b["easy_top_25pct"]["mean_delta_psnr"]

    checks = {
        "mean_psnr_delta": {
            "observed": c["mean_psnr_delta"],
            "required": ">= +0.020",
            "pass": c["mean_psnr_delta"] >= 0.020,
        },
        "mean_ssim_delta": {
            "observed": c["mean_ssim_delta"],
            "required": ">= 0",
            "pass": c["mean_ssim_delta"] >= 0,
        },
        "hard_bottom_25pct_delta": {
            "observed": hard,
            "required": ">= +0.080",
            "pass": hard >= 0.080,
        },
        "easy_top_25pct_delta": {
            "observed": easy,
            "required": ">= -0.010",
            "pass": easy >= -0.010,
        },
        "strong_reference_regressions": {
            "observed": c["strong_regression_count_delta_le_-0.05"],
            "required": "<= 30 / 250",
            "pass": c["strong_regression_count_delta_le_-0.05"] <= 30,
        },
        "severe_regressions": {
            "observed": c["worst_regression_count_delta_le_-0.20"],
            "required": "<= 10 / 1000",
            "pass": c["worst_regression_count_delta_le_-0.20"] <= 10,
        },
        "worst10img_mean_delta": {
            "observed": overall["worst_10_mean_delta_psnr"],
            "required": "> -0.300",
            "pass": overall["worst_10_mean_delta_psnr"] > -0.300,
        },
        "median_psnr_delta": {
            "observed": c["median_psnr_delta"],
            "required": ">= 0",
            "pass": c["median_psnr_delta"] >= 0,
        },
    }
    result = {
        "stage": args.stage,
        "compare_json": args.compare_json,
        "bucket_json": args.bucket_json,
        "checks": checks,
        "pass": all(item["pass"] for item in checks.values()),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
