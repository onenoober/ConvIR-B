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
    elif op == "<":
        passed = value < threshold
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
    parser.add_argument("--stage", required=True, choices=["B1", "B2", "B3"])
    parser.add_argument("--compare_json", required=True)
    parser.add_argument("--bucket_json", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    compare = json.loads(Path(args.compare_json).read_text(encoding="utf-8"))
    buckets = json.loads(Path(args.bucket_json).read_text(encoding="utf-8"))
    comparison = compare["comparison"]
    difficulty = buckets["difficulty_buckets_by_original_psnr"]
    hard = difficulty["hard_bottom_25pct"]["mean_delta_psnr"]
    easy = difficulty["easy_top_25pct"]["mean_delta_psnr"]
    global_delta = comparison["mean_psnr_delta"]
    strong_regressions = comparison["strong_regression_count_delta_le_-0.05"]
    severe_regressions = comparison["worst_regression_count_delta_le_-0.20"]

    if args.stage == "B1":
        checks = [
            check("global_psnr_delta_vs_a1", global_delta, ">=", -0.05),
            check("easy_top25_mean_delta", easy, ">=", -0.05),
            check("strong_reference_regressions", strong_regressions, "<=", 50),
            check("severe_all_image_regressions", severe_regressions, "<", 444),
        ]
    elif args.stage == "B2":
        checks = [
            check("global_psnr_delta_vs_b1", global_delta, ">=", -0.03),
            check("hard_bottom25_delta_vs_b1", hard, ">", 0.0),
            check("easy_top25_mean_delta_vs_b1", easy, ">=", -0.05),
            check("strong_reference_regressions_vs_b1", strong_regressions, "<=", 50),
        ]
    else:
        checks = [
            check("global_psnr_delta_vs_b2", global_delta, ">=", -0.03),
            check("hard_bottom25_delta_vs_b2", hard, ">=", 0.0),
            check("easy_top25_drop_vs_b2", easy, ">", -0.03),
            check("strong_reference_regressions_vs_b2", strong_regressions, "<=", 50),
        ]

    result = {
        "stage": args.stage,
        "compare_json": args.compare_json,
        "bucket_json": args.bucket_json,
        "metrics": {
            "global_mean_delta_psnr": global_delta,
            "hard_bottom25_mean_delta_psnr": hard,
            "easy_top25_mean_delta_psnr": easy,
            "strong_reference_regressions_delta_le_-0.05": strong_regressions,
            "severe_all_image_regressions_delta_le_-0.20": severe_regressions,
        },
        "checks": checks,
        "automatic_pass": all(item["pass"] for item in checks),
        "manual_visual_artifact_check": "pending",
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["automatic_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
