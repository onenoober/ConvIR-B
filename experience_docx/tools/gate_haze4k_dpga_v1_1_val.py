#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def load_compare(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    comparison = payload["comparison"]
    return payload, comparison


def check_value(observed, required_text, passed):
    return {
        "observed": observed,
        "required": required_text,
        "pass": bool(passed),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--best_compare_json", required=True)
    parser.add_argument("--final_compare_json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stage", default="DPGA-v1.1 val_inner scout")
    parser.add_argument("--strong_regression_ratio_limit", type=float, default=0.42)
    args = parser.parse_args()

    _best_payload, best = load_compare(args.best_compare_json)
    _final_payload, final = load_compare(args.final_compare_json)
    best_count = best["common_count"]
    best_strong_count = max(1, best["strong_reference_count"])
    final_count = final["common_count"]

    best_strong_ratio = best["strong_regression_count_delta_le_-0.05"] / best_strong_count
    best_worst_limit = max(1, round(0.12 * best_count))
    final_worst_limit = max(1, round(0.20 * final_count))

    checks = {
        "best_mean_psnr_delta": check_value(
            best["mean_psnr_delta"],
            ">= +0.030 dB",
            best["mean_psnr_delta"] >= 0.030,
        ),
        "best_mean_ssim_delta": check_value(
            best["mean_ssim_delta"],
            ">= 0",
            best["mean_ssim_delta"] >= 0,
        ),
        "best_hard_bottom25_delta": check_value(
            best["hard_bottom25_psnr_delta"],
            ">= +0.030 dB",
            best["hard_bottom25_psnr_delta"] >= 0.030,
        ),
        "best_easy_top25_delta": check_value(
            best["easy_top25_psnr_delta"],
            ">= 0",
            best["easy_top25_psnr_delta"] >= 0,
        ),
        "best_positive_ratio": check_value(
            best["positive_ratio"],
            ">= 0.55",
            best["positive_ratio"] >= 0.55,
        ),
        "best_strong_regression_ratio": check_value(
            best_strong_ratio,
            f"<= {args.strong_regression_ratio_limit:.2f}",
            best_strong_ratio <= args.strong_regression_ratio_limit,
        ),
        "best_worst_regressions": check_value(
            best["worst_regression_count_delta_le_-0.20"],
            f"<= {best_worst_limit} / {best_count}",
            best["worst_regression_count_delta_le_-0.20"] <= best_worst_limit,
        ),
        "final_mean_psnr_delta": check_value(
            final["mean_psnr_delta"],
            ">= 0",
            final["mean_psnr_delta"] >= 0,
        ),
        "final_worst_regressions": check_value(
            final["worst_regression_count_delta_le_-0.20"],
            f"<= {final_worst_limit} / {final_count}",
            final["worst_regression_count_delta_le_-0.20"] <= final_worst_limit,
        ),
    }
    passed = all(item["pass"] for item in checks.values())
    result = {
        "stage": args.stage,
        "best_compare_json": args.best_compare_json,
        "final_compare_json": args.final_compare_json,
        "selection_split": "val_inner",
        "diagnostic_only": True,
        "locked_test_allowed": passed,
        "checks": checks,
        "pass": passed,
        "next_step": (
            "Run one locked Haze4K test evaluation for the selected Best checkpoint."
            if passed
            else "Do not run locked test yet; inspect val_inner failure modes and training dynamics."
        ),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
