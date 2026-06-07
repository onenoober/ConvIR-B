#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def load_compare(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload["comparison"]


def strong_regression_ratio(compare):
    strong_count = compare.get("strong_reference_count", 0)
    regressions = compare.get("strong_regression_count_delta_le_-0.05", 0)
    return regressions / strong_count if strong_count else 0.0


def check(observed, required, passed):
    return {"observed": observed, "required": required, "pass": bool(passed)}


def build_checks(regular, hard):
    regular_strong_ratio = strong_regression_ratio(regular)
    continue_checks = {
        "regular_mean_delta": check(regular["mean_psnr_delta"], ">= +0.035 dB", regular["mean_psnr_delta"] >= 0.035),
        "regular_positive_ratio": check(regular["positive_ratio"], ">= 0.620", regular["positive_ratio"] >= 0.62),
        "regular_worst_count": check(
            regular["worst_regression_count_delta_le_-0.20"],
            "<= 8",
            regular["worst_regression_count_delta_le_-0.20"] <= 8,
        ),
        "regular_strong_regression_ratio": check(regular_strong_ratio, "<= 0.180", regular_strong_ratio <= 0.18),
        "regular_ssim_delta": check(regular["mean_ssim_delta"], ">= 0", regular["mean_ssim_delta"] >= 0),
        "hard_mean_delta": check(hard["mean_psnr_delta"], ">= +0.030 dB", hard["mean_psnr_delta"] >= 0.030),
        "hard_bottom25_delta": check(
            hard["hard_bottom25_psnr_delta"],
            ">= +0.035 dB",
            hard["hard_bottom25_psnr_delta"] >= 0.035,
        ),
        "hard_worst_count": check(
            hard["worst_regression_count_delta_le_-0.20"],
            "<= 4",
            hard["worst_regression_count_delta_le_-0.20"] <= 4,
        ),
        "hard_ssim_delta": check(hard["mean_ssim_delta"], ">= 0", hard["mean_ssim_delta"] >= 0),
    }
    locked_checks = {
        "regular_mean_delta": check(regular["mean_psnr_delta"], ">= +0.040 dB", regular["mean_psnr_delta"] >= 0.040),
        "regular_positive_ratio": check(regular["positive_ratio"], ">= 0.620", regular["positive_ratio"] >= 0.62),
        "regular_strong_regression_ratio": check(regular_strong_ratio, "<= 0.160", regular_strong_ratio <= 0.16),
        "regular_worst_count": check(
            regular["worst_regression_count_delta_le_-0.20"],
            "<= 12",
            regular["worst_regression_count_delta_le_-0.20"] <= 12,
        ),
        "regular_ssim_delta": check(regular["mean_ssim_delta"], ">= 0", regular["mean_ssim_delta"] >= 0),
        "hard_mean_delta": check(hard["mean_psnr_delta"], ">= +0.030 dB", hard["mean_psnr_delta"] >= 0.030),
        "hard_bottom25_delta": check(
            hard["hard_bottom25_psnr_delta"],
            ">= +0.050 dB",
            hard["hard_bottom25_psnr_delta"] >= 0.050,
        ),
        "hard_ssim_delta": check(hard["mean_ssim_delta"], ">= 0", hard["mean_ssim_delta"] >= 0),
    }
    return continue_checks, locked_checks


def final_sanity_checks(final_regular, final_hard):
    return {
        "final_regular_mean_nonnegative": check(
            final_regular["mean_psnr_delta"],
            ">= 0 dB",
            final_regular["mean_psnr_delta"] >= 0,
        ),
        "final_hard_mean_nonnegative": check(
            final_hard["mean_psnr_delta"],
            ">= 0 dB",
            final_hard["mean_psnr_delta"] >= 0,
        ),
        "final_regular_worst_count": check(
            final_regular["worst_regression_count_delta_le_-0.20"],
            "<= 20",
            final_regular["worst_regression_count_delta_le_-0.20"] <= 20,
        ),
        "final_hard_worst_count": check(
            final_hard["worst_regression_count_delta_le_-0.20"],
            "<= 20",
            final_hard["worst_regression_count_delta_le_-0.20"] <= 20,
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--best_regular_compare_json", required=True)
    parser.add_argument("--best_hard_compare_json", required=True)
    parser.add_argument("--final_regular_compare_json", required=True)
    parser.add_argument("--final_hard_compare_json", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    best_regular = load_compare(args.best_regular_compare_json)
    best_hard = load_compare(args.best_hard_compare_json)
    final_regular = load_compare(args.final_regular_compare_json)
    final_hard = load_compare(args.final_hard_compare_json)
    continue_checks, locked_checks = build_checks(best_regular, best_hard)
    sanity_checks = final_sanity_checks(final_regular, final_hard)
    continue_pass = all(item["pass"] for item in continue_checks.values()) and all(
        item["pass"] for item in sanity_checks.values()
    )
    locked_pass = all(item["pass"] for item in locked_checks.values()) and all(
        item["pass"] for item in sanity_checks.values()
    )
    result = {
        "stage": "ConvIR-Dehaze-v1.4B-BiDPFM1 regular+hard gate",
        "diagnostic_only": True,
        "continue_allowed": continue_pass,
        "locked_test_allowed": locked_pass,
        "continue_checks": continue_checks,
        "locked_test_checks": locked_checks,
        "final_sanity_checks": sanity_checks,
        "pass": continue_pass,
        "next_step": (
            "Run the single locked Haze4K test only if this configuration and checkpoint were selected before seeing locked-test data."
            if locked_pass
            else "Do not run locked Haze4K test. If continue_allowed is true, proceed to v1.4C full-res fusion-neighbor adapter; otherwise stop this adapter-only form."
        ),
        "inputs": vars(args),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not continue_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
