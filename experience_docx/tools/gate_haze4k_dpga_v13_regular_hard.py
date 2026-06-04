#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def load_compare(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload["comparison"]


def check(observed, required, passed):
    return {"observed": observed, "required": required, "pass": bool(passed)}


def gate_stage(args, regular, hard, prefix):
    worst_budget = args.worst_budget
    regular_strong_count = regular.get("strong_reference_count", 0)
    regular_strong_regressions = regular.get("strong_regression_count_delta_le_-0.05", 0)
    strong_ratio = (
        regular_strong_regressions / regular_strong_count
        if regular_strong_count
        else 0.0
    )
    checks = {
        f"{prefix}_regular_mean_delta": check(
            regular["mean_psnr_delta"],
            f">= {args.regular_mean_min:+.3f} dB",
            regular["mean_psnr_delta"] >= args.regular_mean_min,
        ),
        f"{prefix}_regular_easy_delta": check(
            regular["easy_top25_psnr_delta"],
            ">= 0 dB",
            regular["easy_top25_psnr_delta"] >= 0,
        ),
        f"{prefix}_regular_worst_count": check(
            regular["worst_regression_count_delta_le_-0.20"],
            f"<= {worst_budget}",
            regular["worst_regression_count_delta_le_-0.20"] <= worst_budget,
        ),
        f"{prefix}_regular_ssim_delta": check(
            regular["mean_ssim_delta"],
            ">= 0",
            regular["mean_ssim_delta"] >= 0,
        ),
        f"{prefix}_hard_bottom25_delta": check(
            hard["hard_bottom25_psnr_delta"],
            f">= {args.hard_bottom25_min:+.3f} dB",
            hard["hard_bottom25_psnr_delta"] >= args.hard_bottom25_min,
        ),
        f"{prefix}_hard_mean_delta": check(
            hard["mean_psnr_delta"],
            f">= {args.hard_mean_min:+.3f} dB",
            hard["mean_psnr_delta"] >= args.hard_mean_min,
        ),
        f"{prefix}_hard_ssim_delta": check(
            hard["mean_ssim_delta"],
            ">= 0",
            hard["mean_ssim_delta"] >= 0,
        ),
    }
    if args.positive_ratio_min > 0:
        checks[f"{prefix}_regular_positive_ratio"] = check(
            regular["positive_ratio"],
            f">= {args.positive_ratio_min:.3f}",
            regular["positive_ratio"] >= args.positive_ratio_min,
        )
    if args.strong_ratio_max >= 0:
        checks[f"{prefix}_regular_strong_regression_ratio"] = check(
            strong_ratio,
            f"<= {args.strong_ratio_max:.3f}",
            strong_ratio <= args.strong_ratio_max,
        )
    score = (
        regular["mean_psnr_delta"]
        + args.hard_score_weight * hard["hard_bottom25_psnr_delta"]
        + args.easy_score_weight * regular["easy_top25_psnr_delta"]
        - args.worst_penalty_weight
        * max(0, regular["worst_regression_count_delta_le_-0.20"] - worst_budget)
        / max(1, worst_budget)
    )
    return checks, score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--best_regular_compare_json", required=True)
    parser.add_argument("--best_hard_compare_json", required=True)
    parser.add_argument("--final_regular_compare_json", required=True)
    parser.add_argument("--final_hard_compare_json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stage", default="DPGA-v1.3A regular+hard gate")
    parser.add_argument("--regular_mean_min", type=float, default=0.035)
    parser.add_argument("--hard_bottom25_min", type=float, default=0.040)
    parser.add_argument("--hard_mean_min", type=float, default=0.0)
    parser.add_argument("--worst_budget", type=int, default=12)
    parser.add_argument("--positive_ratio_min", type=float, default=0.0)
    parser.add_argument("--strong_ratio_max", type=float, default=-1.0)
    parser.add_argument("--hard_score_weight", type=float, default=0.8)
    parser.add_argument("--easy_score_weight", type=float, default=0.2)
    parser.add_argument("--worst_penalty_weight", type=float, default=0.2)
    parser.add_argument(
        "--failure_next_step",
        default="Do not run locked Haze4K test; inspect mask/sampler/runtime audits before any separately justified follow-up route.",
    )
    args = parser.parse_args()

    best_regular = load_compare(args.best_regular_compare_json)
    best_hard = load_compare(args.best_hard_compare_json)
    final_regular = load_compare(args.final_regular_compare_json)
    final_hard = load_compare(args.final_hard_compare_json)
    best_checks, best_score = gate_stage(args, best_regular, best_hard, "best")
    final_checks = {
        "final_regular_mean_delta": check(
            final_regular["mean_psnr_delta"],
            ">= 0 dB",
            final_regular["mean_psnr_delta"] >= 0,
        ),
        "final_regular_worst_count": check(
            final_regular["worst_regression_count_delta_le_-0.20"],
            f"<= {max(args.worst_budget, 20)}",
            final_regular["worst_regression_count_delta_le_-0.20"] <= max(args.worst_budget, 20),
        ),
        "final_hard_mean_delta": check(
            final_hard["mean_psnr_delta"],
            ">= 0 dB",
            final_hard["mean_psnr_delta"] >= 0,
        ),
    }
    checks = {**best_checks, **final_checks}
    passed = all(item["pass"] for item in checks.values())
    result = {
        "stage": args.stage,
        "selection_score": best_score,
        "diagnostic_only": True,
        "locked_test_allowed": False,
        "checks": checks,
        "pass": passed,
        "next_step": (
            "If v1.3A hard gain is near +0.04 dB, keep this as the loss-mask root-cause route and avoid v1.3B for now."
            if passed
            else args.failure_next_step
        ),
        "inputs": {
            "best_regular_compare_json": args.best_regular_compare_json,
            "best_hard_compare_json": args.best_hard_compare_json,
            "final_regular_compare_json": args.final_regular_compare_json,
            "final_hard_compare_json": args.final_hard_compare_json,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
