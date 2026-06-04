#!/usr/bin/env python3
import argparse
import csv
import datetime as dt
import json
import math
from pathlib import Path


MODULE_CSV = "dpga_module_ablation_best_final.csv"
SCALE_CSV = "dpga_scale_sweep_best_final.csv"


METRICS = (
    "mean_delta",
    "median_delta",
    "hard_bottom25_delta",
    "easy_top25_delta",
    "mean_ssim_delta",
    "positive_ratio",
    "strong_ref_regressions",
    "worst_0p20_count",
    "p5_delta",
    "p95_delta",
    "bright_low_gradient_delta",
    "low_saturation_bright_delta",
    "sky_bright_proxy_delta",
    "hard_bright_low_gradient_delta",
    "easy_bright_low_gradient_delta",
)


def as_float(row, key, default=None):
    value = row.get(key, "")
    if value in ("", None):
        return default
    try:
        return float(value)
    except ValueError:
        return default


def mean(values):
    values = [value for value in values if value is not None and not math.isnan(value)]
    if not values:
        return None
    return sum(values) / len(values)


def read_rows(path):
    with open(path, "r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def row_key(row):
    return (row["checkpoint"], row["variant"])


def summarize_rows(rows):
    summary = {}
    for key in METRICS:
        summary[key] = mean(as_float(row, key) for row in rows)
    checkpoints = sorted({row["checkpoint"] for row in rows})
    summary["checkpoints"] = checkpoints
    summary["rows"] = len(rows)
    summary["common_count"] = min(int(as_float(row, "common_count", 0) or 0) for row in rows)
    return summary


def baseline_by_checkpoint(rows, variant):
    return {row["checkpoint"]: row for row in rows if row["variant"] == variant}


def count_delta(summary, baseline_summary, key):
    value = summary.get(key)
    baseline = baseline_summary.get(key)
    if value is None or baseline is None:
        return None
    return baseline - value


def quality_score(summary, baseline_summary):
    score = 0.0
    for key, weight in (
        ("mean_delta", 1.0),
        ("hard_bottom25_delta", 0.55),
        ("easy_top25_delta", 0.20),
        ("p5_delta", 0.06),
        ("bright_low_gradient_delta", 0.10),
        ("sky_bright_proxy_delta", 0.10),
    ):
        value = summary.get(key)
        if value is not None:
            score += weight * value
    positive_ratio = summary.get("positive_ratio")
    if positive_ratio is not None:
        score += 0.015 * (positive_ratio - 0.5)
    mean_ssim = summary.get("mean_ssim_delta")
    if mean_ssim is not None:
        score += 2.0 * mean_ssim
    strong_improve = count_delta(summary, baseline_summary, "strong_ref_regressions")
    worst_improve = count_delta(summary, baseline_summary, "worst_0p20_count")
    if strong_improve is not None:
        score += 0.00020 * strong_improve
    if worst_improve is not None:
        score += 0.00012 * worst_improve
    return score


def module_candidates(rows, args):
    all_summary = summarize_rows([row for row in rows if row["variant"] == "all_adapters"])
    grouped = {}
    for row in rows:
        grouped.setdefault(row["variant"], []).append(row)

    candidates = []
    for variant, variant_rows in grouped.items():
        summary = summarize_rows(variant_rows)
        active_adapters = variant_rows[0]["active_adapters"]
        checks = {
            "has_best_and_final": set(summary["checkpoints"]) == {"best", "final"},
            "common_count_ok": summary["common_count"] >= args.min_common_count,
            "mean_delta_ok": (summary["mean_delta"] or -999) >= args.min_module_mean_delta,
            "hard_delta_ok": (summary["hard_bottom25_delta"] or -999) >= args.min_module_hard_delta,
            "easy_delta_ok": (summary["easy_top25_delta"] or -999) >= args.min_module_easy_delta,
            "ssim_delta_ok": (summary["mean_ssim_delta"] or -999) >= args.min_ssim_delta,
        }
        strong_improve = count_delta(summary, all_summary, "strong_ref_regressions")
        worst_improve = count_delta(summary, all_summary, "worst_0p20_count")
        candidate = {
            "variant": variant,
            "active_adapters": active_adapters,
            "summary": summary,
            "strong_ref_regression_improvement_vs_all": strong_improve,
            "worst_0p20_improvement_vs_all": worst_improve,
            "score": quality_score(summary, all_summary),
            "checks": checks,
            "eligible": all(checks.values()),
        }
        candidates.append(candidate)
    return sorted(candidates, key=lambda item: item["score"], reverse=True), all_summary


def scale_candidates(rows, args):
    scale1_summary = summarize_rows([row for row in rows if as_float(row, "scale_multiplier") == 1.0])
    grouped = {}
    for row in rows:
        grouped.setdefault(as_float(row, "scale_multiplier"), []).append(row)

    candidates = []
    for scale, scale_rows in grouped.items():
        summary = summarize_rows(scale_rows)
        checks = {
            "has_best_and_final": set(summary["checkpoints"]) == {"best", "final"},
            "common_count_ok": summary["common_count"] >= args.min_common_count,
            "scale_positive": (scale or 0.0) > 0.0,
            "mean_delta_ok": (summary["mean_delta"] or -999) >= args.min_scale_mean_delta,
            "hard_delta_ok": (summary["hard_bottom25_delta"] or -999) >= args.min_scale_hard_delta,
            "easy_delta_ok": (summary["easy_top25_delta"] or -999) >= args.min_scale_easy_delta,
            "ssim_delta_ok": (summary["mean_ssim_delta"] or -999) >= args.min_ssim_delta,
        }
        strong_improve = count_delta(summary, scale1_summary, "strong_ref_regressions")
        worst_improve = count_delta(summary, scale1_summary, "worst_0p20_count")
        score = quality_score(summary, scale1_summary)
        if scale is not None:
            score -= 0.003 * scale
        candidate = {
            "variant": scale_rows[0]["variant"],
            "scale_multiplier": scale,
            "summary": summary,
            "strong_ref_regression_improvement_vs_scale1": strong_improve,
            "worst_0p20_improvement_vs_scale1": worst_improve,
            "score": score,
            "checks": checks,
            "eligible": all(checks.values()),
        }
        candidates.append(candidate)
    return sorted(candidates, key=lambda item: item["score"], reverse=True), scale1_summary


def choose(candidates):
    for candidate in candidates:
        if candidate["eligible"]:
            return candidate
    return candidates[0] if candidates else None


def md_value(value, precision=6):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{precision}f}"
    return str(value)


def write_markdown(path, decision):
    module = decision["selected_module"]
    scale = decision["selected_scale"]
    lines = [
        "# DPGA-v1.1 Tail-Control Training Decision",
        "",
        f"Generated: {decision['generated_at']}",
        f"Launch allowed: `{str(decision['launch_allowed']).lower()}`",
        "",
        "## Selected Config",
        "",
        f"- Active adapters: `{decision['selected_config'].get('dpga_active_adapters', '')}`",
        f"- Scale multiplier: `{decision['selected_config'].get('dpga_scale_multiplier', '')}`",
        f"- Diagnostic-only source: `{str(decision['diagnostic_only']).lower()}`",
        "",
        "## Module Selection",
        "",
        "| variant | active | score | mean | hard | easy | ssim | strong improvement | worst improvement | eligible |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for candidate in decision["module_candidates"][:8]:
        summary = candidate["summary"]
        lines.append(
            "| {variant} | `{active}` | {score} | {mean} | {hard} | {easy} | {ssim} | {strong} | {worst} | `{eligible}` |".format(
                variant=candidate["variant"],
                active=candidate["active_adapters"],
                score=md_value(candidate["score"]),
                mean=md_value(summary.get("mean_delta")),
                hard=md_value(summary.get("hard_bottom25_delta")),
                easy=md_value(summary.get("easy_top25_delta")),
                ssim=md_value(summary.get("mean_ssim_delta"), 8),
                strong=md_value(candidate.get("strong_ref_regression_improvement_vs_all"), 3),
                worst=md_value(candidate.get("worst_0p20_improvement_vs_all"), 3),
                eligible=str(candidate["eligible"]).lower(),
            )
        )
    lines.extend(
        [
            "",
            "## Scale Selection",
            "",
            "| scale | score | mean | hard | easy | ssim | strong improvement | worst improvement | eligible |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for candidate in decision["scale_candidates"]:
        summary = candidate["summary"]
        lines.append(
            "| {scale} | {score} | {mean} | {hard} | {easy} | {ssim} | {strong} | {worst} | `{eligible}` |".format(
                scale=md_value(candidate["scale_multiplier"], 2),
                score=md_value(candidate["score"]),
                mean=md_value(summary.get("mean_delta")),
                hard=md_value(summary.get("hard_bottom25_delta")),
                easy=md_value(summary.get("easy_top25_delta")),
                ssim=md_value(summary.get("mean_ssim_delta"), 8),
                strong=md_value(candidate.get("strong_ref_regression_improvement_vs_scale1"), 3),
                worst=md_value(candidate.get("worst_0p20_improvement_vs_scale1"), 3),
                eligible=str(candidate["eligible"]).lower(),
            )
        )
    lines.extend(["", "## Blockers", ""])
    if decision["launch_blockers"]:
        lines.extend(f"- {blocker}" for blocker in decision["launch_blockers"])
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Recommended Training Args",
            "",
            "```json",
            json.dumps(decision["training_args"], indent=2),
            "```",
            "",
        ]
    )
    if module and scale:
        lines.append(
            "Decision note: diagnostics choose the v1.1 starting configuration only; "
            "the next checkpoint must be selected on `val_inner`, not Haze4K test."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostics_dir", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    parser.add_argument("--model_name_prefix", default="ConvIR-Haze4K-DPGA-v1.1-tail-control")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--min_common_count", type=int, default=1000)
    parser.add_argument("--min_module_mean_delta", type=float, default=0.005)
    parser.add_argument("--min_module_hard_delta", type=float, default=0.0)
    parser.add_argument("--min_module_easy_delta", type=float, default=-0.02)
    parser.add_argument("--min_scale_mean_delta", type=float, default=0.005)
    parser.add_argument("--min_scale_hard_delta", type=float, default=-0.005)
    parser.add_argument("--min_scale_easy_delta", type=float, default=-0.02)
    parser.add_argument("--min_ssim_delta", type=float, default=-0.0005)
    args = parser.parse_args()

    diagnostics_dir = Path(args.diagnostics_dir)
    module_path = diagnostics_dir / MODULE_CSV
    scale_path = diagnostics_dir / SCALE_CSV
    launch_blockers = []
    if not module_path.is_file():
        launch_blockers.append(f"missing {module_path}")
    if not scale_path.is_file():
        launch_blockers.append(f"missing {scale_path}")

    module_rows = read_rows(module_path) if module_path.is_file() else []
    scale_rows = read_rows(scale_path) if scale_path.is_file() else []
    module_rank, module_all = module_candidates(module_rows, args) if module_rows else ([], {})
    scale_rank, scale1 = scale_candidates(scale_rows, args) if scale_rows else ([], {})
    selected_module = choose(module_rank)
    selected_scale = choose(scale_rank)

    if selected_module is None:
        launch_blockers.append("no module diagnostic candidate was available")
    elif not selected_module["eligible"]:
        launch_blockers.append(f"top module candidate {selected_module['variant']} failed eligibility checks")
    if selected_scale is None:
        launch_blockers.append("no scale diagnostic candidate was available")
    elif not selected_scale["eligible"]:
        launch_blockers.append(f"top scale candidate {selected_scale['variant']} failed eligibility checks")

    launch_allowed = not launch_blockers
    active_adapters = selected_module["active_adapters"] if selected_module else ""
    scale_multiplier = selected_scale["scale_multiplier"] if selected_scale else None
    safe_active = active_adapters.replace(",", "-") if active_adapters else "unknown"
    safe_scale = str(scale_multiplier).replace(".", "p") if scale_multiplier is not None else "unknown"
    model_name = f"{args.model_name_prefix}-{safe_active}-scale{safe_scale}-seed{args.seed}-20260604"
    training_args = {
        "model_name": model_name,
        "dpga_active_adapters": active_adapters,
        "dpga_scale_multiplier": scale_multiplier,
        "dpga_adapter_residual_scale": 0.1,
        "dpga_tc_rec_loss": "charbonnier",
        "dpga_tc_fft_lambda": 0.05,
        "dpga_tc_anchor_lambda": 0.08,
        "dpga_tc_chroma_lambda": 0.03,
        "dpga_tc_delta_lambda": 0.0002,
        "dpga_tc_delta_tv_lambda": 0.00005,
        "dpga_tc_anchor_error_threshold": 0.035,
        "learning_rate": 0.0003,
        "weight_decay": 0.0001,
        "stop_epoch": 20,
        "seed": args.seed,
    }
    decision = {
        "generated_at": dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds"),
        "diagnostic_only": True,
        "launch_allowed": launch_allowed,
        "launch_blockers": launch_blockers,
        "selected_config": {
            "dpga_active_adapters": active_adapters,
            "dpga_scale_multiplier": scale_multiplier,
            "dpga_adapter_residual_scale": 0.1,
        },
        "training_args": training_args,
        "selected_module": selected_module,
        "selected_scale": selected_scale,
        "module_baseline_all": module_all,
        "scale_baseline_scale1": scale1,
        "module_candidates": module_rank,
        "scale_candidates": scale_rank,
        "source_files": {
            "module_ablation": str(module_path),
            "scale_sweep": str(scale_path),
        },
    }

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    write_markdown(output_md, decision)
    print(json.dumps({
        "launch_allowed": launch_allowed,
        "launch_blockers": launch_blockers,
        "model_name": model_name,
        "active_adapters": active_adapters,
        "scale_multiplier": scale_multiplier,
        "decision_json": str(output_json),
        "decision_md": str(output_md),
    }, indent=2))


if __name__ == "__main__":
    main()
