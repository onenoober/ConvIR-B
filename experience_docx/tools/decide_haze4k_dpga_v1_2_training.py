#!/usr/bin/env python3
import argparse
import datetime as dt
import json
from pathlib import Path


def check_value(observed, required, passed):
    return {
        "observed": observed,
        "required": required,
        "pass": bool(passed),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--failure_analysis_json", required=True)
    parser.add_argument("--previous_decision_json", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    parser.add_argument("--date_tag", default=dt.datetime.now().strftime("%Y%m%d"))
    args = parser.parse_args()

    failure = json.loads(Path(args.failure_analysis_json).read_text(encoding="utf-8"))
    previous = json.loads(Path(args.previous_decision_json).read_text(encoding="utf-8"))
    gate = failure["gate"]
    best_groups = failure["best"]["summary"]["groups"]
    all_best = best_groups["all"]
    hard_best = best_groups["hard_bottom25"]
    strong_best = best_groups["strong_reference_top25"]

    checks = {
        "previous_gate_failed": check_value(gate.get("pass"), "false", gate.get("pass") is False),
        "locked_test_blocked": check_value(
            gate.get("locked_test_allowed"),
            "false",
            gate.get("locked_test_allowed") is False,
        ),
        "mean_gain_ok": check_value(
            all_best["delta_psnr_mean"],
            ">= +0.030 dB",
            all_best["delta_psnr_mean"] >= 0.030,
        ),
        "hard_gain_shortfall": check_value(
            hard_best["delta_psnr_mean"],
            "< +0.030 dB",
            hard_best["delta_psnr_mean"] < 0.030,
        ),
        "tail_regressions_safe": check_value(
            all_best["worst_regression_count_delta_le_-0.20"],
            "<= 12 / 300",
            all_best["worst_regression_count_delta_le_-0.20"] <= 12,
        ),
        "strong_regressions_safe": check_value(
            strong_best["strong_regression_count_delta_le_-0.05"],
            "<= 15 / 75",
            strong_best["strong_regression_count_delta_le_-0.05"] <= 15,
        ),
    }
    launch_allowed = all(item["pass"] for item in checks.values())
    model_name = (
        "ConvIR-Haze4K-DPGA-v1.2-hard-gain-"
        f"shallow-scale0p5-anchor0p04-seed3407-{args.date_tag}"
    )
    training_args = {
        "model_name": model_name,
        "dpga_active_adapters": "shallow",
        "dpga_scale_multiplier": 0.5,
        "dpga_adapter_residual_scale": previous["training_args"]["dpga_adapter_residual_scale"],
        "dpga_tc_rec_loss": "charbonnier",
        "dpga_tc_fft_lambda": 0.05,
        "dpga_tc_anchor_lambda": 0.04,
        "dpga_tc_chroma_lambda": 0.03,
        "dpga_tc_delta_lambda": 0.00025,
        "dpga_tc_delta_tv_lambda": 5e-05,
        "dpga_tc_anchor_error_threshold": previous["training_args"]["dpga_tc_anchor_error_threshold"],
        "learning_rate": previous["training_args"]["learning_rate"],
        "weight_decay": previous["training_args"]["weight_decay"],
        "stop_epoch": previous["training_args"]["stop_epoch"],
        "seed": previous["training_args"]["seed"],
    }
    decision = {
        "generated_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "diagnostic_only": True,
        "source_failure_analysis_json": args.failure_analysis_json,
        "previous_decision_json": args.previous_decision_json,
        "launch_allowed": launch_allowed,
        "launch_blockers": [name for name, item in checks.items() if not item["pass"]],
        "checks": checks,
        "selected_config": {
            "dpga_active_adapters": training_args["dpga_active_adapters"],
            "dpga_scale_multiplier": training_args["dpga_scale_multiplier"],
            "dpga_adapter_residual_scale": training_args["dpga_adapter_residual_scale"],
            "dpga_tc_anchor_lambda": training_args["dpga_tc_anchor_lambda"],
        },
        "training_args": training_args,
        "rationale": (
            "v1.1 passed mean/positive/tail safety on val_inner but missed hard-bottom "
            "gain. v1.2 keeps the same adapter family, raises scale from 0.25 to 0.5, "
            "and lowers anchor pressure from 0.08 to 0.04 to test whether hard gain "
            "can cross the gate without reviving tail regressions."
        ),
        "locked_test_allowed": False,
        "next_gate": "Run Best/Final against A0 on val_inner only; locked test remains blocked unless that gate passes.",
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# DPGA-v1.2 Training Decision",
        "",
        f"Generated: {decision['generated_at']}",
        f"Launch allowed: `{str(launch_allowed).lower()}`",
        "",
        "## Checks",
        "",
        "| check | observed | required | pass |",
        "| --- | ---: | --- | --- |",
    ]
    for name, item in checks.items():
        observed = item["observed"]
        if isinstance(observed, float):
            observed = f"{observed:.6f}"
        lines.append(f"| {name} | {observed} | {item['required']} | `{str(item['pass']).lower()}` |")
    lines.extend(
        [
            "",
            "## Selected Config",
            "",
            "```json",
            json.dumps(training_args, indent=2),
            "```",
            "",
            "## Rationale",
            "",
            decision["rationale"],
            "",
            "Locked test remains blocked until this config passes the same `val_inner` gate.",
            "",
        ]
    )
    Path(args.output_md).write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(decision, indent=2))
    if not launch_allowed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
