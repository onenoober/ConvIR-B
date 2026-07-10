#!/usr/bin/env python3
"""Static source preflight for CHD-RM v3b.

This script reads source files only. It does not import the model, allocate
CUDA tensors, run inference, evaluate metrics, or train.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROUTE_ID = "haze4k_v5_chd_rm_v3b_rarm_preflight_design_20260710"
DECISION_LABEL = "V3B_RARM_PREFLIGHT_BLOCKED_GATE_PIPELINE_ABSENT_NO_RARM_TRAINING"


def find_line(lines: list[str], needle: str) -> int | None:
    for index, line in enumerate(lines, start=1):
        if needle in line:
            return index
    return None


def read_lines(repo_root: Path, relative_path: str) -> list[str]:
    return (repo_root / relative_path).read_text(encoding="utf-8").splitlines()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()

    conv_lines = read_lines(repo_root, "Dehazing/ITS/models/ConvIR.py")
    train_lines = read_lines(repo_root, "Dehazing/ITS/train.py")
    valid_lines = read_lines(repo_root, "Dehazing/ITS/valid.py")
    eval_lines = read_lines(repo_root, "Dehazing/ITS/eval.py")

    audit = {
        "route_id": ROUTE_ID,
        "decision_label": DECISION_LABEL,
        "scope": "static_source_preflight_only",
        "runtime_execution": "none",
        "local_model_forward": "none",
        "source_evidence": {
            "d7c_noop_forward_requires_gate": {
                "path": "Dehazing/ITS/models/ConvIR.py",
                "line": find_line(
                    conv_lines,
                    "raise ValueError('d7c_gate is required for d7c_noop FAM mode')",
                ),
                "pass": "raise ValueError('d7c_gate is required for d7c_noop FAM mode')"
                in "\n".join(conv_lines),
            },
            "model_forward_accepts_gate": {
                "path": "Dehazing/ITS/models/ConvIR.py",
                "line": find_line(conv_lines, "def forward(self, x, d7c_gate=None):"),
                "pass": "def forward(self, x, d7c_gate=None):" in "\n".join(conv_lines),
            },
            "train_forward_call_has_no_gate": {
                "path": "Dehazing/ITS/train.py",
                "line": find_line(train_lines, "pred_img = model(input_img)"),
                "pass": "pred_img = model(input_img)" in "\n".join(train_lines)
                and "d7c_gate" not in "\n".join(train_lines),
            },
            "valid_forward_call_has_no_gate": {
                "path": "Dehazing/ITS/valid.py",
                "line": find_line(valid_lines, "pred = model(input_img)[2]"),
                "pass": "pred = model(input_img)[2]" in "\n".join(valid_lines)
                and "d7c_gate" not in "\n".join(valid_lines),
            },
            "eval_forward_call_has_no_gate": {
                "path": "Dehazing/ITS/eval.py",
                "line": find_line(eval_lines, "pred = model(input_img)[2]"),
                "pass": "pred = model(input_img)[2]" in "\n".join(eval_lines)
                and "d7c_gate" not in "\n".join(eval_lines),
            },
            "modulation_stats_call_has_no_gate": {
                "path": "Dehazing/ITS/train.py",
                "line": find_line(train_lines, "batch_stats = model.collect_modulation_stats(input_img)"),
                "pass": "batch_stats = model.collect_modulation_stats(input_img)"
                in "\n".join(train_lines),
            },
        },
        "metric_contract": {
            "question": "Can current runnable training/eval entrypoints support fam2_d7c_noop without redesign?",
            "pass_rule": "All active entrypoints must produce and pass a nontrivial d7c_gate without forbidden training/RARM scope expansion.",
            "result": "fail",
            "reason": "fam2_d7c_noop requires d7c_gate, while train/valid/eval and modulation stats calls do not pass it.",
        },
        "forbidden_flow_audit": {
            "training": "not_run",
            "RARM": "not_connected_or_trained",
            "adapter_training": "none",
            "ConvIR_B_unfreeze": "none",
            "locked_haze4k_test_usage": "none",
            "canary_expansion": "none",
        },
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        print(json.dumps(audit, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
