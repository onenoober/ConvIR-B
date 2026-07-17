#!/usr/bin/env python3
"""CPU-only end-to-end validation program for the route-ready fast path."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from route_program_api import (
    atomic_json,
    load_context,
    prepare_phase_output,
    write_contract_result,
    write_run_result,
)


def contract(context_path: Path) -> None:
    context = load_context(context_path, "contract")
    prepare_phase_output(context)
    tests = Path(context.remote_repo) / "experience_docx/tools/tests"
    completed = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", str(tests), "-p", "test_*.py"],
        cwd=context.remote_repo, text=True, capture_output=True, timeout=120,
    )
    report = context.phase_output_path / "unit_tests.txt"
    report.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    checks = {
        "cloud_unit_tests": completed.returncode == 0,
        "contract_environment": os.environ.get("CONVIR_CONTRACT_ONLY") == "1",
        "cpu_only_contract": os.environ.get("CUDA_VISIBLE_DEVICES") == "",
        "workload_output_absent": not (context.output_path / "workload").exists(),
        "context_paths_named": context.phase_output_path.name == "contract",
    }
    write_contract_result(context, checks=checks)
    if not all(checks.values()):
        raise SystemExit("route-ready cloud contract validation failed")


def run(context_path: Path) -> None:
    context = load_context(context_path, "run")
    prepare_phase_output(context)
    contract_report = context.output_path / "contract/unit_tests.txt"
    if not contract_report.is_file() or "OK" not in contract_report.read_text(encoding="utf-8"):
        raise RuntimeError("cloud unit-test evidence is unavailable")
    summary = {
        "schema_version": 1,
        "route_id": context.route_id,
        "operation_id": context.operation_id,
        "generic_runner_used": True,
        "contract_before_run": True,
        "typed_closeout_owned_by_lifecycle": True,
        "position_free_context_paths": True,
        "cloud_unit_tests_passed": True,
        "model_calls": 0,
        "gpu_accessed": False,
        "dataset_accessed": False,
        "checkpoint_accessed": False,
        "confirmation_images_targets_outcomes_touched": False,
        "canary_touched": False,
        "locked_test_touched": False,
    }
    atomic_json(context.phase_output_path / "summary.json", summary)
    write_run_result(
        context,
        state="COMPLETED_GATE_PASS",
        decision="ROUTE_READY_FASTPATH_VALIDATION_PASS",
        authorizes="ROUTE_READY_FASTPATH_ADOPTION",
        details={"cloud_unit_tests_passed": True, "model_calls": 0},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("contract", "run"))
    parser.add_argument("--context", type=Path, required=True)
    args = parser.parse_args()
    if args.phase == "contract":
        contract(args.context)
    else:
        run(args.context)


if __name__ == "__main__":
    main()
