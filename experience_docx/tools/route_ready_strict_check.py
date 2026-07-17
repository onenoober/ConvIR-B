#!/usr/bin/env python3
"""Minimal route program used only to exercise strict staged validation."""

import argparse
from pathlib import Path

from route_program_api import (
    atomic_json,
    load_context,
    prepare_phase_output,
    write_contract_result,
    write_run_result,
)


def contract(context_path):
    context = load_context(context_path, "contract")
    prepare_phase_output(context)
    write_contract_result(context, checks={"context_contract": True})


def run(context_path):
    context = load_context(context_path, "run")
    prepare_phase_output(context)
    atomic_json(context.phase_output_path / "summary.json", {"strict_check": True})
    write_run_result(
        context,
        state="COMPLETED_GATE_PASS",
        decision="ROUTE_READY_STRICT_CHECK_PASS",
        authorizes="NONE",
    )


def main():
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
