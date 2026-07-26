#!/usr/bin/env python3
"""Data-free entrypoint for one generic control-plane acceptance lifecycle."""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

from route_program_api import (
    atomic_json,
    load_completed_unit_ledger,
    load_context,
    output_file,
    prepare_phase_output,
    record_completed_unit,
    write_contract_progress,
    write_contract_result,
    write_gate_result,
    write_workload_progress,
)


def contract(context_path):
    context = load_context(Path(context_path), "contract")
    prepare_phase_output(context)
    write_contract_progress(
        context, completed_iterations=1, total_iterations=1, stage="metadata_contract",
    )
    engineering = {
        "mode": "metadata_only",
        "device": "cpu",
        "fixture": None,
        "production_path_exercised": False,
        "protected_data_touched": False,
        "scientific_output_created": False,
        "scientific_training_occurred": False,
    }
    write_contract_result(
        context,
        checks={
            "no_external_assets": True,
            "metadata_only_mode": True,
            "protected_permissions_disabled": True,
        },
        engineering=engineering,
    )


def run(context_path):
    context = load_context(Path(context_path), "run")
    prepare_phase_output(context)
    if load_completed_unit_ledger(context):
        raise RuntimeError("fresh acceptance workload unexpectedly has completed units")
    summary = {
        "schema_version": 1,
        "route_id": context.route_id,
        "operation_id": context.operation_id,
        "run_id": context.run_id,
        "acceptance_kind": "generic_control_plane_no_data",
        "external_assets_read": 0,
        "dataset_access": 0,
        "checkpoint_access": 0,
        "model_calls": 0,
        "protected_data_touched": False,
    }
    atomic_json(output_file(context, "acceptance_summary.json"), summary)
    record_completed_unit(
        context,
        unit_id="lifecycle_control_record",
        input_sha256=hashlib.sha256(b"general-science-control-plane-e2e-v1").hexdigest(),
        output_relpath="acceptance_summary.json",
    )
    write_workload_progress(
        context, completed_units=1, stage="no_data_lifecycle_verified",
    )
    time.sleep(45)
    write_gate_result(
        context,
        gate_outcomes={
            "integrity": "pass",
            "lifecycle_safety": "safe",
        },
        details={
            "acceptance_kind": "generic_control_plane_no_data",
            "scientific_authorization": "NONE",
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("contract", "run"))
    parser.add_argument("--context", required=True)
    args = parser.parse_args()
    if args.phase == "contract":
        contract(args.context)
    else:
        run(args.context)


if __name__ == "__main__":
    main()
