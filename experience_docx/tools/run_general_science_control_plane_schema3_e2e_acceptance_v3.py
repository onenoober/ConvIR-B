#!/usr/bin/env python3
"""Data-free entrypoint for the schema-3 control-plane acceptance lifecycle."""

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
    write_review_facts,
    write_workload_progress,
)


SUMMARY_FILENAME = "acceptance_summary.json"
PUBLISHED_SUMMARY_FILENAME = (
    "general_science_control_plane_schema3_e2e_acceptance_v3_summary.json"
)
REVIEW_FACTS_FILENAME = (
    "general_science_control_plane_schema3_e2e_acceptance_v3_review_facts.json"
)


def contract(context_path):
    context = load_context(Path(context_path), "contract")
    prepare_phase_output(context)
    write_contract_progress(
        context,
        completed_iterations=1,
        total_iterations=1,
        stage="metadata_contract",
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
        "acceptance_kind": "schema3_generic_control_plane_no_data",
        "completed_control_units": 1,
        "required_control_units": 1,
        "lifecycle_completion_ratio": 1.0,
        "required_completion_ratio": 1.0,
        "external_assets_read": 0,
        "dataset_access": 0,
        "checkpoint_access": 0,
        "model_calls": 0,
        "protected_data_access_count": 0,
        "protected_data_access_limit": 0,
        "protected_data_touched": False,
        "gate_outcomes": {
            "integrity": "pass",
            "lifecycle_safety": "safe",
        },
    }
    summary_path = output_file(context, SUMMARY_FILENAME)
    atomic_json(summary_path, summary)
    summary_sha256 = hashlib.sha256(summary_path.read_bytes()).hexdigest()

    write_review_facts(
        context,
        relpath=REVIEW_FACTS_FILENAME,
        facts=[
            {
                "fact_id": "lifecycle_completion_ratio",
                "claim_id": "schema3_lifecycle_completed",
                "metric": "control plane lifecycle completion ratio",
                "unit": "control_record",
                "population": "one frozen no-data control record",
                "grouping": "control_record",
                "point": 1.0,
                "ci_lower": None,
                "ci_upper": None,
                "confidence_level": None,
                "threshold": 1.0,
                "threshold_operator": "==",
                "gate_outcome": "pass",
                "source_filename": PUBLISHED_SUMMARY_FILENAME,
                "source_sha256": summary_sha256,
                "json_pointers": {
                    "point": "/lifecycle_completion_ratio",
                    "ci_lower": None,
                    "ci_upper": None,
                    "confidence_level": None,
                    "threshold": "/required_completion_ratio",
                    "gate_outcome": "/gate_outcomes/integrity",
                },
            },
            {
                "fact_id": "protected_data_access_count",
                "claim_id": "protected_data_boundary_preserved",
                "metric": "protected data access count",
                "unit": "accesses",
                "population": "one frozen no-data control record",
                "grouping": "control_record",
                "point": 0,
                "ci_lower": None,
                "ci_upper": None,
                "confidence_level": None,
                "threshold": 0,
                "threshold_operator": "==",
                "gate_outcome": "safe",
                "source_filename": PUBLISHED_SUMMARY_FILENAME,
                "source_sha256": summary_sha256,
                "json_pointers": {
                    "point": "/protected_data_access_count",
                    "ci_lower": None,
                    "ci_upper": None,
                    "confidence_level": None,
                    "threshold": "/protected_data_access_limit",
                    "gate_outcome": "/gate_outcomes/lifecycle_safety",
                },
            },
        ],
    )
    record_completed_unit(
        context,
        unit_id="schema3_lifecycle_control_record",
        input_sha256=hashlib.sha256(
            b"general-science-control-plane-schema3-e2e-v3"
        ).hexdigest(),
        output_relpath=SUMMARY_FILENAME,
    )
    write_workload_progress(
        context,
        completed_units=1,
        stage="schema3_no_data_lifecycle_verified",
    )
    time.sleep(45)
    write_gate_result(
        context,
        gate_outcomes={
            "integrity": "pass",
            "lifecycle_safety": "safe",
        },
        details={
            "acceptance_kind": "schema3_generic_control_plane_no_data",
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
