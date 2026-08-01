#!/usr/bin/env python3
"""Read-only S1 measurement-validity and precision evidence review."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from route_program_api import (
    asset_path,
    atomic_json,
    build_gate_review_fact,
    load_context,
    output_file,
    write_contract_result,
    write_gate_result,
    write_review_facts,
)


ROUTE_ID = "daytime-dehazing-s1-validity-precision-review-v1"
OPERATION_ID = "DAYTIME_DEHAZING_S1_VALIDITY_PRECISION_REVIEW"
SUMMARY_FILE = "daytime_dehazing_s1_validity_precision_review_v1_summary.json"
GATE_SUMMARY_FILE = (
    "daytime_dehazing_s1_validity_precision_review_v1_gate_summary.json"
)
REVIEW_FACTS_FILE = (
    "daytime_dehazing_s1_validity_precision_review_v1_review_facts.json"
)
INPUT_ASSETS = (
    "s0_closeout",
    "s0_role_summary",
    "s1_v1_closeout",
    "s1_v1_summary",
    "pairing_v2_closeout",
    "pairing_v2_summary",
    "pairing_v2_identity_summary",
    "s1_v2_closeout",
    "s1_v2_summary",
    "s1_v2_gate_summary",
)
PAIRING_LEDGER_SHA256 = (
    "d3e889b3cce92c0a11f25a6a49db0e3a3854b7eb427721b2410871ad65d39c25"
)
S1_V2_SUMMARY_SHA256 = (
    "fdb15c9da34b6e9bba815927578898957b1980f87d7bb3295dfb1b31fb3fa068"
)
S1_V2_GATE_SUMMARY_SHA256 = (
    "a498559a645d547472185d889af9c19ce589f56ed91564cb3931deda4e338a69"
)
PAIRING_V2_SUMMARY_SHA256 = (
    "d7e4900fab8701e7ae764775cd527510795cf0c357964a7e7cf0f1b2254c3699"
)
PAIRING_V2_IDENTITY_SHA256 = (
    "123acad0dcf73b958d2d02845c8c51f0eac89023a469de551102ca40a714794a"
)
S0_ROLE_SUMMARY_SHA256 = (
    "d2262c8ba28c56a21b992c8f2c445d92099d7c9861f4263171c522c2efd8e7b1"
)
S1_V1_SUMMARY_SHA256 = (
    "5e3b678c8968a0c243f6bf79fe40b8ca80e860a376e475b61f90599eded0df1f"
)


def _nested(value: Any, *keys: str) -> Any:
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _load_inputs(context: Any) -> tuple[dict[str, dict[str, Any]], list[str]]:
    documents: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for identifier in INPUT_ASSETS:
        try:
            value = json.loads(
                asset_path(context, identifier, kind="file").read_text(
                    encoding="utf-8"
                )
            )
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{identifier}:{type(exc).__name__}")
            continue
        if not isinstance(value, dict):
            errors.append(f"{identifier}:not_object")
            continue
        documents[identifier] = value
    return documents, errors


def _identity_checks(documents: dict[str, dict[str, Any]]) -> dict[str, bool]:
    s0_closeout = documents.get("s0_closeout", {})
    s1_v1_closeout = documents.get("s1_v1_closeout", {})
    pairing_closeout = documents.get("pairing_v2_closeout", {})
    s1_v2_closeout = documents.get("s1_v2_closeout", {})
    return {
        "all_compact_inputs_loaded": set(documents) == set(INPUT_ASSETS),
        "s0_terminal_identity": (
            s0_closeout.get("route_id") == "daytime-dehazing-haze4k-identity-v1"
            and s0_closeout.get("state") == "COMPLETED_GATE_PASS"
            and s0_closeout.get("decision")
            == "DAYTIME_DEHAZING_PROGRAM_FOUNDATION_PASS"
        ),
        "s0_role_summary_bound": _nested(
            s0_closeout,
            "evidence_sha256",
            "daytime_dehazing_haze4k_identity_v1_role_summary.json",
        ) == S0_ROLE_SUMMARY_SHA256,
        "s1_v1_terminal_identity": (
            s1_v1_closeout.get("route_id")
            == "daytime-dehazing-local-restoration-need-qualification-v1"
            and s1_v1_closeout.get("state") == "COMPLETED_INCONCLUSIVE"
            and s1_v1_closeout.get("authorizes")
            == "S1_MEASUREMENT_VALIDITY_OR_PRECISION_REVIEW_ONLY"
        ),
        "s1_v1_summary_bound": _nested(
            s1_v1_closeout,
            "evidence_sha256",
            "daytime_dehazing_local_restoration_need_v1_summary.json",
        ) == S1_V1_SUMMARY_SHA256,
        "pairing_v2_terminal_identity": (
            pairing_closeout.get("route_id")
            == "daytime-dehazing-reside-pairing-contract-v2"
            and pairing_closeout.get("state") == "COMPLETED_GATE_PASS"
            and pairing_closeout.get("authorizes")
            == "AUTHOR_REVISED_S1_UTILITY_QUALIFICATION_CONTRACT_ONLY"
        ),
        "pairing_v2_summary_bound": _nested(
            pairing_closeout,
            "evidence_sha256",
            "daytime_dehazing_reside_pairing_contract_v2_summary.json",
        ) == PAIRING_V2_SUMMARY_SHA256,
        "pairing_v2_identity_bound": _nested(
            pairing_closeout,
            "evidence_sha256",
            "daytime_dehazing_reside_pairing_contract_v2_identity_summary.json",
        ) == PAIRING_V2_IDENTITY_SHA256,
        "s1_v2_terminal_identity": (
            s1_v2_closeout.get("route_id")
            == "daytime-dehazing-local-restoration-need-qualification-v2"
            and s1_v2_closeout.get("state") == "COMPLETED_INCONCLUSIVE"
            and s1_v2_closeout.get("decision")
            == "DAYTIME_DEHAZING_LOCAL_RESTORATION_NEED_INCONCLUSIVE"
            and s1_v2_closeout.get("authorizes")
            == "S1_MEASUREMENT_VALIDITY_OR_PRECISION_REVIEW_ONLY"
        ),
        "s1_v2_summary_bound": _nested(
            s1_v2_closeout,
            "evidence_sha256",
            "daytime_dehazing_local_restoration_need_v2_summary.json",
        ) == S1_V2_SUMMARY_SHA256,
        "s1_v2_gate_summary_bound": _nested(
            s1_v2_closeout,
            "evidence_sha256",
            "daytime_dehazing_local_restoration_need_v2_gate_summary.json",
        ) == S1_V2_GATE_SUMMARY_SHA256,
    }


def _pairing_checks(documents: dict[str, dict[str, Any]]) -> dict[str, bool]:
    roles = documents.get("s0_role_summary", {})
    pairing = documents.get("pairing_v2_summary", {})
    identity = documents.get("pairing_v2_identity_summary", {})
    parent = documents.get("s1_v2_summary", {})
    checks = {
        "s0_roles_disjoint": _nested(roles, "roles", "all_roles_disjoint") is True,
        "haze4k_capacity_150": _nested(
            roles, "roles", "datasets", "HAZE4K_TRAIN", "role_counts",
            "development_screening",
        ) == 150,
        "its_capacity_at_least_150": (
            _nested(
                roles, "roles", "datasets", "ITS", "role_counts",
                "development_screening",
            ) or 0
        ) >= 150,
        "ots_capacity_at_least_150": (
            _nested(
                roles, "roles", "datasets", "OTS", "role_counts",
                "development_screening",
            ) or 0
        ) >= 150,
        "pairing_valid_scene_count_300": pairing.get("valid_scene_count") == 300,
        "pairing_ledger_rows_300": _nested(pairing, "mapping_ledger", "rows") == 300,
        "pairing_ledger_identity": (
            _nested(pairing, "mapping_ledger", "sha256") == PAIRING_LEDGER_SHA256
            and identity.get("mapping_ledger_sha256") == PAIRING_LEDGER_SHA256
            and _nested(
                parent, "ledger_summary", "reside_pairing", "mapping_ledger_sha256"
            ) == PAIRING_LEDGER_SHA256
        ),
        "no_pairing_backfill": _nested(
            pairing, "forbidden_activity_receipt", "planned_scene_backfill_occurred"
        ) is False,
        "parent_coverage_passed": _nested(
            parent, "gate_outcomes", "development_scene_coverage"
        ) == "pass",
    }
    for dataset in ("ITS", "OTS"):
        prefix = dataset.lower()
        summary = _nested(pairing, "datasets", dataset) or {}
        planned = _nested(identity, "planned_rosters", dataset) or {}
        checks[f"{prefix}_planned_selected_150"] = (
            summary.get("planned_scene_count") == 150
            and summary.get("selected_scene_count") == 150
            and planned.get("planned_scene_count") == 150
            and planned.get("selected_scene_count") == 150
        )
        checks[f"{prefix}_nested_not_independent"] = (
            _nested(summary, "independence_contract", "haze_observations_are_nested")
            is True
            and _nested(
                summary,
                "independence_contract",
                "haze_observations_are_independent_replicates",
            )
            is False
        )
        checks[f"{prefix}_all_identity_geometry_checks_150"] = all(
            value == 150 for value in (summary.get("check_counts") or {}).values()
        ) and len(summary.get("check_counts") or {}) == 9
    return checks


def _geometry_checks(documents: dict[str, dict[str, Any]]) -> dict[str, bool]:
    pairing = documents.get("pairing_v2_summary", {})
    identity = documents.get("pairing_v2_identity_summary", {})
    parent = documents.get("s1_v2_summary", {})
    constraints = pairing.get("future_s1_constraints") or {}
    checks = {
        "whole_image_inference_before_crop": (
            constraints.get("whole_image_inference_before_scoring_crop") is True
            and parent.get("intervention", {}).get("inference_context")
            == "the complete image is padded to a multiple of 32 and inferred once before the frozen square scoring crop is extracted"
        ),
        "crop_minimum_256": constraints.get("scoring_crop_minimum") == 256,
        "crop_maximum_512": constraints.get("scoring_crop_maximum") == 512,
        "crop_multiple_32": constraints.get("scoring_crop_multiple") == 32,
        "edge_interior_available_as_descriptive_only": _is_number(
            _nested(parent, "diagnostics", "edge_minus_interior_local_advantage_db")
        ),
    }
    expected_rule = (
        "largest square multiple of 32 not exceeding 512 or either image dimension, "
        "with minimum 256"
    )
    for dataset in ("ITS", "OTS"):
        envelope = _nested(identity, "planned_rosters", dataset, "shape_envelope") or {}
        prefix = dataset.lower()
        checks[f"{prefix}_crop_rule_bound"] = (
            envelope.get("scoring_crop_rule") == expected_rule
        )
        checks[f"{prefix}_observed_shape_within_bound"] = (
            (envelope.get("max_observed_height") or math.inf)
            <= (envelope.get("max_allowed_height") or -math.inf)
            and (envelope.get("max_observed_width") or math.inf)
            <= (envelope.get("max_allowed_width") or -math.inf)
            and (envelope.get("max_observed_padded_pixels") or math.inf)
            <= (envelope.get("max_allowed_padded_pixels") or -math.inf)
        )
    return checks


def _review_scope_checks(documents: dict[str, dict[str, Any]]) -> dict[str, bool]:
    parent = documents.get("s1_v2_summary", {})
    gates = documents.get("s1_v2_gate_summary", {}).get("gate_outcomes") or {}
    return {
        "its_null_ucb_archived": _is_number(
            _nested(parent, "null_control", "by_dataset", "ITS", "upper")
        ),
        "haze4k_negative_tail_archived": _is_number(
            _nested(
                parent,
                "prevalence",
                "negative_tail_scene",
                "by_dataset",
                "HAZE4K_TRAIN",
                "estimate",
            )
        ),
        "cross_observation_transfer_archived": _is_number(
            _nested(parent, "cross_observation_transfer", "overall", "estimate")
        ),
        "crop_edge_interior_archived": _is_number(
            _nested(parent, "diagnostics", "edge_minus_interior_local_advantage_db")
        ),
        "prior_null_failure_preserved": gates.get("measurement_null_control") == "fail",
        "prior_utility_unfavorable_preserved": (
            gates.get("local_utility_over_global") == "unfavorable"
        ),
        "prior_transfer_unfavorable_preserved": (
            gates.get("bidirectional_repeatability") == "unfavorable"
        ),
        "prior_precision_met_preserved": gates.get("primary_precision") == "met",
    }


def _outcome(checks: dict[str, bool], *, identity_ok: bool) -> str:
    if not identity_ok:
        return "invalid"
    return "pass" if checks and all(checks.values()) else "fail"


def contract(context_path: str) -> None:
    context = load_context(Path(context_path), "contract")
    write_contract_result(
        context,
        checks={
            "route_identity": (
                context.route_id == ROUTE_ID
                and context.operation_id == OPERATION_ID
            ),
            "review_only_zero_units": context.total_units == 0,
            "metadata_only_engineering": (
                context.engineering_contract.get("mode") == "metadata_only"
            ),
            "no_protected_permission": not any(
                context.protected_data_permissions.values()
            ),
            "no_model_structure_change": True,
            "no_training_or_inference": True,
            "no_raw_image_weight_or_array_access": True,
            "typed_gate_writer_only": True,
        },
        engineering={
            "mode": "metadata_only",
            "device": context.device,
            "fixture": None,
            "production_path_exercised": False,
            "protected_data_touched": False,
            "scientific_output_created": False,
            "scientific_training_occurred": False,
        },
    )


def run(context_path: str) -> None:
    context = load_context(Path(context_path), "run")
    if context.route_id != ROUTE_ID or context.operation_id != OPERATION_ID:
        raise RuntimeError("route identity mismatch")
    if context.total_units != 0 or any(context.protected_data_permissions.values()):
        raise RuntimeError("review runtime contract mismatch")

    documents, input_errors = _load_inputs(context)
    identity_checks = _identity_checks(documents)
    identity_ok = not input_errors and all(identity_checks.values())
    pairing_checks = _pairing_checks(documents)
    geometry_checks = _geometry_checks(documents)
    review_scope_checks = _review_scope_checks(documents)

    planned_groups = {
        "HAZE4K_TRAIN_DEVELOPMENT": _nested(
            documents.get("s1_v2_summary", {}),
            "dataset_scene_counts",
            "HAZE4K_TRAIN",
        ),
        "ITS_DEVELOPMENT": _nested(
            documents.get("pairing_v2_summary", {}),
            "datasets",
            "ITS",
            "selected_scene_count",
        ),
        "OTS_DEVELOPMENT": _nested(
            documents.get("pairing_v2_summary", {}),
            "datasets",
            "OTS",
            "selected_scene_count",
        ),
    }
    planning_sd_upper_bound_db = 0.25
    critical_value = 2.58
    target_half_width_db = 0.055
    required_groups = math.ceil(
        (critical_value * planning_sd_upper_bound_db / target_half_width_db) ** 2
    )
    precision_valid = identity_ok and all(
        isinstance(value, int) and not isinstance(value, bool)
        for value in planned_groups.values()
    )
    precision_met = precision_valid and all(
        value >= required_groups for value in planned_groups.values()
    )

    gate_outcomes = {
        "evidence_identity": "pass" if identity_ok else "invalid",
        "pairing_geometry": _outcome(pairing_checks, identity_ok=identity_ok),
        "geometry_contract": _outcome(geometry_checks, identity_ok=identity_ok),
        "null_control_validity": _outcome(
            review_scope_checks, identity_ok=identity_ok
        ),
        "review_contract_resolution": "invalid",
        "precision_feasibility": (
            "met" if precision_met else "unmet" if precision_valid else "invalid"
        ),
    }
    validity_pass = all(
        gate_outcomes[key] == "pass"
        for key in (
            "evidence_identity",
            "pairing_geometry",
            "geometry_contract",
            "null_control_validity",
        )
    )
    gate_outcomes["review_contract_resolution"] = (
        "favorable" if validity_pass else "indeterminate"
        if identity_ok else "invalid"
    )

    parent = documents.get("s1_v2_summary", {})
    archived_observations = {
        "its_shifted_target_null_ucb_db": _nested(
            parent, "null_control", "by_dataset", "ITS", "upper"
        ),
        "haze4k_negative_tail_prevalence": _nested(
            parent,
            "prevalence",
            "negative_tail_scene",
            "by_dataset",
            "HAZE4K_TRAIN",
            "estimate",
        ),
        "cross_observation_transfer_overall_db": _nested(
            parent, "cross_observation_transfer", "overall", "estimate"
        ),
        "edge_minus_interior_local_advantage_db_descriptive": _nested(
            parent, "diagnostics", "edge_minus_interior_local_advantage_db"
        ),
        "primary_precision_prior_gate": _nested(
            parent, "gate_outcomes", "primary_precision"
        ),
    }
    missing_evidence = [
        "A frozen diagnostic that can distinguish an invalid ITS shifted-target null from a genuine target-construction response.",
        "A predeclared check of whether Haze4K negative-tail behavior persists under the valid measurement contract.",
        "A predeclared cross-observation transfer check under the same valid measurement contract.",
        "Uncertainty-qualified crop and edge/interior diagnostics; the archived edge/interior value remains descriptive only.",
    ]
    summary = {
        "schema_version": 1,
        "route_id": context.route_id,
        "operation_id": context.operation_id,
        "run_id": context.run_id,
        "scope": "archived compact measurement-validity and precision review only",
        "input_errors": input_errors,
        "github_archived_facts": {
            "parent_terminal": {
                "state": _nested(documents.get("s1_v2_closeout", {}), "state"),
                "decision": _nested(documents.get("s1_v2_closeout", {}), "decision"),
                "authorizes": _nested(
                    documents.get("s1_v2_closeout", {}), "authorizes"
                ),
            },
            "pairing_terminal": {
                "state": _nested(documents.get("pairing_v2_closeout", {}), "state"),
                "decision": _nested(
                    documents.get("pairing_v2_closeout", {}), "decision"
                ),
                "authorizes": _nested(
                    documents.get("pairing_v2_closeout", {}), "authorizes"
                ),
            },
            "archived_observations": archived_observations,
        },
        "ai_inference": (
            "The compact identity, pairing, geometry, diagnostic-scope, and planning-capacity records are sufficient only to author a revised S1 utility-qualification contract; they do not resolve the archived S1 scientific terminal."
            if validity_pass and precision_met
            else "The compact records do not yet provide a complete basis for revised S1 contract authoring."
        ),
        "missing_evidence": missing_evidence,
        "checks": {
            "evidence_identity": identity_checks,
            "pairing_geometry": pairing_checks,
            "geometry_contract": geometry_checks,
            "review_scope": review_scope_checks,
        },
        "precision_feasibility": {
            "unit": "independent original-clear scenes per dataset stratum",
            "planning_sd_upper_bound_db": planning_sd_upper_bound_db,
            "critical_value": critical_value,
            "target_half_width_db": target_half_width_db,
            "required_groups_per_stratum": required_groups,
            "planned_groups_per_stratum": planned_groups,
        },
        "gate_outcomes": gate_outcomes,
        "forbidden_activity_receipt": {
            "raw_images_accessed": False,
            "weights_or_checkpoints_accessed": False,
            "arrays_or_large_tables_accessed": False,
            "historical_metric_recomputed": False,
            "historical_terminal_modified": False,
            "model_structure_modified": False,
            "training_or_inference_occurred": False,
            "protected_data_touched": False,
        },
    }
    atomic_json(output_file(context, SUMMARY_FILE), summary)

    gate_summary = {
        "schema_version": 1,
        "route_id": context.route_id,
        "operation_id": context.operation_id,
        "run_id": context.run_id,
        "gate_outcomes": gate_outcomes,
        "validity_veto_gates": [
            "evidence_identity",
            "pairing_geometry",
            "geometry_contract",
            "null_control_validity",
        ],
        "inconclusive_only_gates": ["precision_feasibility"],
        "descriptive_observations": [
            "edge_minus_interior_local_advantage_db_descriptive"
        ],
        "summary_filename": SUMMARY_FILE,
    }
    gate_path = output_file(context, GATE_SUMMARY_FILE)
    atomic_json(gate_path, gate_summary)
    gate_sha256 = hashlib.sha256(gate_path.read_bytes()).hexdigest()
    facts = [
        build_gate_review_fact(
            fact_id=gate_id,
            metric=f"{gate_id} typed gate outcome",
            unit="typed outcome",
            population="archived development-screening compact evidence",
            grouping="original clear scene; haze observations and spatial regions remain nested",
            gate_outcome=outcome,
            source_filename=GATE_SUMMARY_FILE,
            source_sha256=gate_sha256,
        )
        for gate_id, outcome in gate_outcomes.items()
    ]
    write_review_facts(context, relpath=REVIEW_FACTS_FILE, facts=facts)
    write_gate_result(
        context,
        gate_outcomes=gate_outcomes,
        details={
            "summary_file": SUMMARY_FILE,
            "gate_summary_file": GATE_SUMMARY_FILE,
            "review_facts_file": REVIEW_FACTS_FILE,
            "compact_input_count": len(INPUT_ASSETS),
            "raw_or_protected_data_touched": False,
            "historical_results_modified": False,
            "model_or_training_activity": False,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["contract", "run"])
    parser.add_argument("--context", required=True)
    args = parser.parse_args()
    if args.phase == "contract":
        contract(args.context)
    else:
        run(args.context)


if __name__ == "__main__":
    main()
