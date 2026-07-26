#!/usr/bin/env python3
"""Qualify an outcome-blind role ledger for the existing ConvIR datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

from route_program_api import (
    asset_path,
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


ROUTE_ID = "convir-existing-dataset-role-ledger-v1"
OPERATION_ID = "CONVIR_EXISTING_DATASET_ROLE_LEDGER_QUALIFY"
SUMMARY_NAME = "convir_existing_dataset_role_ledger_v1_summary.json"
RAW_LEDGER_NAME = "reside_scene_role_ledger.jsonl"

MEASUREMENT_SELECTION_SALT = "reside-known-overlap-continuous-utility-measurement-v1"
LEDGER_ALLOCATION_SALT = "convir-existing-dataset-role-ledger-v1"
HISTORICAL_SCENES_PER_STRATUM = 1587
TARGET_HALF_WIDTH_DB = 0.025
UTILITY_MARGIN_DB = 0.05
SIMULTANEOUS_Z = 2.241402727604947
SD_ASSURANCE_MULTIPLIER = 1.05

EXPECTED = {
    "ITS": {
        "official": 11000,
        "excluded": 2187,
        "eligible": 8813,
        "historical": HISTORICAL_SCENES_PER_STRATUM,
        "unassigned": 7226,
        "planning_sd": 0.602251735447599,
        "planning_sd_upper": 0.632364322219979,
        "required": 3215,
        "historical_digest": "29b68fb172c766cf50a5e2193f6e3809db9037426784bc356b585596ab675775",
    },
    "OTS": {
        "official": 8970,
        "excluded": 964,
        "eligible": 8006,
        "historical": HISTORICAL_SCENES_PER_STRATUM,
        "unassigned": 6419,
        "planning_sd": 0.5170106753649101,
        "planning_sd_upper": 0.5428612091331556,
        "required": 2369,
        "historical_digest": "084274e75172fdb90df2489af8c440527226e6176db99648cb0c98873dc56fec",
    },
}

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def read_json_asset(context: Any, asset_id: str) -> dict[str, Any]:
    value = json.loads(
        asset_path(context, asset_id, kind="file").read_text(encoding="utf-8")
    )
    if not isinstance(value, dict):
        raise ValueError(f"{asset_id} must contain a JSON object")
    return value


def read_unique_lines(path: Path) -> tuple[list[str], set[str]]:
    values = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return values, set(values)


def image_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"dataset directory is unavailable: {directory}")
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def digest_lines(values: Iterable[str]) -> str:
    return hashlib.sha256(
        "\n".join(sorted(values)).encode("utf-8")
    ).hexdigest()


def digest_order(values: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def salted_order(values: Iterable[str], salt: str, namespace: str) -> list[str]:
    return sorted(
        values,
        key=lambda value: (
            hashlib.sha256(f"{salt}|{namespace}|{value}".encode()).hexdigest(),
            value,
        ),
    )


def close_enough(actual: Any, expected: float) -> bool:
    return isinstance(actual, (int, float)) and not isinstance(actual, bool) \
        and math.isfinite(float(actual)) and abs(float(actual) - expected) <= 1e-12


def parent_evidence_checks(context: Any) -> tuple[dict[str, bool], dict[str, Any]]:
    haze_train_closeout = read_json_asset(context, "haze4k_train_split_closeout")
    haze_train_summary = read_json_asset(context, "haze4k_train_split_summary")
    haze_test_closeout = read_json_asset(context, "haze4k_test_split_closeout")
    haze_test_summary = read_json_asset(context, "haze4k_test_split_summary")
    its_closeout = read_json_asset(context, "its_quarantine_closeout")
    its_summary = read_json_asset(context, "its_quarantine_summary")
    ots_closeout = read_json_asset(context, "ots_quarantine_closeout")
    ots_summary = read_json_asset(context, "ots_quarantine_summary")
    measurement_closeout = read_json_asset(context, "reside_measurement_closeout")
    measurement_summary = read_json_asset(context, "reside_measurement_summary")
    nhhaze_closeout = read_json_asset(context, "nhhaze_exposure_closeout")
    nhhaze_identity = read_json_asset(context, "nhhaze_identity")

    checks = {
        "haze4k_train_terminal": (
            haze_train_closeout.get("state") == "COMPLETED_GATE_PASS"
            and haze_train_closeout.get("decision") == "HAZE4K_TRAIN_SCENE_SPLIT_PASS"
            and haze_train_closeout.get("details", {}).get("canonical_scenes") == 750
            and haze_train_summary.get("frozen_split", {}).get("training_scene_count") == 600
            and haze_train_summary.get("frozen_split", {}).get(
                "internal_development_scene_count"
            ) == 150
            and haze_train_summary.get("frozen_split", {}).get("assignment_digest")
            == "7b21d3af455475f7bb29198081a2ef2e651cffaac6149fd27741863c765b4efc"
        ),
        "haze4k_test_terminal": (
            haze_test_closeout.get("state") == "COMPLETED_GATE_PASS"
            and haze_test_closeout.get("decision") == "HAZE4K_TEST_SCENE_SPLIT_PASS"
            and haze_test_summary.get("exposure_class")
            == "baseline_exposed_candidate_unseen"
            and haze_test_summary.get("frozen_split", {}).get(
                "development_scene_count"
            ) == 100
            and haze_test_summary.get("frozen_split", {}).get(
                "candidate_confirmation_scene_count"
            ) == 150
            and haze_test_summary.get("frozen_split", {}).get("assignment_digest")
            == "6ca5174470dad2b4eef4ae15c5a13a99d8ae9fc0bc2ea1116b199c4d4bc05582"
        ),
        "its_quarantine_terminal": (
            its_closeout.get("state") == "COMPLETED_GATE_PASS"
            and its_closeout.get("decision") == "ITS_VERIFIED_OVERLAP_QUARANTINE_PASS"
            and its_closeout.get("authorizes")
            == "ITS_KNOWN_OVERLAP_QUARANTINED_MEASUREMENT_DESIGN"
            and its_summary.get("quarantine_tiers", {}).get(
                "train_and_selection_exposure", {}
            ).get("authorized_exclusion_ids") == EXPECTED["ITS"]["excluded"]
            and its_summary.get("quarantine_tiers", {}).get(
                "train_and_selection_exposure", {}
            ).get("eligible_its_scenes") == EXPECTED["ITS"]["eligible"]
        ),
        "ots_quarantine_terminal": (
            ots_closeout.get("state") == "COMPLETED_GATE_PASS"
            and ots_closeout.get("decision") == "OTS_TARGETED_GEOMETRY_PASS"
            and ots_closeout.get("authorizes") == "OTS_OUTDOOR_MEASUREMENT_DESIGN"
            and ots_summary.get("exclusion_pool", {}).get(
                "deduplicated_exclusion_count"
            ) == EXPECTED["OTS"]["excluded"]
            and ots_summary.get("exclusion_pool", {}).get("eligible_ots_scenes")
            == EXPECTED["OTS"]["eligible"]
        ),
        "reside_measurement_terminal": (
            measurement_closeout.get("state") == "COMPLETED_INCONCLUSIVE"
            and measurement_closeout.get("decision")
            == "RESIDE_CONTINUOUS_UTILITY_MEASUREMENT_INCONCLUSIVE"
            and measurement_closeout.get("authorizes") == "NONE"
            and measurement_summary.get("selection", {}).get("salt")
            == MEASUREMENT_SELECTION_SALT
            and measurement_summary.get("selection", {}).get("scenes_per_stratum")
            == HISTORICAL_SCENES_PER_STRATUM
            and close_enough(
                measurement_summary.get("estimand", {}).get("utility_margin_db"),
                UTILITY_MARGIN_DB,
            )
            and close_enough(
                measurement_summary.get("estimand", {}).get("precision_distance_db"),
                TARGET_HALF_WIDTH_DB,
            )
            and all(
                measurement_summary.get("selection", {}).get("scene_digests", {}).get(
                    stratum
                ) == EXPECTED[stratum]["historical_digest"]
                and close_enough(
                    measurement_summary.get("strata", {}).get(stratum, {}).get(
                        "sample_sd_db"
                    ),
                    EXPECTED[stratum]["planning_sd"],
                )
                for stratum in ("ITS", "OTS")
            )
        ),
        "nhhaze_development_only": (
            nhhaze_closeout.get("state") == "COMPLETED_GATE_FAIL"
            and nhhaze_closeout.get("decision")
            == "R15_S0E_REAL_ACTION_HEADROOM_FAIL_STRATEGIC_RESET"
            and nhhaze_identity.get("data_role")
            == "development_screening_previously_used_by_v2_7"
            and nhhaze_identity.get("eligible_as_final_external_validation") is False
            and nhhaze_identity.get("pass1_image_count") == 55
        ),
    }
    evidence = {
        "haze4k_train": {
            "independent_scenes": 750,
            "training_scenes": 600,
            "internal_development_scenes": 150,
            "assignment_digest": haze_train_summary.get("frozen_split", {}).get(
                "assignment_digest"
            ),
            "role": "development_only",
        },
        "haze4k_test": {
            "independent_scenes": 250,
            "historical_development_scenes": 100,
            "candidate_confirmation_scenes": 150,
            "assignment_digest": haze_test_summary.get("frozen_split", {}).get(
                "assignment_digest"
            ),
            "role": "split_development_and_candidate_confirmation",
            "protected_confirmation_preserved": True,
        },
        "nhhaze": {
            "independent_scenes": 55,
            "historical_development_scenes": 55,
            "unassigned_scenes": 0,
            "eligible_as_final_external_validation": False,
            "role": "development_only",
        },
    }
    return checks, evidence


def enumerate_its(reside_root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    specifications = (
        ("ITS_TRAIN", reside_root / "official/ITS/train/ITS_clear"),
        ("ITS_VALIDATION", reside_root / "official/ITS/val/clear"),
    )
    for namespace, directory in specifications:
        for path in image_files(directory):
            scene_id = f"{namespace}:{path.stem}"
            if scene_id in result:
                raise ValueError(f"duplicate ITS scene id: {scene_id}")
            result[scene_id] = path
    return result


def enumerate_ots(reside_root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in image_files(reside_root / "official/OTS_ALPHA/clear_images"):
        if path.stem in result:
            raise ValueError(f"duplicate OTS scene id: {path.stem}")
        result[path.stem] = path
    return result


def reconstruct_stratum(
    stratum: str,
    official: dict[str, Path],
    exclusion_lines: list[str],
    exclusions: set[str],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, bool]]:
    expected = EXPECTED[stratum]
    official_ids = set(official)
    eligible = official_ids - exclusions
    previous_namespace = f"{stratum}-scenes"
    historical = set(
        salted_order(
            eligible, MEASUREMENT_SELECTION_SALT, previous_namespace
        )[:HISTORICAL_SCENES_PER_STRATUM]
    )
    unassigned = eligible - historical
    allocation_order = salted_order(
        unassigned, LEDGER_ALLOCATION_SALT, f"{stratum}-unassigned-scenes"
    )
    allocation_rank = {
        scene_id: rank for rank, scene_id in enumerate(allocation_order)
    }

    namespace_checks = {
        "official_count": len(official_ids) == expected["official"],
        "exclusion_rows_unique": len(exclusion_lines) == len(exclusions),
        "exclusion_count": len(exclusions) == expected["excluded"],
        "all_exclusions_official": exclusions.issubset(official_ids),
        "eligible_count": len(eligible) == expected["eligible"],
    }
    role_checks = {
        "historical_count": len(historical) == expected["historical"],
        "historical_digest": digest_lines(historical)
        == expected["historical_digest"],
        "unassigned_count": len(unassigned) == expected["unassigned"],
        "eligible_partition": historical.isdisjoint(unassigned)
        and historical | unassigned == eligible,
        "official_partition": exclusions.isdisjoint(eligible)
        and exclusions | eligible == official_ids,
        "allocation_order_complete": set(allocation_order) == unassigned
        and len(allocation_order) == len(unassigned),
    }
    required = expected["required"]
    summary = {
        "official_scenes": len(official_ids),
        "known_overlap_excluded_scenes": len(exclusions),
        "known_overlap_quarantined_eligible_scenes": len(eligible),
        "historical_development_exposure_scenes": len(historical),
        "unassigned_scenes": len(unassigned),
        "historical_exposure_digest": digest_lines(historical),
        "unassigned_membership_digest": digest_lines(unassigned),
        "unassigned_allocation_order_digest": digest_order(allocation_order),
        "future_formal_scene_requirement": required,
        "capacity_margin_scenes": len(unassigned) - required,
        "capacity_feasible": len(unassigned) >= required,
        "planning_sd_db": expected["planning_sd"],
        "planning_sd_upper_bound_db": expected["planning_sd_upper"],
    }
    rows = []
    for scene_id in sorted(official_ids):
        if scene_id in exclusions:
            role = "known_overlap_excluded"
            rank = None
        elif scene_id in historical:
            role = "historical_development_exposure"
            rank = None
        else:
            role = "unassigned_known_overlap_quarantined"
            rank = allocation_rank[scene_id]
        rows.append({
            "schema_version": 1,
            "dataset": stratum,
            "scene_id": scene_id,
            "independent_unit": "original_clear_scene",
            "role": role,
            "allocation_rank": rank,
        })
    return summary, rows, {**namespace_checks, **role_checks}


def write_once_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"write-once ledger already exists: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(
                    json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                )
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(
                f"write-once ledger already exists: {path}"
            ) from exc
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unit_input_sha256(
    stratum: str, summary: dict[str, Any], checks: dict[str, bool]
) -> str:
    value = {
        "schema_version": 1,
        "route_id": ROUTE_ID,
        "operation_id": OPERATION_ID,
        "stratum": stratum,
        "measurement_selection_salt": MEASUREMENT_SELECTION_SALT,
        "ledger_allocation_salt": LEDGER_ALLOCATION_SALT,
        "summary": summary,
        "checks": checks,
    }
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def contract(context_path: Path) -> None:
    context = load_context(context_path, "contract")
    prepare_phase_output(context)
    write_contract_progress(
        context,
        completed_iterations=1,
        total_iterations=1,
        stage="metadata_role_contract",
    )
    write_contract_result(
        context,
        checks={
            "cpu_contract": context.device == "cpu",
            "metadata_only_mode": context.engineering_contract.get("mode")
            == "metadata_only",
            "protected_permissions_disabled": not any(
                context.protected_data_permissions.values()
            ),
            "reside_dataset_hidden": "reside_root" not in context.assets,
        },
        engineering={
            "mode": "metadata_only",
            "device": "cpu",
            "fixture": None,
            "production_path_exercised": False,
            "protected_data_touched": False,
            "scientific_output_created": False,
            "scientific_training_occurred": False,
        },
    )


def run(context_path: Path) -> None:
    context = load_context(context_path, "run")
    prepare_phase_output(context)
    if context.total_units != 2 or context.device != "cpu":
        raise RuntimeError("role-ledger runtime contract mismatch")
    if any(context.protected_data_permissions.values()):
        raise RuntimeError("role-ledger qualification forbids protected permissions")
    if load_completed_unit_ledger(context):
        raise RuntimeError("fresh role-ledger workload unexpectedly has completed units")

    evidence_checks, fixed_roles = parent_evidence_checks(context)
    reside_root = asset_path(context, "reside_root", kind="directory")
    its_official = enumerate_its(reside_root)
    its_lines, its_exclusions = read_unique_lines(
        asset_path(context, "its_exclusions", kind="file")
    )
    its_summary, its_rows, its_checks = reconstruct_stratum(
        "ITS", its_official, its_lines, its_exclusions
    )
    its_output_relpath = "units/its_role_reconstruction.json"
    atomic_json(
        output_file(context, its_output_relpath),
        {
            "schema_version": 1,
            "stratum": "ITS",
            "summary": its_summary,
            "checks": its_checks,
        },
    )
    record_completed_unit(
        context,
        unit_id="its_role_reconstruction",
        input_sha256=unit_input_sha256("ITS", its_summary, its_checks),
        output_relpath=its_output_relpath,
    )
    write_workload_progress(
        context, completed_units=1, stage="its_namespace_reconstructed"
    )

    ots_official = enumerate_ots(reside_root)
    ots_lines, ots_exclusions = read_unique_lines(
        asset_path(context, "ots_exclusions", kind="file")
    )
    ots_summary, ots_rows, ots_checks = reconstruct_stratum(
        "OTS", ots_official, ots_lines, ots_exclusions
    )
    ots_output_relpath = "units/ots_role_reconstruction.json"
    atomic_json(
        output_file(context, ots_output_relpath),
        {
            "schema_version": 1,
            "stratum": "OTS",
            "summary": ots_summary,
            "checks": ots_checks,
        },
    )
    record_completed_unit(
        context,
        unit_id="ots_role_reconstruction",
        input_sha256=unit_input_sha256("OTS", ots_summary, ots_checks),
        output_relpath=ots_output_relpath,
    )

    evidence_identity_ok = all(evidence_checks.values())
    namespace_keys = {
        "official_count",
        "exclusion_rows_unique",
        "exclusion_count",
        "all_exclusions_official",
        "eligible_count",
    }
    role_keys = set(its_checks) - namespace_keys
    namespace_ok = all(
        checks[key]
        for checks in (its_checks, ots_checks)
        for key in namespace_keys
    )
    roles_ok = all(
        checks[key]
        for checks in (its_checks, ots_checks)
        for key in role_keys
    )
    capacity_ok = all(
        item["capacity_feasible"] for item in (its_summary, ots_summary)
    )

    ledger_path = output_file(context, RAW_LEDGER_NAME)
    write_once_jsonl(ledger_path, [*its_rows, *ots_rows])
    summary = {
        "schema_version": 1,
        "route_id": ROUTE_ID,
        "operation_id": OPERATION_ID,
        "run_id": context.run_id,
        "scope": "outcome-blind role and exposure qualification for the existing ConvIR dataset inventory",
        "independent_unit": "original_clear_scene",
        "evidence_identity_checks": evidence_checks,
        "fixed_dataset_roles": fixed_roles,
        "reside_strata": {
            "ITS": its_summary,
            "OTS": ots_summary,
        },
        "namespace_and_role_checks": {
            "ITS": its_checks,
            "OTS": ots_checks,
        },
        "precision_feasibility": {
            "method": "normal mean with two-stratum simultaneous 95 percent critical value",
            "simultaneous_critical_value": SIMULTANEOUS_Z,
            "target_half_width_db": TARGET_HALF_WIDTH_DB,
            "utility_margin_db": UTILITY_MARGIN_DB,
            "planning_sd_assurance_multiplier": SD_ASSURANCE_MULTIPLIER,
            "planning_sd_rule": "Multiply each observed 1,587-scene sample SD by 1.05, a conservative bound above the one-sided 95 percent normal-theory SD multiplier implied by the chi-square lower-tail concentration bound.",
            "feasible": capacity_ok,
        },
        "future_claim_protocol": {
            "allocation_salt": LEDGER_ALLOCATION_SALT,
            "ordering": "SHA-256(salt + vertical-bar + stratum + -unassigned-scenes + vertical-bar + scene_id), then scene_id",
            "claim_unit": "contiguous offset and count within each stratum-specific unassigned order",
            "claim_status": "no scene is claimed or exposed by this qualification route",
            "authorization_required": "a later typed contract must freeze offset, count, estimand, gates, and data role before measurement",
        },
        "reporting_policy": {
            "ITS": "separate indoor synthetic stratum",
            "OTS": "separate outdoor synthetic stratum",
            "pooling": "no sample-size-weighted exchangeable pooling",
            "cross_dataset_estimand": "only a separately predeclared stratified estimand such as the maximum of stratum means",
        },
        "raw_cloud_ledger": {
            "filename": RAW_LEDGER_NAME,
            "rows": len(its_rows) + len(ots_rows),
            "sha256": sha256_file(ledger_path),
            "archived_to_github": False,
        },
        "protected_data_touched": False,
        "model_or_checkpoint_accessed": False,
        "image_content_decoded": False,
        "training_or_inference_occurred": False,
        "limitations": [
            "ITS and OTS are known-overlap-quarantined populations, not fully source-disjoint populations.",
            "The remaining 574 unresolved ITS parent groups are not interpreted as non-overlaps.",
            "Haze variants, tiles, regions, pixels, actions, and resamples never increase the independent scene count.",
            "NH-Haze is development-only because all 55 scenes participated in prior decisions.",
            "The Haze4K 150-scene candidate-confirmation role is preserved without reading its images, targets, or outcomes.",
            "This route authorizes contract authoring only and does not authorize a measurement, training run, model selection, confirmation, or deployment claim.",
        ],
        "marker": "CONVIR_EXISTING_DATASET_ROLE_LEDGER_V1_COMPLETE",
    }
    atomic_json(output_file(context, SUMMARY_NAME), summary)
    write_workload_progress(
        context, completed_units=2, stage="dataset_role_ledger_finalized"
    )

    validity_ok = evidence_identity_ok and namespace_ok and roles_ok
    write_gate_result(
        context,
        gate_outcomes={
            "evidence_identity": "pass" if evidence_identity_ok else "fail",
            "namespace_reconstruction": "pass" if namespace_ok else "fail",
            "role_disjointness": "pass" if roles_ok else "fail",
            "remaining_capacity": (
                "favorable" if capacity_ok else "unfavorable"
            ) if validity_ok else "invalid",
            "future_precision_feasibility": (
                "met" if capacity_ok else "unmet"
            ) if validity_ok else "invalid",
        },
        details={
            "summary_file": SUMMARY_NAME,
            "raw_cloud_ledger_file": RAW_LEDGER_NAME,
            "raw_cloud_ledger_sha256": summary["raw_cloud_ledger"]["sha256"],
            "unassigned_scenes": {
                "ITS": its_summary["unassigned_scenes"],
                "OTS": ots_summary["unassigned_scenes"],
            },
            "required_scenes": {
                "ITS": EXPECTED["ITS"]["required"],
                "OTS": EXPECTED["OTS"]["required"],
            },
            "protected_data_touched": False,
            "training_or_inference_occurred": False,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("contract", "run"))
    parser.add_argument("--context", required=True, type=Path)
    arguments = parser.parse_args()
    if arguments.phase == "contract":
        contract(arguments.context)
    else:
        run(arguments.context)


if __name__ == "__main__":
    main()
