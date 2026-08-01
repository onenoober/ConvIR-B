#!/usr/bin/env python3
"""Requalify S1 utility from SHA-bound archived scene-level evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from route_program_api import (
    asset_path,
    atomic_json,
    build_gate_review_fact,
    load_completed_unit_ledger,
    load_context,
    output_file,
    record_completed_unit,
    write_contract_result,
    write_gate_result,
    write_review_facts,
)


ROUTE_ID = "daytime-dehazing-local-restoration-need-qualification-v3"
OPERATION_ID = "DAYTIME_DEHAZING_LOCAL_RESTORATION_NEED_QUALIFY"
SUMMARY_FILE = "daytime_dehazing_local_restoration_need_v3_summary.json"
GEOMETRY_FILE = "daytime_dehazing_local_restoration_need_v3_geometry.json"
GATE_FILE = "daytime_dehazing_local_restoration_need_v3_gate_summary.json"
REVIEW_FACTS_FILE = "daytime_dehazing_local_restoration_need_v3_review_facts.json"

DATASETS = ("HAZE4K_TRAIN", "ITS", "OTS")
SCENES_PER_DATASET = 150
TOTAL_SCENES = SCENES_PER_DATASET * len(DATASETS)
BOOTSTRAP_RESAMPLES = 20_000
BOOTSTRAP_LOWER_QUANTILE = 0.005
BOOTSTRAP_UPPER_QUANTILE = 0.995
BOOTSTRAP_SEED = 20260801
NULL_OVERALL_MARGIN_DB = 0.05
NULL_DATASET_MARGIN_DB = 0.10

ASSET_SHA256 = {
    "s1_v2_closeout": "64831258a340a0e3f9a3bad82c8138ce144c837b3e01cd42ded67daa82d59992",
    "s1_v2_summary": "fdb15c9da34b6e9bba815927578898957b1980f87d7bb3295dfb1b31fb3fa068",
    "s1_v2_gate_summary": "a498559a645d547472185d889af9c19ce589f56ed91564cb3931deda4e338a69",
    "s1_v2_scene_metrics": "db6cfe210b3eedf08e06d2b6fd4447f523b1777661d87ccb53e0bf5e99bb1f8e",
    "s1_review_closeout": "b05665be761bbe90707cc622ce24cba285027b3c43a34041a007bcbe0e321670",
}

REQUIRED_COLUMNS = {
    "dataset",
    "scene_id_sha256",
    "nested_variants",
    "null_local_minus_global_psnr_db",
    "negative_tail_scene",
    "cross_observation_transfer_psnr_db",
    "primary_local_minus_global_psnr_db",
    "edge_local_minus_global_psnr_db",
    "interior_local_minus_global_psnr_db",
    "maximum_full_image_padded_pixels",
    "whole_image_inference_before_scoring_crop",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path.name}")
    return value


def nested(value: Any, *keys: str) -> Any:
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def parse_bool(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"invalid boolean token: {value!r}")


def quantile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values or not 0.0 <= probability <= 1.0:
        raise ValueError("invalid quantile input")
    position = probability * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return (
        sorted_values[lower] * (1.0 - weight)
        + sorted_values[upper] * weight
    )


def bootstrap_family(
    values_by_dataset: dict[str, list[float]], seed: int
) -> dict[str, Any]:
    if set(values_by_dataset) != set(DATASETS) or any(
        len(values_by_dataset[dataset]) != SCENES_PER_DATASET
        for dataset in DATASETS
    ):
        raise ValueError("bootstrap requires the frozen 150-scene strata")
    rng = random.Random(seed)
    draws = {dataset: [] for dataset in DATASETS}
    overall_draws: list[float] = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        dataset_means = []
        for dataset in DATASETS:
            values = values_by_dataset[dataset]
            sampled_mean = sum(
                values[rng.randrange(SCENES_PER_DATASET)]
                for _ in range(SCENES_PER_DATASET)
            ) / SCENES_PER_DATASET
            draws[dataset].append(sampled_mean)
            dataset_means.append(sampled_mean)
        overall_draws.append(sum(dataset_means) / len(dataset_means))
    result: dict[str, Any] = {"by_dataset": {}}
    for dataset in DATASETS:
        samples = sorted(draws[dataset])
        estimate = sum(values_by_dataset[dataset]) / SCENES_PER_DATASET
        result["by_dataset"][dataset] = {
            "estimate": estimate,
            "lower": quantile(samples, BOOTSTRAP_LOWER_QUANTILE),
            "upper": quantile(samples, BOOTSTRAP_UPPER_QUANTILE),
            "scene_count": SCENES_PER_DATASET,
            "resamples": BOOTSTRAP_RESAMPLES,
        }
    overall_samples = sorted(overall_draws)
    overall_estimate = sum(
        result["by_dataset"][dataset]["estimate"] for dataset in DATASETS
    ) / len(DATASETS)
    result["overall_equal_dataset_weight"] = {
        "estimate": overall_estimate,
        "lower": quantile(overall_samples, BOOTSTRAP_LOWER_QUANTILE),
        "upper": quantile(overall_samples, BOOTSTRAP_UPPER_QUANTILE),
        "scene_count": TOTAL_SCENES,
        "resamples": BOOTSTRAP_RESAMPLES,
        "aggregation": "equal_weight_over_three_dataset_means",
    }
    return result


def load_rows(path: Path) -> tuple[list[dict[str, Any]], dict[str, int], set[str]]:
    rows: list[dict[str, Any]] = []
    counts = defaultdict(int)
    seen = set()
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = set(reader.fieldnames or ())
        if not REQUIRED_COLUMNS <= columns:
            raise RuntimeError("archived scene CSV lacks required frozen columns")
        for raw in reader:
            dataset = raw["dataset"]
            scene_id = raw["scene_id_sha256"]
            if dataset not in DATASETS or len(scene_id) != 64:
                raise RuntimeError("archived scene identity is invalid")
            int(scene_id, 16)
            if scene_id in seen:
                raise RuntimeError("archived scene identity is duplicated")
            seen.add(scene_id)
            row = {
                "dataset": dataset,
                "scene_id_sha256": scene_id,
                "nested_variants": int(raw["nested_variants"]),
                "null": float(raw["null_local_minus_global_psnr_db"]),
                "negative_tail": parse_bool(raw["negative_tail_scene"]),
                "transfer": float(raw["cross_observation_transfer_psnr_db"]),
                "primary": float(raw["primary_local_minus_global_psnr_db"]),
                "edge": float(raw["edge_local_minus_global_psnr_db"]),
                "interior": float(raw["interior_local_minus_global_psnr_db"]),
                "maximum_padded_pixels": int(
                    raw["maximum_full_image_padded_pixels"]
                ),
                "whole_image_before_crop": parse_bool(
                    raw["whole_image_inference_before_scoring_crop"]
                ),
            }
            numeric = (
                row["null"],
                row["transfer"],
                row["primary"],
                row["edge"],
                row["interior"],
            )
            if row["nested_variants"] != 2 or not all(
                math.isfinite(float(item)) for item in numeric
            ):
                raise RuntimeError("archived scene metric row is invalid")
            if row["maximum_padded_pixels"] <= 0 or not row["whole_image_before_crop"]:
                raise RuntimeError("archived scene geometry row is invalid")
            rows.append(row)
            counts[dataset] += 1
    return rows, dict(counts), columns


def identity_checks(context: Any) -> tuple[dict[str, Path], dict[str, Any], dict[str, bool]]:
    paths = {
        identifier: asset_path(context, identifier, kind="file")
        for identifier in ASSET_SHA256
    }
    checks = {
        f"{identifier}_sha256": sha256_file(path) == ASSET_SHA256[identifier]
        for identifier, path in paths.items()
    }
    documents = {
        identifier: load_json(paths[identifier])
        for identifier in (
            "s1_v2_closeout",
            "s1_v2_summary",
            "s1_v2_gate_summary",
            "s1_review_closeout",
        )
    }
    v2 = documents["s1_v2_closeout"]
    review = documents["s1_review_closeout"]
    checks.update({
        "v2_terminal": (
            v2.get("route_id")
            == "daytime-dehazing-local-restoration-need-qualification-v2"
            and v2.get("state") == "COMPLETED_INCONCLUSIVE"
            and v2.get("decision")
            == "DAYTIME_DEHAZING_LOCAL_RESTORATION_NEED_INCONCLUSIVE"
            and v2.get("authorizes")
            == "S1_MEASUREMENT_VALIDITY_OR_PRECISION_REVIEW_ONLY"
        ),
        "v2_summary_bound": nested(
            v2,
            "evidence_sha256",
            "daytime_dehazing_local_restoration_need_v2_summary.json",
        ) == ASSET_SHA256["s1_v2_summary"],
        "v2_gate_summary_bound": nested(
            v2,
            "evidence_sha256",
            "daytime_dehazing_local_restoration_need_v2_gate_summary.json",
        ) == ASSET_SHA256["s1_v2_gate_summary"],
        "v2_scene_metrics_bound": nested(
            v2,
            "evidence_sha256",
            "daytime_dehazing_local_restoration_need_v2_scene_metrics.csv",
        ) == ASSET_SHA256["s1_v2_scene_metrics"],
        "review_terminal": (
            review.get("route_id")
            == "daytime-dehazing-s1-validity-precision-review-v1"
            and review.get("state") == "COMPLETED_GATE_PASS"
            and review.get("decision")
            == "DAYTIME_DEHAZING_S1_VALIDITY_REVIEW_PASS"
            and review.get("authorizes")
            == "AUTHOR_REVISED_S1_UTILITY_QUALIFICATION_CONTRACT_ONLY"
        ),
    })
    return paths, documents, checks


def null_outcomes(summary: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    overall = nested(summary, "null_control", "overall")
    by_dataset = nested(summary, "null_control", "by_dataset")
    records = [overall] + [
        by_dataset.get(dataset) if isinstance(by_dataset, dict) else None
        for dataset in DATASETS
    ]
    if any(
        not isinstance(record, dict)
        or not all(finite_number(record.get(key)) for key in ("estimate", "lower", "upper"))
        for record in records
    ):
        return "invalid", "invalid", {"status": "invalid"}
    artifact = (
        float(overall["lower"]) >= NULL_OVERALL_MARGIN_DB
        or any(
            float(by_dataset[dataset]["lower"]) >= NULL_DATASET_MARGIN_DB
            for dataset in DATASETS
        )
    )
    precision_met = (
        float(overall["upper"]) < NULL_OVERALL_MARGIN_DB
        and all(
            float(by_dataset[dataset]["upper"]) < NULL_DATASET_MARGIN_DB
            for dataset in DATASETS
        )
    )
    return (
        "fail" if artifact else "pass",
        "met" if precision_met else "unmet",
        {
            "status": "affirmative_artifact" if artifact else "no_affirmative_artifact",
            "overall": overall,
            "by_dataset": by_dataset,
            "overall_margin_db": NULL_OVERALL_MARGIN_DB,
            "dataset_margin_db": NULL_DATASET_MARGIN_DB,
            "precision_certifies_absence": precision_met,
        },
    )


def archived_gate(
    gate_summary: dict[str, Any], gate_id: str, allowed: set[str]
) -> str:
    value = nested(gate_summary, "gate_outcomes", gate_id)
    return value if value in allowed else "invalid"


def geometry_diagnostics(
    rows: list[dict[str, Any]], columns: set[str]
) -> dict[str, Any]:
    values = {
        "edge": {dataset: [] for dataset in DATASETS},
        "interior": {dataset: [] for dataset in DATASETS},
        "edge_minus_interior": {dataset: [] for dataset in DATASETS},
    }
    shape_strata = {
        dataset: {"at_most_262144": 0, "above_262144": 0}
        for dataset in DATASETS
    }
    for row in rows:
        dataset = row["dataset"]
        values["edge"][dataset].append(row["edge"])
        values["interior"][dataset].append(row["interior"])
        values["edge_minus_interior"][dataset].append(
            row["edge"] - row["interior"]
        )
        key = (
            "at_most_262144"
            if row["maximum_padded_pixels"] <= 512 * 512
            else "above_262144"
        )
        shape_strata[dataset][key] += 1
    return {
        "schema_version": 1,
        "route_id": ROUTE_ID,
        "operation_id": OPERATION_ID,
        "independent_unit": "original_clear_scene",
        "confidence_level": 0.99,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "edge_local_minus_global_psnr_db": bootstrap_family(
            values["edge"], BOOTSTRAP_SEED + 1
        ),
        "interior_local_minus_global_psnr_db": bootstrap_family(
            values["interior"], BOOTSTRAP_SEED + 2
        ),
        "edge_minus_interior_local_advantage_db": bootstrap_family(
            values["edge_minus_interior"], BOOTSTRAP_SEED + 3
        ),
        "full_image_padded_area_strata": shape_strata,
        "scoring_crop_side_available": "scoring_crop_side" in columns,
        "scope_note": (
            "Edge/interior intervals are uncertainty-qualified. The archived CSV "
            "does not contain scoring-crop side, so padded-area strata are descriptive "
            "and cannot be relabeled as a crop-size effect."
        ),
    }


def contract(context_path: str) -> None:
    context = load_context(Path(context_path), "contract")
    checks = {
        "route_identity": (
            context.route_id == ROUTE_ID
            and context.operation_id == OPERATION_ID
        ),
        "one_aggregate_workload_unit": context.total_units == 1,
        "metadata_only_engineering": (
            context.engineering_contract.get("mode") == "metadata_only"
        ),
        "no_iteration_cost_contract": (
            context.engineering_contract.get("cost_contract") is None
        ),
        "no_protected_permission": not any(
            context.protected_data_permissions.values()
        ),
        "no_model_weight_or_image_path": True,
        "typed_gate_writer_only": True,
    }
    write_contract_result(
        context,
        checks=checks,
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
    if context.total_units != 1 or any(context.protected_data_permissions.values()):
        raise RuntimeError("runtime contract mismatch")
    if load_completed_unit_ledger(context):
        raise RuntimeError("new-output route unexpectedly has completed units")

    paths, documents, checks = identity_checks(context)
    identity_ok = all(checks.values())
    rows: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    columns: set[str] = set()
    row_error = None
    try:
        rows, counts, columns = load_rows(paths["s1_v2_scene_metrics"])
    except (OSError, ValueError, RuntimeError) as exc:
        row_error = f"{type(exc).__name__}:{str(exc)[:256]}"
    coverage_ok = (
        row_error is None
        and len(rows) == TOTAL_SCENES
        and counts == {dataset: SCENES_PER_DATASET for dataset in DATASETS}
    )

    null_validity, null_precision, null_diagnostic = null_outcomes(
        documents["s1_v2_summary"]
    )
    if not identity_ok or not coverage_ok:
        null_validity = "invalid"
        null_precision = "invalid"

    archived = documents["s1_v2_gate_summary"]
    gate_outcomes = {
        "evidence_identity": "pass" if identity_ok else "invalid",
        "archived_scene_coverage": (
            "pass" if coverage_ok and identity_ok else "fail"
            if identity_ok else "invalid"
        ),
        "measurement_null_artifact": null_validity,
        "local_utility_over_global": archived_gate(
            archived,
            "local_utility_over_global",
            {"favorable", "unfavorable", "indeterminate", "invalid"},
        ) if identity_ok and coverage_ok else "invalid",
        "bidirectional_repeatability": archived_gate(
            archived,
            "bidirectional_repeatability",
            {"favorable", "unfavorable", "indeterminate", "invalid"},
        ) if identity_ok and coverage_ok else "invalid",
        "near_clear_fidelity": archived_gate(
            archived,
            "near_clear_fidelity",
            {"safe", "unsafe", "indeterminate", "invalid"},
        ) if identity_ok and coverage_ok else "invalid",
        "primary_precision": archived_gate(
            archived, "primary_precision", {"met", "unmet", "invalid"}
        ) if identity_ok and coverage_ok else "invalid",
    }

    geometry: dict[str, Any]
    try:
        geometry = geometry_diagnostics(rows, columns) if coverage_ok else {
            "status": "invalid",
            "reason": row_error or "coverage_failed",
        }
        geometry_ok = coverage_ok and "edge_local_minus_global_psnr_db" in geometry
    except (ValueError, RuntimeError) as exc:
        geometry = {
            "status": "invalid",
            "reason": f"{type(exc).__name__}:{str(exc)[:256]}",
        }
        geometry_ok = False
    crop_side_available = geometry.get("scoring_crop_side_available") is True
    archived_primary_precision = gate_outcomes["primary_precision"]
    auxiliary_precision = (
        "met" if null_precision == "met" and geometry_ok and crop_side_available
        else "unmet" if identity_ok and coverage_ok and geometry_ok
        else "invalid"
    )
    gate_outcomes["primary_precision"] = (
        "met"
        if archived_primary_precision == "met" and auxiliary_precision == "met"
        else "unmet"
        if archived_primary_precision in {"met", "unmet"}
        and auxiliary_precision in {"met", "unmet"}
        else "invalid"
    )

    geometry_path = output_file(context, GEOMETRY_FILE)
    atomic_json(geometry_path, geometry)
    summary = {
        "schema_version": 1,
        "route_id": context.route_id,
        "operation_id": context.operation_id,
        "run_id": context.run_id,
        "scope": "SHA-bound archived S1 scene-level compact reanalysis",
        "source_bindings": dict(sorted(ASSET_SHA256.items())),
        "identity_checks": checks,
        "coverage": {
            "row_error": row_error,
            "scene_count": len(rows),
            "counts_by_dataset": counts,
            "expected_counts_by_dataset": {
                dataset: SCENES_PER_DATASET for dataset in DATASETS
            },
        },
        "measurement_null": {
            "validity_outcome": null_validity,
            "precision_outcome": null_precision,
            "diagnostic": null_diagnostic,
            "classification_rule": (
                "Affirmative artifact requires a simultaneous LCB at or above "
                "the unchanged 0.05 dB overall or 0.10 dB dataset margin. "
                "Failure to put every UCB below those margins is precision-unmet, "
                "not a validity failure."
            ),
        },
        "precision_components": {
            "archived_primary_precision": archived_primary_precision,
            "auxiliary_precision": auxiliary_precision,
            "combined_typed_gate": gate_outcomes["primary_precision"],
            "auxiliary_requirements": (
                "null-control absence certified, edge/interior geometry valid, "
                "and scoring-crop side explicitly archived"
            ),
        },
        "archived_scientific_gate_outcomes": {
            key: nested(archived, "gate_outcomes", key)
            for key in (
                "local_utility_over_global",
                "bidirectional_repeatability",
                "near_clear_fidelity",
                "primary_precision",
            )
        },
        "geometry_diagnostics_file": GEOMETRY_FILE,
        "gate_outcomes": gate_outcomes,
        "limitations": [
            "This development-screening reanalysis is not independent confirmation.",
            "Historical v2 estimates, thresholds, gate outcomes, and terminal are not modified.",
            "The archived scene CSV has no scoring-crop-side column; padded full-image area cannot be relabeled as crop size.",
            "Edge/interior intervals are descriptive and cannot by themselves establish a crop-boundary artifact.",
            "No model, checkpoint, image, array, training, inference, S2, or protected data is accessed.",
        ],
        "forbidden_activity_receipt": {
            "historical_metric_recomputed": False,
            "historical_terminal_modified": False,
            "raw_images_accessed": False,
            "weights_or_checkpoints_accessed": False,
            "model_structure_modified": False,
            "training_or_inference_occurred": False,
            "protected_data_touched": False,
        },
    }
    summary_path = output_file(context, SUMMARY_FILE)
    atomic_json(summary_path, summary)
    input_identity = hashlib.sha256(
        "|".join(ASSET_SHA256[key] for key in sorted(ASSET_SHA256)).encode("ascii")
    ).hexdigest()
    record_completed_unit(
        context,
        unit_id="archived_s1_compact_reanalysis",
        input_sha256=input_identity,
        output_relpath=SUMMARY_FILE,
    )

    gate_summary = {
        "schema_version": 1,
        "route_id": context.route_id,
        "operation_id": context.operation_id,
        "run_id": context.run_id,
        "gate_outcomes": gate_outcomes,
        "validity_veto_gates": [
            "evidence_identity",
            "archived_scene_coverage",
            "measurement_null_artifact",
        ],
        "decisive_gates": [
            "local_utility_over_global",
            "bidirectional_repeatability",
            "near_clear_fidelity",
        ],
        "inconclusive_only_gates": [
            "primary_precision",
        ],
        "precision_components": {
            "archived_primary_precision": archived_primary_precision,
            "auxiliary_precision": auxiliary_precision,
        },
        "descriptive_results": [
            "edge_local_minus_global_psnr_db",
            "interior_local_minus_global_psnr_db",
            "edge_minus_interior_local_advantage_db",
            "full_image_padded_area_strata",
        ],
        "summary_filename": SUMMARY_FILE,
        "geometry_filename": GEOMETRY_FILE,
    }
    gate_path = output_file(context, GATE_FILE)
    atomic_json(gate_path, gate_summary)
    gate_sha256 = sha256_file(gate_path)
    facts = [
        build_gate_review_fact(
            fact_id=gate_id,
            metric=f"{gate_id} typed gate outcome",
            unit="typed outcome",
            population="archived development-screening S1 scene evidence",
            grouping=(
                "original clear scene; haze observations and spatial regions "
                "remain nested"
            ),
            gate_outcome=outcome,
            source_filename=GATE_FILE,
            source_sha256=gate_sha256,
        )
        for gate_id, outcome in gate_outcomes.items()
    ]
    write_review_facts(context, relpath=REVIEW_FACTS_FILE, facts=facts)
    if len(load_completed_unit_ledger(context)) != 1:
        raise RuntimeError("completed-unit ledger coverage mismatch")
    write_gate_result(
        context,
        gate_outcomes=gate_outcomes,
        details={
            "summary_file": SUMMARY_FILE,
            "geometry_file": GEOMETRY_FILE,
            "gate_summary_file": GATE_FILE,
            "review_facts_file": REVIEW_FACTS_FILE,
            "completed_units": 1,
            "total_units": 1,
            "historical_results_modified": False,
            "model_or_training_activity": False,
            "raw_or_protected_data_touched": False,
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
