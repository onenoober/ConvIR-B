#!/usr/bin/env python3
"""Run the authorized ITS measurement supplement with physical haze mapping."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

import calibrate_reside_its_local_measurement as base
from route_program_api import (
    asset_path,
    load_context,
    output_file,
    prepare_phase_output,
    write_contract_result,
    write_run_result,
    write_workload_progress,
)


PAIRING_GAP_MINIMUM = 1e-8


def select_paired_haze(
    clear: np.ndarray,
    transmission: np.ndarray,
    haze_candidates: dict[str, np.ndarray],
) -> tuple[str, np.ndarray, float, float]:
    scored = sorted(
        (
            base.reconstruction_rmse(haze, clear, transmission),
            stem,
            haze,
        )
        for stem, haze in haze_candidates.items()
    )
    if len(scored) != 10:
        raise ValueError(f"expected ten haze candidates, observed {len(scored)}")
    best_rmse, best_stem, best_haze = scored[0]
    second_rmse = scored[1][0]
    if not (
        math.isfinite(best_rmse)
        and math.isfinite(second_rmse)
        and best_rmse + PAIRING_GAP_MINIMUM < second_rmse
    ):
        raise ValueError("physical haze/transmission mapping has no unique RMSE minimum")
    return best_stem, best_haze, best_rmse, second_rmse


def process_split_mapping(
    *,
    split: str,
    selected_ids: list[str],
    severity_records: dict[str, dict[str, dict[str, Any]]],
    clear_by_id: dict[str, Path],
    haze_by_scene: dict[str, list[Path]],
    size: int,
    progress_offset: int,
    context: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    available_ids = [identifier for identifier in selected_ids if identifier in severity_records]
    if available_ids:
        shift = base.PERMUTATION_SHIFT % len(available_ids)
        if shift == 0:
            shift = 1
        permuted = {
            identifier: available_ids[(index + shift) % len(available_ids)]
            for index, identifier in enumerate(available_ids)
        }
    else:
        permuted = {}

    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for index, identifier in enumerate(selected_ids, start=1):
        try:
            if identifier not in severity_records:
                raise ValueError("severity scan was unavailable")
            candidate_paths = haze_by_scene.get(identifier, [])
            if len(candidate_paths) != 10:
                raise ValueError(
                    f"expected ten same-scene haze candidates, observed {len(candidate_paths)}"
                )
            clear = base.load_rgb(clear_by_id[identifier], size)
            haze_candidates = {
                path.stem: base.load_rgb(path, size) for path in candidate_paths
            }
            fields = base.content_fields(clear)
            severity_results: dict[str, dict[str, Any]] = {}
            chosen_haze_stems: list[str] = []
            q_scene_pass = True
            identifiable_scene = True
            negative_control_wins = 0
            maximum_q_error = 0.0
            local_associations: dict[str, list[float]] = {
                "tau_luminance": [],
                "tau_gradient": [],
                "tau_saturation": [],
            }
            for severity in base.SEVERITIES:
                selected = severity_records[identifier][severity]
                transmission_path = selected["transmission_path"]
                transmission = base.load_transmission(transmission_path, size)
                mapped_stem, haze, true_rmse, second_rmse = select_paired_haze(
                    clear, transmission, haze_candidates,
                )
                chosen_haze_stems.append(mapped_stem)
                permutation_id = permuted[identifier]
                if permutation_id == identifier:
                    raise ValueError("negative-control permutation contains a fixed point")
                permuted_path = severity_records[permutation_id][severity]["transmission_path"]
                permuted_transmission = base.load_transmission(permuted_path, size)
                permuted_rmse = base.reconstruction_rmse(haze, clear, permuted_transmission)
                improvement = permuted_rmse - true_rmse
                negative_control_wins += int(improvement > base.NEGATIVE_CONTROL_MARGIN)
                tau = -np.log(transmission)
                cells = base.q_cells(haze, clear, tau)
                identifiable = cells["identifiable_regions"] >= 2
                identifiable_scene = identifiable_scene and identifiable
                q_scene_pass = (
                    q_scene_pass
                    and identifiable
                    and cells["q_pass"]
                    and cells["absolute_error_finite"]
                )
                maximum_q_error = max(maximum_q_error, cells["maximum_q_error"])
                correlations = {
                    "tau_luminance": base.local_correlation(tau, fields["luminance"]),
                    "tau_gradient": base.local_correlation(tau, fields["gradient"]),
                    "tau_saturation": base.local_correlation(tau, fields["saturation"]),
                }
                for key, value in correlations.items():
                    if value is not None:
                        local_associations[key].append(value)
                severity_results[severity] = {
                    "tau_mean": float(selected["tau_mean"]),
                    "transmission_stem_digest": hashlib.sha256(
                        transmission_path.stem.encode()
                    ).hexdigest(),
                    "mapped_haze_stem_digest": hashlib.sha256(mapped_stem.encode()).hexdigest(),
                    "true_reconstruction_rmse": true_rmse,
                    "mapping_second_best_rmse": second_rmse,
                    "mapping_best_to_second_ratio": true_rmse / second_rmse,
                    "permuted_reconstruction_rmse": permuted_rmse,
                    "negative_control_improvement": improvement,
                    "negative_control_pass": improvement > base.NEGATIVE_CONTROL_MARGIN,
                    "identifiable_regions": cells["identifiable_regions"],
                    "low_denominator_regions": cells["low_denominator_regions"],
                    "q_pass": cells["q_pass"],
                    "maximum_q_error": cells["maximum_q_error"],
                }
            if len(set(chosen_haze_stems)) != len(base.SEVERITIES):
                raise ValueError("three selected transmission fields did not map to distinct haze images")
            tau_range = (
                severity_results["heavy"]["tau_mean"]
                - severity_results["light"]["tau_mean"]
            )
            negative_control_pass = negative_control_wins >= 2
            results.append({
                "scene_id": identifier,
                "canonical_clear_digest": base.canonical_clear_digest(clear_by_id[identifier]),
                "severity": severity_results,
                "tau_range": tau_range,
                "severity_coverage": tau_range >= base.TAU_RANGE_MINIMUM,
                "identifiable_scene": identifiable_scene,
                "q_scene_pass": q_scene_pass,
                "negative_control_pass": negative_control_pass,
                "joint_qualification": (
                    identifiable_scene and q_scene_pass and negative_control_pass
                ),
                "maximum_q_error": maximum_q_error,
                "mean_luminance": fields["mean_luminance"],
                "mean_gradient": fields["mean_gradient"],
                "mean_saturation": fields["mean_saturation"],
                "local_associations": {
                    key: (float(np.mean(values)) if values else None)
                    for key, values in local_associations.items()
                },
            })
        except (OSError, ValueError) as exc:
            failures.append({"scene_id": identifier, "reason": str(exc)[:256]})
        completed = progress_offset + index
        if index == 1 or index % 25 == 0 or index == len(selected_ids):
            write_workload_progress(
                context,
                completed_units=completed,
                stage=f"{split}_physical_mapping_measurement",
            )
    return results, failures


def mapping_summary(split_results: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    ratios: dict[str, list[float]] = {"definition": [], "validation": []}
    best_rmses: dict[str, list[float]] = {"definition": [], "validation": []}
    mapping_records: dict[str, list[str]] = {"definition": [], "validation": []}
    for split, results in split_results.items():
        for item in results:
            for severity in base.SEVERITIES:
                value = item["severity"][severity]
                ratios[split].append(value["mapping_best_to_second_ratio"])
                best_rmses[split].append(value["true_reconstruction_rmse"])
                mapping_records[split].append(
                    "|".join([
                        item["scene_id"], severity,
                        value["transmission_stem_digest"],
                        value["mapped_haze_stem_digest"],
                    ])
                )
    return {
        split: {
            "mapped_scene_count": len(split_results[split]),
            "mapped_variant_count": 3 * len(split_results[split]),
            "mapping_digest": base.digest_lines(mapping_records[split]),
            "best_to_second_ratio": base.quantiles(ratios[split]),
            "best_reconstruction_rmse": base.quantiles(best_rmses[split]),
        }
        for split in ("definition", "validation")
    }


def contract(context_path: Path) -> None:
    context = load_context(context_path, "contract")
    prepare_phase_output(context)
    qualification = context.assets.get("reside_qualification")
    prior_closeout = context.assets.get("prior_inconclusive_closeout")
    checks = {
        "metadata_only_mode": context.engineering_contract["mode"] == "metadata_only",
        "cpu_contract": context.device == "cpu",
        "dataset_hidden_from_contract": "reside_root" not in context.assets,
        "qualification_identity_bound": (
            qualification is not None and qualification.contract_access is True
        ),
        "prior_inconclusive_identity_bound": (
            prior_closeout is not None and prior_closeout.contract_access is True
        ),
        "protected_roles_disabled": not any(context.protected_data_permissions.values()),
        "mapping_only_supplement": True,
        "no_model_or_training_path": True,
        "output_and_finalizer_contract": True,
    }
    write_contract_result(
        context,
        checks=checks,
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
    size = int(os.environ.get("CONVIR_ROUTE_ITS_IMAGE_SIZE", "96"))
    if size != 96:
        raise ValueError("ITS image size differs from the frozen 96-pixel contract")
    reside = asset_path(context, "reside_root", kind="directory")
    qualification_path = asset_path(context, "reside_qualification", kind="file")
    prior_closeout_path = asset_path(
        context, "prior_inconclusive_closeout", kind="file",
    )
    qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
    prior_closeout = json.loads(prior_closeout_path.read_text(encoding="utf-8"))
    if qualification.get("marker") != "RESIDE_MINIMAL_MEASUREMENT_QUALIFICATION_RECORDED":
        raise ValueError("RESIDE qualification marker is missing")
    if (
        qualification.get("decision", {}).get("its_training_role")
        != "ELIGIBLE_FOR_SCENE_LEVEL_SUBSAMPLING"
    ):
        raise ValueError("RESIDE qualification does not authorize ITS scene sampling")
    expected_prior_terminal = {
        "route_id": "reside-its-local-measurement-v1",
        "operation_id": "ITS_CALIBRATE",
        "state": "COMPLETED_INCONCLUSIVE",
        "decision": "ITS_LOCAL_MEASUREMENT_INCONCLUSIVE",
        "authorizes": "MEASUREMENT_SUPPLEMENT_ONLY",
    }
    if any(
        prior_closeout.get(key) != value
        for key, value in expected_prior_terminal.items()
    ):
        raise ValueError("prior ITS terminal does not authorize this mapping supplement")
    dataset_identity = qualification["dataset_identity"]
    identity_files = {
        "archive_manifest_sha256": reside / "ARCHIVE_SHA256SUMS.txt",
        "pairing_report_sha256": reside / "PAIRING_VALIDATION.txt",
        "layout_record_sha256": reside / "DATASET_LAYOUT.txt",
    }
    observed_identity = {
        key: base.sha256_file(path) for key, path in identity_files.items()
    }
    if any(observed_identity[key] != dataset_identity[key] for key in observed_identity):
        raise ValueError("RESIDE dataset identity differs from the qualified snapshot")

    directories = {
        "train_clear": reside / "official/ITS/train/ITS_clear",
        "train_haze": reside / "official/ITS/train/ITS_haze",
        "train_transmission": reside / "official/ITS/train/ITS_trans",
        "validation_clear": reside / "official/ITS/val/clear",
        "validation_haze": reside / "official/ITS/val/haze",
        "validation_transmission": reside / "official/ITS/val/trans",
    }
    files = {key: base.image_files(path) for key, path in directories.items()}
    observed_counts = {
        "train_clear": len(files["train_clear"]),
        "train_variants": len(files["train_haze"]),
        "train_transmissions": len(files["train_transmission"]),
        "validation_clear": len(files["validation_clear"]),
        "validation_variants": len(files["validation_haze"]),
        "validation_transmissions": len(files["validation_transmission"]),
    }
    count_integrity = (
        observed_counts["train_clear"] == base.EXPECTED_COUNTS["train_clear"]
        and observed_counts["train_variants"] == base.EXPECTED_COUNTS["train_variants"]
        and observed_counts["train_transmissions"] == base.EXPECTED_COUNTS["train_variants"]
        and observed_counts["validation_clear"] == base.EXPECTED_COUNTS["validation_clear"]
        and observed_counts["validation_variants"] == base.EXPECTED_COUNTS["validation_variants"]
        and observed_counts["validation_transmissions"] == base.EXPECTED_COUNTS["validation_variants"]
    )
    clear = {
        "definition": {path.stem: path for path in files["train_clear"]},
        "validation": {path.stem: path for path in files["validation_clear"]},
    }
    haze_by_scene = {
        "definition": base.grouped_by_scene(files["train_haze"]),
        "validation": base.grouped_by_scene(files["validation_haze"]),
    }
    transmissions = {
        "definition": base.grouped_by_scene(files["train_transmission"]),
        "validation": base.grouped_by_scene(files["validation_transmission"]),
    }
    mapping_integrity = all(
        set(clear[split]) == set(haze_by_scene[split]) == set(transmissions[split])
        and all(len(haze_by_scene[split][identifier]) == 10 for identifier in clear[split])
        and all(len(transmissions[split][identifier]) == 10 for identifier in clear[split])
        for split in ("definition", "validation")
    )

    definition_order = base.hashed_scene_order(clear["definition"], "definition")[:1000]
    validation_order = base.hashed_scene_order(clear["validation"], "validation")
    scans: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    scan_failures: dict[str, list[dict[str, str]]] = {}
    scans["definition"], scan_failures["definition"] = base.scan_split(
        split="definition",
        selected_ids=definition_order,
        transmissions=transmissions["definition"],
        size=size,
        progress_offset=0,
        context=context,
    )
    scans["validation"], scan_failures["validation"] = base.scan_split(
        split="validation",
        selected_ids=validation_order,
        transmissions=transmissions["validation"],
        size=size,
        progress_offset=1000,
        context=context,
    )

    split_results: dict[str, list[dict[str, Any]]] = {}
    process_failures: dict[str, list[dict[str, str]]] = {}
    split_results["definition"], process_failures["definition"] = process_split_mapping(
        split="definition",
        selected_ids=definition_order,
        severity_records=scans["definition"],
        clear_by_id=clear["definition"],
        haze_by_scene=haze_by_scene["definition"],
        size=size,
        progress_offset=2000,
        context=context,
    )
    split_results["validation"], process_failures["validation"] = process_split_mapping(
        split="validation",
        selected_ids=validation_order,
        severity_records=scans["validation"],
        clear_by_id=clear["validation"],
        haze_by_scene=haze_by_scene["validation"],
        size=size,
        progress_offset=3000,
        context=context,
    )

    aggregates = {
        "definition": base.aggregate_split(split_results["definition"], 1000),
        "validation": base.aggregate_split(split_results["validation"], 1000),
    }
    definition_digests = {
        item["canonical_clear_digest"] for item in split_results["definition"]
    }
    validation_digests = {
        item["canonical_clear_digest"] for item in split_results["validation"]
    }
    overlap = definition_digests & validation_digests
    state, decision, authorizes, gate_reasons = base.terminal_from_gates(
        aggregates["validation"], len(overlap),
    )
    decision_by_state = {
        "COMPLETED_GATE_PASS": "ITS_LOCAL_MEASUREMENT_MAPPING_PASS",
        "COMPLETED_GATE_FAIL": "ITS_LOCAL_MEASUREMENT_MAPPING_FAIL",
        "COMPLETED_INCONCLUSIVE": "ITS_LOCAL_MEASUREMENT_MAPPING_INCONCLUSIVE",
    }
    decision = decision_by_state[state]
    if state == "COMPLETED_INCONCLUSIVE":
        authorizes = "NONE"
    if not count_integrity or not mapping_integrity:
        state = "COMPLETED_INCONCLUSIVE"
        decision = "ITS_LOCAL_MEASUREMENT_MAPPING_INCONCLUSIVE"
        authorizes = "NONE"
        gate_reasons = [
            "qualified ITS counts or same-scene candidate multiplicities changed"
        ]

    failures = {
        split: (scan_failures[split] + process_failures[split])[:20]
        for split in ("definition", "validation")
    }
    summary = {
        "schema_version": 1,
        "route_id": context.route_id,
        "operation_id": context.operation_id,
        "run_id": context.run_id,
        "measurement_scope": "synthetic indoor ITS only",
        "supplement_scope": "haze/transmission physical mapping only",
        "dataset_identity": {
            "root": str(reside),
            "observed_identity": observed_identity,
            "qualified_identity_match": True,
            "observed_counts": observed_counts,
            "count_integrity": count_integrity,
            "mapping_candidate_integrity": mapping_integrity,
        },
        "sampling": {
            "definition_available_scenes": len(clear["definition"]),
            "definition_selected_scenes": len(definition_order),
            "validation_available_scenes": len(clear["validation"]),
            "validation_selected_scenes": len(validation_order),
            "variants_per_scene": 3,
            "severity_selection": (
                "rank ten paired transmission fields by mean "
                "tau=-log(clip(t,1/255,1)); choose first, index four, and last"
            ),
            "definition_scene_digest": base.digest_lines(definition_order),
            "validation_scene_digest": base.digest_lines(validation_order),
            "selection_salt_sha256": hashlib.sha256(base.SAMPLE_SALT.encode()).hexdigest(),
            "permutation_shift": base.PERMUTATION_SHIFT,
            "effective_independent_unit": "clear_scene",
            "effective_independent_scene_count": (
                len(split_results["definition"]) + len(split_results["validation"])
            ),
            "nested_pair_count": 3 * (
                len(split_results["definition"]) + len(split_results["validation"])
            ),
        },
        "physical_mapping": {
            "method": (
                "For each selected transmission and ten same-scene haze candidates, "
                "fit constant RGB airlight under I=J*t+A*(1-t) at 96x96 and choose "
                "the unique minimum reconstruction RMSE."
            ),
            "unique_minimum_gap": PAIRING_GAP_MINIMUM,
            "aggregates": mapping_summary(split_results),
        },
        "measurement": {
            "tau": "-log(clip(t,1/255,1)); synthetic optical thickness only",
            "q": "<Y-J,I-J>/(||I-J||^2+1e-12)",
            "controlled_alpha_grid": list(base.ALPHAS),
            "q_tolerance": base.Q_TOLERANCE,
            "q_denominator_mse_minimum": base.Q_DENOMINATOR_MSE_MINIMUM,
            "negative_control_margin": base.NEGATIVE_CONTROL_MARGIN,
            "near_clear_rule": (
                "q is not used for low-denominator cells; finite absolute error "
                "is recorded instead"
            ),
        },
        "aggregates": aggregates,
        "split_overlap": {
            "canonical_overlap_count": len(overlap),
            "overlap_digest": base.digest_lines(overlap),
        },
        "failures": failures,
        "failure_counts": {
            split: len(scan_failures[split]) + len(process_failures[split])
            for split in ("definition", "validation")
        },
        "gates": {
            "validation_scene_count": aggregates["validation"]["completed_scene_count"] == 1000,
            "physical_mapping_coverage": (
                aggregates["validation"]["completed_scene_count"] == 1000
                and mapping_summary(split_results)["validation"]["mapped_variant_count"]
                == 3000
            ),
            "split_overlap_zero": len(overlap) == 0,
            "severity_coverage": aggregates["validation"]["severity_coverage"]["lower"] >= 0.9,
            "identifiable_region_coverage": (
                aggregates["validation"]["identifiable_scene_coverage"]["lower"] >= 0.9
            ),
            "controlled_q_recovery": (
                aggregates["validation"]["controlled_q_recovery"]["lower"] >= 0.99
            ),
            "paired_transmission_negative_control": (
                aggregates["validation"]["negative_control_separation"]["lower"] >= 0.85
            ),
            "primary_scene_qualification": (
                aggregates["validation"]["joint_qualification"]["lower"] >= 0.85
            ),
        },
        "terminal": {
            "state": state,
            "decision": decision,
            "authorizes": authorizes,
            "gate_reasons": gate_reasons,
        },
        "limitations": [
            "Passing supports an L1 synthetic-indoor measurement/mechanism-feasibility claim only.",
            "Controlled q recovery validates direction and implementation but is algebraic, not causal evidence.",
            "Transmission is synthetic optical thickness and is not a validated real-haze restoration-demand label.",
            "Brightness, gradient, and saturation associations are descriptive and may indicate construct confounding.",
            "Outdoor and real-haze calibration remain mandatory independent stages; no ITS result is transported to them.",
            "No model, insertion site, loss, hyperparameter, training run, inference, SOTS, OTS, Haze4K test, or NH-HAZE access occurred.",
        ],
        "marker": "RESIDE_ITS_LOCAL_MEASUREMENT_MAPPING_SUPPLEMENT_COMPLETE",
    }
    output_file(context, "measurement_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    base.write_strata_csv(
        output_file(context, "aggregate_strata.csv"), split_results,
    )
    write_run_result(
        context,
        state=state,
        decision=decision,
        authorizes=authorizes,
        details={
            "summary_file": "measurement_summary.json",
            "validation_scenes": len(split_results["validation"]),
            "joint_qualification_lower_95": (
                aggregates["validation"]["joint_qualification"]["lower"]
            ),
            "gate_reasons": gate_reasons,
            "physical_mapping_applied": True,
            "outdoor_calibration_required": True,
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
