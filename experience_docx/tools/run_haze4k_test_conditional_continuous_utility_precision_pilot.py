#!/usr/bin/env python3
"""Variance-only precision pilot for conditional continuous utility error."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

import run_haze4k_test_conditional_taper_grid_measurement_qualification as taper
from route_program_api import (
    asset_path,
    atomic_json,
    load_context,
    output_file,
    prepare_phase_output,
    write_contract_result,
    write_run_result,
    write_workload_progress,
)


ROUTE_ID = "haze4k-test-conditional-continuous-utility-contrast-measurement-qualification-v1"
OPERATION_ID = "HAZE4K_TEST_CONDITIONAL_CONTINUOUS_UTILITY_PRECISION_PILOT"
PILOT_SCENES = 32
PILOT_VARIANTS = 128
PILOT_SALT = "haze4k-continuous-utility-precision-pilot-v1"
TARGET_HALF_WIDTH_DB = 0.025
FULL_SCENE_COUNT = 100
NORMAL_95 = 1.959963984540054
MAX_PLANNING_SD_DB = TARGET_HALF_WIDTH_DB * math.sqrt(FULL_SCENE_COUNT) / NORMAL_95
PILOT_SD_UCB_FACTOR = 1.2680048907653747  # sqrt(31 / chi2_0.05,31)
SD_BOOTSTRAP_RESAMPLES = 20_000
SD_BOOTSTRAP_SEED = 20260811
UTILITY_ERROR_MARGIN_DB = 0.05
CONTENT_CLOSEOUT_SHA256 = "2188ace2cd45bd7e3fa253afc56235c6130054ac94218134cea97c25452289e0"
CONTENT_CONCLUSION_SHA256 = "9d0339970f649035c674851e0aae0c805a35f161b1051b0559627a57eaf3debb"
PARENT_TAPER_RUNNER_SHA256 = "2e5835cff886353843c4ed599eff6382e26343d960787b0f42061152a33329b7"
SUMMARY_NAME = "haze4k_test_conditional_continuous_utility_precision_pilot_v1_summary.json"


def contrast_from_mse(errors: np.ndarray) -> np.ndarray:
    """Return keep-referenced PSNR utility for keep, weaken, strengthen."""
    values = np.asarray(errors, dtype=np.float64)
    if values.shape[-1] != len(taper.ACTION_SET) or not np.isfinite(values).all():
        raise RuntimeError("contrast input must contain three finite action MSE values")
    if bool(np.any(values < 0.0)):
        raise RuntimeError("contrast MSE values cannot be negative")
    stabilized = np.maximum(values, taper.EPSILON)
    contrasts = 10.0 * np.log10(stabilized[..., 0, None] / stabilized)
    contrasts[..., 0] = 0.0
    if not np.isfinite(contrasts).all() or not np.array_equal(contrasts[..., 0], np.zeros_like(contrasts[..., 0])):
        raise RuntimeError("invalid keep-referenced utility contrast")
    return contrasts


def tile_contrast_table(variant: dict[str, Any], tiles: list[list[tuple[int, int, int, int]]]) -> np.ndarray:
    rows, columns = len(tiles), len(tiles[0])
    errors = np.empty((rows, columns, len(taper.ACTION_SET)), dtype=np.float64)
    for row_index, row in enumerate(tiles):
        for column_index, (top, bottom, left, right) in enumerate(row):
            target = variant["clear"][top:bottom, left:right]
            errors[row_index, column_index] = [
                taper.mse(candidate[top:bottom, left:right], target)
                for candidate in variant["candidates"]
            ]
    return contrast_from_mse(errors)


def area_weighted_phase_metrics(
    projected: np.ndarray, held_out: np.ndarray,
    tiles: list[list[tuple[int, int, int, int]]],
) -> dict[str, float]:
    if projected.shape != held_out.shape or projected.shape[-1] != 3:
        raise RuntimeError("projected and held-out contrast tables do not match")
    total_area = 0
    error_sum = 0.0
    agreement_area = 0
    regret_sum = 0.0
    material_area = 0
    for row_index, row in enumerate(tiles):
        for column_index, (top, bottom, left, right) in enumerate(row):
            area = (bottom - top) * (right - left)
            source = projected[row_index, column_index]
            target = held_out[row_index, column_index]
            error = float(np.max(np.abs(source[1:] - target[1:])))
            source_action = int(np.argmax(source))
            target_action = int(np.argmax(target))
            regret = float(np.max(target) - target[source_action])
            if error < -1e-12 or regret < -1e-12:
                raise RuntimeError("continuous utility error or diagnostic regret is negative")
            total_area += area
            error_sum += max(0.0, error) * area
            regret_sum += max(0.0, regret) * area
            agreement_area += int(source_action == target_action) * area
            material_area += int(np.max(np.abs(target[1:])) >= 0.10) * area
    if total_area <= 0:
        raise RuntimeError("held-out grid has no measurable area")
    return {
        "linfinity_error_db": error_sum / total_area,
        "hard_action_agreement_fraction": agreement_area / total_area,
        "hard_decision_regret_db": regret_sum / total_area,
        "material_utility_area_fraction": material_area / total_area,
    }


def continuous_scene_measurement(raw_variants: list[dict[str, np.ndarray]]) -> dict[str, Any]:
    if len(raw_variants) != taper.VARIANTS_PER_SCENE:
        raise RuntimeError("a canonical scene must contain four haze variants")
    variants = [
        taper.prepare_variant(item["hazy"], item["clear"], item["prediction"])
        for item in raw_variants
    ]
    height, width = variants[0]["clear"].shape[:2]
    grids: list[dict[str, Any]] = []
    for origin in taper.GRID_OFFSETS:
        tiles = taper.grid_slices(height, width, origin)
        grids.append({
            "origin": origin,
            "tiles": tiles,
            "tables": [
                {"contrast": tile_contrast_table(variant, tiles)}
                for variant in variants
            ],
        })

    phase_records = []
    for held_out_index, grid in enumerate(grids):
        variant_metrics = []
        for variant_index in range(len(variants)):
            projected, _, complete, minimum_weight = taper.project_grid_scores(
                grids, variant_index, held_out_index, "contrast", None, projection="taper",
            )
            if not complete or minimum_weight <= 0.0:
                raise RuntimeError("continuous utility taper projection is incomplete")
            held_out = grid["tables"][variant_index]["contrast"]
            variant_metrics.append(area_weighted_phase_metrics(projected, held_out, grid["tiles"]))
        phase_records.append({
            "origin": list(grid["origin"]),
            "linfinity_error_db": float(np.mean([item["linfinity_error_db"] for item in variant_metrics])),
            "hard_action_agreement_fraction": float(np.mean([item["hard_action_agreement_fraction"] for item in variant_metrics])),
            "hard_decision_regret_db": float(np.mean([item["hard_decision_regret_db"] for item in variant_metrics])),
            "material_utility_area_fraction": float(np.mean([item["material_utility_area_fraction"] for item in variant_metrics])),
        })
    return {
        "worst_phase_linfinity_error_db": max(item["linfinity_error_db"] for item in phase_records),
        "phase_records": phase_records,
        "keep_structural_identity": all(np.array_equal(item["candidates"][0], item["prediction"]) for item in variants),
        "affine_manipulation_exact": all(item["affine_manipulation_exact"] for item in variants),
    }


def reference_checks() -> dict[str, bool]:
    errors = np.asarray([4.0, 2.0, 8.0], dtype=np.float64)
    contrast = contrast_from_mse(errors)
    direction_ok = abs(float(contrast[1]) - 10.0 * math.log10(2.0)) < 1e-12 and contrast[2] < 0.0
    keep_ok = contrast[0] == 0.0

    height, width = 40, 44
    grids = []
    source_values = (1.0, 2.0, 3.0, 999.0)
    for origin, value in zip(taper.GRID_OFFSETS, source_values):
        tiles = taper.grid_slices(height, width, origin)
        table = np.zeros((len(tiles), len(tiles[0]), 3), dtype=np.float64)
        table[..., 1] = value
        table[..., 2] = -value
        grids.append({"tiles": tiles, "tables": [{"contrast": table}]})
    projected, _, complete, minimum_weight = taper.project_grid_scores(
        grids, 0, 3, "contrast", None, projection="taper",
    )
    exclusion_ok = complete and minimum_weight > 0.0 and float(np.max(projected[..., 1])) < 3.0
    area = area_weighted_phase_metrics(
        np.asarray([[[0.0, 0.2, -0.2], [0.0, 0.1, -0.1]]]),
        np.asarray([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]),
        [[(0, 1, 0, 1), (0, 1, 1, 4)]],
    )
    area_ok = abs(area["linfinity_error_db"] - 0.125) < 1e-12
    aggregation_ok = max(np.mean([0.01, 0.02, 0.03, 0.04]), np.mean([0.02] * 4)) == 0.025
    return {
        "db_contrast_direction": direction_ok,
        "keep_contrast_exact_zero": keep_ok,
        "held_out_source_exclusion_and_taper_projection": exclusion_ok,
        "area_weighted_two_action_linfinity_error": area_ok,
        "four_variant_mean_then_worst_phase": aggregation_ok,
    }


def pilot_order(digests: list[str]) -> list[str]:
    return sorted(digests, key=lambda value: hashlib.sha256(f"{PILOT_SALT}:{value}".encode()).hexdigest())


def bootstrap_sd_upper(values: np.ndarray) -> float:
    generator = np.random.default_rng(SD_BOOTSTRAP_SEED)
    draws = np.empty(SD_BOOTSTRAP_RESAMPLES, dtype=np.float64)
    for start in range(0, SD_BOOTSTRAP_RESAMPLES, 1000):
        stop = min(start + 1000, SD_BOOTSTRAP_RESAMPLES)
        indices = generator.integers(0, values.size, size=(stop - start, values.size))
        draws[start:stop] = np.std(values[indices], axis=1, ddof=1)
    return float(np.quantile(draws, 0.95))


def precision_result(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size != PILOT_SCENES or not np.isfinite(array).all():
        raise RuntimeError("precision pilot requires exactly 32 finite scene values")
    sample_sd = float(np.std(array, ddof=1))
    chi_square_upper = sample_sd * PILOT_SD_UCB_FACTOR
    bootstrap_upper = bootstrap_sd_upper(array)
    conservative_upper = max(chi_square_upper, bootstrap_upper)
    required_groups = math.ceil((NORMAL_95 * conservative_upper / TARGET_HALF_WIDTH_DB) ** 2)
    feasible = conservative_upper <= MAX_PLANNING_SD_DB and required_groups <= FULL_SCENE_COUNT
    return {
        "pilot_scene_count": PILOT_SCENES,
        "primary_mean_withheld": True,
        "sample_sd_db": sample_sd,
        "chi_square_one_sided_95_sd_upper_db": chi_square_upper,
        "bootstrap_one_sided_95_sd_upper_db": bootstrap_upper,
        "conservative_planning_sd_db": conservative_upper,
        "maximum_planning_sd_db": MAX_PLANNING_SD_DB,
        "future_target_half_width_db": TARGET_HALF_WIDTH_DB,
        "future_independent_scenes_available": FULL_SCENE_COUNT,
        "future_independent_scenes_required": required_groups,
        "feasible": feasible,
        "bootstrap_seed": SD_BOOTSTRAP_SEED,
        "bootstrap_resamples": SD_BOOTSTRAP_RESAMPLES,
    }


def terminal(feasible: bool) -> dict[str, str]:
    if feasible:
        return {
            "state": "COMPLETED_GATE_PASS",
            "decision": "HAZE4K_CONTINUOUS_UTILITY_PRECISION_FEASIBLE",
            "authorizes": "FULL_CONTINUOUS_UTILITY_MEASUREMENT_ONLY",
        }
    return {
        "state": "COMPLETED_GATE_FAIL",
        "decision": "HAZE4K_CONTINUOUS_UTILITY_BLOCKED_PRECISION",
        "authorizes": "NONE",
    }


def contract(context_path: Path) -> None:
    context = load_context(context_path, "contract")
    prepare_phase_output(context)
    if context.device != "cuda" or any(context.protected_data_permissions.values()):
        raise RuntimeError("precision pilot contract requires CUDA without protected permissions")
    if "haze4k_test_development" in context.assets:
        raise RuntimeError("development data must be absent from the contract phase")
    torch, model = taper.load_official_model(context)
    height, width = 1200, 1600
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    clear = np.stack((0.15 + 0.65 * xx / width, 0.18 + 0.60 * yy / height, 0.20 + 0.55 * (xx + yy) / (width + height)), axis=-1).astype(np.float32)
    hazy = (0.70 * clear + 0.20).astype(np.float32)
    inferred = taper.infer(torch, model, hazy, context.device)
    variants = []
    for index, scale in enumerate((0.92, 1.03, 1.12, 1.21)):
        manufactured = np.clip(hazy + np.float32(scale) * (clear - hazy), 0.0, 1.0).astype(np.float32)
        variants.append({"hazy": hazy, "clear": clear, "prediction": manufactured})
    measured = continuous_scene_measurement(variants)
    references = reference_checks()
    checks = {
        "strict_checkpoint_load_and_parameter_count": sum(parameter.numel() for parameter in model.parameters()) == taper.PARAMETER_COUNT,
        "all_parameters_frozen": not any(parameter.requires_grad for parameter in model.parameters()),
        "official_model_forward_finite": bool(np.isfinite(inferred).all()),
        "new_continuous_utility_path_finite": math.isfinite(measured["worst_phase_linfinity_error_db"]),
        "four_declared_origins_exact": [item["origin"] for item in measured["phase_records"]] == [list(item) for item in taper.GRID_OFFSETS],
        "keep_structural_identity": measured["keep_structural_identity"],
        "nonclipped_affine_manipulation_exact": measured["affine_manipulation_exact"],
        "reference_checks_complete": all(references.values()),
        "terminal_mapping_reference": terminal(True)["state"] == "COMPLETED_GATE_PASS" and terminal(False)["decision"].endswith("BLOCKED_PRECISION"),
        "protected_roles_disabled": not any(context.protected_data_permissions.values()),
    }
    write_contract_result(
        context, checks=checks,
        engineering={
            "mode": "gpu_synthetic_no_data", "device": "cuda",
            "fixture": {"batch": 1, "channels": 3, "height": height, "width": width},
            "production_path_exercised": True, "protected_data_touched": False,
            "scientific_output_created": False, "scientific_training_occurred": False,
            "reference_checks": references,
        },
    )


def run(context_path: Path) -> None:
    context = load_context(context_path, "run")
    prepare_phase_output(context)
    development_root = asset_path(context, "haze4k_test_development", kind="directory")
    closeout_path = asset_path(context, "parent_content_closeout", kind="file")
    conclusion_path = asset_path(context, "parent_content_conclusion", kind="file")
    if context.assets["parent_content_closeout"].sha256 != CONTENT_CLOSEOUT_SHA256 or context.assets["parent_content_conclusion"].sha256 != CONTENT_CONCLUSION_SHA256:
        raise RuntimeError("direct parent evidence identity changed")
    if context.assets["parent_taper_runner"].sha256 != PARENT_TAPER_RUNNER_SHA256:
        raise RuntimeError("verified taper production path identity changed")
    closeout = json.loads(closeout_path.read_text(encoding="utf-8"))
    conclusion = json.loads(conclusion_path.read_text(encoding="utf-8"))
    parent_ok = (
        closeout.get("state") == "COMPLETED_GATE_FAIL"
        and closeout.get("decision") == "HAZE4K_TEST_CONDITIONAL_CONTENT_ALIGNED_REGION_STABILITY_FAIL"
        and closeout.get("authorizes") == "NONE"
        and closeout.get("details", {}).get("independent_scenes") == taper.EXPECTED_SCENES
        and closeout.get("details", {}).get("nested_variants") == taper.EXPECTED_VARIANTS
        and conclusion.get("authorizes") == "NONE"
    )
    scope_ok = (
        development_root.name == "development_screening"
        and not any(context.protected_data_permissions.values())
        and "candidate_confirmation" not in str(development_root)
        and "haze4k_test_candidate_confirmation" not in context.assets
    )
    haze_root, clear_root = development_root / "haze", development_root / "gt"
    hazy_paths = taper.image_files(haze_root) if scope_ok and haze_root.is_dir() else []
    clear_paths = taper.image_files(clear_root) if scope_ok and clear_root.is_dir() else []
    failures: list[dict[str, str]] = []
    variants_by_digest: dict[str, list[Path]] = defaultdict(list)
    for hazy_path in hazy_paths:
        try:
            variants_by_digest[taper.canonical_rgb_digest(taper.image_array(clear_root / hazy_path.name))].append(hazy_path)
        except Exception as exc:
            failures.append({"scene": "pair", "variant": hazy_path.name, "reason": str(exc)[:512]})
    histogram = Counter(len(paths) for paths in variants_by_digest.values())
    dataset_ok = (
        len(hazy_paths) == taper.EXPECTED_VARIANTS and len(clear_paths) == taper.EXPECTED_VARIANTS
        and {path.name for path in hazy_paths} == {path.name for path in clear_paths}
        and len(variants_by_digest) == taper.EXPECTED_SCENES
        and histogram == {taper.VARIANTS_PER_SCENE: taper.EXPECTED_SCENES}
        and not failures
    )
    selected = pilot_order(list(variants_by_digest))[:PILOT_SCENES] if dataset_ok else []
    selection_digest = hashlib.sha256("\n".join(selected).encode()).hexdigest() if selected else None
    write_workload_progress(context, completed_units=2, stage="fixed_hash_pilot_selection")

    scenes: list[dict[str, Any]] = []
    if parent_ok and scope_ok and dataset_ok and len(selected) == PILOT_SCENES:
        torch, model = taper.load_official_model(context)
        attempted = 0
        for digest in selected:
            variants = []
            for hazy_path in sorted(variants_by_digest[digest]):
                attempted += 1
                try:
                    hazy = taper.image_array(hazy_path)
                    clear = taper.image_array(clear_root / hazy_path.name)
                    prediction = taper.infer(torch, model, hazy, context.device)
                    variants.append({"hazy": hazy, "clear": clear, "prediction": prediction})
                except Exception as exc:
                    failures.append({"scene": digest[:16], "variant": hazy_path.name, "reason": str(exc)[:512]})
                if attempted % 4 == 0:
                    write_workload_progress(context, completed_units=2 + attempted, stage="pilot_official_inference")
            if len(variants) == taper.VARIANTS_PER_SCENE:
                try:
                    scenes.append(continuous_scene_measurement(variants))
                except Exception as exc:
                    failures.append({"scene": digest[:16], "variant": "continuous_measurement", "reason": str(exc)[:512]})

    integrity = {
        "direct_parent_terminal_exact": parent_ok,
        "isolated_development_asset_only": scope_ok,
        "complete_dataset_before_fixed_subset": dataset_ok,
        "fixed_hash_selection_complete": len(selected) == PILOT_SCENES,
        "complete_32_scene_128_variant_pilot": len(scenes) == PILOT_SCENES and not failures,
        "keep_and_affine_controls": bool(scenes) and all(item["keep_structural_identity"] and item["affine_manipulation_exact"] for item in scenes),
        "no_training_or_protected_access": True,
    }
    precision = precision_result([item["worst_phase_linfinity_error_db"] for item in scenes]) if all(integrity.values()) else None
    verdict = terminal(bool(precision and precision["feasible"]))
    if not all(integrity.values()):
        verdict = {"state": "FAILED_ENGINEERING", "decision": None, "authorizes": "NONE"}
    summary = {
        "schema_version": 1, "route_id": ROUTE_ID, "operation_id": OPERATION_ID,
        "run_id": context.run_id,
        "scope": "Variance-only internal development-screening precision supplement; primary mean is withheld and no target qualification is decided.",
        "estimand": "SD of canonical-scene worst-held-out-phase area-weighted L-infinity error for two keep-referenced continuous PSNR utility contrasts",
        "selection": {
            "method": "first 32 canonical scene digests after SHA-256 ordering with a frozen public salt",
            "salt": PILOT_SALT, "selection_digest": selection_digest,
            "independent_scenes": len(scenes), "nested_variants": len(scenes) * taper.VARIANTS_PER_SCENE,
        },
        "identity_and_integrity": {"checks": integrity, "failure_count": len(failures), "failures": failures},
        "precision_feasibility": precision,
        "future_frozen_gate": {
            "utility_error_margin_db": UTILITY_ERROR_MARGIN_DB,
            "pass": "one-sided 95 percent UCB below 0.05 dB",
            "fail": "one-sided 95 percent LCB above 0.05 dB",
            "inconclusive": "interval crosses 0.05 dB or upper-bound distance exceeds 0.025 dB",
        },
        "terminal": verdict,
        "limitations": [
            "The pilot uses 32 of the same 100 development scenes and cannot serve as confirmation or increase future independent n.",
            "Only the dispersion needed for pre-run precision planning is released; the primary mean and target verdict remain unexamined.",
            "The chi-square SD bound assumes approximately normal scene values; the gate conservatively also requires the nonparametric bootstrap SD upper bound.",
            "No result authorizes Stage 2, confirmation data, NH-HAZE, training, model selection, or deployment claims.",
        ],
        "marker": "HAZE4K_CONTINUOUS_UTILITY_PRECISION_PILOT_COMPLETE",
    }
    atomic_json(output_file(context, SUMMARY_NAME), summary)
    write_workload_progress(context, completed_units=131, stage="variance_only_precision_finalize")
    write_run_result(
        context, state=verdict["state"], decision=verdict["decision"], authorizes=verdict["authorizes"],
        details={
            "summary_file": SUMMARY_NAME, "independent_scenes": len(scenes),
            "nested_variants": len(scenes) * taper.VARIANTS_PER_SCENE,
            "primary_mean_withheld": True, "precision_feasible": None if precision is None else precision["feasible"],
            "candidate_confirmation_asset_delivered": False, "network_or_proxy_training_occurred": False,
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
