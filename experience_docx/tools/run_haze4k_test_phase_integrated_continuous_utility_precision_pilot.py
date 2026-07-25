#!/usr/bin/env python3
"""Variance-only precision pilot for phase-integrated continuous utility error."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

import run_haze4k_test_conditional_continuous_utility_precision_pilot as parent
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


ROUTE_ID = "haze4k-test-phase-integrated-continuous-utility-measurement-v1"
OPERATION_ID = "HAZE4K_TEST_PHASE_INTEGRATED_CONTINUOUS_UTILITY_PRECISION_PILOT"
OUTPUT_ID = "haze4k-test-phase-integrated-continuous-utility-measurement-v1-precision-pilot-r1"
SUMMARY_NAME = "haze4k_test_phase_integrated_continuous_utility_precision_pilot_v1_summary.json"
PILOT_SALT = parent.PILOT_SALT
PARENT_CLOSEOUT_SHA256 = "23983919fb2f18aae8e621b80d5f87bd3502f8c3e1a8a3765adc43157cc5880a"
PARENT_CONCLUSION_SHA256 = "725f9fd7f5e47699febddfffa4a1311c5aeb539a4d7507f4ce041fc8f879d3b3"
PARENT_SUMMARY_SHA256 = "4c6cdb30b854be1d5a721cd9de1f2439368425d1647db16991a82f1ab4918302"
PARENT_TAPER_RUNNER_SHA256 = "2e5835cff886353843c4ed599eff6382e26343d960787b0f42061152a33329b7"
PARENT_CONTINUOUS_RUNNER_SHA256 = "46faa7ce0c75a38beb3c52f25c573e17ed1e6e55aca5b2ceea138ae81dbeb89d"
PARENT_SAMPLE_SD_DB = 0.3503289895035461
PARENT_SD_ABS_TOLERANCE_DB = 1e-9


def scene_measurement(raw_variants: list[dict[str, np.ndarray]]) -> dict[str, Any]:
    measured = parent.continuous_scene_measurement(raw_variants)
    phase_values = np.asarray(
        [item["linfinity_error_db"] for item in measured["phase_records"]],
        dtype=np.float64,
    )
    if phase_values.shape != (4,) or not np.isfinite(phase_values).all():
        raise RuntimeError("phase integration requires four finite held-out phase errors")
    measured["phase_integrated_linfinity_error_db"] = float(np.mean(phase_values))
    measured["within_scene_phase_sd_db"] = float(np.std(phase_values, ddof=1))
    return measured


def reference_checks() -> dict[str, bool]:
    inherited = parent.reference_checks()
    example = np.asarray([0.01, 0.02, 0.03, 0.04], dtype=np.float64)
    phase_mean_ok = abs(float(np.mean(example)) - 0.025) < 1e-15
    phase_permutation_ok = all(
        abs(float(np.mean(example[list(order)])) - 0.025) < 1e-15
        for order in ((3, 2, 1, 0), (1, 3, 0, 2))
    )
    return {
        **inherited,
        "four_phase_equal_weight_mean": phase_mean_ok,
        "phase_order_invariance": phase_permutation_ok,
    }


def precision_result(values: list[float]) -> dict[str, Any]:
    return parent.precision_result(values)


def terminal(feasible: bool) -> dict[str, str]:
    if feasible:
        return {
            "state": "COMPLETED_GATE_PASS",
            "decision": "HAZE4K_PHASE_INTEGRATED_CONTINUOUS_UTILITY_PRECISION_FEASIBLE",
            "authorizes": "FULL_PHASE_INTEGRATED_CONTINUOUS_UTILITY_MEASUREMENT_ONLY",
        }
    return {
        "state": "COMPLETED_GATE_FAIL",
        "decision": "HAZE4K_PHASE_INTEGRATED_CONTINUOUS_UTILITY_BLOCKED_PRECISION",
        "authorizes": "NONE",
    }


def contract(context_path: Path) -> None:
    context = load_context(context_path, "contract")
    prepare_phase_output(context)
    if context.device != "cuda" or any(context.protected_data_permissions.values()):
        raise RuntimeError("phase-integrated pilot requires CUDA without protected permissions")
    if "haze4k_test_development" in context.assets:
        raise RuntimeError("development data must be absent from the contract phase")
    torch, model = taper.load_official_model(context)
    height, width = 1200, 1600
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    clear = np.stack(
        (0.15 + 0.65 * xx / width, 0.18 + 0.60 * yy / height,
         0.20 + 0.55 * (xx + yy) / (width + height)), axis=-1,
    ).astype(np.float32)
    hazy = (0.70 * clear + 0.20).astype(np.float32)
    inferred = taper.infer(torch, model, hazy, context.device)
    variants = []
    for scale in (0.92, 1.03, 1.12, 1.21):
        manufactured = np.clip(
            hazy + np.float32(scale) * (clear - hazy), 0.0, 1.0,
        ).astype(np.float32)
        variants.append({"hazy": hazy, "clear": clear, "prediction": manufactured})
    measured = scene_measurement(variants)
    references = reference_checks()
    checks = {
        "strict_checkpoint_load_and_parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ) == taper.PARAMETER_COUNT,
        "all_parameters_frozen": not any(parameter.requires_grad for parameter in model.parameters()),
        "official_model_forward_finite": bool(np.isfinite(inferred).all()),
        "phase_integrated_path_finite": math.isfinite(
            measured["phase_integrated_linfinity_error_db"]
        ),
        "four_declared_origins_exact": [
            item["origin"] for item in measured["phase_records"]
        ] == [list(item) for item in taper.GRID_OFFSETS],
        "keep_structural_identity": measured["keep_structural_identity"],
        "nonclipped_affine_manipulation_exact": measured["affine_manipulation_exact"],
        "reference_checks_complete": all(references.values()),
        "phase_mean_not_worse_than_worst_reference": measured[
            "phase_integrated_linfinity_error_db"
        ] <= measured["worst_phase_linfinity_error_db"] + 1e-15,
        "terminal_mapping_reference": (
            terminal(True)["state"] == "COMPLETED_GATE_PASS"
            and terminal(False)["decision"].endswith("BLOCKED_PRECISION")
        ),
        "protected_roles_disabled": not any(context.protected_data_permissions.values()),
    }
    write_contract_result(
        context,
        checks=checks,
        engineering={
            "mode": "gpu_synthetic_no_data",
            "device": "cuda",
            "fixture": {"batch": 1, "channels": 3, "height": height, "width": width},
            "production_path_exercised": True,
            "protected_data_touched": False,
            "scientific_output_created": False,
            "scientific_training_occurred": False,
        },
    )


def run(context_path: Path) -> None:
    context = load_context(context_path, "run")
    prepare_phase_output(context)
    development_root = asset_path(context, "haze4k_test_development", kind="directory")
    parent_closeout_path = asset_path(context, "parent_precision_closeout", kind="file")
    parent_conclusion_path = asset_path(context, "parent_precision_conclusion", kind="file")
    parent_summary_path = asset_path(context, "parent_precision_summary", kind="file")
    expected_hashes = {
        "parent_precision_closeout": PARENT_CLOSEOUT_SHA256,
        "parent_precision_conclusion": PARENT_CONCLUSION_SHA256,
        "parent_precision_summary": PARENT_SUMMARY_SHA256,
        "parent_taper_runner": PARENT_TAPER_RUNNER_SHA256,
        "parent_continuous_runner": PARENT_CONTINUOUS_RUNNER_SHA256,
    }
    identities_ok = all(
        context.assets[name].sha256 == expected for name, expected in expected_hashes.items()
    )
    closeout = json.loads(parent_closeout_path.read_text(encoding="utf-8"))
    conclusion = json.loads(parent_conclusion_path.read_text(encoding="utf-8"))
    parent_summary = json.loads(parent_summary_path.read_text(encoding="utf-8"))
    parent_ok = (
        identities_ok
        and closeout.get("state") == "COMPLETED_GATE_FAIL"
        and closeout.get("decision") == "HAZE4K_CONTINUOUS_UTILITY_BLOCKED_PRECISION"
        and closeout.get("authorizes") == "NONE"
        and conclusion.get("authorizes") == "NONE"
        and parent_summary.get("precision_feasibility", {}).get("primary_mean_withheld") is True
        and abs(
            float(parent_summary["precision_feasibility"]["sample_sd_db"])
            - PARENT_SAMPLE_SD_DB
        ) <= PARENT_SD_ABS_TOLERANCE_DB
    )
    scope_ok = (
        development_root.name == "development_screening"
        and not any(context.protected_data_permissions.values())
        and "candidate_confirmation" not in str(development_root)
        and "haze4k_test_candidate_confirmation" not in context.assets
        and "haze4k_train" not in context.assets
    )
    haze_root, clear_root = development_root / "haze", development_root / "gt"
    hazy_paths = taper.image_files(haze_root) if scope_ok and haze_root.is_dir() else []
    clear_paths = taper.image_files(clear_root) if scope_ok and clear_root.is_dir() else []
    failures: list[dict[str, str]] = []
    variants_by_digest: dict[str, list[Path]] = defaultdict(list)
    for hazy_path in hazy_paths:
        try:
            digest = taper.canonical_rgb_digest(taper.image_array(clear_root / hazy_path.name))
            variants_by_digest[digest].append(hazy_path)
        except Exception as exc:
            failures.append({"scene": "pair", "variant": hazy_path.name, "reason": str(exc)[:512]})
    histogram = Counter(len(paths) for paths in variants_by_digest.values())
    dataset_ok = (
        len(hazy_paths) == taper.EXPECTED_VARIANTS
        and len(clear_paths) == taper.EXPECTED_VARIANTS
        and {path.name for path in hazy_paths} == {path.name for path in clear_paths}
        and len(variants_by_digest) == taper.EXPECTED_SCENES
        and histogram == {taper.VARIANTS_PER_SCENE: taper.EXPECTED_SCENES}
        and not failures
    )
    selected = parent.pilot_order(list(variants_by_digest))[:parent.PILOT_SCENES] if dataset_ok else []
    selection_digest = hashlib.sha256("\n".join(selected).encode()).hexdigest() if selected else None
    parent_selection_ok = selection_digest == parent_summary.get("selection", {}).get("selection_digest")
    write_workload_progress(context, completed_units=2, stage="fixed_hash_phase_integrated_selection")

    scenes: list[dict[str, Any]] = []
    if parent_ok and scope_ok and dataset_ok and parent_selection_ok:
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
                    failures.append({
                        "scene": digest[:16], "variant": hazy_path.name,
                        "reason": str(exc)[:512],
                    })
                if attempted % 4 == 0:
                    write_workload_progress(
                        context, completed_units=2 + attempted,
                        stage="phase_integrated_official_inference",
                    )
            if len(variants) == taper.VARIANTS_PER_SCENE:
                try:
                    scenes.append(scene_measurement(variants))
                except Exception as exc:
                    failures.append({
                        "scene": digest[:16], "variant": "phase_integrated_measurement",
                        "reason": str(exc)[:512],
                    })

    legacy_precision = None
    integrated_precision = None
    legacy_reproduced = False
    if len(scenes) == parent.PILOT_SCENES and not failures:
        legacy_precision = precision_result([
            item["worst_phase_linfinity_error_db"] for item in scenes
        ])
        integrated_precision = precision_result([
            item["phase_integrated_linfinity_error_db"] for item in scenes
        ])
        legacy_reproduced = abs(
            legacy_precision["sample_sd_db"] - PARENT_SAMPLE_SD_DB
        ) <= PARENT_SD_ABS_TOLERANCE_DB
    integrity = {
        "blocked_parent_terminal_exact": parent_ok,
        "isolated_development_asset_only": scope_ok,
        "complete_dataset_before_fixed_subset": dataset_ok,
        "same_fixed_hash_selection_as_parent": parent_selection_ok,
        "complete_32_scene_128_variant_pilot": len(scenes) == parent.PILOT_SCENES and not failures,
        "keep_and_affine_controls": bool(scenes) and all(
            item["keep_structural_identity"] and item["affine_manipulation_exact"]
            for item in scenes
        ),
        "legacy_worst_phase_sd_exactly_reproduced": legacy_reproduced,
        "no_training_or_protected_access": True,
    }
    verdict = terminal(bool(integrated_precision and integrated_precision["feasible"]))
    if not all(integrity.values()):
        verdict = {"state": "FAILED_ENGINEERING", "decision": None, "authorizes": "NONE"}
    variance_attribution = None
    if legacy_precision and integrated_precision:
        variance_attribution = {
            "legacy_worst_phase_sample_sd_db": legacy_precision["sample_sd_db"],
            "phase_integrated_sample_sd_db": integrated_precision["sample_sd_db"],
            "phase_integrated_to_worst_sample_variance_ratio": (
                integrated_precision["sample_sd_db"] / legacy_precision["sample_sd_db"]
            ) ** 2 if legacy_precision["sample_sd_db"] > 0.0 else None,
            "mean_within_scene_phase_sd_db": float(np.mean([
                item["within_scene_phase_sd_db"] for item in scenes
            ])),
            "primary_means_withheld": True,
        }
    summary = {
        "schema_version": 1,
        "route_id": ROUTE_ID,
        "operation_id": OPERATION_ID,
        "run_id": context.run_id,
        "scope": (
            "Variance-only development-screening precision supplement for a predeclared "
            "uniform-random grid-origin estimand; all primary means are withheld."
        ),
        "estimand": (
            "SD of canonical-scene equally phase-integrated area-weighted L-infinity "
            "error for two keep-referenced continuous PSNR utility contrasts"
        ),
        "selection": {
            "method": "same first 32 canonical scene digests under the frozen parent salt",
            "salt": PILOT_SALT,
            "selection_digest": selection_digest,
            "independent_scenes": len(scenes),
            "nested_variants": len(scenes) * taper.VARIANTS_PER_SCENE,
        },
        "identity_and_integrity": {
            "checks": integrity, "failure_count": len(failures), "failures": failures,
        },
        "phase_integrated_precision_feasibility": integrated_precision,
        "variance_attribution": variance_attribution,
        "future_frozen_gate": {
            "utility_error_margin_db": parent.UTILITY_ERROR_MARGIN_DB,
            "population": "uniform random draw over the four frozen grid origins within each unseen canonical scene",
            "pass": "one-sided 95 percent UCB below 0.05 dB",
            "fail": "one-sided 95 percent LCB above 0.05 dB",
            "inconclusive": "interval crosses 0.05 dB or upper-bound distance exceeds 0.025 dB",
        },
        "terminal": verdict,
        "limitations": [
            "The estimand applies only to the predeclared equal mixture of four grid origins; it does not bound a fixed or worst origin.",
            "The same 32 development scenes support paired variance diagnosis but are not confirmation and do not increase independent n.",
            "All primary means are withheld; feasibility cannot establish target stability.",
            "Training-overlap data, confirmation, NH-HAZE, canary, and locked test are not accessed or counted.",
            "No result authorizes Stage 2, training, model selection, or deployment.",
        ],
        "marker": "HAZE4K_PHASE_INTEGRATED_CONTINUOUS_UTILITY_PRECISION_PILOT_COMPLETE",
    }
    atomic_json(output_file(context, SUMMARY_NAME), summary)
    write_workload_progress(context, completed_units=131, stage="phase_integrated_variance_finalize")
    write_run_result(
        context,
        state=verdict["state"], decision=verdict["decision"], authorizes=verdict["authorizes"],
        details={
            "summary_file": SUMMARY_NAME,
            "independent_scenes": len(scenes),
            "nested_variants": len(scenes) * taper.VARIANTS_PER_SCENE,
            "primary_mean_withheld": True,
            "precision_feasible": None if integrated_precision is None else integrated_precision["feasible"],
            "training_asset_delivered": False,
            "candidate_confirmation_asset_delivered": False,
            "network_or_proxy_training_occurred": False,
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
