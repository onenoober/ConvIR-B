#!/usr/bin/env python3
"""Measure frozen continuous utility on quarantined ITS and OTS scene strata."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

import calibrate_reside_its_local_measurement as its_reader
import measure_reside_ots_baseline_behavior_supplement as ots_reader
import run_haze4k_test_conditional_continuous_utility_precision_pilot as utility
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


ROUTE_ID = "reside-known-overlap-continuous-utility-measurement-v1"
OPERATION_ID = "RESIDE_KNOWN_OVERLAP_CONTINUOUS_UTILITY_MEASURE"
SCENES_PER_STRATUM = 1587
VARIANTS_PER_SCENE = 4
TOTAL_INFERENCES = 2 * SCENES_PER_STRATUM * VARIANTS_PER_SCENE
TOTAL_UNITS = TOTAL_INFERENCES + 2
COST_PROBE_MAX_SECONDS = 840.0
COST_PROBE_MAX_CUDA_MIB = 8192.0
SELECTION_SALT = "reside-known-overlap-continuous-utility-measurement-v1"
LONG_EDGE = 320
MODEL_MIN_EDGE = 256
UTILITY_MARGIN_DB = 0.05
PRECISION_DISTANCE_DB = 0.025
SIMULTANEOUS_Z = 2.241402727604947
BOOTSTRAP_RESAMPLES = 20_000
BOOTSTRAP_SEED = 20260903
PLANNING_SD_DB = 0.444218872067388
ITS_ELIGIBLE_SCENES = 8813
OTS_ELIGIBLE_SCENES = 8006
ITS_EXCLUSION_COUNT = 2187
OTS_EXCLUSION_COUNT = 964
SUMMARY_NAME = "reside_known_overlap_continuous_utility_measurement_v1_summary.json"
STRATA_NAME = "reside_known_overlap_continuous_utility_measurement_v1_strata.csv"


def read_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def digest_lines(values: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(values)).encode("utf-8")).hexdigest()


def salted_order(values: Iterable[str], namespace: str) -> list[str]:
    return sorted(
        values,
        key=lambda value: (
            hashlib.sha256(f"{SELECTION_SALT}|{namespace}|{value}".encode()).hexdigest(),
            value,
        ),
    )


def choose_variants(paths: list[Path], stratum: str, scene_id: str) -> list[Path]:
    if len(paths) < VARIANTS_PER_SCENE:
        raise ValueError(f"{stratum} scene {scene_id} has fewer than four haze variants")
    ordered = sorted(
        paths,
        key=lambda path: (
            hashlib.sha256(
                f"{SELECTION_SALT}|{stratum}|{scene_id}|{path.name}".encode()
            ).hexdigest(),
            path.name,
        ),
    )
    return ordered[:VARIANTS_PER_SCENE]


def resize_pair(clear_path: Path, haze_path: Path) -> tuple[np.ndarray, np.ndarray]:
    width, height = ots_reader.target_size(clear_path, LONG_EDGE)
    clear = ots_reader.image_array(clear_path, (width, height))
    haze = ots_reader.image_array(haze_path, (width, height))
    if clear.shape != haze.shape or clear.shape[:2] != (height, width):
        raise ValueError("paired clear and haze content sizes do not match")
    return clear, haze


def infer_numpy(torch: Any, model: Any, haze: np.ndarray, device: str) -> np.ndarray:
    prediction = ots_reader.infer(
        torch, model, haze, device, model_min_edge=MODEL_MIN_EDGE,
    )
    value = np.transpose(prediction.squeeze(0).cpu().numpy(), (1, 2, 0))
    if value.shape != haze.shape or not np.isfinite(value).all():
        raise RuntimeError("official prediction is not finite or content-aligned")
    return value


def terminal_from_intervals(strata: dict[str, dict[str, Any]]) -> tuple[str, str, str, list[str]]:
    imprecise = [
        name for name, item in strata.items()
        if item["precision_distance_db"] > PRECISION_DISTANCE_DB
    ]
    if imprecise:
        return (
            "COMPLETED_INCONCLUSIVE",
            "RESIDE_CONTINUOUS_UTILITY_MEASUREMENT_INCONCLUSIVE",
            "NONE",
            [f"{name} precision distance exceeds 0.025 dB" for name in imprecise],
        )
    if all(item["upper_db"] < UTILITY_MARGIN_DB for item in strata.values()):
        return (
            "COMPLETED_GATE_PASS",
            "RESIDE_CONTINUOUS_UTILITY_MEASUREMENT_PASS",
            "STRATIFIED_CONTINUOUS_UTILITY_RESULT_ONLY",
            ["both simultaneous stratum upper bounds are below 0.05 dB"],
        )
    failing = [name for name, item in strata.items() if item["lower_db"] > UTILITY_MARGIN_DB]
    if failing:
        return (
            "COMPLETED_GATE_FAIL",
            "RESIDE_CONTINUOUS_UTILITY_MEASUREMENT_FAIL",
            "NONE",
            [f"simultaneous lower bound exceeds 0.05 dB in {name}" for name in failing],
        )
    return (
        "COMPLETED_INCONCLUSIVE",
        "RESIDE_CONTINUOUS_UTILITY_MEASUREMENT_INCONCLUSIVE",
        "NONE",
        ["at least one simultaneous stratum interval crosses 0.05 dB"],
    )


def bootstrap_mean_bounds(values: np.ndarray, seed_offset: int) -> tuple[float, float]:
    generator = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    means = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    for start in range(0, BOOTSTRAP_RESAMPLES, 500):
        stop = min(start + 500, BOOTSTRAP_RESAMPLES)
        indices = generator.integers(0, values.size, size=(stop - start, values.size))
        means[start:stop] = np.mean(values[indices], axis=1)
    return float(np.quantile(means, 0.0125)), float(np.quantile(means, 0.9875))


def summarize_stratum(values: list[float], seed_offset: int) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size != SCENES_PER_STRATUM or not np.isfinite(array).all():
        raise RuntimeError("stratum requires exactly 1,587 finite scene values")
    mean = float(np.mean(array))
    sample_sd = float(np.std(array, ddof=1))
    normal_distance = SIMULTANEOUS_Z * sample_sd / math.sqrt(array.size)
    bootstrap_lower, bootstrap_upper = bootstrap_mean_bounds(array, seed_offset)
    lower = min(mean - normal_distance, bootstrap_lower)
    upper = max(mean + normal_distance, bootstrap_upper)
    return {
        "independent_scenes": int(array.size),
        "nested_variants": int(array.size * VARIANTS_PER_SCENE),
        "mean_db": mean,
        "sample_sd_db": sample_sd,
        "normal_simultaneous_distance_db": normal_distance,
        "bootstrap_lower_db": bootstrap_lower,
        "bootstrap_upper_db": bootstrap_upper,
        "lower_db": lower,
        "upper_db": upper,
        "precision_distance_db": max(mean - lower, upper - mean),
        "utility_margin_db": UTILITY_MARGIN_DB,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED + seed_offset,
    }


def parent_identity(context: Any) -> tuple[bool, dict[str, Any]]:
    its_closeout = json.loads(asset_path(context, "its_quarantine_closeout", kind="file").read_text(encoding="utf-8"))
    its_conclusion = json.loads(asset_path(context, "its_quarantine_conclusion", kind="file").read_text(encoding="utf-8"))
    ots_closeout = json.loads(asset_path(context, "ots_quarantine_closeout", kind="file").read_text(encoding="utf-8"))
    ots_conclusion = json.loads(asset_path(context, "ots_quarantine_conclusion", kind="file").read_text(encoding="utf-8"))
    precision_closeout = json.loads(asset_path(context, "precision_pilot_closeout", kind="file").read_text(encoding="utf-8"))
    precision_summary = json.loads(asset_path(context, "precision_pilot_summary", kind="file").read_text(encoding="utf-8"))
    its_q_closeout = json.loads(asset_path(context, "its_q_fail_closeout", kind="file").read_text(encoding="utf-8"))
    checks = {
        "its_quarantine_terminal": (
            its_closeout.get("state") == "COMPLETED_GATE_PASS"
            and its_closeout.get("decision") == "ITS_VERIFIED_OVERLAP_QUARANTINE_PASS"
            and its_closeout.get("authorizes") == "ITS_KNOWN_OVERLAP_QUARANTINED_MEASUREMENT_DESIGN"
            and its_conclusion.get("authorizes") == "ITS_KNOWN_OVERLAP_QUARANTINED_MEASUREMENT_DESIGN"
        ),
        "ots_quarantine_terminal": (
            ots_closeout.get("state") == "COMPLETED_GATE_PASS"
            and ots_closeout.get("decision") == "OTS_TARGETED_GEOMETRY_PASS"
            and ots_closeout.get("authorizes") == "OTS_OUTDOOR_MEASUREMENT_DESIGN"
            and ots_conclusion.get("authorizes") == "OTS_OUTDOOR_MEASUREMENT_DESIGN"
        ),
        "precision_pilot_terminal": (
            precision_closeout.get("state") == "COMPLETED_GATE_FAIL"
            and precision_closeout.get("decision") == "HAZE4K_CONTINUOUS_UTILITY_BLOCKED_PRECISION"
            and precision_closeout.get("authorizes") == "NONE"
            and abs(
                float(precision_summary["precision_feasibility"]["conservative_planning_sd_db"])
                - PLANNING_SD_DB
            ) < 1e-12
        ),
        "its_q_construct_remains_blocked": (
            its_q_closeout.get("state") == "COMPLETED_GATE_FAIL"
            and its_q_closeout.get("decision") == "ITS_LOCAL_MEASUREMENT_MAPPING_FAIL"
            and its_q_closeout.get("authorizes") == "NONE"
        ),
    }
    return all(checks.values()), checks


def enumerate_its(reside: Path, exclusions: set[str]) -> tuple[dict[str, Path], dict[str, list[Path]], dict[str, Any]]:
    split_specs = (
        ("ITS_TRAIN", reside / "official/ITS/train/ITS_clear", reside / "official/ITS/train/ITS_haze"),
        ("ITS_VALIDATION", reside / "official/ITS/val/clear", reside / "official/ITS/val/haze"),
    )
    clear: dict[str, Path] = {}
    variants: dict[str, list[Path]] = {}
    counts: dict[str, int] = {}
    for prefix, clear_dir, haze_dir in split_specs:
        clear_paths = its_reader.image_files(clear_dir)
        haze_paths = its_reader.image_files(haze_dir)
        grouped = its_reader.grouped_by_scene(haze_paths)
        counts[f"{prefix.lower()}_clear"] = len(clear_paths)
        counts[f"{prefix.lower()}_variants"] = len(haze_paths)
        for path in clear_paths:
            key = f"{prefix}:{path.stem}"
            clear[key] = path
            variants[key] = grouped.get(path.stem, [])
    eligible = {key: path for key, path in clear.items() if key not in exclusions}
    eligible_variants = {key: variants[key] for key in eligible}
    checks = {
        "official_clear_count": len(clear) == 11000,
        "official_variant_count": sum(count for key, count in counts.items() if key.endswith("_variants")) == 110000,
        "exclusion_count": len(exclusions) == ITS_EXCLUSION_COUNT,
        "all_exclusions_are_official": exclusions.issubset(clear),
        "eligible_count": len(eligible) == ITS_ELIGIBLE_SCENES,
        "ten_variants_per_scene": all(len(paths) == 10 for paths in eligible_variants.values()),
    }
    return eligible, eligible_variants, {"counts": counts, "checks": checks}


def enumerate_ots(reside: Path, exclusions: set[str]) -> tuple[dict[str, Path], dict[str, list[Path]], dict[str, Any]]:
    clear_paths = ots_reader.image_files(reside / "official/OTS_ALPHA/clear_images")
    haze_paths = ots_reader.image_files(reside / "official/OTS_ALPHA/OTS")
    clear = {path.stem: path for path in clear_paths}
    variants: dict[str, list[Path]] = defaultdict(list)
    parse_failures: list[str] = []
    for path in haze_paths:
        try:
            scene_id, _, _ = ots_reader.parse_variant(path)
            variants[scene_id].append(path)
        except ValueError:
            parse_failures.append(path.name)
    eligible = {key: path for key, path in clear.items() if key not in exclusions}
    eligible_variants = {key: sorted(variants.get(key, [])) for key in eligible}
    checks = {
        "official_clear_count": len(clear) == 8970,
        "official_variant_count": len(haze_paths) == 313950,
        "all_variants_parse": not parse_failures,
        "exclusion_count": len(exclusions) == OTS_EXCLUSION_COUNT,
        "all_exclusions_are_official": exclusions.issubset(clear),
        "eligible_count": len(eligible) == OTS_ELIGIBLE_SCENES,
        "thirty_five_variants_per_scene": all(len(paths) == 35 for paths in eligible_variants.values()),
    }
    return eligible, eligible_variants, {"parse_failures": parse_failures[:20], "checks": checks}


def contract(context_path: Path) -> None:
    context = load_context(context_path, "contract")
    prepare_phase_output(context)
    if context.device != "cuda" or any(context.protected_data_permissions.values()):
        raise RuntimeError("measurement contract requires CUDA with protected roles disabled")
    if "reside_root" in context.assets:
        raise RuntimeError("scientific RESIDE data must be hidden from contract phase")
    torch, model = ots_reader.load_official_model(context)
    height, width = 256, 320
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    clear = np.stack(
        (0.15 + 0.65 * xx / width, 0.18 + 0.60 * yy / height, 0.20 + 0.55 * (xx + yy) / (width + height)),
        axis=-1,
    ).astype(np.float32)
    synthetic_variants = []
    for haze_scale in (0.68, 0.72, 0.76, 0.80):
        synthetic_variants.append(
            np.clip(haze_scale * clear + (1.0 - haze_scale) * 0.75, 0.0, 1.0).astype(np.float32)
        )
    raw_variants = []
    finite_forwards = True
    torch.cuda.reset_peak_memory_stats()
    probe_started = time.perf_counter()
    attempted = 0
    for iteration in range(TOTAL_INFERENCES):
        haze = synthetic_variants[iteration % len(synthetic_variants)]
        prediction = infer_numpy(torch, model, haze, context.device)
        finite_forwards = finite_forwards and bool(np.isfinite(prediction).all())
        attempted += 1
        if iteration < VARIANTS_PER_SCENE:
            raw_variants.append({"hazy": haze, "clear": clear, "prediction": prediction})
        if attempted == 1 or attempted % 512 == 0 or attempted == TOTAL_INFERENCES:
            print(
                json.dumps({
                    "marker": "RESIDE_CONTINUOUS_UTILITY_COST_PROBE_PROGRESS",
                    "completed_iterations": attempted,
                    "total_iterations": TOTAL_INFERENCES,
                }, sort_keys=True),
                flush=True,
            )
    torch.cuda.synchronize()
    probe_wall_seconds = time.perf_counter() - probe_started
    peak_cuda_mem_mib = float(torch.cuda.max_memory_allocated() / 1024**2)
    measured = utility.continuous_scene_measurement(raw_variants)
    planning_required = math.ceil(
        (SIMULTANEOUS_Z * PLANNING_SD_DB / PRECISION_DISTANCE_DB) ** 2
    )
    checks = {
        "strict_official_checkpoint_load": sum(parameter.numel() for parameter in model.parameters()) == 8630665,
        "all_parameters_frozen": not any(parameter.requires_grad for parameter in model.parameters()),
        "same_scale_12696_iteration_bound": attempted == TOTAL_INFERENCES,
        "same_scale_all_official_forwards_finite": finite_forwards,
        "same_scale_wall_under_840_seconds": probe_wall_seconds <= COST_PROBE_MAX_SECONDS,
        "peak_cuda_memory_under_8192_mib": peak_cuda_mem_mib <= COST_PROBE_MAX_CUDA_MIB,
        "exact_frozen_continuous_utility_path": math.isfinite(measured["worst_phase_linfinity_error_db"]),
        "keep_and_affine_controls": measured["keep_structural_identity"] and measured["affine_manipulation_exact"],
        "simultaneous_precision_reference": planning_required == SCENES_PER_STRATUM,
        "terminal_mapping_reference": terminal_from_intervals({
            "ITS": {"upper_db": 0.04, "lower_db": 0.02, "precision_distance_db": 0.02},
            "OTS": {"upper_db": 0.04, "lower_db": 0.02, "precision_distance_db": 0.02},
        })[0] == "COMPLETED_GATE_PASS",
        "precision_vetoes_apparent_pass": terminal_from_intervals({
            "ITS": {"upper_db": 0.04, "lower_db": 0.02, "precision_distance_db": 0.026},
            "OTS": {"upper_db": 0.04, "lower_db": 0.02, "precision_distance_db": 0.02},
        })[0] == "COMPLETED_INCONCLUSIVE",
        "precision_vetoes_apparent_fail": terminal_from_intervals({
            "ITS": {"upper_db": 0.08, "lower_db": 0.06, "precision_distance_db": 0.026},
            "OTS": {"upper_db": 0.04, "lower_db": 0.02, "precision_distance_db": 0.02},
        })[0] == "COMPLETED_INCONCLUSIVE",
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
    if context.total_units != TOTAL_UNITS:
        raise RuntimeError("runtime total_units differs from frozen measurement budget")
    parents_ok, parent_checks = parent_identity(context)
    its_exclusions = set(read_lines(asset_path(context, "its_exclusions", kind="file")))
    ots_exclusions = set(read_lines(asset_path(context, "ots_exclusions", kind="file")))
    reside = asset_path(context, "reside_root", kind="directory")
    its_clear, its_variants, its_identity = enumerate_its(reside, its_exclusions)
    ots_clear, ots_variants, ots_identity = enumerate_ots(reside, ots_exclusions)
    identity_ok = parents_ok and all(its_identity["checks"].values()) and all(ots_identity["checks"].values())

    selected = {
        "ITS": salted_order(its_clear, "ITS-scenes")[:SCENES_PER_STRATUM] if identity_ok else [],
        "OTS": salted_order(ots_clear, "OTS-scenes")[:SCENES_PER_STRATUM] if identity_ok else [],
    }
    selected_variants = {
        "ITS": {scene: choose_variants(its_variants[scene], "ITS", scene) for scene in selected["ITS"]},
        "OTS": {scene: choose_variants(ots_variants[scene], "OTS", scene) for scene in selected["OTS"]},
    }
    selection_checks = {
        "exact_scene_count_per_stratum": all(len(values) == SCENES_PER_STRATUM for values in selected.values()),
        "exact_variant_count_per_scene": all(
            len(paths) == VARIANTS_PER_SCENE
            for values in selected_variants.values() for paths in values.values()
        ),
        "selected_scenes_not_excluded": (
            set(selected["ITS"]).isdisjoint(its_exclusions)
            and set(selected["OTS"]).isdisjoint(ots_exclusions)
        ),
    }
    write_workload_progress(context, completed_units=2, stage="identity_and_frozen_selection")

    results: dict[str, list[float]] = {"ITS": [], "OTS": []}
    failures: list[dict[str, str]] = []
    attempted = 0
    if identity_ok and all(selection_checks.values()):
        torch, model = ots_reader.load_official_model(context)
        for stratum, clear_map in (("ITS", its_clear), ("OTS", ots_clear)):
            for scene_id in selected[stratum]:
                raw_variants = []
                for haze_path in selected_variants[stratum][scene_id]:
                    attempted += 1
                    try:
                        clear, haze = resize_pair(clear_map[scene_id], haze_path)
                        prediction = infer_numpy(torch, model, haze, context.device)
                        raw_variants.append({"hazy": haze, "clear": clear, "prediction": prediction})
                    except Exception as exc:
                        failures.append({
                            "stratum": stratum,
                            "scene": scene_id,
                            "variant_digest": hashlib.sha256(haze_path.name.encode()).hexdigest()[:16],
                            "reason": str(exc)[:512],
                        })
                    if attempted == 1 or attempted % 32 == 0 or attempted == TOTAL_INFERENCES:
                        write_workload_progress(
                            context,
                            completed_units=2 + attempted,
                            stage=f"{stratum.lower()}_official_inference",
                        )
                if len(raw_variants) == VARIANTS_PER_SCENE:
                    try:
                        measured = utility.continuous_scene_measurement(raw_variants)
                        results[stratum].append(float(measured["worst_phase_linfinity_error_db"]))
                    except Exception as exc:
                        failures.append({
                            "stratum": stratum,
                            "scene": scene_id,
                            "variant_digest": "continuous_measurement",
                            "reason": str(exc)[:512],
                        })

    coverage_ok = (
        not failures
        and attempted == TOTAL_INFERENCES
        and all(len(values) == SCENES_PER_STRATUM for values in results.values())
    )
    strata = {
        "ITS": summarize_stratum(results["ITS"], 0),
        "OTS": summarize_stratum(results["OTS"], 1),
    } if identity_ok and all(selection_checks.values()) and coverage_ok else {}
    if strata:
        state, decision, authorizes, gate_reasons = terminal_from_intervals(strata)
    else:
        state, decision, authorizes = (
            "COMPLETED_INCONCLUSIVE",
            "RESIDE_CONTINUOUS_UTILITY_MEASUREMENT_INCONCLUSIVE",
            "NONE",
        )
        gate_reasons = ["parent identity, dataset identity, frozen selection, or complete-scene coverage failed"]

    summary = {
        "schema_version": 1,
        "route_id": ROUTE_ID,
        "operation_id": OPERATION_ID,
        "run_id": context.run_id,
        "scope": "separately reported ITS indoor and OTS outdoor known-overlap-quarantined synthetic-scene populations",
        "parent_identity_checks": parent_checks,
        "dataset_identity": {"ITS": its_identity, "OTS": ots_identity},
        "selection": {
            "method": "first 1,587 eligible clear-scene IDs and first four same-scene official haze filenames after namespace-specific salted SHA-256 ordering",
            "salt": SELECTION_SALT,
            "independent_unit": "original_clear_scene",
            "scenes_per_stratum": SCENES_PER_STRATUM,
            "nested_variants_per_scene": VARIANTS_PER_SCENE,
            "scene_digests": {name: digest_lines(values) for name, values in selected.items()},
            "checks": selection_checks,
        },
        "estimand": {
            "per_stratum": "mean original-clear-scene worst-held-out-phase area-weighted L-infinity error for two keep-referenced continuous PSNR utility contrasts",
            "primary": "maximum of the separately estimated ITS and OTS stratum means; no sample-size-weighted pooling",
            "utility_margin_db": UTILITY_MARGIN_DB,
            "precision_distance_db": PRECISION_DISTANCE_DB,
            "simultaneous_z": SIMULTANEOUS_Z,
            "familywise_confidence": 0.95,
        },
        "strata": strata,
        "complete_scene_coverage": coverage_ok,
        "completed_inferences": attempted - len([item for item in failures if item["variant_digest"] != "continuous_measurement"]),
        "failure_count": len(failures),
        "failures": failures[:20],
        "terminal": {
            "state": state,
            "decision": decision,
            "authorizes": authorizes,
            "gate_reasons": gate_reasons,
        },
        "limitations": [
            "The populations quarantine verified known overlaps only and are not claimed fully source-disjoint.",
            "ITS and OTS are separate synthetic-domain strata; neither is transported to real haze and they are not pooled as exchangeable scenes.",
            "Haze variants, phases, tiles, pixels, actions and bootstrap draws are nested and never increase independent n.",
            "The failed ITS q construct is not used, reopened or reinterpreted by this continuous-utility route.",
            "This development-screening measurement cannot authorize training, model selection, confirmation, canary, locked-test or deployment use.",
        ],
        "marker": "RESIDE_KNOWN_OVERLAP_CONTINUOUS_UTILITY_MEASUREMENT_COMPLETE",
    }
    atomic_json(output_file(context, SUMMARY_NAME), summary)
    with output_file(context, STRATA_NAME).open("w", encoding="utf-8", newline="") as stream:
        fields = [
            "stratum", "independent_scenes", "nested_variants", "mean_db", "sample_sd_db",
            "lower_db", "upper_db", "precision_distance_db", "utility_margin_db",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name in ("ITS", "OTS"):
            row = {"stratum": name}
            row.update({key: strata.get(name, {}).get(key, "") for key in fields if key != "stratum"})
            writer.writerow(row)
    write_workload_progress(context, completed_units=TOTAL_UNITS, stage="stratified_measurement_finalize")
    write_run_result(
        context,
        state=state,
        decision=decision,
        authorizes=authorizes,
        details={
            "summary_file": SUMMARY_NAME,
            "strata_file": STRATA_NAME,
            "independent_scenes_per_stratum": {
                name: len(values) for name, values in results.items()
            },
            "nested_variants_per_scene": VARIANTS_PER_SCENE,
            "gate_reasons": gate_reasons,
            "known_overlap_quarantined_only": True,
            "training_occurred": False,
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
