#!/usr/bin/env python3
"""Replicate local-error constructs on isolated Haze4K test development scenes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

from route_program_api import (
    asset_path, atomic_json, load_context, output_file, prepare_phase_output,
    write_contract_result, write_run_result, write_workload_progress,
)


IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}
EXPECTED_SCENES = 100
EXPECTED_VARIANTS = 400
VARIANTS_PER_SCENE = 4
SPLIT_SALT = "haze4k-test-local-error-replication-v1"
SPLIT_ASSIGNMENT_DIGEST = "6ca5174470dad2b4eef4ae15c5a13a99d8ae9fc0bc2ea1116b199c4d4bc05582"
PARENT_CLOSEOUT_SHA256 = "d3dbd88b25eee35c5922b2459a2f48b39b3fc8f686588cde9edb1d9f267f8a9f"
CHECKPOINT_SHA256 = "6f42037d57a4e3de3a10ac0ab909d66a3415864a19433c29204a975f4efa4088"
MODEL_SOURCE_SHA256 = "3fa227af396464a7f07ac773f92e9cdb746e0fa6ae63adef711c765a02c3d4cd"
MODEL_LAYERS_SHA256 = "ac8a05bd626d9adda16308dedb9466f36d7ff44cfb666f64e7e14ddf8cdf43a4"
PARAMETER_COUNT = 8_630_665
TILE_SIZE = 32
REGION_FRACTION = 0.20
TILE_FRACTION_MARGIN = 0.20
SCENE_REPEAT_VARIANTS = 3
GLOBAL_COMPETENCE_MARGIN = 0.80
REPLICATION_OVERSHOOT_MIN = 0.05
UNDER_RECOVERY_NONPREVALENCE_MARGIN = 0.20
HIGH_SPECIFICITY_MARGIN = 0.05
ORACLE_PREVALENCE_MARGIN = 0.20
UNDER_ALPHA_MAX = 0.80
UNDER_RESIDUAL_RATIO_MIN = 0.25
OVERSHOOT_ALPHA_MIN = 1.05
ABSOLUTE_ERROR_MARGIN = 1.0 / 255.0
LOW_HARM_RATIO_MIN = 1.10
BOOTSTRAP_SEED = 20260724
BOOTSTRAP_RESAMPLES = 20_000
EPSILON = 1e-8


def image_files(directory: Path) -> list[Path]:
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def canonical_rgb_digest(array: np.ndarray) -> str:
    height, width = array.shape[:2]
    payload = np.rint(array * 255.0).clip(0, 255).astype(np.uint8).tobytes()
    digest = hashlib.sha256()
    digest.update(width.to_bytes(8, "big"))
    digest.update(height.to_bytes(8, "big"))
    digest.update(payload)
    return digest.hexdigest()


def image_array(path: Path) -> np.ndarray:
    with Image.open(path) as source:
        value = np.asarray(source.convert("RGB"), dtype=np.float32) / 255.0
    if value.ndim != 3 or value.shape[2] != 3 or min(value.shape[:2]) <= TILE_SIZE:
        raise RuntimeError(f"unsupported image shape for {path.name}: {value.shape}")
    return value


def wilson(successes: int, total: int, z_value: float = 1.96) -> dict[str, float | int]:
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("invalid Wilson inputs")
    estimate = successes / total
    denominator = 1.0 + z_value * z_value / total
    center = (estimate + z_value * z_value / (2.0 * total)) / denominator
    half = z_value * math.sqrt(
        estimate * (1.0 - estimate) / total
        + z_value * z_value / (4.0 * total * total)
    ) / denominator
    return {
        "successes": successes, "total": total, "estimate": estimate,
        "lower": max(0.0, center - half), "upper": min(1.0, center + half),
    }


def aggregate(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "mean": None, "median": None, "q25": None, "q75": None}
    if not np.isfinite(array).all():
        raise RuntimeError("aggregate received non-finite values")
    return {
        "count": int(array.size), "mean": float(np.mean(array)),
        "median": float(np.median(array)), "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
    }


def paired_bootstrap(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise RuntimeError("paired bootstrap requires finite scene values")
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    for start in range(0, BOOTSTRAP_RESAMPLES, 1000):
        stop = min(start + 1000, BOOTSTRAP_RESAMPLES)
        indices = generator.integers(0, array.size, size=(stop - start, array.size))
        draws[start:stop] = np.mean(array[indices], axis=1)
    return {
        "scene_count": int(array.size), "estimate": float(np.mean(array)),
        "lower": float(np.quantile(draws, 0.025)),
        "upper": float(np.quantile(draws, 0.975)),
        "seed": BOOTSTRAP_SEED, "resamples": BOOTSTRAP_RESAMPLES,
    }


def image_tiles(array: np.ndarray) -> np.ndarray:
    height, width = array.shape[:2]
    padded = np.pad(
        array, ((0, (-height) % TILE_SIZE), (0, (-width) % TILE_SIZE), (0, 0)),
        mode="reflect",
    )
    grid_height = padded.shape[0] // TILE_SIZE
    grid_width = padded.shape[1] // TILE_SIZE
    return padded.reshape(
        grid_height, TILE_SIZE, grid_width, TILE_SIZE, 3,
    ).transpose(0, 2, 1, 3, 4).astype(np.float64)


def region_indices(score_tiles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = score_tiles.reshape(-1)
    count = max(1, int(math.floor(flat.size * REGION_FRACTION)))
    if count * 2 > flat.size:
        raise RuntimeError("too few tiles for disjoint demand regions")
    order = np.argsort(flat, kind="mergesort")
    return order[:count], order[-count:]


def variant_measurement(hazy: np.ndarray, clear: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    if hazy.shape != clear.shape or prediction.shape != clear.shape:
        raise RuntimeError("paired image shapes changed")
    hazy_tiles = image_tiles(hazy)
    clear_tiles = image_tiles(clear)
    prediction_tiles = image_tiles(prediction)
    direction = clear_tiles - hazy_tiles
    change = prediction_tiles - hazy_tiles
    input_squared = np.mean(direction * direction, axis=(2, 3, 4))
    output_squared = np.mean((prediction_tiles - clear_tiles) ** 2, axis=(2, 3, 4))
    direction_norm = np.sum(direction * direction, axis=(2, 3, 4))
    alpha = np.sum(change * direction, axis=(2, 3, 4)) / (direction_norm + EPSILON)
    demand_rms = np.sqrt(input_squared)
    residual_ratio = output_squared / (input_squared + EPSILON)
    projected_overshoot = np.maximum(alpha - 1.0, 0.0) * demand_rms
    under = (alpha <= UNDER_ALPHA_MAX) & (residual_ratio >= UNDER_RESIDUAL_RATIO_MIN)
    overshoot = (alpha >= OVERSHOOT_ALPHA_MIN) & (projected_overshoot >= ABSOLUTE_ERROR_MARGIN)
    harm = (
        (output_squared >= LOW_HARM_RATIO_MIN * input_squared)
        & ((output_squared - input_squared) >= ABSOLUTE_ERROR_MARGIN ** 2)
    )
    if not all(np.isfinite(item).all() for item in (input_squared, output_squared, alpha)):
        raise RuntimeError("non-finite local measurement")
    low_indices, high_indices = region_indices(input_squared)
    flat_input = input_squared.reshape(-1)
    flat_output = output_squared.reshape(-1)
    flat_overshoot = overshoot.reshape(-1)
    flat_under = under.reshape(-1)
    flat_harm = harm.reshape(-1)
    input_mse = float(np.mean(flat_input))
    output_mse = float(np.mean(flat_output))
    high_under = float(np.mean(flat_under[high_indices]))
    high_overshoot = float(np.mean(flat_overshoot[high_indices]))
    low_overshoot = float(np.mean(flat_overshoot[low_indices]))
    all_overshoot = float(np.mean(flat_overshoot))
    low_harm = float(np.mean(flat_harm[low_indices]))
    return {
        "input_mse": input_mse, "output_mse": output_mse,
        "input_psnr": -10.0 * math.log10(max(input_mse, 1e-12)),
        "output_psnr": -10.0 * math.log10(max(output_mse, 1e-12)),
        "high_under_recovery_fraction": high_under,
        "high_material_overshoot_fraction": high_overshoot,
        "low_material_overshoot_fraction": low_overshoot,
        "all_material_overshoot_fraction": all_overshoot,
        "low_demand_harm_fraction": low_harm,
        "variant_under_recovery": high_under >= TILE_FRACTION_MARGIN,
        "variant_signed_overshoot": all_overshoot >= TILE_FRACTION_MARGIN,
        "variant_low_demand_harm": low_harm >= TILE_FRACTION_MARGIN,
    }


def mean_key(records: list[dict[str, Any]], key: str) -> float:
    values = np.asarray([float(record[key]) for record in records], dtype=np.float64)
    if values.size != VARIANTS_PER_SCENE or not np.isfinite(values).all():
        raise RuntimeError(f"invalid scene values for {key}")
    return float(np.mean(values))


def scene_measurement(records: list[dict[str, Any]]) -> dict[str, Any]:
    if len(records) != VARIANTS_PER_SCENE:
        raise RuntimeError("scene does not contain four complete variants")
    under_count = sum(bool(record["variant_under_recovery"]) for record in records)
    overshoot_count = sum(bool(record["variant_signed_overshoot"]) for record in records)
    harm_count = sum(bool(record["variant_low_demand_harm"]) for record in records)
    result = {
        "globally_competent": mean_key(records, "output_mse") < mean_key(records, "input_mse"),
        "input_psnr": mean_key(records, "input_psnr"),
        "output_psnr": mean_key(records, "output_psnr"),
        "high_under_recovery_fraction": mean_key(records, "high_under_recovery_fraction"),
        "high_material_overshoot_fraction": mean_key(records, "high_material_overshoot_fraction"),
        "low_material_overshoot_fraction": mean_key(records, "low_material_overshoot_fraction"),
        "all_material_overshoot_fraction": mean_key(records, "all_material_overshoot_fraction"),
        "low_demand_harm_fraction": mean_key(records, "low_demand_harm_fraction"),
        "replication_high_demand_overshoot": mean_key(records, "high_material_overshoot_fraction") >= TILE_FRACTION_MARGIN,
        "replication_high_demand_under_recovery": mean_key(records, "high_under_recovery_fraction") >= TILE_FRACTION_MARGIN,
        "stable_under_recovery": under_count >= SCENE_REPEAT_VARIANTS,
        "stable_signed_overshoot": overshoot_count >= SCENE_REPEAT_VARIANTS,
        "stable_low_demand_harm": harm_count >= SCENE_REPEAT_VARIANTS,
    }
    result["stable_any_local_error"] = bool(
        result["stable_under_recovery"]
        or result["stable_signed_overshoot"]
        or result["stable_low_demand_harm"]
    )
    return result


def load_official_model(context):
    import torch

    checkpoint = asset_path(context, "official_checkpoint", kind="file")
    model_source = asset_path(context, "model_source", kind="file")
    model_layers = asset_path(context, "model_layers", kind="file")
    asset_path(context, "measurement_entrypoint", kind="file")
    expected = {
        "official_checkpoint": CHECKPOINT_SHA256,
        "model_source": MODEL_SOURCE_SHA256,
        "model_layers": MODEL_LAYERS_SHA256,
    }
    for identifier, identity in expected.items():
        if context.assets[identifier].sha256 != identity:
            raise RuntimeError(f"verified identity changed for {identifier}")
    if str(context.remote_repo) not in sys.path:
        sys.path.insert(0, str(context.remote_repo))
    from Dehazing.ITS.models.ConvIR import build_net

    module = sys.modules[build_net.__module__]
    layers_module = sys.modules.get("Dehazing.ITS.models.layers")
    if Path(module.__file__).resolve() != model_source.resolve():
        raise RuntimeError("official model import resolved to a different file")
    if layers_module is None or Path(layers_module.__file__).resolve() != model_layers.resolve():
        raise RuntimeError("official layer import resolved to a different file")
    model = build_net("base", "Haze4K", fam_mode="original")
    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location="cpu")
    if not isinstance(state, dict) or not isinstance(state.get("model"), dict):
        raise RuntimeError("official checkpoint lacks state_dict['model']")
    model.load_state_dict(state["model"], strict=True)
    if sum(parameter.numel() for parameter in model.parameters()) != PARAMETER_COUNT:
        raise RuntimeError("official parameter count changed")
    model.requires_grad_(False).to(context.device).eval()
    return torch, model


def infer(torch, model, hazy: np.ndarray, device: str) -> np.ndarray:
    import torch.nn.functional as functional

    tensor = torch.from_numpy(np.transpose(hazy, (2, 0, 1)).copy()).unsqueeze(0).to(device)
    height, width = tensor.shape[-2:]
    padded = functional.pad(tensor, (0, (-width) % 32, 0, (-height) % 32), mode="reflect")
    with torch.inference_mode():
        outputs = model(padded)
        if not isinstance(outputs, list) or len(outputs) != 3:
            raise RuntimeError("official three-scale output contract changed")
        prediction = outputs[2][:, :, :height, :width].clamp(0.0, 1.0)
        if not bool(torch.isfinite(prediction).all().item()):
            raise RuntimeError("official model produced non-finite output")
    return np.transpose(prediction.squeeze(0).cpu().numpy(), (1, 2, 0))


def finalize(context, integrity: dict[str, bool], scenes: list[dict[str, Any]], failures: list[dict[str, str]]) -> dict[str, Any]:
    complete = all(integrity.values()) and len(scenes) == EXPECTED_SCENES and not failures
    intervals = {
        "replication_high_demand_overshoot": wilson(sum(bool(s["replication_high_demand_overshoot"]) for s in scenes), EXPECTED_SCENES),
        "replication_high_demand_under_recovery": wilson(sum(bool(s["replication_high_demand_under_recovery"]) for s in scenes), EXPECTED_SCENES),
        "stable_any_local_error": wilson(sum(bool(s["stable_any_local_error"]) for s in scenes), EXPECTED_SCENES),
        "stable_under_recovery": wilson(sum(bool(s["stable_under_recovery"]) for s in scenes), EXPECTED_SCENES),
        "stable_signed_overshoot": wilson(sum(bool(s["stable_signed_overshoot"]) for s in scenes), EXPECTED_SCENES),
        "stable_low_demand_harm": wilson(sum(bool(s["stable_low_demand_harm"]) for s in scenes), EXPECTED_SCENES),
        "global_competence": wilson(sum(bool(s["globally_competent"]) for s in scenes), EXPECTED_SCENES),
    }
    contrast_values = [
        float(scene["high_material_overshoot_fraction"])
        - float(scene["low_material_overshoot_fraction"]) for scene in scenes
    ]
    contrast = paired_bootstrap(contrast_values) if scenes else None
    competence_pass = float(intervals["global_competence"]["lower"]) >= GLOBAL_COMPETENCE_MARGIN
    pattern_pass = bool(
        complete and competence_pass
        and float(intervals["replication_high_demand_overshoot"]["lower"]) >= REPLICATION_OVERSHOOT_MIN
        and float(intervals["replication_high_demand_under_recovery"]["upper"]) < UNDER_RECOVERY_NONPREVALENCE_MARGIN
        and contrast is not None and float(contrast["upper"]) < HIGH_SPECIFICITY_MARGIN
    )
    pattern_fail = bool(
        complete and competence_pass and contrast is not None and (
            float(intervals["replication_high_demand_overshoot"]["upper"]) < REPLICATION_OVERSHOOT_MIN
            or float(intervals["replication_high_demand_under_recovery"]["lower"]) >= UNDER_RECOVERY_NONPREVALENCE_MARGIN
            or float(contrast["lower"]) >= HIGH_SPECIFICITY_MARGIN
        )
    )
    oracle = intervals["stable_any_local_error"]
    oracle_pass = float(oracle["lower"]) >= ORACLE_PREVALENCE_MARGIN
    oracle_fail = float(oracle["upper"]) < ORACLE_PREVALENCE_MARGIN
    if not complete or not competence_pass:
        state, decision, authorizes = (
            "COMPLETED_INCONCLUSIVE",
            "HAZE4K_TEST_LOCAL_ERROR_REPLICATION_INCONCLUSIVE",
            "HAZE4K_TEST_LOCAL_ERROR_SUPPLEMENT_ONLY",
        )
        reasons = ["complete isolated-development coverage or global baseline competence failed"]
    elif pattern_pass and oracle_pass:
        state, decision, authorizes = (
            "COMPLETED_GATE_PASS",
            "HAZE4K_TEST_LOCAL_ERROR_REPLICATION_ORACLE_QUALIFIED",
            "HAZE4K_TEST_BOUNDED_LOCAL_ACTION_ORACLE",
        )
        reasons = ["SOTS-consistent pattern replicated and stable-any-local-error lower bound met 0.20"]
    elif pattern_pass and oracle_fail:
        state, decision, authorizes = (
            "COMPLETED_GATE_PASS",
            "HAZE4K_TEST_LOCAL_ERROR_PATTERN_REPLICATED_ORACLE_NOT_QUALIFIED",
            "NH_HAZE_DEVELOPMENT_CONSTRUCT_AUDIT",
        )
        reasons = ["SOTS-consistent pattern replicated but stable-any-local-error upper bound was below 0.20"]
    elif pattern_pass:
        state, decision, authorizes = (
            "COMPLETED_INCONCLUSIVE",
            "HAZE4K_TEST_LOCAL_ERROR_PATTERN_REPLICATED_ORACLE_INCONCLUSIVE",
            "HAZE4K_TEST_LOCAL_ERROR_SUPPLEMENT_ONLY",
        )
        reasons = ["SOTS-consistent pattern replicated but oracle-qualification interval crossed 0.20"]
    elif pattern_fail:
        state, decision, authorizes = (
            "COMPLETED_GATE_FAIL", "HAZE4K_TEST_LOCAL_ERROR_REPLICATION_FAIL", "NONE",
        )
        reasons = ["at least one frozen cross-domain pattern component was contradicted"]
    else:
        state, decision, authorizes = (
            "COMPLETED_INCONCLUSIVE",
            "HAZE4K_TEST_LOCAL_ERROR_REPLICATION_INCONCLUSIVE",
            "HAZE4K_TEST_LOCAL_ERROR_SUPPLEMENT_ONLY",
        )
        reasons = ["at least one frozen cross-domain pattern interval crossed its margin"]

    summary = {
        "schema_version": 1, "route_id": context.route_id,
        "operation_id": context.operation_id, "run_id": context.run_id,
        "scope": "fixed official Haze4K ConvIR-B on isolated 100-scene Haze4K official-test development partition",
        "identity_and_coverage": {
            "integrity_checks": integrity, "completed_scenes": len(scenes),
            "completed_variants": len(scenes) * VARIANTS_PER_SCENE,
            "failure_count": len(failures), "failures": failures[:20],
            "assignment_digest": SPLIT_ASSIGNMENT_DIGEST,
            "candidate_confirmation_asset_delivered": False,
        },
        "measurement": {
            "tile_size": TILE_SIZE, "region_fraction": REGION_FRACTION,
            "variant_material_fraction": TILE_FRACTION_MARGIN,
            "scene_repeatability": "same construct material in at least three of four variants",
            "alpha_min_overshoot": OVERSHOOT_ALPHA_MIN,
            "projected_absolute_margin": ABSOLUTE_ERROR_MARGIN,
        },
        "cross_domain_pattern": {
            "passed": pattern_pass,
            "high_demand_overshoot_scene_prevalence": intervals["replication_high_demand_overshoot"],
            "high_demand_under_recovery_scene_prevalence": intervals["replication_high_demand_under_recovery"],
            "high_minus_low_overshoot_fraction": contrast,
            "margins": {
                "overshoot_lower_min": REPLICATION_OVERSHOOT_MIN,
                "under_recovery_upper_max": UNDER_RECOVERY_NONPREVALENCE_MARGIN,
                "high_specificity_upper_max": HIGH_SPECIFICITY_MARGIN,
            },
        },
        "oracle_qualification": {
            "stable_construct_prevalence": {key: value for key, value in intervals.items() if key.startswith("stable_")},
            "stable_any_lower_min": ORACLE_PREVALENCE_MARGIN,
            "passed": oracle_pass, "failed": oracle_fail,
        },
        "global_competence": {"scene_prevalence": intervals["global_competence"], "passed": competence_pass},
        "secondary_scene_aggregates": {
            "input_psnr": aggregate(s["input_psnr"] for s in scenes),
            "output_psnr": aggregate(s["output_psnr"] for s in scenes),
            "high_material_overshoot_fraction": aggregate(s["high_material_overshoot_fraction"] for s in scenes),
            "low_material_overshoot_fraction": aggregate(s["low_material_overshoot_fraction"] for s in scenes),
        },
        "terminal": {"state": state, "decision": decision, "authorizes": authorizes, "gate_reasons": reasons},
        "limitations": [
            "This is development-screening evidence; the 150-scene candidate-confirmation asset was not delivered.",
            "GT-derived demand and error labels are measurement-only and unavailable at deployment.",
            "Signed RGB projection is an operational construct, not a causal or perceptual proof.",
            "Pattern replication and oracle qualification are separate gates.",
        ],
        "marker": "HAZE4K_TEST_DEVELOPMENT_LOCAL_ERROR_REPLICATION_COMPLETE",
    }
    atomic_json(output_file(context, "haze4k_test_development_local_error_summary.json"), summary)
    rows = []
    for name, interval in intervals.items():
        rows.append({
            "stratum": name, "scenes": interval["total"],
            "positive_scenes": interval["successes"], "estimate": interval["estimate"],
            "lower_95": interval["lower"], "upper_95": interval["upper"],
        })
    with output_file(context, "haze4k_test_development_local_error_strata.csv").open(
        "w", encoding="utf-8", newline="",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {
        "state": state, "decision": decision, "authorizes": authorizes,
        "details": {
            "summary_file": "haze4k_test_development_local_error_summary.json",
            "strata_file": "haze4k_test_development_local_error_strata.csv",
            "independent_scenes": len(scenes), "nested_variants": len(scenes) * 4,
            "pattern_replicated": pattern_pass, "oracle_qualified": oracle_pass,
            "gate_reasons": reasons, "training_occurred": False,
            "candidate_confirmation_asset_delivered": False,
        },
    }


def contract(context_path: Path) -> None:
    context = load_context(context_path, "contract")
    prepare_phase_output(context)
    if context.device != "cuda" or any(context.protected_data_permissions.values()):
        raise RuntimeError("local-error contract requires CUDA and no protected permission")
    if "haze4k_test_development" in context.assets:
        raise RuntimeError("scientific development data must be absent from contract phase")
    torch, model = load_official_model(context)
    height, width = 256, 320
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    clear = np.stack((xx / width, yy / height, (xx + yy) / (width + height)), axis=-1)
    transmission = np.exp(-0.18 * (1.0 + xx / width + yy / height)).astype(np.float32)
    hazy = (clear * transmission[..., None] + 0.90 * (1.0 - transmission[..., None])).astype(np.float32)
    prediction = infer(torch, model, hazy, context.device)
    measured = variant_measurement(hazy, clear, prediction)
    unchanged = variant_measurement(hazy, clear, hazy)
    exact = variant_measurement(hazy, clear, clear)
    partial = variant_measurement(hazy, clear, hazy + 0.40 * (clear - hazy))
    extended = variant_measurement(hazy, clear, hazy + 10.0 * (clear - hazy))
    checks = {
        "strict_checkpoint_load_and_parameter_count": sum(p.numel() for p in model.parameters()) == PARAMETER_COUNT,
        "all_parameters_frozen": not any(p.requires_grad for p in model.parameters()),
        "three_scale_finite_forward": bool(np.isfinite(prediction).all()),
        "unchanged_reference": abs(float(unchanged["output_mse"]) - float(unchanged["input_mse"])) < 1e-12,
        "exact_gt_reference": exact["output_mse"] == 0.0,
        "partial_recovery_reference": partial["high_under_recovery_fraction"] == 1.0,
        "strong_overshoot_reference": extended["all_material_overshoot_fraction"] >= 0.99,
        "measurement_finalizer_finite": all(
            np.isfinite(float(v)) for v in measured.values() if isinstance(v, (int, float)) and not isinstance(v, bool)
        ),
        "wilson_direction_reference": wilson(20, 100)["lower"] < wilson(21, 100)["lower"],
        "bootstrap_direction_reference": abs(float(paired_bootstrap([0.1, 0.2, 0.3, 0.4])["estimate"]) - 0.25) < 1e-12,
    }
    write_contract_result(
        context, checks=checks,
        engineering={
            "mode": "gpu_synthetic_no_data", "device": "cuda",
            "fixture": {"batch": 1, "channels": 3, "height": height, "width": width},
            "production_path_exercised": True, "protected_data_touched": False,
            "scientific_output_created": False, "scientific_training_occurred": False,
        },
    )


def run(context_path: Path) -> None:
    context = load_context(context_path, "run")
    prepare_phase_output(context)
    development_root = asset_path(context, "haze4k_test_development", kind="directory")
    parent_path = asset_path(context, "prior_test_split_closeout", kind="file")
    if context.assets["prior_test_split_closeout"].sha256 != PARENT_CLOSEOUT_SHA256:
        raise RuntimeError("parent closeout identity changed")
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    parent_ok = (
        parent.get("state") == "COMPLETED_GATE_PASS"
        and parent.get("decision") == "HAZE4K_TEST_SCENE_SPLIT_PASS"
        and parent.get("authorizes") == "HAZE4K_TEST_DEVELOPMENT_LOCAL_ERROR_REPLICATION"
        and parent.get("details", {}).get("split_assignment_digest") == SPLIT_ASSIGNMENT_DIGEST
        and parent.get("details", {}).get("development_scenes") == EXPECTED_SCENES
    )
    scope_ok = (
        development_root.name == "development_screening"
        and not any(context.protected_data_permissions.values())
        and "candidate_confirmation" not in str(development_root)
        and "haze4k_test_candidate_confirmation" not in context.assets
    )
    haze_root, clear_root = development_root / "haze", development_root / "gt"
    hazy_paths = image_files(haze_root) if scope_ok and haze_root.is_dir() else []
    clear_paths = image_files(clear_root) if scope_ok and clear_root.is_dir() else []
    exact_names = {p.name for p in hazy_paths} == {p.name for p in clear_paths}
    write_workload_progress(context, completed_units=1, stage="isolated_development_pairing")

    clear_by_digest: dict[str, list[Path]] = defaultdict(list)
    failures: list[dict[str, str]] = []
    for clear_path in clear_paths:
        try:
            clear_by_digest[canonical_rgb_digest(image_array(clear_path))].append(clear_path)
        except Exception as exc:
            failures.append({"scene": "clear", "variant": clear_path.name, "reason": str(exc)[:512]})
    variants_by_digest: dict[str, list[Path]] = defaultdict(list)
    for hazy_path in hazy_paths:
        clear_path = clear_root / hazy_path.name
        try:
            variants_by_digest[canonical_rgb_digest(image_array(clear_path))].append(hazy_path)
        except Exception as exc:
            failures.append({"scene": "pair", "variant": hazy_path.name, "reason": str(exc)[:512]})
    histogram = Counter(len(paths) for paths in variants_by_digest.values())
    dataset_ok = (
        len(hazy_paths) == EXPECTED_VARIANTS and len(clear_paths) == EXPECTED_VARIANTS
        and exact_names and len(clear_by_digest) == EXPECTED_SCENES
        and len(variants_by_digest) == EXPECTED_SCENES
        and histogram == {VARIANTS_PER_SCENE: EXPECTED_SCENES} and not failures
    )
    write_workload_progress(context, completed_units=2, stage="canonical_scene_grouping")

    scenes: list[dict[str, Any]] = []
    if parent_ok and scope_ok and dataset_ok:
        torch, model = load_official_model(context)
        attempted = 0
        for digest in sorted(variants_by_digest):
            records = []
            for hazy_path in sorted(variants_by_digest[digest]):
                attempted += 1
                try:
                    hazy = image_array(hazy_path)
                    clear = image_array(clear_root / hazy_path.name)
                    prediction = infer(torch, model, hazy, context.device)
                    records.append(variant_measurement(hazy, clear, prediction))
                except Exception as exc:
                    failures.append({"scene": digest[:16], "variant": hazy_path.name, "reason": str(exc)[:512]})
                if attempted % 5 == 0:
                    write_workload_progress(context, completed_units=2 + attempted, stage="official_haze4k_inference")
            if len(records) == VARIANTS_PER_SCENE:
                scenes.append(scene_measurement(records))

    integrity = {
        "parent_split_authorization": parent_ok,
        "isolated_development_asset_only": scope_ok,
        "complete_100_scene_400_variant_grouping": dataset_ok,
        "complete_finite_inference": len(scenes) == EXPECTED_SCENES and not failures,
        "candidate_confirmation_asset_not_delivered": True,
        "no_training": True,
    }
    result = finalize(context, integrity, scenes, failures)
    write_workload_progress(context, completed_units=403, stage="scene_level_finalize")
    write_run_result(
        context, state=result["state"], decision=result["decision"],
        authorizes=result["authorizes"], details=result["details"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("contract", "run"))
    parser.add_argument("--context", required=True, type=Path)
    args = parser.parse_args()
    if args.phase == "contract":
        contract(args.context)
    else:
        run(args.context)


if __name__ == "__main__":
    main()
