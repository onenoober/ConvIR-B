#!/usr/bin/env python3
"""Measure fixed ConvIR-B local errors on Haze4K train internal development."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

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


IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}
INPUT_DIRECTORY_NAMES = ("IN", "haze", "hazy")
LABEL_DIRECTORY_NAMES = ("GT", "gt")
EXPECTED_FILES = 3000
EXPECTED_SCENES = 750
DEVELOPMENT_SCENES = 150
DEVELOPMENT_VARIANTS = 600
VARIANTS_PER_SCENE = 4
SPLIT_SALT = "haze4k-local-error-qualification-v2"
SPLIT_ASSIGNMENT_DIGEST = "7b21d3af455475f7bb29198081a2ef2e651cffaac6149fd27741863c765b4efc"
PARENT_CLOSEOUT_SHA256 = "dde7f4654674f776d2f2b0a687128019e477eea7b29fb8d69f01d621aa6c5887"
CHECKPOINT_SHA256 = "6f42037d57a4e3de3a10ac0ab909d66a3415864a19433c29204a975f4efa4088"
MODEL_SOURCE_SHA256 = "3fa227af396464a7f07ac773f92e9cdb746e0fa6ae63adef711c765a02c3d4cd"
MODEL_LAYERS_SHA256 = "ac8a05bd626d9adda16308dedb9466f36d7ff44cfb666f64e7e14ddf8cdf43a4"
PARAMETER_COUNT = 8_630_665
TILE_SIZE = 32
REGION_FRACTION = 0.20
TILE_FRACTION_MARGIN = 0.20
SCENE_REPEAT_VARIANTS = 3
PRIMARY_PREVALENCE_MARGIN = 0.20
GLOBAL_COMPETENCE_MARGIN = 0.80
UNDER_ALPHA_MAX = 0.80
UNDER_RESIDUAL_RATIO_MIN = 0.25
OVERSHOOT_ALPHA_MIN = 1.05
ABSOLUTE_ERROR_MARGIN = 1.0 / 255.0
LOW_HARM_RATIO_MIN = 1.10
BOOTSTRAP_SEED = 20260723
BOOTSTRAP_RESAMPLES = 20_000
EPSILON = 1e-8


def image_files(directory: Path) -> list[Path]:
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def supported_directories(root: Path, names: tuple[str, ...]) -> list[Path]:
    return [root / name for name in names if (root / name).is_dir()]


def selected_label(image_name: str, label_dir: Path) -> Path | None:
    stem, extension = os.path.splitext(image_name)
    names = [image_name]
    if "_" in stem:
        prefix = stem.split("_", 1)[0]
        names.extend((f"{prefix}{extension}", f"{prefix}.png"))
    seen: set[Path] = set()
    for name in names:
        candidate = label_dir / name
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            return candidate
    return None


def canonical_rgb_digest(width: int, height: int, payload: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(width.to_bytes(8, "big"))
    digest.update(height.to_bytes(8, "big"))
    digest.update(payload)
    return digest.hexdigest()


def inspect_clear(path: Path) -> dict[str, Any]:
    try:
        with Image.open(path) as source:
            rgb = source.convert("RGB")
            rgb.load()
            width, height = rgb.size
            digest = canonical_rgb_digest(width, height, rgb.tobytes())
        return {
            "path": path, "width": width, "height": height,
            "digest": digest, "error": None,
        }
    except Exception as exc:
        return {
            "path": path, "width": None, "height": None,
            "digest": None, "error": f"{type(exc).__name__}: {exc}"[:240],
        }


def image_array(path: Path) -> np.ndarray:
    with Image.open(path) as source:
        array = np.asarray(source.convert("RGB"), dtype=np.float32) / 255.0
    if array.ndim != 3 or array.shape[2] != 3 or min(array.shape[:2]) <= TILE_SIZE:
        raise RuntimeError(f"unsupported image shape for {path.name}: {array.shape}")
    return array


def assignment_digest(scene_digests: Iterable[str], development: set[str]) -> str:
    lines = [
        f"{digest},{'internal_development' if digest in development else 'training'}"
        for digest in scene_digests
    ]
    return hashlib.sha256("\n".join(sorted(lines)).encode("utf-8")).hexdigest()


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
        return {
            "count": 0, "mean": None, "median": None, "q10": None,
            "q25": None, "q75": None, "q90": None, "min": None, "max": None,
        }
    if not np.isfinite(array).all():
        raise RuntimeError("aggregate received non-finite values")
    return {
        "count": int(array.size), "mean": float(np.mean(array)),
        "median": float(np.median(array)), "q10": float(np.quantile(array, 0.10)),
        "q25": float(np.quantile(array, 0.25)), "q75": float(np.quantile(array, 0.75)),
        "q90": float(np.quantile(array, 0.90)), "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def paired_bootstrap(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise RuntimeError("paired bootstrap requires finite scene values")
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    batch_size = 1000
    for start in range(0, BOOTSTRAP_RESAMPLES, batch_size):
        stop = min(start + batch_size, BOOTSTRAP_RESAMPLES)
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
    pad_height = (-height) % TILE_SIZE
    pad_width = (-width) % TILE_SIZE
    padded = np.pad(
        array, ((0, pad_height), (0, pad_width), (0, 0)), mode="reflect",
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


def variant_measurement(
    hazy: np.ndarray, clear: np.ndarray, prediction: np.ndarray,
) -> dict[str, float | bool | int]:
    if hazy.shape != clear.shape or prediction.shape != clear.shape:
        raise RuntimeError("paired image shapes changed")
    hazy_tiles = image_tiles(hazy)
    clear_tiles = image_tiles(clear)
    prediction_tiles = image_tiles(prediction)
    direction = clear_tiles - hazy_tiles
    change = prediction_tiles - hazy_tiles
    input_squared = np.mean(direction * direction, axis=(2, 3, 4))
    output_squared = np.mean(
        (prediction_tiles - clear_tiles) ** 2, axis=(2, 3, 4),
    )
    direction_norm = np.sum(direction * direction, axis=(2, 3, 4))
    alpha = np.sum(change * direction, axis=(2, 3, 4)) / (direction_norm + EPSILON)
    demand_rms = np.sqrt(input_squared)
    residual_ratio = output_squared / (input_squared + EPSILON)
    projected_overshoot = np.maximum(alpha - 1.0, 0.0) * demand_rms
    under = (alpha <= UNDER_ALPHA_MAX) & (residual_ratio >= UNDER_RESIDUAL_RATIO_MIN)
    overshoot = (
        (alpha >= OVERSHOOT_ALPHA_MIN)
        & (projected_overshoot >= ABSOLUTE_ERROR_MARGIN)
    )
    harm = (
        (output_squared >= LOW_HARM_RATIO_MIN * input_squared)
        & ((output_squared - input_squared) >= ABSOLUTE_ERROR_MARGIN ** 2)
    )
    if not all(np.isfinite(item).all() for item in (
        input_squared, output_squared, alpha, residual_ratio, projected_overshoot,
    )):
        raise RuntimeError("non-finite local measurement")
    low_indices, high_indices = region_indices(input_squared)
    flat_input = input_squared.reshape(-1)
    flat_output = output_squared.reshape(-1)
    flat_alpha = alpha.reshape(-1)
    flat_under = under.reshape(-1)
    flat_overshoot = overshoot.reshape(-1)
    flat_harm = harm.reshape(-1)
    input_mse = float(np.mean(flat_input))
    output_mse = float(np.mean(flat_output))
    all_overshoot_fraction = float(np.mean(flat_overshoot))
    high_under_fraction = float(np.mean(flat_under[high_indices]))
    low_harm_fraction = float(np.mean(flat_harm[low_indices]))
    return {
        "height": int(clear.shape[0]), "width": int(clear.shape[1]),
        "tile_count": int(flat_input.size),
        "input_mse": input_mse, "output_mse": output_mse,
        "input_psnr": -10.0 * math.log10(max(input_mse, 1e-12)),
        "output_psnr": -10.0 * math.log10(max(output_mse, 1e-12)),
        "global_improvement": output_mse < input_mse,
        "high_demand_input_mse": float(np.mean(flat_input[high_indices])),
        "low_demand_input_mse": float(np.mean(flat_input[low_indices])),
        "high_demand_output_mse": float(np.mean(flat_output[high_indices])),
        "low_demand_output_mse": float(np.mean(flat_output[low_indices])),
        "high_alpha_mean": float(np.mean(flat_alpha[high_indices])),
        "low_alpha_mean": float(np.mean(flat_alpha[low_indices])),
        "high_under_recovery_fraction": high_under_fraction,
        "all_material_overshoot_fraction": all_overshoot_fraction,
        "high_material_overshoot_fraction": float(np.mean(flat_overshoot[high_indices])),
        "low_material_overshoot_fraction": float(np.mean(flat_overshoot[low_indices])),
        "low_demand_harm_fraction": low_harm_fraction,
        "variant_under_recovery": high_under_fraction >= TILE_FRACTION_MARGIN,
        "variant_signed_overshoot": all_overshoot_fraction >= TILE_FRACTION_MARGIN,
        "variant_low_demand_harm": low_harm_fraction >= TILE_FRACTION_MARGIN,
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
        "variant_count": len(records),
        "globally_competent": mean_key(records, "output_mse") < mean_key(records, "input_mse"),
        "input_psnr": mean_key(records, "input_psnr"),
        "output_psnr": mean_key(records, "output_psnr"),
        "high_alpha_mean": mean_key(records, "high_alpha_mean"),
        "low_alpha_mean": mean_key(records, "low_alpha_mean"),
        "high_under_recovery_fraction": mean_key(records, "high_under_recovery_fraction"),
        "all_material_overshoot_fraction": mean_key(records, "all_material_overshoot_fraction"),
        "high_material_overshoot_fraction": mean_key(records, "high_material_overshoot_fraction"),
        "low_material_overshoot_fraction": mean_key(records, "low_material_overshoot_fraction"),
        "low_demand_harm_fraction": mean_key(records, "low_demand_harm_fraction"),
        "under_recovery_variant_count": under_count,
        "signed_overshoot_variant_count": overshoot_count,
        "low_demand_harm_variant_count": harm_count,
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
    if Path(module.__file__).resolve() != model_source.resolve():
        raise RuntimeError("official model import resolved to a different file")
    layers_module = sys.modules.get("Dehazing.ITS.models.layers")
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
        raise RuntimeError("official Haze4K base parameter count changed")
    model.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("official baseline was not fully frozen")
    model.to(context.device)
    model.eval()
    return torch, model


def infer(torch, model, hazy: np.ndarray, device: str) -> np.ndarray:
    import torch.nn.functional as functional

    tensor = torch.from_numpy(np.transpose(hazy, (2, 0, 1)).copy()).unsqueeze(0).to(device)
    height, width = tensor.shape[-2:]
    pad_height = (-height) % 32
    pad_width = (-width) % 32
    padded = functional.pad(tensor, (0, pad_width, 0, pad_height), mode="reflect")
    with torch.inference_mode():
        outputs = model(padded)
        if not isinstance(outputs, list) or len(outputs) != 3:
            raise RuntimeError("official three-scale output contract changed")
        prediction = outputs[2][:, :, :height, :width].clamp(0.0, 1.0)
        if not bool(torch.isfinite(prediction).all().item()):
            raise RuntimeError("official model produced non-finite output")
    return np.transpose(prediction.squeeze(0).cpu().numpy(), (1, 2, 0))


def finalize(
    context, *, integrity: dict[str, bool], scenes: list[dict[str, Any]],
    variants: list[dict[str, Any]], failures: list[dict[str, str]],
    observed_assignment_digest: str | None,
) -> dict[str, Any]:
    coverage_complete = (
        all(integrity.values()) and len(scenes) == DEVELOPMENT_SCENES
        and len(variants) == DEVELOPMENT_VARIANTS and not failures
    )
    stable_counts = {
        "any": sum(bool(scene["stable_any_local_error"]) for scene in scenes),
        "under_recovery": sum(bool(scene["stable_under_recovery"]) for scene in scenes),
        "signed_overshoot": sum(bool(scene["stable_signed_overshoot"]) for scene in scenes),
        "low_demand_harm": sum(bool(scene["stable_low_demand_harm"]) for scene in scenes),
    }
    prevalence = {
        key: wilson(count, DEVELOPMENT_SCENES) for key, count in stable_counts.items()
    }
    competence_count = sum(bool(scene["globally_competent"]) for scene in scenes)
    competence = wilson(competence_count, DEVELOPMENT_SCENES)
    competence_pass = float(competence["lower"]) >= GLOBAL_COMPETENCE_MARGIN
    primary = prevalence["any"]

    if not coverage_complete:
        state = "COMPLETED_INCONCLUSIVE"
        decision = "HAZE4K_TRAIN_BASELINE_LOCAL_ERROR_INCONCLUSIVE"
        authorizes = "HAZE4K_TRAIN_LOCAL_ERROR_SUPPLEMENT_ONLY"
        gate_reasons = [
            "parent authorization, train-only split identity, complete 150-scene coverage, strict inference, or finite measurement failed"
        ]
    elif not competence_pass:
        state = "COMPLETED_INCONCLUSIVE"
        decision = "HAZE4K_TRAIN_BASELINE_LOCAL_ERROR_INCONCLUSIVE"
        authorizes = "HAZE4K_TRAIN_LOCAL_ERROR_SUPPLEMENT_ONLY"
        gate_reasons = [
            "fixed official baseline did not meet the frozen scene-level global-competence prerequisite"
        ]
    elif float(primary["lower"]) >= PRIMARY_PREVALENCE_MARGIN:
        state = "COMPLETED_GATE_PASS"
        decision = "HAZE4K_TRAIN_BASELINE_LOCAL_ERROR_PASS"
        authorizes = "HAZE4K_TRAIN_BOUNDED_LOCAL_ACTION_ORACLE"
        gate_reasons = [
            "the 95 percent lower bound for stable material local-error prevalence met the frozen 0.20 margin"
        ]
    elif float(primary["upper"]) < PRIMARY_PREVALENCE_MARGIN:
        state = "COMPLETED_GATE_FAIL"
        decision = "HAZE4K_TRAIN_BASELINE_LOCAL_ERROR_FAIL"
        authorizes = "NONE"
        gate_reasons = [
            "the 95 percent upper bound for stable material local-error prevalence was below the frozen 0.20 margin"
        ]
    else:
        state = "COMPLETED_INCONCLUSIVE"
        decision = "HAZE4K_TRAIN_BASELINE_LOCAL_ERROR_INCONCLUSIVE"
        authorizes = "HAZE4K_TRAIN_LOCAL_ERROR_SUPPLEMENT_ONLY"
        gate_reasons = [
            "the stable material local-error prevalence interval crossed the frozen 0.20 margin"
        ]

    demand_differences = [
        float(scene["high_material_overshoot_fraction"])
        - float(scene["low_material_overshoot_fraction"])
        for scene in scenes
    ]
    psnr_gains = [
        float(scene["output_psnr"]) - float(scene["input_psnr"]) for scene in scenes
    ]
    summary = {
        "schema_version": 1,
        "route_id": context.route_id, "operation_id": context.operation_id,
        "run_id": context.run_id,
        "scope": "fixed official Haze4K ConvIR-B on 150 train-derived internal-development clear scenes and 600 nested haze variants; official test excluded",
        "identity_and_coverage": {
            "integrity_checks": integrity,
            "observed_split_assignment_digest": observed_assignment_digest,
            "expected_split_assignment_digest": SPLIT_ASSIGNMENT_DIGEST,
            "completed_clear_scenes": len(scenes),
            "completed_nested_variants": len(variants),
            "failure_count": len(failures),
            "failures": failures[:20],
        },
        "measurement": {
            "tile_size_pixels": TILE_SIZE,
            "tile_stride_pixels": TILE_SIZE,
            "edge_rule": "reflect-pad bottom and right to a multiple of 32",
            "demand_score": "paired input-versus-GT RGB MSE per tile",
            "high_and_low_region_fraction": REGION_FRACTION,
            "signed_projection_alpha": "dot(prediction-input, GT-input)/(squared_norm(GT-input)+1e-8)",
            "under_recovery": {
                "alpha_max": UNDER_ALPHA_MAX,
                "output_to_input_error_ratio_min": UNDER_RESIDUAL_RATIO_MIN,
                "evaluated_region": "highest-demand 20 percent of tiles",
            },
            "signed_overshoot": {
                "alpha_min": OVERSHOOT_ALPHA_MIN,
                "projected_absolute_margin": ABSOLUTE_ERROR_MARGIN,
                "evaluated_region": "all tiles; high/low demand contrast is secondary",
            },
            "low_demand_harm": {
                "output_to_input_error_ratio_min": LOW_HARM_RATIO_MIN,
                "absolute_mse_increase_min": ABSOLUTE_ERROR_MARGIN ** 2,
                "evaluated_region": "lowest-demand 20 percent of tiles",
            },
            "variant_material_fraction_margin": TILE_FRACTION_MARGIN,
            "scene_repeatability_rule": "the same construct is material in at least three of four nested haze variants",
            "scene_aggregation": "arithmetic mean within clear scene followed by equal-weight scene inference",
        },
        "primary_estimand": {
            "stable_any_material_local_error_prevalence": primary,
            "material_prevalence_margin": PRIMARY_PREVALENCE_MARGIN,
            "independent_unit": "canonical_clear_scene",
        },
        "construct_prevalence": prevalence,
        "global_competence": {
            "scene_prevalence": competence,
            "required_lower_95": GLOBAL_COMPETENCE_MARGIN,
            "passed": competence_pass,
            "scene_mean_psnr_gain": paired_bootstrap(psnr_gains) if scenes else None,
        },
        "demand_contrast_secondary": {
            "paired_scene_mean_high_minus_low_material_overshoot_fraction": (
                paired_bootstrap(demand_differences) if scenes else None
            ),
            "direction_was_not_a_gate": True,
        },
        "secondary_scene_aggregates": {
            "input_psnr": aggregate(scene["input_psnr"] for scene in scenes),
            "output_psnr": aggregate(scene["output_psnr"] for scene in scenes),
            "high_alpha_mean": aggregate(scene["high_alpha_mean"] for scene in scenes),
            "low_alpha_mean": aggregate(scene["low_alpha_mean"] for scene in scenes),
            "high_under_recovery_fraction": aggregate(
                scene["high_under_recovery_fraction"] for scene in scenes
            ),
            "all_material_overshoot_fraction": aggregate(
                scene["all_material_overshoot_fraction"] for scene in scenes
            ),
            "high_material_overshoot_fraction": aggregate(
                scene["high_material_overshoot_fraction"] for scene in scenes
            ),
            "low_material_overshoot_fraction": aggregate(
                scene["low_material_overshoot_fraction"] for scene in scenes
            ),
            "low_demand_harm_fraction": aggregate(
                scene["low_demand_harm_fraction"] for scene in scenes
            ),
        },
        "terminal": {
            "state": state, "decision": decision, "authorizes": authorizes,
            "gate_reasons": gate_reasons,
        },
        "limitations": [
            "This is Haze4K train-derived development-screening evidence for one fixed official checkpoint, not official-test confirmation or deployment evidence.",
            "GT-derived demand and error labels are measurement-only quantities and are unavailable as deployment inputs.",
            "Signed RGB projection and MSE materiality are operational constructs, not direct proofs of perceptual over-dehazing, haze density, or causal mechanism.",
            "The high-versus-low demand contrast is reported without assuming monotonicity or a required direction.",
            "Passing authorizes only a non-deployable bounded local-action oracle, not proxy fitting or module design.",
        ],
        "marker": "HAZE4K_TRAIN_BASELINE_LOCAL_ERROR_V1_COMPLETE",
    }
    atomic_json(output_file(context, "haze4k_baseline_local_error_summary.json"), summary)

    rows = []
    for name, interval in (
        ("stable_any_local_error", prevalence["any"]),
        ("stable_under_recovery", prevalence["under_recovery"]),
        ("stable_signed_overshoot", prevalence["signed_overshoot"]),
        ("stable_low_demand_harm", prevalence["low_demand_harm"]),
        ("global_competence", competence),
    ):
        rows.append({
            "stratum": name, "scenes": interval["total"],
            "positive_scenes": interval["successes"],
            "estimate": interval["estimate"],
            "lower_95": interval["lower"], "upper_95": interval["upper"],
        })
    with output_file(context, "haze4k_baseline_local_error_strata.csv").open(
        "w", encoding="utf-8", newline="",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {
        "state": state, "decision": decision, "authorizes": authorizes,
        "details": {
            "summary_file": "haze4k_baseline_local_error_summary.json",
            "strata_file": "haze4k_baseline_local_error_strata.csv",
            "independent_clear_scenes": len(scenes),
            "nested_haze_variants": len(variants),
            "stable_local_error_prevalence": primary,
            "global_competence": competence,
            "gate_reasons": gate_reasons,
            "training_occurred": False,
            "official_test_accessed": False,
        },
    }


def contract(context_path: Path) -> None:
    context = load_context(context_path, "contract")
    prepare_phase_output(context)
    if context.device != "cuda" or any(context.protected_data_permissions.values()):
        raise RuntimeError("baseline local-error contract requires CUDA and no protected permission")
    if "haze4k_train" in context.assets:
        raise RuntimeError("scientific Haze4K train data must be absent from contract phase")
    torch, model = load_official_model(context)
    height, width = 256, 320
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    clear = np.stack((xx / width, yy / height, (xx + yy) / (width + height)), axis=-1)
    transmission = np.exp(-0.18 * (1.0 + xx / width + yy / height)).astype(np.float32)
    hazy = clear * transmission[..., None] + 0.90 * (1.0 - transmission[..., None])
    hazy = hazy.astype(np.float32)
    prediction = infer(torch, model, hazy, context.device)
    measured = variant_measurement(hazy, clear, prediction)
    unchanged = variant_measurement(hazy, clear, hazy)
    exact = variant_measurement(hazy, clear, clear)
    partial = variant_measurement(hazy, clear, hazy + 0.40 * (clear - hazy))
    extended = variant_measurement(hazy, clear, hazy + 10.0 * (clear - hazy))
    opposite = variant_measurement(
        hazy, clear, hazy - (10.0 / 255.0) * np.sign(clear - hazy),
    )
    reference_scene = scene_measurement([
        {**partial, "variant_under_recovery": True},
        {**partial, "variant_under_recovery": True},
        {**partial, "variant_under_recovery": True},
        {**exact, "variant_under_recovery": False},
    ])
    numeric = [
        float(value) for value in measured.values()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    checks = {
        "strict_official_checkpoint_load": True,
        "official_parameter_count": True,
        "all_parameters_frozen": not any(parameter.requires_grad for parameter in model.parameters()),
        "three_scale_finite_forward": bool(np.isfinite(prediction).all()),
        "measurement_finalizer_finite": bool(numeric and np.isfinite(numeric).all()),
        "unchanged_reference": bool(
            abs(float(unchanged["output_mse"]) - float(unchanged["input_mse"])) < 1e-12
            and unchanged["all_material_overshoot_fraction"] == 0.0
        ),
        "exact_gt_reference": bool(
            exact["output_mse"] == 0.0
            and exact["all_material_overshoot_fraction"] == 0.0
            and exact["high_under_recovery_fraction"] == 0.0
        ),
        "partial_recovery_reference": bool(partial["high_under_recovery_fraction"] == 1.0),
        "strong_overshoot_reference": bool(
            extended["all_material_overshoot_fraction"] >= 0.99
        ),
        "opposite_low_demand_harm_reference": bool(opposite["low_demand_harm_fraction"] >= 0.99),
        "scene_repeatability_reference": bool(
            reference_scene["stable_under_recovery"]
            and reference_scene["under_recovery_variant_count"] == 3
        ),
        "wilson_direction_reference": bool(
            wilson(40, 150)["lower"] < wilson(41, 150)["lower"]
        ),
        "bootstrap_direction_reference": bool(
            abs(float(paired_bootstrap([0.10, 0.20, 0.30, 0.40])["estimate"]) - 0.25) < 1e-12
        ),
    }
    atomic_json(output_file(context, "haze4k_baseline_local_error_contract_details.json"), {
        "parameter_count": PARAMETER_COUNT,
        "fixture": {"batch": 1, "channels": 3, "height": height, "width": width},
        "measurement_keys": sorted(measured),
        "bootstrap": {"seed": BOOTSTRAP_SEED, "resamples": BOOTSTRAP_RESAMPLES},
    })
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
    train_root = asset_path(context, "haze4k_train", kind="directory")
    parent_path = asset_path(context, "prior_scene_split_closeout", kind="file")
    if context.assets["prior_scene_split_closeout"].sha256 != PARENT_CLOSEOUT_SHA256:
        raise RuntimeError("parent scene-split closeout identity changed")
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    parent_ok = (
        parent.get("state") == "COMPLETED_GATE_PASS"
        and parent.get("decision") == "HAZE4K_TRAIN_SCENE_SPLIT_PASS"
        and parent.get("authorizes") == "HAZE4K_TRAIN_BASELINE_LOCAL_ERROR_MEASUREMENT"
        and parent.get("details", {}).get("split_assignment_digest") == SPLIT_ASSIGNMENT_DIGEST
        and parent.get("details", {}).get("official_test_accessed") is False
        and parent.get("details", {}).get("model_or_checkpoint_accessed") is False
    )
    scope_ok = (
        train_root.name.lower() == "train"
        and "test" not in {part.lower() for part in train_root.parts}
        and not any(context.protected_data_permissions.values())
    )
    input_dirs = supported_directories(train_root, INPUT_DIRECTORY_NAMES) if scope_ok else []
    label_dirs = supported_directories(train_root, LABEL_DIRECTORY_NAMES) if scope_ok else []
    directory_ok = len(input_dirs) == 1 and len(label_dirs) == 1
    hazy_paths = image_files(input_dirs[0]) if directory_ok else []
    clear_paths = image_files(label_dirs[0]) if directory_ok else []
    pairing = {}
    if directory_ok:
        for hazy_path in hazy_paths:
            clear_path = selected_label(hazy_path.name, label_dirs[0])
            if clear_path is not None:
                pairing[hazy_path] = clear_path
    write_workload_progress(context, completed_units=1, stage="train_scope_and_pairing")

    workers = int(os.environ.get("CONVIR_ROUTE_HAZE4K_MEASURE_WORKERS", "8"))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        clear_info = list(executor.map(inspect_clear, clear_paths, chunksize=16))
    clear_by_path = {item["path"]: item for item in clear_info}
    decode_failures = sum(item["error"] is not None for item in clear_info)
    all_digest_to_paths: dict[str, list[Path]] = defaultdict(list)
    for item in clear_info:
        if item["digest"] is not None:
            all_digest_to_paths[item["digest"]].append(item["path"])
    scene_digests = sorted(all_digest_to_paths)
    ranked = sorted(
        scene_digests,
        key=lambda digest: (
            hashlib.sha256(f"{SPLIT_SALT}|{digest}".encode("utf-8")).hexdigest(), digest,
        ),
    )
    development = set(ranked[:DEVELOPMENT_SCENES])
    observed_assignment = assignment_digest(scene_digests, development) if scene_digests else None
    variants_by_scene: dict[str, list[Path]] = defaultdict(list)
    for hazy_path, clear_path in pairing.items():
        item = clear_by_path.get(clear_path)
        if item and item["digest"] in development:
            variants_by_scene[item["digest"]].append(hazy_path)
    variant_histogram = Counter(len(paths) for paths in variants_by_scene.values())
    split_ok = (
        len(hazy_paths) == EXPECTED_FILES and len(clear_paths) == EXPECTED_FILES
        and len(pairing) == EXPECTED_FILES and decode_failures == 0
        and len(scene_digests) == EXPECTED_SCENES
        and len(development) == DEVELOPMENT_SCENES
        and observed_assignment == SPLIT_ASSIGNMENT_DIGEST
        and variant_histogram == {VARIANTS_PER_SCENE: DEVELOPMENT_SCENES}
    )
    write_workload_progress(context, completed_units=2, stage="reproduce_internal_development_split")

    scene_records: list[dict[str, Any]] = []
    variant_records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    if parent_ok and scope_ok and split_ok:
        torch, model = load_official_model(context)
        attempted = 0
        for digest in sorted(development):
            records = []
            representative = sorted(all_digest_to_paths[digest])[0]
            try:
                clear = image_array(representative)
            except Exception as exc:
                failures.append({
                    "scene": digest[:16], "variant": "clear",
                    "reason": str(exc)[:512],
                })
                continue
            for hazy_path in sorted(variants_by_scene[digest]):
                attempted += 1
                try:
                    hazy = image_array(hazy_path)
                    if hazy.shape != clear.shape:
                        raise RuntimeError(
                            f"paired shape mismatch: clear={clear.shape}, hazy={hazy.shape}"
                        )
                    prediction = infer(torch, model, hazy, context.device)
                    measured = variant_measurement(hazy, clear, prediction)
                    records.append(measured)
                    variant_records.append(measured)
                except Exception as exc:
                    failures.append({
                        "scene": digest[:16], "variant": hazy_path.name,
                        "reason": str(exc)[:512],
                    })
                if attempted % 5 == 0:
                    write_workload_progress(
                        context, completed_units=2 + attempted, stage="official_haze4k_inference",
                    )
            if len(records) == VARIANTS_PER_SCENE:
                scene_records.append(scene_measurement(records))

    integrity = {
        "parent_split_authorization": parent_ok,
        "train_only_asset_scope": scope_ok,
        "complete_pairing_and_clear_decode": directory_ok and len(pairing) == EXPECTED_FILES and decode_failures == 0,
        "frozen_split_reproduced": split_ok,
        "complete_internal_development_inference": (
            len(scene_records) == DEVELOPMENT_SCENES
            and len(variant_records) == DEVELOPMENT_VARIANTS and not failures
        ),
        "official_test_excluded": True,
        "no_training": True,
    }
    result = finalize(
        context, integrity=integrity, scenes=scene_records, variants=variant_records,
        failures=failures, observed_assignment_digest=observed_assignment,
    )
    write_workload_progress(context, completed_units=603, stage="scene_level_finalize")
    write_run_result(
        context, state=result["state"], decision=result["decision"],
        authorizes=result["authorizes"], details=result["details"],
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
