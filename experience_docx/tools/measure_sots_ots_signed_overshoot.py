#!/usr/bin/env python3
"""Measure signed high-demand paired-GT overshoot of official OTS ConvIR-B."""

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
    asset_path,
    atomic_json,
    load_context,
    output_file,
    prepare_phase_output,
    write_contract_result,
    write_run_result,
    write_workload_progress,
)


EXPECTED_GT = 492
EXPECTED_HAZY = 500
EXPECTED_GROUPS = 492
TOTAL_UNITS = 1492
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
CHECKPOINT_SHA256 = "dc28713ad92af0a2594b1964451602614e1b24a1e898647a6cf41c94e9e533e6"
DATASET_SHA256 = "ac6921cc9a2afd32c9976465fd9a50555dce2f217f8f4ddb0614c907f681e5b2"
PARENT_CLOSEOUT_SHA256 = "8e1a4a6c2e4dc0572977fac58db20e1e9dbd6bcdea353bcee808e72984245943"
MODEL_SOURCE_SHA256 = "140cec905c64f429359d675fff46d9ae77465d7dc972879c458095beb0d63a6d"
MODEL_LAYERS_SHA256 = "023e27cb28b9ba13663dda0cd7da181e2f28f60a67c833bc1933e6d5b1ebd580"
PARAMETER_COUNT = 8630665
TILE_SIZE = 32
REGION_FRACTION = 0.20
ALPHA_MARGIN = 1.05
ABSOLUTE_OVERSHOOT_MARGIN = 1.0 / 255.0
HIGH_TILE_FRACTION_MARGIN = 0.20
OVERSHOOT_PREVALENCE_MARGIN = 0.20
GLOBAL_COMPETENCE_MARGIN = 0.80
EPSILON = 1e-8


def image_files(path: Path) -> list[Path]:
    return sorted(
        item for item in path.iterdir()
        if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def aggregate_dataset_digest(
    root: Path, files: list[Path], context,
) -> str:
    digest = hashlib.sha256()
    for index, path in enumerate(files, start=1):
        identity = sha256_file(path)
        relative = path.relative_to(root).as_posix()
        digest.update(f"{relative}\t{path.stat().st_size}\t{identity}\n".encode("utf-8"))
        if index % 25 == 0:
            write_workload_progress(
                context, completed_units=index, stage="dataset_identity",
            )
    return digest.hexdigest()


def wilson(successes: int, total: int, z_value: float = 1.96) -> dict[str, float | int]:
    if total <= 0 or successes < 0 or successes > total:
        raise ValueError("invalid Wilson inputs")
    proportion = successes / total
    denominator = 1.0 + z_value * z_value / total
    center = (proportion + z_value * z_value / (2.0 * total)) / denominator
    half_width = z_value * math.sqrt(
        proportion * (1.0 - proportion) / total
        + z_value * z_value / (4.0 * total * total)
    ) / denominator
    return {
        "successes": successes,
        "total": total,
        "estimate": proportion,
        "lower": max(0.0, center - half_width),
        "upper": min(1.0, center + half_width),
    }


def aggregate(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {
            "count": 0, "mean": None, "median": None,
            "q25": None, "q75": None, "min": None, "max": None,
        }
    if not np.isfinite(array).all():
        raise RuntimeError("aggregate received non-finite values")
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def image_array(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    if array.ndim != 3 or array.shape[2] != 3 or min(array.shape[:2]) <= TILE_SIZE:
        raise RuntimeError(f"unsupported image shape for {path.name}: {array.shape}")
    return array


def pad_image(array: np.ndarray) -> np.ndarray:
    height, width = array.shape[:2]
    pad_height = (-height) % TILE_SIZE
    pad_width = (-width) % TILE_SIZE
    return np.pad(
        array, ((0, pad_height), (0, pad_width), (0, 0)), mode="reflect",
    )


def image_tiles(array: np.ndarray) -> np.ndarray:
    padded = pad_image(array)
    grid_height = padded.shape[0] // TILE_SIZE
    grid_width = padded.shape[1] // TILE_SIZE
    return padded.reshape(
        grid_height, TILE_SIZE, grid_width, TILE_SIZE, 3,
    ).transpose(0, 2, 1, 3, 4)


def region_indices(score_tiles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = score_tiles.reshape(-1)
    region_count = max(1, int(math.floor(flat.size * REGION_FRACTION)))
    if region_count * 2 > flat.size:
        raise RuntimeError("too few tiles for disjoint low and high regions")
    order = np.argsort(flat, kind="mergesort")
    return order[:region_count], order[-region_count:]


def signed_tile_metrics(
    hazy: np.ndarray, clear: np.ndarray, prediction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hazy_tiles = image_tiles(hazy).astype(np.float64)
    clear_tiles = image_tiles(clear).astype(np.float64)
    prediction_tiles = image_tiles(prediction).astype(np.float64)
    direction = clear_tiles - hazy_tiles
    change = prediction_tiles - hazy_tiles
    denominator = np.sum(direction * direction, axis=(2, 3, 4))
    numerator = np.sum(change * direction, axis=(2, 3, 4))
    alpha = numerator / (denominator + EPSILON)
    demand_mse = denominator / float(TILE_SIZE * TILE_SIZE * 3)
    demand_rms = np.sqrt(demand_mse)
    overshoot_magnitude = np.maximum(alpha - 1.0, 0.0) * demand_rms
    material = (
        (alpha >= ALPHA_MARGIN)
        & (overshoot_magnitude >= ABSOLUTE_OVERSHOOT_MARGIN)
    )
    return demand_mse, alpha, overshoot_magnitude, material


def variant_measurement(
    hazy: np.ndarray, clear: np.ndarray, prediction: np.ndarray,
    output_ssim: float,
) -> dict[str, float | bool | int]:
    if hazy.shape != clear.shape or prediction.shape != clear.shape:
        raise RuntimeError("paired image shapes changed")
    input_error_map = np.mean((hazy - clear) ** 2, axis=2, dtype=np.float64)
    output_error_map = np.mean((prediction - clear) ** 2, axis=2, dtype=np.float64)
    if not np.isfinite(input_error_map).all() or not np.isfinite(output_error_map).all():
        raise RuntimeError("non-finite paired error map")
    input_mse = float(np.mean(input_error_map))
    output_mse = float(np.mean(output_error_map))
    demand_mse, alpha, overshoot_magnitude, material = signed_tile_metrics(
        hazy, clear, prediction,
    )
    if not all(np.isfinite(item).all() for item in (
        demand_mse, alpha, overshoot_magnitude,
    )):
        raise RuntimeError("non-finite signed tile metric")
    low_indices, high_indices = region_indices(demand_mse)
    flat_alpha = alpha.reshape(-1)
    flat_overshoot = overshoot_magnitude.reshape(-1)
    flat_material = material.reshape(-1)
    high_material_fraction = float(np.mean(flat_material[high_indices]))
    low_material_fraction = float(np.mean(flat_material[low_indices]))
    return {
        "height": int(clear.shape[0]),
        "width": int(clear.shape[1]),
        "tile_count": int(demand_mse.size),
        "high_tile_count": int(high_indices.size),
        "input_mse": input_mse,
        "output_mse": output_mse,
        "input_psnr": -10.0 * math.log10(max(input_mse, 1e-12)),
        "output_psnr": -10.0 * math.log10(max(output_mse, 1e-12)),
        "output_ssim": float(output_ssim),
        "global_improvement": output_mse < input_mse,
        "high_demand_mse": float(np.mean(demand_mse.reshape(-1)[high_indices])),
        "low_demand_mse": float(np.mean(demand_mse.reshape(-1)[low_indices])),
        "high_alpha_mean": float(np.mean(flat_alpha[high_indices])),
        "high_alpha_median": float(np.median(flat_alpha[high_indices])),
        "low_alpha_mean": float(np.mean(flat_alpha[low_indices])),
        "low_alpha_median": float(np.median(flat_alpha[low_indices])),
        "high_material_overshoot_fraction": high_material_fraction,
        "low_material_overshoot_fraction": low_material_fraction,
        "high_projected_overshoot_mean": float(np.mean(flat_overshoot[high_indices])),
        "high_projected_overshoot_max": float(np.max(flat_overshoot[high_indices])),
        "variant_high_demand_overshoot": high_material_fraction >= HIGH_TILE_FRACTION_MARGIN,
    }


def load_official_model(context):
    import torch

    checkpoint = asset_path(context, "official_checkpoint", kind="file")
    model_source = asset_path(context, "model_source", kind="file")
    asset_path(context, "model_layers", kind="file")
    identities = {
        "official_checkpoint": CHECKPOINT_SHA256,
        "model_source": MODEL_SOURCE_SHA256,
        "model_layers": MODEL_LAYERS_SHA256,
    }
    for asset_id, expected in identities.items():
        if context.assets[asset_id].sha256 != expected:
            raise RuntimeError(f"verified identity changed for {asset_id}")

    ots_root = context.remote_repo / "Dehazing" / "OTS"
    if str(ots_root) not in sys.path:
        sys.path.insert(0, str(ots_root))
    from models.ConvIR import build_net

    module = sys.modules[build_net.__module__]
    if Path(module.__file__).resolve() != model_source.resolve():
        raise RuntimeError("official OTS model import resolved to a different file")
    model = build_net("base")
    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location="cpu")
    if not isinstance(state, dict) or not isinstance(state.get("model"), dict):
        raise RuntimeError("official checkpoint lacks state_dict['model']")
    model.load_state_dict(state["model"], strict=True)
    if sum(parameter.numel() for parameter in model.parameters()) != PARAMETER_COUNT:
        raise RuntimeError("official OTS base parameter count changed")
    model.to(context.device)
    model.eval()
    return torch, model


def infer(torch, model, hazy: np.ndarray, clear: np.ndarray, device: str):
    import torch.nn.functional as functional
    from pytorch_msssim import ssim

    input_tensor = torch.from_numpy(
        np.transpose(hazy, (2, 0, 1)).copy(),
    ).unsqueeze(0).to(device)
    clear_tensor = torch.from_numpy(
        np.transpose(clear, (2, 0, 1)).copy(),
    ).unsqueeze(0).to(device)
    height, width = input_tensor.shape[-2:]
    pad_height = (-height) % 32
    pad_width = (-width) % 32
    padded = functional.pad(
        input_tensor, (0, pad_width, 0, pad_height), mode="reflect",
    )
    with torch.inference_mode():
        outputs = model(padded)
        if not isinstance(outputs, list) or len(outputs) != 3:
            raise RuntimeError("official three-scale output contract changed")
        prediction = outputs[2][:, :, :height, :width].clamp(0.0, 1.0)
        if not bool(torch.isfinite(prediction).all().item()):
            raise RuntimeError("official model produced non-finite output")
        down_ratio = max(1, round(min(height, width) / 256))
        target = (max(1, height // down_ratio), max(1, width // down_ratio))
        output_ssim = float(ssim(
            functional.adaptive_avg_pool2d(prediction, target),
            functional.adaptive_avg_pool2d(clear_tensor, target),
            data_range=1.0,
            size_average=True,
        ).item())
    prediction_array = np.transpose(
        prediction.squeeze(0).cpu().numpy(), (1, 2, 0),
    )
    return prediction_array, output_ssim


def mean_key(records: list[dict[str, Any]], key: str) -> float:
    values = np.asarray([float(record[key]) for record in records], dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise RuntimeError(f"invalid group values for {key}")
    return float(np.mean(values))


def group_measurement(records: list[dict[str, Any]]) -> dict[str, Any]:
    group = {
        "variant_count": len(records),
        "input_mse": mean_key(records, "input_mse"),
        "output_mse": mean_key(records, "output_mse"),
        "input_psnr": mean_key(records, "input_psnr"),
        "output_psnr": mean_key(records, "output_psnr"),
        "output_ssim": mean_key(records, "output_ssim"),
        "high_demand_mse": mean_key(records, "high_demand_mse"),
        "low_demand_mse": mean_key(records, "low_demand_mse"),
        "high_alpha_mean": mean_key(records, "high_alpha_mean"),
        "high_alpha_median": mean_key(records, "high_alpha_median"),
        "low_alpha_mean": mean_key(records, "low_alpha_mean"),
        "low_alpha_median": mean_key(records, "low_alpha_median"),
        "high_material_overshoot_fraction": mean_key(
            records, "high_material_overshoot_fraction",
        ),
        "low_material_overshoot_fraction": mean_key(
            records, "low_material_overshoot_fraction",
        ),
        "high_projected_overshoot_mean": mean_key(
            records, "high_projected_overshoot_mean",
        ),
        "high_projected_overshoot_max": mean_key(
            records, "high_projected_overshoot_max",
        ),
    }
    group["globally_competent"] = group["output_mse"] < group["input_mse"]
    group["group_high_demand_overshoot"] = (
        group["high_material_overshoot_fraction"] >= HIGH_TILE_FRACTION_MARGIN
    )
    return group


def stratum_row(name: str, groups: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(groups)
    overshot = sum(bool(group["group_high_demand_overshoot"]) for group in groups)
    competent = sum(bool(group["globally_competent"]) for group in groups)
    return {
        "stratum": name,
        "source_groups": count,
        "nested_variants": sum(int(group["variant_count"]) for group in groups),
        "overshoot_group_count": overshot,
        "overshoot_group_rate": overshot / count if count else "",
        "global_competence_count": competent,
        "global_competence_rate": competent / count if count else "",
        "high_overshoot_fraction_mean": float(np.mean([group["high_material_overshoot_fraction"] for group in groups])) if groups else "",
        "high_overshoot_fraction_median": float(np.median([group["high_material_overshoot_fraction"] for group in groups])) if groups else "",
        "high_alpha_mean": float(np.mean([group["high_alpha_mean"] for group in groups])) if groups else "",
        "high_alpha_median": float(np.median([group["high_alpha_median"] for group in groups])) if groups else "",
        "low_alpha_median": float(np.median([group["low_alpha_median"] for group in groups])) if groups else "",
        "high_projected_overshoot_mean": float(np.mean([group["high_projected_overshoot_mean"] for group in groups])) if groups else "",
        "input_psnr_median": float(np.median([group["input_psnr"] for group in groups])) if groups else "",
        "output_psnr_median": float(np.median([group["output_psnr"] for group in groups])) if groups else "",
        "output_ssim_median": float(np.median([group["output_ssim"] for group in groups])) if groups else "",
    }


def finalize(
    context,
    *,
    parent_identity: bool,
    dataset_identity: bool,
    dataset_digest: str | None,
    pairing: dict[str, Any],
    groups: list[dict[str, Any]],
    variants: list[dict[str, Any]],
    failures: list[dict[str, str]],
    completed_units: int,
) -> dict[str, Any]:
    coverage_complete = (
        dataset_identity
        and len(groups) == EXPECTED_GROUPS
        and len(variants) == EXPECTED_HAZY
        and not failures
    )
    overshoot_count = sum(bool(group["group_high_demand_overshoot"]) for group in groups)
    competence_count = sum(bool(group["globally_competent"]) for group in groups)
    overshoot_interval = wilson(overshoot_count, EXPECTED_GROUPS)
    competence_interval = wilson(competence_count, EXPECTED_GROUPS)
    globally_competent = float(competence_interval["lower"]) >= GLOBAL_COMPETENCE_MARGIN

    if not parent_identity or not dataset_identity or not coverage_complete:
        state = "COMPLETED_INCONCLUSIVE"
        decision = "SOTS_HIGH_DEMAND_SIGNED_OVERSHOOT_INCONCLUSIVE"
        authorizes = "SOTS_SIGNED_OVERSHOOT_SUPPLEMENT_ONLY"
        gate_reasons = [
            "prior diagnostic, exact asset identity, pairing, finite inference, or complete source coverage failed"
        ]
    elif not globally_competent:
        state = "COMPLETED_INCONCLUSIVE"
        decision = "SOTS_HIGH_DEMAND_SIGNED_OVERSHOOT_INCONCLUSIVE"
        authorizes = "SOTS_SIGNED_OVERSHOOT_SUPPLEMENT_ONLY"
        gate_reasons = [
            "official OTS baseline lacked the frozen minimum global SOTS competence"
        ]
    elif float(overshoot_interval["lower"]) >= OVERSHOOT_PREVALENCE_MARGIN:
        state = "COMPLETED_GATE_PASS"
        decision = "SOTS_HIGH_DEMAND_SIGNED_OVERSHOOT_PASS"
        authorizes = "HIGH_DEMAND_OVERSHOOT_MECHANISM_REVIEW"
        gate_reasons = [
            "high-demand signed-overshoot prevalence lower 95 percent bound met the frozen material margin"
        ]
    elif float(overshoot_interval["upper"]) < OVERSHOOT_PREVALENCE_MARGIN:
        state = "COMPLETED_GATE_FAIL"
        decision = "SOTS_HIGH_DEMAND_SIGNED_OVERSHOOT_FAIL"
        authorizes = "NONE"
        gate_reasons = [
            "high-demand signed-overshoot prevalence upper 95 percent bound was below the frozen material margin"
        ]
    else:
        state = "COMPLETED_INCONCLUSIVE"
        decision = "SOTS_HIGH_DEMAND_SIGNED_OVERSHOOT_INCONCLUSIVE"
        authorizes = "SOTS_SIGNED_OVERSHOOT_SUPPLEMENT_ONLY"
        gate_reasons = [
            "high-demand signed-overshoot prevalence interval crossed the frozen material margin"
        ]

    summary = {
        "schema_version": 1,
        "route_id": context.route_id,
        "operation_id": context.operation_id,
        "run_id": context.run_id,
        "scope": "fixed official OTS ConvIR-B on development-screening SOTS-Outdoor paired images",
        "dataset_identity": {
            "prior_diagnostic_identity_match": parent_identity,
            "pairing_and_count_match": bool(pairing.get("pairing_and_count_match")),
            "aggregate_digest_match": dataset_digest == DATASET_SHA256,
            "observed_aggregate_sha256": dataset_digest,
            "expected_source_groups": EXPECTED_GROUPS,
            "completed_source_groups": len(groups),
            "expected_nested_variants": EXPECTED_HAZY,
            "completed_nested_variants": len(variants),
            "variant_count_histogram": pairing.get("variant_count_histogram", {}),
        },
        "measurement": {
            "tile_size_pixels": TILE_SIZE,
            "region_fraction": REGION_FRACTION,
            "projection_coefficient": "dot(prediction-input, GT-input)/(squared_norm(GT-input)+1e-8)",
            "alpha_margin": ALPHA_MARGIN,
            "absolute_projected_overshoot_margin": ABSOLUTE_OVERSHOOT_MARGIN,
            "high_tile_fraction_margin": HIGH_TILE_FRACTION_MARGIN,
            "variant_aggregation": "arithmetic mean within clear-source group before group classification",
        },
        "primary_estimand": {
            "high_demand_signed_overshoot_prevalence": overshoot_interval,
            "material_margin": OVERSHOOT_PREVALENCE_MARGIN,
        },
        "global_competence": {
            "source_group_prevalence": competence_interval,
            "required_lower_95": GLOBAL_COMPETENCE_MARGIN,
            "passed": globally_competent,
        },
        "secondary_group_aggregates": {
            "input_psnr": aggregate(group["input_psnr"] for group in groups),
            "output_psnr": aggregate(group["output_psnr"] for group in groups),
            "output_ssim": aggregate(group["output_ssim"] for group in groups),
            "high_demand_alpha_mean": aggregate(group["high_alpha_mean"] for group in groups),
            "high_demand_alpha_median": aggregate(group["high_alpha_median"] for group in groups),
            "low_demand_alpha_median": aggregate(group["low_alpha_median"] for group in groups),
            "high_demand_material_overshoot_fraction": aggregate(group["high_material_overshoot_fraction"] for group in groups),
            "low_demand_material_overshoot_fraction": aggregate(group["low_material_overshoot_fraction"] for group in groups),
            "high_demand_projected_overshoot_mean": aggregate(group["high_projected_overshoot_mean"] for group in groups),
            "high_demand_projected_overshoot_max": aggregate(group["high_projected_overshoot_max"] for group in groups),
        },
        "failure_count": len(failures),
        "failures": failures[:20],
        "terminal": {
            "state": state,
            "decision": decision,
            "authorizes": authorizes,
            "gate_reasons": gate_reasons,
        },
        "limitations": [
            "This is development-screening diagnosis for one fixed official OTS ConvIR-B instance and cannot be reused as unseen confirmation evidence.",
            "Paired input error is model-output-independent but is not a pure haze-density or perceptual restoration-demand label.",
            "Signed RGB projection operationalizes crossing beyond GT but does not by itself prove perceptual over-dehazing.",
            "Clear-source overlap with OTS training is not excluded, so the result cannot establish unseen-source generalization.",
            "No module, training, Haze4K outcome, NH-HAZE, confirmation, canary or locked-test evidence was used.",
        ],
        "marker": "SOTS_OTS_SIGNED_OVERSHOOT_V1_COMPLETE",
    }
    atomic_json(output_file(context, "signed_overshoot_summary.json"), summary)

    fields = [
        "stratum", "source_groups", "nested_variants", "overshoot_group_count",
        "overshoot_group_rate", "global_competence_count", "global_competence_rate",
        "high_overshoot_fraction_mean", "high_overshoot_fraction_median",
        "high_alpha_mean", "high_alpha_median", "low_alpha_median",
        "high_projected_overshoot_mean", "input_psnr_median", "output_psnr_median",
        "output_ssim_median",
    ]
    rows = [stratum_row("all", groups)]
    for count in sorted({int(group["variant_count"]) for group in groups}):
        subset = [group for group in groups if int(group["variant_count"]) == count]
        rows.append(stratum_row(f"variant_count_{count}", subset))
    with output_file(context, "signed_overshoot_strata.csv").open(
        "w", encoding="utf-8", newline="",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    write_workload_progress(
        context, completed_units=completed_units, stage="signed_overshoot_finalize",
    )
    return {
        "state": state,
        "decision": decision,
        "authorizes": authorizes,
        "details": {
            "summary_file": "signed_overshoot_summary.json",
            "independent_source_groups": len(groups),
            "nested_hazy_variants": len(variants),
            "signed_overshoot_prevalence": overshoot_interval,
            "global_competence": competence_interval,
            "gate_reasons": gate_reasons,
            "training_occurred": False,
            "sots_development_screening_only": True,
        },
    }


def contract(context_path: Path) -> None:
    context = load_context(context_path, "contract")
    prepare_phase_output(context)
    if context.device != "cuda" or any(context.protected_data_permissions.values()):
        raise RuntimeError("signed-overshoot contract requires CUDA and no protected-data permission")
    if "sots_outdoor" in context.assets:
        raise RuntimeError("scientific SOTS data must be absent from the contract phase")
    torch, model = load_official_model(context)
    height, width = 256, 320
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    clear = np.stack(
        (xx / width, yy / height, (xx + yy) / (width + height)), axis=-1,
    )
    transmission = np.exp(-0.10 * (1.0 + xx / width + yy / height)).astype(np.float32)
    hazy = clear * transmission[..., None] + 0.90 * (1.0 - transmission[..., None])
    prediction, output_ssim = infer(
        torch, model, hazy.astype(np.float32), clear.astype(np.float32), context.device,
    )
    measured = variant_measurement(hazy, clear, prediction, output_ssim)
    unchanged = variant_measurement(hazy, clear, hazy, 0.0)
    exact = variant_measurement(hazy, clear, clear, 1.0)
    extended = variant_measurement(
        hazy, clear, hazy + 1.10 * (clear - hazy), 0.0,
    )
    numeric = [
        float(value) for value in measured.values()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    checks = {
        "strict_official_checkpoint_load": True,
        "official_parameter_count": True,
        "three_scale_finite_forward": bool(np.isfinite(prediction).all()),
        "finalizer_finite": bool(numeric and np.isfinite(numeric).all()),
        "unchanged_output_reference": bool(
            abs(float(unchanged["output_mse"]) - float(unchanged["input_mse"])) < 1e-12
            and abs(float(unchanged["high_alpha_mean"])) < 1e-7
            and unchanged["high_material_overshoot_fraction"] == 0.0
        ),
        "exact_gt_projection_reference": bool(
            exact["output_mse"] == 0.0
            and exact["global_improvement"]
            and abs(float(exact["high_alpha_mean"]) - 1.0) < 1e-6
            and exact["high_material_overshoot_fraction"] == 0.0
        ),
        "ten_percent_extension_reference": bool(
            abs(float(extended["high_alpha_mean"]) - 1.10) < 1e-5
            and extended["high_material_overshoot_fraction"] == 1.0
        ),
        "wilson_reference": abs(float(wilson(98, 492)["estimate"]) - (98 / 492)) < 1e-12,
    }
    atomic_json(output_file(context, "signed_overshoot_contract_details.json"), {
        "parameter_count": PARAMETER_COUNT,
        "fixture": {"batch": 1, "channels": 3, "height": height, "width": width},
        "measurement_keys": sorted(measured),
    })
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
    sots = asset_path(context, "sots_outdoor", kind="directory")
    parent_closeout_path = asset_path(context, "prior_local_error_closeout", kind="file")
    if context.assets["prior_local_error_closeout"].sha256 != PARENT_CLOSEOUT_SHA256:
        raise RuntimeError("prior local-error closeout identity changed")
    parent = json.loads(parent_closeout_path.read_text(encoding="utf-8"))
    parent_identity = (
        parent.get("state") == "COMPLETED_GATE_FAIL"
        and parent.get("decision") == "SOTS_LOCAL_GT_ERROR_MISMATCH_FAIL"
        and parent.get("authorizes") == "NONE"
        and parent.get("details", {}).get("independent_source_groups") == EXPECTED_GROUPS
        and parent.get("details", {}).get("nested_hazy_variants") == EXPECTED_HAZY
        and parent.get("details", {}).get("training_occurred") is False
    )

    gt_dir = sots / "gt"
    hazy_dir = sots / "hazy"
    gt = image_files(gt_dir) if gt_dir.is_dir() else []
    hazy = image_files(hazy_dir) if hazy_dir.is_dir() else []
    gt_by_stem = {path.stem: path for path in gt}
    duplicate_gt_stems = len(gt_by_stem) != len(gt)
    hazy_by_source: dict[str, list[Path]] = defaultdict(list)
    unmapped_hazy = []
    for path in hazy:
        source = path.stem.split("_", 1)[0]
        if source not in gt_by_stem:
            unmapped_hazy.append(path.name)
        else:
            hazy_by_source[source].append(path)
    unused_gt = sorted(set(gt_by_stem) - set(hazy_by_source))
    counts = Counter(len(paths) for paths in hazy_by_source.values())
    pairing_match = (
        not duplicate_gt_stems
        and not unmapped_hazy
        and not unused_gt
        and len(gt) == EXPECTED_GT
        and len(hazy) == EXPECTED_HAZY
        and len(hazy_by_source) == EXPECTED_GROUPS
        and sum(len(paths) for paths in hazy_by_source.values()) == EXPECTED_HAZY
    )
    pairing = {
        "pairing_and_count_match": pairing_match,
        "variant_count_histogram": {str(key): value for key, value in sorted(counts.items())},
    }
    ordered = sorted(gt + hazy, key=lambda path: path.relative_to(sots).as_posix())
    dataset_digest = aggregate_dataset_digest(sots, ordered, context) if ordered else None
    dataset_identity = pairing_match and dataset_digest == DATASET_SHA256
    completed_units = len(ordered)

    group_records: list[dict[str, Any]] = []
    variant_records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    if dataset_identity and parent_identity:
        torch, model = load_official_model(context)
        attempted = 0
        for source in sorted(gt_by_stem):
            source_records = []
            for hazy_path in sorted(hazy_by_source[source]):
                attempted += 1
                try:
                    clear = image_array(gt_by_stem[source])
                    hazy_image = image_array(hazy_path)
                    if clear.shape != hazy_image.shape:
                        raise RuntimeError(
                            f"paired shape mismatch: GT={clear.shape}, hazy={hazy_image.shape}"
                        )
                    prediction, output_ssim = infer(
                        torch, model, hazy_image, clear, context.device,
                    )
                    measured = variant_measurement(
                        hazy_image, clear, prediction, output_ssim,
                    )
                    source_records.append(measured)
                    variant_records.append(measured)
                except Exception as exc:
                    failures.append({
                        "source_group": source,
                        "variant": hazy_path.name,
                        "reason": str(exc)[:512],
                    })
                completed_units = len(ordered) + attempted
                if attempted % 10 == 0:
                    write_workload_progress(
                        context,
                        completed_units=completed_units,
                        stage="official_ots_inference",
                    )
            if len(source_records) == len(hazy_by_source[source]):
                group_records.append(group_measurement(source_records))

    result = finalize(
        context,
        parent_identity=parent_identity,
        dataset_identity=dataset_identity,
        dataset_digest=dataset_digest,
        pairing=pairing,
        groups=group_records,
        variants=variant_records,
        failures=failures,
        completed_units=completed_units,
    )
    write_run_result(
        context,
        state=result["state"],
        decision=result["decision"],
        authorizes=result["authorizes"],
        details=result["details"],
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
