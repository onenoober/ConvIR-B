#!/usr/bin/env python3
"""Measure local paired-GT error behavior of the official OTS ConvIR-B model."""

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
PARENT_CLOSEOUT_SHA256 = "2772599de94979be9486b752b2bfc6fc3cb887e49c498a5d70249681b4db6c56"
MODEL_SOURCE_SHA256 = "140cec905c64f429359d675fff46d9ae77465d7dc972879c458095beb0d63a6d"
MODEL_LAYERS_SHA256 = "023e27cb28b9ba13663dda0cd7da181e2f28f60a67c833bc1933e6d5b1ebd580"
PARAMETER_COUNT = 8630665
TILE_SIZE = 32
REGION_FRACTION = 0.20
CORRECTION_IMBALANCE_MARGIN = 0.10
ALIGNMENT_MARGIN = 0.05
MISMATCH_PREVALENCE_MARGIN = 0.20
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


def error_tiles(error_map: np.ndarray) -> np.ndarray:
    height, width = error_map.shape
    pad_height = (-height) % TILE_SIZE
    pad_width = (-width) % TILE_SIZE
    padded = np.pad(
        error_map, ((0, pad_height), (0, pad_width)), mode="reflect",
    )
    grid_height = padded.shape[0] // TILE_SIZE
    grid_width = padded.shape[1] // TILE_SIZE
    return padded.reshape(
        grid_height, TILE_SIZE, grid_width, TILE_SIZE,
    ).mean(axis=(1, 3), dtype=np.float64)


def region_indices(score_tiles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = score_tiles.reshape(-1)
    region_count = max(1, int(math.floor(flat.size * REGION_FRACTION)))
    if region_count * 2 > flat.size:
        raise RuntimeError("too few tiles for disjoint low and high regions")
    order = np.argsort(flat, kind="mergesort")
    return order[:region_count], order[-region_count:]


def correction(
    input_tiles: np.ndarray, output_tiles: np.ndarray, indices: np.ndarray,
) -> tuple[float, float, float]:
    flat_input = input_tiles.reshape(-1)
    flat_output = output_tiles.reshape(-1)
    input_error = float(np.mean(flat_input[indices]))
    output_error = float(np.mean(flat_output[indices]))
    relative = (input_error - output_error) / (input_error + EPSILON)
    return input_error, output_error, float(relative)


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
    input_tiles = error_tiles(input_error_map)
    output_tiles = error_tiles(output_error_map)

    low_indices, high_indices = region_indices(input_tiles)
    low_input, low_output, low_correction = correction(
        input_tiles, output_tiles, low_indices,
    )
    high_input, high_output, high_correction = correction(
        input_tiles, output_tiles, high_indices,
    )
    true_imbalance = low_correction - high_correction

    rotated_scores = np.flip(input_tiles, axis=(0, 1))
    control_low_indices, control_high_indices = region_indices(rotated_scores)
    _, _, control_low_correction = correction(
        input_tiles, output_tiles, control_low_indices,
    )
    _, _, control_high_correction = correction(
        input_tiles, output_tiles, control_high_indices,
    )
    control_imbalance = control_low_correction - control_high_correction
    alignment_separation = true_imbalance - control_imbalance
    variant_mismatch = (
        true_imbalance >= CORRECTION_IMBALANCE_MARGIN
        and alignment_separation >= ALIGNMENT_MARGIN
    )
    return {
        "height": int(clear.shape[0]),
        "width": int(clear.shape[1]),
        "tile_count": int(input_tiles.size),
        "input_mse": input_mse,
        "output_mse": output_mse,
        "input_psnr": -10.0 * math.log10(max(input_mse, 1e-12)),
        "output_psnr": -10.0 * math.log10(max(output_mse, 1e-12)),
        "output_ssim": float(output_ssim),
        "global_improvement": output_mse < input_mse,
        "low_input_error": low_input,
        "low_output_error": low_output,
        "high_input_error": high_input,
        "high_output_error": high_output,
        "low_correction": low_correction,
        "high_correction": high_correction,
        "true_imbalance": true_imbalance,
        "control_imbalance": control_imbalance,
        "alignment_separation": alignment_separation,
        "low_excess_harm": (low_output - low_input) / (input_mse + EPSILON),
        "high_residual_ratio": high_output / (high_input + EPSILON),
        "variant_mismatch": variant_mismatch,
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
        "low_input_error": mean_key(records, "low_input_error"),
        "low_output_error": mean_key(records, "low_output_error"),
        "high_input_error": mean_key(records, "high_input_error"),
        "high_output_error": mean_key(records, "high_output_error"),
        "low_correction": mean_key(records, "low_correction"),
        "high_correction": mean_key(records, "high_correction"),
        "true_imbalance": mean_key(records, "true_imbalance"),
        "control_imbalance": mean_key(records, "control_imbalance"),
        "alignment_separation": mean_key(records, "alignment_separation"),
        "low_excess_harm": mean_key(records, "low_excess_harm"),
        "high_residual_ratio": mean_key(records, "high_residual_ratio"),
    }
    group["globally_competent"] = group["output_mse"] < group["input_mse"]
    group["low_region_harmed"] = group["low_output_error"] > group["low_input_error"]
    group["group_mismatch"] = (
        group["true_imbalance"] >= CORRECTION_IMBALANCE_MARGIN
        and group["alignment_separation"] >= ALIGNMENT_MARGIN
    )
    return group


def stratum_row(name: str, groups: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(groups)
    mismatches = sum(bool(group["group_mismatch"]) for group in groups)
    competent = sum(bool(group["globally_competent"]) for group in groups)
    harmed = sum(bool(group["low_region_harmed"]) for group in groups)
    return {
        "stratum": name,
        "source_groups": count,
        "nested_variants": sum(int(group["variant_count"]) for group in groups),
        "mismatch_count": mismatches,
        "mismatch_rate": mismatches / count if count else "",
        "global_competence_count": competent,
        "global_competence_rate": competent / count if count else "",
        "low_region_harm_count": harmed,
        "low_region_harm_rate": harmed / count if count else "",
        "true_imbalance_mean": float(np.mean([group["true_imbalance"] for group in groups])) if groups else "",
        "true_imbalance_median": float(np.median([group["true_imbalance"] for group in groups])) if groups else "",
        "control_imbalance_median": float(np.median([group["control_imbalance"] for group in groups])) if groups else "",
        "alignment_separation_median": float(np.median([group["alignment_separation"] for group in groups])) if groups else "",
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
    mismatch_count = sum(bool(group["group_mismatch"]) for group in groups)
    competence_count = sum(bool(group["globally_competent"]) for group in groups)
    harm_count = sum(bool(group["low_region_harmed"]) for group in groups)
    mismatch_interval = wilson(mismatch_count, EXPECTED_GROUPS)
    competence_interval = wilson(competence_count, EXPECTED_GROUPS)
    harm_interval = wilson(harm_count, EXPECTED_GROUPS)
    globally_competent = float(competence_interval["lower"]) >= GLOBAL_COMPETENCE_MARGIN

    if not parent_identity or not dataset_identity or not coverage_complete:
        state = "COMPLETED_INCONCLUSIVE"
        decision = "SOTS_LOCAL_GT_ERROR_MISMATCH_INCONCLUSIVE"
        authorizes = "SOTS_LOCAL_ERROR_MEASUREMENT_SUPPLEMENT_ONLY"
        gate_reasons = [
            "parent authorization, exact asset identity, pairing, finite inference, or complete source coverage failed"
        ]
    elif not globally_competent:
        state = "COMPLETED_INCONCLUSIVE"
        decision = "SOTS_LOCAL_GT_ERROR_MISMATCH_INCONCLUSIVE"
        authorizes = "SOTS_LOCAL_ERROR_MEASUREMENT_SUPPLEMENT_ONLY"
        gate_reasons = [
            "official OTS baseline lacked the frozen minimum global SOTS competence"
        ]
    elif float(mismatch_interval["lower"]) >= MISMATCH_PREVALENCE_MARGIN:
        state = "COMPLETED_GATE_PASS"
        decision = "SOTS_LOCAL_GT_ERROR_MISMATCH_PASS"
        authorizes = "SOTS_LOCAL_ERROR_MECHANISM_DESIGN_REVIEW"
        gate_reasons = [
            "paired-GT local mismatch prevalence lower 95 percent bound met the frozen material margin"
        ]
    elif float(mismatch_interval["upper"]) < MISMATCH_PREVALENCE_MARGIN:
        state = "COMPLETED_GATE_FAIL"
        decision = "SOTS_LOCAL_GT_ERROR_MISMATCH_FAIL"
        authorizes = "NONE"
        gate_reasons = [
            "paired-GT local mismatch prevalence upper 95 percent bound was below the frozen material margin"
        ]
    else:
        state = "COMPLETED_INCONCLUSIVE"
        decision = "SOTS_LOCAL_GT_ERROR_MISMATCH_INCONCLUSIVE"
        authorizes = "SOTS_LOCAL_ERROR_MEASUREMENT_SUPPLEMENT_ONLY"
        gate_reasons = [
            "paired-GT local mismatch prevalence interval crossed the frozen material margin"
        ]

    summary = {
        "schema_version": 1,
        "route_id": context.route_id,
        "operation_id": context.operation_id,
        "run_id": context.run_id,
        "scope": "fixed official OTS ConvIR-B on development-screening SOTS-Outdoor paired images",
        "dataset_identity": {
            "parent_authorization_match": parent_identity,
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
            "relative_correction": "(regional input MSE - regional output MSE)/(regional input MSE + 1e-8)",
            "group_mismatch_margins": {
                "true_low_minus_high_correction": CORRECTION_IMBALANCE_MARGIN,
                "true_minus_rotated_control_imbalance": ALIGNMENT_MARGIN,
            },
            "negative_control": "within-image 180-degree rotation of the tile input-error score field",
            "variant_aggregation": "arithmetic mean within clear-source group before group classification",
        },
        "primary_estimand": {
            "paired_gt_local_mismatch_prevalence": mismatch_interval,
            "material_margin": MISMATCH_PREVALENCE_MARGIN,
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
            "low_error_relative_correction": aggregate(group["low_correction"] for group in groups),
            "high_error_relative_correction": aggregate(group["high_correction"] for group in groups),
            "true_correction_imbalance": aggregate(group["true_imbalance"] for group in groups),
            "rotated_control_imbalance": aggregate(group["control_imbalance"] for group in groups),
            "alignment_separation": aggregate(group["alignment_separation"] for group in groups),
            "low_error_region_harm_prevalence": harm_interval,
            "low_error_excess_harm": aggregate(group["low_excess_harm"] for group in groups),
            "high_error_residual_ratio": aggregate(group["high_residual_ratio"] for group in groups),
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
            "Paired input error is model-output-independent but is not a pure haze-density or restoration-demand label.",
            "The rotated control reduces a simple spatial-content explanation but does not establish a causal mechanism.",
            "Clear-source overlap with OTS training is not excluded, so the result cannot establish unseen-source generalization.",
            "No module, training, Haze4K outcome, NH-HAZE, confirmation, canary or locked-test evidence was used.",
        ],
        "marker": "SOTS_OTS_LOCAL_ERROR_MEASUREMENT_V2_COMPLETE",
    }
    atomic_json(output_file(context, "local_error_summary.json"), summary)

    fields = [
        "stratum", "source_groups", "nested_variants", "mismatch_count",
        "mismatch_rate", "global_competence_count", "global_competence_rate",
        "low_region_harm_count", "low_region_harm_rate", "true_imbalance_mean",
        "true_imbalance_median", "control_imbalance_median",
        "alignment_separation_median", "input_psnr_median", "output_psnr_median",
        "output_ssim_median",
    ]
    rows = [stratum_row("all", groups)]
    for count in sorted({int(group["variant_count"]) for group in groups}):
        subset = [group for group in groups if int(group["variant_count"]) == count]
        rows.append(stratum_row(f"variant_count_{count}", subset))
    with output_file(context, "local_error_strata.csv").open(
        "w", encoding="utf-8", newline="",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    write_workload_progress(
        context, completed_units=completed_units, stage="local_error_finalize",
    )
    return {
        "state": state,
        "decision": decision,
        "authorizes": authorizes,
        "details": {
            "summary_file": "local_error_summary.json",
            "independent_source_groups": len(groups),
            "nested_hazy_variants": len(variants),
            "mismatch_prevalence": mismatch_interval,
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
        raise RuntimeError("local-error contract requires CUDA and no protected-data permission")
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
    perfect = variant_measurement(hazy, clear, clear, 1.0)
    unchanged = variant_measurement(hazy, clear, hazy, 0.0)
    numeric = [
        float(value) for value in measured.values()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    checks = {
        "strict_official_checkpoint_load": True,
        "official_parameter_count": True,
        "three_scale_finite_forward": bool(np.isfinite(prediction).all()),
        "finalizer_finite": bool(numeric and np.isfinite(numeric).all()),
        "perfect_output_reference": bool(
            perfect["output_mse"] == 0.0
            and perfect["global_improvement"]
            and perfect["low_correction"] > 0.99
            and perfect["high_correction"] > 0.99
            and not perfect["variant_mismatch"]
        ),
        "unchanged_output_reference": bool(
            abs(float(unchanged["output_mse"]) - float(unchanged["input_mse"])) < 1e-12
            and abs(float(unchanged["low_correction"])) < 1e-7
            and abs(float(unchanged["high_correction"])) < 1e-7
        ),
        "wilson_reference": abs(float(wilson(98, 492)["estimate"]) - (98 / 492)) < 1e-12,
    }
    atomic_json(output_file(context, "local_error_contract_details.json"), {
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
    parent_closeout_path = asset_path(context, "parent_closeout", kind="file")
    if context.assets["parent_closeout"].sha256 != PARENT_CLOSEOUT_SHA256:
        raise RuntimeError("parent closeout identity changed")
    parent = json.loads(parent_closeout_path.read_text(encoding="utf-8"))
    parent_identity = (
        parent.get("state") == "COMPLETED_GATE_PASS"
        and parent.get("decision") == "SOTS_OTS_ASSET_AUDIT_PASS"
        and parent.get("authorizes") == "SOTS_OTS_LOCAL_ERROR_MEASUREMENT_CONTRACT"
        and parent.get("details", {}).get("checkpoint_sha256") == CHECKPOINT_SHA256
        and parent.get("details", {}).get("dataset_sha256") == DATASET_SHA256
        and parent.get("details", {}).get("source_groups") == EXPECTED_GROUPS
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
