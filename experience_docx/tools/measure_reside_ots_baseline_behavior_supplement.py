#!/usr/bin/env python3
"""Complete the frozen OTS baseline measurement with safe aspect-ratio sizing."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
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


EXPECTED_SCENES = 625
AIRLIGHT = 0.90
BETAS = (0.04, 0.10, 0.20)
LOW_QUANTILE = 0.20
HIGH_QUANTILE = 0.80
CORRECTION_IMBALANCE_MARGIN = 0.10
ALIGNMENT_MARGIN = 0.05
SCENE_REQUIRED_VARIANTS = 2
MISMATCH_PREVALENCE_MARGIN = 0.20
GLOBAL_COMPETENCE_MARGIN = 0.80
TOTAL_UNITS = 2500
PARAMETER_COUNT = 8630665
EPSILON = 1e-8
CHECKPOINT_SHA256 = "6f42037d57a4e3de3a10ac0ab909d66a3415864a19433c29204a975f4efa4088"
MODEL_SOURCE_SHA256 = "3fa227af396464a7f07ac773f92e9cdb746e0fa6ae63adef711c765a02c3d4cd"
MODEL_LAYERS_SHA256 = "ac8a05bd626d9adda16308dedb9466f36d7ff44cfb666f64e7e14ddf8cdf43a4"
PARENT_CLOSEOUT_SHA256 = "34c29841585b517611b287e4bc8678a25451417dfd308c640feeff014ba30b86"
PARENT_SCENES_SHA256 = "f6677092a82fe80719e4eb184f6ae9dc71c514e9ee57cd8f96bbc5682af21c39"


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> dict[str, float | int]:
    if total <= 0:
        return {"successes": successes, "total": total, "estimate": 0.0, "lower": 0.0, "upper": 1.0}
    estimate = successes / total
    denominator = 1.0 + z * z / total
    center = (estimate + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(
        estimate * (1.0 - estimate) / total + z * z / (4.0 * total * total)
    ) / denominator
    return {
        "successes": successes,
        "total": total,
        "estimate": estimate,
        "lower": max(0.0, center - half),
        "upper": min(1.0, center + half),
    }


def aggregate(values: Iterable[float]) -> dict[str, float | None]:
    array = np.asarray(list(values), dtype=np.float64)
    if not array.size:
        return {key: None for key in ("min", "q05", "median", "mean", "q95", "max")}
    return {
        "min": float(array.min()),
        "q05": float(np.quantile(array, 0.05)),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "q95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def parse_variant(path: Path) -> tuple[str, float, float]:
    parts = path.stem.split("_")
    if len(parts) != 3:
        raise ValueError(f"invalid OTS haze filename: {path.name}")
    scene_id, airlight, beta = parts
    values = float(airlight), float(beta)
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"non-finite OTS haze filename token: {path.name}")
    return scene_id, values[0], values[1]


def image_files(path: Path) -> list[Path]:
    suffixes = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    return sorted(item for item in path.iterdir() if item.is_file() and item.suffix.lower() in suffixes)


def target_size(path: Path, long_edge: int) -> tuple[int, int]:
    with Image.open(path) as image:
        width, height = image.size
    scale = min(1.0, float(long_edge) / max(width, height))
    return max(32, int(round(width * scale))), max(32, int(round(height * scale)))


def image_array(path: Path, size: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(
            image.convert("RGB").resize(size, Image.Resampling.BILINEAR), dtype=np.float32,
        ) / 255.0


def _octave_quote(path: Path) -> str:
    return str(path).replace("'", "''")


def export_depths(
    entries: list[tuple[Path, int, int, int, int, Path]], script_path: Path, *, timeout: int,
) -> None:
    lines = ["more off;"]
    for source, source_height, source_width, target_height, target_width, destination in entries:
        lines.extend([
            f"s=load('{_octave_quote(source)}');",
            "if ~isfield(s,'depth'), error('missing depth field'); end;",
            "d=double(s.depth);",
            f"if size(d,1)=={source_width} && size(d,2)=={source_height}, d=d'; end;",
            f"if size(d,1)~={source_height} || size(d,2)~={source_width}, error('depth shape mismatch'); end;",
            f"[xq,yq]=meshgrid(linspace(1,size(d,2),{target_width}),linspace(1,size(d,1),{target_height}));",
            "small=interp2(d,xq,yq,'linear');",
            "if any(~isfinite(small(:))), error('nonfinite depth'); end;",
            f"fid=fopen('{_octave_quote(destination)}','wb');",
            "if fid<0, error('depth output open failed'); end;",
            "fwrite(fid,single(small),'single'); fclose(fid);",
        ])
    lines.append("disp('OTS_BASELINE_SUPPLEMENT_DEPTH_EXPORT_OK');")
    script_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    completed = subprocess.run(
        ["/usr/bin/octave", "--quiet", "--no-gui", str(script_path)],
        text=True, capture_output=True, timeout=timeout, check=False,
    )
    if completed.returncode or "OTS_BASELINE_SUPPLEMENT_DEPTH_EXPORT_OK" not in completed.stdout:
        detail = (completed.stdout + completed.stderr)[-4096:]
        raise RuntimeError(f"Octave depth export failed: {detail}")
    for _, _, _, target_height, target_width, destination in entries:
        if not destination.is_file() or destination.stat().st_size != target_height * target_width * 4:
            raise RuntimeError(f"invalid exported depth: {destination}")


def load_depth(path: Path, height: int, width: int) -> np.ndarray:
    values = np.fromfile(path, dtype=np.float32)
    if values.size != height * width or not np.isfinite(values).all():
        raise ValueError(f"invalid depth transport: {path}")
    return values.reshape(height, width, order="F")


def region_correction(input_error: np.ndarray, output_error: np.ndarray, mask: np.ndarray) -> float:
    input_mean = float(np.mean(input_error[mask], dtype=np.float64))
    output_mean = float(np.mean(output_error[mask], dtype=np.float64))
    return (input_mean - output_mean) / (input_mean + EPSILON)


def variant_measurement(
    hazy: np.ndarray,
    clear: np.ndarray,
    prediction: np.ndarray,
    depth: np.ndarray,
    beta: float,
    ssim_value: float,
) -> dict[str, float | bool]:
    input_error = np.mean(np.square(hazy - clear), axis=2, dtype=np.float64)
    output_error = np.mean(np.square(prediction - clear), axis=2, dtype=np.float64)
    tau = np.asarray(depth, dtype=np.float64) * float(beta)
    low_cut, high_cut = np.quantile(tau, [LOW_QUANTILE, HIGH_QUANTILE])
    low_mask, high_mask = tau <= low_cut, tau >= high_cut
    rotated = np.flip(tau, axis=(0, 1))
    rotated_low, rotated_high = rotated <= low_cut, rotated >= high_cut
    low_correction = region_correction(input_error, output_error, low_mask)
    high_correction = region_correction(input_error, output_error, high_mask)
    control_low_correction = region_correction(input_error, output_error, rotated_low)
    control_high_correction = region_correction(input_error, output_error, rotated_high)
    true_imbalance = low_correction - high_correction
    control_imbalance = control_low_correction - control_high_correction
    alignment_separation = true_imbalance - control_imbalance
    input_mse = float(np.mean(input_error, dtype=np.float64))
    output_mse = float(np.mean(output_error, dtype=np.float64))
    low_excess_harm = float(
        np.mean(np.maximum(output_error[low_mask] - input_error[low_mask], 0.0), dtype=np.float64)
        / (float(np.mean(input_error[low_mask], dtype=np.float64)) + EPSILON)
    )
    return {
        "input_mse": input_mse,
        "output_mse": output_mse,
        "input_psnr": float(-10.0 * math.log10(max(input_mse, 1e-12))),
        "output_psnr": float(-10.0 * math.log10(max(output_mse, 1e-12))),
        "output_ssim": float(ssim_value),
        "global_improvement": bool(output_mse < input_mse),
        "low_correction": low_correction,
        "high_correction": high_correction,
        "true_imbalance": true_imbalance,
        "control_imbalance": control_imbalance,
        "alignment_separation": alignment_separation,
        "low_excess_harm": low_excess_harm,
        "variant_mismatch": bool(
            true_imbalance >= CORRECTION_IMBALANCE_MARGIN
            and alignment_separation >= ALIGNMENT_MARGIN
        ),
    }


def load_official_model(context):
    import torch

    checkpoint = asset_path(context, "official_checkpoint", kind="file")
    identities = {
        "official_checkpoint": CHECKPOINT_SHA256,
        "model_source": MODEL_SOURCE_SHA256,
        "model_layers": MODEL_LAYERS_SHA256,
    }
    for asset_id, expected in identities.items():
        if context.assets[asset_id].sha256 != expected:
            raise RuntimeError(f"frozen asset identity changed: {asset_id}")
    sys.path.insert(0, str(context.remote_repo / "Dehazing" / "ITS"))
    from models.ConvIR import build_net

    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location="cpu")
    if not isinstance(state, dict) or "model" not in state or not isinstance(state["model"], dict):
        raise RuntimeError("official checkpoint does not contain state_dict['model']")
    model = build_net("base", "Haze4K").to(context.device).eval()
    model.load_state_dict(state["model"], strict=True)
    if sum(parameter.numel() for parameter in model.parameters()) != PARAMETER_COUNT:
        raise RuntimeError("official model parameter count changed")
    return torch, model


def reflection_indices(length: int, target: int) -> np.ndarray:
    if length < 2 or target < length:
        raise ValueError(f"invalid reflection extent: length={length}, target={target}")
    positions = np.arange(target, dtype=np.int64)
    period = 2 * length - 2
    folded = positions % period
    return np.where(folded < length, folded, period - folded).astype(np.int64)


def safe_model_canvas(image: np.ndarray, model_min_edge: int = 256) -> np.ndarray:
    height, width = image.shape[:2]
    target_height = int(math.ceil(max(height, model_min_edge) / 32.0) * 32)
    target_width = int(math.ceil(max(width, model_min_edge) / 32.0) * 32)
    rows = reflection_indices(height, target_height)
    columns = reflection_indices(width, target_width)
    return image[rows[:, None], columns[None, :], :]


def infer(torch, model, hazy: np.ndarray, device: str, model_min_edge: int = 256):
    height, width = hazy.shape[:2]
    canvas = safe_model_canvas(hazy, model_min_edge=model_min_edge)
    tensor = torch.from_numpy(np.transpose(canvas, (2, 0, 1)).copy()).unsqueeze(0).to(device)
    with torch.inference_mode():
        outputs = model(tensor)
        if not isinstance(outputs, list) or len(outputs) != 3:
            raise RuntimeError("official three-scale output contract changed")
        prediction = outputs[2][:, :, :height, :width].clamp(0.0, 1.0)
    if not bool(torch.isfinite(prediction).all().item()):
        raise RuntimeError("official model produced non-finite output")
    return prediction


def contract(context_path: Path) -> None:
    context = load_context(context_path, "contract")
    prepare_phase_output(context)
    if context.device != "cuda" or any(context.protected_data_permissions.values()):
        raise RuntimeError("supplement contract requires CUDA and no protected-data permission")
    torch, model = load_official_model(context)
    from pytorch_msssim import ssim

    fixture_results = []
    perfect = None
    for content_height, content_width in ((32, 320), (320, 32)):
        yy, xx = np.mgrid[0:content_height, 0:content_width].astype(np.float32)
        clear = np.stack((
            xx / content_width,
            yy / content_height,
            (xx + yy) / (content_width + content_height),
        ), axis=-1)
        depth = 1.0 + xx / content_width + 0.5 * yy / content_height
        transmission = np.exp(-0.10 * depth).astype(np.float32)
        hazy = (
            clear * transmission[..., None]
            + AIRLIGHT * (1.0 - transmission[..., None])
        ).astype(np.float32)
        prediction_tensor = infer(torch, model, hazy, context.device)
        clear_tensor = torch.from_numpy(
            np.transpose(clear, (2, 0, 1)).copy()
        ).unsqueeze(0).to(context.device)
        ssim_value = float(ssim(
            prediction_tensor, clear_tensor, data_range=1.0, size_average=True,
        ).item())
        prediction = np.transpose(
            prediction_tensor.squeeze(0).cpu().numpy(), (1, 2, 0),
        )
        fixture_results.append(
            variant_measurement(hazy, clear, prediction, depth, 0.10, ssim_value)
        )
        perfect = variant_measurement(hazy, clear, clear, depth, 0.10, 1.0)

    reference = np.arange(289 * 317 * 3, dtype=np.float32).reshape(289, 317, 3)
    reference_tensor = torch.from_numpy(
        np.transpose(reference, (2, 0, 1)).copy()
    ).unsqueeze(0)
    import torch.nn.functional as functional
    torch_padded = functional.pad(reference_tensor, (0, 3, 0, 31), mode="reflect")
    torch_reference = np.transpose(torch_padded.squeeze(0).numpy(), (1, 2, 0))
    canvas_reference = safe_model_canvas(reference, model_min_edge=256)
    reflection_reference_match = bool(np.array_equal(canvas_reference, torch_reference))
    measured = fixture_results[0]
    checks = {
        "strict_official_checkpoint_load": True,
        "official_parameter_count": True,
        "both_formal_aspect_boundaries_forward": all(
            all(
                isinstance(value, bool)
                or (isinstance(value, (int, float)) and math.isfinite(float(value)))
                for value in result.values()
            )
            for result in fixture_results
        ),
        "reflection_matches_torch_when_one_pass_valid": reflection_reference_match,
        "finalizer_finite": all(
            isinstance(value, bool) or (isinstance(value, (int, float)) and math.isfinite(float(value)))
            for value in measured.values()
        ),
        "metric_direction_reference": bool(
            perfect["output_mse"] == 0.0 and perfect["global_improvement"]
            and perfect["low_correction"] > 0.99 and perfect["high_correction"] > 0.99
        ),
        "wilson_reference": abs(float(wilson(125, 625)["estimate"]) - 0.20) < 1e-12,
    }
    atomic_json(output_file(context, "supplement_contract_details.json"), {
        "parameter_count": PARAMETER_COUNT,
        "content_fixtures": [
            {"batch": 1, "channels": 3, "height": 32, "width": 320},
            {"batch": 1, "channels": 3, "height": 320, "width": 32}
        ],
        "model_canvas_fixtures": [
            {"batch": 1, "channels": 3, "height": 256, "width": 320},
            {"batch": 1, "channels": 3, "height": 320, "width": 256}
        ],
        "measurement_keys": sorted(measured),
    })
    write_contract_result(
        context, checks=checks,
        engineering={
            "mode": "gpu_synthetic_no_data",
            "device": "cuda",
            "fixture": {"batch": 1, "channels": 3, "height": 256, "width": 320},
            "production_path_exercised": True,
            "protected_data_touched": False,
            "scientific_output_created": False,
            "scientific_training_occurred": False,
        },
    )


def run(context_path: Path) -> None:
    context = load_context(context_path, "run")
    prepare_phase_output(context)
    import torch
    from pytorch_msssim import ssim

    reside = asset_path(context, "reside_root", kind="directory")
    parent_closeout_path = asset_path(context, "parent_closeout", kind="file")
    parent_scenes_path = asset_path(context, "parent_scenes", kind="file")
    if context.assets["parent_closeout"].sha256 != PARENT_CLOSEOUT_SHA256 \
            or context.assets["parent_scenes"].sha256 != PARENT_SCENES_SHA256:
        raise RuntimeError("parent evidence identity changed")
    parent = json.loads(parent_closeout_path.read_text(encoding="utf-8"))
    parent_identity = (
        parent.get("state") == "COMPLETED_INCONCLUSIVE"
        and parent.get("decision") == "OTS_BASELINE_LOCAL_MISMATCH_INCONCLUSIVE"
        and parent.get("authorizes") == "OTS_BASELINE_MEASUREMENT_SUPPLEMENT_ONLY"
        and parent.get("details", {}).get("independent_scenes") == 407
    )
    validation_ids = [
        line.strip() for line in parent_scenes_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    scene_list_identity = len(validation_ids) == EXPECTED_SCENES and len(set(validation_ids)) == EXPECTED_SCENES
    output_file(context, "validation_scene_ids.txt").write_text(
        "".join(f"{scene_id}\n" for scene_id in validation_ids), encoding="utf-8",
    )

    clear_paths = image_files(reside / "official/OTS_ALPHA/clear_images")
    depth_paths = sorted((reside / "official/OTS_ALPHA/depth").glob("*.mat"))
    clear_by_id = {path.stem: path for path in clear_paths}
    depth_by_id = {path.stem: path for path in depth_paths}
    selected = set(validation_ids)
    selected_haze: dict[str, dict[float, Path]] = defaultdict(dict)
    for path in image_files(reside / "official/OTS_ALPHA/OTS"):
        try:
            scene_id, airlight, beta = parse_variant(path)
        except ValueError:
            continue
        if scene_id in selected and abs(airlight - AIRLIGHT) < 1e-8:
            for target_beta in BETAS:
                if abs(beta - target_beta) < 1e-8:
                    if target_beta in selected_haze[scene_id]:
                        raise RuntimeError(f"duplicate selected OTS variant: {scene_id}, {target_beta}")
                    selected_haze[scene_id][target_beta] = path
    dataset_identity = (
        parent_identity and scene_list_identity
        and all(scene_id in clear_by_id and scene_id in depth_by_id for scene_id in validation_ids)
        and all(set(selected_haze[scene_id]) == set(BETAS) for scene_id in validation_ids)
    )
    write_workload_progress(context, completed_units=100, stage="dataset_identity")

    long_edge = int(os.environ.get("CONVIR_ROUTE_OTS_LONG_EDGE", "320"))
    model_min_edge = int(os.environ.get("CONVIR_ROUTE_OTS_MODEL_MIN_EDGE", "256"))
    if long_edge != 320 or model_min_edge != 256:
        raise RuntimeError("frozen supplement sizing changed")
    sizes: dict[str, tuple[int, int]] = {}
    raw_root = output_file(context, "depth_resized")
    raw_root.mkdir()
    entries = []
    max_target_long_edge = 0
    for scene_id in validation_ids:
        with Image.open(clear_by_id[scene_id]) as image:
            source_width, source_height = image.size
        width, height = target_size(clear_by_id[scene_id], long_edge)
        sizes[scene_id] = (width, height)
        max_target_long_edge = max(max_target_long_edge, width, height)
        entries.append((
            depth_by_id[scene_id], source_height, source_width, height, width,
            raw_root / f"{scene_id}.f32",
        ))
    export_depths(entries, output_file(context, "export_depths.m"), timeout=1800)
    write_workload_progress(context, completed_units=625, stage="depth_export")

    torch, model = load_official_model(context)
    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    completed_variants = 0
    for scene_id in validation_ids:
        try:
            width, height = sizes[scene_id]
            clear = image_array(clear_by_id[scene_id], (width, height))
            depth = load_depth(raw_root / f"{scene_id}.f32", height, width)
            variant_records = []
            for beta in BETAS:
                hazy = image_array(selected_haze[scene_id][beta], (width, height))
                prediction_tensor = infer(
                    torch, model, hazy, context.device, model_min_edge=model_min_edge,
                )
                clear_tensor = torch.from_numpy(
                    np.transpose(clear, (2, 0, 1)).copy()
                ).unsqueeze(0).to(context.device)
                ssim_value = float(ssim(
                    prediction_tensor, clear_tensor, data_range=1.0, size_average=True,
                ).item())
                prediction = np.transpose(
                    prediction_tensor.squeeze(0).cpu().numpy(), (1, 2, 0),
                )
                measured = variant_measurement(hazy, clear, prediction, depth, beta, ssim_value)
                measured["beta"] = beta
                variant_records.append(measured)
                completed_variants += 1
                if completed_variants % 25 == 0:
                    write_workload_progress(
                        context, completed_units=625 + completed_variants, stage="official_inference",
                    )
            mismatch_variants = sum(bool(item["variant_mismatch"]) for item in variant_records)
            competent_variants = sum(bool(item["global_improvement"]) for item in variant_records)
            records.append({
                "scene_id": scene_id,
                "scene_mismatch": mismatch_variants >= SCENE_REQUIRED_VARIANTS,
                "scene_globally_competent": competent_variants >= SCENE_REQUIRED_VARIANTS,
                "mismatch_variants": mismatch_variants,
                "competent_variants": competent_variants,
                "variants": variant_records,
            })
        except Exception as exc:
            failures.append({"scene_id": scene_id, "reason": str(exc)[:512]})

    coverage_complete = len(records) == EXPECTED_SCENES and not failures
    mismatch_interval = wilson(sum(bool(item["scene_mismatch"]) for item in records), EXPECTED_SCENES)
    competence_interval = wilson(
        sum(bool(item["scene_globally_competent"]) for item in records), EXPECTED_SCENES,
    )
    globally_competent = float(competence_interval["lower"]) >= GLOBAL_COMPETENCE_MARGIN
    if not dataset_identity or not coverage_complete:
        state, decision, authorizes = (
            "COMPLETED_INCONCLUSIVE", "OTS_BASELINE_LOCAL_MISMATCH_INCONCLUSIVE", "NONE",
        )
        gate_reasons = ["supplement authorization, frozen assets, or complete 625-scene coverage failed"]
    elif not globally_competent:
        state, decision, authorizes = (
            "COMPLETED_INCONCLUSIVE", "OTS_BASELINE_LOCAL_MISMATCH_INCONCLUSIVE", "NONE",
        )
        gate_reasons = ["official Haze4K baseline lacked the frozen minimum global competence on OTS"]
    elif float(mismatch_interval["lower"]) >= MISMATCH_PREVALENCE_MARGIN:
        state, decision, authorizes = (
            "COMPLETED_GATE_PASS", "OTS_BASELINE_LOCAL_MISMATCH_PASS",
            "OUTDOOR_LOCAL_MISMATCH_MECHANISM_DESIGN_REVIEW",
        )
        gate_reasons = ["mismatch prevalence lower 95 percent bound met the frozen material margin"]
    elif float(mismatch_interval["upper"]) < MISMATCH_PREVALENCE_MARGIN:
        state, decision, authorizes = (
            "COMPLETED_GATE_FAIL", "OTS_BASELINE_LOCAL_MISMATCH_FAIL", "NONE",
        )
        gate_reasons = ["mismatch prevalence upper 95 percent bound was below the frozen material margin"]
    else:
        state, decision, authorizes = (
            "COMPLETED_INCONCLUSIVE", "OTS_BASELINE_LOCAL_MISMATCH_INCONCLUSIVE", "NONE",
        )
        gate_reasons = ["mismatch prevalence interval crossed the frozen material margin"]

    flattened = [variant for record in records for variant in record["variants"]]
    summary = {
        "schema_version": 1,
        "route_id": context.route_id,
        "operation_id": context.operation_id,
        "run_id": context.run_id,
        "scope": "complete frozen official Haze4K ConvIR-B measurement on OTS synthetic outdoor scenes",
        "parent": {
            "identity_match": parent_identity,
            "decision": "OTS_BASELINE_LOCAL_MISMATCH_INCONCLUSIVE",
            "authorized_action": "OTS_BASELINE_MEASUREMENT_SUPPLEMENT_ONLY",
        },
        "dataset_identity": {
            "parent_scene_list_match": scene_list_identity,
            "selected_variant_and_pairing_match": dataset_identity,
            "expected_independent_scenes": EXPECTED_SCENES,
            "completed_independent_scenes": len(records),
            "completed_nested_variants": len(flattened),
        },
        "preprocessing_supplement": {
            "long_edge_nominal_cap": long_edge,
            "model_canvas_minimum_edge": model_min_edge,
            "maximum_observed_target_long_edge": max_target_long_edge,
            "content_resize_unchanged_from_parent": True,
            "padded_pixels_in_scientific_metrics": False,
            "all_625_scenes_recomputed": True,
            "parent_outcomes_pooled": False,
        },
        "measurement": {
            "tau": "beta times official OTS depth; model-output-independent synthetic optical thickness",
            "airlight": AIRLIGHT,
            "betas": list(BETAS),
            "region_quantiles": {"low_max": LOW_QUANTILE, "high_min": HIGH_QUANTILE},
            "relative_correction": "(regional input MSE - regional output MSE)/(regional input MSE + 1e-8)",
            "variant_mismatch_margins": {
                "true_low_minus_high_correction": CORRECTION_IMBALANCE_MARGIN,
                "true_minus_rotated_tau_imbalance": ALIGNMENT_MARGIN,
            },
            "scene_required_variant_count": SCENE_REQUIRED_VARIANTS,
            "negative_control": "within-scene 180-degree spatial rotation of tau",
        },
        "primary_estimand": {
            "tau_aligned_scene_mismatch_prevalence": mismatch_interval,
            "material_margin": MISMATCH_PREVALENCE_MARGIN,
        },
        "global_competence": {
            "scene_prevalence": competence_interval,
            "required_lower_95": GLOBAL_COMPETENCE_MARGIN,
            "passed": globally_competent,
        },
        "secondary_aggregates": {
            "input_psnr": aggregate(float(item["input_psnr"]) for item in flattened),
            "output_psnr": aggregate(float(item["output_psnr"]) for item in flattened),
            "output_ssim": aggregate(float(item["output_ssim"]) for item in flattened),
            "low_tau_relative_correction": aggregate(float(item["low_correction"]) for item in flattened),
            "high_tau_relative_correction": aggregate(float(item["high_correction"]) for item in flattened),
            "true_correction_imbalance": aggregate(float(item["true_imbalance"]) for item in flattened),
            "rotated_tau_correction_imbalance": aggregate(float(item["control_imbalance"]) for item in flattened),
            "alignment_separation": aggregate(float(item["alignment_separation"]) for item in flattened),
            "low_tau_excess_harm": aggregate(float(item["low_excess_harm"]) for item in flattened),
        },
        "failure_count": len(failures),
        "failures": failures[:20],
        "terminal": {
            "state": state, "decision": decision, "authorizes": authorizes,
            "gate_reasons": gate_reasons,
        },
        "limitations": [
            "This is L1 descriptive development-screening evidence for one fixed model on OTS synthetic outdoor optical thickness.",
            "The rotated-tau control is descriptive and does not identify a causal mechanism.",
            "This supplement exhausts the authorized preprocessing correction and cannot authorize another retry.",
            "No training, candidate module, SOTS, Haze4K outcome, NH-HAZE, confirmation, canary, or locked-test evidence was used.",
        ],
        "marker": "RESIDE_OTS_BASELINE_BEHAVIOR_SUPPLEMENT_V1_COMPLETE",
    }
    atomic_json(output_file(context, "baseline_behavior_summary.json"), summary)

    fields = [
        "beta", "independent_scenes", "global_improvement_rate", "variant_mismatch_rate",
        "input_psnr_median", "output_psnr_median", "output_ssim_median",
        "low_correction_median", "high_correction_median", "true_imbalance_median",
        "rotated_imbalance_median", "alignment_separation_median", "low_excess_harm_median",
    ]
    with output_file(context, "aggregate_strata.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for beta in BETAS:
            values = [
                variant for record in records for variant in record["variants"]
                if abs(float(variant["beta"]) - beta) < 1e-8
            ]
            writer.writerow({
                "beta": format(beta, ".2f"),
                "independent_scenes": len(values),
                "global_improvement_rate": float(np.mean([item["global_improvement"] for item in values])) if values else "",
                "variant_mismatch_rate": float(np.mean([item["variant_mismatch"] for item in values])) if values else "",
                "input_psnr_median": float(np.median([item["input_psnr"] for item in values])) if values else "",
                "output_psnr_median": float(np.median([item["output_psnr"] for item in values])) if values else "",
                "output_ssim_median": float(np.median([item["output_ssim"] for item in values])) if values else "",
                "low_correction_median": float(np.median([item["low_correction"] for item in values])) if values else "",
                "high_correction_median": float(np.median([item["high_correction"] for item in values])) if values else "",
                "true_imbalance_median": float(np.median([item["true_imbalance"] for item in values])) if values else "",
                "rotated_imbalance_median": float(np.median([item["control_imbalance"] for item in values])) if values else "",
                "alignment_separation_median": float(np.median([item["alignment_separation"] for item in values])) if values else "",
                "low_excess_harm_median": float(np.median([item["low_excess_harm"] for item in values])) if values else "",
            })
    write_workload_progress(context, completed_units=TOTAL_UNITS, stage="supplement_finalize")
    write_run_result(
        context, state=state, decision=decision, authorizes=authorizes,
        details={
            "summary_file": "baseline_behavior_summary.json",
            "independent_scenes": len(records),
            "nested_variants": len(flattened),
            "mismatch_prevalence": mismatch_interval,
            "global_competence": competence_interval,
            "gate_reasons": gate_reasons,
            "training_occurred": False,
            "synthetic_outdoor_only": True,
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
