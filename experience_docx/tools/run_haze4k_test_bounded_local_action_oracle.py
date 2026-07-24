#!/usr/bin/env python3
"""Run a bounded, GT-privileged local-action oracle on Haze4K development scenes."""

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
TILE_SIZE = 32
ACTION_SET = (("keep", 1.0), ("weaken", 0.8), ("strengthen", 1.2))
SPLIT_ASSIGNMENT_DIGEST = "6ca5174470dad2b4eef4ae15c5a13a99d8ae9fc0bc2ea1116b199c4d4bc05582"
PARENT_CLOSEOUT_SHA256 = "08298720243c9d12dd1b2a0c27923ac460f65c0c2bfef858a1a70a9edb1a6012"
PARENT_SUMMARY_SHA256 = "f788db97f4d99937a5603b1c4789d417621bb5a5a2f94fb87d15756199e412be"
CHECKPOINT_SHA256 = "6f42037d57a4e3de3a10ac0ab909d66a3415864a19433c29204a975f4efa4088"
MODEL_SOURCE_SHA256 = "3fa227af396464a7f07ac773f92e9cdb746e0fa6ae63adef711c765a02c3d4cd"
MODEL_LAYERS_SHA256 = "ac8a05bd626d9adda16308dedb9466f36d7ff44cfb666f64e7e14ddf8cdf43a4"
PARAMETER_COUNT = 8_630_665
BOOTSTRAP_SEED = 20260725
BOOTSTRAP_RESAMPLES = 20_000
MIN_MEAN_GAIN_DB = 0.10
MATERIAL_SCENE_GAIN_DB = 0.10
MIN_MATERIAL_SCENE_PREVALENCE = 0.20
SSIM_HARM_MARGIN = 0.005
COLOR_HARM_MARGIN = 1.0 / 255.0
MAX_HARM_PREVALENCE = 0.10
EPSILON = 1e-12


def image_files(directory: Path) -> list[Path]:
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def image_array(path: Path) -> np.ndarray:
    with Image.open(path) as source:
        value = np.asarray(source.convert("RGB"), dtype=np.float32) / 255.0
    if value.ndim != 3 or value.shape[2] != 3 or min(value.shape[:2]) <= TILE_SIZE:
        raise RuntimeError(f"unsupported image shape for {path.name}: {value.shape}")
    return value


def canonical_rgb_digest(array: np.ndarray) -> str:
    height, width = array.shape[:2]
    payload = np.rint(array * 255.0).clip(0, 255).astype(np.uint8).tobytes()
    digest = hashlib.sha256()
    digest.update(width.to_bytes(8, "big"))
    digest.update(height.to_bytes(8, "big"))
    digest.update(payload)
    return digest.hexdigest()


def mse(value: np.ndarray, target: np.ndarray) -> float:
    result = float(np.mean((value.astype(np.float64) - target.astype(np.float64)) ** 2))
    if not math.isfinite(result):
        raise RuntimeError("non-finite MSE")
    return result


def psnr_from_mse(value: float) -> float:
    return -10.0 * math.log10(max(float(value), EPSILON))


def color_bias(value: np.ndarray, target: np.ndarray) -> float:
    channel_bias = np.mean(value.astype(np.float64) - target.astype(np.float64), axis=(0, 1))
    result = float(np.mean(np.abs(channel_bias)))
    if not math.isfinite(result):
        raise RuntimeError("non-finite color bias")
    return result


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
        return {"count": 0, "mean": None, "median": None, "q10": None, "q25": None, "q75": None, "q90": None}
    if not np.isfinite(array).all():
        raise RuntimeError("aggregate received non-finite values")
    return {
        "count": int(array.size), "mean": float(np.mean(array)),
        "median": float(np.median(array)), "q10": float(np.quantile(array, 0.10)),
        "q25": float(np.quantile(array, 0.25)), "q75": float(np.quantile(array, 0.75)),
        "q90": float(np.quantile(array, 0.90)),
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


def apply_scale(hazy: np.ndarray, prediction: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return prediction.copy()
    return np.clip(hazy + np.float32(scale) * (prediction - hazy), 0.0, 1.0).astype(np.float32)


def uniform_oracle(hazy: np.ndarray, clear: np.ndarray, prediction: np.ndarray) -> tuple[np.ndarray, str, float]:
    candidates = [(name, scale, apply_scale(hazy, prediction, scale)) for name, scale in ACTION_SET]
    errors = [mse(candidate, clear) for _, _, candidate in candidates]
    index = int(np.argmin(np.asarray(errors, dtype=np.float64)))
    name, _, output = candidates[index]
    return output, name, errors[index]


def spatial_oracle(hazy: np.ndarray, clear: np.ndarray, prediction: np.ndarray) -> tuple[np.ndarray, Counter[str], float]:
    height, width = clear.shape[:2]
    output = np.empty_like(prediction, dtype=np.float32)
    actions: Counter[str] = Counter()
    for top in range(0, height, TILE_SIZE):
        bottom = min(top + TILE_SIZE, height)
        for left in range(0, width, TILE_SIZE):
            right = min(left + TILE_SIZE, width)
            hazy_tile = hazy[top:bottom, left:right]
            prediction_tile = prediction[top:bottom, left:right]
            clear_tile = clear[top:bottom, left:right]
            candidates = [
                (name, apply_scale(hazy_tile, prediction_tile, scale))
                for name, scale in ACTION_SET
            ]
            errors = [mse(candidate, clear_tile) for _, candidate in candidates]
            index = int(np.argmin(np.asarray(errors, dtype=np.float64)))
            name, selected = candidates[index]
            output[top:bottom, left:right] = selected
            actions[name] += 1
    return output, actions, mse(output, clear)


def rgb_ssim(torch, outputs: list[np.ndarray], clear: np.ndarray, device: str) -> list[float]:
    import torch.nn.functional as functional

    stacked = np.stack(outputs, axis=0).transpose(0, 3, 1, 2).copy()
    reference = np.repeat(clear[None, ...], len(outputs), axis=0).transpose(0, 3, 1, 2).copy()
    x = torch.from_numpy(stacked).to(device=device, dtype=torch.float32)
    y = torch.from_numpy(reference).to(device=device, dtype=torch.float32)

    def local_mean(value):
        return functional.avg_pool2d(functional.pad(value, (5, 5, 5, 5), mode="reflect"), 11, stride=1)

    mu_x, mu_y = local_mean(x), local_mean(y)
    variance_x = torch.clamp(local_mean(x * x) - mu_x * mu_x, min=0.0)
    variance_y = torch.clamp(local_mean(y * y) - mu_y * mu_y, min=0.0)
    covariance = local_mean(x * y) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    score = ((2.0 * mu_x * mu_y + c1) * (2.0 * covariance + c2)) / (
        (mu_x * mu_x + mu_y * mu_y + c1) * (variance_x + variance_y + c2)
    )
    values = score.mean(dim=(1, 2, 3)).detach().cpu().numpy().astype(np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError("non-finite RGB SSIM")
    return [float(value) for value in values]


def variant_oracle(torch, hazy: np.ndarray, clear: np.ndarray, prediction: np.ndarray, device: str) -> dict[str, Any]:
    baseline_error = mse(prediction, clear)
    uniform, uniform_action, uniform_error = uniform_oracle(hazy, clear, prediction)
    spatial, spatial_actions, spatial_error = spatial_oracle(hazy, clear, prediction)
    if uniform_error > baseline_error + 1e-11:
        raise RuntimeError("best uniform action failed to include the keep baseline")
    if spatial_error > uniform_error + 1e-11:
        raise RuntimeError("spatial oracle failed deterministic MSE dominance")
    baseline_ssim, uniform_ssim, spatial_ssim = rgb_ssim(
        torch, [prediction, uniform, spatial], clear, device,
    )
    baseline_psnr = psnr_from_mse(baseline_error)
    uniform_psnr = psnr_from_mse(uniform_error)
    spatial_psnr = psnr_from_mse(spatial_error)
    total_tiles = sum(spatial_actions.values())
    if total_tiles <= 0:
        raise RuntimeError("spatial oracle selected no tiles")
    return {
        "baseline_mse": baseline_error, "uniform_mse": uniform_error, "spatial_mse": spatial_error,
        "baseline_psnr": baseline_psnr, "uniform_psnr": uniform_psnr, "spatial_psnr": spatial_psnr,
        "spatial_minus_uniform_psnr": spatial_psnr - uniform_psnr,
        "uniform_minus_baseline_psnr": uniform_psnr - baseline_psnr,
        "spatial_minus_baseline_psnr": spatial_psnr - baseline_psnr,
        "baseline_ssim": baseline_ssim, "uniform_ssim": uniform_ssim, "spatial_ssim": spatial_ssim,
        "spatial_minus_uniform_ssim": spatial_ssim - uniform_ssim,
        "baseline_color_bias": color_bias(prediction, clear),
        "uniform_color_bias": color_bias(uniform, clear),
        "spatial_color_bias": color_bias(spatial, clear),
        "spatial_minus_uniform_color_bias": color_bias(spatial, clear) - color_bias(uniform, clear),
        "uniform_action": uniform_action,
        "spatial_actions": dict(spatial_actions),
        "spatial_action_fractions": {name: spatial_actions[name] / total_tiles for name, _ in ACTION_SET},
        "tile_count": total_tiles,
    }


def mean_key(records: list[dict[str, Any]], key: str) -> float:
    values = np.asarray([float(record[key]) for record in records], dtype=np.float64)
    if values.size != VARIANTS_PER_SCENE or not np.isfinite(values).all():
        raise RuntimeError(f"invalid scene values for {key}")
    return float(np.mean(values))


def scene_oracle(records: list[dict[str, Any]]) -> dict[str, Any]:
    if len(records) != VARIANTS_PER_SCENE:
        raise RuntimeError("scene does not contain four complete variants")
    gain = mean_key(records, "spatial_minus_uniform_psnr")
    ssim_delta = mean_key(records, "spatial_minus_uniform_ssim")
    color_delta = mean_key(records, "spatial_minus_uniform_color_bias")
    result = {
        "spatial_minus_uniform_psnr": gain,
        "uniform_minus_baseline_psnr": mean_key(records, "uniform_minus_baseline_psnr"),
        "spatial_minus_baseline_psnr": mean_key(records, "spatial_minus_baseline_psnr"),
        "spatial_minus_uniform_ssim": ssim_delta,
        "spatial_minus_uniform_color_bias": color_delta,
        "materially_benefited": gain >= MATERIAL_SCENE_GAIN_DB,
        "ssim_harmed": ssim_delta <= -SSIM_HARM_MARGIN,
        "color_harmed": color_delta >= COLOR_HARM_MARGIN,
    }
    for name, _ in ACTION_SET:
        result[f"spatial_{name}_fraction"] = float(np.mean([
            record["spatial_action_fractions"][name] for record in records
        ]))
    return result


def load_official_model(context):
    import torch

    checkpoint = asset_path(context, "official_checkpoint", kind="file")
    model_source = asset_path(context, "model_source", kind="file")
    model_layers = asset_path(context, "model_layers", kind="file")
    asset_path(context, "oracle_entrypoint", kind="file")
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


def finalize(context, integrity: dict[str, bool], scenes: list[dict[str, Any]], failures: list[dict[str, str]], action_counts: dict[str, Any]) -> dict[str, Any]:
    complete = all(integrity.values()) and len(scenes) == EXPECTED_SCENES and not failures
    primary = paired_bootstrap(scene["spatial_minus_uniform_psnr"] for scene in scenes) if scenes else None
    benefit = wilson(sum(bool(scene["materially_benefited"]) for scene in scenes), EXPECTED_SCENES)
    ssim_harm = wilson(sum(bool(scene["ssim_harmed"]) for scene in scenes), EXPECTED_SCENES)
    color_harm = wilson(sum(bool(scene["color_harmed"]) for scene in scenes), EXPECTED_SCENES)
    utility_pass = bool(
        complete and primary is not None
        and float(primary["lower"]) >= MIN_MEAN_GAIN_DB
        and float(benefit["lower"]) >= MIN_MATERIAL_SCENE_PREVALENCE
    )
    safety_pass = bool(
        complete
        and float(ssim_harm["upper"]) < MAX_HARM_PREVALENCE
        and float(color_harm["upper"]) < MAX_HARM_PREVALENCE
    )
    utility_fail = bool(
        complete and primary is not None and (
            float(primary["upper"]) < MIN_MEAN_GAIN_DB
            or float(benefit["upper"]) < MIN_MATERIAL_SCENE_PREVALENCE
        )
    )
    safety_fail = bool(
        complete and (
            float(ssim_harm["lower"]) >= MAX_HARM_PREVALENCE
            or float(color_harm["lower"]) >= MAX_HARM_PREVALENCE
        )
    )
    if not complete:
        state, decision, authorizes = (
            "COMPLETED_INCONCLUSIVE", "HAZE4K_TEST_BOUNDED_LOCAL_ACTION_ORACLE_INCONCLUSIVE",
            "HAZE4K_TEST_BOUNDED_LOCAL_ACTION_ORACLE_SUPPLEMENT_ONLY",
        )
        reasons = ["complete isolated-development coverage or integrity failed"]
    elif utility_pass and safety_pass:
        state, decision, authorizes = (
            "COMPLETED_GATE_PASS", "HAZE4K_TEST_BOUNDED_LOCAL_ACTION_ORACLE_HEADROOM_PASS",
            "HAZE4K_TEST_LOCAL_ACTION_PROXY_PREDICTABILITY",
        )
        reasons = ["spatial oracle exceeded the per-image best uniform oracle by the frozen materiality and breadth margins while passing both harm gates"]
    elif utility_fail or safety_fail:
        state, decision, authorizes = (
            "COMPLETED_GATE_FAIL", "HAZE4K_TEST_BOUNDED_LOCAL_ACTION_ORACLE_HEADROOM_FAIL", "NONE",
        )
        reasons = ["at least one frozen utility or safety interval clearly contradicted the required margin"]
    else:
        state, decision, authorizes = (
            "COMPLETED_INCONCLUSIVE", "HAZE4K_TEST_BOUNDED_LOCAL_ACTION_ORACLE_INCONCLUSIVE",
            "HAZE4K_TEST_BOUNDED_LOCAL_ACTION_ORACLE_SUPPLEMENT_ONLY",
        )
        reasons = ["at least one frozen utility or safety interval crossed its margin"]

    summary = {
        "schema_version": 1, "route_id": context.route_id, "operation_id": context.operation_id,
        "run_id": context.run_id,
        "scope": "GT-privileged bounded local-action oracle on isolated 100-scene Haze4K official-test development partition",
        "identity_and_coverage": {
            "integrity_checks": integrity, "completed_scenes": len(scenes),
            "completed_variants": len(scenes) * VARIANTS_PER_SCENE,
            "failure_count": len(failures), "failures": failures[:20],
            "assignment_digest": SPLIT_ASSIGNMENT_DIGEST,
            "candidate_confirmation_asset_delivered": False, "training_occurred": False,
        },
        "action_contract": {
            "formula": "clip(hazy + scale * (official_prediction - hazy), 0, 1)",
            "actions": [{"name": name, "scale": scale} for name, scale in ACTION_SET],
            "spatial_grid_pixels": TILE_SIZE,
            "uniform_reference": "GT-selected per-image best action from the identical action set",
            "spatial_policy": "GT-selected action independently per origin-aligned tile; boundary tiles use valid pixels only",
        },
        "primary_spatial_vs_best_uniform": {
            "scene_mean_delta_psnr_db": primary,
            "minimum_lower_bound_db": MIN_MEAN_GAIN_DB,
            "material_scene_gain_db": MATERIAL_SCENE_GAIN_DB,
            "material_scene_prevalence": benefit,
            "minimum_prevalence_lower_bound": MIN_MATERIAL_SCENE_PREVALENCE,
            "utility_passed": utility_pass,
        },
        "safety": {
            "ssim_harm_definition": f"scene-mean spatial-minus-uniform SSIM at most -{SSIM_HARM_MARGIN}",
            "ssim_harm_prevalence": ssim_harm,
            "color_harm_definition": f"scene-mean spatial-minus-uniform RGB mean-bias at least {COLOR_HARM_MARGIN}",
            "color_harm_prevalence": color_harm,
            "maximum_harm_upper_bound": MAX_HARM_PREVALENCE,
            "safety_passed": safety_pass,
        },
        "secondary_scene_aggregates": {
            "spatial_minus_uniform_psnr_db": aggregate(scene["spatial_minus_uniform_psnr"] for scene in scenes),
            "uniform_minus_baseline_psnr_db": aggregate(scene["uniform_minus_baseline_psnr"] for scene in scenes),
            "spatial_minus_baseline_psnr_db": aggregate(scene["spatial_minus_baseline_psnr"] for scene in scenes),
            "spatial_minus_uniform_ssim": aggregate(scene["spatial_minus_uniform_ssim"] for scene in scenes),
            "spatial_minus_uniform_color_bias": aggregate(scene["spatial_minus_uniform_color_bias"] for scene in scenes),
        },
        "action_usage": action_counts,
        "terminal": {"state": state, "decision": decision, "authorizes": authorizes, "gate_reasons": reasons},
        "limitations": [
            "This is a non-deployable GT-privileged development oracle, not a learned predictor or module.",
            "The action family is limited to three residual scales and does not prove the optimum over other action spaces or insertion locations.",
            "The per-image best uniform reference is also GT-privileged; the primary contrast isolates spatial selection within this frozen family.",
            "An 11-by-11 uniform-window RGB SSIM is used as a frozen safety metric and does not replace perceptual validation.",
            "The 150-scene candidate-confirmation asset was not delivered or accessed.",
        ],
        "marker": "HAZE4K_TEST_BOUNDED_LOCAL_ACTION_ORACLE_COMPLETE",
    }
    atomic_json(output_file(context, "haze4k_test_bounded_local_action_oracle_summary.json"), summary)
    rows = [
        {"stratum": "scene_mean_spatial_minus_uniform_psnr_db", "kind": "paired_bootstrap", "scenes": EXPECTED_SCENES, "events": "", "estimate": primary["estimate"] if primary else "", "lower_95": primary["lower"] if primary else "", "upper_95": primary["upper"] if primary else "", "threshold": MIN_MEAN_GAIN_DB},
        {"stratum": "materially_benefited_scene", "kind": "wilson", "scenes": EXPECTED_SCENES, "events": benefit["successes"], "estimate": benefit["estimate"], "lower_95": benefit["lower"], "upper_95": benefit["upper"], "threshold": MIN_MATERIAL_SCENE_PREVALENCE},
        {"stratum": "ssim_harmed_scene", "kind": "wilson", "scenes": EXPECTED_SCENES, "events": ssim_harm["successes"], "estimate": ssim_harm["estimate"], "lower_95": ssim_harm["lower"], "upper_95": ssim_harm["upper"], "threshold": MAX_HARM_PREVALENCE},
        {"stratum": "color_harmed_scene", "kind": "wilson", "scenes": EXPECTED_SCENES, "events": color_harm["successes"], "estimate": color_harm["estimate"], "lower_95": color_harm["lower"], "upper_95": color_harm["upper"], "threshold": MAX_HARM_PREVALENCE},
    ]
    with output_file(context, "haze4k_test_bounded_local_action_oracle_strata.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {
        "state": state, "decision": decision, "authorizes": authorizes,
        "details": {
            "summary_file": "haze4k_test_bounded_local_action_oracle_summary.json",
            "strata_file": "haze4k_test_bounded_local_action_oracle_strata.csv",
            "independent_scenes": len(scenes), "nested_variants": len(scenes) * VARIANTS_PER_SCENE,
            "utility_passed": utility_pass, "safety_passed": safety_pass,
            "gate_reasons": reasons, "training_occurred": False,
            "candidate_confirmation_asset_delivered": False,
        },
    }


def contract(context_path: Path) -> None:
    context = load_context(context_path, "contract")
    prepare_phase_output(context)
    if context.device != "cuda" or any(context.protected_data_permissions.values()):
        raise RuntimeError("bounded oracle contract requires CUDA and no protected permission")
    if "haze4k_test_development" in context.assets:
        raise RuntimeError("scientific development data must be absent from contract phase")
    torch, model = load_official_model(context)
    height, width = 256, 320
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    clear = np.stack((0.15 + 0.55 * xx / width, 0.20 + 0.50 * yy / height, 0.18 + 0.52 * (xx + yy) / (width + height)), axis=-1).astype(np.float32)
    hazy = (0.72 * clear + 0.18).astype(np.float32)
    prediction = infer(torch, model, hazy, context.device)
    exercised = variant_oracle(torch, hazy, clear, prediction, context.device)
    direction = clear - hazy
    manufactured = np.empty_like(clear)
    manufactured[:, :width // 2] = hazy[:, :width // 2] + 1.25 * direction[:, :width // 2]
    manufactured[:, width // 2:] = hazy[:, width // 2:] + 0.75 * direction[:, width // 2:]
    manufactured = np.clip(manufactured, 0.0, 1.0).astype(np.float32)
    reference = variant_oracle(torch, hazy, clear, manufactured, context.device)
    checks = {
        "strict_checkpoint_load_and_parameter_count": sum(p.numel() for p in model.parameters()) == PARAMETER_COUNT,
        "all_parameters_frozen": not any(p.requires_grad for p in model.parameters()),
        "three_scale_finite_forward": bool(np.isfinite(prediction).all()),
        "production_action_finalizer_finite": all(math.isfinite(float(value)) for key, value in exercised.items() if isinstance(value, (int, float)) and not isinstance(value, bool)),
        "uniform_oracle_includes_keep": float(exercised["uniform_mse"]) <= float(exercised["baseline_mse"]) + 1e-11,
        "spatial_oracle_mse_dominance": float(exercised["spatial_mse"]) <= float(exercised["uniform_mse"]) + 1e-11,
        "manufactured_spatial_gain_positive": float(reference["spatial_minus_uniform_psnr"]) > 0.10,
        "manufactured_uses_weaken_and_strengthen": reference["spatial_actions"].get("weaken", 0) > 0 and reference["spatial_actions"].get("strengthen", 0) > 0,
        "bootstrap_direction_reference": abs(float(paired_bootstrap([0.1, 0.2, 0.3, 0.4])["estimate"]) - 0.25) < 1e-12,
        "ssim_identity_reference": abs(rgb_ssim(torch, [clear], clear, context.device)[0] - 1.0) < 1e-6,
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
    parent_closeout_path = asset_path(context, "prior_replication_closeout", kind="file")
    parent_summary_path = asset_path(context, "prior_replication_summary", kind="file")
    if context.assets["prior_replication_closeout"].sha256 != PARENT_CLOSEOUT_SHA256:
        raise RuntimeError("parent closeout identity changed")
    if context.assets["prior_replication_summary"].sha256 != PARENT_SUMMARY_SHA256:
        raise RuntimeError("parent summary identity changed")
    parent = json.loads(parent_closeout_path.read_text(encoding="utf-8"))
    parent_summary = json.loads(parent_summary_path.read_text(encoding="utf-8"))
    parent_ok = (
        parent.get("state") == "COMPLETED_GATE_PASS"
        and parent.get("decision") == "HAZE4K_TEST_LOCAL_ERROR_REPLICATION_ORACLE_QUALIFIED"
        and parent.get("authorizes") == "HAZE4K_TEST_BOUNDED_LOCAL_ACTION_ORACLE"
        and parent.get("details", {}).get("independent_scenes") == EXPECTED_SCENES
        and parent.get("details", {}).get("candidate_confirmation_asset_delivered") is False
        and parent_summary.get("identity_and_coverage", {}).get("assignment_digest") == SPLIT_ASSIGNMENT_DIGEST
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
    exact_names = {path.name for path in hazy_paths} == {path.name for path in clear_paths}
    write_workload_progress(context, completed_units=1, stage="isolated_development_pairing")

    failures: list[dict[str, str]] = []
    clear_by_digest: dict[str, list[Path]] = defaultdict(list)
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
    spatial_action_counts: Counter[str] = Counter()
    uniform_action_counts: Counter[str] = Counter()
    total_tiles = 0
    if parent_ok and scope_ok and dataset_ok:
        torch, model = load_official_model(context)
        attempted = 0
        for digest in sorted(variants_by_digest):
            records: list[dict[str, Any]] = []
            for hazy_path in sorted(variants_by_digest[digest]):
                attempted += 1
                try:
                    hazy = image_array(hazy_path)
                    clear = image_array(clear_root / hazy_path.name)
                    prediction = infer(torch, model, hazy, context.device)
                    record = variant_oracle(torch, hazy, clear, prediction, context.device)
                    records.append(record)
                    spatial_action_counts.update(record["spatial_actions"])
                    uniform_action_counts[record["uniform_action"]] += 1
                    total_tiles += int(record["tile_count"])
                except Exception as exc:
                    failures.append({"scene": digest[:16], "variant": hazy_path.name, "reason": str(exc)[:512]})
                if attempted % 5 == 0:
                    write_workload_progress(context, completed_units=2 + attempted, stage="official_inference_and_bounded_oracle")
            if len(records) == VARIANTS_PER_SCENE:
                scenes.append(scene_oracle(records))

    integrity = {
        "parent_oracle_authorization": parent_ok,
        "isolated_development_asset_only": scope_ok,
        "complete_100_scene_400_variant_grouping": dataset_ok,
        "complete_finite_inference_and_oracle": len(scenes) == EXPECTED_SCENES and not failures,
        "candidate_confirmation_asset_not_delivered": True,
        "identical_bounded_action_set_for_uniform_and_spatial": True,
        "no_training": True,
    }
    action_counts = {
        "spatial_tile_counts": {name: spatial_action_counts[name] for name, _ in ACTION_SET},
        "spatial_tile_fractions": {name: (spatial_action_counts[name] / total_tiles if total_tiles else None) for name, _ in ACTION_SET},
        "total_spatial_tiles": total_tiles,
        "best_uniform_variant_counts": {name: uniform_action_counts[name] for name, _ in ACTION_SET},
    }
    result = finalize(context, integrity, scenes, failures, action_counts)
    write_workload_progress(context, completed_units=403, stage="scene_level_oracle_finalize")
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
