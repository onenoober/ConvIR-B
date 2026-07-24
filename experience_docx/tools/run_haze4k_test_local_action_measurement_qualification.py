#!/usr/bin/env python3
"""Qualify Haze4K local-action utility as a stable measurement target."""

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


IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}
EXPECTED_SCENES = 100
EXPECTED_VARIANTS = 400
VARIANTS_PER_SCENE = 4
TILE_SIZE = 32
GRID_OFFSETS = ((0, 0), (0, 16), (16, 0), (16, 16))
ACTION_SET = (("keep", 1.0), ("weaken", 0.8), ("strengthen", 1.2))
SPLIT_ASSIGNMENT_DIGEST = "6ca5174470dad2b4eef4ae15c5a13a99d8ae9fc0bc2ea1116b199c4d4bc05582"
ORACLE_CLOSEOUT_SHA256 = "96357cbaaee5aa338fb0f9c9835a975a27e7f048c78a096fd644e6acdd3e383c"
ORACLE_SUMMARY_SHA256 = "794ef27733f51f2fa70fab5c94bc661564d7b988ef4247a1b26e33a21b4de7cb"
PROXY_CLOSEOUT_SHA256 = "c12a4b013db12b0828823ae8cba7fe8a51208971d29e667966635c157a67cba4"
PROXY_CONCLUSION_SHA256 = "c8419d0b0d23b079b55b3490c54347033fa79221bcc035d995d5e55dcaa1695b"
CHECKPOINT_SHA256 = "6f42037d57a4e3de3a10ac0ab909d66a3415864a19433c29204a975f4efa4088"
MODEL_SOURCE_SHA256 = "3fa227af396464a7f07ac773f92e9cdb746e0fa6ae63adef711c765a02c3d4cd"
MODEL_LAYERS_SHA256 = "ac8a05bd626d9adda16308dedb9466f36d7ff44cfb666f64e7e14ddf8cdf43a4"
PARAMETER_COUNT = 8_630_665
BOOTSTRAP_RESAMPLES = 20_000
BOOTSTRAP_SEEDS = {
    "regret": 20260729,
    "aligned_shift": 20260730,
    "nonclipped_gain": 20260731,
}
REGRET_MARGIN_DB = 0.10
PRECISION_DISTANCE_DB = 0.05
MATERIAL_GAIN_DB = 0.10
MIN_MATERIAL_SCENE_PREVALENCE = 0.20
SSIM_HARM_MARGIN = 0.005
COLOR_HARM_MARGIN = 1.0 / 255.0
MAX_HARM_PREVALENCE = 0.10
EPSILON = 1e-12
NUMERICAL_TIE_TOLERANCE = 1e-12


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


def mse(value: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> float:
    difference = (value.astype(np.float64) - target.astype(np.float64)) ** 2
    if mask is not None:
        if mask.shape != value.shape[:2] or not bool(np.any(mask)):
            raise RuntimeError("MSE mask is empty or has the wrong shape")
        difference = difference[mask]
    result = float(np.mean(difference))
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
        "successes": successes,
        "total": total,
        "estimate": estimate,
        "lower": max(0.0, center - half),
        "upper": min(1.0, center + half),
    }


def aggregate(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {
            "count": 0, "mean": None, "sd": None, "min": None, "q05": None,
            "median": None, "q95": None, "max": None,
        }
    if not np.isfinite(array).all():
        raise RuntimeError("aggregate received non-finite values")
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "sd": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "min": float(np.min(array)),
        "q05": float(np.quantile(array, 0.05)),
        "median": float(np.median(array)),
        "q95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def scene_bootstrap(values: Iterable[float], seed: int) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise RuntimeError("scene bootstrap requires finite scene values")
    generator = np.random.default_rng(seed)
    draws = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    for start in range(0, BOOTSTRAP_RESAMPLES, 1000):
        stop = min(start + 1000, BOOTSTRAP_RESAMPLES)
        indices = generator.integers(0, array.size, size=(stop - start, array.size))
        draws[start:stop] = np.mean(array[indices], axis=1)
    estimate = float(np.mean(array))
    lower_one_sided = float(np.quantile(draws, 0.05))
    upper_one_sided = float(np.quantile(draws, 0.95))
    return {
        "scene_count": int(array.size),
        "estimate": estimate,
        "lower_one_sided_95": lower_one_sided,
        "upper_one_sided_95": upper_one_sided,
        "lower_two_sided_95": float(np.quantile(draws, 0.025)),
        "upper_two_sided_95": float(np.quantile(draws, 0.975)),
        "upper_distance": upper_one_sided - estimate,
        "seed": seed,
        "resamples": BOOTSTRAP_RESAMPLES,
    }


def axis_slices(length: int, offset: int) -> list[tuple[int, int]]:
    if length <= 0 or not 0 <= offset < TILE_SIZE:
        raise ValueError("invalid shifted-grid axis")
    boundaries = {0, length}
    cursor = offset
    while cursor < length:
        if cursor > 0:
            boundaries.add(cursor)
        cursor += TILE_SIZE
    ordered = sorted(boundaries)
    slices = [(ordered[index], ordered[index + 1]) for index in range(len(ordered) - 1)]
    if sum(stop - start for start, stop in slices) != length:
        raise RuntimeError("shifted-grid axis does not cover exactly once")
    return slices


def grid_slices(height: int, width: int, offset: tuple[int, int]) -> list[list[tuple[int, int, int, int]]]:
    rows = axis_slices(height, offset[0])
    columns = axis_slices(width, offset[1])
    tiles = [
        [(top, bottom, left, right) for left, right in columns]
        for top, bottom in rows
    ]
    covered = sum((bottom - top) * (right - left) for row in tiles for top, bottom, left, right in row)
    if covered != height * width:
        raise RuntimeError("shifted grid does not cover image exactly once")
    return tiles


def apply_scale(hazy: np.ndarray, prediction: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return prediction.copy()
    return np.clip(
        hazy + np.float32(scale) * (prediction - hazy), 0.0, 1.0,
    ).astype(np.float32)


def prepare_variant(hazy: np.ndarray, clear: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    if hazy.shape != clear.shape or hazy.shape != prediction.shape:
        raise RuntimeError("variant arrays have inconsistent shapes")
    candidates = [apply_scale(hazy, prediction, scale) for _, scale in ACTION_SET]
    raw_candidates = [
        hazy + np.float32(scale) * (prediction - hazy) for _, scale in ACTION_SET
    ]
    clip_masks = [
        np.any((raw < 0.0) | (raw > 1.0), axis=2) for raw in raw_candidates
    ]
    common_nonclipped = ~np.logical_or.reduce(clip_masks)
    errors = [mse(candidate, clear) for candidate in candidates]
    uniform_index = int(np.argmin(np.asarray(errors, dtype=np.float64)))
    if not np.array_equal(candidates[0], prediction):
        raise RuntimeError("keep action does not exactly reproduce official prediction")
    return {
        "hazy": hazy,
        "clear": clear,
        "prediction": prediction,
        "candidates": candidates,
        "clip_masks": clip_masks,
        "clip_rates": [float(np.mean(mask)) for mask in clip_masks],
        "common_nonclipped": common_nonclipped,
        "common_nonclipped_fraction": float(np.mean(common_nonclipped)),
        "uniform_index": uniform_index,
        "uniform_output": candidates[uniform_index],
    }


def tile_utility_table(
    variant: dict[str, Any], tiles: list[list[tuple[int, int, int, int]]], *,
    common_nonclipped: bool,
) -> tuple[np.ndarray, np.ndarray]:
    rows, columns = len(tiles), len(tiles[0])
    utilities = np.zeros((rows, columns, len(ACTION_SET)), dtype=np.float64)
    valid = np.ones((rows, columns), dtype=bool)
    clear = variant["clear"]
    candidates = variant["candidates"]
    for row_index, row in enumerate(tiles):
        for column_index, (top, bottom, left, right) in enumerate(row):
            mask = None
            if common_nonclipped:
                mask = variant["common_nonclipped"][top:bottom, left:right]
                if not bool(np.any(mask)):
                    valid[row_index, column_index] = False
                    continue
            target = clear[top:bottom, left:right]
            errors = [
                mse(candidate[top:bottom, left:right], target, mask)
                for candidate in candidates
            ]
            utilities[row_index, column_index] = errors[0] - np.asarray(errors)
    return utilities, valid


def select_actions(scores: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    if scores.ndim != 3 or scores.shape[2] != len(ACTION_SET):
        raise RuntimeError("action score table has the wrong shape")
    selected = np.argmax(scores, axis=2).astype(np.int16)
    if valid is not None:
        selected = np.where(valid, selected, 0).astype(np.int16)
    return selected


def compose_action_field(
    variant: dict[str, Any], tiles: list[list[tuple[int, int, int, int]]],
    selected: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = variant["clear"].shape[:2]
    output = np.empty_like(variant["prediction"])
    dense_actions = np.empty((height, width), dtype=np.int16)
    for row_index, row in enumerate(tiles):
        for column_index, (top, bottom, left, right) in enumerate(row):
            action = int(selected[row_index, column_index])
            output[top:bottom, left:right] = variant["candidates"][action][top:bottom, left:right]
            dense_actions[top:bottom, left:right] = action
    return output, dense_actions


def output_from_dense_actions(variant: dict[str, Any], dense_actions: np.ndarray) -> np.ndarray:
    output = np.empty_like(variant["prediction"])
    for action_index in range(len(ACTION_SET)):
        mask = dense_actions == action_index
        output[mask] = variant["candidates"][action_index][mask]
    return output


def rgb_ssim(torch, outputs: list[np.ndarray], clear: np.ndarray, device: str) -> list[float]:
    import torch.nn.functional as functional

    stacked = np.stack(outputs, axis=0).transpose(0, 3, 1, 2).copy()
    reference = np.repeat(clear[None, ...], len(outputs), axis=0).transpose(0, 3, 1, 2).copy()
    x = torch.from_numpy(stacked).to(device=device, dtype=torch.float32)
    y = torch.from_numpy(reference).to(device=device, dtype=torch.float32)

    def local_mean(value):
        return functional.avg_pool2d(
            functional.pad(value, (5, 5, 5, 5), mode="reflect"), 11, stride=1,
        )

    mu_x, mu_y = local_mean(x), local_mean(y)
    variance_x = torch.clamp(local_mean(x * x) - mu_x * mu_x, min=0.0)
    variance_y = torch.clamp(local_mean(y * y) - mu_y * mu_y, min=0.0)
    covariance = local_mean(x * y) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    score = ((2.0 * mu_x * mu_y + c1) * (2.0 * covariance + c2)) / (
        (mu_x * mu_x + mu_y * mu_y + c1)
        * (variance_x + variance_y + c2)
    )
    values = score.mean(dim=(1, 2, 3)).detach().cpu().numpy().astype(np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError("non-finite RGB SSIM")
    return [float(value) for value in values]


def tile_margin_summary(
    scores: np.ndarray, tiles: list[list[tuple[int, int, int, int]]],
) -> dict[str, float]:
    ordered = np.sort(scores, axis=2)
    gaps = ordered[:, :, -1] - ordered[:, :, -2]
    weighted_sum = 0.0
    tie_area = 0
    nonkeep_area = 0
    total_area = 0
    selected = select_actions(scores)
    for row_index, row in enumerate(tiles):
        for column_index, (top, bottom, left, right) in enumerate(row):
            area = (bottom - top) * (right - left)
            total_area += area
            weighted_sum += float(gaps[row_index, column_index]) * area
            if float(gaps[row_index, column_index]) <= NUMERICAL_TIE_TOLERANCE:
                tie_area += area
            if int(selected[row_index, column_index]) != 0:
                nonkeep_area += area
    return {
        "mean_top_two_mse_margin": weighted_sum / total_area,
        "exact_near_tie_pixel_fraction": tie_area / total_area,
        "nonkeep_pixel_fraction": nonkeep_area / total_area,
    }


def target_measurement(
    torch,
    prepared: list[dict[str, Any]],
    tables: list[dict[str, Any]],
    target_index: int,
    tiles: list[list[tuple[int, int, int, int]]],
    device: str,
) -> dict[str, float]:
    source_indices = [index for index in range(VARIANTS_PER_SCENE) if index != target_index]
    source_scores = np.mean(
        np.stack([tables[index]["utility"] for index in source_indices], axis=0), axis=0,
    )
    source_common_scores = np.mean(
        np.stack([tables[index]["common_utility"] for index in source_indices], axis=0), axis=0,
    )
    source_common_valid = np.logical_and.reduce(
        [tables[index]["common_valid"] for index in source_indices],
    )
    target = prepared[target_index]
    transferred_actions = select_actions(source_scores)
    nonclipped_actions = select_actions(source_common_scores, source_common_valid)
    oracle_actions = select_actions(tables[target_index]["utility"])
    transferred, dense_actions = compose_action_field(target, tiles, transferred_actions)
    nonclipped, _ = compose_action_field(target, tiles, nonclipped_actions)
    oracle, _ = compose_action_field(target, tiles, oracle_actions)
    shifted_actions = np.roll(dense_actions, shift=(TILE_SIZE, TILE_SIZE), axis=(0, 1))
    shifted = output_from_dense_actions(target, shifted_actions)
    uniform = target["uniform_output"]

    oracle_psnr = psnr_from_mse(mse(oracle, target["clear"]))
    transferred_psnr = psnr_from_mse(mse(transferred, target["clear"]))
    uniform_psnr = psnr_from_mse(mse(uniform, target["clear"]))
    shifted_psnr = psnr_from_mse(mse(shifted, target["clear"]))
    nonclipped_psnr = psnr_from_mse(mse(nonclipped, target["clear"]))
    regret = oracle_psnr - transferred_psnr
    if regret < -1e-8:
        raise RuntimeError("held-out GT oracle was worse than the transferred action field")
    uniform_ssim, transferred_ssim = rgb_ssim(
        torch, [uniform, transferred], target["clear"], device,
    )
    margin = tile_margin_summary(source_scores, tiles)
    selected_clip = np.zeros(dense_actions.shape, dtype=bool)
    for action_index, clip_mask in enumerate(target["clip_masks"]):
        selected_clip |= (dense_actions == action_index) & clip_mask
    return {
        "regret_psnr_db": max(0.0, regret),
        "transferred_minus_uniform_psnr_db": transferred_psnr - uniform_psnr,
        "aligned_minus_shifted_psnr_db": transferred_psnr - shifted_psnr,
        "nonclipped_source_minus_uniform_psnr_db": nonclipped_psnr - uniform_psnr,
        "transferred_minus_uniform_ssim": transferred_ssim - uniform_ssim,
        "transferred_minus_uniform_color_bias": (
            color_bias(transferred, target["clear"]) - color_bias(uniform, target["clear"])
        ),
        "mean_top_two_mse_margin": margin["mean_top_two_mse_margin"],
        "exact_near_tie_pixel_fraction": margin["exact_near_tie_pixel_fraction"],
        "nonkeep_pixel_fraction": margin["nonkeep_pixel_fraction"],
        "selected_clip_pixel_fraction": float(np.mean(selected_clip)),
        "common_nonclipped_pixel_fraction": float(np.mean([
            prepared[index]["common_nonclipped_fraction"] for index in source_indices
        ])),
        "keep_duplicate_utility": mse(target["prediction"], target["clear"]) - mse(
            apply_scale(target["hazy"], target["prediction"], 1.0), target["clear"],
        ),
    }


def scene_measurement(
    torch, variants: list[dict[str, np.ndarray]], device: str,
) -> dict[str, Any]:
    if len(variants) != VARIANTS_PER_SCENE:
        raise RuntimeError("scene measurement requires exactly four haze variants")
    shapes = {record["clear"].shape for record in variants}
    if len(shapes) != 1:
        raise RuntimeError("nested variants are not spatially registered")
    prepared = [
        prepare_variant(record["hazy"], record["clear"], record["prediction"])
        for record in variants
    ]
    height, width = prepared[0]["clear"].shape[:2]
    grid_records: list[dict[str, Any]] = []
    for offset in GRID_OFFSETS:
        tiles = grid_slices(height, width, offset)
        tables = []
        for variant in prepared:
            utility, valid = tile_utility_table(variant, tiles, common_nonclipped=False)
            common_utility, common_valid = tile_utility_table(
                variant, tiles, common_nonclipped=True,
            )
            tables.append({
                "utility": utility,
                "valid": valid,
                "common_utility": common_utility,
                "common_valid": common_valid,
            })
        target_records = [
            target_measurement(torch, prepared, tables, index, tiles, device)
            for index in range(VARIANTS_PER_SCENE)
        ]
        grid_records.append({
            "offset": list(offset),
            "tile_rows": len(tiles),
            "tile_columns": len(tiles[0]),
            **{
                key: float(np.mean([record[key] for record in target_records]))
                for key in target_records[0]
            },
        })
    return {
        "regret_psnr_db": max(record["regret_psnr_db"] for record in grid_records),
        "robust_gain_psnr_db": min(
            record["transferred_minus_uniform_psnr_db"] for record in grid_records
        ),
        "aligned_minus_shifted_psnr_db": min(
            record["aligned_minus_shifted_psnr_db"] for record in grid_records
        ),
        "nonclipped_source_minus_uniform_psnr_db": min(
            record["nonclipped_source_minus_uniform_psnr_db"] for record in grid_records
        ),
        "transferred_minus_uniform_ssim": min(
            record["transferred_minus_uniform_ssim"] for record in grid_records
        ),
        "transferred_minus_uniform_color_bias": max(
            record["transferred_minus_uniform_color_bias"] for record in grid_records
        ),
        "mean_top_two_mse_margin": float(np.mean([
            record["mean_top_two_mse_margin"] for record in grid_records
        ])),
        "exact_near_tie_pixel_fraction": float(np.mean([
            record["exact_near_tie_pixel_fraction"] for record in grid_records
        ])),
        "nonkeep_pixel_fraction": float(np.mean([
            record["nonkeep_pixel_fraction"] for record in grid_records
        ])),
        "selected_clip_pixel_fraction": float(np.mean([
            record["selected_clip_pixel_fraction"] for record in grid_records
        ])),
        "common_nonclipped_pixel_fraction": float(np.mean([
            record["common_nonclipped_pixel_fraction"] for record in grid_records
        ])),
        "maximum_absolute_keep_duplicate_utility": max(
            abs(record["keep_duplicate_utility"]) for record in grid_records
        ),
        "grid_records": grid_records,
    }


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


def evaluate_gates(
    integrity: dict[str, bool], scenes: list[dict[str, Any]], failures: list[dict[str, str]],
) -> dict[str, Any]:
    complete = all(integrity.values()) and len(scenes) == EXPECTED_SCENES and not failures
    regret = scene_bootstrap(
        (scene["regret_psnr_db"] for scene in scenes), BOOTSTRAP_SEEDS["regret"],
    ) if scenes else None
    aligned_shift = scene_bootstrap(
        (scene["aligned_minus_shifted_psnr_db"] for scene in scenes),
        BOOTSTRAP_SEEDS["aligned_shift"],
    ) if scenes else None
    nonclipped_gain = scene_bootstrap(
        (scene["nonclipped_source_minus_uniform_psnr_db"] for scene in scenes),
        BOOTSTRAP_SEEDS["nonclipped_gain"],
    ) if scenes else None
    benefit = wilson(
        sum(scene["robust_gain_psnr_db"] >= MATERIAL_GAIN_DB for scene in scenes),
        EXPECTED_SCENES,
    )
    ssim_harm = wilson(
        sum(scene["transferred_minus_uniform_ssim"] <= -SSIM_HARM_MARGIN for scene in scenes),
        EXPECTED_SCENES,
    )
    color_harm = wilson(
        sum(scene["transferred_minus_uniform_color_bias"] >= COLOR_HARM_MARGIN for scene in scenes),
        EXPECTED_SCENES,
    )
    precision_pass = bool(
        regret is not None
        and regret["upper_one_sided_95"] < REGRET_MARGIN_DB
        and regret["upper_distance"] <= PRECISION_DISTANCE_DB
    )
    coverage_pass = benefit["lower"] >= MIN_MATERIAL_SCENE_PREVALENCE
    negative_control_pass = bool(
        aligned_shift is not None and aligned_shift["lower_one_sided_95"] > 0.0
    )
    nonclipped_pass = bool(
        nonclipped_gain is not None and nonclipped_gain["lower_one_sided_95"] >= 0.0
    )
    safety_pass = bool(
        ssim_harm["upper"] < MAX_HARM_PREVALENCE
        and color_harm["upper"] < MAX_HARM_PREVALENCE
    )
    regret_fail = bool(
        regret is not None and regret["lower_one_sided_95"] >= REGRET_MARGIN_DB
    )
    coverage_fail = benefit["upper"] < MIN_MATERIAL_SCENE_PREVALENCE
    negative_control_fail = bool(
        aligned_shift is not None and aligned_shift["upper_one_sided_95"] <= 0.0
    )
    nonclipped_fail = bool(
        nonclipped_gain is not None and nonclipped_gain["upper_one_sided_95"] < 0.0
    )
    safety_fail = bool(
        ssim_harm["lower"] >= MAX_HARM_PREVALENCE
        or color_harm["lower"] >= MAX_HARM_PREVALENCE
    )
    if not complete:
        terminal = {
            "state": "COMPLETED_INCONCLUSIVE",
            "decision": "HAZE4K_TEST_LOCAL_ACTION_MEASUREMENT_INCONCLUSIVE",
            "authorizes": "STAGE1_REASSESSMENT_ONLY",
            "gate_reasons": ["complete 100-scene measurement integrity was not established"],
        }
    elif regret_fail or coverage_fail or negative_control_fail or nonclipped_fail or safety_fail:
        terminal = {
            "state": "COMPLETED_GATE_FAIL",
            "decision": "HAZE4K_TEST_LOCAL_ACTION_MEASUREMENT_FAIL",
            "authorizes": "NONE",
            "gate_reasons": ["at least one frozen measurement-validity interval clearly failed"],
        }
    elif precision_pass and coverage_pass and negative_control_pass and nonclipped_pass and safety_pass:
        terminal = {
            "state": "COMPLETED_GATE_PASS",
            "decision": "HAZE4K_TEST_LOCAL_ACTION_MEASUREMENT_QUALIFIED",
            "authorizes": "STAGE1_REASSESSMENT_ONLY",
            "gate_reasons": ["the frozen regret, precision, coverage, negative-control, clipping, and safety gates all passed"],
        }
    else:
        terminal = {
            "state": "COMPLETED_INCONCLUSIVE",
            "decision": "HAZE4K_TEST_LOCAL_ACTION_MEASUREMENT_INCONCLUSIVE",
            "authorizes": "STAGE1_REASSESSMENT_ONLY",
            "gate_reasons": ["at least one frozen interval crossed its margin or the achieved regret precision exceeded 0.05 dB"],
        }
    return {
        "complete": complete,
        "regret": regret,
        "aligned_shift": aligned_shift,
        "nonclipped_gain": nonclipped_gain,
        "benefit": benefit,
        "ssim_harm": ssim_harm,
        "color_harm": color_harm,
        "gates": {
            "primary_regret_and_precision": precision_pass,
            "material_scene_coverage": coverage_pass,
            "aligned_location_negative_control": negative_control_pass,
            "nonclipped_direction": nonclipped_pass,
            "ssim_and_color_safety": safety_pass,
        },
        "terminal": terminal,
    }


def finalize(
    context, integrity: dict[str, bool], scenes: list[dict[str, Any]],
    failures: list[dict[str, str]],
) -> dict[str, Any]:
    evaluated = evaluate_gates(integrity, scenes, failures)
    terminal = evaluated["terminal"]
    summary = {
        "schema_version": 1,
        "route_id": context.route_id,
        "operation_id": context.operation_id,
        "run_id": context.run_id,
        "scope": "GT-derived local-action measurement qualification on isolated 100-scene Haze4K development partition",
        "identity_and_coverage": {
            "integrity_checks": integrity,
            "completed_scenes": len(scenes),
            "completed_variants": len(scenes) * VARIANTS_PER_SCENE,
            "failure_count": len(failures),
            "failures": failures[:20],
            "assignment_digest": SPLIT_ASSIGNMENT_DIGEST,
            "candidate_confirmation_asset_delivered": False,
            "network_or_proxy_training_occurred": False,
        },
        "measurement_contract": {
            "actions": [{"name": name, "scale": scale} for name, scale in ACTION_SET],
            "formula": "clip(hazy + scale * (official_prediction - hazy), 0, 1)",
            "leave_one_variant_out": "mean continuous per-pixel RGB-MSE utility over the other three variants selects each tile action",
            "grid_offsets": [list(offset) for offset in GRID_OFFSETS],
            "primary_scene_value": "maximum across grids of mean four-held-out-variant GT-oracle-minus-transferred PSNR regret",
            "independent_unit": "canonical_clear_scene",
            "nested_repeats": "four haze variants, tiles, pixels, grid offsets, and metrics never increase n",
        },
        "primary_regret": {
            "scene_mean_regret_psnr_db": evaluated["regret"],
            "maximum_upper_bound_db": REGRET_MARGIN_DB,
            "maximum_upper_distance_db": PRECISION_DISTANCE_DB,
            "pre_run_precision_mode": "descriptive_capacity because the target regret scene SD was unavailable before this supplement",
        },
        "coverage": {
            "material_scene_gain_db": MATERIAL_GAIN_DB,
            "robust_material_scene_prevalence": evaluated["benefit"],
            "minimum_prevalence_lower_bound": MIN_MATERIAL_SCENE_PREVALENCE,
        },
        "controls": {
            "aligned_minus_circularly_shifted_psnr_db": evaluated["aligned_shift"],
            "nonclipped_source_minus_uniform_psnr_db": evaluated["nonclipped_gain"],
            "maximum_absolute_keep_duplicate_utility": max(
                (scene["maximum_absolute_keep_duplicate_utility"] for scene in scenes),
                default=None,
            ),
        },
        "safety": {
            "ssim_harm_definition": f"worst-grid scene mean transferred-minus-uniform SSIM at most -{SSIM_HARM_MARGIN}",
            "ssim_harm_prevalence": evaluated["ssim_harm"],
            "color_harm_definition": f"worst-grid scene mean transferred-minus-uniform RGB mean-bias at least {COLOR_HARM_MARGIN}",
            "color_harm_prevalence": evaluated["color_harm"],
            "maximum_harm_upper_bound": MAX_HARM_PREVALENCE,
        },
        "secondary_scene_aggregates": {
            key: aggregate(scene[key] for scene in scenes)
            for key in (
                "regret_psnr_db",
                "robust_gain_psnr_db",
                "aligned_minus_shifted_psnr_db",
                "nonclipped_source_minus_uniform_psnr_db",
                "transferred_minus_uniform_ssim",
                "transferred_minus_uniform_color_bias",
                "mean_top_two_mse_margin",
                "exact_near_tie_pixel_fraction",
                "nonkeep_pixel_fraction",
                "selected_clip_pixel_fraction",
                "common_nonclipped_pixel_fraction",
            )
        },
        "gates": evaluated["gates"],
        "terminal": terminal,
        "limitations": [
            "This is development-screening measurement evidence from 100 canonical synthetic Haze4K scenes.",
            "The pre-run continuous-regret variance was unavailable, so the route records descriptive capacity and requires stage-1 reassessment even after a gate pass.",
            "GT selects source utilities and held-out oracle outcomes; no result establishes deployment-time predictability.",
            "RGB reconstruction utility and limited SSIM/color safety do not establish perceptual quality or physical haze state.",
            "The result is specific to the fixed three residual scales and four declared 32-pixel grid origins.",
            "The protected candidate-confirmation asset and NH-HAZE were not delivered or accessed.",
        ],
        "marker": "HAZE4K_TEST_LOCAL_ACTION_MEASUREMENT_QUALIFICATION_COMPLETE",
    }
    summary_name = "haze4k_test_local_action_measurement_qualification_v1_summary.json"
    strata_name = "haze4k_test_local_action_measurement_qualification_v1_strata.csv"
    atomic_json(output_file(context, summary_name), summary)
    rows = [
        {
            "estimand": "scene_mean_worst_grid_regret_psnr_db",
            "method": "scene_bootstrap",
            "scenes": EXPECTED_SCENES,
            "events": "",
            "estimate": evaluated["regret"]["estimate"] if evaluated["regret"] else "",
            "lower_one_sided_95": evaluated["regret"]["lower_one_sided_95"] if evaluated["regret"] else "",
            "upper_one_sided_95": evaluated["regret"]["upper_one_sided_95"] if evaluated["regret"] else "",
            "threshold": REGRET_MARGIN_DB,
        },
        {
            "estimand": "robust_material_gain_scene_prevalence",
            "method": "wilson",
            "scenes": EXPECTED_SCENES,
            "events": evaluated["benefit"]["successes"],
            "estimate": evaluated["benefit"]["estimate"],
            "lower_one_sided_95": evaluated["benefit"]["lower"],
            "upper_one_sided_95": evaluated["benefit"]["upper"],
            "threshold": MIN_MATERIAL_SCENE_PREVALENCE,
        },
        {
            "estimand": "aligned_minus_circular_shifted_psnr_db",
            "method": "scene_bootstrap",
            "scenes": EXPECTED_SCENES,
            "events": "",
            "estimate": evaluated["aligned_shift"]["estimate"] if evaluated["aligned_shift"] else "",
            "lower_one_sided_95": evaluated["aligned_shift"]["lower_one_sided_95"] if evaluated["aligned_shift"] else "",
            "upper_one_sided_95": evaluated["aligned_shift"]["upper_one_sided_95"] if evaluated["aligned_shift"] else "",
            "threshold": 0.0,
        },
        {
            "estimand": "nonclipped_source_minus_uniform_psnr_db",
            "method": "scene_bootstrap",
            "scenes": EXPECTED_SCENES,
            "events": "",
            "estimate": evaluated["nonclipped_gain"]["estimate"] if evaluated["nonclipped_gain"] else "",
            "lower_one_sided_95": evaluated["nonclipped_gain"]["lower_one_sided_95"] if evaluated["nonclipped_gain"] else "",
            "upper_one_sided_95": evaluated["nonclipped_gain"]["upper_one_sided_95"] if evaluated["nonclipped_gain"] else "",
            "threshold": 0.0,
        },
        {
            "estimand": "ssim_harm_scene_prevalence",
            "method": "wilson",
            "scenes": EXPECTED_SCENES,
            "events": evaluated["ssim_harm"]["successes"],
            "estimate": evaluated["ssim_harm"]["estimate"],
            "lower_one_sided_95": evaluated["ssim_harm"]["lower"],
            "upper_one_sided_95": evaluated["ssim_harm"]["upper"],
            "threshold": MAX_HARM_PREVALENCE,
        },
        {
            "estimand": "color_harm_scene_prevalence",
            "method": "wilson",
            "scenes": EXPECTED_SCENES,
            "events": evaluated["color_harm"]["successes"],
            "estimate": evaluated["color_harm"]["estimate"],
            "lower_one_sided_95": evaluated["color_harm"]["lower"],
            "upper_one_sided_95": evaluated["color_harm"]["upper"],
            "threshold": MAX_HARM_PREVALENCE,
        },
    ]
    with output_file(context, strata_name).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {
        "state": terminal["state"],
        "decision": terminal["decision"],
        "authorizes": terminal["authorizes"],
        "details": {
            "summary_file": summary_name,
            "strata_file": strata_name,
            "independent_scenes": len(scenes),
            "nested_variants": len(scenes) * VARIANTS_PER_SCENE,
            "gate_reasons": terminal["gate_reasons"],
            "candidate_confirmation_asset_delivered": False,
            "network_or_proxy_training_occurred": False,
        },
    }


def contract(context_path: Path) -> None:
    context = load_context(context_path, "contract")
    prepare_phase_output(context)
    if context.device != "cuda" or any(context.protected_data_permissions.values()):
        raise RuntimeError("measurement contract requires CUDA and no protected permission")
    if "haze4k_test_development" in context.assets:
        raise RuntimeError("scientific development data must be absent from contract phase")
    torch, model = load_official_model(context)
    height, width = 256, 320
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    clear = np.stack(
        (
            0.12 + 0.62 * xx / width,
            0.18 + 0.56 * yy / height,
            0.16 + 0.58 * (xx + 2.0 * yy) / (width + 2.0 * height),
        ),
        axis=-1,
    ).astype(np.float32)
    synthetic_hazy = (0.72 * clear + 0.18).astype(np.float32)
    inferred = infer(torch, model, synthetic_hazy, context.device)
    variants = []
    for index, transmission in enumerate((0.62, 0.68, 0.74, 0.80)):
        hazy = (transmission * clear + (1.0 - transmission) * 0.72).astype(np.float32)
        horizontal = np.where(xx < width * 0.43, 1.24, 0.76).astype(np.float32)
        vertical = np.where(yy < height * 0.61, 1.0, 1.10).astype(np.float32)
        factor = horizontal * vertical + np.float32((index - 1.5) * 0.01)
        manufactured = np.clip(
            hazy + factor[..., None] * (clear - hazy), 0.0, 1.0,
        ).astype(np.float32)
        variants.append({"hazy": hazy, "clear": clear, "prediction": manufactured})
    synthetic_scene = scene_measurement(torch, variants, context.device)
    repeated = [synthetic_scene for _ in range(EXPECTED_SCENES)]
    exercised = evaluate_gates(
        {
            "parent_oracle_evidence": True,
            "closed_proxy_evidence": True,
            "isolated_development_asset_only": True,
            "complete_100_scene_400_variant_grouping": True,
            "complete_finite_inference_and_measurement": True,
            "candidate_confirmation_asset_not_delivered": True,
            "no_training": True,
            "measurement_controls_exact": True,
        },
        repeated,
        [],
    )
    checks = {
        "strict_checkpoint_load_and_parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ) == PARAMETER_COUNT,
        "all_parameters_frozen": not any(parameter.requires_grad for parameter in model.parameters()),
        "three_scale_finite_forward": bool(np.isfinite(inferred).all()),
        "four_shifted_grids_complete": len(synthetic_scene["grid_records"]) == len(GRID_OFFSETS),
        "keep_duplicate_exact": synthetic_scene["maximum_absolute_keep_duplicate_utility"] == 0.0,
        "held_out_regret_nonnegative": synthetic_scene["regret_psnr_db"] >= 0.0,
        "negative_control_finite": math.isfinite(
            synthetic_scene["aligned_minus_shifted_psnr_db"],
        ),
        "bootstrap_and_terminal_finite": exercised["regret"] is not None
        and exercised["terminal"]["state"] in {
            "COMPLETED_GATE_PASS", "COMPLETED_GATE_FAIL", "COMPLETED_INCONCLUSIVE",
        },
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
    oracle_closeout_path = asset_path(context, "prior_oracle_closeout", kind="file")
    oracle_summary_path = asset_path(context, "prior_oracle_summary", kind="file")
    proxy_closeout_path = asset_path(context, "prior_proxy_closeout", kind="file")
    proxy_conclusion_path = asset_path(context, "prior_proxy_conclusion", kind="file")
    expected_assets = {
        "prior_oracle_closeout": ORACLE_CLOSEOUT_SHA256,
        "prior_oracle_summary": ORACLE_SUMMARY_SHA256,
        "prior_proxy_closeout": PROXY_CLOSEOUT_SHA256,
        "prior_proxy_conclusion": PROXY_CONCLUSION_SHA256,
    }
    for identifier, identity in expected_assets.items():
        if context.assets[identifier].sha256 != identity:
            raise RuntimeError(f"formal evidence identity changed for {identifier}")
    oracle_closeout = json.loads(oracle_closeout_path.read_text(encoding="utf-8"))
    oracle_summary = json.loads(oracle_summary_path.read_text(encoding="utf-8"))
    proxy_closeout = json.loads(proxy_closeout_path.read_text(encoding="utf-8"))
    proxy_conclusion = json.loads(proxy_conclusion_path.read_text(encoding="utf-8"))
    oracle_ok = (
        oracle_closeout.get("state") == "COMPLETED_GATE_PASS"
        and oracle_closeout.get("decision") == "HAZE4K_TEST_BOUNDED_LOCAL_ACTION_ORACLE_HEADROOM_PASS"
        and oracle_closeout.get("authorizes") == "HAZE4K_TEST_LOCAL_ACTION_PROXY_PREDICTABILITY"
        and oracle_summary.get("identity_and_coverage", {}).get("assignment_digest")
        == SPLIT_ASSIGNMENT_DIGEST
    )
    proxy_ok = (
        proxy_closeout.get("state") == "COMPLETED_GATE_FAIL"
        and proxy_closeout.get("decision") == "HAZE4K_TEST_LOCAL_ACTION_PROXY_PREDICTABILITY_FAIL"
        and proxy_closeout.get("authorizes") == "NONE"
        and proxy_conclusion.get("authorizes") == "NONE"
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
        len(hazy_paths) == EXPECTED_VARIANTS
        and len(clear_paths) == EXPECTED_VARIANTS
        and exact_names
        and len(clear_by_digest) == EXPECTED_SCENES
        and len(variants_by_digest) == EXPECTED_SCENES
        and histogram == {VARIANTS_PER_SCENE: EXPECTED_SCENES}
        and not failures
    )
    write_workload_progress(context, completed_units=2, stage="canonical_scene_grouping")

    scenes: list[dict[str, Any]] = []
    if oracle_ok and proxy_ok and scope_ok and dataset_ok:
        torch, model = load_official_model(context)
        attempted = 0
        for digest in sorted(variants_by_digest):
            variants = []
            for hazy_path in sorted(variants_by_digest[digest]):
                attempted += 1
                try:
                    hazy = image_array(hazy_path)
                    clear = image_array(clear_root / hazy_path.name)
                    prediction = infer(torch, model, hazy, context.device)
                    variants.append({"hazy": hazy, "clear": clear, "prediction": prediction})
                except Exception as exc:
                    failures.append({
                        "scene": digest[:16], "variant": hazy_path.name,
                        "reason": str(exc)[:512],
                    })
                if attempted % 5 == 0:
                    write_workload_progress(
                        context,
                        completed_units=2 + attempted,
                        stage="official_inference_and_measurement",
                    )
            if len(variants) == VARIANTS_PER_SCENE:
                try:
                    scenes.append(scene_measurement(torch, variants, context.device))
                except Exception as exc:
                    failures.append({
                        "scene": digest[:16], "variant": "scene_measurement",
                        "reason": str(exc)[:512],
                    })

    integrity = {
        "parent_oracle_evidence": oracle_ok,
        "closed_proxy_evidence": proxy_ok,
        "isolated_development_asset_only": scope_ok,
        "complete_100_scene_400_variant_grouping": dataset_ok,
        "complete_finite_inference_and_measurement": len(scenes) == EXPECTED_SCENES and not failures,
        "candidate_confirmation_asset_not_delivered": True,
        "no_training": True,
        "measurement_controls_exact": all(
            scene["maximum_absolute_keep_duplicate_utility"] == 0.0 for scene in scenes
        ) if scenes else False,
    }
    result = finalize(context, integrity, scenes, failures)
    write_workload_progress(context, completed_units=403, stage="scene_level_measurement_finalize")
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
