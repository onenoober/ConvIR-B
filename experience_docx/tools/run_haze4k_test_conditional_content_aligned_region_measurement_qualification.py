#!/usr/bin/env python3
"""Qualify source-only content-aligned regions for conditional local utility."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.morphology import binary_dilation
from skimage.segmentation import find_boundaries, watershed

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
REGION_PHASES = GRID_OFFSETS
REGION_SPACING = 32
ACTION_SET = (("keep", 1.0), ("weaken", 0.8), ("strengthen", 1.2))
SPLIT_ASSIGNMENT_DIGEST = "6ca5174470dad2b4eef4ae15c5a13a99d8ae9fc0bc2ea1116b199c4d4bc05582"
ORACLE_CLOSEOUT_SHA256 = "96357cbaaee5aa338fb0f9c9835a975a27e7f048c78a096fd644e6acdd3e383c"
ORACLE_SUMMARY_SHA256 = "794ef27733f51f2fa70fab5c94bc661564d7b988ef4247a1b26e33a21b4de7cb"
PROXY_CLOSEOUT_SHA256 = "c12a4b013db12b0828823ae8cba7fe8a51208971d29e667966635c157a67cba4"
PROXY_CONCLUSION_SHA256 = "c8419d0b0d23b079b55b3490c54347033fa79221bcc035d995d5e55dcaa1695b"
PRIOR_MEASUREMENT_CLOSEOUT_SHA256 = "ee323807f1addb504b866d74f198e94d2df34cce458abafcd7e84e4c4b801163"
PRIOR_MEASUREMENT_SUMMARY_SHA256 = "97ac2086751c5e6ef1f08c0ee0f6777045c347503c9be24d2d82506645bbbcbe"
PRIOR_MEASUREMENT_CONCLUSION_SHA256 = "5bd16cdef5c238c2cdea107028c259b0f31847c90dae05b5055929b8ae5225b5"
PARENT_CONDITIONAL_CLOSEOUT_SHA256 = "d63ace91dad6cbab3cff8a18c37c01a705ea5cd7e1ccab9da1317bc3ee4a0bca"
PARENT_CONDITIONAL_SUMMARY_SHA256 = "ca00a1365f7e804e0254936569b28ec11f038514252e3c73a283740b03e03517"
PARENT_CONDITIONAL_CONCLUSION_SHA256 = "1de48c122a1621922de4602c8ef732bb00f66d735ae5eb2ebd1f7a03e422d543"
PARENT_TAPER_CLOSEOUT_SHA256 = "a41759573c5d8ccf1db09f30ac59ecf67cf9f09072d79122e625d8f8fa4af497"
PARENT_TAPER_SUMMARY_SHA256 = "6ca61e3473ef1c92eca1415523b56c7b592bf5c663f03b5b01214861409619fb"
PARENT_TAPER_CONCLUSION_SHA256 = "3fe676574c1f0126409be55cd9eaef7a71830f80fc857553d581c596514ce8f8"
CHECKPOINT_SHA256 = "6f42037d57a4e3de3a10ac0ab909d66a3415864a19433c29204a975f4efa4088"
MODEL_SOURCE_SHA256 = "3fa227af396464a7f07ac773f92e9cdb746e0fa6ae63adef711c765a02c3d4cd"
MODEL_LAYERS_SHA256 = "ac8a05bd626d9adda16308dedb9466f36d7ff44cfb666f64e7e14ddf8cdf43a4"
PARAMETER_COUNT = 8_630_665
BOOTSTRAP_RESAMPLES = 20_000
BOOTSTRAP_SEEDS = {
    "regret": 20260801,
    "aligned_shift": 20260802,
    "nonclipped_gain": 20260803,
}
REGRET_MARGIN_DB = 0.10
PRECISION_DISTANCE_DB = 0.05
BOX_CONTROL_EXPECTED_REGRET_DB = 0.1423354325027467
BOX_CONTROL_TOLERANCE_DB = 1e-8
TAPER_CONTROL_EXPECTED_REGRET_DB = 0.1340031213991598
TAPER_CONTROL_TOLERANCE_DB = 1e-8
MATERIAL_GAIN_DB = 0.10
MIN_MATERIAL_SCENE_PREVALENCE = 0.20
SSIM_HARM_MARGIN = 0.005
COLOR_HARM_MARGIN = 1.0 / 255.0
MAX_HARM_PREVALENCE = 0.10
EPSILON = 1e-12
NUMERICAL_TIE_TOLERANCE = 1e-12
KEEP_DIAGNOSTIC_ERROR_BOUND = 64.0 * np.finfo(np.float64).eps
BOUNDARY_DIAGNOSTIC_BAND_PIXELS = 4
CONTRACT_MAX_SECONDS = 600.0


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
    affine_manipulation_exact = all(
        np.array_equal(candidate[~clip_mask], raw[~clip_mask])
        for index, (candidate, raw, clip_mask) in enumerate(
            zip(candidates, raw_candidates, clip_masks)
        )
        if index != 0
    )
    errors = [mse(candidate, clear) for candidate in candidates]
    uniform_index = int(np.argmin(np.asarray(errors, dtype=np.float64)))
    common_uniform_index = 0
    if bool(np.any(common_nonclipped)):
        common_errors = [
            mse(candidate, clear, common_nonclipped) for candidate in candidates
        ]
        common_uniform_index = int(np.argmin(np.asarray(common_errors, dtype=np.float64)))
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
        "affine_manipulation_exact": affine_manipulation_exact,
        "uniform_index": uniform_index,
        "uniform_output": candidates[uniform_index],
        "common_uniform_output": candidates[common_uniform_index],
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


def raised_cosine_tile_weight(height: int, width: int) -> np.ndarray:
    """Return a strictly-positive separable half-sample raised-cosine window."""
    if height <= 0 or width <= 0:
        raise ValueError("tile weight requires positive dimensions")
    row_phase = (np.arange(height, dtype=np.float64) + 0.5) / height
    column_phase = (np.arange(width, dtype=np.float64) + 0.5) / width
    row_weight = np.sin(np.pi * row_phase) ** 2
    column_weight = np.sin(np.pi * column_phase) ** 2
    weight = row_weight[:, None] * column_weight[None, :]
    if not np.isfinite(weight).all() or not bool(np.all(weight > 0.0)):
        raise RuntimeError("raised-cosine weight is not finite and strictly positive")
    return weight


def tile_boundary_mask(
    tiles: list[list[tuple[int, int, int, int]]], height: int, width: int, band: int,
) -> np.ndarray:
    """Mark a fixed-width band around internal held-out-grid boundaries."""
    mask = np.zeros((height, width), dtype=bool)
    row_boundaries = sorted({bottom for row in tiles for _, bottom, _, _ in row if bottom < height})
    column_boundaries = sorted({right for row in tiles for _, _, _, right in row if right < width})
    for boundary in row_boundaries:
        mask[max(0, boundary - band):min(height, boundary + band), :] = True
    for boundary in column_boundaries:
        mask[:, max(0, boundary - band):min(width, boundary + band)] = True
    return mask


def project_grid_scores(
    grids: list[dict[str, Any]], variant_index: int, held_out_index: int,
    table_key: str, valid_key: str | None, pixel_mask: np.ndarray | None = None,
    *, projection: str = "taper",
) -> tuple[np.ndarray, np.ndarray, bool, float]:
    target_tiles = grids[held_out_index]["tiles"]
    height = target_tiles[-1][-1][1]
    width = target_tiles[-1][-1][3]
    if projection not in {"box", "taper"}:
        raise ValueError("projection must be box or taper")
    dense_sum = np.zeros((height, width, len(ACTION_SET)), dtype=np.float64)
    dense_weight = np.zeros((height, width), dtype=np.float64)
    dense_contributors = np.zeros((height, width), dtype=np.int16)
    source_count = len(grids) - 1
    for grid_index, grid in enumerate(grids):
        if grid_index == held_out_index:
            continue
        table = grid["tables"][variant_index]
        scores = table[table_key]
        valid = (
            table[valid_key] if valid_key is not None
            else np.ones(scores.shape[:2], dtype=bool)
        )
        for row_index, row in enumerate(grid["tiles"]):
            for column_index, (top, bottom, left, right) in enumerate(row):
                if not bool(valid[row_index, column_index]):
                    continue
                tile_height, tile_width = bottom - top, right - left
                weight = (
                    raised_cosine_tile_weight(tile_height, tile_width)
                    if projection == "taper"
                    else np.ones((tile_height, tile_width), dtype=np.float64)
                )
                support = (
                    np.ones((tile_height, tile_width), dtype=bool)
                    if pixel_mask is None else pixel_mask[top:bottom, left:right]
                )
                sum_region = dense_sum[top:bottom, left:right]
                weight_region = dense_weight[top:bottom, left:right]
                contributor_region = dense_contributors[top:bottom, left:right]
                sum_region[support] += weight[support, None] * scores[row_index, column_index]
                weight_region[support] += weight[support]
                contributor_region[support] += 1

    if pixel_mask is None:
        projection_complete = bool(
            np.all(dense_contributors == source_count)
            and np.all(np.isfinite(dense_weight))
            and np.all(dense_weight > 0.0)
        )
    else:
        projection_complete = bool(
            np.all(dense_contributors[pixel_mask] == source_count)
            and np.all(dense_contributors[~pixel_mask] == 0)
            and np.all(np.isfinite(dense_weight[pixel_mask]))
            and np.all(dense_weight[pixel_mask] > 0.0)
            and np.all(dense_weight[~pixel_mask] == 0.0)
        )
    target_scores = np.zeros(
        (len(target_tiles), len(target_tiles[0]), len(ACTION_SET)), dtype=np.float64,
    )
    target_valid = np.zeros(target_scores.shape[:2], dtype=bool)
    for row_index, row in enumerate(target_tiles):
        for column_index, (top, bottom, left, right) in enumerate(row):
            weights = dense_weight[top:bottom, left:right]
            support = weights > 0.0
            if not bool(np.any(support)):
                continue
            pixel_scores = dense_sum[top:bottom, left:right][support] / weights[support, None]
            target_scores[row_index, column_index] = np.mean(pixel_scores, axis=0)
            target_valid[row_index, column_index] = True
    return (
        target_scores,
        target_valid,
        projection_complete,
        float(np.mean(dense_weight > 0.0)),
    )


def projection_reference_check() -> bool:
    height, width = 64, 80
    grids: list[dict[str, Any]] = []
    for grid_index, offset in enumerate(GRID_OFFSETS):
        tiles = grid_slices(height, width, offset)
        scores = np.zeros(
            (len(tiles), len(tiles[0]), len(ACTION_SET)), dtype=np.float64,
        )
        for row_index, row in enumerate(tiles):
            for column_index, _ in enumerate(row):
                scores[row_index, column_index] = (
                    0.0,
                    (grid_index + 1) * 0.01 + row_index * 0.003,
                    (grid_index + 1) * 0.02 + column_index * 0.005,
                )
        grids.append({
            "offset": list(offset),
            "tiles": tiles,
            "tables": [{"utility": scores}],
        })

    for held_out_index, target_grid in enumerate(grids):
        for projection in ("box", "taper"):
            actual, valid, complete, coverage = project_grid_scores(
                grids, 0, held_out_index, "utility", None, projection=projection,
            )
            dense_sum = np.zeros((height, width, len(ACTION_SET)), dtype=np.float64)
            dense_weight = np.zeros((height, width), dtype=np.float64)
            dense_contributors = np.zeros((height, width), dtype=np.int16)
            for source_index, source_grid in enumerate(grids):
                if source_index == held_out_index:
                    continue
                source_scores = source_grid["tables"][0]["utility"]
                for source_row, source_tiles in enumerate(source_grid["tiles"]):
                    for source_column, (top, bottom, left, right) in enumerate(source_tiles):
                        weight = (
                            raised_cosine_tile_weight(bottom - top, right - left)
                            if projection == "taper"
                            else np.ones((bottom - top, right - left), dtype=np.float64)
                        )
                        dense_sum[top:bottom, left:right] += (
                            weight[..., None] * source_scores[source_row, source_column]
                        )
                        dense_weight[top:bottom, left:right] += weight
                        dense_contributors[top:bottom, left:right] += 1
            if not bool(np.all(dense_contributors == len(grids) - 1)):
                return False
            if not bool(np.all(dense_weight > 0.0)):
                return False
            dense_scores = dense_sum / dense_weight[..., None]
            expected = np.zeros_like(actual)
            for target_row, target_tiles in enumerate(target_grid["tiles"]):
                for target_column, (top, bottom, left, right) in enumerate(target_tiles):
                    expected[target_row, target_column] = np.mean(
                        dense_scores[top:bottom, left:right], axis=(0, 1),
                    )
            if not complete or coverage != 1.0 or not bool(np.all(valid)):
                return False
            if not np.allclose(
                actual, expected, rtol=0.0, atol=KEEP_DIAGNOSTIC_ERROR_BOUND,
            ):
                return False

        constant_grids = []
        constant_score = np.asarray((0.0, 0.125, -0.25), dtype=np.float64)
        for grid in grids:
            rows, columns = len(grid["tiles"]), len(grid["tiles"][0])
            scores = np.broadcast_to(
                constant_score, (rows, columns, len(ACTION_SET)),
            ).copy()
            constant_grids.append({
                "offset": grid["offset"], "tiles": grid["tiles"],
                "tables": [{"utility": scores}],
            })
        constant_actual, _, constant_complete, _ = project_grid_scores(
            constant_grids, 0, held_out_index, "utility", None, projection="taper",
        )
        if not constant_complete or not np.allclose(
            constant_actual, constant_score, rtol=0.0, atol=KEEP_DIAGNOSTIC_ERROR_BOUND,
        ):
            return False
    return True


def conditional_grid_measurement(
    torch,
    prepared: list[dict[str, Any]],
    grids: list[dict[str, Any]],
    variant_index: int,
    held_out_index: int,
    device: str,
) -> dict[str, float | bool]:
    target_grid = grids[held_out_index]
    tiles = target_grid["tiles"]
    target = prepared[variant_index]
    source_scores, source_valid, projection_complete, _ = project_grid_scores(
        grids, variant_index, held_out_index, "utility", None, projection="taper",
    )
    box_scores, box_valid, box_projection_complete, _ = project_grid_scores(
        grids, variant_index, held_out_index, "utility", None, projection="box",
    )
    source_common_scores, source_common_valid, common_projection_complete, common_projection_coverage = project_grid_scores(
        grids, variant_index, held_out_index, "common_utility", "common_valid",
        target["common_nonclipped"], projection="taper",
    )
    transferred_actions = select_actions(source_scores, source_valid)
    box_actions = select_actions(box_scores, box_valid)
    nonclipped_actions = select_actions(source_common_scores, source_common_valid)
    oracle_actions = select_actions(target_grid["tables"][variant_index]["utility"])
    transferred, dense_actions = compose_action_field(target, tiles, transferred_actions)
    box_transferred, box_dense_actions = compose_action_field(target, tiles, box_actions)
    nonclipped, _ = compose_action_field(target, tiles, nonclipped_actions)
    oracle, oracle_dense_actions = compose_action_field(target, tiles, oracle_actions)
    shifted_actions = np.roll(dense_actions, shift=(TILE_SIZE, TILE_SIZE), axis=(0, 1))
    shifted = output_from_dense_actions(target, shifted_actions)
    uniform = target["uniform_output"]

    oracle_psnr = psnr_from_mse(mse(oracle, target["clear"]))
    transferred_psnr = psnr_from_mse(mse(transferred, target["clear"]))
    box_transferred_psnr = psnr_from_mse(mse(box_transferred, target["clear"]))
    uniform_psnr = psnr_from_mse(mse(uniform, target["clear"]))
    shifted_psnr = psnr_from_mse(mse(shifted, target["clear"]))
    nonclipped_psnr = psnr_from_mse(mse(
        nonclipped, target["clear"], target["common_nonclipped"],
    ))
    uniform_nonclipped_psnr = psnr_from_mse(mse(
        target["common_uniform_output"], target["clear"], target["common_nonclipped"],
    ))
    regret = oracle_psnr - transferred_psnr
    box_regret = oracle_psnr - box_transferred_psnr
    if regret < -1e-8:
        raise RuntimeError("held-out GT oracle was worse than the transferred action field")
    if box_regret < -1e-8:
        raise RuntimeError("held-out GT oracle was worse than the box control field")
    uniform_ssim, transferred_ssim = rgb_ssim(
        torch, [uniform, transferred], target["clear"], device,
    )
    margin = tile_margin_summary(source_scores, tiles)
    selected_clip = np.zeros(dense_actions.shape, dtype=bool)
    for action_index, clip_mask in enumerate(target["clip_masks"]):
        selected_clip |= (dense_actions == action_index) & clip_mask
    boundary_mask = np.zeros(dense_actions.shape, dtype=bool)
    for source_index, source_grid in enumerate(grids):
        if source_index != held_out_index:
            boundary_mask |= tile_boundary_mask(
                source_grid["tiles"], dense_actions.shape[0], dense_actions.shape[1],
                BOUNDARY_DIAGNOSTIC_BAND_PIXELS,
            )
    interior_mask = ~boundary_mask
    action_disagreement = dense_actions != box_dense_actions
    if not bool(np.any(boundary_mask)) or not bool(np.any(interior_mask)):
        raise RuntimeError("boundary diagnostic masks are empty")
    boundary_regret = psnr_from_mse(mse(oracle, target["clear"], boundary_mask)) - (
        psnr_from_mse(mse(transferred, target["clear"], boundary_mask))
    )
    interior_regret = psnr_from_mse(mse(oracle, target["clear"], interior_mask)) - (
        psnr_from_mse(mse(transferred, target["clear"], interior_mask))
    )
    return {
        "regret_psnr_db": max(0.0, regret),
        "box_regret_psnr_db": max(0.0, box_regret),
        "taper_minus_box_psnr_db": transferred_psnr - box_transferred_psnr,
        "boundary_band_regret_psnr_db": boundary_regret,
        "interior_regret_psnr_db": interior_regret,
        "boundary_action_disagreement_fraction": float(np.mean(
            action_disagreement[boundary_mask],
        )),
        "interior_action_disagreement_fraction": float(np.mean(
            action_disagreement[interior_mask],
        )),
        "transferred_minus_uniform_psnr_db": transferred_psnr - uniform_psnr,
        "aligned_minus_shifted_psnr_db": transferred_psnr - shifted_psnr,
        "nonclipped_source_minus_uniform_psnr_db": (
            nonclipped_psnr - uniform_nonclipped_psnr
        ),
        "transferred_minus_uniform_ssim": transferred_ssim - uniform_ssim,
        "transferred_minus_uniform_color_bias": (
            color_bias(transferred, target["clear"]) - color_bias(uniform, target["clear"])
        ),
        "mean_top_two_mse_margin": margin["mean_top_two_mse_margin"],
        "exact_near_tie_pixel_fraction": margin["exact_near_tie_pixel_fraction"],
        "nonkeep_pixel_fraction": margin["nonkeep_pixel_fraction"],
        "selected_clip_pixel_fraction": float(np.mean(selected_clip)),
        "common_nonclipped_pixel_fraction": target["common_nonclipped_fraction"],
        "common_source_projection_coverage_fraction": common_projection_coverage,
        "hard_action_agreement_fraction": float(np.mean(
            dense_actions == oracle_dense_actions,
        )),
        "source_grid_projection_complete": projection_complete,
        "box_source_grid_projection_complete": box_projection_complete,
        "common_source_projection_complete": common_projection_complete,
        "keep_structural_identity": bool(np.array_equal(
            target["candidates"][0], target["prediction"],
        )),
        "affine_manipulation_exact": bool(target["affine_manipulation_exact"]),
        "keep_duplicate_utility": mse(target["prediction"], target["clear"]) - mse(
            apply_scale(target["hazy"], target["prediction"], 1.0), target["clear"],
        ),
    }


def taper_control_scene_measurement(
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
    grids: list[dict[str, Any]] = []
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
        grids.append({"offset": list(offset), "tiles": tiles, "tables": tables})

    mean_keys = (
        "regret_psnr_db",
        "box_regret_psnr_db",
        "taper_minus_box_psnr_db",
        "boundary_band_regret_psnr_db",
        "interior_regret_psnr_db",
        "boundary_action_disagreement_fraction",
        "interior_action_disagreement_fraction",
        "transferred_minus_uniform_psnr_db",
        "aligned_minus_shifted_psnr_db",
        "nonclipped_source_minus_uniform_psnr_db",
        "transferred_minus_uniform_ssim",
        "transferred_minus_uniform_color_bias",
        "mean_top_two_mse_margin",
        "exact_near_tie_pixel_fraction",
        "nonkeep_pixel_fraction",
        "selected_clip_pixel_fraction",
        "common_nonclipped_pixel_fraction",
        "common_source_projection_coverage_fraction",
        "hard_action_agreement_fraction",
    )
    grid_records: list[dict[str, Any]] = []
    for held_out_index, grid in enumerate(grids):
        target_records = [
            conditional_grid_measurement(
                torch, prepared, grids, variant_index, held_out_index, device,
            )
            for variant_index in range(VARIANTS_PER_SCENE)
        ]
        grid_records.append({
            "held_out_offset": grid["offset"],
            "tile_rows": len(grid["tiles"]),
            "tile_columns": len(grid["tiles"][0]),
            "condition_count": len(target_records),
            **{
                key: float(np.mean([record[key] for record in target_records]))
                for key in mean_keys
            },
            "source_grid_projection_complete": all(
                bool(record["source_grid_projection_complete"])
                for record in target_records
            ),
            "box_source_grid_projection_complete": all(
                bool(record["box_source_grid_projection_complete"])
                for record in target_records
            ),
            "common_source_projection_complete": all(
                bool(record["common_source_projection_complete"])
                for record in target_records
            ),
            "keep_structural_identity": all(
                bool(record["keep_structural_identity"])
                for record in target_records
            ),
            "affine_manipulation_exact": all(
                bool(record["affine_manipulation_exact"])
                for record in target_records
            ),
            "maximum_absolute_keep_duplicate_utility": max(
                abs(float(record["keep_duplicate_utility"]))
                for record in target_records
            ),
        })
    return {
        "regret_psnr_db": max(record["regret_psnr_db"] for record in grid_records),
        "box_regret_psnr_db": max(
            record["box_regret_psnr_db"] for record in grid_records
        ),
        "taper_minus_box_psnr_db": min(
            record["taper_minus_box_psnr_db"] for record in grid_records
        ),
        "boundary_band_regret_psnr_db": max(
            record["boundary_band_regret_psnr_db"] for record in grid_records
        ),
        "interior_regret_psnr_db": max(
            record["interior_regret_psnr_db"] for record in grid_records
        ),
        "boundary_action_disagreement_fraction": float(np.mean([
            record["boundary_action_disagreement_fraction"] for record in grid_records
        ])),
        "interior_action_disagreement_fraction": float(np.mean([
            record["interior_action_disagreement_fraction"] for record in grid_records
        ])),
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
        "common_source_projection_coverage_fraction": float(np.mean([
            record["common_source_projection_coverage_fraction"]
            for record in grid_records
        ])),
        "hard_action_agreement_fraction": float(np.mean([
            record["hard_action_agreement_fraction"] for record in grid_records
        ])),
        "source_grid_projection_complete": all(
            record["source_grid_projection_complete"] for record in grid_records
        ),
        "box_source_grid_projection_complete": all(
            record["box_source_grid_projection_complete"] for record in grid_records
        ),
        "common_source_projection_complete": all(
            record["common_source_projection_complete"] for record in grid_records
        ),
        "keep_structural_identity": all(
            record["keep_structural_identity"] for record in grid_records
        ),
        "affine_manipulation_exact": all(
            record["affine_manipulation_exact"] for record in grid_records
        ),
        "maximum_absolute_keep_duplicate_utility": max(
            record["maximum_absolute_keep_duplicate_utility"]
            for record in grid_records
        ),
        "grid_records": grid_records,
    }


def content_aligned_partition(
    hazy: np.ndarray, phase: tuple[int, int],
) -> np.ndarray:
    """Build a deterministic edge-watershed partition from hazy RGB only."""
    if hazy.ndim != 3 or hazy.shape[2] != 3 or phase not in REGION_PHASES:
        raise ValueError("invalid content-aligned partition input")
    height, width = hazy.shape[:2]
    row_start = (REGION_SPACING // 2 + phase[0]) % REGION_SPACING
    column_start = (REGION_SPACING // 2 + phase[1]) % REGION_SPACING
    rows = np.arange(row_start, height, REGION_SPACING, dtype=np.int64)
    columns = np.arange(column_start, width, REGION_SPACING, dtype=np.int64)
    if rows.size == 0 or columns.size == 0:
        raise RuntimeError("content-aligned marker lattice is empty")
    markers = np.zeros((height, width), dtype=np.int32)
    marker_ids = np.arange(1, rows.size * columns.size + 1, dtype=np.int32).reshape(
        rows.size, columns.size,
    )
    markers[np.ix_(rows, columns)] = marker_ids
    elevation = sobel(rgb2gray(hazy).astype(np.float64, copy=False))
    labels = watershed(
        elevation, markers=markers, connectivity=1, watershed_line=False,
    ).astype(np.int32, copy=False) - 1
    unique = np.unique(labels)
    if (
        labels.shape != (height, width)
        or int(np.min(labels)) != 0
        or not np.array_equal(unique, np.arange(unique.size, dtype=np.int32))
        or unique.size != marker_ids.size
    ):
        raise RuntimeError("content-aligned partition is incomplete or non-contiguous")
    return labels


def region_utility_table(
    variant: dict[str, Any], labels: np.ndarray, *, common_nonclipped: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if labels.shape != variant["clear"].shape[:2] or int(np.min(labels)) != 0:
        raise RuntimeError("region labels do not cover the variant")
    region_count = int(labels.max()) + 1
    flat_labels = labels.ravel()
    support = (
        variant["common_nonclipped"] if common_nonclipped
        else np.ones(labels.shape, dtype=bool)
    )
    counts = np.bincount(
        flat_labels, weights=support.ravel().astype(np.float64), minlength=region_count,
    )
    valid = counts > 0.0
    keep_error = np.mean(
        (variant["candidates"][0].astype(np.float64) - variant["clear"].astype(np.float64)) ** 2,
        axis=2,
    )
    utilities = np.zeros((region_count, len(ACTION_SET)), dtype=np.float64)
    for action_index, candidate in enumerate(variant["candidates"]):
        action_error = np.mean(
            (candidate.astype(np.float64) - variant["clear"].astype(np.float64)) ** 2,
            axis=2,
        )
        pixel_utility = (keep_error - action_error) * support
        sums = np.bincount(
            flat_labels, weights=pixel_utility.ravel(), minlength=region_count,
        )
        utilities[valid, action_index] = sums[valid] / counts[valid]
    if not np.isfinite(utilities[valid]).all():
        raise RuntimeError("non-finite content-aligned region utility")
    return utilities, valid, counts


def select_region_actions(
    scores: np.ndarray, valid: np.ndarray | None = None,
) -> np.ndarray:
    if scores.ndim != 2 or scores.shape[1] != len(ACTION_SET):
        raise RuntimeError("region score table has the wrong shape")
    selected = np.argmax(scores, axis=1).astype(np.int16)
    if valid is not None:
        selected = np.where(valid, selected, 0).astype(np.int16)
    return selected


def compose_region_action_field(
    variant: dict[str, Any], labels: np.ndarray, selected: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if selected.shape != (int(labels.max()) + 1,):
        raise RuntimeError("region action table does not match labels")
    dense_actions = selected[labels]
    return output_from_dense_actions(variant, dense_actions), dense_actions


def region_margin_summary(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    ordered = np.sort(scores, axis=1)
    gaps = ordered[:, -1] - ordered[:, -2]
    counts = np.bincount(labels.ravel(), minlength=scores.shape[0]).astype(np.float64)
    selected = select_region_actions(scores)
    total = float(np.sum(counts))
    return {
        "mean_top_two_mse_margin": float(np.sum(gaps * counts) / total),
        "exact_near_tie_pixel_fraction": float(
            np.sum(counts[gaps <= NUMERICAL_TIE_TOLERANCE]) / total
        ),
        "nonkeep_pixel_fraction": float(np.sum(counts[selected != 0]) / total),
    }


def project_region_scores(
    partitions: list[dict[str, Any]], held_out_index: int,
    table_key: str, valid_key: str, pixel_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, bool, float]:
    target_labels = partitions[held_out_index]["labels"]
    height, width = target_labels.shape
    dense_sum = np.zeros((height, width, len(ACTION_SET)), dtype=np.float64)
    dense_contributors = np.zeros((height, width), dtype=np.int16)
    source_count = len(partitions) - 1
    declared_support = (
        np.ones((height, width), dtype=bool) if pixel_mask is None else pixel_mask
    )
    for partition_index, partition in enumerate(partitions):
        if partition_index == held_out_index:
            continue
        labels = partition["labels"]
        scores = partition[table_key]
        valid = partition[valid_key]
        support = declared_support & valid[labels]
        dense_sum[support] += scores[labels[support]]
        dense_contributors[support] += 1
    complete = bool(
        np.all(dense_contributors[declared_support] == source_count)
        and np.all(dense_contributors[~declared_support] == 0)
    )
    dense_scores = np.zeros_like(dense_sum)
    covered = dense_contributors > 0
    dense_scores[covered] = dense_sum[covered] / dense_contributors[covered, None]
    region_count = int(target_labels.max()) + 1
    counts = np.bincount(
        target_labels.ravel(), weights=covered.ravel().astype(np.float64),
        minlength=region_count,
    )
    target_valid = counts > 0.0
    target_scores = np.zeros((region_count, len(ACTION_SET)), dtype=np.float64)
    for action_index in range(len(ACTION_SET)):
        sums = np.bincount(
            target_labels.ravel(),
            weights=dense_scores[..., action_index].ravel(),
            minlength=region_count,
        )
        target_scores[target_valid, action_index] = sums[target_valid] / counts[target_valid]
    return target_scores, target_valid, complete, float(np.mean(covered))


def region_boundary_mask(labels: np.ndarray, band: int) -> np.ndarray:
    boundary = find_boundaries(labels, connectivity=1, mode="thick")
    footprint = np.ones((2 * band + 1, 2 * band + 1), dtype=bool)
    return binary_dilation(boundary, footprint=footprint)


def region_projection_reference_check() -> bool:
    height, width = 12, 16
    yy, xx = np.mgrid[0:height, 0:width]
    partitions: list[dict[str, Any]] = []
    for phase_index, (row_shift, column_shift) in enumerate(REGION_PHASES):
        raw = ((yy + row_shift // 4) // 4) * 8 + ((xx + column_shift // 4) // 4)
        _, labels = np.unique(raw, return_inverse=True)
        labels = labels.reshape(height, width).astype(np.int32)
        region_count = int(labels.max()) + 1
        scores = np.stack(
            (
                np.zeros(region_count, dtype=np.float64),
                0.01 * (phase_index + 1) + 0.001 * np.arange(region_count),
                0.02 * (phase_index + 1) - 0.0005 * np.arange(region_count),
            ),
            axis=1,
        )
        partitions.append({
            "labels": labels,
            "utility": scores,
            "valid": np.ones(region_count, dtype=bool),
        })
    for held_out_index, target in enumerate(partitions):
        actual, valid, complete, coverage = project_region_scores(
            partitions, held_out_index, "utility", "valid",
        )
        dense = np.mean(
            [
                source["utility"][source["labels"]]
                for index, source in enumerate(partitions)
                if index != held_out_index
            ],
            axis=0,
        )
        expected = np.zeros_like(actual)
        for region_index in range(expected.shape[0]):
            expected[region_index] = np.mean(
                dense[target["labels"] == region_index], axis=0,
            )
        if (
            not complete or coverage != 1.0 or not bool(np.all(valid))
            or not np.allclose(actual, expected, rtol=0.0, atol=KEEP_DIAGNOSTIC_ERROR_BOUND)
        ):
            return False
    return True


def conditional_region_measurement(
    torch, target: dict[str, Any], partitions: list[dict[str, Any]],
    held_out_index: int, device: str,
) -> dict[str, float | bool | int]:
    target_partition = partitions[held_out_index]
    labels = target_partition["labels"]
    source_scores, source_valid, projection_complete, _ = project_region_scores(
        partitions, held_out_index, "utility", "valid",
    )
    common_scores, common_valid, common_complete, common_coverage = project_region_scores(
        partitions, held_out_index, "common_utility", "common_valid",
        target["common_nonclipped"],
    )
    transferred_actions = select_region_actions(source_scores, source_valid)
    nonclipped_actions = select_region_actions(common_scores, common_valid)
    oracle_actions = select_region_actions(target_partition["utility"])
    transferred, dense_actions = compose_region_action_field(
        target, labels, transferred_actions,
    )
    nonclipped, _ = compose_region_action_field(target, labels, nonclipped_actions)
    oracle, oracle_dense_actions = compose_region_action_field(target, labels, oracle_actions)
    shifted = output_from_dense_actions(
        target, np.roll(dense_actions, shift=(REGION_SPACING, REGION_SPACING), axis=(0, 1)),
    )
    uniform = target["uniform_output"]
    oracle_psnr = psnr_from_mse(mse(oracle, target["clear"]))
    transferred_psnr = psnr_from_mse(mse(transferred, target["clear"]))
    uniform_psnr = psnr_from_mse(mse(uniform, target["clear"]))
    shifted_psnr = psnr_from_mse(mse(shifted, target["clear"]))
    nonclipped_psnr = psnr_from_mse(mse(
        nonclipped, target["clear"], target["common_nonclipped"],
    ))
    uniform_nonclipped_psnr = psnr_from_mse(mse(
        target["common_uniform_output"], target["clear"], target["common_nonclipped"],
    ))
    regret = oracle_psnr - transferred_psnr
    if regret < -1e-8:
        raise RuntimeError("held-out region oracle was worse than the transferred field")
    uniform_ssim, transferred_ssim = rgb_ssim(
        torch, [uniform, transferred], target["clear"], device,
    )
    margin = region_margin_summary(source_scores, labels)
    selected_clip = np.zeros(labels.shape, dtype=bool)
    for action_index, clip_mask in enumerate(target["clip_masks"]):
        selected_clip |= (dense_actions == action_index) & clip_mask
    boundary = region_boundary_mask(labels, BOUNDARY_DIAGNOSTIC_BAND_PIXELS)
    interior = ~boundary
    if not bool(np.any(boundary)) or not bool(np.any(interior)):
        raise RuntimeError("same-phase region boundary diagnostics are empty")
    boundary_regret = psnr_from_mse(mse(oracle, target["clear"], boundary)) - (
        psnr_from_mse(mse(transferred, target["clear"], boundary))
    )
    interior_regret = psnr_from_mse(mse(oracle, target["clear"], interior)) - (
        psnr_from_mse(mse(transferred, target["clear"], interior))
    )
    disagreement = dense_actions != oracle_dense_actions
    region_counts = target_partition["counts"]
    return {
        "regret_psnr_db": max(0.0, regret),
        "boundary_band_regret_psnr_db": boundary_regret,
        "interior_regret_psnr_db": interior_regret,
        "boundary_minus_interior_regret_psnr_db": boundary_regret - interior_regret,
        "boundary_action_disagreement_fraction": float(np.mean(disagreement[boundary])),
        "interior_action_disagreement_fraction": float(np.mean(disagreement[interior])),
        "transferred_minus_uniform_psnr_db": transferred_psnr - uniform_psnr,
        "aligned_minus_shifted_psnr_db": transferred_psnr - shifted_psnr,
        "nonclipped_source_minus_uniform_psnr_db": nonclipped_psnr - uniform_nonclipped_psnr,
        "transferred_minus_uniform_ssim": transferred_ssim - uniform_ssim,
        "transferred_minus_uniform_color_bias": color_bias(
            transferred, target["clear"],
        ) - color_bias(uniform, target["clear"]),
        "mean_top_two_mse_margin": margin["mean_top_two_mse_margin"],
        "exact_near_tie_pixel_fraction": margin["exact_near_tie_pixel_fraction"],
        "nonkeep_pixel_fraction": margin["nonkeep_pixel_fraction"],
        "selected_clip_pixel_fraction": float(np.mean(selected_clip)),
        "common_nonclipped_pixel_fraction": target["common_nonclipped_fraction"],
        "common_source_projection_coverage_fraction": common_coverage,
        "hard_action_agreement_fraction": float(np.mean(dense_actions == oracle_dense_actions)),
        "source_region_projection_complete": projection_complete,
        "common_source_region_projection_complete": common_complete,
        "keep_structural_identity": bool(np.array_equal(
            target["candidates"][0], target["prediction"],
        )),
        "affine_manipulation_exact": bool(target["affine_manipulation_exact"]),
        "keep_duplicate_utility": mse(target["prediction"], target["clear"]) - mse(
            apply_scale(target["hazy"], target["prediction"], 1.0), target["clear"],
        ),
        "region_count": int(region_counts.size),
        "minimum_region_area": int(np.min(region_counts)),
        "maximum_region_area": int(np.max(region_counts)),
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
    variant_partitions: list[list[dict[str, Any]]] = []
    for variant in prepared:
        partitions = []
        for phase in REGION_PHASES:
            labels = content_aligned_partition(variant["hazy"], phase)
            utility, valid, counts = region_utility_table(
                variant, labels, common_nonclipped=False,
            )
            common_utility, common_valid, _ = region_utility_table(
                variant, labels, common_nonclipped=True,
            )
            partitions.append({
                "phase": list(phase),
                "labels": labels,
                "utility": utility,
                "valid": valid,
                "common_utility": common_utility,
                "common_valid": common_valid,
                "counts": counts.astype(np.int64),
            })
        variant_partitions.append(partitions)
    mean_keys = (
        "regret_psnr_db", "boundary_band_regret_psnr_db",
        "interior_regret_psnr_db", "boundary_minus_interior_regret_psnr_db",
        "boundary_action_disagreement_fraction", "interior_action_disagreement_fraction",
        "transferred_minus_uniform_psnr_db", "aligned_minus_shifted_psnr_db",
        "nonclipped_source_minus_uniform_psnr_db", "transferred_minus_uniform_ssim",
        "transferred_minus_uniform_color_bias", "mean_top_two_mse_margin",
        "exact_near_tie_pixel_fraction", "nonkeep_pixel_fraction",
        "selected_clip_pixel_fraction", "common_nonclipped_pixel_fraction",
        "common_source_projection_coverage_fraction", "hard_action_agreement_fraction",
        "region_count", "minimum_region_area", "maximum_region_area",
    )
    phase_records: list[dict[str, Any]] = []
    for held_out_index, phase in enumerate(REGION_PHASES):
        condition_records = [
            conditional_region_measurement(
                torch, prepared[index], variant_partitions[index], held_out_index, device,
            )
            for index in range(VARIANTS_PER_SCENE)
        ]
        phase_records.append({
            "held_out_phase": list(phase),
            "condition_count": len(condition_records),
            **{
                key: float(np.mean([record[key] for record in condition_records]))
                for key in mean_keys
            },
            "source_region_projection_complete": all(
                bool(record["source_region_projection_complete"])
                for record in condition_records
            ),
            "common_source_region_projection_complete": all(
                bool(record["common_source_region_projection_complete"])
                for record in condition_records
            ),
            "keep_structural_identity": all(
                bool(record["keep_structural_identity"]) for record in condition_records
            ),
            "affine_manipulation_exact": all(
                bool(record["affine_manipulation_exact"]) for record in condition_records
            ),
            "maximum_absolute_keep_duplicate_utility": max(
                abs(float(record["keep_duplicate_utility"]))
                for record in condition_records
            ),
        })
    worst_index = int(np.argmax([
        record["regret_psnr_db"] for record in phase_records
    ]))
    worst = phase_records[worst_index]
    taper_control = taper_control_scene_measurement(torch, variants, device)
    return {
        "regret_psnr_db": worst["regret_psnr_db"],
        "taper_control_regret_psnr_db": taper_control["regret_psnr_db"],
        "taper_control_minus_content_aligned_regret_psnr_db": (
            taper_control["regret_psnr_db"] - worst["regret_psnr_db"]
        ),
        "boundary_band_regret_psnr_db": worst["boundary_band_regret_psnr_db"],
        "interior_regret_psnr_db": worst["interior_regret_psnr_db"],
        "boundary_minus_interior_regret_psnr_db": worst[
            "boundary_minus_interior_regret_psnr_db"
        ],
        "boundary_action_disagreement_fraction": worst[
            "boundary_action_disagreement_fraction"
        ],
        "interior_action_disagreement_fraction": worst[
            "interior_action_disagreement_fraction"
        ],
        "robust_gain_psnr_db": min(
            record["transferred_minus_uniform_psnr_db"] for record in phase_records
        ),
        "aligned_minus_shifted_psnr_db": min(
            record["aligned_minus_shifted_psnr_db"] for record in phase_records
        ),
        "nonclipped_source_minus_uniform_psnr_db": min(
            record["nonclipped_source_minus_uniform_psnr_db"] for record in phase_records
        ),
        "transferred_minus_uniform_ssim": min(
            record["transferred_minus_uniform_ssim"] for record in phase_records
        ),
        "transferred_minus_uniform_color_bias": max(
            record["transferred_minus_uniform_color_bias"] for record in phase_records
        ),
        **{
            key: float(np.mean([record[key] for record in phase_records]))
            for key in (
                "mean_top_two_mse_margin", "exact_near_tie_pixel_fraction",
                "nonkeep_pixel_fraction", "selected_clip_pixel_fraction",
                "common_nonclipped_pixel_fraction",
                "common_source_projection_coverage_fraction",
                "hard_action_agreement_fraction", "region_count",
                "minimum_region_area", "maximum_region_area",
            )
        },
        "source_region_projection_complete": all(
            record["source_region_projection_complete"] for record in phase_records
        ),
        "common_source_region_projection_complete": all(
            record["common_source_region_projection_complete"] for record in phase_records
        ),
        "keep_structural_identity": all(
            record["keep_structural_identity"] for record in phase_records
        ),
        "affine_manipulation_exact": all(
            record["affine_manipulation_exact"] for record in phase_records
        ),
        "maximum_absolute_keep_duplicate_utility": max(
            record["maximum_absolute_keep_duplicate_utility"] for record in phase_records
        ),
        "partition_records": phase_records,
        "taper_control": taper_control,
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
    taper_control_regret = scene_bootstrap(
        (scene["taper_control_regret_psnr_db"] for scene in scenes),
        BOOTSTRAP_SEEDS["regret"],
    ) if scenes else None
    content_aligned_minus_taper = scene_bootstrap(
        (scene["taper_control_minus_content_aligned_regret_psnr_db"] for scene in scenes),
        BOOTSTRAP_SEEDS["regret"],
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
            "decision": "HAZE4K_TEST_CONDITIONAL_CONTENT_ALIGNED_REGION_STABILITY_INCONCLUSIVE",
            "authorizes": "STAGE1_REASSESSMENT_ONLY",
            "gate_reasons": ["complete 100-scene conditional measurement integrity was not established"],
        }
    elif regret_fail or coverage_fail or negative_control_fail or nonclipped_fail or safety_fail:
        terminal = {
            "state": "COMPLETED_GATE_FAIL",
            "decision": "HAZE4K_TEST_CONDITIONAL_CONTENT_ALIGNED_REGION_STABILITY_FAIL",
            "authorizes": "NONE",
            "gate_reasons": ["at least one frozen conditional measurement-validity interval clearly failed"],
        }
    elif precision_pass and coverage_pass and negative_control_pass and nonclipped_pass and safety_pass:
        terminal = {
            "state": "COMPLETED_GATE_PASS",
            "decision": "HAZE4K_TEST_CONDITIONAL_CONTENT_ALIGNED_REGION_STABILITY_SCREEN_PASS",
            "authorizes": "STAGE1_REASSESSMENT_ONLY",
            "gate_reasons": ["the frozen regret, precision, coverage, negative-control, clipping, and safety gates all passed"],
        }
    else:
        terminal = {
            "state": "COMPLETED_INCONCLUSIVE",
            "decision": "HAZE4K_TEST_CONDITIONAL_CONTENT_ALIGNED_REGION_STABILITY_INCONCLUSIVE",
            "authorizes": "STAGE1_REASSESSMENT_ONLY",
            "gate_reasons": ["at least one frozen interval crossed its margin or the achieved regret precision exceeded 0.05 dB"],
        }
    return {
        "complete": complete,
        "regret": regret,
        "taper_control_regret": taper_control_regret,
        "content_aligned_minus_taper": content_aligned_minus_taper,
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
        "scope": "Haze-conditional source-only edge-watershed region qualification on the isolated 100-scene Haze4K development partition",
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
            "haze_condition_role": "each of four haze variants is analyzed as a substantive observed condition; no utility transfers across variants",
            "leave_one_partition_out": "within each haze condition, construct four edge-watershed partitions from hazy RGB only, lift region-mean RGB-MSE utility from three source partitions to pixels, average source scores, freeze the hard action field, and score only then on the held-out partition",
            "matched_control": "the archived raised-cosine taper projection is recomputed from the same official inferences and must reproduce its parent mean regret within 1e-8 dB",
            "region_algorithm": "skimage edge-watershed with Sobel grayscale elevation, fixed 32-pixel marker spacing, connectivity 1, no watershed line, and no GT or outcome input",
            "partition_phases": [list(phase) for phase in REGION_PHASES],
            "primary_scene_value": "maximum held-out-partition regret after equal averaging of the four separately conditioned haze variants",
            "independent_unit": "canonical_clear_scene",
            "nested_repeats": "four observed haze conditions, regions, pixels, partition phases, and metrics never increase n",
        },
        "primary_regret": {
            "scene_mean_regret_psnr_db": evaluated["regret"],
            "maximum_upper_bound_db": REGRET_MARGIN_DB,
            "maximum_upper_distance_db": PRECISION_DISTANCE_DB,
            "pre_run_precision_mode": "formal normal-mean feasibility: directly matched taper-regret scene SD 0.0985213295 dB requires 15 independent scenes for a two-sided 0.05 dB half-width; n=100 is feasible",
        },
        "matched_taper_control": {
            "scene_mean_taper_control_regret_psnr_db": evaluated["taper_control_regret"],
            "scene_mean_taper_control_minus_content_aligned_regret_psnr_db": evaluated[
                "content_aligned_minus_taper"
            ],
            "expected_parent_taper_regret_db": TAPER_CONTROL_EXPECTED_REGRET_DB,
            "absolute_reproduction_tolerance_db": TAPER_CONTROL_TOLERANCE_DB,
            "reproduction_pass": integrity.get("matched_taper_parent_control", False),
            "decision_role": "integrity for taper reproduction; content-aligned minus taper contrast is diagnostic only",
        },
        "coverage": {
            "material_scene_gain_db": MATERIAL_GAIN_DB,
            "robust_material_scene_prevalence": evaluated["benefit"],
            "minimum_prevalence_lower_bound": MIN_MATERIAL_SCENE_PREVALENCE,
        },
        "controls": {
            "aligned_minus_circularly_shifted_psnr_db": evaluated["aligned_shift"],
            "nonclipped_source_minus_uniform_psnr_db": evaluated["nonclipped_gain"],
            "keep_structural_identity_all": all(
                scene["keep_structural_identity"] for scene in scenes
            ) if scenes else False,
            "source_region_projection_complete_all": all(
                scene["source_region_projection_complete"] for scene in scenes
            ) if scenes else False,
            "common_source_region_projection_complete_all": all(
                scene["common_source_region_projection_complete"] for scene in scenes
            ) if scenes else False,
            "matched_taper_parent_control_mean_regret_db": (
                float(np.mean([scene["taper_control_regret_psnr_db"] for scene in scenes]))
                if scenes else None
            ),
            "matched_taper_parent_control_expected_regret_db": TAPER_CONTROL_EXPECTED_REGRET_DB,
            "matched_taper_parent_control_tolerance_db": TAPER_CONTROL_TOLERANCE_DB,
            "nonclipped_affine_manipulation_exact_all": all(
                scene["affine_manipulation_exact"] for scene in scenes
            ) if scenes else False,
            "maximum_absolute_keep_duplicate_utility": max(
                (scene["maximum_absolute_keep_duplicate_utility"] for scene in scenes),
                default=None,
            ),
            "keep_duplicate_utility_diagnostic_error_bound": KEEP_DIAGNOSTIC_ERROR_BOUND,
            "keep_duplicate_utility_within_diagnostic_bound": all(
                scene["maximum_absolute_keep_duplicate_utility"]
                <= KEEP_DIAGNOSTIC_ERROR_BOUND for scene in scenes
            ) if scenes else False,
        },
        "safety": {
            "ssim_harm_definition": f"worst-partition scene mean transferred-minus-uniform SSIM at most -{SSIM_HARM_MARGIN}",
            "ssim_harm_prevalence": evaluated["ssim_harm"],
            "color_harm_definition": f"worst-partition scene mean transferred-minus-uniform RGB mean-bias at least {COLOR_HARM_MARGIN}",
            "color_harm_prevalence": evaluated["color_harm"],
            "maximum_harm_upper_bound": MAX_HARM_PREVALENCE,
        },
        "secondary_scene_aggregates": {
            key: aggregate(scene[key] for scene in scenes)
            for key in (
                "regret_psnr_db",
                "taper_control_regret_psnr_db",
                "taper_control_minus_content_aligned_regret_psnr_db",
                "boundary_band_regret_psnr_db",
                "interior_regret_psnr_db",
                "boundary_minus_interior_regret_psnr_db",
                "boundary_action_disagreement_fraction",
                "interior_action_disagreement_fraction",
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
                "common_source_projection_coverage_fraction",
                "hard_action_agreement_fraction",
                "region_count",
                "minimum_region_area",
                "maximum_region_area",
            )
        },
        "per_partition_diagnostics": {
            str(tuple(REGION_PHASES[origin_index])): {
                key: aggregate(
                    scene["partition_records"][origin_index][key] for scene in scenes
                )
                for key in (
                    "regret_psnr_db",
                    "boundary_band_regret_psnr_db", "interior_regret_psnr_db",
                    "boundary_minus_interior_regret_psnr_db",
                    "boundary_action_disagreement_fraction",
                    "interior_action_disagreement_fraction",
                    "selected_clip_pixel_fraction", "region_count",
                )
            }
            for origin_index in range(len(REGION_PHASES))
        },
        "gates": evaluated["gates"],
        "terminal": terminal,
        "limitations": [
            "This is development-screening measurement evidence from 100 canonical synthetic Haze4K scenes.",
            "The directly matched taper-regret scene SD supports the frozen 0.05 dB precision target at n=100, but the result remains development-screening evidence.",
            "GT selects source-region utilities and held-out-region oracle outcomes within the same haze condition; no result establishes deployment-time predictability.",
            "A pass qualifies only this fixed source-only edge-watershed measurement target conditional on observed haze and cannot overturn archived cross-variant instability.",
            "RGB reconstruction utility and limited SSIM/color safety do not establish perceptual quality or physical haze state.",
            "The result is specific to the fixed three residual scales, Sobel edge cost, 32-pixel marker spacing, and four declared marker phases.",
            "The protected candidate-confirmation asset and NH-HAZE were not delivered or accessed.",
        ],
        "marker": "HAZE4K_TEST_CONDITIONAL_CONTENT_ALIGNED_REGION_MEASUREMENT_QUALIFICATION_COMPLETE",
    }
    summary_name = "haze4k_test_conditional_content_aligned_region_measurement_qualification_v1_summary.json"
    strata_name = "haze4k_test_conditional_content_aligned_region_measurement_qualification_v1_strata.csv"
    atomic_json(output_file(context, summary_name), summary)
    rows = [
        {
            "estimand": "scene_mean_worst_held_out_partition_conditional_regret_psnr_db",
            "method": "scene_bootstrap",
            "scenes": EXPECTED_SCENES,
            "events": "",
            "estimate": evaluated["regret"]["estimate"] if evaluated["regret"] else "",
            "lower_one_sided_95": evaluated["regret"]["lower_one_sided_95"] if evaluated["regret"] else "",
            "upper_one_sided_95": evaluated["regret"]["upper_one_sided_95"] if evaluated["regret"] else "",
            "threshold": REGRET_MARGIN_DB,
        },
        {
            "estimand": "scene_mean_worst_held_out_partition_taper_control_regret_psnr_db",
            "method": "scene_bootstrap",
            "scenes": EXPECTED_SCENES,
            "events": "",
            "estimate": evaluated["taper_control_regret"]["estimate"] if evaluated["taper_control_regret"] else "",
            "lower_one_sided_95": evaluated["taper_control_regret"]["lower_one_sided_95"] if evaluated["taper_control_regret"] else "",
            "upper_one_sided_95": evaluated["taper_control_regret"]["upper_one_sided_95"] if evaluated["taper_control_regret"] else "",
            "threshold": TAPER_CONTROL_EXPECTED_REGRET_DB,
        },
        {
            "estimand": "scene_mean_taper_control_minus_content_aligned_regret_psnr_db",
            "method": "scene_bootstrap",
            "scenes": EXPECTED_SCENES,
            "events": "",
            "estimate": evaluated["content_aligned_minus_taper"]["estimate"] if evaluated["content_aligned_minus_taper"] else "",
            "lower_one_sided_95": evaluated["content_aligned_minus_taper"]["lower_one_sided_95"] if evaluated["content_aligned_minus_taper"] else "",
            "upper_one_sided_95": evaluated["content_aligned_minus_taper"]["upper_one_sided_95"] if evaluated["content_aligned_minus_taper"] else "",
            "threshold": "diagnostic_only",
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
    started_at = time.monotonic()
    torch, model = load_official_model(context)
    height, width = 1200, 1600
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    clear = np.stack(
        (
            0.12 + 0.62 * xx / width,
            0.18 + 0.56 * yy / height,
            0.16 + 0.58 * (xx + 2.0 * yy) / (width + 2.0 * height),
        ),
        axis=-1,
    ).astype(np.float32)
    variants = []
    predictions = []
    for transmission in (0.62, 0.68, 0.74, 0.80):
        hazy = (transmission * clear + (1.0 - transmission) * 0.72).astype(np.float32)
        prediction = infer(torch, model, hazy, context.device)
        predictions.append(prediction)
        variants.append({"hazy": hazy, "clear": clear, "prediction": prediction})
    synthetic_scene = scene_measurement(torch, variants, context.device)
    first_partition = content_aligned_partition(variants[0]["hazy"], REGION_PHASES[0])
    partition_deterministic = np.array_equal(
        first_partition,
        content_aligned_partition(variants[0]["hazy"], REGION_PHASES[0]),
    )
    repeated = [synthetic_scene for _ in range(EXPECTED_SCENES)]
    exercised = evaluate_gates(
        {
            "parent_taper_terminal_evidence": True,
            "isolated_development_asset_only": True,
            "complete_100_scene_400_variant_grouping": True,
            "complete_finite_inference_and_measurement": True,
            "candidate_confirmation_asset_not_delivered": True,
            "no_training": True,
            "measurement_controls_structural": True,
            "matched_taper_parent_control": True,
        },
        repeated,
        [],
    )
    checks = {
        "strict_checkpoint_load_and_parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ) == PARAMETER_COUNT,
        "all_parameters_frozen": not any(parameter.requires_grad for parameter in model.parameters()),
        "three_scale_finite_forward": all(np.isfinite(value).all() for value in predictions),
        "four_content_aligned_partitions_complete": len(
            synthetic_scene["partition_records"]
        ) == len(REGION_PHASES),
        "partition_deterministic": partition_deterministic,
        "keep_structural_identity": synthetic_scene["keep_structural_identity"],
        "source_region_projection_complete": synthetic_scene[
            "source_region_projection_complete"
        ],
        "common_source_region_projection_complete": synthetic_scene[
            "common_source_region_projection_complete"
        ],
        "region_projection_independent_reference": region_projection_reference_check(),
        "matched_taper_projection_reference": projection_reference_check(),
        "matched_taper_path_complete": len(
            synthetic_scene["taper_control"]["grid_records"]
        ) == len(GRID_OFFSETS),
        "nonclipped_affine_manipulation_exact": synthetic_scene["affine_manipulation_exact"],
        "keep_duplicate_diagnostic_finite": math.isfinite(
            synthetic_scene["maximum_absolute_keep_duplicate_utility"],
        ),
        "held_out_regret_nonnegative": synthetic_scene["regret_psnr_db"] >= 0.0,
        "negative_control_finite": math.isfinite(
            synthetic_scene["aligned_minus_shifted_psnr_db"],
        ),
        "bootstrap_and_terminal_finite": exercised["regret"] is not None
        and exercised["terminal"]["state"] in {
            "COMPLETED_GATE_PASS", "COMPLETED_GATE_FAIL", "COMPLETED_INCONCLUSIVE",
        },
        "protected_roles_disabled": not any(context.protected_data_permissions.values()),
        "full_scale_fixture_within_frozen_bound": (
            time.monotonic() - started_at <= CONTRACT_MAX_SECONDS
        ),
    }
    write_contract_result(
        context,
        checks=checks,
        engineering={
            "mode": "gpu_synthetic_no_data",
            "device": "cuda",
            "fixture": {"batch": 1, "channels": 3, "height": height, "width": width},
            "full_scale_fixture_max_seconds": CONTRACT_MAX_SECONDS,
            "full_scale_fixture_elapsed_seconds": time.monotonic() - started_at,
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
    parent_closeout_path = asset_path(
        context, "parent_taper_closeout", kind="file",
    )
    parent_summary_path = asset_path(
        context, "parent_taper_summary", kind="file",
    )
    parent_conclusion_path = asset_path(
        context, "parent_taper_conclusion", kind="file",
    )
    expected_assets = {
        "parent_taper_closeout": PARENT_TAPER_CLOSEOUT_SHA256,
        "parent_taper_summary": PARENT_TAPER_SUMMARY_SHA256,
        "parent_taper_conclusion": PARENT_TAPER_CONCLUSION_SHA256,
    }
    for identifier, identity in expected_assets.items():
        if context.assets[identifier].sha256 != identity:
            raise RuntimeError(f"formal evidence identity changed for {identifier}")
    parent_closeout = json.loads(parent_closeout_path.read_text(encoding="utf-8"))
    parent_summary = json.loads(parent_summary_path.read_text(encoding="utf-8"))
    parent_conclusion = json.loads(parent_conclusion_path.read_text(encoding="utf-8"))
    parent_taper_ok = (
        parent_closeout.get("state") == "COMPLETED_GATE_FAIL"
        and parent_closeout.get("decision")
        == "HAZE4K_TEST_CONDITIONAL_TAPER_GRID_STABILITY_FAIL"
        and parent_closeout.get("authorizes") == "NONE"
        and parent_summary.get("identity_and_coverage", {}).get("completed_scenes")
        == EXPECTED_SCENES
        and parent_summary.get("identity_and_coverage", {}).get("completed_variants")
        == EXPECTED_VARIANTS
        and parent_summary.get("identity_and_coverage", {}).get("assignment_digest")
        == SPLIT_ASSIGNMENT_DIGEST
        and abs(
            float(parent_summary["primary_regret"]["scene_mean_regret_psnr_db"]["estimate"])
            - TAPER_CONTROL_EXPECTED_REGRET_DB
        ) <= TAPER_CONTROL_TOLERANCE_DB
        and parent_conclusion.get("authorizes") == "NONE"
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
    if parent_taper_ok and scope_ok and dataset_ok:
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
        "parent_taper_terminal_evidence": parent_taper_ok,
        "isolated_development_asset_only": scope_ok,
        "complete_100_scene_400_variant_grouping": dataset_ok,
        "complete_finite_inference_and_measurement": len(scenes) == EXPECTED_SCENES and not failures,
        "candidate_confirmation_asset_not_delivered": True,
        "no_training": True,
        "measurement_controls_structural": all(
            scene["keep_structural_identity"]
            and scene["source_region_projection_complete"]
            and scene["common_source_region_projection_complete"]
            and scene["affine_manipulation_exact"]
            for scene in scenes
        ) if scenes else False,
        "matched_taper_parent_control": (
            len(scenes) == EXPECTED_SCENES
            and abs(
                float(np.mean([
                    scene["taper_control_regret_psnr_db"] for scene in scenes
                ]))
                - TAPER_CONTROL_EXPECTED_REGRET_DB
            ) <= TAPER_CONTROL_TOLERANCE_DB
        ),
    }
    result = finalize(context, integrity, scenes, failures)
    write_workload_progress(
        context, completed_units=403,
        stage="scene_level_conditional_content_aligned_region_measurement_finalize",
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
