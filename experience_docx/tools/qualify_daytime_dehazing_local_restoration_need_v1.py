#!/usr/bin/env python3
"""Qualify development-only bidirectional local restoration need."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

from route_program_api import (
    asset_path,
    atomic_json,
    load_completed_unit_ledger,
    load_context,
    output_file,
    prepare_phase_output,
    record_completed_unit,
    write_contract_progress,
    write_contract_result,
    write_gate_result,
    write_workload_progress,
)


ROUTE_ID = "daytime-dehazing-local-restoration-need-qualification-v1"
OPERATION_ID = "DAYTIME_DEHAZING_LOCAL_RESTORATION_NEED_QUALIFY"
SUMMARY_NAME = "daytime_dehazing_local_restoration_need_v1_summary.json"
SCENE_METRICS_NAME = "daytime_dehazing_local_restoration_need_v1_scene_metrics.csv"
GATE_NAME = "daytime_dehazing_local_restoration_need_v1_gate_summary.json"
REVIEW_FACTS_NAME = "daytime_dehazing_local_restoration_need_v1_review_facts.json"

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
DATASET_ORDER = ("HAZE4K_TRAIN", "ITS", "OTS")
DEVELOPMENT_COUNTS = {"HAZE4K_TRAIN": 150, "ITS": 869, "OTS": 801}
PLANNED_SCENES = {dataset: 150 for dataset in DATASET_ORDER}
VARIANTS_PER_SCENE = 2
GRID_SIDE = 4
MAX_CROP_SIDE = 512
MIN_CROP_SIDE = 256
ALPHA_GRID = np.asarray([0.5, 0.75, 1.0, 1.25, 1.5], dtype=np.float64)
KEEP_ALPHA = 1.0
ALPHA_TIE_DB = 0.005
LOCAL_DIRECTION_MARGIN_DB = 0.05
PRIMARY_MARGIN_DB = 0.05
PRIMARY_CLIP_DB = 0.25
MATERIAL_SCENE_PREVALENCE = 0.20
STABLE_CELL_COUNT = 12
BIDIRECTIONAL_CELL_COUNT = 2
KEEP_CELL_COUNT = 4
STABILITY_PREVALENCE = 0.50
BIDIRECTIONAL_PREVALENCE = 0.20
KEEP_PREVALENCE = 0.20
NEAR_CLEAR_PSNR_DB = 30.0
NEAR_CLEAR_DAMAGE_DB = 0.10
NEAR_CLEAR_MIN_SCENES = 42
NEAR_CLEAR_MITIGATION_PREVALENCE = 0.10
NULL_UCB_MARGIN_DB = 0.05
NULL_DATASET_UCB_MARGIN_DB = 0.10
PRECISION_HALF_WIDTH_DB = 0.055
BOOTSTRAP_RESAMPLES = 20_000
BOOTSTRAP_SEED = 20260731
SIMULTANEOUS_CRITICAL_VALUE = 2.58
SCENE_SELECTION_SALT = "daytime-dehazing-local-restoration-need-v1|scenes"
VARIANT_SELECTION_SALT = "daytime-dehazing-local-restoration-need-v1|variants"
CROP_SELECTION_SALT = "daytime-dehazing-local-restoration-need-v1|crop"
TOTAL_UNITS = 451
PROBE_ITERATIONS = 8
FORMAL_INFERENCE_ITERATIONS = 900
EPSILON = 1.0e-12

ANCHOR_COMMIT = "3b4da35440c8c26a7d1bcaf1daf342e11d9a3898"
CHECKPOINT_SHA256 = "6f42037d57a4e3de3a10ac0ab909d66a3415864a19433c29204a975f4efa4088"
MODEL_SOURCE_SHA256 = "9681defa95c6602d0e0abae05e635771bd65d48a18555198ff00d573242a0005"
MODEL_LAYERS_SHA256 = "ac8a05bd626d9adda16308dedb9466f36d7ff44cfb666f64e7e14ddf8cdf43a4"
RUNTIME_ENVIRONMENT_SHA256 = "ede9744c12b6f154092277bb8d7b6ad1d7ade5011de3e186df19afac2a3b0fde"
S0_CLOSEOUT_SHA256 = "9e132828fb98615241d5e8dea0b0fecffa542397f4ff71bf686a901ca8959346"
S0_ROLE_SUMMARY_SHA256 = "d2262c8ba28c56a21b992c8f2c445d92099d7c9861f4263171c522c2efd8e7b1"
S0_LEDGER_SHA256 = "4cff8e7aecea5d8e19165ac4e725f69746342521115a3ab1f03ea1474f280960"
PARAMETER_COUNT = 8_630_665


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def image_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def image_array(path: Path) -> np.ndarray:
    with Image.open(path) as source:
        value = np.asarray(source.convert("RGB"), dtype=np.float32) / 255.0
    if value.ndim != 3 or value.shape[2] != 3:
        raise RuntimeError(f"unsupported RGB image: {path.name}")
    return value


def canonical_rgb_digest(value: np.ndarray) -> str:
    height, width = value.shape[:2]
    payload = np.rint(value * 255.0).clip(0, 255).astype(np.uint8).tobytes()
    digest = hashlib.sha256()
    digest.update(width.to_bytes(8, "big"))
    digest.update(height.to_bytes(8, "big"))
    digest.update(payload)
    return digest.hexdigest()


def selected_label(image_name: str, label_dir: Path) -> Path | None:
    stem, extension = os.path.splitext(image_name)
    names = [image_name]
    if "_" in stem:
        prefix = stem.split("_", 1)[0]
        names.extend((f"{prefix}{extension}", f"{prefix}.png"))
    seen: set[Path] = set()
    for name in names:
        candidate = label_dir / name
        if candidate not in seen and candidate.is_file():
            return candidate
        seen.add(candidate)
    return None


def deterministic_rank(values: Iterable[str], salt: str) -> list[str]:
    return sorted(values, key=lambda value: (sha256_text(f"{salt}|{value}"), value))


def read_s0_ledger(path: Path) -> tuple[dict[str, set[str]], dict[str, Any]]:
    roles: dict[str, set[str]] = defaultdict(set)
    row_count = 0
    excluded_count = 0
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            row_count += 1
            if row.get("schema_version") != 1 \
                    or row.get("program_id") != "daytime_dehazing_spatially_adaptive_restoration_v1" \
                    or row.get("independent_unit") != "original_clear_scene":
                raise RuntimeError("S0 scene-role ledger schema or program identity changed")
            if row.get("exclusion_reason") is not None:
                excluded_count += 1
                continue
            if row.get("role") == "development_screening":
                roles[str(row.get("dataset"))].add(str(row.get("scene_id")))
    counts = {dataset: len(roles[dataset]) for dataset in DATASET_ORDER}
    return roles, {
        "row_count": row_count,
        "excluded_count": excluded_count,
        "development_counts": counts,
        "expected_counts_match": counts == DEVELOPMENT_COUNTS,
    }


def enumerate_haze4k(
    root: Path, allowed_scenes: set[str],
) -> dict[str, list[tuple[Path, Path]]]:
    input_dirs = [root / name for name in ("IN", "haze", "hazy") if (root / name).is_dir()]
    label_dirs = [root / name for name in ("GT", "gt") if (root / name).is_dir()]
    if len(input_dirs) != 1 or len(label_dirs) != 1:
        return {}
    clear_digest_cache: dict[Path, str] = {}
    groups: dict[str, list[tuple[Path, Path]]] = defaultdict(list)
    for hazy_path in image_files(input_dirs[0]):
        clear_path = selected_label(hazy_path.name, label_dirs[0])
        if clear_path is None:
            continue
        digest = clear_digest_cache.get(clear_path)
        if digest is None:
            digest = canonical_rgb_digest(image_array(clear_path))
            clear_digest_cache[clear_path] = digest
        if digest in allowed_scenes:
            groups[digest].append((hazy_path, clear_path))
    return dict(groups)


def stem_map(directory: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in image_files(directory):
        if path.stem in result:
            return {}
        result[path.stem] = path
    return result


def haze_prefix_map(directory: Path) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = defaultdict(list)
    for path in image_files(directory):
        result[path.stem.split("_", 1)[0]].append(path)
    return dict(result)


def enumerate_its(
    reside_root: Path, allowed_scenes: set[str],
) -> dict[str, list[tuple[Path, Path]]]:
    specifications = (
        ("ITS_TRAIN", reside_root / "official/ITS/train/ITS_clear", reside_root / "official/ITS/train/ITS_hazy"),
        ("ITS_VALIDATION", reside_root / "official/ITS/val/clear", reside_root / "official/ITS/val/hazy"),
    )
    groups: dict[str, list[tuple[Path, Path]]] = {}
    for namespace, clear_dir, hazy_dir in specifications:
        clears = stem_map(clear_dir)
        hazy = haze_prefix_map(hazy_dir)
        for stem, clear_path in clears.items():
            scene = f"{namespace}:{stem}"
            if scene in allowed_scenes:
                groups[scene] = [(path, clear_path) for path in hazy.get(stem, [])]
    return groups


def enumerate_ots(
    reside_root: Path, allowed_scenes: set[str],
) -> dict[str, list[tuple[Path, Path]]]:
    clear_dir = reside_root / "official/OTS_ALPHA/clear_images"
    hazy_dir = reside_root / "official/OTS_ALPHA/hazy_images"
    clears = stem_map(clear_dir)
    hazy = haze_prefix_map(hazy_dir)
    return {
        scene: [(path, clears[scene]) for path in hazy.get(scene, [])]
        for scene in allowed_scenes if scene in clears
    }


def select_scene_pairs(
    dataset: str,
    groups: dict[str, list[tuple[Path, Path]]],
) -> tuple[dict[str, list[tuple[Path, Path]]], dict[str, Any]]:
    eligible = {
        scene: items for scene, items in groups.items()
        if len(items) >= VARIANTS_PER_SCENE
    }
    order = deterministic_rank(eligible, f"{SCENE_SELECTION_SALT}|{dataset}")
    selected = order[:PLANNED_SCENES[dataset]]
    result = {}
    for scene in selected:
        variants = sorted(
            eligible[scene],
            key=lambda pair: (
                sha256_text(
                    f"{VARIANT_SELECTION_SALT}|{dataset}|{scene}|{pair[0].name}"
                ),
                pair[0].name,
            ),
        )[:VARIANTS_PER_SCENE]
        result[scene] = variants
    return result, {
        "allowed_scene_count": len(groups),
        "paired_scene_count": len(eligible),
        "selected_scene_count": len(result),
        "two_variants_per_scene": all(
            len(items) == VARIANTS_PER_SCENE for items in result.values()
        ),
        "selection_digest": sha256_text("\n".join(selected)),
    }


def load_official_model(context):
    import torch

    expected = {
        "official_checkpoint": CHECKPOINT_SHA256,
        "model_source": MODEL_SOURCE_SHA256,
        "model_layers": MODEL_LAYERS_SHA256,
    }
    for identifier, identity in expected.items():
        if context.assets[identifier].sha256 != identity:
            raise RuntimeError(f"verified identity changed for {identifier}")
    anchor_root = asset_path(context, "official_anchor_checkout", kind="git_checkout")
    if str(anchor_root) not in sys.path:
        sys.path.insert(0, str(anchor_root))
    from Dehazing.ITS.models.ConvIR import build_net

    model_source = asset_path(context, "model_source", kind="file")
    model_layers = asset_path(context, "model_layers", kind="file")
    model_module = sys.modules[build_net.__module__]
    layer_module = sys.modules.get("Dehazing.ITS.models.layers")
    if Path(model_module.__file__).resolve() != model_source.resolve() \
            or layer_module is None \
            or Path(layer_module.__file__).resolve() != model_layers.resolve():
        raise RuntimeError("official model import resolved outside bound source assets")
    model = build_net("base", "Haze4K", fam_mode="original")
    checkpoint = asset_path(context, "official_checkpoint", kind="file")
    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location="cpu")
    if not isinstance(state, dict) or not isinstance(state.get("model"), dict):
        raise RuntimeError("official checkpoint lacks state_dict['model']")
    model.load_state_dict(state["model"], strict=True)
    if sum(parameter.numel() for parameter in model.parameters()) != PARAMETER_COUNT:
        raise RuntimeError("official model parameter count changed")
    model.requires_grad_(False).to(context.device).eval()
    return torch, model


def deterministic_crop(
    hazy: np.ndarray, clear: np.ndarray, scene: str,
) -> tuple[np.ndarray, np.ndarray]:
    if hazy.shape != clear.shape or min(hazy.shape[:2]) < MIN_CROP_SIDE:
        raise RuntimeError(f"pair shape outside frozen crop contract for {scene[:24]}")
    height, width = hazy.shape[:2]
    crop_h, crop_w = min(height, MAX_CROP_SIDE), min(width, MAX_CROP_SIDE)
    seed = int(sha256_text(f"{CROP_SELECTION_SALT}|{scene}")[:16], 16)
    y0 = seed % (height - crop_h + 1)
    x0 = (seed // 65537) % (width - crop_w + 1)
    return (
        hazy[y0:y0 + crop_h, x0:x0 + crop_w].copy(),
        clear[y0:y0 + crop_h, x0:x0 + crop_w].copy(),
    )


def infer_official(torch, model, hazy: np.ndarray, device: str) -> np.ndarray:
    import torch.nn.functional as functional

    tensor = torch.from_numpy(hazy.transpose(2, 0, 1).copy()).unsqueeze(0).to(device)
    height, width = hazy.shape[:2]
    padded = functional.pad(
        tensor, (0, (-width) % 32, 0, (-height) % 32), mode="reflect",
    )
    with torch.inference_mode():
        outputs = model(padded)
    if not isinstance(outputs, list) or len(outputs) != 3:
        raise RuntimeError("official model output contract changed")
    prediction = outputs[2][:, :, :height, :width].clamp(0.0, 1.0)
    if not bool(torch.isfinite(prediction).all().item()):
        raise RuntimeError("official model produced non-finite output")
    return prediction.squeeze(0).permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)


def psnr(value: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> float:
    error = (value.astype(np.float64) - target.astype(np.float64)) ** 2
    if mask is not None:
        error = error[mask]
    mse = float(np.mean(error))
    return -10.0 * math.log10(max(mse, EPSILON))


def select_alpha(utilities: np.ndarray) -> int:
    best = float(np.max(utilities))
    eligible = np.flatnonzero(utilities >= best - ALPHA_TIE_DB)
    return int(min(eligible, key=lambda index: (abs(ALPHA_GRID[index] - 1.0), ALPHA_GRID[index])))


def direction(alpha: float) -> str:
    if alpha < KEEP_ALPHA:
        return "weaken"
    if alpha > KEEP_ALPHA:
        return "strengthen"
    return "keep"


def cell_slices(height: int, width: int) -> list[tuple[slice, slice]]:
    cells = []
    for row in range(GRID_SIDE):
        outer_y0 = math.floor(row * height / GRID_SIDE)
        outer_y1 = math.floor((row + 1) * height / GRID_SIDE)
        for column in range(GRID_SIDE):
            outer_x0 = math.floor(column * width / GRID_SIDE)
            outer_x1 = math.floor((column + 1) * width / GRID_SIDE)
            margin = max(4, math.floor(min(outer_y1 - outer_y0, outer_x1 - outer_x0) * 0.125))
            y0, y1 = outer_y0 + margin, outer_y1 - margin
            x0, x1 = outer_x0 + margin, outer_x1 - margin
            if y1 - y0 < 32 or x1 - x0 < 32:
                raise RuntimeError("scored tile core is smaller than 32 by 32")
            cells.append((slice(y0, y1), slice(x0, x1)))
    return cells


def candidate_errors(
    hazy: np.ndarray, prediction: np.ndarray, target: np.ndarray,
    region: tuple[slice, slice], mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    y, x = region
    local_hazy = hazy[y, x]
    residual = prediction[y, x] - local_hazy
    local_target = target[y, x]
    sums = np.empty(len(ALPHA_GRID), dtype=np.float64)
    counts = np.empty(len(ALPHA_GRID), dtype=np.float64)
    for index, alpha in enumerate(ALPHA_GRID):
        candidate = np.clip(local_hazy + float(alpha) * residual, 0.0, 1.0)
        error = (candidate.astype(np.float64) - local_target.astype(np.float64)) ** 2
        sums[index] = float(np.sum(error[mask]))
        counts[index] = float(np.count_nonzero(mask) * 3)
    return sums, counts


def utility_from_errors(sums: np.ndarray, counts: np.ndarray) -> np.ndarray:
    keep_index = int(np.flatnonzero(ALPHA_GRID == KEEP_ALPHA)[0])
    keep_mse = sums[keep_index] / counts[keep_index]
    return 10.0 * np.log10(
        np.maximum(keep_mse, EPSILON) / np.maximum(sums / counts, EPSILON)
    )


def analyze_variant(
    hazy: np.ndarray, clear: np.ndarray, prediction: np.ndarray,
) -> dict[str, Any]:
    height, width = hazy.shape[:2]
    shifted = np.roll(clear, shift=(height // 2, width // 2), axis=(0, 1))
    tile_records = []
    true_sums = [np.zeros(len(ALPHA_GRID)), np.zeros(len(ALPHA_GRID))]
    true_counts = [np.zeros(len(ALPHA_GRID)), np.zeros(len(ALPHA_GRID))]
    null_sums = [np.zeros(len(ALPHA_GRID)), np.zeros(len(ALPHA_GRID))]
    local_weighted_utility = 0.0
    null_weighted_utility = 0.0
    total_weight = 0.0
    for cell_index, region in enumerate(cell_slices(height, width)):
        y, x = region
        yy, xx = np.indices((y.stop - y.start, x.stop - x.start))
        masks = ((yy + xx) % 2 == 0, (yy + xx) % 2 == 1)
        true_utilities = []
        null_utilities = []
        for half in range(2):
            sums, counts = candidate_errors(hazy, prediction, clear, region, masks[half])
            wrong_sums, _ = candidate_errors(hazy, prediction, shifted, region, masks[half])
            true_sums[half] += sums
            true_counts[half] += counts
            null_sums[half] += wrong_sums
            true_utilities.append(utility_from_errors(sums, counts))
            null_utilities.append(utility_from_errors(wrong_sums, counts))
        selected_true = [select_alpha(true_utilities[0]), select_alpha(true_utilities[1])]
        selected_null = [select_alpha(null_utilities[0]), select_alpha(null_utilities[1])]
        cross_utility = 0.5 * (
            true_utilities[1][selected_true[0]] + true_utilities[0][selected_true[1]]
        )
        null_cross_utility = 0.5 * (
            true_utilities[1][selected_null[0]] + true_utilities[0][selected_null[1]]
        )
        selected_directions = [direction(float(ALPHA_GRID[index])) for index in selected_true]
        cross_halves = (
            float(true_utilities[1][selected_true[0]]),
            float(true_utilities[0][selected_true[1]]),
        )
        label = "keep"
        if selected_directions[0] == selected_directions[1] != "keep" \
                and min(cross_halves) >= LOCAL_DIRECTION_MARGIN_DB:
            label = selected_directions[0]
        local_hazy, local_clear, local_prediction = hazy[y, x], clear[y, x], prediction[y, x]
        hazy_psnr = psnr(local_hazy, local_clear)
        prediction_psnr = psnr(local_prediction, local_clear)
        weight = float(sum(np.count_nonzero(mask) for mask in masks))
        local_weighted_utility += weight * float(cross_utility)
        null_weighted_utility += weight * float(null_cross_utility)
        total_weight += weight
        tile_records.append({
            "cell": cell_index,
            "label": label,
            "selected_alpha_a": float(ALPHA_GRID[selected_true[0]]),
            "selected_alpha_b": float(ALPHA_GRID[selected_true[1]]),
            "cross_utility_db": float(cross_utility),
            "near_clear": hazy_psnr >= NEAR_CLEAR_PSNR_DB,
            "baseline_damage": prediction_psnr <= hazy_psnr - NEAR_CLEAR_DAMAGE_DB,
        })
    global_selected = [
        select_alpha(utility_from_errors(true_sums[half], true_counts[half]))
        for half in range(2)
    ]
    global_utilities = [
        utility_from_errors(true_sums[half], true_counts[half]) for half in range(2)
    ]
    global_cross = 0.5 * (
        global_utilities[1][global_selected[0]]
        + global_utilities[0][global_selected[1]]
    )
    null_selected = [
        select_alpha(utility_from_errors(null_sums[half], true_counts[half]))
        for half in range(2)
    ]
    null_global_cross = 0.5 * (
        global_utilities[1][null_selected[0]]
        + global_utilities[0][null_selected[1]]
    )
    return {
        "tiles": tile_records,
        "local_utility_db": local_weighted_utility / total_weight,
        "global_utility_db": float(global_cross),
        "local_minus_global_db": local_weighted_utility / total_weight - float(global_cross),
        "null_local_minus_global_db": null_weighted_utility / total_weight - float(null_global_cross),
        "baseline_psnr_db": psnr(prediction, clear),
        "hazy_psnr_db": psnr(hazy, clear),
    }


def analyze_scene(
    torch, model, dataset: str, scene: str,
    pairs: list[tuple[Path, Path]], device: str,
) -> tuple[dict[str, Any], str]:
    variants = []
    input_parts = ["local-restoration-need-v1", dataset, scene]
    expected_shape = None
    for hazy_path, clear_path in pairs:
        hazy, clear = deterministic_crop(
            image_array(hazy_path), image_array(clear_path), scene,
        )
        if expected_shape is None:
            expected_shape = hazy.shape
        elif hazy.shape != expected_shape:
            raise RuntimeError(f"nested variant crop shapes differ for {scene[:24]}")
        prediction = infer_official(torch, model, hazy, device)
        variants.append(analyze_variant(hazy, clear, prediction))
        input_parts.extend([
            hazy_path.name, sha256_file(hazy_path), clear_path.name, sha256_file(clear_path),
        ])
    if len(variants) != VARIANTS_PER_SCENE:
        raise RuntimeError("scene analysis did not use exactly two nested variants")
    labels = [[tile["label"] for tile in variant["tiles"]] for variant in variants]
    repeat_counts = Counter(
        labels[0][index] if labels[0][index] == labels[1][index] else "discordant"
        for index in range(GRID_SIDE * GRID_SIDE)
    )
    repeat_near = []
    repeat_mitigated = []
    for index in range(GRID_SIDE * GRID_SIDE):
        left, right = variants[0]["tiles"][index], variants[1]["tiles"][index]
        near = bool(left["near_clear"] and right["near_clear"])
        repeat_near.append(near)
        repeat_mitigated.append(bool(
            near
            and left["baseline_damage"] and right["baseline_damage"]
            and left["label"] == right["label"] == "weaken"
        ))
    primary = float(np.clip(
        np.mean([item["local_minus_global_db"] for item in variants]),
        -PRIMARY_CLIP_DB,
        PRIMARY_CLIP_DB,
    ))
    null = float(np.clip(
        np.mean([item["null_local_minus_global_db"] for item in variants]),
        -PRIMARY_CLIP_DB,
        PRIMARY_CLIP_DB,
    ))
    row = {
        "dataset": dataset,
        "scene_id_sha256": sha256_text(scene),
        "nested_variants": VARIANTS_PER_SCENE,
        "primary_local_minus_global_psnr_db": primary,
        "null_local_minus_global_psnr_db": null,
        "local_minus_keep_psnr_db": float(np.mean([item["local_utility_db"] for item in variants])),
        "best_global_minus_keep_psnr_db": float(np.mean([item["global_utility_db"] for item in variants])),
        "baseline_psnr_db": float(np.mean([item["baseline_psnr_db"] for item in variants])),
        "hazy_psnr_db": float(np.mean([item["hazy_psnr_db"] for item in variants])),
        "stable_cell_count": sum(value != "discordant" for value in (
            labels[0][index] if labels[0][index] == labels[1][index] else "discordant"
            for index in range(GRID_SIDE * GRID_SIDE)
        )),
        "repeat_weaken_cells": repeat_counts["weaken"],
        "repeat_keep_cells": repeat_counts["keep"],
        "repeat_strengthen_cells": repeat_counts["strengthen"],
        "direction_stable_scene": repeat_counts["discordant"] <= GRID_SIDE * GRID_SIDE - STABLE_CELL_COUNT,
        "bidirectional_scene": (
            repeat_counts["weaken"] >= BIDIRECTIONAL_CELL_COUNT
            and repeat_counts["strengthen"] >= BIDIRECTIONAL_CELL_COUNT
        ),
        "keep_scene": repeat_counts["keep"] >= KEEP_CELL_COUNT,
        "material_utility_scene": primary >= PRIMARY_MARGIN_DB,
        "near_clear_eligible_scene": any(repeat_near),
        "near_clear_mitigated_scene": any(repeat_mitigated),
    }
    return row, sha256_text("|".join(input_parts))


def bootstrap_interval(values: Iterable[float], seed: int) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size < 2 or not np.isfinite(array).all():
        raise RuntimeError("bootstrap requires at least two finite scene values")
    generator = np.random.default_rng(seed)
    draws = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    for start in range(0, BOOTSTRAP_RESAMPLES, 500):
        stop = min(start + 500, BOOTSTRAP_RESAMPLES)
        indices = generator.integers(0, array.size, size=(stop - start, array.size))
        draws[start:stop] = np.mean(array[indices], axis=1)
    point = float(np.mean(array))
    lower = float(np.quantile(draws, 0.00625))
    upper = float(np.quantile(draws, 0.99375))
    return {
        "scene_count": int(array.size),
        "estimate": point,
        "lower": lower,
        "upper": upper,
        "max_half_width": max(point - lower, upper - point),
        "resamples": BOOTSTRAP_RESAMPLES,
        "simultaneous_family": "overall_plus_three_dataset_means",
    }


def stratified_bootstrap(rows_by_dataset: dict[str, list[dict[str, Any]]], key: str, seed: int) -> dict[str, Any]:
    arrays = {
        dataset: np.asarray([row[key] for row in rows_by_dataset[dataset]], dtype=np.float64)
        for dataset in DATASET_ORDER
    }
    generator = np.random.default_rng(seed)
    draws = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    for start in range(0, BOOTSTRAP_RESAMPLES, 500):
        stop = min(start + 500, BOOTSTRAP_RESAMPLES)
        means = []
        for dataset in DATASET_ORDER:
            array = arrays[dataset]
            indices = generator.integers(0, array.size, size=(stop - start, array.size))
            means.append(np.mean(array[indices], axis=1))
        draws[start:stop] = np.mean(np.stack(means), axis=0)
    point = float(np.mean([np.mean(array) for array in arrays.values()]))
    lower = float(np.quantile(draws, 0.00625))
    upper = float(np.quantile(draws, 0.99375))
    return {
        "scene_count": sum(len(array) for array in arrays.values()),
        "estimate": point,
        "lower": lower,
        "upper": upper,
        "max_half_width": max(point - lower, upper - point),
        "resamples": BOOTSTRAP_RESAMPLES,
        "aggregation": "equal_weight_over_three_dataset_means",
        "simultaneous_family": "overall_plus_three_dataset_means",
    }


def wilson(successes: int, total: int) -> dict[str, Any]:
    if total <= 0 or not 0 <= successes <= total:
        raise RuntimeError("invalid Wilson interval inputs")
    z = SIMULTANEOUS_CRITICAL_VALUE
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
        "critical_value": z,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def evidence_identity(context, include_ledger: bool) -> dict[str, bool]:
    closeout = asset_path(context, "s0_closeout", kind="file")
    role_summary = asset_path(context, "s0_role_summary", kind="file")
    checks = {
        "s0_closeout_sha256": sha256_file(closeout) == S0_CLOSEOUT_SHA256,
        "s0_terminal": (
            read_json(closeout).get("state") == "COMPLETED_GATE_PASS"
            and read_json(closeout).get("decision") == "DAYTIME_DEHAZING_PROGRAM_FOUNDATION_PASS"
            and read_json(closeout).get("authorizes") == "AUTHOR_S1_UTILITY_QUALIFICATION_CONTRACT_ONLY"
        ),
        "s0_role_summary_sha256": sha256_file(role_summary) == S0_ROLE_SUMMARY_SHA256,
    }
    if include_ledger:
        ledger = asset_path(context, "s0_scene_role_ledger", kind="file")
        checks["s0_scene_role_ledger_sha256"] = sha256_file(ledger) == S0_LEDGER_SHA256
    return checks


def terminal_summary(
    context, gate_outcomes: dict[str, str], summary: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    summary["gate_outcomes"] = gate_outcomes
    summary_path = output_file(context, SUMMARY_NAME)
    metrics_path = output_file(context, SCENE_METRICS_NAME)
    gate_path = output_file(context, GATE_NAME)
    review_path = output_file(context, REVIEW_FACTS_NAME)
    atomic_json(summary_path, summary)
    write_csv(metrics_path, rows or [{"status": "no_valid_scene_rows"}])
    atomic_json(gate_path, {
        "schema_version": 1,
        "route_id": ROUTE_ID,
        "operation_id": OPERATION_ID,
        "gate_outcomes": gate_outcomes,
        "primary": summary.get("primary"),
        "prevalence": summary.get("prevalence"),
    })
    summary_sha = sha256_file(summary_path)
    facts = []
    for gate_id, outcome in gate_outcomes.items():
        facts.append({
            "fact_id": gate_id,
            "claim_id": gate_id,
            "metric": f"{gate_id} typed gate outcome",
            "unit": "typed outcome",
            "population": "450 planned retained development-screening original clear scenes",
            "grouping": "original clear scene; haze variants, cells, pixels, and resamples are nested",
            "point": None,
            "ci_lower": None,
            "ci_upper": None,
            "confidence_level": 0.95,
            "threshold": None,
            "threshold_operator": None,
            "gate_outcome": outcome,
            "source_filename": SUMMARY_NAME,
            "source_sha256": summary_sha,
            "json_pointers": {
                "point": None,
                "ci_lower": None,
                "ci_upper": None,
                "confidence_level": None,
                "threshold": None,
                "gate_outcome": f"/gate_outcomes/{gate_id}",
            },
        })
    atomic_json(review_path, {
        "schema_version": 2,
        "route_id": ROUTE_ID,
        "operation_id": OPERATION_ID,
        "run_id": context.run_id,
        "facts": facts,
    })


def contract(context_path: Path) -> None:
    context = load_context(context_path, "contract")
    prepare_phase_output(context)
    started = time.monotonic()
    import torch

    if context.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    checks = evidence_identity(context, include_ledger=False)
    runtime_environment = asset_path(context, "runtime_environment", kind="file")
    entrypoint = asset_path(context, "local_need_entrypoint", kind="file")
    anchor = context.assets.get("official_anchor_checkout")
    checks.update({
        "entrypoint_identity": context.assets["local_need_entrypoint"].sha256 == sha256_file(entrypoint),
        "runtime_environment_identity": sha256_file(runtime_environment) == RUNTIME_ENVIRONMENT_SHA256,
        "runtime_environment_contract": read_json(runtime_environment).get("device_class") == "cuda_sm89",
        "official_anchor_identity": anchor is not None and anchor.commit == ANCHOR_COMMIT,
        "run_only_assets_absent": all(
            identifier not in context.assets
            for identifier in ("s0_scene_role_ledger", "haze4k_train", "reside_root")
        ),
        "fixed_cost_contract": (
            context.engineering_contract["cost_contract"]["formal_iterations"]
            == FORMAL_INFERENCE_ITERATIONS
            and context.engineering_contract["cost_contract"]["probe_iterations"]
            == PROBE_ITERATIONS
        ),
    })
    torch, model = load_official_model(context)
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    finite = True
    for index in range(PROBE_ITERATIONS):
        hazy = generator.uniform(0.05, 0.95, size=(512, 512, 3)).astype(np.float32)
        clear = np.clip(hazy + generator.normal(0.0, 0.03, size=hazy.shape), 0.0, 1.0).astype(np.float32)
        prediction = infer_official(torch, model, hazy, context.device)
        result = analyze_variant(hazy, clear, prediction)
        finite = finite and all(math.isfinite(float(result[key])) for key in (
            "local_utility_db", "global_utility_db", "local_minus_global_db",
            "null_local_minus_global_db", "baseline_psnr_db", "hazy_psnr_db",
        ))
        write_contract_progress(
            context,
            completed_iterations=index + 1,
            total_iterations=PROBE_ITERATIONS,
            stage="synthetic_official_inference_and_local_measurement",
        )
    elapsed = time.monotonic() - started
    peak = float(torch.cuda.max_memory_allocated() / (1024 * 1024)) if context.device == "cuda" else 0.0
    checks.update({
        "strict_official_graph_loaded": sum(parameter.numel() for parameter in model.parameters()) == PARAMETER_COUNT,
        "five_action_crossfit_path": len(result["tiles"]) == GRID_SIDE * GRID_SIDE,
        "finite_synthetic_measurement": finite,
        "probe_iteration_count": PROBE_ITERATIONS == context.engineering_contract["cost_contract"]["probe_iterations"],
    })
    write_contract_result(
        context,
        checks=checks,
        engineering={
            "mode": "gpu_synthetic_no_data",
            "device": context.device,
            "fixture": {"batch": 1, "channels": 3, "height": 512, "width": 512},
            "production_path_exercised": True,
            "protected_data_touched": False,
            "scientific_output_created": False,
            "scientific_training_occurred": False,
            "cost": {
                "observed_iterations": PROBE_ITERATIONS,
                "observed_wall_seconds": elapsed,
                "observed_peak_memory_mib": peak,
            },
        },
    )


def run(context_path: Path) -> None:
    context = load_context(context_path, "run")
    prepare_phase_output(context)
    if context.total_units != TOTAL_UNITS \
            or context.evidence_role != "development_screening" \
            or any(context.protected_data_permissions.values()):
        raise RuntimeError("runtime role, unit, or protected-data contract changed")
    if load_completed_unit_ledger(context):
        raise RuntimeError("fresh S1 workload unexpectedly contains completed units")
    identity_checks = evidence_identity(context, include_ledger=True)
    ledger_roles, ledger_summary = read_s0_ledger(
        asset_path(context, "s0_scene_role_ledger", kind="file")
    )
    identity_checks["ledger_development_counts"] = ledger_summary["expected_counts_match"]
    write_workload_progress(context, completed_units=0, stage="identity_and_development_scope")

    haze4k_groups = enumerate_haze4k(
        asset_path(context, "haze4k_train", kind="directory"),
        ledger_roles["HAZE4K_TRAIN"],
    )
    reside_root = asset_path(context, "reside_root", kind="directory")
    all_groups = {
        "HAZE4K_TRAIN": haze4k_groups,
        "ITS": enumerate_its(reside_root, ledger_roles["ITS"]),
        "OTS": enumerate_ots(reside_root, ledger_roles["OTS"]),
    }
    selected_groups = {}
    coverage = {}
    for dataset in DATASET_ORDER:
        selected_groups[dataset], coverage[dataset] = select_scene_pairs(
            dataset, all_groups[dataset],
        )
    coverage_checks = {
        "exact_planned_scene_counts": all(
            coverage[dataset]["selected_scene_count"] == PLANNED_SCENES[dataset]
            for dataset in DATASET_ORDER
        ),
        "two_nested_variants_each": all(
            coverage[dataset]["two_variants_per_scene"] for dataset in DATASET_ORDER
        ),
        "development_only_membership": all(
            set(selected_groups[dataset]) <= ledger_roles[dataset]
            for dataset in DATASET_ORDER
        ),
        "confirmation_and_sealed_assets_absent": all(
            identifier not in context.assets
            for identifier in ("confirmation", "sealed_final", "nh_haze")
        ),
    }
    identity_pass = all(identity_checks.values())
    coverage_pass = all(coverage_checks.values())
    if not identity_pass or not coverage_pass:
        gate_outcomes = {
            "evidence_identity": "pass" if identity_pass else "fail",
            "development_scene_coverage": "pass" if coverage_pass else "fail",
            "measurement_null_control": "invalid",
            "local_utility_over_global": "invalid",
            "bidirectional_repeatability": "invalid",
            "near_clear_fidelity": "invalid",
            "primary_precision": "invalid",
        }
        write_workload_progress(context, completed_units=TOTAL_UNITS, stage="typed_identity_or_coverage_inconclusive")
        terminal_summary(context, gate_outcomes, {
            "schema_version": 1,
            "route_id": ROUTE_ID,
            "operation_id": OPERATION_ID,
            "run_id": context.run_id,
            "scope": "development-only privileged local-restoration-need qualification",
            "identity_checks": identity_checks,
            "ledger_summary": ledger_summary,
            "coverage": coverage,
            "coverage_checks": coverage_checks,
            "primary": None,
            "prevalence": None,
            "limitations": [
                "No scientific inference is permitted when S0 identity or planned paired-scene coverage is invalid.",
                "No confirmation, sealed-final, canary, or NH-Haze image is delivered to this route.",
            ],
        }, [])
        write_gate_result(
            context,
            gate_outcomes=gate_outcomes,
            details={
                "independent_scene_count": 0,
                "nested_variant_count": 0,
                "summary_file": SUMMARY_NAME,
                "scene_metrics_file": SCENE_METRICS_NAME,
                "gate_summary_file": GATE_NAME,
                "privileged_oracle_only": True,
                "network_training_occurred": False,
                "confirmation_or_sealed_data_touched": False,
            },
        )
        return

    torch, model = load_official_model(context)
    rows: list[dict[str, Any]] = []
    rows_by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    completed = 0
    for dataset in DATASET_ORDER:
        for index, scene in enumerate(sorted(selected_groups[dataset]), start=1):
            row, input_identity = analyze_scene(
                torch, model, dataset, scene, selected_groups[dataset][scene], context.device,
            )
            rows.append(row)
            rows_by_dataset[dataset].append(row)
            relative = f"units/{dataset.lower()}_{index:03d}.json"
            atomic_json(output_file(context, relative), {
                "schema_version": 1,
                "dataset": dataset,
                "scene_id_sha256": row["scene_id_sha256"],
                "nested_variants": VARIANTS_PER_SCENE,
                "metrics": row,
            })
            completed += 1
            record_completed_unit(
                context,
                unit_id=f"scene_{completed:03d}_{dataset.lower()}",
                input_sha256=input_identity,
                output_relpath=relative,
            )
            if completed % 10 == 0 or completed == 450:
                write_workload_progress(
                    context, completed_units=completed, stage="privileged_crossfit_scene_measurement",
                )

    primary_by_dataset = {
        dataset: bootstrap_interval(
            (row["primary_local_minus_global_psnr_db"] for row in rows_by_dataset[dataset]),
            BOOTSTRAP_SEED + index,
        )
        for index, dataset in enumerate(DATASET_ORDER, start=1)
    }
    primary_overall = stratified_bootstrap(
        rows_by_dataset, "primary_local_minus_global_psnr_db", BOOTSTRAP_SEED,
    )
    null_by_dataset = {
        dataset: bootstrap_interval(
            (row["null_local_minus_global_psnr_db"] for row in rows_by_dataset[dataset]),
            BOOTSTRAP_SEED + 10 + index,
        )
        for index, dataset in enumerate(DATASET_ORDER, start=1)
    }
    null_overall = stratified_bootstrap(
        rows_by_dataset, "null_local_minus_global_psnr_db", BOOTSTRAP_SEED + 10,
    )
    prevalence = {
        "material_utility_scene": wilson(sum(row["material_utility_scene"] for row in rows), len(rows)),
        "direction_stable_scene": wilson(sum(row["direction_stable_scene"] for row in rows), len(rows)),
        "bidirectional_scene": wilson(sum(row["bidirectional_scene"] for row in rows), len(rows)),
        "keep_scene": wilson(sum(row["keep_scene"] for row in rows), len(rows)),
    }
    near_rows = [row for row in rows if row["near_clear_eligible_scene"]]
    near_interval = (
        wilson(sum(row["near_clear_mitigated_scene"] for row in near_rows), len(near_rows))
        if near_rows else None
    )
    null_pass = (
        null_overall["upper"] < NULL_UCB_MARGIN_DB
        and all(item["upper"] < NULL_DATASET_UCB_MARGIN_DB for item in null_by_dataset.values())
    )
    utility_pass = (
        primary_overall["lower"] > PRIMARY_MARGIN_DB
        and sum(item["lower"] > 0.0 for item in primary_by_dataset.values()) >= 2
        and all(item["upper"] >= 0.0 for item in primary_by_dataset.values())
        and prevalence["material_utility_scene"]["lower"] > MATERIAL_SCENE_PREVALENCE
    )
    repeatability_pass = (
        prevalence["direction_stable_scene"]["lower"] > STABILITY_PREVALENCE
        and prevalence["bidirectional_scene"]["lower"] > BIDIRECTIONAL_PREVALENCE
        and prevalence["keep_scene"]["lower"] > KEEP_PREVALENCE
    )
    near_coverage = len(near_rows) >= NEAR_CLEAR_MIN_SCENES
    near_pass = bool(
        near_coverage and near_interval is not None
        and near_interval["lower"] > NEAR_CLEAR_MITIGATION_PREVALENCE
    )
    precision_intervals = [primary_overall, *primary_by_dataset.values()]
    precision_met = all(item["max_half_width"] <= PRECISION_HALF_WIDTH_DB for item in precision_intervals)
    gate_outcomes = {
        "evidence_identity": "pass",
        "development_scene_coverage": "pass",
        "measurement_null_control": "pass" if null_pass else "fail",
        "local_utility_over_global": "favorable" if utility_pass else "unfavorable",
        "bidirectional_repeatability": "favorable" if repeatability_pass else "unfavorable",
        "near_clear_fidelity": (
            "safe" if near_pass else "unsafe"
        ) if near_coverage else "indeterminate",
        "primary_precision": "met" if precision_met else "unmet",
    }
    aggregate_unit = output_file(context, "units/aggregate_inference.json")
    atomic_json(aggregate_unit, {
        "schema_version": 1,
        "primary_overall": primary_overall,
        "primary_by_dataset": primary_by_dataset,
        "null_overall": null_overall,
        "prevalence": prevalence,
        "near_clear": near_interval,
        "gate_outcomes": gate_outcomes,
    })
    record_completed_unit(
        context,
        unit_id="aggregate_scene_grouped_inference",
        input_sha256=sha256_text("|".join(sorted(row["scene_id_sha256"] for row in rows))),
        output_relpath="units/aggregate_inference.json",
    )
    completed_ledger = load_completed_unit_ledger(context)
    coverage_checks["completed_unit_ledger"] = len(completed_ledger) == TOTAL_UNITS
    if not coverage_checks["completed_unit_ledger"]:
        gate_outcomes["development_scene_coverage"] = "fail"
        for gate in (
            "measurement_null_control", "local_utility_over_global",
            "bidirectional_repeatability", "near_clear_fidelity", "primary_precision",
        ):
            gate_outcomes[gate] = "invalid"
    write_workload_progress(context, completed_units=TOTAL_UNITS, stage="scene_grouped_inference_complete")
    terminal_summary(context, gate_outcomes, {
        "schema_version": 1,
        "route_id": ROUTE_ID,
        "operation_id": OPERATION_ID,
        "run_id": context.run_id,
        "scope": "development-only privileged bidirectional local-restoration-need qualification",
        "independent_unit": "original_clear_scene",
        "dataset_scene_counts": {dataset: len(rows_by_dataset[dataset]) for dataset in DATASET_ORDER},
        "nested_variant_count": len(rows) * VARIANTS_PER_SCENE,
        "identity_checks": identity_checks,
        "ledger_summary": ledger_summary,
        "coverage": coverage,
        "coverage_checks": coverage_checks,
        "intervention": {
            "formula": "clip(hazy + alpha * (official_output - hazy), 0, 1)",
            "alpha_grid": ALPHA_GRID.tolist(),
            "grid": "4x4 equal-area cells with central 75 percent scored cores",
            "selection": "two-way checkerboard-pixel cross-selection and evaluation",
            "replication": "two outcome-blind haze observations nested within each clear scene",
            "privileged_gt_use": "offline diagnostic oracle only; not deployable inference and not a proposed post-processing method",
        },
        "primary": {
            "minimum_meaningful_effect_db": PRIMARY_MARGIN_DB,
            "clip_db": [-PRIMARY_CLIP_DB, PRIMARY_CLIP_DB],
            "overall_equal_dataset_weight": primary_overall,
            "by_dataset": primary_by_dataset,
            "precision_target_half_width_db": PRECISION_HALF_WIDTH_DB,
        },
        "null_control": {
            "selection_target": "clear target circularly shifted by half image height and width",
            "overall": null_overall,
            "by_dataset": null_by_dataset,
        },
        "prevalence": prevalence,
        "near_clear_fidelity": {
            "definition": "hazy-input core PSNR at least 30 dB in both nested observations",
            "eligible_scenes": len(near_rows),
            "minimum_eligible_scenes": NEAR_CLEAR_MIN_SCENES,
            "mitigation_interval": near_interval,
        },
        "forbidden_activity_receipt": {
            "network_training_or_fitting_occurred": False,
            "observable_signal_or_mechanism_selected": False,
            "confirmation_or_sealed_final_touched": False,
            "images_or_arrays_archived_to_github": False,
        },
        "limitations": [
            "GT is used only by a privileged offline diagnostic oracle; S1 makes no inference-time observability claim.",
            "The residual-strength intervention qualifies the research problem and is not a deployable method or authorized post-processing solution.",
            "Each original clear scene contributes once; haze variants, cells, pixels, actions, and bootstrap draws remain nested.",
            "Exact RGB and archived known-overlap controls do not prove complete capture or source-provenance disjointness.",
            "A PASS authorizes only S2 mechanism-discovery contract authoring and never module training or protected-data access.",
        ],
        "marker": "DAYTIME_DEHAZING_LOCAL_RESTORATION_NEED_V1_COMPLETE",
    }, rows)
    write_gate_result(
        context,
        gate_outcomes=gate_outcomes,
        details={
            "independent_scene_count": len(rows),
            "nested_variant_count": len(rows) * VARIANTS_PER_SCENE,
            "summary_file": SUMMARY_NAME,
            "scene_metrics_file": SCENE_METRICS_NAME,
            "gate_summary_file": GATE_NAME,
            "privileged_oracle_only": True,
            "network_training_occurred": False,
            "confirmation_or_sealed_data_touched": False,
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
