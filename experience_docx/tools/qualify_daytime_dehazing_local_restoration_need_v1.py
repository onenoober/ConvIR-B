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
INFERENCE_HALO = 64
CROSSFIT_BLOCK_SIDE = 8
ALPHA_GRID = np.asarray([0.5, 0.75, 1.0, 1.25, 1.5], dtype=np.float64)
DENSE_GLOBAL_ALPHA_GRID = np.linspace(0.5, 1.5, 1001, dtype=np.float64)
KEEP_ALPHA = 1.0
ALPHA_TIE_DB = 0.005
GLOBAL_ALPHA_TIE_DB = 0.0001
LOCAL_DIRECTION_MARGIN_DB = 0.05
KEEP_EQUIVALENCE_DB = 0.02
PRIMARY_MARGIN_DB = 0.10
TRANSFER_MARGIN_DB = 0.0
SPATIAL_SPECIFICITY_MARGIN_DB = 0.05
PRIMARY_CLIP_DB = 0.25
MATERIAL_SCENE_PREVALENCE = 0.20
JOINT_SCENE_PREVALENCE = 0.15
STABLE_CELL_COUNT = 12
STRENGTH_STABLE_CELL_COUNT = 10
BIDIRECTIONAL_CELL_COUNT = 2
KEEP_CELL_COUNT = 4
STABILITY_PREVALENCE = 0.50
BIDIRECTIONAL_PREVALENCE = 0.20
KEEP_PREVALENCE = 0.20
NEGATIVE_TAIL_DB = -0.10
NEGATIVE_TAIL_PREVALENCE = 0.10
NEAR_CLEAR_PSNR_DB = 30.0
NEAR_CLEAR_DAMAGE_DB = 0.10
NEAR_CLEAR_MITIGATION_DB = 0.05
NEAR_CLEAR_MIN_EXPOSED_SCENES = 42
NEAR_CLEAR_MIN_EXPOSED_DATASETS = 2
NEAR_CLEAR_MIN_DAMAGED_SCENES = 42
NEAR_CLEAR_DAMAGE_PREVALENCE = 0.10
NEAR_CLEAR_MITIGATION_PREVALENCE = 0.50
NEAR_CLEAR_CELL_MITIGATION_FRACTION = 0.50
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


def read_s0_ledger(path: Path) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    roles: dict[str, dict[str, str]] = defaultdict(dict)
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
                dataset = str(row.get("dataset"))
                scene = str(row.get("scene_id"))
                canonical_digest = str(row.get("canonical_digest"))
                if len(canonical_digest) != 64 or any(
                    character not in "0123456789abcdef" for character in canonical_digest
                ):
                    raise RuntimeError("S0 ledger contains an invalid canonical RGB digest")
                if scene in roles[dataset] and roles[dataset][scene] != canonical_digest:
                    raise RuntimeError("S0 ledger maps one scene to multiple canonical digests")
                roles[dataset][scene] = canonical_digest
    counts = {dataset: len(roles[dataset]) for dataset in DATASET_ORDER}
    return roles, {
        "row_count": row_count,
        "excluded_count": excluded_count,
        "development_counts": counts,
        "expected_counts_match": counts == DEVELOPMENT_COUNTS,
    }


def enumerate_haze4k(
    root: Path, allowed_scenes: dict[str, str],
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
    reside_root: Path, allowed_scenes: dict[str, str],
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
    reside_root: Path, allowed_scenes: dict[str, str],
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
    allowed_scenes: dict[str, str],
) -> tuple[dict[str, list[tuple[Path, Path]]], dict[str, Any]]:
    planned = deterministic_rank(allowed_scenes, f"{SCENE_SELECTION_SALT}|{dataset}")[
        :PLANNED_SCENES[dataset]
    ]
    result = {}
    canonical_match = {}
    distinct_variant_payloads = {}
    selected_roster = []
    for scene in planned:
        items = groups.get(scene, [])
        if not items:
            continue
        clear_digests = {canonical_rgb_digest(image_array(pair[1])) for pair in items}
        canonical_match[scene] = clear_digests == {allowed_scenes[scene]}
        if len(items) < VARIANTS_PER_SCENE or not canonical_match[scene]:
            continue
        variants = sorted(
            items,
            key=lambda pair: (
                sha256_text(
                    f"{VARIANT_SELECTION_SALT}|{dataset}|{scene}|{pair[0].name}"
                ),
                pair[0].name,
            ),
        )[:VARIANTS_PER_SCENE]
        variant_hashes = [sha256_file(pair[0]) for pair in variants]
        distinct_variant_payloads[scene] = len(set(variant_hashes)) == VARIANTS_PER_SCENE
        if not distinct_variant_payloads[scene]:
            continue
        result[scene] = variants
        selected_roster.extend(
            f"{dataset}|{scene}|{pair[0].name}|{variant_hash}|{pair[1].name}|{sha256_file(pair[1])}"
            for pair, variant_hash in zip(variants, variant_hashes)
        )
    return result, {
        "allowed_scene_count": len(groups),
        "planned_scene_count": len(planned),
        "planned_scene_digest": sha256_text("\n".join(planned)),
        "selected_scene_count": len(result),
        "two_variants_per_scene": all(
            len(items) == VARIANTS_PER_SCENE for items in result.values()
        ),
        "exact_planned_roster_retained": set(result) == set(planned),
        "canonical_clear_identity_match": (
            len(canonical_match) == len(planned) and all(canonical_match.values())
        ),
        "distinct_haze_payloads": (
            len(distinct_variant_payloads) == len(planned)
            and all(distinct_variant_payloads.values())
        ),
        "selected_input_roster_digest": sha256_text("\n".join(selected_roster)),
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


def deterministic_crop_box(
    hazy: np.ndarray, clear: np.ndarray, scene: str,
) -> tuple[int, int, int, int]:
    if hazy.shape != clear.shape or min(hazy.shape[:2]) < MIN_CROP_SIDE:
        raise RuntimeError(f"pair shape outside frozen crop contract for {scene[:24]}")
    height, width = hazy.shape[:2]
    crop_h, crop_w = min(height, MAX_CROP_SIDE), min(width, MAX_CROP_SIDE)
    seed = int(sha256_text(f"{CROP_SELECTION_SALT}|{scene}")[:16], 16)
    y0 = seed % (height - crop_h + 1)
    x0 = (seed // 65537) % (width - crop_w + 1)
    return y0, y0 + crop_h, x0, x0 + crop_w


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


def infer_scored_crop(
    torch, model, hazy: np.ndarray, clear: np.ndarray, scene: str, device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    y0, y1, x0, x1 = deterministic_crop_box(hazy, clear, scene)
    height, width = hazy.shape[:2]
    halo_y0, halo_y1 = max(0, y0 - INFERENCE_HALO), min(height, y1 + INFERENCE_HALO)
    halo_x0, halo_x1 = max(0, x0 - INFERENCE_HALO), min(width, x1 + INFERENCE_HALO)
    halo_prediction = infer_official(
        torch, model, hazy[halo_y0:halo_y1, halo_x0:halo_x1], device,
    )
    local_y0, local_y1 = y0 - halo_y0, y1 - halo_y0
    local_x0, local_x1 = x0 - halo_x0, x1 - halo_x0
    return (
        hazy[y0:y1, x0:x1].copy(),
        clear[y0:y1, x0:x1].copy(),
        halo_prediction[local_y0:local_y1, local_x0:local_x1].copy(),
        {
            "top": y0 - halo_y0,
            "bottom": halo_y1 - y1,
            "left": x0 - halo_x0,
            "right": halo_x1 - x1,
        },
    )


def psnr(value: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> float:
    error = (value.astype(np.float64) - target.astype(np.float64)) ** 2
    if mask is not None:
        error = error[mask]
    mse = float(np.mean(error))
    return -10.0 * math.log10(max(mse, EPSILON))


def select_alpha(
    utilities: np.ndarray, grid: np.ndarray = ALPHA_GRID,
    tie_db: float = ALPHA_TIE_DB,
) -> int:
    best = float(np.max(utilities))
    eligible = np.flatnonzero(utilities >= best - tie_db)
    return int(min(eligible, key=lambda index: (abs(grid[index] - 1.0), grid[index])))


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


def crossfit_masks(region: tuple[slice, slice]) -> tuple[np.ndarray, np.ndarray]:
    y, x = region
    yy, xx = np.indices((y.stop - y.start, x.stop - x.start))
    parity = (yy // CROSSFIT_BLOCK_SIDE + xx // CROSSFIT_BLOCK_SIDE) % 2
    masks = (parity == 0, parity == 1)
    if min(np.count_nonzero(mask) for mask in masks) == 0:
        raise RuntimeError("macroblock cross-fit produced an empty half")
    return masks


def candidate_errors(
    hazy: np.ndarray, prediction: np.ndarray, target: np.ndarray,
    region: tuple[slice, slice], mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    y, x = region
    local_hazy = hazy[y, x].astype(np.float64)
    residual = prediction[y, x].astype(np.float64) - local_hazy
    local_target = target[y, x].astype(np.float64)
    sums = np.empty(len(ALPHA_GRID), dtype=np.float64)
    counts = np.empty(len(ALPHA_GRID), dtype=np.float64)
    for index, alpha in enumerate(ALPHA_GRID):
        candidate = np.clip(local_hazy + float(alpha) * residual, 0.0, 1.0)
        error = (candidate - local_target) ** 2
        sums[index] = float(np.sum(error[mask]))
        counts[index] = float(np.count_nonzero(mask) * 3)
    return sums, counts


def dense_global_errors(
    hazy: np.ndarray, prediction: np.ndarray, target: np.ndarray,
    region: tuple[slice, slice], mask: np.ndarray,
) -> tuple[np.ndarray, float]:
    y, x = region
    local_hazy = hazy[y, x][mask].astype(np.float64).reshape(-1)
    residual = (prediction[y, x][mask] - hazy[y, x][mask]).astype(np.float64).reshape(-1)
    local_target = target[y, x][mask].astype(np.float64).reshape(-1)
    offset = local_hazy - local_target
    coefficient_a = float(np.sum(residual * residual))
    coefficient_b = float(np.sum(residual * offset))
    coefficient_c = float(np.sum(offset * offset))

    positive = residual > 0.0
    negative = residual < 0.0
    breakpoints = np.full_like(residual, np.inf)
    bounds = np.zeros_like(residual)
    breakpoints[positive] = (1.0 - local_hazy[positive]) / residual[positive]
    bounds[positive] = 1.0
    breakpoints[negative] = -local_hazy[negative] / residual[negative]
    bounds[negative] = 0.0
    active = (positive | negative) & (breakpoints >= 1.0) & (breakpoints <= 1.5)

    bins_a = np.zeros(len(DENSE_GLOBAL_ALPHA_GRID), dtype=np.float64)
    bins_b = np.zeros(len(DENSE_GLOBAL_ALPHA_GRID), dtype=np.float64)
    bins_c = np.zeros(len(DENSE_GLOBAL_ALPHA_GRID), dtype=np.float64)
    if np.any(active):
        active_breakpoints = breakpoints[active]
        indices = np.ceil(
            (active_breakpoints - DENSE_GLOBAL_ALPHA_GRID[0])
            / (DENSE_GLOBAL_ALPHA_GRID[1] - DENSE_GLOBAL_ALPHA_GRID[0])
            - 1.0e-10
        ).astype(np.int64)
        indices = np.clip(indices, 0, len(DENSE_GLOBAL_ALPHA_GRID) - 1)
        active_residual = residual[active]
        active_offset = offset[active]
        saturated_error = (bounds[active] - local_target[active]) ** 2
        bins_a += np.bincount(
            indices, weights=-(active_residual ** 2), minlength=len(bins_a),
        )
        bins_b += np.bincount(
            indices, weights=-(active_residual * active_offset), minlength=len(bins_b),
        )
        bins_c += np.bincount(
            indices,
            weights=saturated_error - active_offset ** 2,
            minlength=len(bins_c),
        )
    cumulative_a = coefficient_a + np.cumsum(bins_a)
    cumulative_b = coefficient_b + np.cumsum(bins_b)
    cumulative_c = coefficient_c + np.cumsum(bins_c)
    grid = DENSE_GLOBAL_ALPHA_GRID
    sums = cumulative_a * grid * grid + 2.0 * cumulative_b * grid + cumulative_c
    return np.maximum(sums, 0.0), float(local_hazy.size)


def utility_from_errors(sums: np.ndarray, counts: np.ndarray) -> np.ndarray:
    keep_index = int(np.flatnonzero(ALPHA_GRID == KEEP_ALPHA)[0])
    keep_mse = sums[keep_index] / counts[keep_index]
    return 10.0 * np.log10(
        np.maximum(keep_mse, EPSILON) / np.maximum(sums / counts, EPSILON)
    )


def utility_from_dense_errors(sums: np.ndarray, count: float) -> np.ndarray:
    keep_index = int(np.flatnonzero(np.isclose(DENSE_GLOBAL_ALPHA_GRID, KEEP_ALPHA))[0])
    keep_mse = sums[keep_index] / count
    return 10.0 * np.log10(
        max(keep_mse, EPSILON) / np.maximum(sums / count, EPSILON)
    )


def pooled_db_gain(reference_sse: float, candidate_sse: float) -> float:
    return 10.0 * math.log10(
        max(float(reference_sse), EPSILON) / max(float(candidate_sse), EPSILON)
    )


def candidate_saturation_fraction(
    hazy: np.ndarray, prediction: np.ndarray,
    region: tuple[slice, slice], alpha: float,
) -> float:
    y, x = region
    raw = hazy[y, x].astype(np.float64) + float(alpha) * (
        prediction[y, x].astype(np.float64) - hazy[y, x].astype(np.float64)
    )
    return float(np.mean((raw < 0.0) | (raw > 1.0)))


def analyze_variant(
    hazy: np.ndarray, clear: np.ndarray, prediction: np.ndarray,
) -> dict[str, Any]:
    height, width = hazy.shape[:2]
    shifted = np.roll(clear, shift=(height // 2, width // 2), axis=(0, 1))
    tile_records: list[dict[str, Any]] = []
    cell_true_sums: list[list[np.ndarray]] = []
    cell_dense_true_sums: list[list[np.ndarray]] = []
    selected_true_by_cell: list[list[int]] = []
    selected_null_by_cell: list[list[int]] = []
    dense_true_sums = [np.zeros(len(DENSE_GLOBAL_ALPHA_GRID)), np.zeros(len(DENSE_GLOBAL_ALPHA_GRID))]
    dense_true_counts = [0.0, 0.0]
    dense_null_sums = [np.zeros(len(DENSE_GLOBAL_ALPHA_GRID)), np.zeros(len(DENSE_GLOBAL_ALPHA_GRID))]
    scored_mask = np.zeros((height, width), dtype=bool)
    for cell_index, region in enumerate(cell_slices(height, width)):
        y, x = region
        scored_mask[y, x] = True
        masks = crossfit_masks(region)
        true_utilities = []
        null_utilities = []
        half_true_sums = []
        half_dense_true_sums = []
        for half in range(2):
            sums, counts = candidate_errors(hazy, prediction, clear, region, masks[half])
            wrong_sums, _ = candidate_errors(hazy, prediction, shifted, region, masks[half])
            true_utilities.append(utility_from_errors(sums, counts))
            null_utilities.append(utility_from_errors(wrong_sums, counts))
            half_true_sums.append(sums)
            dense_sums, dense_count = dense_global_errors(
                hazy, prediction, clear, region, masks[half],
            )
            dense_wrong_sums, _ = dense_global_errors(
                hazy, prediction, shifted, region, masks[half],
            )
            half_dense_true_sums.append(dense_sums)
            dense_true_sums[half] += dense_sums
            dense_true_counts[half] += dense_count
            dense_null_sums[half] += dense_wrong_sums
        selected_true = [select_alpha(true_utilities[0]), select_alpha(true_utilities[1])]
        selected_null = [select_alpha(null_utilities[0]), select_alpha(null_utilities[1])]
        selected_true_by_cell.append(selected_true)
        selected_null_by_cell.append(selected_null)
        cell_true_sums.append(half_true_sums)
        cell_dense_true_sums.append(half_dense_true_sums)
        cross_utility = float(0.5 * (
            true_utilities[1][selected_true[0]] + true_utilities[0][selected_true[1]]
        ))
        selected_directions = [direction(float(ALPHA_GRID[index])) for index in selected_true]
        cross_halves = (
            float(true_utilities[1][selected_true[0]]),
            float(true_utilities[0][selected_true[1]]),
        )
        label = "unresolved"
        recommended_index = int(np.flatnonzero(ALPHA_GRID == KEEP_ALPHA)[0])
        if selected_true[0] == selected_true[1] \
                and selected_directions[0] == selected_directions[1] != "keep" \
                and min(cross_halves) >= LOCAL_DIRECTION_MARGIN_DB:
            label = selected_directions[0]
            recommended_index = selected_true[0]
        elif selected_true[0] == selected_true[1] == recommended_index:
            non_keep = np.flatnonzero(ALPHA_GRID != KEEP_ALPHA)
            if max(
                float(np.max(true_utilities[half][non_keep])) for half in range(2)
            ) <= KEEP_EQUIVALENCE_DB:
                label = "keep_supported"
        local_hazy, local_clear, local_prediction = hazy[y, x], clear[y, x], prediction[y, x]
        hazy_psnr = psnr(local_hazy, local_clear)
        prediction_psnr = psnr(local_prediction, local_clear)
        full_sums = half_true_sums[0] + half_true_sums[1]
        keep_index = int(np.flatnonzero(ALPHA_GRID == KEEP_ALPHA)[0])
        mitigation_db = pooled_db_gain(full_sums[keep_index], full_sums[recommended_index])
        row_index, column_index = divmod(cell_index, GRID_SIDE)
        tile_records.append({
            "cell": cell_index,
            "label": label,
            "selected_alpha_a": float(ALPHA_GRID[selected_true[0]]),
            "selected_alpha_b": float(ALPHA_GRID[selected_true[1]]),
            "recommended_alpha": (
                float(ALPHA_GRID[recommended_index]) if label != "unresolved" else None
            ),
            "recommended_index": recommended_index,
            "cross_utility_db": cross_utility,
            "near_clear": hazy_psnr >= NEAR_CLEAR_PSNR_DB,
            "baseline_damage": prediction_psnr <= hazy_psnr - NEAR_CLEAR_DAMAGE_DB,
            "mitigation_db": mitigation_db,
            "selected_candidate_saturation_fraction": candidate_saturation_fraction(
                hazy, prediction, region, float(ALPHA_GRID[recommended_index]),
            ),
            "edge_cell": row_index in {0, GRID_SIDE - 1} or column_index in {0, GRID_SIDE - 1},
        })

    global_selected = [
        select_alpha(
            utility_from_dense_errors(dense_true_sums[half], dense_true_counts[half]),
            DENSE_GLOBAL_ALPHA_GRID,
            GLOBAL_ALPHA_TIE_DB,
        )
        for half in range(2)
    ]
    null_global_selected = [
        select_alpha(
            utility_from_dense_errors(dense_null_sums[half], dense_true_counts[half]),
            DENSE_GLOBAL_ALPHA_GRID,
            GLOBAL_ALPHA_TIE_DB,
        )
        for half in range(2)
    ]

    keep_index = int(np.flatnonzero(ALPHA_GRID == KEEP_ALPHA)[0])
    local_sse = global_sse = keep_sse = 0.0
    null_local_sse = null_global_sse = 0.0
    permuted_local_sse = 0.0
    edge_local_sse = edge_global_sse = 0.0
    interior_local_sse = interior_global_sse = 0.0
    permutation_offset = 5
    for selection_half in range(2):
        evaluation_half = 1 - selection_half
        global_index = global_selected[selection_half]
        null_global_index = null_global_selected[selection_half]
        for cell_index, tile in enumerate(tile_records):
            selected_index = selected_true_by_cell[cell_index][selection_half]
            null_index = selected_null_by_cell[cell_index][selection_half]
            permuted_index = selected_true_by_cell[
                (cell_index + permutation_offset) % (GRID_SIDE * GRID_SIDE)
            ][selection_half]
            local_value = float(cell_true_sums[cell_index][evaluation_half][selected_index])
            global_value = float(cell_dense_true_sums[cell_index][evaluation_half][global_index])
            local_sse += local_value
            global_sse += global_value
            keep_sse += float(cell_true_sums[cell_index][evaluation_half][keep_index])
            null_local_sse += float(cell_true_sums[cell_index][evaluation_half][null_index])
            null_global_sse += float(
                cell_dense_true_sums[cell_index][evaluation_half][null_global_index]
            )
            permuted_local_sse += float(
                cell_true_sums[cell_index][evaluation_half][permuted_index]
            )
            if tile["edge_cell"]:
                edge_local_sse += local_value
                edge_global_sse += global_value
            else:
                interior_local_sse += local_value
                interior_global_sse += global_value

    return {
        "tiles": tile_records,
        "cell_true_sums": cell_true_sums,
        "dense_true_sums": dense_true_sums,
        "dense_true_counts": dense_true_counts,
        "local_utility_db": pooled_db_gain(keep_sse, local_sse),
        "global_utility_db": pooled_db_gain(keep_sse, global_sse),
        "local_minus_global_db": pooled_db_gain(global_sse, local_sse),
        "null_local_minus_global_db": pooled_db_gain(null_global_sse, null_local_sse),
        "aligned_minus_permuted_db": pooled_db_gain(permuted_local_sse, local_sse),
        "edge_local_minus_global_db": pooled_db_gain(edge_global_sse, edge_local_sse),
        "interior_local_minus_global_db": pooled_db_gain(
            interior_global_sse, interior_local_sse,
        ),
        "global_selected_alpha_a": float(DENSE_GLOBAL_ALPHA_GRID[global_selected[0]]),
        "global_selected_alpha_b": float(DENSE_GLOBAL_ALPHA_GRID[global_selected[1]]),
        "global_oracle_sse": float(np.min(dense_true_sums[0] + dense_true_sums[1])),
        "baseline_psnr_db": psnr(prediction, clear, scored_mask),
        "hazy_psnr_db": psnr(hazy, clear, scored_mask),
    }


def analyze_scene(
    torch, model, dataset: str, scene: str,
    pairs: list[tuple[Path, Path]], expected_clear_digest: str,
    expected_pair_identities: list[dict[str, str]], device: str,
) -> tuple[dict[str, Any], str]:
    variants: list[dict[str, Any]] = []
    input_parts = ["local-restoration-need-v1", dataset, scene]
    expected_shape = None
    halo_records = []
    for pair_index, (hazy_path, clear_path) in enumerate(pairs):
        identity = expected_pair_identities[pair_index]
        if hazy_path.name != identity["hazy_name"] \
                or clear_path.name != identity["clear_name"] \
                or sha256_file(hazy_path) != identity["hazy_sha256"] \
                or sha256_file(clear_path) != identity["clear_sha256"]:
            raise RuntimeError("selected input roster changed after preflight")
        full_hazy, full_clear = image_array(hazy_path), image_array(clear_path)
        if canonical_rgb_digest(full_clear) != expected_clear_digest:
            raise RuntimeError("clear-scene canonical identity changed after S0")
        hazy, clear, prediction, halo = infer_scored_crop(
            torch, model, full_hazy, full_clear, scene, device,
        )
        if expected_shape is None:
            expected_shape = hazy.shape
        elif hazy.shape != expected_shape:
            raise RuntimeError(f"nested variant crop shapes differ for {scene[:24]}")
        variants.append(analyze_variant(hazy, clear, prediction))
        halo_records.append(halo)
        if sha256_file(hazy_path) != identity["hazy_sha256"] \
                or sha256_file(clear_path) != identity["clear_sha256"]:
            raise RuntimeError("selected input payload changed during inference")
        input_parts.extend([
            identity["hazy_name"], identity["hazy_sha256"],
            identity["clear_name"], identity["clear_sha256"], expected_clear_digest,
        ])
    if len(variants) != VARIANTS_PER_SCENE:
        raise RuntimeError("scene analysis did not use exactly two nested variants")
    labels = [[tile["label"] for tile in variant["tiles"]] for variant in variants]
    repeated_labels = [
        labels[0][index]
        if labels[0][index] == labels[1][index] != "unresolved"
        else "discordant"
        for index in range(GRID_SIDE * GRID_SIDE)
    ]
    repeat_counts = Counter(repeated_labels)
    exact_alpha_agreement = []
    repeat_near: list[bool] = []
    repeat_damaged: list[bool] = []
    repeat_mitigated: list[bool] = []
    for index in range(GRID_SIDE * GRID_SIDE):
        left, right = variants[0]["tiles"][index], variants[1]["tiles"][index]
        exact_alpha_agreement.append(bool(
            repeated_labels[index] != "discordant"
            and left["recommended_alpha"] == right["recommended_alpha"]
        ))
        near = bool(left["near_clear"] and right["near_clear"])
        damaged = bool(near and left["baseline_damage"] and right["baseline_damage"])
        repeat_near.append(near)
        repeat_damaged.append(damaged)
        repeat_mitigated.append(bool(
            damaged
            and left["label"] == right["label"] == "weaken"
            and min(left["mitigation_db"], right["mitigation_db"])
            >= NEAR_CLEAR_MITIGATION_DB
        ))
    transfer_local_sse = 0.0
    transfer_global_sse = 0.0
    keep_index = int(np.flatnonzero(ALPHA_GRID == KEEP_ALPHA)[0])
    for source_index, target_index in ((0, 1), (1, 0)):
        source, target = variants[source_index], variants[target_index]
        for cell_index, source_tile in enumerate(source["tiles"]):
            selected_index = (
                int(source_tile["recommended_index"])
                if source_tile["label"] != "unresolved" else keep_index
            )
            transfer_local_sse += float(
                target["cell_true_sums"][cell_index][0][selected_index]
                + target["cell_true_sums"][cell_index][1][selected_index]
            )
        transfer_global_sse += float(target["global_oracle_sse"])
    transfer_raw = pooled_db_gain(transfer_global_sse, transfer_local_sse)
    primary_raw = float(np.mean([item["local_minus_global_db"] for item in variants]))
    specificity_raw = float(np.mean([item["aligned_minus_permuted_db"] for item in variants]))
    primary = float(np.clip(primary_raw, -PRIMARY_CLIP_DB, PRIMARY_CLIP_DB))
    transfer = float(np.clip(transfer_raw, -PRIMARY_CLIP_DB, PRIMARY_CLIP_DB))
    specificity = float(np.clip(specificity_raw, -PRIMARY_CLIP_DB, PRIMARY_CLIP_DB))
    damaged_cell_count = sum(repeat_damaged)
    mitigated_cell_count = sum(repeat_mitigated)
    mitigation_fraction = (
        mitigated_cell_count / damaged_cell_count if damaged_cell_count else 0.0
    )
    stable_cell_count = sum(value != "discordant" for value in repeated_labels)
    exact_alpha_count = sum(exact_alpha_agreement)
    bidirectional_scene = (
        repeat_counts["weaken"] >= BIDIRECTIONAL_CELL_COUNT
        and repeat_counts["strengthen"] >= BIDIRECTIONAL_CELL_COUNT
    )
    material_scene = primary >= PRIMARY_MARGIN_DB
    transfer_scene = transfer >= TRANSFER_MARGIN_DB
    row = {
        "dataset": dataset,
        "scene_id_sha256": sha256_text(scene),
        "nested_variants": VARIANTS_PER_SCENE,
        "primary_local_minus_global_psnr_db": primary,
        "primary_raw_local_minus_global_psnr_db": primary_raw,
        "primary_clipped_low": primary_raw < -PRIMARY_CLIP_DB,
        "primary_clipped_high": primary_raw > PRIMARY_CLIP_DB,
        "null_local_minus_global_psnr_db": float(np.mean([
            item["null_local_minus_global_db"] for item in variants
        ])),
        "spatial_specificity_psnr_db": specificity,
        "spatial_specificity_raw_psnr_db": specificity_raw,
        "cross_observation_transfer_psnr_db": transfer,
        "cross_observation_transfer_raw_psnr_db": transfer_raw,
        "local_minus_keep_psnr_db": float(np.mean([item["local_utility_db"] for item in variants])),
        "best_global_minus_keep_psnr_db": float(np.mean([item["global_utility_db"] for item in variants])),
        "baseline_psnr_db": float(np.mean([item["baseline_psnr_db"] for item in variants])),
        "hazy_psnr_db": float(np.mean([item["hazy_psnr_db"] for item in variants])),
        "edge_local_minus_global_psnr_db": float(np.mean([
            item["edge_local_minus_global_db"] for item in variants
        ])),
        "interior_local_minus_global_psnr_db": float(np.mean([
            item["interior_local_minus_global_db"] for item in variants
        ])),
        "stable_cell_count": stable_cell_count,
        "exact_alpha_stable_cell_count": exact_alpha_count,
        "repeat_weaken_cells": repeat_counts["weaken"],
        "repeat_keep_supported_cells": repeat_counts["keep_supported"],
        "repeat_strengthen_cells": repeat_counts["strengthen"],
        "unresolved_cell_fraction": float(np.mean([
            tile["label"] == "unresolved" for variant in variants for tile in variant["tiles"]
        ])),
        "exact_alpha_agreement_fraction": exact_alpha_count / (GRID_SIDE * GRID_SIDE),
        "local_endpoint_selection_fraction": float(np.mean([
            alpha in {float(ALPHA_GRID[0]), float(ALPHA_GRID[-1])}
            for variant in variants for tile in variant["tiles"]
            for alpha in (tile["selected_alpha_a"], tile["selected_alpha_b"])
        ])),
        "global_endpoint_selection_fraction": float(np.mean([
            alpha in {float(DENSE_GLOBAL_ALPHA_GRID[0]), float(DENSE_GLOBAL_ALPHA_GRID[-1])}
            for variant in variants
            for alpha in (variant["global_selected_alpha_a"], variant["global_selected_alpha_b"])
        ])),
        "selected_candidate_saturation_fraction": float(np.mean([
            tile["selected_candidate_saturation_fraction"]
            for variant in variants for tile in variant["tiles"]
        ])),
        "minimum_real_halo_pixels": min(
            value for halo in halo_records for value in halo.values()
        ),
        "direction_and_strength_stable_scene": (
            stable_cell_count >= STABLE_CELL_COUNT
            and exact_alpha_count >= STRENGTH_STABLE_CELL_COUNT
        ),
        "bidirectional_scene": bidirectional_scene,
        "keep_supported_scene": repeat_counts["keep_supported"] >= KEEP_CELL_COUNT,
        "material_utility_scene": material_scene,
        "negative_tail_scene": primary_raw < NEGATIVE_TAIL_DB,
        "joint_useful_bidirectional_scene": (
            material_scene and transfer_scene and bidirectional_scene
            and specificity >= SPATIAL_SPECIFICITY_MARGIN_DB
        ),
        "near_clear_exposed_scene": any(repeat_near),
        "near_clear_damaged_scene": damaged_cell_count > 0,
        "near_clear_mitigated_scene": (
            damaged_cell_count > 0
            and mitigation_fraction >= NEAR_CLEAR_CELL_MITIGATION_FRACTION
        ),
        "near_clear_exposed_cell_count": sum(repeat_near),
        "near_clear_damaged_cell_count": damaged_cell_count,
        "near_clear_mitigated_cell_count": mitigated_cell_count,
        "near_clear_cell_mitigation_fraction": mitigation_fraction,
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


def prevalence_family(
    rows_by_dataset: dict[str, list[dict[str, Any]]], key: str, seed: int,
) -> dict[str, Any]:
    return {
        "overall_equal_dataset_weight": stratified_bootstrap(rows_by_dataset, key, seed),
        "by_dataset": {
            dataset: wilson(
                sum(bool(row[key]) for row in rows_by_dataset[dataset]),
                len(rows_by_dataset[dataset]),
            )
            for dataset in DATASET_ORDER
        },
    }


def combine_states(states: Iterable[str]) -> str:
    values = list(states)
    if any(value == "unfavorable" for value in values):
        return "unfavorable"
    if values and all(value == "favorable" for value in values):
        return "favorable"
    return "indeterminate"


def mean_cross_dataset_state(
    family: dict[str, Any], overall_threshold: float,
    dataset_support_threshold: float, no_harm_threshold: float,
) -> str:
    overall = family["overall"]
    datasets = list(family["by_dataset"].values())
    favorable = (
        overall["lower"] > overall_threshold
        and sum(item["lower"] > dataset_support_threshold for item in datasets) >= 2
        and all(item["upper"] >= no_harm_threshold for item in datasets)
    )
    unfavorable = (
        overall["upper"] <= overall_threshold
        or sum(item["upper"] > dataset_support_threshold for item in datasets) < 2
        or any(item["upper"] < no_harm_threshold for item in datasets)
    )
    return "favorable" if favorable else "unfavorable" if unfavorable else "indeterminate"


def prevalence_cross_dataset_state(
    family: dict[str, Any], overall_threshold: float,
    dataset_support_threshold: float, no_support_floor: float,
) -> str:
    overall = family["overall_equal_dataset_weight"]
    datasets = list(family["by_dataset"].values())
    favorable = (
        overall["lower"] > overall_threshold
        and sum(item["lower"] > dataset_support_threshold for item in datasets) >= 2
        and all(item["upper"] >= no_support_floor for item in datasets)
    )
    unfavorable = (
        overall["upper"] <= overall_threshold
        or sum(item["upper"] > dataset_support_threshold for item in datasets) < 2
        or any(item["upper"] < no_support_floor for item in datasets)
    )
    return "favorable" if favorable else "unfavorable" if unfavorable else "indeterminate"


def maximum_prevalence_state(
    family: dict[str, Any], overall_threshold: float, dataset_threshold: float,
) -> str:
    overall = family["overall_equal_dataset_weight"]
    datasets = list(family["by_dataset"].values())
    if overall["upper"] < overall_threshold \
            and all(item["upper"] < dataset_threshold for item in datasets):
        return "favorable"
    if overall["lower"] >= overall_threshold \
            or any(item["lower"] >= dataset_threshold for item in datasets):
        return "unfavorable"
    return "indeterminate"


def near_clear_state(
    exposed_count: int,
    damage_interval: dict[str, Any] | None,
    damaged_count: int,
    mitigation_interval: dict[str, Any] | None,
) -> tuple[str, str]:
    if exposed_count < NEAR_CLEAR_MIN_EXPOSED_SCENES or damage_interval is None:
        return "indeterminate", "insufficient_near_clear_exposure"
    if damage_interval["upper"] < NEAR_CLEAR_DAMAGE_PREVALENCE:
        return "safe", "safe_no_material_damage"
    if damage_interval["lower"] > NEAR_CLEAR_DAMAGE_PREVALENCE:
        if damaged_count < NEAR_CLEAR_MIN_DAMAGED_SCENES or mitigation_interval is None:
            return "indeterminate", "material_damage_but_insufficient_conditional_precision"
        if mitigation_interval["lower"] > NEAR_CLEAR_MITIGATION_PREVALENCE:
            return "safe", "safe_damage_conditionally_mitigated"
        if mitigation_interval["upper"] <= NEAR_CLEAR_MITIGATION_PREVALENCE:
            return "unsafe", "material_damage_not_conditionally_mitigated"
    return "indeterminate", "damage_or_mitigation_interval_crosses_decision_threshold"


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


def freeze_selected_input_roster(
    selected_groups: dict[str, dict[str, list[tuple[Path, Path]]]],
) -> tuple[dict[str, dict[str, list[dict[str, str]]]], str]:
    identities: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(dict)
    records = []
    for dataset in DATASET_ORDER:
        for scene in sorted(selected_groups[dataset]):
            scene_records = []
            for hazy_path, clear_path in selected_groups[dataset][scene]:
                record = {
                    "hazy_name": hazy_path.name,
                    "hazy_sha256": sha256_file(hazy_path),
                    "clear_name": clear_path.name,
                    "clear_sha256": sha256_file(clear_path),
                }
                scene_records.append(record)
                records.append(
                    "|".join((dataset, scene, *record.values()))
                )
            identities[dataset][scene] = scene_records
    return identities, sha256_text("\n".join(records))


def complete_invalid_units(context, reason: str) -> None:
    reason_digest = sha256_text(reason)
    completed = 0
    for dataset in DATASET_ORDER:
        for index in range(1, PLANNED_SCENES[dataset] + 1):
            completed += 1
            relative = f"units/invalid_{dataset.lower()}_{index:03d}.json"
            atomic_json(output_file(context, relative), {
                "schema_version": 1,
                "status": "not_evaluated_due_to_pre_inference_validity_veto",
                "dataset": dataset,
                "planned_index": index,
                "reason_sha256": reason_digest,
            })
            record_completed_unit(
                context,
                unit_id=f"scene_{completed:03d}_{dataset.lower()}",
                input_sha256=sha256_text(f"{reason_digest}|{dataset}|{index}"),
                output_relpath=relative,
            )
    relative = "units/invalid_aggregate_inference.json"
    atomic_json(output_file(context, relative), {
        "schema_version": 1,
        "status": "not_evaluated_due_to_pre_inference_validity_veto",
        "reason_sha256": reason_digest,
    })
    record_completed_unit(
        context,
        unit_id="aggregate_scene_grouped_inference",
        input_sha256=reason_digest,
        output_relpath=relative,
    )
    if len(load_completed_unit_ledger(context)) != TOTAL_UNITS:
        raise RuntimeError("invalid terminal unit ledger is incomplete")


def reference_cases() -> dict[str, bool]:
    height = width = 256
    hazy = np.full((height, width, 3), 0.25, dtype=np.float32)
    prediction = np.full((height, width, 3), 0.65, dtype=np.float32)
    residual = prediction - hazy

    uniform_target = hazy + 1.25 * residual
    uniform = analyze_variant(hazy, uniform_target, prediction)

    heterogeneous_target = hazy + residual
    desired = [0.5] * 6 + [1.0] * 4 + [1.5] * 6
    for cell_index, region in enumerate(cell_slices(height, width)):
        y, x = region
        heterogeneous_target[y, x] = hazy[y, x] + desired[cell_index] * residual[y, x]
    heterogeneous = analyze_variant(hazy, heterogeneous_target, prediction)

    keep_target = prediction.copy()
    keep = analyze_variant(hazy, keep_target, prediction)

    unresolved_target = prediction.copy()
    for region in cell_slices(height, width):
        y, x = region
        masks = crossfit_masks(region)
        local = unresolved_target[y, x]
        local[masks[0]] = hazy[y, x][masks[0]] + 0.5 * residual[y, x][masks[0]]
        local[masks[1]] = hazy[y, x][masks[1]] + 1.5 * residual[y, x][masks[1]]
    unresolved = analyze_variant(hazy, unresolved_target, prediction)

    favorable_family = {
        "overall": {"lower": 0.11, "upper": 0.20},
        "by_dataset": {
            dataset: {"lower": 0.01, "upper": 0.20} for dataset in DATASET_ORDER
        },
    }
    unfavorable_family = {
        "overall": {"lower": 0.01, "upper": 0.09},
        "by_dataset": {
            dataset: {"lower": -0.01, "upper": 0.09} for dataset in DATASET_ORDER
        },
    }
    indeterminate_family = {
        "overall": {"lower": 0.09, "upper": 0.11},
        "by_dataset": {
            dataset: {"lower": -0.01, "upper": 0.10} for dataset in DATASET_ORDER
        },
    }
    same_action_reference_sse = np.asarray([0.01, 1.0], dtype=np.float64)
    small_hazy = np.linspace(0.0, 1.0, 48, dtype=np.float32).reshape(4, 4, 3)
    small_prediction = np.flip(small_hazy, axis=1).copy()
    small_target = np.clip(0.2 + 0.6 * small_hazy, 0.0, 1.0)
    small_mask = np.ones((4, 4), dtype=bool)
    dense_reference, _ = dense_global_errors(
        small_hazy, small_prediction, small_target,
        (slice(0, 4), slice(0, 4)), small_mask,
    )
    brute_reference = np.asarray([
        np.sum((
            np.clip(
                small_hazy.astype(np.float64)
                + float(alpha) * (
                    small_prediction.astype(np.float64) - small_hazy.astype(np.float64)
                ),
                0.0,
                1.0,
            )
            - small_target.astype(np.float64)
        ) ** 2)
        for alpha in DENSE_GLOBAL_ALPHA_GRID
    ])
    return {
        "pooled_same_action_zero": abs(
            pooled_db_gain(float(np.sum(same_action_reference_sse)),
                           float(np.sum(same_action_reference_sse)))
        ) < 1.0e-12,
        "dense_global_histogram_matches_direct_reference": bool(
            np.max(np.abs(dense_reference - brute_reference)) < 1.0e-8
        ),
        "uniform_optimum_has_zero_local_advantage": abs(uniform["local_minus_global_db"]) < 1.0e-9,
        "uniform_dense_global_recovers_known_alpha": all(
            abs(uniform[key] - 1.25) < 1.0e-12
            for key in ("global_selected_alpha_a", "global_selected_alpha_b")
        ),
        "heterogeneous_known_case_positive": heterogeneous["local_minus_global_db"] > 0.5,
        "heterogeneous_spatial_placement_matters": heterogeneous[
            "aligned_minus_permuted_db"
        ] > 0.5,
        "heterogeneous_known_directions": (
            sum(tile["label"] == "weaken" for tile in heterogeneous["tiles"]) == 6
            and sum(tile["label"] == "keep_supported" for tile in heterogeneous["tiles"]) == 4
            and sum(tile["label"] == "strengthen" for tile in heterogeneous["tiles"]) == 6
        ),
        "keep_requires_positive_equivalence_support": all(
            tile["label"] == "keep_supported" for tile in keep["tiles"]
        ),
        "conflicting_halves_are_unresolved": all(
            tile["label"] == "unresolved" for tile in unresolved["tiles"]
        ),
        "favorable_gate_reference": mean_cross_dataset_state(
            favorable_family, 0.10, 0.0, 0.0,
        ) == "favorable",
        "unfavorable_gate_reference": mean_cross_dataset_state(
            unfavorable_family, 0.10, 0.0, 0.0,
        ) == "unfavorable",
        "indeterminate_gate_reference": mean_cross_dataset_state(
            indeterminate_family, 0.10, 0.0, 0.0,
        ) == "indeterminate",
        "near_clear_no_damage_is_safe": near_clear_state(
            42, {"lower": 0.0, "upper": 0.05}, 0, None,
        ) == ("safe", "safe_no_material_damage"),
        "near_clear_mitigated_damage_is_safe": near_clear_state(
            80, {"lower": 0.20, "upper": 0.30}, 50,
            {"lower": 0.60, "upper": 0.80},
        ) == ("safe", "safe_damage_conditionally_mitigated"),
        "near_clear_unmitigated_damage_is_unsafe": near_clear_state(
            80, {"lower": 0.20, "upper": 0.30}, 50,
            {"lower": 0.10, "upper": 0.40},
        ) == ("unsafe", "material_damage_not_conditionally_mitigated"),
        "near_clear_threshold_crossing_is_indeterminate": near_clear_state(
            80, {"lower": 0.05, "upper": 0.20}, 20,
            {"lower": 0.20, "upper": 0.70},
        )[0] == "indeterminate",
    }


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
    checks.update({
        f"reference_{name}": passed for name, passed in reference_cases().items()
    })
    torch, model = load_official_model(context)
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    finite = True
    for index in range(PROBE_ITERATIONS):
        full_hazy = generator.uniform(0.05, 0.95, size=(640, 640, 3)).astype(np.float32)
        full_clear = np.clip(
            full_hazy + generator.normal(0.0, 0.03, size=full_hazy.shape), 0.0, 1.0,
        ).astype(np.float32)
        hazy, clear, prediction, _ = infer_scored_crop(
            torch, model, full_hazy, full_clear, f"synthetic-reference-{index}", context.device,
        )
        result = analyze_variant(hazy, clear, prediction)
        finite = finite and all(math.isfinite(float(result[key])) for key in (
            "local_utility_db", "global_utility_db", "local_minus_global_db",
            "null_local_minus_global_db", "aligned_minus_permuted_db",
            "baseline_psnr_db", "hazy_psnr_db",
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
        "five_local_action_crossfit_path": len(result["tiles"]) == GRID_SIDE * GRID_SIDE,
        "dense_global_comparator_path": len(result["dense_true_sums"][0]) == 1001,
        "finite_synthetic_measurement": finite,
        "probe_iteration_count": PROBE_ITERATIONS == context.engineering_contract["cost_contract"]["probe_iterations"],
    })
    write_contract_result(
        context,
        checks=checks,
        engineering={
            "mode": "gpu_synthetic_no_data",
            "device": context.device,
            "fixture": {"batch": 1, "channels": 3, "height": 640, "width": 640},
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
    ledger_roles: dict[str, dict[str, str]] = defaultdict(dict)
    ledger_summary: dict[str, Any] = {"status": "not_parsed"}
    try:
        ledger_roles, ledger_summary = read_s0_ledger(
            asset_path(context, "s0_scene_role_ledger", kind="file")
        )
        identity_checks["ledger_schema_and_counts"] = ledger_summary["expected_counts_match"]
    except Exception as error:
        identity_checks["ledger_schema_and_counts"] = False
        ledger_summary = {
            "status": "invalid",
            "error_type": type(error).__name__,
            "error_sha256": sha256_text(str(error)),
        }
    write_workload_progress(context, completed_units=0, stage="identity_and_development_scope")

    selected_groups: dict[str, dict[str, list[tuple[Path, Path]]]] = {
        dataset: {} for dataset in DATASET_ORDER
    }
    coverage: dict[str, Any] = {}
    input_roster_identities: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(dict)
    selected_input_roster_digest = None
    preflight_error = None
    if all(identity_checks.values()):
        try:
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
            for dataset in DATASET_ORDER:
                selected_groups[dataset], coverage[dataset] = select_scene_pairs(
                    dataset, all_groups[dataset], ledger_roles[dataset],
                )
            input_roster_identities, selected_input_roster_digest = freeze_selected_input_roster(
                selected_groups,
            )
        except Exception as error:
            preflight_error = {
                "error_type": type(error).__name__,
                "error_sha256": sha256_text(str(error)),
            }
    else:
        preflight_error = {"error_type": "IdentityVeto", "error_sha256": sha256_text("identity_veto")}

    coverage_checks = {
        "exact_planned_scene_counts": all(
            coverage.get(dataset, {}).get("selected_scene_count") == PLANNED_SCENES[dataset]
            for dataset in DATASET_ORDER
        ),
        "two_nested_variants_each": all(
            coverage.get(dataset, {}).get("two_variants_per_scene") is True
            for dataset in DATASET_ORDER
        ),
        "exact_planned_roster_without_backfill": all(
            coverage.get(dataset, {}).get("exact_planned_roster_retained") is True
            for dataset in DATASET_ORDER
        ),
        "s0_canonical_clear_identity": all(
            coverage.get(dataset, {}).get("canonical_clear_identity_match") is True
            for dataset in DATASET_ORDER
        ),
        "distinct_haze_observation_payloads": all(
            coverage.get(dataset, {}).get("distinct_haze_payloads") is True
            for dataset in DATASET_ORDER
        ),
        "development_only_membership": all(
            set(selected_groups[dataset]) <= set(ledger_roles[dataset])
            for dataset in DATASET_ORDER
        ),
        "selected_input_roster_frozen": selected_input_roster_digest is not None,
        "preflight_completed_without_error": preflight_error is None,
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
        complete_invalid_units(
            context,
            json.dumps(
                {
                    "identity_checks": identity_checks,
                    "coverage_checks": coverage_checks,
                    "preflight_error": preflight_error,
                },
                sort_keys=True,
            ),
        )
        write_workload_progress(
            context, completed_units=TOTAL_UNITS,
            stage="typed_identity_or_coverage_inconclusive",
        )
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
            "preflight_error": preflight_error,
            "selected_input_roster_digest": selected_input_roster_digest,
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
                torch, model, dataset, scene, selected_groups[dataset][scene],
                ledger_roles[dataset][scene], input_roster_identities[dataset][scene],
                context.device,
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

    def mean_family(key: str, seed: int) -> dict[str, Any]:
        return {
            "overall": stratified_bootstrap(rows_by_dataset, key, seed),
            "by_dataset": {
                dataset: bootstrap_interval(
                    (row[key] for row in rows_by_dataset[dataset]), seed + index,
                )
                for index, dataset in enumerate(DATASET_ORDER, start=1)
            },
        }

    primary = mean_family("primary_local_minus_global_psnr_db", BOOTSTRAP_SEED)
    null = mean_family("null_local_minus_global_psnr_db", BOOTSTRAP_SEED + 10)
    specificity = mean_family("spatial_specificity_psnr_db", BOOTSTRAP_SEED + 20)
    transfer = mean_family("cross_observation_transfer_psnr_db", BOOTSTRAP_SEED + 30)
    prevalence = {
        "material_utility_scene": prevalence_family(
            rows_by_dataset, "material_utility_scene", BOOTSTRAP_SEED + 40,
        ),
        "direction_and_strength_stable_scene": prevalence_family(
            rows_by_dataset, "direction_and_strength_stable_scene", BOOTSTRAP_SEED + 50,
        ),
        "bidirectional_scene": prevalence_family(
            rows_by_dataset, "bidirectional_scene", BOOTSTRAP_SEED + 60,
        ),
        "keep_supported_scene": prevalence_family(
            rows_by_dataset, "keep_supported_scene", BOOTSTRAP_SEED + 70,
        ),
        "joint_useful_bidirectional_scene": prevalence_family(
            rows_by_dataset, "joint_useful_bidirectional_scene", BOOTSTRAP_SEED + 80,
        ),
        "negative_tail_scene": prevalence_family(
            rows_by_dataset, "negative_tail_scene", BOOTSTRAP_SEED + 90,
        ),
    }
    near_rows = [row for row in rows if row["near_clear_exposed_scene"]]
    damaged_rows = [row for row in near_rows if row["near_clear_damaged_scene"]]
    near_damage_interval = (
        wilson(sum(row["near_clear_damaged_scene"] for row in near_rows), len(near_rows))
        if near_rows else None
    )
    near_mitigation_interval = (
        wilson(sum(row["near_clear_mitigated_scene"] for row in damaged_rows), len(damaged_rows))
        if damaged_rows else None
    )
    near_outcome, near_basis = near_clear_state(
        len(near_rows), near_damage_interval, len(damaged_rows), near_mitigation_interval,
    )
    near_by_dataset = {}
    for dataset in DATASET_ORDER:
        dataset_near = [
            row for row in rows_by_dataset[dataset] if row["near_clear_exposed_scene"]
        ]
        dataset_damaged = [
            row for row in dataset_near if row["near_clear_damaged_scene"]
        ]
        near_by_dataset[dataset] = {
            "exposed_scenes": len(dataset_near),
            "damage_interval": (
                wilson(len(dataset_damaged), len(dataset_near)) if dataset_near else None
            ),
            "damaged_scenes": len(dataset_damaged),
            "mitigation_interval": (
                wilson(
                    sum(row["near_clear_mitigated_scene"] for row in dataset_damaged),
                    len(dataset_damaged),
                ) if dataset_damaged else None
            ),
        }
    near_dataset_decisions = {}
    for dataset, details in near_by_dataset.items():
        if details["exposed_scenes"] < NEAR_CLEAR_MIN_EXPOSED_SCENES:
            near_dataset_decisions[dataset] = {
                "outcome": "indeterminate",
                "basis": "insufficient_dataset_near_clear_exposure",
            }
            continue
        dataset_outcome, dataset_basis = near_clear_state(
            details["exposed_scenes"],
            details["damage_interval"],
            details["damaged_scenes"],
            details["mitigation_interval"],
        )
        near_dataset_decisions[dataset] = {
            "outcome": dataset_outcome,
            "basis": dataset_basis,
        }
    sufficiently_exposed_datasets = sum(
        details["exposed_scenes"] >= NEAR_CLEAR_MIN_EXPOSED_SCENES
        for details in near_by_dataset.values()
    )
    if any(item["outcome"] == "unsafe" for item in near_dataset_decisions.values()):
        near_outcome, near_basis = "unsafe", "dataset_specific_material_damage_not_mitigated"
    elif near_outcome == "safe" and (
        sufficiently_exposed_datasets < NEAR_CLEAR_MIN_EXPOSED_DATASETS
        or any(
            item["outcome"] != "safe" for item in near_dataset_decisions.values()
            if item["basis"] != "insufficient_dataset_near_clear_exposure"
        )
    ):
        near_outcome, near_basis = "indeterminate", "insufficient_cross_dataset_near_clear_support"
    null_pass = (
        null["overall"]["upper"] < NULL_UCB_MARGIN_DB
        and all(
            item["upper"] < NULL_DATASET_UCB_MARGIN_DB
            for item in null["by_dataset"].values()
        )
    )
    utility_state = combine_states((
        mean_cross_dataset_state(primary, PRIMARY_MARGIN_DB, 0.0, 0.0),
        mean_cross_dataset_state(
            specificity, SPATIAL_SPECIFICITY_MARGIN_DB, 0.0, 0.0,
        ),
        prevalence_cross_dataset_state(
            prevalence["material_utility_scene"], MATERIAL_SCENE_PREVALENCE, 0.10, 0.05,
        ),
        maximum_prevalence_state(
            prevalence["negative_tail_scene"], NEGATIVE_TAIL_PREVALENCE, 0.15,
        ),
    ))
    repeatability_state = combine_states((
        mean_cross_dataset_state(transfer, TRANSFER_MARGIN_DB, 0.0, 0.0),
        prevalence_cross_dataset_state(
            prevalence["direction_and_strength_stable_scene"],
            STABILITY_PREVALENCE, 0.35, 0.25,
        ),
        prevalence_cross_dataset_state(
            prevalence["bidirectional_scene"], BIDIRECTIONAL_PREVALENCE, 0.10, 0.05,
        ),
        prevalence_cross_dataset_state(
            prevalence["keep_supported_scene"], KEEP_PREVALENCE, 0.10, 0.05,
        ),
        prevalence_cross_dataset_state(
            prevalence["joint_useful_bidirectional_scene"],
            JOINT_SCENE_PREVALENCE, 0.05, 0.02,
        ),
    ))
    precision_intervals = [primary["overall"], *primary["by_dataset"].values()]
    precision_met = all(item["max_half_width"] <= PRECISION_HALF_WIDTH_DB for item in precision_intervals)
    gate_outcomes = {
        "evidence_identity": "pass",
        "development_scene_coverage": "pass",
        "measurement_null_control": "pass" if null_pass else "fail",
        "local_utility_over_global": utility_state,
        "bidirectional_repeatability": repeatability_state,
        "near_clear_fidelity": near_outcome,
        "primary_precision": "met" if precision_met else "unmet",
    }
    aggregate_unit = output_file(context, "units/aggregate_inference.json")
    atomic_json(aggregate_unit, {
        "schema_version": 1,
        "primary": primary,
        "spatial_specificity": specificity,
        "cross_observation_transfer": transfer,
        "null": null,
        "prevalence": prevalence,
        "near_clear": {
            "exposed_scenes": len(near_rows),
            "damage_interval": near_damage_interval,
            "damaged_scenes": len(damaged_rows),
            "mitigation_interval": near_mitigation_interval,
            "decision_basis": near_basis,
            "by_dataset": near_by_dataset,
            "dataset_decisions": near_dataset_decisions,
            "sufficiently_exposed_dataset_count": sufficiently_exposed_datasets,
        },
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
        "selected_input_roster_digest": selected_input_roster_digest,
        "intervention": {
            "formula": "clip(hazy + alpha * (official_output - hazy), 0, 1)",
            "local_alpha_grid": ALPHA_GRID.tolist(),
            "global_alpha_grid": {
                "minimum": float(DENSE_GLOBAL_ALPHA_GRID[0]),
                "maximum": float(DENSE_GLOBAL_ALPHA_GRID[-1]),
                "step": float(DENSE_GLOBAL_ALPHA_GRID[1] - DENSE_GLOBAL_ALPHA_GRID[0]),
                "count": len(DENSE_GLOBAL_ALPHA_GRID),
            },
            "grid": "4x4 equal-area cells; central 75 percent of each side, 56.25 percent area, is scored without a taper",
            "selection": "two-way 8x8 macroblock checkerboard cross-selection and opposite-half evaluation",
            "inference_context": "the scored crop is inferred with up to 64 real image pixels of halo on each available side",
            "replication": "two outcome-blind haze observations nested within each clear scene",
            "privileged_gt_use": "offline diagnostic oracle only; not deployable inference and not a proposed post-processing method",
        },
        "primary": {
            "minimum_meaningful_effect_db": PRIMARY_MARGIN_DB,
            "clip_db": [-PRIMARY_CLIP_DB, PRIMARY_CLIP_DB],
            "overall_equal_dataset_weight": primary["overall"],
            "by_dataset": primary["by_dataset"],
            "precision_target_half_width_db": PRECISION_HALF_WIDTH_DB,
            "clipping_diagnostics": {
                "low_scene_count": sum(row["primary_clipped_low"] for row in rows),
                "high_scene_count": sum(row["primary_clipped_high"] for row in rows),
            },
        },
        "spatial_specificity_control": specificity,
        "cross_observation_transfer": transfer,
        "null_control": {
            "selection_target": "clear target circularly shifted by half image height and width",
            "overall": null["overall"],
            "by_dataset": null["by_dataset"],
        },
        "prevalence": prevalence,
        "near_clear_fidelity": {
            "definition": "exposure, repeatable baseline damage, and damage-conditional mitigation are separate scene-grouped estimands",
            "exposed_scenes": len(near_rows),
            "minimum_exposed_scenes": NEAR_CLEAR_MIN_EXPOSED_SCENES,
            "minimum_exposed_datasets": NEAR_CLEAR_MIN_EXPOSED_DATASETS,
            "damage_interval": near_damage_interval,
            "damaged_scenes": len(damaged_rows),
            "minimum_damaged_scenes_for_conditional_decision": NEAR_CLEAR_MIN_DAMAGED_SCENES,
            "mitigation_interval": near_mitigation_interval,
            "decision_basis": near_basis,
            "by_dataset": near_by_dataset,
            "dataset_decisions": near_dataset_decisions,
            "sufficiently_exposed_dataset_count": sufficiently_exposed_datasets,
        },
        "diagnostics": {
            "mean_unresolved_cell_fraction": float(np.mean([
                row["unresolved_cell_fraction"] for row in rows
            ])),
            "mean_exact_alpha_agreement_fraction": float(np.mean([
                row["exact_alpha_agreement_fraction"] for row in rows
            ])),
            "mean_local_endpoint_selection_fraction": float(np.mean([
                row["local_endpoint_selection_fraction"] for row in rows
            ])),
            "mean_global_endpoint_selection_fraction": float(np.mean([
                row["global_endpoint_selection_fraction"] for row in rows
            ])),
            "mean_selected_candidate_saturation_fraction": float(np.mean([
                row["selected_candidate_saturation_fraction"] for row in rows
            ])),
            "edge_minus_interior_local_advantage_db": float(np.mean([
                row["edge_local_minus_global_psnr_db"]
                - row["interior_local_minus_global_psnr_db"] for row in rows
            ])),
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
            "The global claim is limited to one scene-global residual strength on the frozen 1001-point [0.5, 1.5] grid; it is not a claim over arbitrary restoration operators.",
            "The first two haze filenames are selected deterministically and their names and payload hashes are frozen at run preflight because S0 did not archive a hazy-payload manifest.",
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
