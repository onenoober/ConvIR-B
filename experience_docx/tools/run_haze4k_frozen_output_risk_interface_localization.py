#!/usr/bin/env python3
"""Localize the weighted-BCE risk interface using frozen development outputs."""

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
from statistics import NormalDist
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


ROUTE_ID = "haze4k-frozen-output-risk-interface-localization-v1"
OPERATION_ID = "HAZE4K_FROZEN_OUTPUT_RISK_INTERFACE_LOCALIZE"
PARENT_ROUTE_ID = "haze4k-observable-signal-calibration-decomposition-v1"
PARENT_ROUTE_COMMIT = "72698dfae556a6364d4e6de33735a8d5a99e43bb"
PARENT_RECEIPT = "612e38e1bc6e7b70b07604b494102b35eb420e482cdbc21b9e8784a509c1e7e1"
PARENT_RAW_INVENTORY_ID = "301264df5f2aa5f8ac68d79e5e145edbcf1008cf43ca827cca66768021573a91"
PARENT_ENTRYPOINT_SHA256 = "4d421c4a9123a000e4679837e597275d1faffbd29219a4b63b4bc1286a667812"
ANCHOR_COMMIT = "3b4da35440c8c26a7d1bcaf1daf342e11d9a3898"
CHECKPOINT_SHA256 = "6f42037d57a4e3de3a10ac0ab909d66a3415864a19433c29204a975f4efa4088"
MODEL_SOURCE_SHA256 = "9681defa95c6602d0e0abae05e635771bd65d48a18555198ff00d573242a0005"
MODEL_LAYERS_SHA256 = "ac8a05bd626d9adda16308dedb9466f36d7ff44cfb666f64e7e14ddf8cdf43a4"
TRAIN_ASSIGNMENT_DIGEST = "7b21d3af455475f7bb29198081a2ef2e651cffaac6149fd27741863c765b4efc"
TEST_ASSIGNMENT_DIGEST = "6ca5174470dad2b4eef4ae15c5a13a99d8ae9fc0bc2ea1116b199c4d4bc05582"
TRAIN_SPLIT_SALT = "haze4k-local-error-qualification-v2"
VARIANT_SELECTION_SALT = "haze4k-observable-conditional-utility-risk-feasibility-v1"
SHUFFLE_SALT = "haze4k-frozen-output-risk-interface-localization-v1-shuffle"
PERMUTATION_SALT = "haze4k-frozen-output-risk-interface-localization-v1-permutation"

IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}
TRAIN_INPUT_DIRECTORIES = ("IN", "haze", "hazy")
TRAIN_LABEL_DIRECTORIES = ("GT", "gt")
EXPECTED_TRAIN_SCENES = 750
TRAINING_SCENES = 600
CALIBRATION_SCENES = 150
EXPECTED_TEST_SCENES = 100
VARIANTS_PER_TEST_SCENE = 4
OOF_FOLDS = 5
TILE_SIZE = 32
ACTION_NAMES = ("weaken", "strengthen")
ACTION_SCALES = np.asarray([0.8, 1.2], dtype=np.float32)
POLICY_IDS = (
    "q_weighted",
    "p_deweighted",
    "utility_only",
    "observable_utility_gt_risk",
    "gt_utility_observable_risk",
    "gt_gt",
    "uniform",
    "shuffled",
    "permuted",
)
MAIN_POLICY = "p_deweighted"
FORMAL_FIELDS = (
    "p_deweighted_minus_keep_psnr_db",
    "p_deweighted_minus_q_weighted_psnr_db",
    "p_deweighted_minus_uniform_psnr_db",
    "p_deweighted_minus_shuffled_psnr_db",
)
FORMAL_FAMILY_SIZE = len(FORMAL_FIELDS)
SAFETY_FAMILY_SIZE = 8
ORACLE_FAMILY_SIZE = 2
FRONTIER_RISK_THRESHOLDS = (0.025, 0.05, 0.10, 0.20, 0.50)
RISK_POSITIVE_WEIGHT = 2.0
UTILITY_MARGIN_DB = 0.05
RISK_ACCEPTANCE_MAX = 0.10
PRECISION_TARGET_DB = 0.025
PSNR_HARM_MARGIN_DB = 0.10
SSIM_HARM_MARGIN = 0.005
COLOR_HARM_MARGIN = 1.0 / 255.0
MAX_HARM_PREVALENCE = 0.10
MIN_MATERIAL_SCENE_PREVALENCE = 0.20
MIN_ACTIVE_SCENE_PREVALENCE = 0.20
MIN_ACTION_AREA_FRACTION = 0.01
BOOTSTRAP_RESAMPLES = 100_000
BOOTSTRAP_SEED = 20260727
TOTAL_UNITS = 701
PROBE_ITERATIONS = 8
PARAMETER_COUNT = 8_630_665
EPSILON = 1.0e-12


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def digest_lines(lines: Iterable[str]) -> str:
    return sha256_text("\n".join(sorted(lines)))


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    if not fields:
        raise RuntimeError(f"CSV rows are empty: {path.name}")
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


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
    for name in dict.fromkeys(names):
        candidate = label_dir / name
        if candidate.is_file():
            return candidate
    return None


def enumerate_train_groups(root: Path) -> tuple[
    dict[str, list[tuple[Path, Path]]], list[str], list[str], str,
]:
    input_dirs = [root / name for name in TRAIN_INPUT_DIRECTORIES if (root / name).is_dir()]
    label_dirs = [root / name for name in TRAIN_LABEL_DIRECTORIES if (root / name).is_dir()]
    if len(input_dirs) != 1 or len(label_dirs) != 1:
        raise RuntimeError("Haze4K train input/label directories are ambiguous")
    hazy_paths = image_files(input_dirs[0])
    clear_paths = image_files(label_dirs[0])
    if len(hazy_paths) != 3000 or len(clear_paths) != 3000:
        raise RuntimeError("Haze4K train file census changed")
    clear_digest_cache: dict[Path, str] = {}
    groups: dict[str, list[tuple[Path, Path]]] = defaultdict(list)
    for hazy_path in hazy_paths:
        clear_path = selected_label(hazy_path.name, label_dirs[0])
        if clear_path is None:
            raise RuntimeError(f"missing train target for {hazy_path.name}")
        digest = clear_digest_cache.get(clear_path)
        if digest is None:
            hazy, clear = image_array(hazy_path), image_array(clear_path)
            if hazy.shape != clear.shape:
                raise RuntimeError(f"train pair dimensions differ for {hazy_path.name}")
            digest = canonical_rgb_digest(clear)
            clear_digest_cache[clear_path] = digest
        groups[digest].append((hazy_path, clear_path))
    if len(groups) != EXPECTED_TRAIN_SCENES \
            or Counter(len(items) for items in groups.values()) != {
                4: EXPECTED_TRAIN_SCENES,
            }:
        raise RuntimeError("Haze4K train canonical scene structure changed")
    ranked = sorted(
        groups,
        key=lambda scene: (sha256_text(f"{TRAIN_SPLIT_SALT}|{scene}"), scene),
    )
    calibration = ranked[:CALIBRATION_SCENES]
    training = ranked[CALIBRATION_SCENES:]
    calibration_set = set(calibration)
    assignment = digest_lines(
        f"{scene},{'internal_development' if scene in calibration_set else 'training'}"
        for scene in sorted(groups)
    )
    return dict(groups), training, calibration, assignment


def enumerate_test_groups(root: Path) -> tuple[
    dict[str, list[tuple[Path, Path]]], str,
]:
    haze_root, clear_root = root / "haze", root / "gt"
    if root.name != "development_screening" or not haze_root.is_dir() \
            or not clear_root.is_dir():
        raise RuntimeError("isolated Haze4K test-development contract changed")
    haze_paths = image_files(haze_root)
    clear_paths = image_files(clear_root)
    if len(haze_paths) != 400 or len(clear_paths) != 400:
        raise RuntimeError("Haze4K test-development census changed")
    groups: dict[str, list[tuple[Path, Path]]] = defaultdict(list)
    assignment_lines = []
    for hazy_path in haze_paths:
        clear_path = clear_root / hazy_path.name
        if not clear_path.is_file():
            raise RuntimeError(f"missing test-development target for {hazy_path.name}")
        hazy, clear = image_array(hazy_path), image_array(clear_path)
        if hazy.shape != clear.shape:
            raise RuntimeError(f"test pair dimensions differ for {hazy_path.name}")
        scene = canonical_rgb_digest(clear)
        groups[scene].append((hazy_path, clear_path))
        assignment_lines.append(f"{hazy_path.name},development_screening,{scene}")
    if len(groups) != EXPECTED_TEST_SCENES \
            or Counter(len(items) for items in groups.values()) != {
                VARIANTS_PER_TEST_SCENE: EXPECTED_TEST_SCENES,
            }:
        raise RuntimeError("Haze4K test-development canonical structure changed")
    return dict(groups), digest_lines(assignment_lines)


def choose_one_variant(scene: str, items: list[tuple[Path, Path]]) -> tuple[Path, Path]:
    if len(items) != 4:
        raise RuntimeError(f"scene {scene[:16]} does not contain four variants")
    return min(
        items,
        key=lambda pair: (
            sha256_text(f"{VARIANT_SELECTION_SALT}|{scene}|{pair[0].name}"),
            pair[0].name,
        ),
    )


def load_official_model(context):
    import torch

    anchor = asset_path(context, "official_anchor_checkout", kind="git_checkout")
    checkpoint = asset_path(context, "official_checkpoint", kind="file")
    model_source = asset_path(context, "model_source", kind="file")
    model_layers = asset_path(context, "model_layers", kind="file")
    expected = {
        "official_checkpoint": CHECKPOINT_SHA256,
        "model_source": MODEL_SOURCE_SHA256,
        "model_layers": MODEL_LAYERS_SHA256,
    }
    if context.assets["official_anchor_checkout"].commit != ANCHOR_COMMIT:
        raise RuntimeError("official anchor commit changed")
    for identifier, identity in expected.items():
        if context.assets[identifier].sha256 != identity:
            raise RuntimeError(f"verified identity changed for {identifier}")
    anchor_text = str(anchor)
    if anchor_text in sys.path:
        sys.path.remove(anchor_text)
    sys.path.insert(0, anchor_text)
    from Dehazing.ITS.models.ConvIR import build_net

    model_module = sys.modules[build_net.__module__]
    layer_module = sys.modules.get("Dehazing.ITS.models.layers")
    if Path(model_module.__file__).resolve() != model_source.resolve():
        raise RuntimeError("official model import resolved to a different file")
    if layer_module is None or Path(layer_module.__file__).resolve() != model_layers.resolve():
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


def official_prediction(torch, model, hazy: np.ndarray, device: str) -> np.ndarray:
    import torch.nn.functional as functional

    tensor = torch.from_numpy(hazy.transpose(2, 0, 1).copy()).unsqueeze(0).to(device)
    height, width = hazy.shape[:2]
    padded = functional.pad(
        tensor, (0, (-width) % TILE_SIZE, 0, (-height) % TILE_SIZE), mode="reflect",
    )
    with torch.inference_mode():
        outputs = model(padded)
    if not isinstance(outputs, list) or len(outputs) != 3:
        raise RuntimeError("official output contract changed")
    prediction = outputs[2][:, :, :height, :width].clamp(0.0, 1.0)
    if not bool(torch.isfinite(prediction).all().item()):
        raise RuntimeError("official prediction is non-finite")
    return prediction.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float32)


def deweight_probability(q_value: np.ndarray) -> np.ndarray:
    q_value = np.asarray(q_value, dtype=np.float64)
    if not np.isfinite(q_value).all() or np.any(q_value < 0.0) or np.any(q_value > 1.0):
        raise RuntimeError("weighted harm score is outside [0, 1]")
    return np.clip(q_value / np.maximum(2.0 - q_value, EPSILON), 0.0, 1.0)


def choose_actions(
    utility: np.ndarray,
    risk: np.ndarray,
    *,
    utility_threshold: float = UTILITY_MARGIN_DB,
    risk_threshold: float = RISK_ACCEPTANCE_MAX,
) -> np.ndarray:
    if utility.shape != risk.shape or utility.ndim != 3 or utility.shape[0] != 2:
        raise RuntimeError("utility/risk action grid contract changed")
    eligible = (utility > utility_threshold) & (risk <= risk_threshold)
    scores = np.where(eligible, utility, -np.inf)
    best = np.argmax(scores, axis=0)
    best_score = np.max(scores, axis=0)
    actions = np.where(np.isfinite(best_score), best + 1, 0)
    return actions.astype(np.int8)


def choose_uniform_action(
    utility: np.ndarray, risk: np.ndarray, areas: np.ndarray,
) -> np.ndarray:
    weights = areas.astype(np.float64)
    weights /= float(np.sum(weights))
    mean_utility = np.sum(utility.astype(np.float64) * weights[None, :, :], axis=(1, 2))
    mean_risk = np.sum(risk.astype(np.float64) * weights[None, :, :], axis=(1, 2))
    eligible = (mean_utility > UTILITY_MARGIN_DB) & (mean_risk <= RISK_ACCEPTANCE_MAX)
    scores = np.where(eligible, mean_utility, -np.inf)
    action = int(np.argmax(scores) + 1) if np.isfinite(np.max(scores)) else 0
    return np.full(areas.shape, action, dtype=np.int8)


def policy_actions(raw: dict[str, np.ndarray], scene_key: str) -> dict[str, np.ndarray]:
    utility = raw["raw_utility"].astype(np.float64)
    q_value = raw["raw_risk"].astype(np.float64)
    p_value = deweight_probability(q_value)
    target_utility = raw["target_utility"].astype(np.float64)
    target_harm = raw["target_harm"].astype(np.float64)
    areas = raw["areas"].astype(np.float64)
    zero_risk = np.zeros_like(p_value)
    gt_risk = target_harm
    actions = {
        "q_weighted": choose_actions(utility, q_value),
        "p_deweighted": choose_actions(utility, p_value),
        "utility_only": choose_actions(utility, zero_risk),
        "observable_utility_gt_risk": choose_actions(utility, gt_risk),
        "gt_utility_observable_risk": choose_actions(target_utility, p_value),
        "gt_gt": choose_actions(target_utility, gt_risk),
        "uniform": choose_uniform_action(utility, p_value, areas),
    }
    main_actions = actions[MAIN_POLICY]
    actions["shuffled"] = np.random.default_rng(
        int(sha256_text(f"{SHUFFLE_SALT}|{scene_key}")[:16], 16),
    ).permutation(main_actions.reshape(-1)).reshape(main_actions.shape).astype(np.int8)
    permutation = np.random.default_rng(
        int(sha256_text(f"{PERMUTATION_SALT}|{scene_key}")[:16], 16),
    ).permutation(areas.size)
    permuted_utility = utility.reshape(2, -1)[:, permutation].reshape(utility.shape)
    permuted_risk = p_value.reshape(2, -1)[:, permutation].reshape(p_value.shape)
    actions["permuted"] = choose_actions(permuted_utility, permuted_risk)
    if set(actions) != set(POLICY_IDS):
        raise RuntimeError("policy action set is incomplete")
    return actions


def action_scale(action: int) -> float:
    return 1.0 if action == 0 else float(ACTION_SCALES[action - 1])


def apply_spatial(hazy: np.ndarray, prediction: np.ndarray, actions: np.ndarray) -> np.ndarray:
    height, width = hazy.shape[:2]
    expected = (math.ceil(height / TILE_SIZE), math.ceil(width / TILE_SIZE))
    if prediction.shape != hazy.shape or actions.shape != expected:
        raise RuntimeError("image prediction/action grid contract changed")
    output = prediction.copy()
    for row in range(actions.shape[0]):
        for column in range(actions.shape[1]):
            action = int(actions[row, column])
            if action:
                y0, y1 = row * TILE_SIZE, min(height, (row + 1) * TILE_SIZE)
                x0, x1 = column * TILE_SIZE, min(width, (column + 1) * TILE_SIZE)
                output[y0:y1, x0:x1] = np.clip(
                    hazy[y0:y1, x0:x1]
                    + action_scale(action)
                    * (prediction[y0:y1, x0:x1] - hazy[y0:y1, x0:x1]),
                    0.0,
                    1.0,
                )
    return output.astype(np.float32)


def mse(value: np.ndarray, target: np.ndarray) -> float:
    result = float(np.mean((value.astype(np.float64) - target.astype(np.float64)) ** 2))
    if not math.isfinite(result):
        raise RuntimeError("non-finite MSE")
    return result


def psnr(value: np.ndarray, target: np.ndarray) -> float:
    return -10.0 * math.log10(max(mse(value, target), EPSILON))


def color_bias(value: np.ndarray, target: np.ndarray) -> float:
    channel_bias = np.mean(
        value.astype(np.float64) - target.astype(np.float64), axis=(0, 1),
    )
    return float(np.mean(np.abs(channel_bias)))


def rgb_ssim(torch, values: list[np.ndarray], target: np.ndarray, device: str) -> list[float]:
    import torch.nn.functional as functional

    results: list[float] = []
    for start in range(0, len(values), 3):
        batch = values[start:start + 3]
        stack = np.stack(batch).transpose(0, 3, 1, 2).copy()
        reference = np.repeat(
            target[None, ...], len(batch), axis=0,
        ).transpose(0, 3, 1, 2).copy()
        x = torch.from_numpy(stack).to(device=device, dtype=torch.float32)
        y = torch.from_numpy(reference).to(device=device, dtype=torch.float32)

        def local_mean(item):
            return functional.avg_pool2d(
                functional.pad(item, (5, 5, 5, 5), mode="reflect"),
                11,
                stride=1,
            )

        mu_x, mu_y = local_mean(x), local_mean(y)
        variance_x = torch.clamp(local_mean(x * x) - mu_x * mu_x, min=0.0)
        variance_y = torch.clamp(local_mean(y * y) - mu_y * mu_y, min=0.0)
        covariance = local_mean(x * y) - mu_x * mu_y
        c1, c2 = 0.01**2, 0.03**2
        scores = ((2.0 * mu_x * mu_y + c1) * (2.0 * covariance + c2)) / (
            (mu_x * mu_x + mu_y * mu_y + c1)
            * (variance_x + variance_y + c2)
        )
        observed = scores.mean(dim=(1, 2, 3)).detach().cpu().numpy().astype(np.float64)
        if not np.isfinite(observed).all():
            raise RuntimeError("non-finite RGB SSIM")
        results.extend(float(item) for item in observed)
    return results


def weighted_mean(value: np.ndarray, weights: np.ndarray) -> float:
    numerator = float(np.sum(value.astype(np.float64) * weights.astype(np.float64)))
    denominator = float(np.sum(weights))
    if denominator <= 0.0:
        raise RuntimeError("weighted mean denominator is zero")
    return numerator / denominator


def selected_tile_quantities(
    actions: np.ndarray,
    target_utility: np.ndarray,
    target_harm: np.ndarray,
    areas: np.ndarray,
) -> dict[str, float]:
    selected_area = 0.0
    harmed_area = 0.0
    realized_utility = 0.0
    total_area = float(np.sum(areas))
    for action in (1, 2):
        mask = actions == action
        selected_area += float(np.sum(areas[mask]))
        harmed_area += float(np.sum(areas[mask] * target_harm[action - 1][mask]))
        realized_utility += float(
            np.sum(areas[mask] * target_utility[action - 1][mask])
        )
    return {
        "selected_area": selected_area,
        "harmed_selected_area": harmed_area,
        "total_area": total_area,
        "coverage": selected_area / total_area,
        "tile_utility_db": realized_utility / total_area,
        "selected_harm_fraction": (
            harmed_area / selected_area if selected_area > 0.0 else math.nan
        ),
    }


def load_raw_prediction(path: Path) -> dict[str, np.ndarray]:
    required = {
        "raw_utility", "raw_risk", "target_utility", "target_harm", "areas",
        "utility_correction", "risk_correction", "actions_p00",
    }
    with np.load(path, allow_pickle=False) as source:
        if not required <= set(source.files):
            raise RuntimeError(f"raw prediction contract changed: {path.name}")
        result = {name: source[name].copy() for name in required}
    arrays = [
        result[name] for name in (
            "raw_utility", "raw_risk", "target_utility", "target_harm", "areas",
        )
    ]
    if not all(np.isfinite(value).all() for value in arrays):
        raise RuntimeError(f"non-finite raw prediction: {path.name}")
    return result


def evaluate_variant(
    torch,
    raw: dict[str, np.ndarray],
    hazy: np.ndarray,
    clear: np.ndarray,
    prediction: np.ndarray,
    scene_key: str,
    device: str,
) -> tuple[dict[str, dict[str, float]], dict[str, bool], list[dict[str, Any]]]:
    actions = policy_actions(raw, scene_key)
    q_reproduction = np.array_equal(actions["q_weighted"], raw["actions_p00"])
    outputs = [prediction]
    for policy_id in POLICY_IDS:
        outputs.append(apply_spatial(hazy, prediction, actions[policy_id]))
    ssim_values = rgb_ssim(torch, outputs, clear, device)
    keep_psnr = psnr(prediction, clear)
    keep_ssim = ssim_values[0]
    keep_color = color_bias(prediction, clear)
    metrics: dict[str, dict[str, float]] = {}
    for index, policy_id in enumerate(POLICY_IDS, start=1):
        output = outputs[index]
        tile = selected_tile_quantities(
            actions[policy_id],
            raw["target_utility"],
            raw["target_harm"],
            raw["areas"],
        )
        gain = psnr(output, clear) - keep_psnr
        ssim_gain = ssim_values[index] - keep_ssim
        color_delta = color_bias(output, clear) - keep_color
        metrics[policy_id] = {
            **tile,
            "psnr_gain_db": gain,
            "ssim_gain": ssim_gain,
            "color_bias_delta": color_delta,
            "psnr_harm": float(gain <= -PSNR_HARM_MARGIN_DB),
            "ssim_harm": float(ssim_gain <= -SSIM_HARM_MARGIN),
            "color_harm": float(color_delta >= COLOR_HARM_MARGIN),
            "active": float(tile["selected_area"] > 0.0),
            "material": float(gain > UTILITY_MARGIN_DB),
        }
    frontier_rows = []
    p_value = deweight_probability(raw["raw_risk"])
    for threshold in FRONTIER_RISK_THRESHOLDS:
        frontier_actions = choose_actions(
            raw["raw_utility"], p_value, risk_threshold=threshold,
        )
        frontier_rows.append({
            "risk_threshold": threshold,
            **selected_tile_quantities(
                frontier_actions,
                raw["target_utility"],
                raw["target_harm"],
                raw["areas"],
            ),
        })
    checks = {
        "q_action_reproduction": q_reproduction,
        "finite_metrics": all(
            math.isfinite(value)
            for policy in metrics.values()
            for key, value in policy.items()
            if key != "selected_harm_fraction" or not math.isnan(value)
        ),
    }
    return metrics, checks, frontier_rows


def mean_nested_variant_results(
    variants: list[dict[str, dict[str, float]]],
) -> dict[str, dict[str, float]]:
    if len(variants) != VARIANTS_PER_TEST_SCENE:
        raise RuntimeError("test scene does not have four completed variants")
    result: dict[str, dict[str, float]] = {}
    for policy_id in POLICY_IDS:
        selected_area = sum(item[policy_id]["selected_area"] for item in variants)
        harmed_area = sum(item[policy_id]["harmed_selected_area"] for item in variants)
        total_area = sum(item[policy_id]["total_area"] for item in variants)
        fields = (
            "psnr_gain_db", "ssim_gain", "color_bias_delta", "tile_utility_db",
        )
        result[policy_id] = {
            field: float(np.mean([item[policy_id][field] for item in variants]))
            for field in fields
        }
        result[policy_id].update({
            "selected_area": selected_area,
            "harmed_selected_area": harmed_area,
            "total_area": total_area,
            "coverage": selected_area / total_area,
            "selected_harm_fraction": (
                harmed_area / selected_area if selected_area > 0.0 else math.nan
            ),
        })
        gain = result[policy_id]["psnr_gain_db"]
        ssim_gain = result[policy_id]["ssim_gain"]
        color_delta = result[policy_id]["color_bias_delta"]
        result[policy_id].update({
            "psnr_harm": float(gain <= -PSNR_HARM_MARGIN_DB),
            "ssim_harm": float(ssim_gain <= -SSIM_HARM_MARGIN),
            "color_harm": float(color_delta >= COLOR_HARM_MARGIN),
            "active": float(selected_area > 0.0),
            "material": float(gain > UTILITY_MARGIN_DB),
        })
    return result


def compact_parent_inventory(root: Path) -> dict[str, Any]:
    workload = root / "workload"
    patterns = {
        "head_checkpoint": workload / "head_checkpoints",
        "raw_prediction": workload / "raw_predictions",
        "scene_cache": workload / "scene_cache",
        "scene_replay": workload / "test_development_replay",
    }
    suffixes = {
        "head_checkpoint": ".pt",
        "raw_prediction": ".npz",
        "scene_cache": ".npz",
        "scene_replay": ".json",
    }
    items = []
    classes = {}
    for artifact_class, directory in patterns.items():
        if not directory.is_dir():
            raise RuntimeError(f"parent raw directory is missing: {directory}")
        selected = sorted(
            path for path in directory.rglob(f"*{suffixes[artifact_class]}")
            if path.is_file() and not path.is_symlink()
        )
        class_items = []
        for path in selected:
            item = {
                "artifact_class": artifact_class,
                "relative_path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            items.append(item)
            class_items.append(item)
        classes[artifact_class] = {
            "file_count": len(class_items),
            "total_bytes": sum(item["bytes"] for item in class_items),
            "inventory_sha256": digest_lines(
                f"{item['relative_path']}|{item['sha256']}|{item['bytes']}"
                for item in class_items
            ),
        }
    return {
        "file_count": len(items),
        "total_bytes": sum(item["bytes"] for item in items),
        "inventory_sha256": digest_lines(
            f"{item['artifact_class']}|{item['relative_path']}|{item['sha256']}|{item['bytes']}"
            for item in items
        ),
        "classes": classes,
    }


def bonferroni_z(family_size: int, *, confidence: float = 0.95) -> float:
    alpha = 1.0 - confidence
    return NormalDist().inv_cdf(1.0 - alpha / (2.0 * family_size))


def wilson_interval(successes: int, total: int, family_size: int) -> dict[str, float | int]:
    if total <= 0 or not 0 <= successes <= total:
        raise RuntimeError("Wilson count contract is invalid")
    z = bonferroni_z(family_size)
    estimate = successes / total
    denominator = 1.0 + z * z / total
    center = (estimate + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(
        estimate * (1.0 - estimate) / total + z * z / (4.0 * total * total)
    ) / denominator
    return {
        "estimate": estimate,
        "lower": max(0.0, center - radius),
        "upper": min(1.0, center + radius),
        "successes": successes,
        "total": total,
        "critical_value": z,
        "family_size": family_size,
    }


def bootstrap_mean_interval(
    values: np.ndarray,
    folds: np.ndarray | None,
    *,
    family_size: int,
    seed: int,
) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise RuntimeError("bootstrap values are invalid")
    rng = np.random.default_rng(seed)
    samples = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    chunk_size = 2000
    if folds is None:
        groups = [np.arange(len(values))]
    else:
        folds = np.asarray(folds)
        groups = [np.flatnonzero(folds == fold) for fold in sorted(set(folds.tolist()))]
        if len(groups) != OOF_FOLDS or any(len(group) == 0 for group in groups):
            raise RuntimeError("OOF bootstrap fold structure changed")
    offset = 0
    while offset < BOOTSTRAP_RESAMPLES:
        count = min(chunk_size, BOOTSTRAP_RESAMPLES - offset)
        group_means = []
        for group in groups:
            indices = rng.integers(0, len(group), size=(count, len(group)))
            group_means.append(np.mean(values[group][indices], axis=1))
        samples[offset:offset + count] = np.mean(np.stack(group_means), axis=0)
        offset += count
    tail = (1.0 - 0.95) / (2.0 * family_size)
    estimate = float(np.mean([np.mean(values[group]) for group in groups]))
    lower, upper = np.quantile(samples, [tail, 1.0 - tail])
    return {
        "estimate": estimate,
        "lower": float(lower),
        "upper": float(upper),
        "max_half_width": max(estimate - float(lower), float(upper) - estimate),
        "scene_count": len(values),
        "resamples": BOOTSTRAP_RESAMPLES,
        "seed": seed,
        "family_size": family_size,
        "critical_value": bonferroni_z(family_size),
    }


def bootstrap_ratio_interval(
    numerators: np.ndarray,
    denominators: np.ndarray,
    folds: np.ndarray | None,
    *,
    family_size: int,
    seed: int,
) -> dict[str, float | int | None]:
    numerators = np.asarray(numerators, dtype=np.float64)
    denominators = np.asarray(denominators, dtype=np.float64)
    if numerators.shape != denominators.shape or numerators.ndim != 1 \
            or np.any(numerators < 0.0) or np.any(denominators < numerators):
        raise RuntimeError("bootstrap ratio inputs are invalid")
    total_denominator = float(np.sum(denominators))
    if total_denominator <= 0.0:
        return {
            "estimate": None,
            "lower": None,
            "upper": None,
            "scene_count": len(numerators),
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": seed,
            "family_size": family_size,
        }
    if folds is None:
        groups = [np.arange(len(numerators))]
    else:
        groups = [np.flatnonzero(folds == fold) for fold in sorted(set(folds.tolist()))]
    rng = np.random.default_rng(seed)
    samples = []
    chunk_size = 2000
    completed = 0
    while completed < BOOTSTRAP_RESAMPLES:
        count = min(chunk_size, BOOTSTRAP_RESAMPLES - completed)
        numerator_parts = []
        denominator_parts = []
        for group in groups:
            indices = rng.integers(0, len(group), size=(count, len(group)))
            numerator_parts.append(np.sum(numerators[group][indices], axis=1))
            denominator_parts.append(np.sum(denominators[group][indices], axis=1))
        numerator_sum = np.sum(np.stack(numerator_parts), axis=0)
        denominator_sum = np.sum(np.stack(denominator_parts), axis=0)
        valid = denominator_sum > 0.0
        samples.extend((numerator_sum[valid] / denominator_sum[valid]).tolist())
        completed += count
    if not samples:
        return {
            "estimate": None,
            "lower": None,
            "upper": None,
            "scene_count": len(numerators),
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": seed,
            "family_size": family_size,
        }
    tail = (1.0 - 0.95) / (2.0 * family_size)
    lower, upper = np.quantile(np.asarray(samples), [tail, 1.0 - tail])
    return {
        "estimate": float(np.sum(numerators) / total_denominator),
        "lower": float(lower),
        "upper": float(upper),
        "scene_count": len(numerators),
        "resamples": len(samples),
        "seed": seed,
        "family_size": family_size,
    }


def calibration_scene_metrics(raw: dict[str, np.ndarray]) -> dict[str, float]:
    q_value = raw["raw_risk"].astype(np.float64)
    p_value = deweight_probability(q_value)
    harm = raw["target_harm"].astype(np.float64)
    areas = np.broadcast_to(raw["areas"][None, :, :], harm.shape).astype(np.float64)
    return {
        "q_brier": weighted_mean((q_value - harm) ** 2, areas),
        "p_brier": weighted_mean((p_value - harm) ** 2, areas),
        "q_mean": weighted_mean(q_value, areas),
        "p_mean": weighted_mean(p_value, areas),
        "harm_mean": weighted_mean(harm, areas),
    }


def flatten_unit(
    population: str,
    scene: str,
    fold: int | None,
    policies: dict[str, dict[str, float]],
    calibration: dict[str, float],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "population": population,
        "scene": scene,
        "fold": fold,
        **calibration,
    }
    for policy_id, metrics in policies.items():
        for key, value in metrics.items():
            result[f"{policy_id}_{key}"] = value
    result["p_deweighted_minus_q_weighted_psnr_db"] = (
        result["p_deweighted_psnr_gain_db"] - result["q_weighted_psnr_gain_db"]
    )
    result["p_deweighted_minus_uniform_psnr_db"] = (
        result["p_deweighted_psnr_gain_db"] - result["uniform_psnr_gain_db"]
    )
    result["p_deweighted_minus_shuffled_psnr_db"] = (
        result["p_deweighted_psnr_gain_db"] - result["shuffled_psnr_gain_db"]
    )
    result["p_deweighted_minus_keep_psnr_db"] = result["p_deweighted_psnr_gain_db"]
    return result


def aggregate_frontier(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    keys = sorted({(row["population"], row["risk_threshold"]) for row in rows})
    for population, threshold in keys:
        selected = [
            row for row in rows
            if row["population"] == population and row["risk_threshold"] == threshold
        ]
        selected_area = sum(row["selected_area"] for row in selected)
        harmed_area = sum(row["harmed_selected_area"] for row in selected)
        total_area = sum(row["total_area"] for row in selected)
        result.append({
            "population": population,
            "risk_threshold": threshold,
            "utility_threshold_db": UTILITY_MARGIN_DB,
            "scene_count": len(selected),
            "coverage": selected_area / total_area,
            "selected_harm_fraction": (
                harmed_area / selected_area if selected_area > 0.0 else None
            ),
            "mean_tile_utility_db": float(np.mean([
                row["tile_utility_db"] for row in selected
            ])),
        })
    return result


def parent_identity_checks(context, observed_inventory: dict[str, Any]) -> dict[str, bool]:
    closeout = read_json(asset_path(context, "parent_closeout", kind="file"))
    conclusion = read_json(asset_path(context, "parent_conclusion", kind="file"))
    summary = read_json(asset_path(context, "parent_summary", kind="file"))
    archived_inventory = read_json(
        asset_path(context, "parent_raw_inventory", kind="file"),
    )
    experiment_spec = read_json(
        asset_path(context, "parent_experiment_spec", kind="file"),
    )
    fixed_factors = experiment_spec["operations"][
        "HAZE4K_OBSERVABLE_SIGNAL_CALIBRATION_DECOMPOSE"
    ]["scientific_contract"]["intervention"]["fixed_factors"]
    return {
        "parent_route_commit": closeout.get("route_commit") == PARENT_ROUTE_COMMIT,
        "parent_terminal_tuple": (
            closeout.get("state") == "COMPLETED_GATE_FAIL"
            and closeout.get("decision")
            == "HAZE4K_OBSERVABLE_SIGNAL_CALIBRATION_DECOMPOSITION_FAIL"
            and closeout.get("authorizes")
            == "OBSERVABLE_REPRESENTATION_OR_LEARNER_CONTRACT_AUTHORING_ONLY"
        ),
        "parent_receipt_reference": PARENT_RECEIPT == (
            "612e38e1bc6e7b70b07604b494102b35eb420e482cdbc21b9e8784a509c1e7e1"
        ),
        "parent_conclusion": conclusion.get("route_id") == PARENT_ROUTE_ID,
        "parent_summary": summary.get("marker")
        == "HAZE4K_OBSERVABLE_SIGNAL_CALIBRATION_DECOMPOSITION_COMPLETE",
        "parent_entrypoint_identity": any(
            item.get("id") == "observable_entrypoint"
            and item.get("sha256") == PARENT_ENTRYPOINT_SHA256
            for item in closeout.get("verified_assets", [])
        ),
        "positive_weight_frozen": any(
            "positive weight 2.0" in item for item in fixed_factors
        ),
        "risk_threshold_frozen": any(
            "risk acceptance remains 0.10" in item for item in fixed_factors
        ),
        "archived_raw_inventory_id": archived_inventory.get("inventory_sha256")
        == PARENT_RAW_INVENTORY_ID,
        "summary_raw_inventory_id": summary.get(
            "raw_artifact_inventory", {},
        ).get("inventory_sha256") == PARENT_RAW_INVENTORY_ID,
        "observed_raw_inventory_id": observed_inventory.get("inventory_sha256")
        == PARENT_RAW_INVENTORY_ID,
        "raw_file_count": observed_inventory.get("file_count") == 1856,
        "raw_prediction_count": observed_inventory.get("classes", {}).get(
            "raw_prediction", {},
        ).get("file_count") == 1000,
        "scene_cache_count": observed_inventory.get("classes", {}).get(
            "scene_cache", {},
        ).get("file_count") == 750,
    }


def summarize_results(
    context,
    oof_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    frontier_rows: list[dict[str, Any]],
    identity_checks: dict[str, bool],
    role_checks: dict[str, bool],
    observed_inventory: dict[str, Any],
) -> tuple[dict[str, str], dict[str, Any], dict[str, Path]]:
    folds = np.asarray([int(row["fold"]) for row in oof_rows])
    formal_intervals = {}
    for index, field in enumerate(FORMAL_FIELDS):
        formal_intervals[field] = bootstrap_mean_interval(
            np.asarray([float(row[field]) for row in oof_rows]),
            folds,
            family_size=FORMAL_FAMILY_SIZE,
            seed=BOOTSTRAP_SEED + index,
        )
    precision_half_width = max(
        item["max_half_width"] for item in formal_intervals.values()
    )
    precision_outcome = "met" if precision_half_width <= PRECISION_TARGET_DB else "unmet"

    main_gain = formal_intervals["p_deweighted_minus_keep_psnr_db"]
    semantic_gain = formal_intervals["p_deweighted_minus_q_weighted_psnr_db"]
    uniform_gain = formal_intervals["p_deweighted_minus_uniform_psnr_db"]
    shuffled_gain = formal_intervals["p_deweighted_minus_shuffled_psnr_db"]
    material = wilson_interval(
        sum(int(row["p_deweighted_material"]) for row in oof_rows),
        len(oof_rows),
        FORMAL_FAMILY_SIZE,
    )
    actionable_favorable = (
        main_gain["lower"] > UTILITY_MARGIN_DB
        and semantic_gain["lower"] > 0.0
        and uniform_gain["lower"] > 0.0
        and shuffled_gain["lower"] > 0.0
        and material["lower"] > MIN_MATERIAL_SCENE_PREVALENCE
    )
    actionable_unfavorable = (
        main_gain["upper"] <= UTILITY_MARGIN_DB
        or semantic_gain["upper"] <= 0.0
        or uniform_gain["upper"] <= 0.0
        or shuffled_gain["upper"] <= 0.0
        or material["upper"] <= MIN_MATERIAL_SCENE_PREVALENCE
    )
    actionable_outcome = (
        "favorable" if actionable_favorable
        else "unfavorable" if actionable_unfavorable
        else "indeterminate"
    )

    coverage_interval = bootstrap_mean_interval(
        np.asarray([float(row["p_deweighted_coverage"]) for row in oof_rows]),
        folds,
        family_size=2,
        seed=BOOTSTRAP_SEED + 20,
    )
    active_interval = wilson_interval(
        sum(int(row["p_deweighted_active"]) for row in oof_rows),
        len(oof_rows),
        2,
    )
    activation_favorable = (
        coverage_interval["lower"] >= MIN_ACTION_AREA_FRACTION
        and active_interval["lower"] >= MIN_ACTIVE_SCENE_PREVALENCE
    )
    activation_unfavorable = (
        coverage_interval["upper"] < MIN_ACTION_AREA_FRACTION
        or active_interval["upper"] < MIN_ACTIVE_SCENE_PREVALENCE
    )
    activation_outcome = (
        "favorable" if activation_favorable
        else "unfavorable" if activation_unfavorable
        else "indeterminate"
    )

    safety = {}
    safety_rows = []
    for population, rows, population_folds, seed_offset in (
        ("haze4k_train_oof", oof_rows, folds, 30),
        ("haze4k_test_development_stress", test_rows, None, 40),
    ):
        risk = bootstrap_ratio_interval(
            np.asarray([row["p_deweighted_harmed_selected_area"] for row in rows]),
            np.asarray([row["p_deweighted_selected_area"] for row in rows]),
            population_folds,
            family_size=SAFETY_FAMILY_SIZE,
            seed=BOOTSTRAP_SEED + seed_offset,
        )
        safety[population] = {"selected_area_harm": risk}
        safety_rows.append({
            "population": population,
            "metric": "selected_area_harm",
            **risk,
        })
        for metric in ("psnr", "ssim", "color"):
            interval = wilson_interval(
                sum(int(row[f"p_deweighted_{metric}_harm"]) for row in rows),
                len(rows),
                SAFETY_FAMILY_SIZE,
            )
            safety[population][f"{metric}_harm"] = interval
            safety_rows.append({
                "population": population,
                "metric": f"{metric}_harm",
                **interval,
            })
    safety_intervals = [
        interval for population in safety.values() for interval in population.values()
    ]
    safety_safe = all(
        item.get("upper") is not None and item["upper"] <= MAX_HARM_PREVALENCE
        for item in safety_intervals
    )
    safety_unsafe = any(
        item.get("lower") is not None and item["lower"] > MAX_HARM_PREVALENCE
        for item in safety_intervals
    )
    safety_outcome = "safe" if safety_safe else "unsafe" if safety_unsafe else "indeterminate"

    oracle_interval = bootstrap_mean_interval(
        np.asarray([row["gt_gt_psnr_gain_db"] for row in oof_rows]),
        folds,
        family_size=ORACLE_FAMILY_SIZE,
        seed=BOOTSTRAP_SEED + 50,
    )
    oracle_material = wilson_interval(
        sum(int(row["gt_gt_material"]) for row in oof_rows),
        len(oof_rows),
        ORACLE_FAMILY_SIZE,
    )
    oracle_outcome = (
        "favorable"
        if oracle_interval["lower"] > UTILITY_MARGIN_DB
        and oracle_material["lower"] > MIN_MATERIAL_SCENE_PREVALENCE
        else "unfavorable"
        if oracle_interval["upper"] <= UTILITY_MARGIN_DB
        or oracle_material["upper"] <= MIN_MATERIAL_SCENE_PREVALENCE
        else "indeterminate"
    )

    q_brier_interval = bootstrap_mean_interval(
        np.asarray([row["q_brier"] for row in oof_rows]),
        folds,
        family_size=2,
        seed=BOOTSTRAP_SEED + 60,
    )
    p_brier_interval = bootstrap_mean_interval(
        np.asarray([row["p_brier"] for row in oof_rows]),
        folds,
        family_size=2,
        seed=BOOTSTRAP_SEED + 61,
    )
    brier_difference = bootstrap_mean_interval(
        np.asarray([row["q_brier"] - row["p_brier"] for row in oof_rows]),
        folds,
        family_size=1,
        seed=BOOTSTRAP_SEED + 62,
    )
    semantics_outcome = (
        "favorable" if brier_difference["lower"] > 0.0
        else "unfavorable" if brier_difference["upper"] <= 0.0
        else "indeterminate"
    )

    identity_outcome = "pass" if all(identity_checks.values()) else "fail"
    coverage_outcome = "pass" if all(role_checks.values()) else "fail"
    gate_outcomes = {
        "evidence_identity": identity_outcome,
        "scene_role_and_coverage": coverage_outcome,
        "oracle_headroom_control": oracle_outcome,
        "interface_activation": activation_outcome,
        "corrected_actionable_utility": actionable_outcome,
        "selected_area_and_image_safety": safety_outcome,
        "precision": precision_outcome,
        "probability_semantics_diagnostic": semantics_outcome,
    }

    policy_summary_rows = []
    for population, rows in (
        ("haze4k_train_oof", oof_rows),
        ("haze4k_test_development_stress", test_rows),
    ):
        for policy_id in POLICY_IDS:
            selected_area = sum(row[f"{policy_id}_selected_area"] for row in rows)
            harmed_area = sum(row[f"{policy_id}_harmed_selected_area"] for row in rows)
            total_area = sum(row[f"{policy_id}_total_area"] for row in rows)
            policy_summary_rows.append({
                "population": population,
                "policy": policy_id,
                "scene_count": len(rows),
                "mean_psnr_gain_db": float(np.mean([
                    row[f"{policy_id}_psnr_gain_db"] for row in rows
                ])),
                "mean_ssim_gain": float(np.mean([
                    row[f"{policy_id}_ssim_gain"] for row in rows
                ])),
                "mean_color_bias_delta": float(np.mean([
                    row[f"{policy_id}_color_bias_delta"] for row in rows
                ])),
                "action_area_fraction": selected_area / total_area,
                "active_scene_prevalence": float(np.mean([
                    row[f"{policy_id}_active"] for row in rows
                ])),
                "material_scene_prevalence": float(np.mean([
                    row[f"{policy_id}_material"] for row in rows
                ])),
                "selected_area_harm_fraction": (
                    harmed_area / selected_area if selected_area > 0.0 else None
                ),
                "psnr_harm_prevalence": float(np.mean([
                    row[f"{policy_id}_psnr_harm"] for row in rows
                ])),
                "ssim_harm_prevalence": float(np.mean([
                    row[f"{policy_id}_ssim_harm"] for row in rows
                ])),
                "color_harm_prevalence": float(np.mean([
                    row[f"{policy_id}_color_harm"] for row in rows
                ])),
            })

    fold_rows = []
    for fold in range(OOF_FOLDS):
        rows = [row for row in oof_rows if int(row["fold"]) == fold]
        fold_rows.append({
            "fold": fold,
            "scene_count": len(rows),
            "p_deweighted_gain_db": float(np.mean([
                row["p_deweighted_psnr_gain_db"] for row in rows
            ])),
            "q_weighted_gain_db": float(np.mean([
                row["q_weighted_psnr_gain_db"] for row in rows
            ])),
            "semantic_gain_db": float(np.mean([
                row["p_deweighted_minus_q_weighted_psnr_db"] for row in rows
            ])),
            "coverage": float(np.mean([
                row["p_deweighted_coverage"] for row in rows
            ])),
            "material_scene_prevalence": float(np.mean([
                row["p_deweighted_material"] for row in rows
            ])),
            "q_minus_p_brier": float(np.mean([
                row["q_brier"] - row["p_brier"] for row in rows
            ])),
        })

    v00 = np.asarray([row["p_deweighted_psnr_gain_db"] for row in oof_rows])
    v10 = np.asarray([row["gt_utility_observable_risk_psnr_gain_db"] for row in oof_rows])
    v01 = np.asarray([row["observable_utility_gt_risk_psnr_gain_db"] for row in oof_rows])
    v11 = np.asarray([row["gt_gt_psnr_gain_db"] for row in oof_rows])
    utility_shapley = 0.5 * ((v10 - v00) + (v11 - v01))
    risk_shapley = 0.5 * ((v01 - v00) + (v11 - v10))
    regret = {
        "schema_version": 1,
        "independent_unit": "original_clear_scene",
        "semantic_recovery": formal_intervals[
            "p_deweighted_minus_q_weighted_psnr_db"
        ],
        "total_oracle_gap": bootstrap_mean_interval(
            v11 - v00,
            folds,
            family_size=2,
            seed=BOOTSTRAP_SEED + 70,
        ),
        "utility_shapley_regret": bootstrap_mean_interval(
            utility_shapley,
            folds,
            family_size=2,
            seed=BOOTSTRAP_SEED + 71,
        ),
        "risk_shapley_regret": bootstrap_mean_interval(
            risk_shapley,
            folds,
            family_size=2,
            seed=BOOTSTRAP_SEED + 72,
        ),
        "identity": "utility_shapley_regret + risk_shapley_regret equals total_oracle_gap scene by scene",
        "maximum_identity_error": float(
            np.max(np.abs(utility_shapley + risk_shapley - (v11 - v00)))
        ),
    }

    calibration_rows = [
        {
            "population": population,
            "metric": metric,
            "estimate": float(np.mean([row[metric] for row in rows])),
            "scene_count": len(rows),
        }
        for population, rows in (
            ("haze4k_train_oof", oof_rows),
            ("haze4k_test_development_stress", test_rows),
        )
        for metric in ("q_brier", "p_brier", "q_mean", "p_mean", "harm_mean")
    ]
    calibration_summary = {
        "q_brier_interval": q_brier_interval,
        "p_brier_interval": p_brier_interval,
        "q_minus_p_brier_interval": brier_difference,
        "outcome": semantics_outcome,
    }

    summary = {
        "schema_version": 2,
        "route_id": context.route_id,
        "operation_id": context.operation_id,
        "run_id": context.run_id,
        "scope": "development-only frozen-output weighted-BCE probability-semantics localization",
        "independent_unit": "original_clear_scene",
        "data_roles": {
            "haze4k_train_oof": len(oof_rows),
            "haze4k_train_fixed_calibration_nuisance": CALIBRATION_SCENES,
            "haze4k_test_development_stress": len(test_rows),
            "haze4k_test_development_nested_variants": len(test_rows)
            * VARIANTS_PER_TEST_SCENE,
            "candidate_confirmation_touched": False,
            "nh_haze_touched": False,
            "reside_its_ots_touched": False,
        },
        "frozen_interface": {
            "weighted_score": "q=sigmoid(z) from positive-class BCE weight 2.0",
            "candidate_probability": "p=q/(2-q)",
            "candidate_is_predeclared_not_assumed_calibrated": True,
            "utility_margin_db": UTILITY_MARGIN_DB,
            "risk_acceptance_max": RISK_ACCEPTANCE_MAX,
            "precision_target_db": PRECISION_TARGET_DB,
            "anti_abstention_area_lcb": MIN_ACTION_AREA_FRACTION,
            "anti_abstention_active_scene_lcb": MIN_ACTIVE_SCENE_PREVALENCE,
            "action_scales": {"keep": 1.0, "weaken": 0.8, "strengthen": 1.2},
        },
        "formal_intervals": formal_intervals,
        "material_scene_prevalence": material,
        "interface_activation": {
            "coverage_interval": coverage_interval,
            "active_scene_interval": active_interval,
            "outcome": activation_outcome,
        },
        "safety": safety,
        "oracle_control": {
            "gain_interval": oracle_interval,
            "material_scene_prevalence": oracle_material,
            "outcome": oracle_outcome,
        },
        "probability_semantics": calibration_summary,
        "precision": {
            "target_half_width_db": PRECISION_TARGET_DB,
            "observed_max_simultaneous_half_width_db": precision_half_width,
            "outcome": precision_outcome,
            "planning_sd_upper_bound_db": 0.2358753949,
            "candidate_variance_not_reused_from_parent_abstention": True,
        },
        "identity_checks": identity_checks,
        "role_and_coverage_checks": role_checks,
        "gate_outcomes": gate_outcomes,
        "raw_reuse_inventory": observed_inventory,
        "limitations": [
            "All evidence is development_screening; candidate-confirmation, canary, locked-test, NH-Haze, ITS, and OTS data were not read.",
            "The candidate p=q/(2-q) is the exact class-weight inverse under the ideal weighted-BCE population optimum, but this route does not assume the finite learned head is calibrated.",
            "The 600 OOF original clear scenes are the only primary independent units; tiles and haze variants are nested and never increase sample size.",
            "The 100 test-development scenes are secondary stress evidence with four variants averaged within scene.",
            "The archived 150 calibration scenes remain nuisance-only and do not increase primary sample size.",
            "GT-risk and GT-utility controls are privileged and non-deployable; they localize regret only.",
            "A precision miss can block a provisional PASS but cannot hide a decisive utility, activation, or safety FAIL.",
        ],
        "marker": "HAZE4K_FROZEN_OUTPUT_RISK_INTERFACE_LOCALIZATION_COMPLETE",
    }

    gate_summary = {
        "schema_version": 2,
        "gate_outcomes": gate_outcomes,
        "actionable_utility": {
            "favorable": actionable_favorable,
            "unfavorable": actionable_unfavorable,
            "formal_intervals": formal_intervals,
            "material_scene_prevalence": material,
        },
        "activation": summary["interface_activation"],
        "safety": safety,
        "oracle_control": summary["oracle_control"],
        "precision": summary["precision"],
    }

    paths = {
        "summary": output_file(
            context, "haze4k_frozen_output_risk_interface_localization_summary.json",
        ),
        "gate_summary": output_file(
            context, "haze4k_frozen_output_risk_interface_localization_gate_summary.json",
        ),
        "formal_intervals": output_file(
            context, "haze4k_frozen_output_risk_interface_localization_formal_intervals.json",
        ),
        "regret": output_file(
            context, "haze4k_frozen_output_risk_interface_localization_regret_decomposition.json",
        ),
        "policy_summary": output_file(
            context, "haze4k_frozen_output_risk_interface_localization_policy_summary.csv",
        ),
        "fold_stability": output_file(
            context, "haze4k_frozen_output_risk_interface_localization_fold_stability.csv",
        ),
        "frontier": output_file(
            context, "haze4k_frozen_output_risk_interface_localization_risk_frontier.csv",
        ),
        "calibration": output_file(
            context, "haze4k_frozen_output_risk_interface_localization_calibration_diagnostics.csv",
        ),
        "safety": output_file(
            context, "haze4k_frozen_output_risk_interface_localization_replay_safety.csv",
        ),
        "inventory": output_file(
            context, "haze4k_frozen_output_risk_interface_localization_raw_reuse_inventory.json",
        ),
    }
    atomic_json(paths["summary"], summary)
    atomic_json(paths["gate_summary"], gate_summary)
    atomic_json(paths["formal_intervals"], {
        "schema_version": 1,
        "comparison_family": "haze4k_train_oof_four_endpoint_family",
        "independent_unit": "original_clear_scene",
        "intervals": formal_intervals,
        "material_scene_prevalence": material,
    })
    atomic_json(paths["regret"], regret)
    write_csv(paths["policy_summary"], policy_summary_rows)
    write_csv(paths["fold_stability"], fold_rows)
    write_csv(paths["frontier"], aggregate_frontier(frontier_rows))
    write_csv(paths["calibration"], calibration_rows)
    write_csv(paths["safety"], safety_rows)
    atomic_json(paths["inventory"], {
        "schema_version": 1,
        "parent_route_id": PARENT_ROUTE_ID,
        "parent_route_commit": PARENT_ROUTE_COMMIT,
        "parent_receipt": PARENT_RECEIPT,
        "parent_inventory": observed_inventory,
        "unit_output_count": len(oof_rows) + len(test_rows),
        "unit_output_inventory_sha256": digest_lines(
            f"{row['population']}|{row['scene']}|{row['unit_output_sha256']}"
            for row in oof_rows + test_rows
        ),
        "raw_retention_scope": "cloud_only",
    })
    return gate_outcomes, summary, paths


def contract(context_path: Path) -> None:
    context = load_context(context_path, "contract")
    if context.route_id != ROUTE_ID or context.operation_id != OPERATION_ID:
        raise RuntimeError("route or operation identity mismatch")
    prepare_phase_output(context)
    write_contract_progress(
        context,
        completed_iterations=0,
        total_iterations=PROBE_ITERATIONS,
        stage="frozen_output_interface_fixture",
    )
    torch, model = load_official_model(context)
    torch.manual_seed(20260727)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    checks = {
        "route_identity": True,
        "probability_inverse": True,
        "policy_fixture": True,
        "official_model_path": True,
        "image_replay_fixture": True,
        "result_finalizer": True,
    }
    q_fixture = np.asarray([0.01, 0.05, 0.10, 0.25, 0.50, 0.90], dtype=np.float64)
    p_fixture = deweight_probability(q_fixture)
    q_roundtrip = RISK_POSITIVE_WEIGHT * p_fixture / (
        1.0 - p_fixture + RISK_POSITIVE_WEIGHT * p_fixture
    )
    checks["probability_inverse"] = bool(np.allclose(q_fixture, q_roundtrip, atol=1e-12))
    for index in range(PROBE_ITERATIONS):
        hazy_tensor = torch.rand((1, 3, 256, 320), device=context.device)
        with torch.inference_mode():
            outputs = model(hazy_tensor)
        prediction = outputs[2].clamp(0.0, 1.0).squeeze(0).permute(1, 2, 0).cpu().numpy()
        hazy = hazy_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        clear = np.clip(0.8 * prediction + 0.2 * hazy, 0.0, 1.0).astype(np.float32)
        utility = np.stack([
            np.full((8, 10), 0.06, dtype=np.float32),
            np.full((8, 10), 0.08, dtype=np.float32),
        ])
        q_value = np.stack([
            np.full((8, 10), 0.15, dtype=np.float32),
            np.full((8, 10), 0.18, dtype=np.float32),
        ])
        raw = {
            "raw_utility": utility,
            "raw_risk": q_value,
            "target_utility": utility,
            "target_harm": np.zeros_like(utility),
            "areas": np.full((8, 10), TILE_SIZE * TILE_SIZE, dtype=np.float32),
            "actions_p00": choose_actions(utility, q_value),
        }
        actions = policy_actions(raw, f"fixture-{index}")
        replay = apply_spatial(hazy, prediction, actions[MAIN_POLICY])
        scores = rgb_ssim(torch, [prediction, replay], clear, context.device)
        checks["policy_fixture"] = bool(
            checks["policy_fixture"]
            and actions["q_weighted"].sum() == 0
            and actions["p_deweighted"].sum() > 0
        )
        checks["image_replay_fixture"] = checks["image_replay_fixture"] and all(
            math.isfinite(item) for item in scores
        )
        write_contract_progress(
            context,
            completed_iterations=index + 1,
            total_iterations=PROBE_ITERATIONS,
            stage="frozen_output_interface_fixture",
        )
    wall_seconds = time.monotonic() - started
    peak_memory_mib = torch.cuda.max_memory_allocated() / (1024 * 1024)
    checks["official_model_path"] = bool(
        isinstance(outputs, list) and len(outputs) == 3
    )
    write_contract_result(
        context,
        checks=checks,
        engineering={
            "mode": "gpu_synthetic_no_data",
            "device": context.device,
            "fixture": {"batch": 1, "channels": 3, "height": 256, "width": 320},
            "production_path_exercised": True,
            "protected_data_touched": False,
            "scientific_output_created": False,
            "scientific_training_occurred": False,
            "cost": {
                "observed_iterations": PROBE_ITERATIONS,
                "observed_wall_seconds": wall_seconds,
                "observed_peak_memory_mib": peak_memory_mib,
            },
        },
    )


def run(context_path: Path) -> None:
    context = load_context(context_path, "run")
    if context.route_id != ROUTE_ID or context.operation_id != OPERATION_ID \
            or context.total_units != TOTAL_UNITS:
        raise RuntimeError("route, operation, or unit identity mismatch")
    if any(context.protected_data_permissions.values()):
        raise RuntimeError("protected-data permission is forbidden")
    forbidden_assets = {"candidate_confirmation", "nh_haze", "reside_its", "reside_ots"}
    if forbidden_assets & set(context.assets):
        raise RuntimeError("forbidden scientific asset was delivered")
    prepare_phase_output(context)
    if load_completed_unit_ledger(context):
        raise RuntimeError("fresh frozen-output route cannot preload completed units")
    write_workload_progress(
        context, completed_units=0, stage="parent_raw_inventory_verification",
    )

    parent_output = asset_path(context, "parent_raw_output", kind="directory")
    observed_inventory = compact_parent_inventory(parent_output)
    identity_checks = parent_identity_checks(context, observed_inventory)

    train_root = asset_path(context, "haze4k_train", kind="directory")
    test_root = asset_path(context, "haze4k_test_development", kind="directory")
    train_groups, training_scenes, calibration_scenes, train_digest = (
        enumerate_train_groups(train_root)
    )
    test_groups, _test_content_digest = enumerate_test_groups(test_root)
    train_split_summary = read_json(
        asset_path(context, "train_split_summary", kind="file"),
    )
    test_split_summary = read_json(
        asset_path(context, "test_split_summary", kind="file"),
    )
    torch, model = load_official_model(context)
    parent_workload = parent_output / "workload"
    raw_oof_root = parent_workload / "raw_predictions" / "haze4k_train_oof"
    raw_test_root = (
        parent_workload / "raw_predictions" / "haze4k_test_development_stress"
    )
    scene_cache_root = parent_workload / "scene_cache"
    unit_root = output_file(context, "units")
    unit_root.mkdir()

    role_checks = {
        "train_assignment_digest": (
            train_digest == TRAIN_ASSIGNMENT_DIGEST
            and train_split_summary.get("frozen_split", {}).get("assignment_digest")
            == TRAIN_ASSIGNMENT_DIGEST
        ),
        "test_assignment_digest": test_split_summary.get(
            "frozen_split", {},
        ).get("assignment_digest") == TEST_ASSIGNMENT_DIGEST,
        "train_role_counts": (
            len(training_scenes) == TRAINING_SCENES
            and len(calibration_scenes) == CALIBRATION_SCENES
        ),
        "train_role_disjointness": not (set(training_scenes) & set(calibration_scenes)),
        "test_scene_count": len(test_groups) == EXPECTED_TEST_SCENES,
        "four_variants_nested_within_test_scene": all(
            len(items) == VARIANTS_PER_TEST_SCENE for items in test_groups.values()
        ),
        "parent_raw_inventory_complete": all(identity_checks.values()),
        "q_action_reproduction": True,
        "all_scene_metrics_finite": True,
        "completed_unit_ledger_coverage": False,
        "candidate_confirmation_absent": "candidate_confirmation" not in context.assets,
        "nh_haze_absent": "nh_haze" not in context.assets,
        "reside_its_ots_absent": (
            "reside_its" not in context.assets and "reside_ots" not in context.assets
        ),
    }
    oof_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    frontier_rows: list[dict[str, Any]] = []

    for index, scene in enumerate(sorted(training_scenes), start=1):
        raw_matches = sorted(raw_oof_root.glob(f"oof_fold_*_{scene[:24]}.npz"))
        cache_path = scene_cache_root / f"training_{scene[:24]}.npz"
        if len(raw_matches) != 1 or not cache_path.is_file():
            raise RuntimeError(f"parent OOF raw/cache identity missing for {scene[:16]}")
        raw_path = raw_matches[0]
        parts = raw_path.stem.split("_")
        if len(parts) < 4 or parts[0:2] != ["oof", "fold"]:
            raise RuntimeError(f"OOF raw filename contract changed: {raw_path.name}")
        fold = int(parts[2])
        if fold not in range(OOF_FOLDS):
            raise RuntimeError("OOF fold id is invalid")
        raw = load_raw_prediction(raw_path)
        with np.load(cache_path, allow_pickle=False) as source:
            prediction = source["prediction"].copy()
        hazy_path, clear_path = choose_one_variant(scene, train_groups[scene])
        hazy, clear = image_array(hazy_path), image_array(clear_path)
        metrics, checks, variant_frontier = evaluate_variant(
            torch, raw, hazy, clear, prediction, scene, context.device,
        )
        calibration = calibration_scene_metrics(raw)
        row = flatten_unit("haze4k_train_oof", scene, fold, metrics, calibration)
        unit_relpath = f"units/oof_{index:04d}_{scene[:16]}.json"
        unit_path = output_file(context, unit_relpath)
        atomic_json(unit_path, row)
        row["unit_output_sha256"] = sha256_file(unit_path)
        oof_rows.append(row)
        for frontier in variant_frontier:
            frontier_rows.append({
                "population": "haze4k_train_oof", "scene": scene, **frontier,
            })
        role_checks["q_action_reproduction"] = (
            role_checks["q_action_reproduction"] and checks["q_action_reproduction"]
        )
        role_checks["all_scene_metrics_finite"] = (
            role_checks["all_scene_metrics_finite"] and checks["finite_metrics"]
        )
        record_completed_unit(
            context,
            unit_id=f"oof_{index:04d}_{scene[:16]}",
            input_sha256=sha256_text(
                "|".join([
                    "frozen-output-risk-interface-oof-v1",
                    scene,
                    str(fold),
                    sha256_file(raw_path),
                    sha256_file(cache_path),
                    sha256_file(hazy_path),
                    sha256_file(clear_path),
                ])
            ),
            output_relpath=unit_relpath,
        )
        if index == 1 or index % 25 == 0 or index == TRAINING_SCENES:
            write_workload_progress(
                context, completed_units=index, stage="oof_frozen_output_replay",
            )

    for scene_index, scene in enumerate(sorted(test_groups), start=1):
        variant_metrics = []
        variant_calibration = []
        variant_frontiers = []
        input_parts = ["frozen-output-risk-interface-test-v1", scene]
        all_checks = []
        for variant_index, (hazy_path, clear_path) in enumerate(
            sorted(test_groups[scene]), start=1,
        ):
            raw_path = raw_test_root / (
                f"scene_{scene_index:03d}_{scene[:16]}_variant_{variant_index}.npz"
            )
            if not raw_path.is_file():
                raise RuntimeError(f"parent test raw identity missing: {raw_path.name}")
            raw = load_raw_prediction(raw_path)
            hazy, clear = image_array(hazy_path), image_array(clear_path)
            prediction = official_prediction(torch, model, hazy, context.device)
            metrics, checks, variant_frontier = evaluate_variant(
                torch,
                raw,
                hazy,
                clear,
                prediction,
                f"{scene}|{hazy_path.name}",
                context.device,
            )
            variant_metrics.append(metrics)
            variant_calibration.append(calibration_scene_metrics(raw))
            variant_frontiers.append(variant_frontier)
            all_checks.append(checks)
            input_parts.extend([
                hazy_path.name,
                sha256_file(raw_path),
                sha256_file(hazy_path),
                sha256_file(clear_path),
            ])
        policies = mean_nested_variant_results(variant_metrics)
        calibration = {
            field: float(np.mean([item[field] for item in variant_calibration]))
            for field in variant_calibration[0]
        }
        row = flatten_unit(
            "haze4k_test_development_stress", scene, None, policies, calibration,
        )
        unit_relpath = f"units/test_{scene_index:03d}_{scene[:16]}.json"
        unit_path = output_file(context, unit_relpath)
        atomic_json(unit_path, row)
        row["unit_output_sha256"] = sha256_file(unit_path)
        test_rows.append(row)
        for frontier_index, threshold in enumerate(FRONTIER_RISK_THRESHOLDS):
            selected = [items[frontier_index] for items in variant_frontiers]
            selected_area = sum(item["selected_area"] for item in selected)
            harmed_area = sum(item["harmed_selected_area"] for item in selected)
            total_area = sum(item["total_area"] for item in selected)
            frontier_rows.append({
                "population": "haze4k_test_development_stress",
                "scene": scene,
                "risk_threshold": threshold,
                "selected_area": selected_area,
                "harmed_selected_area": harmed_area,
                "total_area": total_area,
                "coverage": selected_area / total_area,
                "selected_harm_fraction": (
                    harmed_area / selected_area if selected_area > 0.0 else math.nan
                ),
                "tile_utility_db": float(np.mean([
                    item["tile_utility_db"] for item in selected
                ])),
            })
        role_checks["q_action_reproduction"] = (
            role_checks["q_action_reproduction"]
            and all(item["q_action_reproduction"] for item in all_checks)
        )
        role_checks["all_scene_metrics_finite"] = (
            role_checks["all_scene_metrics_finite"]
            and all(item["finite_metrics"] for item in all_checks)
        )
        completed_units = TRAINING_SCENES + scene_index
        record_completed_unit(
            context,
            unit_id=f"test_{scene_index:03d}_{scene[:16]}",
            input_sha256=sha256_text("|".join(input_parts)),
            output_relpath=unit_relpath,
        )
        if scene_index == 1 or scene_index % 10 == 0 or scene_index == EXPECTED_TEST_SCENES:
            write_workload_progress(
                context,
                completed_units=completed_units,
                stage="test_development_nested_replay",
            )

    role_checks["completed_unit_ledger_coverage"] = (
        len(load_completed_unit_ledger(context)) == TOTAL_UNITS - 1
    )
    gate_outcomes, summary, evidence_paths = summarize_results(
        context,
        oof_rows,
        test_rows,
        frontier_rows,
        identity_checks,
        role_checks,
        observed_inventory,
    )
    final_relpath = "finalization_unit.json"
    final_path = output_file(context, final_relpath)
    atomic_json(final_path, {
        "schema_version": 1,
        "route_id": context.route_id,
        "operation_id": context.operation_id,
        "completed_scene_units": len(oof_rows) + len(test_rows),
        "evidence_sha256": {
            key: sha256_file(path) for key, path in sorted(evidence_paths.items())
        },
        "gate_outcomes": gate_outcomes,
    })
    record_completed_unit(
        context,
        unit_id="finalization",
        input_sha256=sha256_text(
            "|".join([
                "frozen-output-risk-interface-finalization-v1",
                observed_inventory["inventory_sha256"],
                digest_lines(
                    f"{row['population']}|{row['scene']}|{row['unit_output_sha256']}"
                    for row in oof_rows + test_rows
                ),
            ])
        ),
        output_relpath=final_relpath,
    )
    if len(load_completed_unit_ledger(context)) != TOTAL_UNITS:
        raise RuntimeError("completed-unit ledger is not complete")
    write_workload_progress(
        context, completed_units=TOTAL_UNITS, stage="aggregate_gate_finalization",
    )
    write_gate_result(
        context,
        gate_outcomes=gate_outcomes,
        details={
            "independent_oof_scenes": len(oof_rows),
            "fixed_calibration_nuisance_scenes": CALIBRATION_SCENES,
            "test_development_stress_scenes": len(test_rows),
            "nested_test_variants": len(test_rows) * VARIANTS_PER_TEST_SCENE,
            "completed_unit_ledger_count": TOTAL_UNITS,
            "network_training_occurred": False,
            "parent_raw_output_reused": True,
            "candidate_confirmation_touched": False,
            "formal_family_size": FORMAL_FAMILY_SIZE,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "parent_raw_inventory_sha256": observed_inventory["inventory_sha256"],
            "summary_file": evidence_paths["summary"].name,
            "gate_summary_file": evidence_paths["gate_summary"].name,
            "formal_intervals_file": evidence_paths["formal_intervals"].name,
            "regret_decomposition_file": evidence_paths["regret"].name,
            "policy_summary_file": evidence_paths["policy_summary"].name,
            "fold_stability_file": evidence_paths["fold_stability"].name,
            "risk_frontier_file": evidence_paths["frontier"].name,
            "calibration_diagnostics_file": evidence_paths["calibration"].name,
            "replay_safety_file": evidence_paths["safety"].name,
            "raw_reuse_inventory_file": evidence_paths["inventory"].name,
            "primary_estimate_db": summary["formal_intervals"][
                "p_deweighted_minus_keep_psnr_db"
            ]["estimate"],
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
