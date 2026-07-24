#!/usr/bin/env python3
"""Cross-fit a low-capacity bounded local-action proxy on Haze4K development scenes."""

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

from route_program_api import (
    asset_path, atomic_json, load_context, output_file, prepare_phase_output,
    write_contract_result, write_run_result, write_workload_progress,
)


IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}
EXPECTED_SCENES = 100
EXPECTED_VARIANTS = 400
VARIANTS_PER_SCENE = 4
TILE_SIZE = 32
ACTION_NAMES = ("keep", "weaken", "strengthen")
ACTION_SCALES = np.asarray([1.0, 0.8, 1.2], dtype=np.float32)
OUTER_FOLDS = 5
INNER_FOLDS = 4
ALPHAS = (0.0001, 0.001, 0.01, 0.1, 1.0)
ACTION_THRESHOLDS = (0.0, 0.0025, 0.005, 0.01, 0.02, 2.0)
HANDCRAFTED_DIM = 40
FULL_FEATURE_DIM = 104
INPUT_FEATURE_INDICES = np.asarray(
    [0, 1, 2, 12, 15, 17, 20, 21, 22, 32, 35, 37], dtype=np.int64,
)
SPLIT_ASSIGNMENT_DIGEST = "6ca5174470dad2b4eef4ae15c5a13a99d8ae9fc0bc2ea1116b199c4d4bc05582"
PARENT_CLOSEOUT_SHA256 = "96357cbaaee5aa338fb0f9c9835a975a27e7f048c78a096fd644e6acdd3e383c"
PARENT_SUMMARY_SHA256 = "794ef27733f51f2fa70fab5c94bc661564d7b988ef4247a1b26e33a21b4de7cb"
CHECKPOINT_SHA256 = "6f42037d57a4e3de3a10ac0ab909d66a3415864a19433c29204a975f4efa4088"
MODEL_SOURCE_SHA256 = "3fa227af396464a7f07ac773f92e9cdb746e0fa6ae63adef711c765a02c3d4cd"
MODEL_LAYERS_SHA256 = "ac8a05bd626d9adda16308dedb9466f36d7ff44cfb666f64e7e14ddf8cdf43a4"
PARAMETER_COUNT = 8_630_665
BOOTSTRAP_SEED = 20260726
BOOTSTRAP_RESAMPLES = 20_000
MIN_MEAN_GAIN_DB = 0.10
MIN_SPATIAL_GAIN_DB = 0.10
MIN_SHUFFLE_CONTRAST_DB = 0.10
MATERIAL_SCENE_GAIN_DB = 0.10
MIN_MATERIAL_SCENE_PREVALENCE = 0.20
PSNR_HARM_MARGIN_DB = 0.10
SSIM_HARM_MARGIN = 0.005
COLOR_HARM_MARGIN = 1.0 / 255.0
MAX_HARM_PREVALENCE = 0.10
INNER_MAX_HARM_RATE = 0.05
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


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


def apply_scale(hazy: np.ndarray, prediction: np.ndarray, action: int) -> np.ndarray:
    scale = float(ACTION_SCALES[action])
    if action == 0:
        return prediction.copy()
    return np.clip(hazy + scale * (prediction - hazy), 0.0, 1.0).astype(np.float32)


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


def paired_bootstrap(values: Iterable[float], seed_offset: int = 0) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size != EXPECTED_SCENES or not np.isfinite(array).all():
        raise RuntimeError("paired bootstrap requires 100 finite scene values")
    generator = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    draws = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    for start in range(0, BOOTSTRAP_RESAMPLES, 1000):
        stop = min(start + 1000, BOOTSTRAP_RESAMPLES)
        indices = generator.integers(0, array.size, size=(stop - start, array.size))
        draws[start:stop] = np.mean(array[indices], axis=1)
    return {
        "scene_count": int(array.size), "estimate": float(np.mean(array)),
        "lower": float(np.quantile(draws, 0.025)),
        "upper": float(np.quantile(draws, 0.975)),
        "seed": BOOTSTRAP_SEED + seed_offset, "resamples": BOOTSTRAP_RESAMPLES,
    }


def aggregate(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "mean": None, "median": None, "q10": None, "q90": None}
    if not np.isfinite(array).all():
        raise RuntimeError("non-finite aggregate")
    return {
        "count": int(array.size), "mean": float(np.mean(array)),
        "median": float(np.median(array)), "q10": float(np.quantile(array, 0.10)),
        "q90": float(np.quantile(array, 0.90)),
    }


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


def load_official_model(context):
    import torch

    checkpoint = asset_path(context, "official_checkpoint", kind="file")
    model_source = asset_path(context, "model_source", kind="file")
    model_layers = asset_path(context, "model_layers", kind="file")
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


def tile_moments(torch, functional, tensor, height: int, width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tensor = tensor[:, :, :height, :width]
    pad_h, pad_w = (-height) % TILE_SIZE, (-width) % TILE_SIZE
    padded = functional.pad(tensor, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
    mask = torch.ones((1, 1, height, width), dtype=tensor.dtype, device=tensor.device)
    mask = functional.pad(mask, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
    counts = functional.avg_pool2d(mask, TILE_SIZE, stride=TILE_SIZE, divisor_override=1)
    sums = functional.avg_pool2d(padded, TILE_SIZE, stride=TILE_SIZE, divisor_override=1)
    squares = functional.avg_pool2d(padded * padded, TILE_SIZE, stride=TILE_SIZE, divisor_override=1)
    means = sums / counts
    variances = torch.clamp(squares / counts - means * means, min=0.0)
    mean_array = means.squeeze(0).permute(1, 2, 0).reshape(-1, means.shape[1]).detach().cpu().numpy()
    std_array = torch.sqrt(variances).squeeze(0).permute(1, 2, 0).reshape(-1, means.shape[1]).detach().cpu().numpy()
    area_array = counts.reshape(-1).detach().cpu().numpy()
    return mean_array.astype(np.float32), std_array.astype(np.float32), area_array.astype(np.float64)


def luma_gradient(functional, value):
    horizontal = functional.pad(torch_abs(value[:, :, :, 1:] - value[:, :, :, :-1]), (0, 1, 0, 0))
    vertical = functional.pad(torch_abs(value[:, :, 1:, :] - value[:, :, :-1, :]), (0, 0, 0, 1))
    return 0.5 * (horizontal + vertical)


def torch_abs(value):
    return value.abs()


def extract_variant(torch, model, hazy: np.ndarray, clear: np.ndarray, device: str) -> dict[str, Any]:
    import torch.nn.functional as functional

    hazy_tensor = torch.from_numpy(hazy.transpose(2, 0, 1).copy()).unsqueeze(0).to(device)
    clear_tensor = torch.from_numpy(clear.transpose(2, 0, 1).copy()).unsqueeze(0).to(device)
    height, width = hazy.shape[:2]
    padded = functional.pad(hazy_tensor, (0, (-width) % 32, 0, (-height) % 32), mode="reflect")
    captured: dict[str, Any] = {}

    def hook(_module, _inputs, output):
        captured["decoder"] = output

    handle = model.Decoder[2].register_forward_hook(hook)
    try:
        with torch.inference_mode():
            outputs = model(padded)
            if not isinstance(outputs, list) or len(outputs) != 3 or "decoder" not in captured:
                raise RuntimeError("official output or decoder feature contract changed")
            prediction_tensor = outputs[2][:, :, :height, :width].clamp(0.0, 1.0)
            decoder_tensor = captured["decoder"][:, :, :height, :width]
            if (
                not bool(torch.isfinite(prediction_tensor).all().item())
                or not bool(torch.isfinite(decoder_tensor).all().item())
            ):
                raise RuntimeError("official model produced non-finite output or features")

            residual = prediction_tensor - hazy_tensor
            weights = torch.tensor([0.299, 0.587, 0.114], device=device, dtype=torch.float32).view(1, 3, 1, 1)
            hazy_luma = torch.sum(hazy_tensor * weights, dim=1, keepdim=True)
            prediction_luma = torch.sum(prediction_tensor * weights, dim=1, keepdim=True)
            residual_luma = prediction_luma - hazy_luma
            handcrafted_channels = torch.cat([
                hazy_tensor, prediction_tensor, residual, residual.abs(),
                hazy_luma, prediction_luma, residual_luma,
                hazy_tensor.max(dim=1, keepdim=True).values - hazy_tensor.min(dim=1, keepdim=True).values,
                prediction_tensor.max(dim=1, keepdim=True).values - prediction_tensor.min(dim=1, keepdim=True).values,
                luma_gradient(functional, hazy_luma),
                luma_gradient(functional, prediction_luma),
                luma_gradient(functional, residual_luma),
            ], dim=1)
            if handcrafted_channels.shape[1] != 20 or decoder_tensor.shape[1] != 32:
                raise RuntimeError("frozen feature dimensionality changed")
            hand_mean, hand_std, areas = tile_moments(
                torch, functional, handcrafted_channels, height, width,
            )
            decoder_mean, decoder_std, decoder_areas = tile_moments(
                torch, functional, decoder_tensor, height, width,
            )
            if not np.array_equal(areas, decoder_areas):
                raise RuntimeError("handcrafted and decoder tile grids differ")
            features = np.concatenate([hand_mean, hand_std, decoder_mean, decoder_std], axis=1)
            if features.shape[1] != FULL_FEATURE_DIM or not np.isfinite(features).all():
                raise RuntimeError("invalid frozen tile features")

            candidate_sse = []
            for scale in ACTION_SCALES:
                candidate = torch.clamp(
                    hazy_tensor + float(scale) * (prediction_tensor - hazy_tensor), 0.0, 1.0,
                )
                error = torch.sum((candidate - clear_tensor) ** 2, dim=1, keepdim=True)
                sums, _, error_areas = tile_moments(torch, functional, error, height, width)
                if not np.array_equal(areas, error_areas):
                    raise RuntimeError("candidate error tile grid differs")
                candidate_sse.append(sums[:, 0] * areas)
            action_sse = np.stack(candidate_sse, axis=1).astype(np.float64)
            denominators = np.maximum(action_sse[:, [0]], EPSILON * areas[:, None] * 3.0)
            targets = np.clip(
                (action_sse[:, [0]] - action_sse[:, 1:]) / denominators,
                -1.0, 1.0,
            ).astype(np.float64)
            prediction = prediction_tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)
    finally:
        handle.remove()
    return {
        "features": features.astype(np.float32), "targets": targets,
        "action_sse": action_sse, "areas": areas, "prediction": prediction,
        "height": height, "width": width,
    }


def feature_view(features: np.ndarray, family: str) -> np.ndarray:
    if family == "full":
        return features
    if family == "observable":
        return features[:, :HANDCRAFTED_DIM]
    if family == "input":
        return features[:, INPUT_FEATURE_INDICES]
    raise ValueError(f"unknown feature family: {family}")


def stack_records(
    records: list[dict[str, Any]], family: str, *, shuffle_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.concatenate([feature_view(record["features"], family) for record in records], axis=0).astype(np.float64)
    y = np.concatenate([record["targets"] for record in records], axis=0).astype(np.float64)
    weights = []
    for record in records:
        normalized = record["areas"] / float(np.sum(record["areas"])) / VARIANTS_PER_SCENE
        weights.append(normalized)
    w = np.concatenate(weights).astype(np.float64)
    w /= float(np.sum(w))
    if shuffle_seed is not None:
        generator = np.random.default_rng(shuffle_seed)
        y = y[generator.permutation(y.shape[0])]
    if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(w).all():
        raise RuntimeError("non-finite ridge input")
    return x, y, w


def ridge_family(
    records: list[dict[str, Any]], family: str, alphas: Iterable[float], *,
    shuffle_seed: int | None = None,
) -> dict[float, dict[str, np.ndarray]]:
    x, y, weights = stack_records(records, family, shuffle_seed=shuffle_seed)
    mean = np.sum(x * weights[:, None], axis=0)
    variance = np.sum((x - mean) ** 2 * weights[:, None], axis=0)
    scale = np.sqrt(np.maximum(variance, 1e-10))
    standardized = (x - mean) / scale
    y_mean = np.sum(y * weights[:, None], axis=0)
    centered_y = y - y_mean
    gram = standardized.T @ (standardized * weights[:, None])
    rhs = standardized.T @ (centered_y * weights[:, None])
    identity = np.eye(gram.shape[0], dtype=np.float64)
    models = {}
    for alpha in alphas:
        beta = np.linalg.solve(gram + float(alpha) * identity, rhs)
        models[float(alpha)] = {"mean": mean, "scale": scale, "y_mean": y_mean, "beta": beta}
    return models


def ridge_predict(model: dict[str, np.ndarray], features: np.ndarray, family: str) -> np.ndarray:
    x = feature_view(features, family).astype(np.float64)
    prediction = (x - model["mean"]) / model["scale"] @ model["beta"] + model["y_mean"]
    prediction = np.clip(prediction, -1.0, 1.0)
    if not np.isfinite(prediction).all():
        raise RuntimeError("non-finite ridge prediction")
    return prediction


def choose_actions(predicted_utilities: np.ndarray, areas: np.ndarray, threshold: float) -> tuple[np.ndarray, int]:
    best_nonkeep = np.argmax(predicted_utilities, axis=1) + 1
    best_value = np.max(predicted_utilities, axis=1)
    actions = np.where(best_value >= threshold, best_nonkeep, 0).astype(np.int64)
    weighted = np.sum(predicted_utilities * areas[:, None], axis=0) / float(np.sum(areas))
    uniform = int(np.argmax(weighted) + 1) if float(np.max(weighted)) >= threshold else 0
    return actions, uniform


def mse_from_actions(record: dict[str, Any], actions: np.ndarray) -> float:
    if actions.shape != (record["action_sse"].shape[0],):
        raise RuntimeError("action shape mismatch")
    selected = record["action_sse"][np.arange(actions.size), actions]
    return float(np.sum(selected) / (3.0 * np.sum(record["areas"])))


def record_policy_psnr(record: dict[str, Any], prediction: np.ndarray, threshold: float) -> dict[str, Any]:
    actions, uniform = choose_actions(prediction, record["areas"], threshold)
    baseline_mse = float(np.sum(record["action_sse"][:, 0]) / (3.0 * np.sum(record["areas"])))
    spatial_mse = mse_from_actions(record, actions)
    uniform_actions = np.full(actions.shape, uniform, dtype=np.int64)
    uniform_mse = mse_from_actions(record, uniform_actions)
    return {
        "actions": actions, "uniform": uniform,
        "spatial_minus_keep": psnr_from_mse(spatial_mse) - psnr_from_mse(baseline_mse),
        "spatial_minus_uniform": psnr_from_mse(spatial_mse) - psnr_from_mse(uniform_mse),
        "uniform_minus_keep": psnr_from_mse(uniform_mse) - psnr_from_mse(baseline_mse),
    }


def scene_psnr_summary(records: list[dict[str, Any]], predictions: dict[str, np.ndarray], threshold: float) -> dict[str, Any]:
    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_scene[record["scene"]].append(record_policy_psnr(record, predictions[record["id"]], threshold))
    scene_rows = []
    for scene in sorted(by_scene):
        rows = by_scene[scene]
        if len(rows) != VARIANTS_PER_SCENE:
            raise RuntimeError("inner scene lacks four variants")
        scene_rows.append({
            "spatial_minus_keep": float(np.mean([row["spatial_minus_keep"] for row in rows])),
            "spatial_minus_uniform": float(np.mean([row["spatial_minus_uniform"] for row in rows])),
        })
    gains = np.asarray([row["spatial_minus_keep"] for row in scene_rows], dtype=np.float64)
    spatial = np.asarray([row["spatial_minus_uniform"] for row in scene_rows], dtype=np.float64)
    return {
        "scene_count": len(scene_rows), "mean_spatial_minus_keep": float(np.mean(gains)),
        "mean_spatial_minus_uniform": float(np.mean(spatial)),
        "harm_rate": float(np.mean(gains <= -PSNR_HARM_MARGIN_DB)),
    }


def select_inner_config(records: list[dict[str, Any]], outer_fold: int) -> tuple[dict[str, float], dict[str, Any]]:
    scenes = sorted({record["scene"] for record in records})
    inner_assignment = {scene: index % INNER_FOLDS for index, scene in enumerate(scenes)}
    predictions_by_alpha: dict[float, dict[str, np.ndarray]] = {alpha: {} for alpha in ALPHAS}
    for inner_fold in range(INNER_FOLDS):
        train = [record for record in records if inner_assignment[record["scene"]] != inner_fold]
        validation = [record for record in records if inner_assignment[record["scene"]] == inner_fold]
        models = ridge_family(train, "full", ALPHAS)
        for alpha, model in models.items():
            for record in validation:
                predictions_by_alpha[alpha][record["id"]] = ridge_predict(model, record["features"], "full")
    candidates = []
    for alpha in ALPHAS:
        if len(predictions_by_alpha[alpha]) != len(records):
            raise RuntimeError("inner OOF prediction coverage incomplete")
        for threshold in ACTION_THRESHOLDS:
            metrics = scene_psnr_summary(records, predictions_by_alpha[alpha], threshold)
            safe = metrics["harm_rate"] <= INNER_MAX_HARM_RATE
            score = metrics["mean_spatial_minus_keep"] + metrics["mean_spatial_minus_uniform"]
            candidates.append({
                "alpha": float(alpha), "threshold": float(threshold), "safe": safe,
                "score": float(score), **metrics,
            })
    safe_candidates = [candidate for candidate in candidates if candidate["safe"]]
    if not safe_candidates:
        raise RuntimeError("keep-only inner configuration unexpectedly failed the harm screen")
    selected = max(
        safe_candidates,
        key=lambda item: (
            item["score"],
            min(item["mean_spatial_minus_keep"], item["mean_spatial_minus_uniform"]),
            item["threshold"], item["alpha"],
        ),
    )
    diagnostic = {
        "outer_fold": outer_fold, "candidate_count": len(candidates),
        "selected_alpha": selected["alpha"], "selected_threshold": selected["threshold"],
        "selected_inner_mean_spatial_minus_keep": selected["mean_spatial_minus_keep"],
        "selected_inner_mean_spatial_minus_uniform": selected["mean_spatial_minus_uniform"],
        "selected_inner_harm_rate": selected["harm_rate"],
    }
    return {"alpha": selected["alpha"], "threshold": selected["threshold"]}, diagnostic


def assign_outer_policy(
    records: list[dict[str, Any]], model: dict[str, np.ndarray], family: str,
    threshold: float, policy_name: str,
) -> None:
    for record in records:
        predicted = ridge_predict(model, record["features"], family)
        actions, uniform = choose_actions(predicted, record["areas"], threshold)
        record.setdefault("policies", {})[policy_name] = {
            "actions": actions, "uniform": uniform, "predicted": predicted,
        }


def build_spatial_output(
    hazy: np.ndarray, prediction: np.ndarray, actions: np.ndarray,
) -> np.ndarray:
    candidates = [apply_scale(hazy, prediction, action) for action in range(3)]
    output = np.empty_like(prediction, dtype=np.float32)
    height, width = hazy.shape[:2]
    index = 0
    for top in range(0, height, TILE_SIZE):
        bottom = min(top + TILE_SIZE, height)
        for left in range(0, width, TILE_SIZE):
            right = min(left + TILE_SIZE, width)
            output[top:bottom, left:right] = candidates[int(actions[index])][top:bottom, left:right]
            index += 1
    if index != actions.size:
        raise RuntimeError("spatial replay tile count mismatch")
    return output


def replay_variant(torch, record: dict[str, Any], device: str) -> dict[str, Any]:
    hazy = image_array(record["hazy_path"])
    clear = image_array(record["clear_path"])
    prediction = np.load(record["cache_path"], allow_pickle=False).astype(np.float32)
    if prediction.shape != hazy.shape or clear.shape != hazy.shape:
        raise RuntimeError("cached replay shape mismatch")
    primary = record["policies"]["primary"]
    shuffled = record["policies"]["shuffled"]
    spatial = build_spatial_output(hazy, prediction, primary["actions"])
    uniform = apply_scale(hazy, prediction, int(primary["uniform"]))
    shuffled_output = build_spatial_output(hazy, prediction, shuffled["actions"])
    baseline_ssim, spatial_ssim, uniform_ssim, shuffled_ssim = rgb_ssim(
        torch, [prediction, spatial, uniform, shuffled_output], clear, device,
    )
    baseline_psnr = psnr_from_mse(mse(prediction, clear))
    spatial_psnr = psnr_from_mse(mse(spatial, clear))
    uniform_psnr = psnr_from_mse(mse(uniform, clear))
    shuffled_psnr = psnr_from_mse(mse(shuffled_output, clear))
    oracle_actions = np.argmin(record["action_sse"], axis=1)

    result = {
        "spatial_minus_keep_psnr": spatial_psnr - baseline_psnr,
        "spatial_minus_uniform_psnr": spatial_psnr - uniform_psnr,
        "spatial_minus_shuffle_psnr": spatial_psnr - shuffled_psnr,
        "uniform_minus_keep_psnr": uniform_psnr - baseline_psnr,
        "shuffle_minus_keep_psnr": shuffled_psnr - baseline_psnr,
        "spatial_minus_keep_ssim": spatial_ssim - baseline_ssim,
        "spatial_minus_uniform_ssim": spatial_ssim - uniform_ssim,
        "spatial_minus_keep_color_bias": color_bias(spatial, clear) - color_bias(prediction, clear),
        "spatial_minus_uniform_color_bias": color_bias(spatial, clear) - color_bias(uniform, clear),
        "oracle_action_agreement": float(np.sum(record["areas"] * (primary["actions"] == oracle_actions)) / np.sum(record["areas"])),
    }
    for family in ("observable", "input"):
        policy = record["policies"][family]
        family_mse = mse_from_actions(record, policy["actions"])
        baseline_mse = float(np.sum(record["action_sse"][:, 0]) / (3.0 * np.sum(record["areas"])))
        result[f"{family}_minus_keep_psnr"] = psnr_from_mse(family_mse) - psnr_from_mse(baseline_mse)
    return result


def scene_replay(records: list[dict[str, Any]], torch, device: str) -> list[dict[str, Any]]:
    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_scene[record["scene"]].append(replay_variant(torch, record, device))
    scenes = []
    keys = (
        "spatial_minus_keep_psnr", "spatial_minus_uniform_psnr", "spatial_minus_shuffle_psnr",
        "uniform_minus_keep_psnr", "shuffle_minus_keep_psnr", "spatial_minus_keep_ssim",
        "spatial_minus_uniform_ssim", "spatial_minus_keep_color_bias",
        "spatial_minus_uniform_color_bias", "oracle_action_agreement",
        "observable_minus_keep_psnr", "input_minus_keep_psnr",
    )
    for scene in sorted(by_scene):
        rows = by_scene[scene]
        if len(rows) != VARIANTS_PER_SCENE:
            raise RuntimeError("outer replay scene lacks four variants")
        scenes.append({key: float(np.mean([row[key] for row in rows])) for key in keys})
    if len(scenes) != EXPECTED_SCENES:
        raise RuntimeError("outer replay lacks 100 scenes")
    return scenes


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"write-once CSV already exists: {path}")
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def terminal_summary(
    integrity: dict[str, bool], scenes: list[dict[str, Any]], fold_rows: list[dict[str, Any]],
    action_counts: dict[str, Counter[int]],
) -> dict[str, Any]:
    complete = all(integrity.values()) and len(scenes) == EXPECTED_SCENES
    if complete:
        keep = paired_bootstrap((scene["spatial_minus_keep_psnr"] for scene in scenes), 0)
        uniform = paired_bootstrap((scene["spatial_minus_uniform_psnr"] for scene in scenes), 1)
        shuffle = paired_bootstrap((scene["spatial_minus_shuffle_psnr"] for scene in scenes), 2)
        material = wilson(sum(scene["spatial_minus_keep_psnr"] >= MATERIAL_SCENE_GAIN_DB for scene in scenes), EXPECTED_SCENES)
        psnr_harm = wilson(sum(scene["spatial_minus_keep_psnr"] <= -PSNR_HARM_MARGIN_DB for scene in scenes), EXPECTED_SCENES)
        ssim_harm = wilson(sum(scene["spatial_minus_keep_ssim"] <= -SSIM_HARM_MARGIN for scene in scenes), EXPECTED_SCENES)
        color_harm = wilson(sum(scene["spatial_minus_keep_color_bias"] >= COLOR_HARM_MARGIN for scene in scenes), EXPECTED_SCENES)
        utility_pass = bool(
            keep["lower"] >= MIN_MEAN_GAIN_DB
            and uniform["lower"] >= MIN_SPATIAL_GAIN_DB
            and shuffle["lower"] >= MIN_SHUFFLE_CONTRAST_DB
            and material["lower"] >= MIN_MATERIAL_SCENE_PREVALENCE
        )
        safety_pass = bool(
            psnr_harm["upper"] < MAX_HARM_PREVALENCE
            and ssim_harm["upper"] < MAX_HARM_PREVALENCE
            and color_harm["upper"] < MAX_HARM_PREVALENCE
        )
        utility_fail = bool(
            keep["upper"] < MIN_MEAN_GAIN_DB or uniform["upper"] < MIN_SPATIAL_GAIN_DB
            or shuffle["upper"] < MIN_SHUFFLE_CONTRAST_DB
            or material["upper"] < MIN_MATERIAL_SCENE_PREVALENCE
        )
        safety_fail = bool(
            psnr_harm["lower"] >= MAX_HARM_PREVALENCE
            or ssim_harm["lower"] >= MAX_HARM_PREVALENCE
            or color_harm["lower"] >= MAX_HARM_PREVALENCE
        )
    else:
        unavailable_mean = {
            "scene_count": len(scenes), "estimate": None, "lower": None, "upper": None,
            "seed": None, "resamples": 0,
        }
        unavailable_prevalence = {
            "successes": None, "total": len(scenes), "estimate": None,
            "lower": None, "upper": None,
        }
        keep = dict(unavailable_mean)
        uniform = dict(unavailable_mean)
        shuffle = dict(unavailable_mean)
        material = dict(unavailable_prevalence)
        psnr_harm = dict(unavailable_prevalence)
        ssim_harm = dict(unavailable_prevalence)
        color_harm = dict(unavailable_prevalence)
        utility_pass = False
        safety_pass = False
        utility_fail = True
        safety_fail = False
    if utility_pass and safety_pass:
        state = "COMPLETED_GATE_PASS"
        decision = "HAZE4K_TEST_LOCAL_ACTION_PROXY_PREDICTABILITY_PASS"
        authorizes = "HAZE4K_LOCAL_ACTION_MODULE_CONTRACT_REVIEW_ONLY"
        reasons = ["cross-fitted spatial proxy passed all frozen utility, spatial-specificity, falsification, breadth, and tail-safety gates"]
    elif utility_fail or safety_fail or not complete:
        state = "COMPLETED_GATE_FAIL"
        decision = "HAZE4K_TEST_LOCAL_ACTION_PROXY_PREDICTABILITY_FAIL"
        authorizes = "NONE"
        reasons = ["at least one frozen utility or safety interval established futility, harm, or incomplete integrity"]
    else:
        state = "COMPLETED_INCONCLUSIVE"
        decision = "HAZE4K_TEST_LOCAL_ACTION_PROXY_PREDICTABILITY_INCONCLUSIVE"
        authorizes = "HAZE4K_TEST_LOCAL_ACTION_PROXY_PREDICTABILITY_SUPPLEMENT_ONLY"
        reasons = ["all planned units completed but at least one frozen qualification interval crossed its decision margin"]

    total_actions = sum(action_counts["primary"].values())
    action_usage = {
        name: {
            "tiles": action_counts["primary"][index],
            "fraction": action_counts["primary"][index] / total_actions if total_actions else None,
        }
        for index, name in enumerate(ACTION_NAMES)
    }
    summary = {
        "schema_version": 1,
        "marker": "HAZE4K_TEST_LOCAL_ACTION_PROXY_PREDICTABILITY_COMPLETE",
        "scope": "nested scene-cross-fitted low-capacity proxy on isolated 100-scene Haze4K official-test development partition",
        "identity_and_coverage": {
            "completed_scenes": len(scenes), "completed_variants": EXPECTED_VARIANTS,
            "assignment_digest": SPLIT_ASSIGNMENT_DIGEST, "outer_folds": OUTER_FOLDS,
            "inner_folds": INNER_FOLDS, "candidate_confirmation_asset_delivered": False,
            "network_training_occurred": False, "proxy_fitting_occurred": True,
            "integrity_checks": integrity,
        },
        "predictor_contract": {
            "primary_family": "weighted multi-output ridge over 40 deployment-visible local image/output features plus 64 pooled frozen Decoder[2] moments",
            "targets": "GT-derived clipped normalized tile MSE advantage of weaken 0.8 and strengthen 1.2 relative to keep; GT unavailable to test prediction",
            "outer_split": "five deterministic folds of 20 canonical clear scenes; all four variants remain in one fold",
            "inner_selection": "four scene folds choose alpha and conservative action threshold from frozen grids using only outer-training scenes",
            "uniform_control": "area-average the identical tile utility predictions within each image and apply the identical threshold once",
            "shuffle_control": "permute paired action-utility targets across outer-training tile rows and refit the identical ridge with selected alpha and threshold",
            "alpha_grid": list(ALPHAS), "action_threshold_grid": list(ACTION_THRESHOLDS),
            "action_formula": "clip(hazy + scale * (official_prediction - hazy), 0, 1)",
        },
        "primary": {
            "spatial_minus_keep_scene_mean_delta_psnr_db": keep,
            "spatial_minus_same_predictor_uniform_scene_mean_delta_psnr_db": uniform,
            "spatial_minus_shuffled_target_scene_mean_delta_psnr_db": shuffle,
            "material_scene_prevalence": material,
            "utility_passed": utility_pass,
        },
        "safety": {
            "psnr_harm_prevalence": psnr_harm, "ssim_harm_prevalence": ssim_harm,
            "color_harm_prevalence": color_harm, "maximum_harm_upper_bound": MAX_HARM_PREVALENCE,
            "safety_passed": safety_pass,
        },
        "secondary": {
            "same_predictor_uniform_minus_keep_psnr_db": aggregate(scene["uniform_minus_keep_psnr"] for scene in scenes),
            "shuffled_target_minus_keep_psnr_db": aggregate(scene["shuffle_minus_keep_psnr"] for scene in scenes),
            "observable_only_minus_keep_psnr_db": aggregate(scene["observable_minus_keep_psnr"] for scene in scenes),
            "input_only_minus_keep_psnr_db": aggregate(scene["input_minus_keep_psnr"] for scene in scenes),
            "oracle_action_agreement": aggregate(scene["oracle_action_agreement"] for scene in scenes),
            "spatial_minus_keep_ssim": aggregate(scene["spatial_minus_keep_ssim"] for scene in scenes),
            "spatial_minus_keep_color_bias": aggregate(scene["spatial_minus_keep_color_bias"] for scene in scenes),
        },
        "action_usage": action_usage,
        "fold_selection": fold_rows,
        "limitations": [
            "This is development-screening proxy fitting and OOF replay, not module training or candidate confirmation.",
            "GT is used only inside outer-training partitions for targets and inner selection; the OOF test policy receives no GT-derived input.",
            "The low-capacity ridge, feature moments, three residual scales, hard 32-pixel grid, and selected thresholds do not identify an optimal module.",
            "The frozen decoder feature can encode synthetic Haze4K or official-checkpoint regularities that may not transfer to real haze.",
            "PSNR is primary; the frozen SSIM and color gates do not replace later perceptual and real-domain validation.",
            "The protected 150-scene candidate-confirmation asset and NH-HAZE were not delivered or accessed.",
        ],
        "terminal": {"state": state, "decision": decision, "authorizes": authorizes, "gate_reasons": reasons},
    }
    return summary


def synthetic_record(generator: np.random.Generator, scene: str, variant: int, tiles: int = 228) -> dict[str, Any]:
    features = generator.normal(size=(tiles, FULL_FEATURE_DIM)).astype(np.float32)
    signal = 0.03 * features[:, [0]] - 0.02 * features[:, [40]]
    targets = np.concatenate([-signal, signal], axis=1) + generator.normal(scale=0.01, size=(tiles, 2))
    targets = np.clip(targets, -1.0, 1.0)
    areas = np.full(tiles, TILE_SIZE * TILE_SIZE, dtype=np.float64)
    keep = areas * 3.0 * 0.01
    action_sse = np.stack([
        keep,
        keep * np.clip(1.0 - targets[:, 0], 0.2, 2.0),
        keep * np.clip(1.0 - targets[:, 1], 0.2, 2.0),
    ], axis=1)
    return {
        "id": f"{scene}-{variant}", "scene": scene, "features": features,
        "targets": targets.astype(np.float64), "action_sse": action_sse,
        "areas": areas,
    }


def contract(context_path: Path) -> None:
    context = load_context(context_path, "contract")
    prepare_phase_output(context)
    parent_closeout = asset_path(context, "prior_oracle_closeout", kind="file")
    parent_summary = asset_path(context, "prior_oracle_summary", kind="file")
    entrypoint = asset_path(context, "proxy_entrypoint", kind="file")
    parent_ok = (
        sha256_file(parent_closeout) == PARENT_CLOSEOUT_SHA256
        and sha256_file(parent_summary) == PARENT_SUMMARY_SHA256
        and read_json(parent_closeout).get("decision") == "HAZE4K_TEST_BOUNDED_LOCAL_ACTION_ORACLE_HEADROOM_PASS"
        and read_json(parent_closeout).get("authorizes") == "HAZE4K_TEST_LOCAL_ACTION_PROXY_PREDICTABILITY"
    )
    torch, model = load_official_model(context)
    generator = np.random.default_rng(20260726)
    hazy = generator.uniform(0.05, 0.95, size=(256, 320, 3)).astype(np.float32)
    clear = np.clip(hazy + generator.normal(scale=0.03, size=hazy.shape), 0.0, 1.0).astype(np.float32)
    fixture = extract_variant(torch, model, hazy, clear, context.device)
    fixture_record = {
        "id": "fixture", "scene": "fixture", "features": fixture["features"],
        "targets": fixture["targets"], "action_sse": fixture["action_sse"], "areas": fixture["areas"],
    }
    fitted = ridge_family([fixture_record] * VARIANTS_PER_SCENE, "full", [0.01])[0.01]
    fixture_prediction = ridge_predict(fitted, fixture_record["features"], "full")
    fixture_policy = record_policy_psnr(fixture_record, fixture_prediction, 0.01)
    fixture_output = build_spatial_output(hazy, fixture["prediction"], fixture_policy["actions"])
    fixture_ssim = rgb_ssim(torch, [fixture["prediction"], fixture_output], clear, context.device)

    started = time.monotonic()
    probe_records = [
        synthetic_record(generator, f"scene-{scene:03d}", variant)
        for scene in range(EXPECTED_SCENES) for variant in range(VARIANTS_PER_SCENE)
    ]
    probe_train = [record for record in probe_records if int(record["scene"].split("-")[1]) % OUTER_FOLDS != 0]
    selected, diagnostic = select_inner_config(probe_train, 0)
    probe_model = ridge_family(probe_train, "full", [selected["alpha"]])[selected["alpha"]]
    probe_test = [record for record in probe_records if int(record["scene"].split("-")[1]) % OUTER_FOLDS == 0]
    probe_predictions = {record["id"]: ridge_predict(probe_model, record["features"], "full") for record in probe_test}
    probe_metrics = scene_psnr_summary(probe_test, probe_predictions, selected["threshold"])
    elapsed = time.monotonic() - started
    checks = {
        "parent_authorization": parent_ok,
        "entrypoint_identity": context.assets["proxy_entrypoint"].sha256 == sha256_file(entrypoint),
        "official_graph_strict_loaded": sum(parameter.numel() for parameter in model.parameters()) == PARAMETER_COUNT,
        "frozen_feature_shape": fixture["features"].shape[1] == FULL_FEATURE_DIM,
        "finite_proxy_target": bool(np.isfinite(fixture["targets"]).all()),
        "finite_ridge_prediction": bool(np.isfinite(fixture_prediction).all()),
        "bounded_actions": bool(np.isin(fixture_policy["actions"], [0, 1, 2]).all()),
        "finite_replay": bool(np.isfinite(fixture_output).all() and np.isfinite(fixture_ssim).all()),
        "same_scale_probe_complete": len(probe_records) == EXPECTED_VARIANTS and probe_metrics["scene_count"] == 20,
        "finite_nested_selection": bool(math.isfinite(diagnostic["selected_inner_mean_spatial_minus_keep"])),
        "bounded_scale_probe_time": elapsed * OUTER_FOLDS < 600.0,
        "protected_data_absent": "haze4k_test_development" not in context.assets,
    }
    write_contract_result(
        context, checks=checks,
        engineering={
            "mode": "gpu_synthetic_no_data", "device": context.device,
            "fixture": {"batch": 1, "channels": 3, "height": 256, "width": 320},
            "production_path_exercised": True,
            "protected_data_touched": False, "scientific_output_created": False,
            "scientific_training_occurred": False,
        },
    )


def run(context_path: Path) -> None:
    context = load_context(context_path, "run")
    prepare_phase_output(context)
    development_root = asset_path(context, "haze4k_test_development", kind="directory")
    parent_closeout_path = asset_path(context, "prior_oracle_closeout", kind="file")
    parent_summary_path = asset_path(context, "prior_oracle_summary", kind="file")
    parent_closeout = read_json(parent_closeout_path)
    parent_summary = read_json(parent_summary_path)
    parent_ok = (
        sha256_file(parent_closeout_path) == PARENT_CLOSEOUT_SHA256
        and sha256_file(parent_summary_path) == PARENT_SUMMARY_SHA256
        and parent_closeout.get("state") == "COMPLETED_GATE_PASS"
        and parent_closeout.get("decision") == "HAZE4K_TEST_BOUNDED_LOCAL_ACTION_ORACLE_HEADROOM_PASS"
        and parent_closeout.get("authorizes") == "HAZE4K_TEST_LOCAL_ACTION_PROXY_PREDICTABILITY"
        and parent_summary.get("identity_and_coverage", {}).get("assignment_digest") == SPLIT_ASSIGNMENT_DIGEST
    )
    scope_ok = (
        development_root.name == "development_screening"
        and context.evidence_role == "development_screening"
        and not any(context.protected_data_permissions.values())
        and "candidate_confirmation" not in str(development_root)
    )
    write_workload_progress(context, completed_units=1, stage="authorization_and_scope")

    haze_root, clear_root = development_root / "haze", development_root / "gt"
    haze_paths = image_files(haze_root) if scope_ok and haze_root.is_dir() else []
    clear_paths = image_files(clear_root) if scope_ok and clear_root.is_dir() else []
    variants_by_digest: dict[str, list[tuple[Path, Path]]] = defaultdict(list)
    failures: list[dict[str, str]] = []
    for hazy_path in haze_paths:
        clear_path = clear_root / hazy_path.name
        if not clear_path.is_file():
            failures.append({"variant": hazy_path.name, "reason": "missing paired GT"})
            continue
        try:
            digest = canonical_rgb_digest(image_array(clear_path))
            variants_by_digest[digest].append((hazy_path, clear_path))
        except Exception as exc:
            failures.append({"variant": hazy_path.name, "reason": str(exc)[:512]})
    histogram = Counter(len(items) for items in variants_by_digest.values())
    dataset_ok = (
        len(haze_paths) == EXPECTED_VARIANTS and len(clear_paths) == EXPECTED_VARIANTS
        and len(variants_by_digest) == EXPECTED_SCENES
        and histogram == {VARIANTS_PER_SCENE: EXPECTED_SCENES} and not failures
    )
    write_workload_progress(context, completed_units=2, stage="canonical_scene_grouping")

    records: list[dict[str, Any]] = []
    cache_root = output_file(context, "prediction_cache")
    cache_root.mkdir()
    if parent_ok and scope_ok and dataset_ok:
        torch, model = load_official_model(context)
        attempted = 0
        for scene in sorted(variants_by_digest):
            for variant_index, (hazy_path, clear_path) in enumerate(sorted(variants_by_digest[scene])):
                attempted += 1
                try:
                    hazy = image_array(hazy_path)
                    clear = image_array(clear_path)
                    extracted = extract_variant(torch, model, hazy, clear, context.device)
                    record_id = f"{scene[:16]}-{variant_index}"
                    cache_path = cache_root / f"{record_id}.npy"
                    np.save(cache_path, extracted.pop("prediction"), allow_pickle=False)
                    records.append({
                        "id": record_id, "scene": scene, "variant": variant_index,
                        "hazy_path": hazy_path, "clear_path": clear_path, "cache_path": cache_path,
                        **extracted,
                    })
                except Exception as exc:
                    failures.append({"variant": hazy_path.name, "reason": str(exc)[:512]})
                if attempted % 5 == 0 or attempted == EXPECTED_VARIANTS:
                    write_workload_progress(
                        context, completed_units=2 + attempted,
                        stage="official_inference_and_proxy_target_extraction",
                    )
    inference_ok = len(records) == EXPECTED_VARIANTS and not failures

    fold_rows: list[dict[str, Any]] = []
    if inference_ok:
        scenes = sorted(variants_by_digest)
        outer_assignment = {scene: index % OUTER_FOLDS for index, scene in enumerate(scenes)}
        for outer_fold in range(OUTER_FOLDS):
            train = [record for record in records if outer_assignment[record["scene"]] != outer_fold]
            test = [record for record in records if outer_assignment[record["scene"]] == outer_fold]
            config, diagnostic = select_inner_config(train, outer_fold)
            alpha, threshold = config["alpha"], config["threshold"]
            primary_model = ridge_family(train, "full", [alpha])[alpha]
            shuffled_model = ridge_family(
                train, "full", [alpha], shuffle_seed=BOOTSTRAP_SEED + 100 + outer_fold,
            )[alpha]
            observable_model = ridge_family(train, "observable", [alpha])[alpha]
            input_model = ridge_family(train, "input", [alpha])[alpha]
            assign_outer_policy(test, primary_model, "full", threshold, "primary")
            assign_outer_policy(test, shuffled_model, "full", threshold, "shuffled")
            assign_outer_policy(test, observable_model, "observable", threshold, "observable")
            assign_outer_policy(test, input_model, "input", threshold, "input")
            test_predictions = {
                record["id"]: record["policies"]["primary"]["predicted"] for record in test
            }
            test_metrics = scene_psnr_summary(test, test_predictions, threshold)
            primary_actions = np.concatenate([record["policies"]["primary"]["actions"] for record in test])
            fold_rows.append({
                **diagnostic,
                "outer_test_scenes": len({record["scene"] for record in test}),
                "outer_test_mean_spatial_minus_keep": test_metrics["mean_spatial_minus_keep"],
                "outer_test_mean_spatial_minus_uniform": test_metrics["mean_spatial_minus_uniform"],
                "outer_test_harm_rate": test_metrics["harm_rate"],
                "outer_test_keep_fraction": float(np.mean(primary_actions == 0)),
                "outer_test_weaken_fraction": float(np.mean(primary_actions == 1)),
                "outer_test_strengthen_fraction": float(np.mean(primary_actions == 2)),
            })
            write_workload_progress(
                context, completed_units=402 + outer_fold + 1,
                stage="nested_scene_cross_fitted_proxy",
            )

    crossfit_ok = inference_ok and all("policies" in record for record in records) and len(fold_rows) == OUTER_FOLDS
    replay_scenes: list[dict[str, Any]] = []
    if crossfit_ok:
        for index, record in enumerate(records):
            record["replay"] = replay_variant(torch, record, context.device)
            if (index + 1) % 5 == 0 or index + 1 == len(records):
                write_workload_progress(
                    context, completed_units=407 + index + 1,
                    stage="oof_image_replay_and_safety",
                )
        by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            by_scene[record["scene"]].append(record["replay"])
        keys = tuple(records[0]["replay"])
        for scene in sorted(by_scene):
            rows = by_scene[scene]
            if len(rows) != VARIANTS_PER_SCENE:
                raise RuntimeError("replay scene lacks four variants")
            replay_scenes.append({key: float(np.mean([row[key] for row in rows])) for key in keys})

    integrity = {
        "parent_proxy_authorization": parent_ok,
        "isolated_development_asset_only": scope_ok,
        "complete_100_scene_400_variant_grouping": dataset_ok,
        "complete_single_official_inference_per_variant": inference_ok,
        "five_outer_scene_folds_complete": crossfit_ok,
        "all_selection_inside_outer_training_scenes": crossfit_ok,
        "complete_oof_image_replay": len(replay_scenes) == EXPECTED_SCENES,
        "candidate_confirmation_asset_not_delivered": True,
        "no_network_or_module_training": True,
    }
    action_counts = {name: Counter() for name in ("primary", "shuffled", "observable", "input")}
    for record in records:
        for name in action_counts:
            if "policies" in record:
                action_counts[name].update(int(item) for item in record["policies"][name]["actions"])
    summary = terminal_summary(integrity, replay_scenes, fold_rows, action_counts)
    summary_path = output_file(context, "haze4k_test_local_action_proxy_predictability_summary.json")
    strata_path = output_file(context, "haze4k_test_local_action_proxy_predictability_strata.csv")
    folds_path = output_file(context, "haze4k_test_local_action_proxy_predictability_folds.csv")
    atomic_json(summary_path, summary)
    primary = summary["primary"]
    safety = summary["safety"]
    strata_rows = [
        {"estimand": "spatial_minus_keep_psnr_db", "method": "paired_bootstrap", **primary["spatial_minus_keep_scene_mean_delta_psnr_db"], "threshold": MIN_MEAN_GAIN_DB},
        {"estimand": "spatial_minus_uniform_psnr_db", "method": "paired_bootstrap", **primary["spatial_minus_same_predictor_uniform_scene_mean_delta_psnr_db"], "threshold": MIN_SPATIAL_GAIN_DB},
        {"estimand": "spatial_minus_shuffle_psnr_db", "method": "paired_bootstrap", **primary["spatial_minus_shuffled_target_scene_mean_delta_psnr_db"], "threshold": MIN_SHUFFLE_CONTRAST_DB},
        {"estimand": "material_benefit_prevalence", "method": "wilson", **primary["material_scene_prevalence"], "threshold": MIN_MATERIAL_SCENE_PREVALENCE},
        {"estimand": "psnr_harm_prevalence", "method": "wilson", **safety["psnr_harm_prevalence"], "threshold": MAX_HARM_PREVALENCE},
        {"estimand": "ssim_harm_prevalence", "method": "wilson", **safety["ssim_harm_prevalence"], "threshold": MAX_HARM_PREVALENCE},
        {"estimand": "color_harm_prevalence", "method": "wilson", **safety["color_harm_prevalence"], "threshold": MAX_HARM_PREVALENCE},
    ]
    strata_fields = sorted({key for row in strata_rows for key in row})
    write_csv(strata_path, strata_fields, strata_rows)
    fold_fields = list(fold_rows[0]) if fold_rows else ["outer_fold"]
    write_csv(folds_path, fold_fields, fold_rows)
    write_workload_progress(context, completed_units=808, stage="scene_level_proxy_finalize")
    terminal = summary["terminal"]
    write_run_result(
        context, state=terminal["state"], decision=terminal["decision"],
        authorizes=terminal["authorizes"],
        details={
            "independent_scenes": len(replay_scenes), "nested_variants": len(records),
            "utility_passed": summary["primary"]["utility_passed"],
            "safety_passed": summary["safety"]["safety_passed"],
            "network_training_occurred": False, "proxy_fitting_occurred": True,
            "candidate_confirmation_asset_delivered": False,
            "summary_file": summary_path.name, "strata_file": strata_path.name,
            "folds_file": folds_path.name, "gate_reasons": terminal["gate_reasons"],
        },
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
