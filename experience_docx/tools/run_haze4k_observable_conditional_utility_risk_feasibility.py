#!/usr/bin/env python3
"""Qualify one observable contextual utility-risk head on development-only Haze4K."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
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
    load_completed_unit_ledger,
    output_file,
    prepare_phase_output,
    record_completed_unit,
    write_contract_progress,
    write_contract_result,
    write_gate_result,
    write_workload_progress,
)


IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}
TRAIN_INPUT_DIRECTORIES = ("IN", "haze", "hazy")
TRAIN_LABEL_DIRECTORIES = ("GT", "gt")
EXPECTED_TRAIN_SCENES = 750
TRAINING_SCENES = 600
CALIBRATION_SCENES = 150
EXPECTED_TEST_SCENES = 100
VARIANTS_PER_SCENE = 4
TILE_SIZE = 32
FEATURE_CHANNELS = 104
ACTION_NAMES = ("weaken", "strengthen")
ACTION_SCALES = np.asarray([0.8, 1.2], dtype=np.float32)
TRAIN_SPLIT_SALT = "haze4k-local-error-qualification-v2"
VARIANT_SELECTION_SALT = "haze4k-observable-conditional-utility-risk-feasibility-v1"
OOF_FOLD_SALT = "haze4k-observable-conditional-utility-risk-feasibility-v1-oof"
SHUFFLE_SALT = "haze4k-observable-conditional-utility-risk-feasibility-v1-shuffle"
TRAIN_ASSIGNMENT_DIGEST = "7b21d3af455475f7bb29198081a2ef2e651cffaac6149fd27741863c765b4efc"
TEST_ASSIGNMENT_DIGEST = "6ca5174470dad2b4eef4ae15c5a13a99d8ae9fc0bc2ea1116b199c4d4bc05582"
ANCHOR_COMMIT = "3b4da35440c8c26a7d1bcaf1daf342e11d9a3898"
CHECKPOINT_SHA256 = "6f42037d57a4e3de3a10ac0ab909d66a3415864a19433c29204a975f4efa4088"
MODEL_SOURCE_SHA256 = "3fa227af396464a7f07ac773f92e9cdb746e0fa6ae63adef711c765a02c3d4cd"
MODEL_LAYERS_SHA256 = "ac8a05bd626d9adda16308dedb9466f36d7ff44cfb666f64e7e14ddf8cdf43a4"
ROLE_LEDGER_CLOSEOUT_SHA256 = "54027119140165b981f8f464c898065430d1be7cbc297b472ca3c347560859c5"
ROLE_LEDGER_SUMMARY_SHA256 = "6353eb0f485437d9879e8adec4ca5833c17769cd97c514caf428cef22d3e0ea2"
TRAIN_SPLIT_CLOSEOUT_SHA256 = "dde7f4654674f776d2f2b0a687128019e477eea7b29fb8d69f01d621aa6c5887"
TRAIN_SPLIT_SUMMARY_SHA256 = "a9aaa2b40d73eccb133763ce75d030c1a41013079178b39b4ebd5693a22099a9"
TEST_SPLIT_CLOSEOUT_SHA256 = "d3dbd88b25eee35c5922b2459a2f48b39b3fc8f686588cde9edb1d9f267f8a9f"
TEST_SPLIT_SUMMARY_SHA256 = "656407f3183f92e75e357004b04b505d2b92cd4b0f4b02b3c6a22aef57d069cf"
PROXY_CLOSEOUT_SHA256 = "c12a4b013db12b0828823ae8cba7fe8a51208971d29e667966635c157a67cba4"
PROXY_SUMMARY_SHA256 = "052e2b20b12dde785db3929df0f353713592a340ccafa9d48f50646706501bc7"
ORACLE_CLOSEOUT_SHA256 = "96357cbaaee5aa338fb0f9c9835a975a27e7f048c78a096fd644e6acdd3e383c"
ORACLE_SUMMARY_SHA256 = "794ef27733f51f2fa70fab5c94bc661564d7b988ef4247a1b26e33a21b4de7cb"
RUNTIME_ENVIRONMENT_SHA256 = "35600a8354cfca6c0f3ed3c6159a362377e5c558795208f325e59d74b59b569b"
PARAMETER_COUNT = 8_630_665
OOF_FOLDS = 5
EPOCHS = 12
LEARNING_RATE = 3.0e-4
WEIGHT_DECAY = 1.0e-4
GRADIENT_CLIP_NORM = 1.0
UTILITY_HUBER_BETA_DB = 0.25
RISK_LOSS_WEIGHT = 0.5
RISK_POSITIVE_WEIGHT = 2.0
TARGET_CLIP_DB = 3.0
UTILITY_MARGIN_DB = 0.05
TRANSFER_POINT_MARGIN_DB = 0.025
RISK_ACCEPTANCE_MAX = 0.10
PSNR_HARM_MARGIN_DB = 0.10
SSIM_HARM_MARGIN = 0.005
COLOR_HARM_MARGIN = 1.0 / 255.0
MAX_HARM_PREVALENCE = 0.10
MIN_MATERIAL_SCENE_PREVALENCE = 0.20
CALIBRATION_QUANTILE = 0.95
BOOTSTRAP_RESAMPLES = 20_000
BOOTSTRAP_SEED = 20260726
TRAINING_SEED = 20260726
TOTAL_UNITS = 851
FORMAL_COST_ITERATIONS = 39_050
PROBE_COST_ITERATIONS = 160
EPSILON = 1e-12


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
    seen: set[Path] = set()
    for name in names:
        candidate = label_dir / name
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            return candidate
    return None


def digest_lines(lines: Iterable[str]) -> str:
    return sha256_text("\n".join(sorted(lines)))


def deterministic_seed(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:16], 16) % (2**32)


def choose_one_variant(scene: str, items: list[tuple[Path, Path]]) -> tuple[Path, Path]:
    if len(items) != VARIANTS_PER_SCENE:
        raise RuntimeError(f"scene {scene[:16]} does not contain four variants")
    return min(
        items,
        key=lambda pair: (
            sha256_text(f"{VARIANT_SELECTION_SALT}|{scene}|{pair[0].name}"),
            pair[0].name,
        ),
    )


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
            clear = image_array(clear_path)
            hazy = image_array(hazy_path)
            if clear.shape != hazy.shape:
                raise RuntimeError(f"train pair dimensions differ for {hazy_path.name}")
            digest = canonical_rgb_digest(clear)
            clear_digest_cache[clear_path] = digest
        groups[digest].append((hazy_path, clear_path))
    if len(groups) != EXPECTED_TRAIN_SCENES \
            or Counter(len(items) for items in groups.values()) != {VARIANTS_PER_SCENE: EXPECTED_TRAIN_SCENES}:
        raise RuntimeError("Haze4K train canonical scene structure changed")
    ranked = sorted(
        groups,
        key=lambda scene: (sha256_text(f"{TRAIN_SPLIT_SALT}|{scene}"), scene),
    )
    calibration = ranked[:CALIBRATION_SCENES]
    training = ranked[CALIBRATION_SCENES:]
    assignment = digest_lines(
        f"{scene},{'internal_development' if scene in set(calibration) else 'training'}"
        for scene in sorted(groups)
    )
    return dict(groups), training, calibration, assignment


def enumerate_test_groups(root: Path) -> dict[str, list[tuple[Path, Path]]]:
    haze_root, clear_root = root / "haze", root / "gt"
    if root.name != "development_screening" or not haze_root.is_dir() or not clear_root.is_dir():
        raise RuntimeError("isolated Haze4K test-development asset contract changed")
    haze_paths = image_files(haze_root)
    clear_paths = image_files(clear_root)
    if len(haze_paths) != 400 or len(clear_paths) != 400:
        raise RuntimeError("Haze4K test-development file census changed")
    groups: dict[str, list[tuple[Path, Path]]] = defaultdict(list)
    for hazy_path in haze_paths:
        clear_path = clear_root / hazy_path.name
        if not clear_path.is_file():
            raise RuntimeError(f"missing test-development target for {hazy_path.name}")
        hazy, clear = image_array(hazy_path), image_array(clear_path)
        if hazy.shape != clear.shape:
            raise RuntimeError(f"test-development dimensions differ for {hazy_path.name}")
        groups[canonical_rgb_digest(clear)].append((hazy_path, clear_path))
    if len(groups) != EXPECTED_TEST_SCENES \
            or Counter(len(items) for items in groups.values()) != {VARIANTS_PER_SCENE: EXPECTED_TEST_SCENES}:
        raise RuntimeError("Haze4K test-development canonical scene structure changed")
    return dict(groups)


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


def tile_moments(torch, functional, tensor, height: int, width: int):
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
    return means, torch.sqrt(variances), counts


def luma_gradient(functional, value):
    horizontal = functional.pad((value[:, :, :, 1:] - value[:, :, :, :-1]).abs(), (0, 1, 0, 0))
    vertical = functional.pad((value[:, :, 1:, :] - value[:, :, :-1, :]).abs(), (0, 0, 0, 1))
    return 0.5 * (horizontal + vertical)


def extract_variant(torch, model, hazy: np.ndarray, clear: np.ndarray, device: str) -> dict[str, Any]:
    import torch.nn.functional as functional

    hazy_tensor = torch.from_numpy(hazy.transpose(2, 0, 1).copy()).unsqueeze(0).to(device)
    clear_tensor = torch.from_numpy(clear.transpose(2, 0, 1).copy()).unsqueeze(0).to(device)
    height, width = hazy.shape[:2]
    padded = functional.pad(
        hazy_tensor, (0, (-width) % TILE_SIZE, 0, (-height) % TILE_SIZE), mode="reflect",
    )
    captured: dict[str, Any] = {}

    def hook(_module, _inputs, output):
        captured["decoder"] = output

    handle = model.Decoder[2].register_forward_hook(hook)
    try:
        with torch.inference_mode():
            outputs = model(padded)
            if not isinstance(outputs, list) or len(outputs) != 3 or "decoder" not in captured:
                raise RuntimeError("official output or Decoder[2] feature contract changed")
            prediction_tensor = outputs[2][:, :, :height, :width].clamp(0.0, 1.0)
            decoder_tensor = captured["decoder"][:, :, :height, :width]
            if not bool(torch.isfinite(prediction_tensor).all().item()) \
                    or not bool(torch.isfinite(decoder_tensor).all().item()):
                raise RuntimeError("official model produced non-finite output or features")
            residual = prediction_tensor - hazy_tensor
            weights = torch.tensor(
                [0.299, 0.587, 0.114], dtype=torch.float32, device=device,
            ).view(1, 3, 1, 1)
            hazy_luma = torch.sum(hazy_tensor * weights, dim=1, keepdim=True)
            prediction_luma = torch.sum(prediction_tensor * weights, dim=1, keepdim=True)
            residual_luma = prediction_luma - hazy_luma
            handcrafted = torch.cat([
                hazy_tensor,
                prediction_tensor,
                residual,
                residual.abs(),
                hazy_luma,
                prediction_luma,
                residual_luma,
                hazy_tensor.max(dim=1, keepdim=True).values
                - hazy_tensor.min(dim=1, keepdim=True).values,
                prediction_tensor.max(dim=1, keepdim=True).values
                - prediction_tensor.min(dim=1, keepdim=True).values,
                luma_gradient(functional, hazy_luma),
                luma_gradient(functional, prediction_luma),
                luma_gradient(functional, residual_luma),
            ], dim=1)
            if handcrafted.shape[1] != 20 or decoder_tensor.shape[1] != 32:
                raise RuntimeError("frozen feature dimensionality changed")
            hand_mean, hand_std, areas = tile_moments(
                torch, functional, handcrafted, height, width,
            )
            decoder_mean, decoder_std, decoder_areas = tile_moments(
                torch, functional, decoder_tensor, height, width,
            )
            if not torch.equal(areas, decoder_areas):
                raise RuntimeError("handcrafted and decoder tile grids differ")
            features = torch.cat([hand_mean, hand_std, decoder_mean, decoder_std], dim=1)
            if features.shape[1] != FEATURE_CHANNELS:
                raise RuntimeError("observable feature channel count changed")
            keep_error = (prediction_tensor - clear_tensor) ** 2
            keep_sums, _, error_areas = tile_moments(
                torch, functional, keep_error, height, width,
            )
            if not torch.equal(areas, error_areas):
                raise RuntimeError("keep-error tile grid differs")
            keep_mse = keep_sums.mean(dim=1, keepdim=True)
            utilities = []
            for scale in ACTION_SCALES:
                candidate = torch.clamp(
                    hazy_tensor + float(scale) * (prediction_tensor - hazy_tensor),
                    0.0,
                    1.0,
                )
                candidate_error = (candidate - clear_tensor) ** 2
                candidate_sums, _, candidate_areas = tile_moments(
                    torch, functional, candidate_error, height, width,
                )
                if not torch.equal(areas, candidate_areas):
                    raise RuntimeError("candidate-error tile grid differs")
                candidate_mse = candidate_sums.mean(dim=1, keepdim=True)
                utility = 10.0 * torch.log10(
                    torch.clamp(keep_mse, min=EPSILON)
                    / torch.clamp(candidate_mse, min=EPSILON)
                )
                utilities.append(utility)
            utility = torch.cat(utilities, dim=1).clamp(-TARGET_CLIP_DB, TARGET_CLIP_DB)
            harm = (utility <= -PSNR_HARM_MARGIN_DB).to(dtype=torch.float32)
            arrays = {
                "features": features.squeeze(0).detach().cpu().numpy().astype(np.float32),
                "utility": utility.squeeze(0).detach().cpu().numpy().astype(np.float32),
                "harm": harm.squeeze(0).detach().cpu().numpy().astype(np.float32),
                "areas": areas.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32),
                "prediction": prediction_tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy().astype(np.float32),
            }
            if not all(np.isfinite(array).all() for array in arrays.values()):
                raise RuntimeError("extracted observable tensors are non-finite")
            return arrays
    finally:
        handle.remove()


def make_head(torch):
    import torch.nn as nn

    return nn.Sequential(
        nn.Conv2d(FEATURE_CHANNELS, 64, kernel_size=3, padding=1),
        nn.GroupNorm(8, 64),
        nn.GELU(),
        nn.Conv2d(64, 64, kernel_size=3, padding=1),
        nn.GroupNorm(8, 64),
        nn.GELU(),
        nn.Conv2d(64, 4, kernel_size=1),
    )


def load_cache(path: Path, *, include_prediction: bool = False) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        names = ["features", "utility", "harm", "areas"]
        if include_prediction:
            names.append("prediction")
        result = {name: source[name].copy() for name in names}
    return result


def feature_normalization(records: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    scene_means = []
    for record in records:
        features, areas = record["features"], record["areas"].astype(np.float64)
        weights = areas / float(np.sum(areas))
        scene_means.append(np.sum(features.astype(np.float64) * weights[None, :, :], axis=(1, 2)))
    mean = np.mean(np.stack(scene_means), axis=0)
    scene_variances = []
    for record in records:
        features, areas = record["features"].astype(np.float64), record["areas"].astype(np.float64)
        weights = areas / float(np.sum(areas))
        scene_variances.append(np.sum((features - mean[:, None, None]) ** 2 * weights[None, :, :], axis=(1, 2)))
    scale = np.sqrt(np.maximum(np.mean(np.stack(scene_variances), axis=0), 1.0e-8))
    return mean.astype(np.float32), scale.astype(np.float32)


def normalized_tensor(torch, record: dict[str, Any], mean: np.ndarray, scale: np.ndarray, device: str):
    value = (record["features"] - mean[:, None, None]) / scale[:, None, None]
    return torch.from_numpy(value.copy()).unsqueeze(0).to(device=device, dtype=torch.float32)


def train_head(torch, records: list[dict[str, Any]], *, seed: int, device: str):
    import torch.nn.functional as functional

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    head = make_head(torch).to(device)
    mean, scale = feature_normalization(records)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
    )
    history = []
    for epoch in range(EPOCHS):
        order = np.random.default_rng(seed + epoch).permutation(len(records))
        total_loss = 0.0
        for index in order:
            record = records[int(index)]
            x = normalized_tensor(torch, record, mean, scale, device)
            target = torch.from_numpy(record["utility"].copy()).unsqueeze(0).to(device)
            harm = torch.from_numpy(record["harm"].copy()).unsqueeze(0).to(device)
            area = torch.from_numpy(record["areas"].copy()).to(device)
            area = area / torch.sum(area)
            output = head(x)
            utility_loss = functional.smooth_l1_loss(
                output[:, :2], target, beta=UTILITY_HUBER_BETA_DB, reduction="none",
            ).mean(dim=1)
            risk_loss = functional.binary_cross_entropy_with_logits(
                output[:, 2:], harm, reduction="none",
            )
            risk_loss = risk_loss * torch.where(
                harm > 0.5,
                torch.full_like(harm, RISK_POSITIVE_WEIGHT),
                torch.ones_like(harm),
            )
            risk_loss = risk_loss.mean(dim=1)
            loss = torch.sum(area * utility_loss.squeeze(0)) \
                + RISK_LOSS_WEIGHT * torch.sum(area * risk_loss.squeeze(0))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), GRADIENT_CLIP_NORM)
            optimizer.step()
            total_loss += float(loss.detach().cpu().item())
        history.append(total_loss / len(records))
    return head.eval(), mean, scale, history


def predict_head(torch, head, record, mean, scale, device: str) -> tuple[np.ndarray, np.ndarray]:
    with torch.inference_mode():
        output = head(normalized_tensor(torch, record, mean, scale, device))
        utility = output[:, :2].squeeze(0).detach().cpu().numpy().astype(np.float64)
        risk = torch.sigmoid(output[:, 2:]).squeeze(0).detach().cpu().numpy().astype(np.float64)
    if not np.isfinite(utility).all() or not np.isfinite(risk).all():
        raise RuntimeError("contextual head produced non-finite predictions")
    return utility, risk


def upper_quantile(values: Iterable[float], probability: float) -> float:
    array = np.sort(np.asarray(list(values), dtype=np.float64))
    if array.size == 0 or not np.isfinite(array).all():
        raise RuntimeError("calibration quantile requires finite scene scores")
    index = min(array.size - 1, max(0, math.ceil((array.size + 1) * probability) - 1))
    return float(array[index])


def calibrate_head(torch, head, records, mean, scale, device: str) -> dict[str, Any]:
    utility_scores = [[], []]
    risk_scores = [[], []]
    for record in records:
        utility, risk = predict_head(torch, head, record, mean, scale, device)
        weights = record["areas"].astype(np.float64)
        weights /= float(np.sum(weights))
        for action in range(2):
            utility_residual = float(np.sum(weights * (utility[action] - record["utility"][action])))
            risk_residual = float(np.sum(weights * (record["harm"][action] - risk[action])))
            utility_scores[action].append(max(0.0, utility_residual))
            risk_scores[action].append(max(0.0, risk_residual))
    utility_correction = np.asarray([
        upper_quantile(scores, CALIBRATION_QUANTILE) for scores in utility_scores
    ], dtype=np.float64)
    risk_correction = np.asarray([
        upper_quantile(scores, CALIBRATION_QUANTILE) for scores in risk_scores
    ], dtype=np.float64)
    return {
        "utility_correction": utility_correction,
        "risk_correction": risk_correction,
        "scene_count": len(records),
        "quantile": CALIBRATION_QUANTILE,
    }


def corrected_predictions(utility, risk, calibration) -> tuple[np.ndarray, np.ndarray]:
    utility_lower = utility - calibration["utility_correction"][:, None, None]
    risk_upper = np.clip(risk + calibration["risk_correction"][:, None, None], 0.0, 1.0)
    return utility_lower, risk_upper


def choose_actions(utility_lower: np.ndarray, risk_upper: np.ndarray) -> np.ndarray:
    eligible = (utility_lower > UTILITY_MARGIN_DB) & (risk_upper <= RISK_ACCEPTANCE_MAX)
    scores = np.where(eligible, utility_lower, -np.inf)
    best = np.argmax(scores, axis=0)
    any_eligible = np.any(eligible, axis=0)
    return np.where(any_eligible, best + 1, 0).astype(np.int64)


def choose_uniform_action(
    utility_lower: np.ndarray, risk_upper: np.ndarray, areas: np.ndarray,
) -> int:
    weights = areas.astype(np.float64) / float(np.sum(areas))
    utility = np.sum(utility_lower * weights[None, :, :], axis=(1, 2))
    risk = np.sum(risk_upper * weights[None, :, :], axis=(1, 2))
    eligible = (utility > UTILITY_MARGIN_DB) & (risk <= RISK_ACCEPTANCE_MAX)
    if not np.any(eligible):
        return 0
    return int(np.argmax(np.where(eligible, utility, -np.inf)) + 1)


def action_scale(action: int) -> float:
    return 1.0 if action == 0 else float(ACTION_SCALES[action - 1])


def apply_uniform(hazy: np.ndarray, prediction: np.ndarray, action: int) -> np.ndarray:
    if action == 0:
        return prediction.copy()
    return np.clip(
        hazy + action_scale(action) * (prediction - hazy), 0.0, 1.0,
    ).astype(np.float32)


def apply_spatial(hazy: np.ndarray, prediction: np.ndarray, actions: np.ndarray) -> np.ndarray:
    height, width = hazy.shape[:2]
    expected = (math.ceil(height / TILE_SIZE), math.ceil(width / TILE_SIZE))
    if actions.shape != expected:
        raise RuntimeError(f"action grid {actions.shape} differs from image grid {expected}")
    output = prediction.copy()
    for row in range(actions.shape[0]):
        for column in range(actions.shape[1]):
            y0, y1 = row * TILE_SIZE, min(height, (row + 1) * TILE_SIZE)
            x0, x1 = column * TILE_SIZE, min(width, (column + 1) * TILE_SIZE)
            action = int(actions[row, column])
            if action:
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
    channel_bias = np.mean(value.astype(np.float64) - target.astype(np.float64), axis=(0, 1))
    return float(np.mean(np.abs(channel_bias)))


def rgb_ssim(torch, values: list[np.ndarray], target: np.ndarray, device: str) -> list[float]:
    import torch.nn.functional as functional

    stack = np.stack(values).transpose(0, 3, 1, 2).copy()
    reference = np.repeat(target[None, ...], len(values), axis=0).transpose(0, 3, 1, 2).copy()
    x = torch.from_numpy(stack).to(device=device, dtype=torch.float32)
    y = torch.from_numpy(reference).to(device=device, dtype=torch.float32)

    def local_mean(item):
        return functional.avg_pool2d(
            functional.pad(item, (5, 5, 5, 5), mode="reflect"), 11, stride=1,
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
    result = scores.mean(dim=(1, 2, 3)).detach().cpu().numpy().astype(np.float64)
    if not np.isfinite(result).all():
        raise RuntimeError("non-finite RGB SSIM")
    return [float(item) for item in result]


def evaluate_extracted(
    torch,
    head,
    record,
    mean,
    scale,
    calibration,
    hazy,
    clear,
    prediction,
    scene: str,
    variant: str,
    device: str,
) -> dict[str, Any]:
    utility, risk = predict_head(torch, head, record, mean, scale, device)
    utility_lower, risk_upper = corrected_predictions(utility, risk, calibration)
    actions = choose_actions(utility_lower, risk_upper)
    uniform_action = choose_uniform_action(utility_lower, risk_upper, record["areas"])
    shuffled = np.random.default_rng(
        deterministic_seed(f"{SHUFFLE_SALT}|{scene}|{variant}"),
    ).permutation(actions.reshape(-1)).reshape(actions.shape)
    spatial_output = apply_spatial(hazy, prediction, actions)
    uniform_output = apply_uniform(hazy, prediction, uniform_action)
    shuffled_output = apply_spatial(hazy, prediction, shuffled)
    keep_psnr = psnr(prediction, clear)
    spatial_psnr = psnr(spatial_output, clear)
    uniform_psnr = psnr(uniform_output, clear)
    shuffled_psnr = psnr(shuffled_output, clear)
    keep_ssim, spatial_ssim = rgb_ssim(
        torch, [prediction, spatial_output], clear, device,
    )
    keep_color = color_bias(prediction, clear)
    spatial_color = color_bias(spatial_output, clear)
    counts = Counter(int(item) for item in actions.reshape(-1))
    total = int(actions.size)
    return {
        "spatial_minus_keep_psnr_db": spatial_psnr - keep_psnr,
        "spatial_minus_uniform_psnr_db": spatial_psnr - uniform_psnr,
        "spatial_minus_shuffled_psnr_db": spatial_psnr - shuffled_psnr,
        "spatial_minus_keep_ssim": spatial_ssim - keep_ssim,
        "spatial_minus_keep_color_bias": spatial_color - keep_color,
        "keep_fraction": counts.get(0, 0) / total,
        "weaken_fraction": counts.get(1, 0) / total,
        "strengthen_fraction": counts.get(2, 0) / total,
    }


def mean_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = tuple(rows[0])
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def paired_bootstrap(values: Iterable[float], *, seed: int) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size < 2 or not np.isfinite(array).all():
        raise RuntimeError("paired bootstrap requires finite independent-scene values")
    generator = np.random.default_rng(seed)
    draws = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    for start in range(0, BOOTSTRAP_RESAMPLES, 500):
        stop = min(start + 500, BOOTSTRAP_RESAMPLES)
        indices = generator.integers(0, array.size, size=(stop - start, array.size))
        draws[start:stop] = np.mean(array[indices], axis=1)
    estimate = float(np.mean(array))
    lower = float(np.quantile(draws, 0.025))
    upper = float(np.quantile(draws, 0.975))
    return {
        "scene_count": int(array.size),
        "estimate": estimate,
        "lower": lower,
        "upper": upper,
        "max_half_width": max(estimate - lower, upper - estimate),
        "resamples": BOOTSTRAP_RESAMPLES,
        "seed": seed,
    }


def wilson(successes: int, total: int) -> dict[str, Any]:
    if total <= 0 or not 0 <= successes <= total:
        raise RuntimeError("invalid Wilson interval inputs")
    z = 1.959963984540054
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


def summarize_population(rows: list[dict[str, Any]], *, seed_offset: int) -> dict[str, Any]:
    utility = paired_bootstrap(
        (row["spatial_minus_keep_psnr_db"] for row in rows),
        seed=BOOTSTRAP_SEED + seed_offset,
    )
    uniform = paired_bootstrap(
        (row["spatial_minus_uniform_psnr_db"] for row in rows),
        seed=BOOTSTRAP_SEED + seed_offset + 1,
    )
    shuffled = paired_bootstrap(
        (row["spatial_minus_shuffled_psnr_db"] for row in rows),
        seed=BOOTSTRAP_SEED + seed_offset + 2,
    )
    material = wilson(
        sum(row["spatial_minus_keep_psnr_db"] >= UTILITY_MARGIN_DB for row in rows),
        len(rows),
    )
    harms = {
        "psnr": wilson(
            sum(row["spatial_minus_keep_psnr_db"] <= -PSNR_HARM_MARGIN_DB for row in rows),
            len(rows),
        ),
        "ssim": wilson(
            sum(row["spatial_minus_keep_ssim"] <= -SSIM_HARM_MARGIN for row in rows),
            len(rows),
        ),
        "color": wilson(
            sum(row["spatial_minus_keep_color_bias"] >= COLOR_HARM_MARGIN for row in rows),
            len(rows),
        ),
    }
    return {
        "scene_count": len(rows),
        "spatial_minus_keep_psnr_db": utility,
        "spatial_minus_uniform_psnr_db": uniform,
        "spatial_minus_shuffled_psnr_db": shuffled,
        "material_scene_prevalence": material,
        "harm_prevalence": harms,
        "action_fractions": {
            key: float(np.mean([row[key] for row in rows]))
            for key in ("keep_fraction", "weaken_fraction", "strengthen_fraction")
        },
    }


def state_dict_digest(head) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(head.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def evidence_identity(context) -> dict[str, bool]:
    assets = {
        "role_ledger_closeout": (ROLE_LEDGER_CLOSEOUT_SHA256, "CONVIR_EXISTING_DATASET_ROLE_LEDGER_PASS"),
        "train_split_closeout": (TRAIN_SPLIT_CLOSEOUT_SHA256, "HAZE4K_TRAIN_SCENE_SPLIT_PASS"),
        "test_split_closeout": (TEST_SPLIT_CLOSEOUT_SHA256, "HAZE4K_TEST_SCENE_SPLIT_PASS"),
        "prior_proxy_closeout": (PROXY_CLOSEOUT_SHA256, "HAZE4K_TEST_LOCAL_ACTION_PROXY_PREDICTABILITY_FAIL"),
        "prior_oracle_closeout": (ORACLE_CLOSEOUT_SHA256, "HAZE4K_TEST_BOUNDED_LOCAL_ACTION_ORACLE_HEADROOM_PASS"),
    }
    checks = {}
    for identifier, (expected_sha, expected_decision) in assets.items():
        path = asset_path(context, identifier, kind="file")
        checks[identifier] = (
            sha256_file(path) == expected_sha
            and read_json(path).get("decision") == expected_decision
        )
    checks.update({
        "role_ledger_summary": sha256_file(asset_path(context, "role_ledger_summary", kind="file")) == ROLE_LEDGER_SUMMARY_SHA256,
        "train_split_summary": sha256_file(asset_path(context, "train_split_summary", kind="file")) == TRAIN_SPLIT_SUMMARY_SHA256,
        "test_split_summary": sha256_file(asset_path(context, "test_split_summary", kind="file")) == TEST_SPLIT_SUMMARY_SHA256,
        "prior_proxy_summary": sha256_file(asset_path(context, "prior_proxy_summary", kind="file")) == PROXY_SUMMARY_SHA256,
        "prior_oracle_summary": sha256_file(asset_path(context, "prior_oracle_summary", kind="file")) == ORACLE_SUMMARY_SHA256,
        "parent_authorization": read_json(
            asset_path(context, "role_ledger_closeout", kind="file"),
        ).get("authorizes") == "UNIFIED_DEVELOPMENT_CONTRACT_AUTHORING_ONLY",
    })
    return checks


def contract(context_path: Path) -> None:
    context = load_context(context_path, "contract")
    prepare_phase_output(context)
    import torch

    checks = evidence_identity(context)
    runtime_environment = asset_path(context, "runtime_environment", kind="file")
    entrypoint = asset_path(context, "observable_entrypoint", kind="file")
    anchor = context.assets.get("official_anchor_checkout")
    checks.update({
        "entrypoint_identity": (
            context.assets["observable_entrypoint"].sha256 == sha256_file(entrypoint)
        ),
        "runtime_environment_identity": sha256_file(runtime_environment) == RUNTIME_ENVIRONMENT_SHA256,
        "runtime_environment_contract": read_json(runtime_environment).get("device_class") == "cuda_sm89",
        "official_anchor_identity": anchor is not None and anchor.commit == ANCHOR_COMMIT,
        "protected_datasets_absent": (
            "haze4k_train" not in context.assets
            and "haze4k_test_development" not in context.assets
            and "candidate_confirmation" not in context.assets
        ),
        "fixed_cost_contract": (
            context.engineering_contract["cost_contract"]["formal_iterations"] == FORMAL_COST_ITERATIONS
            and context.engineering_contract["cost_contract"]["probe_iterations"] == PROBE_COST_ITERATIONS
        ),
    })
    torch, model = load_official_model(context)
    generator = np.random.default_rng(TRAINING_SEED)
    hazy = generator.uniform(0.05, 0.95, size=(256, 320, 3)).astype(np.float32)
    clear = np.clip(
        hazy + generator.normal(scale=0.03, size=hazy.shape), 0.0, 1.0,
    ).astype(np.float32)
    extracted = None
    official_iterations = 5
    for index in range(official_iterations):
        extracted = extract_variant(torch, model, hazy, clear, context.device)
        write_contract_progress(
            context,
            completed_iterations=index + 1,
            total_iterations=PROBE_COST_ITERATIONS,
            stage="synthetic_official_feature_path",
        )
    assert extracted is not None
    fixture_record = {
        key: extracted[key] for key in ("features", "utility", "harm", "areas")
    }
    mean = np.zeros(FEATURE_CHANNELS, dtype=np.float32)
    scale = np.ones(FEATURE_CHANNELS, dtype=np.float32)
    head = make_head(torch).to(context.device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    import torch.nn.functional as functional

    training_iterations = 148
    for index in range(training_iterations):
        output = head(normalized_tensor(torch, fixture_record, mean, scale, context.device))
        target = torch.from_numpy(fixture_record["utility"]).unsqueeze(0).to(context.device)
        harm = torch.from_numpy(fixture_record["harm"]).unsqueeze(0).to(context.device)
        loss = functional.smooth_l1_loss(output[:, :2], target, beta=UTILITY_HUBER_BETA_DB) \
            + RISK_LOSS_WEIGHT * functional.binary_cross_entropy_with_logits(output[:, 2:], harm)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), GRADIENT_CLIP_NORM)
        optimizer.step()
        if (index + 1) % 37 == 0:
            write_contract_progress(
                context,
                completed_iterations=official_iterations + index + 1,
                total_iterations=PROBE_COST_ITERATIONS,
                stage="synthetic_context_head_path",
            )
    evaluation_iterations = PROBE_COST_ITERATIONS - official_iterations - training_iterations
    calibration = {
        "utility_correction": np.zeros(2, dtype=np.float64),
        "risk_correction": np.zeros(2, dtype=np.float64),
    }
    finite_eval = True
    for _ in range(evaluation_iterations):
        utility, risk = predict_head(
            torch, head.eval(), fixture_record, mean, scale, context.device,
        )
        lower, upper = corrected_predictions(utility, risk, calibration)
        actions = choose_actions(lower, upper)
        output = apply_spatial(hazy, extracted["prediction"], actions)
        finite_eval = finite_eval and bool(np.isfinite(output).all())
    checks.update({
        "official_graph_strict_loaded": sum(parameter.numel() for parameter in model.parameters()) == PARAMETER_COUNT,
        "observable_feature_shape": extracted["features"].shape[0] == FEATURE_CHANNELS,
        "continuous_target_shape": extracted["utility"].shape[0] == 2,
        "risk_target_shape": extracted["harm"].shape[0] == 2,
        "contextual_head_output_shape": tuple(output.shape) == tuple(hazy.shape),
        "finite_synthetic_replay": finite_eval,
        "probe_iteration_count": official_iterations + training_iterations + evaluation_iterations == PROBE_COST_ITERATIONS,
    })
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
        },
    )


def run(context_path: Path) -> None:
    context = load_context(context_path, "run")
    prepare_phase_output(context)
    if context.total_units != TOTAL_UNITS or context.evidence_role != "development_screening" \
            or any(context.protected_data_permissions.values()):
        raise RuntimeError("runtime role, unit, or protected-data contract changed")
    identity_checks = evidence_identity(context)
    train_root = asset_path(context, "haze4k_train", kind="directory")
    test_root = asset_path(context, "haze4k_test_development", kind="directory")
    if train_root.name.lower() != "train" or "candidate_confirmation" in str(test_root):
        raise RuntimeError("development dataset scope changed")
    write_workload_progress(context, completed_units=0, stage="identity_and_scope")
    train_groups, training_scenes, calibration_scenes, assignment = enumerate_train_groups(train_root)
    role_checks = {
        "train_assignment_digest": assignment == TRAIN_ASSIGNMENT_DIGEST,
        "train_role_counts": len(training_scenes) == TRAINING_SCENES and len(calibration_scenes) == CALIBRATION_SCENES,
        "train_role_disjointness": not (set(training_scenes) & set(calibration_scenes)),
        "one_outcome_blind_variant_per_train_scene": True,
        "test_assignment_digest": read_json(
            asset_path(context, "test_split_summary", kind="file"),
        ).get("frozen_split", {}).get("assignment_digest") == TEST_ASSIGNMENT_DIGEST,
        "candidate_confirmation_absent": "candidate_confirmation" not in context.assets,
    }
    torch, model = load_official_model(context)
    cache_root = output_file(context, "scene_cache")
    cache_root.mkdir()
    extraction_records: dict[str, dict[str, Any]] = {}
    extraction_identities = []
    ordered_train_scenes = training_scenes + calibration_scenes
    for completed, scene in enumerate(ordered_train_scenes, start=1):
        hazy_path, clear_path = choose_one_variant(scene, train_groups[scene])
        hazy, clear = image_array(hazy_path), image_array(clear_path)
        if hazy.shape != clear.shape:
            raise RuntimeError(f"selected train pair dimensions differ for {hazy_path.name}")
        extracted = extract_variant(torch, model, hazy, clear, context.device)
        role = "training" if scene in set(training_scenes) else "calibration"
        relative = f"scene_cache/{role}_{scene[:24]}.npz"
        cache_path = output_file(context, relative)
        np.savez(cache_path, **extracted)
        input_identity = sha256_text(
            "|".join([
                "observable-extraction-v1",
                scene,
                sha256_file(hazy_path),
                sha256_file(clear_path),
                hazy_path.name,
            ])
        )
        cache_identity = sha256_file(cache_path)
        record_completed_unit(
            context,
            unit_id=f"extract_{completed:04d}_{scene[:16]}",
            input_sha256=input_identity,
            output_relpath=relative,
        )
        extraction_identities.append(f"{input_identity}:{cache_identity}")
        extraction_records[scene] = {
            "scene": scene,
            "role": role,
            "cache_path": cache_path,
            "hazy_path": hazy_path,
            "clear_path": clear_path,
            **{key: extracted[key] for key in ("features", "utility", "harm", "areas")},
        }
        if completed % 10 == 0 or completed == EXPECTED_TRAIN_SCENES:
            write_workload_progress(
                context,
                completed_units=completed,
                stage="outcome_blind_scene_extraction",
            )
    role_checks["complete_train_extraction"] = len(extraction_records) == EXPECTED_TRAIN_SCENES

    fold_order = sorted(
        training_scenes,
        key=lambda scene: (sha256_text(f"{OOF_FOLD_SALT}|{scene}"), scene),
    )
    fold_assignment = {scene: index % OOF_FOLDS for index, scene in enumerate(fold_order)}
    role_checks["balanced_oof_folds"] = Counter(fold_assignment.values()) == {fold: 120 for fold in range(OOF_FOLDS)}
    calibration_records = [extraction_records[scene] for scene in calibration_scenes]
    primary_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    for fold in range(OOF_FOLDS):
        train_records = [
            extraction_records[scene] for scene in training_scenes
            if fold_assignment[scene] != fold
        ]
        eval_records = [
            extraction_records[scene] for scene in training_scenes
            if fold_assignment[scene] == fold
        ]
        head, mean, scale, history = train_head(
            torch,
            train_records,
            seed=TRAINING_SEED + fold,
            device=context.device,
        )
        calibration = calibrate_head(
            torch, head, calibration_records, mean, scale, context.device,
        )
        for record in eval_records:
            cached = load_cache(record["cache_path"], include_prediction=True)
            hazy, clear = image_array(record["hazy_path"]), image_array(record["clear_path"])
            metrics = evaluate_extracted(
                torch,
                head,
                record,
                mean,
                scale,
                calibration,
                hazy,
                clear,
                cached["prediction"],
                record["scene"],
                record["hazy_path"].name,
                context.device,
            )
            primary_rows.append({
                "population": "haze4k_train_oof",
                "scene": record["scene"],
                "fold": fold,
                **metrics,
            })
        training_rows.append({
            "model": f"oof_fold_{fold}",
            "train_scenes": len(train_records),
            "calibration_scenes": len(calibration_records),
            "evaluation_scenes": len(eval_records),
            "epochs": EPOCHS,
            "final_loss": history[-1],
            "utility_correction_weaken": calibration["utility_correction"][0],
            "utility_correction_strengthen": calibration["utility_correction"][1],
            "risk_correction_weaken": calibration["risk_correction"][0],
            "risk_correction_strengthen": calibration["risk_correction"][1],
        })
    role_checks["complete_oof_replay"] = len(primary_rows) == TRAINING_SCENES \
        and len({row["scene"] for row in primary_rows}) == TRAINING_SCENES
    final_train_records = [extraction_records[scene] for scene in training_scenes]
    final_head, final_mean, final_scale, final_history = train_head(
        torch,
        final_train_records,
        seed=TRAINING_SEED + OOF_FOLDS,
        device=context.device,
    )
    final_calibration = calibrate_head(
        torch,
        final_head,
        calibration_records,
        final_mean,
        final_scale,
        context.device,
    )
    final_head_sha = state_dict_digest(final_head)
    training_rows.append({
        "model": "frozen_test_development_candidate",
        "train_scenes": len(final_train_records),
        "calibration_scenes": len(calibration_records),
        "evaluation_scenes": EXPECTED_TEST_SCENES,
        "epochs": EPOCHS,
        "final_loss": final_history[-1],
        "utility_correction_weaken": final_calibration["utility_correction"][0],
        "utility_correction_strengthen": final_calibration["utility_correction"][1],
        "risk_correction_weaken": final_calibration["risk_correction"][0],
        "risk_correction_strengthen": final_calibration["risk_correction"][1],
    })
    training_unit_path = output_file(context, "training_and_calibration_unit.json")
    atomic_json(training_unit_path, {
        "schema_version": 1,
        "oof_folds": OOF_FOLDS,
        "epochs": EPOCHS,
        "training_scenes": TRAINING_SCENES,
        "calibration_scenes": CALIBRATION_SCENES,
        "head_sha256": final_head_sha,
        "architecture": "conv104x64_gn_gelu_conv64x64_gn_gelu_conv64x4",
        "thresholds": {
            "utility_margin_db": UTILITY_MARGIN_DB,
            "risk_acceptance_max": RISK_ACCEPTANCE_MAX,
        },
    })
    training_input_sha = sha256_text(
        "training-calibration-v1|" + "|".join(sorted(extraction_identities)),
    )
    record_completed_unit(
        context,
        unit_id="training_and_calibration",
        input_sha256=training_input_sha,
        output_relpath="training_and_calibration_unit.json",
    )
    write_workload_progress(
        context,
        completed_units=EXPECTED_TRAIN_SCENES + 1,
        stage="oof_training_calibration_and_replay",
    )

    test_groups = enumerate_test_groups(test_root)
    test_rows: list[dict[str, Any]] = []
    test_root_output = output_file(context, "test_development_replay")
    test_root_output.mkdir()
    for index, scene in enumerate(sorted(test_groups), start=1):
        variant_rows = []
        input_parts = ["test-development-replay-v1", scene, final_head_sha]
        for hazy_path, clear_path in sorted(test_groups[scene]):
            hazy, clear = image_array(hazy_path), image_array(clear_path)
            extracted = extract_variant(torch, model, hazy, clear, context.device)
            record = {
                key: extracted[key] for key in ("features", "utility", "harm", "areas")
            }
            variant_rows.append(evaluate_extracted(
                torch,
                final_head,
                record,
                final_mean,
                final_scale,
                final_calibration,
                hazy,
                clear,
                extracted["prediction"],
                scene,
                hazy_path.name,
                context.device,
            ))
            input_parts.extend([hazy_path.name, sha256_file(hazy_path), sha256_file(clear_path)])
        metrics = mean_metrics(variant_rows)
        row = {
            "population": "haze4k_test_development_stress",
            "scene": scene,
            "fold": "frozen_candidate",
            **metrics,
        }
        test_rows.append(row)
        relative = f"test_development_replay/scene_{scene[:24]}.json"
        atomic_json(output_file(context, relative), {
            "schema_version": 1,
            "scene": scene,
            "nested_variants": VARIANTS_PER_SCENE,
            "metrics": metrics,
        })
        record_completed_unit(
            context,
            unit_id=f"test_replay_{index:03d}_{scene[:16]}",
            input_sha256=sha256_text("|".join(input_parts)),
            output_relpath=relative,
        )
        if index % 5 == 0 or index == EXPECTED_TEST_SCENES:
            write_workload_progress(
                context,
                completed_units=EXPECTED_TRAIN_SCENES + 1 + index,
                stage="frozen_test_development_stress",
            )
    role_checks["complete_test_development_stress"] = len(test_rows) == EXPECTED_TEST_SCENES
    role_checks["four_variants_nested_within_test_scene"] = all(
        len(items) == VARIANTS_PER_SCENE for items in test_groups.values()
    )
    completed_unit_ledger = load_completed_unit_ledger(context)
    role_checks["completed_unit_ledger_coverage"] = (
        len(completed_unit_ledger) == TOTAL_UNITS
    )

    primary = summarize_population(primary_rows, seed_offset=0)
    transfer = summarize_population(test_rows, seed_offset=10)
    materiality_checks = {
        "oof_utility_lcb_above_margin": primary["spatial_minus_keep_psnr_db"]["lower"] > UTILITY_MARGIN_DB,
        "oof_material_prevalence_lcb": primary["material_scene_prevalence"]["lower"] > MIN_MATERIAL_SCENE_PREVALENCE,
        "oof_spatial_over_uniform_lcb": primary["spatial_minus_uniform_psnr_db"]["lower"] > 0.0,
        "oof_spatial_over_shuffle_lcb": primary["spatial_minus_shuffled_psnr_db"]["lower"] > 0.0,
        "test_development_transfer_point": transfer["spatial_minus_keep_psnr_db"]["estimate"] >= TRANSFER_POINT_MARGIN_DB,
    }
    safety_checks = {}
    for population, summary in (("oof", primary), ("test_development", transfer)):
        for metric, interval in summary["harm_prevalence"].items():
            safety_checks[f"{population}_{metric}_harm_ucb"] = interval["upper"] <= MAX_HARM_PREVALENCE
    precision_met = primary["spatial_minus_keep_psnr_db"]["max_half_width"] <= 0.025
    identity_pass = all(identity_checks.values())
    role_pass = all(role_checks.values())
    materiality_pass = all(materiality_checks.values())
    safety_pass = all(safety_checks.values())
    gate_outcomes = {
        "evidence_identity": "pass" if identity_pass else "fail",
        "scene_role_and_coverage": "pass" if role_pass else "fail",
        "utility_and_spatial_specificity": "favorable" if materiality_pass else "unfavorable",
        "image_replay_safety": "safe" if safety_pass else "unsafe",
        "precision": "met" if precision_met else "unmet",
    }
    summary = {
        "schema_version": 1,
        "route_id": context.route_id,
        "operation_id": context.operation_id,
        "run_id": context.run_id,
        "scope": "development-only observable contextual utility-risk qualification",
        "independent_unit": "original_clear_scene",
        "data_roles": {
            "haze4k_train_training": TRAINING_SCENES,
            "haze4k_train_fixed_calibration": CALIBRATION_SCENES,
            "haze4k_test_development_stress": EXPECTED_TEST_SCENES,
            "haze4k_test_candidate_confirmation_touched": False,
            "nh_haze_touched": False,
            "reside_its_ots_touched": False,
        },
        "head_contract": {
            "inputs": "hazy image, official output, and frozen Decoder[2] features only",
            "architecture": "Conv2d(104,64,3)-GroupNorm-GELU-Conv2d(64,64,3)-GroupNorm-GELU-Conv2d(64,4,1)",
            "outputs": "weaken/strengthen continuous utility and harm logits",
            "action_scales": {"keep": 1.0, "weaken": 0.8, "strengthen": 1.2},
            "target_clip_db": TARGET_CLIP_DB,
            "utility_margin_db": UTILITY_MARGIN_DB,
            "risk_acceptance_max": RISK_ACCEPTANCE_MAX,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip_norm": GRADIENT_CLIP_NORM,
            "calibration": "95 percent finite-sample upper quantile of nonnegative scene-mean utility and risk residuals",
            "abstention": "keep unless corrected utility exceeds 0.05 dB and corrected risk is at most 0.10",
        },
        "identity_checks": identity_checks,
        "role_and_coverage_checks": role_checks,
        "materiality_checks": materiality_checks,
        "safety_checks": safety_checks,
        "primary_haze4k_train_oof": primary,
        "secondary_haze4k_test_development_stress": transfer,
        "precision": {
            "target_half_width_db": 0.025,
            "observed_max_half_width_db": primary["spatial_minus_keep_psnr_db"]["max_half_width"],
            "met": precision_met,
        },
        "gate_outcomes": gate_outcomes,
        "limitations": [
            "All evidence is development-only; no protected candidate-confirmation scene was delivered or read.",
            "The 600-scene OOF estimand uses one outcome-blind selected haze variant per original clear scene.",
            "The four test-development haze variants are nested repeats and are averaged within each original clear scene.",
            "The fixed Haze4K calibration set is shared across OOF folds as a nuisance calibration resource and is not counted in the primary estimand.",
            "A PASS authorizes only an ITS/OTS cross-domain contract, not measurement execution, promotion, deployment, or a source-disjoint claim.",
        ],
        "marker": "HAZE4K_OBSERVABLE_CONDITIONAL_UTILITY_RISK_FEASIBILITY_COMPLETE",
    }
    summary_path = output_file(context, "haze4k_observable_conditional_utility_risk_feasibility_summary.json")
    metrics_path = output_file(context, "haze4k_observable_conditional_utility_risk_scene_metrics.csv")
    training_path = output_file(context, "haze4k_observable_conditional_utility_risk_training_calibration.csv")
    gate_path = output_file(context, "haze4k_observable_conditional_utility_risk_gate_summary.json")
    atomic_json(summary_path, summary)
    write_csv(metrics_path, primary_rows + test_rows)
    write_csv(training_path, training_rows)
    atomic_json(gate_path, {
        "schema_version": 1,
        "gate_outcomes": gate_outcomes,
        "materiality_checks": materiality_checks,
        "safety_checks": safety_checks,
        "precision_met": precision_met,
    })
    write_gate_result(
        context,
        gate_outcomes=gate_outcomes,
        details={
            "independent_oof_scenes": len(primary_rows),
            "fixed_calibration_scenes": len(calibration_records),
            "test_development_stress_scenes": len(test_rows),
            "nested_test_variants": len(test_rows) * VARIANTS_PER_SCENE,
            "completed_unit_ledger_count": len(completed_unit_ledger),
            "candidate_confirmation_touched": False,
            "network_training_occurred": True,
            "summary_file": summary_path.name,
            "scene_metrics_file": metrics_path.name,
            "training_calibration_file": training_path.name,
            "gate_summary_file": gate_path.name,
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
