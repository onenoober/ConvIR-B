#!/usr/bin/env python3
"""Decompose observable signal, calibration, and risk on development-only Haze4K."""

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
POLICY_IDS = ("P00", "P10", "P01", "P11")
TRAIN_SPLIT_SALT = "haze4k-local-error-qualification-v2"
VARIANT_SELECTION_SALT = "haze4k-observable-conditional-utility-risk-feasibility-v1"
OOF_FOLD_SALT = "haze4k-observable-conditional-utility-risk-feasibility-v1-oof"
SHUFFLE_SALT = "haze4k-observable-conditional-utility-risk-feasibility-v1-shuffle"
PERMUTATION_SALT = "haze4k-observable-signal-calibration-decomposition-v1-permutation"
TRAIN_ASSIGNMENT_DIGEST = "7b21d3af455475f7bb29198081a2ef2e651cffaac6149fd27741863c765b4efc"
TEST_ASSIGNMENT_DIGEST = "6ca5174470dad2b4eef4ae15c5a13a99d8ae9fc0bc2ea1116b199c4d4bc05582"
ANCHOR_COMMIT = "3b4da35440c8c26a7d1bcaf1daf342e11d9a3898"
CHECKPOINT_SHA256 = "6f42037d57a4e3de3a10ac0ab909d66a3415864a19433c29204a975f4efa4088"
MODEL_SOURCE_SHA256 = "9681defa95c6602d0e0abae05e635771bd65d48a18555198ff00d573242a0005"
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
PARENT_CLOSEOUT_SHA256 = "8b0192d32d0ca1cb35c0f2de5c1d0b211e50aa10504498680d404cf642686581"
PARENT_CONCLUSION_SHA256 = "560c05a29c890ce257d9f04a430b3d99234d2a71c68893a7e6605ab5bd36b1f8"
PARENT_SUMMARY_SHA256 = "d7dd7740c98abfe3203647cad44d61645e9ea4bb76be79e67ba83ca64ecdd2c4"
PARENT_GATE_SHA256 = "7cf8fa570e411038c65bf462faf0d5230f53dca2a99c3bb3ff751f59eaeeeb50"
PARENT_TRAINING_SHA256 = "8ee9175c558afd89759336fc98fe90159fda75e3cfae31ac59ff6abc90d9e2bd"
PARENT_SCENE_METRICS_SHA256 = "3736651ff6eedfc8aca0e690bfc163792848037255bea0115509573fdbd00505"
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
BOOTSTRAP_RESAMPLES = 100_000
BOOTSTRAP_SEED = 20260726
TRAINING_SEED = 20260726
FORMAL_CONTINUOUS_FAMILY_SIZE = 5
MATERIAL_PREVALENCE_FAMILY_SIZE = 5
SAFETY_FAMILY_SIZE = 24
CONTROL_FAMILY_SIZE = 2
TOP_FRACTION = 0.10
P11_REPRODUCTION_TOLERANCE = 1.0e-12
PARENT_TRAINING_TOLERANCE = 1.0e-6
TOTAL_UNITS = 851
FORMAL_COST_ITERATIONS = 46_050
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


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

    anchor_checkout = asset_path(
        context, "official_anchor_checkout", kind="git_checkout",
    )
    checkpoint = asset_path(context, "official_checkpoint", kind="file")
    model_source = asset_path(context, "model_source", kind="file")
    model_layers = asset_path(context, "model_layers", kind="file")
    if context.assets["official_anchor_checkout"].commit != ANCHOR_COMMIT:
        raise RuntimeError("official anchor commit changed")
    expected = {
        "official_checkpoint": CHECKPOINT_SHA256,
        "model_source": MODEL_SOURCE_SHA256,
        "model_layers": MODEL_LAYERS_SHA256,
    }
    for identifier, identity in expected.items():
        if context.assets[identifier].sha256 != identity:
            raise RuntimeError(f"verified identity changed for {identifier}")
    if any(
        name in sys.modules
        for name in ("Dehazing.ITS.models.ConvIR", "Dehazing.ITS.models.layers")
    ):
        raise RuntimeError("official model modules loaded before anchor binding")
    anchor_text = str(anchor_checkout)
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


def train_head(
    torch,
    records: list[dict[str, Any]],
    *,
    seed: int,
    device: str,
    progress_callback=None,
):
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
        total_utility_loss = 0.0
        total_risk_loss = 0.0
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
            total_utility_loss += float(
                torch.sum(area * utility_loss.squeeze(0)).detach().cpu().item()
            )
            total_risk_loss += float(
                torch.sum(area * risk_loss.squeeze(0)).detach().cpu().item()
            )
        history.append({
            "epoch": epoch + 1,
            "total_loss": total_loss / len(records),
            "utility_loss": total_utility_loss / len(records),
            "risk_loss": total_risk_loss / len(records),
        })
        if progress_callback is not None:
            progress_callback(epoch + 1)
    return head.eval(), mean, scale, history


def predict_head(torch, head, record, mean, scale, device: str) -> tuple[np.ndarray, np.ndarray]:
    with torch.inference_mode():
        output = head(normalized_tensor(torch, record, mean, scale, device))
        utility = output[:, :2].squeeze(0).detach().cpu().numpy().astype(np.float64)
        risk = torch.sigmoid(output[:, 2:]).squeeze(0).detach().cpu().numpy().astype(np.float64)
    if not np.isfinite(utility).all() or not np.isfinite(risk).all():
        raise RuntimeError("contextual head produced non-finite predictions")
    return utility, risk


def constant_baselines(records: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    utility_means = []
    harm_means = []
    for record in records:
        weights = record["areas"].astype(np.float64)
        weights /= float(np.sum(weights))
        utility_means.append(np.sum(record["utility"] * weights[None, :, :], axis=(1, 2)))
        harm_means.append(np.sum(record["harm"] * weights[None, :, :], axis=(1, 2)))
    return {
        "utility": np.mean(np.stack(utility_means), axis=0).astype(np.float64),
        "risk": np.mean(np.stack(harm_means), axis=0).astype(np.float64),
    }


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    total = float(np.sum(weights))
    if values.size != weights.size or values.size == 0 or total <= 0.0:
        raise RuntimeError("invalid weighted-mean inputs")
    result = float(np.sum(values * weights) / total)
    if not math.isfinite(result):
        raise RuntimeError("non-finite weighted mean")
    return result


def average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def weighted_correlation(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    weights = weights / float(np.sum(weights))
    mean_x = float(np.sum(weights * x))
    mean_y = float(np.sum(weights * y))
    centered_x = x - mean_x
    centered_y = y - mean_y
    variance_x = float(np.sum(weights * centered_x * centered_x))
    variance_y = float(np.sum(weights * centered_y * centered_y))
    if variance_x <= EPSILON or variance_y <= EPSILON:
        return None
    return float(np.sum(weights * centered_x * centered_y) / math.sqrt(variance_x * variance_y))


def weighted_binary_auc(
    scores: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
) -> float | None:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.bool_).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    positive = float(np.sum(weights[labels]))
    negative = float(np.sum(weights[~labels]))
    if positive <= 0.0 or negative <= 0.0:
        return None
    order = np.argsort(scores, kind="mergesort")
    cumulative_negative = 0.0
    favorable_pairs = 0.0
    start = 0
    while start < order.size:
        stop = start + 1
        while stop < order.size and scores[order[stop]] == scores[order[start]]:
            stop += 1
        group = order[start:stop]
        group_positive = float(np.sum(weights[group][labels[group]]))
        group_negative = float(np.sum(weights[group][~labels[group]]))
        favorable_pairs += group_positive * (cumulative_negative + 0.5 * group_negative)
        cumulative_negative += group_negative
        start = stop
    return favorable_pairs / (positive * negative)


def signal_metrics(
    record: dict[str, Any],
    utility_prediction: np.ndarray,
    risk_prediction: np.ndarray,
    baselines: dict[str, np.ndarray],
    *,
    permutation_key: str,
) -> list[dict[str, Any]]:
    weights = record["areas"].astype(np.float64).reshape(-1)
    weights /= float(np.sum(weights))
    result = []
    for action, name in enumerate(ACTION_NAMES):
        utility = record["utility"][action].astype(np.float64).reshape(-1)
        harm = record["harm"][action].astype(np.float64).reshape(-1)
        predicted_utility = utility_prediction[action].astype(np.float64).reshape(-1)
        predicted_risk = risk_prediction[action].astype(np.float64).reshape(-1)
        constant_utility = np.full_like(utility, baselines["utility"][action])
        constant_risk = np.full_like(harm, baselines["risk"][action])
        permutation = np.random.default_rng(
            deterministic_seed(f"{PERMUTATION_SALT}|{permutation_key}|{name}"),
        ).permutation(utility.size)
        permuted_utility_prediction = predicted_utility[permutation]
        permuted_risk_prediction = predicted_risk[permutation]
        top_count = max(1, int(math.ceil(TOP_FRACTION * utility.size)))
        top = np.argsort(predicted_utility, kind="mergesort")[-top_count:]
        permuted_top = np.argsort(permuted_utility_prediction, kind="mergesort")[-top_count:]
        beneficial = utility > UTILITY_MARGIN_DB
        harmful = harm > 0.5
        utility_mae = weighted_mean(np.abs(predicted_utility - utility), weights)
        constant_utility_mae = weighted_mean(np.abs(constant_utility - utility), weights)
        risk_brier = weighted_mean((predicted_risk - harm) ** 2, weights)
        constant_risk_brier = weighted_mean((constant_risk - harm) ** 2, weights)
        top_lift = weighted_mean(utility[top], weights[top]) - weighted_mean(utility, weights)
        permuted_top_lift = (
            weighted_mean(utility[permuted_top], weights[permuted_top])
            - weighted_mean(utility, weights)
        )
        permuted_utility_mae = weighted_mean(
            np.abs(permuted_utility_prediction - utility), weights,
        )
        permuted_risk_brier = weighted_mean(
            (permuted_risk_prediction - harm) ** 2, weights,
        )
        result.append({
            "action": name,
            "utility_mae": utility_mae,
            "constant_utility_mae": constant_utility_mae,
            "utility_mae_improvement": constant_utility_mae - utility_mae,
            "utility_rank_correlation": weighted_correlation(
                average_ranks(predicted_utility), average_ranks(utility), weights,
            ),
            "beneficial_auc": weighted_binary_auc(
                predicted_utility, beneficial, weights,
            ),
            "top_fraction_utility_lift": top_lift,
            "risk_brier": risk_brier,
            "constant_risk_brier": constant_risk_brier,
            "risk_brier_improvement": constant_risk_brier - risk_brier,
            "risk_auc": weighted_binary_auc(predicted_risk, harmful, weights),
            "beneficial_area_prevalence": weighted_mean(beneficial, weights),
            "harm_area_prevalence": weighted_mean(harmful, weights),
            "permuted_utility_mae": permuted_utility_mae,
            "permuted_utility_mae_improvement": constant_utility_mae - permuted_utility_mae,
            "permuted_beneficial_auc": weighted_binary_auc(
                permuted_utility_prediction, beneficial, weights,
            ),
            "permuted_top_fraction_utility_lift": permuted_top_lift,
            "observed_minus_permuted_top_lift": top_lift - permuted_top_lift,
            "permuted_risk_brier": permuted_risk_brier,
            "permuted_risk_brier_improvement": constant_risk_brier - permuted_risk_brier,
            "permuted_risk_auc": weighted_binary_auc(
                permuted_risk_prediction, harmful, weights,
            ),
        })
    return result


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
        "utility_scores": utility_scores,
        "risk_scores": risk_scores,
        "scene_count": len(records),
        "quantile": CALIBRATION_QUANTILE,
    }


def corrected_predictions(utility, risk, calibration) -> tuple[np.ndarray, np.ndarray]:
    utility_lower = utility - calibration["utility_correction"][:, None, None]
    risk_upper = np.clip(risk + calibration["risk_correction"][:, None, None], 0.0, 1.0)
    return utility_lower, risk_upper


def policy_prediction_pairs(
    utility: np.ndarray,
    risk: np.ndarray,
    calibration: dict[str, Any],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    utility_lower, risk_upper = corrected_predictions(utility, risk, calibration)
    return {
        "P00": (utility, risk),
        "P10": (utility_lower, risk),
        "P01": (utility, risk_upper),
        "P11": (utility_lower, risk_upper),
    }


def eligibility(utility_value: np.ndarray, risk_value: np.ndarray) -> np.ndarray:
    return (utility_value > UTILITY_MARGIN_DB) & (risk_value <= RISK_ACCEPTANCE_MAX)


def choose_actions(utility_lower: np.ndarray, risk_upper: np.ndarray) -> np.ndarray:
    eligible = eligibility(utility_lower, risk_upper)
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
    baselines: dict[str, np.ndarray],
    device: str,
) -> dict[str, Any]:
    utility, risk = predict_head(torch, head, record, mean, scale, device)
    pairs = policy_prediction_pairs(utility, risk, calibration)
    action_maps = {
        policy_id: choose_actions(*pairs[policy_id]) for policy_id in POLICY_IDS
    }
    uniform_actions = {
        policy_id: choose_uniform_action(
            *pairs[policy_id], record["areas"],
        )
        for policy_id in POLICY_IDS
    }
    gt_utility = record["utility"].astype(np.float64)
    gt_risk = record["harm"].astype(np.float64)
    action_maps["GT"] = choose_actions(gt_utility, gt_risk)
    uniform_actions["GT"] = choose_uniform_action(
        gt_utility, gt_risk, record["areas"],
    )
    spatial_outputs = {}
    uniform_outputs = {}
    shuffled_outputs = {}
    for policy_id in (*POLICY_IDS, "GT"):
        actions = action_maps[policy_id]
        shuffle_suffix = "" if policy_id == "P11" else f"|{policy_id}"
        shuffled = np.random.default_rng(
            deterministic_seed(
                f"{SHUFFLE_SALT}|{scene}|{variant}{shuffle_suffix}"
            ),
        ).permutation(actions.reshape(-1)).reshape(actions.shape)
        spatial_outputs[policy_id] = apply_spatial(hazy, prediction, actions)
        uniform_outputs[policy_id] = apply_uniform(
            hazy, prediction, uniform_actions[policy_id],
        )
        shuffled_outputs[policy_id] = apply_spatial(hazy, prediction, shuffled)
    keep_psnr = psnr(prediction, clear)
    ssim_values = rgb_ssim(
        torch,
        [prediction] + [spatial_outputs[item] for item in (*POLICY_IDS, "GT")],
        clear,
        device,
    )
    keep_ssim = ssim_values[0]
    keep_color = color_bias(prediction, clear)
    weights = record["areas"].astype(np.float64)
    weights /= float(np.sum(weights))
    metrics = {}
    for index, policy_id in enumerate((*POLICY_IDS, "GT"), start=1):
        spatial_output = spatial_outputs[policy_id]
        spatial_psnr = psnr(spatial_output, clear)
        uniform_psnr = psnr(uniform_outputs[policy_id], clear)
        shuffled_psnr = psnr(shuffled_outputs[policy_id], clear)
        spatial_color = color_bias(spatial_output, clear)
        actions = action_maps[policy_id]
        metrics[policy_id] = {
            "spatial_minus_keep_psnr_db": spatial_psnr - keep_psnr,
            "spatial_minus_uniform_psnr_db": spatial_psnr - uniform_psnr,
            "spatial_minus_shuffled_psnr_db": spatial_psnr - shuffled_psnr,
            "spatial_minus_keep_ssim": ssim_values[index] - keep_ssim,
            "spatial_minus_keep_color_bias": spatial_color - keep_color,
            "keep_fraction": weighted_mean(actions == 0, weights),
            "weaken_fraction": weighted_mean(actions == 1, weights),
            "strengthen_fraction": weighted_mean(actions == 2, weights),
        }
    eligible_maps = {
        policy_id: eligibility(*pairs[policy_id]) for policy_id in POLICY_IDS
    }
    pass_rows = []
    for action, name in enumerate(ACTION_NAMES):
        raw_utility_pass = utility[action] > UTILITY_MARGIN_DB
        corrected_utility_pass = (
            utility[action] - calibration["utility_correction"][action]
        ) > UTILITY_MARGIN_DB
        raw_risk_pass = risk[action] <= RISK_ACCEPTANCE_MAX
        corrected_risk_pass = (
            risk[action] + calibration["risk_correction"][action]
        ) <= RISK_ACCEPTANCE_MAX
        eligible_area = {
            policy_id: weighted_mean(eligible_maps[policy_id][action], weights)
            for policy_id in POLICY_IDS
        }
        pass_rows.append({
            "action": name,
            "raw_utility_pass_area": weighted_mean(raw_utility_pass, weights),
            "corrected_utility_pass_area": weighted_mean(
                corrected_utility_pass, weights,
            ),
            "raw_risk_pass_area": weighted_mean(raw_risk_pass, weights),
            "corrected_risk_pass_area": weighted_mean(corrected_risk_pass, weights),
            "raw_utility_rejection_area": weighted_mean(~raw_utility_pass, weights),
            "raw_risk_rejection_area": weighted_mean(~raw_risk_pass, weights),
            "p00_joint_pass_area": eligible_area["P00"],
            "p10_joint_pass_area": eligible_area["P10"],
            "p01_joint_pass_area": eligible_area["P01"],
            "p11_joint_pass_area": eligible_area["P11"],
            "p00_joint_rejection_area": 1.0 - eligible_area["P00"],
            "utility_correction_exclusion_area": (
                eligible_area["P00"] - eligible_area["P10"]
            ),
            "risk_correction_exclusion_area": (
                eligible_area["P00"] - eligible_area["P01"]
            ),
            "combined_correction_exclusion_area": (
                eligible_area["P00"] - eligible_area["P11"]
            ),
            "utility_correction_shapley": 0.5 * (
                eligible_area["P00"] - eligible_area["P10"]
                + eligible_area["P01"] - eligible_area["P11"]
            ),
            "risk_correction_shapley": 0.5 * (
                eligible_area["P00"] - eligible_area["P01"]
                + eligible_area["P10"] - eligible_area["P11"]
            ),
            "eligibility_interaction": (
                eligible_area["P00"] - eligible_area["P10"]
                - eligible_area["P01"] + eligible_area["P11"]
            ),
            "selected_area_p00": weighted_mean(
                action_maps["P00"] == action + 1, weights,
            ),
            "selected_area_p10": weighted_mean(
                action_maps["P10"] == action + 1, weights,
            ),
            "selected_area_p01": weighted_mean(
                action_maps["P01"] == action + 1, weights,
            ),
            "selected_area_p11": weighted_mean(
                action_maps["P11"] == action + 1, weights,
            ),
            "utility_correction": float(calibration["utility_correction"][action]),
            "risk_correction": float(calibration["risk_correction"][action]),
            "effective_raw_utility_threshold": (
                UTILITY_MARGIN_DB + float(calibration["utility_correction"][action])
            ),
            "effective_raw_risk_maximum": (
                RISK_ACCEPTANCE_MAX - float(calibration["risk_correction"][action])
            ),
            "risk_structurally_unselectable": bool(
                calibration["risk_correction"][action] >= RISK_ACCEPTANCE_MAX
            ),
        })
    return {
        "metrics": metrics,
        "pass_rows": pass_rows,
        "signal_rows": signal_metrics(
            record,
            utility,
            risk,
            baselines,
            permutation_key=f"{scene}|{variant}",
        ),
        "raw_utility": utility,
        "raw_risk": risk,
        "actions": action_maps,
        "keep_identity": bool(np.array_equal(
            apply_spatial(hazy, prediction, np.zeros_like(action_maps["P11"])),
            prediction,
        )),
    }


def mean_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = tuple(rows[0])
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def flatten_policy_metrics(evaluation: dict[str, Any]) -> dict[str, float]:
    flattened = {}
    for policy_id, metrics in evaluation["metrics"].items():
        prefix = policy_id.lower()
        for key, value in metrics.items():
            flattened[f"{prefix}_{key}"] = float(value)
    flattened["p00_minus_p11_psnr_db"] = (
        flattened["p00_spatial_minus_keep_psnr_db"]
        - flattened["p11_spatial_minus_keep_psnr_db"]
    )
    return flattened


def mean_optional(values: Iterable[Any]) -> float | None:
    finite = [
        float(value) for value in values
        if value is not None and math.isfinite(float(value))
    ]
    return None if not finite else float(np.mean(finite))


def average_nested_rows(
    variants: list[list[dict[str, Any]]],
    *,
    identity_key: str,
) -> list[dict[str, Any]]:
    if not variants:
        raise RuntimeError("nested row aggregation requires variants")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rows in variants:
        for row in rows:
            grouped[str(row[identity_key])].append(row)
    result = []
    for identity in sorted(grouped):
        rows = grouped[identity]
        output: dict[str, Any] = {identity_key: identity}
        fields = sorted(set().union(*(row.keys() for row in rows)) - {identity_key})
        for field in fields:
            values = [row.get(field) for row in rows]
            if all(isinstance(value, bool) for value in values):
                output[field] = bool(all(values))
            else:
                output[field] = mean_optional(values)
        result.append(output)
    return result


def bonferroni_z(family_size: int) -> float:
    if family_size <= 0:
        raise RuntimeError("family size must be positive")
    return NormalDist().inv_cdf(1.0 - 0.05 / (2.0 * family_size))


def normal_interval(values: Iterable[float], *, family_size: int) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size < 2 or not np.isfinite(array).all():
        raise RuntimeError("normal interval requires finite independent-scene values")
    estimate = float(np.mean(array))
    standard_error = float(np.std(array, ddof=1) / math.sqrt(array.size))
    half_width = bonferroni_z(family_size) * standard_error
    return {
        "scene_count": int(array.size),
        "estimate": estimate,
        "lower": estimate - half_width,
        "upper": estimate + half_width,
        "max_half_width": half_width,
        "family_size": family_size,
        "critical_value": bonferroni_z(family_size),
    }


def stratified_bootstrap_family(
    rows: list[dict[str, Any]],
    fields: list[str],
    *,
    family_size: int,
    seed: int,
) -> dict[str, dict[str, Any]]:
    if family_size != len(fields):
        raise RuntimeError("formal family size must equal the frozen field count")
    folds = sorted({int(row["fold"]) for row in rows})
    if folds != list(range(OOF_FOLDS)):
        raise RuntimeError("formal bootstrap requires the five frozen OOF folds")
    matrices = {}
    for fold in folds:
        fold_rows = [row for row in rows if int(row["fold"]) == fold]
        if len(fold_rows) != TRAINING_SCENES // OOF_FOLDS:
            raise RuntimeError("formal bootstrap fold size changed")
        matrices[fold] = np.asarray(
            [[float(row[field]) for field in fields] for row in fold_rows],
            dtype=np.float64,
        )
    generator = np.random.default_rng(seed)
    draws = np.empty((BOOTSTRAP_RESAMPLES, len(fields)), dtype=np.float64)
    for start in range(0, BOOTSTRAP_RESAMPLES, 250):
        stop = min(start + 250, BOOTSTRAP_RESAMPLES)
        chunk = np.zeros((stop - start, len(fields)), dtype=np.float64)
        for fold in folds:
            values = matrices[fold]
            indices = generator.integers(
                0, values.shape[0], size=(stop - start, values.shape[0]),
            )
            chunk += np.mean(values[indices], axis=1) / len(folds)
        draws[start:stop] = chunk
    tail = 0.05 / (2.0 * family_size)
    result = {}
    for index, field in enumerate(fields):
        observed = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
        estimate = float(np.mean(observed))
        lower = float(np.quantile(draws[:, index], tail))
        upper = float(np.quantile(draws[:, index], 1.0 - tail))
        result[field] = {
            "scene_count": len(rows),
            "estimate": estimate,
            "lower": lower,
            "upper": upper,
            "max_half_width": max(estimate - lower, upper - estimate),
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": seed,
            "family_size": family_size,
            "tail_probability": tail,
            "stratified_by": "oof_fold",
        }
    return result


def aggregate_signal_quality(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = (
        "utility_mae",
        "constant_utility_mae",
        "utility_mae_improvement",
        "utility_rank_correlation",
        "beneficial_auc",
        "top_fraction_utility_lift",
        "risk_brier",
        "constant_risk_brier",
        "risk_brier_improvement",
        "risk_auc",
        "beneficial_area_prevalence",
        "harm_area_prevalence",
        "permuted_utility_mae_improvement",
        "permuted_beneficial_auc",
        "permuted_top_fraction_utility_lift",
        "observed_minus_permuted_top_lift",
        "permuted_risk_brier_improvement",
        "permuted_risk_auc",
    )
    output = []
    populations = sorted({str(row["population"]) for row in rows})
    for population in populations:
        for action in ACTION_NAMES:
            selected = [
                row for row in rows
                if row["population"] == population and row["action"] == action
            ]
            for metric in metrics:
                values = [
                    float(row[metric]) for row in selected
                    if row.get(metric) is not None and math.isfinite(float(row[metric]))
                ]
                if len(values) < 2:
                    continue
                interval = normal_interval(values, family_size=36)
                output.append({
                    "population": population,
                    "action": action,
                    "metric": metric,
                    **interval,
                })
    return output


def summarize_model_signal(
    rows: list[dict[str, Any]],
    *,
    model: str,
    split: str,
    fold: int | str,
    history: list[dict[str, float]],
) -> list[dict[str, Any]]:
    output = []
    excluded = {"scene", "population", "model", "split", "fold", "action"}
    for action in ACTION_NAMES:
        selected = [row for row in rows if row["action"] == action]
        fields = sorted(set().union(*(row.keys() for row in selected)) - excluded)
        item = {
            "model": model,
            "split": split,
            "fold": fold,
            "action": action,
            "scene_count": len(selected),
            "first_epoch_total_loss": history[0]["total_loss"],
            "first_epoch_utility_loss": history[0]["utility_loss"],
            "first_epoch_risk_loss": history[0]["risk_loss"],
            "final_epoch_total_loss": history[-1]["total_loss"],
            "final_epoch_utility_loss": history[-1]["utility_loss"],
            "final_epoch_risk_loss": history[-1]["risk_loss"],
        }
        for field in fields:
            item[field] = mean_optional(row.get(field) for row in selected)
        output.append(item)
    return output


def calibration_rows(model: str, calibration: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for action, name in enumerate(ACTION_NAMES):
        utility_scores = np.asarray(
            calibration["utility_scores"][action], dtype=np.float64,
        )
        risk_scores = np.asarray(
            calibration["risk_scores"][action], dtype=np.float64,
        )
        output.append({
            "model": model,
            "action": name,
            "calibration_scenes": calibration["scene_count"],
            "quantile": calibration["quantile"],
            "utility_correction": calibration["utility_correction"][action],
            "risk_correction": calibration["risk_correction"][action],
            "effective_raw_utility_threshold": (
                UTILITY_MARGIN_DB + calibration["utility_correction"][action]
            ),
            "effective_raw_risk_maximum": (
                RISK_ACCEPTANCE_MAX - calibration["risk_correction"][action]
            ),
            "risk_structurally_unselectable": bool(
                calibration["risk_correction"][action] >= RISK_ACCEPTANCE_MAX
            ),
            "utility_residual_mean": float(np.mean(utility_scores)),
            "utility_residual_median": float(np.median(utility_scores)),
            "utility_residual_max": float(np.max(utility_scores)),
            "risk_residual_mean": float(np.mean(risk_scores)),
            "risk_residual_median": float(np.median(risk_scores)),
            "risk_residual_max": float(np.max(risk_scores)),
        })
    return output


def flatten_scene_pass_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    flattened = {}
    for row in rows:
        prefix = str(row["action"])
        for key, value in row.items():
            if key == "action":
                continue
            flattened[f"{prefix}_{key}"] = value
    return flattened


def aggregate_pass_rejection(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    groups = sorted({
        (str(row["population"]), str(row["model"]), str(row["action"]))
        for row in rows
    })
    excluded = {"population", "model", "fold", "scene", "action"}
    for population, model, action in groups:
        selected = [
            row for row in rows
            if row["population"] == population
            and str(row["model"]) == model
            and row["action"] == action
        ]
        fields = sorted(set().union(*(row.keys() for row in selected)) - excluded)
        item = {
            "population": population,
            "model": model,
            "action": action,
            "scene_count": len(selected),
        }
        for field in fields:
            values = [row.get(field) for row in selected]
            if all(isinstance(value, bool) for value in values):
                item[field] = bool(all(values))
            else:
                item[field] = mean_optional(values)
        output.append(item)
    return output


def wilson(successes: int, total: int, *, family_size: int = 1) -> dict[str, Any]:
    if total <= 0 or not 0 <= successes <= total:
        raise RuntimeError("invalid Wilson interval inputs")
    z = bonferroni_z(family_size)
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
        "family_size": family_size,
        "critical_value": z,
    }


def state_dict_digest(head) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(head.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def compare_parent_training(
    context,
    training_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    parent = {
        row["model"]: row
        for row in read_csv(
            asset_path(context, "parent_training_calibration", kind="file"),
        )
    }
    current = {str(row["model"]): row for row in training_rows}
    fields = (
        "final_loss",
        "utility_correction_weaken",
        "utility_correction_strengthen",
        "risk_correction_weaken",
        "risk_correction_strengthen",
    )
    differences = []
    for model in sorted(parent):
        if model not in current:
            return {
                "matched": False,
                "reason": f"missing model {model}",
                "tolerance": PARENT_TRAINING_TOLERANCE,
            }
        for field in fields:
            differences.append(abs(float(parent[model][field]) - float(current[model][field])))
    maximum = max(differences, default=math.inf)
    return {
        "matched": set(parent) == set(current) and maximum <= PARENT_TRAINING_TOLERANCE,
        "model_count": len(current),
        "maximum_absolute_difference": maximum,
        "tolerance": PARENT_TRAINING_TOLERANCE,
        "fields": list(fields),
    }


def add_inventory_item(
    items: list[dict[str, Any]],
    context,
    path: Path,
    artifact_class: str,
) -> None:
    items.append({
        "artifact_class": artifact_class,
        "relative_path": path.relative_to(context.output_path).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    })


def compact_inventory(items: list[dict[str, Any]]) -> dict[str, Any]:
    classes = {}
    for artifact_class in sorted({item["artifact_class"] for item in items}):
        selected = [item for item in items if item["artifact_class"] == artifact_class]
        classes[artifact_class] = {
            "file_count": len(selected),
            "total_bytes": sum(int(item["bytes"]) for item in selected),
            "inventory_sha256": digest_lines(
                f"{item['relative_path']}|{item['sha256']}|{item['bytes']}"
                for item in selected
            ),
        }
    checkpoints = [
        {
            "relative_path": item["relative_path"],
            "sha256": item["sha256"],
            "bytes": item["bytes"],
        }
        for item in items
        if item["artifact_class"] == "head_checkpoint"
    ]
    return {
        "schema_version": 1,
        "raw_retention_scope": "cloud_only",
        "file_count": len(items),
        "total_bytes": sum(int(item["bytes"]) for item in items),
        "inventory_sha256": digest_lines(
            f"{item['artifact_class']}|{item['relative_path']}|{item['sha256']}|{item['bytes']}"
            for item in items
        ),
        "classes": classes,
        "head_checkpoints": checkpoints,
        "github_exclusions": [
            "datasets",
            "weights",
            "checkpoints",
            "images",
            "arrays",
            "raw_predictions",
            "feature_tables",
            "action_maps",
        ],
    }


def save_raw_prediction(
    path: Path,
    evaluation: dict[str, Any],
    record: dict[str, Any],
    calibration: dict[str, Any],
    *,
    include_features: bool,
) -> None:
    payload = {
        "raw_utility": evaluation["raw_utility"].astype(np.float32),
        "raw_risk": evaluation["raw_risk"].astype(np.float32),
        "target_utility": record["utility"].astype(np.float32),
        "target_harm": record["harm"].astype(np.float32),
        "areas": record["areas"].astype(np.float32),
        "utility_correction": calibration["utility_correction"].astype(np.float64),
        "risk_correction": calibration["risk_correction"].astype(np.float64),
    }
    if include_features:
        payload["features"] = record["features"].astype(np.float32)
    for policy_id, actions in evaluation["actions"].items():
        payload[f"actions_{policy_id.lower()}"] = actions.astype(np.int8)
    np.savez(path, **payload)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def evidence_identity(context) -> dict[str, bool]:
    assets = {
        "train_split_closeout": (TRAIN_SPLIT_CLOSEOUT_SHA256, "HAZE4K_TRAIN_SCENE_SPLIT_PASS"),
        "test_split_closeout": (TEST_SPLIT_CLOSEOUT_SHA256, "HAZE4K_TEST_SCENE_SPLIT_PASS"),
        "prior_proxy_closeout": (PROXY_CLOSEOUT_SHA256, "HAZE4K_TEST_LOCAL_ACTION_PROXY_PREDICTABILITY_FAIL"),
        "prior_oracle_closeout": (ORACLE_CLOSEOUT_SHA256, "HAZE4K_TEST_BOUNDED_LOCAL_ACTION_ORACLE_HEADROOM_PASS"),
        "parent_closeout": (
            PARENT_CLOSEOUT_SHA256,
            "HAZE4K_OBSERVABLE_CONDITIONAL_UTILITY_RISK_FAIL",
        ),
        "parent_conclusion": (
            PARENT_CONCLUSION_SHA256,
            "HAZE4K_OBSERVABLE_CONDITIONAL_UTILITY_RISK_FAIL",
        ),
    }
    checks = {}
    for identifier, (expected_sha, expected_decision) in assets.items():
        path = asset_path(context, identifier, kind="file")
        checks[identifier] = (
            sha256_file(path) == expected_sha
            and read_json(path).get("decision") == expected_decision
        )
    checks.update({
        "train_split_summary": sha256_file(asset_path(context, "train_split_summary", kind="file")) == TRAIN_SPLIT_SUMMARY_SHA256,
        "test_split_summary": sha256_file(asset_path(context, "test_split_summary", kind="file")) == TEST_SPLIT_SUMMARY_SHA256,
        "prior_proxy_summary": sha256_file(asset_path(context, "prior_proxy_summary", kind="file")) == PROXY_SUMMARY_SHA256,
        "prior_oracle_summary": sha256_file(asset_path(context, "prior_oracle_summary", kind="file")) == ORACLE_SUMMARY_SHA256,
        "parent_summary": sha256_file(
            asset_path(context, "parent_summary", kind="file"),
        ) == PARENT_SUMMARY_SHA256,
        "parent_gate_summary": sha256_file(
            asset_path(context, "parent_gate_summary", kind="file"),
        ) == PARENT_GATE_SHA256,
        "parent_training_calibration": sha256_file(
            asset_path(context, "parent_training_calibration", kind="file"),
        ) == PARENT_TRAINING_SHA256,
        "parent_scene_metrics": sha256_file(
            asset_path(context, "parent_scene_metrics", kind="file"),
        ) == PARENT_SCENE_METRICS_SHA256,
    })
    parent_closeout = read_json(asset_path(context, "parent_closeout", kind="file"))
    parent_summary = read_json(asset_path(context, "parent_summary", kind="file"))
    checks.update({
        "parent_terminal_tuple": (
            parent_closeout.get("state") == "COMPLETED_GATE_FAIL"
            and parent_closeout.get("authorizes") == "NONE"
            and parent_closeout.get("route_commit")
            == "1b9e935bfcb436929dbd16720bc0f3a7e6b3eed2"
        ),
        "parent_all_keep_oof": (
            parent_summary.get("primary_haze4k_train_oof", {})
            .get("action_fractions", {}).get("keep_fraction") == 1
        ),
        "parent_all_keep_test_development": (
            parent_summary.get("secondary_haze4k_test_development_stress", {})
            .get("action_fractions", {}).get("keep_fraction") == 1
        ),
    })
    return checks


def contract(context_path: Path) -> None:
    context = load_context(context_path, "contract")
    prepare_phase_output(context)
    started = time.monotonic()
    import torch

    if context.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
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
    initial_head_digest = state_dict_digest(head)
    first_loss = None
    last_loss = None
    nonzero_finite_gradient = False
    for index in range(training_iterations):
        output = head(normalized_tensor(torch, fixture_record, mean, scale, context.device))
        target = torch.from_numpy(fixture_record["utility"]).unsqueeze(0).to(context.device)
        harm = torch.from_numpy(fixture_record["harm"]).unsqueeze(0).to(context.device)
        loss = functional.smooth_l1_loss(output[:, :2], target, beta=UTILITY_HUBER_BETA_DB) \
            + RISK_LOSS_WEIGHT * functional.binary_cross_entropy_with_logits(output[:, 2:], harm)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradients = [
            parameter.grad for parameter in head.parameters()
            if parameter.grad is not None
        ]
        nonzero_finite_gradient = nonzero_finite_gradient or (
            bool(gradients)
            and all(bool(torch.isfinite(value).all().item()) for value in gradients)
            and any(bool(torch.any(value != 0).item()) for value in gradients)
        )
        torch.nn.utils.clip_grad_norm_(head.parameters(), GRADIENT_CLIP_NORM)
        optimizer.step()
        current_loss = float(loss.detach().cpu().item())
        if first_loss is None:
            first_loss = current_loss
        last_loss = current_loss
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
    baselines = constant_baselines([fixture_record])
    finite_eval = True
    evaluation = None
    for _ in range(evaluation_iterations):
        evaluation = evaluate_extracted(
            torch,
            head.eval(),
            fixture_record,
            mean,
            scale,
            calibration,
            hazy,
            clear,
            extracted["prediction"],
            "synthetic_scene",
            "synthetic_variant",
            baselines,
            context.device,
        )
        finite_eval = finite_eval and bool(
            np.isfinite(evaluation["raw_utility"]).all()
            and np.isfinite(evaluation["raw_risk"]).all()
        )
    assert evaluation is not None
    elapsed = time.monotonic() - started
    peak = (
        float(torch.cuda.max_memory_allocated() / (1024 * 1024))
        if context.device == "cuda" else 0.0
    )
    checks.update({
        "official_graph_strict_loaded": sum(parameter.numel() for parameter in model.parameters()) == PARAMETER_COUNT,
        "observable_feature_shape": extracted["features"].shape[0] == FEATURE_CHANNELS,
        "continuous_target_shape": extracted["utility"].shape[0] == 2,
        "risk_target_shape": extracted["harm"].shape[0] == 2,
        "contextual_head_policy_set": set(evaluation["metrics"]) == {
            *POLICY_IDS, "GT",
        },
        "shared_prediction_policy_actions": set(evaluation["actions"]) == {
            *POLICY_IDS, "GT",
        },
        "pass_rejection_schema": len(evaluation["pass_rows"]) == len(ACTION_NAMES),
        "signal_diagnostic_schema": len(evaluation["signal_rows"]) == len(ACTION_NAMES),
        "keep_identity_control": evaluation["keep_identity"],
        "nonzero_finite_gradient": nonzero_finite_gradient,
        "head_parameters_changed": state_dict_digest(head) != initial_head_digest,
        "synthetic_loss_decreased": (
            first_loss is not None and last_loss is not None and last_loss < first_loss
        ),
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
            "cost": {
                "observed_iterations": PROBE_COST_ITERATIONS,
                "observed_wall_seconds": elapsed,
                "observed_peak_memory_mib": peak,
            },
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

    train_groups, training_scenes, calibration_scenes, assignment = enumerate_train_groups(
        train_root,
    )
    training_scene_set = set(training_scenes)
    role_checks = {
        "train_assignment_digest": assignment == TRAIN_ASSIGNMENT_DIGEST,
        "train_role_counts": (
            len(training_scenes) == TRAINING_SCENES
            and len(calibration_scenes) == CALIBRATION_SCENES
        ),
        "train_role_disjointness": not (
            set(training_scenes) & set(calibration_scenes)
        ),
        "one_outcome_blind_variant_per_train_scene": True,
        "test_assignment_digest": read_json(
            asset_path(context, "test_split_summary", kind="file"),
        ).get("frozen_split", {}).get("assignment_digest") == TEST_ASSIGNMENT_DIGEST,
        "candidate_confirmation_absent": "candidate_confirmation" not in context.assets,
        "nh_haze_absent": "nh_haze" not in context.assets,
        "reside_its_ots_absent": (
            "reside_its" not in context.assets and "reside_ots" not in context.assets
        ),
    }

    torch, model = load_official_model(context)
    cache_root = output_file(context, "scene_cache")
    cache_root.mkdir()
    checkpoint_root = output_file(context, "head_checkpoints")
    checkpoint_root.mkdir()
    raw_oof_root = output_file(context, "raw_predictions/haze4k_train_oof")
    raw_oof_root.mkdir(parents=True)
    raw_test_root = output_file(
        context, "raw_predictions/haze4k_test_development_stress",
    )
    raw_test_root.mkdir(parents=True)
    inventory_items: list[dict[str, Any]] = []
    extraction_records: dict[str, dict[str, Any]] = {}
    extraction_identities = []
    ordered_train_scenes = training_scenes + calibration_scenes
    for completed, scene in enumerate(ordered_train_scenes, start=1):
        hazy_path, clear_path = choose_one_variant(scene, train_groups[scene])
        hazy, clear = image_array(hazy_path), image_array(clear_path)
        if hazy.shape != clear.shape:
            raise RuntimeError(
                f"selected train pair dimensions differ for {hazy_path.name}"
            )
        extracted = extract_variant(torch, model, hazy, clear, context.device)
        role = "training" if scene in training_scene_set else "calibration"
        relative = f"scene_cache/{role}_{scene[:24]}.npz"
        cache_path = output_file(context, relative)
        np.savez(cache_path, **extracted)
        add_inventory_item(inventory_items, context, cache_path, "scene_cache")
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
            **{
                key: extracted[key]
                for key in ("features", "utility", "harm", "areas")
            },
        }
        if completed % 10 == 0 or completed == EXPECTED_TRAIN_SCENES:
            write_workload_progress(
                context,
                completed_units=completed,
                stage="outcome_blind_scene_extraction",
            )
    role_checks["complete_train_extraction"] = (
        len(extraction_records) == EXPECTED_TRAIN_SCENES
    )

    fold_order = sorted(
        training_scenes,
        key=lambda scene: (sha256_text(f"{OOF_FOLD_SALT}|{scene}"), scene),
    )
    fold_assignment = {
        scene: index % OOF_FOLDS for index, scene in enumerate(fold_order)
    }
    role_checks["balanced_oof_folds"] = (
        Counter(fold_assignment.values())
        == {fold: TRAINING_SCENES // OOF_FOLDS for fold in range(OOF_FOLDS)}
    )
    calibration_records = [
        extraction_records[scene] for scene in calibration_scenes
    ]
    primary_rows: list[dict[str, Any]] = []
    pass_scene_rows: list[dict[str, Any]] = []
    signal_scene_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    training_diagnostic_rows: list[dict[str, Any]] = []
    calibration_diagnostic_rows: list[dict[str, Any]] = []

    for fold in range(OOF_FOLDS):
        train_records = [
            extraction_records[scene]
            for scene in training_scenes
            if fold_assignment[scene] != fold
        ]
        eval_records = [
            extraction_records[scene]
            for scene in training_scenes
            if fold_assignment[scene] == fold
        ]
        baselines = constant_baselines(train_records)
        head, mean, scale, history = train_head(
            torch,
            train_records,
            seed=TRAINING_SEED + fold,
            device=context.device,
            progress_callback=lambda epoch, fold=fold: write_workload_progress(
                context,
                completed_units=EXPECTED_TRAIN_SCENES,
                stage=f"oof_training_fold_{fold}_epoch_{epoch}_of_{EPOCHS}",
            ),
        )
        calibration = calibrate_head(
            torch, head, calibration_records, mean, scale, context.device,
        )
        model_id = f"oof_fold_{fold}"
        checkpoint_path = checkpoint_root / f"{model_id}.pt"
        torch.save({
            "schema_version": 1,
            "model": model_id,
            "state_dict": {
                name: value.detach().cpu()
                for name, value in head.state_dict().items()
            },
            "normalization_mean": mean,
            "normalization_scale": scale,
            "utility_correction": calibration["utility_correction"],
            "risk_correction": calibration["risk_correction"],
            "constant_baselines": baselines,
            "history": history,
        }, checkpoint_path)
        add_inventory_item(
            inventory_items, context, checkpoint_path, "head_checkpoint",
        )

        fit_signal_rows = []
        for record in train_records:
            fit_utility, fit_risk = predict_head(
                torch, head, record, mean, scale, context.device,
            )
            for signal_row in signal_metrics(
                record,
                fit_utility,
                fit_risk,
                baselines,
                permutation_key=f"train-fit|{model_id}|{record['scene']}",
            ):
                fit_signal_rows.append({"scene": record["scene"], **signal_row})
        training_diagnostic_rows.extend(summarize_model_signal(
            fit_signal_rows,
            model=model_id,
            split="train_fit",
            fold=fold,
            history=history,
        ))

        for record in eval_records:
            cached = load_cache(record["cache_path"], include_prediction=True)
            hazy = image_array(record["hazy_path"])
            clear = image_array(record["clear_path"])
            evaluation = evaluate_extracted(
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
                baselines,
                context.device,
            )
            row = {
                "population": "haze4k_train_oof",
                "scene": record["scene"],
                "fold": fold,
                "model": model_id,
                **flatten_policy_metrics(evaluation),
                **flatten_scene_pass_rows(evaluation["pass_rows"]),
            }
            primary_rows.append(row)
            for pass_row in evaluation["pass_rows"]:
                pass_scene_rows.append({
                    "population": "haze4k_train_oof",
                    "scene": record["scene"],
                    "fold": fold,
                    "model": model_id,
                    **pass_row,
                })
            for signal_row in evaluation["signal_rows"]:
                signal_scene_rows.append({
                    "population": "haze4k_train_oof",
                    "scene": record["scene"],
                    "fold": fold,
                    "model": model_id,
                    **signal_row,
                })
            raw_path = raw_oof_root / f"{model_id}_{record['scene'][:24]}.npz"
            save_raw_prediction(
                raw_path,
                evaluation,
                record,
                calibration,
                include_features=False,
            )
            add_inventory_item(
                inventory_items, context, raw_path, "raw_prediction",
            )
            role_checks.setdefault("keep_identity_control", True)
            role_checks["keep_identity_control"] = (
                role_checks["keep_identity_control"] and evaluation["keep_identity"]
            )

        training_rows.append({
            "model": model_id,
            "train_scenes": len(train_records),
            "calibration_scenes": len(calibration_records),
            "evaluation_scenes": len(eval_records),
            "epochs": EPOCHS,
            "first_epoch_loss": history[0]["total_loss"],
            "first_epoch_utility_loss": history[0]["utility_loss"],
            "first_epoch_risk_loss": history[0]["risk_loss"],
            "final_loss": history[-1]["total_loss"],
            "final_utility_loss": history[-1]["utility_loss"],
            "final_risk_loss": history[-1]["risk_loss"],
            "utility_correction_weaken": calibration["utility_correction"][0],
            "utility_correction_strengthen": calibration["utility_correction"][1],
            "risk_correction_weaken": calibration["risk_correction"][0],
            "risk_correction_strengthen": calibration["risk_correction"][1],
        })
        calibration_diagnostic_rows.extend(
            calibration_rows(model_id, calibration)
        )

    role_checks["complete_oof_replay"] = (
        len(primary_rows) == TRAINING_SCENES
        and len({row["scene"] for row in primary_rows}) == TRAINING_SCENES
    )

    final_train_records = [
        extraction_records[scene] for scene in training_scenes
    ]
    final_baselines = constant_baselines(final_train_records)
    final_head, final_mean, final_scale, final_history = train_head(
        torch,
        final_train_records,
        seed=TRAINING_SEED + OOF_FOLDS,
        device=context.device,
        progress_callback=lambda epoch: write_workload_progress(
            context,
            completed_units=EXPECTED_TRAIN_SCENES,
            stage=f"final_training_epoch_{epoch}_of_{EPOCHS}",
        ),
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
    final_model_id = "frozen_test_development_candidate"
    final_checkpoint_path = checkpoint_root / f"{final_model_id}.pt"
    torch.save({
        "schema_version": 1,
        "model": final_model_id,
        "state_dict": {
            name: value.detach().cpu()
            for name, value in final_head.state_dict().items()
        },
        "normalization_mean": final_mean,
        "normalization_scale": final_scale,
        "utility_correction": final_calibration["utility_correction"],
        "risk_correction": final_calibration["risk_correction"],
        "constant_baselines": final_baselines,
        "history": final_history,
    }, final_checkpoint_path)
    add_inventory_item(
        inventory_items, context, final_checkpoint_path, "head_checkpoint",
    )

    final_fit_signal_rows = []
    for record in final_train_records:
        fit_utility, fit_risk = predict_head(
            torch, final_head, record, final_mean, final_scale, context.device,
        )
        for signal_row in signal_metrics(
            record,
            fit_utility,
            fit_risk,
            final_baselines,
            permutation_key=f"train-fit|{final_model_id}|{record['scene']}",
        ):
            final_fit_signal_rows.append({"scene": record["scene"], **signal_row})
    training_diagnostic_rows.extend(summarize_model_signal(
        final_fit_signal_rows,
        model=final_model_id,
        split="train_fit",
        fold="final",
        history=final_history,
    ))
    training_rows.append({
        "model": final_model_id,
        "train_scenes": len(final_train_records),
        "calibration_scenes": len(calibration_records),
        "evaluation_scenes": EXPECTED_TEST_SCENES,
        "epochs": EPOCHS,
        "first_epoch_loss": final_history[0]["total_loss"],
        "first_epoch_utility_loss": final_history[0]["utility_loss"],
        "first_epoch_risk_loss": final_history[0]["risk_loss"],
        "final_loss": final_history[-1]["total_loss"],
        "final_utility_loss": final_history[-1]["utility_loss"],
        "final_risk_loss": final_history[-1]["risk_loss"],
        "utility_correction_weaken": final_calibration["utility_correction"][0],
        "utility_correction_strengthen": final_calibration["utility_correction"][1],
        "risk_correction_weaken": final_calibration["risk_correction"][0],
        "risk_correction_strengthen": final_calibration["risk_correction"][1],
    })
    calibration_diagnostic_rows.extend(
        calibration_rows(final_model_id, final_calibration)
    )
    parent_training_reproduction = compare_parent_training(
        context, training_rows,
    )

    checkpoint_items = [
        item for item in inventory_items
        if item["artifact_class"] == "head_checkpoint"
    ]
    training_unit_path = output_file(
        context, "training_and_calibration_unit.json",
    )
    atomic_json(training_unit_path, {
        "schema_version": 2,
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
        "policies": list(POLICY_IDS),
        "parent_training_reproduction": parent_training_reproduction,
        "head_checkpoints": [
            {
                "relative_path": item["relative_path"],
                "sha256": item["sha256"],
            }
            for item in checkpoint_items
        ],
    })
    training_input_sha = sha256_text(
        "training-calibration-decomposition-v1|"
        + "|".join(sorted(extraction_identities))
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
        stage="oof_training_calibration_decomposition_and_replay",
    )

    test_groups = enumerate_test_groups(test_root)
    test_rows: list[dict[str, Any]] = []
    test_root_output = output_file(context, "test_development_replay")
    test_root_output.mkdir()
    for index, scene in enumerate(sorted(test_groups), start=1):
        variant_metric_rows = []
        variant_pass_rows = []
        variant_signal_rows = []
        variant_keep_identity = []
        input_parts = [
            "test-development-decomposition-v1",
            scene,
            final_head_sha,
        ]
        for variant_index, (hazy_path, clear_path) in enumerate(
            sorted(test_groups[scene]), start=1,
        ):
            hazy = image_array(hazy_path)
            clear = image_array(clear_path)
            extracted = extract_variant(
                torch, model, hazy, clear, context.device,
            )
            record = {
                key: extracted[key]
                for key in ("features", "utility", "harm", "areas")
            }
            evaluation = evaluate_extracted(
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
                final_baselines,
                context.device,
            )
            variant_metric_rows.append(flatten_policy_metrics(evaluation))
            variant_pass_rows.append(evaluation["pass_rows"])
            variant_signal_rows.append(evaluation["signal_rows"])
            variant_keep_identity.append(evaluation["keep_identity"])
            raw_path = (
                raw_test_root
                / f"scene_{index:03d}_{scene[:16]}_variant_{variant_index}.npz"
            )
            save_raw_prediction(
                raw_path,
                evaluation,
                record,
                final_calibration,
                include_features=True,
            )
            add_inventory_item(
                inventory_items, context, raw_path, "raw_prediction",
            )
            input_parts.extend([
                hazy_path.name,
                sha256_file(hazy_path),
                sha256_file(clear_path),
            ])

        metrics = mean_metrics(variant_metric_rows)
        scene_pass = average_nested_rows(
            variant_pass_rows, identity_key="action",
        )
        scene_signal = average_nested_rows(
            variant_signal_rows, identity_key="action",
        )
        row = {
            "population": "haze4k_test_development_stress",
            "scene": scene,
            "fold": "frozen_candidate",
            "model": final_model_id,
            **metrics,
            **flatten_scene_pass_rows(scene_pass),
        }
        test_rows.append(row)
        for pass_row in scene_pass:
            pass_scene_rows.append({
                "population": "haze4k_test_development_stress",
                "scene": scene,
                "fold": "frozen_candidate",
                "model": final_model_id,
                **pass_row,
            })
        for signal_row in scene_signal:
            signal_scene_rows.append({
                "population": "haze4k_test_development_stress",
                "scene": scene,
                "fold": "frozen_candidate",
                "model": final_model_id,
                **signal_row,
            })
        relative = f"test_development_replay/scene_{scene[:24]}.json"
        replay_path = output_file(context, relative)
        atomic_json(replay_path, {
            "schema_version": 2,
            "scene": scene,
            "nested_variants": VARIANTS_PER_SCENE,
            "metrics": metrics,
            "pass_rejection": scene_pass,
            "signal": scene_signal,
            "keep_identity": bool(all(variant_keep_identity)),
        })
        add_inventory_item(
            inventory_items, context, replay_path, "scene_replay",
        )
        role_checks["keep_identity_control"] = (
            role_checks.get("keep_identity_control", True)
            and bool(all(variant_keep_identity))
        )
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
                stage="frozen_test_development_decomposition_stress",
            )

    role_checks["complete_test_development_stress"] = (
        len(test_rows) == EXPECTED_TEST_SCENES
    )
    role_checks["four_variants_nested_within_test_scene"] = all(
        len(items) == VARIANTS_PER_SCENE for items in test_groups.values()
    )
    completed_unit_ledger = load_completed_unit_ledger(context)
    role_checks["completed_unit_ledger_coverage"] = (
        len(completed_unit_ledger) == TOTAL_UNITS
    )
    raw_inventory = compact_inventory(inventory_items)
    role_checks["raw_artifact_inventory_complete"] = (
        raw_inventory["file_count"] == 1856
        and raw_inventory["classes"].get("scene_cache", {}).get("file_count") == 750
        and raw_inventory["classes"].get("head_checkpoint", {}).get("file_count") == 6
        and raw_inventory["classes"].get("raw_prediction", {}).get("file_count") == 1000
        and raw_inventory["classes"].get("scene_replay", {}).get("file_count") == 100
    )
    write_workload_progress(
        context,
        completed_units=TOTAL_UNITS,
        stage="scene_grouped_formal_inference",
    )

    formal_fields = [
        "p00_spatial_minus_keep_psnr_db",
        "p10_spatial_minus_keep_psnr_db",
        "p01_spatial_minus_keep_psnr_db",
        "p00_spatial_minus_uniform_psnr_db",
        "p00_spatial_minus_shuffled_psnr_db",
    ]
    formal_intervals = stratified_bootstrap_family(
        primary_rows,
        formal_fields,
        family_size=FORMAL_CONTINUOUS_FAMILY_SIZE,
        seed=BOOTSTRAP_SEED,
    )
    oracle_fields = [
        "gt_spatial_minus_keep_psnr_db",
        "gt_spatial_minus_uniform_psnr_db",
    ]
    oracle_intervals = stratified_bootstrap_family(
        primary_rows,
        oracle_fields,
        family_size=CONTROL_FAMILY_SIZE,
        seed=BOOTSTRAP_SEED + 100,
    )

    population_rows = {
        "haze4k_train_oof": primary_rows,
        "haze4k_test_development_stress": test_rows,
    }
    material_prevalence = {}
    for population, rows in population_rows.items():
        material_prevalence[population] = {}
        for policy_id in (*POLICY_IDS, "GT"):
            prefix = policy_id.lower()
            metric = (
                f"{prefix}_spatial_minus_uniform_psnr_db"
                if policy_id == "GT"
                else f"{prefix}_spatial_minus_keep_psnr_db"
            )
            material_prevalence[population][policy_id] = wilson(
                sum(float(row[metric]) >= UTILITY_MARGIN_DB for row in rows),
                len(rows),
                family_size=MATERIAL_PREVALENCE_FAMILY_SIZE,
            )

    safety_intervals = {}
    safety_definitions = {
        "psnr": lambda row, prefix: (
            float(row[f"{prefix}_spatial_minus_keep_psnr_db"])
            <= -PSNR_HARM_MARGIN_DB
        ),
        "ssim": lambda row, prefix: (
            float(row[f"{prefix}_spatial_minus_keep_ssim"])
            <= -SSIM_HARM_MARGIN
        ),
        "color": lambda row, prefix: (
            float(row[f"{prefix}_spatial_minus_keep_color_bias"])
            >= COLOR_HARM_MARGIN
        ),
    }
    for population, rows in population_rows.items():
        safety_intervals[population] = {}
        for policy_id in POLICY_IDS:
            prefix = policy_id.lower()
            safety_intervals[population][policy_id] = {}
            for metric, predicate in safety_definitions.items():
                safety_intervals[population][policy_id][metric] = wilson(
                    sum(predicate(row, prefix) for row in rows),
                    len(rows),
                    family_size=SAFETY_FAMILY_SIZE,
                )

    policy_summary_rows = []
    for population, rows in population_rows.items():
        for policy_id in (*POLICY_IDS, "GT"):
            prefix = policy_id.lower()
            material = material_prevalence[population][policy_id]
            item = {
                "population": population,
                "policy": policy_id,
                "scene_count": len(rows),
                "spatial_minus_keep_psnr_db": float(np.mean([
                    row[f"{prefix}_spatial_minus_keep_psnr_db"] for row in rows
                ])),
                "spatial_minus_uniform_psnr_db": float(np.mean([
                    row[f"{prefix}_spatial_minus_uniform_psnr_db"] for row in rows
                ])),
                "spatial_minus_shuffled_psnr_db": float(np.mean([
                    row[f"{prefix}_spatial_minus_shuffled_psnr_db"] for row in rows
                ])),
                "material_scene_prevalence": material["estimate"],
                "material_scene_prevalence_lower": material["lower"],
                "material_scene_prevalence_upper": material["upper"],
                "keep_fraction": float(np.mean([
                    row[f"{prefix}_keep_fraction"] for row in rows
                ])),
                "weaken_fraction": float(np.mean([
                    row[f"{prefix}_weaken_fraction"] for row in rows
                ])),
                "strengthen_fraction": float(np.mean([
                    row[f"{prefix}_strengthen_fraction"] for row in rows
                ])),
            }
            if policy_id in POLICY_IDS:
                for metric in ("psnr", "ssim", "color"):
                    interval = safety_intervals[population][policy_id][metric]
                    item[f"{metric}_harm_prevalence"] = interval["estimate"]
                    item[f"{metric}_harm_lower"] = interval["lower"]
                    item[f"{metric}_harm_upper"] = interval["upper"]
            if population == "haze4k_train_oof":
                gain_field = f"{prefix}_spatial_minus_keep_psnr_db"
                if gain_field in formal_intervals:
                    item["formal_gain_lower"] = formal_intervals[gain_field]["lower"]
                    item["formal_gain_upper"] = formal_intervals[gain_field]["upper"]
                if policy_id == "GT":
                    item["oracle_uniform_lower"] = oracle_intervals[
                        "gt_spatial_minus_uniform_psnr_db"
                    ]["lower"]
                    item["oracle_uniform_upper"] = oracle_intervals[
                        "gt_spatial_minus_uniform_psnr_db"
                    ]["upper"]
            policy_summary_rows.append(item)

    p00_gain = formal_intervals["p00_spatial_minus_keep_psnr_db"]
    p00_uniform = formal_intervals["p00_spatial_minus_uniform_psnr_db"]
    p00_shuffle = formal_intervals["p00_spatial_minus_shuffled_psnr_db"]
    p00_prevalence = material_prevalence["haze4k_train_oof"]["P00"]
    material_favorable = (
        p00_gain["lower"] > UTILITY_MARGIN_DB
        and p00_prevalence["lower"] > MIN_MATERIAL_SCENE_PREVALENCE
        and p00_uniform["lower"] > 0.0
        and p00_shuffle["lower"] > 0.0
    )
    material_futile = (
        p00_gain["upper"] <= UTILITY_MARGIN_DB
        or p00_prevalence["upper"] <= MIN_MATERIAL_SCENE_PREVALENCE
        or p00_uniform["upper"] <= 0.0
        or p00_shuffle["upper"] <= 0.0
    )
    raw_actionable_utility = (
        "favorable" if material_favorable
        else "unfavorable" if material_futile
        else "indeterminate"
    )

    oracle_gain = oracle_intervals["gt_spatial_minus_uniform_psnr_db"]
    oracle_prevalence = material_prevalence["haze4k_train_oof"]["GT"]
    oracle_headroom = (
        "favorable"
        if (
            oracle_gain["lower"] > UTILITY_MARGIN_DB
            and oracle_prevalence["lower"] > MIN_MATERIAL_SCENE_PREVALENCE
        )
        else "unfavorable"
        if (
            oracle_gain["upper"] <= UTILITY_MARGIN_DB
            or oracle_prevalence["upper"] <= MIN_MATERIAL_SCENE_PREVALENCE
        )
        else "indeterminate"
    )

    p11_rows = primary_rows + test_rows
    p11_max_abs_gain = max(
        abs(float(row["p11_spatial_minus_keep_psnr_db"]))
        for row in p11_rows
    )
    p11_max_nonkeep_fraction = max(
        float(row["p11_weaken_fraction"]) + float(row["p11_strengthen_fraction"])
        for row in p11_rows
    )
    parent_reproduction_checks = {
        "all_p11_scenes_keep": (
            p11_max_nonkeep_fraction <= P11_REPRODUCTION_TOLERANCE
        ),
        "all_p11_scene_gains_zero": (
            p11_max_abs_gain <= P11_REPRODUCTION_TOLERANCE
        ),
        "parent_training_calibration_match": parent_training_reproduction["matched"],
        "keep_identity_control": role_checks["keep_identity_control"],
    }
    parent_reproduction_pass = all(parent_reproduction_checks.values())

    raw_safety_intervals = [
        safety_intervals[population]["P00"][metric]
        for population in population_rows
        for metric in ("psnr", "ssim", "color")
    ]
    raw_image_replay_safety = (
        "safe"
        if all(item["upper"] <= MAX_HARM_PREVALENCE for item in raw_safety_intervals)
        else "unsafe"
        if any(item["lower"] > MAX_HARM_PREVALENCE for item in raw_safety_intervals)
        else "indeterminate"
    )

    calibration_suppression = (
        "favorable"
        if parent_reproduction_pass and p00_gain["lower"] > UTILITY_MARGIN_DB
        else "unfavorable"
        if parent_reproduction_pass and p00_gain["upper"] <= UTILITY_MARGIN_DB
        else "indeterminate"
    )
    precision_half_width = max(
        interval["max_half_width"] for interval in formal_intervals.values()
    )
    precision_met = precision_half_width <= 0.025

    signal_quality_rows = aggregate_signal_quality(signal_scene_rows)
    signal_lookup = {
        (row["population"], row["action"], row["metric"]): row
        for row in signal_quality_rows
    }
    signal_present_actions = []
    signal_absent_actions = []
    for action in ACTION_NAMES:
        improvement = signal_lookup[
            ("haze4k_train_oof", action, "utility_mae_improvement")
        ]
        top_contrast = signal_lookup[
            ("haze4k_train_oof", action, "observed_minus_permuted_top_lift")
        ]
        beneficial_auc = signal_lookup.get(
            ("haze4k_train_oof", action, "beneficial_auc")
        )
        present = (
            improvement["lower"] > 0.0
            and top_contrast["lower"] > 0.0
            and beneficial_auc is not None
            and beneficial_auc["lower"] > 0.5
        )
        absent = (
            improvement["upper"] <= 0.0
            or top_contrast["upper"] <= 0.0
            or beneficial_auc is None
            or beneficial_auc["upper"] <= 0.5
        )
        if present:
            signal_present_actions.append(action)
        if absent:
            signal_absent_actions.append(action)
    oof_signal_diagnostic = (
        "oof_signal_present"
        if signal_present_actions
        else "oof_signal_absent"
        if len(signal_absent_actions) == len(ACTION_NAMES)
        else "mixed_or_imprecise"
    )

    fold_fit = [
        row for row in training_diagnostic_rows
        if row["model"].startswith("oof_fold_")
    ]
    fit_positive_counts = {
        action: sum(
            row["utility_mae_improvement"] > 0.0
            for row in fold_fit
            if row["action"] == action
        )
        for action in ACTION_NAMES
    }
    training_fit_diagnostic = (
        "fit_present"
        if max(fit_positive_counts.values()) >= 4
        else "fit_absent"
        if max(fit_positive_counts.values()) <= 1
        else "mixed"
    )

    partial_intervals = {
        "P10": formal_intervals["p10_spatial_minus_keep_psnr_db"],
        "P01": formal_intervals["p01_spatial_minus_keep_psnr_db"],
    }
    partial_states = {}
    for policy_id, interval in partial_intervals.items():
        prevalence = material_prevalence["haze4k_train_oof"][policy_id]
        partial_states[policy_id] = (
            "material"
            if (
                interval["lower"] > UTILITY_MARGIN_DB
                and prevalence["lower"] > MIN_MATERIAL_SCENE_PREVALENCE
            )
            else "futile"
            if (
                interval["upper"] <= UTILITY_MARGIN_DB
                or prevalence["upper"] <= MIN_MATERIAL_SCENE_PREVALENCE
            )
            else "indeterminate"
        )
    if partial_states == {"P10": "futile", "P01": "material"}:
        calibration_attribution = "utility_correction_dominant"
    elif partial_states == {"P10": "material", "P01": "futile"}:
        calibration_attribution = "risk_correction_dominant"
    elif partial_states == {"P10": "material", "P01": "material"}:
        calibration_attribution = "joint_interaction_required"
    elif partial_states == {"P10": "futile", "P01": "futile"}:
        calibration_attribution = "either_correction_sufficient"
    else:
        calibration_attribution = "component_attribution_unresolved"

    identity_pass = all(identity_checks.values())
    role_pass = all(role_checks.values()) and parent_reproduction_pass
    gate_outcomes = {
        "evidence_identity": "pass" if identity_pass else "fail",
        "scene_role_and_coverage": "pass" if role_pass else "fail",
        "oracle_headroom_control": oracle_headroom,
        "raw_actionable_utility": raw_actionable_utility,
        "raw_image_replay_safety": raw_image_replay_safety,
        "calibration_suppression_specificity": calibration_suppression,
        "precision": "met" if precision_met else "unmet",
        "training_fit_diagnostic": (
            "favorable" if training_fit_diagnostic == "fit_present"
            else "unfavorable" if training_fit_diagnostic == "fit_absent"
            else "indeterminate"
        ),
    }

    pass_rejection_rows = aggregate_pass_rejection(pass_scene_rows)
    permutation_control = {
        "schema_version": 1,
        "independent_unit": "original_clear_scene",
        "permutation": "deterministic within-scene prediction permutation; tiles never counted as independent samples",
        "salt": PERMUTATION_SALT,
        "oof_signal_diagnostic": oof_signal_diagnostic,
        "signal_present_actions": signal_present_actions,
        "signal_absent_actions": signal_absent_actions,
        "metrics": [
            row for row in signal_quality_rows
            if row["population"] == "haze4k_train_oof"
            and (
                row["metric"].startswith("permuted_")
                or row["metric"] == "observed_minus_permuted_top_lift"
            )
        ],
    }

    summary = {
        "schema_version": 2,
        "route_id": context.route_id,
        "operation_id": context.operation_id,
        "run_id": context.run_id,
        "scope": "development-only raw-signal and calibration decomposition",
        "independent_unit": "original_clear_scene",
        "data_roles": {
            "haze4k_train_oof": TRAINING_SCENES,
            "haze4k_train_fixed_calibration": CALIBRATION_SCENES,
            "haze4k_test_development_stress": EXPECTED_TEST_SCENES,
            "haze4k_test_candidate_confirmation_touched": False,
            "nh_haze_touched": False,
            "reside_its_ots_touched": False,
        },
        "policy_contract": {
            "shared_predictor": True,
            "P00": "raw utility plus raw risk",
            "P10": "utility-corrected plus raw risk",
            "P01": "raw utility plus risk-corrected",
            "P11": "utility-corrected plus risk-corrected historical mechanism",
            "GT": "privileged positive control only; never an inference input",
            "utility_margin_db": UTILITY_MARGIN_DB,
            "risk_acceptance_max": RISK_ACCEPTANCE_MAX,
            "action_scales": {
                "keep": 1.0,
                "weaken": 0.8,
                "strengthen": 1.2,
            },
        },
        "formal_continuous_family": {
            "fields": formal_fields,
            "family_size": FORMAL_CONTINUOUS_FAMILY_SIZE,
            "critical_value": bonferroni_z(FORMAL_CONTINUOUS_FAMILY_SIZE),
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "intervals": formal_intervals,
        },
        "oracle_control": {
            "intervals": oracle_intervals,
            "material_prevalence": oracle_prevalence,
            "outcome": oracle_headroom,
        },
        "identity_checks": identity_checks,
        "role_and_coverage_checks": role_checks,
        "parent_reproduction": {
            "checks": parent_reproduction_checks,
            "training_calibration": parent_training_reproduction,
            "max_absolute_scene_gain_db": p11_max_abs_gain,
            "max_nonkeep_area_fraction": p11_max_nonkeep_fraction,
            "tolerance": P11_REPRODUCTION_TOLERANCE,
        },
        "policy_summaries": policy_summary_rows,
        "safety_intervals": safety_intervals,
        "signal_diagnostics": {
            "oof_signal": oof_signal_diagnostic,
            "training_fit": training_fit_diagnostic,
            "fit_positive_fold_counts": fit_positive_counts,
            "calibration_attribution": calibration_attribution,
            "partial_policy_states": partial_states,
        },
        "precision": {
            "target_half_width_db": 0.025,
            "observed_max_simultaneous_half_width_db": precision_half_width,
            "met": precision_met,
            "planning_assumption": (
                "conditional on the predeclared 0.2358753949 dB planning-SD "
                "upper bound inherited from the archived proxy interval"
            ),
        },
        "gate_outcomes": gate_outcomes,
        "raw_artifact_inventory": raw_inventory,
        "limitations": [
            "All evidence is development-only; no candidate-confirmation, canary, locked-test, NH-Haze, RESIDE ITS, or RESIDE OTS data were read.",
            "The 600-scene OOF estimand uses one outcome-blind selected haze variant per original clear scene.",
            "The four test-development haze variants are nested repeats and are averaged within each original clear scene.",
            "The fixed 150-scene calibration set is a shared nuisance resource and is not counted in the primary estimand.",
            "The GT oracle is a privileged positive control and cannot support deployment or promotion.",
            "The precision feasibility certificate is conditional on its archived planning-SD assumption; the observed simultaneous half-width remains decisive.",
            "A PASS authorizes calibration/risk-interface contract authoring only, never ITS/OTS execution, promotion, deployment, or a source-disjoint claim.",
        ],
        "marker": "HAZE4K_OBSERVABLE_SIGNAL_CALIBRATION_DECOMPOSITION_COMPLETE",
    }

    summary_path = output_file(
        context, "haze4k_observable_signal_calibration_decomposition_summary.json",
    )
    gate_path = output_file(
        context, "haze4k_observable_signal_calibration_decomposition_gate_summary.json",
    )
    policy_path = output_file(
        context, "haze4k_observable_signal_calibration_decomposition_policy_summary.csv",
    )
    signal_path = output_file(
        context, "haze4k_observable_signal_calibration_decomposition_signal_quality.csv",
    )
    pass_path = output_file(
        context, "haze4k_observable_signal_calibration_decomposition_pass_rejection.csv",
    )
    training_path = output_file(
        context, "haze4k_observable_signal_calibration_decomposition_training_diagnostics.csv",
    )
    calibration_path = output_file(
        context, "haze4k_observable_signal_calibration_decomposition_calibration_diagnostics.csv",
    )
    scene_policy_path = output_file(
        context,
        "haze4k_observable_signal_calibration_decomposition_scene_policy_metrics.csv",
    )
    scene_pass_path = output_file(
        context,
        "haze4k_observable_signal_calibration_decomposition_scene_pass_rejection.csv",
    )
    permutation_path = output_file(
        context, "haze4k_observable_signal_calibration_decomposition_permutation_control.json",
    )
    inventory_path = output_file(
        context, "haze4k_observable_signal_calibration_decomposition_raw_inventory.json",
    )
    atomic_json(summary_path, summary)
    atomic_json(gate_path, {
        "schema_version": 2,
        "gate_outcomes": gate_outcomes,
        "raw_materiality": {
            "favorable": material_favorable,
            "futile": material_futile,
            "gain": p00_gain,
            "material_prevalence": p00_prevalence,
            "spatial_over_uniform": p00_uniform,
            "spatial_over_shuffle": p00_shuffle,
        },
        "raw_safety": safety_intervals,
        "parent_reproduction": parent_reproduction_checks,
        "precision_met": precision_met,
    })
    write_csv(policy_path, policy_summary_rows)
    write_csv(signal_path, signal_quality_rows)
    write_csv(pass_path, pass_rejection_rows)
    write_csv(training_path, training_diagnostic_rows)
    write_csv(calibration_path, calibration_diagnostic_rows)
    all_scene_rows = sorted(primary_rows + test_rows, key=lambda row: (
        str(row["population"]), str(row["scene"]),
    ))
    identity_fields = {"population", "scene", "fold", "model"}
    policy_prefixes = tuple(
        f"{policy_id.lower()}_" for policy_id in (*POLICY_IDS, "GT")
    )
    write_csv(scene_policy_path, [
        {
            key: value for key, value in row.items()
            if key in identity_fields or key.startswith(policy_prefixes)
        }
        for row in all_scene_rows
    ])
    write_csv(scene_pass_path, sorted(pass_scene_rows, key=lambda row: (
        str(row["population"]), str(row["scene"]), str(row["action"]),
    )))
    atomic_json(permutation_path, permutation_control)
    atomic_json(inventory_path, raw_inventory)

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
            "formal_family_size": FORMAL_CONTINUOUS_FAMILY_SIZE,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "raw_inventory_sha256": raw_inventory["inventory_sha256"],
            "summary_file": summary_path.name,
            "gate_summary_file": gate_path.name,
            "policy_summary_file": policy_path.name,
            "signal_quality_file": signal_path.name,
            "pass_rejection_file": pass_path.name,
            "training_diagnostics_file": training_path.name,
            "calibration_diagnostics_file": calibration_path.name,
            "scene_policy_metrics_file": scene_policy_path.name,
            "scene_pass_rejection_file": scene_pass_path.name,
            "permutation_control_file": permutation_path.name,
            "raw_inventory_file": inventory_path.name,
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
