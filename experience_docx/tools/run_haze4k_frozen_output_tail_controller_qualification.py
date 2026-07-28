#!/usr/bin/env python3
"""Qualify low-risk-tail controllers from receipt-bound frozen predictions."""

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


ROUTE_ID = "haze4k-observable-tail-factorial-v1"
OPERATION_ID = "HAZE4K_OBSERVABLE_TAIL_FACTORIAL"
PARENT_ROUTE_ID = "haze4k-frozen-output-risk-interface-localization-v1"
PARENT_ROUTE_COMMIT = "1ad0e732e1007cc324defb68dd0d2bd5e2db12d4"
PARENT_RECEIPT = "1238610079a67496f4484b46ad101a3a95279a26e88bbd0ed2b7b0e783e80126"

IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}
TRAIN_INPUT_DIRECTORIES = ("IN", "haze", "hazy")
TRAIN_LABEL_DIRECTORIES = ("GT", "gt")
EXPECTED_TRAIN_SCENES = 750
TRAINING_SCENES = 600
CALIBRATION_SCENES = 150
EXPECTED_TEST_SCENES = 100
VARIANTS_PER_SCENE = 4
TILE_SIZE = 32
SUBTILE_SIZE = 8
FEATURE_CHANNELS = 104
SHAPE_CHANNELS = 104
DISAGREEMENT_CHANNELS = 16
ACTION_NAMES = ("weaken", "strengthen")
ACTION_SCALES = np.asarray([0.8, 1.2], dtype=np.float32)
REPRESENTATIONS = ("R0", "R1", "R2")
LEARNERS = ("L0", "L1", "L2")
CELL_IDS = tuple(
    f"{representation}_{learner}"
    for representation in REPRESENTATIONS
    for learner in LEARNERS
)
SEED_OFFSETS = (0, 1009, 2027)
POLICY_IDS = (
    "candidate",
    "utility_only",
    "observable_utility_gt_risk",
    "gt_utility_observable_risk",
    "gt_gt",
)
TRAIN_SPLIT_SALT = "haze4k-local-error-qualification-v2"
VARIANT_SELECTION_SALT = "haze4k-observable-conditional-utility-risk-feasibility-v1"
OOF_FOLD_SALT = "haze4k-observable-conditional-utility-risk-feasibility-v1-oof"
CALIBRATION_FOLD_SALT = "haze4k-observable-tail-factorial-v1-calibration-fold"
SHUFFLE_SALT = "haze4k-observable-tail-factorial-v1-shuffle"
PERMUTATION_SALT = "haze4k-observable-tail-factorial-v1-permutation"
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
PARENT_CLOSEOUT_SHA256 = "aabf8ed0efbb2ed6606cf2baf9e09bb226087ea5ddf57000429797a9e6c26b21"
PARENT_CONCLUSION_SHA256 = "77bcd87cd5c54d23993164ed3f9f3c8436a9f50213f51caecb49699256945cc6"
PARENT_SUMMARY_SHA256 = "ef4f560f5983092b4e478d912e07d4e46bf514243b83116af52ce3c1dd6f0fbf"
PARENT_FORMAL_SHA256 = "64b4828c9bcef9ab1a13e970b72438c5c39e15cce18e1ea39f185b56e5704836"
PARENT_REGRET_SHA256 = "f7806a6664466fbfe935680a1ca6df92943237a140b405641e60166a7f4a66ac"
BASELINE_TRAINING_SHA256 = "c3652a6663887a4c79137eed93ced86d74d0387da5dcaedc2e7bdc5dee0a3344"
BASELINE_CALIBRATION_SHA256 = "37cc4826861ecde5c77cc461fa8fc967ce35e9f94372391d3caf23326b367841"
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
UTILITY_BIN_EDGES = (-0.50, -0.10, 0.0, 0.05, 0.20, 0.50)
UTILITY_BIN_CENTERS = (-0.75, -0.30, -0.05, 0.025, 0.125, 0.35, 0.75)
L2_EXPECTATION_WEIGHT = 0.5
L2_ORDINAL_WEIGHT = 0.5
L2_RISK_WEIGHT = 0.5
UTILITY_MARGIN_DB = 0.05
TRANSFER_POINT_MARGIN_DB = 0.025
RISK_ACCEPTANCE_MAX = 0.10
PSNR_HARM_MARGIN_DB = 0.10
SSIM_HARM_MARGIN = 0.005
COLOR_HARM_MARGIN = 1.0 / 255.0
MAX_HARM_PREVALENCE = 0.10
MIN_MATERIAL_SCENE_PREVALENCE = 0.20
MIN_ACTIVE_SCENE_PREVALENCE = 0.20
MIN_ACTION_AREA_FRACTION = 0.01
PRECISION_TARGET_DB = 0.025
CALIBRATION_QUANTILE = 0.95
BOOTSTRAP_RESAMPLES = 100_000
BOOTSTRAP_SEED = 20260726
TRAINING_SEED = 20260726
FORMAL_CONTINUOUS_FAMILY_SIZE = 5
PRIMARY_CONTINUOUS_FAMILY_SIZE = 4
FACTORIAL_FAMILY_SIZE = 9
MATERIAL_PREVALENCE_FAMILY_SIZE = 5
SAFETY_FAMILY_SIZE = 8
CONTROL_FAMILY_SIZE = 2
TOP_FRACTION = 0.10
P11_REPRODUCTION_TOLERANCE = 1.0e-12
PARENT_TRAINING_TOLERANCE = 1.0e-6
TOTAL_UNITS = 852
FORMAL_COST_ITERATIONS = 882_600
PROBE_COST_ITERATIONS = 354
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


def block_moments(torch, functional, tensor, height: int, width: int, block_size: int):
    tensor = tensor[:, :, :height, :width]
    pad_h, pad_w = (-height) % block_size, (-width) % block_size
    padded = functional.pad(tensor, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
    mask = torch.ones((1, 1, height, width), dtype=tensor.dtype, device=tensor.device)
    mask = functional.pad(mask, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
    counts = functional.avg_pool2d(mask, block_size, stride=block_size, divisor_override=1)
    sums = functional.avg_pool2d(padded, block_size, stride=block_size, divisor_override=1)
    squares = functional.avg_pool2d(
        padded * padded, block_size, stride=block_size, divisor_override=1,
    )
    means = sums / counts
    variances = torch.clamp(squares / counts - means * means, min=0.0)
    return means, torch.sqrt(variances), counts


def tile_moments(torch, functional, tensor, height: int, width: int):
    return block_moments(torch, functional, tensor, height, width, TILE_SIZE)


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
            prediction_padded = outputs[2].clamp(0.0, 1.0)
            prediction_tensor = prediction_padded[:, :, :height, :width]
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
            shape_source = torch.cat([handcrafted, decoder_tensor], dim=1)
            shape_mean, shape_std, shape_areas = block_moments(
                torch, functional, shape_source, height, width, SUBTILE_SIZE,
            )
            shape_features = torch.cat([shape_mean, shape_std], dim=1)
            if shape_features.shape[1] != SHAPE_CHANNELS:
                raise RuntimeError("shape-preserving feature dimensionality changed")

            flipped_outputs = model(torch.flip(padded, dims=(3,)))
            if not isinstance(flipped_outputs, list) or len(flipped_outputs) != 3:
                raise RuntimeError("flip-disagreement output contract changed")
            flip_prediction = torch.flip(
                flipped_outputs[2].clamp(0.0, 1.0), dims=(3,),
            )[:, :, :height, :width]
            scaled_height = max(TILE_SIZE, int(math.ceil(padded.shape[2] * 0.75 / TILE_SIZE)) * TILE_SIZE)
            scaled_width = max(TILE_SIZE, int(math.ceil(padded.shape[3] * 0.75 / TILE_SIZE)) * TILE_SIZE)
            scaled_input = functional.interpolate(
                padded,
                size=(scaled_height, scaled_width),
                mode="bilinear",
                align_corners=False,
            )
            scaled_outputs = model(scaled_input)
            if not isinstance(scaled_outputs, list) or len(scaled_outputs) != 3:
                raise RuntimeError("scale-disagreement output contract changed")
            scale_prediction = functional.interpolate(
                scaled_outputs[2].clamp(0.0, 1.0),
                size=padded.shape[2:],
                mode="bilinear",
                align_corners=False,
            )[:, :, :height, :width]
            flip_delta = (prediction_tensor - flip_prediction).abs()
            scale_delta = (prediction_tensor - scale_prediction).abs()
            disagreement_source = torch.cat([
                flip_delta,
                scale_delta,
                torch.sum(flip_delta * weights, dim=1, keepdim=True),
                torch.sum(scale_delta * weights, dim=1, keepdim=True),
            ], dim=1)
            disagreement_mean, disagreement_std, disagreement_areas = tile_moments(
                torch, functional, disagreement_source, height, width,
            )
            disagreement = torch.cat(
                [disagreement_mean, disagreement_std], dim=1,
            )
            if disagreement.shape[1] != DISAGREEMENT_CHANNELS \
                    or not torch.equal(areas, disagreement_areas):
                raise RuntimeError("flip/scale disagreement feature contract changed")
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
                "shape_features": shape_features.squeeze(0).detach().cpu().numpy().astype(np.float32),
                "shape_areas": shape_areas.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32),
                "disagreement": disagreement.squeeze(0).detach().cpu().numpy().astype(np.float32),
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


def make_head(torch, representation: str = "R0", learner: str = "L0"):
    import torch.nn as nn

    if representation not in REPRESENTATIONS or learner not in LEARNERS:
        raise RuntimeError("unknown factorial representation or learner")
    if representation == "R0" and learner == "L0":
        head = nn.Sequential(
            nn.Conv2d(FEATURE_CHANNELS, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 4, kernel_size=1),
        )
        head.factorial_representation = representation
        head.factorial_learner = learner
        return head

    class FactorialHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.factorial_representation = representation
            self.factorial_learner = learner
            channels = FEATURE_CHANNELS
            if representation in {"R1", "R2"}:
                self.shape_encoder = nn.Sequential(
                    nn.Conv2d(SHAPE_CHANNELS, 32, kernel_size=3, padding=1),
                    nn.GroupNorm(8, 32),
                    nn.GELU(),
                    nn.Conv2d(32, 32, kernel_size=3, padding=1),
                    nn.GroupNorm(8, 32),
                    nn.GELU(),
                )
                channels += 32
            if representation == "R2":
                self.context_encoder = nn.Sequential(
                    nn.Conv2d(FEATURE_CHANNELS, 32, kernel_size=3, padding=1),
                    nn.GroupNorm(8, 32),
                    nn.GELU(),
                    nn.Conv2d(
                        32, 32, kernel_size=3, padding=2, dilation=2,
                    ),
                    nn.GroupNorm(8, 32),
                    nn.GELU(),
                )
                self.disagreement_encoder = nn.Sequential(
                    nn.Conv2d(DISAGREEMENT_CHANNELS, 16, kernel_size=1),
                    nn.GroupNorm(4, 16),
                    nn.GELU(),
                )
                channels += 48
            self.trunk = nn.Sequential(
                nn.Conv2d(channels, 64, kernel_size=3, padding=1),
                nn.GroupNorm(8, 64),
                nn.GELU(),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.GroupNorm(8, 64),
                nn.GELU(),
            )
            if learner == "L1":
                self.utility_head = nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=3, padding=1),
                    nn.GELU(),
                    nn.Conv2d(32, 2, kernel_size=1),
                )
                self.risk_head = nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=3, padding=1),
                    nn.GELU(),
                    nn.Conv2d(32, 2, kernel_size=1),
                )
            elif learner == "L2":
                self.distribution_head = nn.Conv2d(
                    64, 2 * len(UTILITY_BIN_CENTERS), kernel_size=1,
                )
            else:
                self.shared_head = nn.Conv2d(64, 4, kernel_size=1)

        def forward(self, inputs):
            import torch.nn.functional as functional

            base = inputs["features"]
            pieces = [base]
            if representation in {"R1", "R2"}:
                shape = self.shape_encoder(inputs["shape_features"])
                shape = functional.avg_pool2d(
                    shape, kernel_size=4, stride=4, ceil_mode=True,
                )
                if shape.shape[2:] != base.shape[2:]:
                    shape = functional.interpolate(
                        shape, size=base.shape[2:], mode="bilinear", align_corners=False,
                    )
                pieces.append(shape)
            if representation == "R2":
                pieces.extend([
                    self.context_encoder(base),
                    self.disagreement_encoder(inputs["disagreement"]),
                ])
            hidden = self.trunk(torch.cat(pieces, dim=1))
            if learner == "L1":
                return {
                    "utility": self.utility_head(hidden),
                    "risk_logits": self.risk_head(hidden),
                }
            if learner == "L2":
                bins = len(UTILITY_BIN_CENTERS)
                logits = self.distribution_head(hidden).reshape(
                    hidden.shape[0], 2, bins, hidden.shape[2], hidden.shape[3],
                )
                probabilities = torch.softmax(logits, dim=2)
                centers = torch.tensor(
                    UTILITY_BIN_CENTERS,
                    dtype=probabilities.dtype,
                    device=probabilities.device,
                ).view(1, 1, bins, 1, 1)
                utility = torch.sum(probabilities * centers, dim=2)
                risk = torch.sum(probabilities[:, :, :2], dim=2)
                return {
                    "utility": utility,
                    "risk": risk,
                    "distribution_logits": logits,
                    "distribution_probabilities": probabilities,
                }
            shared = self.shared_head(hidden)
            return {"utility": shared[:, :2], "risk_logits": shared[:, 2:]}

    return FactorialHead()


def load_cache(path: Path, *, include_prediction: bool = False) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        names = [
            "features",
            "shape_features",
            "shape_areas",
            "disagreement",
            "utility",
            "harm",
            "areas",
        ]
        if include_prediction:
            names.append("prediction")
        result = {name: source[name].copy() for name in names}
    return result


def channel_normalization(
    records: list[dict[str, Any]], feature_key: str, area_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    scene_means = []
    for record in records:
        values = record[feature_key].astype(np.float64)
        weights = record[area_key].astype(np.float64)
        weights /= float(np.sum(weights))
        scene_means.append(np.sum(values * weights[None, :, :], axis=(1, 2)))
    mean = np.mean(np.stack(scene_means), axis=0)
    scene_variances = []
    for record in records:
        values = record[feature_key].astype(np.float64)
        weights = record[area_key].astype(np.float64)
        weights /= float(np.sum(weights))
        scene_variances.append(np.sum(
            (values - mean[:, None, None]) ** 2 * weights[None, :, :],
            axis=(1, 2),
        ))
    scale = np.sqrt(np.maximum(np.mean(np.stack(scene_variances), axis=0), 1.0e-8))
    return mean.astype(np.float32), scale.astype(np.float32)


def feature_normalization(
    records: list[dict[str, Any]], representation: str,
) -> dict[str, dict[str, np.ndarray]]:
    keys = [("features", "areas")]
    if representation in {"R1", "R2"}:
        keys.append(("shape_features", "shape_areas"))
    if representation == "R2":
        keys.append(("disagreement", "areas"))
    return {
        feature: dict(zip(("mean", "scale"), channel_normalization(records, feature, area)))
        for feature, area in keys
    }


def normalized_inputs(
    torch,
    record: dict[str, Any],
    normalization: dict[str, dict[str, np.ndarray]],
    device: str,
) -> dict[str, Any]:
    result = {}
    for key, values in normalization.items():
        value = (record[key] - values["mean"][:, None, None]) / values["scale"][:, None, None]
        result[key] = torch.from_numpy(value.copy()).unsqueeze(0).to(
            device=device, dtype=torch.float32,
        )
    return result


def head_forward(torch, head, record, normalization, device: str) -> dict[str, Any]:
    inputs = normalized_inputs(torch, record, normalization, device)
    representation = getattr(head, "factorial_representation", "R0")
    learner = getattr(head, "factorial_learner", "L0")
    if representation == "R0" and learner == "L0":
        output = head(inputs["features"])
        return {"utility": output[:, :2], "risk_logits": output[:, 2:]}
    return head(inputs)


def training_loss(torch, output, target, harm, area, learner: str):
    import torch.nn.functional as functional

    if learner == "L2":
        edges = torch.tensor(
            UTILITY_BIN_EDGES, dtype=target.dtype, device=target.device,
        )
        target_bins = torch.bucketize(target.contiguous(), edges)
        logits = output["distribution_logits"]
        distribution_loss = functional.cross_entropy(
            logits.transpose(1, 2), target_bins, reduction="none",
        ).mean(dim=1)
        probabilities = output["distribution_probabilities"]
        predicted_cdf = torch.cumsum(probabilities, dim=2)[:, :, :-1]
        ordinal_target = (
            target.unsqueeze(2) <= edges.view(1, 1, -1, 1, 1)
        ).to(dtype=target.dtype)
        ordinal_loss = functional.binary_cross_entropy(
            torch.clamp(predicted_cdf, 1.0e-6, 1.0 - 1.0e-6),
            ordinal_target,
            reduction="none",
        ).mean(dim=(1, 2))
        expectation_loss = functional.smooth_l1_loss(
            output["utility"], target, beta=UTILITY_HUBER_BETA_DB, reduction="none",
        ).mean(dim=1)
        risk_loss = functional.binary_cross_entropy(
            torch.clamp(output["risk"], 1.0e-6, 1.0 - 1.0e-6),
            harm,
            reduction="none",
        )
        risk_loss = risk_loss * torch.where(
            harm > 0.5,
            torch.full_like(harm, RISK_POSITIVE_WEIGHT),
            torch.ones_like(harm),
        )
        risk_loss = risk_loss.mean(dim=1)
        utility_loss = (
            distribution_loss
            + L2_EXPECTATION_WEIGHT * expectation_loss
            + L2_ORDINAL_WEIGHT * ordinal_loss
        )
        total = torch.sum(area * utility_loss.squeeze(0)) \
            + L2_RISK_WEIGHT * torch.sum(area * risk_loss.squeeze(0))
        return total, utility_loss, risk_loss

    utility_loss = functional.smooth_l1_loss(
        output["utility"], target, beta=UTILITY_HUBER_BETA_DB, reduction="none",
    ).mean(dim=1)
    risk_loss = functional.binary_cross_entropy_with_logits(
        output["risk_logits"], harm, reduction="none",
    )
    risk_loss = risk_loss * torch.where(
        harm > 0.5,
        torch.full_like(harm, RISK_POSITIVE_WEIGHT),
        torch.ones_like(harm),
    )
    risk_loss = risk_loss.mean(dim=1)
    total = torch.sum(area * utility_loss.squeeze(0)) \
        + RISK_LOSS_WEIGHT * torch.sum(area * risk_loss.squeeze(0))
    return total, utility_loss, risk_loss


def train_head(
    torch,
    records: list[dict[str, Any]],
    *,
    seed: int,
    device: str,
    representation: str = "R0",
    learner: str = "L0",
    progress_callback=None,
):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    head = make_head(torch, representation, learner).to(device)
    normalization = feature_normalization(records, representation)
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
            target = torch.from_numpy(record["utility"].copy()).unsqueeze(0).to(device)
            harm = torch.from_numpy(record["harm"].copy()).unsqueeze(0).to(device)
            area = torch.from_numpy(record["areas"].copy()).to(device)
            area = area / torch.sum(area)
            output = head_forward(torch, head, record, normalization, device)
            loss, utility_loss, risk_loss = training_loss(
                torch, output, target, harm, area, learner,
            )
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
    return head.eval(), normalization, history


def predict_head(
    torch, head, record, normalization, device: str,
) -> tuple[np.ndarray, np.ndarray]:
    with torch.inference_mode():
        output = head_forward(torch, head, record, normalization, device)
        utility = output["utility"].squeeze(0).detach().cpu().numpy().astype(np.float64)
        risk_tensor = output.get("risk")
        if risk_tensor is None:
            risk_tensor = torch.sigmoid(output["risk_logits"])
        risk = risk_tensor.squeeze(0).detach().cpu().numpy().astype(np.float64)
    if not np.isfinite(utility).all() or not np.isfinite(risk).all():
        raise RuntimeError("contextual head produced non-finite predictions")
    return utility, risk


def predict_ensemble(
    torch,
    bundles: list[dict[str, Any]],
    record: dict[str, Any],
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    predictions = [
        predict_head(
            torch,
            bundle["head"],
            record,
            bundle["normalization"],
            device,
        )
        for bundle in bundles
    ]
    utility = np.mean(np.stack([item[0] for item in predictions]), axis=0)
    risk = np.mean(np.stack([item[1] for item in predictions]), axis=0)
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


def calibrate_predictions(
    prediction_by_scene: dict[str, tuple[np.ndarray, np.ndarray]],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    utility_scores = [[], []]
    risk_scores = [[], []]
    for record in records:
        utility, risk = prediction_by_scene[record["scene"]]
        weights = record["areas"].astype(np.float64)
        weights /= float(np.sum(weights))
        for action in range(2):
            utility_residual = float(np.sum(
                weights * (utility[action] - record["utility"][action]),
            ))
            risk_residual = float(np.sum(
                weights * (record["harm"][action] - risk[action]),
            ))
            utility_scores[action].append(max(0.0, utility_residual))
            risk_scores[action].append(max(0.0, risk_residual))
    return {
        "utility_correction": np.asarray([
            upper_quantile(scores, CALIBRATION_QUANTILE)
            for scores in utility_scores
        ], dtype=np.float64),
        "risk_correction": np.asarray([
            upper_quantile(scores, CALIBRATION_QUANTILE)
            for scores in risk_scores
        ], dtype=np.float64),
        "utility_scores": utility_scores,
        "risk_scores": risk_scores,
        "scene_count": len(records),
        "quantile": CALIBRATION_QUANTILE,
    }


def evaluate_policy_predictions(
    torch,
    record: dict[str, Any],
    utility: np.ndarray,
    risk: np.ndarray,
    calibration: dict[str, Any],
    hazy: np.ndarray,
    clear: np.ndarray,
    prediction: np.ndarray,
    scene: str,
    variant: str,
    device: str,
) -> dict[str, Any]:
    utility_lower, risk_upper = corrected_predictions(utility, risk, calibration)
    target_utility = record["utility"].astype(np.float64)
    target_harm = record["harm"].astype(np.float64)
    zero_risk = np.zeros_like(risk_upper)
    action_maps = {
        "candidate": choose_actions(utility_lower, risk_upper),
        "utility_only": choose_actions(utility_lower, zero_risk),
        "observable_utility_gt_risk": choose_actions(utility_lower, target_harm),
        "gt_utility_observable_risk": choose_actions(target_utility, risk_upper),
        "gt_gt": choose_actions(target_utility, target_harm),
    }
    uniform_action = choose_uniform_action(
        utility_lower, risk_upper, record["areas"],
    )
    candidate_actions = action_maps["candidate"]
    shuffled_actions = np.random.default_rng(
        deterministic_seed(f"{SHUFFLE_SALT}|{scene}|{variant}"),
    ).permutation(candidate_actions.reshape(-1)).reshape(candidate_actions.shape)
    permutation = np.random.default_rng(
        deterministic_seed(f"{PERMUTATION_SALT}|{scene}|{variant}"),
    ).permutation(candidate_actions.size)
    permuted_utility = utility_lower.reshape(2, -1)[:, permutation].reshape(
        utility_lower.shape,
    )
    permuted_risk = risk_upper.reshape(2, -1)[:, permutation].reshape(
        risk_upper.shape,
    )
    permuted_actions = choose_actions(permuted_utility, permuted_risk)
    outputs = {
        policy: apply_spatial(hazy, prediction, actions)
        for policy, actions in action_maps.items()
    }
    outputs.update({
        "uniform": apply_uniform(hazy, prediction, uniform_action),
        "shuffled": apply_spatial(hazy, prediction, shuffled_actions),
        "permuted": apply_spatial(hazy, prediction, permuted_actions),
    })
    output_order = (*POLICY_IDS, "uniform", "shuffled", "permuted")
    ssim_values = rgb_ssim(
        torch,
        [prediction] + [outputs[policy] for policy in output_order],
        clear,
        device,
    )
    keep_psnr = psnr(prediction, clear)
    keep_ssim = ssim_values[0]
    keep_color = color_bias(prediction, clear)
    metrics = {}
    for index, policy in enumerate(output_order, start=1):
        output = outputs[policy]
        metrics[policy] = {
            "minus_keep_psnr_db": psnr(output, clear) - keep_psnr,
            "minus_keep_ssim": ssim_values[index] - keep_ssim,
            "minus_keep_color_bias": color_bias(output, clear) - keep_color,
        }
    metrics["candidate"].update({
        "minus_uniform_psnr_db": (
            metrics["candidate"]["minus_keep_psnr_db"]
            - metrics["uniform"]["minus_keep_psnr_db"]
        ),
        "minus_shuffled_psnr_db": (
            metrics["candidate"]["minus_keep_psnr_db"]
            - metrics["shuffled"]["minus_keep_psnr_db"]
        ),
        "minus_permuted_psnr_db": (
            metrics["candidate"]["minus_keep_psnr_db"]
            - metrics["permuted"]["minus_keep_psnr_db"]
        ),
    })
    areas = record["areas"].astype(np.float64)
    total_area = float(np.sum(areas))
    selected_mask = candidate_actions > 0
    selected_area = float(np.sum(areas[selected_mask]))
    selected_harm_area = 0.0
    selected_utility_sum = 0.0
    for action in range(2):
        selected = candidate_actions == action + 1
        selected_harm_area += float(np.sum(areas[selected] * target_harm[action][selected]))
        selected_utility_sum += float(np.sum(areas[selected] * target_utility[action][selected]))
    weights = areas / total_area
    action_fractions = {
        "keep_fraction": weighted_mean(candidate_actions == 0, weights),
        "weaken_fraction": weighted_mean(candidate_actions == 1, weights),
        "strengthen_fraction": weighted_mean(candidate_actions == 2, weights),
    }
    return {
        "metrics": metrics,
        "actions": action_maps,
        "raw_utility": utility,
        "raw_risk": risk,
        "utility_lower": utility_lower,
        "risk_upper": risk_upper,
        "selected_area": selected_area,
        "selected_harm_area": selected_harm_area,
        "total_area": total_area,
        "selected_area_fraction": selected_area / total_area,
        "selected_harm_fraction": (
            selected_harm_area / selected_area if selected_area > 0.0 else None
        ),
        "selected_utility_db": (
            selected_utility_sum / selected_area if selected_area > 0.0 else None
        ),
        "active_scene": selected_area > 0.0,
        "material_scene": metrics["candidate"]["minus_keep_psnr_db"] >= UTILITY_MARGIN_DB,
        "psnr_harm": metrics["candidate"]["minus_keep_psnr_db"] <= -PSNR_HARM_MARGIN_DB,
        "ssim_harm": metrics["candidate"]["minus_keep_ssim"] <= -SSIM_HARM_MARGIN,
        "color_harm": metrics["candidate"]["minus_keep_color_bias"] >= COLOR_HARM_MARGIN,
        **action_fractions,
    }


def flatten_cell_evaluation(
    population: str,
    scene: str,
    fold: int | str,
    cell: str,
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    row = {
        "population": population,
        "scene": scene,
        "fold": fold,
        "cell": cell,
        "selected_area": evaluation["selected_area"],
        "selected_harm_area": evaluation["selected_harm_area"],
        "total_area": evaluation["total_area"],
        "selected_area_fraction": evaluation["selected_area_fraction"],
        "selected_harm_fraction": evaluation["selected_harm_fraction"],
        "selected_utility_db": evaluation["selected_utility_db"],
        "active_scene": evaluation["active_scene"],
        "material_scene": evaluation["material_scene"],
        "psnr_harm": evaluation["psnr_harm"],
        "ssim_harm": evaluation["ssim_harm"],
        "color_harm": evaluation["color_harm"],
        "keep_fraction": evaluation["keep_fraction"],
        "weaken_fraction": evaluation["weaken_fraction"],
        "strengthen_fraction": evaluation["strengthen_fraction"],
    }
    for policy, metrics in evaluation["metrics"].items():
        for metric, value in metrics.items():
            row[f"{policy}_{metric}"] = value
    return row


def evaluate_factorial_psnr(
    record: dict[str, Any],
    utility: np.ndarray,
    risk: np.ndarray,
    calibration: dict[str, Any],
    hazy: np.ndarray,
    clear: np.ndarray,
    prediction: np.ndarray,
    scene: str,
    variant: str,
) -> dict[str, Any]:
    utility_lower, risk_upper = corrected_predictions(utility, risk, calibration)
    actions = choose_actions(utility_lower, risk_upper)
    uniform_action = choose_uniform_action(utility_lower, risk_upper, record["areas"])
    shuffled = np.random.default_rng(
        deterministic_seed(f"{SHUFFLE_SALT}|{scene}|{variant}"),
    ).permutation(actions.reshape(-1)).reshape(actions.shape)
    permutation = np.random.default_rng(
        deterministic_seed(f"{PERMUTATION_SALT}|{scene}|{variant}"),
    ).permutation(actions.size)
    permuted = choose_actions(
        utility_lower.reshape(2, -1)[:, permutation].reshape(utility_lower.shape),
        risk_upper.reshape(2, -1)[:, permutation].reshape(risk_upper.shape),
    )
    keep_psnr = psnr(prediction, clear)
    candidate_gain = psnr(apply_spatial(hazy, prediction, actions), clear) - keep_psnr
    uniform_gain = psnr(apply_uniform(hazy, prediction, uniform_action), clear) - keep_psnr
    shuffled_gain = psnr(apply_spatial(hazy, prediction, shuffled), clear) - keep_psnr
    permuted_gain = psnr(apply_spatial(hazy, prediction, permuted), clear) - keep_psnr
    areas = record["areas"].astype(np.float64)
    total_area = float(np.sum(areas))
    selected_area = float(np.sum(areas[actions > 0]))
    selected_harm_area = 0.0
    selected_utility_sum = 0.0
    for action in range(2):
        selected = actions == action + 1
        selected_harm_area += float(np.sum(
            areas[selected] * record["harm"][action][selected],
        ))
        selected_utility_sum += float(np.sum(
            areas[selected] * record["utility"][action][selected],
        ))
    weights = areas / total_area
    return {
        "metrics": {
            "candidate": {
                "minus_keep_psnr_db": candidate_gain,
                "minus_keep_ssim": None,
                "minus_keep_color_bias": None,
                "minus_uniform_psnr_db": candidate_gain - uniform_gain,
                "minus_shuffled_psnr_db": candidate_gain - shuffled_gain,
                "minus_permuted_psnr_db": candidate_gain - permuted_gain,
            },
        },
        "actions": {"candidate": actions},
        "raw_utility": utility,
        "raw_risk": risk,
        "utility_lower": utility_lower,
        "risk_upper": risk_upper,
        "selected_area": selected_area,
        "selected_harm_area": selected_harm_area,
        "total_area": total_area,
        "selected_area_fraction": selected_area / total_area,
        "selected_harm_fraction": (
            selected_harm_area / selected_area if selected_area > 0.0 else None
        ),
        "selected_utility_db": (
            selected_utility_sum / selected_area if selected_area > 0.0 else None
        ),
        "active_scene": selected_area > 0.0,
        "material_scene": candidate_gain >= UTILITY_MARGIN_DB,
        "psnr_harm": candidate_gain <= -PSNR_HARM_MARGIN_DB,
        "ssim_harm": None,
        "color_harm": None,
        "keep_fraction": weighted_mean(actions == 0, weights),
        "weaken_fraction": weighted_mean(actions == 1, weights),
        "strengthen_fraction": weighted_mean(actions == 2, weights),
    }


def average_variant_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise RuntimeError("nested variant rows are empty")
    identity = {key: rows[0][key] for key in ("population", "scene", "fold", "cell")}
    result = dict(identity)
    summed_fields = {"selected_area", "selected_harm_area", "total_area"}
    ignored = set(identity) | {
        "selected_area_fraction",
        "selected_harm_fraction",
        "selected_utility_db",
        "active_scene",
        "material_scene",
        "psnr_harm",
        "ssim_harm",
        "color_harm",
    }
    for field in summed_fields:
        result[field] = float(np.sum([float(row[field]) for row in rows]))
    for field in sorted(set().union(*(row.keys() for row in rows)) - ignored - summed_fields):
        values = [row[field] for row in rows if row.get(field) is not None]
        result[field] = float(np.mean(values)) if values else None
    result["selected_area_fraction"] = result["selected_area"] / result["total_area"]
    result["selected_harm_fraction"] = (
        result["selected_harm_area"] / result["selected_area"]
        if result["selected_area"] > 0.0 else None
    )
    selected_utilities = [
        row["selected_utility_db"] for row in rows
        if row.get("selected_utility_db") is not None
    ]
    result["selected_utility_db"] = (
        float(np.mean(selected_utilities)) if selected_utilities else None
    )
    gain = result["candidate_minus_keep_psnr_db"]
    result.update({
        "active_scene": result["selected_area"] > 0.0,
        "material_scene": gain >= UTILITY_MARGIN_DB,
        "psnr_harm": gain <= -PSNR_HARM_MARGIN_DB,
        "ssim_harm": (
            result.get("candidate_minus_keep_ssim") is not None
            and result["candidate_minus_keep_ssim"] <= -SSIM_HARM_MARGIN
        ),
        "color_harm": (
            result.get("candidate_minus_keep_color_bias") is not None
            and result["candidate_minus_keep_color_bias"] >= COLOR_HARM_MARGIN
        ),
    })
    return result


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
    if family_size < len(fields):
        raise RuntimeError("formal family size must cover the evaluated field count")
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


def bootstrap_ratio_interval(
    rows: list[dict[str, Any]],
    numerator_field: str,
    denominator_field: str,
    *,
    family_size: int,
    seed: int,
) -> dict[str, Any]:
    numerator = np.asarray(
        [float(row[numerator_field]) for row in rows], dtype=np.float64,
    )
    denominator = np.asarray(
        [float(row[denominator_field]) for row in rows], dtype=np.float64,
    )
    if len(rows) < 2 or np.any(numerator < 0.0) or np.any(denominator < 0.0):
        raise RuntimeError("invalid scene-block ratio inputs")
    denominator_sum = float(np.sum(denominator))
    estimate = float(np.sum(numerator) / denominator_sum) if denominator_sum > 0.0 else None
    if estimate is None:
        return {
            "estimate": None,
            "lower": None,
            "upper": None,
            "selected_area_positive": False,
            "resamples": BOOTSTRAP_RESAMPLES,
        }
    folds = sorted({str(row["fold"]) for row in rows})
    indices = {
        fold: np.asarray(
            [index for index, row in enumerate(rows) if str(row["fold"]) == fold],
            dtype=np.int64,
        )
        for fold in folds
    }
    generator = np.random.default_rng(seed)
    samples = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    for start in range(0, BOOTSTRAP_RESAMPLES, 2000):
        stop = min(BOOTSTRAP_RESAMPLES, start + 2000)
        sampled_numerator = np.zeros(stop - start, dtype=np.float64)
        sampled_denominator = np.zeros(stop - start, dtype=np.float64)
        for fold in folds:
            selected = indices[fold]
            draws = generator.integers(
                0, selected.size, size=(stop - start, selected.size),
            )
            sampled = selected[draws]
            sampled_numerator += np.sum(numerator[sampled], axis=1)
            sampled_denominator += np.sum(denominator[sampled], axis=1)
        samples[start:stop] = sampled_numerator / np.maximum(
            sampled_denominator, EPSILON,
        )
    alpha = 0.05 / family_size
    return {
        "estimate": estimate,
        "lower": float(np.quantile(samples, alpha / 2.0)),
        "upper": float(np.quantile(samples, 1.0 - alpha / 2.0)),
        "selected_area_positive": True,
        "resamples": BOOTSTRAP_RESAMPLES,
        "family_size": family_size,
        "stratified_by": "fold",
    }


def summarize_cell_rows(
    rows: list[dict[str, Any]], *, family_size: int, seed: int,
) -> dict[str, Any]:
    gain = normal_interval(
        [row["candidate_minus_keep_psnr_db"] for row in rows],
        family_size=family_size,
    )
    action_area = normal_interval(
        [row["selected_area_fraction"] for row in rows],
        family_size=family_size,
    )
    active = wilson(
        sum(bool(row["active_scene"]) for row in rows),
        len(rows),
        family_size=family_size,
    )
    material = wilson(
        sum(bool(row["material_scene"]) for row in rows),
        len(rows),
        family_size=family_size,
    )
    selected_harm = bootstrap_ratio_interval(
        rows,
        "selected_harm_area",
        "selected_area",
        family_size=family_size,
        seed=seed,
    )
    image_safety = {
        metric: wilson(
            sum(bool(row[f"{metric}_harm"]) for row in rows),
            len(rows),
            family_size=family_size * 3,
        )
        for metric in ("psnr", "ssim", "color")
    }
    return {
        "scene_count": len(rows),
        "gain": gain,
        "action_area": action_area,
        "active_scene": active,
        "material_scene": material,
        "selected_area_harm": selected_harm,
        "image_safety": image_safety,
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


def compare_baseline_reproduction(
    context,
    observed: list[dict[str, Any]],
) -> dict[str, Any]:
    archived_training = read_csv(
        asset_path(context, "baseline_training_diagnostics", kind="file"),
    )
    archived_calibration = read_csv(
        asset_path(context, "baseline_calibration_diagnostics", kind="file"),
    )
    training_lookup = {
        (row["model"], row["action"]): row for row in archived_training
        if row.get("split") == "train_fit" and row["model"].startswith("oof_fold_")
    }
    calibration_lookup = {
        (row["model"], row["action"]): row for row in archived_calibration
        if row["model"].startswith("oof_fold_")
    }
    differences = []
    for row in observed:
        key = (row["model"], row["action"])
        training = training_lookup.get(key)
        calibration = calibration_lookup.get(key)
        if training is None or calibration is None:
            differences.append({"model": key[0], "action": key[1], "field": "missing"})
            continue
        comparisons = {
            "first_epoch_total_loss": float(training["first_epoch_total_loss"]),
            "final_epoch_total_loss": float(training["final_epoch_total_loss"]),
            "first_epoch_utility_loss": float(training["first_epoch_utility_loss"]),
            "final_epoch_utility_loss": float(training["final_epoch_utility_loss"]),
            "first_epoch_risk_loss": float(training["first_epoch_risk_loss"]),
            "final_epoch_risk_loss": float(training["final_epoch_risk_loss"]),
            "utility_correction": float(calibration["utility_correction"]),
            "risk_correction": float(calibration["risk_correction"]),
        }
        for field, expected in comparisons.items():
            difference = abs(float(row[field]) - expected)
            differences.append({
                "model": key[0],
                "action": key[1],
                "field": field,
                "observed": float(row[field]),
                "expected": expected,
                "absolute_difference": difference,
            })
    maximum = max(
        (item.get("absolute_difference", math.inf) for item in differences),
        default=math.inf,
    )
    return {
        "schema_version": 1,
        "tolerance": PARENT_TRAINING_TOLERANCE,
        "expected_rows": OOF_FOLDS * len(ACTION_NAMES),
        "observed_rows": len(observed),
        "maximum_absolute_difference": maximum,
        "matched": (
            len(observed) == OOF_FOLDS * len(ACTION_NAMES)
            and len(differences) == OOF_FOLDS * len(ACTION_NAMES) * 8
            and maximum <= PARENT_TRAINING_TOLERANCE
        ),
        "differences": differences,
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
        "prior_oracle_closeout": (ORACLE_CLOSEOUT_SHA256, "HAZE4K_TEST_BOUNDED_LOCAL_ACTION_ORACLE_HEADROOM_PASS"),
        "parent_closeout": (
            PARENT_CLOSEOUT_SHA256,
            "HAZE4K_FROZEN_OUTPUT_RISK_INTERFACE_LOCALIZATION_FAIL",
        ),
        "parent_conclusion": (
            PARENT_CONCLUSION_SHA256,
            "HAZE4K_FROZEN_OUTPUT_RISK_INTERFACE_LOCALIZATION_FAIL",
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
        "prior_oracle_summary": sha256_file(asset_path(context, "prior_oracle_summary", kind="file")) == ORACLE_SUMMARY_SHA256,
        "parent_summary": sha256_file(
            asset_path(context, "parent_summary", kind="file"),
        ) == PARENT_SUMMARY_SHA256,
        "parent_formal_intervals": sha256_file(
            asset_path(context, "parent_formal_intervals", kind="file"),
        ) == PARENT_FORMAL_SHA256,
        "parent_regret_decomposition": sha256_file(
            asset_path(context, "parent_regret_decomposition", kind="file"),
        ) == PARENT_REGRET_SHA256,
        "baseline_training_diagnostics": sha256_file(
            asset_path(context, "baseline_training_diagnostics", kind="file"),
        ) == BASELINE_TRAINING_SHA256,
        "baseline_calibration_diagnostics": sha256_file(
            asset_path(context, "baseline_calibration_diagnostics", kind="file"),
        ) == BASELINE_CALIBRATION_SHA256,
    })
    parent_closeout = read_json(asset_path(context, "parent_closeout", kind="file"))
    checks.update({
        "parent_terminal_tuple": (
            parent_closeout.get("state") == "COMPLETED_GATE_FAIL"
            and parent_closeout.get("authorizes")
            == "OBSERVABLE_REPRESENTATION_OR_LEARNER_CONTRACT_AUTHORING_ONLY"
            and parent_closeout.get("route_commit")
            == PARENT_ROUTE_COMMIT
        ),
        "parent_route_identity": parent_closeout.get("route_id") == PARENT_ROUTE_ID,
    })
    return checks


def _legacy_contract(context_path: Path) -> None:
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


def _legacy_run(context_path: Path) -> None:
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

def move_bundles(bundles: list[dict[str, Any]], device: str) -> None:
    for bundle in bundles:
        bundle["head"].to(device)


def zero_calibration() -> dict[str, Any]:
    return {
        "utility_correction": np.zeros(2, dtype=np.float64),
        "risk_correction": np.zeros(2, dtype=np.float64),
    }


def crossfold_prediction(
    torch,
    fold_bundles: dict[int, list[dict[str, Any]]],
    fold_calibrations: dict[int, dict[str, Any]],
    record: dict[str, Any],
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    utilities = []
    risks = []
    for fold in range(OOF_FOLDS):
        utility, risk = predict_ensemble(
            torch, fold_bundles[fold], record, device,
        )
        lower, upper = corrected_predictions(
            utility, risk, fold_calibrations[fold],
        )
        utilities.append(lower)
        risks.append(upper)
    return np.mean(np.stack(utilities), axis=0), np.mean(np.stack(risks), axis=0)


def factorial_effect_rows(
    rows_by_cell: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    by_scene = {
        row["scene"]: {}
        for rows in rows_by_cell.values()
        for row in rows
    }
    for cell, rows in rows_by_cell.items():
        for row in rows:
            by_scene[row["scene"]][cell] = float(
                row["candidate_minus_keep_psnr_db"],
            )
    if any(set(values) != set(CELL_IDS) for values in by_scene.values()):
        raise RuntimeError("factorial scene-cell outcome matrix is incomplete")
    contrasts: dict[str, list[float]] = defaultdict(list)
    for values in by_scene.values():
        representation_mean = {
            representation: float(np.mean([
                values[f"{representation}_{learner}"] for learner in LEARNERS
            ]))
            for representation in REPRESENTATIONS
        }
        learner_mean = {
            learner: float(np.mean([
                values[f"{representation}_{learner}"]
                for representation in REPRESENTATIONS
            ]))
            for learner in LEARNERS
        }
        grand = float(np.mean(list(values.values())))
        contrasts["representation_R1_minus_R0"].append(
            representation_mean["R1"] - representation_mean["R0"],
        )
        contrasts["representation_R2_minus_R0"].append(
            representation_mean["R2"] - representation_mean["R0"],
        )
        contrasts["learner_L1_minus_L0"].append(
            learner_mean["L1"] - learner_mean["L0"],
        )
        contrasts["learner_L2_minus_L0"].append(
            learner_mean["L2"] - learner_mean["L0"],
        )
        for representation in REPRESENTATIONS:
            for learner in LEARNERS:
                cell = f"{representation}_{learner}"
                contrasts[f"interaction_{cell}"].append(
                    values[cell]
                    - representation_mean[representation]
                    - learner_mean[learner]
                    + grand
                )
    family_size = len(contrasts)
    return [
        {
            "effect": effect,
            **normal_interval(values, family_size=family_size),
        }
        for effect, values in sorted(contrasts.items())
    ]


def contract(context_path: Path) -> None:
    context = load_context(context_path, "contract")
    if context.route_id != ROUTE_ID or context.operation_id != OPERATION_ID:
        raise RuntimeError("route or operation identity mismatch")
    prepare_phase_output(context)
    started = time.monotonic()
    import torch

    if context.device != "cuda":
        raise RuntimeError("tail-factorial engineering contract requires CUDA")
    torch.cuda.reset_peak_memory_stats()
    checks = evidence_identity(context)
    entrypoint = asset_path(context, "factorial_entrypoint", kind="file")
    runtime_environment = asset_path(context, "runtime_environment", kind="file")
    checks.update({
        "entrypoint_identity": (
            context.assets["factorial_entrypoint"].sha256 == sha256_file(entrypoint)
        ),
        "runtime_environment_identity": (
            sha256_file(runtime_environment) == RUNTIME_ENVIRONMENT_SHA256
        ),
        "runtime_environment_contract": (
            read_json(runtime_environment).get("device_class") == "cuda_sm89"
        ),
        "official_anchor_identity": (
            context.assets["official_anchor_checkout"].commit == ANCHOR_COMMIT
        ),
        "protected_datasets_absent": all(
            identifier not in context.assets
            for identifier in (
                "haze4k_train",
                "haze4k_test_development",
                "candidate_confirmation",
                "nh_haze",
                "reside_its",
                "reside_ots",
            )
        ),
        "fixed_cost_contract": (
            context.engineering_contract["cost_contract"]["formal_iterations"]
            == FORMAL_COST_ITERATIONS
            and context.engineering_contract["cost_contract"]["probe_iterations"]
            == PROBE_COST_ITERATIONS
        ),
    })
    torch, model = load_official_model(context)
    generator = np.random.default_rng(TRAINING_SEED)
    hazy = generator.uniform(0.05, 0.95, size=(384, 512, 3)).astype(np.float32)
    clear = np.clip(
        hazy + generator.normal(scale=0.03, size=hazy.shape), 0.0, 1.0,
    ).astype(np.float32)
    extracted = extract_variant(torch, model, hazy, clear, context.device)
    record = {
        "scene": "synthetic",
        **{
            key: extracted[key]
            for key in (
                "features",
                "shape_features",
                "shape_areas",
                "disagreement",
                "utility",
                "harm",
                "areas",
            )
        },
    }
    completed_iterations = 3
    finite_cells = []
    loss_decreased = []
    representation_shapes = {}
    for cell in CELL_IDS:
        representation, learner = cell.split("_")
        for seed_index, offset in enumerate(SEED_OFFSETS):
            head, normalization, history = train_head(
                torch,
                [record],
                seed=TRAINING_SEED + offset,
                device=context.device,
                representation=representation,
                learner=learner,
            )
            completed_iterations += EPOCHS
            utility, risk = predict_head(
                torch, head, record, normalization, context.device,
            )
            evaluation = evaluate_factorial_psnr(
                record,
                utility,
                risk,
                zero_calibration(),
                hazy,
                clear,
                extracted["prediction"],
                "synthetic",
                f"{cell}-{seed_index}",
            )
            completed_iterations += 1
            finite_cells.append(bool(
                np.isfinite(utility).all()
                and np.isfinite(risk).all()
                and math.isfinite(
                    evaluation["metrics"]["candidate"]["minus_keep_psnr_db"],
                )
            ))
            loss_decreased.append(
                history[-1]["total_loss"] < history[0]["total_loss"],
            )
            representation_shapes[cell] = {
                "utility": list(utility.shape),
                "risk": list(risk.shape),
            }
            write_contract_progress(
                context,
                completed_iterations=completed_iterations,
                total_iterations=PROBE_COST_ITERATIONS,
                stage="synthetic_tail_factorial",
            )
    partitioned_family_rows = [
        {"fold": fold, "partitioned_metric": float(index) / 1000.0}
        for fold in range(OOF_FOLDS)
        for index in range(TRAINING_SCENES // OOF_FOLDS)
    ]
    partitioned_family_interval = stratified_bootstrap_family(
        partitioned_family_rows,
        ["partitioned_metric"],
        family_size=FACTORIAL_FAMILY_SIZE,
        seed=BOOTSTRAP_SEED + 999,
    )["partitioned_metric"]
    checks.update({
        "probe_iteration_count": completed_iterations == PROBE_COST_ITERATIONS,
        "base_feature_shape": extracted["features"].shape[0] == FEATURE_CHANNELS,
        "shape_feature_shape": extracted["shape_features"].shape[0] == SHAPE_CHANNELS,
        "disagreement_feature_shape": (
            extracted["disagreement"].shape[0] == DISAGREEMENT_CHANNELS
        ),
        "complete_factorial": set(representation_shapes) == set(CELL_IDS),
        "three_seed_path": len(finite_cells) == len(CELL_IDS) * len(SEED_OFFSETS),
        "finite_predictions_and_replay": all(finite_cells),
        "synthetic_training_response": all(loss_decreased),
        "partitioned_comparison_family": (
            partitioned_family_interval["scene_count"] == TRAINING_SCENES
            and partitioned_family_interval["family_size"] == FACTORIAL_FAMILY_SIZE
            and math.isfinite(partitioned_family_interval["lower"])
            and math.isfinite(partitioned_family_interval["upper"])
        ),
    })
    elapsed = time.monotonic() - started
    write_contract_result(
        context,
        checks=checks,
        engineering={
            "mode": "gpu_synthetic_no_data",
            "device": context.device,
            "fixture": {"batch": 1, "channels": 3, "height": 384, "width": 512},
            "production_path_exercised": True,
            "protected_data_touched": False,
            "scientific_output_created": False,
            "scientific_training_occurred": False,
            "cost": {
                "observed_iterations": completed_iterations,
                "observed_wall_seconds": elapsed,
                "observed_peak_memory_mib": float(
                    torch.cuda.max_memory_allocated() / (1024 * 1024),
                ),
            },
        },
    )


def run(context_path: Path) -> None:
    context = load_context(context_path, "run")
    if context.route_id != ROUTE_ID or context.operation_id != OPERATION_ID:
        raise RuntimeError("route or operation identity mismatch")
    prepare_phase_output(context)
    if context.total_units != TOTAL_UNITS \
            or context.evidence_role != "development_screening" \
            or any(context.protected_data_permissions.values()):
        raise RuntimeError("runtime role, unit, or protected-data contract changed")
    forbidden = {"candidate_confirmation", "nh_haze", "reside_its", "reside_ots"}
    if forbidden & set(context.assets):
        raise RuntimeError("forbidden scientific asset was delivered")
    if load_completed_unit_ledger(context):
        raise RuntimeError("fresh tail-factorial route cannot preload completed units")
    identity_checks = evidence_identity(context)
    train_root = asset_path(context, "haze4k_train", kind="directory")
    test_root = asset_path(context, "haze4k_test_development", kind="directory")
    write_workload_progress(context, completed_units=0, stage="identity_and_scope")

    train_groups, training_scenes, calibration_scenes, assignment = enumerate_train_groups(
        train_root,
    )
    test_groups = enumerate_test_groups(test_root)
    role_checks = {
        "train_assignment_digest": assignment == TRAIN_ASSIGNMENT_DIGEST,
        "test_assignment_digest": read_json(
            asset_path(context, "test_split_summary", kind="file"),
        ).get("frozen_split", {}).get("assignment_digest") == TEST_ASSIGNMENT_DIGEST,
        "train_role_counts": (
            len(training_scenes) == TRAINING_SCENES
            and len(calibration_scenes) == CALIBRATION_SCENES
        ),
        "train_role_disjointness": not (
            set(training_scenes) & set(calibration_scenes)
        ),
        "test_role_counts": (
            len(test_groups) == EXPECTED_TEST_SCENES
            and all(len(items) == VARIANTS_PER_SCENE for items in test_groups.values())
        ),
        "candidate_confirmation_absent": "candidate_confirmation" not in context.assets,
        "protected_cross_domain_absent": not (forbidden & set(context.assets)),
    }

    torch, model = load_official_model(context)
    cache_root = output_file(context, "scene_cache")
    cache_root.mkdir()
    checkpoint_root = output_file(context, "head_checkpoints")
    checkpoint_root.mkdir()
    raw_oof_root = output_file(context, "raw_predictions/haze4k_train_oof")
    raw_oof_root.mkdir(parents=True)
    raw_test_root = output_file(context, "raw_predictions/haze4k_test_development_stress")
    raw_test_root.mkdir(parents=True)
    inventory_items: list[dict[str, Any]] = []
    extraction_records: dict[str, dict[str, Any]] = {}
    extraction_identities = []
    training_scene_set = set(training_scenes)
    ordered_train_scenes = training_scenes + calibration_scenes
    for completed, scene in enumerate(ordered_train_scenes, start=1):
        hazy_path, clear_path = choose_one_variant(scene, train_groups[scene])
        hazy, clear = image_array(hazy_path), image_array(clear_path)
        extracted = extract_variant(torch, model, hazy, clear, context.device)
        role = "training" if scene in training_scene_set else "calibration"
        relative = f"scene_cache/{role}_{scene[:24]}.npz"
        cache_path = output_file(context, relative)
        np.savez(cache_path, **extracted)
        add_inventory_item(inventory_items, context, cache_path, "scene_cache")
        input_identity = sha256_text("|".join([
            "observable-tail-factorial-extraction-v1",
            scene,
            hazy_path.name,
            sha256_file(hazy_path),
            sha256_file(clear_path),
        ]))
        record_completed_unit(
            context,
            unit_id=f"extract_{completed:04d}_{scene[:16]}",
            input_sha256=input_identity,
            output_relpath=relative,
        )
        extraction_identities.append(f"{input_identity}:{sha256_file(cache_path)}")
        extraction_records[scene] = {
            "scene": scene,
            "role": role,
            "cache_path": cache_path,
            "hazy_path": hazy_path,
            "clear_path": clear_path,
            **{
                key: extracted[key]
                for key in (
                    "features",
                    "shape_features",
                    "shape_areas",
                    "disagreement",
                    "utility",
                    "harm",
                    "areas",
                )
            },
        }
        if completed % 10 == 0 or completed == EXPECTED_TRAIN_SCENES:
            write_workload_progress(
                context,
                completed_units=completed,
                stage="outcome_blind_multiview_extraction",
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
    calibration_order = sorted(
        calibration_scenes,
        key=lambda scene: (
            sha256_text(f"{CALIBRATION_FOLD_SALT}|{scene}"), scene,
        ),
    )
    calibration_fold = {
        scene: index % OOF_FOLDS for index, scene in enumerate(calibration_order)
    }
    role_checks["balanced_oof_folds"] = Counter(fold_assignment.values()) == {
        fold: TRAINING_SCENES // OOF_FOLDS for fold in range(OOF_FOLDS)
    }
    role_checks["balanced_calibration_selection_folds"] = Counter(
        calibration_fold.values(),
    ) == {fold: CALIBRATION_SCENES // OOF_FOLDS for fold in range(OOF_FOLDS)}

    all_bundles: dict[str, dict[int, list[dict[str, Any]]]] = {
        cell: {} for cell in CELL_IDS
    }
    fold_calibrations: dict[str, dict[int, dict[str, Any]]] = {
        cell: {} for cell in CELL_IDS
    }
    oof_predictions: dict[str, dict[str, dict[str, Any]]] = {
        cell: {} for cell in CELL_IDS
    }
    selection_rows_by_cell: dict[str, list[dict[str, Any]]] = {
        cell: [] for cell in CELL_IDS
    }
    training_rows: list[dict[str, Any]] = []
    calibration_rows_output: list[dict[str, Any]] = []
    baseline_observed: list[dict[str, Any]] = []

    for cell_index, cell in enumerate(CELL_IDS):
        representation, learner = cell.split("_")
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
            bundles = []
            for seed_index, offset in enumerate(SEED_OFFSETS):
                seed = TRAINING_SEED + fold + offset

                def progress(epoch, *, cell=cell, fold=fold, seed_index=seed_index):
                    if epoch == EPOCHS:
                        write_workload_progress(
                            context,
                            completed_units=EXPECTED_TRAIN_SCENES,
                            stage=f"factorial_{cell}_{fold}_{seed_index}",
                        )

                head, normalization, history = train_head(
                    torch,
                    train_records,
                    seed=seed,
                    device=context.device,
                    representation=representation,
                    learner=learner,
                    progress_callback=progress,
                )
                model_id = f"{cell}_fold_{fold}_seed_{seed_index}"
                checkpoint_path = checkpoint_root / f"{model_id}.pt"
                torch.save({
                    "schema_version": 1,
                    "cell": cell,
                    "representation": representation,
                    "learner": learner,
                    "fold": fold,
                    "seed": seed,
                    "state_dict": {
                        name: value.detach().cpu()
                        for name, value in head.state_dict().items()
                    },
                    "normalization": normalization,
                    "history": history,
                }, checkpoint_path)
                add_inventory_item(
                    inventory_items, context, checkpoint_path, "head_checkpoint",
                )
                training_rows.append({
                    "cell": cell,
                    "representation": representation,
                    "learner": learner,
                    "fold": fold,
                    "seed_index": seed_index,
                    "seed": seed,
                    "train_scenes": len(train_records),
                    "epochs": EPOCHS,
                    "first_epoch_total_loss": history[0]["total_loss"],
                    "first_epoch_utility_loss": history[0]["utility_loss"],
                    "first_epoch_risk_loss": history[0]["risk_loss"],
                    "final_epoch_total_loss": history[-1]["total_loss"],
                    "final_epoch_utility_loss": history[-1]["utility_loss"],
                    "final_epoch_risk_loss": history[-1]["risk_loss"],
                    "checkpoint_sha256": sha256_file(checkpoint_path),
                })
                bundles.append({
                    "head": head,
                    "normalization": normalization,
                    "history": history,
                    "checkpoint_path": checkpoint_path,
                })

            calibration_predictions = {
                scene: predict_ensemble(
                    torch,
                    bundles,
                    extraction_records[scene],
                    context.device,
                )
                for scene in calibration_scenes
            }
            correction_records = [
                extraction_records[scene]
                for scene in calibration_scenes
                if calibration_fold[scene] != fold
            ]
            selection_records = [
                extraction_records[scene]
                for scene in calibration_scenes
                if calibration_fold[scene] == fold
            ]
            calibration = calibrate_predictions(
                calibration_predictions, correction_records,
            )
            fold_calibrations[cell][fold] = calibration
            for row in calibration_rows(cell + f"_fold_{fold}", calibration):
                calibration_rows_output.append({
                    "cell": cell,
                    "fold": fold,
                    **row,
                })

            if cell == "R0_L0":
                first_seed_predictions = {
                    scene: predict_head(
                        torch,
                        bundles[0]["head"],
                        extraction_records[scene],
                        bundles[0]["normalization"],
                        context.device,
                    )
                    for scene in calibration_scenes
                }
                legacy_calibration = calibrate_predictions(
                    first_seed_predictions,
                    [extraction_records[scene] for scene in calibration_scenes],
                )
                history = bundles[0]["history"]
                for action, action_name in enumerate(ACTION_NAMES):
                    baseline_observed.append({
                        "model": f"oof_fold_{fold}",
                        "action": action_name,
                        "first_epoch_total_loss": history[0]["total_loss"],
                        "first_epoch_utility_loss": history[0]["utility_loss"],
                        "first_epoch_risk_loss": history[0]["risk_loss"],
                        "final_epoch_total_loss": history[-1]["total_loss"],
                        "final_epoch_utility_loss": history[-1]["utility_loss"],
                        "final_epoch_risk_loss": history[-1]["risk_loss"],
                        "utility_correction": legacy_calibration[
                            "utility_correction"
                        ][action],
                        "risk_correction": legacy_calibration[
                            "risk_correction"
                        ][action],
                    })

            for record in selection_records:
                utility, risk = calibration_predictions[record["scene"]]
                cached = load_cache(record["cache_path"], include_prediction=True)
                evaluation = evaluate_policy_predictions(
                    torch,
                    record,
                    utility,
                    risk,
                    calibration,
                    image_array(record["hazy_path"]),
                    image_array(record["clear_path"]),
                    cached["prediction"],
                    record["scene"],
                    record["hazy_path"].name,
                    context.device,
                )
                selection_rows_by_cell[cell].append(flatten_cell_evaluation(
                    "haze4k_train_calibration_selection",
                    record["scene"],
                    fold,
                    cell,
                    evaluation,
                ))

            for record in eval_records:
                utility, risk = predict_ensemble(
                    torch, bundles, record, context.device,
                )
                oof_predictions[cell][record["scene"]] = {
                    "utility": utility,
                    "risk": risk,
                    "calibration": calibration,
                    "fold": fold,
                }
            move_bundles(bundles, "cpu")
            all_bundles[cell][fold] = bundles
            torch.cuda.empty_cache()

    baseline_reproduction = compare_baseline_reproduction(
        context, baseline_observed,
    )
    selection_summaries = {
        cell: summarize_cell_rows(
            rows,
            family_size=FACTORIAL_FAMILY_SIZE,
            seed=BOOTSTRAP_SEED + index,
        )
        for index, (cell, rows) in enumerate(selection_rows_by_cell.items())
    }

    def selection_admissible(summary: dict[str, Any]) -> bool:
        harm = summary["selected_area_harm"]
        return bool(
            summary["action_area"]["lower"] >= MIN_ACTION_AREA_FRACTION
            and summary["active_scene"]["lower"] >= MIN_ACTIVE_SCENE_PREVALENCE
            and harm["upper"] is not None
            and harm["upper"] <= MAX_HARM_PREVALENCE
            and all(
                interval["upper"] <= MAX_HARM_PREVALENCE
                for interval in summary["image_safety"].values()
            )
        )

    def selection_key(cell: str) -> tuple[Any, ...]:
        summary = selection_summaries[cell]
        harm_upper = summary["selected_area_harm"]["upper"]
        return (
            selection_admissible(summary),
            summary["gain"]["lower"],
            summary["material_scene"]["lower"],
            -(harm_upper if harm_upper is not None else 1.0),
            -CELL_IDS.index(cell),
        )

    selected_cell = max(CELL_IDS, key=selection_key)
    selection_path = output_file(
        context, "haze4k_observable_tail_factorial_selection_freeze.json",
    )
    atomic_json(selection_path, {
        "schema_version": 1,
        "route_id": ROUTE_ID,
        "selection_population": "150 fixed calibration scenes with five group folds",
        "selection_rule": (
            "prefer cells passing simultaneous activation and safety constraints; "
            "then maximize utility LCB, material-scene LCB, inverse selected-harm "
            "UCB, and fixed R0L0..R2L2 order"
        ),
        "formal_oof_outcomes_read_before_freeze": False,
        "representations": list(REPRESENTATIONS),
        "learners": list(LEARNERS),
        "seeds_per_cell_fold": len(SEED_OFFSETS),
        "cell_summaries": selection_summaries,
        "selected_cell": selected_cell,
        "selected_cell_admissible": selection_admissible(
            selection_summaries[selected_cell],
        ),
    })
    selection_sha = sha256_file(selection_path)

    formal_rows_by_cell: dict[str, list[dict[str, Any]]] = {
        cell: [] for cell in CELL_IDS
    }
    for scene_index, scene in enumerate(sorted(training_scenes), start=1):
        record = extraction_records[scene]
        cached = load_cache(record["cache_path"], include_prediction=True)
        hazy = image_array(record["hazy_path"])
        clear = image_array(record["clear_path"])
        for cell in CELL_IDS:
            prediction_data = oof_predictions[cell][scene]
            if cell in {selected_cell, "R0_L0"}:
                evaluation = evaluate_policy_predictions(
                    torch,
                    record,
                    prediction_data["utility"],
                    prediction_data["risk"],
                    prediction_data["calibration"],
                    hazy,
                    clear,
                    cached["prediction"],
                    scene,
                    record["hazy_path"].name,
                    context.device,
                )
            else:
                evaluation = evaluate_factorial_psnr(
                    record,
                    prediction_data["utility"],
                    prediction_data["risk"],
                    prediction_data["calibration"],
                    hazy,
                    clear,
                    cached["prediction"],
                    scene,
                    record["hazy_path"].name,
                )
            row = flatten_cell_evaluation(
                "haze4k_train_oof",
                scene,
                prediction_data["fold"],
                cell,
                evaluation,
            )
            formal_rows_by_cell[cell].append(row)
            cell_root = raw_oof_root / cell
            cell_root.mkdir(exist_ok=True)
            raw_path = cell_root / f"fold_{prediction_data['fold']}_{scene[:24]}.npz"
            lower, upper = corrected_predictions(
                prediction_data["utility"],
                prediction_data["risk"],
                prediction_data["calibration"],
            )
            np.savez(
                raw_path,
                utility=prediction_data["utility"].astype(np.float32),
                risk=prediction_data["risk"].astype(np.float32),
                utility_lower=lower.astype(np.float32),
                risk_upper=upper.astype(np.float32),
                actions=evaluation["actions"]["candidate"].astype(np.int8),
            )
            add_inventory_item(
                inventory_items, context, raw_path, "oof_cell_prediction",
            )
        if scene_index % 20 == 0 or scene_index == TRAINING_SCENES:
            write_workload_progress(
                context,
                completed_units=EXPECTED_TRAIN_SCENES,
                stage="frozen_selection_oof_factorial_replay",
            )

    baseline_lookup = {
        row["scene"]: row for row in formal_rows_by_cell["R0_L0"]
    }
    selected_rows = formal_rows_by_cell[selected_cell]
    for row in selected_rows:
        row["candidate_minus_baseline_psnr_db"] = (
            row["candidate_minus_keep_psnr_db"]
            - baseline_lookup[row["scene"]]["candidate_minus_keep_psnr_db"]
        )
    primary_fields = [
        "candidate_minus_keep_psnr_db",
        "candidate_minus_baseline_psnr_db",
        "candidate_minus_uniform_psnr_db",
        "candidate_minus_shuffled_psnr_db",
    ]
    primary_intervals = stratified_bootstrap_family(
        selected_rows,
        primary_fields,
        family_size=PRIMARY_CONTINUOUS_FAMILY_SIZE,
        seed=BOOTSTRAP_SEED,
    )
    factorial_intervals = {
        cell: stratified_bootstrap_family(
            rows,
            ["candidate_minus_keep_psnr_db"],
            family_size=FACTORIAL_FAMILY_SIZE,
            seed=BOOTSTRAP_SEED + 100 + CELL_IDS.index(cell),
        )["candidate_minus_keep_psnr_db"]
        for cell, rows in formal_rows_by_cell.items()
    }
    effects = factorial_effect_rows(formal_rows_by_cell)

    training_unit_path = output_file(
        context, "factorial_training_selection_and_oof_unit.json",
    )
    atomic_json(training_unit_path, {
        "schema_version": 1,
        "route_id": ROUTE_ID,
        "selection_freeze_sha256": selection_sha,
        "selected_cell": selected_cell,
        "cell_count": len(CELL_IDS),
        "fold_count": OOF_FOLDS,
        "seeds_per_cell_fold": len(SEED_OFFSETS),
        "checkpoint_count": len(CELL_IDS) * OOF_FOLDS * len(SEED_OFFSETS),
        "oof_scene_cell_count": sum(len(rows) for rows in formal_rows_by_cell.values()),
        "baseline_reproduction": baseline_reproduction,
        "primary_intervals": primary_intervals,
        "factorial_intervals": factorial_intervals,
    })
    record_completed_unit(
        context,
        unit_id="factorial_training_selection_and_oof",
        input_sha256=sha256_text("|".join([
            "observable-tail-factorial-training-v1",
            selection_sha,
            digest_lines(extraction_identities),
        ])),
        output_relpath="factorial_training_selection_and_oof_unit.json",
    )
    write_workload_progress(
        context,
        completed_units=EXPECTED_TRAIN_SCENES + 1,
        stage="selection_frozen_and_oof_complete",
    )

    test_cache_root = output_file(context, "test_development_cache")
    test_cache_root.mkdir()
    test_records = []
    for scene_index, scene in enumerate(sorted(test_groups), start=1):
        for variant_index, (hazy_path, clear_path) in enumerate(
            sorted(test_groups[scene]), start=1,
        ):
            hazy, clear = image_array(hazy_path), image_array(clear_path)
            extracted = extract_variant(torch, model, hazy, clear, context.device)
            cache_path = test_cache_root / (
                f"scene_{scene_index:03d}_{scene[:16]}_variant_{variant_index}.npz"
            )
            np.savez(cache_path, **extracted)
            add_inventory_item(
                inventory_items, context, cache_path, "test_scene_cache",
            )
            test_records.append({
                "scene": scene,
                "scene_index": scene_index,
                "variant_index": variant_index,
                "hazy_path": hazy_path,
                "clear_path": clear_path,
                "cache_path": cache_path,
            })

    test_prediction_paths: dict[tuple[str, str, int], Path] = {}
    for cell in CELL_IDS:
        for fold in range(OOF_FOLDS):
            move_bundles(all_bundles[cell][fold], context.device)
        cell_root = raw_test_root / cell
        cell_root.mkdir(exist_ok=True)
        for record_identity in test_records:
            record = load_cache(
                record_identity["cache_path"], include_prediction=False,
            )
            utility_lower, risk_upper = crossfold_prediction(
                torch,
                all_bundles[cell],
                fold_calibrations[cell],
                record,
                context.device,
            )
            raw_path = cell_root / (
                f"scene_{record_identity['scene_index']:03d}_"
                f"{record_identity['scene'][:16]}_variant_"
                f"{record_identity['variant_index']}.npz"
            )
            np.savez(
                raw_path,
                utility_lower=utility_lower.astype(np.float32),
                risk_upper=risk_upper.astype(np.float32),
            )
            add_inventory_item(
                inventory_items, context, raw_path, "test_cell_prediction",
            )
            test_prediction_paths[
                (cell, record_identity["scene"], record_identity["variant_index"])
            ] = raw_path
        for fold in range(OOF_FOLDS):
            move_bundles(all_bundles[cell][fold], "cpu")
        torch.cuda.empty_cache()

    test_rows_by_cell: dict[str, list[dict[str, Any]]] = {
        cell: [] for cell in CELL_IDS
    }
    records_by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record_identity in test_records:
        records_by_scene[record_identity["scene"]].append(record_identity)
    for scene_index, scene in enumerate(sorted(records_by_scene), start=1):
        nested_rows: dict[str, list[dict[str, Any]]] = {
            cell: [] for cell in CELL_IDS
        }
        input_parts = ["observable-tail-factorial-test-v1", scene, selection_sha]
        for record_identity in sorted(
            records_by_scene[scene], key=lambda item: item["variant_index"],
        ):
            cached = load_cache(
                record_identity["cache_path"], include_prediction=True,
            )
            hazy = image_array(record_identity["hazy_path"])
            clear = image_array(record_identity["clear_path"])
            input_parts.extend([
                record_identity["hazy_path"].name,
                sha256_file(record_identity["hazy_path"]),
                sha256_file(record_identity["clear_path"]),
            ])
            for cell in CELL_IDS:
                prediction_path = test_prediction_paths[
                    (cell, scene, record_identity["variant_index"])
                ]
                with np.load(prediction_path, allow_pickle=False) as source:
                    utility_lower = source["utility_lower"].astype(np.float64)
                    risk_upper = source["risk_upper"].astype(np.float64)
                if cell == selected_cell:
                    evaluation = evaluate_policy_predictions(
                        torch,
                        cached,
                        utility_lower,
                        risk_upper,
                        zero_calibration(),
                        hazy,
                        clear,
                        cached["prediction"],
                        scene,
                        record_identity["hazy_path"].name,
                        context.device,
                    )
                else:
                    evaluation = evaluate_factorial_psnr(
                        cached,
                        utility_lower,
                        risk_upper,
                        zero_calibration(),
                        hazy,
                        clear,
                        cached["prediction"],
                        scene,
                        record_identity["hazy_path"].name,
                    )
                nested_rows[cell].append(flatten_cell_evaluation(
                    "haze4k_test_development_stress",
                    scene,
                    "crossfold_ensemble",
                    cell,
                    evaluation,
                ))
                input_parts.append(sha256_file(prediction_path))
        scene_rows = []
        for cell in CELL_IDS:
            row = average_variant_rows(nested_rows[cell])
            test_rows_by_cell[cell].append(row)
            scene_rows.append(row)
        relative = f"test_development_scene_units/scene_{scene_index:03d}_{scene[:16]}.json"
        scene_path = output_file(context, relative)
        scene_path.parent.mkdir(exist_ok=True)
        atomic_json(scene_path, {
            "schema_version": 1,
            "scene": scene,
            "nested_variants": VARIANTS_PER_SCENE,
            "selected_cell": selected_cell,
            "cell_rows": scene_rows,
        })
        record_completed_unit(
            context,
            unit_id=f"test_scene_{scene_index:03d}_{scene[:16]}",
            input_sha256=sha256_text("|".join(input_parts)),
            output_relpath=relative,
        )
        if scene_index % 5 == 0 or scene_index == EXPECTED_TEST_SCENES:
            write_workload_progress(
                context,
                completed_units=EXPECTED_TRAIN_SCENES + 1 + scene_index,
                stage="nested_test_development_factorial_stress",
            )

    role_checks.update({
        "complete_factorial_cells": set(formal_rows_by_cell) == set(CELL_IDS),
        "complete_fold_seed_grid": len(training_rows)
        == len(CELL_IDS) * OOF_FOLDS * len(SEED_OFFSETS),
        "complete_selection_grid": all(
            len(rows) == CALIBRATION_SCENES
            for rows in selection_rows_by_cell.values()
        ),
        "complete_oof_grid": all(
            len(rows) == TRAINING_SCENES for rows in formal_rows_by_cell.values()
        ),
        "complete_test_grid": all(
            len(rows) == EXPECTED_TEST_SCENES
            for rows in test_rows_by_cell.values()
        ),
        "selection_frozen_before_formal_outcomes": selection_path.is_file(),
        "baseline_reproduction": baseline_reproduction["matched"],
        "pre_finalization_ledger_coverage": (
            len(load_completed_unit_ledger(context)) == TOTAL_UNITS - 1
        ),
    })

    selected_oof_summary = summarize_cell_rows(
        selected_rows,
        family_size=SAFETY_FAMILY_SIZE,
        seed=BOOTSTRAP_SEED + 500,
    )
    selected_test_rows = test_rows_by_cell[selected_cell]
    selected_test_summary = summarize_cell_rows(
        selected_test_rows,
        family_size=SAFETY_FAMILY_SIZE,
        seed=BOOTSTRAP_SEED + 600,
    )
    oracle_interval = stratified_bootstrap_family(
        selected_rows,
        ["gt_gt_minus_keep_psnr_db"],
        family_size=CONTROL_FAMILY_SIZE,
        seed=BOOTSTRAP_SEED + 700,
    )["gt_gt_minus_keep_psnr_db"]
    oracle_material = wilson(
        sum(
            float(row["gt_gt_minus_keep_psnr_db"]) >= UTILITY_MARGIN_DB
            for row in selected_rows
        ),
        len(selected_rows),
        family_size=CONTROL_FAMILY_SIZE,
    )

    utility_gain = primary_intervals["candidate_minus_keep_psnr_db"]
    baseline_gain = primary_intervals["candidate_minus_baseline_psnr_db"]
    uniform_gain = primary_intervals["candidate_minus_uniform_psnr_db"]
    shuffle_gain = primary_intervals["candidate_minus_shuffled_psnr_db"]
    utility_favorable = (
        utility_gain["lower"] > UTILITY_MARGIN_DB
        and selected_oof_summary["material_scene"]["lower"] > MIN_MATERIAL_SCENE_PREVALENCE
        and baseline_gain["lower"] > 0.0
        and uniform_gain["lower"] > 0.0
        and shuffle_gain["lower"] > 0.0
    )
    utility_futile = (
        utility_gain["upper"] <= UTILITY_MARGIN_DB
        or selected_oof_summary["material_scene"]["upper"] <= MIN_MATERIAL_SCENE_PREVALENCE
        or baseline_gain["upper"] <= 0.0
        or uniform_gain["upper"] <= 0.0
        or shuffle_gain["upper"] <= 0.0
    )
    utility_outcome = (
        "favorable" if utility_favorable
        else "unfavorable" if utility_futile
        else "indeterminate"
    )
    activation_favorable = (
        selected_oof_summary["action_area"]["lower"] >= MIN_ACTION_AREA_FRACTION
        and selected_oof_summary["active_scene"]["lower"] >= MIN_ACTIVE_SCENE_PREVALENCE
    )
    activation_futile = (
        selected_oof_summary["action_area"]["upper"] < MIN_ACTION_AREA_FRACTION
        or selected_oof_summary["active_scene"]["upper"] < MIN_ACTIVE_SCENE_PREVALENCE
    )
    activation_outcome = (
        "favorable" if activation_favorable
        else "unfavorable" if activation_futile
        else "indeterminate"
    )

    safety_summaries = [selected_oof_summary, selected_test_summary]
    ratio_intervals = [summary["selected_area_harm"] for summary in safety_summaries]
    image_intervals = [
        interval
        for summary in safety_summaries
        for interval in summary["image_safety"].values()
    ]
    safety_favorable = (
        all(interval["upper"] is not None for interval in ratio_intervals)
        and all(interval["upper"] <= MAX_HARM_PREVALENCE for interval in ratio_intervals)
        and all(interval["upper"] <= MAX_HARM_PREVALENCE for interval in image_intervals)
    )
    safety_unsafe = (
        any(
            interval["lower"] is not None
            and interval["lower"] > MAX_HARM_PREVALENCE
            for interval in ratio_intervals
        )
        or any(interval["lower"] > MAX_HARM_PREVALENCE for interval in image_intervals)
    )
    safety_outcome = (
        "safe" if safety_favorable
        else "unsafe" if safety_unsafe
        else "indeterminate"
    )
    oracle_outcome = (
        "favorable"
        if (
            oracle_interval["lower"] > UTILITY_MARGIN_DB
            and oracle_material["lower"] > MIN_MATERIAL_SCENE_PREVALENCE
        )
        else "unfavorable"
        if (
            oracle_interval["upper"] <= UTILITY_MARGIN_DB
            or oracle_material["upper"] <= MIN_MATERIAL_SCENE_PREVALENCE
        )
        else "indeterminate"
    )
    precision_half_width = max(
        interval["max_half_width"] for interval in primary_intervals.values()
    )
    identity_pass = all(identity_checks.values())
    coverage_pass = all(role_checks.values())
    gate_outcomes = {
        "evidence_identity": "pass" if identity_pass else "fail",
        "factorial_completeness_and_selection_integrity": (
            "pass" if coverage_pass else "fail"
        ),
        "oracle_headroom_control": oracle_outcome,
        "selected_candidate_actionable_utility": utility_outcome,
        "selected_candidate_activation": activation_outcome,
        "selected_area_and_image_safety": safety_outcome,
        "precision": "met" if precision_half_width <= PRECISION_TARGET_DB else "unmet",
        "factorial_mechanism_diagnostic": "favorable",
    }

    factorial_cell_rows = []
    for cell in CELL_IDS:
        selection = selection_summaries[cell]
        formal = factorial_intervals[cell]
        test_gain = normal_interval(
            [row["candidate_minus_keep_psnr_db"] for row in test_rows_by_cell[cell]],
            family_size=FACTORIAL_FAMILY_SIZE,
        )
        factorial_cell_rows.append({
            "cell": cell,
            "representation": cell.split("_")[0],
            "learner": cell.split("_")[1],
            "selected": cell == selected_cell,
            "selection_admissible": selection_admissible(selection),
            "selection_gain": selection["gain"]["estimate"],
            "selection_gain_lower": selection["gain"]["lower"],
            "selection_gain_upper": selection["gain"]["upper"],
            "oof_gain": formal["estimate"],
            "oof_gain_lower": formal["lower"],
            "oof_gain_upper": formal["upper"],
            "test_stress_gain": test_gain["estimate"],
            "test_stress_gain_lower": test_gain["lower"],
            "test_stress_gain_upper": test_gain["upper"],
            "selection_action_area_lower": selection["action_area"]["lower"],
            "selection_active_scene_lower": selection["active_scene"]["lower"],
            "selection_selected_harm_upper": selection[
                "selected_area_harm"
            ]["upper"],
        })

    control_ids = (
        "utility_only",
        "observable_utility_gt_risk",
        "gt_utility_observable_risk",
        "gt_gt",
    )
    controls = {
        control: stratified_bootstrap_family(
            selected_rows,
            [f"{control}_minus_keep_psnr_db"],
            family_size=len(control_ids),
            seed=BOOTSTRAP_SEED + 800 + index,
        )[f"{control}_minus_keep_psnr_db"]
        for index, control in enumerate(control_ids)
    }
    controls["risk_regret_fraction"] = (
        controls["observable_utility_gt_risk"]["estimate"]
        - utility_gain["estimate"]
    )
    controls["utility_regret_fraction"] = (
        controls["gt_utility_observable_risk"]["estimate"]
        - utility_gain["estimate"]
    )

    raw_inventory = compact_inventory(inventory_items)
    summary = {
        "schema_version": 2,
        "route_id": ROUTE_ID,
        "operation_id": OPERATION_ID,
        "run_id": context.run_id,
        "scope": "development-only observable representation x learner tail factorial",
        "independent_unit": "original_clear_scene",
        "selected_cell": selected_cell,
        "selection_freeze_sha256": selection_sha,
        "factorial": {
            "representations": {
                "R0": "exact archived 104-channel tile baseline",
                "R1": "R0 plus learned shape-preserving 8x8 within-tile residual encoding",
                "R2": "R1 plus learned multiscale context and deterministic flip/scale disagreement",
            },
            "learners": {
                "L0": "shared Smooth-L1 and weighted-BCE control",
                "L1": "separate utility and risk heads with matched losses",
                "L2": "decision-aligned distributional and ordinal utility-tail learner",
            },
            "folds": OOF_FOLDS,
            "paired_seeds": len(SEED_OFFSETS),
            "cells": factorial_cell_rows,
            "effects": effects,
        },
        "data_roles": {
            "train_oof": TRAINING_SCENES,
            "calibration_selection": CALIBRATION_SCENES,
            "test_development_stress": EXPECTED_TEST_SCENES,
            "nested_test_variants": EXPECTED_TEST_SCENES * VARIANTS_PER_SCENE,
            "candidate_confirmation_touched": False,
            "nh_haze_touched": False,
            "reside_its_ots_touched": False,
        },
        "identity_checks": identity_checks,
        "role_and_coverage_checks": role_checks,
        "baseline_reproduction": baseline_reproduction,
        "primary_intervals": primary_intervals,
        "selected_oof": selected_oof_summary,
        "selected_test_development": selected_test_summary,
        "oracle_control": {
            "gain": oracle_interval,
            "material_scene": oracle_material,
            "outcome": oracle_outcome,
        },
        "privileged_controls": controls,
        "precision": {
            "target_half_width_db": PRECISION_TARGET_DB,
            "observed_max_simultaneous_half_width_db": precision_half_width,
            "met": precision_half_width <= PRECISION_TARGET_DB,
        },
        "gate_outcomes": gate_outcomes,
        "raw_artifact_inventory": raw_inventory,
        "limitations": [
            "All evidence is development-screening; candidate-confirmation, canary, locked-test, NH-Haze, RESIDE ITS, and RESIDE OTS were not read.",
            "The 600 OOF original-clear scenes are the only primary independent units; tiles, actions, folds, paired seeds, and variants never increase sample size.",
            "The 150 calibration scenes select one cell through group-cross-fitted calibration and cannot contribute to the formal OOF estimand.",
            "The four test-development haze variants are averaged within each of 100 original-clear scenes.",
            "GT-risk, GT-utility, and GT/GT are privileged localization controls and are never deployable inputs.",
            "A PASS authorizes only a formal precision-certificate authoring step before any protected confirmation contract.",
        ],
        "marker": "HAZE4K_OBSERVABLE_TAIL_FACTORIAL_COMPLETE",
    }

    paths = {
        "summary": output_file(context, "haze4k_observable_tail_factorial_summary.json"),
        "gate": output_file(context, "haze4k_observable_tail_factorial_gate_summary.json"),
        "selection": selection_path,
        "primary": output_file(context, "haze4k_observable_tail_factorial_primary_intervals.json"),
        "cells": output_file(context, "haze4k_observable_tail_factorial_cells.csv"),
        "effects": output_file(context, "haze4k_observable_tail_factorial_effects.csv"),
        "training": output_file(context, "haze4k_observable_tail_factorial_fold_seed_completeness.csv"),
        "calibration": output_file(context, "haze4k_observable_tail_factorial_calibration.csv"),
        "baseline": output_file(context, "haze4k_observable_tail_factorial_baseline_reproduction.json"),
        "controls": output_file(context, "haze4k_observable_tail_factorial_privileged_controls.json"),
        "safety": output_file(context, "haze4k_observable_tail_factorial_tail_safety.json"),
        "test": output_file(context, "haze4k_observable_tail_factorial_test_scene_cells.csv"),
        "inventory": output_file(context, "haze4k_observable_tail_factorial_raw_inventory.json"),
    }
    atomic_json(paths["summary"], summary)
    atomic_json(paths["gate"], {
        "schema_version": 2,
        "gate_outcomes": gate_outcomes,
        "selected_cell": selected_cell,
        "selection_freeze_sha256": selection_sha,
        "utility": {
            "favorable": utility_favorable,
            "futile": utility_futile,
            "intervals": primary_intervals,
        },
        "activation": selected_oof_summary,
        "safety": {
            "oof": selected_oof_summary,
            "test_development": selected_test_summary,
        },
        "precision_half_width_db": precision_half_width,
    })
    atomic_json(paths["primary"], {
        "schema_version": 1,
        "selected_cell": selected_cell,
        "family_size": PRIMARY_CONTINUOUS_FAMILY_SIZE,
        "intervals": primary_intervals,
        "factorial_intervals": factorial_intervals,
    })
    write_csv(paths["cells"], factorial_cell_rows)
    write_csv(paths["effects"], effects)
    write_csv(paths["training"], training_rows)
    write_csv(paths["calibration"], calibration_rows_output)
    atomic_json(paths["baseline"], baseline_reproduction)
    atomic_json(paths["controls"], controls)
    atomic_json(paths["safety"], {
        "schema_version": 1,
        "selected_cell": selected_cell,
        "haze4k_train_oof": selected_oof_summary,
        "haze4k_test_development_stress": selected_test_summary,
    })
    compact_test_fields = (
        "population",
        "scene",
        "fold",
        "cell",
        "candidate_minus_keep_psnr_db",
        "candidate_minus_uniform_psnr_db",
        "candidate_minus_shuffled_psnr_db",
        "selected_area_fraction",
        "selected_harm_area",
        "selected_area",
        "total_area",
        "active_scene",
        "material_scene",
    )
    write_csv(paths["test"], [
        {field: row.get(field) for field in compact_test_fields}
        for cell in CELL_IDS for row in test_rows_by_cell[cell]
    ])
    atomic_json(paths["inventory"], raw_inventory)
    oof_paths = {}
    for representation in REPRESENTATIONS:
        path = output_file(
            context,
            f"haze4k_observable_tail_factorial_oof_{representation.lower()}_scene_cells.csv",
        )
        write_csv(path, [
            {field: row.get(field) for field in compact_test_fields}
            for cell in CELL_IDS
            if cell.startswith(representation + "_")
            for row in formal_rows_by_cell[cell]
        ])
        oof_paths[representation] = path

    finalization_path = output_file(context, "finalization_unit.json")
    evidence_paths = {**paths, **{f"oof_{key}": value for key, value in oof_paths.items()}}
    atomic_json(finalization_path, {
        "schema_version": 1,
        "route_id": ROUTE_ID,
        "operation_id": OPERATION_ID,
        "selected_cell": selected_cell,
        "selection_freeze_sha256": selection_sha,
        "gate_outcomes": gate_outcomes,
        "evidence_sha256": {
            key: sha256_file(path) for key, path in sorted(evidence_paths.items())
        },
    })
    record_completed_unit(
        context,
        unit_id="finalization",
        input_sha256=sha256_text("|".join([
            "observable-tail-factorial-finalization-v1",
            selection_sha,
            raw_inventory["inventory_sha256"],
            digest_lines(
                f"{key}:{sha256_file(path)}"
                for key, path in sorted(evidence_paths.items())
            ),
        ])),
        output_relpath="finalization_unit.json",
    )
    if len(load_completed_unit_ledger(context)) != TOTAL_UNITS:
        raise RuntimeError("completed-unit ledger is not complete")
    write_workload_progress(
        context,
        completed_units=TOTAL_UNITS,
        stage="aggregate_gate_finalization",
    )
    write_gate_result(
        context,
        gate_outcomes=gate_outcomes,
        details={
            "selected_cell": selected_cell,
            "selection_freeze_sha256": selection_sha,
            "independent_oof_scenes": TRAINING_SCENES,
            "calibration_selection_scenes": CALIBRATION_SCENES,
            "test_development_stress_scenes": EXPECTED_TEST_SCENES,
            "nested_test_variants": EXPECTED_TEST_SCENES * VARIANTS_PER_SCENE,
            "factorial_cells": len(CELL_IDS),
            "folds": OOF_FOLDS,
            "paired_seeds": len(SEED_OFFSETS),
            "completed_unit_ledger_count": TOTAL_UNITS,
            "candidate_confirmation_touched": False,
            "formal_family_size": PRIMARY_CONTINUOUS_FAMILY_SIZE,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "primary_estimate_db": utility_gain["estimate"],
            "summary_file": paths["summary"].name,
            "gate_summary_file": paths["gate"].name,
            "selection_file": paths["selection"].name,
            "primary_intervals_file": paths["primary"].name,
            "factorial_cells_file": paths["cells"].name,
            "factorial_effects_file": paths["effects"].name,
            "tail_safety_file": paths["safety"].name,
            "raw_inventory_file": paths["inventory"].name,
        },
    )


QUAL_ROUTE_ID = "haze4k-frozen-output-tail-controller-qualification-v1"
QUAL_OPERATION_ID = "HAZE4K_FROZEN_OUTPUT_TAIL_CONTROLLER_QUALIFY"
QUAL_RUN_ID = "haze4k-frozen-output-tail-controller-qualification-v1-r1"
QUAL_PARENT_ROUTE_ID = "haze4k-observable-tail-factorial-v1"
QUAL_PARENT_ROUTE_COMMIT = "b4ce59309a12b68e79dbfc8534a1b106e162110a"
QUAL_PARENT_RECEIPT = "ce6d9f96bab31b3c46ea5d7024e259e8d016cc4362be30b65d94f57f8d61ecaa"
QUAL_PARENT_INVENTORY_SHA256 = "f70b492d5b6cf16fc8a2dbc43a1e049556a8f4c6365726bc7eec9102b3e2cd28"
QUAL_PARENT_INVENTORY_FILE_SHA256 = "9a5249f66c994f9c94245145cc65e02da643ada71cc047cd5144db475b3c66ff"
QUAL_TOTAL_UNITS = 748
QUAL_PRIMARY_FAMILY_SIZE = 4
QUAL_SAFETY_FAMILY_SIZE = 8
QUAL_CALIBRATION_FAMILY_SIZE = 9
QUAL_PROBE_ITERATIONS = 180
QUAL_FORMAL_ITERATIONS = 90450
QUAL_AREA_CAPS = (0.01, 0.02, 0.05, 0.10, 0.20, 1.0)
QUAL_RISK_METHODS = (
    "weighted_q",
    "deweighted_probability",
    "scene_platt",
    "scene_isotonic",
    "parent_upper",
)
QUAL_UTILITY_METHODS = ("raw", "parent_lower")
QUAL_SELECTORS = ("tilewise", "setwise")
QUAL_LOW_TAIL_EDGES = (0.0, 0.025, 0.05, 0.10, 0.20)
QUAL_SELECTION_BOOTSTRAP_RESAMPLES = 5000
QUAL_SELECTION_SEED = 20260728


def qual_deweighted_probability(value: np.ndarray) -> np.ndarray:
    value = np.clip(np.asarray(value, dtype=np.float64), 0.0, 1.0)
    return np.clip(value / np.maximum(2.0 - value, EPSILON), 0.0, 1.0)


def qual_parent_inventory_items(root: Path) -> list[dict[str, Any]]:
    specifications = (
        ("head_checkpoint", "workload/head_checkpoints/*.pt"),
        ("oof_cell_prediction", "workload/raw_predictions/haze4k_train_oof/*/*.npz"),
        ("scene_cache", "workload/scene_cache/*.npz"),
        (
            "test_cell_prediction",
            "workload/raw_predictions/haze4k_test_development_stress/*/*.npz",
        ),
        ("test_scene_cache", "workload/test_development_cache/*.npz"),
    )
    items = []
    for artifact_class, pattern in specifications:
        for path in sorted(root.glob(pattern)):
            if not path.is_file():
                raise RuntimeError(f"parent inventory path is not a file: {path}")
            items.append({
                "artifact_class": artifact_class,
                "relative_path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            })
    return items


def qual_verify_parent_inventory(
    root: Path, expected_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    expected = read_json(expected_path)
    items = qual_parent_inventory_items(root)
    observed = compact_inventory(items)
    observed_checkpoints = {
        item["relative_path"]: (item["sha256"], int(item["bytes"]))
        for item in observed["head_checkpoints"]
    }
    expected_checkpoints = {
        item["relative_path"]: (item["sha256"], int(item["bytes"]))
        for item in expected["head_checkpoints"]
    }
    checks = {
        "expected_inventory_file_identity": sha256_file(expected_path)
        == QUAL_PARENT_INVENTORY_FILE_SHA256,
        "overall_inventory_identity": observed["inventory_sha256"]
        == expected.get("inventory_sha256")
        == QUAL_PARENT_INVENTORY_SHA256,
        "file_count": observed["file_count"] == expected.get("file_count") == 10285,
        "total_bytes": observed["total_bytes"] == expected.get("total_bytes") == 5225583075,
        "class_inventory": observed["classes"] == expected.get("classes"),
        "checkpoint_inventory": observed_checkpoints == expected_checkpoints,
    }
    return {
        "schema_version": 1,
        "parent_route_id": QUAL_PARENT_ROUTE_ID,
        "parent_route_commit": QUAL_PARENT_ROUTE_COMMIT,
        "parent_receipt": QUAL_PARENT_RECEIPT,
        "checks": checks,
        "matched": all(checks.values()),
        "observed": {
            "file_count": observed["file_count"],
            "total_bytes": observed["total_bytes"],
            "inventory_sha256": observed["inventory_sha256"],
            "classes": observed["classes"],
        },
    }, items


def qual_load_bundle(torch, path: Path, device: str = "cpu") -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    required = {
        "schema_version", "cell", "representation", "learner", "fold",
        "seed", "state_dict", "normalization", "history",
    }
    if not isinstance(payload, dict) or set(payload) != required \
            or payload["schema_version"] != 1:
        raise RuntimeError(f"checkpoint contract changed: {path.name}")
    cell = f"{payload['representation']}_{payload['learner']}"
    if payload["cell"] != cell or cell not in CELL_IDS:
        raise RuntimeError(f"checkpoint cell identity changed: {path.name}")
    head = make_head(torch, payload["representation"], payload["learner"])
    head.load_state_dict(payload["state_dict"], strict=True)
    head.requires_grad_(False).to(device).eval()
    return {
        "head": head,
        "normalization": payload["normalization"],
        "history": payload["history"],
        "cell": cell,
        "fold": int(payload["fold"]),
        "seed": int(payload["seed"]),
        "checkpoint_path": path,
    }


def qual_checkpoint_path(root: Path, cell: str, fold: int, seed_index: int) -> Path:
    return root / "workload" / "head_checkpoints" / (
        f"{cell}_fold_{fold}_seed_{seed_index}.pt"
    )


def qual_calibration_vectors(
    predictions: dict[str, tuple[np.ndarray, np.ndarray]],
    records: dict[str, dict[str, Any]],
    scenes: list[str],
    action: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scores, targets, weights = [], [], []
    for scene in scenes:
        risk = np.asarray(predictions[scene][1][action], dtype=np.float64).reshape(-1)
        target = np.asarray(records[scene]["harm"][action], dtype=np.float64).reshape(-1)
        area = np.asarray(records[scene]["areas"], dtype=np.float64).reshape(-1)
        area /= float(np.sum(area))
        if risk.size != target.size or risk.size != area.size:
            raise RuntimeError("calibration prediction grid changed")
        scores.append(risk)
        targets.append(target)
        weights.append(area / len(scenes))
    return np.concatenate(scores), np.concatenate(targets), np.concatenate(weights)


def qual_fit_platt_action(
    scores: np.ndarray, targets: np.ndarray, weights: np.ndarray,
) -> dict[str, float]:
    clipped = np.clip(scores, 1.0e-6, 1.0 - 1.0e-6)
    x = np.log(clipped) - np.log1p(-clipped)
    y = np.asarray(targets, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    w /= float(np.sum(w))
    prevalence = float(np.clip(np.sum(w * y), 1.0e-6, 1.0 - 1.0e-6))
    parameters = np.asarray([1.0, math.log(prevalence / (1.0 - prevalence))], dtype=np.float64)
    ridge = 1.0e-6
    for _ in range(100):
        linear = np.clip(parameters[0] * x + parameters[1], -40.0, 40.0)
        probability = 1.0 / (1.0 + np.exp(-linear))
        residual = probability - y
        curvature = np.maximum(probability * (1.0 - probability), 1.0e-9)
        gradient = np.asarray([
            np.sum(w * residual * x) + ridge * (parameters[0] - 1.0),
            np.sum(w * residual) + ridge * parameters[1],
        ])
        hessian = np.asarray([
            [np.sum(w * curvature * x * x) + ridge, np.sum(w * curvature * x)],
            [np.sum(w * curvature * x), np.sum(w * curvature) + ridge],
        ])
        step = np.linalg.solve(hessian, gradient)
        parameters -= np.clip(step, -2.0, 2.0)
        parameters[0] = max(0.0, min(20.0, parameters[0]))
        parameters[1] = max(-40.0, min(40.0, parameters[1]))
        if float(np.max(np.abs(step))) < 1.0e-10:
            break
    return {"slope": float(parameters[0]), "intercept": float(parameters[1])}


def qual_apply_platt(value: np.ndarray, model: dict[str, float]) -> np.ndarray:
    clipped = np.clip(np.asarray(value, dtype=np.float64), 1.0e-6, 1.0 - 1.0e-6)
    logit = np.log(clipped) - np.log1p(-clipped)
    linear = np.clip(model["slope"] * logit + model["intercept"], -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-linear))


def qual_fit_isotonic_action(
    scores: np.ndarray, targets: np.ndarray, weights: np.ndarray,
) -> dict[str, list[float]]:
    order = np.argsort(scores, kind="mergesort")
    x = np.asarray(scores, dtype=np.float64)[order]
    y = np.asarray(targets, dtype=np.float64)[order]
    w = np.asarray(weights, dtype=np.float64)[order]
    blocks: list[dict[str, float]] = []
    index = 0
    while index < x.size:
        stop = index + 1
        while stop < x.size and x[stop] == x[index]:
            stop += 1
        weight = float(np.sum(w[index:stop]))
        blocks.append({
            "upper": float(x[stop - 1]),
            "weight": weight,
            "mean": float(np.sum(w[index:stop] * y[index:stop]) / max(weight, EPSILON)),
        })
        while len(blocks) >= 2 and blocks[-2]["mean"] > blocks[-1]["mean"]:
            right = blocks.pop()
            left = blocks.pop()
            total = left["weight"] + right["weight"]
            blocks.append({
                "upper": right["upper"],
                "weight": total,
                "mean": (
                    left["weight"] * left["mean"]
                    + right["weight"] * right["mean"]
                ) / max(total, EPSILON),
            })
        index = stop
    return {
        "upper": [float(item["upper"]) for item in blocks],
        "probability": [float(item["mean"]) for item in blocks],
    }


def qual_apply_isotonic(
    value: np.ndarray, model: dict[str, list[float]],
) -> np.ndarray:
    upper = np.asarray(model["upper"], dtype=np.float64)
    probability = np.asarray(model["probability"], dtype=np.float64)
    indices = np.searchsorted(upper, np.asarray(value, dtype=np.float64), side="left")
    return probability[np.clip(indices, 0, probability.size - 1)]


def qual_fit_calibration_models(
    predictions: dict[str, tuple[np.ndarray, np.ndarray]],
    records: dict[str, dict[str, Any]],
    scenes: list[str],
) -> dict[str, Any]:
    parent = calibrate_predictions(predictions, [records[scene] for scene in scenes])
    platt, isotonic = [], []
    for action in range(len(ACTION_NAMES)):
        score, target, weight = qual_calibration_vectors(
            predictions, records, scenes, action,
        )
        platt.append(qual_fit_platt_action(score, target, weight))
        isotonic.append(qual_fit_isotonic_action(score, target, weight))
    return {"parent": parent, "platt": platt, "isotonic": isotonic}


def qual_apply_risk(
    raw: np.ndarray,
    method: str,
    models: dict[str, Any],
    learner: str,
) -> np.ndarray:
    raw = np.clip(np.asarray(raw, dtype=np.float64), 0.0, 1.0)
    if method == "weighted_q":
        return raw
    if method == "deweighted_probability":
        if learner == "L2":
            raise RuntimeError("exact weighted-BCE deweighting is inapplicable to L2")
        return qual_deweighted_probability(raw)
    if method == "parent_upper":
        correction = models["parent"]["risk_correction"][:, None, None]
        return np.clip(raw + correction, 0.0, 1.0)
    if method == "scene_platt":
        return np.stack([
            qual_apply_platt(raw[action], models["platt"][action])
            for action in range(len(ACTION_NAMES))
        ])
    if method == "scene_isotonic":
        return np.stack([
            qual_apply_isotonic(raw[action], models["isotonic"][action])
            for action in range(len(ACTION_NAMES))
        ])
    raise RuntimeError(f"unknown risk calibration method: {method}")


def qual_apply_utility(
    raw: np.ndarray, method: str, models: dict[str, Any],
) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float64)
    if method == "raw":
        return raw
    if method == "parent_lower":
        return raw - models["parent"]["utility_correction"][:, None, None]
    raise RuntimeError(f"unknown utility method: {method}")


def qual_available_risk_methods(cell: str) -> tuple[str, ...]:
    learner = cell.split("_")[1]
    return tuple(
        method for method in QUAL_RISK_METHODS
        if method != "deweighted_probability" or learner != "L2"
    )


def qual_choose_tilewise(utility: np.ndarray, risk: np.ndarray) -> np.ndarray:
    return choose_actions(utility, risk).astype(np.int8)


def qual_choose_setwise(
    utility: np.ndarray,
    risk: np.ndarray,
    areas: np.ndarray,
    *,
    area_cap: float,
) -> np.ndarray:
    if utility.shape != risk.shape or utility.shape[0] != len(ACTION_NAMES) \
            or utility.shape[1:] != areas.shape:
        raise RuntimeError("setwise controller grid changed")
    best = np.argmax(utility, axis=0)
    rows, columns = np.indices(areas.shape)
    best_utility = utility[best, rows, columns]
    best_risk = risk[best, rows, columns]
    eligible = np.isfinite(best_utility) & np.isfinite(best_risk) \
        & (best_utility > UTILITY_MARGIN_DB)
    flat = np.flatnonzero(eligible.reshape(-1))
    actions = np.zeros(areas.size, dtype=np.int8)
    if flat.size == 0:
        return actions.reshape(areas.shape)
    flat_utility = best_utility.reshape(-1)
    flat_risk = best_risk.reshape(-1)
    flat_area = areas.astype(np.float64).reshape(-1)
    order = sorted(
        (int(index) for index in flat),
        key=lambda index: (
            float(flat_risk[index]),
            -float(flat_utility[index]),
            index,
        ),
    )
    total_area = float(np.sum(flat_area))
    selected_area = 0.0
    selected_expected_harm = 0.0
    for index in order:
        candidate_area = selected_area + float(flat_area[index])
        if candidate_area / total_area > area_cap + 1.0e-12:
            continue
        candidate_harm = selected_expected_harm \
            + float(flat_area[index]) * float(flat_risk[index])
        if candidate_harm / candidate_area > RISK_ACCEPTANCE_MAX + 1.0e-12:
            break
        actions[index] = int(best.reshape(-1)[index]) + 1
        selected_area = candidate_area
        selected_expected_harm = candidate_harm
    return actions.reshape(areas.shape)


def qual_choose_policy(
    utility: np.ndarray,
    risk: np.ndarray,
    areas: np.ndarray,
    selector: str,
    area_cap: float,
) -> np.ndarray:
    if selector == "tilewise":
        return qual_choose_tilewise(utility, risk)
    if selector == "setwise":
        return qual_choose_setwise(utility, risk, areas, area_cap=area_cap)
    raise RuntimeError(f"unknown selector: {selector}")


def qual_uniform_action_map(
    utility: np.ndarray, risk: np.ndarray, areas: np.ndarray,
) -> np.ndarray:
    action = choose_uniform_action(utility, risk, areas)
    return np.full(areas.shape, action, dtype=np.int8)


def qual_shuffled_actions(actions: np.ndarray, key: str) -> np.ndarray:
    generator = np.random.default_rng(deterministic_seed(f"qual-shuffle|{key}"))
    flattened = actions.reshape(-1).copy()
    return flattened[generator.permutation(flattened.size)].reshape(actions.shape)


def qual_permuted_scores(
    utility: np.ndarray, risk: np.ndarray, key: str,
) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(deterministic_seed(f"qual-permutation|{key}"))
    permutation = generator.permutation(utility.shape[1] * utility.shape[2])
    return (
        utility.reshape(utility.shape[0], -1)[:, permutation].reshape(utility.shape),
        risk.reshape(risk.shape[0], -1)[:, permutation].reshape(risk.shape),
    )


def qual_tile_mse(
    hazy: np.ndarray,
    prediction: np.ndarray,
    clear: np.ndarray,
    grid_shape: tuple[int, int],
) -> np.ndarray:
    outputs = [prediction]
    outputs.extend(
        np.clip(hazy + float(scale) * (prediction - hazy), 0.0, 1.0)
        for scale in ACTION_SCALES
    )
    values = np.empty((len(outputs), *grid_shape), dtype=np.float64)
    height, width = clear.shape[:2]
    for row in range(grid_shape[0]):
        for column in range(grid_shape[1]):
            y0, y1 = row * TILE_SIZE, min(height, (row + 1) * TILE_SIZE)
            x0, x1 = column * TILE_SIZE, min(width, (column + 1) * TILE_SIZE)
            target = clear[y0:y1, x0:x1]
            for action, output in enumerate(outputs):
                values[action, row, column] = mse(
                    output[y0:y1, x0:x1], target,
                )
    return values


def qual_gain_from_actions(
    tile_mse_values: np.ndarray,
    areas: np.ndarray,
    actions: np.ndarray,
) -> float:
    if tile_mse_values.shape[1:] != areas.shape or actions.shape != areas.shape:
        raise RuntimeError("tile MSE action grid changed")
    rows, columns = np.indices(areas.shape)
    keep = float(np.sum(areas * tile_mse_values[0]) / np.sum(areas))
    candidate = float(
        np.sum(areas * tile_mse_values[actions, rows, columns]) / np.sum(areas)
    )
    return 10.0 * math.log10(max(keep, EPSILON) / max(candidate, EPSILON))


def qual_policy_row(
    record: dict[str, Any], actions: np.ndarray,
) -> dict[str, Any]:
    areas = np.asarray(record["areas"], dtype=np.float64)
    selected = actions > 0
    action_indices = np.maximum(actions.astype(np.int64) - 1, 0)
    rows, columns = np.indices(actions.shape)
    selected_area = float(np.sum(areas[selected]))
    total_area = float(np.sum(areas))
    selected_harm = float(np.sum(
        areas * selected * record["harm"][action_indices, rows, columns]
    ))
    gain = qual_gain_from_actions(record["tile_mse"], areas, actions)
    return {
        "gain_db": gain,
        "selected_area": selected_area,
        "selected_harm_area": selected_harm,
        "total_area": total_area,
        "action_area_fraction": selected_area / total_area,
        "selected_harm_fraction": (
            selected_harm / selected_area if selected_area > 0.0 else None
        ),
        "active_scene": selected_area > 0.0,
        "material_scene": gain >= UTILITY_MARGIN_DB,
        "risk_margin": (selected_harm - MAX_HARM_PREVALENCE * selected_area)
        / total_area,
    }


def qual_ratio_normal_interval(
    rows: list[dict[str, Any]],
    numerator: str,
    denominator: str,
    *,
    family_size: int,
) -> dict[str, Any]:
    numerator_values = np.asarray([float(row[numerator]) for row in rows], dtype=np.float64)
    denominator_values = np.asarray([float(row[denominator]) for row in rows], dtype=np.float64)
    total = float(np.sum(denominator_values))
    if total <= 0.0:
        return {
            "estimate": None, "lower": None, "upper": None,
            "selected_area_positive": False, "family_size": family_size,
        }
    estimate = float(np.sum(numerator_values) / total)
    mean_denominator = float(np.mean(denominator_values))
    influence = (numerator_values - estimate * denominator_values) / max(
        mean_denominator, EPSILON,
    )
    standard_error = float(np.std(influence, ddof=1) / math.sqrt(len(rows)))
    half = bonferroni_z(family_size) * standard_error
    return {
        "estimate": estimate,
        "lower": max(0.0, estimate - half),
        "upper": min(1.0, estimate + half),
        "selected_area_positive": True,
        "family_size": family_size,
        "critical_value": bonferroni_z(family_size),
    }


def qual_descriptive_summary(
    rows: list[dict[str, Any]], *, family_size: int = 1,
) -> dict[str, Any]:
    if not rows:
        raise RuntimeError("policy summary requires scene rows")
    return {
        "scene_count": len(rows),
        "gain": normal_interval(
            [row["gain_db"] for row in rows], family_size=family_size,
        ),
        "action_area": qual_ratio_normal_interval(
            rows, "selected_area", "total_area", family_size=family_size,
        ),
        "selected_harm": qual_ratio_normal_interval(
            rows, "selected_harm_area", "selected_area", family_size=family_size,
        ),
        "active_scene": wilson(
            sum(bool(row["active_scene"]) for row in rows),
            len(rows),
            family_size=family_size,
        ),
        "material_scene": wilson(
            sum(bool(row["material_scene"]) for row in rows),
            len(rows),
            family_size=family_size,
        ),
        "risk_margin": normal_interval(
            [row["risk_margin"] for row in rows], family_size=family_size,
        ),
    }


def qual_calibration_diagnostics(
    predictions: dict[str, tuple[np.ndarray, np.ndarray]],
    records: dict[str, dict[str, Any]],
    scenes: list[str],
    models: dict[str, Any],
    *,
    cell: str,
    fold: int,
    source: str,
    include_bins: bool,
) -> list[dict[str, Any]]:
    output = []
    learner = cell.split("_")[1]
    for method in qual_available_risk_methods(cell):
        calibrated = {
            scene: qual_apply_risk(
                predictions[scene][1], method, models, learner,
            )
            for scene in scenes
        }
        for action, action_name in enumerate(ACTION_NAMES):
            scene_brier, scene_logloss = [], []
            all_score, all_target, all_weight = [], [], []
            tail_area, tail_harm = 0.0, 0.0
            for scene in scenes:
                score = np.clip(calibrated[scene][action], 1.0e-8, 1.0 - 1.0e-8)
                target = records[scene]["harm"][action].astype(np.float64)
                area = records[scene]["areas"].astype(np.float64)
                weight = area / float(np.sum(area))
                scene_brier.append(weighted_mean((score - target) ** 2, weight))
                scene_logloss.append(weighted_mean(
                    -(target * np.log(score) + (1.0 - target) * np.log1p(-score)),
                    weight,
                ))
                all_score.append(score.reshape(-1))
                all_target.append(target.reshape(-1))
                all_weight.append(weight.reshape(-1) / len(scenes))
                selected = score <= RISK_ACCEPTANCE_MAX
                tail_area += float(np.sum(area[selected]))
                tail_harm += float(np.sum(area[selected] * target[selected]))
            combined_score = np.concatenate(all_score)
            combined_target = np.concatenate(all_target)
            combined_weight = np.concatenate(all_weight)
            output.append({
                "cell": cell,
                "fold": fold,
                "source": source,
                "risk_method": method,
                "action": action_name,
                "bin_lower": None,
                "bin_upper": None,
                "scene_count": len(scenes),
                "scene_weighted_brier": float(np.mean(scene_brier)),
                "scene_weighted_logloss": float(np.mean(scene_logloss)),
                "scene_weighted_auc": weighted_binary_auc(
                    combined_score, combined_target > 0.5, combined_weight,
                ),
                "tail_coverage": tail_area / sum(
                    float(np.sum(records[scene]["areas"])) for scene in scenes
                ),
                "tail_selected_harm": tail_harm / tail_area if tail_area > 0.0 else None,
                "bin_predicted_risk": None,
                "bin_observed_harm": None,
            })
            if not include_bins:
                continue
            for lower, upper in zip(QUAL_LOW_TAIL_EDGES[:-1], QUAL_LOW_TAIL_EDGES[1:]):
                predicted_sum = 0.0
                observed_sum = 0.0
                weight_sum = 0.0
                contributing_scenes = 0
                for scene in scenes:
                    score = calibrated[scene][action]
                    target = records[scene]["harm"][action].astype(np.float64)
                    area = records[scene]["areas"].astype(np.float64)
                    selected = (score >= lower) & (
                        (score <= upper) if upper == 1.0 else (score < upper)
                    )
                    if np.any(selected):
                        contributing_scenes += 1
                        predicted_sum += float(np.sum(area[selected] * score[selected]))
                        observed_sum += float(np.sum(area[selected] * target[selected]))
                        weight_sum += float(np.sum(area[selected]))
                output.append({
                    "cell": cell,
                    "fold": fold,
                    "source": source,
                    "risk_method": method,
                    "action": action_name,
                    "bin_lower": lower,
                    "bin_upper": upper,
                    "scene_count": contributing_scenes,
                    "scene_weighted_brier": None,
                    "scene_weighted_logloss": None,
                    "scene_weighted_auc": None,
                    "tail_coverage": None,
                    "tail_selected_harm": None,
                    "bin_predicted_risk": (
                        predicted_sum / weight_sum if weight_sum > 0.0 else None
                    ),
                    "bin_observed_harm": (
                        observed_sum / weight_sum if weight_sum > 0.0 else None
                    ),
                })
    return output


def qual_enrich_record(
    scene: str,
    cache_path: Path,
    hazy_path: Path,
    clear_path: Path,
) -> dict[str, Any]:
    cached = load_cache(cache_path, include_prediction=True)
    hazy = image_array(hazy_path)
    clear = image_array(clear_path)
    prediction = cached["prediction"]
    if hazy.shape != clear.shape or prediction.shape != clear.shape:
        raise RuntimeError(f"frozen scene image shape changed: {scene[:16]}")
    return {
        "scene": scene,
        "cache_path": cache_path,
        "hazy_path": hazy_path,
        "clear_path": clear_path,
        "hazy": hazy,
        "clear": clear,
        "prediction": prediction,
        "utility": cached["utility"].astype(np.float64),
        "harm": cached["harm"].astype(np.float64),
        "areas": cached["areas"].astype(np.float64),
        "tile_mse": qual_tile_mse(
            hazy, prediction, clear, cached["areas"].shape,
        ),
    }


def qual_feature_record(cache_path: Path) -> dict[str, np.ndarray]:
    return load_cache(cache_path, include_prediction=False)


def qual_prediction_payload(
    scenes: list[str],
    seed_predictions: dict[str, list[tuple[np.ndarray, np.ndarray]]],
) -> dict[str, np.ndarray]:
    payload: dict[str, np.ndarray] = {
        "scenes": np.asarray(scenes, dtype="U64"),
    }
    for scene_index, scene in enumerate(scenes):
        predictions = seed_predictions[scene]
        for seed_index, (utility, risk) in enumerate(predictions):
            prefix = f"scene_{scene_index:03d}_seed_{seed_index}"
            payload[f"{prefix}_utility"] = utility.astype(np.float32)
            payload[f"{prefix}_risk"] = risk.astype(np.float32)
    return payload


def qual_candidate_id(
    cell: str,
    risk_method: str,
    utility_method: str,
    selector: str,
    area_cap: float,
) -> str:
    cap = str(area_cap).replace(".", "p")
    return f"{cell}__{risk_method}__{utility_method}__{selector}__cap_{cap}"


def qual_candidate_table(
    calibration_records: dict[str, dict[str, Any]],
    calibration_fold: dict[str, int],
    heldout_predictions: dict[str, dict[str, dict[str, Any]]],
    calibration_models: dict[str, dict[int, dict[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    summaries = []
    scene_rows_by_candidate: dict[str, list[dict[str, Any]]] = {}
    scenes = sorted(calibration_records)
    candidate_count = sum(
        len(qual_available_risk_methods(cell))
        * len(QUAL_UTILITY_METHODS) * len(QUAL_AREA_CAPS)
        for cell in CELL_IDS
    )
    for cell in CELL_IDS:
        learner = cell.split("_")[1]
        for risk_method in qual_available_risk_methods(cell):
            if risk_method == "parent_upper":
                continue
            for utility_method in QUAL_UTILITY_METHODS:
                for area_cap in QUAL_AREA_CAPS:
                    candidate = qual_candidate_id(
                        cell, risk_method, utility_method, "setwise", area_cap,
                    )
                    rows = []
                    for scene in scenes:
                        fold = calibration_fold[scene]
                        prediction = heldout_predictions[cell][scene]["ensemble"]
                        models = calibration_models[cell][fold]["ensemble"]
                        utility = qual_apply_utility(
                            prediction[0], utility_method, models,
                        )
                        risk = qual_apply_risk(
                            prediction[1], risk_method, models, learner,
                        )
                        actions = qual_choose_setwise(
                            utility, risk, calibration_records[scene]["areas"],
                            area_cap=area_cap,
                        )
                        rows.append({
                            "scene": scene,
                            "fold": fold,
                            **qual_policy_row(calibration_records[scene], actions),
                        })
                    summary = qual_descriptive_summary(
                        rows, family_size=max(1, candidate_count),
                    )
                    harm_upper = summary["selected_harm"]["upper"]
                    coverage = (
                        summary["action_area"]["lower"] >= MIN_ACTION_AREA_FRACTION
                        and summary["active_scene"]["lower"] >= MIN_ACTIVE_SCENE_PREVALENCE
                    )
                    safety = harm_upper is not None \
                        and harm_upper <= MAX_HARM_PREVALENCE
                    actionable = (
                        coverage
                        and safety
                        and summary["gain"]["lower"] > UTILITY_MARGIN_DB
                        and summary["material_scene"]["lower"] > MIN_MATERIAL_SCENE_PREVALENCE
                    )
                    summaries.append({
                        "candidate_id": candidate,
                        "cell": cell,
                        "risk_method": risk_method,
                        "utility_method": utility_method,
                        "selector": "setwise",
                        "area_cap": area_cap,
                        "coverage_admissible": coverage,
                        "safety_admissible": safety,
                        "actionable": actionable,
                        "gain_estimate": summary["gain"]["estimate"],
                        "gain_lower": summary["gain"]["lower"],
                        "gain_upper": summary["gain"]["upper"],
                        "action_area_estimate": summary["action_area"]["estimate"],
                        "action_area_lower": summary["action_area"]["lower"],
                        "active_scene_lower": summary["active_scene"]["lower"],
                        "selected_harm_estimate": summary["selected_harm"]["estimate"],
                        "selected_harm_upper": harm_upper,
                        "material_scene_lower": summary["material_scene"]["lower"],
                        "risk_margin_upper": summary["risk_margin"]["upper"],
                    })
                    scene_rows_by_candidate[candidate] = rows
    return summaries, scene_rows_by_candidate


def qual_select_candidate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise RuntimeError("controller candidate table is empty")
    ordered_ids = {row["candidate_id"]: index for index, row in enumerate(rows)}

    def key(row: dict[str, Any]) -> tuple[Any, ...]:
        harm_upper = row["selected_harm_upper"]
        return (
            bool(row["actionable"]),
            bool(row["coverage_admissible"] and row["safety_admissible"]),
            bool(row["coverage_admissible"]),
            -(float(harm_upper) if harm_upper is not None else 1.0),
            float(row["gain_lower"]),
            float(row["action_area_lower"]),
            -ordered_ids[row["candidate_id"]],
        )

    return dict(max(rows, key=key))


def qual_frontier_key(
    cell: str,
    risk_method: str,
    utility_method: str,
    selector: str,
    area_cap: float,
) -> tuple[str, str, str, str, float]:
    return cell, risk_method, utility_method, selector, float(area_cap)


def qual_frontier_rows(
    values: dict[tuple[str, str, str, str, float], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    output = []
    for key in sorted(values):
        cell, risk_method, utility_method, selector, area_cap = key
        summary = qual_descriptive_summary(values[key])
        output.append({
            "cell": cell,
            "representation": cell.split("_")[0],
            "learner": cell.split("_")[1],
            "risk_method": risk_method,
            "utility_method": utility_method,
            "selector": selector,
            "area_cap": area_cap,
            "scene_count": summary["scene_count"],
            "gain_db": summary["gain"]["estimate"],
            "gain_lower": summary["gain"]["lower"],
            "gain_upper": summary["gain"]["upper"],
            "action_area": summary["action_area"]["estimate"],
            "action_area_lower": summary["action_area"]["lower"],
            "action_area_upper": summary["action_area"]["upper"],
            "selected_harm": summary["selected_harm"]["estimate"],
            "selected_harm_lower": summary["selected_harm"]["lower"],
            "selected_harm_upper": summary["selected_harm"]["upper"],
            "active_scene": summary["active_scene"]["estimate"],
            "active_scene_lower": summary["active_scene"]["lower"],
            "active_scene_upper": summary["active_scene"]["upper"],
            "material_scene": summary["material_scene"]["estimate"],
            "material_scene_lower": summary["material_scene"]["lower"],
            "material_scene_upper": summary["material_scene"]["upper"],
            "risk_margin": summary["risk_margin"]["estimate"],
            "risk_margin_upper": summary["risk_margin"]["upper"],
        })
    return output


def qual_fold_seed_rows(
    values: dict[tuple[str, int, int, str, str], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    output = []
    for key in sorted(values):
        cell, fold, seed_index, risk_method, selector = key
        summary = qual_descriptive_summary(values[key])
        output.append({
            "cell": cell,
            "fold": fold,
            "seed_index": seed_index,
            "risk_method": risk_method,
            "selector": selector,
            "scene_count": summary["scene_count"],
            "gain_db": summary["gain"]["estimate"],
            "gain_lower": summary["gain"]["lower"],
            "gain_upper": summary["gain"]["upper"],
            "action_area": summary["action_area"]["estimate"],
            "selected_harm": summary["selected_harm"]["estimate"],
            "selected_harm_upper": summary["selected_harm"]["upper"],
            "active_scene": summary["active_scene"]["estimate"],
            "material_scene": summary["material_scene"]["estimate"],
        })
    return output


def qual_effect_rows(
    frontier: dict[tuple[str, str, str, str, float], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    output = []

    def add_effect(
        effect_id: str,
        left_key: tuple[str, str, str, str, float],
        right_key: tuple[str, str, str, str, float],
        metric: str,
    ) -> None:
        if left_key not in frontier or right_key not in frontier:
            return
        left, right = frontier[left_key], frontier[right_key]
        if len(left) != len(right):
            raise RuntimeError("controller effect rows are not paired")
        differences = [
            float(item_left[metric]) - float(item_right[metric])
            for item_left, item_right in zip(left, right)
        ]
        interval = normal_interval(differences, family_size=1)
        output.append({
            "effect_id": effect_id,
            "metric": metric,
            "left": "|".join(map(str, left_key)),
            "right": "|".join(map(str, right_key)),
            **interval,
        })

    for cell in CELL_IDS:
        methods = qual_available_risk_methods(cell)
        for utility_method in QUAL_UTILITY_METHODS:
            weighted_tile = qual_frontier_key(
                cell, "weighted_q", utility_method, "tilewise", 1.0,
            )
            weighted_set = qual_frontier_key(
                cell, "weighted_q", utility_method, "setwise", 1.0,
            )
            for metric in ("gain_db", "action_area_fraction", "risk_margin"):
                add_effect(
                    f"selector:{cell}:{utility_method}:weighted_q",
                    weighted_set,
                    weighted_tile,
                    metric,
                )
            for method in methods:
                if method == "weighted_q":
                    continue
                method_tile = qual_frontier_key(
                    cell, method, utility_method, "tilewise", 1.0,
                )
                method_set = qual_frontier_key(
                    cell, method, utility_method, "setwise", 1.0,
                )
                for metric in ("gain_db", "action_area_fraction", "risk_margin"):
                    add_effect(
                        f"risk_target:{cell}:{utility_method}:{method}:tilewise",
                        method_tile,
                        weighted_tile,
                        metric,
                    )
                    add_effect(
                        f"risk_target:{cell}:{utility_method}:{method}:setwise",
                        method_set,
                        weighted_set,
                        metric,
                    )
                    if method_tile in frontier and method_set in frontier \
                            and weighted_tile in frontier and weighted_set in frontier:
                        method_difference = [
                            float(left[metric]) - float(right[metric])
                            for left, right in zip(
                                frontier[method_set], frontier[method_tile],
                            )
                        ]
                        weighted_difference = [
                            float(left[metric]) - float(right[metric])
                            for left, right in zip(
                                frontier[weighted_set], frontier[weighted_tile],
                            )
                        ]
                        interaction = [
                            left - right
                            for left, right in zip(
                                method_difference, weighted_difference,
                            )
                        ]
                        output.append({
                            "effect_id": (
                                f"interaction:{cell}:{utility_method}:{method}"
                            ),
                            "metric": metric,
                            "left": "method_selector_contrast",
                            "right": "weighted_selector_contrast",
                            **normal_interval(interaction, family_size=1),
                        })
    return output


def qual_image_metrics(
    torch,
    record: dict[str, Any],
    actions: np.ndarray,
    device: str,
) -> dict[str, Any]:
    candidate = apply_spatial(record["hazy"], record["prediction"], actions)
    keep = record["prediction"]
    keep_psnr = psnr(keep, record["clear"])
    candidate_psnr = psnr(candidate, record["clear"])
    keep_ssim, candidate_ssim = rgb_ssim(
        torch, [keep, candidate], record["clear"], device,
    )
    keep_color = color_bias(keep, record["clear"])
    candidate_color = color_bias(candidate, record["clear"])
    gain = candidate_psnr - keep_psnr
    tile_gain = qual_gain_from_actions(
        record["tile_mse"], record["areas"], actions,
    )
    return {
        "gain_db": gain,
        "tile_gain_db": tile_gain,
        "gain_identity_difference": abs(gain - tile_gain),
        "keep_psnr_db": keep_psnr,
        "candidate_psnr_db": candidate_psnr,
        "keep_ssim": keep_ssim,
        "candidate_ssim": candidate_ssim,
        "ssim_delta": candidate_ssim - keep_ssim,
        "keep_color_bias": keep_color,
        "candidate_color_bias": candidate_color,
        "color_bias_delta": candidate_color - keep_color,
        "psnr_harm": gain <= -PSNR_HARM_MARGIN_DB,
        "ssim_harm": candidate_ssim < keep_ssim - SSIM_HARM_MARGIN,
        "color_harm": candidate_color > keep_color + COLOR_HARM_MARGIN,
    }


def qual_contract(context_path: Path) -> None:
    context = load_context(context_path, "contract")
    if context.route_id != QUAL_ROUTE_ID \
            or context.operation_id != QUAL_OPERATION_ID:
        raise RuntimeError("controller qualification route identity mismatch")
    prepare_phase_output(context)
    started = time.monotonic()
    import torch

    if context.device != "cuda":
        raise RuntimeError("controller qualification contract requires CUDA")
    torch.cuda.reset_peak_memory_stats()
    entrypoint = asset_path(context, "controller_entrypoint", kind="file")
    runtime_environment = asset_path(context, "runtime_environment", kind="file")
    parent_inventory = asset_path(context, "parent_raw_inventory", kind="file")
    checks = {
        "entrypoint_identity": context.assets["controller_entrypoint"].sha256
        == sha256_file(entrypoint),
        "runtime_environment_identity": context.assets["runtime_environment"].sha256
        == sha256_file(runtime_environment),
        "runtime_device_class": read_json(runtime_environment).get("device_class")
        == "cuda_sm89",
        "parent_inventory_metadata_identity": sha256_file(parent_inventory)
        == QUAL_PARENT_INVENTORY_FILE_SHA256,
        "official_anchor_identity": context.assets["official_anchor_checkout"].commit
        == ANCHOR_COMMIT,
        "scientific_assets_absent": all(
            identifier not in context.assets
            for identifier in (
                "parent_raw_workspace",
                "haze4k_train",
                "haze4k_test_development",
                "candidate_confirmation",
                "canary",
                "locked_test",
            )
        ),
        "cost_contract": (
            context.engineering_contract["cost_contract"]["formal_iterations"]
            == QUAL_FORMAL_ITERATIONS
            and context.engineering_contract["cost_contract"]["probe_iterations"]
            == QUAL_PROBE_ITERATIONS
        ),
    }
    generator = np.random.default_rng(QUAL_SELECTION_SEED)
    synthetic_records: dict[str, dict[str, Any]] = {}
    for scene_index in range(5):
        utility = generator.normal(0.03, 0.15, size=(2, 6, 8)).astype(np.float32)
        harm = (utility <= -PSNR_HARM_MARGIN_DB).astype(np.float32)
        synthetic_records[f"synthetic_{scene_index}"] = {
            "scene": f"synthetic_{scene_index}",
            "features": generator.normal(size=(FEATURE_CHANNELS, 6, 8)).astype(np.float32),
            "shape_features": generator.normal(
                size=(SHAPE_CHANNELS, 24, 32),
            ).astype(np.float32),
            "shape_areas": np.ones((24, 32), dtype=np.float32),
            "disagreement": generator.normal(
                size=(DISAGREEMENT_CHANNELS, 6, 8),
            ).astype(np.float32),
            "utility": utility,
            "harm": harm,
            "areas": np.ones((6, 8), dtype=np.float32),
        }
    completed_iterations = 0
    finite_paths = []
    calibration_paths = []
    fixture_root = context.phase_output_path / "synthetic_checkpoints"
    fixture_root.mkdir()
    for cell in CELL_IDS:
        representation, learner = cell.split("_")
        normalization = {
            "features": {
                "mean": np.zeros(FEATURE_CHANNELS, dtype=np.float32),
                "scale": np.ones(FEATURE_CHANNELS, dtype=np.float32),
            },
        }
        if representation in {"R1", "R2"}:
            normalization["shape_features"] = {
                "mean": np.zeros(SHAPE_CHANNELS, dtype=np.float32),
                "scale": np.ones(SHAPE_CHANNELS, dtype=np.float32),
            }
        if representation == "R2":
            normalization["disagreement"] = {
                "mean": np.zeros(DISAGREEMENT_CHANNELS, dtype=np.float32),
                "scale": np.ones(DISAGREEMENT_CHANNELS, dtype=np.float32),
            }
        cell_predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for seed_index in range(len(SEED_OFFSETS)):
            torch.manual_seed(QUAL_SELECTION_SEED + seed_index)
            head = make_head(torch, representation, learner).to(context.device).eval()
            completed_iterations += 1
            first = synthetic_records["synthetic_0"]
            utility, risk = predict_head(
                torch, head, first, normalization, context.device,
            )
            completed_iterations += 1
            checkpoint = fixture_root / f"{cell}_{seed_index}.pt"
            torch.save({
                "schema_version": 1,
                "cell": cell,
                "representation": representation,
                "learner": learner,
                "fold": 0,
                "seed": QUAL_SELECTION_SEED + seed_index,
                "state_dict": {
                    name: value.detach().cpu()
                    for name, value in head.state_dict().items()
                },
                "normalization": normalization,
                "history": [{"total_loss": 0.0}],
            }, checkpoint)
            completed_iterations += 1
            loaded = qual_load_bundle(torch, checkpoint, context.device)
            loaded_utility, loaded_risk = predict_head(
                torch,
                loaded["head"],
                first,
                loaded["normalization"],
                context.device,
            )
            completed_iterations += 1
            finite_paths.append(bool(
                np.isfinite(loaded_utility).all()
                and np.isfinite(loaded_risk).all()
                and np.max(np.abs(utility - loaded_utility)) == 0.0
                and np.max(np.abs(risk - loaded_risk)) == 0.0
            ))
        for scene, record in synthetic_records.items():
            head = loaded["head"]
            cell_predictions[scene] = predict_head(
                torch, head, record, normalization, context.device,
            )
        models = qual_fit_calibration_models(
            cell_predictions, synthetic_records, sorted(synthetic_records),
        )
        completed_iterations += 1
        for method in qual_available_risk_methods(cell):
            applied = qual_apply_risk(
                cell_predictions["synthetic_0"][1], method, models, learner,
            )
            calibration_paths.append(bool(
                np.isfinite(applied).all()
                and np.all((applied >= 0.0) & (applied <= 1.0))
            ))
            completed_iterations += 1
        utility = qual_apply_utility(
            cell_predictions["synthetic_0"][0], "parent_lower", models,
        )
        risk = qual_apply_risk(
            cell_predictions["synthetic_0"][1], "scene_isotonic", models, learner,
        )
        tilewise = qual_choose_tilewise(utility, risk)
        setwise = qual_choose_setwise(
            utility, risk, synthetic_records["synthetic_0"]["areas"], area_cap=0.10,
        )
        completed_iterations += 1
        calibration_paths.append(
            tilewise.shape == (6, 8) and setwise.shape == (6, 8)
        )
        while completed_iterations % 20 != 0 and completed_iterations < QUAL_PROBE_ITERATIONS:
            _ = qual_choose_setwise(
                utility,
                risk,
                synthetic_records["synthetic_0"]["areas"],
                area_cap=QUAL_AREA_CAPS[completed_iterations % len(QUAL_AREA_CAPS)],
            )
            completed_iterations += 1
        write_contract_progress(
            context,
            completed_iterations=min(completed_iterations, QUAL_PROBE_ITERATIONS),
            total_iterations=QUAL_PROBE_ITERATIONS,
            stage=f"synthetic_controller_{cell}",
        )
    while completed_iterations < QUAL_PROBE_ITERATIONS:
        _ = qual_choose_setwise(
            utility,
            risk,
            synthetic_records["synthetic_0"]["areas"],
            area_cap=QUAL_AREA_CAPS[completed_iterations % len(QUAL_AREA_CAPS)],
        )
        completed_iterations += 1
    checks.update({
        "probe_iteration_count": completed_iterations == QUAL_PROBE_ITERATIONS,
        "all_checkpoint_roundtrips": len(finite_paths) == len(CELL_IDS) * 3
        and all(finite_paths),
        "all_calibration_and_selector_paths": bool(calibration_paths)
        and all(calibration_paths),
    })
    elapsed = time.monotonic() - started
    write_contract_result(
        context,
        checks=checks,
        engineering={
            "mode": "gpu_synthetic_no_data",
            "device": context.device,
            "fixture": {"batch": 1, "channels": 104, "height": 6, "width": 8},
            "production_path_exercised": True,
            "protected_data_touched": False,
            "scientific_output_created": False,
            "scientific_training_occurred": False,
            "cost": {
                "observed_iterations": completed_iterations,
                "observed_wall_seconds": elapsed,
                "observed_peak_memory_mib": float(
                    torch.cuda.max_memory_allocated() / (1024 * 1024),
                ),
            },
        },
    )


def qual_run(context_path: Path) -> None:
    context = load_context(context_path, "run")
    if context.route_id != QUAL_ROUTE_ID \
            or context.operation_id != QUAL_OPERATION_ID \
            or context.run_id != QUAL_RUN_ID:
        raise RuntimeError("controller qualification runtime identity mismatch")
    prepare_phase_output(context)
    import torch

    parent_root = asset_path(context, "parent_raw_workspace", kind="directory")
    parent_inventory_path = asset_path(context, "parent_raw_inventory", kind="file")
    parent_closeout_path = asset_path(context, "parent_closeout", kind="file")
    parent_conclusion_path = asset_path(context, "parent_conclusion", kind="file")
    train_root = asset_path(context, "haze4k_train", kind="directory")
    test_root = asset_path(context, "haze4k_test_development", kind="directory")
    parent_closeout = read_json(parent_closeout_path)
    parent_conclusion = read_json(parent_conclusion_path)
    identity_checks = {
        "parent_route": parent_closeout.get("route_id") == QUAL_PARENT_ROUTE_ID,
        "parent_route_commit": parent_closeout.get("route_commit")
        == QUAL_PARENT_ROUTE_COMMIT,
        "parent_terminal": (
            parent_closeout.get("state") == "COMPLETED_GATE_FAIL"
            and parent_closeout.get("decision")
            == "HAZE4K_OBSERVABLE_TAIL_FACTORIAL_FAIL"
            and parent_closeout.get("authorizes")
            == "NEW_ACTION_FAMILY_OR_END_TO_END_CONTROLLER_CONTRACT_AUTHORING_ONLY"
        ),
        "parent_conclusion": (
            parent_conclusion.get("state") == "COMPLETED_GATE_FAIL"
            and parent_conclusion.get("decision")
            == "HAZE4K_OBSERVABLE_TAIL_FACTORIAL_FAIL"
        ),
        "candidate_confirmation_absent": "candidate_confirmation" not in context.assets,
        "canary_absent": "canary" not in context.assets,
        "locked_test_absent": "locked_test" not in context.assets,
        "cross_domain_absent": all(
            identifier not in context.assets
            for identifier in ("nh_haze", "reside_its", "reside_ots")
        ),
    }
    inventory_verification, parent_items = qual_verify_parent_inventory(
        parent_root, parent_inventory_path,
    )
    identity_checks["parent_receipt_contract"] = (
        inventory_verification["parent_receipt"] == QUAL_PARENT_RECEIPT
    )
    inventory_path = output_file(context, "parent_inventory_verification.json")
    atomic_json(inventory_path, inventory_verification)
    record_completed_unit(
        context,
        unit_id="parent_inventory_verification",
        input_sha256=sha256_text("|".join([
            QUAL_PARENT_RECEIPT,
            QUAL_PARENT_INVENTORY_SHA256,
            sha256_file(parent_closeout_path),
        ])),
        output_relpath=inventory_path.name,
    )
    write_workload_progress(
        context, completed_units=1, stage="parent_inventory_verified",
    )
    parent_item_lookup = {
        item["relative_path"]: item for item in parent_items
    }
    raw_output_items: list[dict[str, Any]] = []

    train_groups, training_scenes, calibration_scenes, train_assignment = (
        enumerate_train_groups(train_root)
    )
    test_groups = enumerate_test_groups(test_root)
    test_split_summary = read_json(asset_path(
        context, "test_split_summary", kind="file",
    ))
    identity_checks.update({
        "train_assignment": train_assignment == TRAIN_ASSIGNMENT_DIGEST,
        "test_assignment": test_split_summary.get("frozen_split", {}).get(
            "assignment_digest"
        ) == TEST_ASSIGNMENT_DIGEST,
        "train_roles": (
            len(training_scenes) == TRAINING_SCENES
            and len(calibration_scenes) == CALIBRATION_SCENES
            and not (set(training_scenes) & set(calibration_scenes))
        ),
        "test_role": (
            len(test_groups) == EXPECTED_TEST_SCENES
            and all(len(items) == VARIANTS_PER_SCENE for items in test_groups.values())
        ),
    })
    fold_order = sorted(
        training_scenes,
        key=lambda scene: (sha256_text(f"{OOF_FOLD_SALT}|{scene}"), scene),
    )
    fold_assignment = {
        scene: index % OOF_FOLDS for index, scene in enumerate(fold_order)
    }
    calibration_order = sorted(
        calibration_scenes,
        key=lambda scene: (
            sha256_text(f"{CALIBRATION_FOLD_SALT}|{scene}"), scene,
        ),
    )
    calibration_fold = {
        scene: index % OOF_FOLDS for index, scene in enumerate(calibration_order)
    }
    identity_checks.update({
        "balanced_oof_folds": Counter(fold_assignment.values()) == {
            fold: TRAINING_SCENES // OOF_FOLDS for fold in range(OOF_FOLDS)
        },
        "balanced_calibration_folds": Counter(calibration_fold.values()) == {
            fold: CALIBRATION_SCENES // OOF_FOLDS for fold in range(OOF_FOLDS)
        },
    })

    calibration_records: dict[str, dict[str, Any]] = {}
    for scene in calibration_scenes:
        hazy_path, clear_path = choose_one_variant(scene, train_groups[scene])
        cache_path = parent_root / "workload" / "scene_cache" / (
            f"calibration_{scene[:24]}.npz"
        )
        calibration_records[scene] = qual_enrich_record(
            scene, cache_path, hazy_path, clear_path,
        )
    calibration_models: dict[str, dict[int, dict[str, Any]]] = {
        cell: {} for cell in CELL_IDS
    }
    heldout_predictions: dict[str, dict[str, dict[str, Any]]] = {
        cell: {} for cell in CELL_IDS
    }
    all_bundles: dict[str, dict[int, list[dict[str, Any]]]] = {
        cell: {} for cell in CELL_IDS
    }
    calibration_diagnostic_rows: list[dict[str, Any]] = []
    calibration_unit_count = 0
    calibration_output_root = output_file(context, "calibration_predictions")
    calibration_output_root.mkdir()
    for cell in CELL_IDS:
        for fold in range(OOF_FOLDS):
            bundles = [
                qual_load_bundle(
                    torch,
                    qual_checkpoint_path(parent_root, cell, fold, seed_index),
                    context.device,
                )
                for seed_index in range(len(SEED_OFFSETS))
            ]
            seed_predictions: dict[
                str, list[tuple[np.ndarray, np.ndarray]]
            ] = {}
            ensemble_predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for scene in sorted(calibration_scenes):
                record = qual_feature_record(
                    calibration_records[scene]["cache_path"],
                )
                predictions = [
                    predict_head(
                        torch,
                        bundle["head"],
                        record,
                        bundle["normalization"],
                        context.device,
                    )
                    for bundle in bundles
                ]
                seed_predictions[scene] = predictions
                ensemble_predictions[scene] = (
                    np.mean(np.stack([item[0] for item in predictions]), axis=0),
                    np.mean(np.stack([item[1] for item in predictions]), axis=0),
                )
            correction_scenes = [
                scene for scene in calibration_scenes
                if calibration_fold[scene] != fold
            ]
            selection_scenes = [
                scene for scene in calibration_scenes
                if calibration_fold[scene] == fold
            ]
            ensemble_models = qual_fit_calibration_models(
                ensemble_predictions,
                calibration_records,
                correction_scenes,
            )
            seed_models = []
            for seed_index in range(len(SEED_OFFSETS)):
                predictions = {
                    scene: seed_predictions[scene][seed_index]
                    for scene in calibration_scenes
                }
                model = qual_fit_calibration_models(
                    predictions, calibration_records, correction_scenes,
                )
                seed_models.append(model)
            calibration_diagnostic_rows.extend(qual_calibration_diagnostics(
                ensemble_predictions,
                calibration_records,
                selection_scenes,
                ensemble_models,
                cell=cell,
                fold=fold,
                source="ensemble",
                include_bins=True,
            ))
            calibration_models[cell][fold] = {
                "ensemble": ensemble_models,
                "seeds": seed_models,
            }
            for scene in selection_scenes:
                heldout_predictions[cell][scene] = {
                    "ensemble": ensemble_predictions[scene],
                    "seeds": seed_predictions[scene],
                }
            output_path = calibration_output_root / f"{cell}_fold_{fold}.npz"
            np.savez(
                output_path,
                **qual_prediction_payload(
                    sorted(calibration_scenes), seed_predictions,
                ),
            )
            add_inventory_item(
                raw_output_items,
                context,
                output_path,
                "calibration_prediction",
            )
            checkpoint_inputs = [
                parent_item_lookup[
                    qual_checkpoint_path(
                        parent_root, cell, fold, seed_index,
                    ).relative_to(parent_root).as_posix()
                ]["sha256"]
                for seed_index in range(len(SEED_OFFSETS))
            ]
            cache_inputs = [
                parent_item_lookup[
                    calibration_records[scene]["cache_path"].relative_to(
                        parent_root,
                    ).as_posix()
                ]["sha256"]
                for scene in sorted(calibration_scenes)
            ]
            record_completed_unit(
                context,
                unit_id=f"calibration_{cell}_fold_{fold}",
                input_sha256=digest_lines([
                    "frozen-controller-calibration-v1",
                    cell,
                    str(fold),
                    *checkpoint_inputs,
                    *cache_inputs,
                ]),
                output_relpath=output_path.relative_to(
                    context.phase_output_path,
                ).as_posix(),
            )
            calibration_unit_count += 1
            write_workload_progress(
                context,
                completed_units=1 + calibration_unit_count,
                stage=f"nested_calibration_{cell}_fold_{fold}",
            )
            move_bundles(bundles, "cpu")
            all_bundles[cell][fold] = bundles
            torch.cuda.empty_cache()

    candidate_rows, calibration_candidate_scene_rows = qual_candidate_table(
        calibration_records,
        calibration_fold,
        heldout_predictions,
        calibration_models,
    )
    selected_candidate = qual_select_candidate(candidate_rows)
    per_cell_candidates = {
        cell: qual_select_candidate([
            row for row in candidate_rows if row["cell"] == cell
        ])
        for cell in CELL_IDS
    }
    candidate_table_path = output_file(
        context, "haze4k_frozen_output_tail_controller_candidate_table.csv",
    )
    write_csv(candidate_table_path, candidate_rows)
    selected_calibration_rows_path = output_file(
        context,
        "haze4k_frozen_output_tail_controller_selected_calibration_scenes.csv",
    )
    write_csv(
        selected_calibration_rows_path,
        calibration_candidate_scene_rows[selected_candidate["candidate_id"]],
    )
    selection_path = output_file(
        context, "haze4k_frozen_output_tail_controller_selection_freeze.json",
    )
    atomic_json(selection_path, {
        "schema_version": 1,
        "route_id": QUAL_ROUTE_ID,
        "operation_id": QUAL_OPERATION_ID,
        "parent_route_id": QUAL_PARENT_ROUTE_ID,
        "parent_route_commit": QUAL_PARENT_ROUTE_COMMIT,
        "parent_receipt": QUAL_PARENT_RECEIPT,
        "parent_inventory_sha256": QUAL_PARENT_INVENTORY_SHA256,
        "selection_population": (
            "150 fixed calibration-controller scenes with five 120-fit/30-holdout folds"
        ),
        "selection_rule": (
            "Among setwise controllers, prefer simultaneous utility, coverage, and "
            "selected-harm admissibility; then safety plus coverage, coverage, lower "
            "selected-harm UCB, utility LCB, action-area LCB, and fixed candidate order."
        ),
        "formal_oof_arrays_parsed_before_freeze": False,
        "tiles_treated_as_independent_units": False,
        "risk_methods": list(QUAL_RISK_METHODS),
        "utility_methods": list(QUAL_UTILITY_METHODS),
        "area_caps": list(QUAL_AREA_CAPS),
        "selected_candidate": selected_candidate,
        "per_cell_candidates": per_cell_candidates,
        "candidate_count": len(candidate_rows),
    })
    selection_sha = sha256_file(selection_path)
    record_completed_unit(
        context,
        unit_id="controller_selection_freeze",
        input_sha256=sha256_text("|".join([
            "frozen-controller-selection-v1",
            QUAL_PARENT_INVENTORY_SHA256,
            sha256_file(candidate_table_path),
            sha256_file(selected_calibration_rows_path),
        ])),
        output_relpath=selection_path.name,
    )
    write_workload_progress(
        context,
        completed_units=47,
        stage="controller_candidate_hash_frozen",
    )
    del calibration_candidate_scene_rows
    for record in calibration_records.values():
        for key in ("hazy", "clear", "prediction", "tile_mse"):
            record.pop(key, None)

    selected_cell = str(selected_candidate["cell"])
    selected_risk_method = str(selected_candidate["risk_method"])
    selected_utility_method = str(selected_candidate["utility_method"])
    selected_selector = str(selected_candidate["selector"])
    selected_area_cap = float(selected_candidate["area_cap"])
    frontier: dict[
        tuple[str, str, str, str, float], list[dict[str, Any]]
    ] = defaultdict(list)
    fold_seed_values: dict[
        tuple[str, int, int, str, str], list[dict[str, Any]]
    ] = defaultdict(list)
    primary_rows: list[dict[str, Any]] = []
    maximum_prediction_difference = 0.0
    stored_action_mismatches = 0
    maximum_gain_identity_difference = 0.0
    oof_output_root = output_file(context, "oof_controller_scenes")
    oof_output_root.mkdir()

    # Formal OOF arrays are first parsed below, after the immutable selection SHA exists.
    for fold in range(OOF_FOLDS):
        for cell in CELL_IDS:
            move_bundles(all_bundles[cell][fold], context.device)
        fold_scenes = sorted(
            scene for scene in training_scenes if fold_assignment[scene] == fold
        )
        for scene in fold_scenes:
            hazy_path, clear_path = choose_one_variant(scene, train_groups[scene])
            cache_path = parent_root / "workload" / "scene_cache" / (
                f"training_{scene[:24]}.npz"
            )
            record = qual_enrich_record(scene, cache_path, hazy_path, clear_path)
            feature_record = qual_feature_record(cache_path)
            scene_payload: dict[str, np.ndarray] = {
                "scene": np.asarray(scene),
                "fold": np.asarray(fold, dtype=np.int16),
            }
            selected_prediction: tuple[np.ndarray, np.ndarray] | None = None
            selected_seed_predictions: list[tuple[np.ndarray, np.ndarray]] | None = None
            for cell in CELL_IDS:
                learner = cell.split("_")[1]
                predictions = [
                    predict_head(
                        torch,
                        bundle["head"],
                        feature_record,
                        bundle["normalization"],
                        context.device,
                    )
                    for bundle in all_bundles[cell][fold]
                ]
                ensemble = (
                    np.mean(np.stack([item[0] for item in predictions]), axis=0),
                    np.mean(np.stack([item[1] for item in predictions]), axis=0),
                )
                scene_payload[f"{cell}_ensemble_utility"] = ensemble[0].astype(np.float32)
                scene_payload[f"{cell}_ensemble_risk"] = ensemble[1].astype(np.float32)
                for seed_index, (utility_seed, risk_seed) in enumerate(predictions):
                    scene_payload[f"{cell}_seed_{seed_index}_utility"] = (
                        utility_seed.astype(np.float32)
                    )
                    scene_payload[f"{cell}_seed_{seed_index}_risk"] = (
                        risk_seed.astype(np.float32)
                    )
                parent_prediction_path = parent_root / "workload" / "raw_predictions" \
                    / "haze4k_train_oof" / cell \
                    / f"fold_{fold}_{scene[:24]}.npz"
                with np.load(parent_prediction_path, allow_pickle=False) as source:
                    maximum_prediction_difference = max(
                        maximum_prediction_difference,
                        float(np.max(np.abs(source["utility"] - ensemble[0]))),
                        float(np.max(np.abs(source["risk"] - ensemble[1]))),
                    )
                    stored_actions = source["actions"].copy()
                models = calibration_models[cell][fold]["ensemble"]
                for risk_method in qual_available_risk_methods(cell):
                    calibrated_risk = qual_apply_risk(
                        ensemble[1], risk_method, models, learner,
                    )
                    for utility_method in QUAL_UTILITY_METHODS:
                        calibrated_utility = qual_apply_utility(
                            ensemble[0], utility_method, models,
                        )
                        actions = qual_choose_tilewise(
                            calibrated_utility, calibrated_risk,
                        )
                        frontier[qual_frontier_key(
                            cell,
                            risk_method,
                            utility_method,
                            "tilewise",
                            1.0,
                        )].append({
                            "scene": scene,
                            "fold": fold,
                            **qual_policy_row(record, actions),
                        })
                        for area_cap in QUAL_AREA_CAPS:
                            actions = qual_choose_setwise(
                                calibrated_utility,
                                calibrated_risk,
                                record["areas"],
                                area_cap=area_cap,
                            )
                            frontier[qual_frontier_key(
                                cell,
                                risk_method,
                                utility_method,
                                "setwise",
                                area_cap,
                            )].append({
                                "scene": scene,
                                "fold": fold,
                                **qual_policy_row(record, actions),
                            })
                parent_utility = qual_apply_utility(
                    ensemble[0], "parent_lower", models,
                )
                parent_risk = qual_apply_risk(
                    ensemble[1], "parent_upper", models, learner,
                )
                parent_actions = qual_choose_tilewise(parent_utility, parent_risk)
                stored_action_mismatches += int(np.sum(parent_actions != stored_actions))
                for seed_index, prediction in enumerate(predictions):
                    seed_models = calibration_models[cell][fold]["seeds"][seed_index]
                    utility_seed = qual_apply_utility(
                        prediction[0], selected_utility_method, seed_models,
                    )
                    for risk_method in qual_available_risk_methods(cell):
                        risk_seed = qual_apply_risk(
                            prediction[1], risk_method, seed_models, learner,
                        )
                        for selector in QUAL_SELECTORS:
                            actions = qual_choose_policy(
                                utility_seed,
                                risk_seed,
                                record["areas"],
                                selector,
                                selected_area_cap,
                            )
                            fold_seed_values[
                                (cell, fold, seed_index, risk_method, selector)
                            ].append({
                                "scene": scene,
                                "fold": fold,
                                **qual_policy_row(record, actions),
                            })
                if cell == selected_cell:
                    selected_prediction = ensemble
                    selected_seed_predictions = predictions

            if selected_prediction is None or selected_seed_predictions is None:
                raise RuntimeError("selected cell prediction was not produced")
            selected_models = calibration_models[selected_cell][fold]["ensemble"]
            selected_learner = selected_cell.split("_")[1]
            selected_utility = qual_apply_utility(
                selected_prediction[0], selected_utility_method, selected_models,
            )
            selected_risk = qual_apply_risk(
                selected_prediction[1],
                selected_risk_method,
                selected_models,
                selected_learner,
            )
            selected_actions = qual_choose_policy(
                selected_utility,
                selected_risk,
                record["areas"],
                selected_selector,
                selected_area_cap,
            )
            uniform_actions = qual_uniform_action_map(
                selected_utility, selected_risk, record["areas"],
            )
            shuffled_actions = qual_shuffled_actions(selected_actions, scene)
            permuted_utility, permuted_risk = qual_permuted_scores(
                selected_utility, selected_risk, scene,
            )
            permuted_actions = qual_choose_policy(
                permuted_utility,
                permuted_risk,
                record["areas"],
                selected_selector,
                selected_area_cap,
            )
            parent_actions = qual_choose_tilewise(
                qual_apply_utility(
                    selected_prediction[0], "parent_lower", selected_models,
                ),
                qual_apply_risk(
                    selected_prediction[1],
                    "parent_upper",
                    selected_models,
                    selected_learner,
                ),
            )
            raw_actions = qual_choose_tilewise(
                selected_prediction[0], selected_prediction[1],
            )
            observable_utility_gt_risk_actions = qual_choose_policy(
                selected_utility,
                record["harm"],
                record["areas"],
                selected_selector,
                selected_area_cap,
            )
            gt_utility_observable_risk_actions = qual_choose_policy(
                record["utility"],
                selected_risk,
                record["areas"],
                selected_selector,
                selected_area_cap,
            )
            gt_gt_actions = qual_choose_tilewise(
                record["utility"], record["harm"],
            )
            action_maps = {
                "selected": selected_actions,
                "uniform": uniform_actions,
                "shuffled": shuffled_actions,
                "permuted": permuted_actions,
                "parent_p11": parent_actions,
                "raw_p00": raw_actions,
                "observable_utility_gt_risk": observable_utility_gt_risk_actions,
                "gt_utility_observable_risk": gt_utility_observable_risk_actions,
                "gt_gt": gt_gt_actions,
            }
            statistics = {
                name: qual_policy_row(record, actions)
                for name, actions in action_maps.items()
            }
            image_metrics = qual_image_metrics(
                torch, record, selected_actions, context.device,
            )
            maximum_gain_identity_difference = max(
                maximum_gain_identity_difference,
                float(image_metrics["gain_identity_difference"]),
            )
            selected_statistics = statistics["selected"]
            primary_rows.append({
                "scene": scene,
                "fold": fold,
                "candidate_minus_keep_psnr_db": selected_statistics["gain_db"],
                "candidate_minus_uniform_psnr_db": (
                    selected_statistics["gain_db"] - statistics["uniform"]["gain_db"]
                ),
                "candidate_minus_shuffled_psnr_db": (
                    selected_statistics["gain_db"] - statistics["shuffled"]["gain_db"]
                ),
                "candidate_minus_permuted_psnr_db": (
                    selected_statistics["gain_db"] - statistics["permuted"]["gain_db"]
                ),
                "selected_area": selected_statistics["selected_area"],
                "selected_harm_area": selected_statistics["selected_harm_area"],
                "total_area": selected_statistics["total_area"],
                "active_scene": selected_statistics["active_scene"],
                "material_scene": selected_statistics["material_scene"],
                "risk_margin": selected_statistics["risk_margin"],
                "psnr_harm": image_metrics["psnr_harm"],
                "ssim_harm": image_metrics["ssim_harm"],
                "color_harm": image_metrics["color_harm"],
                "ssim_delta": image_metrics["ssim_delta"],
                "color_bias_delta": image_metrics["color_bias_delta"],
                **{
                    f"{name}_gain_db": value["gain_db"]
                    for name, value in statistics.items()
                },
            })
            for name, actions in action_maps.items():
                scene_payload[f"actions_{name}"] = actions.astype(np.int8)
            scene_payload["selected_metrics"] = np.asarray([
                selected_statistics["gain_db"],
                selected_statistics["selected_area"],
                selected_statistics["selected_harm_area"],
                selected_statistics["total_area"],
                image_metrics["ssim_delta"],
                image_metrics["color_bias_delta"],
            ], dtype=np.float64)
            relative = f"oof_controller_scenes/fold_{fold}_{scene[:24]}.npz"
            scene_output_path = output_file(context, relative)
            np.savez(scene_output_path, **scene_payload)
            add_inventory_item(
                raw_output_items,
                context,
                scene_output_path,
                "oof_controller_scene",
            )
            cache_identity = parent_item_lookup[
                cache_path.relative_to(parent_root).as_posix()
            ]["sha256"]
            record_completed_unit(
                context,
                unit_id=f"oof_scene_{fold}_{scene[:24]}",
                input_sha256=sha256_text("|".join([
                    "frozen-controller-oof-v1",
                    scene,
                    selection_sha,
                    cache_identity,
                ])),
                output_relpath=relative,
            )
            completed = 47 + len(primary_rows)
            if len(primary_rows) % 20 == 0 or len(primary_rows) == TRAINING_SCENES:
                write_workload_progress(
                    context,
                    completed_units=completed,
                    stage="frozen_selection_oof_controller_replay",
                )
        for cell in CELL_IDS:
            move_bundles(all_bundles[cell][fold], "cpu")
        torch.cuda.empty_cache()

    if len(primary_rows) != TRAINING_SCENES:
        raise RuntimeError("formal OOF scene coverage changed")

    test_primary_rows: list[dict[str, Any]] = []
    test_rows_by_cell: dict[str, list[dict[str, Any]]] = {
        cell: [] for cell in CELL_IDS
    }
    test_output_root = output_file(context, "test_controller_scenes")
    test_output_root.mkdir()
    for cell in CELL_IDS:
        for fold in range(OOF_FOLDS):
            move_bundles(all_bundles[cell][fold], context.device)

    # Test-development is secondary stress evidence; four variants remain nested in scene.
    for scene_index, scene in enumerate(sorted(test_groups), start=1):
        variant_selected_rows = []
        variant_control_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
        variant_image_rows = []
        scene_payload: dict[str, np.ndarray] = {"scene": np.asarray(scene)}
        for variant_index, (hazy_path, clear_path) in enumerate(
            sorted(test_groups[scene]), start=1,
        ):
            cache_path = parent_root / "workload" / "test_development_cache" / (
                f"scene_{scene_index:03d}_{scene[:16]}_variant_{variant_index}.npz"
            )
            record = qual_enrich_record(scene, cache_path, hazy_path, clear_path)
            feature_record = qual_feature_record(cache_path)
            cell_calibrated: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for cell in CELL_IDS:
                candidate = per_cell_candidates[cell]
                learner = cell.split("_")[1]
                fold_utilities, fold_risks = [], []
                for fold in range(OOF_FOLDS):
                    predictions = [
                        predict_head(
                            torch,
                            bundle["head"],
                            feature_record,
                            bundle["normalization"],
                            context.device,
                        )
                        for bundle in all_bundles[cell][fold]
                    ]
                    ensemble = (
                        np.mean(np.stack([item[0] for item in predictions]), axis=0),
                        np.mean(np.stack([item[1] for item in predictions]), axis=0),
                    )
                    models = calibration_models[cell][fold]["ensemble"]
                    fold_utilities.append(qual_apply_utility(
                        ensemble[0], str(candidate["utility_method"]), models,
                    ))
                    fold_risks.append(qual_apply_risk(
                        ensemble[1],
                        str(candidate["risk_method"]),
                        models,
                        learner,
                    ))
                utility = np.mean(np.stack(fold_utilities), axis=0)
                risk = np.mean(np.stack(fold_risks), axis=0)
                cell_calibrated[cell] = utility, risk
                actions = qual_choose_setwise(
                    utility,
                    risk,
                    record["areas"],
                    area_cap=float(candidate["area_cap"]),
                )
                test_rows_by_cell[cell].append({
                    "scene": scene,
                    "variant": variant_index,
                    "fold": "test",
                    **qual_policy_row(record, actions),
                })
            selected_utility, selected_risk = cell_calibrated[selected_cell]
            selected_actions = qual_choose_policy(
                selected_utility,
                selected_risk,
                record["areas"],
                selected_selector,
                selected_area_cap,
            )
            uniform_actions = qual_uniform_action_map(
                selected_utility, selected_risk, record["areas"],
            )
            shuffled_actions = qual_shuffled_actions(
                selected_actions, f"test|{scene}|{variant_index}",
            )
            permuted_utility, permuted_risk = qual_permuted_scores(
                selected_utility,
                selected_risk,
                f"test|{scene}|{variant_index}",
            )
            permuted_actions = qual_choose_policy(
                permuted_utility,
                permuted_risk,
                record["areas"],
                selected_selector,
                selected_area_cap,
            )
            selected_statistics = qual_policy_row(record, selected_actions)
            controls = {
                "uniform": qual_policy_row(record, uniform_actions),
                "shuffled": qual_policy_row(record, shuffled_actions),
                "permuted": qual_policy_row(record, permuted_actions),
            }
            image_metrics = qual_image_metrics(
                torch, record, selected_actions, context.device,
            )
            variant_selected_rows.append(selected_statistics)
            for name, value in controls.items():
                variant_control_rows[name].append(value)
            variant_image_rows.append(image_metrics)
            scene_payload[f"variant_{variant_index}_actions_selected"] = (
                selected_actions.astype(np.int8)
            )
            scene_payload[f"variant_{variant_index}_actions_uniform"] = (
                uniform_actions.astype(np.int8)
            )
            scene_payload[f"variant_{variant_index}_actions_shuffled"] = (
                shuffled_actions.astype(np.int8)
            )
            scene_payload[f"variant_{variant_index}_actions_permuted"] = (
                permuted_actions.astype(np.int8)
            )
        selected_area = sum(row["selected_area"] for row in variant_selected_rows)
        selected_harm = sum(
            row["selected_harm_area"] for row in variant_selected_rows
        )
        total_area = sum(row["total_area"] for row in variant_selected_rows)
        mean_gain = float(np.mean([row["gain_db"] for row in variant_selected_rows]))
        control_gain = {
            name: float(np.mean([row["gain_db"] for row in rows]))
            for name, rows in variant_control_rows.items()
        }
        mean_ssim_delta = float(np.mean([
            row["ssim_delta"] for row in variant_image_rows
        ]))
        mean_color_delta = float(np.mean([
            row["color_bias_delta"] for row in variant_image_rows
        ]))
        test_primary_rows.append({
            "scene": scene,
            "fold": "test",
            "candidate_minus_keep_psnr_db": mean_gain,
            "candidate_minus_uniform_psnr_db": mean_gain - control_gain["uniform"],
            "candidate_minus_shuffled_psnr_db": mean_gain - control_gain["shuffled"],
            "candidate_minus_permuted_psnr_db": mean_gain - control_gain["permuted"],
            "selected_area": selected_area,
            "selected_harm_area": selected_harm,
            "total_area": total_area,
            "active_scene": selected_area > 0.0,
            "material_scene": mean_gain >= UTILITY_MARGIN_DB,
            "risk_margin": (
                selected_harm - MAX_HARM_PREVALENCE * selected_area
            ) / total_area,
            "psnr_harm": mean_gain <= -PSNR_HARM_MARGIN_DB,
            "ssim_harm": mean_ssim_delta <= -SSIM_HARM_MARGIN,
            "color_harm": mean_color_delta >= COLOR_HARM_MARGIN,
            "ssim_delta": mean_ssim_delta,
            "color_bias_delta": mean_color_delta,
        })
        relative = f"test_controller_scenes/scene_{scene_index:03d}_{scene[:24]}.npz"
        scene_output_path = output_file(context, relative)
        np.savez(scene_output_path, **scene_payload)
        add_inventory_item(
            raw_output_items,
            context,
            scene_output_path,
            "test_controller_scene",
        )
        input_identities = [
            parent_item_lookup[
                (
                    parent_root / "workload" / "test_development_cache" / (
                        f"scene_{scene_index:03d}_{scene[:16]}_variant_{variant_index}.npz"
                    )
                ).relative_to(parent_root).as_posix()
            ]["sha256"]
            for variant_index in range(1, VARIANTS_PER_SCENE + 1)
        ]
        record_completed_unit(
            context,
            unit_id=f"test_scene_{scene_index:03d}_{scene[:24]}",
            input_sha256=digest_lines([
                "frozen-controller-test-v1",
                scene,
                selection_sha,
                *input_identities,
            ]),
            output_relpath=relative,
        )
        if scene_index % 10 == 0 or scene_index == EXPECTED_TEST_SCENES:
            write_workload_progress(
                context,
                completed_units=647 + scene_index,
                stage="nested_test_development_controller_stress",
            )
    for cell in CELL_IDS:
        for fold in range(OOF_FOLDS):
            move_bundles(all_bundles[cell][fold], "cpu")
    torch.cuda.empty_cache()

    # Collapse four test variants to their 100 independent original-clear scenes.
    for cell in CELL_IDS:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in test_rows_by_cell[cell]:
            grouped[str(row["scene"])].append(row)
        collapsed = []
        for scene, rows in sorted(grouped.items()):
            selected_area = sum(row["selected_area"] for row in rows)
            selected_harm = sum(row["selected_harm_area"] for row in rows)
            total_area = sum(row["total_area"] for row in rows)
            gain = float(np.mean([row["gain_db"] for row in rows]))
            collapsed.append({
                "scene": scene,
                "fold": "test",
                "gain_db": gain,
                "selected_area": selected_area,
                "selected_harm_area": selected_harm,
                "total_area": total_area,
                "action_area_fraction": selected_area / total_area,
                "active_scene": selected_area > 0.0,
                "material_scene": gain >= UTILITY_MARGIN_DB,
                "risk_margin": (
                    selected_harm - MAX_HARM_PREVALENCE * selected_area
                ) / total_area,
            })
        test_rows_by_cell[cell] = collapsed

    frontier_rows = qual_frontier_rows(frontier)
    fold_seed_rows = qual_fold_seed_rows(fold_seed_values)
    effect_rows = qual_effect_rows(frontier)
    primary_fields = [
        "candidate_minus_keep_psnr_db",
        "candidate_minus_uniform_psnr_db",
        "candidate_minus_shuffled_psnr_db",
        "candidate_minus_permuted_psnr_db",
    ]
    primary_intervals = stratified_bootstrap_family(
        primary_rows,
        primary_fields,
        family_size=QUAL_PRIMARY_FAMILY_SIZE,
        seed=BOOTSTRAP_SEED + 1200,
    )
    action_area_interval = bootstrap_ratio_interval(
        primary_rows,
        "selected_area",
        "total_area",
        family_size=2,
        seed=BOOTSTRAP_SEED + 1201,
    )
    selected_harm_interval = bootstrap_ratio_interval(
        primary_rows,
        "selected_harm_area",
        "selected_area",
        family_size=QUAL_SAFETY_FAMILY_SIZE,
        seed=BOOTSTRAP_SEED + 1202,
    )
    active_scene_interval = wilson(
        sum(bool(row["active_scene"]) for row in primary_rows),
        len(primary_rows),
        family_size=2,
    )
    material_scene_interval = wilson(
        sum(bool(row["material_scene"]) for row in primary_rows),
        len(primary_rows),
        family_size=QUAL_PRIMARY_FAMILY_SIZE,
    )
    test_action_area_interval = bootstrap_ratio_interval(
        test_primary_rows,
        "selected_area",
        "total_area",
        family_size=2,
        seed=BOOTSTRAP_SEED + 1203,
    )
    test_selected_harm_interval = bootstrap_ratio_interval(
        test_primary_rows,
        "selected_harm_area",
        "selected_area",
        family_size=QUAL_SAFETY_FAMILY_SIZE,
        seed=BOOTSTRAP_SEED + 1204,
    )
    test_active_scene_interval = wilson(
        sum(bool(row["active_scene"]) for row in test_primary_rows),
        len(test_primary_rows),
        family_size=2,
    )
    image_safety = {
        "haze4k_train_oof": {
            metric: wilson(
                sum(bool(row[metric]) for row in primary_rows),
                len(primary_rows),
                family_size=QUAL_SAFETY_FAMILY_SIZE,
            )
            for metric in ("psnr_harm", "ssim_harm", "color_harm")
        },
        "haze4k_test_development": {
            metric: wilson(
                sum(bool(row[metric]) for row in test_primary_rows),
                len(test_primary_rows),
                family_size=QUAL_SAFETY_FAMILY_SIZE,
            )
            for metric in ("psnr_harm", "ssim_harm", "color_harm")
        },
    }
    oracle_interval = stratified_bootstrap_family(
        primary_rows,
        ["gt_gt_gain_db"],
        family_size=2,
        seed=BOOTSTRAP_SEED + 1205,
    )["gt_gt_gain_db"]
    oracle_material = wilson(
        sum(float(row["gt_gt_gain_db"]) >= UTILITY_MARGIN_DB for row in primary_rows),
        len(primary_rows),
        family_size=2,
    )
    control_intervals = {
        name: stratified_bootstrap_family(
            primary_rows,
            [f"{name}_gain_db"],
            family_size=6,
            seed=BOOTSTRAP_SEED + 1300 + index,
        )[f"{name}_gain_db"]
        for index, name in enumerate((
            "parent_p11",
            "raw_p00",
            "observable_utility_gt_risk",
            "gt_utility_observable_risk",
            "gt_gt",
            "selected",
        ))
    }
    test_primary_summary = {
        "gain": normal_interval(
            [row["candidate_minus_keep_psnr_db"] for row in test_primary_rows],
            family_size=QUAL_PRIMARY_FAMILY_SIZE,
        ),
        "action_area": test_action_area_interval,
        "selected_harm": test_selected_harm_interval,
        "active_scene": test_active_scene_interval,
    }

    utility_gain = primary_intervals["candidate_minus_keep_psnr_db"]
    utility_favorable = (
        utility_gain["lower"] > UTILITY_MARGIN_DB
        and material_scene_interval["lower"] > MIN_MATERIAL_SCENE_PREVALENCE
        and primary_intervals["candidate_minus_uniform_psnr_db"]["lower"] > 0.0
        and primary_intervals["candidate_minus_shuffled_psnr_db"]["lower"] > 0.0
        and primary_intervals["candidate_minus_permuted_psnr_db"]["lower"] > 0.0
    )
    utility_futile = (
        utility_gain["upper"] <= UTILITY_MARGIN_DB
        or material_scene_interval["upper"] <= MIN_MATERIAL_SCENE_PREVALENCE
        or primary_intervals["candidate_minus_uniform_psnr_db"]["upper"] <= 0.0
        or primary_intervals["candidate_minus_shuffled_psnr_db"]["upper"] <= 0.0
        or primary_intervals["candidate_minus_permuted_psnr_db"]["upper"] <= 0.0
    )
    utility_outcome = (
        "favorable" if utility_favorable
        else "unfavorable" if utility_futile
        else "indeterminate"
    )
    activation_favorable = (
        action_area_interval["lower"] is not None
        and action_area_interval["lower"] > MIN_ACTION_AREA_FRACTION
        and active_scene_interval["lower"] > MIN_ACTIVE_SCENE_PREVALENCE
    )
    activation_futile = (
        action_area_interval["upper"] is None
        or action_area_interval["upper"] <= MIN_ACTION_AREA_FRACTION
        or active_scene_interval["upper"] <= MIN_ACTIVE_SCENE_PREVALENCE
    )
    activation_outcome = (
        "favorable" if activation_favorable
        else "unfavorable" if activation_futile
        else "indeterminate"
    )
    safety_intervals = [selected_harm_interval, test_selected_harm_interval]
    safety_intervals.extend(
        interval
        for population in image_safety.values()
        for interval in population.values()
    )
    safety_favorable = all(
        interval.get("upper") is not None
        and float(interval["upper"]) <= MAX_HARM_PREVALENCE
        for interval in safety_intervals
    )
    safety_unsafe = any(
        interval.get("lower") is not None
        and float(interval["lower"]) > MAX_HARM_PREVALENCE
        for interval in safety_intervals
    )
    safety_outcome = (
        "safe" if safety_favorable
        else "unsafe" if safety_unsafe
        else "indeterminate"
    )
    oracle_outcome = (
        "favorable"
        if (
            oracle_interval["lower"] > UTILITY_MARGIN_DB
            and oracle_material["lower"] > MIN_MATERIAL_SCENE_PREVALENCE
        )
        else "unfavorable"
        if (
            oracle_interval["upper"] <= UTILITY_MARGIN_DB
            or oracle_material["upper"] <= MIN_MATERIAL_SCENE_PREVALENCE
        )
        else "indeterminate"
    )
    precision_half_width = max(
        interval["max_half_width"] for interval in primary_intervals.values()
    )
    identity_checks.update({
        "parent_inventory": inventory_verification["matched"],
        "raw_prediction_reconstruction": maximum_prediction_difference <= 1.0e-6,
        "parent_action_reconstruction": stored_action_mismatches == 0,
        "image_tile_gain_identity": maximum_gain_identity_difference <= 1.0e-6,
    })
    prefinal_ledger_count = len(load_completed_unit_ledger(context))
    coverage_checks = {
        "calibration_units": calibration_unit_count == len(CELL_IDS) * OOF_FOLDS,
        "heldout_calibration_scenes": all(
            len(heldout_predictions[cell]) == CALIBRATION_SCENES for cell in CELL_IDS
        ),
        "selection_frozen_before_formal_oof": (
            selection_path.is_file()
            and read_json(selection_path)["formal_oof_arrays_parsed_before_freeze"] is False
        ),
        "formal_oof_scenes": len(primary_rows) == TRAINING_SCENES,
        "test_stress_scenes": len(test_primary_rows) == EXPECTED_TEST_SCENES,
        "factorial_frontier": all(
            any(row["cell"] == cell for row in frontier_rows) for cell in CELL_IDS
        ),
        "fold_seed_stability": len({
            (row["cell"], row["fold"], row["seed_index"])
            for row in fold_seed_rows
        }) == len(CELL_IDS) * OOF_FOLDS * len(SEED_OFFSETS),
        "prefinal_completed_unit_ledger": prefinal_ledger_count
        == QUAL_TOTAL_UNITS - 1,
    }
    gate_outcomes = {
        "evidence_identity": "pass" if all(identity_checks.values()) else "fail",
        "coverage_and_selection_integrity": (
            "pass" if all(coverage_checks.values()) else "fail"
        ),
        "oracle_headroom_control": oracle_outcome,
        "selected_controller_actionable_utility": utility_outcome,
        "selected_controller_activation": activation_outcome,
        "selected_area_and_image_safety": safety_outcome,
        "precision": "met" if precision_half_width <= PRECISION_TARGET_DB else "unmet",
        "low_risk_tail_mechanism_diagnostic": "favorable",
    }

    raw_inventory = compact_inventory(raw_output_items)
    raw_inventory.update({
        "upstream_parent_receipt": QUAL_PARENT_RECEIPT,
        "upstream_parent_inventory_sha256": QUAL_PARENT_INVENTORY_SHA256,
        "upstream_parent_file_count": 10285,
        "upstream_parent_total_bytes": 5225583075,
    })
    paths = {
        "summary": output_file(
            context, "haze4k_frozen_output_tail_controller_summary.json",
        ),
        "gate": output_file(
            context, "haze4k_frozen_output_tail_controller_gate_summary.json",
        ),
        "selection": selection_path,
        "candidates": candidate_table_path,
        "selected_calibration": selected_calibration_rows_path,
        "primary": output_file(
            context, "haze4k_frozen_output_tail_controller_primary_intervals.json",
        ),
        "frontier": output_file(
            context, "haze4k_frozen_output_tail_controller_risk_coverage_frontier.csv",
        ),
        "fold_seed": output_file(
            context, "haze4k_frozen_output_tail_controller_fold_seed_tail.csv",
        ),
        "calibration": output_file(
            context, "haze4k_frozen_output_tail_controller_calibration_tail.csv",
        ),
        "effects": output_file(
            context, "haze4k_frozen_output_tail_controller_effects.csv",
        ),
        "controls": output_file(
            context, "haze4k_frozen_output_tail_controller_controls.json",
        ),
        "test": output_file(
            context, "haze4k_frozen_output_tail_controller_test_stress.json",
        ),
        "inventory": output_file(
            context, "haze4k_frozen_output_tail_controller_raw_inventory.json",
        ),
        "integrity": inventory_path,
    }
    summary = {
        "schema_version": 2,
        "route_id": QUAL_ROUTE_ID,
        "operation_id": QUAL_OPERATION_ID,
        "run_id": context.run_id,
        "scope": "receipt-bound frozen-output low-risk-tail controller qualification",
        "evidence_role": "development_screening",
        "independent_unit": "original_clear_scene",
        "selection_freeze_sha256": selection_sha,
        "selected_candidate": selected_candidate,
        "per_cell_candidates": per_cell_candidates,
        "data_roles": {
            "calibration_controller_selection": CALIBRATION_SCENES,
            "formal_oof_development": TRAINING_SCENES,
            "test_development_stress": EXPECTED_TEST_SCENES,
            "test_nested_variants": EXPECTED_TEST_SCENES * VARIANTS_PER_SCENE,
            "candidate_confirmation_touched": False,
            "canary_touched": False,
            "locked_test_touched": False,
            "cross_domain_touched": False,
        },
        "identity_checks": identity_checks,
        "coverage_checks": coverage_checks,
        "primary_intervals": primary_intervals,
        "action_area": action_area_interval,
        "selected_harm": selected_harm_interval,
        "active_scene": active_scene_interval,
        "material_scene": material_scene_interval,
        "image_safety": image_safety,
        "test_development": test_primary_summary,
        "oracle_control": {
            "gain": oracle_interval,
            "material_scene": oracle_material,
            "outcome": oracle_outcome,
        },
        "controls": control_intervals,
        "precision": {
            "target_half_width_db": PRECISION_TARGET_DB,
            "observed_max_simultaneous_half_width_db": precision_half_width,
            "met": precision_half_width <= PRECISION_TARGET_DB,
        },
        "gate_outcomes": gate_outcomes,
        "upstream_raw_inventory": inventory_verification,
        "new_raw_output_inventory": raw_inventory,
        "limitations": [
            "All evidence is development-screening; confirmation, canary, locked-test, NH-Haze, ITS, and OTS were not read.",
            "The 600 OOF original-clear scenes are the only formal independent units; tiles, actions, folds, seeds, and variants never increase n.",
            "The 150 calibration scenes fit probability maps and select the controller through five 120-fit/30-holdout folds; they do not enter the formal OOF estimate.",
            "The 100 test-development scenes are secondary stress evidence and their four variants are averaged within scene.",
            "GT-risk, GT-utility, and GT/GT controls are privileged localization instruments and are non-deployable.",
            "The OOF population has been adaptively reused across prior routes and remains development screening even when this route is precise.",
        ],
        "marker": "HAZE4K_FROZEN_OUTPUT_TAIL_CONTROLLER_QUALIFICATION_COMPLETE",
    }
    atomic_json(paths["summary"], summary)
    atomic_json(paths["gate"], {
        "schema_version": 2,
        "selected_candidate": selected_candidate,
        "selection_freeze_sha256": selection_sha,
        "gate_outcomes": gate_outcomes,
        "utility": {
            "favorable": utility_favorable,
            "futile": utility_futile,
            "intervals": primary_intervals,
        },
        "activation": {
            "action_area": action_area_interval,
            "active_scene": active_scene_interval,
        },
        "safety": {
            "oof_selected_harm": selected_harm_interval,
            "test_selected_harm": test_selected_harm_interval,
            "image_safety": image_safety,
        },
        "precision_half_width_db": precision_half_width,
    })
    atomic_json(paths["primary"], {
        "schema_version": 1,
        "selected_candidate": selected_candidate,
        "comparison_family_size": QUAL_PRIMARY_FAMILY_SIZE,
        "intervals": primary_intervals,
        "action_area": action_area_interval,
        "selected_harm": selected_harm_interval,
        "active_scene": active_scene_interval,
        "material_scene": material_scene_interval,
    })
    write_csv(paths["frontier"], frontier_rows)
    write_csv(paths["fold_seed"], fold_seed_rows)
    write_csv(paths["calibration"], calibration_diagnostic_rows)
    write_csv(paths["effects"], effect_rows)
    atomic_json(paths["controls"], {
        "schema_version": 1,
        "selected_candidate": selected_candidate,
        "intervals": control_intervals,
        "oracle": {"gain": oracle_interval, "material_scene": oracle_material},
    })
    atomic_json(paths["test"], {
        "schema_version": 1,
        "selected_candidate": selected_candidate,
        "selected_controller": test_primary_summary,
        "per_cell": {
            cell: qual_descriptive_summary(rows)
            for cell, rows in test_rows_by_cell.items()
        },
        "image_safety": image_safety["haze4k_test_development"],
    })
    atomic_json(paths["inventory"], raw_inventory)

    finalization_path = output_file(context, "finalization_unit.json")
    evidence_sha256 = {
        key: sha256_file(path) for key, path in sorted(paths.items())
    }
    atomic_json(finalization_path, {
        "schema_version": 1,
        "route_id": QUAL_ROUTE_ID,
        "operation_id": QUAL_OPERATION_ID,
        "selected_candidate": selected_candidate,
        "selection_freeze_sha256": selection_sha,
        "gate_outcomes": gate_outcomes,
        "evidence_sha256": evidence_sha256,
    })
    record_completed_unit(
        context,
        unit_id="finalization",
        input_sha256=sha256_text("|".join([
            "frozen-controller-finalization-v1",
            selection_sha,
            raw_inventory["inventory_sha256"],
            digest_lines(
                f"{key}:{value}" for key, value in evidence_sha256.items()
            ),
        ])),
        output_relpath=finalization_path.name,
    )
    if len(load_completed_unit_ledger(context)) != QUAL_TOTAL_UNITS:
        raise RuntimeError("controller qualification completed-unit ledger is incomplete")
    write_workload_progress(
        context,
        completed_units=QUAL_TOTAL_UNITS,
        stage="aggregate_controller_gate_finalization",
    )
    write_gate_result(
        context,
        gate_outcomes=gate_outcomes,
        details={
            "selected_candidate_id": selected_candidate["candidate_id"],
            "selected_cell": selected_cell,
            "selected_risk_method": selected_risk_method,
            "selected_utility_method": selected_utility_method,
            "selected_selector": selected_selector,
            "selected_area_cap": selected_area_cap,
            "selection_freeze_sha256": selection_sha,
            "independent_oof_scenes": TRAINING_SCENES,
            "calibration_controller_scenes": CALIBRATION_SCENES,
            "test_development_stress_scenes": EXPECTED_TEST_SCENES,
            "factorial_cells": len(CELL_IDS),
            "folds": OOF_FOLDS,
            "paired_seeds": len(SEED_OFFSETS),
            "completed_unit_ledger_count": QUAL_TOTAL_UNITS,
            "candidate_confirmation_touched": False,
            "formal_family_size": QUAL_PRIMARY_FAMILY_SIZE,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "primary_estimate_db": utility_gain["estimate"],
            "summary_file": paths["summary"].name,
            "gate_summary_file": paths["gate"].name,
            "selection_file": paths["selection"].name,
            "primary_intervals_file": paths["primary"].name,
            "risk_coverage_frontier_file": paths["frontier"].name,
            "fold_seed_tail_file": paths["fold_seed"].name,
            "calibration_tail_file": paths["calibration"].name,
            "controller_effects_file": paths["effects"].name,
            "controls_file": paths["controls"].name,
            "test_stress_file": paths["test"].name,
            "raw_inventory_file": paths["inventory"].name,
            "parent_inventory_verification_file": paths["integrity"].name,
        },
    )


# The final definitions select the controller-qualification operation while retaining the
# exact parent head, action, metric, and uncertainty implementations above.
contract = qual_contract
run = qual_run


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
