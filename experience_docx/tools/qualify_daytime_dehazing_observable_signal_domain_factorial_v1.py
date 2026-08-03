#!/usr/bin/env python3
"""Qualify observable signal classes for bounded ConvIR-B feature modulation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

from route_program_api import (
    asset_path,
    atomic_json,
    build_gate_review_fact,
    load_completed_unit_ledger,
    load_context,
    output_file,
    prepare_phase_output,
    record_completed_unit,
    write_contract_progress,
    write_contract_result,
    write_gate_result,
    write_review_facts,
    write_workload_progress,
)


ROUTE_ID = "daytime-dehazing-observable-signal-domain-factorial-v1"
OPERATION_ID = "DAYTIME_DEHAZING_OBSERVABLE_SIGNAL_DOMAIN_FACTORIAL_QUALIFY"
SUMMARY_NAME = "daytime_dehazing_observable_signal_domain_factorial_v1_summary.json"
GATE_NAME = "daytime_dehazing_observable_signal_domain_factorial_v1_gate_summary.json"
FACTS_NAME = "daytime_dehazing_observable_signal_domain_factorial_v1_review_facts.json"

ANCHOR_COMMIT = "3b4da35440c8c26a7d1bcaf1daf342e11d9a3898"
CHECKPOINT_SHA256 = "6f42037d57a4e3de3a10ac0ab909d66a3415864a19433c29204a975f4efa4088"
MODEL_SOURCE_SHA256 = "9681defa95c6602d0e0abae05e635771bd65d48a18555198ff00d573242a0005"
MODEL_LAYERS_SHA256 = "ac8a05bd626d9adda16308dedb9466f36d7ff44cfb666f64e7e14ddf8cdf43a4"
S0_LEDGER_SHA256 = "4cff8e7aecea5d8e19165ac4e725f69746342521115a3ab1f03ea1474f280960"
ROLE_LEDGER_SHA256 = "2f387c0992c1f0717b75cdfacea0c39d6d11e8b98ad8a8d3e5c513d572b001b2"
PARAMETER_COUNT = 8_630_665

ARMS = ("a1_global_sham", "a2_physics_spatial", "a3_learned_spatial")
DOMAINS = ("HAZE4K_INTERNAL", "ITS_UNASSIGNED", "OTS_UNASSIGNED", "NHHAZE_HISTORICAL")
PAIRED_DOMAINS = DOMAINS[:3]
PLANNED_COUNTS = {
    "HAZE4K_INTERNAL": 150,
    "ITS_UNASSIGNED": 150,
    "OTS_UNASSIGNED": 150,
    "NHHAZE_HISTORICAL": 55,
}
TRAINING_SCENES = 600
TRAINING_STEPS = 2000
TRAINING_BLOCK = 10
TRAINING_UNITS = len(ARMS) * (TRAINING_STEPS // TRAINING_BLOCK)
EVALUATION_UNITS = sum(PLANNED_COUNTS.values())
TOTAL_UNITS = TRAINING_UNITS + EVALUATION_UNITS + 1
FORMAL_ITERATIONS = 10_780
TRAINING_SEED = 20_260_803
BOOTSTRAP_SEED = 20_260_804
TRAIN_CROP = 256
MAX_EVAL_CROP = 512
PRIMARY_CLIP_DB = 0.25
UTILITY_MARGIN_DB = 0.10
SPECIFICITY_MARGIN_DB = 0.05
PRECISION_HALF_WIDTH_DB = 0.095
NEAR_CLEAR_PSNR_DB = 30.0
NEAR_CLEAR_MEAN_FLOOR_DB = -0.05
NEGATIVE_TAIL_DB = -0.10
NEGATIVE_TAIL_LIMIT = 0.10
KEEP_SEVERITY_MAX = 0.15
KEEP_DELTA_MAX_DB = 0.02
LOOKS = (
    ("look_050", 0.50, 3.88),
    ("look_075", 0.75, 3.16),
    ("look_100", 1.00, 2.74),
)
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
EPSILON = 1.0e-12


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def image_files(directory: Path, *, recursive: bool = False) -> list[Path]:
    iterator = directory.rglob("*") if recursive else directory.iterdir()
    return sorted(
        path for path in iterator
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def image_array(path: Path) -> np.ndarray:
    with Image.open(path) as source:
        value = np.asarray(source.convert("RGB"), dtype=np.float32) / 255.0
    if value.ndim != 3 or value.shape[2] != 3:
        raise RuntimeError(f"unsupported RGB payload: {path.name}")
    return value


def canonical_rgb_digest(path: Path) -> str:
    with Image.open(path) as source:
        rgb = source.convert("RGB")
        width, height = rgb.size
        payload = np.asarray(rgb, dtype=np.uint8).tobytes()
    digest = hashlib.sha256()
    digest.update(width.to_bytes(8, "big"))
    digest.update(height.to_bytes(8, "big"))
    digest.update(payload)
    return digest.hexdigest()


def deterministic_order(values: Iterable[Any], salt: str, key) -> list[Any]:
    return sorted(values, key=lambda value: (sha256_text(f"{salt}|{key(value)}"), key(value)))


def selected_label(image_name: str, label_dir: Path) -> Path | None:
    stem, extension = os.path.splitext(image_name)
    names = [image_name]
    if "_" in stem:
        prefix = stem.split("_", 1)[0]
        names.extend((f"{prefix}{extension}", f"{prefix}.png", f"{prefix}.jpg"))
    for name in dict.fromkeys(names):
        candidate = label_dir / name
        if candidate.is_file():
            return candidate
    return None


def read_haze4k_roles(path: Path) -> dict[str, set[str]]:
    roles = {"training": set(), "development_screening": set()}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("dataset") != "HAZE4K_TRAIN":
                continue
            role = row.get("role")
            scene = row.get("scene_id")
            digest = row.get("canonical_digest")
            if role in roles and isinstance(scene, str) and scene == digest:
                roles[role].add(scene)
    if len(roles["training"]) != 600 or len(roles["development_screening"]) != 150:
        raise RuntimeError("Haze4K role ledger does not preserve the frozen 600/150 split")
    return roles


def enumerate_haze4k(root: Path, roles: dict[str, set[str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    input_dirs = [root / name for name in ("IN", "haze", "hazy") if (root / name).is_dir()]
    label_dirs = [root / name for name in ("GT", "gt", "clear") if (root / name).is_dir()]
    if len(input_dirs) != 1 or len(label_dirs) != 1:
        raise RuntimeError("Haze4K train root does not have one input and one target directory")
    groups: dict[str, dict[str, Any]] = {}
    clear_cache: dict[Path, str] = {}
    for hazy in image_files(input_dirs[0]):
        clear = selected_label(hazy.name, label_dirs[0])
        if clear is None:
            continue
        digest = clear_cache.setdefault(clear, canonical_rgb_digest(clear))
        item = groups.setdefault(digest, {"scene_id": digest, "clear": clear, "observations": []})
        item["observations"].append(hazy)
    if len(groups) != 750 or any(len(item["observations"]) != 4 for item in groups.values()):
        raise RuntimeError("Haze4K census differs from the frozen 750 scenes and four variants")
    if set(groups) != roles["training"] | roles["development_screening"]:
        raise RuntimeError("Haze4K decoded scene identities differ from the S0 role ledger")
    training = [groups[key] for key in sorted(roles["training"])]
    development = []
    for key in sorted(roles["development_screening"]):
        item = dict(groups[key])
        item["dataset"] = "HAZE4K_INTERNAL"
        item["observations"] = deterministic_order(
            item["observations"], "c1-haze4k-evaluation-observations", lambda path: path.name,
        )[:2]
        development.append(item)
    return training, development


def read_reside_claims(path: Path) -> dict[str, list[str]]:
    claims = {"ITS": [], "OTS": []}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            dataset = row.get("dataset")
            if row.get("schema_version") == 1 and dataset in claims \
                    and row.get("role") == "unassigned_known_overlap_quarantined" \
                    and isinstance(row.get("scene_id"), str) \
                    and isinstance(row.get("allocation_rank"), int) \
                    and 0 <= row["allocation_rank"] < 150:
                claims[dataset].append((row["allocation_rank"], row["scene_id"]))
    result = {}
    for dataset, values in claims.items():
        values.sort()
        if [rank for rank, _ in values] != list(range(150)):
            raise RuntimeError(f"{dataset} does not expose the frozen first 150 allocation ranks")
        result[dataset] = [scene for _, scene in values]
    return result


def index_by_stem(directory: Path) -> dict[str, Path]:
    result = {}
    for path in image_files(directory, recursive=True):
        if path.stem in result:
            raise RuntimeError(f"duplicate clear stem in {directory.name}: {path.stem}")
        result[path.stem] = path
    return result


def index_haze_prefix(directory: Path) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = defaultdict(list)
    for path in image_files(directory, recursive=True):
        result[path.stem.split("_", 1)[0]].append(path)
    return result


def enumerate_reside(context: Any, claims: dict[str, list[str]]) -> list[dict[str, Any]]:
    its_train_clear = index_by_stem(asset_path(context, "its_train_clear", kind="directory"))
    its_val_clear = index_by_stem(asset_path(context, "its_val_clear", kind="directory"))
    its_train_haze = index_haze_prefix(asset_path(context, "its_train_haze", kind="directory"))
    its_val_haze = index_haze_prefix(asset_path(context, "its_val_haze", kind="directory"))
    ots_clear = index_by_stem(asset_path(context, "ots_clear", kind="directory"))
    ots_haze = index_haze_prefix(asset_path(context, "ots_haze", kind="directory"))
    records = []
    for dataset, scene_ids in claims.items():
        for scene_id in scene_ids:
            if dataset == "ITS":
                namespace, stem = scene_id.split(":", 1)
                clear_index, haze_index = (
                    (its_train_clear, its_train_haze)
                    if namespace == "ITS_TRAIN" else (its_val_clear, its_val_haze)
                )
                domain = "ITS_UNASSIGNED"
            else:
                stem = scene_id
                clear_index, haze_index = ots_clear, ots_haze
                domain = "OTS_UNASSIGNED"
            clear = clear_index.get(stem)
            variants = haze_index.get(stem, [])
            if clear is None or len(variants) < 2:
                raise RuntimeError(f"{dataset} claimed scene does not resolve two observations")
            selected = deterministic_order(
                variants, f"c1-{dataset.lower()}-evaluation-observations", lambda path: path.name,
            )[:2]
            records.append({
                "dataset": domain,
                "scene_id": scene_id,
                "clear": clear,
                "observations": selected,
            })
    return records


def nh_scene_key(path: Path) -> str:
    stem = path.stem.casefold()
    for suffix in ("_hazy", "_haze", "_gt", "_clear"):
        if stem.endswith(suffix):
            stem = stem[:-len(suffix)]
    return stem


def enumerate_nhhaze(root: Path) -> list[dict[str, Any]]:
    files = image_files(root, recursive=True)
    hazy = [path for path in files if "hazy" in path.stem.casefold() or "haze" in path.stem.casefold()]
    clear = [path for path in files if "gt" in path.stem.casefold() or "clear" in path.stem.casefold()]
    clear_index = {nh_scene_key(path): path for path in clear}
    records = []
    for path in hazy:
        key = nh_scene_key(path)
        if key in clear_index:
            records.append({
                "dataset": "NHHAZE_HISTORICAL",
                "scene_id": f"NHHAZE_{key}",
                "clear": clear_index[key],
                "observations": [path],
            })
    records.sort(key=lambda item: item["scene_id"])
    if len(records) != 55:
        raise RuntimeError("NH-HAZE historical-development roster must contain exactly 55 pairs")
    return records


def load_official_model(context: Any):
    import torch

    expected = {
        "official_checkpoint": CHECKPOINT_SHA256,
        "model_source": MODEL_SOURCE_SHA256,
        "model_layers": MODEL_LAYERS_SHA256,
    }
    for identifier, digest in expected.items():
        asset = context.assets.get(identifier)
        if asset is None or asset.sha256 != digest:
            raise RuntimeError(f"bound model identity changed: {identifier}")
    anchor = asset_path(context, "official_anchor_checkout", kind="git_checkout")
    if str(anchor) not in sys.path:
        sys.path.insert(0, str(anchor))
    from Dehazing.ITS.models.ConvIR import build_net

    module = sys.modules[build_net.__module__]
    layers = sys.modules.get("Dehazing.ITS.models.layers")
    if Path(module.__file__).resolve() != asset_path(context, "model_source", kind="file").resolve() \
            or layers is None \
            or Path(layers.__file__).resolve() != asset_path(context, "model_layers", kind="file").resolve():
        raise RuntimeError("official ConvIR-B import escaped the bound source files")
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
        raise RuntimeError("official ConvIR-B parameter count changed")
    model.requires_grad_(False).to(context.device).eval()
    return torch, model


def adapter_class(torch):
    class SignalAdapter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.body = torch.nn.Sequential(
                torch.nn.Conv2d(4, 16, 3, padding=1),
                torch.nn.GELU(),
                torch.nn.Conv2d(16, 64, 1),
            )
            torch.nn.init.zeros_(self.body[-1].weight)
            torch.nn.init.zeros_(self.body[-1].bias)

        def forward(self, value):
            gamma, beta = self.body(value).chunk(2, dim=1)
            return 0.1 * torch.tanh(gamma), 0.1 * torch.tanh(beta)

    return SignalAdapter


def physics_signal(torch, image):
    import torch.nn.functional as functional

    minimum = image.min(dim=1, keepdim=True).values
    maximum = image.max(dim=1, keepdim=True).values
    dark = -functional.max_pool2d(-minimum, 15, stride=1, padding=7)
    saturation = (maximum - minimum) / maximum.clamp_min(1.0e-4)
    luminance = 0.299 * image[:, :1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
    local_mean = functional.avg_pool2d(luminance, 9, stride=1, padding=4)
    contrast = (luminance - local_mean).abs().clamp(0.0, 1.0)
    dx = functional.pad((luminance[:, :, :, 1:] - luminance[:, :, :, :-1]).abs(), (0, 1, 0, 0))
    dy = functional.pad((luminance[:, :, 1:, :] - luminance[:, :, :-1, :]).abs(), (0, 0, 0, 1))
    gradient = (dx + dy).clamp(0.0, 1.0)
    return torch.cat((dark, saturation, contrast, gradient), dim=1)


def signal_for_arm(torch, image, arm: str, *, shuffled: bool = False):
    if arm in {"a1_global_sham", "a2_physics_spatial"}:
        value = physics_signal(torch, image)
        if arm == "a1_global_sham":
            value = value.mean(dim=(2, 3), keepdim=True).expand_as(value)
    else:
        luminance = 0.299 * image[:, :1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
        value = torch.cat((image, luminance), dim=1)
    if shuffled:
        value = torch.roll(value, shifts=(value.shape[2] // 2, value.shape[3] // 2), dims=(2, 3))
    return value


def final_feature(torch, model, image):
    import torch.nn.functional as functional

    with torch.no_grad():
        x2 = functional.interpolate(image, scale_factor=0.5)
        x4 = functional.interpolate(x2, scale_factor=0.5)
        z2 = model.SCM2(x2)
        z4 = model.SCM1(x4)
        x0 = model.feat_extract[0](image)
        res1 = model.Encoder[0](x0)
        z = model.FAM2(model.feat_extract[1](res1), z2)
        res2 = model.Encoder[1](z)
        z = model.FAM1(model.feat_extract[2](res2), z4)
        z = model.Decoder[0](model.Encoder[2](z))
        z = model.feat_extract[3](z)
        z = model.Decoder[1](model.Convs[0](torch.cat((z, res2), dim=1)))
        z = model.feat_extract[4](z)
        z = model.Decoder[2](model.Convs[1](torch.cat((z, res1), dim=1)))
    return z.detach()


def adapted_output(torch, model, adapter, image, arm: str, *, shuffled: bool = False):
    feature = final_feature(torch, model, image)
    signal = signal_for_arm(torch, image, arm, shuffled=shuffled)
    gamma, beta = adapter(signal)
    prediction = model.feat_extract[5](feature * (1.0 + gamma) + beta) + image
    return prediction.clamp(0.0, 1.0), gamma, beta


def edge_loss(torch, value, target):
    dx = (value[:, :, :, 1:] - value[:, :, :, :-1]) - (target[:, :, :, 1:] - target[:, :, :, :-1])
    dy = (value[:, :, 1:, :] - value[:, :, :-1, :]) - (target[:, :, 1:, :] - target[:, :, :-1, :])
    return dx.abs().mean() + dy.abs().mean()


def training_loss(torch, prediction, target, gamma, beta):
    charbonnier = torch.sqrt((prediction - target).square() + 1.0e-6).mean()
    regularizer = gamma.square().mean() + beta.square().mean()
    return charbonnier + 0.1 * edge_loss(torch, prediction, target) + 0.01 * regularizer


def crop_pair(hazy: np.ndarray, clear: np.ndarray, scene_id: str, side: int) -> tuple[np.ndarray, np.ndarray]:
    if hazy.shape != clear.shape or min(hazy.shape[:2]) < side:
        raise RuntimeError(f"paired geometry is outside the frozen crop contract: {scene_id[:24]}")
    height, width = hazy.shape[:2]
    seed = int(sha256_text(f"c1-eval-crop|{scene_id}")[:16], 16)
    y0 = seed % (height - side + 1)
    x0 = (seed // 65537) % (width - side + 1)
    return hazy[y0:y0 + side, x0:x0 + side].copy(), clear[y0:y0 + side, x0:x0 + side].copy()


def training_batch(torch, records: list[dict[str, Any]], step: int):
    rng = np.random.default_rng(TRAINING_SEED + step)
    hazy_values, clear_values = [], []
    for _ in range(2):
        record = records[int(rng.integers(0, len(records)))]
        observation = record["observations"][int(rng.integers(0, len(record["observations"])))]
        hazy = image_array(observation)
        clear = image_array(record["clear"])
        height, width = hazy.shape[:2]
        if hazy.shape != clear.shape or min(height, width) < TRAIN_CROP:
            raise RuntimeError("Haze4K training pair violates the 256-pixel crop contract")
        y0 = int(rng.integers(0, height - TRAIN_CROP + 1))
        x0 = int(rng.integers(0, width - TRAIN_CROP + 1))
        hazy = hazy[y0:y0 + TRAIN_CROP, x0:x0 + TRAIN_CROP]
        clear = clear[y0:y0 + TRAIN_CROP, x0:x0 + TRAIN_CROP]
        if bool(rng.integers(0, 2)):
            hazy, clear = hazy[:, ::-1], clear[:, ::-1]
        hazy_values.append(np.ascontiguousarray(hazy.transpose(2, 0, 1)))
        clear_values.append(np.ascontiguousarray(clear.transpose(2, 0, 1)))
    return (
        torch.from_numpy(np.stack(hazy_values)).to("cuda"),
        torch.from_numpy(np.stack(clear_values)).to("cuda"),
    )


def psnr(torch, prediction, target) -> float:
    mse = float((prediction.double() - target.double()).square().mean().item())
    return -10.0 * math.log10(max(mse, EPSILON))


def evaluate_scene(torch, model, adapters, record: dict[str, Any]) -> dict[str, Any]:
    clear_full = image_array(record["clear"])
    observations = []
    input_parts = [record["dataset"], record["scene_id"], sha256_file(record["clear"])]
    for observation in record["observations"]:
        hazy_full = image_array(observation)
        side = (min(hazy_full.shape[0], hazy_full.shape[1], MAX_EVAL_CROP) // 32) * 32
        if side < 256:
            raise RuntimeError("evaluation pair is below the frozen 256-pixel minimum")
        hazy, clear = crop_pair(hazy_full, clear_full, record["scene_id"], side)
        hazy_tensor = torch.from_numpy(hazy.transpose(2, 0, 1).copy()).unsqueeze(0).to("cuda")
        clear_tensor = torch.from_numpy(clear.transpose(2, 0, 1).copy()).unsqueeze(0).to("cuda")
        with torch.no_grad():
            baseline = model(hazy_tensor)[2].clamp(0.0, 1.0)
            outputs = {}
            modulation = {}
            for arm in ARMS:
                outputs[arm], gamma, beta = adapted_output(
                    torch, model, adapters[arm], hazy_tensor, arm,
                )
                modulation[arm] = float((gamma.abs().mean() + beta.abs().mean()).item())
            shuffled_a2, _, _ = adapted_output(
                torch, model, adapters["a2_physics_spatial"], hazy_tensor,
                "a2_physics_spatial", shuffled=True,
            )
            shuffled_a3, _, _ = adapted_output(
                torch, model, adapters["a3_learned_spatial"], hazy_tensor,
                "a3_learned_spatial", shuffled=True,
            )
        scores = {"a0": psnr(torch, baseline, clear_tensor)}
        scores.update({arm: psnr(torch, value, clear_tensor) for arm, value in outputs.items()})
        scores["a2_shuffle"] = psnr(torch, shuffled_a2, clear_tensor)
        scores["a3_shuffle"] = psnr(torch, shuffled_a3, clear_tensor)
        severity = float(physics_signal(torch, hazy_tensor).mean().item())
        observations.append({"scores": scores, "modulation": modulation, "severity": severity})
        input_parts.append(sha256_file(observation))
    mean = lambda key: float(np.mean([item["scores"][key] for item in observations]))
    result = {
        "dataset": record["dataset"],
        "scene_id": record["scene_id"],
        "observation_count": len(observations),
        "baseline_psnr_db": mean("a0"),
        "severity": float(np.mean([item["severity"] for item in observations])),
        "a2_minus_a1": float(np.clip(mean("a2_physics_spatial") - mean("a1_global_sham"), -PRIMARY_CLIP_DB, PRIMARY_CLIP_DB)),
        "a3_minus_a1": float(np.clip(mean("a3_learned_spatial") - mean("a1_global_sham"), -PRIMARY_CLIP_DB, PRIMARY_CLIP_DB)),
        "a2_specificity": float(np.clip(mean("a2_physics_spatial") - mean("a2_shuffle"), -PRIMARY_CLIP_DB, PRIMARY_CLIP_DB)),
        "a3_specificity": float(np.clip(mean("a3_learned_spatial") - mean("a3_shuffle"), -PRIMARY_CLIP_DB, PRIMARY_CLIP_DB)),
        "a2_minus_a0": float(np.clip(mean("a2_physics_spatial") - mean("a0"), -PRIMARY_CLIP_DB, PRIMARY_CLIP_DB)),
        "a3_minus_a0": float(np.clip(mean("a3_learned_spatial") - mean("a0"), -PRIMARY_CLIP_DB, PRIMARY_CLIP_DB)),
        "a2_observation_deltas": [item["scores"]["a2_physics_spatial"] - item["scores"]["a1_global_sham"] for item in observations],
        "a3_observation_deltas": [item["scores"]["a3_learned_spatial"] - item["scores"]["a1_global_sham"] for item in observations],
        "a2_modulation": float(np.mean([item["modulation"]["a2_physics_spatial"] for item in observations])),
        "a3_modulation": float(np.mean([item["modulation"]["a3_learned_spatial"] for item in observations])),
    }
    return result | {"input_sha256": sha256_text("|".join(input_parts))}


def mean_interval(values: list[float], critical: float) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    point = float(array.mean())
    if len(array) < 2:
        half = float("inf")
    else:
        half = critical * float(array.std(ddof=1)) / math.sqrt(len(array))
    return {
        "count": len(array), "estimate": point, "lower": point - half,
        "upper": point + half, "half_width": half,
    }


def contrast_summary(records: list[dict[str, Any]], key: str, critical: float) -> dict[str, Any]:
    by_domain = {
        domain: mean_interval([float(item[key]) for item in records if item["dataset"] == domain], critical)
        for domain in DOMAINS
    }
    estimates = [by_domain[domain]["estimate"] for domain in DOMAINS]
    variances = []
    for domain in DOMAINS:
        values = np.asarray([float(item[key]) for item in records if item["dataset"] == domain])
        variances.append(float(values.var(ddof=1)) / len(values) if len(values) >= 2 else float("inf"))
    point = float(np.mean(estimates))
    half = critical * math.sqrt(sum(variances)) / len(DOMAINS)
    return {
        "by_domain": by_domain,
        "overall_equal_domain": {
            "estimate": point, "lower": point - half, "upper": point + half,
            "half_width": half,
        },
    }


def wilson(successes: int, total: int, critical: float) -> dict[str, float | int]:
    if total == 0:
        return {"successes": 0, "total": 0, "estimate": 0.0, "lower": 0.0, "upper": 1.0}
    point = successes / total
    denominator = 1.0 + critical * critical / total
    center = (point + critical * critical / (2.0 * total)) / denominator
    half = critical * math.sqrt(point * (1.0 - point) / total + critical * critical / (4.0 * total * total)) / denominator
    return {"successes": successes, "total": total, "estimate": point, "lower": max(0.0, center - half), "upper": min(1.0, center + half)}


def prevalence_summary(records: list[dict[str, Any]], predicate, domains: tuple[str, ...], critical: float) -> dict[str, Any]:
    by_domain = {}
    for domain in domains:
        selected = [item for item in records if item["dataset"] == domain]
        by_domain[domain] = wilson(sum(bool(predicate(item)) for item in selected), len(selected), critical)
    return {
        "by_domain": by_domain,
        "overall_equal_domain": {
            "estimate": float(np.mean([item["estimate"] for item in by_domain.values()])),
            "lower": float(np.mean([item["lower"] for item in by_domain.values()])),
            "upper": float(np.mean([item["upper"] for item in by_domain.values()])),
        },
    }


def selected_arm(primary: dict[str, Any]) -> str:
    a2 = primary["a2_minus_a1"]["overall_equal_domain"]["estimate"]
    a3 = primary["a3_minus_a1"]["overall_equal_domain"]["estimate"]
    return "a3" if a3 > a2 + 0.005 else "a2"


def scientific_summaries(records: list[dict[str, Any]], critical: float) -> dict[str, Any]:
    primary = {
        "a2_minus_a1": contrast_summary(records, "a2_minus_a1", critical),
        "a3_minus_a1": contrast_summary(records, "a3_minus_a1", critical),
    }
    selected = selected_arm(primary)
    specificity_key = f"{selected}_specificity"
    a0_key = f"{selected}_minus_a0"
    primary_key = f"{selected}_minus_a1"
    observation_key = f"{selected}_observation_deltas"
    modulation_key = f"{selected}_modulation"
    specificity = contrast_summary(records, specificity_key, critical)
    repeatability = prevalence_summary(
        records,
        lambda item: len(item[observation_key]) == 2
        and min(item[observation_key]) > 0.0
        and abs(item[observation_key][0] - item[observation_key][1]) <= 0.10,
        PAIRED_DOMAINS,
        critical,
    )
    near_clear_records = [item for item in records if item["baseline_psnr_db"] >= NEAR_CLEAR_PSNR_DB]
    near_clear = {
        "mean_delta": contrast_summary(near_clear_records, a0_key, critical) if all(
            any(item["dataset"] == domain for item in near_clear_records) for domain in DOMAINS
        ) else None,
        "negative_tail": prevalence_summary(
            near_clear_records, lambda item: item[a0_key] < NEGATIVE_TAIL_DB, DOMAINS, critical,
        ),
        "count": len(near_clear_records),
    }
    negative_tail = prevalence_summary(
        records, lambda item: item[primary_key] < NEGATIVE_TAIL_DB, DOMAINS, critical,
    )
    keep_supported = prevalence_summary(
        [item for item in records if item["severity"] <= KEEP_SEVERITY_MAX],
        lambda item: abs(item[primary_key]) <= KEEP_DELTA_MAX_DB and item[modulation_key] <= 0.02,
        DOMAINS,
        critical,
    )
    return {
        "primary": primary,
        "selected_arm": selected,
        "specificity": specificity,
        "repeatability": repeatability,
        "near_clear": near_clear,
        "negative_tail": negative_tail,
        "keep_supported": keep_supported,
    }


def gate_outcomes(summaries: dict[str, Any]) -> tuple[dict[str, str], dict[str, Any]]:
    primary = summaries["primary"]
    selected = summaries["selected_arm"]
    selected_primary = primary[f"{selected}_minus_a1"]
    overall = selected_primary["overall_equal_domain"]
    domains = selected_primary["by_domain"]
    favorable_primary = overall["lower"] > UTILITY_MARGIN_DB \
        and sum(item["lower"] > 0.0 for item in domains.values()) >= 3 \
        and all(item["upper"] >= 0.0 for item in domains.values())
    impossible_primary = all(
        item["overall_equal_domain"]["upper"] <= UTILITY_MARGIN_DB
        for item in primary.values()
    )
    primary_gate = "favorable" if favorable_primary else "unfavorable" if impossible_primary else "indeterminate"

    specificity = summaries["specificity"]
    favorable_specificity = specificity["overall_equal_domain"]["lower"] > SPECIFICITY_MARGIN_DB \
        and sum(item["lower"] > 0.0 for item in specificity["by_domain"].values()) >= 3 \
        and all(item["upper"] >= 0.0 for item in specificity["by_domain"].values())
    specificity_gate = "favorable" if favorable_specificity else (
        "unfavorable" if specificity["overall_equal_domain"]["upper"] <= SPECIFICITY_MARGIN_DB else "indeterminate"
    )

    repeatability = summaries["repeatability"]
    repeat_overall = repeatability["overall_equal_domain"]
    favorable_repeat = repeat_overall["lower"] >= 0.50 \
        and sum(item["lower"] >= 0.40 for item in repeatability["by_domain"].values()) >= 2 \
        and all(item["upper"] >= 0.30 for item in repeatability["by_domain"].values())
    impossible_repeat = repeat_overall["upper"] < 0.50 \
        or sum(item["upper"] < 0.40 for item in repeatability["by_domain"].values()) >= 2
    repeat_gate = "favorable" if favorable_repeat else "unfavorable" if impossible_repeat else "indeterminate"

    near_clear = summaries["near_clear"]
    if near_clear["mean_delta"] is None:
        near_gate = "indeterminate"
    else:
        near_mean = near_clear["mean_delta"]["overall_equal_domain"]
        near_tail = near_clear["negative_tail"]["overall_equal_domain"]
        near_gate = "safe" if near_mean["lower"] >= NEAR_CLEAR_MEAN_FLOOR_DB \
            and near_tail["upper"] <= NEGATIVE_TAIL_LIMIT else (
                "unsafe" if near_mean["upper"] < NEAR_CLEAR_MEAN_FLOOR_DB
                or near_tail["lower"] > NEGATIVE_TAIL_LIMIT else "indeterminate"
            )

    tail = summaries["negative_tail"]["overall_equal_domain"]
    keep = summaries["keep_supported"]["overall_equal_domain"]
    tail_keep_gate = "safe" if tail["upper"] <= NEGATIVE_TAIL_LIMIT and keep["lower"] >= 0.50 else (
        "unsafe" if tail["lower"] > NEGATIVE_TAIL_LIMIT or keep["upper"] < 0.50 else "indeterminate"
    )

    precision_met = all(
        item["half_width"] <= PRECISION_HALF_WIDTH_DB
        for contrast in primary.values()
        for item in [contrast["overall_equal_domain"], *contrast["by_domain"].values()]
    )
    outcomes = {
        "evidence_identity_integrity_coverage": "pass",
        "primary_observable_utility": primary_gate,
        "spatial_signal_specificity": specificity_gate,
        "cross_observation_repeatability": repeat_gate,
        "near_clear_fidelity": near_gate,
        "negative_tail_and_keep_supported": tail_keep_gate,
        "primary_precision": "met" if precision_met else "unmet",
    }
    return outcomes, {"precision_met": precision_met}


def sequential_stop(records: list[dict[str, Any]], critical: float) -> str | None:
    summaries = scientific_summaries(records, critical)
    if all(
        contrast["overall_equal_domain"]["upper"] <= UTILITY_MARGIN_DB
        for contrast in summaries["primary"].values()
    ):
        return "predeclared_primary_futility"
    selected = summaries["selected_arm"]
    a0_key = f"{selected}_minus_a0"
    near = [item for item in records if item["baseline_psnr_db"] >= NEAR_CLEAR_PSNR_DB]
    if near:
        tail = prevalence_summary(
            near, lambda item: item[a0_key] < NEGATIVE_TAIL_DB, DOMAINS, critical,
        )["overall_equal_domain"]
        if tail["lower"] > NEGATIVE_TAIL_LIMIT:
            return "predeclared_near_clear_safety_failure"
    return None


def look_segments(records: list[dict[str, Any]]) -> list[tuple[str, float, list[dict[str, Any]], float]]:
    by_domain = {domain: sorted(
        [item for item in records if item["dataset"] == domain], key=lambda item: item["scene_id"],
    ) for domain in DOMAINS}
    previous = {domain: 0 for domain in DOMAINS}
    result = []
    for look_id, fraction, critical in LOOKS:
        segment = []
        for domain in DOMAINS:
            stop = math.ceil(len(by_domain[domain]) * fraction)
            segment.extend(by_domain[domain][previous[domain]:stop])
            previous[domain] = stop
        result.append((look_id, fraction, segment, critical))
    return result


def contract(context_path: Path) -> None:
    context = load_context(context_path, "contract")
    prepare_phase_output(context)
    import torch

    started = time.monotonic()
    torch.manual_seed(TRAINING_SEED)
    torch.cuda.reset_peak_memory_stats()
    torch, model = load_official_model(context)
    SignalAdapter = adapter_class(torch)
    adapters = {arm: SignalAdapter().to(context.device) for arm in ARMS}
    optimizers = {
        arm: torch.optim.AdamW(adapters[arm].parameters(), lr=2.0e-4, betas=(0.9, 0.999), weight_decay=0.0)
        for arm in ARMS
    }
    hazy = torch.rand((2, 3, TRAIN_CROP, TRAIN_CROP), device=context.device)
    target = torch.rand_like(hazy)
    finite = True
    for index in range(FORMAL_ITERATIONS):
        if index < len(ARMS) * TRAINING_STEPS:
            arm = ARMS[(index // TRAINING_STEPS) % len(ARMS)]
            optimizer = optimizers[arm]
            optimizer.zero_grad(set_to_none=True)
            prediction, gamma, beta = adapted_output(torch, model, adapters[arm], hazy, arm)
            loss = training_loss(torch, prediction, target, gamma, beta)
            finite = finite and bool(torch.isfinite(loss).item())
            loss.backward()
            optimizer.step()
        else:
            arm = ARMS[index % len(ARMS)]
            with torch.no_grad():
                prediction, _, _ = adapted_output(
                    torch, model, adapters[arm], hazy[:1], arm,
                    shuffled=arm != "a1_global_sham" and index % 2 == 0,
                )
            finite = finite and bool(torch.isfinite(prediction).all().item())
        if (index + 1) % 100 == 0 or index + 1 == FORMAL_ITERATIONS:
            write_contract_progress(
                context, completed_iterations=index + 1,
                total_iterations=FORMAL_ITERATIONS, stage="synthetic_same_scale_path",
            )
    elapsed = time.monotonic() - started
    peak = float(torch.cuda.max_memory_allocated() / (1024 * 1024))
    checks = {
        "cuda_available": context.device == "cuda" and torch.cuda.is_available(),
        "official_anchor_identity": context.assets["official_anchor_checkout"].commit == ANCHOR_COMMIT,
        "official_parameter_count": sum(parameter.numel() for parameter in model.parameters()) == PARAMETER_COUNT,
        "equal_adapter_parameter_count": len({sum(parameter.numel() for parameter in adapter.parameters()) for adapter in adapters.values()}) == 1,
        "same_scale_iteration_count": context.engineering_contract["cost_contract"]["formal_iterations"] == FORMAL_ITERATIONS,
        "finite_synthetic_training_and_evaluation": finite,
        "run_only_data_absent": all(identifier not in context.assets for identifier in (
            "s0_scene_role_ledger", "reside_role_ledger", "haze4k_train", "its_train_clear",
            "its_train_haze", "its_val_clear", "its_val_haze", "ots_clear", "ots_haze", "nhhaze",
        )),
    }
    engineering = {
        "mode": "gpu_synthetic_no_data",
        "device": context.device,
        "fixture": {"batch": 2, "channels": 3, "height": TRAIN_CROP, "width": TRAIN_CROP},
        "production_path_exercised": True,
        "protected_data_touched": False,
        "scientific_output_created": False,
        "scientific_training_occurred": False,
        "cost": {
            "observed_iterations": FORMAL_ITERATIONS,
            "observed_wall_seconds": elapsed,
            "observed_peak_memory_mib": peak,
        },
    }
    write_contract_result(context, checks=checks, engineering=engineering)


def run(context_path: Path) -> None:
    context = load_context(context_path, "run")
    prepare_phase_output(context)
    if context.total_units != TOTAL_UNITS or context.evidence_role != "development_screening" \
            or any(context.protected_data_permissions.values()):
        raise RuntimeError("C1 runtime role, unit, or protected-data contract changed")
    if load_completed_unit_ledger(context):
        raise RuntimeError("fresh C1 run unexpectedly contains completed units")

    roles_path = asset_path(context, "s0_scene_role_ledger", kind="file")
    reside_ledger = asset_path(context, "reside_role_ledger", kind="file")
    if sha256_file(roles_path) != S0_LEDGER_SHA256 or sha256_file(reside_ledger) != ROLE_LEDGER_SHA256:
        raise RuntimeError("development role-ledger identity changed")
    roles = read_haze4k_roles(roles_path)
    training_records, haze4k_development = enumerate_haze4k(
        asset_path(context, "haze4k_train", kind="directory"), roles,
    )
    claims = read_reside_claims(reside_ledger)
    evaluation_records = [
        *haze4k_development,
        *enumerate_reside(context, claims),
        *enumerate_nhhaze(asset_path(context, "nhhaze", kind="directory")),
    ]
    counts = {domain: sum(item["dataset"] == domain for item in evaluation_records) for domain in DOMAINS}
    if counts != PLANNED_COUNTS or len(training_records) != TRAINING_SCENES:
        raise RuntimeError("frozen training or evaluation roster count changed")

    import torch

    torch.manual_seed(TRAINING_SEED)
    torch, model = load_official_model(context)
    SignalAdapter = adapter_class(torch)
    adapters = {arm: SignalAdapter().to(context.device) for arm in ARMS}
    completed = 0
    roster_digest = sha256_text("\n".join(
        [item["scene_id"] for item in training_records]
        + [f"{item['dataset']}|{item['scene_id']}" for item in sorted(evaluation_records, key=lambda item: (item["dataset"], item["scene_id"]))]
    ))
    for arm in ARMS:
        optimizer = torch.optim.AdamW(
            adapters[arm].parameters(), lr=2.0e-4,
            betas=(0.9, 0.999), weight_decay=0.0,
        )
        adapters[arm].train()
        for step in range(TRAINING_STEPS):
            hazy, clear = training_batch(torch, training_records, step)
            optimizer.zero_grad(set_to_none=True)
            prediction, gamma, beta = adapted_output(torch, model, adapters[arm], hazy, arm)
            loss = training_loss(torch, prediction, clear, gamma, beta)
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError("adapter training produced a non-finite loss")
            loss.backward()
            optimizer.step()
            if (step + 1) % TRAINING_BLOCK == 0:
                block = (step + 1) // TRAINING_BLOCK
                relpath = f"training/{arm}_{block:03d}.pt"
                destination = output_file(context, relpath)
                destination.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"arm": arm, "step": step + 1, "adapter": adapters[arm].state_dict()}, destination)
                record_completed_unit(
                    context, unit_id=f"train_{arm}_{block:03d}",
                    input_sha256=sha256_text(f"{roster_digest}|{arm}|{block}|{TRAINING_SEED}"),
                    output_relpath=relpath,
                )
                completed += 1
                write_workload_progress(context, completed_units=completed, stage="adapter_training")
        adapters[arm].eval()

    measured: list[dict[str, Any]] = []
    skipped = []
    looks_observed = []
    stop_reason = None
    remaining_after_stop: list[dict[str, Any]] = []
    for look_id, fraction, segment, critical in look_segments(evaluation_records):
        if stop_reason is not None:
            remaining_after_stop.extend(segment)
            continue
        for record in segment:
            result = evaluate_scene(torch, model, adapters, record)
            unit_id = f"eval_{sha256_text(record['dataset'] + '|' + record['scene_id'])[:24]}"
            relpath = f"evaluation/{unit_id}.json"
            atomic_json(output_file(context, relpath), result)
            record_completed_unit(
                context, unit_id=unit_id, input_sha256=result.pop("input_sha256"),
                output_relpath=relpath,
            )
            measured.append(result)
            completed += 1
            write_workload_progress(context, completed_units=completed, stage="development_evaluation")
        look_counts = {domain: sum(item["dataset"] == domain for item in measured) for domain in DOMAINS}
        stop_reason = sequential_stop(measured, critical) if fraction < 1.0 else None
        looks_observed.append({
            "look_id": look_id, "information_fraction": fraction,
            "critical_value": critical, "evaluated_counts": look_counts,
            "stop_reason": stop_reason,
        })
    for record in remaining_after_stop:
        unit_id = f"eval_{sha256_text(record['dataset'] + '|' + record['scene_id'])[:24]}"
        relpath = f"evaluation/{unit_id}.json"
        value = {
            "dataset": record["dataset"], "scene_id": record["scene_id"],
            "status": "SKIPPED_PREDECLARED_GROUP_SEQUENTIAL",
            "reason": stop_reason,
        }
        atomic_json(output_file(context, relpath), value)
        record_completed_unit(
            context, unit_id=unit_id,
            input_sha256=sha256_text(f"{roster_digest}|{record['dataset']}|{record['scene_id']}|skipped"),
            output_relpath=relpath,
        )
        skipped.append(value)
        completed += 1
        write_workload_progress(context, completed_units=completed, stage="predeclared_skipped_units")

    final_critical = next(item[2] for item in LOOKS if item[0] == looks_observed[-1]["look_id"])
    summaries = scientific_summaries(measured, final_critical)
    outcomes, precision = gate_outcomes(summaries)
    aggregate = {
        "schema_version": 1,
        "route_id": ROUTE_ID,
        "operation_id": OPERATION_ID,
        "roster_digest": roster_digest,
        "planned_counts": PLANNED_COUNTS,
        "evaluated_counts": {domain: sum(item["dataset"] == domain for item in measured) for domain in DOMAINS},
        "skipped_count": len(skipped),
        "sequential_looks": looks_observed,
        "selected_arm": summaries["selected_arm"],
        "gate_outcomes": outcomes,
    }
    aggregate_relpath = "aggregate/decision.json"
    atomic_json(output_file(context, aggregate_relpath), aggregate)
    record_completed_unit(
        context, unit_id="aggregate_decision",
        input_sha256=sha256_text(f"{roster_digest}|aggregate|{len(measured)}|{len(skipped)}"),
        output_relpath=aggregate_relpath,
    )
    completed += 1
    if completed != TOTAL_UNITS or len(load_completed_unit_ledger(context)) != TOTAL_UNITS:
        raise RuntimeError("completed-unit ledger does not exactly cover the frozen workload")
    write_workload_progress(context, completed_units=completed, stage="aggregate_complete")

    summary = {
        "schema_version": 1,
        "route_id": ROUTE_ID,
        "operation_id": OPERATION_ID,
        "run_id": context.run_id,
        "design": {
            "arms": ["a0_official", *ARMS],
            "training_steps_per_adapter": TRAINING_STEPS,
            "batch_size": 2,
            "crop_size": TRAIN_CROP,
            "seed": TRAINING_SEED,
            "primary_clip_db": [-PRIMARY_CLIP_DB, PRIMARY_CLIP_DB],
            "minimum_meaningful_effect_db": UTILITY_MARGIN_DB,
            "precision_half_width_db": PRECISION_HALF_WIDTH_DB,
        },
        "roster": aggregate,
        "summaries": summaries,
        "precision": precision,
        "gate_outcomes": outcomes,
        "protected_data_touched": False,
        "confirmation_or_sealed_data_touched": False,
        "marker": "DAYTIME_DEHAZING_OBSERVABLE_SIGNAL_DOMAIN_FACTORIAL_V1_COMPLETE",
    }
    atomic_json(output_file(context, SUMMARY_NAME), summary)
    gate_summary = {
        "schema_version": 1,
        "route_id": ROUTE_ID,
        "operation_id": OPERATION_ID,
        "gate_outcomes": outcomes,
        "selected_arm": summaries["selected_arm"],
        "primary": summaries["primary"],
        "specificity": summaries["specificity"],
        "repeatability": summaries["repeatability"],
        "near_clear": summaries["near_clear"],
        "negative_tail": summaries["negative_tail"],
        "keep_supported": summaries["keep_supported"],
        "sequential_looks": looks_observed,
    }
    atomic_json(output_file(context, GATE_NAME), gate_summary)
    source_sha = sha256_file(output_file(context, SUMMARY_NAME))
    facts = [
        build_gate_review_fact(
            fact_id=gate_id,
            metric=f"{gate_id} typed gate outcome",
            unit="typed outcome",
            population="505 frozen development-only original clear scenes",
            grouping="original clear scene; haze observations, pixels, training crops, and sequential looks are nested",
            gate_outcome=outcome,
            source_filename=SUMMARY_NAME,
            source_sha256=source_sha,
        )
        for gate_id, outcome in outcomes.items()
    ]
    write_review_facts(context, relpath=FACTS_NAME, facts=facts)
    write_gate_result(
        context,
        gate_outcomes=outcomes,
        details={
            "summary_file": SUMMARY_NAME,
            "gate_summary_file": GATE_NAME,
            "review_facts_file": FACTS_NAME,
            "selected_arm": summaries["selected_arm"],
            "evaluated_scene_count": len(measured),
            "skipped_scene_count": len(skipped),
            "training_steps_per_adapter": TRAINING_STEPS,
            "completed_unit_count": completed,
            "network_training_occurred": True,
            "protected_data_touched": False,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("contract", "run"))
    parser.add_argument("--context", required=True, type=Path)
    arguments = parser.parse_args()
    if arguments.phase == "contract":
        contract(arguments.context)
    else:
        run(arguments.context)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
