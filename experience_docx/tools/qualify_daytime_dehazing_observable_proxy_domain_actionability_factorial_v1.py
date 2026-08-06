#!/usr/bin/env python3
"""Test inference-observable proxy actionability in a matched full factorial."""

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
    build_primary_review_fact,
    load_completed_unit_ledger,
    load_context,
    output_file,
    prepare_phase_output,
    record_completed_unit,
    write_contract_progress,
    write_contract_result,
    write_gate_result,
    write_scientific_review_facts,
    write_workload_progress,
)


ROUTE_ID = "daytime-dehazing-observable-proxy-domain-actionability-factorial-v1"
OPERATION_ID = "DAYTIME_DEHAZING_OBSERVABLE_PROXY_DOMAIN_ACTIONABILITY_FACTORIAL_QUALIFY"
SUMMARY_NAME = "daytime_dehazing_observable_proxy_domain_actionability_factorial_v1_summary.json"
GATE_NAME = "daytime_dehazing_observable_proxy_domain_actionability_factorial_v1_gate_summary.json"
FACTS_NAME = "daytime_dehazing_observable_proxy_domain_actionability_factorial_v1_review_facts.json"

ANCHOR_COMMIT = "3b4da35440c8c26a7d1bcaf1daf342e11d9a3898"
CHECKPOINT_SHA256 = "6f42037d57a4e3de3a10ac0ab909d66a3415864a19433c29204a975f4efa4088"
MODEL_SOURCE_SHA256 = "9681defa95c6602d0e0abae05e635771bd65d48a18555198ff00d573242a0005"
MODEL_LAYERS_SHA256 = "ac8a05bd626d9adda16308dedb9466f36d7ff44cfb666f64e7e14ddf8cdf43a4"
S0_LEDGER_SHA256 = "4cff8e7aecea5d8e19165ac4e725f69746342521115a3ab1f03ea1474f280960"
ROLE_LEDGER_SHA256 = "2f387c0992c1f0717b75cdfacea0c39d6d11e8b98ad8a8d3e5c513d572b001b2"
PARAMETER_COUNT = 8_630_665

ARM_CONFIG = {
    "physics_permuted": ("physics", "permuted"),
    "physics_aligned": ("physics", "aligned"),
    "uncertainty_permuted": ("uncertainty", "permuted"),
    "uncertainty_aligned": ("uncertainty", "aligned"),
}
ARMS = tuple(ARM_CONFIG)
PROXIES = ("physics", "uncertainty")
DOMAINS = ("HAZE4K_INTERNAL", "ITS_UNASSIGNED", "OTS_UNASSIGNED", "NHHAZE_HISTORICAL")
PAIRED_DOMAINS = DOMAINS[:3]
PLANNED_COUNTS = {domain: 55 for domain in DOMAINS}
TRAINING_SCENES = 600
TRAINING_SEEDS = (20_260_803, 20_260_817, 20_260_831, 20_260_914, 20_260_928)
TRAINING_STEPS = 2000
TRAINING_BLOCK = 100
TRAINING_UNITS = len(ARMS) * len(TRAINING_SEEDS) * (TRAINING_STEPS // TRAINING_BLOCK)
EVALUATION_UNITS = sum(PLANNED_COUNTS.values()) * len(TRAINING_SEEDS)
TOTAL_UNITS = TRAINING_UNITS + EVALUATION_UNITS + 1
FORMAL_ITERATIONS = len(ARMS) * len(TRAINING_SEEDS) * TRAINING_STEPS + EVALUATION_UNITS
TRAIN_CROP = 256
MAX_EVAL_CROP = 512
UTILITY_MARGIN_DB = 0.10
BASELINE_MARGIN_DB = 0.05
PRECISION_HALF_WIDTH_DB = 0.39
SIMULTANEOUS_CRITICAL = 2.9551668474978343
NEAR_CLEAR_PSNR_DB = 30.0
NEAR_CLEAR_MEAN_FLOOR_DB = -0.05
NEGATIVE_TAIL_DB = -0.10
NEGATIVE_TAIL_LIMIT = 0.10
SEED_NEGATIVE_LIMIT_DB = -0.10
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
    for key in sorted(roles["development_screening"])[:55]:
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
        if [rank for rank, _ in values] != list(range(55)):
            raise RuntimeError(f"{dataset} does not expose the frozen first 55 allocation ranks")
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
    class ProxyAdapter(torch.nn.Module):
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

    return ProxyAdapter


def physics_proxy(torch, image):
    import torch.nn.functional as functional

    minimum = image.min(dim=1, keepdim=True).values
    maximum = image.max(dim=1, keepdim=True).values
    dark = -functional.max_pool2d(-minimum, 15, stride=1, padding=7)
    saturation = (maximum - minimum) / maximum.clamp_min(1.0e-4)
    luminance = 0.299 * image[:, :1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
    local_mean = functional.avg_pool2d(luminance, 9, stride=1, padding=4)
    contrast = (luminance - local_mean).abs().clamp(0.0, 1.0)
    dx = functional.pad(
        (luminance[:, :, :, 1:] - luminance[:, :, :, :-1]).abs(), (0, 1, 0, 0),
    )
    dy = functional.pad(
        (luminance[:, :, 1:, :] - luminance[:, :, :-1, :]).abs(), (0, 0, 0, 1),
    )
    gradient = (dx + dy).clamp(0.0, 1.0)
    return torch.cat((dark, saturation, contrast, gradient), dim=1)


def uncertainty_proxy(torch, model, image):
    with torch.no_grad():
        direct = model(image)[2].clamp(0.0, 1.0)
        flipped_input = torch.flip(image, dims=(3,))
        flipped = torch.flip(model(flipped_input)[2].clamp(0.0, 1.0), dims=(3,))
        disagreement = (direct - flipped).abs()
        luminance = (
            0.299 * disagreement[:, :1]
            + 0.587 * disagreement[:, 1:2]
            + 0.114 * disagreement[:, 2:3]
        )
    return torch.cat((disagreement, luminance), dim=1)


def permute_spatial_proxy(torch, value, keys: list[str]):
    if len(keys) != value.shape[0]:
        raise RuntimeError("proxy permutation keys do not match the batch")
    height, width = value.shape[-2:]
    permuted = []
    for index, key in enumerate(keys):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(sha256_text(f"proxy-permutation|{key}|{height}|{width}")[:16], 16))
        order = torch.randperm(height * width, generator=generator).to(value.device)
        flattened = value[index:index + 1].reshape(1, value.shape[1], -1)
        permuted.append(flattened.index_select(2, order).reshape(1, value.shape[1], height, width))
    return torch.cat(permuted, dim=0)


def proxy_for_arm(torch, model, image, arm: str, keys: list[str]):
    proxy, alignment = ARM_CONFIG[arm]
    with torch.no_grad():
        value = (
            physics_proxy(torch, image)
            if proxy == "physics"
            else uncertainty_proxy(torch, model, image)
        )
        if alignment == "permuted":
            value = permute_spatial_proxy(torch, value, keys)
    return value.detach()


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


def adapted_output(torch, model, adapter, image, arm: str, keys: list[str]):
    feature = final_feature(torch, model, image)
    proxy = proxy_for_arm(torch, model, image, arm, keys)
    gamma, beta = adapter(proxy)
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


def training_batch(
    torch, records: list[dict[str, Any]], seed: int, step: int,
):
    rng = np.random.default_rng(seed + step)
    indexes = rng.choice(len(records), size=2, replace=False)
    hazy_values, clear_values, keys = [], [], []
    for record_index in indexes:
        record = records[int(record_index)]
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
        flipped = bool(rng.integers(0, 2))
        if flipped:
            hazy, clear = hazy[:, ::-1], clear[:, ::-1]
        hazy_values.append(np.ascontiguousarray(hazy.transpose(2, 0, 1)))
        clear_values.append(np.ascontiguousarray(clear.transpose(2, 0, 1)))
        keys.append(
            f"train|{seed}|{step}|{record['scene_id']}|{observation.name}|"
            f"{y0}|{x0}|{int(flipped)}"
        )
    return (
        torch.from_numpy(np.stack(hazy_values)).to("cuda"),
        torch.from_numpy(np.stack(clear_values)).to("cuda"),
        keys,
    )


def psnr(torch, prediction, target) -> float:
    mse = float((prediction.double() - target.double()).square().mean().item())
    return -10.0 * math.log10(max(mse, EPSILON))


def evaluate_scene(
    torch, model, adapters, record: dict[str, Any], seed: int,
) -> dict[str, Any]:
    clear_full = image_array(record["clear"])
    observations = []
    input_parts = [
        record["dataset"], record["scene_id"], str(seed), sha256_file(record["clear"]),
    ]
    for observation in record["observations"]:
        hazy_full = image_array(observation)
        side = (min(hazy_full.shape[0], hazy_full.shape[1], MAX_EVAL_CROP) // 32) * 32
        if side < 256:
            raise RuntimeError("evaluation pair is below the frozen 256-pixel minimum")
        hazy, clear = crop_pair(hazy_full, clear_full, record["scene_id"], side)
        hazy_tensor = torch.from_numpy(hazy.transpose(2, 0, 1).copy()).unsqueeze(0).to("cuda")
        clear_tensor = torch.from_numpy(clear.transpose(2, 0, 1).copy()).unsqueeze(0).to("cuda")
        key = f"eval|{seed}|{record['dataset']}|{record['scene_id']}|{observation.name}"
        with torch.no_grad():
            baseline = model(hazy_tensor)[2].clamp(0.0, 1.0)
            outputs = {}
            modulation = {}
            for arm in ARMS:
                outputs[arm], gamma, beta = adapted_output(
                    torch, model, adapters[arm], hazy_tensor, arm, [key],
                )
                modulation[arm] = float((gamma.abs().mean() + beta.abs().mean()).item())
        scores = {"official": psnr(torch, baseline, clear_tensor)}
        scores.update({arm: psnr(torch, value, clear_tensor) for arm, value in outputs.items()})
        if not all(math.isfinite(value) for value in scores.values()):
            raise RuntimeError("raw scene PSNR is empty or non-finite")
        observations.append({"scores": scores, "modulation": modulation})
        input_parts.append(sha256_file(observation))
    mean = lambda key: float(np.mean([item["scores"][key] for item in observations]))
    result = {
        "dataset": record["dataset"],
        "scene_id": record["scene_id"],
        "seed": seed,
        "observation_count": len(observations),
        "baseline_psnr_db": mean("official"),
    }
    for proxy in PROXIES:
        aligned = f"{proxy}_aligned"
        permuted = f"{proxy}_permuted"
        result[f"{proxy}_actionability"] = mean(aligned) - mean(permuted)
        result[f"{proxy}_versus_baseline"] = mean(aligned) - mean("official")
        result[f"{proxy}_modulation"] = float(
            np.mean([item["modulation"][aligned] for item in observations])
        )
    result["proxy_interaction"] = (
        result["physics_actionability"] - result["uncertainty_actionability"]
    )
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


def collapse_scenes(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["dataset"], record["scene_id"])].append(record)
    collapsed = []
    for (dataset, scene_id), values in sorted(grouped.items()):
        observed_seeds = sorted(item["seed"] for item in values)
        if observed_seeds != list(TRAINING_SEEDS):
            raise RuntimeError("scene does not contain every frozen training-seed block")
        item = {
            "dataset": dataset,
            "scene_id": scene_id,
            "baseline_psnr_db": float(np.mean([row["baseline_psnr_db"] for row in values])),
        }
        for proxy in PROXIES:
            item[f"{proxy}_actionability"] = float(
                np.mean([row[f"{proxy}_actionability"] for row in values])
            )
            item[f"{proxy}_versus_baseline"] = float(
                np.mean([row[f"{proxy}_versus_baseline"] for row in values])
            )
            item[f"{proxy}_seed_effects"] = {
                str(row["seed"]): float(row[f"{proxy}_actionability"]) for row in values
            }
        item["proxy_interaction"] = (
            item["physics_actionability"] - item["uncertainty_actionability"]
        )
        collapsed.append(item)
    return collapsed


def seed_equal_domain(records: list[dict[str, Any]], proxy: str) -> dict[str, float]:
    result = {}
    for seed in TRAINING_SEEDS:
        seed_records = [item for item in records if item["seed"] == seed]
        domain_means = []
        for domain in DOMAINS:
            values = [
                item[f"{proxy}_actionability"]
                for item in seed_records if item["dataset"] == domain
            ]
            if len(values) != PLANNED_COUNTS[domain]:
                raise RuntimeError("seed block does not contain the frozen domain roster")
            domain_means.append(float(np.mean(values)))
        result[str(seed)] = float(np.mean(domain_means))
    return result


def selected_proxy(effect_summaries: dict[str, Any]) -> str:
    physics = effect_summaries["physics"]["overall_equal_domain"]["estimate"]
    uncertainty = effect_summaries["uncertainty"]["overall_equal_domain"]["estimate"]
    return "uncertainty" if uncertainty > physics + 0.005 else "physics"


def scientific_summaries(records: list[dict[str, Any]]) -> dict[str, Any]:
    collapsed = collapse_scenes(records)
    effects = {
        proxy: contrast_summary(
            collapsed, f"{proxy}_actionability", SIMULTANEOUS_CRITICAL,
        )
        for proxy in PROXIES
    }
    baseline = {
        proxy: contrast_summary(
            collapsed, f"{proxy}_versus_baseline", SIMULTANEOUS_CRITICAL,
        )
        for proxy in PROXIES
    }
    interaction = contrast_summary(
        collapsed, "proxy_interaction", SIMULTANEOUS_CRITICAL,
    )
    seed_blocks = {proxy: seed_equal_domain(records, proxy) for proxy in PROXIES}
    selected = selected_proxy(effects)

    proxy_status = {}
    for proxy in PROXIES:
        overall = effects[proxy]["overall_equal_domain"]
        domains = effects[proxy]["by_domain"]
        absolute = baseline[proxy]["overall_equal_domain"]
        seed_values = list(seed_blocks[proxy].values())
        favorable = (
            overall["lower"] > UTILITY_MARGIN_DB
            and sum(item["lower"] > 0.0 for item in domains.values()) >= 3
            and all(item["upper"] >= 0.0 for item in domains.values())
            and absolute["lower"] > BASELINE_MARGIN_DB
            and sum(value > 0.0 for value in seed_values) >= 4
            and min(seed_values) > SEED_NEGATIVE_LIMIT_DB
        )
        impossible = (
            overall["upper"] <= UTILITY_MARGIN_DB
            or absolute["upper"] <= BASELINE_MARGIN_DB
            or any(item["upper"] < 0.0 for item in domains.values())
            or sum(value > 0.0 for value in seed_values) <= 2
            or min(seed_values) <= SEED_NEGATIVE_LIMIT_DB
        )
        proxy_status[proxy] = {
            "favorable": favorable,
            "impossible": impossible,
            "positive_seed_blocks": sum(value > 0.0 for value in seed_values),
            "minimum_seed_effect_db": min(seed_values),
        }
    if any(item["favorable"] for item in proxy_status.values()):
        actionability_outcome = "favorable"
        favorable_proxies = [
            proxy for proxy in PROXIES if proxy_status[proxy]["favorable"]
        ]
        preferred = selected_proxy(effects)
        selected = (
            preferred
            if preferred in favorable_proxies
            else favorable_proxies[0]
        )
    elif all(item["impossible"] for item in proxy_status.values()):
        actionability_outcome = "unfavorable"
    else:
        actionability_outcome = "indeterminate"

    selected_baseline_key = f"{selected}_versus_baseline"
    near_records = [
        item for item in collapsed if item["baseline_psnr_db"] >= NEAR_CLEAR_PSNR_DB
    ]
    near_has_all_domains = all(
        any(item["dataset"] == domain for item in near_records) for domain in DOMAINS
    )
    near_mean = (
        contrast_summary(near_records, selected_baseline_key, SIMULTANEOUS_CRITICAL)
        if near_has_all_domains else None
    )
    near_tail = prevalence_summary(
        near_records,
        lambda item: item[selected_baseline_key] < NEGATIVE_TAIL_DB,
        DOMAINS,
        SIMULTANEOUS_CRITICAL,
    )
    overall_tail = prevalence_summary(
        collapsed,
        lambda item: item[selected_baseline_key] < NEGATIVE_TAIL_DB,
        DOMAINS,
        SIMULTANEOUS_CRITICAL,
    )
    safe = (
        near_mean is not None
        and near_mean["overall_equal_domain"]["lower"] >= NEAR_CLEAR_MEAN_FLOOR_DB
        and all(item["upper"] <= NEGATIVE_TAIL_LIMIT for item in near_tail["by_domain"].values())
        and all(item["upper"] <= NEGATIVE_TAIL_LIMIT for item in overall_tail["by_domain"].values())
    )
    unsafe = (
        (
            near_mean is not None
            and near_mean["overall_equal_domain"]["upper"] < NEAR_CLEAR_MEAN_FLOOR_DB
        )
        or any(item["lower"] > NEGATIVE_TAIL_LIMIT for item in near_tail["by_domain"].values())
        or any(item["lower"] > NEGATIVE_TAIL_LIMIT for item in overall_tail["by_domain"].values())
    )
    safety_outcome = "safe" if safe else "unsafe" if unsafe else "indeterminate"

    family_intervals = []
    for proxy in PROXIES:
        family_intervals.append(effects[proxy]["overall_equal_domain"])
        family_intervals.extend(effects[proxy]["by_domain"].values())
        family_intervals.append(baseline[proxy]["overall_equal_domain"])
    family_intervals.extend(interaction["by_domain"].values())
    if len(family_intervals) != 16:
        raise RuntimeError("simultaneous comparison family does not contain 16 intervals")
    precision_met = all(
        item["half_width"] <= PRECISION_HALF_WIDTH_DB for item in family_intervals
    )
    return {
        "collapsed_scene_count": len(collapsed),
        "effects": effects,
        "aligned_versus_official": baseline,
        "proxy_interaction": interaction,
        "seed_blocks": seed_blocks,
        "proxy_status": proxy_status,
        "selected_proxy": selected,
        "scientific_actionability_outcome": actionability_outcome,
        "safety_outcome": safety_outcome,
        "near_clear": {
            "count": len(near_records),
            "mean": near_mean,
            "negative_tail": near_tail,
        },
        "overall_negative_tail": overall_tail,
        "comparison_family_size": len(family_intervals),
        "precision_met": precision_met,
    }


def gate_outcomes(summaries: dict[str, Any]) -> dict[str, str]:
    return {
        "evidence_identity_integrity_coverage": "pass",
        "scientific_proxy_actionability": summaries["scientific_actionability_outcome"],
        "selected_proxy_safety": summaries["safety_outcome"],
        "primary_precision": "met" if summaries["precision_met"] else "unmet",
    }


def contract(context_path: Path) -> None:
    context = load_context(context_path, "contract")
    prepare_phase_output(context)
    import torch

    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    torch, model = load_official_model(context)
    ProxyAdapter = adapter_class(torch)
    train_hazy = torch.rand((2, 3, TRAIN_CROP, TRAIN_CROP), device=context.device)
    train_target = torch.rand_like(train_hazy)
    eval_hazy = torch.rand((1, 3, MAX_EVAL_CROP, MAX_EVAL_CROP), device=context.device)
    finite = True
    completed_iterations = 0
    equal_initialization = True
    last_adapters = None
    for seed in TRAINING_SEEDS:
        torch.manual_seed(seed)
        template = ProxyAdapter().to(context.device)
        template_state = {
            key: value.detach().clone() for key, value in template.state_dict().items()
        }
        adapters = {arm: ProxyAdapter().to(context.device) for arm in ARMS}
        for adapter in adapters.values():
            adapter.load_state_dict(template_state, strict=True)
        equal_initialization = equal_initialization and all(
            all(torch.equal(value, template_state[key]) for key, value in adapter.state_dict().items())
            for adapter in adapters.values()
        )
        for arm in ARMS:
            optimizer = torch.optim.AdamW(
                adapters[arm].parameters(), lr=2.0e-4,
                betas=(0.9, 0.999), weight_decay=0.0,
            )
            keys = [f"contract|{seed}|{arm}|0", f"contract|{seed}|{arm}|1"]
            for _ in range(TRAINING_STEPS):
                optimizer.zero_grad(set_to_none=True)
                prediction, gamma, beta = adapted_output(
                    torch, model, adapters[arm], train_hazy, arm, keys,
                )
                loss = training_loss(torch, prediction, train_target, gamma, beta)
                finite = finite and bool(torch.isfinite(loss).item())
                loss.backward()
                optimizer.step()
                completed_iterations += 1
                if completed_iterations % 100 == 0:
                    write_contract_progress(
                        context,
                        completed_iterations=completed_iterations,
                        total_iterations=FORMAL_ITERATIONS,
                        stage="synthetic_exact_training_map",
                    )
        last_adapters = adapters
    if last_adapters is None:
        raise RuntimeError("synthetic adapter set was not created")
    for index in range(EVALUATION_UNITS):
        key = f"contract-eval|{index}"
        with torch.no_grad():
            baseline = model(eval_hazy)[2]
            finite = finite and bool(torch.isfinite(baseline).all().item())
            for arm in ARMS:
                prediction, _, _ = adapted_output(
                    torch, model, last_adapters[arm], eval_hazy, arm, [key],
                )
                finite = finite and bool(torch.isfinite(prediction).all().item())
        completed_iterations += 1
        if completed_iterations % 25 == 0 or completed_iterations == FORMAL_ITERATIONS:
            write_contract_progress(
                context,
                completed_iterations=completed_iterations,
                total_iterations=FORMAL_ITERATIONS,
                stage="synthetic_exact_evaluation_map",
            )
    elapsed = time.monotonic() - started
    peak = float(torch.cuda.max_memory_allocated() / (1024 * 1024))
    checks = {
        "cuda_available": context.device == "cuda" and torch.cuda.is_available(),
        "official_anchor_identity": context.assets["official_anchor_checkout"].commit == ANCHOR_COMMIT,
        "official_parameter_count": sum(parameter.numel() for parameter in model.parameters()) == PARAMETER_COUNT,
        "equal_adapter_parameter_count": len({
            sum(parameter.numel() for parameter in adapter.parameters())
            for adapter in last_adapters.values()
        }) == 1,
        "identical_arm_initialization_within_seed": equal_initialization,
        "same_scale_iteration_count": (
            context.engineering_contract["cost_contract"]["formal_iterations"]
            == completed_iterations == FORMAL_ITERATIONS
        ),
        "finite_synthetic_training_and_evaluation": finite,
        "run_only_data_absent": all(identifier not in context.assets for identifier in (
            "s0_scene_role_ledger", "reside_role_ledger", "haze4k_train",
            "its_train_clear", "its_train_haze", "its_val_clear", "its_val_haze",
            "ots_clear", "ots_haze", "nhhaze",
        )),
    }
    engineering = {
        "mode": "gpu_synthetic_no_data",
        "device": context.device,
        "fixture": {
            "batch": 2, "channels": 3,
            "training_height": TRAIN_CROP, "training_width": TRAIN_CROP,
            "evaluation_height": MAX_EVAL_CROP, "evaluation_width": MAX_EVAL_CROP,
        },
        "production_path_exercised": True,
        "protected_data_touched": False,
        "scientific_output_created": False,
        "scientific_training_occurred": False,
        "cost": {
            "observed_iterations": completed_iterations,
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
        raise RuntimeError("proxy-factorial runtime role, unit, or protected-data contract changed")
    if load_completed_unit_ledger(context):
        raise RuntimeError("fresh proxy-factorial run unexpectedly contains completed units")

    roles_path = asset_path(context, "s0_scene_role_ledger", kind="file")
    reside_ledger = asset_path(context, "reside_role_ledger", kind="file")
    if sha256_file(roles_path) != S0_LEDGER_SHA256 \
            or sha256_file(reside_ledger) != ROLE_LEDGER_SHA256:
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
    counts = {
        domain: sum(item["dataset"] == domain for item in evaluation_records)
        for domain in DOMAINS
    }
    if counts != PLANNED_COUNTS or len(training_records) != TRAINING_SCENES:
        raise RuntimeError("frozen training or evaluation roster count changed")

    import torch

    torch, model = load_official_model(context)
    ProxyAdapter = adapter_class(torch)
    completed = 0
    roster_digest = sha256_text("\n".join(
        [item["scene_id"] for item in training_records]
        + [
            f"{item['dataset']}|{item['scene_id']}"
            for item in sorted(
                evaluation_records,
                key=lambda item: (item["dataset"], item["scene_id"]),
            )
        ]
    ))
    final_checkpoints: dict[tuple[int, str], Path] = {}
    for seed in TRAINING_SEEDS:
        torch.manual_seed(seed)
        template = ProxyAdapter().to(context.device)
        template_state = {
            key: value.detach().clone() for key, value in template.state_dict().items()
        }
        for arm in ARMS:
            adapter = ProxyAdapter().to(context.device)
            adapter.load_state_dict(template_state, strict=True)
            optimizer = torch.optim.AdamW(
                adapter.parameters(), lr=2.0e-4,
                betas=(0.9, 0.999), weight_decay=0.0,
            )
            adapter.train()
            for step in range(TRAINING_STEPS):
                hazy, clear, keys = training_batch(
                    torch, training_records, seed, step,
                )
                optimizer.zero_grad(set_to_none=True)
                prediction, gamma, beta = adapted_output(
                    torch, model, adapter, hazy, arm, keys,
                )
                loss = training_loss(torch, prediction, clear, gamma, beta)
                if not bool(torch.isfinite(loss).item()):
                    raise RuntimeError("adapter training produced a non-finite loss")
                loss.backward()
                optimizer.step()
                if (step + 1) % TRAINING_BLOCK == 0:
                    block = (step + 1) // TRAINING_BLOCK
                    relpath = f"training/{seed}/{arm}_{block:03d}.pt"
                    destination = output_file(context, relpath)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({
                        "seed": seed, "arm": arm, "step": step + 1,
                        "adapter": adapter.state_dict(),
                    }, destination)
                    record_completed_unit(
                        context,
                        unit_id=f"train_{seed}_{arm}_{block:03d}",
                        input_sha256=sha256_text(
                            f"{roster_digest}|{seed}|{arm}|{block}"
                        ),
                        output_relpath=relpath,
                    )
                    completed += 1
                    write_workload_progress(
                        context, completed_units=completed, stage="adapter_training",
                    )
                    if step + 1 == TRAINING_STEPS:
                        final_checkpoints[(seed, arm)] = destination
            del optimizer
            del adapter

    measured: list[dict[str, Any]] = []
    for seed in TRAINING_SEEDS:
        adapters = {}
        for arm in ARMS:
            adapter = ProxyAdapter().to(context.device)
            try:
                payload = torch.load(
                    final_checkpoints[(seed, arm)],
                    map_location=context.device,
                    weights_only=True,
                )
            except TypeError:
                payload = torch.load(
                    final_checkpoints[(seed, arm)], map_location=context.device,
                )
            if payload.get("seed") != seed or payload.get("arm") != arm:
                raise RuntimeError("final adapter checkpoint identity changed")
            adapter.load_state_dict(payload["adapter"], strict=True)
            adapters[arm] = adapter.eval()
        for record in evaluation_records:
            result = evaluate_scene(torch, model, adapters, record, seed)
            input_sha = result.pop("input_sha256")
            unit_hash = sha256_text(
                f"{seed}|{record['dataset']}|{record['scene_id']}"
            )[:24]
            unit_id = f"eval_{unit_hash}"
            relpath = f"evaluation/{seed}/{unit_id}.json"
            atomic_json(output_file(context, relpath), result)
            record_completed_unit(
                context, unit_id=unit_id,
                input_sha256=input_sha, output_relpath=relpath,
            )
            measured.append(result)
            completed += 1
            write_workload_progress(
                context, completed_units=completed, stage="development_evaluation",
            )
        del adapters

    summaries = scientific_summaries(measured)
    outcomes = gate_outcomes(summaries)
    aggregate = {
        "schema_version": 1,
        "route_id": ROUTE_ID,
        "operation_id": OPERATION_ID,
        "roster_digest": roster_digest,
        "planned_counts": PLANNED_COUNTS,
        "seed_blocks": list(TRAINING_SEEDS),
        "evaluated_scene_seed_units": len(measured),
        "gate_outcomes": outcomes,
        "selected_proxy": summaries["selected_proxy"],
    }
    aggregate_relpath = "aggregate/decision.json"
    atomic_json(output_file(context, aggregate_relpath), aggregate)
    record_completed_unit(
        context, unit_id="aggregate_decision",
        input_sha256=sha256_text(
            f"{roster_digest}|aggregate|{len(measured)}|{summaries['selected_proxy']}"
        ),
        output_relpath=aggregate_relpath,
    )
    completed += 1
    if completed != TOTAL_UNITS or len(load_completed_unit_ledger(context)) != TOTAL_UNITS:
        raise RuntimeError("completed-unit ledger does not exactly cover the frozen workload")
    write_workload_progress(
        context, completed_units=completed, stage="aggregate_complete",
    )

    summary = {
        "schema_version": 1,
        "route_id": ROUTE_ID,
        "operation_id": OPERATION_ID,
        "run_id": context.run_id,
        "design": {
            "arms": ["official_baseline", *ARMS],
            "factors": {
                "proxy_class": list(PROXIES),
                "alignment": ["permuted", "aligned"],
            },
            "training_steps_per_arm_seed": TRAINING_STEPS,
            "training_seed_blocks": list(TRAINING_SEEDS),
            "batch_size": 2,
            "crop_size": TRAIN_CROP,
            "minimum_meaningful_actionability_db": UTILITY_MARGIN_DB,
            "minimum_aligned_baseline_utility_db": BASELINE_MARGIN_DB,
            "precision_half_width_db": PRECISION_HALF_WIDTH_DB,
            "simultaneous_critical_value": SIMULTANEOUS_CRITICAL,
            "confidence_level": 0.95,
            "scene_effect_clipping": None,
            "outcome_look": "terminal_only",
        },
        "roster": aggregate,
        "summaries": summaries,
        "gate_outcomes": outcomes,
        "protected_data_touched": False,
        "confirmation_or_sealed_data_touched": False,
        "marker": "DAYTIME_DEHAZING_OBSERVABLE_PROXY_DOMAIN_ACTIONABILITY_FACTORIAL_V1_COMPLETE",
    }
    atomic_json(output_file(context, SUMMARY_NAME), summary)
    gate_summary = {
        "schema_version": 1,
        "route_id": ROUTE_ID,
        "operation_id": OPERATION_ID,
        "gate_outcomes": outcomes,
        "selected_proxy": summaries["selected_proxy"],
        "effects": summaries["effects"],
        "aligned_versus_official": summaries["aligned_versus_official"],
        "proxy_interaction": summaries["proxy_interaction"],
        "seed_blocks": summaries["seed_blocks"],
        "proxy_status": summaries["proxy_status"],
        "near_clear": summaries["near_clear"],
        "overall_negative_tail": summaries["overall_negative_tail"],
        "comparison_family_size": summaries["comparison_family_size"],
    }
    atomic_json(output_file(context, GATE_NAME), gate_summary)
    source_sha = sha256_file(output_file(context, SUMMARY_NAME))
    selected = summaries["selected_proxy"]
    primary = summaries["effects"][selected]["overall_equal_domain"]
    primary_prefix = f"/summaries/effects/{selected}/overall_equal_domain"
    facts = [
        build_primary_review_fact(
            fact_id="selected_proxy_actionability_primary",
            claim_id="observable_proxy_spatial_actionability",
            metric="selected proxy aligned-minus-permuted equal-domain scene PSNR",
            unit="dB",
            population="220 frozen development-only original clear scenes",
            grouping=(
                "original clear scene; haze observations and five training-seed "
                "blocks are nested; four domains receive equal weight"
            ),
            point=primary["estimate"],
            point_pointer=f"{primary_prefix}/estimate",
            ci_lower=primary["lower"],
            ci_upper=primary["upper"],
            confidence_level=0.95,
            ci_lower_pointer=f"{primary_prefix}/lower",
            ci_upper_pointer=f"{primary_prefix}/upper",
            confidence_level_pointer="/design/confidence_level",
            threshold=UTILITY_MARGIN_DB,
            threshold_operator=">",
            threshold_pointer="/design/minimum_meaningful_actionability_db",
            gate_outcome=outcomes["scientific_proxy_actionability"],
            gate_pointer="/gate_outcomes/scientific_proxy_actionability",
            source_filename=SUMMARY_NAME,
            source_sha256=source_sha,
        ),
        *[
        build_gate_review_fact(
            fact_id=gate_id,
            metric=f"{gate_id} typed gate outcome",
            unit="typed outcome",
            population="220 frozen development-only original clear scenes",
            grouping=(
                "original clear scene; haze observations and five training-seed "
                "blocks are nested; one terminal-only outcome look"
            ),
            gate_outcome=outcome,
            source_filename=SUMMARY_NAME,
            source_sha256=source_sha,
        )
        for gate_id, outcome in outcomes.items()
        ],
    ]
    write_scientific_review_facts(context, relpath=FACTS_NAME, facts=facts)
    write_gate_result(
        context,
        gate_outcomes=outcomes,
        details={
            "summary_file": SUMMARY_NAME,
            "gate_summary_file": GATE_NAME,
            "review_facts_file": FACTS_NAME,
            "selected_proxy": summaries["selected_proxy"],
            "evaluated_scene_seed_units": len(measured),
            "independent_scene_count": summaries["collapsed_scene_count"],
            "training_steps_per_arm_seed": TRAINING_STEPS,
            "training_seed_count": len(TRAINING_SEEDS),
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
