#!/usr/bin/env python3
"""Run the frozen seed-blocked CARL by density-consistency 2x2 factorial."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

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


ROUTE_ID = "daytime-dehazing-density-consistency-carl-multiseed-factorial-v1"
OPERATION_ID = "DAYTIME_DEHAZING_DENSITY_CONSISTENCY_CARL_MULTISEED_FACTORIAL_QUALIFY"
ANCHOR_COMMIT = "3b4da35440c8c26a7d1bcaf1daf342e11d9a3898"
CHECKPOINT_SHA256 = "6f42037d57a4e3de3a10ac0ab909d66a3415864a19433c29204a975f4efa4088"
MODEL_SOURCE_SHA256 = "9681defa95c6602d0e0abae05e635771bd65d48a18555198ff00d573242a0005"
MODEL_LAYERS_SHA256 = "ac8a05bd626d9adda16308dedb9466f36d7ff44cfb666f64e7e14ddf8cdf43a4"
VGG19_SHA256 = "dcbb9e9dad569fff7a846263a77324fc34978fea2bfb039c012d710e1776ae44"
VGG19_SIZE_BYTES = 574_673_361
S0_LEDGER_SHA256 = "4cff8e7aecea5d8e19165ac4e725f69746342521115a3ab1f03ea1474f280960"

ARMS = ("b00_baseline", "b10_carl", "b01_consistency", "b11_combined")
SEEDS = (20_260_803, 20_260_817, 20_260_831, 20_260_914, 20_260_928)
FIRST_LOOK_SEED_COUNT = 3
TRAINING_SCENES = 600
DEVELOPMENT_SCENES = 150
OBSERVATIONS_PER_SCENE = 4
TRAINING_STEPS = 2000
FORMAL_ITERATIONS = len(SEEDS) * len(ARMS) * TRAINING_STEPS
PROBE_STEPS_PER_CELL = 200
PROBE_ITERATIONS = len(SEEDS) * len(ARMS) * PROBE_STEPS_PER_CELL
TOTAL_UNITS = len(SEEDS) * len(ARMS) * 2 + 1
BATCH_SIZE = 2
TRAIN_CROP = 256
EVAL_CROP = 256
DESCRIPTIVE_CLIP_DB = 0.25
MEANINGFUL_EFFECT_DB = 0.10
MECHANISM_EQUIVALENCE_DB = 0.05
WORST_STRATUM_FLOOR_DB = -0.10
CRITICAL_VALUE = 2.865
PRECISION_HALF_WIDTH_DB = 0.260
EMA_DECAY = 0.999
LAMBDA_CONSISTENCY = 1.0
LAMBDA_CARL = 10.0
CARL_MARGIN = 0.5
VGG_LAYER_INDICES = (1, 3, 5, 9, 13)
VGG_LAYER_WEIGHTS = (1.0 / 32.0, 1.0 / 16.0, 1.0 / 8.0, 1.0 / 4.0, 1.0)
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def require_context(context: Any) -> None:
    if context.route_id != ROUTE_ID or context.operation_id != OPERATION_ID:
        raise RuntimeError("multiseed route or operation identity mismatch")
    if context.total_units != TOTAL_UNITS:
        raise RuntimeError("multiseed workload total differs from the frozen contract")
    if any(context.protected_data_permissions.values()):
        raise RuntimeError("multiseed route does not permit protected-data access")


def verify_file_asset(
    context: Any,
    asset_id: str,
    expected_sha: str,
    *,
    size: int | None = None,
) -> Path:
    asset = context.assets.get(asset_id)
    if asset is None or asset.kind != "file" or asset.sha256 != expected_sha:
        raise RuntimeError(f"bound asset identity changed: {asset_id}")
    path = asset_path(context, asset_id, kind="file")
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"bound asset is unavailable: {asset_id}")
    if size is not None and path.stat().st_size != size:
        raise RuntimeError(f"bound asset size changed: {asset_id}")
    if sha256_file(path) != expected_sha:
        raise RuntimeError(f"bound asset bytes changed: {asset_id}")
    return path


def verify_anchor(context: Any) -> Path:
    asset = context.assets.get("official_anchor_checkout")
    if asset is None or asset.kind != "git_checkout" or asset.commit != ANCHOR_COMMIT:
        raise RuntimeError("official anchor identity changed")
    anchor = asset_path(context, "official_anchor_checkout", kind="git_checkout")
    completed = subprocess.run(
        ["/usr/bin/git", "-C", str(anchor), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if completed.returncode or completed.stdout.strip() != ANCHOR_COMMIT:
        raise RuntimeError("official anchor checkout HEAD differs from the frozen commit")
    return anchor


def load_vgg19(context: Any, torch: Any) -> Any:
    path = verify_file_asset(
        context,
        "vgg19_imagenet1k_v1",
        VGG19_SHA256,
        size=VGG19_SIZE_BYTES,
    )
    import torchvision

    network = torchvision.models.vgg19(weights=None)
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict):
        raise RuntimeError("the bound VGG-19 state dict is invalid")
    network.load_state_dict(state, strict=True)
    features = network.features[: max(VGG_LAYER_INDICES) + 1]
    features.requires_grad_(False).to(context.device).eval()
    return features


def load_convir(context: Any, torch: Any, *, trainable: bool) -> Any:
    anchor = verify_anchor(context)
    checkpoint = verify_file_asset(context, "official_checkpoint", CHECKPOINT_SHA256)
    source = verify_file_asset(context, "model_source", MODEL_SOURCE_SHA256)
    layers = verify_file_asset(context, "model_layers", MODEL_LAYERS_SHA256)
    if str(anchor) not in sys.path:
        sys.path.insert(0, str(anchor))
    from Dehazing.ITS.models.ConvIR import build_net

    module = sys.modules[build_net.__module__]
    layer_module = sys.modules.get("Dehazing.ITS.models.layers")
    if (
        Path(module.__file__).resolve() != source.resolve()
        or layer_module is None
        or Path(layer_module.__file__).resolve() != layers.resolve()
    ):
        raise RuntimeError("ConvIR import escaped the bound official source")
    model = build_net("base", "Haze4K", fam_mode="original")
    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location="cpu")
    if not isinstance(state, dict) or not isinstance(state.get("model"), dict):
        raise RuntimeError("official checkpoint does not expose a model state dict")
    model.load_state_dict(state["model"], strict=True)
    model.to(context.device)
    model.train(mode=trainable)
    model.requires_grad_(trainable)
    return model


def unwrap_prediction(value: Any) -> Any:
    if isinstance(value, (tuple, list)):
        value = value[-1]
    if not hasattr(value, "ndim") or value.ndim != 4 or value.shape[1] != 3:
        raise RuntimeError("ConvIR prediction has an invalid RGB tensor contract")
    return value


def normalized_vgg_input(torch: Any, value: Any) -> Any:
    mean = torch.tensor((0.485, 0.456, 0.406), device=value.device).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device=value.device).view(1, 3, 1, 1)
    return (value.clamp(0.0, 1.0) - mean) / std


def feature_maps(torch: Any, features: Any, value: Any) -> list[Any]:
    value = normalized_vgg_input(torch, value)
    result = []
    for index, layer in enumerate(features):
        value = layer(value)
        if index in VGG_LAYER_INDICES:
            result.append(value)
    if len(result) != len(VGG_LAYER_INDICES):
        raise RuntimeError("VGG-19 feature layer contract changed")
    return result


def carl_loss(
    torch: Any,
    features: Any,
    prediction: Any,
    clear: Any,
    hazy: list[Any],
) -> Any:
    """Fixed K=4 triplet-margin transfer, not a paper reproduction."""
    import torch.nn.functional as functional

    prediction_maps = feature_maps(torch, features, prediction)
    with torch.no_grad():
        clear_maps = feature_maps(torch, features, clear)
        hazy_maps = [feature_maps(torch, features, item) for item in hazy]
    loss = prediction.new_zeros(())
    for weight, predicted, target, *negative_layers in zip(
        VGG_LAYER_WEIGHTS,
        prediction_maps,
        clear_maps,
        *hazy_maps,
    ):
        positive = functional.l1_loss(predicted, target)
        negative = torch.stack(
            [functional.l1_loss(predicted, item) for item in negative_layers]
        ).mean()
        loss = loss + weight * torch.relu(positive - negative + CARL_MARGIN)
    return loss


def image_files(directory: Path) -> list[Path]:
    return sorted(
        item
        for item in directory.iterdir()
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
    )


def canonical_rgb_digest(path: Path) -> str:
    from PIL import Image
    import numpy as np

    with Image.open(path) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        payload = np.asarray(rgb, dtype=np.uint8).tobytes()
    digest = hashlib.sha256()
    digest.update(width.to_bytes(8, "big"))
    digest.update(height.to_bytes(8, "big"))
    digest.update(payload)
    return digest.hexdigest()


def selected_label(image_name: str, label_dir: Path) -> Path | None:
    stem, suffix = os.path.splitext(image_name)
    choices = [image_name]
    if "_" in stem:
        prefix = stem.split("_", 1)[0]
        choices.extend((f"{prefix}{suffix}", f"{prefix}.png", f"{prefix}.jpg"))
    for name in dict.fromkeys(choices):
        candidate = label_dir / name
        if candidate.is_file():
            return candidate
    return None


def read_roles(context: Any) -> dict[str, set[str]]:
    ledger = verify_file_asset(context, "s0_scene_role_ledger", S0_LEDGER_SHA256)
    roles = {"training": set(), "development_screening": set()}
    with ledger.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("dataset") != "HAZE4K_TRAIN":
                continue
            role = row.get("role")
            scene = row.get("scene_id")
            digest = row.get("canonical_digest")
            if role in roles and isinstance(scene, str) and scene == digest:
                roles[role].add(scene)
    if (
        len(roles["training"]) != TRAINING_SCENES
        or len(roles["development_screening"]) != DEVELOPMENT_SCENES
    ):
        raise RuntimeError("the frozen Haze4K 600/150 role split is unavailable")
    return roles


def enumerate_haze4k(
    context: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    root = asset_path(context, "haze4k_train", kind="directory")
    roles = read_roles(context)
    input_dirs = [
        root / name for name in ("IN", "haze", "hazy") if (root / name).is_dir()
    ]
    label_dirs = [
        root / name for name in ("GT", "gt", "clear") if (root / name).is_dir()
    ]
    if len(input_dirs) != 1 or len(label_dirs) != 1:
        raise RuntimeError("Haze4K root lacks one input and one target directory")
    groups: dict[str, dict[str, Any]] = {}
    clear_cache: dict[Path, str] = {}
    for hazy in image_files(input_dirs[0]):
        clear = selected_label(hazy.name, label_dirs[0])
        if clear is None:
            continue
        scene_id = clear_cache.setdefault(clear, canonical_rgb_digest(clear))
        group = groups.setdefault(
            scene_id,
            {"scene_id": scene_id, "clear": clear, "observations": []},
        )
        group["observations"].append(hazy)
    if len(groups) != 750 or any(
        len(item["observations"]) != OBSERVATIONS_PER_SCENE
        for item in groups.values()
    ):
        raise RuntimeError(
            "Haze4K does not preserve 750 clear scenes with four observations each"
        )
    if set(groups) != roles["training"] | roles["development_screening"]:
        raise RuntimeError("Haze4K scene identities differ from the bound role ledger")
    training = [groups[key] for key in sorted(roles["training"])]
    development = [groups[key] for key in sorted(roles["development_screening"])]
    return training, development


def load_rgb(path: Path, *, crop: int, crop_key: str, center: bool) -> Any:
    from PIL import Image
    import numpy as np

    with Image.open(path) as image:
        value = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    height, width = value.shape[:2]
    if height < crop or width < crop:
        raise RuntimeError(
            f"image is smaller than the frozen {crop}-pixel crop: {path.name}"
        )
    if center:
        top = (height - crop) // 2
        left = (width - crop) // 2
    else:
        digest = hashlib.sha256(crop_key.encode("utf-8")).digest()
        top = int.from_bytes(digest[:8], "big") % (height - crop + 1)
        left = int.from_bytes(digest[8:16], "big") % (width - crop + 1)
    return value[top : top + crop, left : left + crop]


def to_tensor(torch: Any, array: Any, device: str) -> Any:
    import numpy as np

    contiguous = np.ascontiguousarray(array.transpose(2, 0, 1))
    return torch.from_numpy(contiguous).unsqueeze(0).to(device)


def scene_batch(
    torch: Any,
    group: dict[str, Any],
    *,
    crop: int,
    crop_key: str,
    center: bool,
    device: str,
) -> tuple[Any, list[Any]]:
    clear = load_rgb(group["clear"], crop=crop, crop_key=crop_key, center=center)
    hazy = [
        load_rgb(path, crop=crop, crop_key=crop_key, center=center)
        for path in sorted(group["observations"])
    ]
    return to_tensor(torch, clear, device), [
        to_tensor(torch, item, device) for item in hazy
    ]


def density_proxy(group: dict[str, Any]) -> float:
    import numpy as np

    clear = load_rgb(
        group["clear"],
        crop=EVAL_CROP,
        crop_key=group["scene_id"],
        center=True,
    )
    values = []
    for hazy in group["observations"]:
        value = load_rgb(
            hazy,
            crop=EVAL_CROP,
            crop_key=group["scene_id"],
            center=True,
        )
        values.append(float(np.mean(np.square(value - clear))))
    return float(sum(values) / len(values))


def fixed_strata(
    development: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    ranked = sorted(
        development,
        key=lambda item: (density_proxy(item), item["scene_id"]),
    )
    if len(ranked) != DEVELOPMENT_SCENES:
        raise RuntimeError("development roster count differs from the frozen contract")
    return {
        "DENSITY_LOW": ranked[:50],
        "DENSITY_MID": ranked[50:100],
        "DENSITY_HIGH": ranked[100:],
    }


def psnr(torch: Any, prediction: Any, target: Any) -> float:
    mse = float(
        torch.mean(torch.square(prediction.clamp(0.0, 1.0) - target)).item()
    )
    return 99.0 if mse <= 1.0e-12 else -10.0 * math.log10(mse)


def update_ema(teacher: Any, student: Any) -> None:
    for target, source in zip(teacher.parameters(), student.parameters()):
        target.mul_(EMA_DECAY).add_(source, alpha=1.0 - EMA_DECAY)


def write_unit(context: Any, unit_id: str, payload: dict[str, Any]) -> None:
    relpath = f"units/{unit_id}.json"
    atomic_json(output_file(context, relpath), payload)
    record_completed_unit(
        context,
        unit_id=unit_id,
        input_sha256=canonical_sha256(payload["input"]),
        output_relpath=relpath,
    )


def train_arm(
    context: Any,
    torch: Any,
    features: Any,
    training: list[dict[str, Any]],
    arm: str,
    seed: int,
) -> Path:
    import torch.nn.functional as functional

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = load_convir(context, torch, trainable=True)
    teacher = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2.0e-4,
        betas=(0.9, 0.999),
        weight_decay=0.0,
    )
    for step in range(TRAINING_STEPS):
        group = training[step % len(training)]
        crop_key = f"{group['scene_id']}|{seed}|{step}"
        clear_single, hazy_single = scene_batch(
            torch,
            group,
            crop=TRAIN_CROP,
            crop_key=crop_key,
            center=False,
            device=context.device,
        )
        clear = torch.cat([clear_single, clear_single], dim=0)
        hazy = [torch.cat([item, item], dim=0) for item in hazy_single]
        primary = hazy[step % OBSERVATIONS_PER_SCENE]
        paired = hazy[
            (step // OBSERVATIONS_PER_SCENE + 1) % OBSERVATIONS_PER_SCENE
        ]
        prediction = unwrap_prediction(model(primary))
        loss = functional.l1_loss(prediction, clear)
        if arm in {"b01_consistency", "b11_combined"}:
            with torch.no_grad():
                teacher_prediction = unwrap_prediction(teacher(paired))
            loss = loss + LAMBDA_CONSISTENCY * functional.l1_loss(
                prediction,
                teacher_prediction,
            )
        if arm in {"b10_carl", "b11_combined"}:
            loss = loss + LAMBDA_CARL * carl_loss(
                torch,
                features,
                prediction,
                clear,
                hazy,
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            update_ema(teacher, model)
    checkpoint = output_file(context, f"checkpoints/seed_{seed}/{arm}.pt")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "arm": arm, "seed": seed}, checkpoint)
    unit_id = f"train_s{seed}_{arm}"
    payload = {
        "input": {
            "kind": "training_cell",
            "arm": arm,
            "seed": seed,
            "steps": TRAINING_STEPS,
            "training_scene_count": len(training),
        },
        "checkpoint": checkpoint.name,
    }
    record_completed_unit(
        context,
        unit_id=unit_id,
        input_sha256=canonical_sha256(payload["input"]),
        output_relpath=f"checkpoints/seed_{seed}/{arm}.pt",
    )
    return checkpoint


def load_trained_arm(
    context: Any,
    torch: Any,
    arm: str,
    seed: int,
    checkpoint: Path,
) -> Any:
    model = load_convir(context, torch, trainable=False)
    try:
        state = torch.load(
            checkpoint,
            map_location=context.device,
            weights_only=True,
        )
    except TypeError:
        state = torch.load(checkpoint, map_location=context.device)
    if (
        not isinstance(state, dict)
        or state.get("arm") != arm
        or state.get("seed") != seed
        or not isinstance(state.get("model"), dict)
    ):
        raise RuntimeError(f"multiseed training checkpoint is invalid: {seed}/{arm}")
    model.load_state_dict(state["model"], strict=True)
    return model.eval()


def evaluate_arm(
    context: Any,
    torch: Any,
    arm: str,
    seed: int,
    model: Any,
    strata: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    with torch.no_grad():
        for stratum, groups in strata.items():
            for group in groups:
                clear, hazy = scene_batch(
                    torch,
                    group,
                    crop=EVAL_CROP,
                    crop_key=group["scene_id"],
                    center=True,
                    device=context.device,
                )
                observation_scores = [
                    psnr(torch, unwrap_prediction(model(item)), clear)
                    for item in hazy
                ]
                results[group["scene_id"]] = {
                    "score": float(sum(observation_scores) / len(observation_scores)),
                    "stratum": stratum,
                    "observation_scores": observation_scores,
                }
    unit_id = f"eval_s{seed}_{arm}"
    relpath = f"evaluations/seed_{seed}/{arm}.json"
    payload = {
        "input": {
            "kind": "evaluation_cell",
            "arm": arm,
            "seed": seed,
            "scene_count": DEVELOPMENT_SCENES,
            "observations_per_scene": OBSERVATIONS_PER_SCENE,
        },
        "results": results,
    }
    atomic_json(output_file(context, relpath), payload)
    record_completed_unit(
        context,
        unit_id=unit_id,
        input_sha256=canonical_sha256(payload["input"]),
        output_relpath=relpath,
    )
    return results


def interval(values: list[float]) -> dict[str, float]:
    if not values or any(not math.isfinite(value) for value in values):
        raise RuntimeError("raw grouped contrast is empty or non-finite")
    point = float(sum(values) / len(values))
    if len(values) == 1:
        half_width = float("inf")
    else:
        variance = sum((value - point) ** 2 for value in values) / (len(values) - 1)
        half_width = CRITICAL_VALUE * math.sqrt(variance / len(values))
    return {
        "point": point,
        "ci_lower": point - half_width,
        "ci_upper": point + half_width,
        "half_width": half_width,
    }


def clip_descriptive(value: float) -> float:
    return max(-DESCRIPTIVE_CLIP_DB, min(DESCRIPTIVE_CLIP_DB, value))


def summarize(
    scores: dict[int, dict[str, dict[str, dict[str, Any]]]],
    completed_seeds: list[int],
) -> dict[str, Any]:
    if not completed_seeds:
        raise RuntimeError("no completed seed block")
    scene_ids = sorted(scores[completed_seeds[0]]["b00_baseline"])
    per_scene: dict[str, dict[str, Any]] = {}
    seed_combined: dict[str, list[float]] = {
        str(seed): [] for seed in completed_seeds
    }
    for scene_id in scene_ids:
        seed_terms = []
        stratum = scores[completed_seeds[0]]["b00_baseline"][scene_id]["stratum"]
        for seed in completed_seeds:
            if any(
                set(scores[seed][arm]) != set(scene_ids)
                for arm in ARMS
            ):
                raise RuntimeError(
                    f"seed {seed} arms do not cover the frozen scene roster"
                )
            values = {
                arm: scores[seed][arm][scene_id]["score"] for arm in ARMS
            }
            combined = values["b11_combined"] - values["b00_baseline"]
            seed_combined[str(seed)].append(combined)
            seed_terms.append(
                {
                    "combined": combined,
                    "carl_main": 0.5
                    * (
                        values["b10_carl"]
                        - values["b00_baseline"]
                        + values["b11_combined"]
                        - values["b01_consistency"]
                    ),
                    "consistency_main": 0.5
                    * (
                        values["b01_consistency"]
                        - values["b00_baseline"]
                        + values["b11_combined"]
                        - values["b10_carl"]
                    ),
                    "interaction": (
                        values["b11_combined"]
                        - values["b01_consistency"]
                        - values["b10_carl"]
                        + values["b00_baseline"]
                    ),
                }
            )
        per_scene[scene_id] = {
            "stratum": stratum,
            "combined": float(
                sum(item["combined"] for item in seed_terms) / len(seed_terms)
            ),
            "carl_main": float(
                sum(item["carl_main"] for item in seed_terms) / len(seed_terms)
            ),
            "consistency_main": float(
                sum(item["consistency_main"] for item in seed_terms)
                / len(seed_terms)
            ),
            "interaction": float(
                sum(item["interaction"] for item in seed_terms) / len(seed_terms)
            ),
            "clipped_sensitivity": float(
                sum(clip_descriptive(item["combined"]) for item in seed_terms)
                / len(seed_terms)
            ),
        }
    stratum_results = {}
    for stratum in ("DENSITY_LOW", "DENSITY_MID", "DENSITY_HIGH"):
        values = [
            item["combined"]
            for item in per_scene.values()
            if item["stratum"] == stratum
        ]
        if len(values) != 50:
            raise RuntimeError(f"frozen density stratum coverage differs: {stratum}")
        stratum_results[stratum] = interval(values)
    return {
        "confidence_level": 0.95,
        "critical_value": CRITICAL_VALUE,
        "meaningful_effect_db": MEANINGFUL_EFFECT_DB,
        "mechanism_equivalence_margin_db": MECHANISM_EQUIVALENCE_DB,
        "completed_seeds": completed_seeds,
        "planned_seeds": list(SEEDS),
        "combined_vs_baseline": interval(
            [item["combined"] for item in per_scene.values()]
        ),
        "carl_main_effect": interval(
            [item["carl_main"] for item in per_scene.values()]
        ),
        "consistency_main_effect": interval(
            [item["consistency_main"] for item in per_scene.values()]
        ),
        "interaction": interval(
            [item["interaction"] for item in per_scene.values()]
        ),
        "density_strata": stratum_results,
        "seed_specific_combined": {
            seed: interval(values) for seed, values in seed_combined.items()
        },
        "clipped_sensitivity": interval(
            [item["clipped_sensitivity"] for item in per_scene.values()]
        ),
    }


def is_early_futility(summary: dict[str, Any]) -> bool:
    if summary["combined_vs_baseline"]["ci_upper"] <= MEANINGFUL_EFFECT_DB:
        return True
    return any(
        value["ci_upper"] <= 0.0
        for value in summary["seed_specific_combined"].values()
    )


def skip_seed(context: Any, seed: int) -> None:
    for arm in ARMS:
        for phase in ("train", "eval"):
            unit_id = f"{phase}_s{seed}_{arm}"
            write_unit(
                context,
                unit_id,
                {
                    "input": {
                        "kind": f"skipped_{phase}_cell",
                        "arm": arm,
                        "seed": seed,
                        "reason": "predeclared_three_seed_nonbinding_futility",
                    },
                    "status": "SKIPPED_PREDECLARED_FUTILITY",
                },
            )


def mechanism_classification(value: dict[str, float]) -> str:
    if value["ci_lower"] > MECHANISM_EQUIVALENCE_DB:
        return "material_positive"
    if value["ci_upper"] < -MECHANISM_EQUIVALENCE_DB:
        return "material_negative"
    if (
        value["ci_lower"] >= -MECHANISM_EQUIVALENCE_DB
        and value["ci_upper"] <= MECHANISM_EQUIVALENCE_DB
    ):
        return "equivalent_small"
    return "unresolved"


def gate_outcomes(
    summary: dict[str, Any],
    ledger_count: int,
    *,
    early_futility: bool,
) -> tuple[dict[str, str], dict[str, str]]:
    primary = summary["combined_vs_baseline"]
    seed_values = summary["seed_specific_combined"].values()
    strata = summary["density_strata"].values()
    mechanism = {
        "carl_main_effect": mechanism_classification(
            summary["carl_main_effect"]
        ),
        "consistency_main_effect": mechanism_classification(
            summary["consistency_main_effect"]
        ),
        "interaction": mechanism_classification(summary["interaction"]),
    }
    complete_terminal = len(summary["completed_seeds"]) == len(SEEDS)
    if early_futility or primary["ci_upper"] <= MEANINGFUL_EFFECT_DB:
        combined = "unfavorable"
    elif complete_terminal and primary["ci_lower"] > MEANINGFUL_EFFECT_DB:
        combined = "favorable"
    else:
        combined = "indeterminate"
    if any(value["ci_upper"] <= 0.0 for value in seed_values):
        seed_outcome = "unfavorable"
    elif complete_terminal and all(value["ci_lower"] > 0.0 for value in seed_values):
        seed_outcome = "favorable"
    else:
        seed_outcome = "indeterminate"
    if complete_terminal and all(
        value != "unresolved" for value in mechanism.values()
    ):
        mechanism_outcome = "favorable"
    else:
        mechanism_outcome = "indeterminate"
    worst_lower = min(value["ci_lower"] for value in strata)
    worst_upper = min(value["ci_upper"] for value in strata)
    if worst_lower >= WORST_STRATUM_FLOOR_DB:
        safety = "safe"
    elif worst_upper < WORST_STRATUM_FLOOR_DB:
        safety = "unsafe"
    else:
        safety = "indeterminate"
    outcomes = {
        "identity_integrity": "pass",
        "workload_coverage": "pass" if ledger_count == TOTAL_UNITS else "fail",
        "primary_precision": (
            "met"
            if complete_terminal
            and all(
                value["half_width"] <= PRECISION_HALF_WIDTH_DB for value in strata
            )
            else "unmet"
        ),
        "combined_materiality": combined,
        "seed_robustness": seed_outcome,
        "mechanism_separation": mechanism_outcome,
        "worst_stratum_safety": safety,
    }
    return outcomes, mechanism


def synthetic_training_cell(
    context: Any,
    torch: Any,
    features: Any,
    arm: str,
    seed: int,
    *,
    iterations: int,
    completed_iterations: int,
) -> int:
    import torch.nn.functional as functional

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = load_convir(context, torch, trainable=True)
    teacher = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2.0e-4,
        betas=(0.9, 0.999),
        weight_decay=0.0,
    )
    for _ in range(iterations):
        hazy = torch.rand(
            (BATCH_SIZE, 3, TRAIN_CROP, TRAIN_CROP),
            device=context.device,
        )
        paired_hazy = torch.rand_like(hazy)
        clear = torch.rand_like(hazy)
        negatives = [hazy, paired_hazy, torch.rand_like(hazy), torch.rand_like(hazy)]
        prediction = unwrap_prediction(model(hazy))
        loss = functional.l1_loss(prediction, clear)
        if arm in {"b01_consistency", "b11_combined"}:
            with torch.no_grad():
                teacher_prediction = unwrap_prediction(teacher(paired_hazy))
            loss = loss + LAMBDA_CONSISTENCY * functional.l1_loss(
                prediction,
                teacher_prediction,
            )
        if arm in {"b10_carl", "b11_combined"}:
            loss = loss + LAMBDA_CARL * carl_loss(
                torch,
                features,
                prediction,
                clear,
                negatives,
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            update_ema(teacher, model)
        completed_iterations += 1
        if completed_iterations in {1, PROBE_ITERATIONS // 2, PROBE_ITERATIONS}:
            write_contract_progress(
                context,
                completed_iterations=completed_iterations,
                total_iterations=PROBE_ITERATIONS,
                stage="multiseed_synthetic_fixed_linear_probe",
            )
    return completed_iterations


@(
    (
        lambda argv: (
            argv.pop(1)
            if len(argv) > 1 and argv[1] in {"contract", "run"}
            else None
        )
    )(__import__("sys").argv),
    lambda function: function,
)[1]
def contract(context_path):
    context = load_context(context_path, "contract")
    require_context(context)
    if (
        context.device != "cuda"
        or context.engineering_contract.get("mode") != "gpu_synthetic_no_data"
    ):
        raise RuntimeError(
            "multiseed capability qualification requires synthetic CUDA only"
        )
    import torch

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    features = load_vgg19(context, torch)
    torch.cuda.reset_peak_memory_stats(context.device)
    started = time.monotonic()
    completed_iterations = 0
    for seed in SEEDS:
        for arm in ARMS:
            completed_iterations = synthetic_training_cell(
                context,
                torch,
                features,
                arm,
                seed,
                iterations=PROBE_STEPS_PER_CELL,
                completed_iterations=completed_iterations,
            )
    if completed_iterations != PROBE_ITERATIONS:
        raise RuntimeError("synthetic cost probe did not cover the frozen probe map")
    write_contract_result(
        context,
        checks={
            "bound_assets_verified": True,
            "local_vgg_state_dict_verified": True,
            "synthetic_cuda_only": True,
            "five_seed_four_arm_path_exercised": True,
        },
        engineering={
            "mode": "gpu_synthetic_no_data",
            "device": context.device,
            "fixture": {
                "batch": BATCH_SIZE,
                "channels": 3,
                "height": TRAIN_CROP,
                "width": TRAIN_CROP,
            },
            "production_path_exercised": True,
            "protected_data_touched": False,
            "scientific_output_created": False,
            "scientific_training_occurred": False,
            "cost": {
                "observed_iterations": PROBE_ITERATIONS,
                "observed_wall_seconds": time.monotonic() - started,
                "observed_peak_memory_mib": (
                    torch.cuda.max_memory_allocated(context.device) / (1024 * 1024)
                ),
            },
        },
    )


def run(context_path):
    context = load_context(context_path, "run")
    require_context(context)
    if context.device != "cuda":
        raise RuntimeError(
            "multiseed scientific execution requires lifecycle-selected CUDA"
        )
    import torch

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    prepare_phase_output(context)
    training, development = enumerate_haze4k(context)
    strata = fixed_strata(development)
    features = load_vgg19(context, torch)
    scores: dict[int, dict[str, dict[str, dict[str, Any]]]] = {}
    early_futility = False
    completed_seeds: list[int] = []
    first_look_summary: dict[str, Any] | None = None
    for seed_index, seed in enumerate(SEEDS):
        scores[seed] = {}
        for arm in ARMS:
            checkpoint = train_arm(
                context,
                torch,
                features,
                training,
                arm,
                seed,
            )
            write_workload_progress(
                context,
                completed_units=len(load_completed_unit_ledger(context)),
                stage="multiseed_train",
            )
            model = load_trained_arm(context, torch, arm, seed, checkpoint)
            scores[seed][arm] = evaluate_arm(
                context,
                torch,
                arm,
                seed,
                model,
                strata,
            )
            write_workload_progress(
                context,
                completed_units=len(load_completed_unit_ledger(context)),
                stage="multiseed_evaluate",
            )
        completed_seeds.append(seed)
        if seed_index + 1 == FIRST_LOOK_SEED_COUNT:
            first_look_summary = summarize(scores, completed_seeds)
            if is_early_futility(first_look_summary):
                early_futility = True
                for skipped_seed in SEEDS[FIRST_LOOK_SEED_COUNT:]:
                    skip_seed(context, skipped_seed)
                    write_workload_progress(
                        context,
                        completed_units=len(load_completed_unit_ledger(context)),
                        stage="multiseed_predeclared_futility_closure",
                    )
                break
    summary = summarize(scores, completed_seeds)
    summary.update(
        {
            "look_id": (
                "seed_look_060" if early_futility else "seed_look_100"
            ),
            "early_futility": early_futility,
            "first_look": first_look_summary,
        }
    )
    summary_path = output_file(context, "multiseed_summary.json")
    atomic_json(summary_path, summary)
    record_completed_unit(
        context,
        unit_id="multiseed_terminal_summary",
        input_sha256=canonical_sha256(
            {
                "kind": "multiseed_terminal_summary",
                "arms": ARMS,
                "planned_seeds": SEEDS,
                "completed_seeds": completed_seeds,
                "early_futility": early_futility,
            }
        ),
        output_relpath="multiseed_summary.json",
    )
    ledger_count = len(load_completed_unit_ledger(context))
    outcomes, mechanism = gate_outcomes(
        summary,
        ledger_count,
        early_futility=early_futility,
    )
    gate_summary = {
        "route_id": ROUTE_ID,
        "operation_id": OPERATION_ID,
        "gate_outcomes": outcomes,
        "mechanism_classification": mechanism,
        "look_id": summary["look_id"],
        "early_futility": early_futility,
        "completed_seed_count": len(completed_seeds),
        "planned_seed_count": len(SEEDS),
        "completed_units": ledger_count,
        "total_units": TOTAL_UNITS,
    }
    gate_path = output_file(context, "multiseed_gate_summary.json")
    atomic_json(gate_path, gate_summary)
    summary_sha = sha256_file(summary_path)
    gate_sha = sha256_file(gate_path)
    primary = summary["combined_vs_baseline"]
    write_scientific_review_facts(
        context,
        relpath=(
            "daytime_dehazing_density_consistency_carl_"
            "multiseed_factorial_v1_review_facts.json"
        ),
        facts=[
            build_primary_review_fact(
                fact_id="multiseed_combined_vs_baseline_raw_point",
                claim_id="combined_materiality",
                metric="raw seed-blocked combined-versus-baseline scene PSNR effect",
                unit="dB",
                population="150 Haze4K development-screening original clear scenes",
                grouping="original clear scene after averaging four haze observations and completed paired seed blocks",
                point=primary["point"],
                point_pointer="/combined_vs_baseline/point",
                ci_lower=primary["ci_lower"],
                ci_upper=primary["ci_upper"],
                confidence_level=summary["confidence_level"],
                ci_lower_pointer="/combined_vs_baseline/ci_lower",
                ci_upper_pointer="/combined_vs_baseline/ci_upper",
                confidence_level_pointer="/confidence_level",
                threshold=summary["meaningful_effect_db"],
                threshold_operator=">",
                threshold_pointer="/meaningful_effect_db",
                source_filename="multiseed_summary.json",
                source_sha256=summary_sha,
            ),
            *[
                build_gate_review_fact(
                    fact_id=gate_id,
                    metric=gate_id,
                    unit="original_clear_scene",
                    population="Haze4K development screening",
                    grouping="original_clear_scene",
                    gate_outcome=outcome,
                    source_filename="multiseed_gate_summary.json",
                    source_sha256=gate_sha,
                )
                for gate_id, outcome in sorted(outcomes.items())
            ],
        ],
    )
    write_gate_result(
        context,
        gate_outcomes=outcomes,
        details={
            "summary_filename": "multiseed_summary.json",
            "gate_summary_filename": "multiseed_gate_summary.json",
            "comparison_family": (
                "multiseed_factorial_raw_terms_density_and_seed_stability"
            ),
            "look_id": summary["look_id"],
            "early_futility": early_futility,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=Path, required=True)
    args = parser.parse_args()
    raw = json.loads(args.context.read_text(encoding="utf-8"))
    phase = raw.get("phase") if isinstance(raw, dict) else None
    if phase == "contract":
        contract(args.context)
    elif phase == "run":
        run(args.context)
    else:
        raise SystemExit("route lifecycle context phase is invalid")


if __name__ == "__main__":
    main()
