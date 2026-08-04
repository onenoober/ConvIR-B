#!/usr/bin/env python3
"""Run C2's frozen CARL and haze-density-consistency 2x2 factorial.

The contract phase runs only an identity-bound CUDA synthetic qualification.
The run phase is the sole path that may access the declared Haze4K development
role, and it records every completed workload unit in the generic ledger.
"""

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


ROUTE_ID = "daytime-dehazing-density-consistency-carl-factorial-v1"
OPERATION_ID = "DAYTIME_DEHAZING_DENSITY_CONSISTENCY_CARL_FACTORIAL_QUALIFY"
ANCHOR_COMMIT = "3b4da35440c8c26a7d1bcaf1daf342e11d9a3898"
CHECKPOINT_SHA256 = "6f42037d57a4e3de3a10ac0ab909d66a3415864a19433c29204a975f4efa4088"
MODEL_SOURCE_SHA256 = "9681defa95c6602d0e0abae05e635771bd65d48a18555198ff00d573242a0005"
MODEL_LAYERS_SHA256 = "ac8a05bd626d9adda16308dedb9466f36d7ff44cfb666f64e7e14ddf8cdf43a4"
VGG19_SHA256 = "dcbb9e9dad569fff7a846263a77324fc34978fea2bfb039c012d710e1776ae44"
VGG19_SIZE_BYTES = 574_673_361
S0_LEDGER_SHA256 = "4cff8e7aecea5d8e19165ac4e725f69746342521115a3ab1f03ea1474f280960"

ARMS = ("b00_baseline", "b10_carl", "b01_consistency", "b11_combined")
TRAINING_SCENES = 600
DEVELOPMENT_SCENES = 150
OBSERVATIONS_PER_SCENE = 4
TRAINING_STEPS = 2000
FORMAL_ITERATIONS = len(ARMS) * TRAINING_STEPS
TOTAL_UNITS = len(ARMS) + len(ARMS) * DEVELOPMENT_SCENES + 1
BATCH_SIZE = 2
TRAIN_CROP = 256
EVAL_CROP = 256
TRAINING_SEED = 20_260_803
PRIMARY_CLIP_DB = 0.25
MEANINGFUL_EFFECT_DB = 0.10
WORST_STRATUM_FLOOR_DB = -0.10
CRITICAL_VALUE = 2.74
PRECISION_HALF_WIDTH_DB = 0.10
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
        raise RuntimeError("C2 route or operation identity mismatch")
    if context.total_units != TOTAL_UNITS:
        raise RuntimeError("C2 workload total differs from the frozen contract")
    if any(context.protected_data_permissions.values()):
        raise RuntimeError("C2 does not permit protected-data access")


def verify_file_asset(context: Any, asset_id: str, expected_sha: str, *, size: int | None = None) -> Path:
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
        capture_output=True, text=True, timeout=30, check=False,
    )
    if completed.returncode or completed.stdout.strip() != ANCHOR_COMMIT:
        raise RuntimeError("official anchor checkout HEAD differs from the frozen commit")
    return anchor


def load_vgg19(context: Any, torch: Any) -> Any:
    path = verify_file_asset(
        context, "vgg19_imagenet1k_v1", VGG19_SHA256, size=VGG19_SIZE_BYTES,
    )
    import torchvision

    # No torchvision weights enum, cache resolution, or runtime download is used.
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
    if Path(module.__file__).resolve() != source.resolve() or layer_module is None \
            or Path(layer_module.__file__).resolve() != layers.resolve():
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
        value = value[0]
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


def carl_loss(torch: Any, features: Any, prediction: Any, clear: Any, hazy: list[Any]) -> Any:
    """C2's fixed K=4 triplet-margin transfer, not a paper reproduction."""
    import torch.nn.functional as functional

    prediction_maps = feature_maps(torch, features, prediction)
    with torch.no_grad():
        clear_maps = feature_maps(torch, features, clear)
        hazy_maps = [feature_maps(torch, features, item) for item in hazy]
    loss = prediction.new_zeros(())
    for weight, predicted, target, *negative_layers in zip(
        VGG_LAYER_WEIGHTS, prediction_maps, clear_maps, *hazy_maps,
    ):
        positive = functional.l1_loss(predicted, target)
        negative = torch.stack([functional.l1_loss(predicted, item) for item in negative_layers]).mean()
        loss = loss + weight * torch.relu(positive - negative + CARL_MARGIN)
    return loss


def image_files(directory: Path) -> list[Path]:
    return sorted(
        item for item in directory.iterdir()
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
    if len(roles["training"]) != TRAINING_SCENES \
            or len(roles["development_screening"]) != DEVELOPMENT_SCENES:
        raise RuntimeError("the frozen Haze4K 600/150 role split is unavailable")
    return roles


def enumerate_haze4k(context: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    root = asset_path(context, "haze4k_train", kind="directory")
    roles = read_roles(context)
    input_dirs = [root / name for name in ("IN", "haze", "hazy") if (root / name).is_dir()]
    label_dirs = [root / name for name in ("GT", "gt", "clear") if (root / name).is_dir()]
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
            scene_id, {"scene_id": scene_id, "clear": clear, "observations": []},
        )
        group["observations"].append(hazy)
    if len(groups) != 750 or any(len(item["observations"]) != OBSERVATIONS_PER_SCENE for item in groups.values()):
        raise RuntimeError("Haze4K does not preserve 750 clear scenes with four observations each")
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
        raise RuntimeError(f"image is smaller than the frozen {crop}-pixel crop: {path.name}")
    if center:
        top = (height - crop) // 2
        left = (width - crop) // 2
    else:
        digest = hashlib.sha256(crop_key.encode("utf-8")).digest()
        top = int.from_bytes(digest[:8], "big") % (height - crop + 1)
        left = int.from_bytes(digest[8:16], "big") % (width - crop + 1)
    return value[top:top + crop, left:left + crop]


def to_tensor(torch: Any, array: Any, device: str) -> Any:
    import numpy as np

    return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1))).unsqueeze(0).to(device)


def scene_batch(torch: Any, group: dict[str, Any], *, crop: int, crop_key: str, center: bool, device: str) -> tuple[Any, list[Any]]:
    clear = load_rgb(group["clear"], crop=crop, crop_key=crop_key, center=center)
    hazy = [
        load_rgb(path, crop=crop, crop_key=crop_key, center=center)
        for path in sorted(group["observations"])
    ]
    return to_tensor(torch, clear, device), [to_tensor(torch, item, device) for item in hazy]


def density_proxy(group: dict[str, Any]) -> float:
    import numpy as np

    clear = load_rgb(group["clear"], crop=EVAL_CROP, crop_key=group["scene_id"], center=True)
    values = []
    for hazy in group["observations"]:
        value = load_rgb(hazy, crop=EVAL_CROP, crop_key=group["scene_id"], center=True)
        values.append(float(np.mean(np.square(value - clear))))
    return float(sum(values) / len(values))


def fixed_strata(development: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    ranked = sorted(
        development, key=lambda item: (density_proxy(item), item["scene_id"]),
    )
    if len(ranked) != DEVELOPMENT_SCENES:
        raise RuntimeError("development roster count differs from the frozen contract")
    return {
        "DENSITY_LOW": ranked[:50],
        "DENSITY_MID": ranked[50:100],
        "DENSITY_HIGH": ranked[100:],
    }


def psnr(torch: Any, prediction: Any, target: Any) -> float:
    mse = float(torch.mean(torch.square(prediction.clamp(0.0, 1.0) - target)).item())
    return 99.0 if mse <= 1.0e-12 else -10.0 * math.log10(mse)


def update_ema(teacher: Any, student: Any) -> None:
    for target, source in zip(teacher.parameters(), student.parameters()):
        target.mul_(EMA_DECAY).add_(source, alpha=1.0 - EMA_DECAY)


def write_unit(context: Any, unit_id: str, payload: dict[str, Any]) -> None:
    relpath = f"units/{unit_id}.json"
    atomic_json(output_file(context, relpath), payload)
    record_completed_unit(
        context, unit_id=unit_id, input_sha256=canonical_sha256(payload["input"]), output_relpath=relpath,
    )


def train_arm(context: Any, torch: Any, features: Any, training: list[dict[str, Any]], arm: str) -> Path:
    import torch.nn.functional as functional

    torch.manual_seed(TRAINING_SEED)
    model = load_convir(context, torch, trainable=True)
    teacher = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-4, betas=(0.9, 0.999), weight_decay=0.0)
    for step in range(TRAINING_STEPS):
        group = training[step % len(training)]
        crop_key = f"{group['scene_id']}|{step}"
        clear_single, hazy_single = scene_batch(
            torch, group, crop=TRAIN_CROP, crop_key=crop_key, center=False, device=context.device,
        )
        clear = torch.cat([clear_single, clear_single], dim=0)
        hazy = [torch.cat([item, item], dim=0) for item in hazy_single]
        primary = hazy[step % OBSERVATIONS_PER_SCENE]
        paired = hazy[(step // OBSERVATIONS_PER_SCENE + 1) % OBSERVATIONS_PER_SCENE]
        prediction = unwrap_prediction(model(primary))
        loss = functional.l1_loss(prediction, clear)
        if arm in {"b01_consistency", "b11_combined"}:
            with torch.no_grad():
                teacher_prediction = unwrap_prediction(teacher(paired))
            loss = loss + LAMBDA_CONSISTENCY * functional.l1_loss(prediction, teacher_prediction)
        if arm in {"b10_carl", "b11_combined"}:
            loss = loss + LAMBDA_CARL * carl_loss(torch, features, prediction, clear, hazy)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            update_ema(teacher, model)
    checkpoint = output_file(context, f"checkpoints/{arm}.pt")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "arm": arm}, checkpoint)
    payload = {
        "input": {
            "kind": "training_arm", "arm": arm, "steps": TRAINING_STEPS,
            "training_scene_count": len(training), "seed": TRAINING_SEED,
        },
        "checkpoint": checkpoint.name,
    }
    unit_id = f"train_{arm}"
    relpath = f"units/{unit_id}.json"
    atomic_json(output_file(context, relpath), payload)
    record_completed_unit(
        context, unit_id=unit_id, input_sha256=canonical_sha256(payload["input"]),
        output_relpath=f"checkpoints/{arm}.pt",
    )
    return checkpoint


def load_trained_arm(context: Any, torch: Any, arm: str, checkpoint: Path) -> Any:
    model = load_convir(context, torch, trainable=False)
    try:
        state = torch.load(checkpoint, map_location=context.device, weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location=context.device)
    if not isinstance(state, dict) or state.get("arm") != arm or not isinstance(state.get("model"), dict):
        raise RuntimeError(f"C2 training checkpoint is invalid: {arm}")
    model.load_state_dict(state["model"], strict=True)
    return model.eval()


def evaluate_arm(context: Any, torch: Any, arm: str, model: Any, strata: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    with torch.no_grad():
        for stratum, groups in strata.items():
            for group in groups:
                clear, hazy = scene_batch(
                    torch, group, crop=EVAL_CROP, crop_key=group["scene_id"], center=True, device=context.device,
                )
                observation_scores = [psnr(torch, unwrap_prediction(model(item)), clear) for item in hazy]
                score = float(sum(observation_scores) / len(observation_scores))
                unit_id = f"eval_{arm}_{group['scene_id']}"
                payload = {
                    "input": {
                        "kind": "evaluation_scene", "arm": arm, "scene_id": group["scene_id"],
                        "stratum": stratum, "observation_count": OBSERVATIONS_PER_SCENE,
                    },
                    "scene_psnr_db": score,
                    "observation_psnr_db": observation_scores,
                }
                write_unit(context, unit_id, payload)
                results[group["scene_id"]] = {"score": score, "stratum": stratum}
    return results


def interval(values: list[float]) -> dict[str, float]:
    if not values:
        raise RuntimeError("empty grouped contrast")
    point = float(sum(values) / len(values))
    if len(values) == 1:
        half_width = float("inf")
    else:
        variance = sum((value - point) ** 2 for value in values) / (len(values) - 1)
        half_width = CRITICAL_VALUE * math.sqrt(variance / len(values))
    return {
        "point": point, "ci_lower": point - half_width,
        "ci_upper": point + half_width, "half_width": half_width,
    }


def clipped_difference(left: float, right: float) -> float:
    return max(-PRIMARY_CLIP_DB, min(PRIMARY_CLIP_DB, left - right))


def summarize(scores: dict[str, dict[str, dict[str, float]]]) -> dict[str, Any]:
    scene_ids = sorted(scores["b00_baseline"])
    if any(set(scores[arm]) != set(scene_ids) for arm in ARMS):
        raise RuntimeError("C2 arms do not cover the same frozen scene roster")
    per_scene = {}
    for scene_id in scene_ids:
        value = {arm: scores[arm][scene_id]["score"] for arm in ARMS}
        per_scene[scene_id] = {
            "stratum": scores["b00_baseline"][scene_id]["stratum"],
            "combined": clipped_difference(value["b11_combined"], value["b00_baseline"]),
            "carl_main": 0.5 * (
                clipped_difference(value["b10_carl"], value["b00_baseline"])
                + clipped_difference(value["b11_combined"], value["b01_consistency"])
            ),
            "consistency_main": 0.5 * (
                clipped_difference(value["b01_consistency"], value["b00_baseline"])
                + clipped_difference(value["b11_combined"], value["b10_carl"])
            ),
            "interaction": clipped_difference(
                value["b11_combined"] - value["b01_consistency"],
                value["b10_carl"] - value["b00_baseline"],
            ),
        }
    combined = interval([item["combined"] for item in per_scene.values()])
    stratum_results = {}
    for stratum in ("DENSITY_LOW", "DENSITY_MID", "DENSITY_HIGH"):
        values = [item["combined"] for item in per_scene.values() if item["stratum"] == stratum]
        if len(values) != 50:
            raise RuntimeError(f"frozen density stratum coverage differs: {stratum}")
        stratum_results[stratum] = interval(values)
    return {
        "combined_vs_baseline": combined,
        "carl_main_effect": interval([item["carl_main"] for item in per_scene.values()]),
        "consistency_main_effect": interval([item["consistency_main"] for item in per_scene.values()]),
        "interaction": interval([item["interaction"] for item in per_scene.values()]),
        "density_strata": stratum_results,
    }


def gate_outcomes(summary: dict[str, Any], ledger_count: int) -> dict[str, str]:
    primary = summary["combined_vs_baseline"]
    strata = summary["density_strata"]
    worst_lower = min(item["ci_lower"] for item in strata.values())
    return {
        "identity_integrity": "pass",
        "workload_coverage": "pass" if ledger_count == TOTAL_UNITS else "fail",
        "primary_precision": (
            "met" if all(item["half_width"] <= PRECISION_HALF_WIDTH_DB for item in strata.values())
            else "unmet"
        ),
        "combined_materiality": (
            "favorable" if primary["ci_lower"] > MEANINGFUL_EFFECT_DB
            else "unfavorable" if primary["ci_upper"] <= MEANINGFUL_EFFECT_DB
            else "indeterminate"
        ),
        "worst_stratum_safety": (
            "safe" if worst_lower >= WORST_STRATUM_FLOOR_DB else "unsafe"
        ),
    }


# The generic lifecycle supplies its phase as argv[1]; normalize that transport
# token before argparse handles the entrypoint's context-only interface.
@(
    (lambda argv: argv.pop(1) if len(argv) > 1 and argv[1] in {"contract", "run"} else None)(
        __import__("sys").argv
    ),
    lambda function: function,
)[1]
def contract(context_path):
    context = load_context(context_path, "contract")
    require_context(context)
    if context.device != "cuda" or context.engineering_contract.get("mode") != "gpu_synthetic_no_data":
        raise RuntimeError("C2 capability qualification requires synthetic CUDA only")
    import torch

    torch.manual_seed(TRAINING_SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    import torch.nn.functional as functional

    model = load_convir(context, torch, trainable=True)
    teacher = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=2.0e-4, betas=(0.9, 0.999), weight_decay=0.0,
    )
    features = load_vgg19(context, torch)
    torch.cuda.reset_peak_memory_stats(context.device)
    started = time.monotonic()
    for iteration in range(FORMAL_ITERATIONS):
        arm = ARMS[iteration // TRAINING_STEPS]
        hazy = torch.rand((BATCH_SIZE, 3, TRAIN_CROP, TRAIN_CROP), device=context.device)
        paired_hazy = torch.rand_like(hazy)
        clear = torch.rand_like(hazy)
        negatives = [hazy, paired_hazy, torch.rand_like(hazy), torch.rand_like(hazy)]
        prediction = unwrap_prediction(model(hazy))
        loss = functional.l1_loss(prediction, clear)
        if arm in {"b01_consistency", "b11_combined"}:
            with torch.no_grad():
                teacher_prediction = unwrap_prediction(teacher(paired_hazy))
            loss = loss + LAMBDA_CONSISTENCY * functional.l1_loss(
                prediction, teacher_prediction,
            )
        if arm in {"b10_carl", "b11_combined"}:
            loss = loss + LAMBDA_CARL * carl_loss(
                torch, features, prediction, clear, negatives,
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            update_ema(teacher, model)
        if iteration + 1 in {1, FORMAL_ITERATIONS // 2, FORMAL_ITERATIONS}:
            write_contract_progress(
                context, completed_iterations=iteration + 1, total_iterations=FORMAL_ITERATIONS,
                stage="c2_synthetic_full_path",
            )
    write_contract_result(
        context,
        checks={
            "bound_assets_verified": True,
            "local_vgg_state_dict_verified": True,
            "synthetic_cuda_only": True,
            "fixed_carl_path_exercised": True,
        },
        engineering={
            "mode": "gpu_synthetic_no_data",
            "device": context.device,
            "fixture": {"batch": BATCH_SIZE, "channels": 3, "height": TRAIN_CROP, "width": TRAIN_CROP},
            "production_path_exercised": True,
            "protected_data_touched": False,
            "scientific_output_created": False,
            "scientific_training_occurred": False,
            "cost": {
                "observed_iterations": FORMAL_ITERATIONS,
                "observed_wall_seconds": time.monotonic() - started,
                "observed_peak_memory_mib": torch.cuda.max_memory_allocated(context.device) / (1024 * 1024),
            },
        },
    )


def run(context_path):
    context = load_context(context_path, "run")
    require_context(context)
    if context.device != "cuda":
        raise RuntimeError("C2 scientific execution requires the lifecycle-selected CUDA device")
    import torch

    torch.manual_seed(TRAINING_SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    prepare_phase_output(context)
    training, development = enumerate_haze4k(context)
    strata = fixed_strata(development)
    features = load_vgg19(context, torch)
    trained = {}
    for arm in ARMS:
        checkpoint = train_arm(context, torch, features, training, arm)
        trained[arm] = load_trained_arm(context, torch, arm, checkpoint)
        write_workload_progress(context, completed_units=len(load_completed_unit_ledger(context)), stage="c2_train")
    scores = {}
    for arm in ARMS:
        scores[arm] = evaluate_arm(context, torch, arm, trained[arm], strata)
        write_workload_progress(context, completed_units=len(load_completed_unit_ledger(context)), stage="c2_evaluate")
    summary = summarize(scores)
    summary_path = output_file(context, "c2_summary.json")
    atomic_json(summary_path, summary)
    record_completed_unit(
        context, unit_id="c2_terminal_summary",
        input_sha256=canonical_sha256({"kind": "c2_terminal_summary", "arms": ARMS}),
        output_relpath="c2_summary.json",
    )
    outcomes = gate_outcomes(summary, len(load_completed_unit_ledger(context)))
    gate_summary = {
        "route_id": ROUTE_ID, "operation_id": OPERATION_ID,
        "gate_outcomes": outcomes, "summary": summary,
        "completed_units": len(load_completed_unit_ledger(context)), "total_units": TOTAL_UNITS,
    }
    gate_path = output_file(context, "c2_gate_summary.json")
    atomic_json(gate_path, gate_summary)
    gate_sha = sha256_file(gate_path)
    write_review_facts(
        context, relpath="daytime_dehazing_density_consistency_carl_factorial_v1_review_facts.json",
        facts=[
            build_gate_review_fact(
                fact_id=gate_id, metric=gate_id, unit="original_clear_scene",
                population="Haze4K development screening", grouping="original_clear_scene",
                gate_outcome=outcome, source_filename="c2_gate_summary.json", source_sha256=gate_sha,
            ) for gate_id, outcome in sorted(outcomes.items())
        ],
    )
    write_gate_result(context, gate_outcomes=outcomes, details={
        "summary_filename": "c2_summary.json",
        "gate_summary_filename": "c2_gate_summary.json",
        "comparison_family": "c2_factorial_main_interaction_density_safety",
    })


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
