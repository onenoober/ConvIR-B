#!/usr/bin/env python3
"""Frozen Haze4K-train development discrimination for CONVIR_ONLY_RDPCM_V1."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

from route_program_api import (
    asset_path,
    atomic_json,
    load_context,
    output_file,
    prepare_phase_output,
    write_contract_result,
    write_run_result,
    write_workload_progress,
)


ANCHOR_COMMIT = "3b4da35440c8c26a7d1bcaf1daf342e11d9a3898"
CHECKPOINT_SHA256 = "6f42037d57a4e3de3a10ac0ab909d66a3415864a19433c29204a975f4efa4088"
MODEL_SOURCE_SHA256 = "efd964e904d20389deebcda082cac5618b11d3c9a92ee68772b592b13cbf6fe8"
REVIEW_SHA256 = "3d3346823e083290c958e0879ee0c8f139ef6a465ee8624919c001dd90fb90c0"
SPLIT_SHA256 = "7596b775f170f9491af238b95e88a7fcfc737ff1ef68a3a56b8529ecde6c6d0d"
RDPCM_PARAMETER_COUNT = 41699
TRAIN_GROUPS = 512
VALIDATION_GROUPS = 300
TRAIN_STEPS = 600
BATCH_SIZE = 4
CROP_SIZE = 256
LEARNING_RATE = 2e-4
SEEDS = (3407, 7777, 20260721)
BOOTSTRAP_SEED = 20260721
BOOTSTRAP_REPLICATES = 10000
PRIMARY_PASS_LCB_DB = 0.03
PRIMARY_FAIL_UCB_DB = 0.01
ABSOLUTE_PASS_LCB_DB = 0.0
ABSOLUTE_FAIL_UCB_DB = 0.0
ABSOLUTE_CVAR5_PASS_DB = -0.10
ABSOLUTE_CVAR5_FAIL_DB = -0.20
SEED_PASS_FLOOR_DB = 0.0
SEED_FAIL_FLOOR_DB = -0.01
FIXTURE = {"batch": 1, "channels": 3, "height": 256, "width": 256}


def _load_checkpoint(torch, path: Path):
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if isinstance(value, dict) and "model" in value:
        value = value["model"]
    if not isinstance(value, dict) or not value:
        raise RuntimeError("official checkpoint is not a non-empty state dict")
    return value


def _load_candidate(build_net, state, mode, device):
    model = build_net("base", "Haze4K", rdpcm_mode=mode).to(device)
    current = model.state_dict()
    missing = sorted(set(current) - set(state))
    unexpected = sorted(set(state) - set(current))
    mismatched = sorted(
        key for key in set(current) & set(state)
        if tuple(current[key].shape) != tuple(state[key].shape)
    )
    if unexpected or mismatched or len(missing) != 9 \
            or any(not key.startswith("RDPCM.") for key in missing):
        raise RuntimeError(
            f"official state identity failed: missing={missing} unexpected={unexpected} "
            f"mismatched={mismatched}"
        )
    accepted = dict(current)
    accepted.update(state)
    model.load_state_dict(accepted, strict=True)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("RDPCM."))
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if trainable != RDPCM_PARAMETER_COUNT:
        raise RuntimeError(f"RDPCM trainable parameter count changed: {trainable}")
    return model


def _configure_determinism(torch, seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _forward(model, tensor):
    outputs = model(tensor)
    if not isinstance(outputs, list) or len(outputs) != 3:
        raise RuntimeError("official three-scale output contract changed")
    if not all(bool(item.isfinite().all().item()) for item in outputs):
        raise RuntimeError("non-finite model output")
    return outputs


def _contract_checks(context):
    import torch

    if context.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("A1 production-path contract requires CUDA")
    if any(context.protected_data_permissions.values()):
        raise RuntimeError("A1 protected permissions must remain false")
    identities = {
        "official_anchor": context.assets["official_anchor"].commit,
        "official_checkpoint": context.assets["official_checkpoint"].sha256,
        "rdpcm_model_source": context.assets["rdpcm_model_source"].sha256,
        "authoritative_review": context.assets["authoritative_review"].sha256,
    }
    expected = {
        "official_anchor": ANCHOR_COMMIT,
        "official_checkpoint": CHECKPOINT_SHA256,
        "rdpcm_model_source": MODEL_SOURCE_SHA256,
        "authoritative_review": REVIEW_SHA256,
    }
    if identities != expected:
        raise RuntimeError(f"identity-bound assets changed: {identities}")
    checkpoint = asset_path(context, "official_checkpoint", kind="file")
    sys.path.insert(0, str(context.remote_repo / "Dehazing" / "ITS"))
    from models.ConvIR import build_net

    device = torch.device("cuda")
    state = _load_checkpoint(torch, checkpoint)
    base = build_net("base", "Haze4K", rdpcm_mode="off").to(device).eval()
    base.load_state_dict(state, strict=True)
    _configure_determinism(torch, BOOTSTRAP_SEED)
    spatial = _load_candidate(build_net, state, "v1", device).eval()
    _configure_determinism(torch, BOOTSTRAP_SEED)
    global_uniform = _load_candidate(build_net, state, "global_v1", device).eval()
    if list(spatial.RDPCM.state_dict()) != list(global_uniform.RDPCM.state_dict()):
        raise RuntimeError("spatial and global controls have different parameter keys")
    for key in spatial.RDPCM.state_dict():
        if not torch.equal(spatial.RDPCM.state_dict()[key], global_uniform.RDPCM.state_dict()[key]):
            raise RuntimeError(f"matched RDPCM initialization differs: {key}")
    fixture = torch.linspace(
        0.0, 1.0, steps=3 * 256 * 256, dtype=torch.float32, device=device,
    ).reshape(1, 3, 256, 256)
    with torch.no_grad():
        base_out = _forward(base, fixture)
        spatial_out = _forward(spatial, fixture)
        global_out = _forward(global_uniform, fixture)
    for reference, left, right in zip(base_out, spatial_out, global_out):
        if not torch.equal(reference, left) or not torch.equal(reference, right):
            raise RuntimeError("initial no-op is not bitwise exact")
    with torch.no_grad():
        spatial.RDPCM.output_gate.fill_(0.1)
        global_uniform.RDPCM.output_gate.fill_(0.1)
        feature = torch.linspace(
            -1.0, 1.0, steps=128 * 32 * 32, dtype=torch.float32, device=device,
        ).reshape(1, 128, 32, 32)
        spatial_components = spatial.RDPCM.modulation_components(feature)
        global_components = global_uniform.RDPCM.modulation_components(feature)
    if any(tuple(item.shape[-2:]) != (32, 32) for item in spatial_components):
        raise RuntimeError("spatial RDPCM fields lost spatial support")
    if any(tuple(item.shape[-2:]) != (1, 1) for item in global_components):
        raise RuntimeError("global control fields are not uniform")
    for model in (spatial, global_uniform):
        model.zero_grad(set_to_none=True)
        sum(item.square().mean() for item in _forward(model, fixture)).backward()
        for name, parameter in model.named_parameters():
            if parameter.requires_grad and (parameter.grad is None \
                    or not bool(parameter.grad.isfinite().all().item())):
                raise RuntimeError(f"missing/non-finite RDPCM gradient: {name}")
    return {
        "asset_identity": True,
        "official_checkpoint_identity": True,
        "matched_parameter_keys": True,
        "matched_parameter_count": True,
        "matched_initialization": True,
        "bitwise_initial_noop": True,
        "spatial_field_support": True,
        "global_uniform_field_support": True,
        "finite_forward_backward": True,
        "protected_data_untouched": True,
    }


def contract(context_path):
    context = load_context(Path(context_path), "contract")
    prepare_phase_output(context)
    checks = _contract_checks(context)
    write_contract_result(
        context,
        checks=checks,
        engineering={
            "mode": "gpu_synthetic_no_data",
            "device": "cuda",
            "fixture": FIXTURE,
            "production_path_exercised": True,
            "protected_data_touched": False,
            "scientific_output_created": False,
            "scientific_training_occurred": False,
        },
    )


def _image_dirs(data_root: Path):
    train = data_root / "train"
    input_dir = next((train / name for name in ("IN", "haze", "hazy") if (train / name).is_dir()), None)
    label_dir = next((train / name for name in ("GT", "gt") if (train / name).is_dir()), None)
    if input_dir is None or label_dir is None:
        raise RuntimeError("Haze4K train input/GT directories are unavailable")
    return input_dir, label_dir


def _label_path(label_dir: Path, name: str):
    stem = Path(name).stem
    candidates = (label_dir / name, label_dir / f"{stem.split('_')[0]}.png")
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise RuntimeError(f"missing train-derived label for {name}")
    return path


def _tensor_pair(name, input_dir, label_dir, seed, step, torch):
    import numpy as np
    from PIL import Image

    hazy = Image.open(input_dir / name).convert("RGB")
    clean = Image.open(_label_path(label_dir, name)).convert("RGB")
    if hazy.size != clean.size or min(hazy.size) < CROP_SIZE:
        raise RuntimeError(f"invalid paired geometry: {name}")
    digest = hashlib.sha256(f"{seed}:{step}:{name}".encode()).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    left = rng.randrange(hazy.width - CROP_SIZE + 1)
    top = rng.randrange(hazy.height - CROP_SIZE + 1)
    box = (left, top, left + CROP_SIZE, top + CROP_SIZE)
    hazy = hazy.crop(box)
    clean = clean.crop(box)
    if rng.random() < 0.5:
        hazy = hazy.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        clean = clean.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    def convert(image):
        value = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        return torch.from_numpy(value)
    return convert(hazy), convert(clean)


def _batch_names(names, seed, step):
    cycle = (step * BATCH_SIZE) // len(names)
    offset = (step * BATCH_SIZE) % len(names)
    ordered = list(names)
    random.Random(seed + cycle * 1000003).shuffle(ordered)
    if offset + BATCH_SIZE <= len(ordered):
        return ordered[offset:offset + BATCH_SIZE]
    following = list(names)
    random.Random(seed + (cycle + 1) * 1000003).shuffle(following)
    return ordered[offset:] + following[:BATCH_SIZE - (len(ordered) - offset)]


def _loss(outputs, target, torch):
    import torch.nn.functional as F
    targets = (
        F.interpolate(target, scale_factor=0.25, mode="bilinear", align_corners=False),
        F.interpolate(target, scale_factor=0.5, mode="bilinear", align_corners=False),
        target,
    )
    content = sum(F.l1_loss(output, truth) for output, truth in zip(outputs, targets))
    frequency = sum(
        F.l1_loss(torch.view_as_real(torch.fft.fft2(output)),
                  torch.view_as_real(torch.fft.fft2(truth)))
        for output, truth in zip(outputs, targets)
    )
    return content + 0.1 * frequency


def _train_arm(build_net, state, mode, names, input_dir, label_dir, seed, device, torch):
    _configure_determinism(torch, seed)
    model = _load_candidate(build_net, state, mode, device).train()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
    )
    last_loss = None
    for step in range(TRAIN_STEPS):
        pairs = [_tensor_pair(name, input_dir, label_dir, seed, step, torch)
                 for name in _batch_names(names, seed, step)]
        inputs = torch.stack([item[0] for item in pairs]).to(device, non_blocking=True)
        targets = torch.stack([item[1] for item in pairs]).to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = _loss(_forward(model, inputs), targets, torch)
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError(f"non-finite training loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad], 1.0,
        )
        optimizer.step()
        last_loss = float(loss.detach().item())
    model.eval()
    return model, last_loss


def _full_tensor(path, torch):
    import numpy as np
    from PIL import Image
    image = Image.open(path).convert("RGB")
    value = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return torch.from_numpy(value).unsqueeze(0)


def _evaluate(model, names, input_dir, label_dir, device, torch):
    import torch.nn.functional as F
    result = {}
    with torch.no_grad():
        for name in names:
            input_tensor = _full_tensor(input_dir / name, torch).to(device)
            target = _full_tensor(_label_path(label_dir, name), torch).to(device)
            height, width = input_tensor.shape[-2:]
            pad_h = (32 - height % 32) % 32
            pad_w = (32 - width % 32) % 32
            padded = F.pad(input_tensor, (0, pad_w, 0, pad_h), mode="reflect")
            prediction = _forward(model, padded)[2][..., :height, :width].clamp(0, 1)
            mse = float((prediction - target).square().mean().item())
            result[name] = float("inf") if mse == 0 else -10.0 * math.log10(mse)
    return result


def _bootstrap(values, seed):
    import numpy as np
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for start in range(0, BOOTSTRAP_REPLICATES, 1000):
        count = min(1000, BOOTSTRAP_REPLICATES - start)
        indexes = rng.integers(0, len(array), size=(count, len(array)))
        means[start:start + count] = array[indexes].mean(axis=1)
    return {
        "point": float(array.mean()),
        "lcb95": float(np.quantile(means, 0.025)),
        "ucb95": float(np.quantile(means, 0.975)),
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": seed,
    }


def _sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(context_path):
    import torch

    context = load_context(Path(context_path), "run")
    prepare_phase_output(context)
    if context.evidence_role != "development_screening" \
            or any(context.protected_data_permissions.values()):
        raise RuntimeError("A1 run evidence role or permissions changed")
    data_root = asset_path(context, "haze4k_development", kind="directory")
    split_path = asset_path(context, "split_contract", kind="file")
    checkpoint = asset_path(context, "official_checkpoint", kind="file")
    if context.assets["split_contract"].sha256 != SPLIT_SHA256:
        raise RuntimeError("split contract identity changed")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    validation = split.get("validation_names")
    if not isinstance(validation, list) or len(validation) != VALIDATION_GROUPS \
            or len(set(validation)) != VALIDATION_GROUPS:
        raise RuntimeError("validation group contract changed")
    validation_groups = {Path(name).stem.split("_")[0] for name in validation}
    if len(validation_groups) != VALIDATION_GROUPS:
        raise RuntimeError("validation clean-reference groups are not independent")
    input_dir, label_dir = _image_dirs(data_root)
    all_names = sorted(path.name for path in input_dir.iterdir()
                       if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"})
    all_groups = {Path(name).stem.split("_")[0] for name in all_names}
    if len(all_names) != 3000 or len(all_groups) != 3000 \
            or not set(validation) <= set(all_names):
        raise RuntimeError("Haze4K train population identity/count changed")
    training_pool = sorted(set(all_names) - set(validation))
    if len(training_pool) != 2700:
        raise RuntimeError("training/validation complement is invalid")
    training = sorted(
        training_pool,
        key=lambda name: hashlib.sha256(f"3407:{name}".encode()).hexdigest(),
    )[:TRAIN_GROUPS]
    if {Path(name).stem.split("_")[0] for name in training} & validation_groups:
        raise RuntimeError("clean-reference group leakage detected")
    for name in training + validation:
        if not (input_dir / name).is_file():
            raise RuntimeError(f"missing train input: {name}")
        _label_path(label_dir, name)

    sys.path.insert(0, str(context.remote_repo / "Dehazing" / "ITS"))
    from models.ConvIR import build_net
    state = _load_checkpoint(torch, checkpoint)
    device = torch.device("cuda")
    base = build_net("base", "Haze4K", rdpcm_mode="off").to(device).eval()
    base.load_state_dict(state, strict=True)
    base_scores = _evaluate(base, validation, input_dir, label_dir, device, torch)
    del base
    torch.cuda.empty_cache()

    scores = {"global": {}, "spatial": {}}
    arm_records = []
    completed = 0
    checkpoint_dir = output_file(context, "checkpoints")
    checkpoint_dir.mkdir()
    for seed in SEEDS:
        for label, mode in (("global", "global_v1"), ("spatial", "v1")):
            model, last_loss = _train_arm(
                build_net, state, mode, training, input_dir, label_dir, seed, device, torch,
            )
            scores[label][seed] = _evaluate(
                model, validation, input_dir, label_dir, device, torch,
            )
            checkpoint_path = checkpoint_dir / f"{label}_seed{seed}_rdpcm.pt"
            torch.save({
                "seed": seed,
                "arm": label,
                "steps": TRAIN_STEPS,
                "rdpcm": model.RDPCM.state_dict(),
            }, checkpoint_path)
            arm_records.append({
                "arm": label,
                "seed": seed,
                "final_training_loss": last_loss,
                "checkpoint_sha256": _sha256(checkpoint_path),
                "checkpoint_relpath": str(checkpoint_path.relative_to(context.phase_output_path)),
            })
            del model
            torch.cuda.empty_cache()
            completed += 1
            write_workload_progress(context, completed_units=completed, stage=f"trained_{label}_seed{seed}")

    primary_by_name = []
    absolute_by_name = []
    for name in validation:
        primary_by_name.append(sum(
            scores["spatial"][seed][name] - scores["global"][seed][name]
            for seed in SEEDS
        ) / len(SEEDS))
        absolute_by_name.append(sum(
            scores["spatial"][seed][name] - base_scores[name]
            for seed in SEEDS
        ) / len(SEEDS))
    primary = _bootstrap(primary_by_name, BOOTSTRAP_SEED)
    absolute = _bootstrap(absolute_by_name, BOOTSTRAP_SEED + 1)
    ordered_absolute = sorted(absolute_by_name)
    cvar_count = max(1, math.ceil(0.05 * len(ordered_absolute)))
    absolute_cvar5 = sum(ordered_absolute[:cvar_count]) / cvar_count
    seed_deltas = {
        str(seed): sum(
            scores["spatial"][seed][name] - scores["global"][seed][name]
            for name in validation
        ) / len(validation)
        for seed in SEEDS
    }
    pass_gate = (
        primary["lcb95"] >= PRIMARY_PASS_LCB_DB
        and absolute["lcb95"] >= ABSOLUTE_PASS_LCB_DB
        and absolute_cvar5 >= ABSOLUTE_CVAR5_PASS_DB
        and min(seed_deltas.values()) >= SEED_PASS_FLOOR_DB
    )
    fail_gate = (
        primary["ucb95"] <= PRIMARY_FAIL_UCB_DB
        or absolute["ucb95"] <= ABSOLUTE_FAIL_UCB_DB
        or absolute_cvar5 <= ABSOLUTE_CVAR5_FAIL_DB
        or min(seed_deltas.values()) <= SEED_FAIL_FLOOR_DB
    )
    if pass_gate:
        state_name = "COMPLETED_GATE_PASS"
        decision = "RDPCM_A1_SPATIAL_MECHANISM_PASS"
        authorizes = "ONE_INDEPENDENT_REAL_DOMAIN_VALIDATION_CONTRACT_ONLY"
    elif fail_gate:
        state_name = "COMPLETED_GATE_FAIL"
        decision = "RDPCM_A1_SPATIAL_MECHANISM_FAIL_CLOSE_V1"
        authorizes = "NONE"
    else:
        state_name = "COMPLETED_INCONCLUSIVE"
        decision = "RDPCM_A1_SPATIAL_MECHANISM_INCONCLUSIVE"
        authorizes = "PREREGISTERED_EVIDENCE_COMPLETION_ONLY"
    summary = {
        "schema_version": 1,
        "route_id": context.route_id,
        "operation_id": context.operation_id,
        "decision": decision,
        "authorizes": authorizes,
        "population": {
            "source": "Haze4K_train_only",
            "training_groups": len(training),
            "validation_independent_groups": len(validation),
            "group_overlap": 0,
            "test_touched": False,
        },
        "fixed_training": {
            "seeds": list(SEEDS),
            "steps_per_arm_seed": TRAIN_STEPS,
            "batch_size": BATCH_SIZE,
            "crop_size": CROP_SIZE,
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": 0.0,
            "trainable_parameters": RDPCM_PARAMETER_COUNT,
            "base_frozen": True,
            "checkpoint_selection": "none_final_step_only",
        },
        "primary_spatial_minus_global_psnr_db": primary,
        "secondary_spatial_minus_official_psnr_db": absolute,
        "secondary_spatial_minus_official_cvar5_db": absolute_cvar5,
        "spatial_minus_global_by_seed_db": seed_deltas,
        "gates": {
            "pass": pass_gate,
            "fail": fail_gate,
            "inconclusive": not pass_gate and not fail_gate,
        },
        "arm_records": arm_records,
        "protected_access": {
            "confirmation_touched": False,
            "canary_touched": False,
            "locked_test_touched": False,
            "nhhaze_touched": False,
        },
    }
    atomic_json(output_file(context, "rdpcm_a1_summary.json"), summary)
    write_workload_progress(context, completed_units=7, stage="development_decision_complete")
    write_run_result(
        context,
        state=state_name,
        decision=decision,
        authorizes=authorizes,
        details={
            "primary_point_db": primary["point"],
            "primary_lcb95_db": primary["lcb95"],
            "primary_ucb95_db": primary["ucb95"],
            "absolute_point_db": absolute["point"],
            "absolute_lcb95_db": absolute["lcb95"],
            "absolute_cvar5_db": absolute_cvar5,
            "validation_groups": len(validation),
            "training_steps_total": 2 * len(SEEDS) * TRAIN_STEPS,
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("contract", "run"))
    parser.add_argument("--context", required=True)
    args = parser.parse_args()
    if args.phase == "contract":
        contract(args.context)
    else:
        run(args.context)


if __name__ == "__main__":
    main()
