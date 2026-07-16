#!/usr/bin/env python3
"""A1X-v3 S0-only engineering integrity gate.

This initial bundle cannot run D0 or formal evidence. It reads only the sealed
parent128 state identity and the independent A1R debug32 manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROUTE_ID = "haze4k_v5_chd_rm_v4a_a1x_accessibility_v3_20260716"
OPERATORS = ("D_ref", "D_rep")
OUTER_FOLDS = 4
SEED = 3407
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 0.1
MICROFIT_EPOCHS = 2
MICROFIT_MIN_REDUCTION = 0.01
BATCH_SIZE = 8
PARAMETER_LIMIT = 300_000
MAC_LIMIT_LARGEST_EXACT_HALF = 600_000_000
A1R: Any = None
RUN_ARGS: Any = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode()).hexdigest()

def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)

def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module

def route_config() -> dict[str, Any]:
    return {
        "route_id": ROUTE_ID,
        "stage": "s0_only",
        "evidence_role": "engineering_debug",
        "debug_name_count": 32,
        "input_roles": ["hazy_rgb", "base_rgb", "old_0p125_rgb", "old_0p25_rgb", "current_delta_u"],
        "input_channels": 15,
        "operators": list(OPERATORS),
        "microfit_epochs": MICROFIT_EPOCHS,
        "microfit_min_loss_reduction_fraction": MICROFIT_MIN_REDUCTION,
        "seed": SEED,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "grad_clip": GRAD_CLIP,
    }

def heartbeat(phase: str, completed: int, total: int) -> None:
    atomic_json(Path(os.environ["HEARTBEAT_PATH"]), {
        "route_id": ROUTE_ID,
        "run_id": os.environ["RUN_ID"],
        "phase": phase,
        "completed": completed,
        "total": total,
        "timestamp": time.time(),
    })

def exact_half(value: torch.Tensor, shape: tuple[int, int], support: bool = False) -> torch.Tensor:
    result = F.interpolate(value.float(), size=shape, mode="bilinear", align_corners=False, antialias=False)
    return result > 0.0 if support else result

def make_record(
    *, args: Any, v3s: Any, legacy: Any, frozen: Any, name: str,
    historical_fold: int, outer_fold: int, current_model: torch.nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    assert A1R is not None
    sample = A1R.SOURCE.V3W.frozen_output_sample(
        args, v3s, legacy, frozen, name, historical_fold, device,
    )
    padded, _, _ = legacy.pad_to_factor(sample["hazy"])
    with torch.no_grad():
        context, _, _, _ = legacy.full_context_maps(
            frozen["control"], frozen["gate_producer"], padded,
        )
    shape = tuple(int(value) for value in context.shape[-2:])
    bound = sample["base"].new_tensor(frozen["bound"]).view(1, 3, 1, 1)
    delta_bound = sample["base"].new_tensor(args.delta_bound).view(1, 3, 1, 1)
    target_step = sample["support"] * A1R.PARENT.clamp_channelwise(
        4.0 * (sample["label"] - sample["base"]), bound,
    )
    record: dict[str, Any] = {
        "name": name, "outer_fold": outer_fold, "historical_fold": historical_fold,
        "shape": shape, "sample": sample, "operators": {},
    }
    for operator in OPERATORS:
        step = sample["steps"][operator]
        current = A1R.SOURCE.V3W.delta_for("output", current_model, sample, operator)
        target = sample["support"] * A1R.PARENT.clamp_channelwise(target_step - step, delta_bound)
        zero = torch.zeros_like(current)
        old_125, old_250, _, _ = v3s.candidate_predictions(sample["base"], step, zero)
        inputs = torch.cat([
            exact_half(sample["hazy"], shape), exact_half(sample["base"], shape),
            exact_half(old_125, shape), exact_half(old_250, shape), exact_half(current, shape),
        ], dim=1)
        _, correction_low, support_low, low_shape = exact_transport(
            target, current, sample["support"], delta_bound,
        )
        if inputs.shape[1] != 15 or tuple(inputs.shape[-2:]) != low_shape:
            raise RuntimeError("five-input exact-half whitelist drift")
        record["operators"][operator] = {
            "input_low": inputs.detach().cpu(),
            "support_low": support_low.detach().cpu(),
            "current_low": exact_half(current, shape).detach().cpu(),
            "target_correction_low": correction_low.detach().cpu(),
            "current": current.detach(), "target": target.detach(),
        }
    return record

def flat_items(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "name": record["name"], "operator": operator,
            "outer_fold": record["outer_fold"], "shape": tuple(record["shape"]),
            "input_low": record["operators"][operator]["input_low"],
            "support_low": record["operators"][operator]["support_low"],
            "current_low": record["operators"][operator]["current_low"],
            "target_correction_low": record["operators"][operator]["target_correction_low"],
        }
        for record in records for operator in OPERATORS
    ]

def feature_stats(items: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
    total = torch.zeros(15, dtype=torch.float64)
    square = torch.zeros(15, dtype=torch.float64)
    count = 0
    for item in items:
        value = item["input_low"].double()
        total += value.sum(dim=(0, 2, 3))
        square += value.square().sum(dim=(0, 2, 3))
        count += value.shape[0] * value.shape[2] * value.shape[3]
    if count <= 0:
        raise RuntimeError("empty feature statistics")
    mean = total / count
    variance = (square / count - mean.square()).clamp_min(1e-12)
    return mean.float().view(1, 15, 1, 1), torch.sqrt(variance).float().view(1, 15, 1, 1)

def shape_batches(items: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        groups[tuple(item["shape"])].append(item)
    batches: list[list[dict[str, Any]]] = []
    for shape in sorted(groups):
        ordered = sorted(groups[shape], key=lambda item: (item["name"], item["operator"]))
        batches.extend(ordered[offset:offset + BATCH_SIZE] for offset in range(0, len(ordered), BATCH_SIZE))
    return batches

def new_head(device: torch.device, seed: int) -> torch.nn.Module:
    repo = Path(os.environ["REMOTE_REPO"])
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from Dehazing.ITS.models.A1XAccess import A1X_ACCESS_Head
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return A1X_ACCESS_Head().to(device)

def predict_correction(
    head: torch.nn.Module, inputs: torch.Tensor, bound: torch.Tensor,
    mean: torch.Tensor, std: torch.Tensor,
) -> torch.Tensor:
    return 2.0 * bound * head((inputs - mean) / std)

def batch_loss(
    head: torch.nn.Module, batch: list[dict[str, Any]], device: torch.device,
    mean: torch.Tensor, std: torch.Tensor,
) -> torch.Tensor:
    inputs = torch.cat([item["input_low"] for item in batch]).to(device)
    support = torch.cat([item["support_low"] for item in batch]).to(device)
    target = torch.cat([item["target_correction_low"] for item in batch]).to(device)
    bound = target.new_tensor(RUN_ARGS.delta_bound).view(1, 3, 1, 1)
    predicted = predict_correction(head, inputs, bound, mean.to(device), std.to(device))
    active = support.expand_as(predicted).to(predicted.dtype)
    per_pixel = ((predicted - target) / (2.0 * bound).clamp_min(1e-8)).square() * active
    per_image = per_pixel.flatten(1).sum(1) / active.flatten(1).sum(1).clamp_min(1.0)
    return per_image.mean()

def microfit(items: list[dict[str, Any]], device: torch.device) -> dict[str, Any]:
    mean, std = feature_stats(items)
    head = new_head(device, SEED)
    optimizer = torch.optim.AdamW(head.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    batches = shape_batches(items)
    item_count = sum(len(batch) for batch in batches)
    if item_count != len(items):
        raise RuntimeError("S0 batching lost image-operator items")
    full_units = [[batch] for batch in batches if len(batch) == BATCH_SIZE]
    partial_batches = [batch for batch in batches if len(batch) < BATCH_SIZE]
    optimizer_units = full_units
    if partial_batches:
        if sum(len(batch) for batch in partial_batches) != BATCH_SIZE:
            raise RuntimeError("S0 partial shape batches do not form one equal-weight optimizer unit")
        optimizer_units.append(partial_batches)
    if any(sum(len(batch) for batch in unit) != BATCH_SIZE for unit in optimizer_units):
        raise RuntimeError("S0 optimizer unit is not image-operator balanced")

    def balanced_loss() -> float:
        return sum(
            float(batch_loss(head, batch, device, mean, std)) * len(batch)
            for batch in batches
        ) / item_count

    with torch.no_grad():
        initial = balanced_loss()
    first_gradient = 0.0
    for epoch in range(MICROFIT_EPOCHS):
        for unit_index, unit in enumerate(optimizer_units):
            optimizer.zero_grad(set_to_none=True)
            for batch in unit:
                loss = batch_loss(head, batch, device, mean, std)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("nonfinite S0 microfit loss")
                (loss * (len(batch) / BATCH_SIZE)).backward()
            gradient = float(torch.nn.utils.clip_grad_norm_(head.parameters(), GRAD_CLIP).item())
            if epoch == 0 and unit_index == 0:
                first_gradient = gradient
            if not math.isfinite(gradient):
                raise FloatingPointError("nonfinite S0 microfit gradient")
            optimizer.step()
    with torch.no_grad():
        final = balanced_loss()
    reduction = (initial - final) / max(initial, 1e-12)
    return {
        "initial_image_balanced_target_loss": initial,
        "final_image_balanced_target_loss": final,
        "loss_reduction_fraction": reduction,
        "minimum_reduction_fraction": MICROFIT_MIN_REDUCTION,
        "epochs": MICROFIT_EPOCHS,
        "optimizer_steps": MICROFIT_EPOCHS * len(optimizer_units),
        "optimizer_unit_size": BATCH_SIZE,
        "image_operator_item_count": item_count,
        "unequal_shape_batches_sample_weighted": True,
        "first_gradient_norm": first_gradient,
        "pass": bool(initial > 0.0 and final < initial and reduction >= MICROFIT_MIN_REDUCTION and first_gradient > 0.0),
    }

def count_conv_macs(head: torch.nn.Module, shape: tuple[int, int], device: torch.device) -> int:
    total = 0
    handles = []

    def hook(module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        nonlocal total
        if not isinstance(module, torch.nn.Conv2d):
            return
        batch, out_channels, height, width = output.shape
        kernel = module.kernel_size[0] * module.kernel_size[1]
        per_output = module.in_channels * kernel // module.groups
        total += int(batch * out_channels * height * width * per_output)

    for module in head.modules():
        if isinstance(module, torch.nn.Conv2d):
            handles.append(module.register_forward_hook(hook))
    with torch.no_grad():
        head(torch.zeros((1, 15, *shape), device=device))
    for handle in handles:
        handle.remove()
    return total

def smoke_gate(items: list[dict[str, Any]], device: torch.device) -> dict[str, Any]:
    head = new_head(device, SEED)
    parameter_count = sum(parameter.numel() for parameter in head.parameters())
    shapes = sorted({tuple(item["shape"]) for item in items})
    if shapes != [(208, 208), (240, 320)]:
        raise RuntimeError(f"S0 requires both exact A1R shapes: {shapes}")
    trainable_names = [name for name, parameter in head.named_parameters() if parameter.requires_grad]
    if len(trainable_names) != len(list(head.named_parameters())):
        raise RuntimeError("A1X head has nontrainable parameters")
    noop = 0.0
    for item in items:
        inputs = item["input_low"].to(device)
        target = item["target_correction_low"].to(device)
        bound = target.new_tensor(RUN_ARGS.delta_bound).view(1, 3, 1, 1)
        correction = predict_correction(
            head, inputs, bound,
            torch.zeros((1, 15, 1, 1), device=device),
            torch.ones((1, 15, 1, 1), device=device),
        )
        noop = max(noop, float(correction.abs().max().item()))
    micro = microfit(items, device)
    macs = {f"{shape[0]}x{shape[1]}": count_conv_macs(head, shape, device) for shape in shapes}
    passed = bool(
        parameter_count <= PARAMETER_LIMIT and max(macs.values()) <= MAC_LIMIT_LARGEST_EXACT_HALF
        and noop == 0.0 and micro["pass"]
    )
    return {
        "input_channels": 15, "native_exact_half_shapes": [list(shape) for shape in shapes],
        "parameter_count": parameter_count, "parameter_limit": PARAMETER_LIMIT,
        "macs_by_shape": macs, "mac_limit_largest_exact_half": MAC_LIMIT_LARGEST_EXACT_HALF,
        "zero_correction_max_abs": noop,
        "trainable_parameter_scope_exact": True,
        "trainable_parameter_names": trainable_names,
        "microfit": micro,
        "microfit_loss_reduction_fraction": micro["loss_reduction_fraction"],
        "deterministic_cuda": {
            "use_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        },
        "pass": passed,
    }

def load_base_state(args: Any, v3s: Any, legacy: Any, frozen: Any, names: list[str], historical_folds: dict[str, int], device: torch.device) -> torch.nn.Module:
    assert A1R is not None
    A1R.validate_parent_review(
        Path(A1R.AUDIT.a1f_r3_review), A1R.AUDIT.expected_a1f_r3_review_sha256,
    )
    payload, _, _ = A1R.PARENT.load_final_state(Path(A1R.AUDIT.a0r_trace_dir))
    _, current_model, optimizer, parameters = A1R.PARENT.a0p.load_cell(
        payload, args, v3s, legacy, frozen, list(names[:128]), historical_folds, device,
    )
    del optimizer, parameters
    current_model.eval()
    for parameter in current_model.parameters():
        parameter.requires_grad_(False)
    return current_model

def common_closeout(stage: str, started: float) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "route_id": ROUTE_ID,
        "run_id": os.environ["RUN_ID"],
        "route_commit": os.environ["EXPECTED_ROUTE_COMMIT"],
        "runner_sha256": os.environ["RUNNER_SHA256"],
        "stage": stage,
        "config_sha256": canonical_hash(route_config()),
        "wall_seconds": time.monotonic() - started,
        "candidate_selected": False,
        "policy_created": False,
        "canary_touched": False,
        "locked_test_touched": False,
    }

def load_s0_manifest() -> dict[str, Any]:
    path = Path(os.environ["S0_DEBUG_MANIFEST"])
    value = json.loads(path.read_text(encoding="utf-8"))
    contract = value["contract"]
    if value.get("evidence_role") != "engineering_debug":
        raise RuntimeError("S0 manifest has wrong evidence role")
    if len(contract["parent_state_names"]) != 128 or len(contract["debug_names"]) != 32:
        raise RuntimeError("S0 manifest count mismatch")
    if set(contract["parent_state_names"]) & set(contract["debug_names"]):
        raise RuntimeError("S0 parent/debug overlap")
    forbidden = value["forbidden"]
    if forbidden != {
        "confirmation_indices": [768, 1200],
        "confirmation_images_targets_outcomes_touched": False,
        "canary_touched": False,
        "locked_test_touched": False,
    }:
        raise RuntimeError("S0 forbidden-data contract drift")
    return value


def exact_transport(
    target: torch.Tensor, current: torch.Tensor, support: torch.Tensor,
    bound: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int]]:
    observed = {(400, 400): (208, 208), (480, 640): (240, 320)}
    full_shape = tuple(int(v) for v in current.shape[-2:])
    if full_shape not in observed:
        raise RuntimeError(f"unsealed A1C transport shape: {full_shape}")
    low_shape = observed[full_shape]
    low_target = F.interpolate(target, size=low_shape, mode="bilinear", align_corners=False, antialias=False)
    low_current = F.interpolate(current, size=low_shape, mode="bilinear", align_corners=False, antialias=False)
    low_support = F.interpolate(support.float(), size=low_shape, mode="bilinear", align_corners=False, antialias=False)
    low_difference = low_target - low_current
    excess = float(torch.clamp(low_difference.abs() - 2.0 * bound, min=0.0).max().item())
    if excess > 1e-6:
        raise RuntimeError("exact-half correction exceeds +/-2B")
    replay = F.interpolate(low_difference, size=full_shape, mode="bilinear", align_corners=False)
    endpoint_full = support * A1R.PARENT.clamp_channelwise(current + support * replay, bound)
    return endpoint_full, low_difference, low_support, low_shape


def transport_equivalence(records: list[dict[str, Any]]) -> dict[str, Any]:
    reference_path = Path(os.environ["A1C_REFERENCE_SOURCE"])
    reference = load_module(reference_path, "a1x_v3_a1c_reference")
    if reference.UPSTREAM_COMMIT != "9c4bc79cfdadb00aa91ac6c6baed58fdbc6be068":
        raise RuntimeError("vendored A1C upstream commit drift")
    if reference.UPSTREAM_SOURCE_SHA256 != "0b947a36a83178aaa5d8316273a24de835c52af437332b3b2607c74ffe9cac12":
        raise RuntimeError("vendored A1C upstream source hash drift")
    reference.PARENT = A1R.PARENT
    max_abs = 0.0
    range_excess = 0.0
    zero_noop_max_abs = 0.0
    shapes: set[tuple[int, int]] = set()
    for record in records:
        sample = record["sample"]
        support = sample["support"]
        for operator in OPERATORS:
            values = record["operators"][operator]
            current, target = values["current"], values["target"]
            bound = current.new_tensor(RUN_ARGS.delta_bound).view(1, 3, 1, 1)
            ours, difference, soft_support, shape = exact_transport(target, current, support, bound)
            theirs, reference_shape = reference.endpoint(target, current, support, bound, "exact_half")
            zero_endpoint = support * A1R.PARENT.clamp_channelwise(current, bound)
            max_abs = max(max_abs, float((ours - theirs).abs().max().item()))
            zero_noop_max_abs = max(zero_noop_max_abs, float((zero_endpoint - current).abs().max().item()))
            range_excess = max(range_excess, float(torch.clamp(difference.abs() - 2.0 * bound, min=0.0).max().item()))
            shapes.add(shape)
            if soft_support.min().item() < 0.0 or soft_support.max().item() > 1.0:
                raise RuntimeError("soft support escaped [0,1]")
            if shape != reference_shape:
                raise RuntimeError("A1C low shape mismatch")
    return {
        "reference_source_sha256": sha256_file(reference_path),
        "reference_upstream_commit": reference.UPSTREAM_COMMIT,
        "reference_upstream_source_sha256": reference.UPSTREAM_SOURCE_SHA256,
        "max_abs_vs_a1c_reference": max_abs,
        "correction_range_excess": range_excess,
        "zero_endpoint_noop_max_abs": zero_noop_max_abs,
        "observed_low_shapes": [list(shape) for shape in sorted(shapes)],
        "soft_support_preserved": True,
        "pass": max_abs == 0.0 and zero_noop_max_abs == 0.0 and range_excess <= 1e-6 and shapes == {(208, 208), (240, 320)},
    }


def run_a1x(
    args: Any, v3s: Any, legacy: Any, frozen: Any, names: list[str],
    folds: dict[str, int], device: torch.device, output_dir: str,
) -> dict[str, Any]:
    global RUN_ARGS
    RUN_ARGS = args
    started = time.monotonic()
    manifest = load_s0_manifest()
    parent_names = list(manifest["contract"]["parent_state_names"])
    if list(names) != parent_names or set(folds) != set(parent_names):
        raise RuntimeError("parent128 state identity mismatch")
    debug_names = list(manifest["contract"]["debug_names"])
    debug_folds = {name: int(value) for name, value in manifest["contract"]["debug_historical_folds"].items()}
    if set(debug_folds) != set(debug_names):
        raise RuntimeError("debug32 fold identity mismatch")
    current_model = load_base_state(args, v3s, legacy, frozen, parent_names, folds, device)
    frozen_parameter_names = [name for name, parameter in current_model.named_parameters() if not parameter.requires_grad]
    if len(frozen_parameter_names) != len(list(current_model.named_parameters())):
        raise RuntimeError("current model is not fully frozen")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    records: list[dict[str, Any]] = []
    for index, name in enumerate(debug_names):
        records.append(make_record(
            args=args, v3s=v3s, legacy=legacy, frozen=frozen, name=name,
            historical_fold=debug_folds[name], outer_fold=index % OUTER_FOLDS,
            current_model=current_model, device=device,
        ))
        heartbeat("s0_cache", index + 1, len(debug_names))
    transport = transport_equivalence(records)
    gate = smoke_gate(flat_items(records), device)
    if device.type == "cuda":
        peak = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
    else:
        peak = 0.0
    structural_pass = bool(
        transport["pass"] and gate["pass"]
        and gate["microfit"]["loss_reduction_fraction"] >= MICROFIT_MIN_REDUCTION
        and gate["trainable_parameter_scope_exact"]
    )
    state = "COMPLETED_GATE_PASS" if structural_pass else "COMPLETED_GATE_FAIL"
    decision = "A1X_V3_S0_PASS_AUTHORIZE_D0_DESIGN_ONLY" if structural_pass else "A1X_V3_S0_ENGINEERING_FAIL_STOP"
    authorizes = "A1X_V3_D0_DESIGN_ONLY" if structural_pass else "NONE"
    summary = {
        "schema_version": 1, "route_id": ROUTE_ID, "stage": "s0",
        "evidence_role": "engineering_debug", "name_count": 32,
        "transport": transport, "gate": gate, "peak_gpu_memory_mib": peak,
        "frozen_parameter_count": len(frozen_parameter_names),
        "frozen_parameter_names_sha256": canonical_hash(frozen_parameter_names),
        "confirmation_images_targets_outcomes_touched": False,
        "canary_touched": False, "locked_test_touched": False,
    }
    output = Path(output_dir)
    atomic_json(output / "a1x_v3_s0_summary.json", summary)
    closeout = {
        **common_closeout("s0", started), "state": state, "decision": decision,
        "authorizes": authorizes, "evidence_role": "engineering_debug",
        "gate_type": "structural_integrity", "structural_valid": structural_pass,
        "confirmation_images_targets_outcomes_touched": False,
        "canary_touched": False, "locked_test_touched": False,
        "candidate_selected": False, "training_scope": "engineering_microfit_only",
        "summary_sha256": sha256_file(output / "a1x_v3_s0_summary.json"),
    }
    atomic_json(Path(os.environ["CLOSEOUT_PATH"]), closeout)
    print(json.dumps(closeout, sort_keys=True), flush=True)
    return closeout


def audit(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--a1r-root", required=True)
    parser.add_argument("--expected-a1r-commit", required=True)
    own, remainder = parser.parse_known_args(argv)
    root = Path(own.a1r_root).resolve()
    import subprocess
    head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain"], check=True, capture_output=True, text=True).stdout.strip()
    if head != own.expected_a1r_commit or dirty:
        raise RuntimeError("pinned A1R source checkout identity mismatch")
    global A1R
    A1R = load_module(root / "experience_docx" / "tools" / "chd_rm_v4a_a1r_representation_sufficiency.py", "a1x_v3_pinned_a1r")
    A1R.run_a1r = run_a1x
    A1R.audit(remainder)


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] != "audit" or os.environ.get("A1X_STAGE") != "s0":
        raise SystemExit("A1X-v3 initial bundle permits only audit/S0")
    audit(sys.argv[2:])


if __name__ == "__main__":
    main()
