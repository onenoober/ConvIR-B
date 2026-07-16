#!/usr/bin/env python3
"""A1X v2 exact-half deployable-input accessibility experiment.

Pinned A1R/A1F code owns source reconstruction and safe replay. This file owns
only the frozen A1X head, cross-fit units, exact resume, and terminal summary.
Intermediate confirmation metrics are not emitted before all eight units exist.
"""

from __future__ import annotations

import argparse
import csv
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

import numpy as np
import torch
import torch.nn.functional as F


ROUTE_ID = "haze4k_v5_chd_rm_v4a_a1x_accessibility_v2_20260716"
CELLS = ("true", "shuffle")
OPERATORS = ("D_ref", "D_rep")
FORMAL_START = 768
FORMAL_COUNT = 432
OUTER_FOLDS = 4
FORMAL_FOLD_COUNT = 108
SMOKE_START = 256
SMOKE_COUNT = 32
EPOCHS = 8
BATCH_SIZE = 8
SEED = 3407
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 0.1
BOOTSTRAP_REPLICATES = 4000
PARAMETER_LIMIT = 300_000
MAC_LIMIT_LARGEST_EXACT_HALF = 600_000_000
GAIN_THRESHOLD_DB = 0.020
RETENTION_THRESHOLD = 0.25
SHUFFLE_GAP_THRESHOLD_DB = 0.005
ORACLE_ADEQUACY_PASS_DB = 0.080
ORACLE_ADEQUACY_FAIL_DB = 0.020

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


def atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError("refusing to write empty rows")
    fields = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def progress(phase: str, completed: int, total: int) -> None:
    value = {
        "route_id": ROUTE_ID,
        "run_id": os.environ["RUN_ID"],
        "phase": phase,
        "completed": completed,
        "total": total,
        "timestamp": time.time(),
    }
    line = json.dumps(value, sort_keys=True)
    print(line, flush=True)
    status = Path(os.environ["STATUS_PATH"])
    status.parent.mkdir(parents=True, exist_ok=True)
    with status.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    atomic_json(Path(os.environ["HEARTBEAT_PATH"]), value)


def heartbeat(phase: str, completed: int, total: int) -> None:
    atomic_json(Path(os.environ["HEARTBEAT_PATH"]), {
        "route_id": ROUTE_ID,
        "run_id": os.environ["RUN_ID"],
        "phase": phase,
        "completed": completed,
        "total": total,
        "timestamp": time.time(),
    })


def route_config() -> dict[str, Any]:
    return {
        "route_id": ROUTE_ID,
        "input_roles": ["hazy_rgb", "base_rgb", "old_0p125_rgb", "old_0p25_rgb", "current_delta_u"],
        "input_channels": 15,
        "cells": list(CELLS),
        "operators": list(OPERATORS),
        "formal_slice": [FORMAL_START, FORMAL_START + FORMAL_COUNT],
        "outer_folds": OUTER_FOLDS,
        "fold_rule": "local_index_modulo_4",
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "grad_clip": GRAD_CLIP,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "thresholds": {
            "gain_lcb95_db": GAIN_THRESHOLD_DB,
            "retention_lcb95": RETENTION_THRESHOLD,
            "true_minus_shuffle_lcb95_db": SHUFFLE_GAP_THRESHOLD_DB,
            "oracle_adequacy_pass_lcb95_db": ORACLE_ADEQUACY_PASS_DB,
            "oracle_adequacy_fail_ucb95_db": ORACLE_ADEQUACY_FAIL_DB,
        },
    }


def exact_half(value: torch.Tensor, shape: tuple[int, int], support: bool = False) -> torch.Tensor:
    result = F.interpolate(value.float(), size=shape, mode="bilinear", align_corners=False, antialias=False)
    return result > 0.0 if support else result


def make_record(
    *, args: Any, v3s: Any, legacy: Any, frozen: Any, name: str,
    historical_fold: int, outer_fold: int, current_model: torch.nn.Module,
    device: torch.device, retain_sample: bool,
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
        "name": name,
        "outer_fold": outer_fold,
        "historical_fold": historical_fold,
        "shape": shape,
        "operators": {},
    }
    if retain_sample:
        record["sample"] = sample
    for operator in OPERATORS:
        step = sample["steps"][operator]
        current = A1R.SOURCE.V3W.delta_for("output", current_model, sample, operator)
        target = sample["support"] * A1R.PARENT.clamp_channelwise(target_step - step, delta_bound)
        zero = torch.zeros_like(current)
        old_125, old_250, _, _ = v3s.candidate_predictions(sample["base"], step, zero)
        inputs = torch.cat([
            exact_half(sample["hazy"], shape),
            exact_half(sample["base"], shape),
            exact_half(old_125, shape),
            exact_half(old_250, shape),
            exact_half(current, shape),
        ], dim=1)
        if inputs.shape[1] != 15:
            raise RuntimeError("input whitelist did not produce 15 channels")
        record["operators"][operator] = {
            "input_low": inputs.detach().cpu(),
            "support_low": exact_half(sample["support"], shape, support=True).detach().cpu(),
            "current_low": exact_half(current, shape).detach().cpu(),
            "target_low": exact_half(target, shape).detach().cpu(),
            "step": step.detach() if retain_sample else None,
            "current": current.detach() if retain_sample else None,
            "target": target.detach() if retain_sample else None,
        }
    return record


def flat_items(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "name": record["name"],
            "operator": operator,
            "outer_fold": record["outer_fold"],
            "shape": tuple(record["shape"]),
            **{key: value for key, value in record["operators"][operator].items() if key.endswith("_low")},
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


def shuffled_targets(items: list[dict[str, Any]]) -> tuple[dict[tuple[str, str], torch.Tensor], str]:
    groups: dict[tuple[str, tuple[int, int]], list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        groups[(item["operator"], tuple(item["shape"]))].append(item)
    result: dict[tuple[str, str], torch.Tensor] = {}
    identity: dict[str, str] = {}
    for key, group in sorted(groups.items()):
        ordered = sorted(group, key=lambda item: item["name"])
        if len(ordered) < 2:
            raise RuntimeError(f"shuffle block is too small: {key}")
        for index, item in enumerate(ordered):
            other = ordered[(index + 1) % len(ordered)]
            if item["name"] == other["name"]:
                raise RuntimeError("shuffle self-pair")
            result[(item["name"], item["operator"])] = other["target_low"]
            identity[f"{item['operator']}:{item['name']}"] = other["name"]
    return result, canonical_hash(identity)


def new_head(device: torch.device, seed: int) -> torch.nn.Module:
    repo = Path(os.environ["REMOTE_REPO"])
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from Dehazing.ITS.models.A1XAccess import A1X_ACCESS_Head
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return A1X_ACCESS_Head().to(device)


def endpoint(
    head: torch.nn.Module, inputs: torch.Tensor, current: torch.Tensor,
    support: torch.Tensor, bound: torch.Tensor, mean: torch.Tensor, std: torch.Tensor,
) -> torch.Tensor:
    correction = head((inputs - mean) / std)
    value = current + support.to(current.dtype) * bound * correction
    return torch.maximum(torch.minimum(value, bound), -bound)


def train_unit(
    items: list[dict[str, Any]], cell: str, fold: int, device: torch.device,
    *, smoke: bool = False,
) -> tuple[torch.nn.Module, torch.Tensor, torch.Tensor, list[dict[str, Any]], float, str | None]:
    mean, std = feature_stats(items)
    head = new_head(device, SEED + fold * 100)
    optimizer = torch.optim.AdamW(head.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    mapping, mapping_hash = shuffled_targets(items) if cell == "shuffle" else ({}, None)
    batches = shape_batches(items)
    epochs = 1 if smoke else EPOCHS
    first_gradient = 0.0
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        head.train()
        total = 0.0
        used = 0
        for batch in batches[:2] if smoke else batches:
            inputs = torch.cat([item["input_low"] for item in batch]).to(device)
            current = torch.cat([item["current_low"] for item in batch]).to(device)
            support = torch.cat([item["support_low"] for item in batch]).to(device)
            target = torch.cat([
                mapping[(item["name"], item["operator"])] if cell == "shuffle" else item["target_low"]
                for item in batch
            ]).to(device)
            bound = current.new_tensor(RUN_ARGS.delta_bound).view(1, 3, 1, 1)
            optimizer.zero_grad(set_to_none=True)
            predicted = endpoint(
                head, inputs, current, support, bound, mean.to(device), std.to(device),
            )
            active = support.expand_as(predicted).to(predicted.dtype)
            loss = (((predicted - target) / bound.clamp_min(1e-8)).square() * active).sum() / active.sum().clamp_min(1.0)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"nonfinite loss: {cell}/fold{fold}")
            loss.backward()
            gradient = float(torch.nn.utils.clip_grad_norm_(head.parameters(), GRAD_CLIP).item())
            if not math.isfinite(gradient):
                raise FloatingPointError(f"nonfinite gradient: {cell}/fold{fold}")
            if epoch == 1 and used == 0:
                first_gradient = gradient
            optimizer.step()
            total += float(loss.detach().item())
            used += 1
            if used % 8 == 0:
                heartbeat(f"train_{cell}_fold{fold}_epoch{epoch}", used, len(batches))
        history.append({
            "epoch": epoch,
            "batches": used,
            "mean_target_loss": total / max(used, 1),
        })
    return head.eval(), mean, std, history, first_gradient, mapping_hash


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
    if len(shapes) != 2:
        raise RuntimeError(f"S0 requires both exact-half native shapes, observed={shapes}")
    noop = 0.0
    for item in items[:8]:
        current = item["current_low"].to(device)
        bound = current.new_tensor(RUN_ARGS.delta_bound).view(1, 3, 1, 1)
        predicted = endpoint(
            head,
            item["input_low"].to(device),
            current,
            item["support_low"].to(device),
            bound,
            torch.zeros((1, 15, 1, 1), device=device),
            torch.ones((1, 15, 1, 1), device=device),
        )
        noop = max(noop, float((predicted - current).abs().max().item()))
    true = train_unit(items, "true", 0, device, smoke=True)
    shuffled = train_unit(items, "shuffle", 0, device, smoke=True)
    macs = {f"{shape[0]}x{shape[1]}": count_conv_macs(head, shape, device) for shape in shapes}
    return {
        "input_channels": 15,
        "native_exact_half_shapes": [list(shape) for shape in shapes],
        "parameter_count": parameter_count,
        "parameter_limit": PARAMETER_LIMIT,
        "macs_by_shape": macs,
        "mac_limit_largest_exact_half": MAC_LIMIT_LARGEST_EXACT_HALF,
        "zero_noop_max_abs": noop,
        "true_first_gradient_norm": true[4],
        "shuffle_first_gradient_norm": shuffled[4],
        "latency_memory_role": "descriptive_only_not_a_gate",
        "pass": bool(
            parameter_count <= PARAMETER_LIMIT
            and max(macs.values()) <= MAC_LIMIT_LARGEST_EXACT_HALF
            and noop == 0.0
            and true[4] > 0.0
            and shuffled[4] > 0.0
        ),
    }


def unit_contract(fold: int, cell: str, train_names: list[str], heldout_names: list[str]) -> dict[str, Any]:
    return {
        "config_sha256": canonical_hash(route_config()),
        "fold": fold,
        "cell": cell,
        "train_names_sha256": canonical_hash(train_names),
        "heldout_names_sha256": canonical_hash(heldout_names),
        "route_commit": os.environ["EXPECTED_ROUTE_COMMIT"],
        "runner_sha256": os.environ["RUNNER_SHA256"],
    }


def load_or_train_unit(
    *, output: Path, items: list[dict[str, Any]], fold: int, cell: str,
    train_names: list[str], heldout_names: list[str], device: torch.device,
) -> tuple[torch.nn.Module, torch.Tensor, torch.Tensor, dict[str, Any]]:
    contract = unit_contract(fold, cell, train_names, heldout_names)
    contract_hash = canonical_hash(contract)
    path = output / "units" / f"fold{fold}_{cell}.pt"
    if path.exists():
        value = torch.load(path, map_location="cpu")
        if value.get("contract_sha256") != contract_hash or value.get("contract") != contract:
            raise RuntimeError(f"completed unit contract mismatch: {path.name}")
        head = new_head(device, SEED + fold * 100)
        head.load_state_dict(value["head_state"], strict=True)
        return head.eval(), value["feature_mean"], value["feature_std"], {
            "fold": fold,
            "cell": cell,
            "relative_path": str(path.relative_to(output)),
            "sha256": sha256_file(path),
            "contract_sha256": contract_hash,
            "resumed": True,
            "first_gradient_norm": value["first_gradient_norm"],
            "shuffle_mapping_sha256": value["shuffle_mapping_sha256"],
        }
    head, mean, std, history, gradient, mapping_hash = train_unit(items, cell, fold, device)
    value = {
        "contract": contract,
        "contract_sha256": contract_hash,
        "head_state": {key: tensor.detach().cpu() for key, tensor in head.state_dict().items()},
        "feature_mean": mean,
        "feature_std": std,
        "history": history,
        "first_gradient_norm": gradient,
        "shuffle_mapping_sha256": mapping_hash,
    }
    atomic_torch(path, value)
    return head, mean, std, {
        "fold": fold,
        "cell": cell,
        "relative_path": str(path.relative_to(output)),
        "sha256": sha256_file(path),
        "contract_sha256": contract_hash,
        "resumed": False,
        "first_gradient_norm": gradient,
        "shuffle_mapping_sha256": mapping_hash,
    }


def psnr(mse: float) -> float:
    return 10.0 * math.log10(1.0 / max(mse, 1e-30))


def evaluate_record(
    *, record: dict[str, Any], heads: dict[tuple[int, str], tuple[torch.nn.Module, torch.Tensor, torch.Tensor]],
    v3s: Any, device: torch.device,
) -> list[dict[str, Any]]:
    assert A1R is not None
    sample = record["sample"]
    fold = int(record["outer_fold"])
    rows: list[dict[str, Any]] = []
    for operator in OPERATORS:
        values = record["operators"][operator]
        step = values["step"]
        current = values["current"]
        target = values["target"]
        old_125, old_250, _, current_250 = v3s.candidate_predictions(sample["base"], step, current)
        old_125_mse = A1R.PARENT.a0p.scalar(v3s.per_image_mse(old_125, sample["label"]))
        old_250_mse = A1R.PARENT.a0p.scalar(v3s.per_image_mse(old_250, sample["label"]))
        current_250_mse = A1R.PARENT.a0p.scalar(v3s.per_image_mse(current_250, sample["label"]))
        shrink, _ = A1R.canonical_shrink(sample, step, current, old_125_mse, old_250_mse)
        exact_target_full = sample["support"] * F.interpolate(
            values["target_low"].to(device), size=current.shape[-2:], mode="bilinear", align_corners=False,
        )
        delta_bound = current.new_tensor(RUN_ARGS.delta_bound).view(1, 3, 1, 1)
        exact_target_full = A1R.PARENT.clamp_channelwise(exact_target_full, delta_bound)
        oracle = A1R.selected_union(
            sample, step, current, exact_target_full, shrink, old_125_mse, old_250_mse, "exact_half_oracle",
        )
        shrink_psnr = psnr(float(shrink["high_mse"]))
        oracle_gain = psnr(float(oracle["high_mse"])) - shrink_psnr
        for cell in CELLS:
            head, mean, std = heads[(fold, cell)]
            with torch.no_grad():
                predicted_low = endpoint(
                    head,
                    values["input_low"].to(device),
                    values["current_low"].to(device),
                    values["support_low"].to(device),
                    delta_bound,
                    mean.to(device),
                    std.to(device),
                )
            predicted_full = sample["support"] * F.interpolate(
                predicted_low, size=current.shape[-2:], mode="bilinear", align_corners=False,
            )
            predicted_full = A1R.PARENT.clamp_channelwise(predicted_full, delta_bound)
            selected = A1R.selected_union(
                sample, step, current, predicted_full, shrink, old_125_mse, old_250_mse, cell,
            )
            selected_psnr = psnr(float(selected["high_mse"]))
            tolerance = A1R.PARENT.numerical_tolerance(
                float(shrink["high_mse"]), float(selected["high_mse"]),
            )
            target_active = values["support_low"].to(device).expand_as(predicted_low)
            endpoint_mse = float(
                ((predicted_low - values["target_low"].to(device)).square() * target_active).sum().item()
                / max(float(target_active.sum().item()), 1.0)
            )
            cosine = float(F.cosine_similarity(
                predicted_low.flatten(1), values["target_low"].to(device).flatten(1), dim=1,
            ).item())
            row = {
                "name": record["name"],
                "outer_fold": fold,
                "operator": operator,
                "cell": cell,
                "old_125_mse": old_125_mse,
                "old_250_mse": old_250_mse,
                "current_250_mse": current_250_mse,
                "shrink_high_mse": float(shrink["high_mse"]),
                "oracle_high_mse": float(oracle["high_mse"]),
                "selected_low_mse": float(selected["low_mse"]),
                "selected_high_mse": float(selected["high_mse"]),
                "selected_family": str(selected["family"]),
                "selected_grid_value": float(selected["grid_value"]),
                "direction_selected": bool(
                    selected["family"] == cell
                    and float(selected["high_mse"]) < float(shrink["high_mse"]) - tolerance
                ),
                "gain_vs_shrink_db": selected_psnr - shrink_psnr,
                "gain_vs_old25_db": selected_psnr - psnr(old_250_mse),
                "oracle_gain_db": oracle_gain,
                "target_endpoint_mse": endpoint_mse,
                "target_cosine": cosine,
                "anchor_safe": bool(float(selected["low_mse"]) <= old_125_mse + A1R.PARENT.numerical_tolerance(float(selected["low_mse"]), old_125_mse)),
                "predecessor_safe": bool(float(selected["high_mse"]) <= old_250_mse + A1R.PARENT.numerical_tolerance(float(selected["high_mse"]), old_250_mse)),
                "severe_vs_old25": bool(selected_psnr - psnr(old_250_mse) <= -0.2),
                "hard_vs_old25": bool(selected_psnr - psnr(old_250_mse) <= -0.5),
            }
            numeric = [value for value in row.values() if isinstance(value, float)]
            if not all(math.isfinite(value) for value in numeric):
                raise FloatingPointError("nonfinite formal row")
            rows.append(row)
    return rows


def quantile_bounds(values: np.ndarray) -> tuple[float, float]:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    return float(ordered[199]), float(ordered[3799])


def bootstrap_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    names = sorted({str(row["name"]) for row in rows})
    expected = FORMAL_COUNT * len(OPERATORS) * len(CELLS)
    if len(names) != FORMAL_COUNT or len(rows) != expected:
        raise RuntimeError(f"formal rows are incomplete: names={len(names)} rows={len(rows)}")
    keyed = {(str(row["name"]), str(row["operator"]), str(row["cell"])): row for row in rows}
    arrays: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for cell in CELLS:
        for operator in OPERATORS:
            arrays[(cell, operator)] = {
                key: np.asarray([float(keyed[(name, operator, cell)][key]) for name in names])
                for key in ("gain_vs_shrink_db", "oracle_gain_db", "direction_selected")
            }
    generator = np.random.Generator(np.random.PCG64(SEED))
    gain = np.empty(BOOTSTRAP_REPLICATES)
    retention = np.empty(BOOTSTRAP_REPLICATES)
    shuffle_gap = np.empty(BOOTSTRAP_REPLICATES)
    oracle = np.empty(BOOTSTRAP_REPLICATES)
    repairable = np.empty(BOOTSTRAP_REPLICATES)
    for draw in range(BOOTSTRAP_REPLICATES):
        indices = generator.integers(0, len(names), size=len(names), endpoint=False)
        true_by_operator = {
            operator: float(np.mean(arrays[("true", operator)]["gain_vs_shrink_db"][indices]))
            for operator in OPERATORS
        }
        oracle_by_operator = {
            operator: float(np.mean(arrays[("true", operator)]["oracle_gain_db"][indices]))
            for operator in OPERATORS
        }
        gain[draw] = min(true_by_operator.values())
        oracle[draw] = min(oracle_by_operator.values())
        retention[draw] = min(
            true_by_operator[operator] / max(oracle_by_operator[operator], 1e-30)
            for operator in OPERATORS
        )
        shuffle_gap[draw] = min(
            float(np.mean(
                arrays[("true", operator)]["gain_vs_shrink_db"][indices]
                - arrays[("shuffle", operator)]["gain_vs_shrink_db"][indices]
            )) for operator in OPERATORS
        )
        repairable[draw] = min(
            float(np.mean(arrays[("true", operator)]["direction_selected"][indices]))
            for operator in OPERATORS
        )
    result: dict[str, Any] = {
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": SEED,
        "paired_image_unit": True,
        "worst_operator_within_draw": True,
    }
    for name, values in (
        ("safe_gain_db", gain),
        ("exact_half_retention", retention),
        ("true_minus_shuffle_db", shuffle_gap),
        ("exact_half_oracle_gain_db", oracle),
        ("repairable_fraction_supporting", repairable),
    ):
        lower, upper = quantile_bounds(values)
        result[name] = float(np.mean(values))
        result[name + "_lcb95"] = lower
        result[name + "_ucb95"] = upper
    return result


def formal_decision(summary: dict[str, Any], rows: list[dict[str, Any]]) -> tuple[str, str, str]:
    safety = all(
        bool(row["anchor_safe"])
        and bool(row["predecessor_safe"])
        and not bool(row["severe_vs_old25"])
        and not bool(row["hard_vs_old25"])
        for row in rows
    )
    if not safety:
        return "COMPLETED_GATE_FAIL", "A1X_V2_POINTWISE_SAFETY_FAIL", "NONE"
    if float(summary["exact_half_oracle_gain_db_ucb95"]) < ORACLE_ADEQUACY_FAIL_DB:
        return "COMPLETED_GATE_FAIL", "A1X_V2_EXACT_HALF_ADEQUACY_FAIL", "NONE"
    if float(summary["exact_half_oracle_gain_db_lcb95"]) < ORACLE_ADEQUACY_PASS_DB:
        return "COMPLETED_INCONCLUSIVE", "A1X_V2_EXACT_HALF_ADEQUACY_INCONCLUSIVE", "R3_REVIEW_ONLY"
    primary = (
        ("safe_gain_db", GAIN_THRESHOLD_DB),
        ("exact_half_retention", RETENTION_THRESHOLD),
        ("true_minus_shuffle_db", SHUFFLE_GAP_THRESHOLD_DB),
    )
    if all(float(summary[name + "_lcb95"]) >= threshold for name, threshold in primary):
        return "COMPLETED_GATE_PASS", "A1X_V2_ACCESSIBILITY_PASS", "R3_REVIEW_ONLY"
    if any(float(summary[name + "_ucb95"]) < threshold for name, threshold in primary):
        return "COMPLETED_GATE_FAIL", "A1X_V2_ACCESSIBILITY_FAIL", "NONE"
    return "COMPLETED_INCONCLUSIVE", "A1X_V2_ACCESSIBILITY_INCONCLUSIVE", "R3_REVIEW_ONLY"


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


def run_a1x(
    args: Any, v3s: Any, legacy: Any, frozen: Any, names: list[str],
    folds: dict[str, int], device: torch.device, output_dir: str,
) -> dict[str, Any]:
    del names, folds
    global RUN_ARGS
    RUN_ARGS = args
    assert A1R is not None and A1R.AUDIT is not None
    started = time.monotonic()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    evid_stage = Path(os.environ["EVID_STAGE"])
    A1R.PARENT.a0p.validate_frozen_args(args)
    all_names, historical_folds = v3s.load_names_and_folds(args, legacy)
    if len(all_names) < FORMAL_START + FORMAL_COUNT:
        raise RuntimeError("frozen 1200-name source is incomplete")
    current_model = load_base_state(args, v3s, legacy, frozen, all_names, historical_folds, device)
    stage = os.environ["A1X_STAGE"]
    if stage == "s0":
        stage_names = list(all_names[SMOKE_START:SMOKE_START + SMOKE_COUNT])
        if len(stage_names) != SMOKE_COUNT or set(stage_names) & set(all_names[FORMAL_START:FORMAL_START + FORMAL_COUNT]):
            raise RuntimeError("S0 names overlap A1X confirmation")
        records = []
        for index, name in enumerate(stage_names):
            records.append(make_record(
                args=args, v3s=v3s, legacy=legacy, frozen=frozen, name=name,
                historical_fold=historical_folds[name], outer_fold=index % OUTER_FOLDS,
                current_model=current_model, device=device, retain_sample=False,
            ))
            progress("s0_cache", index + 1, len(stage_names))
        gate = smoke_gate(flat_items(records), device)
        state = "COMPLETED_GATE_PASS" if gate["pass"] else "COMPLETED_GATE_FAIL"
        decision = "A1X_V2_S0_PASS_AUTHORIZE_FORMAL" if gate["pass"] else "A1X_V2_S0_ENGINEERING_FAIL_STOP"
        authorizes = "A1X_V2_FORMAL_ONLY" if gate["pass"] else "NONE"
        summary = {
            "route_id": ROUTE_ID,
            "stage": "s0",
            "evidence_role": "engineering_debug",
            "a1x_data_accessed": False,
            "stage_name_count": len(stage_names),
            "gate": gate,
        }
        atomic_json(output / "a1x_v2_s0_summary.json", summary)
        closeout = {
            **common_closeout("s0", started),
            "state": state,
            "decision": decision,
            "authorizes": authorizes,
            "evidence_role": "engineering_debug",
            "a1x_data_accessed": False,
            "structural_gate": gate,
        }
        atomic_json(Path(os.environ["CLOSEOUT_PATH"]), closeout)
        print(json.dumps(closeout, sort_keys=True), flush=True)
        return closeout
    if stage != "formal":
        raise RuntimeError(f"unknown A1X stage: {stage}")

    formal_names = list(all_names[FORMAL_START:FORMAL_START + FORMAL_COUNT])
    if len(formal_names) != FORMAL_COUNT or len(set(formal_names)) != FORMAL_COUNT:
        raise RuntimeError("A1X confirmation names are not exactly 432 unique names")
    outer_folds = {name: index % OUTER_FOLDS for index, name in enumerate(formal_names)}
    if {fold: sum(value == fold for value in outer_folds.values()) for fold in range(OUTER_FOLDS)} != {fold: FORMAL_FOLD_COUNT for fold in range(OUTER_FOLDS)}:
        raise RuntimeError("A1X outer folds are not four balanced 108-name folds")
    records: list[dict[str, Any]] = []
    for index, name in enumerate(formal_names):
        records.append(make_record(
            args=args, v3s=v3s, legacy=legacy, frozen=frozen, name=name,
            historical_fold=historical_folds[name], outer_fold=outer_folds[name],
            current_model=current_model, device=device, retain_sample=False,
        ))
        progress("formal_cache", index + 1, FORMAL_COUNT)
    items = flat_items(records)
    heads: dict[tuple[int, str], tuple[torch.nn.Module, torch.Tensor, torch.Tensor]] = {}
    unit_rows: list[dict[str, Any]] = []
    unit_index = 0
    for fold in range(OUTER_FOLDS):
        train_names = [name for name in formal_names if outer_folds[name] != fold]
        heldout_names = [name for name in formal_names if outer_folds[name] == fold]
        train_items = [item for item in items if int(item["outer_fold"]) != fold]
        if len(train_names) != 324 or len(heldout_names) != 108 or len(train_items) != 648:
            raise RuntimeError(f"invalid fold partition: {fold}")
        for cell in CELLS:
            head, mean, std, unit = load_or_train_unit(
                output=output, items=train_items, fold=fold, cell=cell,
                train_names=train_names, heldout_names=heldout_names, device=device,
            )
            if float(unit["first_gradient_norm"]) <= 0.0:
                raise RuntimeError(f"nonpositive first gradient: fold={fold} cell={cell}")
            heads[(fold, cell)] = (head, mean, std)
            unit_rows.append(unit)
            unit_index += 1
            progress("formal_units", unit_index, OUTER_FOLDS * len(CELLS))
    if len(unit_rows) != 8 or len({row["contract_sha256"] for row in unit_rows}) != 8:
        raise RuntimeError("formal units are incomplete or not distinct")
    unit_manifest = {
        "route_id": ROUTE_ID,
        "config_sha256": canonical_hash(route_config()),
        "unit_count": len(unit_rows),
        "units": unit_rows,
        "resume_policy": "complete_hash_matching_fold_cell_units_only",
    }
    atomic_json(output / "a1x_v2_unit_manifest.json", unit_manifest)

    # Confirmation metrics remain sealed until every unit above is complete.
    rows: list[dict[str, Any]] = []
    for index, name in enumerate(formal_names):
        record = make_record(
            args=args, v3s=v3s, legacy=legacy, frozen=frozen, name=name,
            historical_fold=historical_folds[name], outer_fold=outer_folds[name],
            current_model=current_model, device=device, retain_sample=True,
        )
        rows.extend(evaluate_record(record=record, heads=heads, v3s=v3s, device=device))
        progress("formal_evaluation", index + 1, FORMAL_COUNT)
    summary = bootstrap_summary(rows)
    state, decision, authorizes = formal_decision(summary, rows)
    raw_path = output / "a1x_v2_oof_rows_cloud_only.csv"
    write_rows(raw_path, rows)
    compact = {
        "route_id": ROUTE_ID,
        "stage": "formal",
        "evidence_role": "confirmation",
        "name_count": FORMAL_COUNT,
        "row_count": len(rows),
        "fold_counts": {str(fold): FORMAL_FOLD_COUNT for fold in range(OUTER_FOLDS)},
        "bootstrap": summary,
        "eligibility_guards": {
            "pointwise_safety": all(bool(row["anchor_safe"]) and bool(row["predecessor_safe"]) for row in rows),
            "exact_half_adequacy_lcb95_db": summary["exact_half_oracle_gain_db_lcb95"],
        },
        "supporting": {
            "repairable_fraction": summary["repairable_fraction_supporting"],
            "repairable_fraction_lcb95": summary["repairable_fraction_supporting_lcb95"],
        },
        "decision": decision,
    }
    atomic_json(output / "a1x_v2_formal_summary.json", compact)
    closeout = {
        **common_closeout("formal", started),
        "state": state,
        "decision": decision,
        "authorizes": authorizes,
        "evidence_role": "confirmation",
        "a1x_data_accessed": True,
        "name_count": FORMAL_COUNT,
        "row_count": len(rows),
        "bootstrap": summary,
        "raw_rows_cloud_only": str(raw_path),
        "unit_manifest_sha256": sha256_file(output / "a1x_v2_unit_manifest.json"),
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
    A1R = load_module(
        root / "experience_docx" / "tools" / "chd_rm_v4a_a1r_representation_sufficiency.py",
        "a1x_v2_pinned_a1r",
    )
    A1R.run_a1r = run_a1x
    A1R.audit(remainder)


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] != "audit":
        raise SystemExit("usage: chd_rm_v4a_a1x_accessibility_v2.py audit --a1r-root ...")
    audit(sys.argv[2:])


if __name__ == "__main__":
    main()
