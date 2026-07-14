#!/usr/bin/env python3
"""Fresh-name OOF probe audit for bounded Delta-u representation sufficiency."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ROUTE_ID = "haze4k_v5_chd_rm_v4a_a1r_representation_sufficiency_20260714"
PARENT_ROUTE_ID = "haze4k_v5_chd_rm_v4a_a1f_deltau_action_feasibility_20260714"
EXPECTED_PARENT_REVIEW = {
    "route_id": PARENT_ROUTE_ID,
    "state": "COMPLETED_R3_REVIEW",
    "decision": "V4A_A1F_SAFE_DIRECTION_HEADROOM_PASS_AUTHORIZE_A1R_REPRESENTATION_SUFFICIENCY_DESIGN_ONLY",
    "authorizes": "A1R_ROUTE_DESIGN_ONLY",
}
CELL_SPECS = {
    "output_linear": {"representation": "output", "readout": "linear", "shuffled": False},
    "output_spatial": {"representation": "output", "readout": "spatial", "shuffled": False},
    "context_linear": {"representation": "context", "readout": "linear", "shuffled": False},
    "context_spatial": {"representation": "context", "readout": "spatial", "shuffled": False},
    "context_spatial_shuffled": {"representation": "context", "readout": "spatial", "shuffled": True},
}
PRIMARY_CELL = "context_spatial"
SHUFFLED_CELL = "context_spatial_shuffled"
BOOTSTRAP_REPLICATES = 4000
BOOTSTRAP_SEED = 3407
GAIN_THRESHOLD_DB = 0.020
RETENTION_THRESHOLD = 0.25
SHUFFLE_GAP_THRESHOLD_DB = 0.005
REPAIRABLE_THRESHOLD = 0.20

PARENT: Any = None
SOURCE: Any = None
AUDIT: argparse.Namespace | None = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty rows: {path}")
    fields = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def emit_progress(status_file: Path, value: dict[str, Any]) -> None:
    line = json.dumps(value, sort_keys=True)
    print(line, flush=True)
    with status_file.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def load_module(path: Path, name: str) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_parent_review(path: Path, expected_sha256: str) -> dict[str, Any]:
    if sha256_file(path) != expected_sha256:
        raise RuntimeError("A1F R3 review hash mismatch")
    value = json.loads(path.read_text(encoding="utf-8"))
    for key, expected in EXPECTED_PARENT_REVIEW.items():
        if value.get(key) != expected:
            raise RuntimeError(f"A1F R3 review tuple mismatch: {key}={value.get(key)!r}")
    return value


class DeltaEndpointProbe(nn.Module):
    def __init__(self, input_channels: int, readout: str, width: int, delta_bound: tuple[float, float, float]):
        super().__init__()
        self.readout = readout
        if readout == "linear":
            self.head = nn.Conv2d(input_channels, 3, kernel_size=1)
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)
            self.stem = None
            self.depthwise = None
            self.pointwise = None
        elif readout == "spatial":
            self.stem = nn.Conv2d(input_channels, width, kernel_size=3, padding=1)
            self.depthwise = nn.Conv2d(width, width, kernel_size=3, padding=1, groups=width)
            self.pointwise = nn.Conv2d(width, width, kernel_size=1)
            self.head = nn.Conv2d(width, 3, kernel_size=1)
            for layer in (self.stem, self.depthwise, self.pointwise):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                nn.init.zeros_(layer.bias)
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)
        else:
            raise ValueError(f"unknown readout: {readout}")
        self.activation = nn.GELU()
        self.register_buffer(
            "delta_bound",
            torch.as_tensor(delta_bound, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=True,
        )

    def forward(self, features: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        value = features
        if self.readout == "spatial":
            assert self.stem is not None and self.depthwise is not None and self.pointwise is not None
            value = self.activation(self.stem(value))
            value = self.activation(self.depthwise(value))
            value = self.activation(self.pointwise(value))
        raw = self.head(value)
        return support * (2.0 * self.delta_bound.to(dtype=raw.dtype) * torch.tanh(raw))


def endpoint_loss(
    correction: torch.Tensor,
    current: torch.Tensor,
    target: torch.Tensor,
    support: torch.Tensor,
    bound: torch.Tensor,
) -> torch.Tensor:
    endpoint = torch.maximum(torch.minimum(current + correction, bound), -bound)
    active = (support > 0.0).expand_as(endpoint).to(dtype=endpoint.dtype)
    normalized = (endpoint - target) / bound.clamp_min(1e-8)
    denominator = active.sum().clamp_min(1.0)
    return (normalized.square() * active).sum() / denominator


def item_features(item: dict[str, Any], representation: str) -> torch.Tensor:
    if representation == "output":
        return item["output_features"]
    if representation == "context":
        return torch.cat((item["context_base"], item["output_features"]), dim=1)
    raise ValueError(f"unknown representation: {representation}")


def feature_stats(items: list[dict[str, Any]], representation: str) -> tuple[torch.Tensor, torch.Tensor]:
    total: torch.Tensor | None = None
    square: torch.Tensor | None = None
    count = 0
    for item in items:
        value = item_features(item, representation).to(dtype=torch.float64)
        current_sum = value.sum(dim=(0, 2, 3))
        current_square = value.square().sum(dim=(0, 2, 3))
        total = current_sum if total is None else total + current_sum
        square = current_square if square is None else square + current_square
        count += value.shape[0] * value.shape[2] * value.shape[3]
    if total is None or square is None or count <= 0:
        raise RuntimeError("cannot compute empty feature statistics")
    mean = total / count
    variance = (square / count - mean.square()).clamp_min(1e-12)
    std = torch.sqrt(variance)
    return mean.to(dtype=torch.float32).view(1, -1, 1, 1), std.to(dtype=torch.float32).view(1, -1, 1, 1)


def spatial_shape(item: dict[str, Any]) -> tuple[int, int]:
    return tuple(int(value) for value in item["target_low"].shape[-2:])


def shape_batches(items: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        grouped[spatial_shape(item)].append(item)
    batches: list[list[dict[str, Any]]] = []
    for shape in sorted(grouped):
        values = sorted(grouped[shape], key=lambda row: (row["name"], row["operator"]))
        batches.extend(values[offset:offset + batch_size] for offset in range(0, len(values), batch_size))
    return batches


def shuffled_target_mapping(items: list[dict[str, Any]]) -> dict[tuple[str, str], tuple[str, torch.Tensor]]:
    grouped: dict[tuple[str, tuple[int, int]], list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        grouped[(str(item["operator"]), spatial_shape(item))].append(item)
    result: dict[tuple[str, str], tuple[str, torch.Tensor]] = {}
    for (operator, shape), group in sorted(grouped.items()):
        values = sorted(group, key=lambda row: row["name"])
        if len(values) < 2:
            raise RuntimeError(f"shuffle control requires at least two names per operator/shape block: {operator}/{shape}")
        for index, item in enumerate(values):
            other = values[(index + 1) % len(values)]
            if other["name"] == item["name"]:
                raise RuntimeError("shuffle produced a self-pair")
            if other["target_low"].shape != item["target_low"].shape:
                raise RuntimeError("shape-blocked shuffle produced unequal target shapes")
            result[(str(item["name"]), operator)] = (str(other["name"]), other["target_low"])
    return result


def shuffled_targets(items: list[dict[str, Any]]) -> dict[tuple[str, str], torch.Tensor]:
    return {key: value[1] for key, value in shuffled_target_mapping(items).items()}


def shuffled_mapping_identity(items: list[dict[str, Any]]) -> str:
    by_key = {(str(item["name"]), str(item["operator"])): item for item in items}
    mapping = {
        f"{operator}:{name}": {
            "target_name": target_name,
            "spatial_shape": list(spatial_shape(by_key[(name, operator)])),
        }
        for (name, operator), (target_name, _) in shuffled_target_mapping(items).items()
    }
    return canonical_hash(mapping)


def train_probe(
    *,
    items: list[dict[str, Any]],
    cell: str,
    fold: int,
    audit: argparse.Namespace,
    device: torch.device,
    smoke: bool,
) -> tuple[DeltaEndpointProbe, torch.Tensor, torch.Tensor, list[dict[str, Any]], float]:
    spec = CELL_SPECS[cell]
    representation = str(spec["representation"])
    mean, std = feature_stats(items, representation)
    input_channels = int(item_features(items[0], representation).shape[1])
    base_cell = PRIMARY_CELL if cell == SHUFFLED_CELL else cell
    base_index = list(CELL_SPECS).index(base_cell)
    model_seed = int(audit.seed + fold * 100 + base_index)
    torch.manual_seed(model_seed)
    torch.cuda.manual_seed_all(model_seed)
    model = DeltaEndpointProbe(
        input_channels=input_channels,
        readout=str(spec["readout"]),
        width=audit.probe_width,
        delta_bound=tuple(float(value) for value in items[0]["delta_bound"]),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=audit.learning_rate, weight_decay=audit.weight_decay)
    shuffled = shuffled_targets(items) if bool(spec["shuffled"]) else {}
    ordered_batches = shape_batches(items, audit.batch_size)
    epochs = 1 if smoke else audit.epochs
    max_batches = 2 if smoke else len(ordered_batches)
    history: list[dict[str, Any]] = []
    first_gradient_norm = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0
        for batch in ordered_batches:
            if smoke and batches >= max_batches:
                break
            features = torch.cat([item_features(row, representation) for row in batch]).to(device)
            current = torch.cat([row["current_low"] for row in batch]).to(device)
            support = torch.cat([row["support_low"] for row in batch]).to(device)
            if bool(spec["shuffled"]):
                target = torch.cat([shuffled[(row["name"], row["operator"])] for row in batch]).to(device)
            else:
                target = torch.cat([row["target_low"] for row in batch]).to(device)
            bound = model.delta_bound.to(device=device, dtype=features.dtype)
            normalized = (features - mean.to(device)) / std.to(device)
            optimizer.zero_grad(set_to_none=True)
            correction = model(normalized, support)
            loss = endpoint_loss(correction, current, target, support, bound)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"non-finite A1R loss: {cell}/fold{fold}")
            loss.backward()
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), audit.grad_clip_norm).item())
            if not math.isfinite(gradient_norm):
                raise FloatingPointError(f"non-finite A1R gradient: {cell}/fold{fold}")
            if epoch == 1 and batches == 0:
                first_gradient_norm = gradient_norm
            optimizer.step()
            total_loss += float(loss.detach().item())
            batches += 1
        history.append({
            "schema_version": 1,
            "cell": cell,
            "fold": fold,
            "epoch": epoch,
            "batches": batches,
            "mean_target_loss": total_loss / max(batches, 1),
            "model_seed": model_seed,
        })
        emit_progress(
            Path(audit.status_file),
            {"V4A_A1R_TRAIN_PROGRESS": {"stage": audit.a1r_stage, "cell": cell, "fold": fold, "epoch": epoch, "epochs": epochs}},
        )
    return model.eval(), mean, std, history, first_gradient_norm


def target_and_features(
    *,
    args: Any,
    v3s: Any,
    legacy: Any,
    frozen: Any,
    name: str,
    historical_fold: int,
    outer_fold: int,
    current_model: nn.Module,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    sample = SOURCE.V3W.frozen_output_sample(args, v3s, legacy, frozen, name, historical_fold, device)
    padded, _, _ = legacy.pad_to_factor(sample["hazy"])
    with torch.no_grad():
        context, _, _, _ = legacy.full_context_maps(frozen["control"], frozen["gate_producer"], padded)
    target_size = context.shape[-2:]
    bound = sample["base"].new_tensor(frozen["bound"]).view(1, 3, 1, 1)
    delta_bound = sample["base"].new_tensor(args.delta_bound).view(1, 3, 1, 1)
    target_step = sample["support"] * PARENT.clamp_channelwise(4.0 * (sample["label"] - sample["base"]), bound)
    operator_values: dict[str, torch.Tensor] = {}
    cached: dict[str, Any] = {}
    context_cpu = context.detach().cpu()
    for operator in SOURCE.V3W.OPERATORS:
        step = sample["steps"][operator]
        current_delta = SOURCE.V3W.delta_for("output", current_model, sample, operator)
        target_delta = sample["support"] * PARENT.clamp_channelwise(target_step - step, delta_bound)
        output_parts = [
            F.interpolate(value, size=target_size, mode="bilinear", align_corners=False)
            for value in (sample["hazy"], sample["base"], step, current_delta)
        ]
        output_features = torch.cat(output_parts, dim=1)
        support_low = F.interpolate(sample["support"], size=target_size, mode="bilinear", align_corners=False)
        current_low = F.interpolate(current_delta, size=target_size, mode="bilinear", align_corners=False)
        target_low = F.interpolate(target_delta, size=target_size, mode="bilinear", align_corners=False)
        cached[operator] = {
            "name": name,
            "operator": operator,
            "outer_fold": outer_fold,
            "output_features": output_features.detach().cpu(),
            "context_base": context_cpu,
            "support_low": support_low.detach().cpu(),
            "current_low": current_low.detach().cpu(),
            "target_low": target_low.detach().cpu(),
            "delta_bound": tuple(float(value) for value in args.delta_bound),
        }
        operator_values[f"current:{operator}"] = current_delta.detach()
        operator_values[f"target:{operator}"] = target_delta.detach()
    return cached, {**sample, **operator_values}


def canonical_shrink(sample: dict[str, torch.Tensor], step: torch.Tensor, current_delta: torch.Tensor, old_low_mse: float, old_high_mse: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    zero = torch.zeros_like(current_delta)
    candidates = PARENT.grid_metrics(sample, step, zero, current_delta, PARENT.GRID)
    zero_rows = [row for row in candidates if row["grid_value"] == 0.0]
    if len(zero_rows) != 1:
        raise RuntimeError("A1R shrink grid lacks unique zero action")
    zero_rows[0].update({"low_mse": old_low_mse, "high_mse": old_high_mse, "delta_abs": 0.0})
    for row in candidates:
        row.update({"family": "shrink", "family_rank": 0})
    selected, _ = PARENT.select_safe(candidates, old_low_mse, old_high_mse)
    return selected, candidates


def selected_union(
    sample: dict[str, torch.Tensor],
    step: torch.Tensor,
    current_delta: torch.Tensor,
    endpoint: torch.Tensor,
    shrink: dict[str, Any],
    old_low_mse: float,
    old_high_mse: float,
    family: str,
) -> dict[str, Any]:
    candidates = PARENT.grid_metrics(sample, step, current_delta, endpoint, PARENT.GRID)
    for row in candidates:
        row.update({"family": family, "family_rank": 1})
    selected, _ = PARENT.select_safe([dict(shrink), *candidates], old_low_mse, old_high_mse)
    return selected


def low_alignment(
    model: DeltaEndpointProbe,
    mean: torch.Tensor,
    std: torch.Tensor,
    item: dict[str, Any],
    representation: str,
    device: torch.device,
) -> tuple[torch.Tensor, float, float]:
    features = item_features(item, representation).to(device)
    support = item["support_low"].to(device)
    current = item["current_low"].to(device)
    target = item["target_low"].to(device)
    with torch.no_grad():
        correction = model((features - mean.to(device)) / std.to(device), support)
        endpoint = current + correction
        active = (support > 0.0).expand_as(endpoint)
        bound = model.delta_bound.to(device)
        mse = float((((endpoint - target) / bound.clamp_min(1e-8)).square().masked_select(active)).mean().item()) if bool(active.any()) else 0.0
        predicted_vector = correction.masked_select(active)
        target_vector = (target - current).masked_select(active)
        denominator = float(torch.linalg.vector_norm(predicted_vector).item() * torch.linalg.vector_norm(target_vector).item())
        cosine = float(torch.dot(predicted_vector, target_vector).item() / denominator) if denominator > 1e-30 else 0.0
    return correction, mse, cosine


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["cell"]), str(row["operator"]))].append(row)
    result: list[dict[str, Any]] = []
    for (cell, operator), values in sorted(grouped.items()):
        summary: dict[str, Any] = {
            "schema_version": 1,
            "cell": cell,
            "operator": operator,
            "image_count": len(values),
            "direction_selected_count": sum(bool(row["direction_selected"]) for row in values),
            "direction_selected_fraction": float(np.mean([float(row["direction_selected"]) for row in values])),
            "severe_count": sum(bool(row["severe_vs_old25"]) for row in values),
            "hard_count": sum(bool(row["hard_vs_old25"]) for row in values),
            "max_anchor_excess": max(float(row["selected_low_mse"] - row["old_low_mse"]) for row in values),
            "max_predecessor_excess": max(float(row["selected_high_mse"] - row["old_high_mse"]) for row in values),
        }
        for key in ("gain_vs_shrink_db", "gain_vs_current_db", "gain_vs_old25_db", "oracle_gain_db", "target_endpoint_mse", "target_cosine", "selected_grid_value"):
            array = np.asarray([float(row[key]) for row in values], dtype=np.float64)
            summary[f"mean_{key}"] = float(np.mean(array))
            summary[f"p05_{key}"] = float(np.quantile(array, 0.05, method="linear"))
            summary[f"p10_{key}"] = float(np.quantile(array, 0.10, method="linear"))
        result.append(summary)
    return result


def quantile_bounds(values: np.ndarray) -> tuple[float, float]:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    return float(ordered[199]), float(ordered[3799])


def bootstrap_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    names = sorted({str(row["name"]) for row in rows})
    operators = tuple(SOURCE.V3W.OPERATORS)
    if len(names) != 512 or len(rows) != 512 * len(operators) * len(CELL_SPECS):
        raise RuntimeError("A1R formal bootstrap requires a complete 512x2x5 family")
    keyed = {(str(row["name"]), str(row["operator"]), str(row["cell"])): row for row in rows}
    generator = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    cell_draws = {cell: np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64) for cell in CELL_SPECS}
    repair_draws = {cell: np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64) for cell in CELL_SPECS}
    retention = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    shuffle_gap = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    arrays: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for cell in CELL_SPECS:
        for operator in operators:
            arrays[(cell, operator)] = {
                key: np.asarray([float(keyed[(name, operator, cell)][key]) for name in names], dtype=np.float64)
                for key in ("gain_vs_shrink_db", "oracle_gain_db", "direction_selected")
            }
    for draw in range(BOOTSTRAP_REPLICATES):
        indices = generator.integers(0, len(names), size=len(names), endpoint=False)
        for cell in CELL_SPECS:
            cell_draws[cell][draw] = min(float(np.mean(arrays[(cell, operator)]["gain_vs_shrink_db"][indices])) for operator in operators)
            repair_draws[cell][draw] = min(float(np.mean(arrays[(cell, operator)]["direction_selected"][indices])) for operator in operators)
        retention[draw] = min(
            float(np.mean(arrays[(PRIMARY_CELL, operator)]["gain_vs_shrink_db"][indices]))
            / max(float(np.mean(arrays[(PRIMARY_CELL, operator)]["oracle_gain_db"][indices])), 1e-30)
            for operator in operators
        )
        shuffle_gap[draw] = min(
            float(np.mean(arrays[(PRIMARY_CELL, operator)]["gain_vs_shrink_db"][indices] - arrays[(SHUFFLED_CELL, operator)]["gain_vs_shrink_db"][indices]))
            for operator in operators
        )
    cell_results: dict[str, Any] = {}
    for cell in CELL_SPECS:
        gain_lcb, gain_ucb = quantile_bounds(cell_draws[cell])
        repair_lcb, repair_ucb = quantile_bounds(repair_draws[cell])
        cell_results[cell] = {
            "worst_operator_gain_vs_shrink_db": min(float(np.mean(arrays[(cell, operator)]["gain_vs_shrink_db"])) for operator in operators),
            "worst_operator_gain_vs_shrink_db_lcb95": gain_lcb,
            "worst_operator_gain_vs_shrink_db_ucb95": gain_ucb,
            "worst_operator_repairable_fraction": min(float(np.mean(arrays[(cell, operator)]["direction_selected"])) for operator in operators),
            "worst_operator_repairable_fraction_lcb95": repair_lcb,
            "worst_operator_repairable_fraction_ucb95": repair_ucb,
        }
    retention_lcb, retention_ucb = quantile_bounds(retention)
    shuffle_lcb, shuffle_ucb = quantile_bounds(shuffle_gap)
    return {
        "schema_version": 1,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "operator_pairing_retained": True,
        "cell_pairing_retained": True,
        "worst_operator_within_each_draw": True,
        "cell_results": cell_results,
        "primary_oracle_retention": min(
            float(np.mean(arrays[(PRIMARY_CELL, operator)]["gain_vs_shrink_db"]))
            / max(float(np.mean(arrays[(PRIMARY_CELL, operator)]["oracle_gain_db"])), 1e-30)
            for operator in operators
        ),
        "primary_oracle_retention_lcb95": retention_lcb,
        "primary_oracle_retention_ucb95": retention_ucb,
        "primary_true_minus_shuffle_db": min(
            float(np.mean(arrays[(PRIMARY_CELL, operator)]["gain_vs_shrink_db"] - arrays[(SHUFFLED_CELL, operator)]["gain_vs_shrink_db"]))
            for operator in operators
        ),
        "primary_true_minus_shuffle_db_lcb95": shuffle_lcb,
        "primary_true_minus_shuffle_db_ucb95": shuffle_ucb,
    }


def run_a1r(args: Any, v3s: Any, legacy: Any, frozen: Any, names: list[str], folds: dict[str, int], device: torch.device, output_dir: str) -> dict[str, Any]:
    assert PARENT is not None and SOURCE is not None and AUDIT is not None
    audit = AUDIT
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    PARENT.a0p.validate_frozen_args(args)
    review = validate_parent_review(Path(audit.a1f_r3_review), audit.expected_a1f_r3_review_sha256)
    payload, trace, state_path = PARENT.load_final_state(Path(audit.a0r_trace_dir))
    all_names, historical_folds = v3s.load_names_and_folds(args, legacy)
    if len(all_names) < audit.fresh_start_index + audit.fresh_count:
        raise RuntimeError("frozen name list is too short for A1R fresh range")
    parent_names = list(all_names[:audit.fresh_start_index])
    fresh_names = list(all_names[audit.fresh_start_index:audit.fresh_start_index + audit.fresh_count])
    if len(fresh_names) != audit.fresh_count or set(parent_names) & set(fresh_names):
        raise RuntimeError("A1R fresh-name isolation failed")
    fold_values, _ = legacy.v3l_a1.v3j_b.fold_assignments(fresh_names, audit.outer_folds)
    outer_folds = {name: int(fold) for name, fold in zip(fresh_names, fold_values.tolist())}
    fold_counts = {str(fold): sum(value == fold for value in outer_folds.values()) for fold in range(audit.outer_folds)}
    if set(fold_counts.values()) != {audit.fresh_count // audit.outer_folds}:
        raise RuntimeError(f"A1R outer folds are not balanced: {fold_counts}")
    _, current_model, optimizer, parameters = PARENT.a0p.load_cell(
        payload, args, v3s, legacy, frozen, list(all_names[:128]), historical_folds, device
    )
    del optimizer, parameters
    current_model.eval()
    stage_names = fresh_names[:32] if audit.a1r_stage == "smoke" else fresh_names
    cache: list[dict[str, Any]] = []
    sample_shapes: set[tuple[int, int]] = set()
    for index, name in enumerate(stage_names):
        cached, sample = target_and_features(
            args=args,
            v3s=v3s,
            legacy=legacy,
            frozen=frozen,
            name=name,
            historical_fold=historical_folds[name],
            outer_fold=outer_folds[name],
            current_model=current_model,
            device=device,
        )
        sample_shapes.add(tuple(sample["base"].shape[-2:]))
        cache.extend(cached[operator] for operator in SOURCE.V3W.OPERATORS)
        emit_progress(
            Path(audit.status_file),
            {"V4A_A1R_CACHE_PROGRESS": {"stage": audit.a1r_stage, "completed_images": index + 1, "total_images": len(stage_names)}},
        )
    feature_shapes = {
        (tuple(item["output_features"].shape), tuple(item["context_base"].shape), tuple(item["target_low"].shape))
        for item in cache
    }
    channel_signatures: set[tuple[int, int, int, int, int]] = set()
    shape_name_sets: dict[tuple[int, int], set[str]] = defaultdict(set)
    for item in cache:
        tensor_shapes = {
            tuple(value.shape[-2:])
            for value in (
                item["output_features"], item["context_base"], item["support_low"],
                item["current_low"], item["target_low"],
            )
        }
        if len(tensor_shapes) != 1:
            raise RuntimeError(f"A1R cached tensors disagree spatially for {item['name']}/{item['operator']}")
        if any(int(value.shape[0]) != 1 for value in (
            item["output_features"], item["context_base"], item["support_low"],
            item["current_low"], item["target_low"],
        )):
            raise RuntimeError("A1R cache requires one image per cached item")
        shape = spatial_shape(item)
        shape_name_sets[shape].add(str(item["name"]))
        channel_signatures.add((
            int(item["output_features"].shape[1]), int(item["context_base"].shape[1]),
            int(item["support_low"].shape[1]), int(item["current_low"].shape[1]),
            int(item["target_low"].shape[1]),
        ))
    if len(channel_signatures) != 1:
        raise RuntimeError(f"A1R cached channel signatures differ: {len(channel_signatures)}")
    if any(len(names_in_shape) < 2 for names_in_shape in shape_name_sets.values()):
        raise RuntimeError("A1R shape-blocked shuffle lacks two names in a spatial block")
    spatial_shape_name_counts = {
        f"{shape[0]}x{shape[1]}": len(names_in_shape)
        for shape, names_in_shape in sorted(shape_name_sets.items())
    }
    output = Path(output_dir)
    source_path = output / f"{args.run_tag}_source_manifest.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    config = {
        "cells": CELL_SPECS,
        "primary_cell": PRIMARY_CELL,
        "shuffled_cell": SHUFFLED_CELL,
        "fresh_start_index": audit.fresh_start_index,
        "fresh_count": audit.fresh_count,
        "outer_folds": audit.outer_folds,
        "epochs": audit.epochs,
        "batch_size": audit.batch_size,
        "probe_width": audit.probe_width,
        "learning_rate": audit.learning_rate,
        "weight_decay": audit.weight_decay,
        "grad_clip_norm": audit.grad_clip_norm,
        "seed": audit.seed,
        "batch_rule": "exact_spatial_shape_blocks_then_sorted_name_operator_slices_no_resize_crop_or_padding",
        "shuffle_rule": "within_operator_and_exact_spatial_shape_sorted_training_names_cyclic_plus_one",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "thresholds": {
            "gain_db": GAIN_THRESHOLD_DB,
            "retention": RETENTION_THRESHOLD,
            "shuffle_gap_db": SHUFFLE_GAP_THRESHOLD_DB,
            "repairable_fraction": REPAIRABLE_THRESHOLD,
        },
    }
    source.update({
        "route_id": ROUTE_ID,
        "route_commit": audit.expected_route_commit,
        "route_card_sha256": audit.expected_route_card_sha256,
        "stage": audit.a1r_stage,
        "parent_route_id": PARENT_ROUTE_ID,
        "parent_review_decision": review["decision"],
        "a1f_r3_review_sha256": sha256_file(Path(audit.a1f_r3_review)),
        "a1f_root_commit": audit.expected_a1f_root_commit,
        "a0r_trace_manifest_sha256": sha256_file(Path(audit.a0r_trace_dir) / "trace_manifest.json"),
        "final_state_sha256": sha256_file(state_path),
        "fresh_names_sha256": canonical_hash(fresh_names),
        "fresh_names": fresh_names,
        "parent_names_sha256": canonical_hash(parent_names),
        "outer_fold_sha256": canonical_hash(outer_folds),
        "outer_fold_assignments": outer_folds,
        "outer_fold_counts": fold_counts,
        "stage_name_count": len(stage_names),
        "sample_shapes": [list(value) for value in sorted(sample_shapes)],
        "feature_shapes": [[list(shape) for shape in value] for value in sorted(feature_shapes)],
        "spatial_shape_name_counts": spatial_shape_name_counts,
        "probe_config": config,
        "probe_config_sha256": canonical_hash(config),
        "locked_test_touched": False,
        "canary_touched": False,
    })
    manifest_path = output / "v4a_a1r_source_manifest.json"
    write_json(source_path, source)
    write_json(manifest_path, source)
    zero_max = 0.0
    gradient_min = math.inf
    shuffle_self_pairs = 0
    shuffle_mapping_count = 0
    bound_excess_max = 0.0
    support_excess_max = 0.0
    histories: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    if audit.a1r_stage == "smoke":
        for cell, spec in CELL_SPECS.items():
            representation = str(spec["representation"])
            mean, std = feature_stats(cache, representation)
            input_channels = int(item_features(cache[0], representation).shape[1])
            torch.manual_seed(audit.seed + list(CELL_SPECS).index(PRIMARY_CELL if cell == SHUFFLED_CELL else cell))
            probe = DeltaEndpointProbe(input_channels, str(spec["readout"]), audit.probe_width, tuple(cache[0]["delta_bound"])).to(device)
            with torch.no_grad():
                item = cache[0]
                correction = probe(
                    (item_features(item, representation).to(device) - mean.to(device)) / std.to(device),
                    item["support_low"].to(device),
                )
                zero_max = max(zero_max, float(correction.abs().max().item()))
            _, _, _, history, gradient = train_probe(
                items=cache, cell=cell, fold=0, audit=audit, device=device, smoke=True
            )
            histories.extend(history)
            gradient_min = min(gradient_min, gradient)
            if bool(spec["shuffled"]):
                mapping = shuffled_target_mapping(cache)
                shuffle_self_pairs += sum(name == target_name for (name, _), (target_name, _) in mapping.items())
                shuffle_mapping_count = len(mapping)
        structural_valid = (
            len(cache) == 32 * len(SOURCE.V3W.OPERATORS)
            and len(feature_shapes) == len(shape_name_sets)
            and all(len(names_in_shape) >= 2 for names_in_shape in shape_name_sets.values())
            and zero_max == 0.0
            and math.isfinite(gradient_min)
            and gradient_min > 0.0
            and shuffle_self_pairs == 0
            and shuffle_mapping_count == len(cache)
            and not set(parent_names) & set(stage_names)
        )
        state = "COMPLETED_GATE_PASS" if structural_valid else "COMPLETED_GATE_FAIL"
        decision = "V4A_A1R_S0_INTEGRITY_PASS_AUTHORIZE_FORMAL_ONLY" if structural_valid else "V4A_A1R_S0_INTEGRITY_FAIL_STOP"
        authorizes = "A1R_FORMAL_ONLY" if structural_valid else "NONE"
        bootstrap: dict[str, Any] = {"schema_version": 1, "status": "NOT_RUN_SMOKE"}
        summaries = [{"schema_version": 1, "status": "NOT_RUN_SMOKE"}]
        rows: list[dict[str, Any]] = []
        formal_pass = False
    else:
        models: dict[tuple[int, str], tuple[DeltaEndpointProbe, torch.Tensor, torch.Tensor]] = {}
        model_root = output / "models"
        for fold in range(audit.outer_folds):
            train_items = [item for item in cache if int(item["outer_fold"]) != fold]
            heldout_items = [item for item in cache if int(item["outer_fold"]) == fold]
            if len(train_items) != 384 * len(SOURCE.V3W.OPERATORS) or len(heldout_items) != 128 * len(SOURCE.V3W.OPERATORS):
                raise RuntimeError("A1R fold item counts are invalid")
            for cell in CELL_SPECS:
                model, mean, std, history, gradient = train_probe(
                    items=train_items, cell=cell, fold=fold, audit=audit, device=device, smoke=False
                )
                histories.extend(history)
                gradient_min = min(gradient_min, gradient)
                models[(fold, cell)] = (model, mean, std)
                state_path_out = model_root / cell / f"fold{fold}.pt"
                state_path_out.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "schema_version": 1,
                    "route_id": ROUTE_ID,
                    "cell": cell,
                    "fold": fold,
                    "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                    "feature_mean": mean,
                    "feature_std": std,
                    "probe_config_sha256": canonical_hash(config),
                }, state_path_out)
                state_rows.append({
                    "schema_version": 1,
                    "cell": cell,
                    "fold": fold,
                    "relative_path": str(state_path_out.relative_to(output)),
                    "sha256": sha256_file(state_path_out),
                    "train_item_count": len(train_items),
                    "heldout_item_count": len(heldout_items),
                    "train_names_sha256": canonical_hash(sorted({str(item["name"]) for item in train_items})),
                    "heldout_names_sha256": canonical_hash(sorted({str(item["name"]) for item in heldout_items})),
                    "shuffle_mapping_sha256": shuffled_mapping_identity(train_items) if bool(CELL_SPECS[cell]["shuffled"]) else None,
                    "normalizer_channels": int(mean.shape[1]),
                })
        rows = []
        cached_by_key = {(item["name"], item["operator"]): item for item in cache}
        for name_index, name in enumerate(fresh_names):
            fold = outer_folds[name]
            sample_cached, sample = target_and_features(
                args=args,
                v3s=v3s,
                legacy=legacy,
                frozen=frozen,
                name=name,
                historical_fold=historical_folds[name],
                outer_fold=fold,
                current_model=current_model,
                device=device,
            )
            for operator in SOURCE.V3W.OPERATORS:
                step = sample["steps"][operator]
                current_delta = sample[f"current:{operator}"]
                target_delta = sample[f"target:{operator}"]
                old_low, old_high, _, current_high = v3s.candidate_predictions(sample["base"], step, current_delta)
                old_low_mse = PARENT.a0p.scalar(v3s.per_image_mse(old_low, sample["label"]))
                old_high_mse = PARENT.a0p.scalar(v3s.per_image_mse(old_high, sample["label"]))
                current_high_mse = PARENT.a0p.scalar(v3s.per_image_mse(current_high, sample["label"]))
                shrink, _ = canonical_shrink(sample, step, current_delta, old_low_mse, old_high_mse)
                oracle = selected_union(sample, step, current_delta, target_delta, shrink, old_low_mse, old_high_mse, "gt_direction")
                shrink_psnr = PARENT.metric_psnr(float(shrink["high_mse"]))
                oracle_psnr = PARENT.metric_psnr(float(oracle["high_mse"]))
                current_psnr = PARENT.metric_psnr(current_high_mse)
                old_high_psnr = PARENT.metric_psnr(old_high_mse)
                item = cached_by_key[(name, operator)]
                for cell, spec in CELL_SPECS.items():
                    model, mean, std = models[(fold, cell)]
                    correction_low, target_mse, target_cosine = low_alignment(
                        model, mean, std, item, str(spec["representation"]), device
                    )
                    correction_full = sample["support"] * F.interpolate(
                        correction_low, size=current_delta.shape[-2:], mode="bilinear", align_corners=False
                    )
                    delta_bound = current_delta.new_tensor(args.delta_bound).view(1, 3, 1, 1)
                    predicted_endpoint = PARENT.clamp_channelwise(current_delta + correction_full, delta_bound)
                    bound_excess_max = max(
                        bound_excess_max,
                        float(torch.clamp(predicted_endpoint.abs() - delta_bound, min=0.0).max().item()),
                    )
                    inactive = sample["support"] <= 0.0
                    if bool(inactive.any()):
                        support_excess_max = max(
                            support_excess_max,
                            float(predicted_endpoint.masked_select(inactive.expand_as(predicted_endpoint)).abs().max().item()),
                        )
                    selected = selected_union(
                        sample, step, current_delta, predicted_endpoint, shrink, old_low_mse, old_high_mse, cell
                    )
                    selected_psnr = PARENT.metric_psnr(float(selected["high_mse"]))
                    tolerance = PARENT.numerical_tolerance(float(shrink["high_mse"]), float(selected["high_mse"]))
                    direction_selected = bool(
                        selected["family"] == cell
                        and float(selected["high_mse"]) < float(shrink["high_mse"]) - tolerance
                    )
                    row = {
                        "schema_version": 1,
                        "name": name,
                        "outer_fold": fold,
                        "operator": operator,
                        "cell": cell,
                        "old_low_mse": old_low_mse,
                        "old_high_mse": old_high_mse,
                        "current_high_mse": current_high_mse,
                        "shrink_high_mse": float(shrink["high_mse"]),
                        "oracle_high_mse": float(oracle["high_mse"]),
                        "selected_low_mse": float(selected["low_mse"]),
                        "selected_high_mse": float(selected["high_mse"]),
                        "selected_family": str(selected["family"]),
                        "selected_grid_value": float(selected["grid_value"]),
                        "direction_selected": direction_selected,
                        "gain_vs_shrink_db": selected_psnr - shrink_psnr,
                        "gain_vs_current_db": selected_psnr - current_psnr,
                        "gain_vs_old25_db": selected_psnr - old_high_psnr,
                        "oracle_gain_db": oracle_psnr - shrink_psnr,
                        "target_endpoint_mse": target_mse,
                        "target_cosine": target_cosine,
                        "severe_vs_old25": selected_psnr - old_high_psnr <= -0.2,
                        "hard_vs_old25": selected_psnr - old_high_psnr <= -0.5,
                    }
                    if not all(math.isfinite(float(value)) for key, value in row.items() if key not in {"name", "operator", "cell", "selected_family"}):
                        raise FloatingPointError("non-finite A1R OOF row")
                    rows.append(row)
            if (name_index + 1) % 8 == 0 or name_index + 1 == len(fresh_names):
                emit_progress(
                    Path(audit.status_file),
                    {"V4A_A1R_EVAL_PROGRESS": {"completed_images": name_index + 1, "total_images": len(fresh_names)}},
                )
        summaries = summarize_rows(rows)
        structural_valid = (
            len(rows) == audit.fresh_count * len(SOURCE.V3W.OPERATORS) * len(CELL_SPECS)
            and len(state_rows) == audit.outer_folds * len(CELL_SPECS)
            and math.isfinite(gradient_min)
            and gradient_min > 0.0
            and bound_excess_max <= 1e-7
            and support_excess_max == 0.0
            and all(not row["severe_vs_old25"] and not row["hard_vs_old25"] for row in rows)
            and all(
                float(row["selected_low_mse"]) <= float(row["old_low_mse"]) + PARENT.numerical_tolerance(float(row["selected_low_mse"]), float(row["old_low_mse"]))
                and float(row["selected_high_mse"]) <= float(row["old_high_mse"]) + PARENT.numerical_tolerance(float(row["selected_high_mse"]), float(row["old_high_mse"]))
                for row in rows
            )
        )
        bootstrap = bootstrap_summary(rows) if structural_valid else {"schema_version": 1, "status": "NOT_RUN_STRUCTURAL_FAIL"}
        primary = bootstrap.get("cell_results", {}).get(PRIMARY_CELL, {})
        formal_pass = bool(
            structural_valid
            and float(primary["worst_operator_gain_vs_shrink_db_lcb95"]) >= GAIN_THRESHOLD_DB
            and float(bootstrap["primary_oracle_retention_lcb95"]) >= RETENTION_THRESHOLD
            and float(bootstrap["primary_true_minus_shuffle_db_lcb95"]) >= SHUFFLE_GAP_THRESHOLD_DB
            and float(primary["worst_operator_repairable_fraction_lcb95"]) >= REPAIRABLE_THRESHOLD
        )
        state = "COMPLETED_GATE_PASS" if formal_pass else "COMPLETED_GATE_FAIL"
        decision = (
            "V4A_A1R_CONTEXT_SPATIAL_REPRESENTATION_SUFFICIENT_R3_HANDOFF"
            if formal_pass
            else "V4A_A1R_PRIMARY_REPRESENTATION_SUFFICIENCY_FAIL_R3_HANDOFF"
        )
        authorizes = "R3_REVIEW_ONLY"
    history_path = output / "v4a_a1r_fold_history.csv"
    write_rows(history_path, histories)
    state_manifest_path = output / "v4a_a1r_probe_state_manifest.json"
    write_json(state_manifest_path, {
        "schema_version": 1,
        "route_id": ROUTE_ID,
        "stage": audit.a1r_stage,
        "probe_config_sha256": canonical_hash(config),
        "state_count": len(state_rows),
        "states": state_rows,
        "optimizer_state_retained": False,
        "resume_authorized": False,
    })
    raw_path = output / "v4a_a1r_oof_rows_cloud_only.csv"
    if rows:
        write_rows(raw_path, rows)
    summary_path = output / "v4a_a1r_cell_operator_summary.csv"
    write_rows(summary_path, summaries)
    bootstrap_path = output / "v4a_a1r_bootstrap_summary.json"
    write_json(bootstrap_path, bootstrap)
    closeout = {
        "route_id": ROUTE_ID,
        "run_id": args.run_tag,
        "route_commit": audit.expected_route_commit,
        "stage": f"v4a-A1R-{audit.a1r_stage}",
        "state": state,
        "decision": decision,
        "authorizes": authorizes,
        "evidence_role": "engineering_debug" if audit.a1r_stage == "smoke" else "development_screening",
        "gate_type": "structural_integrity" if audit.a1r_stage == "smoke" else "scientific_utility",
        "metric_contract": "2026-07-14 v4a-A1R card: fresh512 OOF context-spatial direction information",
        "contract_id": audit.expected_route_card_sha256,
        "structural_valid": structural_valid,
        "formal_pass": formal_pass,
        "stage_name_count": len(stage_names),
        "row_count": len(rows),
        "cell_count": len(CELL_SPECS),
        "state_count": len(state_rows),
        "zero_correction_max_abs": zero_max,
        "minimum_first_gradient_norm": gradient_min,
        "bound_excess_max": bound_excess_max,
        "support_excess_max": support_excess_max,
        "source_manifest": str(manifest_path),
        "state_manifest": str(state_manifest_path),
        "fold_history": str(history_path),
        "cell_operator_summary": str(summary_path),
        "bootstrap_summary": str(bootstrap_path),
        "raw_rows_cloud_only": str(raw_path) if rows else None,
        "parent_review_sha256": sha256_file(Path(audit.a1f_r3_review)),
        "final_state_sha256": sha256_file(state_path),
        "wall_seconds": time.perf_counter() - started,
        "peak_gpu_memory_mib": float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0) if device.type == "cuda" else 0.0,
        "training_occurred": True,
        "probe_training_only": True,
        "candidate_selected": False,
        "canary_touched": False,
        "locked_test_touched": False,
    }
    closeout_path = output / "v4a_a1r_closeout.json"
    write_json(closeout_path, closeout)
    print(json.dumps(closeout, sort_keys=True), flush=True)
    return closeout


def audit(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--a1f-root", required=True)
    parser.add_argument("--expected-a1f-root-commit", required=True)
    parser.add_argument("--v3z-root", required=True)
    parser.add_argument("--a1f-r3-review", required=True)
    parser.add_argument("--expected-a1f-r3-review-sha256", required=True)
    parser.add_argument("--a0r-trace-dir", required=True)
    parser.add_argument("--a1r-stage", required=True, choices=("smoke", "formal"))
    parser.add_argument("--expected-route-commit", required=True)
    parser.add_argument("--expected-route-card-sha256", required=True)
    parser.add_argument("--status-file", required=True)
    parser.add_argument("--fresh-start-index", type=int, default=256)
    parser.add_argument("--fresh-count", type=int, default=512)
    parser.add_argument("--outer-folds", type=int, default=4)
    parser.add_argument("--probe-epochs", dest="epochs", type=int, default=8)
    parser.add_argument("--probe-batch-size", dest="batch_size", type=int, default=8)
    parser.add_argument("--probe-width", type=int, default=24)
    parser.add_argument("--probe-learning-rate", dest="learning_rate", type=float, default=5e-4)
    parser.add_argument("--probe-weight-decay", dest="weight_decay", type=float, default=1e-5)
    parser.add_argument("--probe-grad-clip-norm", dest="grad_clip_norm", type=float, default=0.1)
    parser.add_argument("--probe-seed", dest="seed", type=int, default=3407)
    args, source_args = parser.parse_known_args(argv)
    if not source_args:
        raise ValueError("frozen v3z arguments are required after A1R arguments")
    a1f_root = Path(args.a1f_root).resolve()
    if (a1f_root / ".git").exists():
        import subprocess
        commit = subprocess.run(["git", "-C", str(a1f_root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(a1f_root), "status", "--porcelain"], check=True, capture_output=True, text=True).stdout.strip()
        if commit != args.expected_a1f_root_commit or dirty:
            raise RuntimeError("A1F parent checkout identity mismatch")
    global PARENT, SOURCE, AUDIT
    AUDIT = args
    PARENT = load_module(
        a1f_root / "experience_docx" / "tools" / "chd_rm_v4a_a1f_action_feasibility.py",
        "v4a_a1r_parent_a1f",
    )
    SOURCE = PARENT.a0p.load_source(Path(args.v3z_root).resolve())
    PARENT.SOURCE = SOURCE
    PARENT.a0p.SOURCE = SOURCE
    SOURCE.run_projected = run_a1r
    original = sys.argv[:]
    try:
        sys.argv = [str(SOURCE.__file__), *source_args]
        SOURCE.main()
    finally:
        sys.argv = original


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] != "audit":
        raise SystemExit("usage: chd_rm_v4a_a1r_representation_sufficiency.py audit --a1f-root ... --a1r-stage smoke|formal ... projected ...")
    audit(sys.argv[2:])


if __name__ == "__main__":
    main()
