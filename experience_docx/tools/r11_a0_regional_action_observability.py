#!/usr/bin/env python3
"""Frozen R11 regional action observability OOF mechanism screen."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import resource
import time
from pathlib import Path
from typing import Any

from route_program_api import (
    asset_path,
    atomic_json,
    load_context,
    prepare_phase_output,
    write_contract_result,
    write_run_result,
    write_workload_progress,
)


ROUTE_ID = "haze4k_v5_r11_regional_action_observability_20260719"
OPERATION_ID = "R11_A0_REGIONAL_ACTION_OBSERVABILITY_SCREEN"
R10_ROUTE_ID = "haze4k_v5_r10_fixed_region_action_feasibility_20260719"
R10_OPERATION_ID = "R10_A0_FIXED_REGION_ACTION_FEASIBILITY_AUDIT"
R10_RUN_ID = "r10-a0-fixed-region-feasibility-r1"
R10_ROUTE_COMMIT = "c455577905efa8bb6f5c0daa84c3ec43c2ee6ff5"
RUNNER_SHA256 = "336c7e1beccb793229beb533ba12367261e702866497c388ee2a4fa88d12718b"
CACHE_MANIFEST_IDENTITY = "b54603b51c810436c14bc9e993aef9f1294902efbc51fadebdd2c84d7c827a1d"
ACTIONS = ("reference_noop", "state_positive_full", "state_negative_full")
ACTIVE_ACTIONS = (1, 2)
OPERATORS = ("D_ref", "D_rep")
FOLDS = (0, 1)
CELLS = (
    "L1_LOCAL_CANDIDATE_CONTEXT",
    "P0_POOLED_CANDIDATE_RESPONSE",
    "S2_WITHIN_IMAGE_RESPONSE_SHUFFLE",
    "G0_LOCAL_STATE_ONLY",
)
PRIMARY_CELL = CELLS[0]
R5_PRIMARY_CELL = "S1_TRUE_SPATIAL_RESPONSE"
GRID = 8
TILES = GRID * GRID
FEATURE_DIM = 303
SEEDS = (3407, 3411)
EPOCHS = 24
BATCH_SIZE = 256
BOOTSTRAP_DRAWS = 4000
BOOTSTRAP_SEED = 3407
LOCAL_GAIN = 0.005
ABSOLUTE_GAIN = 0.020
INCREMENT_GAIN = 0.005
RETENTION_MIN = 0.25
TAIL_MARGIN = -0.005
SEED_RANGE_MAX = 0.020
SEVERE_GAIN = -0.2
HARD_GAIN = -0.5
EXPECTED_RAW_ROWS = 1536
EXPECTED_NAMES = 384
EXPECTED_UNITS = 768
EXPECTED_ROWS = EXPECTED_NAMES * TILES * len(ACTIVE_ACTIONS)
EXPECTED_R10_GAIN = 0.2389403580740129
EXPECTED_R10_MIXED = 0.90625
EXPECTED_R10_BIDIRECTIONAL = 0.765625
CANDIDATE_HEADER = {
    "action", "cell", "fold", "mean_score", "name", "operator",
    "q05_score", "severe_label", "severe_score", "target_gain_db",
}
ENVIRONMENT = {
    "CONVIR_ROUTE_BOOTSTRAP_DRAWS": str(BOOTSTRAP_DRAWS),
    "CONVIR_ROUTE_BOOTSTRAP_SEED": str(BOOTSTRAP_SEED),
    "CONVIR_ROUTE_GRID": str(GRID),
    "CONVIR_ROUTE_SEEDS": ",".join(str(value) for value in SEEDS),
    "CONVIR_ROUTE_EPOCHS": str(EPOCHS),
    "CONVIR_ROUTE_BATCH_SIZE": str(BATCH_SIZE),
    "CONVIR_ROUTE_LOCAL_GAIN_DB": str(LOCAL_GAIN),
    "CONVIR_ROUTE_SEVERE_GAIN_DB": str(SEVERE_GAIN),
    "CONVIR_ROUTE_HARD_GAIN_DB": str(HARD_GAIN),
}


class ObservabilityInconclusive(RuntimeError):
    """Typed scientific-input or numerical stop with a complete closeout."""


def verify_environment() -> None:
    mismatches = {key: os.environ.get(key) for key, value in ENVIRONMENT.items()
                  if os.environ.get(key) != value}
    if mismatches:
        raise ObservabilityInconclusive(f"frozen environment mismatch: {sorted(mismatches)}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ObservabilityInconclusive(f"invalid JSON: {label}") from exc
    if not isinstance(value, dict):
        raise ObservabilityInconclusive(f"JSON is not an object: {label}")
    return value


def read_csv(path: Path, header: set[str]) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None or set(reader.fieldnames) != header:
                raise ObservabilityInconclusive(f"CSV header mismatch: {path.name}")
            rows = list(reader)
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise ObservabilityInconclusive(f"invalid CSV: {path.name}") from exc
    if not rows or any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ObservabilityInconclusive(f"CSV row contract failed: {path.name}")
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    fields = sorted({key for row in rows for key in row})
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def psnr_gain(reference_sse: float, candidate_sse: float) -> float:
    reference = max(float(reference_sse), 1.0e-30)
    candidate = max(float(candidate_sse), 1.0e-30)
    return 10.0 * math.log10(reference / candidate)


def cvar(values: Any, fraction: float = 0.05) -> float:
    import numpy as np

    array = np.sort(np.asarray(values, dtype=np.float64))
    return float(array[:max(1, math.ceil(fraction * len(array)))].mean())


def interval(point: float, samples: Any) -> dict[str, float]:
    import numpy as np

    values = np.asarray(samples, dtype=np.float64)
    if not math.isfinite(point) or values.size == 0 or not np.isfinite(values).all():
        raise ObservabilityInconclusive("non-finite bootstrap result")
    return {
        "point": float(point),
        "lcb95": float(np.quantile(values, 0.025)),
        "ucb95": float(np.quantile(values, 0.975)),
    }


def safe_pearson(first: list[float], second: list[float]) -> float | None:
    import numpy as np

    x = np.asarray(first, dtype=np.float64)
    y = np.asarray(second, dtype=np.float64)
    if len(x) < 2 or float(x.std()) <= 0.0 or float(y.std()) <= 0.0:
        return None
    value = float(np.corrcoef(x, y)[0, 1])
    return value if math.isfinite(value) else None


def load_label(data_root: Path, name: str) -> Any:
    import numpy as np
    import torch
    from PIL import Image

    stem, extension = os.path.splitext(name)
    for candidate in (name, f"{stem.split('_')[0]}{extension}", f"{stem.split('_')[0]}.png"):
        path = data_root / "train/gt" / candidate
        if path.is_file():
            with Image.open(path) as image:
                array = np.asarray(image.convert("RGB")).copy()
            return torch.from_numpy(array.transpose(2, 0, 1)).float().div_(255.0).unsqueeze(0)
    raise ObservabilityInconclusive(f"development target is missing: {name}")


def metric_psnr(mse: Any) -> Any:
    import torch

    return 10.0 * torch.log10(1.0 / torch.clamp(mse, min=1.0e-30))


def tile_statistics(reference: Any, active: Any, label: Any) -> tuple[Any, Any, Any, list[int]]:
    import numpy as np

    height, width = int(reference.shape[-2]), int(reference.shape[-1])
    state = np.empty((TILES, 6), dtype=np.float32)
    response = np.empty((len(ACTIVE_ACTIONS), TILES, 9), dtype=np.float32)
    sse = np.empty((len(ACTIONS), TILES), dtype=np.float64)
    pixel_counts: list[int] = []
    renders = [reference, active[0:1], active[1:2]]
    tile = 0
    for row in range(GRID):
        y0, y1 = (row * height) // GRID, ((row + 1) * height) // GRID
        for column in range(GRID):
            x0, x1 = (column * width) // GRID, ((column + 1) * width) // GRID
            if y1 <= y0 or x1 <= x0:
                raise ObservabilityInconclusive("native image is smaller than fixed grid")
            ref_patch = reference[0, :, y0:y1, x0:x1].double()
            state[tile] = np.concatenate((
                ref_patch.mean((1, 2)).cpu().numpy(),
                ref_patch.std((1, 2), unbiased=False).cpu().numpy(),
            )).astype(np.float32)
            for action_index, action_render in enumerate(active):
                delta = action_render[:, y0:y1, x0:x1].double() - ref_patch
                response[action_index, tile] = np.concatenate((
                    delta.mean((1, 2)).cpu().numpy(),
                    delta.std((1, 2), unbiased=False).cpu().numpy(),
                    delta.abs().mean((1, 2)).cpu().numpy(),
                )).astype(np.float32)
            target_patch = label[0, :, y0:y1, x0:x1]
            for action, render in enumerate(renders):
                error = render[0, :, y0:y1, x0:x1] - target_patch
                sse[action, tile] = float(error.double().square().sum())
            pixel_counts.append(3 * (y1 - y0) * (x1 - x0))
            tile += 1
    if tile != TILES or sum(pixel_counts) != 3 * height * width:
        raise ObservabilityInconclusive("fixed tile partition is incomplete")
    if not np.isfinite(state).all() or not np.isfinite(response).all() or not np.isfinite(sse).all():
        raise ObservabilityInconclusive("non-finite tile representation")
    return state, response, sse, pixel_counts


def equal_area_permutation(name: str, action: int, pixel_counts: Any) -> Any:
    import numpy as np

    counts = np.asarray(pixel_counts, dtype=np.int64)
    permutation = np.arange(TILES, dtype=np.int64)
    digest = hashlib.sha256(f"{ROUTE_ID}|{name}|action={action}|shuffle".encode()).digest()
    generator = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    for count in sorted(np.unique(counts)):
        indices = np.flatnonzero(counts == count)
        permutation[indices] = generator.permutation(indices)
    return permutation


def neighborhood(values: Any) -> Any:
    import numpy as np

    channels = int(values.shape[-1])
    grid = values.reshape(GRID, GRID, channels)
    padded = np.pad(grid, ((1, 1), (1, 1), (0, 0)), mode="reflect")
    rows = []
    for row in range(GRID):
        for column in range(GRID):
            rows.append(padded[row:row + 3, column:column + 3].reshape(-1))
    return np.asarray(rows, dtype=np.float32)


def build_cell_features(
    name: str, state_map: Any, response_map: Any, pixel_counts: Any, action: int, cell: str,
) -> Any:
    import numpy as np

    action_response = response_map[action - 1]
    if cell == "S2_WITHIN_IMAGE_RESPONSE_SHUFFLE":
        action_response = action_response[equal_area_permutation(name, action, pixel_counts)]
    if cell == "P0_POOLED_CANDIDATE_RESPONSE":
        action_response = np.broadcast_to(action_response.mean(0), action_response.shape).copy()
    if cell == "G0_LOCAL_STATE_ONLY":
        action_response = np.zeros_like(action_response)
    local = np.concatenate((neighborhood(state_map), neighborhood(action_response)), axis=1)
    global_state = np.broadcast_to(state_map.mean(0), (TILES, state_map.shape[1]))
    if cell == "G0_LOCAL_STATE_ONLY":
        global_response = np.zeros((TILES, response_map.shape[-1]), dtype=np.float32)
    else:
        global_response = np.broadcast_to(response_map[action - 1].mean(0),
                                          (TILES, response_map.shape[-1]))
    coordinates = np.asarray([
        (2.0 * (tile // GRID + 0.5) / GRID - 1.0,
         2.0 * (tile % GRID + 0.5) / GRID - 1.0)
        for tile in range(TILES)
    ], dtype=np.float32)
    sign = np.full((TILES, 1), 1.0 if action == 1 else -1.0, dtype=np.float32)
    features = np.concatenate((local, global_state, global_response, coordinates, sign), axis=1)
    if features.shape != (TILES, FEATURE_DIM) or not np.isfinite(features).all():
        raise ObservabilityInconclusive("feature shape or finite-value contract failed")
    return features.astype(np.float32, copy=False)


def oracle_action_map(gains: Any) -> Any:
    import numpy as np

    action_map = []
    for tile in range(TILES):
        scores = [0.0]
        for action_index in range(len(ACTIVE_ACTIONS)):
            values = gains[:, action_index, tile]
            scores.append(float(values.min()) if bool(np.all(values >= LOCAL_GAIN)) else -math.inf)
        action_map.append(max(range(3), key=lambda action: (scores[action], -action)))
    return np.asarray(action_map, dtype=np.int64)


def safe_global_action(sse: Any) -> int:
    scores = [0.0]
    for action in ACTIVE_ACTIONS:
        gains = [psnr_gain(sse[operator, 0].sum(), sse[operator, action].sum())
                 for operator in range(len(OPERATORS))]
        scores.append(min(gains) if all(value >= 0.0 for value in gains) else -math.inf)
    return max(range(3), key=lambda action: (scores[action], -action))


def assign_oracle_budget(scores: Any, oracle_map: Any, pixel_counts: Any) -> Any:
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    assigned = np.zeros(TILES, dtype=np.int64)
    counts = np.asarray(pixel_counts, dtype=np.int64)
    for pixel_count in sorted(np.unique(counts)):
        indices = np.flatnonzero(counts == pixel_count)
        slots = np.asarray([action for action in range(3)
                            for _ in range(int(np.sum(oracle_map[indices] == action)))],
                           dtype=np.int64)
        if len(slots) != len(indices):
            raise ObservabilityInconclusive("oracle budget cardinality mismatch")
        benefits = np.zeros((len(indices), len(slots)), dtype=np.float64)
        for slot_index, action in enumerate(slots):
            if action != 0:
                benefits[:, slot_index] = scores[indices, action - 1]
        rows, columns = linear_sum_assignment(-benefits)
        assigned[indices[rows]] = slots[columns]
    if any(np.sum(assigned == action) != np.sum(oracle_map == action) for action in range(3)):
        raise ObservabilityInconclusive("exact oracle budget was not preserved")
    return assigned


def replay_gain(sse: Any, action_map: Any) -> dict[str, float]:
    result = {}
    for operator_index, operator in enumerate(OPERATORS):
        reference = float(sse[operator_index, 0].sum())
        selected = sum(float(sse[operator_index, action, tile])
                       for tile, action in enumerate(action_map))
        result[operator] = psnr_gain(reference, selected)
    return result


def train_model(features: Any, mean_target: Any, worst_target: Any,
                seed: int, epochs: int = EPOCHS) -> tuple[Any, dict[str, float]]:
    import numpy as np
    import torch

    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    x = torch.as_tensor(features, dtype=torch.float32)
    mean_y = torch.as_tensor(mean_target, dtype=torch.float32)
    worst_y = torch.as_tensor(worst_target, dtype=torch.float32)
    model = torch.nn.Sequential(
        torch.nn.Linear(FEATURE_DIM, 64), torch.nn.ReLU(), torch.nn.Linear(64, 2),
    )
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight, generator=generator)
            torch.nn.init.zeros_(module.bias)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-4)

    def loss_value(prediction: Any, indices: Any) -> Any:
        mean_loss = torch.nn.functional.huber_loss(prediction[:, 0], mean_y[indices])
        worst_loss = torch.nn.functional.huber_loss(prediction[:, 1], worst_y[indices])
        return mean_loss + worst_loss

    def full_loss() -> float:
        weighted = 0.0
        with torch.no_grad():
            for start in range(0, len(x), 4096):
                indices = torch.arange(start, min(start + 4096, len(x)))
                weighted += float(loss_value(model(x[indices]), indices)) * len(indices)
        return weighted / len(x)

    initial = full_loss()
    for epoch in range(epochs):
        order = torch.randperm(len(x), generator=torch.Generator().manual_seed(seed + epoch))
        for start in range(0, len(x), BATCH_SIZE):
            indices = order[start:start + BATCH_SIZE]
            prediction = model(x[indices])
            loss = loss_value(prediction, indices)
            if not bool(torch.isfinite(loss)):
                raise ObservabilityInconclusive("non-finite diagnostic loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    final = full_loss()
    if not math.isfinite(initial) or not math.isfinite(final):
        raise ObservabilityInconclusive("non-finite training summary")
    return model.eval(), {"initial_loss": initial, "final_loss": final}


def predict_model(model: Any, features: Any) -> Any:
    import numpy as np
    import torch

    output = []
    with torch.no_grad():
        tensor = torch.as_tensor(features, dtype=torch.float32)
        for start in range(0, len(tensor), 4096):
            output.append(model(tensor[start:start + 4096]).cpu().numpy())
    values = np.concatenate(output, axis=0)
    if values.shape != (len(features), 2) or not np.isfinite(values).all():
        raise ObservabilityInconclusive("prediction contract failed")
    return values


def evaluate_rows(rows: list[dict[str, Any]], indices: Any) -> dict[str, float]:
    import numpy as np

    result: dict[str, float] = {}
    cell_means: dict[str, dict[str, float]] = {}
    for cell in CELLS:
        cell_means[cell] = {}
        for operator in OPERATORS:
            values = np.asarray([row[f"{cell}_{operator}"] for row in rows],
                                dtype=np.float64)[indices]
            cell_means[cell][operator] = float(values.mean())
            result[f"{cell}_{operator}"] = float(values.mean())
        result[f"{cell}_gain"] = min(cell_means[cell].values())
    oracle_means = {}
    global_means = {}
    for operator in OPERATORS:
        oracle_values = np.asarray([row[f"oracle_{operator}"] for row in rows],
                                   dtype=np.float64)[indices]
        global_values = np.asarray([row[f"global_{operator}"] for row in rows],
                                   dtype=np.float64)[indices]
        oracle_means[operator] = float(oracle_values.mean())
        global_means[operator] = float(global_values.mean())
    result["oracle_gain"] = min(oracle_means.values())
    result["global_gain"] = min(global_means.values())
    result["retention"] = min(
        cell_means[PRIMARY_CELL][operator] / oracle_means[operator] for operator in OPERATORS
    )
    for control in CELLS[1:]:
        result[f"primary_minus_{control}"] = min(
            cell_means[PRIMARY_CELL][operator] - cell_means[control][operator]
            for operator in OPERATORS
        )
    result["primary_minus_global"] = min(
        cell_means[PRIMARY_CELL][operator] - global_means[operator]
        for operator in OPERATORS
    )
    result["primary_minus_global_cvar5"] = min(
        cvar(np.asarray([row[f"{PRIMARY_CELL}_{operator}"] for row in rows],
                       dtype=np.float64)[indices])
        - cvar(np.asarray([row[f"global_{operator}"] for row in rows],
                         dtype=np.float64)[indices])
        for operator in OPERATORS
    )
    return result


def bootstrap(rows: list[dict[str, Any]]) -> dict[str, Any]:
    import numpy as np

    count = len(rows)
    point = evaluate_rows(rows, np.arange(count))
    samples = {key: [] for key in point}
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    fold_indices = {fold: np.asarray([index for index, row in enumerate(rows)
                                     if row["fold"] == fold], dtype=np.int64)
                    for fold in FOLDS}
    for _draw in range(BOOTSTRAP_DRAWS):
        indices = np.concatenate([generator.choice(value, len(value), replace=True)
                                  for value in fold_indices.values()])
        value = evaluate_rows(rows, indices)
        for key in samples:
            samples[key].append(value[key])
    return {key: interval(point[key], samples[key]) for key in point}


def synthetic_contract(context: Any) -> dict[str, bool]:
    import numpy as np

    started = time.perf_counter()
    generator = np.random.default_rng(3407)
    feature_lists: dict[str, list[Any]] = {cell: [] for cell in CELLS}
    for index in range(EXPECTED_NAMES):
        state_map = generator.normal(size=(TILES, 12)).astype(np.float32)
        response_map = generator.normal(size=(2, TILES, 18)).astype(np.float32)
        pixel_counts = np.full(TILES, 3072, dtype=np.int64)
        for cell in CELLS:
            feature_lists[cell].append(np.concatenate([
                build_cell_features(f"synthetic_{index:04d}", state_map, response_map,
                                    pixel_counts, action, cell)
                for action in ACTIVE_ACTIONS
            ], axis=0))
    feature_arrays = {cell: np.concatenate(values, axis=0)
                      for cell, values in feature_lists.items()}
    primary = feature_arrays[PRIMARY_CELL]
    mean_target = (0.1 * primary[:, 0] - 0.05 * primary[:, 1]).astype(np.float32)
    worst_target = (mean_target - 0.02).astype(np.float32)
    train_indices = np.arange(EXPECTED_ROWS // 2)
    normalizer_mean = primary[train_indices].mean(0, dtype=np.float64)
    normalizer_std = primary[train_indices].std(0, dtype=np.float64)
    normalizer_std = np.where(normalizer_std >= 1.0e-6, normalizer_std, 1.0)
    finite_models = True
    loss_decreased = True
    for _fold in FOLDS:
        for cell in CELLS:
            train_features = ((feature_arrays[cell][train_indices] - normalizer_mean)
                              / normalizer_std).astype(np.float32)
            for seed in SEEDS:
                model, summary = train_model(
                    train_features, mean_target[train_indices], worst_target[train_indices],
                    seed, epochs=1,
                )
                prediction = predict_model(model, train_features[:256])
                finite_models = finite_models and np.isfinite(prediction).all() \
                    and math.isfinite(summary["final_loss"])
                loss_decreased = loss_decreased and summary["final_loss"] < summary["initial_loss"]
    oracle = np.asarray(([0] * 20) + ([1] * 24) + ([2] * 20), dtype=np.int64)
    pixel_counts = np.full(TILES, 3072, dtype=np.int64)
    scores = generator.normal(size=(TILES, 2))
    assigned = assign_oracle_budget(scores, oracle, pixel_counts)
    rows = []
    for index in range(EXPECTED_NAMES):
        row = {"fold": index // 192}
        for operator in OPERATORS:
            row[f"oracle_{operator}"] = 0.24
            row[f"global_{operator}"] = 0.15
            for cell_index, cell in enumerate(CELLS):
                row[f"{cell}_{operator}"] = 0.22 - 0.03 * cell_index
        rows.append(row)
    boot = bootstrap(rows)
    return {
        "formal_feature_shape": all(value.shape == (EXPECTED_ROWS, FEATURE_DIM)
                                      for value in feature_arrays.values()),
        "all_cell_transforms_exact": all(np.isfinite(value).all()
                                            for value in feature_arrays.values()),
        "all_models_finite": bool(finite_models),
        "all_models_loss_decreased": bool(loss_decreased),
        "exact_assignment_budget": all(np.sum(assigned == action) == np.sum(oracle == action)
                                           for action in range(3)),
        "full_bootstrap_finite": all(math.isfinite(value["point"]) for value in boot.values()),
        "formal_one_epoch_wall_bounded": time.perf_counter() - started <= 120.0,
        "formal_peak_memory_under_4096_mib": (
            float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0 <= 4096.0
        ),
    }


def contract(context_path: Path) -> None:
    context = load_context(context_path, "contract")
    prepare_phase_output(context)
    verify_environment()
    checks = {
        "route_identity": context.route_id == ROUTE_ID and context.operation_id == OPERATION_ID,
        "cpu_only": context.device == "cpu",
        "protected_roles_blocked": not any(context.protected_data_permissions.values()),
        "frozen_dimensions": GRID == 8 and FEATURE_DIM == 303 and EXPECTED_ROWS == 49152,
        **synthetic_contract(context),
    }
    write_contract_result(context, checks=checks)


def load_formal_dataset(context: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import numpy as np
    import torch

    cache_manifest = read_json(asset_path(context, "a0_cache_manifest", kind="file"),
                               "cache_manifest")
    if cache_manifest.get("cache_manifest_sha256") != CACHE_MANIFEST_IDENTITY:
        raise ObservabilityInconclusive("cache manifest internal identity mismatch")
    try:
        raw_rows = [json.loads(line) for line in
                    asset_path(context, "a0_raw_manifest", kind="file").read_text(
                        encoding="utf-8").splitlines() if line.strip()]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ObservabilityInconclusive("invalid raw cache manifest") from exc
    if len(raw_rows) != EXPECTED_RAW_ROWS or any(not isinstance(row, dict) for row in raw_rows):
        raise ObservabilityInconclusive("raw cache manifest cardinality mismatch")
    raw_by_key = {}
    for row in raw_rows:
        key = row.get("unit_key")
        if not isinstance(key, str) or key in raw_by_key or not isinstance(row.get("cache_sha256"), str):
            raise ObservabilityInconclusive("raw cache manifest key/hash mismatch")
        raw_by_key[key] = row

    candidate_rows = read_csv(asset_path(context, "r5_candidate_scores", kind="file"),
                              CANDIDATE_HEADER)
    primary = [row for row in candidate_rows if row["cell"] == R5_PRIMARY_CELL]
    if len(candidate_rows) != 6144 or len(primary) != 1536:
        raise ObservabilityInconclusive("R5 candidate rows are incomplete")
    target_map: dict[tuple[str, str, int], float] = {}
    fold_by_name: dict[str, int] = {}
    for row in primary:
        try:
            fold = int(row["fold"])
            action = ACTIONS.index(row["action"])
            target = float(row["target_gain_db"])
        except (ValueError, TypeError) as exc:
            raise ObservabilityInconclusive("R5 candidate row parse failure") from exc
        if fold not in FOLDS or action not in ACTIVE_ACTIONS or row["operator"] not in OPERATORS:
            raise ObservabilityInconclusive("R5 candidate identity is outside R11 scope")
        key = (row["name"], row["operator"], action)
        if key in target_map or not math.isfinite(target):
            raise ObservabilityInconclusive("duplicate or non-finite R5 target")
        target_map[key] = target
        if row["name"] in fold_by_name and fold_by_name[row["name"]] != fold:
            raise ObservabilityInconclusive("one name appears in multiple folds")
        fold_by_name[row["name"]] = fold
    names = sorted(fold_by_name)
    required = {(name, operator, action) for name in names for operator in OPERATORS
                for action in ACTIVE_ACTIONS}
    if len(names) != EXPECTED_NAMES or set(target_map) != required \
            or any(sum(fold_by_name[name] == fold for name in names) != 192 for fold in FOLDS):
        raise ObservabilityInconclusive("R11 population grid is incomplete")

    cache_root = asset_path(context, "a0_candidate_cache", kind="directory")
    data_root = asset_path(context, "haze4k_data", kind="directory")
    groups: list[dict[str, Any]] = []
    loaded = 0
    target_mismatches = 0
    target_max_abs_difference = 0.0
    no_op_mismatches = 0
    partition_max_difference = 0.0
    for name in names:
        label = load_label(data_root, name)
        operator_data = []
        for operator in OPERATORS:
            unit_key = hashlib.sha256(f"{name}\0{operator}".encode()).hexdigest()[:32]
            raw = raw_by_key.get(unit_key)
            if raw is None:
                raise ObservabilityInconclusive("cache unit is absent from raw manifest")
            unit_path = cache_root / f"{unit_key}.pt"
            if sha256_file(unit_path) != raw["cache_sha256"]:
                raise ObservabilityInconclusive(f"cache unit hash mismatch: {unit_key}")
            try:
                payload = torch.load(unit_path, map_location="cpu", weights_only=False)
            except Exception as exc:
                raise ObservabilityInconclusive(f"cache unit load failed: {unit_key}") from exc
            candidate_names = list(payload.get("candidate_names", []))
            if any(action not in candidate_names for action in ACTIONS) \
                    or payload.get("name") != name or payload.get("operator") != operator:
                raise ObservabilityInconclusive("cache payload identity/action mismatch")
            selected = [candidate_names.index(action) for action in ACTIONS]
            base = payload["base"].float()
            step = payload["step"].float()
            candidate_delta = payload["candidates"].float()[selected]
            label_used = label[:, :, :base.shape[-2], :base.shape[-1]]
            reference = torch.clamp(base + 0.25 * step, 0.0, 1.0)
            renders = torch.clamp(base + 0.25 * (step + candidate_delta), 0.0, 1.0)
            no_op_mismatches += not bool(torch.equal(reference, renders[0:1]))
            active = renders[1:]
            observed_targets = metric_psnr((active - label_used).square().mean((1, 2, 3))) \
                - metric_psnr((reference - label_used).square().mean())
            for action in ACTIVE_ACTIONS:
                difference = abs(float(observed_targets[action - 1])
                                 - target_map[(name, operator, action)])
                target_max_abs_difference = max(target_max_abs_difference, difference)
                target_mismatches += difference != 0.0
            state, response, sse, pixel_counts = tile_statistics(reference, active, label_used)
            exact = np.asarray([float((render - label_used).double().square().sum())
                                for render in (reference, active[0:1], active[1:2])])
            partition_max_difference = max(partition_max_difference,
                                           float(np.max(np.abs(sse.sum(1) - exact))))
            operator_data.append({"state": state, "response": response, "sse": sse,
                                  "pixel_counts": pixel_counts,
                                  "shape": f"{base.shape[-2]}x{base.shape[-1]}"})
            loaded += 1
            if loaded % 8 == 0 or loaded == EXPECTED_UNITS:
                write_workload_progress(context, completed_units=2 + loaded,
                                        stage="cache_target_feature_replay")
        if operator_data[0]["pixel_counts"] != operator_data[1]["pixel_counts"]:
            raise ObservabilityInconclusive("operator tile partitions differ")
        state_map = np.concatenate([unit["state"] for unit in operator_data], axis=1)
        response_map = np.concatenate([unit["response"] for unit in operator_data], axis=2)
        sse = np.stack([unit["sse"] for unit in operator_data])
        gains = np.empty((len(OPERATORS), len(ACTIVE_ACTIONS), TILES), dtype=np.float64)
        for operator_index in range(len(OPERATORS)):
            for action_index in range(len(ACTIVE_ACTIONS)):
                for tile in range(TILES):
                    gains[operator_index, action_index, tile] = psnr_gain(
                        sse[operator_index, 0, tile], sse[operator_index, action_index + 1, tile],
                    )
        oracle_map = oracle_action_map(gains)
        global_action = safe_global_action(sse)
        feature_by_cell = {cell: np.concatenate([
            build_cell_features(name, state_map, response_map, operator_data[0]["pixel_counts"],
                                action, cell) for action in ACTIVE_ACTIONS
        ], axis=0) for cell in CELLS}
        mean_target = np.concatenate([gains[:, action - 1].mean(0) for action in ACTIVE_ACTIONS])
        worst_target = np.concatenate([np.min(gains[:, action - 1], axis=0)
                                       for action in ACTIVE_ACTIONS])
        eligible = np.concatenate([np.min(gains[:, action - 1], axis=0) >= LOCAL_GAIN
                                   for action in ACTIVE_ACTIONS]).astype(np.float32)
        groups.append({
            "name": name, "fold": fold_by_name[name], "shape": operator_data[0]["shape"],
            "features": feature_by_cell, "mean_target": mean_target.astype(np.float32),
            "worst_target": worst_target.astype(np.float32), "eligible": eligible,
            "gains": gains, "sse": sse, "pixel_counts": operator_data[0]["pixel_counts"],
            "oracle_map": oracle_map, "global_action": global_action,
        })
    if loaded != EXPECTED_UNITS or len(groups) != EXPECTED_NAMES:
        raise ObservabilityInconclusive("formal feature population is incomplete")
    return groups, {
        "candidate_rows": len(candidate_rows), "primary_candidate_rows": len(primary),
        "raw_manifest_rows": len(raw_rows), "evaluated_names": len(groups),
        "evaluated_units": loaded, "tile_action_rows": len(groups) * TILES * 2,
        "target_mismatches": int(target_mismatches),
        "target_max_abs_difference": target_max_abs_difference,
        "no_op_render_mismatches": int(no_op_mismatches),
        "partition_sse_max_abs_difference": partition_max_difference,
        "cache_manifest_internal_identity": cache_manifest.get("cache_manifest_sha256"),
    }


def write_inconclusive_bundle(context: Any, reason: str, started_wall: float,
                              started_cpu: float) -> None:
    common = {"schema_version": 1, "status": "input_or_observability_inconclusive",
              "reason": reason, "r10_terminal_changed": False}
    for filename in (
        "r11_a0_contract_summary.json", "r11_a0_provenance_and_access.json",
        "r11_a0_input_identity.json", "r11_a0_representation_identity.json",
        "r11_a0_oracle_replay.json", "r11_a0_bootstrap_summary.json",
        "r11_a0_label_stability.json", "r11_a0_operator_consistency.json",
        "r11_a0_gate_summary.json",
    ):
        atomic_json(context.phase_output_path / filename, common)
    placeholder = [{"status": "inconclusive", "reason": reason}]
    for filename in ("r11_a0_cell_summary.csv", "r11_a0_fold_seed_stability.csv",
                     "r11_a0_calibration_summary.csv"):
        write_csv(context.phase_output_path / filename, placeholder)
    atomic_json(context.phase_output_path / "r11_a0_resource_summary.json", {
        **common, "wall_seconds": time.perf_counter() - started_wall,
        "cpu_seconds": time.process_time() - started_cpu, "gpu_used": False,
    })
    write_run_result(context, state="COMPLETED_GATE_INCONCLUSIVE",
                     decision="R11_A0_INPUT_OR_OBSERVABILITY_INCONCLUSIVE_STOP",
                     authorizes="NONE",
                     details={"reason": reason, "r10_terminal_changed": False})


def run(context_path: Path) -> None:
    import numpy as np
    import torch

    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    context = load_context(context_path, "run")
    prepare_phase_output(context)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    torch.use_deterministic_algorithms(True)
    try:
        verify_environment()
        r10_closeout = read_json(asset_path(context, "r10_closeout", kind="file"),
                                 "r10_closeout")
        if (r10_closeout.get("route_id"), r10_closeout.get("operation_id"),
            r10_closeout.get("run_id"), r10_closeout.get("route_commit"),
            r10_closeout.get("runner_sha256"), r10_closeout.get("state"),
            r10_closeout.get("decision"), r10_closeout.get("authorizes")) != (
                R10_ROUTE_ID, R10_OPERATION_ID, R10_RUN_ID, R10_ROUTE_COMMIT, RUNNER_SHA256,
                "COMPLETED_GATE_PASS", "R10_A0_FIXED_REGION_ACTION_FEASIBILITY_PASS",
                "R10_REGION_OBSERVABILITY_CONTRACT_REVIEW_ONLY"):
            raise ObservabilityInconclusive("R10 typed closeout identity mismatch")
        r10_gate = read_json(asset_path(context, "r10_gate_summary", kind="file"),
                             "r10_gate_summary")
        r10_distribution = read_json(
            asset_path(context, "r10_action_distribution", kind="file"),
            "r10_action_distribution",
        )
        if not r10_gate.get("passes") or not all(r10_gate.get("gates", {}).values()) \
                or not all(r10_gate.get("structural_checks", {}).values()):
            raise ObservabilityInconclusive("R10 gate evidence is incomplete")
        write_workload_progress(context, completed_units=2, stage="r10_terminal_verified")
        groups, input_metadata = load_formal_dataset(context)
        if input_metadata["target_mismatches"] != 0 \
                or input_metadata["no_op_render_mismatches"] != 0 \
                or input_metadata["partition_sse_max_abs_difference"] > 1.0e-9:
            raise ObservabilityInconclusive("target/render/partition replay mismatch")

        oracle_rows = []
        oracle_counts = np.zeros(3, dtype=np.int64)
        mixed = 0
        bidirectional = 0
        local_violations = 0
        for group in groups:
            oracle_counts += np.bincount(group["oracle_map"], minlength=3)
            pixel_counts = np.asarray(group["pixel_counts"], dtype=np.float64)
            total = float(pixel_counts.sum())
            areas = [float(pixel_counts[group["oracle_map"] == action].sum() / total)
                     for action in range(3)]
            mixed += areas[0] >= 0.10 and areas[1] + areas[2] >= 0.10
            bidirectional += areas[1] >= 0.05 and areas[2] >= 0.05
            for tile, action in enumerate(group["oracle_map"]):
                if action != 0:
                    local_violations += bool(np.any(group["gains"][:, action - 1, tile] < LOCAL_GAIN))
            oracle_gain = replay_gain(group["sse"], group["oracle_map"])
            global_map = np.full(TILES, group["global_action"], dtype=np.int64)
            global_gain = replay_gain(group["sse"], global_map)
            oracle_rows.append({"name": group["name"], "fold": group["fold"],
                                **{f"oracle_{key}": value for key, value in oracle_gain.items()},
                                **{f"global_{key}": value for key, value in global_gain.items()}})
        oracle_point = evaluate_rows([
            {**row, **{f"{cell}_{operator}": row[f"oracle_{operator}"]
                       for cell in CELLS for operator in OPERATORS}}
            for row in oracle_rows
        ], np.arange(len(oracle_rows)))["oracle_gain"]
        distribution_counts = r10_distribution.get("tile_counts", {})
        expected_counts = np.asarray([distribution_counts.get("noop"),
                                      distribution_counts.get("positive"),
                                      distribution_counts.get("negative")])
        if not np.array_equal(oracle_counts, expected_counts) \
                or abs(oracle_point - EXPECTED_R10_GAIN) > 1.0e-12 \
                or mixed / EXPECTED_NAMES != EXPECTED_R10_MIXED \
                or bidirectional / EXPECTED_NAMES != EXPECTED_R10_BIDIRECTIONAL \
                or local_violations != 0:
            raise ObservabilityInconclusive("R10 oracle/action-distribution replay mismatch")
        r10_fold_metrics = r10_gate.get("fold_metrics", {})
        fold_replay_mismatches = 0
        for fold in FOLDS:
            subset = [row for row in oracle_rows if row["fold"] == fold]
            for operator in OPERATORS:
                observed_oracle = float(np.mean([row[f"oracle_{operator}"] for row in subset]))
                observed_global = float(np.mean([row[f"global_{operator}"] for row in subset]))
                expected_fold = r10_fold_metrics.get(str(fold), {})
                fold_replay_mismatches += (
                    abs(observed_oracle - float(expected_fold.get(f"region_{operator}", math.inf)))
                    > 1.0e-12
                )
                fold_replay_mismatches += (
                    abs(observed_global - float(expected_fold.get(f"global_{operator}", math.inf)))
                    > 1.0e-12
                )
        if fold_replay_mismatches != 0:
            raise ObservabilityInconclusive("R10 fold oracle/global replay mismatch")

        predictions: dict[str, dict[int, dict[int, Any]]] = {cell: {} for cell in CELLS}
        training_rows = []
        oof_overlap_violations = 0
        row_groups = np.concatenate([np.full(TILES * 2, index, dtype=np.int64)
                                     for index in range(len(groups))])
        feature_arrays = {cell: np.concatenate([group["features"][cell]
                                                for group in groups], axis=0)
                          for cell in CELLS}
        for group in groups:
            del group["features"]
        mean_target = np.concatenate([group["mean_target"] for group in groups])
        worst_target = np.concatenate([group["worst_target"] for group in groups])
        eligible = np.concatenate([group["eligible"] for group in groups])
        if len(mean_target) != EXPECTED_ROWS or any(values.shape != (EXPECTED_ROWS, FEATURE_DIM)
                                                    for values in feature_arrays.values()):
            raise ObservabilityInconclusive("formal row/feature matrix is incomplete")
        for test_fold in FOLDS:
            train_groups = np.asarray([index for index, group in enumerate(groups)
                                       if group["fold"] != test_fold], dtype=np.int64)
            test_groups = np.asarray([index for index, group in enumerate(groups)
                                      if group["fold"] == test_fold], dtype=np.int64)
            train_indices = np.flatnonzero(np.isin(row_groups, train_groups))
            test_indices = np.flatnonzero(np.isin(row_groups, test_groups))
            oof_overlap_violations += len(set(train_groups.tolist()) & set(test_groups.tolist()))
            if len(train_indices) != EXPECTED_ROWS // 2 or len(test_indices) != EXPECTED_ROWS // 2:
                raise ObservabilityInconclusive("outer fold row boundary failed")
            mean = feature_arrays[PRIMARY_CELL][train_indices].mean(0, dtype=np.float64)
            std = feature_arrays[PRIMARY_CELL][train_indices].std(0, dtype=np.float64)
            std = np.where(std >= 1.0e-6, std, 1.0)
            utility_mean = float(mean_target[train_indices].mean())
            utility_std = max(float(mean_target[train_indices].std()), 1.0e-6)
            target_mean_z = (mean_target[train_indices] - utility_mean) / utility_std
            target_worst_z = (worst_target[train_indices] - utility_mean) / utility_std
            for cell in CELLS:
                train_features = ((feature_arrays[cell][train_indices] - mean) / std).astype(np.float32)
                test_features = ((feature_arrays[cell][test_indices] - mean) / std).astype(np.float32)
                predictions[cell][test_fold] = {}
                for seed in SEEDS:
                    model, training = train_model(
                        train_features, target_mean_z, target_worst_z, seed,
                    )
                    values = predict_model(model, test_features)
                    values[:, :2] = values[:, :2] * utility_std + utility_mean
                    predictions[cell][test_fold][seed] = values
                    training_rows.append({"cell": cell, "test_fold": test_fold, "seed": seed,
                                          **training})
                    write_workload_progress(
                        context, completed_units=771 + len(training_rows) - 1,
                        stage="oof_probe_models",
                    )

        evaluation_rows = []
        seed_rows = []
        cloud_prediction_rows = []
        for test_fold in FOLDS:
            fold_groups = [index for index, group in enumerate(groups) if group["fold"] == test_fold]
            for local_index, group_index in enumerate(fold_groups):
                group = groups[group_index]
                row = dict(oracle_rows[group_index])
                for cell in CELLS:
                    seed_scores = []
                    for seed in SEEDS:
                        start = local_index * TILES * 2
                        stop = start + TILES * 2
                        prediction = predictions[cell][test_fold][seed][start:stop]
                        score = np.stack((prediction[:TILES, 1], prediction[TILES:, 1]), axis=1)
                        seed_scores.append(score)
                    ensemble = np.mean(seed_scores, axis=0)
                    action_map = assign_oracle_budget(ensemble, group["oracle_map"],
                                                      group["pixel_counts"])
                    gains = replay_gain(group["sse"], action_map)
                    for operator, value in gains.items():
                        row[f"{cell}_{operator}"] = value
                    if cell == PRIMARY_CELL:
                        for seed, score in zip(SEEDS, seed_scores):
                            seed_map = assign_oracle_budget(score, group["oracle_map"],
                                                            group["pixel_counts"])
                            seed_gain = replay_gain(group["sse"], seed_map)
                            seed_rows.append({"name": group["name"], "fold": test_fold,
                                              "seed": seed, **seed_gain})
                evaluation_rows.append(row)
                for action_index, action in enumerate(ACTIVE_ACTIONS):
                    start = local_index * TILES * 2 + action_index * TILES
                    prediction = np.mean([predictions[PRIMARY_CELL][test_fold][seed]
                                          for seed in SEEDS], axis=0)[start:start + TILES]
                    for tile in range(TILES):
                        cloud_prediction_rows.append({
                            "name": group["name"], "fold": test_fold, "action": action,
                            "tile": tile, "predicted_mean": float(prediction[tile, 0]),
                            "predicted_worst": float(prediction[tile, 1]),
                            "predicted_eligible": int(float(prediction[tile, 1]) >= LOCAL_GAIN),
                            "actual_mean": float(group["mean_target"][action_index * TILES + tile]),
                            "actual_worst": float(group["worst_target"][action_index * TILES + tile]),
                            "eligible": int(group["eligible"][action_index * TILES + tile]),
                        })
        boot = bootstrap(evaluation_rows)
        write_workload_progress(context, completed_units=4786, stage="bootstrap_complete")

        severe = sum(any(row[f"{PRIMARY_CELL}_{operator}"] <= SEVERE_GAIN
                         for operator in OPERATORS) for row in evaluation_rows)
        hard = sum(any(row[f"{PRIMARY_CELL}_{operator}"] <= HARD_GAIN
                       for operator in OPERATORS) for row in evaluation_rows)
        fold_metrics = {str(fold): evaluate_rows(
            evaluation_rows, np.asarray([index for index, row in enumerate(evaluation_rows)
                                         if row["fold"] == fold], dtype=np.int64),
        ) for fold in FOLDS}
        fold_stability = all(
            fold_metrics[str(fold)][f"{PRIMARY_CELL}_gain"] >= ABSOLUTE_GAIN
            and fold_metrics[str(fold)][f"primary_minus_{control}"] >= INCREMENT_GAIN
            for fold in FOLDS for control in CELLS[1:]
        ) and all(fold_metrics[str(fold)]["primary_minus_global"] >= INCREMENT_GAIN
                  for fold in FOLDS)
        seed_summary = []
        pooled_seed_gains = []
        fold_seed_gains = []
        for seed in SEEDS:
            for fold in (*FOLDS, "pooled"):
                subset = [row for row in seed_rows if row["seed"] == seed
                          and (fold == "pooled" or row["fold"] == fold)]
                means = {operator: float(np.mean([row[operator] for row in subset]))
                         for operator in OPERATORS}
                gain = min(means.values())
                if fold == "pooled":
                    pooled_seed_gains.append(gain)
                else:
                    fold_seed_gains.append(gain)
                seed_summary.append({"record_type": "replay", "seed": seed,
                    "test_fold": fold, "groups": len(subset),
                    "d_ref_gain_db": means["D_ref"],
                    "d_rep_gain_db": means["D_rep"],
                    "worse_operator_gain_db": gain})
        seed_stable = (min(fold_seed_gains) >= 0.0
                       and max(pooled_seed_gains) - min(pooled_seed_gains) <= SEED_RANGE_MAX)
        training_dynamics_valid = all(
            math.isfinite(row["initial_loss"]) and math.isfinite(row["final_loss"])
            and row["final_loss"] < row["initial_loss"]
            for row in training_rows
        )
        gates = {
            "primary_gain_lcb95": boot[f"{PRIMARY_CELL}_gain"]["lcb95"] >= ABSOLUTE_GAIN,
            "oracle_retention_lcb95": boot["retention"]["lcb95"] >= RETENTION_MIN,
            **{f"primary_minus_{control}_lcb95":
               boot[f"primary_minus_{control}"]["lcb95"] >= INCREMENT_GAIN
               for control in CELLS[1:]},
            "primary_minus_global_lcb95":
                boot["primary_minus_global"]["lcb95"] >= INCREMENT_GAIN,
            "primary_minus_global_cvar5_lcb95":
                boot["primary_minus_global_cvar5"]["lcb95"] >= TAIL_MARGIN,
            "zero_primary_severe": severe == 0, "zero_primary_hard": hard == 0,
            "both_folds_material": fold_stability, "seed_stability": seed_stable,
        }
        passes = all(gates.values())
        decisive_fail = (
            boot[f"{PRIMARY_CELL}_gain"]["ucb95"] < ABSOLUTE_GAIN
            or boot["retention"]["ucb95"] < RETENTION_MIN
            or any(boot[f"primary_minus_{control}"]["ucb95"] < INCREMENT_GAIN
                   for control in CELLS[1:])
            or boot["primary_minus_global"]["ucb95"] < INCREMENT_GAIN
            or boot["primary_minus_global_cvar5"]["ucb95"] < TAIL_MARGIN
            or severe > 0 or hard > 0 or not fold_stability or not seed_stable
        )
        if passes:
            state, decision, authorizes = ("COMPLETED_GATE_PASS",
                "R11_A0_REGIONAL_OBSERVABILITY_PASS",
                "R11_SAFE_REGIONAL_COVERAGE_CONTRACT_REVIEW_ONLY")
        elif decisive_fail:
            state, decision, authorizes = ("COMPLETED_GATE_FAIL",
                "R11_A0_REGIONAL_OBSERVABILITY_FAIL_STOP", "NONE")
        else:
            state, decision, authorizes = ("COMPLETED_GATE_INCONCLUSIVE",
                "R11_A0_INPUT_OR_OBSERVABILITY_INCONCLUSIVE_STOP", "NONE")

        cell_rows = []
        for cell in CELLS:
            means = {operator: float(np.mean([row[f"{cell}_{operator}"]
                                              for row in evaluation_rows]))
                     for operator in OPERATORS}
            cell_rows.append({"cell": cell, "groups": len(evaluation_rows),
                              "d_ref_gain_db": means["D_ref"],
                              "d_rep_gain_db": means["D_rep"],
                              "worse_operator_gain_db": min(means.values()),
                              "selected_severe_groups": sum(any(
                                  row[f"{cell}_{operator}"] <= SEVERE_GAIN for operator in OPERATORS)
                                  for row in evaluation_rows),
                              "selected_hard_groups": sum(any(
                                  row[f"{cell}_{operator}"] <= HARD_GAIN for operator in OPERATORS)
                                  for row in evaluation_rows)})
        calibration_rows = []
        sorted_predictions = sorted(cloud_prediction_rows, key=lambda row: row["predicted_worst"])
        for bin_index, chunk in enumerate(np.array_split(sorted_predictions, 10)):
            values = list(chunk)
            calibration_rows.append({"bin": bin_index, "rows": len(values),
                "predicted_worst_mean": float(np.mean([row["predicted_worst"] for row in values])),
                "actual_worst_mean": float(np.mean([row["actual_worst"] for row in values])),
                "predicted_eligible_fraction": float(np.mean([row["predicted_eligible"] for row in values])),
                "actual_eligible_fraction": float(np.mean([row["eligible"] for row in values]))})
        margins = []
        operator_label_agreement = []
        for group in groups:
            for tile in range(TILES):
                scores = [0.0]
                for action_index in range(2):
                    values = group["gains"][:, action_index, tile]
                    scores.append(float(values.min()) if np.all(values >= LOCAL_GAIN) else -math.inf)
                    operator_label_agreement.append(bool((values[0] >= LOCAL_GAIN) ==
                                                         (values[1] >= LOCAL_GAIN)))
                finite = sorted([value for value in scores if math.isfinite(value)], reverse=True)
                margins.append(finite[0] - finite[1] if len(finite) > 1 else finite[0])

        structural = {
            "r10_terminal_and_gate_exact": True,
            "target_replay_exact": input_metadata["target_mismatches"] == 0,
            "no_op_render_exact": input_metadata["no_op_render_mismatches"] == 0,
            "tile_partition_exact": input_metadata["partition_sse_max_abs_difference"] <= 1.0e-9,
            "population_complete": len(groups) == EXPECTED_NAMES,
            "operator_units_complete": input_metadata["evaluated_units"] == EXPECTED_UNITS,
            "tile_action_rows_complete": input_metadata["tile_action_rows"] == EXPECTED_ROWS,
            "oof_group_disjoint": oof_overlap_violations == 0,
            "oracle_replay_exact": fold_replay_mismatches == 0,
            "training_dynamics_valid": training_dynamics_valid,
            "finite_metrics": all(math.isfinite(value["point"]) for value in boot.values()),
            "protected_roles_untouched": not any(context.protected_data_permissions.values()),
        }
        if not all(structural.values()):
            failed = sorted(key for key, value in structural.items() if not value)
            raise ObservabilityInconclusive(f"structural checks failed: {failed}")
        contract_summary = {"schema_version": 1, "route_id": ROUTE_ID,
            "operation_id": OPERATION_ID, "scientific_role": "development_oof_mechanism_screen",
            "cells": list(CELLS), "folds": list(FOLDS), "seeds": list(SEEDS),
            "epochs": EPOCHS, "batch_size": BATCH_SIZE, "feature_dim": FEATURE_DIM,
            "bootstrap_draws": BOOTSTRAP_DRAWS, "bootstrap_seed": BOOTSTRAP_SEED,
            "thresholds": {"absolute_gain_db": ABSOLUTE_GAIN,
                "increment_gain_db": INCREMENT_GAIN, "retention": RETENTION_MIN,
                "tail_margin_db": TAIL_MARGIN, "seed_range_db": SEED_RANGE_MAX,
                "local_gain_db": LOCAL_GAIN, "severe_gain_db": SEVERE_GAIN,
                "hard_gain_db": HARD_GAIN}}
        provenance = {"schema_version": 1, "route_commit": context.route_commit,
            "r10_terminal_preserved": True, "restoration_model_training_run": False,
            "restoration_model_inference_run": False, "diagnostic_probe_fitted": True,
            "candidate_generation_rerun": False, "checkpoint_loaded": False,
            "broader_r3_ledger_opened": False, "confirmation_touched": False,
            "canary_touched": False, "locked_test_touched": False}
        representation = {"schema_version": 1, "feature_dim": FEATURE_DIM,
            "state_channels_paired": 12, "response_channels_paired": 18,
            "neighborhood": [3, 3], "global_channels": 30,
            "coordinate_channels": 2, "action_sign_channels": 1,
            "normalizer_source": "primary_outer_training_rows_only",
            "oracle_budget_is_leakage_ineligible": True}
        oracle_replay = {"schema_version": 1, "worse_operator_gain_db": oracle_point,
            "tile_counts": {ACTIONS[action]: int(oracle_counts[action]) for action in range(3)},
            "mixed_fraction": mixed / EXPECTED_NAMES,
            "bidirectional_fraction": bidirectional / EXPECTED_NAMES,
            "local_materiality_violations": local_violations,
            "fold_metric_mismatches": int(fold_replay_mismatches)}
        gate_summary = {"schema_version": 1, "structural_checks": structural,
            "gates": gates, "fold_metrics": fold_metrics, "passes": passes,
            "decisive_fail": decisive_fail, "primary_severe_groups": severe,
            "primary_hard_groups": hard, "state": state, "decision": decision,
            "authorizes": authorizes, "r10_terminal_changed": False}
        atomic_json(context.phase_output_path / "r11_a0_contract_summary.json", contract_summary)
        atomic_json(context.phase_output_path / "r11_a0_provenance_and_access.json", provenance)
        atomic_json(context.phase_output_path / "r11_a0_input_identity.json", {
            "schema_version": 1, **input_metadata,
            "asset_sha256": {key: context.assets[key].sha256 for key in sorted(context.assets)}})
        atomic_json(context.phase_output_path / "r11_a0_representation_identity.json", representation)
        atomic_json(context.phase_output_path / "r11_a0_oracle_replay.json", oracle_replay)
        write_csv(context.phase_output_path / "r11_a0_cell_summary.csv", cell_rows)
        atomic_json(context.phase_output_path / "r11_a0_bootstrap_summary.json",
                    {"schema_version": 1, **boot})
        write_csv(context.phase_output_path / "r11_a0_fold_seed_stability.csv",
                  seed_summary + [{"record_type": "training", **row} for row in training_rows])
        atomic_json(context.phase_output_path / "r11_a0_label_stability.json", {
            "schema_version": 1, "operator_eligibility_agreement":
                float(np.mean(operator_label_agreement)),
            "oracle_margin_mean_db": float(np.mean(margins)),
            "oracle_margin_median_db": float(np.median(margins)),
            "oracle_margin_p10_db": float(np.quantile(margins, 0.10))})
        write_csv(context.phase_output_path / "r11_a0_calibration_summary.csv", calibration_rows)
        atomic_json(context.phase_output_path / "r11_a0_operator_consistency.json", {
            "schema_version": 1, "primary_gain_pearson": safe_pearson(
                [row[f"{PRIMARY_CELL}_D_ref"] for row in evaluation_rows],
                [row[f"{PRIMARY_CELL}_D_rep"] for row in evaluation_rows]),
            "shared_action_map_by_construction": True})
        atomic_json(context.phase_output_path / "r11_a0_gate_summary.json", gate_summary)
        atomic_json(context.phase_output_path / "r11_a0_resource_summary.json", {
            "schema_version": 1, "wall_seconds": time.perf_counter() - started_wall,
            "cpu_seconds": time.process_time() - started_cpu,
            "max_rss_mib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0,
            "evaluated_units": input_metadata["evaluated_units"], "models": 16,
            "epochs": EPOCHS, "bootstrap_draws": BOOTSTRAP_DRAWS, "gpu_used": False})
        write_csv(context.phase_output_path / "r11_a0_tile_predictions_cloud_only.csv",
                  cloud_prediction_rows)
        write_csv(context.phase_output_path / "r11_a0_per_image_policy_rows_cloud_only.csv",
                  evaluation_rows)
        write_run_result(context, state=state, decision=decision, authorizes=authorizes, details={
            "primary_gain_db": boot[f"{PRIMARY_CELL}_gain"]["point"],
            "oracle_retention": boot["retention"]["point"],
            "primary_minus_pooled_db": boot[f"primary_minus_{CELLS[1]}"]["point"],
            "primary_minus_shuffle_db": boot[f"primary_minus_{CELLS[2]}"]["point"],
            "primary_minus_generic_db": boot[f"primary_minus_{CELLS[3]}"]["point"],
            "primary_minus_global_db": boot["primary_minus_global"]["point"],
            "primary_severe_groups": severe, "primary_hard_groups": hard,
            "both_folds_material": fold_stability, "seed_stability": seed_stable,
            "r10_terminal_changed": False})
    except ObservabilityInconclusive as exc:
        write_inconclusive_bundle(context, str(exc), started_wall, started_cpu)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("contract", "run"))
    parser.add_argument("--context", required=True, type=Path)
    args = parser.parse_args()
    if args.phase == "contract":
        contract(args.context)
    else:
        run(args.context)


if __name__ == "__main__":
    main()
