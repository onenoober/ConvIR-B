#!/usr/bin/env python3
"""Frozen R5 A0 screen of candidate-relative spatial response information."""

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


ROUTE_ID = "haze4k_v5_r5_spatial_candidate_response_sufficiency_20260719"
OPERATION_ID = "R5_A0_FROZEN_SPATIAL_RESPONSE_SUFFICIENCY_SCREEN"
ACTIONS = ("reference_noop", "state_positive_full", "state_negative_full")
ACTIVE_ACTIONS = ACTIONS[1:]
OPERATORS = ("D_ref", "D_rep")
CELLS = (
    "P0_POOLED_DC_ONLY",
    "S1_TRUE_SPATIAL_RESPONSE",
    "S2_SPATIAL_RESPONSE_SHUFFLE",
    "G0_GENERIC_STATE_SPATIAL",
)
PRIMARY_CELL = "S1_TRUE_SPATIAL_RESPONSE"
POOLED_CELL = "P0_POOLED_DC_ONLY"
SHUFFLE_CELL = "S2_SPATIAL_RESPONSE_SHUFFLE"
GENERIC_CELL = "G0_GENERIC_STATE_SPATIAL"
FOLDS = (0, 1)
SEEDS = (3407, 3411)
GRID = 8
EPOCHS = 32
LEARNING_RATE = 1.0e-3
WEIGHT_DECAY = 1.0e-4
BATCH_SIZE = 64
COVERAGE = 0.20
BOOTSTRAP_DRAWS = 4000
BOOTSTRAP_SEED = 3407
FEATURE_UNITS = 1536
TRAINING_UNITS = len(FOLDS) * len(CELLS) * len(SEEDS)
TOTAL_UNITS = FEATURE_UNITS + TRAINING_UNITS
STATE_DIM = 40
ACTION_DIM = 5
RESPONSE_DIM = 3 * GRID * GRID
GENERIC_SPATIAL_DIM = 3 * (GRID * GRID - 1)
INPUT_DIM = STATE_DIM + ACTION_DIM + RESPONSE_DIM + GENERIC_SPATIAL_DIM
HIDDEN_DIM = 64
SEVERE_GAIN = -0.2
HARD_GAIN = -0.5
GAIN_TARGET = 0.020
RETENTION_TARGET = 0.25
INCREMENT_TARGET = 0.005
SEVERE_RATE_UCB_TARGET = 0.010
TAIL_MARGIN = -0.005
PROTECTED_HARM_MARGIN = 0.005
SHAPE_MEAN_FLOOR = -0.020
CACHE_MANIFEST_IDENTITY = "b54603b51c810436c14bc9e993aef9f1294902efbc51fadebdd2c84d7c827a1d"

STATE_START = 0
ACTION_START = STATE_START + STATE_DIM
RESPONSE_START = ACTION_START + ACTION_DIM
GENERIC_START = RESPONSE_START + RESPONSE_DIM


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    fields = sorted({key for row in rows for key in row})
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def metric_psnr(mse: Any) -> Any:
    import torch

    return 10.0 * torch.log10(1.0 / torch.clamp(mse, min=1.0e-30))


def tensor_stats(value: Any) -> Any:
    import torch

    flat = value.float().flatten(2)
    return torch.cat(
        (flat.mean(2), flat.std(2, unbiased=False), flat.abs().mean(2), flat.abs().amax(2)),
        dim=1,
    )


def dct_matrix(*, dtype: Any, device: Any) -> Any:
    import torch

    sample = torch.arange(GRID, dtype=dtype, device=device)
    frequency = sample[:, None]
    matrix = torch.cos(math.pi * (sample[None, :] + 0.5) * frequency / GRID)
    scale = torch.full((GRID,), math.sqrt(2.0 / GRID), dtype=dtype, device=device)
    scale[0] = math.sqrt(1.0 / GRID)
    return scale[:, None] * matrix


def dct2(grid: Any) -> Any:
    matrix = dct_matrix(dtype=grid.dtype, device=grid.device)
    return matrix @ grid @ matrix.t()


def response_dct_features(grid: Any) -> Any:
    return dct2(grid).flatten(1)


def state_non_dc_features(grid: Any) -> Any:
    coefficients = dct2(grid).reshape(grid.shape[0], 3, GRID * GRID)
    return coefficients[:, :, 1:].flatten(1)


def deterministic_permutation(name: str, operator: str, action: str) -> Any:
    import torch

    raw = hashlib.sha256(f"{ROUTE_ID}|{name}|{operator}|{action}".encode()).digest()
    seed = int.from_bytes(raw[:8], "big") % (2**63 - 1)
    return torch.randperm(GRID * GRID, generator=torch.Generator().manual_seed(seed))


def action_features() -> Any:
    import torch

    return torch.tensor(
        ((0.0, 1.0, 0.0, 1.0, 1.0), (0.0, 0.0, 1.0, -1.0, 1.0)),
        dtype=torch.float32,
    )


def response_ac_indices() -> list[int]:
    return [
        RESPONSE_START + channel * GRID * GRID + index
        for channel in range(3)
        for index in range(1, GRID * GRID)
    ]


def cell_mask(cell: str) -> Any:
    import torch

    mask = torch.ones(INPUT_DIM, dtype=torch.float32)
    if cell in {POOLED_CELL, GENERIC_CELL}:
        mask[response_ac_indices()] = 0.0
    if cell in {POOLED_CELL, PRIMARY_CELL, SHUFFLE_CELL}:
        mask[GENERIC_START:] = 0.0
    return mask


def raw_cell_features(unit: dict[str, Any], cell: str) -> Any:
    if cell == SHUFFLE_CELL:
        return unit["shuffle_features"]
    return unit["true_features"]


def normalized_cell_features(
    units: list[dict[str, Any]], cell: str, mean: Any, scale: Any,
) -> Any:
    import torch

    raw = torch.stack([raw_cell_features(unit, cell) for unit in units])
    return ((raw - mean) / scale) * cell_mask(cell)


def fit_normalizer(units: list[dict[str, Any]]) -> tuple[Any, Any]:
    import torch

    rows = torch.stack([unit["true_features"] for unit in units]).reshape(-1, INPUT_DIM)
    mean = rows.mean(0)
    scale = rows.std(0, unbiased=False).clamp(min=1.0e-6)
    return mean, scale


def build_model(seed: int) -> Any:
    import torch

    torch.manual_seed(seed)
    model = torch.nn.Sequential(
        torch.nn.Linear(INPUT_DIM, HIDDEN_DIM),
        torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN_DIM, 3),
    )
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            torch.nn.init.zeros_(module.bias)
    return model


def model_loss(raw: Any, utility: Any, severe: Any) -> Any:
    import torch
    import torch.nn.functional as functional

    mean_loss = functional.smooth_l1_loss(raw[..., 0], utility)
    error = utility - raw[..., 1]
    q05_loss = torch.maximum(0.05 * error, -0.95 * error).mean()
    severe_loss = functional.binary_cross_entropy_with_logits(raw[..., 2], severe)
    return mean_loss + q05_loss + severe_loss


def train_model(features: Any, utility: Any, severe: Any, seed: int) -> tuple[Any, float, float]:
    import torch

    model = build_model(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    generator = torch.Generator().manual_seed(seed + 17)
    with torch.no_grad():
        initial = float(model_loss(model(features), utility, severe))
    for _epoch in range(EPOCHS):
        for indices in torch.randperm(len(features), generator=generator).split(BATCH_SIZE):
            loss = model_loss(model(features[indices]), utility[indices], severe[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        final = float(model_loss(model(features), utility, severe))
    return model, initial, final


def predict(model: Any, features: Any) -> Any:
    import torch

    with torch.no_grad():
        output = model(features)
        output[..., 2] = output[..., 2].sigmoid()
    return output


def tie_key(fold: int, name: str) -> str:
    return hashlib.sha256(f"{ROUTE_ID}|fold={fold}|{name}".encode()).hexdigest()


def select_groups(
    names: list[str], fold: int, q05_by_name: dict[str, tuple[float, float]], coverage: float,
) -> tuple[dict[str, int], set[str], dict[str, float]]:
    actions = {
        name: max(range(2), key=lambda action: (q05_by_name[name][action], -action))
        for name in names
    }
    scores = {name: q05_by_name[name][actions[name]] for name in names}
    ordered = sorted(names, key=lambda name: (-scores[name], tie_key(fold, name)))
    count = max(1, math.ceil(coverage * len(names)))
    return actions, set(ordered[:count]), scores


def make_policy(
    test_units: list[dict[str, Any]], predictions: dict[tuple[str, str], Any],
    fold: int, cell: str, coverage: float = COVERAGE,
) -> list[dict[str, Any]]:
    unit_map = {(unit["name"], unit["operator"]): unit for unit in test_units}
    names = sorted({unit["name"] for unit in test_units})
    if any((name, operator) not in unit_map for name in names for operator in OPERATORS):
        raise RuntimeError("operator pairing is incomplete")
    q05_by_name = {
        name: tuple(
            min(float(predictions[(name, operator)][action, 1]) for operator in OPERATORS)
            for action in range(2)
        )
        for name in names
    }
    actions, selected_names, scores = select_groups(names, fold, q05_by_name, coverage)
    rows = []
    for name in names:
        robust_truth = tuple(
            min(float(unit_map[(name, operator)]["target"][action]) for operator in OPERATORS)
            for action in range(2)
        )
        oracle_action = max(range(2), key=lambda action: (robust_truth[action], -action))
        oracle_selected = oracle_action if robust_truth[oracle_action] > 0.0 else None
        selected_action = actions[name] if name in selected_names else None
        for operator in OPERATORS:
            unit = unit_map[(name, operator)]
            gain = 0.0 if selected_action is None else float(unit["target"][selected_action])
            oracle_gain = 0.0 if oracle_selected is None else float(unit["target"][oracle_selected])
            prediction = predictions[(name, operator)]
            rows.append(
                {
                    "cell": cell,
                    "fold": fold,
                    "name": name,
                    "operator": operator,
                    "selected": 0 if selected_action is None else selected_action + 1,
                    "gain": gain,
                    "oracle_gain": oracle_gain,
                    "oracle_selected": 0 if oracle_selected is None else oracle_selected + 1,
                    "negative_oracle": oracle_selected == 1,
                    "robust_score": scores[name],
                    "mean_score": 0.0 if selected_action is None else float(prediction[selected_action, 0]),
                    "q05_score": 0.0 if selected_action is None else float(prediction[selected_action, 1]),
                    "severe_score": 0.0 if selected_action is None else float(prediction[selected_action, 2]),
                }
            )
    return rows


def cvar(values: Any, fraction: float = 0.05) -> float:
    import numpy as np

    array = np.sort(np.asarray(values, dtype=np.float64))
    return float(array[: max(1, math.ceil(fraction * len(array)))].mean())


def interval(point: float, samples: list[float]) -> dict[str, float]:
    import numpy as np

    return {
        "point": float(point),
        "lcb95": float(np.quantile(samples, 0.025)),
        "ucb95": float(np.quantile(samples, 0.975)),
    }


def policy_bootstrap(policy_rows: list[dict[str, Any]]) -> dict[str, Any]:
    import numpy as np

    names = sorted({row["name"] for row in policy_rows})
    keyed = {(row["cell"], row["name"], row["operator"]): row for row in policy_rows}
    gains = {
        cell: {
            operator: np.asarray(
                [float(keyed[(cell, name, operator)]["gain"]) for name in names], dtype=np.float64
            )
            for operator in OPERATORS
        }
        for cell in CELLS
    }
    oracle = {
        operator: np.asarray(
            [float(keyed[(PRIMARY_CELL, name, operator)]["oracle_gain"]) for name in names],
            dtype=np.float64,
        )
        for operator in OPERATORS
    }

    def evaluate(indices: Any) -> dict[str, float]:
        cell_mean = {
            cell: [float(gains[cell][operator][indices].mean()) for operator in OPERATORS]
            for cell in CELLS
        }
        primary = cell_mean[PRIMARY_CELL]
        oracle_mean = [float(oracle[operator][indices].mean()) for operator in OPERATORS]
        worst = min(range(2), key=lambda index: primary[index])
        return {
            "gain": primary[worst],
            "retention": primary[worst] / max(oracle_mean[worst], 1.0e-12),
            "spatial_minus_pooled": min(
                primary[index] - cell_mean[POOLED_CELL][index] for index in range(2)
            ),
            "true_minus_shuffle": min(
                primary[index] - cell_mean[SHUFFLE_CELL][index] for index in range(2)
            ),
            "spatial_minus_generic": min(
                primary[index] - cell_mean[GENERIC_CELL][index] for index in range(2)
            ),
            "cvar5_spatial_minus_pooled": min(
                cvar(gains[PRIMARY_CELL][operator][indices])
                - cvar(gains[POOLED_CELL][operator][indices])
                for operator in OPERATORS
            ),
        }

    point = evaluate(np.arange(len(names)))
    samples = {key: [] for key in point}
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    for _draw in range(BOOTSTRAP_DRAWS):
        value = evaluate(generator.integers(0, len(names), len(names)))
        for key in samples:
            samples[key].append(value[key])
    return {key: interval(point[key], values) for key, values in samples.items()}


def auc_ap(labels: Any, scores: Any) -> dict[str, float]:
    import numpy as np
    from scipy.stats import rankdata

    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return {"auroc": math.nan, "auprc": math.nan, "prevalence": positives / max(len(labels), 1)}
    ranks = rankdata(scores, method="average")
    auroc = (float(ranks[labels == 1].sum()) - positives * (positives + 1) / 2) / (
        positives * negatives
    )
    order = np.argsort(-scores, kind="mergesort")
    ordered = labels[order]
    precision = np.cumsum(ordered) / np.arange(1, len(ordered) + 1)
    auprc = float(precision[ordered == 1].mean())
    return {"auroc": float(auroc), "auprc": auprc, "prevalence": positives / len(labels)}


def risk_bootstrap(group_rows: list[dict[str, Any]]) -> dict[str, Any]:
    import numpy as np

    names = sorted({row["name"] for row in group_rows})
    keyed = {(row["name"], row["action"]): row for row in group_rows}

    def evaluate(chosen: list[str]) -> dict[str, float]:
        rows = [keyed[(name, action)] for name in chosen for action in (1, 2)]
        metric = auc_ap(
            [int(row["severe_label"]) for row in rows],
            [float(row["severe_score"]) for row in rows],
        )
        return {
            "severe_auroc": metric["auroc"],
            "severe_auprc_lift": metric["auprc"] - metric["prevalence"],
        }

    point = evaluate(names)
    samples = {key: [] for key in point}
    generator = np.random.default_rng(BOOTSTRAP_SEED + 41)
    for _draw in range(BOOTSTRAP_DRAWS):
        chosen = [names[index] for index in generator.integers(0, len(names), len(names))]
        value = evaluate(chosen)
        for key in samples:
            samples[key].append(value[key])
    return {key: interval(point[key], values) for key, values in samples.items()}


def calibration_rows(labels: list[int], scores: list[float], bins: int = 10) -> tuple[list[dict[str, Any]], float, float]:
    import numpy as np

    label_array = np.asarray(labels, dtype=np.float64)
    score_array = np.asarray(scores, dtype=np.float64)
    rows = []
    ece = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        mask = (score_array >= low) & (
            (score_array < high) if index + 1 < bins else (score_array <= high)
        )
        if not mask.any():
            continue
        confidence = float(score_array[mask].mean())
        rate = float(label_array[mask].mean())
        weight = float(mask.mean())
        ece += weight * abs(confidence - rate)
        rows.append(
            {
                "row_type": "severe_bin",
                "bin": index,
                "low": low,
                "high": high,
                "count": int(mask.sum()),
                "mean_probability": confidence,
                "event_rate": rate,
            }
        )
    brier = float(np.mean((score_array - label_array) ** 2))
    return rows, ece, brier


def exact_binomial_ucb(events: int, total: int) -> float:
    from scipy.stats import beta

    if total <= 0 or not 0 <= events <= total:
        raise ValueError("invalid binomial inputs")
    return 1.0 if events == total else float(beta.ppf(0.95, events + 1, total - events))


def pearson(first: list[float], second: list[float]) -> float:
    import numpy as np

    return float(np.corrcoef(np.asarray(first), np.asarray(second))[0, 1])


def policy_group_counts(rows: list[dict[str, Any]]) -> tuple[int, int, int]:
    keyed = {(row["name"], row["operator"]): row for row in rows}
    names = sorted({row["name"] for row in rows})
    severe = sum(
        any(float(keyed[(name, operator)]["gain"]) <= SEVERE_GAIN for operator in OPERATORS)
        for name in names
    )
    hard = sum(
        any(float(keyed[(name, operator)]["gain"]) <= HARD_GAIN for operator in OPERATORS)
        for name in names
    )
    acted = sum(int(keyed[(name, OPERATORS[0])]["selected"]) != 0 for name in names)
    return acted, severe, hard


def protected_harm_rows(
    policy_rows: list[dict[str, Any]], test_units: list[dict[str, Any]],
    fold_thresholds: dict[int, float], cell: str,
) -> list[dict[str, Any]]:
    import torch

    policy = {(row["name"], row["operator"]): row for row in policy_rows if row["cell"] == cell}
    rows = []
    for unit in test_units:
        selected = int(policy[(unit["name"], unit["operator"])]["selected"])
        mask = (unit["local_gain"] <= 0.0).all(0) & (
            unit["reference_cell_mse"] <= fold_thresholds[int(unit["fold"])]
        )
        count = int(mask.sum())
        if count == 0:
            harm = math.nan
        elif selected == 0:
            harm = 0.0
        else:
            harm = float(torch.clamp(-unit["local_gain"][selected - 1, mask], min=0.0).mean())
        rows.append(
            {
                "cell": cell,
                "fold": unit["fold"],
                "name": unit["name"],
                "operator": unit["operator"],
                "protected_cells": count,
                "protected_harm_db": harm,
            }
        )
    return rows


def protected_harm_bootstrap(rows: list[dict[str, Any]]) -> dict[str, float]:
    import numpy as np

    names = sorted({row["name"] for row in rows})
    keyed = {(row["cell"], row["name"], row["operator"]): row for row in rows}

    def evaluate(indices: Any) -> float:
        differences = []
        for operator in OPERATORS:
            primary = np.asarray(
                [float(keyed[(PRIMARY_CELL, names[index], operator)]["protected_harm_db"]) for index in indices],
                dtype=np.float64,
            )
            pooled = np.asarray(
                [float(keyed[(POOLED_CELL, names[index], operator)]["protected_harm_db"]) for index in indices],
                dtype=np.float64,
            )
            valid = np.isfinite(primary) & np.isfinite(pooled)
            if not valid.any():
                raise RuntimeError("protected-cell bootstrap has no valid groups")
            differences.append(float((primary[valid] - pooled[valid]).mean()))
        return max(differences)

    point = evaluate(np.arange(len(names)))
    generator = np.random.default_rng(BOOTSTRAP_SEED + 73)
    samples = [evaluate(generator.integers(0, len(names), len(names))) for _ in range(BOOTSTRAP_DRAWS)]
    return interval(point, samples)


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
    raise FileNotFoundError(name)


def load_feature_unit(
    row: dict[str, Any], cache_root: Path, data_root: Path, fold_lookup: dict[str, int],
) -> tuple[dict[str, Any], float, float]:
    import torch
    import torch.nn.functional as functional

    unit_path = cache_root / f"{row['unit_key']}.pt"
    if sha256_file(unit_path) != row["cache_sha256"]:
        raise RuntimeError(f"cache hash mismatch: {row['unit_key']}")
    payload = torch.load(unit_path, map_location="cpu", weights_only=False)
    candidate_names = list(payload["candidate_names"])
    selected_indices = [candidate_names.index(action) for action in ACTIONS]
    name = payload["name"]
    operator = payload["operator"]
    if name not in fold_lookup or operator not in OPERATORS:
        raise RuntimeError("cache identity is outside the frozen ledger/operator contract")
    base = payload["base"].float()
    step = payload["step"].float()
    current = payload["current"].float()
    support = payload["support"].float()
    candidate_delta = payload["candidates"].float()[selected_indices]
    label = load_label(data_root, name)
    if label.shape[-2:] != base.shape[-2:]:
        label = label[:, :, : base.shape[-2], : base.shape[-1]]
    reference = torch.clamp(base + 0.25 * step, 0.0, 1.0)
    renders = torch.clamp(base + 0.25 * (step + candidate_delta), 0.0, 1.0)
    active_renders = renders[1:]
    target = metric_psnr((active_renders - label).square().mean((1, 2, 3))) - metric_psnr(
        (reference - label).square().mean()
    )

    state = torch.cat(
        (tensor_stats(base), tensor_stats(step), tensor_stats(current), tensor_stats(support)), dim=1
    )
    if state.shape != (1, STATE_DIM):
        raise RuntimeError(f"state feature identity mismatch: {state.shape}")
    response_grid = functional.adaptive_avg_pool2d(active_renders - reference, (GRID, GRID))
    response_features = response_dct_features(response_grid)
    state_grid = functional.adaptive_avg_pool2d(reference, (GRID, GRID))
    generic_features = state_non_dc_features(state_grid).expand(2, -1)
    true_features = torch.cat(
        (state.expand(2, -1), action_features(), response_features, generic_features), dim=1
    )

    shuffled_grids = []
    marginal_max = 0.0
    for action_index, action in enumerate(ACTIVE_ACTIONS):
        permutation = deterministic_permutation(name, operator, action)
        flat = response_grid[action_index].reshape(3, GRID * GRID)
        shuffled = flat[:, permutation].reshape(3, GRID, GRID)
        shuffled_grids.append(shuffled)
        marginal_max = max(
            marginal_max,
            float((flat.sort(1).values - shuffled.reshape(3, GRID * GRID).sort(1).values).abs().max()),
        )
    shuffled_grid = torch.stack(shuffled_grids)
    shuffled_response = response_dct_features(shuffled_grid)
    shuffle_features = torch.cat(
        (state.expand(2, -1), action_features(), shuffled_response, generic_features), dim=1
    )
    dc_positions = [channel * GRID * GRID for channel in range(3)]
    dc_max = float((response_features[:, dc_positions] - shuffled_response[:, dc_positions]).abs().max())

    reference_cell_mse = functional.adaptive_avg_pool2d(
        (reference - label).square(), (GRID, GRID)
    ).mean(1).flatten()
    active_cell_mse = functional.adaptive_avg_pool2d(
        (active_renders - label).square(), (GRID, GRID)
    ).mean(1).flatten(1)
    local_gain = metric_psnr(active_cell_mse) - metric_psnr(reference_cell_mse)[None, :]
    if true_features.shape != (2, INPUT_DIM) or shuffle_features.shape != (2, INPUT_DIM):
        raise RuntimeError("feature dimension contract failed")
    if not all(
        bool(torch.isfinite(value).all())
        for value in (true_features, shuffle_features, target, reference_cell_mse, local_gain)
    ):
        raise RuntimeError("feature or target contains non-finite values")
    unit = {
        "name": name,
        "operator": operator,
        "fold": fold_lookup[name],
        "shape": f"{base.shape[-2]}x{base.shape[-1]}",
        "true_features": true_features,
        "shuffle_features": shuffle_features,
        "target": target,
        "reference_cell_mse": reference_cell_mse,
        "local_gain": local_gain,
    }
    return unit, dc_max, marginal_max


def contract(context_path: Path) -> None:
    import torch

    context = load_context(context_path, "contract")
    prepare_phase_output(context)
    torch.manual_seed(SEEDS[0])
    torch.use_deterministic_algorithms(True)
    grid = torch.randn(4, 3, GRID, GRID)
    coefficients = dct2(grid)
    matrix = dct_matrix(dtype=grid.dtype, device=grid.device)
    reconstructed = matrix.t() @ coefficients @ matrix
    constant = torch.ones(2, 3, GRID, GRID)
    constant_coefficients = dct2(constant).reshape(2, 3, GRID * GRID)
    permutation = torch.randperm(GRID * GRID, generator=torch.Generator().manual_seed(3407))
    shuffled = grid.reshape(4, 3, GRID * GRID)[:, :, permutation].reshape_as(grid)

    features = torch.randn(FEATURE_UNITS, 2, INPUT_DIM)
    utility = torch.randn(FEATURE_UNITS, 2) * 0.1
    severe = (utility <= SEVERE_GAIN).float()
    model = build_model(SEEDS[0])
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    started = time.perf_counter()
    generator = torch.Generator().manual_seed(SEEDS[0] + 17)
    for indices in torch.randperm(FEATURE_UNITS, generator=generator).split(BATCH_SIZE):
        loss = model_loss(model(features[indices]), utility[indices], severe[indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    epoch_seconds = time.perf_counter() - started
    projected_seconds = epoch_seconds * EPOCHS * TRAINING_UNITS

    micro_features = features[:32].clone()
    micro_utility = utility[:32].clone()
    micro_severe = severe[:32].clone()
    micro_model = build_model(SEEDS[1])
    micro_optimizer = torch.optim.AdamW(
        micro_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    with torch.no_grad():
        micro_initial = float(model_loss(micro_model(micro_features), micro_utility, micro_severe))
    gradient_max = 0.0
    for _step in range(64):
        micro_loss = model_loss(micro_model(micro_features), micro_utility, micro_severe)
        micro_optimizer.zero_grad(set_to_none=True)
        micro_loss.backward()
        gradient_max = max(
            gradient_max,
            max(float(parameter.grad.abs().max()) for parameter in micro_model.parameters()),
        )
        micro_optimizer.step()
    with torch.no_grad():
        micro_final = float(model_loss(micro_model(micro_features), micro_utility, micro_severe))

    synthetic_names = [f"synthetic_{index:03d}" for index in range(192)]
    synthetic_scores = {
        name: (float(index), float(-index)) for index, name in enumerate(synthetic_names)
    }
    _actions, selected, _scores = select_groups(synthetic_names, 0, synthetic_scores, COVERAGE)
    expected_environment = {
        "CONVIR_ROUTE_ACTIONS": ",".join(ACTIONS),
        "CONVIR_ROUTE_CELLS": ",".join(CELLS),
        "CONVIR_ROUTE_FOLDS": "0,1",
        "CONVIR_ROUTE_SEEDS": "3407,3411",
        "CONVIR_ROUTE_EPOCHS": "32",
        "CONVIR_ROUTE_GRID": "8",
        "CONVIR_ROUTE_COVERAGE": "0.20",
    }
    checks = {
        "context_cpu_only": context.device == "cpu" and os.environ.get("CUDA_VISIBLE_DEVICES") == "",
        "contract_assets_absent": not context.assets,
        "environment_identity": all(os.environ.get(key) == value for key, value in expected_environment.items()),
        "dct_inverse": float((grid - reconstructed).abs().max()) <= 1.0e-5,
        "constant_has_zero_ac": float(constant_coefficients[:, :, 1:].abs().max()) <= 1.0e-5,
        "shuffle_preserves_marginals": float(
            (grid.flatten(2).sort(2).values - shuffled.flatten(2).sort(2).values).abs().max()
        ) == 0.0,
        "cell_masks_distinct_complete": all(int(cell_mask(cell).numel()) == INPUT_DIM for cell in CELLS)
        and len({cell_mask(cell).numpy().tobytes() for cell in CELLS}) == 3
        and bool(torch.equal(cell_mask(PRIMARY_CELL), cell_mask(SHUFFLE_CELL))),
        "production_model_finite": all(
            bool(torch.isfinite(parameter).all()) for parameter in model.parameters()
        ),
        "production_gradient_nonzero": gradient_max > 0.0 and math.isfinite(gradient_max),
        "microfit_loss_decreased": micro_final < micro_initial,
        "formal_size_probe_bounded": projected_seconds <= 5400.0,
        "formal_size_memory_bounded": features.numel() * features.element_size() <= 32 * 1024 * 1024,
        "fixed_coverage_exact": len(selected) == math.ceil(COVERAGE * 192) == 39,
        "total_units_exact": context.total_units == TOTAL_UNITS,
        "protected_roles_blocked": not any(context.protected_data_permissions.values()),
        "workload_absent": not (context.output_path / "workload").exists(),
    }
    atomic_json(
        context.phase_output_path / "r5_a0_synthetic_contract.json",
        {
            "schema_version": 1,
            "checks": checks,
            "dct_reconstruction_max_abs": float((grid - reconstructed).abs().max()),
            "constant_ac_max_abs": float(constant_coefficients[:, :, 1:].abs().max()),
            "one_epoch_seconds": epoch_seconds,
            "projected_formal_seconds": projected_seconds,
            "microfit_initial": micro_initial,
            "microfit_final": micro_final,
            "gradient_max_abs": gradient_max,
            "synthetic_selected_groups": len(selected),
            "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        },
    )
    write_contract_result(context, checks=checks)


def run(context_path: Path) -> None:
    import numpy as np
    import torch

    context = load_context(context_path, "run")
    prepare_phase_output(context)
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    torch.use_deterministic_algorithms(True)

    ledger_path = asset_path(context, "r4_ledger", kind="file")
    cache_manifest_path = asset_path(context, "a0_cache_manifest", kind="file")
    raw_manifest_path = asset_path(context, "a0_raw_manifest", kind="file")
    cache_root = asset_path(context, "a0_candidate_cache", kind="directory")
    data_root = asset_path(context, "haze4k_data", kind="directory")
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
    raw_rows = [
        json.loads(line)
        for line in raw_manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    folds = {int(key): list(value) for key, value in ledger["development_folds"].items()}
    fold_lookup = {name: fold for fold, names in folds.items() for name in names}
    development = set(ledger["roles"]["development"])
    confirmation = set(ledger["roles"]["confirmation"])
    if set(fold_lookup) != development or development & confirmation:
        raise RuntimeError("data-role isolation failed")
    if len(raw_rows) != FEATURE_UNITS or cache_manifest.get("cache_manifest_sha256") != CACHE_MANIFEST_IDENTITY:
        raise RuntimeError("candidate cache manifest identity failed")

    units = []
    shuffle_dc_max = 0.0
    shuffle_marginal_max = 0.0
    for index, row in enumerate(raw_rows, 1):
        unit, dc_max, marginal_max = load_feature_unit(row, cache_root, data_root, fold_lookup)
        units.append(unit)
        shuffle_dc_max = max(shuffle_dc_max, dc_max)
        shuffle_marginal_max = max(shuffle_marginal_max, marginal_max)
        if index % 8 == 0 or index == FEATURE_UNITS:
            write_workload_progress(context, completed_units=index, stage="spatial_feature_extract")

    by_name: dict[str, list[dict[str, Any]]] = {}
    for unit in units:
        by_name.setdefault(unit["name"], []).append(unit)
    if len(by_name) != 768 or any(sorted(item["operator"] for item in group) != sorted(OPERATORS) for group in by_name.values()):
        raise RuntimeError("clean-image/operator grouping failed")
    for group in by_name.values():
        severe_any = torch.stack([item["target"] for item in group]).le(SEVERE_GAIN).any(0).float()
        for item in group:
            item["severe_any"] = severe_any

    policy_rows: list[dict[str, Any]] = []
    per_seed_rows: list[dict[str, Any]] = []
    ensemble_score_rows: list[dict[str, Any]] = []
    fold_seed_rows: list[dict[str, Any]] = []
    fold_predictions: dict[tuple[int, str], dict[tuple[str, str], Any]] = {}
    fold_test_units: dict[int, list[dict[str, Any]]] = {}
    completed_training = 0
    for fold in FOLDS:
        train_units = [unit for unit in units if int(unit["fold"]) != fold]
        test_units = [unit for unit in units if int(unit["fold"]) == fold]
        fold_test_units[fold] = test_units
        mean, scale = fit_normalizer(train_units)
        utility_train = torch.stack([unit["target"] for unit in train_units])
        severe_train = torch.stack([unit["severe_any"] for unit in train_units])
        for cell in CELLS:
            x_train = normalized_cell_features(train_units, cell, mean, scale)
            x_test = normalized_cell_features(test_units, cell, mean, scale)
            seed_predictions = []
            for seed in SEEDS:
                model, initial_loss, final_loss = train_model(
                    x_train, utility_train, severe_train, seed
                )
                prediction = predict(model, x_test)
                seed_predictions.append(prediction)
                seed_map = {
                    (unit["name"], unit["operator"]): prediction[index]
                    for index, unit in enumerate(test_units)
                }
                seed_policy = make_policy(test_units, seed_map, fold, cell)
                for operator in OPERATORS:
                    subset = [row for row in seed_policy if row["operator"] == operator]
                    acted, severe_count, hard_count = policy_group_counts(seed_policy)
                    fold_seed_rows.append(
                        {
                            "fold": fold,
                            "cell": cell,
                            "seed": seed,
                            "operator": operator,
                            "gain_mean_db": sum(float(row["gain"]) for row in subset) / len(subset),
                            "coverage": acted / len({row["name"] for row in seed_policy}),
                            "severe_groups": severe_count,
                            "hard_groups": hard_count,
                            "initial_loss": initial_loss,
                            "final_loss": final_loss,
                            "epochs": EPOCHS,
                        }
                    )
                for index, unit in enumerate(test_units):
                    for action in range(2):
                        per_seed_rows.append(
                            {
                                "fold": fold,
                                "cell": cell,
                                "seed": seed,
                                "name": unit["name"],
                                "operator": unit["operator"],
                                "action": ACTIVE_ACTIONS[action],
                                "target_gain_db": float(unit["target"][action]),
                                "mean_score": float(prediction[index, action, 0]),
                                "q05_score": float(prediction[index, action, 1]),
                                "severe_score": float(prediction[index, action, 2]),
                            }
                        )
                completed_training += 1
                write_workload_progress(
                    context,
                    completed_units=FEATURE_UNITS + completed_training,
                    stage="spatial_readout_train",
                )
            ensemble = torch.stack(seed_predictions).mean(0)
            prediction_map = {
                (unit["name"], unit["operator"]): ensemble[index]
                for index, unit in enumerate(test_units)
            }
            fold_predictions[(fold, cell)] = prediction_map
            policy_rows.extend(make_policy(test_units, prediction_map, fold, cell))
            for index, unit in enumerate(test_units):
                for action in range(2):
                    ensemble_score_rows.append(
                        {
                            "fold": fold,
                            "cell": cell,
                            "name": unit["name"],
                            "operator": unit["operator"],
                            "action": ACTIVE_ACTIONS[action],
                            "target_gain_db": float(unit["target"][action]),
                            "severe_label": int(unit["severe_any"][action]),
                            "mean_score": float(ensemble[index, action, 0]),
                            "q05_score": float(ensemble[index, action, 1]),
                            "severe_score": float(ensemble[index, action, 2]),
                        }
                    )

    write_csv(context.phase_output_path / "r5_a0_per_seed_predictions_cloud_only.csv", per_seed_rows)
    write_csv(context.phase_output_path / "r5_a0_candidate_scores_cloud_only.csv", ensemble_score_rows)
    write_csv(context.phase_output_path / "r5_a0_policy_rows_cloud_only.csv", policy_rows)

    policy_metrics = policy_bootstrap(policy_rows)
    risk_group_rows = []
    for name in sorted(by_name):
        fold = fold_lookup[name]
        prediction_map = fold_predictions[(fold, PRIMARY_CELL)]
        group = {(unit["operator"]): unit for unit in by_name[name]}
        for action in range(2):
            risk_group_rows.append(
                {
                    "name": name,
                    "action": action + 1,
                    "severe_label": int(group[OPERATORS[0]]["severe_any"][action]),
                    "severe_score": max(
                        float(prediction_map[(name, operator)][action, 2]) for operator in OPERATORS
                    ),
                }
            )
    risk_metrics = risk_bootstrap(risk_group_rows)

    fold_thresholds = {}
    for fold in FOLDS:
        training_reference = torch.cat(
            [unit["reference_cell_mse"] for unit in units if int(unit["fold"]) != fold]
        )
        fold_thresholds[fold] = float(torch.quantile(training_reference, 0.25))
    test_units_all = [unit for unit in units if int(unit["fold"]) in FOLDS]
    protected_rows = []
    for cell in (PRIMARY_CELL, POOLED_CELL):
        protected_rows.extend(
            protected_harm_rows(policy_rows, test_units_all, fold_thresholds, cell)
        )
    protected_metric = protected_harm_bootstrap(protected_rows)
    policy_metrics["protected_harm_increment"] = protected_metric

    cell_summary_rows = []
    for cell in CELLS:
        subset = [row for row in policy_rows if row["cell"] == cell]
        acted, severe_count, hard_count = policy_group_counts(subset)
        operator_gains = {
            operator: sum(float(row["gain"]) for row in subset if row["operator"] == operator)
            / sum(row["operator"] == operator for row in subset)
            for operator in OPERATORS
        }
        cell_summary_rows.append(
            {
                "cell": cell,
                "gain_point_db": min(operator_gains.values()),
                "d_ref_gain_db": operator_gains["D_ref"],
                "d_rep_gain_db": operator_gains["D_rep"],
                "coverage": acted / 384,
                "selected_groups": acted,
                "selected_severe_groups": severe_count,
                "selected_hard_groups": hard_count,
                "negative_selected_groups": sum(
                    int(row["selected"]) == 2 and row["operator"] == OPERATORS[0]
                    for row in subset
                ),
            }
        )

    risk_coverage_rows = []
    for coverage in (0.10, 0.20, 0.30, 0.40, 0.60, 1.00):
        coverage_policy = []
        for fold in FOLDS:
            coverage_policy.extend(
                make_policy(
                    fold_test_units[fold],
                    fold_predictions[(fold, PRIMARY_CELL)],
                    fold,
                    PRIMARY_CELL,
                    coverage,
                )
            )
        acted, severe_count, hard_count = policy_group_counts(coverage_policy)
        for operator in OPERATORS:
            gains = [
                float(row["gain"]) for row in coverage_policy if row["operator"] == operator
            ]
            risk_coverage_rows.append(
                {
                    "coverage_target": coverage,
                    "coverage_realized": acted / 384,
                    "selected_groups": acted,
                    "operator": operator,
                    "population_mean_gain_db": float(np.mean(gains)),
                    "population_p10_gain_db": float(np.quantile(gains, 0.10)),
                    "population_cvar5_gain_db": cvar(gains),
                    "selected_severe_groups": severe_count,
                    "selected_hard_groups": hard_count,
                    "all_group_severe_ucb95_exact": exact_binomial_ucb(severe_count, 384),
                }
            )

    primary_risk_labels = [int(row["severe_label"]) for row in risk_group_rows]
    primary_risk_scores = [float(row["severe_score"]) for row in risk_group_rows]
    calibration, severe_ece, severe_brier = calibration_rows(
        primary_risk_labels, primary_risk_scores
    )
    for cell in CELLS:
        for fold in FOLDS:
            for operator in OPERATORS:
                rows = [
                    row
                    for row in ensemble_score_rows
                    if row["cell"] == cell
                    and int(row["fold"]) == fold
                    and row["operator"] == operator
                ]
                residual = np.asarray(
                    [float(row["mean_score"]) - float(row["target_gain_db"]) for row in rows]
                )
                q05_coverage = np.mean(
                    [float(row["target_gain_db"]) >= float(row["q05_score"]) for row in rows]
                )
                calibration.append(
                    {
                        "row_type": "utility_summary",
                        "cell": cell,
                        "fold": fold,
                        "operator": operator,
                        "count": len(rows),
                        "mean_bias_db": float(residual.mean()),
                        "mean_mae_db": float(np.abs(residual).mean()),
                        "q05_empirical_coverage": float(q05_coverage),
                    }
                )

    oracle_margin_rows = []
    harm_rows = []
    for fold in FOLDS:
        for operator in OPERATORS:
            subset = [
                unit
                for unit in units
                if int(unit["fold"]) == fold and unit["operator"] == operator
            ]
            margins = []
            oracle_gain = []
            positive_oracle = 0
            negative_oracle = 0
            for unit in subset:
                values = [0.0, float(unit["target"][0]), float(unit["target"][1])]
                order = sorted(range(3), key=lambda index: values[index], reverse=True)
                margins.append(values[order[0]] - values[order[1]])
                oracle_gain.append(values[order[0]])
                positive_oracle += order[0] == 1
                negative_oracle += order[0] == 2
            oracle_margin_rows.append(
                {
                    "fold": fold,
                    "operator": operator,
                    "groups": len(subset),
                    "oracle_mean_gain_db": float(np.mean(oracle_gain)),
                    "margin_mean_db": float(np.mean(margins)),
                    "margin_p10_db": float(np.quantile(margins, 0.10)),
                    "near_tie_fraction_le_0_005": float(np.mean(np.asarray(margins) <= 0.005)),
                    "positive_oracle_groups": positive_oracle,
                    "negative_oracle_groups": negative_oracle,
                }
            )
            for action_index, action in enumerate(ACTIVE_ACTIONS):
                targets = [float(unit["target"][action_index]) for unit in subset]
                harm_rows.append(
                    {
                        "fold": fold,
                        "operator": operator,
                        "action": action,
                        "rows": len(targets),
                        "mean_gain_db": float(np.mean(targets)),
                        "harm_count": sum(value < 0.0 for value in targets),
                        "severe_count": sum(value <= SEVERE_GAIN for value in targets),
                        "hard_count": sum(value <= HARD_GAIN for value in targets),
                    }
                )

    oracle_actions = {}
    severe_labels = {}
    for name, group in by_name.items():
        unit_map = {unit["operator"]: unit for unit in group}
        for operator in OPERATORS:
            values = [0.0, *[float(value) for value in unit_map[operator]["target"]]]
            oracle_actions[(name, operator)] = max(range(3), key=lambda index: values[index])
            for action in range(2):
                severe_labels[(name, operator, action)] = values[action + 1] <= SEVERE_GAIN
    action_agreement = np.mean(
        [oracle_actions[(name, OPERATORS[0])] == oracle_actions[(name, OPERATORS[1])] for name in by_name]
    )
    severe_agreement = np.mean(
        [
            severe_labels[(name, OPERATORS[0], action)]
            == severe_labels[(name, OPERATORS[1], action)]
            for name in by_name
            for action in range(2)
        ]
    )
    label_stability = {
        "schema_version": 1,
        "best_action_operator_agreement": float(action_agreement),
        "severe_label_operator_agreement": float(severe_agreement),
        "scope": "D_ref_vs_D_rep_on_frozen_three_action_development_ledger",
    }

    operator_consistency = {"schema_version": 1, "target_pearson": {}}
    for action_index, action in enumerate(ACTIVE_ACTIONS):
        first = [
            float(next(unit for unit in by_name[name] if unit["operator"] == OPERATORS[0])["target"][action_index])
            for name in sorted(by_name)
        ]
        second = [
            float(next(unit for unit in by_name[name] if unit["operator"] == OPERATORS[1])["target"][action_index])
            for name in sorted(by_name)
        ]
        operator_consistency["target_pearson"][action] = pearson(first, second)
    operator_consistency["best_action_agreement"] = float(action_agreement)

    subgroup_tail_rows = []
    for cell in (PRIMARY_CELL, POOLED_CELL):
        subset_cell = [row for row in policy_rows if row["cell"] == cell]
        for operator in OPERATORS:
            for shape in sorted({unit["shape"] for unit in test_units_all}):
                names_shape = {
                    unit["name"]
                    for unit in test_units_all
                    if unit["operator"] == operator and unit["shape"] == shape
                }
                gains = [
                    float(row["gain"])
                    for row in subset_cell
                    if row["operator"] == operator and row["name"] in names_shape
                ]
                subgroup_tail_rows.append(
                    {
                        "row_type": "native_shape_tail",
                        "cell": cell,
                        "operator": operator,
                        "shape": shape,
                        "groups": len(gains),
                        "mean_gain_db": float(np.mean(gains)),
                        "p10_gain_db": float(np.quantile(gains, 0.10)),
                        "cvar5_gain_db": cvar(gains),
                        "worst20_mean_gain_db": float(np.sort(gains)[: min(20, len(gains))].mean()),
                    }
                )
    for cell in (PRIMARY_CELL, POOLED_CELL):
        for operator in OPERATORS:
            rows = [
                row
                for row in protected_rows
                if row["cell"] == cell and row["operator"] == operator
            ]
            valid = [float(row["protected_harm_db"]) for row in rows if math.isfinite(float(row["protected_harm_db"]))]
            subgroup_tail_rows.append(
                {
                    "row_type": "protected_cell_harm",
                    "cell": cell,
                    "operator": operator,
                    "groups": len(rows),
                    "valid_groups": len(valid),
                    "protected_cells": sum(int(row["protected_cells"]) for row in rows),
                    "mean_harm_db": float(np.mean(valid)),
                }
            )

    primary_policy = [row for row in policy_rows if row["cell"] == PRIMARY_CELL]
    acted_groups, selected_severe, selected_hard = policy_group_counts(primary_policy)
    severe_ucb = exact_binomial_ucb(selected_severe, 384)
    native_shape_means = [
        float(row["mean_gain_db"])
        for row in subgroup_tail_rows
        if row["row_type"] == "native_shape_tail" and row["cell"] == PRIMARY_CELL
    ]
    protected_valid_names = {
        row["name"]
        for row in protected_rows
        if row["cell"] == PRIMARY_CELL and math.isfinite(float(row["protected_harm_db"]))
    }

    structural_checks = {
        "feature_units_complete": len(units) == FEATURE_UNITS,
        "development_groups_complete": len(by_name) == 768,
        "evaluated_groups_complete": len({row["name"] for row in primary_policy}) == 384,
        "folds_complete": sorted({int(row["fold"]) for row in policy_rows}) == list(FOLDS),
        "cells_complete": sorted({row["cell"] for row in policy_rows}) == sorted(CELLS),
        "seed_training_units_complete": completed_training == TRAINING_UNITS,
        "policy_rows_complete": len(policy_rows) == 384 * len(OPERATORS) * len(CELLS),
        "per_seed_rows_complete": len(per_seed_rows) == 384 * len(OPERATORS) * 2 * len(CELLS) * len(SEEDS),
        "fixed_coverage_each_fold_cell": all(
            policy_group_counts(
                [row for row in policy_rows if row["cell"] == cell and int(row["fold"]) == fold]
            )[0]
            == 39
            for cell in CELLS
            for fold in FOLDS
        ),
        "shuffle_dc_preserved": shuffle_dc_max <= 1.0e-5,
        "shuffle_marginals_preserved": shuffle_marginal_max == 0.0,
        "protected_group_support": len(protected_valid_names) >= 300,
        "finite_primary_metrics": all(
            math.isfinite(float(value[key]))
            for value in policy_metrics.values()
            for key in ("point", "lcb95", "ucb95")
        )
        and all(
            math.isfinite(float(value[key]))
            for value in risk_metrics.values()
            for key in ("point", "lcb95", "ucb95")
        ),
        "development_confirmation_disjoint": not development & confirmation,
        "protected_roles_untouched": not any(context.protected_data_permissions.values()),
    }
    gates = {
        "gain_point_positive": policy_metrics["gain"]["point"] > 0.0,
        "gain_ucb95_reaches_target": policy_metrics["gain"]["ucb95"] >= GAIN_TARGET,
        "retention_point_positive": policy_metrics["retention"]["point"] > 0.0,
        "retention_ucb95_reaches_target": policy_metrics["retention"]["ucb95"] >= RETENTION_TARGET,
        "spatial_minus_pooled_point_positive": policy_metrics["spatial_minus_pooled"]["point"] > 0.0,
        "spatial_minus_pooled_ucb95_reaches_target": policy_metrics["spatial_minus_pooled"]["ucb95"] >= INCREMENT_TARGET,
        "true_minus_shuffle_point_positive": policy_metrics["true_minus_shuffle"]["point"] > 0.0,
        "true_minus_shuffle_ucb95_reaches_target": policy_metrics["true_minus_shuffle"]["ucb95"] >= INCREMENT_TARGET,
        "spatial_minus_generic_point_positive": policy_metrics["spatial_minus_generic"]["point"] > 0.0,
        "severe_auroc_lcb95": risk_metrics["severe_auroc"]["lcb95"] > 0.5,
        "severe_auprc_lift_lcb95": risk_metrics["severe_auprc_lift"]["lcb95"] > 0.0,
        "coverage_exact": acted_groups == 78,
        "selected_severe_groups_zero": selected_severe == 0,
        "selected_hard_groups_zero": selected_hard == 0,
        "all_group_severe_ucb95": severe_ucb <= SEVERE_RATE_UCB_TARGET,
        "cvar5_spatial_minus_pooled_lcb95": policy_metrics["cvar5_spatial_minus_pooled"]["lcb95"] >= TAIL_MARGIN,
        "protected_harm_increment_ucb95": protected_metric["ucb95"] <= PROTECTED_HARM_MARGIN,
        "native_shape_operator_mean_floor": min(native_shape_means) >= SHAPE_MEAN_FLOOR,
    }
    structural_valid = all(structural_checks.values())
    survives = structural_valid and all(gates.values())
    decisive_fail = structural_valid and (
        policy_metrics["gain"]["ucb95"] < GAIN_TARGET
        or policy_metrics["retention"]["ucb95"] < RETENTION_TARGET
        or policy_metrics["spatial_minus_pooled"]["ucb95"] < INCREMENT_TARGET
        or policy_metrics["true_minus_shuffle"]["ucb95"] < INCREMENT_TARGET
        or policy_metrics["spatial_minus_generic"]["ucb95"] <= 0.0
        or risk_metrics["severe_auroc"]["ucb95"] <= 0.5
        or risk_metrics["severe_auprc_lift"]["ucb95"] <= 0.0
        or not all(
            gates[key]
            for key in (
                "coverage_exact",
                "selected_severe_groups_zero",
                "selected_hard_groups_zero",
                "all_group_severe_ucb95",
                "cvar5_spatial_minus_pooled_lcb95",
                "protected_harm_increment_ucb95",
                "native_shape_operator_mean_floor",
            )
        )
    )
    if survives:
        state = "COMPLETED_GATE_PASS"
        decision = "R5_A0_SPATIAL_RESPONSE_NONFUTILITY_PASS"
        authorizes = "R5_A1_FULL_OOF_CONTRACT_REVIEW_ONLY"
    elif decisive_fail:
        state = "COMPLETED_GATE_FAIL"
        decision = "R5_A0_SPATIAL_RESPONSE_FUTILITY_OR_SAFETY_FAIL_STOP"
        authorizes = "NONE"
    else:
        state = "COMPLETED_GATE_INCONCLUSIVE"
        decision = "R5_A0_SPATIAL_RESPONSE_INCONCLUSIVE_STOP"
        authorizes = "NONE"

    contract_summary = {
        "schema_version": 1,
        "route_id": ROUTE_ID,
        "operation_id": OPERATION_ID,
        "actions": list(ACTIONS),
        "operators": list(OPERATORS),
        "cells": list(CELLS),
        "folds": list(FOLDS),
        "seeds": list(SEEDS),
        "grid": GRID,
        "input_dim": INPUT_DIM,
        "hidden_dim": HIDDEN_DIM,
        "epochs": EPOCHS,
        "optimizer": "AdamW",
        "coverage_per_fold": COVERAGE,
        "selected_groups_per_fold": 39,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }
    provenance = {
        "schema_version": 1,
        "route_commit": context.route_commit,
        "ledger_sha256": sha256_file(ledger_path),
        "cache_manifest_sha256": sha256_file(cache_manifest_path),
        "raw_manifest_sha256": sha256_file(raw_manifest_path),
        "cache_manifest_internal_identity": cache_manifest.get("cache_manifest_sha256"),
        "cache_units_verified": len(units),
        "development_images_targets_accessed": 768,
        "evaluated_development_images": 384,
        "confirmation_images_targets_outcomes_touched": False,
        "historical_a1x_432_outcomes_touched": False,
        "canary_touched": False,
        "locked_test_touched": False,
        "official_checkpoint_loaded": False,
        "candidate_generation_rerun": False,
    }
    representation_identity = {
        "schema_version": 1,
        "grid": [GRID, GRID],
        "transform": "orthonormal_dct_ii_full_63_non_dc_per_rgb_channel",
        "response": "active_render_minus_reference_noop_render",
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "response_dim": RESPONSE_DIM,
        "generic_spatial_dim": GENERIC_SPATIAL_DIM,
        "input_dim": INPUT_DIM,
        "cell_nonzero_dimensions": {cell: int(cell_mask(cell).sum()) for cell in CELLS},
        "shuffle_dc_max_abs": shuffle_dc_max,
        "shuffle_marginal_max_abs": shuffle_marginal_max,
        "normalizer": "outer_train_true_features_shared_then_mask",
    }
    resource_summary = {
        "schema_version": 1,
        "wall_seconds": time.perf_counter() - started_wall,
        "cpu_seconds": time.process_time() - started_cpu,
        "max_rss_mib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0,
        "feature_units": len(units),
        "seed_model_training_units": completed_training,
        "model_parameters": sum(parameter.numel() for parameter in build_model(SEEDS[0]).parameters()),
        "gpu_used": False,
    }
    risk_metrics["severe_ece"] = severe_ece
    risk_metrics["severe_brier"] = severe_brier
    policy_metrics.update(risk_metrics)
    atomic_json(context.phase_output_path / "r5_a0_contract_summary.json", contract_summary)
    atomic_json(context.phase_output_path / "r5_a0_provenance_and_access.json", provenance)
    atomic_json(context.phase_output_path / "r5_a0_representation_identity.json", representation_identity)
    atomic_json(
        context.phase_output_path / "r5_a0_structural_summary.json",
        {"schema_version": 1, "checks": structural_checks, "valid": structural_valid},
    )
    write_csv(context.phase_output_path / "r5_a0_oracle_margin_summary.csv", oracle_margin_rows)
    atomic_json(context.phase_output_path / "r5_a0_label_stability.json", label_stability)
    write_csv(context.phase_output_path / "r5_a0_harm_prevalence.csv", harm_rows)
    atomic_json(context.phase_output_path / "r5_a0_operator_consistency.json", operator_consistency)
    write_csv(context.phase_output_path / "r5_a0_fold_seed_stability.csv", fold_seed_rows)
    write_csv(context.phase_output_path / "r5_a0_cell_summary.csv", cell_summary_rows)
    write_csv(context.phase_output_path / "r5_a0_risk_coverage.csv", risk_coverage_rows)
    write_csv(context.phase_output_path / "r5_a0_calibration_summary.csv", calibration)
    write_csv(context.phase_output_path / "r5_a0_subgroup_tail_summary.csv", subgroup_tail_rows)
    atomic_json(context.phase_output_path / "r5_a0_bootstrap_summary.json", {"schema_version": 1, **policy_metrics})
    atomic_json(
        context.phase_output_path / "r5_a0_gate_summary.json",
        {
            "schema_version": 1,
            "gates": gates,
            "survives": survives,
            "decisive_fail": decisive_fail,
            "selected_severe_groups": selected_severe,
            "selected_hard_groups": selected_hard,
            "all_group_severe_ucb95_exact": severe_ucb,
        },
    )
    atomic_json(context.phase_output_path / "r5_a0_resource_summary.json", resource_summary)
    write_run_result(
        context,
        state=state,
        decision=decision,
        authorizes=authorizes,
        details={
            "survives": survives,
            "structural_valid": structural_valid,
            "gain_ucb95_db": policy_metrics["gain"]["ucb95"],
            "retention_ucb95": policy_metrics["retention"]["ucb95"],
            "spatial_minus_pooled_ucb95_db": policy_metrics["spatial_minus_pooled"]["ucb95"],
            "true_minus_shuffle_ucb95_db": policy_metrics["true_minus_shuffle"]["ucb95"],
            "spatial_minus_generic_point_db": policy_metrics["spatial_minus_generic"]["point"],
            "selected_severe_groups": selected_severe,
            "selected_hard_groups": selected_hard,
            "protected_harm_increment_ucb95_db": protected_metric["ucb95"],
        },
    )


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
