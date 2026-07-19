#!/usr/bin/env python3
"""Frozen R13 image-relative candidate-context OOF mechanism screen."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
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

ROUTE_ID = "haze4k_v5_r13_image_relative_context_observability_20260719"
OPERATION_ID = "R13_A0_IMAGE_RELATIVE_CONTEXT_OBSERVABILITY_SCREEN"
R11_ROUTE_ID = "haze4k_v5_r11_regional_action_observability_20260719"
R11_RUN_ID = "r11-a0-regional-observability-r1"
R11_COMMIT = "c183817e2b3befdeeb12278aa6e6a0574883b6d5"
R11_DECISION = "R11_A0_REGIONAL_OBSERVABILITY_FAIL_STOP"
R11_ENTRYPOINT = "experience_docx/tools/r11_a0_regional_action_observability.py"
CELLS = (
    "C1_ALIGNED_RELATIVE_CONTEXT",
    "C0_LOCAL_ZERO_PAD",
    "G1_GENERIC_NONACTION_CONTEXT",
    "S1_WITHIN_IMAGE_CONTEXT_SHUFFLE",
    "I1_FIXED_ACTION_ID_PERMUTATION",
)
PRIMARY = CELLS[0]
CONTROLS = CELLS[1:]
OPERATORS = ("D_ref", "D_rep")
FOLDS = (0, 1)
SEEDS = (3407, 3411)
GRID = 8
TILES = 64
LOCAL_DIM = 303
CONTEXT_DIM = 108
FEATURE_DIM = LOCAL_DIM + CONTEXT_DIM
EPOCHS = 24
BATCH_SIZE = 256
BOOTSTRAP_DRAWS = 4000
BOOTSTRAP_SEED = 3407
ABSOLUTE_GAIN = 0.020
SPECIFIC_INCREMENT = 0.010
RETENTION_MIN = 0.25
TAIL_MARGIN = -0.005
SEED_RANGE_MAX = 0.020
SEVERE_GAIN = -0.2
HARD_GAIN = -0.5
EXPECTED_NAMES = 384
EXPECTED_UNITS = 768
EXPECTED_ROWS = 49152
EXPECTED_R11_SOURCE_SHA256 = "9c60abd5ac36adf51d505b5686ca56169783f5f34856cde2f20e472d48c8a724"
ENVIRONMENT = {
    "CONVIR_ROUTE_BOOTSTRAP_DRAWS": "4000",
    "CONVIR_ROUTE_BOOTSTRAP_SEED": "3407",
    "CONVIR_ROUTE_GRID": "8",
    "CONVIR_ROUTE_SEEDS": "3407,3411",
    "CONVIR_ROUTE_EPOCHS": "24",
    "CONVIR_ROUTE_BATCH_SIZE": "256",
    "CONVIR_ROUTE_CONTEXT_DIM": "108",
    "CONVIR_ROUTE_LOCAL_GAIN_DB": "0.005",
    "CONVIR_ROUTE_SEVERE_GAIN_DB": "-0.2",
    "CONVIR_ROUTE_HARD_GAIN_DB": "-0.5",
}


class ContextInconclusive(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContextInconclusive(f"JSON object required: {path.name}")
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ContextInconclusive(f"empty CSV refused: {path.name}")
    fields = sorted({key for row in rows for key in row})
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_r11_module(checkout: Path) -> Any:
    source = checkout / R11_ENTRYPOINT
    if not source.is_file():
        raise ContextInconclusive("R11 entrypoint missing from source checkout")
    digest = sha256_file(source)
    if digest != EXPECTED_R11_SOURCE_SHA256:
        raise ContextInconclusive(f"R11 source SHA mismatch: {digest}")
    spec = importlib.util.spec_from_file_location("r13_frozen_r11_source", source)
    if spec is None or spec.loader is None:
        raise ContextInconclusive("cannot bind R11 source module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_environment() -> None:
    mismatches = sorted(key for key, value in ENVIRONMENT.items() if os.environ.get(key) != value)
    if mismatches:
        raise ContextInconclusive(f"frozen environment mismatch: {mismatches}")


def percentile_midrank(values: Any) -> Any:
    import numpy as np
    from scipy.stats import rankdata

    array = np.asarray(values, dtype=np.float64)
    result = np.empty_like(array, dtype=np.float64)
    for channel in range(array.shape[1]):
        ranks = rankdata(array[:, channel], method="average")
        result[:, channel] = 2.0 * (ranks - 1.0) / (len(ranks) - 1.0) - 1.0
    return result.astype(np.float32)


def relation_block(current: Any, opposite: Any) -> Any:
    import numpy as np

    current = np.asarray(current, dtype=np.float64)
    opposite = np.asarray(opposite, dtype=np.float64)
    ranks = percentile_midrank(current)
    opposite_ranks = percentile_midrank(opposite)
    median = np.quantile(current, 0.50, axis=0, method="linear")
    q25 = np.quantile(current, 0.25, axis=0, method="linear")
    q75 = np.quantile(current, 0.75, axis=0, method="linear")
    robust = ((current - median) / np.maximum(q75 - q25, 1.0e-6)).astype(np.float32)
    quantiles = np.concatenate([
        np.quantile(current, level, axis=0, method="linear") for level in (0.10, 0.50, 0.90)
    ]).astype(np.float32)
    broadcast = np.broadcast_to(quantiles, (TILES, len(quantiles))).copy()
    block = np.concatenate((ranks, robust, ranks - opposite_ranks, broadcast), axis=1)
    if block.shape != (TILES, CONTEXT_DIM) or not np.isfinite(block).all():
        raise ContextInconclusive("relation block shape/finite contract failed")
    return block.astype(np.float32, copy=False)


def relation_permutation(name: str, action: int, pixel_counts: Any) -> Any:
    import numpy as np

    counts = np.asarray(pixel_counts, dtype=np.int64)
    result = np.arange(TILES, dtype=np.int64)
    digest = hashlib.sha256(f"{ROUTE_ID}|{name}|action={action}|context-shuffle".encode()).digest()
    generator = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    for count in sorted(np.unique(counts)):
        indices = np.flatnonzero(counts == count)
        result[indices] = generator.permutation(indices)
    return result


def build_group_features(group: dict[str, Any], r11: Any) -> dict[str, Any]:
    import numpy as np

    local = np.asarray(group["features"][r11.PRIMARY_CELL], dtype=np.float32)
    if local.shape != (TILES * 2, LOCAL_DIM):
        raise ContextInconclusive("frozen R11 local feature shape mismatch")
    responses = np.stack((local[:TILES, 180:198], local[TILES:, 180:198]))
    for action in range(2):
        observed = responses[action].mean(axis=0, dtype=np.float64)
        expected = local[action * TILES:(action + 1) * TILES, 282:300][0]
        if float(np.max(np.abs(observed - expected))) > 1.0e-6:
            raise ContextInconclusive("R11 central-response extraction identity failed")
    aligned = [relation_block(responses[action], responses[1 - action]) for action in range(2)]
    generic_map = 0.5 * (np.abs(responses[0]) + np.abs(responses[1]))
    generic = relation_block(generic_map, generic_map)
    blocks: dict[str, list[Any]] = {cell: [] for cell in CELLS}
    for action in range(2):
        blocks[PRIMARY].append(aligned[action])
        blocks["C0_LOCAL_ZERO_PAD"].append(np.zeros((TILES, CONTEXT_DIM), dtype=np.float32))
        blocks["G1_GENERIC_NONACTION_CONTEXT"].append(generic)
        blocks["S1_WITHIN_IMAGE_CONTEXT_SHUFFLE"].append(
            aligned[action][relation_permutation(group["name"], action + 1, group["pixel_counts"])]
        )
        blocks["I1_FIXED_ACTION_ID_PERMUTATION"].append(aligned[1 - action])
    result = {}
    for cell in CELLS:
        context = np.concatenate(blocks[cell], axis=0)
        values = np.concatenate((local, context), axis=1).astype(np.float32)
        if values.shape != (TILES * 2, FEATURE_DIM) or not np.isfinite(values).all():
            raise ContextInconclusive(f"feature contract failed: {cell}")
        result[cell] = values
    return result


def train_model(features: Any, mean_target: Any, worst_target: Any, seed: int, epochs: int) -> tuple[Any, dict[str, float]]:
    import numpy as np
    import torch

    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    x = torch.as_tensor(features, dtype=torch.float32)
    mean_y = torch.as_tensor(mean_target, dtype=torch.float32)
    worst_y = torch.as_tensor(worst_target, dtype=torch.float32)
    model = torch.nn.Sequential(torch.nn.Linear(FEATURE_DIM, 64), torch.nn.ReLU(), torch.nn.Linear(64, 2))
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight, generator=generator)
            torch.nn.init.zeros_(module.bias)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-4)

    def full_loss() -> float:
        total = 0.0
        with torch.no_grad():
            for start in range(0, len(x), 4096):
                indices = torch.arange(start, min(start + 4096, len(x)))
                prediction = model(x[indices])
                loss = torch.nn.functional.huber_loss(prediction[:, 0], mean_y[indices]) \
                    + torch.nn.functional.huber_loss(prediction[:, 1], worst_y[indices])
                total += float(loss) * len(indices)
        return total / len(x)

    initial = full_loss()
    for epoch in range(epochs):
        order = torch.randperm(len(x), generator=torch.Generator().manual_seed(seed + epoch))
        for start in range(0, len(x), BATCH_SIZE):
            indices = order[start:start + BATCH_SIZE]
            prediction = model(x[indices])
            loss = torch.nn.functional.huber_loss(prediction[:, 0], mean_y[indices]) \
                + torch.nn.functional.huber_loss(prediction[:, 1], worst_y[indices])
            if not bool(torch.isfinite(loss)):
                raise ContextInconclusive("non-finite probe loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    final = full_loss()
    if not math.isfinite(initial) or not math.isfinite(final):
        raise ContextInconclusive("non-finite training summary")
    return model.eval(), {"initial_loss": initial, "final_loss": final}


def predict_model(model: Any, features: Any) -> Any:
    import numpy as np
    import torch

    outputs = []
    tensor = torch.as_tensor(features, dtype=torch.float32)
    with torch.no_grad():
        for start in range(0, len(tensor), 4096):
            outputs.append(model(tensor[start:start + 4096]).cpu().numpy())
    values = np.concatenate(outputs, axis=0)
    if values.shape != (len(features), 2) or not np.isfinite(values).all():
        raise ContextInconclusive("prediction contract failed")
    return values


def cvar(values: Any, fraction: float = 0.05) -> float:
    import numpy as np

    array = np.sort(np.asarray(values, dtype=np.float64))
    return float(array[:max(1, math.ceil(fraction * len(array)))].mean())


def evaluate_rows(rows: list[dict[str, Any]], indices: Any) -> dict[str, float]:
    import numpy as np

    result: dict[str, float] = {}
    means: dict[str, dict[str, float]] = {}
    for cell in CELLS:
        means[cell] = {}
        for operator in OPERATORS:
            values = np.asarray([row[f"{cell}_{operator}"] for row in rows], dtype=np.float64)[indices]
            means[cell][operator] = float(values.mean())
            result[f"{cell}_{operator}"] = means[cell][operator]
        result[f"{cell}_gain"] = min(means[cell].values())
    oracle = {operator: float(np.mean(np.asarray([row[f"oracle_{operator}"] for row in rows])[indices])) for operator in OPERATORS}
    global_mean = {operator: float(np.mean(np.asarray([row[f"global_{operator}"] for row in rows])[indices])) for operator in OPERATORS}
    result["oracle_gain"] = min(oracle.values())
    result["global_gain"] = min(global_mean.values())
    result["retention"] = min(means[PRIMARY][operator] / oracle[operator] for operator in OPERATORS)
    result["delta_specific"] = min(
        means[PRIMARY][operator] - max(means[control][operator] for control in CONTROLS)
        for operator in OPERATORS
    )
    for control in CONTROLS:
        result[f"primary_minus_{control}"] = min(
            means[PRIMARY][operator] - means[control][operator] for operator in OPERATORS
        )
    result["primary_minus_global"] = min(
        means[PRIMARY][operator] - global_mean[operator] for operator in OPERATORS
    )
    result["primary_minus_global_cvar5"] = min(
        cvar(np.asarray([row[f"{PRIMARY}_{operator}"] for row in rows])[indices])
        - cvar(np.asarray([row[f"global_{operator}"] for row in rows])[indices])
        for operator in OPERATORS
    )
    return result


def interval(point: float, samples: list[float]) -> dict[str, float]:
    import numpy as np

    values = np.asarray(samples, dtype=np.float64)
    if not math.isfinite(point) or not np.isfinite(values).all():
        raise ContextInconclusive("non-finite bootstrap result")
    return {
        "point": float(point),
        "lcb95": float(np.quantile(values, 0.025)),
        "ucb95": float(np.quantile(values, 0.975)),
    }


def bootstrap(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    import numpy as np

    point = evaluate_rows(rows, np.arange(len(rows)))
    samples = {key: [] for key in point}
    fold_indices = {fold: np.asarray([index for index, row in enumerate(rows) if row["fold"] == fold]) for fold in FOLDS}
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    for _draw in range(BOOTSTRAP_DRAWS):
        indices = np.concatenate([generator.choice(value, len(value), replace=True) for value in fold_indices.values()])
        values = evaluate_rows(rows, indices)
        for key in samples:
            samples[key].append(values[key])
    return {key: interval(point[key], values) for key, values in samples.items()}


def synthetic_contract() -> dict[str, bool]:
    import numpy as np

    started = time.perf_counter()
    generator = np.random.default_rng(3407)
    features = {cell: [] for cell in CELLS}
    for image in range(EXPECTED_NAMES):
        local = generator.normal(size=(TILES * 2, LOCAL_DIM)).astype(np.float32)
        response = generator.normal(size=(2, TILES, 18)).astype(np.float32)
        aligned = [relation_block(response[action], response[1 - action]) for action in range(2)]
        generic_map = 0.5 * (np.abs(response[0]) + np.abs(response[1]))
        generic = relation_block(generic_map, generic_map)
        for action in range(2):
            features[PRIMARY].append(np.concatenate((local[action*TILES:(action+1)*TILES], aligned[action]), axis=1))
            features["C0_LOCAL_ZERO_PAD"].append(np.concatenate((local[action*TILES:(action+1)*TILES], np.zeros((TILES, CONTEXT_DIM), dtype=np.float32)), axis=1))
            features["G1_GENERIC_NONACTION_CONTEXT"].append(np.concatenate((local[action*TILES:(action+1)*TILES], generic), axis=1))
            permutation = relation_permutation(f"synthetic_{image:04d}", action + 1, np.full(TILES, 3072))
            features["S1_WITHIN_IMAGE_CONTEXT_SHUFFLE"].append(np.concatenate((local[action*TILES:(action+1)*TILES], aligned[action][permutation]), axis=1))
            features["I1_FIXED_ACTION_ID_PERMUTATION"].append(np.concatenate((local[action*TILES:(action+1)*TILES], aligned[1-action]), axis=1))
    arrays = {cell: np.concatenate(parts).astype(np.float32) for cell, parts in features.items()}
    target_mean = (0.08 * arrays[PRIMARY][:, 303] - 0.03 * arrays[PRIMARY][:, 339]).astype(np.float32)
    target_worst = (target_mean - 0.02).astype(np.float32)
    train = np.arange(EXPECTED_ROWS // 2)
    mean = arrays[PRIMARY][train].mean(0, dtype=np.float64)
    std = arrays[PRIMARY][train].std(0, dtype=np.float64)
    std = np.where(std >= 1.0e-6, std, 1.0)
    finite = True
    decreased = True
    for cell in CELLS:
        x = ((arrays[cell][train] - mean) / std).astype(np.float32)
        for seed in SEEDS:
            model, summary = train_model(x, target_mean[train], target_worst[train], seed, epochs=1)
            prediction = predict_model(model, x[:256])
            finite = finite and bool(np.isfinite(prediction).all())
            decreased = decreased and summary["final_loss"] < summary["initial_loss"]
    rows = []
    for image in range(EXPECTED_NAMES):
        row: dict[str, Any] = {"fold": image % 2}
        for cell_index, cell in enumerate(CELLS):
            for operator_index, operator in enumerate(OPERATORS):
                row[f"{cell}_{operator}"] = 0.08 - 0.004 * cell_index - 0.001 * operator_index
        row.update({"oracle_D_ref": 0.24, "oracle_D_rep": 0.239, "global_D_ref": 0.10, "global_D_rep": 0.099})
        rows.append(row)
    boot = bootstrap(rows)
    elapsed = time.perf_counter() - started
    return {
        "formal_rows_complete": all(value.shape == (EXPECTED_ROWS, FEATURE_DIM) for value in arrays.values()),
        "all_cells_finite": all(np.isfinite(value).all() for value in arrays.values()),
        "production_fit_finite": finite,
        "production_loss_decreased": decreased,
        "bootstrap_complete": len(boot) >= 10 and all(math.isfinite(value["point"]) for value in boot.values()),
        "formal_scale_wall_bound": elapsed <= 240.0,
        "formal_scale_memory_bound": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0 <= 3072.0,
    }


def contract(context_path: Path) -> None:
    context = load_context(context_path, "contract")
    prepare_phase_output(context)
    checks = synthetic_contract()
    write_contract_result(context, checks=checks)


def conclusion(state: str, decision: str, authorizes: str, boot: dict[str, Any], gates: dict[str, bool], severe: int, hard: int) -> dict[str, Any]:
    primary = boot[f"{PRIMARY}_gain"]
    delta = boot["delta_specific"]
    if state == "COMPLETED_GATE_PASS":
        interpretation = "The frozen aligned image-relative relation block adds material, control-specific regional signed utility and satisfies the privileged-budget image-level tail gates, supporting H2 for this exact transform."
    elif decision == "R13_A0_RELATIVE_CONTEXT_SAFETY_FAIL_STOP":
        interpretation = "The aligned relation block satisfies the frozen utility-specificity gates but fails image-level tail safety, supporting utility-risk separation without authorizing a context mechanism route."
    else:
        interpretation = "The frozen aligned relation block does not establish material utility beyond the strongest matched control, supporting H1 for this exact response family and transform."
    return {
        "schema_version": 1, "route_id": ROUTE_ID, "operation_id": OPERATION_ID,
        "run_id": "r13-a0-image-relative-context-r1", "state": state,
        "decision": decision, "authorizes": authorizes,
        "primary_result": f"Aligned gain {primary['point']:.10f} dB (LCB95 {primary['lcb95']:.10f}); delta-specific {delta['point']:.10f} dB (LCB95 {delta['lcb95']:.10f}, UCB95 {delta['ucb95']:.10f}); severe/hard {severe}/{hard}.",
        "gate_reasons": [f"{key}={value}" for key, value in sorted(gates.items())],
        "competing_explanation": interpretation,
        "limitations": [
            "Development-only OOF mechanism screen on the reused 384 R11 groups.",
            "The exact R10 action-count budget is outcome-derived and nondeployable.",
            "The conclusion applies only to the single frozen relation transform and matched readout.",
            "R12 already rejects a readout-only downside rescue; R13 does not establish external-domain safety.",
            "Confirmation, canary and locked test remain untouched.",
        ],
    }


def write_inconclusive(context: Any, reason: str, started_wall: float, started_cpu: float) -> None:
    common = {"schema_version": 1, "status": "inconclusive", "reason": reason}
    for name in ("r13_a0_contract_summary.json", "r13_a0_provenance_and_access.json", "r13_a0_input_identity.json", "r13_a0_representation_identity.json", "r13_a0_bootstrap_summary.json", "r13_a0_operator_consistency.json", "r13_a0_gate_summary.json"):
        atomic_json(context.phase_output_path / name, common)
    for name in ("r13_a0_cell_summary.csv", "r13_a0_fold_seed_stability.csv"):
        write_csv(context.phase_output_path / name, [{"status": "inconclusive", "reason": reason}])
    atomic_json(context.phase_output_path / "r13_a0_resource_summary.json", {**common, "wall_seconds": time.perf_counter()-started_wall, "cpu_seconds": time.process_time()-started_cpu, "gpu_used": False})
    atomic_json(context.phase_output_path / "r13_a0_scientific_conclusion.json", {
        "schema_version": 1, "route_id": ROUTE_ID, "operation_id": OPERATION_ID,
        "run_id": context.run_id, "state": "COMPLETED_GATE_INCONCLUSIVE",
        "decision": "R13_A0_INPUT_OR_CONTEXT_INCONCLUSIVE_STOP", "authorizes": "NONE",
        "primary_result": reason, "gate_reasons": [reason],
        "competing_explanation": "No scientific mechanism conclusion is valid.",
        "limitations": ["Input or numerical integrity failed before a valid decision."],
    })
    write_run_result(context, state="COMPLETED_GATE_INCONCLUSIVE", decision="R13_A0_INPUT_OR_CONTEXT_INCONCLUSIVE_STOP", authorizes="NONE", details={"reason": reason})


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
        r11_closeout = read_json(asset_path(context, "r11_closeout", kind="file"))
        if (r11_closeout.get("route_id"), r11_closeout.get("run_id"), r11_closeout.get("state"), r11_closeout.get("decision"), r11_closeout.get("authorizes")) != (R11_ROUTE_ID, R11_RUN_ID, "COMPLETED_GATE_FAIL", R11_DECISION, "NONE"):
            raise ContextInconclusive("terminal R11 identity mismatch")
        checkout = asset_path(context, "r11_source_checkout", kind="git_checkout")
        r11 = load_r11_module(checkout)
        groups, input_metadata = r11.load_formal_dataset(context)
        if len(groups) != EXPECTED_NAMES or input_metadata.get("evaluated_units") != EXPECTED_UNITS or input_metadata.get("tile_action_rows") != EXPECTED_ROWS:
            raise ContextInconclusive("formal R11 population mismatch")
        if input_metadata.get("target_mismatches") != 0 or input_metadata.get("no_op_render_mismatches") != 0:
            raise ContextInconclusive("R11 replay identity mismatch")

        feature_arrays = {cell: [] for cell in CELLS}
        for group in groups:
            values = build_group_features(group, r11)
            for cell in CELLS:
                feature_arrays[cell].append(values[cell])
        feature_arrays = {cell: np.concatenate(parts) for cell, parts in feature_arrays.items()}
        mean_target = np.concatenate([group["mean_target"] for group in groups])
        worst_target = np.concatenate([group["worst_target"] for group in groups])
        row_groups = np.concatenate([np.full(TILES*2, index, dtype=np.int64) for index in range(len(groups))])
        predictions: dict[str, dict[int, dict[int, Any]]] = {cell: {} for cell in CELLS}
        training_rows = []
        for test_fold in FOLDS:
            train_groups = np.asarray([index for index, group in enumerate(groups) if group["fold"] != test_fold])
            test_groups = np.asarray([index for index, group in enumerate(groups) if group["fold"] == test_fold])
            if set(train_groups.tolist()) & set(test_groups.tolist()):
                raise ContextInconclusive("OOF image overlap")
            train_indices = np.flatnonzero(np.isin(row_groups, train_groups))
            test_indices = np.flatnonzero(np.isin(row_groups, test_groups))
            mean = feature_arrays[PRIMARY][train_indices].mean(0, dtype=np.float64)
            std = feature_arrays[PRIMARY][train_indices].std(0, dtype=np.float64)
            std = np.where(std >= 1.0e-6, std, 1.0)
            utility_mean = float(mean_target[train_indices].mean())
            utility_std = max(float(mean_target[train_indices].std()), 1.0e-6)
            y_mean = (mean_target[train_indices] - utility_mean) / utility_std
            y_worst = (worst_target[train_indices] - utility_mean) / utility_std
            for cell in CELLS:
                x_train = ((feature_arrays[cell][train_indices] - mean) / std).astype(np.float32)
                x_test = ((feature_arrays[cell][test_indices] - mean) / std).astype(np.float32)
                predictions[cell][test_fold] = {}
                for seed in SEEDS:
                    model, summary = train_model(x_train, y_mean, y_worst, seed, EPOCHS)
                    values = predict_model(model, x_test)
                    values = values * utility_std + utility_mean
                    predictions[cell][test_fold][seed] = values
                    training_rows.append({"record_type": "training", "cell": cell, "test_fold": test_fold, "seed": seed, **summary})
                    write_workload_progress(context, completed_units=771+len(training_rows)-1, stage="oof_probe_models")

        evaluation_rows = []
        seed_rows = []
        for test_fold in FOLDS:
            fold_groups = [index for index, group in enumerate(groups) if group["fold"] == test_fold]
            for local_index, group_index in enumerate(fold_groups):
                group = groups[group_index]
                oracle_gain = r11.replay_gain(group["sse"], group["oracle_map"])
                global_map = np.full(TILES, group["global_action"], dtype=np.int64)
                global_gain = r11.replay_gain(group["sse"], global_map)
                row: dict[str, Any] = {"name": group["name"], "fold": test_fold, **{f"oracle_{key}": value for key, value in oracle_gain.items()}, **{f"global_{key}": value for key, value in global_gain.items()}}
                for cell in CELLS:
                    seed_scores = []
                    for seed in SEEDS:
                        start = local_index*TILES*2
                        values = predictions[cell][test_fold][seed][start:start+TILES*2]
                        seed_scores.append(np.stack((values[:TILES, 1], values[TILES:, 1]), axis=1))
                    ensemble = np.mean(seed_scores, axis=0)
                    action_map = r11.assign_oracle_budget(ensemble, group["oracle_map"], group["pixel_counts"])
                    gains = r11.replay_gain(group["sse"], action_map)
                    for operator, value in gains.items():
                        row[f"{cell}_{operator}"] = value
                    if cell == PRIMARY:
                        for seed, scores in zip(SEEDS, seed_scores):
                            seed_map = r11.assign_oracle_budget(scores, group["oracle_map"], group["pixel_counts"])
                            seed_rows.append({"name": group["name"], "fold": test_fold, "seed": seed, **r11.replay_gain(group["sse"], seed_map)})
                evaluation_rows.append(row)
        boot = bootstrap(evaluation_rows)
        write_workload_progress(context, completed_units=4790, stage="bootstrap_complete")
        severe = sum(any(row[f"{PRIMARY}_{operator}"] <= SEVERE_GAIN for operator in OPERATORS) for row in evaluation_rows)
        hard = sum(any(row[f"{PRIMARY}_{operator}"] <= HARD_GAIN for operator in OPERATORS) for row in evaluation_rows)
        fold_metrics = {str(fold): evaluate_rows(evaluation_rows, np.asarray([index for index, row in enumerate(evaluation_rows) if row["fold"] == fold])) for fold in FOLDS}
        both_folds_specific = all(fold_metrics[str(fold)]["delta_specific"] > 0.0 for fold in FOLDS)
        replay_rows = []
        pooled_seed_gains = []
        fold_seed_gains = []
        for seed in SEEDS:
            for fold in (*FOLDS, "pooled"):
                subset = [row for row in seed_rows if row["seed"] == seed and (fold == "pooled" or row["fold"] == fold)]
                means = {operator: float(np.mean([row[operator] for row in subset])) for operator in OPERATORS}
                gain = min(means.values())
                replay_rows.append({"record_type": "replay", "seed": seed, "test_fold": fold, "groups": len(subset), "d_ref_gain_db": means["D_ref"], "d_rep_gain_db": means["D_rep"], "worse_operator_gain_db": gain})
                (pooled_seed_gains if fold == "pooled" else fold_seed_gains).append(gain)
        seed_stability = min(fold_seed_gains) >= 0.0 and max(pooled_seed_gains)-min(pooled_seed_gains) <= SEED_RANGE_MAX
        gates = {
            "delta_specific_lcb95": boot["delta_specific"]["lcb95"] > SPECIFIC_INCREMENT,
            "primary_gain_lcb95": boot[f"{PRIMARY}_gain"]["lcb95"] >= ABSOLUTE_GAIN,
            "oracle_retention_lcb95": boot["retention"]["lcb95"] >= RETENTION_MIN,
            "both_folds_specific": both_folds_specific, "seed_stability": seed_stability,
            "primary_minus_global_cvar5_lcb95": boot["primary_minus_global_cvar5"]["lcb95"] >= TAIL_MARGIN,
            "zero_primary_severe": severe == 0, "zero_primary_hard": hard == 0,
        }
        utility_keys = ("delta_specific_lcb95", "primary_gain_lcb95", "oracle_retention_lcb95", "both_folds_specific", "seed_stability")
        safety_keys = ("primary_minus_global_cvar5_lcb95", "zero_primary_severe", "zero_primary_hard")
        utility_pass = all(gates[key] for key in utility_keys)
        safety_pass = all(gates[key] for key in safety_keys)
        decisive_utility = boot["delta_specific"]["ucb95"] <= SPECIFIC_INCREMENT or boot[f"{PRIMARY}_gain"]["ucb95"] < ABSOLUTE_GAIN or boot["retention"]["ucb95"] < RETENTION_MIN or not both_folds_specific or not seed_stability
        if utility_pass and safety_pass:
            state, decision, authorizes = "COMPLETED_GATE_PASS", "R13_A0_IMAGE_RELATIVE_CONTEXT_PASS", "R13_FIXED_CONTEXT_MECHANISM_CONTRACT_REVIEW_ONLY"
        elif utility_pass and not safety_pass:
            state, decision, authorizes = "COMPLETED_GATE_FAIL", "R13_A0_RELATIVE_CONTEXT_SAFETY_FAIL_STOP", "NONE"
        elif decisive_utility:
            state, decision, authorizes = "COMPLETED_GATE_FAIL", "R13_A0_RELATIVE_CONTEXT_UTILITY_FAIL_STOP", "NONE"
        else:
            state, decision, authorizes = "COMPLETED_GATE_INCONCLUSIVE", "R13_A0_INPUT_OR_CONTEXT_INCONCLUSIVE_STOP", "NONE"

        cell_rows = []
        for cell in CELLS:
            means = {operator: float(np.mean([row[f"{cell}_{operator}"] for row in evaluation_rows])) for operator in OPERATORS}
            cell_rows.append({"cell": cell, "groups": len(evaluation_rows), "d_ref_gain_db": means["D_ref"], "d_rep_gain_db": means["D_rep"], "worse_operator_gain_db": min(means.values()), "selected_severe_groups": sum(any(row[f"{cell}_{operator}"] <= SEVERE_GAIN for operator in OPERATORS) for row in evaluation_rows), "selected_hard_groups": sum(any(row[f"{cell}_{operator}"] <= HARD_GAIN for operator in OPERATORS) for row in evaluation_rows)})
        atomic_json(context.phase_output_path / "r13_a0_contract_summary.json", {"schema_version":1, "route_id":ROUTE_ID, "operation_id":OPERATION_ID, "cells":list(CELLS), "folds":list(FOLDS), "seeds":list(SEEDS), "epochs":EPOCHS, "feature_dim":FEATURE_DIM, "context_dim":CONTEXT_DIM, "bootstrap_draws":BOOTSTRAP_DRAWS, "thresholds":{"specific_increment_db":SPECIFIC_INCREMENT, "absolute_gain_db":ABSOLUTE_GAIN, "retention":RETENTION_MIN, "tail_margin_db":TAIL_MARGIN, "severe_gain_db":SEVERE_GAIN, "hard_gain_db":HARD_GAIN}})
        atomic_json(context.phase_output_path / "r13_a0_provenance_and_access.json", {"schema_version":1, "route_commit":context.route_commit, "r11_source_commit":R11_COMMIT, "r11_terminal_preserved":True, "r12_terminal_preserved":True, "restoration_model_training_run":False, "restoration_model_inference_run":False, "checkpoint_loaded":False, "candidate_generation_rerun":False, "confirmation_touched":False, "canary_touched":False, "locked_test_touched":False})
        atomic_json(context.phase_output_path / "r13_a0_input_identity.json", {"schema_version":1, **input_metadata, "asset_sha256":{key:context.assets[key].sha256 for key in sorted(context.assets)}})
        atomic_json(context.phase_output_path / "r13_a0_representation_identity.json", {"schema_version":1, "feature_dim":FEATURE_DIM, "local_dim":LOCAL_DIM, "context_dim":CONTEXT_DIM, "rank_channels":18, "robust_z_channels":18, "cross_action_rank_difference_channels":18, "quantile_channels":54, "quantiles":[0.10,0.50,0.90], "tie_method":"average", "normalizer_source":"primary_outer_training_rows_only"})
        write_csv(context.phase_output_path / "r13_a0_cell_summary.csv", cell_rows)
        atomic_json(context.phase_output_path / "r13_a0_bootstrap_summary.json", {"schema_version":1, **boot})
        write_csv(context.phase_output_path / "r13_a0_fold_seed_stability.csv", replay_rows + training_rows)
        atomic_json(context.phase_output_path / "r13_a0_operator_consistency.json", {"schema_version":1, "primary_gain_pearson":r11.safe_pearson([row[f"{PRIMARY}_D_ref"] for row in evaluation_rows], [row[f"{PRIMARY}_D_rep"] for row in evaluation_rows]), "shared_action_map_by_construction":True})
        atomic_json(context.phase_output_path / "r13_a0_gate_summary.json", {"schema_version":1, "gates":gates, "fold_metrics":fold_metrics, "primary_severe_groups":severe, "primary_hard_groups":hard, "state":state, "decision":decision, "authorizes":authorizes, "passes":state=="COMPLETED_GATE_PASS", "r11_terminal_changed":False, "r12_terminal_changed":False})
        atomic_json(context.phase_output_path / "r13_a0_resource_summary.json", {"schema_version":1, "wall_seconds":time.perf_counter()-started_wall, "cpu_seconds":time.process_time()-started_cpu, "max_rss_mib":float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)/1024.0, "evaluated_units":input_metadata["evaluated_units"], "models":20, "epochs":EPOCHS, "bootstrap_draws":BOOTSTRAP_DRAWS, "gpu_used":False})
        atomic_json(context.phase_output_path / "r13_a0_scientific_conclusion.json", conclusion(state, decision, authorizes, boot, gates, severe, hard))
        write_run_result(context, state=state, decision=decision, authorizes=authorizes, details={"primary_gain_db":boot[f"{PRIMARY}_gain"]["point"], "delta_specific_db":boot["delta_specific"]["point"], "oracle_retention":boot["retention"]["point"], "primary_minus_global_cvar5":boot["primary_minus_global_cvar5"]["point"], "primary_severe_groups":severe, "primary_hard_groups":hard, "r11_terminal_changed":False, "r12_terminal_changed":False})
    except (ContextInconclusive, json.JSONDecodeError, OSError, ValueError) as exc:
        write_inconclusive(context, str(exc), started_wall, started_cpu)


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
