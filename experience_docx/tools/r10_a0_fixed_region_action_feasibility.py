#!/usr/bin/env python3
"""Frozen R10 privileged fixed-region action-feasibility audit."""

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


ROUTE_ID = "haze4k_v5_r10_fixed_region_action_feasibility_20260719"
OPERATION_ID = "R10_A0_FIXED_REGION_ACTION_FEASIBILITY_AUDIT"
R5_ROUTE_ID = "haze4k_v5_r5_spatial_candidate_response_sufficiency_20260719"
R5_OPERATION_ID = "R5_A0_FROZEN_SPATIAL_RESPONSE_SUFFICIENCY_SCREEN"
R5_RUN_ID = "r5-a0-spatial-response-screen-r2"
R5_ROUTE_COMMIT = "7e75eed504b2ead65a1971ec250dc7f59a79574d"
R9_ROUTE_ID = "haze4k_v5_r9_r5_decision_factorial_attribution_20260719"
R9_OPERATION_ID = "R9_A0_FROZEN_R5_DECISION_FACTORIAL_ATTRIBUTION_AUDIT"
R9_RUN_ID = "r9-a0-r5-decision-factorial-r1"
R9_ROUTE_COMMIT = "3f72363850d8f51268163f1cad3d15ca7d40cd74"
RUNNER_SHA256 = "336c7e1beccb793229beb533ba12367261e702866497c388ee2a4fa88d12718b"
CACHE_MANIFEST_IDENTITY = "b54603b51c810436c14bc9e993aef9f1294902efbc51fadebdd2c84d7c827a1d"
ACTIONS = ("reference_noop", "state_positive_full", "state_negative_full")
OPERATORS = ("D_ref", "D_rep")
FOLDS = (0, 1)
PRIMARY_CELL = "S1_TRUE_SPATIAL_RESPONSE"
GRID = 8
TILES = GRID * GRID
SHUFFLE_REPLICATES = 16
BOOTSTRAP_DRAWS = 4000
BOOTSTRAP_SEED = 3407
SEVERE_GAIN = -0.2
HARD_GAIN = -0.5
ABSOLUTE_GAIN = 0.020
INCREMENT_GAIN = 0.005
TAIL_MARGIN = -0.005
MIXED_FRACTION = 0.25
BIDIRECTIONAL_FRACTION = 0.10
EXPECTED_RAW_ROWS = 1536
EXPECTED_EVALUATED_NAMES = 384
EXPECTED_EVALUATED_UNITS = 768
CANDIDATE_HEADER = {
    "action", "cell", "fold", "mean_score", "name", "operator",
    "q05_score", "severe_label", "severe_score", "target_gain_db",
}


class FeasibilityInconclusive(RuntimeError):
    """A typed scientific-input stop that still owns a complete closeout."""


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
        raise FeasibilityInconclusive(f"invalid JSON: {label}") from exc
    if not isinstance(value, dict):
        raise FeasibilityInconclusive(f"JSON is not an object: {label}")
    return value


def read_csv(path: Path, header: set[str]) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None or set(reader.fieldnames) != header:
                raise FeasibilityInconclusive(f"CSV header mismatch: {path.name}")
            rows = list(reader)
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise FeasibilityInconclusive(f"invalid CSV: {path.name}") from exc
    if not rows or any(None in row or any(value is None for value in row.values()) for row in rows):
        raise FeasibilityInconclusive(f"CSV row contract failed: {path.name}")
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    fields = sorted({key for row in rows for key in row})
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def interval(point: float, samples: Any) -> dict[str, float]:
    import numpy as np

    values = np.asarray(samples, dtype=np.float64)
    if not math.isfinite(point) or values.size == 0 or not np.isfinite(values).all():
        raise FeasibilityInconclusive("non-finite bootstrap result")
    return {
        "point": float(point),
        "lcb95": float(np.quantile(values, 0.025)),
        "ucb95": float(np.quantile(values, 0.975)),
    }


def cvar(values: Any, fraction: float = 0.05) -> float:
    import numpy as np

    array = np.sort(np.asarray(values, dtype=np.float64))
    return float(array[: max(1, math.ceil(fraction * len(array)))].mean())


def psnr_gain(reference_sse: float, candidate_sse: float) -> float:
    reference = max(float(reference_sse), 1.0e-30)
    candidate = max(float(candidate_sse), 1.0e-30)
    return 10.0 * math.log10(reference / candidate)


def binomial_interval(events: int, total: int) -> dict[str, float]:
    from scipy.stats import beta

    if not 0 <= events <= total or total <= 0:
        raise FeasibilityInconclusive("invalid binomial counts")
    point = events / total
    low = 0.0 if events == 0 else float(beta.ppf(0.025, events, total - events + 1))
    high = 1.0 if events == total else float(beta.ppf(0.975, events + 1, total - events))
    return {"events": events, "total": total, "point": point, "lcb95": low, "ucb95": high}


def metric_psnr(mse: Any) -> Any:
    import torch

    return 10.0 * torch.log10(1.0 / torch.clamp(mse, min=1.0e-30))


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
    raise FeasibilityInconclusive(f"development target is missing: {name}")


def tile_sse(errors: Any) -> tuple[Any, list[int]]:
    import torch

    if errors.ndim != 4 or errors.shape[0] != len(ACTIONS) or errors.shape[1] != 3:
        raise FeasibilityInconclusive("render-error tensor shape mismatch")
    height, width = int(errors.shape[-2]), int(errors.shape[-1])
    values = torch.empty((len(ACTIONS), TILES), dtype=torch.float64)
    pixel_counts = []
    tile = 0
    for row in range(GRID):
        y0, y1 = (row * height) // GRID, ((row + 1) * height) // GRID
        for column in range(GRID):
            x0, x1 = (column * width) // GRID, ((column + 1) * width) // GRID
            if y1 <= y0 or x1 <= x0:
                raise FeasibilityInconclusive("native image is smaller than fixed grid")
            patch = errors[:, :, y0:y1, x0:x1]
            values[:, tile] = patch.double().square().sum((1, 2, 3))
            pixel_counts.append(3 * (y1 - y0) * (x1 - x0))
            tile += 1
    if tile != TILES or sum(pixel_counts) != 3 * height * width or not bool(torch.isfinite(values).all()):
        raise FeasibilityInconclusive("fixed tile partition is incomplete or non-finite")
    return values, pixel_counts


def deterministic_permutation(name: str, replicate: int) -> Any:
    import numpy as np

    digest = hashlib.sha256(f"{ROUTE_ID}|{name}|shuffle={replicate}".encode()).digest()
    seed = int.from_bytes(digest[:8], "big")
    return np.random.default_rng(seed).permutation(TILES)


def analyze_groups(groups: dict[str, dict[str, dict[str, Any]]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import numpy as np

    rows = []
    local_safety_violations = 0
    shuffle_histogram_violations = 0
    for name in sorted(groups):
        operator_units = groups[name]
        if set(operator_units) != set(OPERATORS):
            raise FeasibilityInconclusive("operator pairing is incomplete")
        if len({tuple(unit["pixel_counts"]) for unit in operator_units.values()}) != 1:
            raise FeasibilityInconclusive("paired operators have different normalized tile partitions")
        action_map = []
        for tile in range(TILES):
            scores = [0.0]
            for action in (1, 2):
                gains = [
                    psnr_gain(unit["sse"][0, tile], unit["sse"][action, tile])
                    for unit in operator_units.values()
                ]
                scores.append(min(gains) if all(unit["sse"][action, tile] <= unit["sse"][0, tile]
                                                 for unit in operator_units.values()) else -math.inf)
            action_map.append(max(range(3), key=lambda action: (scores[action], -action)))
        action_map_array = np.asarray(action_map, dtype=np.int64)
        for operator, unit in operator_units.items():
            for tile, action in enumerate(action_map):
                local_safety_violations += bool(
                    unit["sse"][action, tile] > unit["sse"][0, tile]
                )
        region_gain = {}
        global_gain_by_operator = {}
        global_action_scores = [0.0]
        for action in (1, 2):
            gains = {
                operator: psnr_gain(float(unit["sse"][0].sum()), float(unit["sse"][action].sum()))
                for operator, unit in operator_units.items()
            }
            global_action_scores.append(min(gains.values()) if all(value >= 0.0 for value in gains.values()) else -math.inf)
        global_action = max(range(3), key=lambda action: (global_action_scores[action], -action))
        shuffle_gain = {operator: [] for operator in OPERATORS}
        for operator, unit in operator_units.items():
            reference_sse = float(unit["sse"][0].sum())
            selected_sse = sum(float(unit["sse"][action, tile]) for tile, action in enumerate(action_map))
            region_gain[operator] = psnr_gain(reference_sse, selected_sse)
            global_gain_by_operator[operator] = psnr_gain(reference_sse, float(unit["sse"][global_action].sum()))
            original_histogram = np.bincount(action_map_array, minlength=3)
            for replicate in range(SHUFFLE_REPLICATES):
                permuted = action_map_array[deterministic_permutation(name, replicate)]
                shuffle_histogram_violations += not np.array_equal(
                    original_histogram, np.bincount(permuted, minlength=3),
                )
                shuffled_sse = sum(float(unit["sse"][action, tile]) for tile, action in enumerate(permuted))
                shuffle_gain[operator].append(psnr_gain(reference_sse, shuffled_sse))
        counts = np.bincount(action_map_array, minlength=3)
        row = {
            "name": name,
            "fold": next(iter(operator_units.values()))["fold"],
            "shape": next(iter(operator_units.values()))["shape"],
            "noop_tiles": int(counts[0]),
            "positive_tiles": int(counts[1]),
            "negative_tiles": int(counts[2]),
            "mixed_noop_active": bool(counts[0] > 0 and counts[1] + counts[2] > 0),
            "bidirectional": bool(counts[1] > 0 and counts[2] > 0),
            "global_action": int(global_action),
            "action_map": [int(value) for value in action_map],
        }
        for operator in OPERATORS:
            row[f"region_{operator}"] = region_gain[operator]
            row[f"global_{operator}"] = global_gain_by_operator[operator]
            row[f"shuffle_{operator}"] = float(np.mean(shuffle_gain[operator]))
        rows.append(row)
    return rows, {
        "local_safety_violations": int(local_safety_violations),
        "shuffle_histogram_violations": int(shuffle_histogram_violations),
    }


def evaluate_rows(rows: list[dict[str, Any]], indices: Any) -> dict[str, float]:
    import numpy as np

    result: dict[str, float] = {}
    region_means = {}
    global_means = {}
    shuffle_means = {}
    for operator in OPERATORS:
        region = np.asarray([row[f"region_{operator}"] for row in rows], dtype=np.float64)[indices]
        global_values = np.asarray([row[f"global_{operator}"] for row in rows], dtype=np.float64)[indices]
        shuffle = np.asarray([row[f"shuffle_{operator}"] for row in rows], dtype=np.float64)[indices]
        region_means[operator] = float(region.mean())
        global_means[operator] = float(global_values.mean())
        shuffle_means[operator] = float(shuffle.mean())
        result[f"region_{operator}"] = region_means[operator]
        result[f"global_{operator}"] = global_means[operator]
        result[f"shuffle_{operator}"] = shuffle_means[operator]
    result["region_gain"] = min(region_means.values())
    result["global_gain"] = min(global_means.values())
    result["shuffle_gain"] = min(shuffle_means.values())
    result["region_minus_global"] = min(
        region_means[operator] - global_means[operator] for operator in OPERATORS
    )
    result["region_minus_shuffle"] = min(
        region_means[operator] - shuffle_means[operator] for operator in OPERATORS
    )
    result["region_minus_global_cvar5"] = min(
        cvar(np.asarray([row[f"region_{operator}"] for row in rows], dtype=np.float64)[indices])
        - cvar(np.asarray([row[f"global_{operator}"] for row in rows], dtype=np.float64)[indices])
        for operator in OPERATORS
    )
    return result


def bootstrap(rows: list[dict[str, Any]], draws: int, seed: int) -> dict[str, Any]:
    import numpy as np

    count = len(rows)
    point = evaluate_rows(rows, np.arange(count))
    samples = {key: [] for key in point}
    generator = np.random.default_rng(seed)
    for _draw in range(draws):
        value = evaluate_rows(rows, generator.integers(0, count, count))
        for key in samples:
            samples[key].append(value[key])
    return {key: interval(point[key], samples[key]) for key in point}


def safe_pearson(first: list[float], second: list[float]) -> float | None:
    import numpy as np

    x = np.asarray(first, dtype=np.float64)
    y = np.asarray(second, dtype=np.float64)
    if len(x) < 2 or float(x.std()) <= 0.0 or float(y.std()) <= 0.0:
        return None
    value = float(np.corrcoef(x, y)[0, 1])
    return value if math.isfinite(value) else None


def synthetic_groups() -> dict[str, dict[str, dict[str, Any]]]:
    import torch

    groups = {}
    for index in range(EXPECTED_EVALUATED_NAMES):
        name = f"synthetic_{index:04d}"
        operator_units = {}
        for operator_index, operator in enumerate(OPERATORS):
            reference = torch.full((TILES,), 100.0 + operator_index, dtype=torch.float64)
            positive = reference.clone()
            negative = reference.clone()
            magnitude = float(index % 7)
            for tile in range(TILES):
                pattern = tile % 4
                if pattern == 0:
                    positive[tile] = 78.0 + magnitude + operator_index
                    negative[tile] = 112.0 + operator_index
                elif pattern == 1:
                    positive[tile] = 111.0 + operator_index
                    negative[tile] = 76.0 + 0.5 * magnitude + operator_index
                elif pattern == 2:
                    positive[tile] = 108.0 + operator_index
                    negative[tile] = 109.0 + operator_index
                else:
                    positive[tile] = 91.0 + 0.25 * magnitude + operator_index
                    negative[tile] = 96.0 + operator_index
            operator_units[operator] = {
                "fold": index // 192,
                "shape": "256x256",
                "sse": torch.stack((reference, positive, negative)),
                "pixel_counts": [3072] * TILES,
            }
        groups[name] = operator_units
    return groups


def contract(context_path: Path) -> None:
    context = load_context(context_path, "contract")
    prepare_phase_output(context)
    groups = synthetic_groups()
    rows, integrity = analyze_groups(groups)
    first = bootstrap(rows, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED)
    second = bootstrap(rows, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED)
    checks = {
        "route_identity": context.route_id == ROUTE_ID and context.operation_id == OPERATION_ID,
        "cpu_only": context.device == "cpu",
        "protected_roles_blocked": not any(context.protected_data_permissions.values()),
        "formal_group_scale": len(groups) == EXPECTED_EVALUATED_NAMES,
        "paired_operators": all(set(group) == set(OPERATORS) for group in groups.values()),
        "fixed_tiles": all(
            len(unit["pixel_counts"]) == TILES for group in groups.values() for unit in group.values()
        ),
        "local_safety": integrity["local_safety_violations"] == 0,
        "shuffle_histogram": integrity["shuffle_histogram_violations"] == 0,
        "mixed_actions_exercised": all(
            row["mixed_noop_active"] and row["bidirectional"] for row in rows
        ),
        "positive_region_increment": first["region_minus_global"]["point"] > 0.0,
        "positive_spatial_specificity": first["region_minus_shuffle"]["point"] > 0.0,
        "deterministic_full_bootstrap": first == second,
        "finite_outputs": all(math.isfinite(value["point"]) for value in first.values()),
        "bounded_work_class": (
            len(groups) == 384 and TILES == 64 and SHUFFLE_REPLICATES == 16
            and BOOTSTRAP_DRAWS == 4000
        ),
    }
    write_contract_result(context, checks=checks)


def load_formal_groups(context: Any) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, Any]]:
    import torch

    cache_manifest = read_json(asset_path(context, "a0_cache_manifest", kind="file"), "cache_manifest")
    if cache_manifest.get("cache_manifest_sha256") != CACHE_MANIFEST_IDENTITY:
        raise FeasibilityInconclusive("cache manifest internal identity mismatch")
    try:
        raw_rows = [
            json.loads(line)
            for line in asset_path(context, "a0_raw_manifest", kind="file").read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FeasibilityInconclusive("invalid raw cache manifest") from exc
    if len(raw_rows) != EXPECTED_RAW_ROWS or any(not isinstance(row, dict) for row in raw_rows):
        raise FeasibilityInconclusive("raw cache manifest cardinality mismatch")
    raw_by_key = {}
    for row in raw_rows:
        key = row.get("unit_key")
        if not isinstance(key, str) or key in raw_by_key or not isinstance(row.get("cache_sha256"), str):
            raise FeasibilityInconclusive("raw cache manifest key/hash mismatch")
        raw_by_key[key] = row

    candidate_rows = read_csv(
        asset_path(context, "r5_candidate_scores", kind="file"), CANDIDATE_HEADER,
    )
    if len(candidate_rows) != 6144:
        raise FeasibilityInconclusive("R5 candidate-score cardinality mismatch")
    primary = [row for row in candidate_rows if row["cell"] == PRIMARY_CELL]
    if len(primary) != 1536:
        raise FeasibilityInconclusive("R5 primary candidate rows are incomplete")
    target_map = {}
    fold_by_name = {}
    for row in primary:
        try:
            fold = int(row["fold"])
            action = ACTIONS.index(row["action"])
            target = float(row["target_gain_db"])
        except (ValueError, TypeError) as exc:
            raise FeasibilityInconclusive("R5 candidate row parse failure") from exc
        name = row["name"]
        operator = row["operator"]
        if fold not in FOLDS or action not in (1, 2) or operator not in OPERATORS \
                or not math.isfinite(target):
            raise FeasibilityInconclusive("R5 candidate identity is outside R10 scope")
        key = (name, operator, action)
        if key in target_map:
            raise FeasibilityInconclusive("duplicate R5 candidate target")
        target_map[key] = target
        if name in fold_by_name and fold_by_name[name] != fold:
            raise FeasibilityInconclusive("one name appears in multiple R5 folds")
        fold_by_name[name] = fold
    names = sorted(fold_by_name)
    required_targets = {
        (name, operator, action)
        for name in names for operator in OPERATORS for action in (1, 2)
    }
    if len(names) != EXPECTED_EVALUATED_NAMES or set(target_map) != required_targets \
            or any(sum(fold_by_name[name] == fold for name in names) != 192 for fold in FOLDS):
        raise FeasibilityInconclusive("R5 folds0/1 population grid is incomplete")

    cache_root = asset_path(context, "a0_candidate_cache", kind="directory")
    data_root = asset_path(context, "haze4k_data", kind="directory")
    groups: dict[str, dict[str, dict[str, Any]]] = {}
    target_mismatches = 0
    target_max_abs_difference = 0.0
    no_op_render_mismatches = 0
    partition_sse_max_abs_difference = 0.0
    loaded = 0
    for name in names:
        label = load_label(data_root, name)
        for operator in OPERATORS:
            unit_key = hashlib.sha256(f"{name}\0{operator}".encode()).hexdigest()[:32]
            raw = raw_by_key.get(unit_key)
            if raw is None:
                raise FeasibilityInconclusive("evaluated cache unit is absent from raw manifest")
            unit_path = cache_root / f"{unit_key}.pt"
            if sha256_file(unit_path) != raw["cache_sha256"]:
                raise FeasibilityInconclusive(f"cache unit hash mismatch: {unit_key}")
            try:
                payload = torch.load(unit_path, map_location="cpu", weights_only=False)
            except Exception as exc:
                raise FeasibilityInconclusive(f"cache unit load failed: {unit_key}") from exc
            candidate_names = list(payload.get("candidate_names", []))
            if any(action not in candidate_names for action in ACTIONS) \
                    or payload.get("name") != name or payload.get("operator") != operator:
                raise FeasibilityInconclusive("cache payload identity/action mismatch")
            selected_indices = [candidate_names.index(action) for action in ACTIONS]
            base = payload["base"].float()
            step = payload["step"].float()
            candidate_delta = payload["candidates"].float()[selected_indices]
            if label.shape[-2:] != base.shape[-2:]:
                label_used = label[:, :, : base.shape[-2], : base.shape[-1]]
            else:
                label_used = label
            reference = torch.clamp(base + 0.25 * step, 0.0, 1.0)
            renders = torch.clamp(base + 0.25 * (step + candidate_delta), 0.0, 1.0)
            no_op_render_mismatches += not bool(torch.equal(reference, renders[0:1]))
            active = renders[1:]
            observed_targets = metric_psnr((active - label_used).square().mean((1, 2, 3))) \
                - metric_psnr((reference - label_used).square().mean())
            for action in (1, 2):
                observed = float(observed_targets[action - 1])
                expected = target_map[(name, operator, action)]
                difference = abs(observed - expected)
                target_max_abs_difference = max(target_max_abs_difference, difference)
                target_mismatches += observed != expected
            render_errors = torch.cat((reference, active), dim=0) - label_used
            sse, pixel_counts = tile_sse(render_errors)
            exact_sse = render_errors.double().square().sum((1, 2, 3))
            partition_difference = float((sse.sum(1) - exact_sse).abs().max())
            partition_sse_max_abs_difference = max(
                partition_sse_max_abs_difference, partition_difference,
            )
            groups.setdefault(name, {})[operator] = {
                "fold": fold_by_name[name],
                "shape": f"{base.shape[-2]}x{base.shape[-1]}",
                "sse": sse,
                "pixel_counts": pixel_counts,
            }
            loaded += 1
            if loaded % 8 == 0 or loaded == EXPECTED_EVALUATED_UNITS:
                write_workload_progress(
                    context, completed_units=2 + loaded, stage="cache_target_replay",
                )
    if loaded != EXPECTED_EVALUATED_UNITS or any(set(group) != set(OPERATORS) for group in groups.values()):
        raise FeasibilityInconclusive("formal cache/operator grid is incomplete")
    return groups, {
        "candidate_rows": len(candidate_rows),
        "primary_candidate_rows": len(primary),
        "raw_manifest_rows": len(raw_rows),
        "evaluated_names": len(names),
        "evaluated_units": loaded,
        "target_mismatches": target_mismatches,
        "target_max_abs_difference": target_max_abs_difference,
        "no_op_render_mismatches": int(no_op_render_mismatches),
        "partition_sse_max_abs_difference": partition_sse_max_abs_difference,
        "cache_manifest_internal_identity": cache_manifest.get("cache_manifest_sha256"),
    }


def write_inconclusive_bundle(
    context: Any, reason: str, started_wall: float, started_cpu: float,
) -> None:
    common = {
        "schema_version": 1,
        "status": "input_or_feasibility_inconclusive",
        "reason": reason,
        "r5_terminal_changed": False,
        "r9_terminal_changed": False,
    }
    for filename in (
        "r10_a0_contract_summary.json",
        "r10_a0_provenance_and_access.json",
        "r10_a0_input_identity.json",
        "r10_a0_global_target_replay.json",
        "r10_a0_region_action_distribution.json",
        "r10_a0_bootstrap_summary.json",
        "r10_a0_spatial_shuffle_summary.json",
        "r10_a0_operator_consistency.json",
        "r10_a0_gate_summary.json",
    ):
        atomic_json(context.phase_output_path / filename, common)
    placeholder = [{"status": "inconclusive", "reason": reason}]
    write_csv(context.phase_output_path / "r10_a0_feasibility_summary.csv", placeholder)
    write_csv(context.phase_output_path / "r10_a0_native_shape_summary.csv", placeholder)
    atomic_json(context.phase_output_path / "r10_a0_resource_summary.json", {
        **common,
        "wall_seconds": time.perf_counter() - started_wall,
        "cpu_seconds": time.process_time() - started_cpu,
        "gpu_used": False,
    })
    write_run_result(
        context,
        state="COMPLETED_GATE_INCONCLUSIVE",
        decision="R10_A0_INPUT_OR_FEASIBILITY_INCONCLUSIVE_STOP",
        authorizes="NONE",
        details={
            "reason": reason,
            "r5_terminal_changed": False,
            "r9_terminal_changed": False,
        },
    )


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
        r5_closeout = read_json(asset_path(context, "r5_closeout", kind="file"), "r5_closeout")
        if (
            r5_closeout.get("route_id"), r5_closeout.get("operation_id"),
            r5_closeout.get("run_id"), r5_closeout.get("route_commit"),
            r5_closeout.get("runner_sha256"), r5_closeout.get("state"),
            r5_closeout.get("decision"), r5_closeout.get("authorizes"),
        ) != (
            R5_ROUTE_ID, R5_OPERATION_ID, R5_RUN_ID, R5_ROUTE_COMMIT, RUNNER_SHA256,
            "COMPLETED_GATE_FAIL", "R5_A0_SPATIAL_RESPONSE_FUTILITY_OR_SAFETY_FAIL_STOP", "NONE",
        ):
            raise FeasibilityInconclusive("R5 typed closeout identity or terminal mismatch")
        r9_closeout = read_json(asset_path(context, "r9_closeout", kind="file"), "r9_closeout")
        if (
            r9_closeout.get("route_id"), r9_closeout.get("operation_id"),
            r9_closeout.get("run_id"), r9_closeout.get("route_commit"),
            r9_closeout.get("runner_sha256"), r9_closeout.get("state"),
            r9_closeout.get("decision"), r9_closeout.get("authorizes"),
        ) != (
            R9_ROUTE_ID, R9_OPERATION_ID, R9_RUN_ID, R9_ROUTE_COMMIT, RUNNER_SHA256,
            "COMPLETED_GATE_PASS", "R9_A0_DECISION_FACTORIAL_ATTRIBUTION_PASS",
            "R9_NEXT_CONTRACT_REVIEW_ONLY",
        ):
            raise FeasibilityInconclusive("R9 typed closeout identity or authorization mismatch")
        write_workload_progress(context, completed_units=2, stage="terminal_inputs_verified")
        groups, input_metadata = load_formal_groups(context)
        if input_metadata["target_mismatches"] != 0 \
                or input_metadata["no_op_render_mismatches"] != 0 \
                or input_metadata["partition_sse_max_abs_difference"] > 1.0e-9:
            raise FeasibilityInconclusive("R5 target/no-op/tile-partition replay mismatch")
        rows, integrity = analyze_groups(groups)
        if integrity["local_safety_violations"] != 0 \
                or integrity["shuffle_histogram_violations"] != 0:
            raise FeasibilityInconclusive("regional safety or shuffle-histogram construction failed")
        write_workload_progress(context, completed_units=770, stage="region_maps_complete")
        boot = bootstrap(rows, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED)
        write_workload_progress(context, completed_units=4770, stage="bootstrap_complete")

        mixed = binomial_interval(sum(row["mixed_noop_active"] for row in rows), len(rows))
        bidirectional = binomial_interval(sum(row["bidirectional"] for row in rows), len(rows))
        severe = sum(
            any(row[f"region_{operator}"] <= SEVERE_GAIN for operator in OPERATORS)
            for row in rows
        )
        hard = sum(
            any(row[f"region_{operator}"] <= HARD_GAIN for operator in OPERATORS)
            for row in rows
        )
        structural_checks = {
            "r5_target_replay_exact": input_metadata["target_mismatches"] == 0,
            "no_op_render_replay_exact": input_metadata["no_op_render_mismatches"] == 0,
            "tile_partition_exact": input_metadata["partition_sse_max_abs_difference"] <= 1.0e-9,
            "population_complete": len(rows) == EXPECTED_EVALUATED_NAMES,
            "operator_grid_complete": input_metadata["evaluated_units"] == EXPECTED_EVALUATED_UNITS,
            "folds_complete": sorted({row["fold"] for row in rows}) == list(FOLDS),
            "finite_metrics": all(math.isfinite(value["point"]) for value in boot.values()),
            "local_dual_operator_safety": integrity["local_safety_violations"] == 0,
            "shuffle_histograms_exact": integrity["shuffle_histogram_violations"] == 0,
            "protected_roles_untouched": not any(context.protected_data_permissions.values()),
        }
        if not all(structural_checks.values()):
            failed = sorted(key for key, value in structural_checks.items() if not value)
            raise FeasibilityInconclusive(f"structural checks failed: {failed}")
        gates = {
            "region_gain_lcb95": boot["region_gain"]["lcb95"] >= ABSOLUTE_GAIN,
            "region_minus_global_lcb95": boot["region_minus_global"]["lcb95"] >= INCREMENT_GAIN,
            "region_minus_shuffle_lcb95": boot["region_minus_shuffle"]["lcb95"] >= INCREMENT_GAIN,
            "region_minus_global_cvar5_lcb95": (
                boot["region_minus_global_cvar5"]["lcb95"] >= TAIL_MARGIN
            ),
            "zero_region_severe": severe == 0,
            "zero_region_hard": hard == 0,
            "mixed_fraction_lcb95": mixed["lcb95"] >= MIXED_FRACTION,
            "bidirectional_fraction_lcb95": bidirectional["lcb95"] >= BIDIRECTIONAL_FRACTION,
        }
        passes = all(gates.values())
        decisive_fail = (
            boot["region_gain"]["ucb95"] < ABSOLUTE_GAIN
            or boot["region_minus_global"]["ucb95"] < INCREMENT_GAIN
            or boot["region_minus_shuffle"]["ucb95"] < INCREMENT_GAIN
            or boot["region_minus_global_cvar5"]["ucb95"] < TAIL_MARGIN
            or mixed["ucb95"] < MIXED_FRACTION
            or bidirectional["ucb95"] < BIDIRECTIONAL_FRACTION
        )
        if passes:
            state = "COMPLETED_GATE_PASS"
            decision = "R10_A0_FIXED_REGION_ACTION_FEASIBILITY_PASS"
            authorizes = "R10_REGION_OBSERVABILITY_CONTRACT_REVIEW_ONLY"
        elif decisive_fail:
            state = "COMPLETED_GATE_FAIL"
            decision = "R10_A0_FIXED_REGION_ACTION_FEASIBILITY_FAIL_STOP"
            authorizes = "NONE"
        else:
            state = "COMPLETED_GATE_INCONCLUSIVE"
            decision = "R10_A0_INPUT_OR_FEASIBILITY_INCONCLUSIVE_STOP"
            authorizes = "NONE"

        summary_rows = []
        for policy, prefix in (
            ("region_oracle", "region"),
            ("safe_global_oracle", "global"),
            ("spatial_shuffle_control", "shuffle"),
        ):
            gains = {
                operator: np.asarray([row[f"{prefix}_{operator}"] for row in rows], dtype=np.float64)
                for operator in OPERATORS
            }
            means = {operator: float(values.mean()) for operator, values in gains.items()}
            summary_rows.append({
                "policy": policy,
                "groups": len(rows),
                "d_ref_gain_db": means["D_ref"],
                "d_rep_gain_db": means["D_rep"],
                "worse_operator_gain_db": min(means.values()),
                "worse_operator_cvar5_db": min(cvar(values) for values in gains.values()),
                "selected_severe_groups": sum(
                    any(row[f"{prefix}_{operator}"] <= SEVERE_GAIN for operator in OPERATORS)
                    for row in rows
                ),
                "selected_hard_groups": sum(
                    any(row[f"{prefix}_{operator}"] <= HARD_GAIN for operator in OPERATORS)
                    for row in rows
                ),
            })
        shape_rows = []
        for shape in sorted({row["shape"] for row in rows}):
            subset = [row for row in rows if row["shape"] == shape]
            for operator in OPERATORS:
                shape_rows.append({
                    "shape": shape,
                    "operator": operator,
                    "groups": len(subset),
                    "region_mean_gain_db": float(np.mean([row[f"region_{operator}"] for row in subset])),
                    "global_mean_gain_db": float(np.mean([row[f"global_{operator}"] for row in subset])),
                    "shuffle_mean_gain_db": float(np.mean([row[f"shuffle_{operator}"] for row in subset])),
                    "mixed_fraction": float(np.mean([row["mixed_noop_active"] for row in subset])),
                    "bidirectional_fraction": float(np.mean([row["bidirectional"] for row in subset])),
                })
        action_distribution = {
            "schema_version": 1,
            "tile_counts": {
                "noop": sum(row["noop_tiles"] for row in rows),
                "positive": sum(row["positive_tiles"] for row in rows),
                "negative": sum(row["negative_tiles"] for row in rows),
            },
            "mixed_noop_active": mixed,
            "bidirectional_positive_negative": bidirectional,
            "global_action_counts": {
                ACTIONS[action]: sum(row["global_action"] == action for row in rows)
                for action in range(3)
            },
        }
        operator_consistency = {
            "schema_version": 1,
            "region_gain_pearson": safe_pearson(
                [row["region_D_ref"] for row in rows], [row["region_D_rep"] for row in rows],
            ),
            "global_gain_pearson": safe_pearson(
                [row["global_D_ref"] for row in rows], [row["global_D_rep"] for row in rows],
            ),
            "shared_region_map_by_construction": True,
            "local_dual_operator_safety_violations": integrity["local_safety_violations"],
        }
        contract_summary = {
            "schema_version": 1,
            "route_id": ROUTE_ID,
            "operation_id": OPERATION_ID,
            "scientific_role": "post_hoc_privileged_development_feasibility",
            "grid": [GRID, GRID],
            "actions": list(ACTIONS),
            "operators": list(OPERATORS),
            "folds": list(FOLDS),
            "shuffle_replicates": SHUFFLE_REPLICATES,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "thresholds": {
                "absolute_gain_db": ABSOLUTE_GAIN,
                "increment_gain_db": INCREMENT_GAIN,
                "tail_margin_db": TAIL_MARGIN,
                "mixed_fraction": MIXED_FRACTION,
                "bidirectional_fraction": BIDIRECTIONAL_FRACTION,
                "severe_gain_db": SEVERE_GAIN,
                "hard_gain_db": HARD_GAIN,
            },
        }
        provenance = {
            "schema_version": 1,
            "route_commit": context.route_commit,
            "r5_terminal_preserved": True,
            "r9_terminal_preserved": True,
            "training_run": False,
            "model_inference_run": False,
            "candidate_generation_rerun": False,
            "checkpoint_loaded": False,
            "broader_r3_ledger_opened": False,
            "confirmation_identities_images_targets_outcomes_touched": False,
            "canary_touched": False,
            "locked_test_touched": False,
        }
        input_identity = {
            "schema_version": 1,
            **input_metadata,
            "asset_sha256": {
                identifier: context.assets[identifier].sha256 for identifier in sorted(context.assets)
            },
        }
        spatial_shuffle = {
            "schema_version": 1,
            "replicates": SHUFFLE_REPLICATES,
            "histogram_violations": integrity["shuffle_histogram_violations"],
            "region_minus_shuffle": boot["region_minus_shuffle"],
            "control_worse_operator_gain": boot["shuffle_gain"],
        }
        gate_summary = {
            "schema_version": 1,
            "structural_checks": structural_checks,
            "gates": gates,
            "passes": passes,
            "decisive_fail": decisive_fail,
            "region_severe_groups": severe,
            "region_hard_groups": hard,
            "state": state,
            "decision": decision,
            "authorizes": authorizes,
            "r5_terminal_changed": False,
            "r9_terminal_changed": False,
        }
        atomic_json(context.phase_output_path / "r10_a0_contract_summary.json", contract_summary)
        atomic_json(context.phase_output_path / "r10_a0_provenance_and_access.json", provenance)
        atomic_json(context.phase_output_path / "r10_a0_input_identity.json", input_identity)
        atomic_json(context.phase_output_path / "r10_a0_global_target_replay.json", {
            "schema_version": 1,
            "exact": input_metadata["target_mismatches"] == 0,
            "mismatches": input_metadata["target_mismatches"],
            "max_abs_difference": input_metadata["target_max_abs_difference"],
            "no_op_render_mismatches": input_metadata["no_op_render_mismatches"],
            "partition_sse_max_abs_difference": input_metadata["partition_sse_max_abs_difference"],
        })
        atomic_json(context.phase_output_path / "r10_a0_region_action_distribution.json", action_distribution)
        write_csv(context.phase_output_path / "r10_a0_feasibility_summary.csv", summary_rows)
        atomic_json(context.phase_output_path / "r10_a0_bootstrap_summary.json", {
            "schema_version": 1, **boot,
        })
        atomic_json(context.phase_output_path / "r10_a0_spatial_shuffle_summary.json", spatial_shuffle)
        atomic_json(context.phase_output_path / "r10_a0_operator_consistency.json", operator_consistency)
        write_csv(context.phase_output_path / "r10_a0_native_shape_summary.csv", shape_rows)
        atomic_json(context.phase_output_path / "r10_a0_gate_summary.json", gate_summary)
        atomic_json(context.phase_output_path / "r10_a0_resource_summary.json", {
            "schema_version": 1,
            "wall_seconds": time.perf_counter() - started_wall,
            "cpu_seconds": time.process_time() - started_cpu,
            "max_rss_mib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0,
            "evaluated_units": input_metadata["evaluated_units"],
            "tiles_per_unit": TILES,
            "shuffle_replicates": SHUFFLE_REPLICATES,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "gpu_used": False,
        })
        cloud_rows = []
        for row in rows:
            cloud_rows.append({
                **{key: value for key, value in row.items() if key != "action_map"},
                "action_map": ";".join(str(value) for value in row["action_map"]),
            })
        write_csv(context.phase_output_path / "r10_a0_per_image_region_rows_cloud_only.csv", cloud_rows)
        write_run_result(
            context,
            state=state,
            decision=decision,
            authorizes=authorizes,
            details={
                "region_gain_db": boot["region_gain"]["point"],
                "region_minus_global_db": boot["region_minus_global"]["point"],
                "region_minus_shuffle_db": boot["region_minus_shuffle"]["point"],
                "mixed_fraction": mixed["point"],
                "bidirectional_fraction": bidirectional["point"],
                "region_severe_groups": severe,
                "region_hard_groups": hard,
                "r5_terminal_changed": False,
                "r9_terminal_changed": False,
            },
        )
    except FeasibilityInconclusive as exc:
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
