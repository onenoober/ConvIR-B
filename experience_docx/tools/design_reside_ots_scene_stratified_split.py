#!/usr/bin/env python3
"""Freeze scene-disjoint, coverage-qualified roles for the eligible OTS pool."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image
from scipy.io import loadmat

from route_program_api import (
    asset_path,
    load_context,
    output_file,
    prepare_phase_output,
    write_contract_result,
    write_run_result,
    write_workload_progress,
)


EXPECTED_CLEAR = 8970
EXPECTED_DEPTH = 8970
EXPECTED_HAZE = 313950
EXPECTED_EXCLUSIONS = 964
EXPECTED_ELIGIBLE = 8006
EXPECTED_GRID = 35
TOTAL_UNITS = 8007
SEED = "reside-ots-scene-stratified-split-v1"
ROLE_QUOTAS = {
    "training": 4000,
    "model_development": 500,
    "measurement_definition": 1000,
    "measurement_validation": 1000,
    "reserve": 1506,
}
FEATURES = (
    "clear_luminance_mean",
    "clear_luminance_sd",
    "clear_gradient_energy",
    "clear_colorfulness",
    "log_depth_mean",
    "log_depth_sd",
    "log_depth_interdecile_range",
    "normalized_depth_gradient_energy",
)
ALLOCATION_FEATURES = (
    "clear_luminance_mean",
    "clear_gradient_energy",
    "log_depth_mean",
    "log_depth_interdecile_range",
)


def digest_lines(lines: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(lines)).encode("utf-8")).hexdigest()


def image_files(directory: Path, suffix: str) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() == suffix)


def _stable_key(*tokens: str) -> str:
    return hashlib.sha256("|".join((SEED, *tokens)).encode("utf-8")).hexdigest()


def _rank_quartiles(values: np.ndarray, ids: list[str]) -> np.ndarray:
    order = sorted(range(len(ids)), key=lambda index: (float(values[index]), ids[index]))
    bins = np.empty(len(ids), dtype=np.int8)
    for rank, index in enumerate(order):
        bins[index] = min(3, (4 * rank) // len(ids))
    return bins


def allocate_roles(ids: list[str], matrix: np.ndarray) -> tuple[dict[str, list[str]], dict[str, np.ndarray]]:
    if len(ids) != sum(ROLE_QUOTAS.values()) or matrix.shape != (len(ids), len(FEATURES)):
        raise ValueError("allocator input differs from the frozen population")
    feature_index = {name: index for index, name in enumerate(FEATURES)}
    bins = {
        name: _rank_quartiles(matrix[:, feature_index[name]], ids)
        for name in ALLOCATION_FEATURES
    }
    strata: dict[tuple[int, ...], list[str]] = defaultdict(list)
    for index, scene_id in enumerate(ids):
        strata[tuple(int(bins[name][index]) for name in ALLOCATION_FEATURES)].append(scene_id)
    for key in strata:
        strata[key].sort(key=lambda item: _stable_key("within", *map(str, key), item))
    ordered_keys = sorted(strata, key=lambda key: _stable_key("stratum", *map(str, key)))
    sequence: list[str] = []
    position = {key: 0 for key in ordered_keys}
    while len(sequence) < len(ids):
        for key in ordered_keys:
            offset = position[key]
            if offset < len(strata[key]):
                sequence.append(strata[key][offset])
                position[key] += 1
    assigned = {role: [] for role in ROLE_QUOTAS}
    counts = {role: 0 for role in ROLE_QUOTAS}
    total = len(sequence)
    for step, scene_id in enumerate(sequence, start=1):
        available = [role for role, quota in ROLE_QUOTAS.items() if counts[role] < quota]
        role = max(
            available,
            key=lambda item: (
                (step * ROLE_QUOTAS[item] / total) - counts[item],
                _stable_key("role", scene_id, item),
            ),
        )
        assigned[role].append(scene_id)
        counts[role] += 1
    for role in assigned:
        assigned[role].sort()
    return assigned, bins


def empirical_cdf_gap(left: np.ndarray, right: np.ndarray) -> float:
    left = np.sort(np.asarray(left, dtype=np.float64))
    right = np.sort(np.asarray(right, dtype=np.float64))
    points = np.unique(np.concatenate((left, right)))
    left_cdf = np.searchsorted(left, points, side="right") / len(left)
    right_cdf = np.searchsorted(right, points, side="right") / len(right)
    return float(np.max(np.abs(left_cdf - right_cdf)))


def coverage_rows(
    ids: list[str], matrix: np.ndarray, roles: dict[str, list[str]], bins: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    by_id = {scene_id: index for index, scene_id in enumerate(ids)}
    feature_index = {name: index for index, name in enumerate(FEATURES)}
    rows: list[dict[str, Any]] = []
    for role, role_ids in roles.items():
        indices = np.asarray([by_id[item] for item in role_ids], dtype=np.int64)
        for feature in FEATURES:
            column = matrix[:, feature_index[feature]]
            selected = column[indices]
            pool_sd = float(np.std(column, ddof=0))
            smd = abs(float(np.mean(selected) - np.mean(column))) / max(pool_sd, 1e-12)
            q10, q90 = np.quantile(column, [0.10, 0.90])
            record: dict[str, Any] = {
                "role": role,
                "feature": feature,
                "scene_count": len(role_ids),
                "cdf_gap": empirical_cdf_gap(selected, column),
                "absolute_standardized_mean_difference": smd,
                "lower_tail_fraction": float(np.mean(selected <= q10)),
                "upper_tail_fraction": float(np.mean(selected >= q90)),
                "quartile_0_fraction": None,
                "quartile_1_fraction": None,
                "quartile_2_fraction": None,
                "quartile_3_fraction": None,
            }
            if feature in bins:
                for quartile in range(4):
                    record[f"quartile_{quartile}_fraction"] = float(np.mean(bins[feature][indices] == quartile))
            rows.append(record)
    return rows


def _largest_numeric_2d(value: dict[str, Any]) -> np.ndarray:
    candidates: list[np.ndarray] = []
    for key, item in value.items():
        if key.startswith("__") or not isinstance(item, np.ndarray) or not np.issubdtype(item.dtype, np.number):
            continue
        squeezed = np.squeeze(item)
        if squeezed.ndim == 2 and min(squeezed.shape) >= 8:
            candidates.append(np.asarray(squeezed, dtype=np.float32))
    if not candidates:
        raise ValueError("no numeric two-dimensional depth field")
    return max(candidates, key=lambda item: item.size)


def _resize_float(array: np.ndarray, size: tuple[int, int] = (64, 64)) -> np.ndarray:
    return np.asarray(Image.fromarray(array.astype(np.float32), mode="F").resize(size, Image.Resampling.BILINEAR), dtype=np.float32)


def scene_features(scene_id: str, clear_path: Path, depth_path: Path) -> tuple[str, list[float] | None, str | None]:
    try:
        with Image.open(clear_path) as image:
            rgb = np.asarray(image.convert("RGB").resize((64, 64), Image.Resampling.LANCZOS), dtype=np.float32) / 255.0
        luminance = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
        gy, gx = np.gradient(luminance)
        rg = rgb[..., 0] - rgb[..., 1]
        yb = 0.5 * (rgb[..., 0] + rgb[..., 1]) - rgb[..., 2]
        colorfulness = math.sqrt(float(np.var(rg) + np.var(yb))) + 0.3 * math.sqrt(float(np.mean(rg) ** 2 + np.mean(yb) ** 2))
        depth = _largest_numeric_2d(loadmat(depth_path))
        if not np.all(np.isfinite(depth)) or float(np.min(depth)) < 0.0:
            raise ValueError("depth field is nonfinite or negative")
        log_depth = np.log1p(_resize_float(depth))
        dgy, dgx = np.gradient(log_depth)
        d10, d90 = np.quantile(log_depth, [0.10, 0.90])
        depth_range = float(d90 - d10)
        values = [
            float(np.mean(luminance)),
            float(np.std(luminance)),
            float(np.mean(np.hypot(gx, gy))),
            colorfulness,
            float(np.mean(log_depth)),
            float(np.std(log_depth)),
            depth_range,
            float(np.mean(np.hypot(dgx, dgy)) / max(depth_range, 1e-6)),
        ]
        if not np.all(np.isfinite(values)):
            raise ValueError("derived descriptor is nonfinite")
        return scene_id, values, None
    except Exception as exc:  # converted to an integrity terminal, not an engineering crash
        return scene_id, None, f"{type(exc).__name__}:{exc}"


def _haze_grid(paths: list[Path]) -> tuple[dict[str, set[str]], set[tuple[float, float]], dict[tuple[float, float], str], list[str]]:
    suffixes: dict[str, set[str]] = defaultdict(set)
    errors: list[str] = []
    pair_to_suffix: dict[tuple[float, float], str] = {}
    all_pairs: set[tuple[float, float]] = set()
    for path in paths:
        parts = path.stem.split("_")
        if len(parts) != 3:
            if len(errors) < 20:
                errors.append(f"invalid_haze_stem:{path.stem}")
            continue
        scene_id, first_raw, second_raw = parts
        suffix = f"{first_raw}_{second_raw}"
        try:
            pair = (float(first_raw), float(second_raw))
        except ValueError:
            if len(errors) < 20:
                errors.append(f"invalid_haze_parameters:{path.stem}")
            continue
        suffixes[scene_id].add(suffix)
        all_pairs.add(pair)
        previous = pair_to_suffix.setdefault(pair, suffix)
        if previous != suffix and len(errors) < 20:
            errors.append(f"inconsistent_parameter_format:{previous}:{suffix}")
    return dict(suffixes), all_pairs, pair_to_suffix, errors


def _variant_recipe(pairs: set[tuple[float, float]], pair_to_suffix: dict[tuple[float, float], str]) -> dict[str, Any]:
    first = sorted({item[0] for item in pairs})
    second = sorted({item[1] for item in pairs})
    selected_pairs = [
        (first[0], second[0]),
        (first[0], second[-1]),
        (first[-1], second[0]),
        (first[-1], second[-1]),
        (first[len(first) // 2], second[len(second) // 2]),
    ]
    return {
        "parameter_axes": {"first": first, "second": second},
        "full_grid_size": len(pairs),
        "training_and_model_development_pairs": [
            {"first": item[0], "second": item[1], "filename_suffix": pair_to_suffix[item]}
            for item in selected_pairs
        ],
        "training_scenes": ROLE_QUOTAS["training"],
        "training_hazy_pairs": ROLE_QUOTAS["training"] * len(selected_pairs),
        "model_development_scenes": ROLE_QUOTAS["model_development"],
        "model_development_hazy_pairs": ROLE_QUOTAS["model_development"] * len(selected_pairs),
        "measurement_definition_scenes": ROLE_QUOTAS["measurement_definition"],
        "measurement_definition_retained_hazy_pairs": ROLE_QUOTAS["measurement_definition"] * len(pairs),
        "measurement_validation_scenes": ROLE_QUOTAS["measurement_validation"],
        "measurement_validation_retained_hazy_pairs": ROLE_QUOTAS["measurement_validation"] * len(pairs),
        "reserve_scenes": ROLE_QUOTAS["reserve"],
        "reserve_release_policy": "unused unless a later preregistered precision or failure trigger explicitly releases named scenes",
    }


def contract(context_path: Path) -> None:
    context = load_context(context_path, "contract")
    prepare_phase_output(context)
    parent = context.assets.get("parent_closeout")
    checks = {
        "cpu_exact_mode": context.engineering_contract["mode"] == "cpu_exact",
        "cpu_contract": context.device == "cpu",
        "dataset_hidden_from_contract": "reside_root" not in context.assets,
        "parent_identity_bound": parent is not None and parent.contract_access is True,
        "protected_roles_disabled": not any(context.protected_data_permissions.values()),
        "no_model_training_or_inference_path": True,
    }
    generator = np.random.default_rng(3407)
    ids = [f"synthetic-{index:05d}" for index in range(EXPECTED_ELIGIBLE)]
    matrix = generator.normal(size=(EXPECTED_ELIGIBLE, len(FEATURES))).astype(np.float64)
    started = time.monotonic()
    roles, bins = allocate_roles(ids, matrix)
    rows = coverage_rows(ids, matrix, roles, bins)
    elapsed = time.monotonic() - started
    union = set().union(*(set(items) for items in roles.values()))
    checks.update({
        "same_scale_exact_role_counts": all(len(roles[role]) == quota for role, quota in ROLE_QUOTAS.items()),
        "same_scale_disjoint_union": len(union) == EXPECTED_ELIGIBLE and sum(len(items) for items in roles.values()) == EXPECTED_ELIGIBLE,
        "same_scale_complete_coverage_rows": len(rows) == len(ROLE_QUOTAS) * len(FEATURES),
        "same_scale_elapsed_bound": elapsed <= 60.0,
        "same_scale_memory_bound": matrix.nbytes <= 1024 * 1024,
        "output_and_finalizer_contract": True,
    })
    write_contract_result(
        context,
        checks=checks,
        engineering={
            "mode": "cpu_exact",
            "device": "cpu",
            "fixture": {"batch": EXPECTED_ELIGIBLE, "channels": len(FEATURES), "height": 1, "width": len(ROLE_QUOTAS)},
            "production_path_exercised": True,
            "protected_data_touched": False,
            "scientific_output_created": False,
            "scientific_training_occurred": False,
        },
    )


def run(context_path: Path) -> None:
    context = load_context(context_path, "run")
    prepare_phase_output(context)
    if context.total_units != TOTAL_UNITS:
        raise ValueError("runtime total_units differs from the frozen scene census")
    parent = json.loads(asset_path(context, "parent_closeout", kind="file").read_text(encoding="utf-8"))
    exclusions = {
        line.strip() for line in asset_path(context, "authorized_exclusions", kind="file").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    parent_authorization = (
        parent.get("state") == "COMPLETED_GATE_PASS"
        and parent.get("decision") == "OTS_TARGETED_GEOMETRY_PASS"
        and parent.get("authorizes") == "OTS_OUTDOOR_MEASUREMENT_DESIGN"
        and parent.get("details", {}).get("deduplicated_exclusion_count") == EXPECTED_EXCLUSIONS
        and parent.get("details", {}).get("eligible_ots_scenes") == EXPECTED_ELIGIBLE
        and parent.get("details", {}).get("exclusion_digest") == digest_lines(exclusions)
        and len(exclusions) == EXPECTED_EXCLUSIONS
    )
    root = asset_path(context, "reside_root", kind="directory") / "official/OTS_ALPHA"
    clear_paths = image_files(root / "clear_images", ".jpg")
    depth_paths = image_files(root / "depth", ".mat")
    haze_paths = image_files(root / "OTS", ".jpg")
    clear_index = {path.stem: path for path in clear_paths}
    depth_index = {path.stem: path for path in depth_paths}
    eligible = sorted(set(clear_index) - exclusions)
    haze_suffixes, haze_pairs, pair_to_suffix, haze_errors = _haze_grid(haze_paths)
    expected_suffixes = next(iter(haze_suffixes.values()), set())
    axes = ({item[0] for item in haze_pairs}, {item[1] for item in haze_pairs})
    haze_grid_complete = (
        not haze_errors
        and len(haze_pairs) == EXPECTED_GRID
        and len(axes[0]) == 5
        and len(axes[1]) == 7
        and all(haze_suffixes.get(scene_id) == expected_suffixes for scene_id in eligible)
    )
    dataset_pairing = (
        len(clear_paths) == EXPECTED_CLEAR
        and len(depth_paths) == EXPECTED_DEPTH
        and len(haze_paths) == EXPECTED_HAZE
        and set(clear_index) == set(depth_index) == set(haze_suffixes)
        and exclusions.issubset(clear_index)
        and len(eligible) == EXPECTED_ELIGIBLE
    )

    records: dict[str, list[float]] = {}
    feature_errors: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = executor.map(
            lambda scene_id: scene_features(scene_id, clear_index[scene_id], depth_index[scene_id]),
            eligible,
            chunksize=8,
        )
        for index, (scene_id, values, error) in enumerate(futures, start=1):
            if error is None and values is not None:
                records[scene_id] = values
            else:
                feature_errors[scene_id] = error or "unknown_feature_error"
            if index == 1 or index % 100 == 0 or index == len(eligible):
                write_workload_progress(context, completed_units=index, stage="ots_scene_descriptor_census")
    depth_available = len(records) == EXPECTED_ELIGIBLE and not feature_errors

    roles = {role: [] for role in ROLE_QUOTAS}
    coverage: list[dict[str, Any]] = []
    bins: dict[str, np.ndarray] = {}
    if parent_authorization and dataset_pairing and depth_available and haze_grid_complete:
        matrix = np.asarray([records[item] for item in eligible], dtype=np.float64)
        roles, bins = allocate_roles(eligible, matrix)
        coverage = coverage_rows(eligible, matrix, roles, bins)
    union = set().union(*(set(items) for items in roles.values()))
    exact_counts = all(len(roles[role]) == quota for role, quota in ROLE_QUOTAS.items())
    disjoint_union = (
        sum(len(items) for items in roles.values()) == len(union) == EXPECTED_ELIGIBLE
        and union == set(eligible)
    )
    worst_cdf = max((row["cdf_gap"] for row in coverage), default=float("inf"))
    worst_smd = max((row["absolute_standardized_mean_difference"] for row in coverage), default=float("inf"))
    quartile_values = [
        row[f"quartile_{quartile}_fraction"]
        for row in coverage if row["feature"] in ALLOCATION_FEATURES
        for quartile in range(4)
    ]
    minimum_quartile = min(quartile_values, default=0.0)
    tail_values = [value for row in coverage for value in (row["lower_tail_fraction"], row["upper_tail_fraction"])]
    minimum_tail = min(tail_values, default=0.0)
    recipe = _variant_recipe(haze_pairs, pair_to_suffix) if haze_grid_complete else {
        "error": "official eligible haze grid did not meet the frozen 5 by 7 contract",
        "observed_pair_count": len(haze_pairs),
        "parse_errors": haze_errors,
    }
    training_budget = (
        haze_grid_complete
        and recipe.get("training_hazy_pairs") == 20000
        and recipe.get("model_development_hazy_pairs") == 2500
        and len(recipe.get("training_and_model_development_pairs", [])) == 5
    )
    gates = {
        "parent_authorization": parent_authorization,
        "dataset_pairing": dataset_pairing,
        "depth_field_availability": depth_available,
        "haze_grid_completeness": haze_grid_complete,
        "role_counts": exact_counts,
        "scene_disjointness": disjoint_union,
        "worst_cdf_gap": worst_cdf <= 0.08,
        "worst_standardized_mean_difference": worst_smd <= 0.10,
        "marginal_quartile_coverage": minimum_quartile >= 0.15,
        "tail_coverage": minimum_tail >= 0.05,
        "training_haze_budget": training_budget,
    }
    integrity = ["parent_authorization", "dataset_pairing", "depth_field_availability", "haze_grid_completeness"]
    if not all(gates[key] for key in integrity):
        state = "COMPLETED_INCONCLUSIVE"
        decision = "OTS_SCENE_STRATIFIED_SPLIT_INCONCLUSIVE"
        authorizes = "NONE"
    elif all(gates.values()):
        state = "COMPLETED_GATE_PASS"
        decision = "OTS_SCENE_STRATIFIED_SPLIT_PASS"
        authorizes = "OTS_OUTDOOR_MEASUREMENT_DEFINITION"
    else:
        state = "COMPLETED_GATE_FAIL"
        decision = "OTS_SCENE_STRATIFIED_SPLIT_FAIL"
        authorizes = "NONE"
    gate_reasons = [key for key, passed in gates.items() if not passed]
    if not gate_reasons:
        gate_reasons = ["all_frozen_identity_pairing_role_coverage_and_budget_gates_passed"]

    summary = {
        "schema_version": 1,
        "route_id": context.route_id,
        "operation_id": context.operation_id,
        "run_id": context.run_id,
        "scope": "scene-level stratified role and haze-budget design for the source-independent OTS pool",
        "parent": {
            "authorization_matched": parent_authorization,
            "excluded_scenes": len(exclusions),
            "eligible_scenes": len(eligible),
            "exclusion_digest": digest_lines(exclusions),
        },
        "dataset": {
            "clear_files": len(clear_paths),
            "depth_files": len(depth_paths),
            "haze_files": len(haze_paths),
            "paired_id_sets_equal": set(clear_index) == set(depth_index) == set(haze_suffixes),
            "valid_scene_descriptor_count": len(records),
            "invalid_scene_descriptor_count": len(feature_errors),
            "first_descriptor_errors": dict(list(sorted(feature_errors.items()))[:20]),
            "haze_parameter_pair_count": len(haze_pairs),
            "haze_first_axis_count": len(axes[0]),
            "haze_second_axis_count": len(axes[1]),
            "first_haze_parse_errors": haze_errors,
        },
        "allocation": {
            "method": "four marginal rank-quartiles, joint-stratum round-robin, and weighted-deficit role allocation",
            "seed": SEED,
            "role_counts": {role: len(items) for role, items in roles.items()},
            "role_digests": {role: digest_lines(items) for role, items in roles.items()},
            "pairwise_disjoint_complete_union": disjoint_union,
            "allocation_features": list(ALLOCATION_FEATURES),
            "coverage_features": list(FEATURES),
        },
        "coverage": {
            "worst_cdf_gap": worst_cdf if math.isfinite(worst_cdf) else None,
            "maximum_allowed_cdf_gap": 0.08,
            "worst_absolute_standardized_mean_difference": worst_smd if math.isfinite(worst_smd) else None,
            "maximum_allowed_absolute_standardized_mean_difference": 0.10,
            "minimum_marginal_quartile_fraction": minimum_quartile,
            "minimum_required_marginal_quartile_fraction": 0.15,
            "minimum_tail_fraction": minimum_tail,
            "minimum_required_tail_fraction": 0.05,
        },
        "haze_budget": recipe,
        "gates": gates,
        "terminal": {"state": state, "decision": decision, "authorizes": authorizes, "gate_reasons": gate_reasons},
        "limitations": [
            "The split uses observable clear-image and depth descriptors; it does not guarantee semantic-category balance beyond those descriptors.",
            "Haze filename parameters are controlled generation settings, not validated local haze or restoration-demand labels.",
            "The measurement-validation role is protected from measurement-definition selection but remains development-screening evidence, not final model confirmation.",
            "No model training, inference, checkpoint selection, SOTS outcome, Haze4K outcome, NH-HAZE outcome, canary, confirmation, or locked-test metric is accessed.",
        ],
        "marker": "RESIDE_OTS_SCENE_STRATIFIED_SPLIT_COMPLETE",
    }
    output_file(context, "split_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    coverage_fields = [
        "role", "feature", "scene_count", "cdf_gap", "absolute_standardized_mean_difference",
        "lower_tail_fraction", "upper_tail_fraction", "quartile_0_fraction", "quartile_1_fraction",
        "quartile_2_fraction", "quartile_3_fraction",
    ]
    with output_file(context, "coverage_audit.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=coverage_fields)
        writer.writeheader()
        writer.writerows(coverage)
    filenames = {
        "training": "training_scene_ids.txt",
        "model_development": "model_development_scene_ids.txt",
        "measurement_definition": "measurement_definition_scene_ids.txt",
        "measurement_validation": "measurement_validation_scene_ids.txt",
        "reserve": "reserve_scene_ids.txt",
    }
    for role, filename in filenames.items():
        payload = "\n".join(roles[role])
        output_file(context, filename).write_text(payload + ("\n" if payload else ""), encoding="utf-8")
    output_file(context, "haze_variant_recipe.json").write_text(json.dumps(recipe, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_workload_progress(context, completed_units=TOTAL_UNITS, stage="ots_scene_split_finalize")
    write_run_result(
        context,
        state=state,
        decision=decision,
        authorizes=authorizes,
        details={
            "summary_file": "split_summary.json",
            "role_counts": {role: len(items) for role, items in roles.items()},
            "worst_cdf_gap": worst_cdf if math.isfinite(worst_cdf) else None,
            "worst_absolute_standardized_mean_difference": worst_smd if math.isfinite(worst_smd) else None,
            "minimum_marginal_quartile_fraction": minimum_quartile,
            "minimum_tail_fraction": minimum_tail,
            "training_hazy_pairs": recipe.get("training_hazy_pairs"),
            "model_development_hazy_pairs": recipe.get("model_development_hazy_pairs"),
            "gate_reasons": gate_reasons,
            "model_training_executed": False,
            "outdoor_measurement_executed": False,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("contract", "run"))
    parser.add_argument("--context", required=True, type=Path)
    arguments = parser.parse_args()
    if arguments.phase == "contract":
        contract(arguments.context)
    else:
        run(arguments.context)


if __name__ == "__main__":
    main()
