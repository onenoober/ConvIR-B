#!/usr/bin/env python3
"""Frozen R12 action-conditioned severe-downside observability screen."""

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
    asset_path, atomic_json, load_context, prepare_phase_output,
    write_contract_result, write_run_result, write_workload_progress,
)

ROUTE_ID = "haze4k_v5_r12_action_conditioned_downside_observability_20260719"
OPERATION_ID = "R12_A0_ACTION_CONDITIONED_DOWNSIDE_OBSERVABILITY_SCREEN"
R11_ROUTE_ID = "haze4k_v5_r11_regional_action_observability_20260719"
R11_OPERATION_ID = "R11_A0_REGIONAL_ACTION_OBSERVABILITY_SCREEN"
R11_RUN_ID = "r11-a0-regional-observability-r1"
R11_ROUTE_COMMIT = "c183817e2b3befdeeb12278aa6e6a0574883b6d5"
RUNNER_SHA256 = "336c7e1beccb793229beb533ba12367261e702866497c388ee2a4fa88d12718b"
CELLS = ("ACTION_CONDITIONED", "ACTION_AGNOSTIC", "SIGN_SWAP", "LABEL_SHUFFLE")
PRIMARY = CELLS[0]
FOLDS = (0, 1)
EXPECTED_NAMES = 384
EXPECTED_ROWS = 49152
FEATURE_DIM = 10
BOOTSTRAP_DRAWS = 4000
BOOTSTRAP_SEED = 3407
SEVERE = -0.2
REJECT_FRACTION = 0.2
L2 = 1.0e-4
MAX_ITER = 200
AUROC_MIN = 0.75
CONTROL_INCREMENT = 0.03
SHUFFLE_INCREMENT = 0.10
CAPTURE_MIN = 0.60
RETAINED_RATIO_MAX = 0.60
HEADER = {
    "action", "actual_mean", "actual_worst", "eligible", "fold", "name",
    "predicted_eligible", "predicted_mean", "predicted_worst", "tile",
}
ENVIRONMENT = {
    "CONVIR_ROUTE_BOOTSTRAP_DRAWS": str(BOOTSTRAP_DRAWS),
    "CONVIR_ROUTE_BOOTSTRAP_SEED": str(BOOTSTRAP_SEED),
    "CONVIR_ROUTE_SEVERE_GAIN_DB": str(SEVERE),
    "CONVIR_ROUTE_REJECT_FRACTION": str(REJECT_FRACTION),
    "CONVIR_ROUTE_L2": str(L2),
    "CONVIR_ROUTE_MAX_ITER": str(MAX_ITER),
}


class DownsideInconclusive(RuntimeError):
    """Typed scientific-input or numerical stop."""


def verify_environment() -> None:
    bad = {key: os.environ.get(key) for key, value in ENVIRONMENT.items()
           if os.environ.get(key) != value}
    if bad:
        raise DownsideInconclusive(f"frozen environment mismatch: {sorted(bad)}")


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DownsideInconclusive(f"invalid JSON: {path.name}") from exc
    if not isinstance(value, dict):
        raise DownsideInconclusive(f"JSON is not an object: {path.name}")
    return value


def read_rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None or set(reader.fieldnames) != HEADER:
                raise DownsideInconclusive("R11 tile prediction header mismatch")
            rows = list(reader)
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise DownsideInconclusive("R11 tile predictions are unreadable") from exc
    if len(rows) != EXPECTED_ROWS:
        raise DownsideInconclusive("R11 tile prediction row count mismatch")
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    fields = sorted({key for row in rows for key in row})
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sigmoid(values: Any) -> Any:
    import numpy as np
    values = np.clip(np.asarray(values, dtype=np.float64), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-values))


def base_features(rows: list[dict[str, str]]) -> Any:
    import numpy as np
    output = []
    for row in rows:
        action_sign = 1.0 if int(row["action"]) == 1 else -1.0
        tile = int(row["tile"]); y, x = divmod(tile, 8)
        mean = float(row["predicted_mean"]); worst = float(row["predicted_worst"])
        continuous = [mean, worst, mean - worst]
        output.append(continuous + [2.0 * (y + 0.5) / 8.0 - 1.0,
                      2.0 * (x + 0.5) / 8.0 - 1.0, action_sign]
                      + [action_sign * value for value in continuous]
                      + [action_sign * (2.0 * (x + 0.5) / 8.0 - 1.0)])
    value = np.asarray(output, dtype=np.float64)
    if value.shape != (len(rows), FEATURE_DIM) or not np.isfinite(value).all():
        raise DownsideInconclusive("feature construction failed")
    return value


def cell_features(rows: list[dict[str, str]], primary: Any, cell: str) -> Any:
    import numpy as np
    value = primary.copy()
    if cell == "ACTION_AGNOSTIC":
        value[:, 5:] = 0.0
    elif cell == "SIGN_SWAP":
        lookup = {(row["name"], int(row["tile"]), int(row["action"])): index
                  for index, row in enumerate(rows)}
        for index, row in enumerate(rows):
            other = lookup[(row["name"], int(row["tile"]), 3 - int(row["action"]))]
            sign = value[index, 5]
            value[index, :3] = primary[other, :3]
            value[index, 6:9] = sign * value[index, :3]
    return value


def shuffled_training_labels(rows: list[dict[str, str]], labels: Any, indices: Any) -> Any:
    import numpy as np
    result = labels[indices].copy()
    positions: dict[str, list[int]] = {}
    for local, index in enumerate(indices):
        positions.setdefault(rows[int(index)]["name"], []).append(local)
    for name, local_indices in positions.items():
        digest = hashlib.sha256(f"{ROUTE_ID}|{name}|label-shuffle".encode()).digest()
        generator = np.random.default_rng(int.from_bytes(digest[:8], "big"))
        values = result[local_indices].copy()
        result[local_indices] = values[generator.permutation(len(values))]
    return result


def fit_logistic(x: Any, y: Any) -> tuple[Any, dict[str, Any]]:
    import numpy as np
    from scipy.optimize import minimize
    positive = max(float(np.sum(y)), 1.0); negative = max(float(len(y) - np.sum(y)), 1.0)
    weights = np.where(y > 0.5, 0.5 * len(y) / positive, 0.5 * len(y) / negative)
    design = np.concatenate((np.ones((len(x), 1)), x), axis=1)
    def objective(beta: Any) -> tuple[float, Any]:
        z = design @ beta
        loss = float(np.mean(weights * (np.logaddexp(0.0, z) - y * z))
                     + 0.5 * L2 * np.sum(beta[1:] ** 2))
        gradient = design.T @ (weights * (sigmoid(z) - y)) / len(y)
        gradient[1:] += L2 * beta[1:]
        return loss, gradient
    initial = np.zeros(design.shape[1], dtype=np.float64)
    initial_loss = objective(initial)[0]
    result = minimize(objective, initial, jac=True, method="L-BFGS-B",
                      options={"maxiter": MAX_ITER, "ftol": 1.0e-12, "gtol": 1.0e-8})
    if not result.success or not np.isfinite(result.x).all() or not math.isfinite(result.fun):
        raise DownsideInconclusive(f"risk optimizer failed: {result.message}")
    if result.fun >= initial_loss:
        raise DownsideInconclusive("risk optimizer did not reduce loss")
    return result.x, {"initial_loss": initial_loss, "final_loss": float(result.fun),
                      "iterations": int(result.nit), "converged": bool(result.success)}


def predict(beta: Any, x: Any) -> Any:
    import numpy as np
    design = np.concatenate((np.ones((len(x), 1)), x), axis=1)
    result = sigmoid(design @ beta)
    if not np.isfinite(result).all():
        raise DownsideInconclusive("non-finite risk score")
    return result


def row_metrics(labels: Any, scores: Any, reject_fraction: float = REJECT_FRACTION) -> dict[str, float]:
    import numpy as np
    labels = np.asarray(labels, dtype=np.int64); scores = np.asarray(scores, dtype=np.float64)
    weights = np.ones(len(labels), dtype=np.float64)
    positive = float(np.sum(labels)); negative = float(len(labels) - positive)
    if positive <= 0.0 or negative <= 0.0:
        raise DownsideInconclusive("risk metric lacks both label classes")
    order = np.argsort(scores, kind="mergesort")
    y = labels[order]; w = weights[order]; s = scores[order]
    cumulative_negative = np.cumsum(w * (1 - y))
    before_negative = cumulative_negative - w * (1 - y)
    starts = np.r_[0, np.flatnonzero(np.diff(s) != 0.0) + 1]
    tie_positive = np.add.reduceat(w * y, starts)
    tie_negative = np.add.reduceat(w * (1 - y), starts)
    negative_before = np.cumsum(tie_negative) - tie_negative
    auroc = float(np.sum(tie_positive * (negative_before + 0.5 * tie_negative))
                  / (positive * negative))
    descending = np.argsort(-scores, kind="mergesort")
    total_weight = float(len(labels)); budget = reject_fraction * total_weight
    cumulative = np.cumsum(weights[descending]); before = cumulative - weights[descending]
    take = np.clip(budget - before, 0.0, weights[descending])
    rejected = np.zeros(len(labels), dtype=np.float64); rejected[descending] = take
    captured = float(np.sum(rejected * labels) / positive)
    kept = weights - rejected
    baseline_prevalence = positive / total_weight
    retained_prevalence = float(np.sum(kept * labels) / np.sum(kept))
    return {"auroc": auroc, "severe_capture_at_20pct": captured,
            "retained_severe_prevalence_ratio": retained_prevalence / baseline_prevalence}


def interval(point: float, samples: list[float]) -> dict[str, float]:
    import numpy as np
    values = np.asarray(samples, dtype=np.float64)
    return {"point": float(point), "lcb95": float(np.quantile(values, 0.025)),
            "ucb95": float(np.quantile(values, 0.975))}


def per_group_metrics(labels: Any, scores: dict[str, Any], group_ids: Any, group_count: int,
                      reject_fraction: float = REJECT_FRACTION) -> dict[str, Any]:
    import numpy as np
    output = {cell: {metric: np.full(group_count, np.nan, dtype=np.float64)
                     for metric in ("auroc", "severe_capture_at_20pct",
                                    "retained_severe_prevalence_ratio")} for cell in CELLS}
    for group in range(group_count):
        subset = group_ids == group
        if np.sum(labels[subset]) <= 0 or np.sum(labels[subset]) >= np.sum(subset):
            continue
        for cell in CELLS:
            values = row_metrics(labels[subset], scores[cell][subset], reject_fraction)
            for metric, value in values.items():
                output[cell][metric][group] = value
    return output


def grouped_bootstrap(labels: Any, scores: dict[str, Any], group_ids: Any, group_folds: Any) -> dict[str, Any]:
    import numpy as np
    group_values = per_group_metrics(labels, scores, group_ids, len(group_folds))
    point = {cell: {metric: float(np.nanmean(values)) for metric, values in metrics.items()}
             for cell, metrics in group_values.items()}
    samples: dict[str, list[float]] = {}
    for cell in CELLS:
        for metric in point[cell]: samples[f"{cell}_{metric}"] = []
    for control in CELLS[1:]: samples[f"primary_minus_{control}_auroc"] = []
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    fold_groups = {fold: np.flatnonzero(group_folds == fold) for fold in FOLDS}
    for _ in range(BOOTSTRAP_DRAWS):
        chosen_all = []
        for fold in FOLDS:
            chosen = generator.choice(fold_groups[fold], len(fold_groups[fold]), replace=True)
            chosen_all.append(chosen)
        chosen = np.concatenate(chosen_all)
        values = {cell: {metric: float(np.nanmean(array[chosen]))
                         for metric, array in metrics.items()}
                  for cell, metrics in group_values.items()}
        for cell in CELLS:
            for metric, value in values[cell].items(): samples[f"{cell}_{metric}"].append(value)
        for control in CELLS[1:]:
            samples[f"primary_minus_{control}_auroc"].append(
                values[PRIMARY]["auroc"] - values[control]["auroc"])
    result = {}
    for cell in CELLS:
        for metric, value in point[cell].items():
            key = f"{cell}_{metric}"; result[key] = interval(value, samples[key])
    for control in CELLS[1:]:
        key = f"primary_minus_{control}_auroc"
        result[key] = interval(point[PRIMARY]["auroc"] - point[control]["auroc"], samples[key])
    return result


def synthetic_contract() -> dict[str, bool]:
    import numpy as np
    started = time.perf_counter(); generator = np.random.default_rng(3407)
    rows = []
    for group in range(EXPECTED_NAMES):
        for tile in range(64):
            latent = generator.normal()
            for action in (1, 2):
                sign = 1.0 if action == 1 else -1.0
                predicted_mean = 0.15 * latent + 0.05 * sign + generator.normal(scale=0.20)
                predicted_worst = predicted_mean - abs(generator.normal(scale=0.18))
                rows.append({"name": f"synthetic_{group:04d}",
                    "fold": str(group // 192), "tile": str(tile), "action": str(action),
                    "predicted_mean": str(predicted_mean),
                    "predicted_worst": str(predicted_worst),
                    "actual_worst": "0.0", "actual_mean": "0.0",
                    "eligible": "0", "predicted_eligible": "0"})
    groups = np.repeat(np.arange(EXPECTED_NAMES), 128)
    row_folds = np.asarray([int(row["fold"]) for row in rows], dtype=np.int64)
    primary = base_features(rows)
    probability = 0.04 + 0.42 * sigmoid(-1.6 * primary[:, 1] + 0.9 * primary[:, 6])
    labels = (generator.random(EXPECTED_ROWS) < probability).astype(np.float64)
    for group in range(EXPECTED_NAMES):
        labels[group * 128] = 0.0
        labels[group * 128 + 1] = 1.0
    features = {cell: cell_features(rows, primary, cell) for cell in CELLS}
    scores = {cell: np.empty(EXPECTED_ROWS) for cell in CELLS}
    all_fits_valid = True
    for cell in CELLS:
        for fold in FOLDS:
            train = np.flatnonzero(row_folds != fold); test = np.flatnonzero(row_folds == fold)
            mean = primary[train].mean(0); std = primary[train].std(0)
            std = np.where(std >= 1.0e-6, std, 1.0)
            train_labels = shuffled_training_labels(rows, labels, train) if cell == "LABEL_SHUFFLE" else labels[train]
            beta, summary = fit_logistic((features[cell][train] - mean) / std, train_labels)
            scores[cell][test] = predict(beta, (features[cell][test] - mean) / std)
            all_fits_valid = all_fits_valid and summary["converged"] \
                and summary["final_loss"] < summary["initial_loss"]
    boot = grouped_bootstrap(labels, scores, groups, np.arange(EXPECTED_NAMES) // 192)
    return {"formal_rows": len(labels) == EXPECTED_ROWS,
            "all_production_cells_exercised": all(features[cell].shape == (EXPECTED_ROWS, FEATURE_DIM)
                                                     for cell in CELLS),
            "all_fits_valid": bool(all_fits_valid),
            "all_scores_finite": all(np.isfinite(value).all() for value in scores.values()),
            "full_bootstrap_finite": all(math.isfinite(value["point"]) for value in boot.values()),
            "formal_wall_bounded": time.perf_counter() - started <= 120.0,
            "formal_peak_memory_under_1024_mib":
                float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0 <= 1024.0}


def contract(context_path: Path) -> None:
    context = load_context(context_path, "contract"); prepare_phase_output(context); verify_environment()
    checks = {"route_identity": context.route_id == ROUTE_ID and context.operation_id == OPERATION_ID,
              "cpu_only": context.device == "cpu",
              "protected_roles_blocked": not any(context.protected_data_permissions.values()),
              "frozen_dimensions": FEATURE_DIM == 10 and EXPECTED_ROWS == 49152,
              **synthetic_contract()}
    write_contract_result(context, checks=checks)


def write_inconclusive(context: Any, reason: str, started_wall: float, started_cpu: float) -> None:
    common = {"schema_version": 1, "status": "input_or_downside_inconclusive", "reason": reason}
    for filename in ("r12_a0_contract_summary.json", "r12_a0_provenance_and_access.json",
                     "r12_a0_input_identity.json", "r12_a0_bootstrap_summary.json",
                     "r12_a0_gate_summary.json"):
        atomic_json(context.phase_output_path / filename, common)
    placeholder = [{"status": "inconclusive", "reason": reason}]
    for filename in ("r12_a0_cell_summary.csv", "r12_a0_fold_stability.csv",
                     "r12_a0_risk_coverage.csv"):
        write_csv(context.phase_output_path / filename, placeholder)
    atomic_json(context.phase_output_path / "r12_a0_resource_summary.json", {**common,
        "wall_seconds": time.perf_counter() - started_wall,
        "cpu_seconds": time.process_time() - started_cpu, "gpu_used": False})
    write_run_result(context, state="COMPLETED_GATE_INCONCLUSIVE",
                     decision="R12_A0_INPUT_OR_DOWNSIDE_INCONCLUSIVE_STOP",
                     authorizes="NONE", details={"reason": reason, "r11_terminal_changed": False})


def run(context_path: Path) -> None:
    import numpy as np
    started_wall = time.perf_counter(); started_cpu = time.process_time()
    context = load_context(context_path, "run"); prepare_phase_output(context)
    try:
        verify_environment()
        closeout = read_json(asset_path(context, "r11_closeout", kind="file"))
        if (closeout.get("route_id"), closeout.get("operation_id"), closeout.get("run_id"),
            closeout.get("route_commit"), closeout.get("state"), closeout.get("decision"),
            closeout.get("authorizes")) != (R11_ROUTE_ID, R11_OPERATION_ID, R11_RUN_ID,
                R11_ROUTE_COMMIT, "COMPLETED_GATE_FAIL",
                "R11_A0_REGIONAL_OBSERVABILITY_FAIL_STOP", "NONE"):
            raise DownsideInconclusive("R11 typed closeout identity mismatch")
        rows = read_rows(asset_path(context, "r11_tile_predictions", kind="file"))
        keys = [(row["name"], int(row["fold"]), int(row["tile"]), int(row["action"])) for row in rows]
        if len(set(keys)) != EXPECTED_ROWS:
            raise DownsideInconclusive("R11 tile prediction keys are not unique")
        names = sorted({row["name"] for row in rows}); fold_by_name = {}
        for row in rows:
            fold = int(row["fold"]); name = row["name"]
            if fold not in FOLDS or (name in fold_by_name and fold_by_name[name] != fold):
                raise DownsideInconclusive("R11 name/fold identity mismatch")
            fold_by_name[name] = fold
        if len(names) != EXPECTED_NAMES or any(sum(value == fold for value in fold_by_name.values()) != 192 for fold in FOLDS):
            raise DownsideInconclusive("R11 group/fold coverage mismatch")
        labels = np.asarray([float(row["actual_worst"]) <= SEVERE for row in rows], dtype=np.float64)
        primary_x = base_features(rows); features = {cell: cell_features(rows, primary_x, cell) for cell in CELLS}
        row_folds = np.asarray([int(row["fold"]) for row in rows]); name_index = {name: i for i, name in enumerate(names)}
        group_ids = np.asarray([name_index[row["name"]] for row in rows], dtype=np.int64)
        group_folds = np.asarray([fold_by_name[name] for name in names], dtype=np.int64)
        scores = {cell: np.empty(EXPECTED_ROWS, dtype=np.float64) for cell in CELLS}
        training_rows = []
        for fold in FOLDS:
            train = np.flatnonzero(row_folds != fold); test = np.flatnonzero(row_folds == fold)
            mean = primary_x[train].mean(0); std = primary_x[train].std(0); std = np.where(std >= 1.0e-6, std, 1.0)
            for cell in CELLS:
                train_labels = shuffled_training_labels(rows, labels, train) if cell == "LABEL_SHUFFLE" else labels[train]
                beta, summary = fit_logistic((features[cell][train] - mean) / std, train_labels)
                scores[cell][test] = predict(beta, (features[cell][test] - mean) / std)
                training_rows.append({"cell": cell, "test_fold": fold, **summary})
                write_workload_progress(context, completed_units=len(training_rows), stage="cross_fold_risk_fits")
        if not all(np.isfinite(value).all() for value in scores.values()):
            raise DownsideInconclusive("OOF risk scores are non-finite")
        boot = grouped_bootstrap(labels, scores, group_ids, group_folds)
        write_workload_progress(context, completed_units=12, stage="grouped_bootstrap_complete")
        cell_rows = []
        risk_rows = []
        fold_rows = []
        group_values = per_group_metrics(labels, scores, group_ids, len(names))
        point = {cell: {metric: float(np.nanmean(values)) for metric, values in metrics.items()}
                 for cell, metrics in group_values.items()}
        for cell in CELLS:
            cell_rows.append({"cell": cell, "rows": EXPECTED_ROWS,
                              "severe_rows": int(labels.sum()), **point[cell]})
            for reject in (0.05, 0.10, 0.20, 0.30):
                descriptive = per_group_metrics(labels, scores, group_ids, len(names), reject)[cell]
                metrics = {key: float(np.nanmean(value)) for key, value in descriptive.items()}
                risk_rows.append({"cell": cell, "reject_fraction": reject, **metrics})
        for fold in FOLDS:
            group_subset = np.flatnonzero(group_folds == fold)
            values = {cell: {metric: float(np.nanmean(array[group_subset]))
                             for metric, array in metrics.items()}
                      for cell, metrics in group_values.items()}
            row = {"fold": fold}
            for cell in CELLS:
                for key, value in values[cell].items(): row[f"{cell}_{key}"] = value
            for control in CELLS[1:]:
                row[f"primary_minus_{control}_auroc"] = values[PRIMARY]["auroc"] - values[control]["auroc"]
            fold_rows.append(row)
        severe_groups = len({row["name"] for row, label in zip(rows, labels) if label > 0.5})
        gates = {
            "primary_auroc_lcb95": boot[f"{PRIMARY}_auroc"]["lcb95"] >= AUROC_MIN,
            "primary_minus_action_agnostic_lcb95": boot["primary_minus_ACTION_AGNOSTIC_auroc"]["lcb95"] >= CONTROL_INCREMENT,
            "primary_minus_sign_swap_lcb95": boot["primary_minus_SIGN_SWAP_auroc"]["lcb95"] >= CONTROL_INCREMENT,
            "primary_minus_label_shuffle_lcb95": boot["primary_minus_LABEL_SHUFFLE_auroc"]["lcb95"] >= SHUFFLE_INCREMENT,
            "severe_capture_lcb95": boot[f"{PRIMARY}_severe_capture_at_20pct"]["lcb95"] >= CAPTURE_MIN,
            "retained_severe_ratio_ucb95": boot[f"{PRIMARY}_retained_severe_prevalence_ratio"]["ucb95"] <= RETAINED_RATIO_MAX,
            "both_folds_material": all(row[f"{PRIMARY}_auroc"] >= 0.70
                and row[f"{PRIMARY}_severe_capture_at_20pct"] >= 0.50
                and all(row[f"primary_minus_{control}_auroc"] > 0.0 for control in CELLS[1:])
                for row in fold_rows),
            "severe_coverage": int(labels.sum()) >= 1000 and severe_groups >= 300,
        }
        passes = all(gates.values())
        decisive = (boot[f"{PRIMARY}_auroc"]["ucb95"] < AUROC_MIN
            or boot["primary_minus_ACTION_AGNOSTIC_auroc"]["ucb95"] < CONTROL_INCREMENT
            or boot["primary_minus_SIGN_SWAP_auroc"]["ucb95"] < CONTROL_INCREMENT
            or boot["primary_minus_LABEL_SHUFFLE_auroc"]["ucb95"] < SHUFFLE_INCREMENT
            or boot[f"{PRIMARY}_severe_capture_at_20pct"]["ucb95"] < CAPTURE_MIN
            or boot[f"{PRIMARY}_retained_severe_prevalence_ratio"]["lcb95"] > RETAINED_RATIO_MAX
            or not gates["both_folds_material"] or not gates["severe_coverage"])
        if passes:
            state, decision, authorizes = ("COMPLETED_GATE_PASS",
                "R12_A0_ACTION_CONDITIONED_DOWNSIDE_PASS",
                "R12_DECOMPOSED_UTILITY_RISK_DECISION_CONTRACT_REVIEW_ONLY")
        elif decisive:
            state, decision, authorizes = ("COMPLETED_GATE_FAIL",
                "R12_A0_ACTION_CONDITIONED_DOWNSIDE_FAIL_STOP", "NONE")
        else:
            state, decision, authorizes = ("COMPLETED_GATE_INCONCLUSIVE",
                "R12_A0_INPUT_OR_DOWNSIDE_INCONCLUSIVE_STOP", "NONE")
        structural = {"r11_terminal_exact": True, "rows_complete": len(rows) == EXPECTED_ROWS,
            "groups_complete": len(names) == EXPECTED_NAMES, "keys_unique": len(set(keys)) == EXPECTED_ROWS,
            "folds_complete": all(sum(value == fold for value in fold_by_name.values()) == 192 for fold in FOLDS),
            "finite_scores": all(np.isfinite(value).all() for value in scores.values()),
            "training_valid": all(row["converged"] and row["final_loss"] < row["initial_loss"] for row in training_rows),
            "protected_roles_untouched": not any(context.protected_data_permissions.values())}
        if not all(structural.values()):
            raise DownsideInconclusive(f"structural checks failed: {[k for k,v in structural.items() if not v]}")
        atomic_json(context.phase_output_path / "r12_a0_contract_summary.json", {"schema_version": 1,
            "route_id": ROUTE_ID, "operation_id": OPERATION_ID, "cells": list(CELLS),
            "folds": list(FOLDS), "feature_dim": FEATURE_DIM, "optimizer": "L-BFGS-B",
            "l2": L2, "max_iter": MAX_ITER, "bootstrap_draws": BOOTSTRAP_DRAWS,
            "thresholds": {"severe_gain_db": SEVERE, "reject_fraction": 0.20,
                "auroc": AUROC_MIN, "control_increment": CONTROL_INCREMENT,
                "shuffle_increment": SHUFFLE_INCREMENT, "capture": CAPTURE_MIN,
                "retained_ratio": RETAINED_RATIO_MAX}})
        atomic_json(context.phase_output_path / "r12_a0_provenance_and_access.json", {"schema_version": 1,
            "route_commit": context.route_commit, "r11_terminal_preserved": True,
            "immutable_r11_outputs_only": True, "restoration_training_run": False,
            "restoration_inference_run": False, "candidate_generation_rerun": False,
            "image_or_target_opened": False, "confirmation_touched": False,
            "canary_touched": False, "locked_test_touched": False})
        atomic_json(context.phase_output_path / "r12_a0_input_identity.json", {"schema_version": 1,
            "groups": len(names), "rows": len(rows), "fold_counts": {str(f): 192 for f in FOLDS},
            "severe_rows": int(labels.sum()), "severe_groups": severe_groups,
            "asset_sha256": {key: context.assets[key].sha256 for key in sorted(context.assets)}})
        write_csv(context.phase_output_path / "r12_a0_cell_summary.csv", cell_rows)
        write_csv(context.phase_output_path / "r12_a0_fold_stability.csv", fold_rows + training_rows)
        atomic_json(context.phase_output_path / "r12_a0_bootstrap_summary.json", {"schema_version": 1, **boot})
        write_csv(context.phase_output_path / "r12_a0_risk_coverage.csv", risk_rows)
        atomic_json(context.phase_output_path / "r12_a0_gate_summary.json", {"schema_version": 1,
            "structural_checks": structural, "gates": gates, "passes": passes,
            "decisive_fail": decisive, "fold_metrics": fold_rows, "state": state,
            "decision": decision, "authorizes": authorizes, "r11_terminal_changed": False})
        atomic_json(context.phase_output_path / "r12_a0_resource_summary.json", {"schema_version": 1,
            "wall_seconds": time.perf_counter() - started_wall, "cpu_seconds": time.process_time() - started_cpu,
            "max_rss_mib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0,
            "models": 8, "bootstrap_draws": BOOTSTRAP_DRAWS, "gpu_used": False})
        score_rows = [{"name": row["name"], "fold": row["fold"], "tile": row["tile"],
                       "action": row["action"], "severe": int(label),
                       **{f"{cell}_risk": float(scores[cell][index]) for cell in CELLS}}
                      for index, (row, label) in enumerate(zip(rows, labels))]
        write_csv(context.phase_output_path / "r12_a0_oof_risk_scores_cloud_only.csv", score_rows)
        write_run_result(context, state=state, decision=decision, authorizes=authorizes, details={
            "primary_auroc": boot[f"{PRIMARY}_auroc"]["point"],
            "primary_minus_action_agnostic": boot["primary_minus_ACTION_AGNOSTIC_auroc"]["point"],
            "primary_minus_sign_swap": boot["primary_minus_SIGN_SWAP_auroc"]["point"],
            "primary_minus_label_shuffle": boot["primary_minus_LABEL_SHUFFLE_auroc"]["point"],
            "severe_capture_at_20pct": boot[f"{PRIMARY}_severe_capture_at_20pct"]["point"],
            "retained_severe_prevalence_ratio": boot[f"{PRIMARY}_retained_severe_prevalence_ratio"]["point"],
            "r11_terminal_changed": False})
    except DownsideInconclusive as exc:
        write_inconclusive(context, str(exc), started_wall, started_cpu)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("phase", choices=("contract", "run"))
    parser.add_argument("--context", required=True, type=Path); args = parser.parse_args()
    contract(args.context) if args.phase == "contract" else run(args.context)


if __name__ == "__main__":
    main()
