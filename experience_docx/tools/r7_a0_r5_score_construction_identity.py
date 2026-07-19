#!/usr/bin/env python3
"""Frozen finite-population audit of R5 seed-to-ensemble score construction."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
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


ROUTE_ID = "haze4k_v5_r7_r5_score_construction_identity_20260719"
OPERATION_ID = "R7_A0_FROZEN_R5_SCORE_CONSTRUCTION_IDENTITY_AUDIT"
R5_ROUTE_ID = "haze4k_v5_r5_spatial_candidate_response_sufficiency_20260719"
R5_OPERATION_ID = "R5_A0_FROZEN_SPATIAL_RESPONSE_SUFFICIENCY_SCREEN"
R5_RUN_ID = "r5-a0-spatial-response-screen-r2"
R5_ROUTE_COMMIT = "7e75eed504b2ead65a1971ec250dc7f59a79574d"
R5_RUNNER_SHA256 = "336c7e1beccb793229beb533ba12367261e702866497c388ee2a4fa88d12718b"
R6_ROUTE_ID = "haze4k_v5_r6_r5_decision_component_attribution_20260719"
R6_OPERATION_ID = "R6_A0_FROZEN_R5_DECISION_COMPONENT_ATTRIBUTION_AUDIT"
R6_RUN_ID = "r6-a0-r5-decision-attribution-r2"
CELLS = (
    "P0_POOLED_DC_ONLY",
    "S1_TRUE_SPATIAL_RESPONSE",
    "S2_SPATIAL_RESPONSE_SHUFFLE",
    "G0_GENERIC_STATE_SPATIAL",
)
ACTIONS = ("state_positive_full", "state_negative_full")
OPERATORS = ("D_ref", "D_rep")
FOLDS = (0, 1)
SEEDS = (3407, 3411)
SCORE_FIELDS = ("mean_score", "q05_score", "severe_score")
SELECTED_PER_FOLD = 39
SEVERE_GAIN = -0.2
HARD_GAIN = -0.5
EXPECTED_ROWS = {"per_seed": 12288, "candidate": 6144, "policy": 3072}
PER_SEED_HEADER = {
    "action", "cell", "fold", "mean_score", "name", "operator",
    "q05_score", "seed", "severe_score", "target_gain_db",
}
CANDIDATE_HEADER = {
    "action", "cell", "fold", "mean_score", "name", "operator",
    "q05_score", "severe_label", "severe_score", "target_gain_db",
}
POLICY_HEADER = {
    "cell", "fold", "gain", "mean_score", "name", "negative_oracle",
    "operator", "oracle_gain", "oracle_selected", "q05_score",
    "robust_score", "selected", "severe_score",
}
CELL_HEADER = {
    "cell", "coverage", "d_ref_gain_db", "d_rep_gain_db", "gain_point_db",
    "negative_selected_groups", "selected_groups", "selected_hard_groups",
    "selected_severe_groups",
}


class InputInconclusive(RuntimeError):
    pass


def read_csv(path: Path, header: set[str]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or set(reader.fieldnames) != header:
            raise InputInconclusive(f"CSV header mismatch: {path.name}")
        rows = list(reader)
    if not rows or any(None in row or any(value is None for value in row.values()) for row in rows):
        raise InputInconclusive(f"CSV row contract failed: {path.name}")
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    fields = sorted({key for row in rows for key in row})
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def finite_float(value: str, field: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise InputInconclusive(f"non-numeric {field}") from exc
    if not math.isfinite(result):
        raise InputInconclusive(f"non-finite {field}")
    return result


def bool_value(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise InputInconclusive("invalid serialized boolean")


def action_index(value: str) -> int:
    try:
        return ACTIONS.index(value)
    except ValueError as exc:
        raise InputInconclusive(f"unexpected action {value}") from exc


def score_key(row: dict[str, str]) -> tuple[str, int, str, str, int]:
    cell = row["cell"]; fold = int(row["fold"]); operator = row["operator"]
    if cell not in CELLS or fold not in FOLDS or operator not in OPERATORS:
        raise InputInconclusive("score identity is outside the frozen scope")
    return cell, fold, row["name"], operator, action_index(row["action"])


def native_mean(values: tuple[float, float]) -> float:
    import torch

    tensors = [torch.tensor(value, dtype=torch.float32) for value in values]
    return float(torch.stack(tensors).mean(0))


def float64_mean(values: tuple[float, float]) -> float:
    return (values[0] + values[1]) / 2.0


def reconstruct(
    per_seed_rows: list[dict[str, str]], candidate_rows: list[dict[str, str]],
) -> tuple[dict[tuple[str, int, str, str, int], dict[str, Any]], list[dict[str, Any]]]:
    seeds: dict[tuple[str, int, str, str, int, int], dict[str, str]] = {}
    for row in per_seed_rows:
        key = (*score_key(row), int(row["seed"]))
        if key[-1] not in SEEDS or key in seeds:
            raise InputInconclusive("per-seed key is duplicate or outside the frozen seeds")
        seeds[key] = row
    candidates: dict[tuple[str, int, str, str, int], dict[str, Any]] = {}
    summary: dict[tuple[str, str], dict[str, Any]] = {}
    for row in candidate_rows:
        key = score_key(row)
        if key in candidates:
            raise InputInconclusive("duplicate candidate-score key")
        seed_rows = [seeds.get((*key, seed)) for seed in SEEDS]
        if any(item is None for item in seed_rows):
            raise InputInconclusive("candidate score lacks both saved seed rows")
        record: dict[str, Any] = {
            "target_gain_db": finite_float(row["target_gain_db"], "target_gain_db"),
            "severe_label": int(row["severe_label"]),
        }
        for item in seed_rows:
            assert item is not None
            if finite_float(item["target_gain_db"], "target_gain_db") != record["target_gain_db"]:
                raise InputInconclusive("target changes across seed and ensemble rows")
        if record["severe_label"] not in (0, 1) \
                or record["severe_label"] != int(record["target_gain_db"] <= SEVERE_GAIN):
            raise InputInconclusive("saved severe label disagrees with frozen target threshold")
        for field in SCORE_FIELDS:
            values = tuple(finite_float(item[field], field) for item in seed_rows if item is not None)
            if len(values) != 2:
                raise InputInconclusive("seed score pair is incomplete")
            saved = finite_float(row[field], field)
            native = native_mean((values[0], values[1]))
            double = float64_mean((values[0], values[1]))
            record[field] = saved
            record[f"native_{field}"] = native
            record[f"float64_{field}"] = double
            cell_summary = summary.setdefault((key[0], field), {
                "cell": key[0], "field": field, "rows": 0,
                "native_exact_mismatches": 0, "native_max_abs_difference": 0.0,
                "float64_exact_mismatches": 0, "float64_max_abs_difference": 0.0,
            })
            cell_summary["rows"] += 1
            native_difference = abs(saved - native)
            double_difference = abs(saved - double)
            cell_summary["native_exact_mismatches"] += saved != native
            cell_summary["float64_exact_mismatches"] += saved != double
            cell_summary["native_max_abs_difference"] = max(
                cell_summary["native_max_abs_difference"], native_difference,
            )
            cell_summary["float64_max_abs_difference"] = max(
                cell_summary["float64_max_abs_difference"], double_difference,
            )
        candidates[key] = record
    if len(seeds) != EXPECTED_ROWS["per_seed"] or len(candidates) != EXPECTED_ROWS["candidate"]:
        raise InputInconclusive("score key cardinality is incomplete")
    if set(seeds) != {(*key, seed) for key in candidates for seed in SEEDS}:
        raise InputInconclusive("per-seed and ensemble key grids differ")
    return candidates, [summary[key] for key in sorted(summary)]


def tie_key(fold: int, name: str) -> str:
    return hashlib.sha256(f"{R5_ROUTE_ID}|fold={fold}|{name}".encode()).hexdigest()


def score_value(record: dict[str, Any], mode: str, field: str) -> float:
    key = field if mode == "saved" else f"{mode}_{field}"
    return float(record[key])


def build_policies(
    candidates: dict[tuple[str, int, str, str, int], dict[str, Any]], mode: str,
) -> tuple[dict[tuple[str, str, str], dict[str, Any]], dict[tuple[str, int], dict[str, set[str]]]]:
    names_by_cell_fold: dict[tuple[str, int], set[str]] = {}
    for cell, fold, name, _operator, _action in candidates:
        names_by_cell_fold.setdefault((cell, fold), set()).add(name)
    if any(len(names) != 192 for names in names_by_cell_fold.values()) \
            or set(names_by_cell_fold) != {(cell, fold) for cell in CELLS for fold in FOLDS}:
        raise InputInconclusive("cell/fold name groups are not complete 192-image sets")
    policies: dict[tuple[str, str, str], dict[str, Any]] = {}
    selections: dict[tuple[str, int], dict[str, set[str]]] = {}
    for cell in CELLS:
        for fold in FOLDS:
            names = sorted(names_by_cell_fold[(cell, fold)])
            assignments = {}
            scores = {}
            for name in names:
                robust_q05 = tuple(
                    min(
                        score_value(candidates[(cell, fold, name, operator, action)], mode, "q05_score")
                        for operator in OPERATORS
                    )
                    for action in range(2)
                )
                assignment = max(range(2), key=lambda action: (robust_q05[action], -action))
                assignments[name] = assignment
                scores[name] = robust_q05[assignment]
            ordered = sorted(names, key=lambda name: (-scores[name], tie_key(fold, name)))
            selected = set(ordered[:SELECTED_PER_FOLD])
            selections[(cell, fold)] = {"selected": selected, "assignments": {
                f"{name}:{assignments[name]}" for name in names
            }}
            for name in names:
                truths = tuple(
                    min(
                        float(candidates[(cell, fold, name, operator, action)]["target_gain_db"])
                        for operator in OPERATORS
                    )
                    for action in range(2)
                )
                oracle_action = max(range(2), key=lambda action: (truths[action], -action))
                oracle = oracle_action if truths[oracle_action] > 0.0 else None
                selected_action = assignments[name] if name in selected else None
                for operator in OPERATORS:
                    selected_record = candidates[(cell, fold, name, operator, assignments[name])]
                    truth_record = candidates[(cell, fold, name, operator, oracle_action)]
                    policies[(cell, name, operator)] = {
                        "fold": fold,
                        "selected": 0 if selected_action is None else selected_action + 1,
                        "gain": 0.0 if selected_action is None else float(selected_record["target_gain_db"]),
                        "oracle_selected": 0 if oracle is None else oracle + 1,
                        "oracle_gain": 0.0 if oracle is None else float(truth_record["target_gain_db"]),
                        "negative_oracle": oracle == 1,
                        "robust_score": scores[name],
                        "mean_score": 0.0 if selected_action is None else score_value(selected_record, mode, "mean_score"),
                        "q05_score": 0.0 if selected_action is None else score_value(selected_record, mode, "q05_score"),
                        "severe_score": 0.0 if selected_action is None else score_value(selected_record, mode, "severe_score"),
                    }
    return policies, selections


def compare_policy_rows(
    reconstructed: dict[tuple[str, str, str], dict[str, Any]],
    saved_rows: list[dict[str, str]],
) -> tuple[bool, dict[tuple[str, str, str], dict[str, str]], int, float]:
    saved = {}
    for row in saved_rows:
        key = (row["cell"], row["name"], row["operator"] )
        if row["cell"] not in CELLS or int(row["fold"]) not in FOLDS \
                or row["operator"] not in OPERATORS or key in saved:
            raise InputInconclusive("saved policy key is invalid or duplicate")
        saved[key] = row
    if len(saved) != EXPECTED_ROWS["policy"] or set(saved) != set(reconstructed):
        raise InputInconclusive("saved and reconstructed policy key grids differ")
    mismatches = 0
    max_difference = 0.0
    for key, expected in reconstructed.items():
        observed = saved[key]
        exact_fields = ("selected", "oracle_selected")
        if any(int(observed[field]) != int(expected[field]) for field in exact_fields):
            mismatches += 1
            continue
        if bool_value(observed["negative_oracle"]) != bool(expected["negative_oracle"]):
            mismatches += 1
            continue
        if int(observed["fold"]) != int(expected["fold"]):
            mismatches += 1
            continue
        row_failed = False
        for field in (
            "gain", "oracle_gain", "robust_score", "mean_score", "q05_score", "severe_score",
        ):
            observed_value = finite_float(observed[field], field)
            difference = abs(observed_value - float(expected[field]))
            max_difference = max(max_difference, difference)
            row_failed |= observed_value != float(expected[field])
        mismatches += row_failed
    return mismatches == 0, saved, mismatches, max_difference


def summarize_policies(
    policies: dict[tuple[str, str, str], dict[str, Any]],
    saved_cell_rows: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], bool]:
    import numpy as np

    saved_by_cell = {row["cell"]: row for row in saved_cell_rows}
    if set(saved_by_cell) != set(CELLS) or len(saved_cell_rows) != len(CELLS):
        raise InputInconclusive("R5 cell summary is incomplete or duplicate")
    rows = []
    all_exact = True
    for cell in CELLS:
        subset = {key: value for key, value in policies.items() if key[0] == cell}
        names = sorted({key[1] for key in subset})
        gains = {
            operator: [float(subset[(cell, name, operator)]["gain"]) for name in names]
            for operator in OPERATORS
        }
        selected = sum(int(subset[(cell, name, OPERATORS[0])]["selected"]) != 0 for name in names)
        negative = sum(int(subset[(cell, name, OPERATORS[0])]["selected"]) == 2 for name in names)
        severe = sum(any(float(subset[(cell, name, operator)]["gain"]) <= SEVERE_GAIN for operator in OPERATORS) for name in names)
        hard = sum(any(float(subset[(cell, name, operator)]["gain"]) <= HARD_GAIN for operator in OPERATORS) for name in names)
        d_ref = float(np.mean(gains["D_ref"])); d_rep = float(np.mean(gains["D_rep"])); point = min(d_ref, d_rep)
        observed = saved_by_cell[cell]
        checks = {
            "coverage": float(observed["coverage"]) == selected / len(names),
            "d_ref_gain": float(observed["d_ref_gain_db"]) == d_ref,
            "d_rep_gain": float(observed["d_rep_gain_db"]) == d_rep,
            "gain_point": float(observed["gain_point_db"]) == point,
            "selected": int(observed["selected_groups"]) == selected,
            "negative": int(observed["negative_selected_groups"]) == negative,
            "severe": int(observed["selected_severe_groups"]) == severe,
            "hard": int(observed["selected_hard_groups"]) == hard,
        }
        all_exact &= all(checks.values()) and selected == 78
        rows.append({
            "cell": cell, "groups": len(names), "selected_groups": selected,
            "negative_selected_groups": negative, "selected_severe_groups": severe,
            "selected_hard_groups": hard, "d_ref_gain_db": d_ref,
            "d_rep_gain_db": d_rep, "gain_point_db": point,
            "saved_cell_summary_exact": all(checks.values()),
        })
    return rows, all_exact


def downstream_sensitivity(
    native: dict[tuple[str, int], dict[str, set[str]]],
    double: dict[tuple[str, int], dict[str, set[str]]],
) -> dict[str, Any]:
    rows = []
    total_action_changes = 0
    total_selection_difference = 0
    for key in sorted(native):
        action_changes = len(native[key]["assignments"] ^ double[key]["assignments"]) // 2
        selection_difference = len(native[key]["selected"] ^ double[key]["selected"])
        total_action_changes += action_changes
        total_selection_difference += selection_difference
        rows.append({
            "cell": key[0], "fold": key[1], "action_assignment_changes": action_changes,
            "top39_symmetric_difference": selection_difference,
        })
    return {
        "schema_version": 1, "native_vs_float64_by_cell_fold": rows,
        "total_action_assignment_changes": total_action_changes,
        "total_top39_symmetric_difference": total_selection_difference,
        "interpretation_role": "prespecified_sensitivity_not_authoritative_reconstruction",
    }


def synthetic_inputs() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    per_seed = []
    candidates = []
    for cell_index, cell in enumerate(CELLS):
        for fold in FOLDS:
            for name_index in range(192):
                name = f"synthetic_{fold}_{name_index:03d}"
                for operator_index, operator in enumerate(OPERATORS):
                    for action_index_value, action in enumerate(ACTIONS):
                        target = 0.2 - 0.001 * name_index - 0.01 * action_index_value - 0.002 * operator_index
                        seed_values = []
                        for seed_index, seed in enumerate(SEEDS):
                            base = 0.01 * cell_index + 0.001 * name_index + 0.0001 * seed_index
                            values = {
                                "mean_score": base + 0.00003,
                                "q05_score": base - 0.002 * action_index_value,
                                "severe_score": 0.1 + base,
                            }
                            seed_values.append(values)
                            per_seed.append({
                                "action": action, "cell": cell, "fold": str(fold),
                                "name": name, "operator": operator, "seed": str(seed),
                                "target_gain_db": repr(target),
                                **{field: repr(value) for field, value in values.items()},
                            })
                        ensemble = {
                            field: native_mean((seed_values[0][field], seed_values[1][field]))
                            for field in SCORE_FIELDS
                        }
                        candidates.append({
                            "action": action, "cell": cell, "fold": str(fold),
                            "name": name, "operator": operator, "target_gain_db": repr(target),
                            "severe_label": str(int(target <= SEVERE_GAIN)),
                            **{field: repr(value) for field, value in ensemble.items()},
                        })
    return per_seed, candidates


def contract(context_path: Path) -> None:
    context = load_context(context_path, "contract")
    prepare_phase_output(context)
    per_seed, candidates = synthetic_inputs()
    reconstructed, summary = reconstruct(per_seed, candidates)
    native_policies, native_selection = build_policies(reconstructed, "native")
    double_policies, double_selection = build_policies(reconstructed, "float64")
    sensitivity = downstream_sensitivity(native_selection, double_selection)
    checks = {
        "route_identity": context.route_id == ROUTE_ID and context.operation_id == OPERATION_ID,
        "cpu_only": context.device == "cpu",
        "protected_roles_blocked": not any(context.protected_data_permissions.values()),
        "formal_row_scale": len(per_seed) == EXPECTED_ROWS["per_seed"] and len(candidates) == EXPECTED_ROWS["candidate"],
        "native_exactness": all(row["native_exact_mismatches"] == 0 for row in summary),
        "four_cell_policy_scale": len(native_policies) == EXPECTED_ROWS["policy"] == len(double_policies),
        "fixed_coverage": all(len(value["selected"]) == SELECTED_PER_FOLD for value in native_selection.values()),
        "finite_sensitivity": isinstance(sensitivity["total_action_assignment_changes"], int),
        "bounded_work": len(reconstructed) == EXPECTED_ROWS["candidate"],
    }
    write_contract_result(context, checks=checks)


def write_inconclusive(context: Any, reason: str, started_wall: float, started_cpu: float) -> None:
    base = {"schema_version": 1, "status": "input_identity_inconclusive", "reason": reason}
    for filename in (
        "r7_a0_contract_summary.json", "r7_a0_provenance_and_access.json",
        "r7_a0_input_identity.json", "r7_a0_downstream_sensitivity.json",
        "r7_a0_gate_summary.json",
    ):
        atomic_json(context.phase_output_path / filename, base)
    placeholder = [{"status": "inconclusive", "reason": reason}]
    write_csv(context.phase_output_path / "r7_a0_score_reconstruction_summary.csv", placeholder)
    write_csv(context.phase_output_path / "r7_a0_policy_replay_summary.csv", placeholder)
    atomic_json(context.phase_output_path / "r7_a0_resource_summary.json", {
        **base, "wall_seconds": time.perf_counter() - started_wall,
        "cpu_seconds": time.process_time() - started_cpu, "gpu_used": False,
    })
    write_run_result(
        context, state="COMPLETED_GATE_INCONCLUSIVE",
        decision="R7_A0_INPUT_IDENTITY_INCONCLUSIVE_STOP", authorizes="NONE",
        details={"reason": reason, "r5_terminal_changed": False, "r6_terminal_changed": False},
    )


def run(context_path: Path) -> None:
    started_wall = time.perf_counter(); started_cpu = time.process_time()
    context = load_context(context_path, "run")
    prepare_phase_output(context)
    try:
        identity = json.loads(asset_path(context, "r5_lifecycle_identity", kind="file").read_text(encoding="utf-8"))
        expected_identity = {
            "schema_version": 1, "route_id": R5_ROUTE_ID, "operation_id": R5_OPERATION_ID,
            "run_id": R5_RUN_ID, "route_commit": R5_ROUTE_COMMIT, "runner_sha256": R5_RUNNER_SHA256,
        }
        if identity != expected_identity:
            raise InputInconclusive("R5 lifecycle identity mismatch")
        r5_closeout = json.loads(asset_path(context, "r5_closeout", kind="file").read_text(encoding="utf-8"))
        if (r5_closeout.get("state"), r5_closeout.get("decision"), r5_closeout.get("authorizes")) != (
            "COMPLETED_GATE_FAIL", "R5_A0_SPATIAL_RESPONSE_FUTILITY_OR_SAFETY_FAIL_STOP", "NONE"
        ) or any(r5_closeout.get(key) != expected_identity[key] for key in ("route_id", "operation_id", "run_id", "route_commit", "runner_sha256")):
            raise InputInconclusive("R5 typed closeout identity mismatch")
        r6_closeout = json.loads(asset_path(context, "r6_closeout", kind="file").read_text(encoding="utf-8"))
        if (r6_closeout.get("route_id"), r6_closeout.get("operation_id"), r6_closeout.get("run_id")) != (
            R6_ROUTE_ID, R6_OPERATION_ID, R6_RUN_ID
        ) or (r6_closeout.get("state"), r6_closeout.get("decision"), r6_closeout.get("authorizes")) != (
            "COMPLETED_GATE_INCONCLUSIVE", "R6_A0_INPUT_OR_ATTRIBUTION_INCONCLUSIVE_STOP", "NONE"
        ) or r6_closeout.get("details", {}).get("reason") != "seed ensemble replay mismatch: q05_score":
            raise InputInconclusive("R6 triggering closeout identity mismatch")
        per_seed_rows = read_csv(asset_path(context, "r5_per_seed_predictions", kind="file"), PER_SEED_HEADER)
        candidate_rows = read_csv(asset_path(context, "r5_candidate_scores", kind="file"), CANDIDATE_HEADER)
        policy_rows = read_csv(asset_path(context, "r5_policy_rows", kind="file"), POLICY_HEADER)
        if (len(per_seed_rows), len(candidate_rows), len(policy_rows)) != (
            EXPECTED_ROWS["per_seed"], EXPECTED_ROWS["candidate"], EXPECTED_ROWS["policy"],
        ):
            raise InputInconclusive("R5 raw row counts are incomplete")
        write_workload_progress(context, completed_units=1, stage="inputs_verified")
        reconstructed, score_summary = reconstruct(per_seed_rows, candidate_rows)
        write_workload_progress(context, completed_units=2, stage="scores_reconstructed")
        native_policies, native_selection = build_policies(reconstructed, "native")
        double_policies, double_selection = build_policies(reconstructed, "float64")
        policy_exact, _saved_policy, policy_mismatches, policy_max_difference = compare_policy_rows(
            native_policies, policy_rows,
        )
        cell_rows = read_csv(asset_path(context, "r5_cell_summary", kind="file"), CELL_HEADER)
        policy_summary, cell_exact = summarize_policies(native_policies, cell_rows)
        sensitivity = downstream_sensitivity(native_selection, double_selection)
        write_workload_progress(context, completed_units=3, stage="policies_replayed")
        native_exact = all(row["native_exact_mismatches"] == 0 for row in score_summary)
        target_label_exact = True
        structural_valid = (
            len(reconstructed) == EXPECTED_ROWS["candidate"]
            and len(native_policies) == EXPECTED_ROWS["policy"]
            and all(len(value["selected"]) == SELECTED_PER_FOLD for value in native_selection.values())
        )
        passes = structural_valid and native_exact and target_label_exact and policy_exact and cell_exact
        if passes:
            state = "COMPLETED_GATE_PASS"
            decision = "R7_A0_SCORE_CONSTRUCTION_IDENTITY_PASS"
            authorizes = "R7_NEXT_ATTRIBUTION_CONTRACT_REVIEW_ONLY"
        else:
            state = "COMPLETED_GATE_FAIL"
            decision = "R7_A0_SCORE_OR_POLICY_IDENTITY_FAIL_STOP"
            authorizes = "NONE"
        contract_summary = {
            "schema_version": 1, "route_id": ROUTE_ID, "operation_id": OPERATION_ID,
            "audit_role": "engineering_integrity_with_post_hoc_mechanism_implication",
            "authoritative_reconstruction": "torch_float32_stack_two_seed_predictions_then_mean_dim0",
            "sensitivity_reconstruction": "python_float64_arithmetic_mean",
            "cells": list(CELLS), "folds": list(FOLDS), "seeds": list(SEEDS),
            "operators": list(OPERATORS), "actions": list(ACTIONS),
            "selected_groups_per_fold_cell": SELECTED_PER_FOLD,
        }
        provenance = {
            "schema_version": 1, "route_commit": context.route_commit,
            "r5_source_commit": R5_ROUTE_COMMIT, "r5_source_expression": "torch.stack(seed_predictions).mean(0)",
            "r5_terminal_preserved": True, "r6_terminal_preserved": True,
            "training_run": False, "inference_run": False, "candidate_generation_rerun": False,
            "confirmation_images_targets_outcomes_touched": False, "canary_touched": False,
            "locked_test_touched": False,
        }
        input_identity = {
            "schema_version": 1, "r5_lifecycle_identity": identity,
            "row_counts": {"per_seed": len(per_seed_rows), "candidate": len(candidate_rows), "policy": len(policy_rows)},
            "asset_sha256": {identifier: context.assets[identifier].sha256 for identifier in sorted(context.assets)},
        }
        gates = {
            "structural_valid": structural_valid, "native_float32_scores_exact": native_exact,
            "target_and_severe_label_identity_exact": target_label_exact,
            "all_saved_policy_rows_exact": policy_exact, "all_saved_cell_summaries_exact": cell_exact,
        }
        gate_summary = {
            "schema_version": 1, "gates": gates, "passes": passes,
            "state": state, "decision": decision, "authorizes": authorizes,
            "policy_row_mismatches": policy_mismatches,
            "policy_max_abs_difference": policy_max_difference,
            "r5_terminal_changed": False, "r6_terminal_changed": False,
        }
        atomic_json(context.phase_output_path / "r7_a0_contract_summary.json", contract_summary)
        atomic_json(context.phase_output_path / "r7_a0_provenance_and_access.json", provenance)
        atomic_json(context.phase_output_path / "r7_a0_input_identity.json", input_identity)
        write_csv(context.phase_output_path / "r7_a0_score_reconstruction_summary.csv", score_summary)
        write_csv(context.phase_output_path / "r7_a0_policy_replay_summary.csv", policy_summary)
        atomic_json(context.phase_output_path / "r7_a0_downstream_sensitivity.json", sensitivity)
        atomic_json(context.phase_output_path / "r7_a0_gate_summary.json", gate_summary)
        atomic_json(context.phase_output_path / "r7_a0_resource_summary.json", {
            "schema_version": 1, "wall_seconds": time.perf_counter() - started_wall,
            "cpu_seconds": time.process_time() - started_cpu,
            "max_rss_mib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0,
            "raw_rows_read": len(per_seed_rows) + len(candidate_rows) + len(policy_rows),
            "reconstruction_units": len(reconstructed), "gpu_used": False,
        })
        write_workload_progress(context, completed_units=4, stage="finalized")
        float64_mismatches = sum(row["float64_exact_mismatches"] for row in score_summary)
        write_run_result(
            context, state=state, decision=decision, authorizes=authorizes,
            details={
                "native_score_mismatches": sum(row["native_exact_mismatches"] for row in score_summary),
                "float64_score_mismatches": float64_mismatches,
                "policy_row_mismatches": policy_mismatches,
                "float64_action_assignment_changes": sensitivity["total_action_assignment_changes"],
                "float64_top39_symmetric_difference": sensitivity["total_top39_symmetric_difference"],
                "r5_terminal_changed": False, "r6_terminal_changed": False,
            },
        )
    except InputInconclusive as exc:
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
