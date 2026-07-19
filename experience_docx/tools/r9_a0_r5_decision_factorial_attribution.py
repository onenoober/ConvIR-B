#!/usr/bin/env python3
"""Frozen R9 attribution of the verified R5 action, coverage, and risk policy."""

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


ROUTE_ID = "haze4k_v5_r9_r5_decision_factorial_attribution_20260719"
OPERATION_ID = "R9_A0_FROZEN_R5_DECISION_FACTORIAL_ATTRIBUTION_AUDIT"
R5_ROUTE_ID = "haze4k_v5_r5_spatial_candidate_response_sufficiency_20260719"
R5_OPERATION_ID = "R5_A0_FROZEN_SPATIAL_RESPONSE_SUFFICIENCY_SCREEN"
R5_RUN_ID = "r5-a0-spatial-response-screen-r2"
R5_ROUTE_COMMIT = "7e75eed504b2ead65a1971ec250dc7f59a79574d"
R5_RUNNER_SHA256 = "336c7e1beccb793229beb533ba12367261e702866497c388ee2a4fa88d12718b"
R8_ROUTE_ID = "haze4k_v5_r8_r5_score_and_label_identity_20260719"
R8_OPERATION_ID = "R8_A0_FROZEN_R5_SCORE_AND_LABEL_IDENTITY_AUDIT"
R8_RUN_ID = "r8-a0-r5-score-label-identity-r1"
R8_ROUTE_COMMIT = "8a0238f303a8195248b5c06d2c300da82c9d9dbd"
PRIMARY_CELL = "S1_TRUE_SPATIAL_RESPONSE"
ACTIONS = ("state_positive_full", "state_negative_full")
OPERATORS = ("D_ref", "D_rep")
FOLDS = (0, 1)
CELL_BITS = {
    f"P{action}{coverage}{risk}": (action, coverage, risk)
    for action in range(2)
    for coverage in range(2)
    for risk in range(2)
}
ATTRIBUTION_CELLS = ("P100", "P010", "P001", "P110", "P101", "P011")
INTERACTIONS = {
    "action_x_coverage": ("P110", "P100", "P010"),
    "action_x_risk": ("P101", "P100", "P001"),
    "coverage_x_risk": ("P011", "P010", "P001"),
}
COVERAGE = 0.20
SELECTED_PER_FOLD = 39
BOOTSTRAP_DRAWS = 4000
BOOTSTRAP_SEED = 3407
SEVERE_GAIN = -0.2
HARD_GAIN = -0.5
UTILITY_DELTA = 0.005
RECOVERY_FRACTION = 0.25
SAFETY_NONINFERIORITY = -0.005
FLOAT_TOLERANCE = 2.0e-12
EXPECTED_ROWS = {"candidate": 6144, "policy": 3072}
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


class DiagnosticInconclusive(RuntimeError):
    """A scientific input or identifiability stop that must produce evidence."""


def read_csv(path: Path, expected: set[str]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or set(reader.fieldnames) != expected:
            raise DiagnosticInconclusive(f"CSV header mismatch: {path.name}")
        rows = list(reader)
    if not rows or any(None in row or any(value is None for value in row.values()) for row in rows):
        raise DiagnosticInconclusive(f"CSV row contract failed: {path.name}")
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
        raise DiagnosticInconclusive(f"non-numeric {field}") from exc
    if not math.isfinite(result):
        raise DiagnosticInconclusive(f"non-finite {field}")
    return result


def finite_int(value: str, field: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise DiagnosticInconclusive(f"invalid integer {field}") from exc
    return result


def read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DiagnosticInconclusive(f"invalid JSON input: {label}") from exc
    if not isinstance(value, dict):
        raise DiagnosticInconclusive(f"JSON input is not an object: {label}")
    return value


def bool_value(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise DiagnosticInconclusive("invalid serialized boolean")


def action_index(value: str) -> int:
    try:
        return ACTIONS.index(value)
    except ValueError as exc:
        raise DiagnosticInconclusive(f"unexpected action: {value}") from exc


def close(first: float, second: float, tolerance: float = FLOAT_TOLERANCE) -> bool:
    return math.isfinite(first) and math.isfinite(second) and abs(first - second) <= tolerance


def tie_key(fold: int, name: str) -> str:
    return hashlib.sha256(f"{R5_ROUTE_ID}|fold={fold}|{name}".encode()).hexdigest()


def cvar(values: Any, fraction: float = 0.05) -> float:
    import numpy as np

    array = np.sort(np.asarray(values, dtype=np.float64))
    return float(array[: max(1, math.ceil(fraction * len(array)))].mean())


def interval(point: float, samples: Any) -> dict[str, float]:
    import numpy as np

    values = np.asarray(samples, dtype=np.float64)
    if not math.isfinite(point) or values.size == 0 or not np.isfinite(values).all():
        raise DiagnosticInconclusive("non-finite bootstrap result")
    return {
        "point": float(point),
        "lcb95": float(np.quantile(values, 0.025)),
        "ucb95": float(np.quantile(values, 0.975)),
    }


def load_records(context: Any) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    candidate_rows = read_csv(
        asset_path(context, "r5_candidate_scores", kind="file"), CANDIDATE_HEADER,
    )
    policy_rows = read_csv(
        asset_path(context, "r5_policy_rows", kind="file"), POLICY_HEADER,
    )
    if (len(candidate_rows), len(policy_rows)) != (
        EXPECTED_ROWS["candidate"], EXPECTED_ROWS["policy"],
    ):
        raise DiagnosticInconclusive("R5 raw row counts do not match the frozen contract")

    primary_candidates = [row for row in candidate_rows if row["cell"] == PRIMARY_CELL]
    primary_policy = [row for row in policy_rows if row["cell"] == PRIMARY_CELL]
    if (len(primary_candidates), len(primary_policy)) != (1536, 768):
        raise DiagnosticInconclusive("R5 primary-cell row counts are incomplete")

    records: dict[str, dict[str, Any]] = {}
    candidate_keys: set[tuple[str, str, int]] = set()
    for row in primary_candidates:
        fold = finite_int(row["fold"], "fold")
        operator = row["operator"]
        action = action_index(row["action"])
        name = row["name"]
        if fold not in FOLDS or operator not in OPERATORS:
            raise DiagnosticInconclusive("R5 candidate identity is outside the frozen scope")
        key = (name, operator, action)
        if key in candidate_keys:
            raise DiagnosticInconclusive("duplicate R5 candidate-score row")
        candidate_keys.add(key)
        record = records.setdefault(
            name,
            {"fold": fold, "truth": {}, "mean": {}, "q05": {}, "risk": {}, "label": {}},
        )
        if record["fold"] != fold:
            raise DiagnosticInconclusive("one clean-image name appears in multiple folds")
        record["truth"][(operator, action)] = finite_float(row["target_gain_db"], "target_gain_db")
        record["mean"][(operator, action)] = finite_float(row["mean_score"], "mean_score")
        record["q05"][(operator, action)] = finite_float(row["q05_score"], "q05_score")
        record["risk"][(operator, action)] = finite_float(row["severe_score"], "severe_score")
        label = finite_int(row["severe_label"], "severe_label")
        if label not in (0, 1):
            raise DiagnosticInconclusive("saved severe label is not binary")
        record["label"][(operator, action)] = label

    required = {(operator, action) for operator in OPERATORS for action in range(2)}
    if len(records) != 384 or any(set(record["truth"]) != required for record in records.values()):
        raise DiagnosticInconclusive("R5 primary action/operator grid is incomplete")
    if any(sum(record["fold"] == fold for record in records.values()) != 192 for fold in FOLDS):
        raise DiagnosticInconclusive("R5 primary fold sizes are not 192/192")
    for record in records.values():
        if any(set(record[field]) != required for field in ("truth", "mean", "q05", "risk", "label")):
            raise DiagnosticInconclusive("R5 score field grid is incomplete")
        for action in range(2):
            expected_label = int(any(record["truth"][(operator, action)] <= SEVERE_GAIN for operator in OPERATORS))
            observed_labels = {record["label"][(operator, action)] for operator in OPERATORS}
            if observed_labels != {expected_label}:
                raise DiagnosticInconclusive("R5 any-operator severe-label semantics mismatch")

    policy_map: dict[tuple[str, str], dict[str, str]] = {}
    for row in primary_policy:
        fold = finite_int(row["fold"], "fold")
        operator = row["operator"]
        key = (row["name"], operator)
        if (
            key in policy_map
            or row["name"] not in records
            or operator not in OPERATORS
            or fold != records[row["name"]]["fold"]
        ):
            raise DiagnosticInconclusive("R5 policy row identity is invalid")
        policy_map[key] = row
    if len(policy_map) != 768:
        raise DiagnosticInconclusive("R5 primary policy grid is incomplete")
    return records, {
        "candidate_rows": len(candidate_rows),
        "policy_rows": len(policy_rows),
        "primary_candidate_rows": len(primary_candidates),
        "primary_policy_rows": len(primary_policy),
        "policy_map": policy_map,
    }


def robust_value(record: dict[str, Any], field: str, action: int) -> float:
    return min(float(record[field][(operator, action)]) for operator in OPERATORS)


def assigned_action(record: dict[str, Any], oracle: int) -> int:
    field = "truth" if oracle else "q05"
    return max(range(2), key=lambda action: (robust_value(record, field, action), -action))


def oracle_action(record: dict[str, Any]) -> int | None:
    action = assigned_action(record, 1)
    return action if robust_value(record, "truth", action) > 0.0 else None


def build_policy(records: dict[str, dict[str, Any]], bits: tuple[int, int, int]) -> dict[str, Any]:
    action_bit, coverage_bit, risk_bit = bits
    assignments = {name: assigned_action(record, action_bit) for name, record in records.items()}
    rank_field = "truth" if coverage_bit else "q05"
    rank_scores = {
        name: robust_value(record, rank_field, assignments[name])
        for name, record in records.items()
    }
    selected: set[str] = set()
    raw_selected: set[str] = set()
    ineligible: set[str] = set()
    removed: set[str] = set()
    backfilled: set[str] = set()
    selected_by_fold: dict[int, set[str]] = {}
    for fold in FOLDS:
        names = [name for name, record in records.items() if record["fold"] == fold]
        ordered = sorted(names, key=lambda name: (-rank_scores[name], tie_key(fold, name)))
        raw_fold = set(ordered[:SELECTED_PER_FOLD])
        raw_selected.update(raw_fold)
        if risk_bit:
            unsafe = {
                name for name in ordered
                if robust_value(records[name], "truth", assignments[name]) <= SEVERE_GAIN
            }
            ineligible.update(unsafe)
            eligible = [name for name in ordered if name not in unsafe]
        else:
            eligible = ordered
        if len(eligible) < SELECTED_PER_FOLD:
            raise DiagnosticInconclusive("risk veto cannot safely backfill fixed coverage")
        selected_fold = set(eligible[:SELECTED_PER_FOLD])
        if len(selected_fold) != SELECTED_PER_FOLD:
            raise DiagnosticInconclusive("fixed fold coverage is not exact")
        selected_by_fold[fold] = selected_fold
        selected.update(selected_fold)
        removed.update(raw_fold - selected_fold)
        backfilled.update(selected_fold - raw_fold)

    ordered_names = sorted(records)
    gains = {
        operator: [
            float(records[name]["truth"][(operator, assignments[name])]) if name in selected else 0.0
            for name in ordered_names
        ]
        for operator in OPERATORS
    }
    return {
        "assignments": assignments,
        "rank_scores": rank_scores,
        "selected": selected,
        "selected_by_fold": selected_by_fold,
        "raw_selected": raw_selected,
        "ineligible": ineligible,
        "removed": removed,
        "backfilled": backfilled,
        "gains": gains,
    }


def oracle_gains(records: dict[str, dict[str, Any]]) -> dict[str, list[float]]:
    result = {operator: [] for operator in OPERATORS}
    for name in sorted(records):
        action = oracle_action(records[name])
        for operator in OPERATORS:
            result[operator].append(
                0.0 if action is None else float(records[name]["truth"][(operator, action)])
            )
    return result


def evaluate_indices(
    policies: dict[str, dict[str, Any]], oracle: dict[str, list[float]], indices: Any,
) -> dict[str, dict[str, float]]:
    import numpy as np

    oracle_means = {
        operator: float(np.asarray(oracle[operator], dtype=np.float64)[indices].mean())
        for operator in OPERATORS
    }
    result = {}
    for cell, policy in policies.items():
        means = {
            operator: float(np.asarray(policy["gains"][operator], dtype=np.float64)[indices].mean())
            for operator in OPERATORS
        }
        tails = {
            operator: cvar(np.asarray(policy["gains"][operator], dtype=np.float64)[indices])
            for operator in OPERATORS
        }
        worst = min(OPERATORS, key=lambda operator: means[operator])
        if oracle_means[worst] <= 1.0e-12:
            raise DiagnosticInconclusive("R5 oracle denominator is nonpositive")
        result[cell] = {
            "gain": means[worst],
            "retention": means[worst] / oracle_means[worst],
            "cvar5": min(tails.values()),
            "d_ref_gain": means["D_ref"],
            "d_rep_gain": means["D_rep"],
        }
    return result


def bootstrap(
    policies: dict[str, dict[str, Any]], oracle: dict[str, list[float]], draws: int, seed: int,
) -> dict[str, Any]:
    import numpy as np

    count = len(next(iter(oracle.values())))
    point = evaluate_indices(policies, oracle, np.arange(count))
    samples = {cell: {metric: [] for metric in values} for cell, values in point.items()}
    contrast_samples = {
        cell: {"gain_delta": [], "cvar5_delta": [], "regret_recovery": []}
        for cell in CELL_BITS if cell != "P000"
    }
    interaction_samples = {name: [] for name in INTERACTIONS}
    gap_samples = []
    generator = np.random.default_rng(seed)
    for _draw in range(draws):
        draw = evaluate_indices(policies, oracle, generator.integers(0, count, count))
        for cell, values in draw.items():
            for metric, value in values.items():
                samples[cell][metric].append(value)
        gap = draw["P111"]["gain"] - draw["P000"]["gain"]
        if gap <= 1.0e-12:
            raise DiagnosticInconclusive("P111-P000 gain gap is nonpositive in paired bootstrap")
        gap_samples.append(gap)
        for cell in contrast_samples:
            contrast_samples[cell]["gain_delta"].append(draw[cell]["gain"] - draw["P000"]["gain"])
            contrast_samples[cell]["cvar5_delta"].append(draw[cell]["cvar5"] - draw["P000"]["cvar5"])
            contrast_samples[cell]["regret_recovery"].append(
                (draw[cell]["gain"] - draw["P000"]["gain"]) / gap
            )
        for name, (pair, first, second) in INTERACTIONS.items():
            interaction_samples[name].append(
                draw[pair]["gain"]
                - draw[first]["gain"]
                - draw[second]["gain"]
                + draw["P000"]["gain"]
            )

    cell_intervals = {
        cell: {metric: interval(point[cell][metric], values) for metric, values in samples[cell].items()}
        for cell in point
    }
    gap_point = point["P111"]["gain"] - point["P000"]["gain"]
    if gap_point <= 1.0e-12:
        raise DiagnosticInconclusive("P111-P000 gain gap is nonpositive")
    contrasts = {}
    for cell, values in contrast_samples.items():
        contrasts[cell] = {
            "gain_delta": interval(
                point[cell]["gain"] - point["P000"]["gain"], values["gain_delta"],
            ),
            "cvar5_delta": interval(
                point[cell]["cvar5"] - point["P000"]["cvar5"], values["cvar5_delta"],
            ),
            "regret_recovery": interval(
                (point[cell]["gain"] - point["P000"]["gain"]) / gap_point,
                values["regret_recovery"],
            ),
        }
    interactions = {}
    for name, (pair, first, second) in INTERACTIONS.items():
        point_value = (
            point[pair]["gain"] - point[first]["gain"]
            - point[second]["gain"] + point["P000"]["gain"]
        )
        interactions[name] = interval(point_value, interaction_samples[name])
    return {
        "point": point,
        "cells": cell_intervals,
        "contrasts": contrasts,
        "interactions": interactions,
        "diagnostic_gap": interval(gap_point, gap_samples),
    }


def policy_counts(policy: dict[str, Any]) -> dict[str, int]:
    ordered = sorted(policy["assignments"])
    severe = sum(
        name in policy["selected"]
        and any(float(policy["gains"][operator][index]) <= SEVERE_GAIN for operator in OPERATORS)
        for index, name in enumerate(ordered)
    )
    hard = sum(
        name in policy["selected"]
        and any(float(policy["gains"][operator][index]) <= HARD_GAIN for operator in OPERATORS)
        for index, name in enumerate(ordered)
    )
    negative = sum(
        name in policy["selected"] and policy["assignments"][name] == 1 for name in ordered
    )
    return {
        "selected": len(policy["selected"]),
        "severe": severe,
        "hard": hard,
        "negative": negative,
    }


def replay_base(
    records: dict[str, dict[str, Any]],
    policy: dict[str, Any],
    metadata: dict[str, Any],
    bootstrap_result: dict[str, Any],
    context: Any,
) -> dict[str, Any]:
    policy_map = metadata["policy_map"]
    ordered_names = sorted(records)
    name_indices = {name: index for index, name in enumerate(ordered_names)}
    checks: dict[str, bool] = {
        "all_policy_rows_exact": True,
        "assignment_operator_invariant": True,
    }
    policy_row_mismatches = 0
    max_abs_difference = 0.0
    for name in ordered_names:
        assignment = policy["assignments"][name]
        selected = name in policy["selected"]
        robust = robust_value(records[name], "q05", assignment)
        oracle = oracle_action(records[name])
        observed_selected = set()
        for operator in OPERATORS:
            row = policy_map[(name, operator)]
            expected_exact = {
                "fold": records[name]["fold"],
                "selected": assignment + 1 if selected else 0,
                "oracle_selected": 0 if oracle is None else oracle + 1,
                "negative_oracle": oracle == 1,
            }
            exact_ok = (
                finite_int(row["fold"], "fold") == expected_exact["fold"]
                and finite_int(row["selected"], "selected") == expected_exact["selected"]
                and finite_int(row["oracle_selected"], "oracle_selected") == expected_exact["oracle_selected"]
                and bool_value(row["negative_oracle"]) == expected_exact["negative_oracle"]
            )
            observed_selected.add(finite_int(row["selected"], "selected"))
            expected_numeric = {
                "gain": policy["gains"][operator][name_indices[name]],
                "oracle_gain": (
                    0.0 if oracle is None else records[name]["truth"][(operator, oracle)]
                ),
                "robust_score": robust,
                "q05_score": 0.0 if not selected else records[name]["q05"][(operator, assignment)],
                "mean_score": 0.0 if not selected else records[name]["mean"][(operator, assignment)],
                "severe_score": 0.0 if not selected else records[name]["risk"][(operator, assignment)],
            }
            numeric_ok = True
            for field, expected in expected_numeric.items():
                observed = finite_float(row[field], field)
                difference = abs(observed - float(expected))
                max_abs_difference = max(max_abs_difference, difference)
                numeric_ok &= observed == float(expected)
            row_ok = exact_ok and numeric_ok
            policy_row_mismatches += not row_ok
            checks["all_policy_rows_exact"] &= row_ok
        checks["assignment_operator_invariant"] &= len(observed_selected) == 1

    counts = policy_counts(policy)
    cell_rows = read_csv(
        asset_path(context, "r5_cell_summary", kind="file"), CELL_HEADER,
    )
    if len(cell_rows) != 4:
        raise DiagnosticInconclusive("R5 compact cell summary row count mismatch")
    primary_rows = [row for row in cell_rows if row["cell"] == PRIMARY_CELL]
    if len(primary_rows) != 1:
        raise DiagnosticInconclusive("R5 compact cell summary lacks one primary cell")
    primary = primary_rows[0]
    point = bootstrap_result["point"]["P000"]
    checks.update({
        "coverage_exact": counts["selected"] == 78 == finite_int(primary["selected_groups"], "selected_groups"),
        "per_fold_coverage_exact": all(
            len(policy["selected_by_fold"][fold]) == SELECTED_PER_FOLD for fold in FOLDS
        ),
        "severe_exact": counts["severe"] == 10 == finite_int(primary["selected_severe_groups"], "selected_severe_groups"),
        "hard_exact": counts["hard"] == 3 == finite_int(primary["selected_hard_groups"], "selected_hard_groups"),
        "negative_selection_exact": (
            counts["negative"] == 9 == finite_int(primary["negative_selected_groups"], "negative_selected_groups")
        ),
        "d_ref_gain_exact": close(
            point["d_ref_gain"], finite_float(primary["d_ref_gain_db"], "d_ref_gain_db")
        ),
        "d_rep_gain_exact": close(
            point["d_rep_gain"], finite_float(primary["d_rep_gain_db"], "d_rep_gain_db")
        ),
        "gain_point_exact": close(
            point["gain"], finite_float(primary["gain_point_db"], "gain_point_db")
        ),
        "base_d_ref_frozen_value": close(point["d_ref_gain"], 0.02670895556608836),
        "base_d_rep_frozen_value": close(point["d_rep_gain"], 0.025940994421641033),
        "base_worse_gain_frozen_value": close(point["gain"], 0.025940994421641033),
    })
    r5_bootstrap = read_json_object(
        asset_path(context, "r5_bootstrap_summary", kind="file"), "r5_bootstrap_summary",
    )
    for metric in ("gain", "retention"):
        if not isinstance(r5_bootstrap.get(metric), dict):
            raise DiagnosticInconclusive(f"R5 bootstrap summary lacks {metric}")
        for bound in ("point", "lcb95", "ucb95"):
            if bound not in r5_bootstrap[metric]:
                raise DiagnosticInconclusive(f"R5 bootstrap summary lacks {metric}.{bound}")
            checks[f"{metric}_{bound}_exact"] = close(
                bootstrap_result["cells"]["P000"][metric][bound],
                finite_float(str(r5_bootstrap[metric][bound]), f"{metric}.{bound}"),
            )
    r5_gate = read_json_object(
        asset_path(context, "r5_gate_summary", kind="file"), "r5_gate_summary",
    )
    checks["gate_counts_exact"] = (
        r5_gate.get("selected_severe_groups") == 10
        and r5_gate.get("selected_hard_groups") == 3
    )
    return {
        "schema_version": 1,
        "checks": checks,
        "valid": all(checks.values()),
        "policy_row_mismatches": policy_row_mismatches,
        "policy_max_abs_difference": max_abs_difference,
        "counts": counts,
        "point": point,
    }


def attribution_rows(
    policies: dict[str, dict[str, Any]], boot: dict[str, Any],
) -> list[dict[str, Any]]:
    base = policy_counts(policies["P000"])
    rows = []
    for cell in ATTRIBUTION_CELLS:
        counts = policy_counts(policies[cell])
        contrast = boot["contrasts"][cell]
        utility = (
            contrast["gain_delta"]["lcb95"] >= UTILITY_DELTA
            and contrast["regret_recovery"]["lcb95"] >= RECOVERY_FRACTION
            and counts["severe"] <= base["severe"]
            and counts["hard"] <= base["hard"]
            and contrast["cvar5_delta"]["lcb95"] >= SAFETY_NONINFERIORITY
        )
        safety = (
            counts["severe"] == 0
            and counts["hard"] == 0
            and contrast["gain_delta"]["lcb95"] >= SAFETY_NONINFERIORITY
            and contrast["cvar5_delta"]["lcb95"] >= SAFETY_NONINFERIORITY
        )
        utility_excluded = (
            contrast["gain_delta"]["ucb95"] < UTILITY_DELTA
            or contrast["regret_recovery"]["ucb95"] < RECOVERY_FRACTION
            or counts["severe"] > base["severe"]
            or counts["hard"] > base["hard"]
            or contrast["cvar5_delta"]["ucb95"] < SAFETY_NONINFERIORITY
        )
        safety_excluded = (
            counts["severe"] != 0
            or counts["hard"] != 0
            or contrast["gain_delta"]["ucb95"] < SAFETY_NONINFERIORITY
            or contrast["cvar5_delta"]["ucb95"] < SAFETY_NONINFERIORITY
        )
        rows.append({
            "cell": cell,
            "scope": "single" if cell.count("1") == 1 else "pair",
            "gain_delta_db": contrast["gain_delta"]["point"],
            "gain_delta_lcb95_db": contrast["gain_delta"]["lcb95"],
            "gain_delta_ucb95_db": contrast["gain_delta"]["ucb95"],
            "regret_recovery": contrast["regret_recovery"]["point"],
            "regret_recovery_lcb95": contrast["regret_recovery"]["lcb95"],
            "regret_recovery_ucb95": contrast["regret_recovery"]["ucb95"],
            "cvar5_delta_db": contrast["cvar5_delta"]["point"],
            "cvar5_delta_lcb95_db": contrast["cvar5_delta"]["lcb95"],
            "cvar5_delta_ucb95_db": contrast["cvar5_delta"]["ucb95"],
            "selected_severe_groups": counts["severe"],
            "selected_hard_groups": counts["hard"],
            "utility_attributable": utility,
            "safety_attributable": safety,
            "utility_decisively_excluded": utility_excluded,
            "safety_decisively_excluded": safety_excluded,
        })
    return rows


def synthetic_records() -> dict[str, dict[str, Any]]:
    records = {}
    for fold in FOLDS:
        for local_index in range(192):
            name = f"synthetic_{fold}_{local_index:03d}"
            rank = 1.0 - 0.001 * local_index
            truth_positive = 0.40 - 0.0005 * local_index
            truth_negative = truth_positive - 0.10
            if local_index in (3, 17, 31):
                truth_negative = -0.30 - 0.01 * (local_index % 2)
            record = {"fold": fold, "truth": {}, "mean": {}, "q05": {}, "risk": {}, "label": {}}
            for operator, offset in zip(OPERATORS, (0.0, -0.002)):
                for action, target in enumerate((truth_positive, truth_negative)):
                    value = target + offset
                    record["truth"][(operator, action)] = value
                    record["mean"][(operator, action)] = rank + 0.05 * action
                    record["q05"][(operator, action)] = rank + 0.10 * action
                    record["risk"][(operator, action)] = 0.9 if value <= SEVERE_GAIN else 0.1
                    record["label"][(operator, action)] = int(
                        min(truth_negative + candidate_offset for candidate_offset in (0.0, -0.002))
                        <= SEVERE_GAIN
                    ) if action == 1 else 0
            records[name] = record
    return records


def contract(context_path: Path) -> None:
    context = load_context(context_path, "contract")
    prepare_phase_output(context)
    records = synthetic_records()
    policies = {cell: build_policy(records, bits) for cell, bits in CELL_BITS.items()}
    oracle = oracle_gains(records)
    first = bootstrap(policies, oracle, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED)
    second = bootstrap(policies, oracle, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED)
    checks = {
        "route_identity": context.route_id == ROUTE_ID and context.operation_id == OPERATION_ID,
        "cpu_only": context.device == "cpu",
        "protected_roles_blocked": not any(context.protected_data_permissions.values()),
        "formal_group_scale": len(records) == 384,
        "two_folds": {record["fold"] for record in records.values()} == set(FOLDS),
        "eight_cells": set(policies) == set(CELL_BITS),
        "fixed_total_coverage": all(len(policy["selected"]) == 78 for policy in policies.values()),
        "fixed_fold_coverage": all(
            len(policy["selected_by_fold"][fold]) == SELECTED_PER_FOLD
            for policy in policies.values() for fold in FOLDS
        ),
        "risk_backfill_exercised": (
            len(policies["P001"]["removed"]) > 0
            and len(policies["P001"]["removed"]) == len(policies["P001"]["backfilled"])
        ),
        "risk_backfill_safe": (
            policy_counts(policies["P001"])["severe"] == 0
            and policy_counts(policies["P001"])["hard"] == 0
        ),
        "deterministic_bootstrap": first == second,
        "finite_outputs": all(
            math.isfinite(metric["point"])
            for cell in first["cells"].values() for metric in cell.values()
        ),
        "positive_diagnostic_gap": first["diagnostic_gap"]["point"] > 0.0,
        "full_scale_bootstrap_exercised": (
            BOOTSTRAP_DRAWS == 4000
            and first["cells"]["P000"]["gain"]["lcb95"]
            == second["cells"]["P000"]["gain"]["lcb95"]
        ),
        "bounded_work_class": len(records) * len(CELL_BITS) == 3072 and BOOTSTRAP_DRAWS == 4000,
    }
    write_contract_result(context, checks=checks)


def write_inconclusive_bundle(
    context: Any, reason: str, started_wall: float, started_cpu: float,
) -> None:
    common = {
        "schema_version": 1,
        "status": "input_or_attribution_inconclusive",
        "reason": reason,
        "r5_terminal_changed": False,
        "r8_terminal_changed": False,
    }
    for filename in (
        "r9_a0_contract_summary.json",
        "r9_a0_provenance_and_access.json",
        "r9_a0_input_identity.json",
        "r9_a0_base_replay_summary.json",
        "r9_a0_bootstrap_summary.json",
        "r9_a0_gate_summary.json",
    ):
        atomic_json(context.phase_output_path / filename, common)
    placeholder = [{"status": "inconclusive", "reason": reason}]
    for filename in (
        "r9_a0_factorial_cell_summary.csv",
        "r9_a0_component_attribution.csv",
        "r9_a0_interaction_summary.csv",
        "r9_a0_risk_veto_summary.csv",
    ):
        write_csv(context.phase_output_path / filename, placeholder)
    atomic_json(context.phase_output_path / "r9_a0_resource_summary.json", {
        **common,
        "wall_seconds": time.perf_counter() - started_wall,
        "cpu_seconds": time.process_time() - started_cpu,
        "gpu_used": False,
    })
    write_run_result(
        context,
        state="COMPLETED_GATE_INCONCLUSIVE",
        decision="R9_A0_INPUT_OR_ATTRIBUTION_INCONCLUSIVE_STOP",
        authorizes="NONE",
        details={
            "reason": reason,
            "r5_terminal_changed": False,
            "r8_terminal_changed": False,
        },
    )


def run(context_path: Path) -> None:
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    context = load_context(context_path, "run")
    prepare_phase_output(context)
    try:
        r5_closeout = read_json_object(
            asset_path(context, "r5_closeout", kind="file"), "r5_closeout",
        )
        r5_identity = {
            "route_id": R5_ROUTE_ID,
            "operation_id": R5_OPERATION_ID,
            "run_id": R5_RUN_ID,
            "route_commit": R5_ROUTE_COMMIT,
            "runner_sha256": R5_RUNNER_SHA256,
        }
        if any(r5_closeout.get(key) != value for key, value in r5_identity.items()) or (
            r5_closeout.get("state"),
            r5_closeout.get("decision"),
            r5_closeout.get("authorizes"),
        ) != (
            "COMPLETED_GATE_FAIL",
            "R5_A0_SPATIAL_RESPONSE_FUTILITY_OR_SAFETY_FAIL_STOP",
            "NONE",
        ):
            raise DiagnosticInconclusive("R5 typed closeout identity or terminal mismatch")

        r8_closeout = read_json_object(
            asset_path(context, "r8_closeout", kind="file"), "r8_closeout",
        )
        r8_identity = {
            "route_id": R8_ROUTE_ID,
            "operation_id": R8_OPERATION_ID,
            "run_id": R8_RUN_ID,
            "route_commit": R8_ROUTE_COMMIT,
            "runner_sha256": R5_RUNNER_SHA256,
        }
        if any(r8_closeout.get(key) != value for key, value in r8_identity.items()) or (
            r8_closeout.get("state"),
            r8_closeout.get("decision"),
            r8_closeout.get("authorizes"),
        ) != (
            "COMPLETED_GATE_PASS",
            "R8_A0_SCORE_AND_LABEL_IDENTITY_PASS",
            "R8_NEXT_ATTRIBUTION_CONTRACT_REVIEW_ONLY",
        ):
            raise DiagnosticInconclusive("R8 typed closeout identity or authorization mismatch")
        r8_details = r8_closeout.get("details", {})
        if not isinstance(r8_details, dict):
            raise DiagnosticInconclusive("R8 PASS details are not an object")
        if not (
            r8_details.get("native_score_mismatches") == 0
            and r8_details.get("policy_row_mismatches") == 0
            and r8_details.get("any_operator_severe_label_mismatches") == 0
            and r8_details.get("r5_terminal_changed") is False
        ):
            raise DiagnosticInconclusive("R8 PASS detail contract mismatch")

        records, metadata = load_records(context)
        write_workload_progress(context, completed_units=4, stage="inputs_verified")
        policies = {cell: build_policy(records, bits) for cell, bits in CELL_BITS.items()}
        oracle = oracle_gains(records)
        boot = bootstrap(policies, oracle, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED)
        write_workload_progress(context, completed_units=4004, stage="bootstrap_complete")
        base_replay = replay_base(records, policies["P000"], metadata, boot, context)
        if not base_replay["valid"]:
            failed = sorted(key for key, value in base_replay["checks"].items() if not value)
            raise DiagnosticInconclusive(f"R5 base replay mismatch: {failed}")

        attribution = attribution_rows(policies, boot)
        attributed = [
            row["cell"] for row in attribution
            if row["utility_attributable"] or row["safety_attributable"]
        ]
        all_excluded = all(
            row["utility_decisively_excluded"] and row["safety_decisively_excluded"]
            for row in attribution
        )
        if attributed:
            state = "COMPLETED_GATE_PASS"
            decision = "R9_A0_DECISION_FACTORIAL_ATTRIBUTION_PASS"
            authorizes = "R9_NEXT_CONTRACT_REVIEW_ONLY"
        elif all_excluded:
            state = "COMPLETED_GATE_FAIL"
            decision = "R9_A0_NO_SINGLE_OR_PAIR_ATTRIBUTION_STOP"
            authorizes = "NONE"
        else:
            state = "COMPLETED_GATE_INCONCLUSIVE"
            decision = "R9_A0_INPUT_OR_ATTRIBUTION_INCONCLUSIVE_STOP"
            authorizes = "NONE"

        cell_rows = []
        risk_rows = []
        for cell in CELL_BITS:
            policy = policies[cell]
            counts = policy_counts(policy)
            point = boot["point"][cell]
            cell_rows.append({
                "cell": cell,
                "action_oracle": CELL_BITS[cell][0],
                "coverage_oracle": CELL_BITS[cell][1],
                "risk_oracle": CELL_BITS[cell][2],
                "selected_groups": counts["selected"],
                "fold0_selected_groups": len(policy["selected_by_fold"][0]),
                "fold1_selected_groups": len(policy["selected_by_fold"][1]),
                "negative_selected_groups": counts["negative"],
                "selected_severe_groups": counts["severe"],
                "selected_hard_groups": counts["hard"],
                "d_ref_gain_db": point["d_ref_gain"],
                "d_rep_gain_db": point["d_rep_gain"],
                "worse_operator_gain_db": point["gain"],
                "retention": point["retention"],
                "cvar5_gain_db": point["cvar5"],
            })
            selected_risk = [
                max(
                    records[name]["risk"][(operator, policy["assignments"][name])]
                    for operator in OPERATORS
                )
                for name in policy["selected"]
            ]
            removed_risk = [
                max(
                    records[name]["risk"][(operator, policy["assignments"][name])]
                    for operator in OPERATORS
                )
                for name in policy["removed"]
            ]
            risk_rows.append({
                "cell": cell,
                "risk_oracle": CELL_BITS[cell][2],
                "unsafe_ineligible_groups": len(policy["ineligible"]),
                "top39_removed_groups": len(policy["removed"]),
                "safe_backfilled_groups": len(policy["backfilled"]),
                "selected_severe_groups": counts["severe"],
                "selected_hard_groups": counts["hard"],
                "selected_mean_predicted_severe_score": sum(selected_risk) / len(selected_risk),
                "removed_mean_predicted_severe_score": (
                    "" if not removed_risk else sum(removed_risk) / len(removed_risk)
                ),
            })
        interaction_rows = [
            {
                "interaction": name,
                "point_db": value["point"],
                "lcb95_db": value["lcb95"],
                "ucb95_db": value["ucb95"],
                "interpretation": "descriptive_coupling_only",
            }
            for name, value in boot["interactions"].items()
        ]
        contract_summary = {
            "schema_version": 1,
            "route_id": ROUTE_ID,
            "operation_id": OPERATION_ID,
            "scientific_evidence_role": "post_hoc_development_mechanism_diagnostic",
            "runtime_schema_evidence_role": context.evidence_role,
            "candidate_source": "R8_verified_authoritative_R5_ensemble_rows",
            "per_seed_reconstruction": False,
            "cells": list(CELL_BITS),
            "folds": list(FOLDS),
            "operators": list(OPERATORS),
            "actions": list(ACTIONS),
            "coverage": COVERAGE,
            "selected_per_fold": SELECTED_PER_FOLD,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "thresholds": {
                "severe_gain_db": SEVERE_GAIN,
                "hard_gain_db": HARD_GAIN,
                "utility_delta_db": UTILITY_DELTA,
                "regret_recovery": RECOVERY_FRACTION,
                "safety_noninferiority_db": SAFETY_NONINFERIORITY,
            },
        }
        provenance = {
            "schema_version": 1,
            "route_commit": context.route_commit,
            "source_r5_identity": r5_identity,
            "source_r8_identity": r8_identity,
            "r5_terminal_preserved": True,
            "r6_terminal_preserved": True,
            "r7_terminal_preserved": True,
            "r8_terminal_preserved": True,
            "training_run": False,
            "inference_run": False,
            "candidate_generation_rerun": False,
            "per_seed_reconstruction": False,
            "confirmation_images_targets_outcomes_touched": False,
            "canary_touched": False,
            "locked_test_touched": False,
        }
        input_identity = {
            "schema_version": 1,
            "r5_closeout_terminal": {
                key: r5_closeout.get(key) for key in ("state", "decision", "authorizes")
            },
            "r8_closeout_terminal": {
                key: r8_closeout.get(key) for key in ("state", "decision", "authorizes")
            },
            "row_counts": {key: value for key, value in metadata.items() if key != "policy_map"},
            "asset_sha256": {
                identifier: context.assets[identifier].sha256 for identifier in sorted(context.assets)
            },
        }
        gate_summary = {
            "schema_version": 1,
            "structural_valid": True,
            "base_replay_valid": True,
            "diagnostic_gap_positive": boot["diagnostic_gap"]["point"] > 0.0,
            "attributed_cells": attributed,
            "all_single_pair_cells_decisively_excluded": all_excluded,
            "state": state,
            "decision": decision,
            "authorizes": authorizes,
            "r5_terminal_changed": False,
            "r8_terminal_changed": False,
        }
        atomic_json(context.phase_output_path / "r9_a0_contract_summary.json", contract_summary)
        atomic_json(context.phase_output_path / "r9_a0_provenance_and_access.json", provenance)
        atomic_json(context.phase_output_path / "r9_a0_input_identity.json", input_identity)
        atomic_json(context.phase_output_path / "r9_a0_base_replay_summary.json", base_replay)
        write_csv(context.phase_output_path / "r9_a0_factorial_cell_summary.csv", cell_rows)
        atomic_json(context.phase_output_path / "r9_a0_bootstrap_summary.json", {
            "schema_version": 1,
            **boot,
        })
        write_csv(context.phase_output_path / "r9_a0_component_attribution.csv", attribution)
        write_csv(context.phase_output_path / "r9_a0_interaction_summary.csv", interaction_rows)
        write_csv(context.phase_output_path / "r9_a0_risk_veto_summary.csv", risk_rows)
        atomic_json(context.phase_output_path / "r9_a0_gate_summary.json", gate_summary)
        atomic_json(context.phase_output_path / "r9_a0_resource_summary.json", {
            "schema_version": 1,
            "wall_seconds": time.perf_counter() - started_wall,
            "cpu_seconds": time.process_time() - started_cpu,
            "max_rss_mib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0,
            "raw_rows_read": metadata["candidate_rows"] + metadata["policy_rows"],
            "primary_rows_used": (
                metadata["primary_candidate_rows"] + metadata["primary_policy_rows"]
            ),
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "gpu_used": False,
        })
        write_workload_progress(context, completed_units=4008, stage="finalized")
        write_run_result(
            context,
            state=state,
            decision=decision,
            authorizes=authorizes,
            details={
                "attributed_cells": attributed,
                "base_gain_db": boot["point"]["P000"]["gain"],
                "diagnostic_ceiling_gain_db": boot["point"]["P111"]["gain"],
                "diagnostic_gap_db": boot["diagnostic_gap"]["point"],
                "base_selected_severe_groups": policy_counts(policies["P000"])["severe"],
                "base_selected_hard_groups": policy_counts(policies["P000"])["hard"],
                "r5_terminal_changed": False,
                "r8_terminal_changed": False,
            },
        )
    except DiagnosticInconclusive as exc:
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
