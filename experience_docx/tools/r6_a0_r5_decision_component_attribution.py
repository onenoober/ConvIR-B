#!/usr/bin/env python3
"""Frozen post-hoc R6 attribution of the R5 action/coverage/risk decision contract."""

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


ROUTE_ID = "haze4k_v5_r6_r5_decision_component_attribution_20260719"
OPERATION_ID = "R6_A0_FROZEN_R5_DECISION_COMPONENT_ATTRIBUTION_AUDIT"
R5_ROUTE_ID = "haze4k_v5_r5_spatial_candidate_response_sufficiency_20260719"
R5_OPERATION_ID = "R5_A0_FROZEN_SPATIAL_RESPONSE_SUFFICIENCY_SCREEN"
R5_RUN_ID = "r5-a0-spatial-response-screen-r2"
R5_ROUTE_COMMIT = "7e75eed504b2ead65a1971ec250dc7f59a79574d"
R5_RUNNER_SHA256 = "336c7e1beccb793229beb533ba12367261e702866497c388ee2a4fa88d12718b"
PRIMARY_CELL = "S1_TRUE_SPATIAL_RESPONSE"
ACTIONS = ("state_positive_full", "state_negative_full")
OPERATORS = ("D_ref", "D_rep")
FOLDS = (0, 1)
SEEDS = (3407, 3411)
CELL_BITS = {f"P{a}{c}{r}": (a, c, r) for a in range(2) for c in range(2) for r in range(2)}
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
EXPECTED_HEADERS = {
    "r5_per_seed_predictions": {
        "action", "cell", "fold", "mean_score", "name", "operator",
        "q05_score", "seed", "severe_score", "target_gain_db",
    },
    "r5_candidate_scores": {
        "action", "cell", "fold", "mean_score", "name", "operator",
        "q05_score", "severe_label", "severe_score", "target_gain_db",
    },
    "r5_policy_rows": {
        "cell", "fold", "gain", "mean_score", "name", "negative_oracle",
        "operator", "oracle_gain", "oracle_selected", "q05_score",
        "robust_score", "selected", "severe_score",
    },
}


class DiagnosticInconclusive(RuntimeError):
    pass


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    fields = sorted({key for row in rows for key in row})
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path, expected: set[str]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or set(reader.fieldnames) != expected:
            raise DiagnosticInconclusive(f"CSV header mismatch: {path.name}")
        rows = list(reader)
    if not rows or any(None in row or any(value is None for value in row.values()) for row in rows):
        raise DiagnosticInconclusive(f"CSV row contract failed: {path.name}")
    return rows


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


def action_index(value: str) -> int:
    try:
        return ACTIONS.index(value)
    except ValueError as exc:
        raise DiagnosticInconclusive(f"unexpected action: {value}") from exc


def load_records(context: Any) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    seed_path = asset_path(context, "r5_per_seed_predictions", kind="file")
    score_path = asset_path(context, "r5_candidate_scores", kind="file")
    policy_path = asset_path(context, "r5_policy_rows", kind="file")
    seed_rows = read_csv(seed_path, EXPECTED_HEADERS["r5_per_seed_predictions"])
    score_rows = read_csv(score_path, EXPECTED_HEADERS["r5_candidate_scores"])
    policy_rows = read_csv(policy_path, EXPECTED_HEADERS["r5_policy_rows"])
    if (len(seed_rows), len(score_rows), len(policy_rows)) != (12288, 6144, 3072):
        raise DiagnosticInconclusive("R5 raw row counts do not match the frozen contract")

    primary_seed = [row for row in seed_rows if row["cell"] == PRIMARY_CELL]
    primary_scores = [row for row in score_rows if row["cell"] == PRIMARY_CELL]
    primary_policy = [row for row in policy_rows if row["cell"] == PRIMARY_CELL]
    if (len(primary_seed), len(primary_scores), len(primary_policy)) != (3072, 1536, 768):
        raise DiagnosticInconclusive("R5 primary-cell row counts are incomplete")

    seed_map: dict[tuple[str, str, int, int], dict[str, str]] = {}
    for row in primary_seed:
        fold = int(row["fold"]); seed = int(row["seed"]); action = action_index(row["action"])
        if fold not in FOLDS or seed not in SEEDS or row["operator"] not in OPERATORS:
            raise DiagnosticInconclusive("R5 per-seed identity is outside the frozen scope")
        key = (row["name"], row["operator"], action, seed)
        if key in seed_map:
            raise DiagnosticInconclusive("duplicate R5 per-seed row")
        seed_map[key] = row

    records: dict[str, dict[str, Any]] = {}
    score_keys = set()
    for row in primary_scores:
        fold = int(row["fold"]); operator = row["operator"]; action = action_index(row["action"]); name = row["name"]
        if fold not in FOLDS or operator not in OPERATORS:
            raise DiagnosticInconclusive("R5 candidate-score identity is outside the frozen scope")
        key = (name, operator, action)
        if key in score_keys:
            raise DiagnosticInconclusive("duplicate R5 candidate-score row")
        score_keys.add(key)
        record = records.setdefault(
            name,
            {"fold": fold, "truth": {}, "mean": {}, "q05": {}, "risk": {}, "label": {}},
        )
        if record["fold"] != fold:
            raise DiagnosticInconclusive("one clean-image name appears in multiple folds")
        record["truth"][(operator, action)] = float(row["target_gain_db"])
        record["mean"][(operator, action)] = float(row["mean_score"])
        record["q05"][(operator, action)] = float(row["q05_score"])
        record["risk"][(operator, action)] = float(row["severe_score"])
        record["label"][(operator, action)] = int(row["severe_label"])
        seeds = [seed_map.get((name, operator, action, seed)) for seed in SEEDS]
        if any(item is None for item in seeds):
            raise DiagnosticInconclusive("candidate score lacks paired seed rows")
        for field in ("mean_score", "q05_score", "severe_score"):
            observed = float(row[field]); recomputed = sum(float(item[field]) for item in seeds if item is not None) / 2.0
            if not close(observed, recomputed):
                raise DiagnosticInconclusive(f"seed ensemble replay mismatch: {field}")
        if any(not close(float(item["target_gain_db"]), float(row["target_gain_db"])) for item in seeds if item is not None):
            raise DiagnosticInconclusive("target gain changes across R5 seeds")
        if int(row["severe_label"]) != int(float(row["target_gain_db"]) <= SEVERE_GAIN):
            raise DiagnosticInconclusive("saved severe label disagrees with the frozen threshold")

    required = {(operator, action) for operator in OPERATORS for action in range(2)}
    if len(records) != 384 or any(set(record["truth"]) != required for record in records.values()):
        raise DiagnosticInconclusive("R5 primary action/operator grid is incomplete")
    if {record["fold"] for record in records.values()} != set(FOLDS) or any(
        sum(record["fold"] == fold for record in records.values()) != 192 for fold in FOLDS
    ):
        raise DiagnosticInconclusive("R5 primary fold sizes are not 192/192")

    policy_map = {}
    for row in primary_policy:
        key = (row["name"], row["operator"]); fold = int(row["fold"]); selected = int(row["selected"])
        if key in policy_map or row["name"] not in records or row["operator"] not in OPERATORS \
                or fold != records[row["name"]]["fold"] or selected not in (0, 1, 2):
            raise DiagnosticInconclusive("R5 policy row identity is invalid")
        policy_map[key] = row
    if len(policy_map) != 768:
        raise DiagnosticInconclusive("R5 policy grid is incomplete")
    return records, {
        "seed_rows": len(seed_rows), "score_rows": len(score_rows), "policy_rows": len(policy_rows),
        "primary_seed_rows": len(primary_seed), "primary_score_rows": len(primary_scores),
        "primary_policy_rows": len(primary_policy), "policy_map": policy_map,
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
        name: robust_value(record, rank_field, assignments[name]) for name, record in records.items()
    }
    selected: set[str] = set()
    vetoed = set()
    for fold in FOLDS:
        names = [name for name, record in records.items() if record["fold"] == fold]
        ordered = sorted(names, key=lambda name: (-rank_scores[name], tie_key(fold, name)))
        eligible = []
        for name in ordered:
            is_vetoed = bool(risk_bit) and robust_value(records[name], "truth", assignments[name]) <= SEVERE_GAIN
            if is_vetoed:
                vetoed.add(name)
            else:
                eligible.append(name)
        if len(eligible) < SELECTED_PER_FOLD:
            raise DiagnosticInconclusive("risk veto cannot safely backfill fixed coverage")
        selected.update(eligible[:SELECTED_PER_FOLD])
    gains = {
        operator: [
            float(records[name]["truth"][(operator, assignments[name])]) if name in selected else 0.0
            for name in sorted(records)
        ]
        for operator in OPERATORS
    }
    return {
        "assignments": assignments, "rank_scores": rank_scores, "selected": selected,
        "vetoed": vetoed, "gains": gains,
    }


def oracle_gains(records: dict[str, dict[str, Any]]) -> dict[str, list[float]]:
    result = {operator: [] for operator in OPERATORS}
    for name in sorted(records):
        action = oracle_action(records[name])
        for operator in OPERATORS:
            result[operator].append(0.0 if action is None else float(records[name]["truth"][(operator, action)]))
    return result


def evaluate_indices(policies: dict[str, dict[str, Any]], oracle: dict[str, list[float]], indices: Any) -> dict[str, dict[str, float]]:
    import numpy as np

    oracle_means = {operator: float(np.asarray(oracle[operator])[indices].mean()) for operator in OPERATORS}
    result = {}
    for cell, policy in policies.items():
        means = {operator: float(np.asarray(policy["gains"][operator])[indices].mean()) for operator in OPERATORS}
        tails = {operator: cvar(np.asarray(policy["gains"][operator])[indices]) for operator in OPERATORS}
        worst = min(OPERATORS, key=lambda operator: means[operator])
        result[cell] = {
            "gain": means[worst],
            "retention": means[worst] / max(oracle_means[worst], 1.0e-12),
            "cvar5": min(tails.values()),
            "d_ref_gain": means["D_ref"],
            "d_rep_gain": means["D_rep"],
        }
    return result


def bootstrap(policies: dict[str, dict[str, Any]], oracle: dict[str, list[float]], draws: int, seed: int) -> dict[str, Any]:
    import numpy as np

    count = len(next(iter(oracle.values())))
    point = evaluate_indices(policies, oracle, np.arange(count))
    samples = {cell: {metric: [] for metric in values} for cell, values in point.items()}
    contrast_samples = {cell: {"gain_delta": [], "cvar5_delta": [], "regret_recovery": []} for cell in CELL_BITS if cell != "P000"}
    interaction_samples = {name: [] for name in INTERACTIONS}
    generator = np.random.default_rng(seed)
    for _draw in range(draws):
        draw = evaluate_indices(policies, oracle, generator.integers(0, count, count))
        for cell, values in draw.items():
            for metric, value in values.items():
                samples[cell][metric].append(value)
        gap = draw["P111"]["gain"] - draw["P000"]["gain"]
        if gap <= 1.0e-12:
            raise DiagnosticInconclusive("P111-P000 oracle gap is nonpositive in bootstrap")
        for cell in contrast_samples:
            contrast_samples[cell]["gain_delta"].append(draw[cell]["gain"] - draw["P000"]["gain"])
            contrast_samples[cell]["cvar5_delta"].append(draw[cell]["cvar5"] - draw["P000"]["cvar5"])
            contrast_samples[cell]["regret_recovery"].append((draw[cell]["gain"] - draw["P000"]["gain"]) / gap)
        for name, (pair, first, second) in INTERACTIONS.items():
            interaction_samples[name].append(
                draw[pair]["gain"] - draw[first]["gain"] - draw[second]["gain"] + draw["P000"]["gain"]
            )
    cell_intervals = {
        cell: {metric: interval(point[cell][metric], values) for metric, values in samples[cell].items()}
        for cell in point
    }
    gap_point = point["P111"]["gain"] - point["P000"]["gain"]
    if gap_point <= 1.0e-12:
        raise DiagnosticInconclusive("P111-P000 oracle gap is nonpositive")
    contrasts = {}
    for cell, values in contrast_samples.items():
        contrasts[cell] = {
            "gain_delta": interval(point[cell]["gain"] - point["P000"]["gain"], values["gain_delta"]),
            "cvar5_delta": interval(point[cell]["cvar5"] - point["P000"]["cvar5"], values["cvar5_delta"]),
            "regret_recovery": interval((point[cell]["gain"] - point["P000"]["gain"]) / gap_point, values["regret_recovery"]),
        }
    interactions = {}
    for name, (pair, first, second) in INTERACTIONS.items():
        point_value = point[pair]["gain"] - point[first]["gain"] - point[second]["gain"] + point["P000"]["gain"]
        interactions[name] = interval(point_value, interaction_samples[name])
    return {"point": point, "cells": cell_intervals, "contrasts": contrasts, "interactions": interactions}


def policy_counts(policy: dict[str, Any]) -> dict[str, int]:
    ordered = sorted(policy["assignments"])
    severe = sum(
        name in policy["selected"] and any(float(policy["gains"][operator][index]) <= SEVERE_GAIN for operator in OPERATORS)
        for index, name in enumerate(ordered)
    )
    hard = sum(
        name in policy["selected"] and any(float(policy["gains"][operator][index]) <= HARD_GAIN for operator in OPERATORS)
        for index, name in enumerate(ordered)
    )
    negative = sum(name in policy["selected"] and policy["assignments"][name] == 1 for name in ordered)
    return {"selected": len(policy["selected"]), "severe": severe, "hard": hard, "negative": negative}


def replay_base(records: dict[str, dict[str, Any]], policy: dict[str, Any], metadata: dict[str, Any], bootstrap_result: dict[str, Any], context: Any) -> dict[str, Any]:
    policy_map = metadata["policy_map"]
    checks: dict[str, bool] = {}
    for name in sorted(records):
        assignment = policy["assignments"][name]; selected = name in policy["selected"]
        robust = robust_value(records[name], "q05", assignment)
        oracle = oracle_action(records[name])
        for operator in OPERATORS:
            row = policy_map[(name, operator)]
            expected_selected = assignment + 1 if selected else 0
            checks.setdefault("policy_rows_exact", True)
            checks["policy_rows_exact"] &= int(row["selected"]) == expected_selected
            checks["policy_rows_exact"] &= close(float(row["gain"]), policy["gains"][operator][sorted(records).index(name)])
            checks["policy_rows_exact"] &= int(row["oracle_selected"]) == (0 if oracle is None else oracle + 1)
            checks["policy_rows_exact"] &= close(float(row["oracle_gain"]), 0.0 if oracle is None else records[name]["truth"][(operator, oracle)])
            checks["policy_rows_exact"] &= close(float(row["robust_score"]), robust)
            checks["policy_rows_exact"] &= close(float(row["q05_score"]), 0.0 if not selected else records[name]["q05"][(operator, assignment)])
            checks["policy_rows_exact"] &= close(float(row["mean_score"]), 0.0 if not selected else records[name]["mean"][(operator, assignment)])
            checks["policy_rows_exact"] &= close(float(row["severe_score"]), 0.0 if not selected else records[name]["risk"][(operator, assignment)])
    counts = policy_counts(policy)
    cell_rows = read_csv(asset_path(context, "r5_cell_summary", kind="file"), {
        "cell", "coverage", "d_ref_gain_db", "d_rep_gain_db", "gain_point_db",
        "negative_selected_groups", "selected_groups", "selected_hard_groups", "selected_severe_groups",
    })
    primary = next((row for row in cell_rows if row["cell"] == PRIMARY_CELL), None)
    if primary is None:
        raise DiagnosticInconclusive("R5 compact cell summary lacks primary cell")
    point = bootstrap_result["point"]["P000"]
    checks.update({
        "coverage_exact": counts["selected"] == 78 == int(primary["selected_groups"]),
        "severe_exact": counts["severe"] == 10 == int(primary["selected_severe_groups"]),
        "hard_exact": counts["hard"] == 3 == int(primary["selected_hard_groups"]),
        "negative_selection_exact": counts["negative"] == 9 == int(primary["negative_selected_groups"]),
        "d_ref_gain_exact": close(point["d_ref_gain"], float(primary["d_ref_gain_db"])),
        "d_rep_gain_exact": close(point["d_rep_gain"], float(primary["d_rep_gain_db"])),
        "gain_point_exact": close(point["gain"], float(primary["gain_point_db"])),
    })
    r5_bootstrap = json.loads(asset_path(context, "r5_bootstrap_summary", kind="file").read_text(encoding="utf-8"))
    for metric in ("gain", "retention"):
        for bound in ("point", "lcb95", "ucb95"):
            checks[f"{metric}_{bound}_exact"] = close(
                bootstrap_result["cells"]["P000"][metric][bound], float(r5_bootstrap[metric][bound])
            )
    r5_gate = json.loads(asset_path(context, "r5_gate_summary", kind="file").read_text(encoding="utf-8"))
    checks["gate_counts_exact"] = r5_gate.get("selected_severe_groups") == 10 and r5_gate.get("selected_hard_groups") == 3
    return {"schema_version": 1, "checks": checks, "valid": all(checks.values()), "counts": counts, "point": point}


def attribution_rows(policies: dict[str, dict[str, Any]], boot: dict[str, Any]) -> list[dict[str, Any]]:
    base = policy_counts(policies["P000"])
    rows = []
    for cell in ATTRIBUTION_CELLS:
        counts = policy_counts(policies[cell]); contrast = boot["contrasts"][cell]
        utility = (
            contrast["gain_delta"]["lcb95"] >= UTILITY_DELTA
            and contrast["regret_recovery"]["lcb95"] >= RECOVERY_FRACTION
            and counts["severe"] <= base["severe"] and counts["hard"] <= base["hard"]
            and contrast["cvar5_delta"]["lcb95"] >= SAFETY_NONINFERIORITY
        )
        safety = (
            counts["severe"] == 0 and counts["hard"] == 0
            and contrast["gain_delta"]["lcb95"] >= SAFETY_NONINFERIORITY
            and contrast["cvar5_delta"]["lcb95"] >= SAFETY_NONINFERIORITY
        )
        utility_excluded = (
            contrast["gain_delta"]["ucb95"] < UTILITY_DELTA
            or contrast["regret_recovery"]["ucb95"] < RECOVERY_FRACTION
            or counts["severe"] > base["severe"] or counts["hard"] > base["hard"]
            or contrast["cvar5_delta"]["ucb95"] < SAFETY_NONINFERIORITY
        )
        safety_excluded = (
            counts["severe"] != 0 or counts["hard"] != 0
            or contrast["gain_delta"]["ucb95"] < SAFETY_NONINFERIORITY
            or contrast["cvar5_delta"]["ucb95"] < SAFETY_NONINFERIORITY
        )
        rows.append({
            "cell": cell, "scope": "single" if cell.count("1") == 1 else "pair",
            "gain_delta_db": contrast["gain_delta"]["point"],
            "gain_delta_lcb95_db": contrast["gain_delta"]["lcb95"],
            "gain_delta_ucb95_db": contrast["gain_delta"]["ucb95"],
            "regret_recovery": contrast["regret_recovery"]["point"],
            "regret_recovery_lcb95": contrast["regret_recovery"]["lcb95"],
            "regret_recovery_ucb95": contrast["regret_recovery"]["ucb95"],
            "cvar5_delta_db": contrast["cvar5_delta"]["point"],
            "cvar5_delta_lcb95_db": contrast["cvar5_delta"]["lcb95"],
            "cvar5_delta_ucb95_db": contrast["cvar5_delta"]["ucb95"],
            "selected_severe_groups": counts["severe"], "selected_hard_groups": counts["hard"],
            "utility_attributable": utility, "safety_attributable": safety,
            "utility_decisively_excluded": utility_excluded, "safety_decisively_excluded": safety_excluded,
        })
    return rows


def synthetic_records() -> dict[str, dict[str, Any]]:
    records = {}
    for index in range(384):
        fold = index // 192; name = f"synthetic_{index:04d}"
        truth0 = 0.30 - index * 0.0002; truth1 = 0.10 + index * 0.00005
        if index % 41 == 0:
            truth0 = -0.30
        record = {"fold": fold, "truth": {}, "mean": {}, "q05": {}, "risk": {}, "label": {}}
        for operator, offset in zip(OPERATORS, (0.0, -0.002)):
            for action, value in enumerate((truth0, truth1)):
                target = value + offset
                record["truth"][(operator, action)] = target
                record["mean"][(operator, action)] = target
                record["q05"][(operator, action)] = target - 0.01
                record["risk"][(operator, action)] = 0.9 if target <= SEVERE_GAIN else 0.1
                record["label"][(operator, action)] = int(target <= SEVERE_GAIN)
        records[name] = record
    return records


def contract(context_path: Path) -> None:
    context = load_context(context_path, "contract")
    prepare_phase_output(context)
    records = synthetic_records()
    policies = {cell: build_policy(records, bits) for cell, bits in CELL_BITS.items()}
    first = bootstrap(policies, oracle_gains(records), 32, BOOTSTRAP_SEED)
    second = bootstrap(policies, oracle_gains(records), 32, BOOTSTRAP_SEED)
    checks = {
        "route_identity": context.route_id == ROUTE_ID and context.operation_id == OPERATION_ID,
        "cpu_only": context.device == "cpu",
        "protected_roles_blocked": not any(context.protected_data_permissions.values()),
        "formal_group_scale": len(records) == 384,
        "eight_cells": set(policies) == set(CELL_BITS),
        "fixed_fold_coverage": all(len(policy["selected"]) == 78 for policy in policies.values()),
        "oracle_veto_safe": policy_counts(policies["P111"])["severe"] == 0 and policy_counts(policies["P111"])["hard"] == 0,
        "deterministic_bootstrap": first == second,
        "finite_finalizer": all(math.isfinite(value["gain"]["point"]) for value in first["cells"].values()),
        "bounded_work": len(records) * len(CELL_BITS) == 3072 and BOOTSTRAP_DRAWS == 4000,
    }
    write_contract_result(context, checks=checks)


def write_inconclusive_bundle(context: Any, message: str, started_wall: float, started_cpu: float) -> None:
    common = {"schema_version": 1, "status": "input_or_base_replay_inconclusive", "reason": message}
    for name in (
        "r6_a0_contract_summary.json", "r6_a0_provenance_and_access.json",
        "r6_a0_input_identity.json", "r6_a0_base_replay_summary.json",
        "r6_a0_bootstrap_summary.json", "r6_a0_gate_summary.json",
    ):
        atomic_json(context.phase_output_path / name, common)
    placeholder = [{"status": "inconclusive", "reason": message}]
    for name in (
        "r6_a0_factorial_cell_summary.csv", "r6_a0_component_attribution.csv",
        "r6_a0_interaction_summary.csv", "r6_a0_risk_veto_summary.csv",
    ):
        write_csv(context.phase_output_path / name, placeholder)
    atomic_json(context.phase_output_path / "r6_a0_resource_summary.json", {
        **common, "wall_seconds": time.perf_counter() - started_wall,
        "cpu_seconds": time.process_time() - started_cpu, "gpu_used": False,
    })
    write_run_result(
        context, state="COMPLETED_GATE_INCONCLUSIVE",
        decision="R6_A0_INPUT_OR_ATTRIBUTION_INCONCLUSIVE_STOP", authorizes="NONE",
        details={"reason": message, "r5_terminal_decision_changed": False},
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
            raise DiagnosticInconclusive("R5 lifecycle identity mismatch")
        closeout = json.loads(asset_path(context, "r5_closeout", kind="file").read_text(encoding="utf-8"))
        if {key: closeout.get(key) for key in ("route_id", "operation_id", "run_id", "route_commit", "runner_sha256")} != {
            key: expected_identity[key] for key in ("route_id", "operation_id", "run_id", "route_commit", "runner_sha256")
        } or (closeout.get("state"), closeout.get("decision"), closeout.get("authorizes")) != (
            "COMPLETED_GATE_FAIL", "R5_A0_SPATIAL_RESPONSE_FUTILITY_OR_SAFETY_FAIL_STOP", "NONE"
        ):
            raise DiagnosticInconclusive("R5 typed closeout identity or terminal mismatch")
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
        attributed = [row["cell"] for row in attribution if row["utility_attributable"] or row["safety_attributable"]]
        all_excluded = all(row["utility_decisively_excluded"] and row["safety_decisively_excluded"] for row in attribution)
        if attributed:
            state = "COMPLETED_GATE_PASS"; decision = "R6_A0_DECISION_COMPONENT_ATTRIBUTION_PASS"; authorizes = "R6_NEXT_CONTRACT_REVIEW_ONLY"
        elif all_excluded:
            state = "COMPLETED_GATE_FAIL"; decision = "R6_A0_NO_SINGLE_OR_PAIR_COMPONENT_ATTRIBUTION_STOP"; authorizes = "NONE"
        else:
            state = "COMPLETED_GATE_INCONCLUSIVE"; decision = "R6_A0_INPUT_OR_ATTRIBUTION_INCONCLUSIVE_STOP"; authorizes = "NONE"

        cell_rows = []
        risk_rows = []
        for cell in CELL_BITS:
            counts = policy_counts(policies[cell]); point = boot["point"][cell]
            cell_rows.append({
                "cell": cell, "action_oracle": CELL_BITS[cell][0], "coverage_oracle": CELL_BITS[cell][1],
                "risk_oracle": CELL_BITS[cell][2], "selected_groups": counts["selected"],
                "negative_selected_groups": counts["negative"], "selected_severe_groups": counts["severe"],
                "selected_hard_groups": counts["hard"], "d_ref_gain_db": point["d_ref_gain"],
                "d_rep_gain_db": point["d_rep_gain"], "worse_operator_gain_db": point["gain"],
                "retention": point["retention"], "cvar5_gain_db": point["cvar5"],
            })
            selected_risk = [
                max(records[name]["risk"][(operator, policies[cell]["assignments"][name])] for operator in OPERATORS)
                for name in policies[cell]["selected"]
            ]
            vetoed_risk = [
                max(records[name]["risk"][(operator, policies[cell]["assignments"][name])] for operator in OPERATORS)
                for name in policies[cell]["vetoed"]
            ]
            risk_rows.append({
                "cell": cell, "risk_oracle": CELL_BITS[cell][2], "vetoed_groups": len(policies[cell]["vetoed"]),
                "selected_severe_groups": counts["severe"], "selected_hard_groups": counts["hard"],
                "selected_mean_predicted_severe_score": sum(selected_risk) / len(selected_risk),
                "vetoed_mean_predicted_severe_score": "" if not vetoed_risk else sum(vetoed_risk) / len(vetoed_risk),
            })
        interaction_rows = [
            {"interaction": name, "point_db": value["point"], "lcb95_db": value["lcb95"], "ucb95_db": value["ucb95"]}
            for name, value in boot["interactions"].items()
        ]
        contract_summary = {
            "schema_version": 1, "route_id": ROUTE_ID, "operation_id": OPERATION_ID,
            "evidence_role_scientific": "post_hoc_mechanism_diagnostic",
            "runtime_schema_evidence_role": context.evidence_role, "cells": list(CELL_BITS),
            "folds": list(FOLDS), "operators": list(OPERATORS), "actions": list(ACTIONS),
            "coverage": COVERAGE, "selected_per_fold": SELECTED_PER_FOLD,
            "bootstrap_draws": BOOTSTRAP_DRAWS, "bootstrap_seed": BOOTSTRAP_SEED,
            "thresholds": {"severe_gain_db": SEVERE_GAIN, "hard_gain_db": HARD_GAIN, "utility_delta_db": UTILITY_DELTA, "regret_recovery": RECOVERY_FRACTION, "safety_noninferiority_db": SAFETY_NONINFERIORITY},
        }
        provenance = {
            "schema_version": 1, "route_commit": context.route_commit, "source_r5_identity": expected_identity,
            "r5_terminal_preserved": True, "training_run": False, "inference_run": False,
            "candidate_generation_rerun": False, "confirmation_images_targets_outcomes_touched": False,
            "canary_touched": False, "locked_test_touched": False,
        }
        input_identity = {
            "schema_version": 1, "r5_lifecycle_identity": identity,
            "row_counts": {key: value for key, value in metadata.items() if key != "policy_map"},
            "asset_sha256": {identifier: context.assets[identifier].sha256 for identifier in sorted(context.assets)},
        }
        gate_summary = {
            "schema_version": 1, "structural_valid": True, "base_replay_valid": True,
            "attributed_cells": attributed, "all_single_pair_cells_decisively_excluded": all_excluded,
            "state": state, "decision": decision, "authorizes": authorizes,
            "r5_terminal_decision_changed": False,
        }
        atomic_json(context.phase_output_path / "r6_a0_contract_summary.json", contract_summary)
        atomic_json(context.phase_output_path / "r6_a0_provenance_and_access.json", provenance)
        atomic_json(context.phase_output_path / "r6_a0_input_identity.json", input_identity)
        atomic_json(context.phase_output_path / "r6_a0_base_replay_summary.json", base_replay)
        write_csv(context.phase_output_path / "r6_a0_factorial_cell_summary.csv", cell_rows)
        atomic_json(context.phase_output_path / "r6_a0_bootstrap_summary.json", {"schema_version": 1, **boot})
        write_csv(context.phase_output_path / "r6_a0_component_attribution.csv", attribution)
        write_csv(context.phase_output_path / "r6_a0_interaction_summary.csv", interaction_rows)
        write_csv(context.phase_output_path / "r6_a0_risk_veto_summary.csv", risk_rows)
        atomic_json(context.phase_output_path / "r6_a0_gate_summary.json", gate_summary)
        atomic_json(context.phase_output_path / "r6_a0_resource_summary.json", {
            "schema_version": 1, "wall_seconds": time.perf_counter() - started_wall,
            "cpu_seconds": time.process_time() - started_cpu,
            "max_rss_mib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0,
            "raw_rows_read": metadata["seed_rows"] + metadata["score_rows"] + metadata["policy_rows"],
            "bootstrap_draws": BOOTSTRAP_DRAWS, "gpu_used": False,
        })
        write_workload_progress(context, completed_units=4008, stage="finalized")
        write_run_result(
            context, state=state, decision=decision, authorizes=authorizes,
            details={
                "attributed_cells": attributed, "base_gain_db": boot["point"]["P000"]["gain"],
                "diagnostic_ceiling_gain_db": boot["point"]["P111"]["gain"],
                "base_selected_severe_groups": policy_counts(policies["P000"])["severe"],
                "base_selected_hard_groups": policy_counts(policies["P000"])["hard"],
                "r5_terminal_decision_changed": False,
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
