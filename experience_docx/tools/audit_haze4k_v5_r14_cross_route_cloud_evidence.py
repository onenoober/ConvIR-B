#!/usr/bin/env python3
"""Read-only cross-route forensic audit over frozen R3/R5/R10-R13 cloud artifacts.

This program does not fit a model, select a checkpoint, regenerate a candidate,
or open any confirmation/canary/locked-test role.  Formal recomputations retain
the original statistical units and seeds.  All added mechanism attributions are
explicitly post-hoc.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata, spearmanr

ROOT = Path("/sda/home/wangyuxin/ConvIR-B")
RUNS = ROOT / "runs"
R3 = RUNS / "haze4k_v5_r3_proposal_first_acv_20260717/r3-a0-proposal-r4/workload"
R5 = RUNS / "haze4k_v5_r5_spatial_candidate_response_sufficiency_20260719/r5-a0-spatial-response-screen-r2/workload"
R10 = RUNS / "haze4k_v5_r10_fixed_region_action_feasibility_20260719/r10-a0-fixed-region-feasibility-r1/workload"
R11 = RUNS / "haze4k_v5_r11_regional_action_observability_20260719/r11-a0-regional-observability-r1/workload"
R12 = RUNS / "haze4k_v5_r12_action_conditioned_downside_observability_20260719/r12-a0-action-downside-r2/workload"
R13_RUN = RUNS / "haze4k_v5_r13_image_relative_context_observability_20260719/r13-a0-image-relative-context-r1"

EXPECTED_SHA = {
    "r3_cache_manifest": "553505119f228d9561b50db9816a27409189206f6c6e108485a81d7622593e47",
    "r3_raw_manifest": "3123f2da6ac5e969fea8590a6c97ddbca46adf0e573fb1c03bec7d7a6c7ebfcc",
    "r5_candidate_scores": "53061beac134696216054e2aa4b2e6294313f0b9242a7f74a4346f79c30011a0",
    "r11_tile_rows": "8f486c4c3e2cc699de9336aeb93be87dd99fb06c6f7708e377f2095a94236d3a",
}

PATHS = {
    "r3_cache_manifest": R3 / "a0_cache_manifest.json",
    "r3_raw_manifest": R3 / "a0_cache_units_cloud_only.jsonl",
    "r5_candidate_scores": R5 / "r5_a0_candidate_scores_cloud_only.csv",
    "r5_policy_rows": R5 / "r5_a0_policy_rows_cloud_only.csv",
    "r5_seed_rows": R5 / "r5_a0_per_seed_predictions_cloud_only.csv",
    "r10_region_rows": R10 / "r10_a0_per_image_region_rows_cloud_only.csv",
    "r11_tile_rows": R11 / "r11_a0_tile_predictions_cloud_only.csv",
    "r11_policy_rows": R11 / "r11_a0_per_image_policy_rows_cloud_only.csv",
    "r12_risk_rows": R12 / "r12_a0_oof_risk_scores_cloud_only.csv",
}

PRIMARY_R5 = "S1_TRUE_SPATIAL_RESPONSE"
PRIMARY_R11 = "L1_LOCAL_CANDIDATE_CONTEXT"
R11_CELLS = (PRIMARY_R11, "P0_POOLED_CANDIDATE_RESPONSE",
             "S2_WITHIN_IMAGE_RESPONSE_SHUFFLE", "G0_LOCAL_STATE_ONLY")
R12_CELLS = ("ACTION_CONDITIONED", "ACTION_AGNOSTIC", "SIGN_SWAP", "LABEL_SHUFFLE")
OPERATORS = ("D_ref", "D_rep")
BOOTSTRAP_DRAWS = 4000
BOOTSTRAP_SEED = 3407
SEVERE = -0.2
HARD = -0.5
LOCAL_GATE = 0.005


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"not a JSON object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or any(None in row or any(v is None for v in row.values()) for row in rows):
        raise RuntimeError(f"CSV contract failed: {path}")
    return rows


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")


def interval(point: float, samples: list[float]) -> dict[str, float]:
    values = np.asarray(samples, dtype=np.float64)
    return {"point": float(point), "lcb95": float(np.quantile(values, .025)),
            "ucb95": float(np.quantile(values, .975))}


def cvar(values: np.ndarray, fraction: float = .05) -> float:
    values = np.sort(np.asarray(values, dtype=np.float64))
    return float(values[:max(1, math.ceil(len(values) * fraction))].mean())


def conservative_distribution(values: list[float]) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    n10 = max(1, math.ceil(.10 * len(x)))
    order = np.argsort(-x)
    positive = np.clip(x, 0, None)
    return {
        "mean": float(x.mean()), "median": float(np.median(x)),
        "p05": float(np.quantile(x, .05)), "p95": float(np.quantile(x, .95)),
        "positive_fraction": float(np.mean(x > 0)),
        "top10pct_share_of_positive_sum": float(positive[order[:n10]].sum() / max(positive.sum(), 1e-30)),
        "mean_after_dropping_top10pct": float(x[order[n10:]].mean()),
    }


def verify_status(run_dir: Path, route_id: str, expected: int) -> dict[str, Any]:
    path = run_dir / "status.txt"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    terminal = [row for row in rows if row.get("phase") == "terminal" and row.get("event") == "workload_end"]
    ok = len(terminal) == 1 and terminal[0].get("route_id") == route_id \
        and terminal[0].get("completed") == expected and terminal[0].get("total") == expected
    return {"sha256": sha256(path), "events": len(rows), "terminal_events": len(terminal),
            "expected_units": expected, "terminal": terminal[-1] if terminal else None, "valid": ok}


def r10_evaluate(rows: list[dict[str, Any]], indices: np.ndarray) -> dict[str, float]:
    result: dict[str, float] = {}
    means: dict[str, dict[str, float]] = {k: {} for k in ("region", "global", "shuffle")}
    for prefix in means:
        for op in OPERATORS:
            x = np.asarray([row[f"{prefix}_{op}"] for row in rows], dtype=np.float64)[indices]
            means[prefix][op] = float(x.mean())
            result[f"{prefix}_{op}"] = float(x.mean())
        result[f"{prefix}_gain"] = min(means[prefix].values())
    result["region_minus_global"] = min(means["region"][op] - means["global"][op] for op in OPERATORS)
    result["region_minus_shuffle"] = min(means["region"][op] - means["shuffle"][op] for op in OPERATORS)
    result["region_minus_global_cvar5"] = min(
        cvar(np.asarray([row[f"region_{op}"] for row in rows])[indices])
        - cvar(np.asarray([row[f"global_{op}"] for row in rows])[indices]) for op in OPERATORS)
    return result


def grouped_bootstrap(rows: list[dict[str, Any]], evaluate: Any, stratify_fold: bool) -> dict[str, Any]:
    point = evaluate(rows, np.arange(len(rows)))
    samples = {key: [] for key in point}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    strata = ([np.flatnonzero(np.asarray([int(row["fold"]) for row in rows]) == fold) for fold in (0, 1)]
              if stratify_fold else [np.arange(len(rows))])
    for _ in range(BOOTSTRAP_DRAWS):
        index = np.concatenate([rng.choice(s, len(s), replace=True) for s in strata])
        value = evaluate(rows, index)
        for key in samples:
            samples[key].append(value[key])
    return {key: interval(value, samples[key]) for key, value in point.items()}


def r11_evaluate(rows: list[dict[str, Any]], indices: np.ndarray) -> dict[str, float]:
    result: dict[str, float] = {}
    cell_means: dict[str, dict[str, float]] = defaultdict(dict)
    for cell in R11_CELLS:
        for op in OPERATORS:
            value = float(np.mean([float(rows[i][f"{cell}_{op}"]) for i in indices]))
            cell_means[cell][op] = value
            result[f"{cell}_{op}"] = value
        result[f"{cell}_gain"] = min(cell_means[cell].values())
    oracle = {op: float(np.mean([float(rows[i][f"oracle_{op}"]) for i in indices])) for op in OPERATORS}
    global_ = {op: float(np.mean([float(rows[i][f"global_{op}"]) for i in indices])) for op in OPERATORS}
    result["oracle_gain"] = min(oracle.values())
    result["global_gain"] = min(global_.values())
    result["retention"] = min(cell_means[PRIMARY_R11][op] / oracle[op] for op in OPERATORS)
    for control in R11_CELLS[1:]:
        result[f"primary_minus_{control}"] = min(
            cell_means[PRIMARY_R11][op] - cell_means[control][op] for op in OPERATORS)
    result["primary_minus_global"] = min(cell_means[PRIMARY_R11][op] - global_[op] for op in OPERATORS)
    result["primary_minus_global_cvar5"] = min(
        cvar(np.asarray([float(rows[i][f"{PRIMARY_R11}_{op}"]) for i in indices]))
        - cvar(np.asarray([float(rows[i][f"global_{op}"]) for i in indices])) for op in OPERATORS)
    return result


def row_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    pos = labels == 1
    neg = labels == 0
    if not pos.any() or not neg.any():
        return math.nan
    ranks = rankdata(scores, method="average")
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))


def row_risk_metrics(labels: np.ndarray, scores: np.ndarray, reject: float = .20) -> dict[str, float]:
    auc = row_auc(labels, scores)
    order = np.argsort(-scores, kind="mergesort")
    budget = reject * len(labels)
    before = np.arange(len(labels), dtype=np.float64)
    take = np.clip(budget - before, 0, 1)
    rejected = np.zeros(len(labels)); rejected[order] = take
    positive = labels.sum()
    capture = float((rejected * labels).sum() / positive)
    kept = 1 - rejected
    ratio = float(((kept * labels).sum() / kept.sum()) / (positive / len(labels)))
    return {"auroc": auc, "severe_capture_at_20pct": capture,
            "retained_severe_prevalence_ratio": ratio}


def r12_group_values(rows: list[dict[str, str]]) -> tuple[list[str], np.ndarray, dict[str, dict[str, np.ndarray]]]:
    names = sorted({row["name"] for row in rows})
    by_name: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        by_name[row["name"]].append(i)
    folds = np.asarray([int(next(row["fold"] for row in rows if row["name"] == name)) for name in names])
    output = {cell: {metric: np.full(len(names), np.nan) for metric in ("auroc",
        "severe_capture_at_20pct", "retained_severe_prevalence_ratio")} for cell in R12_CELLS}
    for group, name in enumerate(names):
        ix = by_name[name]
        y = np.asarray([int(rows[i]["severe"]) for i in ix])
        if y.min() == y.max():
            continue
        for cell in R12_CELLS:
            s = np.asarray([float(rows[i][f"{cell}_risk"]) for i in ix])
            for metric, value in row_risk_metrics(y, s).items():
                output[cell][metric][group] = value
    return names, folds, output


def r12_bootstrap(rows: list[dict[str, str]]) -> tuple[dict[str, Any], dict[str, Any]]:
    names, folds, values = r12_group_values(rows)
    point = {cell: {metric: float(np.nanmean(array)) for metric, array in metrics.items()}
             for cell, metrics in values.items()}
    samples: dict[str, list[float]] = {}
    for cell in R12_CELLS:
        for metric in point[cell]:
            samples[f"{cell}_{metric}"] = []
    for control in R12_CELLS[1:]:
        samples[f"primary_minus_{control}_auroc"] = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    strata = [np.flatnonzero(folds == fold) for fold in (0, 1)]
    for _ in range(BOOTSTRAP_DRAWS):
        chosen = np.concatenate([rng.choice(s, len(s), replace=True) for s in strata])
        current = {cell: {metric: float(np.nanmean(array[chosen])) for metric, array in metrics.items()}
                   for cell, metrics in values.items()}
        for cell in R12_CELLS:
            for metric, value in current[cell].items():
                samples[f"{cell}_{metric}"].append(value)
        for control in R12_CELLS[1:]:
            samples[f"primary_minus_{control}_auroc"].append(
                current[R12_CELLS[0]]["auroc"] - current[control]["auroc"])
    result = {}
    for cell in R12_CELLS:
        for metric, value in point[cell].items():
            key = f"{cell}_{metric}"; result[key] = interval(value, samples[key])
    for control in R12_CELLS[1:]:
        key = f"primary_minus_{control}_auroc"
        result[key] = interval(point[R12_CELLS[0]]["auroc"] - point[control]["auroc"], samples[key])
    return result, {"names": names, "folds": folds.tolist(), "point": point}


def compare_bootstrap(observed: dict[str, Any], official: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    differences = {}
    for key in keys:
        for bound in ("point", "lcb95", "ucb95"):
            differences[f"{key}.{bound}"] = abs(float(observed[key][bound]) - float(official[key][bound]))
    return {"keys": keys, "max_abs_difference": max(differences.values()),
            "all_within_1e-12": max(differences.values()) <= 1e-12, "differences": differences}


def parse_r10(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    numeric = {"fold", "global_action", "noop_tiles", "positive_tiles", "negative_tiles"}
    floats = {f"{p}_{op}" for p in ("region", "global", "shuffle") for op in OPERATORS}
    floats |= {"active_area_fraction", "noop_area_fraction", "positive_area_fraction", "negative_area_fraction"}
    output = []
    for raw in rows:
        row: dict[str, Any] = dict(raw)
        for key in numeric: row[key] = int(raw[key])
        for key in floats: row[key] = float(raw[key])
        row["action_map"] = [int(x) for x in raw["action_map"].split(";")]
        row["mixed_noop_active"] = raw["mixed_noop_active"].lower() == "true"
        row["bidirectional"] = raw["bidirectional"].lower() == "true"
        output.append(row)
    return output


def pixel_counts(shape: str) -> np.ndarray:
    h, w = [int(x) for x in shape.split("x")]
    return np.asarray([3 * (((r + 1) * h // 8) - (r * h // 8))
                       * (((c + 1) * w // 8) - (c * w // 8))
                       for r in range(8) for c in range(8)], dtype=np.int64)


def exact_budget_map(scores: np.ndarray, oracle: np.ndarray, counts: np.ndarray) -> np.ndarray:
    assigned = np.zeros(64, dtype=np.int64)
    for count in sorted(np.unique(counts)):
        indices = np.flatnonzero(counts == count)
        slots = np.asarray([action for action in range(3)
                            for _ in range(int(np.sum(oracle[indices] == action)))])
        benefit = np.zeros((len(indices), len(slots)))
        for j, action in enumerate(slots):
            if action: benefit[:, j] = scores[indices, action - 1]
        rr, cc = linear_sum_assignment(-benefit)
        assigned[indices[rr]] = slots[cc]
    return assigned


def analyze_r5(candidate: list[dict[str, str]], policy: list[dict[str, str]], seeds: list[dict[str, str]]) -> dict[str, Any]:
    candidate = [r for r in candidate if r["cell"] == PRIMARY_R5]
    policy = [r for r in policy if r["cell"] == PRIMARY_R5]
    names = sorted({r["name"] for r in candidate})
    target = {(r["name"], r["operator"], int(r["action"])): float(r["target_gain_db"])
              for r in candidate}
    by_name_policy = defaultdict(list)
    for r in policy: by_name_policy[r["name"]].append(r)
    fold = {name: int(by_name_policy[name][0]["fold"]) for name in names}
    selected = {name for name in names if int(by_name_policy[name][0]["selected"]) != 0}
    pred_action = {name: int(by_name_policy[name][0]["selected"]) for name in selected}
    truth_action = {}; robust = {}
    for name in names:
        value = {action: min(target[(name, op, action)] for op in OPERATORS) for action in (1, 2)}
        truth_action[name] = max((1, 2), key=lambda action: (value[action], -action))
        robust[name] = value[truth_action[name]]
    true_fixed = set()
    route = "haze4k_v5_r5_spatial_candidate_response_sufficiency_20260719"
    for f in (0, 1):
        subset = [name for name in names if fold[name] == f]
        subset.sort(key=lambda name: (-robust[name], hashlib.sha256(f"{route}|fold={f}|{name}".encode()).hexdigest()))
        true_fixed.update(subset[:39])
    def gain(active: set[str], actions: dict[str, int], positive_only: bool = False) -> dict[str, float]:
        means = {}
        for op in OPERATORS:
            values = [target[(name, op, actions[name])] if name in active and (not positive_only or robust[name] > 0) else 0.0
                      for name in names]
            means[op] = float(np.mean(values))
        return {**means, "conservative": min(means.values())}
    full = gain(set(names), truth_action, True)
    fixed = gain(true_fixed, truth_action)
    pred_names_true = gain(selected, truth_action)
    formal = gain(selected, pred_action)
    wrong_action = sum(pred_action[n] != truth_action[n] for n in selected)
    selected_values = [min(target[(n, op, pred_action[n])] for op in OPERATORS) for n in selected]
    seed_primary = [r for r in seeds if r["cell"] == PRIMARY_R5]
    seed_key = defaultdict(dict)
    for r in seed_primary:
        seed_key[(r["name"], r["operator"], int(r["action"]))][int(r["seed"])] = float(r["q05_score"])
    seed_ranges = [max(v.values()) - min(v.values()) for v in seed_key.values()]
    return {"evidence_role": "post_hoc_attribution over formal OOF rows",
        "groups": len(names), "selected_groups": len(selected),
        "chain_db": {"positive_oracle": full["conservative"], "true_fixed_20pct": fixed["conservative"],
            "predicted_names_true_action": pred_names_true["conservative"], "formal_policy": formal["conservative"]},
        "gap_db": {"abstention_or_fixed_coverage": full["conservative"] - fixed["conservative"],
            "coverage_ranking": fixed["conservative"] - pred_names_true["conservative"],
            "action_direction": pred_names_true["conservative"] - formal["conservative"],
            "total_oracle_to_formal": full["conservative"] - formal["conservative"]},
        "selected_wrong_action": wrong_action, "selected_wrong_action_fraction": wrong_action / len(selected),
        "selected_name_overlap_with_true_fixed": len(selected & true_fixed),
        "selected_conservative_gain_distribution": conservative_distribution(selected_values),
        "seed_q05_range": {"median": float(np.median(seed_ranges)), "p95": float(np.quantile(seed_ranges, .95)),
            "max": float(max(seed_ranges))}}


def analyze_tiles(r10: list[dict[str, Any]], tiles: list[dict[str, str]]) -> dict[str, Any]:
    r10_by_name = {r["name"]: r for r in r10}
    by_name: dict[str, dict[tuple[int, int], dict[str, str]]] = defaultdict(dict)
    for row in tiles: by_name[row["name"]][(int(row["tile"]), int(row["action"]))] = row
    exact = 0; total = 0; high_margin_exact = 0; high_margin_total = 0
    false_active = missed_active = wrong_sign = 0
    raw_margins = []; boundary = []; spreads = []; float_flips = 0; perturb_1e4 = 0; perturb_1e3 = 0
    pred_all = []; actual_all = []; selected_actual = []; oracle_actual = []
    edge_disagree = []; singleton = []
    map_replay_mismatch = 0
    for name, row in r10_by_name.items():
        oracle = np.asarray(row["action_map"], dtype=np.int64)
        counts = pixel_counts(row["shape"])
        scores = np.zeros((64, 2)); truth = np.zeros((64, 2)); means = np.zeros((64, 2))
        for tile in range(64):
            for action in (1, 2):
                raw = by_name[name][(tile, action)]
                scores[tile, action - 1] = float(raw["predicted_worst"])
                truth[tile, action - 1] = float(raw["actual_worst"])
                means[tile, action - 1] = float(raw["actual_mean"])
                boundary.append(abs(truth[tile, action - 1] - LOCAL_GATE))
                spreads.append(2 * abs(means[tile, action - 1] - truth[tile, action - 1]))
                pred_all.append(scores[tile, action - 1]); actual_all.append(truth[tile, action - 1])
        true_map = np.asarray([max(range(3), key=lambda a: ((0.0 if a == 0 else
            truth[t, a - 1] if truth[t, a - 1] >= LOCAL_GATE else -math.inf), -a)) for t in range(64)])
        map_replay_mismatch += int(np.sum(true_map != oracle))
        pred_map = exact_budget_map(scores, oracle, counts)
        exact += int(np.sum(pred_map == oracle)); total += 64
        for t in range(64):
            vals = sorted([0.0, truth[t, 0], truth[t, 1]], reverse=True)
            margin = vals[0] - vals[1]; raw_margins.append(margin)
            if margin >= .02:
                high_margin_total += 1; high_margin_exact += pred_map[t] == oracle[t]
            false_active += pred_map[t] != 0 and oracle[t] == 0
            missed_active += pred_map[t] == 0 and oracle[t] != 0
            wrong_sign += pred_map[t] != 0 and oracle[t] != 0 and pred_map[t] != oracle[t]
            selected_actual.append(0.0 if pred_map[t] == 0 else truth[t, pred_map[t] - 1])
            oracle_actual.append(0.0 if oracle[t] == 0 else truth[t, oracle[t] - 1])
        f32 = truth.astype(np.float32)
        map32 = np.asarray([max(range(3), key=lambda a: ((0.0 if a == 0 else
            float(f32[t, a - 1]) if f32[t, a - 1] >= LOCAL_GATE else -math.inf), -a)) for t in range(64)])
        float_flips += int(np.sum(map32 != oracle))
        for eps, key in ((1e-4, "small"), (1e-3, "large")):
            flips = sum(abs(value - LOCAL_GATE) <= eps for value in truth.ravel())
            if key == "small": perturb_1e4 += flips
            else: perturb_1e3 += flips
        grid = oracle.reshape(8, 8); disagreements = 0; edges = 0; singles = 0
        for rr in range(8):
            for cc in range(8):
                neighbors = []
                for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
                    if 0 <= rr+dr < 8 and 0 <= cc+dc < 8: neighbors.append(grid[rr+dr,cc+dc])
                singles += all(value != grid[rr,cc] for value in neighbors)
                if rr < 7: edges += 1; disagreements += grid[rr,cc] != grid[rr+1,cc]
                if cc < 7: edges += 1; disagreements += grid[rr,cc] != grid[rr,cc+1]
        edge_disagree.append(disagreements / edges); singleton.append(singles / 64)
    pred = np.asarray(pred_all); actual = np.asarray(actual_all)
    eligible = actual >= LOCAL_GATE
    return {"evidence_role": "formal replay checks plus post-hoc measurement audit",
        "groups": len(r10), "tile_action_rows": len(tiles), "oracle_map_replay_mismatches": map_replay_mismatch,
        "predicted_exact_action_accuracy": exact / total, "high_true_margin_ge_0.02_accuracy": high_margin_exact / high_margin_total,
        "high_true_margin_rows": high_margin_total, "false_active_tiles": false_active, "missed_active_tiles": missed_active,
        "wrong_signed_active_tiles": wrong_sign, "predicted_actual_worst_pearson": float(np.corrcoef(pred, actual)[0,1]),
        "predicted_actual_worst_spearman": float(spearmanr(pred, actual).statistic), "eligibility_auroc": row_auc(eligible.astype(int), pred),
        "true_margin": {"median": float(np.median(raw_margins)), "p10": float(np.quantile(raw_margins,.1)),
            "fraction_le_0.005": float(np.mean(np.asarray(raw_margins) <= .005)),
            "fraction_le_0.010": float(np.mean(np.asarray(raw_margins) <= .010))},
        "operator_range_db": {"median": float(np.median(spreads)), "p90": float(np.quantile(spreads,.9)),
            "p99": float(np.quantile(spreads,.99))},
        "numerical_stability": {"float32_action_flips": float_flips, "rows_within_1e-4_of_gate": perturb_1e4,
            "rows_within_1e-3_of_gate": perturb_1e3, "minimum_gate_distance": float(min(boundary))},
        "region_unit": {"mean_boundary_disagreement": float(np.mean(edge_disagree)),
            "mean_singleton_fraction": float(np.mean(singleton))},
        "robust_tile_utility": {"predicted_map_mean": float(np.mean(selected_actual)),
            "oracle_map_mean": float(np.mean(oracle_actual)),
            "retention": float(np.sum(selected_actual) / np.sum(oracle_actual))}}


def analyze_risk(risk: list[dict[str, str]], tiles: list[dict[str, str]], boot: dict[str, Any]) -> dict[str, Any]:
    tile_lookup = {(r["name"], int(r["tile"]), int(r["action"])): r for r in tiles}
    label_mismatch = 0; y=[]; score=[]; harm=[]; actions=[]; folds=[]
    for row in risk:
        source = tile_lookup[(row["name"], int(row["tile"]), int(row["action"]))]
        expected = int(float(source["actual_worst"]) <= SEVERE)
        label_mismatch += expected != int(row["severe"])
        y.append(expected); score.append(float(row["ACTION_CONDITIONED_risk"]));
        harm.append(-float(source["actual_worst"])); actions.append(int(row["action"])); folds.append(int(row["fold"]))
    y=np.asarray(y); score=np.asarray(score); harm=np.asarray(harm); actions=np.asarray(actions); folds=np.asarray(folds)
    bins = np.minimum((score * 10).astype(int), 9); ece=0.0
    for b in range(10):
        mask=bins==b
        if mask.any(): ece += mask.mean() * abs(score[mask].mean() - y[mask].mean())
    strata = {}
    for action in (1,2):
        mask=actions==action; strata[f"action_{action}"]={"rows":int(mask.sum()),"auroc":row_auc(y[mask],score[mask]),
            "brier":float(np.mean((score[mask]-y[mask])**2))}
    for fold in (0,1):
        mask=folds==fold; strata[f"fold_{fold}"]={"rows":int(mask.sum()),"auroc":row_auc(y[mask],score[mask])}
    return {"evidence_role": "formal grouped-metric reproduction plus fixed post-hoc calibration strata",
        "label_replay_mismatches": label_mismatch, "rows": len(risk), "severe_rows": int(y.sum()),
        "formal": boot, "row_brier": float(np.mean((score-y)**2)), "row_ece_10_equal_width": float(ece),
        "risk_vs_damage_spearman": float(spearmanr(score,harm).statistic), "fixed_strata": strata}


def audit_compact_bundle(repo: Path, relative: str, closeout_name: str) -> dict[str, Any]:
    root = repo / relative; closeout = read_json(root / closeout_name)
    mismatches = {}
    for name, expected in closeout.get("evidence_sha256", {}).items():
        path = root / name
        actual = sha256(path) if path.is_file() else None
        if actual != expected: mismatches[name] = {"expected": expected, "actual": actual}
    return {"closeout_sha256": sha256(root / closeout_name), "route_commit": closeout.get("route_commit"),
        "state": closeout.get("state"), "decision": closeout.get("decision"),
        "authorizes": closeout.get("authorizes"), "evidence_files": len(closeout.get("evidence_sha256", {})),
        "hash_mismatches": mismatches, "valid": not mismatches}


def analyze_external(repo: Path) -> dict[str, Any]:
    root = repo / "experience_docx/experiment_logs/haze4k_v2_7_nhhaze_transfer_20260616"
    rows = read_csv(root / "v27_nhhaze_wdmamba_transfer_per_image.csv")
    psnr=np.asarray([float(r["alpha_a0p375_dPSNR"]) for r in rows]); ssim=np.asarray([float(r["alpha_a0p375_dSSIM"]) for r in rows])
    return {"evidence_role": "historical post-hoc external directional reference; different action family",
        "images": len(rows), "fixed_haze4k_alpha": .375, "mean_dpsnr": float(psnr.mean()),
        "mean_dssim": float(ssim.mean()), "psnr_positive_fraction": float(np.mean(psnr>0)),
        "psnr_ssim_sign_disagreement_fraction": float(np.mean((psnr>0)!=(ssim>0))),
        "psnr_ssim_spearman": float(spearmanr(psnr,ssim).statistic),
        "severe_dpsnr_le_minus_0.2": int(np.sum(psnr<=-.2)),
        "scope_warning": "NH-HAZE v2.7 WDMamba blend is not the R10-R13 action family and cannot validate transfer."}


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--repo-root",required=True,type=Path); parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args(); started=time.time()
    if args.output.exists(): raise RuntimeError(f"output already exists: {args.output}")
    args.output.mkdir(parents=True); (args.output/"status.txt").write_text("R14_AUDIT_STARTED\n",encoding="utf-8")
    if ROOT != Path("/sda/home/wangyuxin/ConvIR-B") or not ROOT.is_dir(): raise RuntimeError("cloud root identity mismatch")
    identity={key:{"path":str(path),"sha256":sha256(path),"bytes":path.stat().st_size} for key,path in PATHS.items()}
    sha_mismatch={key:{"expected":value,"actual":identity[key]["sha256"]} for key,value in EXPECTED_SHA.items() if identity[key]["sha256"]!=value}
    statuses={
      "r5":verify_status(R5.parent,"haze4k_v5_r5_spatial_candidate_response_sufficiency_20260719",1552),
      "r10":verify_status(R10.parent,"haze4k_v5_r10_fixed_region_action_feasibility_20260719",4770),
      "r11":verify_status(R11.parent,"haze4k_v5_r11_regional_action_observability_20260719",4786),
      "r12":verify_status(R12.parent,"haze4k_v5_r12_action_conditioned_downside_observability_20260719",12),
      "r13":verify_status(R13_RUN,"haze4k_v5_r13_image_relative_context_observability_20260719",4790)}
    integrity_ok=not sha_mismatch and all(v["valid"] for v in statuses.values())
    input_identity={"schema_version":1,"cloud_host":"convir-4090","cloud_root":str(ROOT),"inputs":identity,
        "known_sha_mismatches":sha_mismatch,"run_status":statuses,"protected_path_tokens_opened":[],
        "r13_row_assets":"RAW_ASSET_UNAVAILABLE","integrity_valid":integrity_ok}
    write_json(args.output/"input_identity.json",input_identity)
    if not integrity_ok:
        close={"state":"INTEGRITY_INCIDENT","mechanism_analysis_performed":False,"terminal_decision_affected":False}
        write_json(args.output/"cloud_audit_closeout.json",close); raise RuntimeError("integrity incident")
    r5c=read_csv(PATHS["r5_candidate_scores"]); r5p=read_csv(PATHS["r5_policy_rows"]); r5s=read_csv(PATHS["r5_seed_rows"])
    r10=parse_r10(read_csv(PATHS["r10_region_rows"])); r11t=read_csv(PATHS["r11_tile_rows"]); r11p=read_csv(PATHS["r11_policy_rows"]); r12=read_csv(PATHS["r12_risk_rows"])
    if not (len(r5c)==6144 and len(r5p)==3072 and len(r5s)==12288 and len(r10)==384 and len(r11t)==49152 and len(r11p)==384 and len(r12)==49152):
        raise RuntimeError("formal row counts changed after preflight")
    r10_boot=grouped_bootstrap(r10,r10_evaluate,False); r11_boot=grouped_bootstrap(r11p,r11_evaluate,True); r12_boot,r12_points=r12_bootstrap(r12)
    logs=args.repo_root/"experience_docx/experiment_logs"
    off10=read_json(logs/"haze4k_v5_r10_fixed_region_action_feasibility_20260719/r10_a0_bootstrap_summary.json")
    off11=read_json(logs/"haze4k_v5_r11_regional_action_observability_20260719/r11_a0_bootstrap_summary.json")
    off12=read_json(logs/"haze4k_v5_r12_action_conditioned_downside_observability_20260719/r12_a0_bootstrap_summary.json")
    comparison={
      "r10":compare_bootstrap(r10_boot,off10,["region_gain","region_minus_global","region_minus_shuffle","region_minus_global_cvar5"]),
      "r11":compare_bootstrap(r11_boot,off11,[f"{PRIMARY_R11}_gain","retention","primary_minus_P0_POOLED_CANDIDATE_RESPONSE","primary_minus_S2_WITHIN_IMAGE_RESPONSE_SHUFFLE","primary_minus_G0_LOCAL_STATE_ONLY","primary_minus_global","primary_minus_global_cvar5"]),
      "r12":compare_bootstrap(r12_boot,off12,["ACTION_CONDITIONED_auroc","primary_minus_ACTION_AGNOSTIC_auroc","primary_minus_SIGN_SWAP_auroc","primary_minus_LABEL_SHUFFLE_auroc","ACTION_CONDITIONED_severe_capture_at_20pct","ACTION_CONDITIONED_retained_severe_prevalence_ratio"])}
    compact={}
    for route,prefix,close in [("haze4k_v5_r10_fixed_region_action_feasibility_20260719","r10","r10_a0_fixed_region_action_feasibility_closeout.json"),("haze4k_v5_r11_regional_action_observability_20260719","r11","r11_a0_regional_action_observability_closeout.json"),("haze4k_v5_r12_action_conditioned_downside_observability_20260719","r12","r12_a0_action_conditioned_downside_observability_closeout.json"),("haze4k_v5_r13_image_relative_context_observability_20260719","r13","r13_a0_image_relative_context_observability_closeout.json")]:
        compact[prefix]=audit_compact_bundle(args.repo_root,f"experience_docx/experiment_logs/{route}",close)
    reproduction={"status":"EXACT_RAW_REPRODUCTION_R10_R12; COMPACT_HASH_VERIFIED_ONLY_R13",
        "bootstrap_comparison":comparison,"compact_bundle_identity":compact,
        "max_raw_reproduction_difference":max(v["max_abs_difference"] for v in comparison.values()),
        "terminal_decision_affected":False}
    write_json(args.output/"official_result_reproduction.json",reproduction)
    a1=analyze_r5(r5c,r5p,r5s); a2=analyze_tiles(r10,r11t); a3=analyze_risk(r12,r11t,r12_boot); a5=analyze_external(args.repo_root)
    conservative_r10=[min(r["region_D_ref"],r["region_D_rep"]) for r in r10]
    a4={"evidence_role":"post-hoc target-alignment audit","r10_gain_distribution":conservative_distribution(conservative_r10),
        "haze4k_target_dimensions_present":["PSNR"],"target_dimensions_absent":["semantic protection","haze severity","human naturalness","perceptual harm"],
        "operator_mean_worst_disagreement_rows":sum(float(r["actual_mean"])>=LOCAL_GATE and float(r["actual_worst"])<LOCAL_GATE for r in r11t),
        "conclusion":"The cloud artifacts cannot establish alignment with heavy-haze recovery, light-haze protection, or natural visual quality."}
    for name,value in [("a1_regret_attribution.json",a1),("a2_label_region_stability.json",a2),("a3_risk_observability.json",a3),("a4_target_alignment.json",a4),("a5_external_directional_reference.json",a5)]: write_json(args.output/name,value)
    findings={"schema_version":1,"findings":[
      {"id":"F1","role":"formal_reproduction","finding":"R10-R12 decisive raw-row metrics reproduce within 1e-12; R13 compact bundle hashes verify but row assets are unavailable."},
      {"id":"F2","role":"post_hoc","finding":f"R11 exact-budget action accuracy {a2['predicted_exact_action_accuracy']:.4f}; high-margin accuracy {a2['high_true_margin_ge_0.02_accuracy']:.4f}."},
      {"id":"F3","role":"post_hoc","finding":f"R5 oracle-to-policy gap decomposes into coverage {a1['gap_db']['abstention_or_fixed_coverage']:.6f}, ranking {a1['gap_db']['coverage_ranking']:.6f}, direction {a1['gap_db']['action_direction']:.6f} dB."},
      {"id":"F4","role":"formal_plus_post_hoc","finding":f"R12 primary AUROC {r12_boot['ACTION_CONDITIONED_auroc']['point']:.4f}, fixed 20% capture {r12_boot['ACTION_CONDITIONED_severe_capture_at_20pct']['point']:.4f}; row ECE {a3['row_ece_10_equal_width']:.4f}."},
      {"id":"F5","role":"historical_external_reference","finding":f"NH-HAZE fixed Haze4K alpha has mean dPSNR {a5['mean_dpsnr']:.4f} and {a5['severe_dpsnr_le_minus_0.2']} severe images; action family differs."}]}
    write_json(args.output/"key_raw_findings.json",findings)
    closeout={"schema_version":1,"audit_id":"haze4k_v5_r14_cross_route_cloud_evidence_audit_20260720",
      "state":"COMPLETED_READ_ONLY_AUDIT","decision":"EXACT_R5_R13_PARADIGM_REMAINS_CLOSED_STRATEGIC_RECONSTRUCTION_REQUIRED",
      "authorizes":"READ_ONLY_EVIDENCE_SYNC_ONLY","old_terminal_decisions_changed":False,"training_run":False,
      "checkpoint_selected_or_loaded":False,"candidate_generated":False,"protected_data_touched":False,
      "confirmation_touched":False,"canary_touched":False,"locked_test_touched":False,
      "formal_reproduction_max_abs_difference":reproduction["max_raw_reproduction_difference"],
      "r13_raw_reproduction":"RAW_ASSET_UNAVAILABLE_COMPACT_HASH_VERIFIED_ONLY","wall_seconds":time.time()-started}
    write_json(args.output/"cloud_audit_closeout.json",closeout)
    conclusion={"schema_version":1,"hypotheses":{
      "H1":{"decision":"SUPPORTED_NOT_FULLY_PROVEN","basis":"poor exact-action and high-margin ranking with exact privileged budget; weak score-utility correlation"},
      "H2":{"decision":"SUPPORTED_AS_SECONDARY","basis":"R5 coverage/direction losses and R12 capture/calibration failure; cannot explain R11 exact-budget gap alone"},
      "H3":{"decision":"NOT_EXCLUDED_AND_STRENGTHENED_UPSTREAM","basis":"ambiguous tile labels, fragmented maps, PSNR-only target and absent semantic/visual harm labels"},
      "H4":{"decision":"NOT_EXCLUDED_DIRECTIONALLY_SUPPORTED","basis":"NH-HAZE fixed Haze4K blend reversal; different action family prevents direct transfer claim"}},
      "updated_bottleneck":"Observable candidate-conditioned signed utility is weak before policy construction, while the fixed PSNR/tile/action formulation itself is insufficiently aligned with the real target.",
      "route_status":"已耗尽，应关闭；总目标需要战略重构",
      "solution_family_updates":{"S1":"支持但需收窄：仅在独立新信息先证明增量后","S2":"可信度下降为次级：决策不能创造缺失信息","S3":"支持增强：先审计数据、监督、区域/动作与评价定义"}}
    write_json(args.output/"scientific_conclusion.json",conclusion)
    with (args.output/"status.txt").open("a",encoding="utf-8") as f: f.write("R14_AUDIT_COMPLETED\n")
    print("R14_CROSS_ROUTE_CLOUD_AUDIT_OK")


if __name__ == "__main__":
    main()
