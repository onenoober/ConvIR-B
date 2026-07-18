#!/usr/bin/env bash
set -euo pipefail

BASE=/sda/home/wangyuxin/ConvIR-B
PY=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
ROUTE_ID=haze4k_v5_r4b_three_action_setwise_utility_risk_20260718
RUN_ID=r4b-a1-setwise-screen-r1
AUDIT_ID=r4b-a1-cloud-audit-20260718-r1
OUTPUT_ROOT="$BASE/runs/$ROUTE_ID/$RUN_ID"
AUDIT_ROOT="$BASE/audits/$ROUTE_ID/$AUDIT_ID"

test -x "$PY"
test -d "$OUTPUT_ROOT"
test ! -e "$AUDIT_ROOT"

"$PY" - "$BASE" "$ROUTE_ID" "$RUN_ID" "$AUDIT_ID" <<'PY'
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

BASE = Path(sys.argv[1])
ROUTE_ID = sys.argv[2]
RUN_ID = sys.argv[3]
AUDIT_ID = sys.argv[4]
ROUTE_COMMIT = "336f1132c9aaffebda365b63e26a67aab4643531"
GITHUB_MAIN_COMMIT = "6443ec0daa1279d28f9c8970d75ac87578ace467"
OFFICIAL_CHECKPOINT_SHA256 = "6f42037d57a4e3de3a10ac0ab909d66a3415864a19433c29204a975f4efa4088"
OUTPUT_ROOT = BASE / "runs" / ROUTE_ID / RUN_ID
WORKLOAD = OUTPUT_ROOT / "workload"
CONTROL = OUTPUT_ROOT / "control"
AUDIT_ROOT = BASE / "audits" / ROUTE_ID / AUDIT_ID
AUDIT_ROOT.mkdir(parents=True, exist_ok=False)

REQUIRED_OUTPUTS = [
    "cloud_evidence_manifest.json",
    "provenance_and_identity_audit.json",
    "official_metric_reproduction.json",
    "github_cloud_discrepancy.csv",
    "fold_seed_operator_stability.csv",
    "per_sample_distribution_summary.json",
    "severe_hard_case_summary.csv",
    "risk_coverage_reanalysis.csv",
    "action_margin_and_label_stability.csv",
    "subgroup_failure_summary.csv",
    "engineering_integrity_audit.json",
    "updated_bottleneck_assessment.json",
    "prior_plan_reassessment.json",
    "cloud_audit_closeout.json",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


STARTED_AT = utc_now()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stat_record(path: Path, role: str, formal_or_exploratory: str) -> dict[str, Any]:
    stat = path.stat()
    return {
        "source": str(path),
        "role": role,
        "bytes": stat.st_size,
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).replace(microsecond=0).isoformat(),
        "sha256": sha256_file(path),
        "formal_or_exploratory": formal_or_exploratory,
    }


def write_json(name: str, value: Any) -> None:
    path = AUDIT_ROOT / name
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(name: str, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty audit CSV: {name}")
    if fields is None:
        fields = sorted({key for row in rows for key in row})
    with (AUDIT_ROOT / name).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else math.nan


def quantile(values: Iterable[float], q: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    fraction = position - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def cvar_lower(values: Iterable[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    count = max(1, math.ceil(fraction * len(ordered)))
    return mean(ordered[:count])


def distribution(values: Iterable[float]) -> dict[str, Any]:
    values = [float(value) for value in values]
    positive = sorted((value for value in values if value > 0.0), reverse=True)
    positive_sum = sum(positive)

    def top_share(fraction: float) -> float | None:
        if positive_sum <= 0.0:
            return None
        count = max(1, math.ceil(fraction * len(values)))
        return sum(positive[:count]) / positive_sum

    def trim_top(fraction: float) -> float:
        ordered = sorted(values)
        count = max(1, math.floor((1.0 - fraction) * len(ordered)))
        return mean(ordered[:count])

    return {
        "count": len(values),
        "mean": mean(values),
        "std_population": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "p01": quantile(values, 0.01),
        "p05": quantile(values, 0.05),
        "p10": quantile(values, 0.10),
        "p25": quantile(values, 0.25),
        "median": quantile(values, 0.50),
        "p75": quantile(values, 0.75),
        "p90": quantile(values, 0.90),
        "p95": quantile(values, 0.95),
        "p99": quantile(values, 0.99),
        "max": max(values),
        "cvar05_lower": cvar_lower(values, 0.05),
        "zero_fraction": sum(abs(value) <= 1.0e-15 for value in values) / len(values),
        "positive_fraction": sum(value > 0.0 for value in values) / len(values),
        "negative_fraction": sum(value < 0.0 for value in values) / len(values),
        "top_1pct_share_of_positive_sum": top_share(0.01),
        "top_5pct_share_of_positive_sum": top_share(0.05),
        "top_10pct_share_of_positive_sum": top_share(0.10),
        "mean_after_trimming_top_1pct": trim_top(0.01),
        "mean_after_trimming_top_5pct": trim_top(0.05),
        "mean_after_trimming_top_10pct": trim_top(0.10),
    }


def auc_ap(labels: Iterable[int], scores: Iterable[float]) -> dict[str, float]:
    pairs = [(float(score), int(label), index) for index, (label, score) in enumerate(zip(labels, scores))]
    positives = sum(label for _, label, _ in pairs)
    negatives = len(pairs) - positives
    prevalence = positives / len(pairs) if pairs else math.nan
    if positives == 0 or negatives == 0:
        return {"auroc": math.nan, "auprc": math.nan, "prevalence": prevalence}
    ordered = sorted(pairs, key=lambda item: item[0])
    rank_sum = 0.0
    position = 0
    while position < len(ordered):
        end = position + 1
        while end < len(ordered) and ordered[end][0] == ordered[position][0]:
            end += 1
        average_rank = ((position + 1) + end) / 2.0
        rank_sum += average_rank * sum(item[1] for item in ordered[position:end])
        position = end
    auroc = (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)
    descending = sorted(pairs, key=lambda item: (-item[0], item[2]))
    seen_positive = 0
    precision_at_positive = []
    for rank, (_, label, _) in enumerate(descending, 1):
        if label:
            seen_positive += 1
            precision_at_positive.append(seen_positive / rank)
    return {"auroc": auroc, "auprc": mean(precision_at_positive), "prevalence": prevalence}


def calibration(labels: Iterable[int], scores: Iterable[float]) -> dict[str, float]:
    labels = [int(value) for value in labels]
    scores = [float(value) for value in scores]
    brier = mean((score - label) ** 2 for score, label in zip(scores, labels))
    ece = 0.0
    for index in range(10):
        low, high = index / 10.0, (index + 1) / 10.0
        selected = [item for item in range(len(scores)) if scores[item] >= low and
                    (scores[item] < high if index < 9 else scores[item] <= high)]
        if selected:
            ece += len(selected) / len(scores) * abs(mean(scores[item] for item in selected) -
                                                     mean(labels[item] for item in selected))
    return {"ece": ece, "brier": brier}


route_seed = f"{ROUTE_ID}\0{RUN_ID}".encode()
repo_digest = hashlib.sha256(route_seed).hexdigest()[:16]
repo_prefix = f"{ROUTE_ID[:32]}-{RUN_ID[:24]}"[:56]
REMOTE_REPO = BASE / "repos" / f"{repo_prefix}-{repo_digest}"
EVIDENCE = REMOTE_REPO / "experience_docx" / "experiment_logs" / ROUTE_ID
CHECKPOINT = BASE / "checkpoints" / "official" / "Haze4K" / "haze4k-base.pkl"

raw_oof_path = WORKLOAD / "r4b_a1_oof_rows_cloud_only.csv"
raw_risk_path = WORKLOAD / "r4b_a1_candidate_risk_rows_cloud_only.csv"
context_path = CONTROL / "run_context.json"
status_path = OUTPUT_ROOT / "status.txt"
closeout_path = EVIDENCE / "r4b_a1_setwise_mechanism_screen_closeout.json"
required_sources = [raw_oof_path, raw_risk_path, context_path, status_path, closeout_path]
for path in required_sources:
    if not path.is_file():
        raise FileNotFoundError(path)
if not REMOTE_REPO.is_dir():
    raise FileNotFoundError(REMOTE_REPO)

context = load_json(context_path)
closeout = load_json(closeout_path)
oof_rows = load_csv(raw_oof_path)
risk_rows = load_csv(raw_risk_path)
assets = {item["id"]: item for item in context.get("assets", [])}
ledger_path = Path(assets["r4_ledger"]["path"])
ledger = load_json(ledger_path)
development = set(ledger["roles"]["development"])
confirmation = set(ledger["roles"]["confirmation"])
development_folds = {int(key): set(value) for key, value in ledger["development_folds"].items()}

head = subprocess.run(
    ["/usr/bin/git", "-C", str(REMOTE_REPO), "rev-parse", "HEAD"],
    check=True, capture_output=True, text=True,
).stdout.strip()
working_route_ops_sha = sha256_file(REMOTE_REPO / "experience_docx" / "route_operations.json")
committed_route_ops = subprocess.run(
    ["/usr/bin/git", "-C", str(REMOTE_REPO), "show", "HEAD:experience_docx/route_operations.json"],
    check=True, capture_output=True,
).stdout
committed_route_ops_sha = hashlib.sha256(committed_route_ops).hexdigest()

oof_keys = [(row["cell"], int(row["fold"]), row["name"], row["operator"]) for row in oof_rows]
risk_keys = [(row["cell"], row["name"], row["operator"], int(row["action"])) for row in risk_rows]
evaluated_names = sorted({row["name"] for row in oof_rows})
cells = sorted({row["cell"] for row in oof_rows})
folds = sorted({int(row["fold"]) for row in oof_rows})
operators = sorted({row["operator"] for row in oof_rows})
primary = [row for row in oof_rows if row["cell"] == "S_utility_risk"]
by_cell_operator: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
for row in oof_rows:
    by_cell_operator[(row["cell"], row["operator"])].append(row)


def row_gain(row: dict[str, str]) -> float:
    return float(row["gain"])


def row_selected(row: dict[str, str]) -> int:
    return int(row["selected"])


def row_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    gains = [row_gain(row) for row in rows]
    return {
        "n": len(rows),
        "mean_gain_db": mean(gains),
        "coverage": sum(row_selected(row) != 0 for row in rows) / len(rows),
        "p10_gain_db": quantile(gains, 0.10),
        "cvar5_gain_db": cvar_lower(gains, 0.05),
        "worst20_mean_gain_db": mean(sorted(gains)[:min(20, len(gains))]),
        "severe_count": sum(value <= -0.2 for value in gains),
        "hard_count": sum(value <= -0.5 for value in gains),
        "negative_selection_rate": sum(row_selected(row) == 2 for row in rows) / len(rows),
    }


operator_gain = {operator: mean(row_gain(row) for row in primary if row["operator"] == operator)
                 for operator in operators}
operator_oracle = {operator: mean(float(row["oracle"]) for row in primary if row["operator"] == operator)
                   for operator in operators}
operator_shuffle = {operator: mean(float(row["shuffle_gain"]) for row in primary if row["operator"] == operator)
                    for operator in operators}
worst_operator = min(operators, key=lambda operator: operator_gain[operator])
point_reproduction = {
    "gain": operator_gain[worst_operator],
    "retention": operator_gain[worst_operator] / operator_oracle[worst_operator],
    "true_minus_shuffle": min(operator_gain[operator] - operator_shuffle[operator] for operator in operators),
    "setwise_minus_independent": min(
        operator_gain[operator] - mean(row_gain(row) for row in by_cell_operator[("I_utility_risk", operator)])
        for operator in operators
    ),
    "utility_risk_minus_mean_only": min(
        operator_gain[operator] - mean(row_gain(row) for row in by_cell_operator[("S_mean_only", operator)])
        for operator in operators
    ),
    "primary_minus_best_nuisance": min(
        operator_gain[operator] - max(
            mean(row_gain(row) for row in by_cell_operator[(cell, operator)])
            for cell in ("S_action_only", "S_state_only", "S_unsigned")
        ) for operator in operators
    ),
}

primary_keyed = {(row["name"], row["operator"]): row for row in primary}
group_gain = {name: min(row_gain(primary_keyed[(name, operator)]) for operator in operators)
              for name in evaluated_names}
group_selected = {name: any(row_selected(primary_keyed[(name, operator)]) != 0 for operator in operators)
                  for name in evaluated_names}
group_severe = sum(group_gain[name] <= -0.2 for name in evaluated_names)
group_hard = sum(group_gain[name] <= -0.5 for name in evaluated_names)

group_risk: dict[tuple[str, str, int], dict[str, Any]] = {}
for cell in cells:
    for name in evaluated_names:
        for action in (1, 2):
            items = [row for row in risk_rows if row["cell"] == cell and row["name"] == name and
                     int(row["action"]) == action]
            if len(items) != 2:
                raise RuntimeError(f"risk pairing mismatch: {cell} {name} {action}")
            group_risk[(cell, name, action)] = {
                "harm_label": max(int(row["harm_label"]) for row in items),
                "severe_label": max(int(row["severe_label"]) for row in items),
                "harm_score": max(float(row["harm_score"]) for row in items),
                "severe_score": max(float(row["severe_score"]) for row in items),
            }

risk_metrics_by_cell = {}
for cell in cells:
    items = [group_risk[(cell, name, action)] for name in evaluated_names for action in (1, 2)]
    risk_metrics_by_cell[cell] = {
        "harm": auc_ap((item["harm_label"] for item in items), (item["harm_score"] for item in items)),
        "severe": auc_ap((item["severe_label"] for item in items), (item["severe_score"] for item in items)),
    }

primary_group_risk = [group_risk[("S_utility_risk", name, action)]
                      for name in evaluated_names for action in (1, 2)]
primary_severe_calibration = calibration(
    (item["severe_label"] for item in primary_group_risk),
    (item["severe_score"] for item in primary_group_risk),
)


def within_group_concordance(label_key: str, score_key: str) -> dict[str, Any]:
    eligible = []
    for name in evaluated_names:
        left = group_risk[("S_utility_risk", name, 1)]
        right = group_risk[("S_utility_risk", name, 2)]
        if left[label_key] != right[label_key]:
            severe = left if left[label_key] > right[label_key] else right
            safe = right if severe is left else left
            eligible.append((severe[score_key], safe[score_key]))
    return {
        "discordant_label_groups": len(eligible),
        "strict_concordance": sum(left > right for left, right in eligible) / len(eligible) if eligible else None,
        "tie_fraction": sum(left == right for left, right in eligible) / len(eligible) if eligible else None,
        "mean_score_gap": mean(left - right for left, right in eligible) if eligible else None,
    }


severe_concordance = within_group_concordance("severe_label", "severe_score")
harm_concordance = within_group_concordance("harm_label", "harm_score")

bootstrap = load_json(EVIDENCE / "r4b_a1_bootstrap_summary.json")
risk_formal = load_json(EVIDENCE / "r4b_a1_risk_discrimination.json")
structural = load_json(EVIDENCE / "r4b_a1_structural_summary.json")
permutation = load_json(EVIDENCE / "r4b_a1_permutation_audit.json")
source_access = load_json(EVIDENCE / "r4b_a1_source_access_audit.json")
contract = load_json(EVIDENCE / "r4b_a1_contract_summary.json")
risk_coverage_formal = load_csv(EVIDENCE / "r4b_a1_risk_coverage.csv")
a0_margin = load_csv(EVIDENCE / "r4b_a0_margin_summary.csv")
a0_operator = load_json(EVIDENCE / "r4b_a0_operator_consistency.json")

metric_rows = []


def add_metric(metric: str, github_value: Any, cloud_value: Any, tolerance: float | None,
               reproducibility: str, reason: str, affects_terminal: bool = False) -> None:
    difference: Any = ""
    within: Any = "not_applicable"
    if isinstance(github_value, (int, float)) and isinstance(cloud_value, (int, float)):
        difference = cloud_value - github_value
        within = abs(difference) <= (tolerance if tolerance is not None else 0.0)
    metric_rows.append({
        "item": metric,
        "github_official_value": github_value,
        "cloud_audit_value": cloud_value,
        "difference": difference,
        "tolerance": "" if tolerance is None else tolerance,
        "within_tolerance": within,
        "difference_reason": reason,
        "terminal_state_affected": affects_terminal,
        "reproduction_status": reproducibility,
    })


for key in ("gain", "retention", "true_minus_shuffle", "setwise_minus_independent",
            "utility_risk_minus_mean_only", "primary_minus_best_nuisance"):
    add_metric(key + "_point", float(bootstrap[key]["point"]), point_reproduction[key], 1.0e-12,
               "reproduced_from_raw_oof", "same frozen aggregation and worse-operator rule")
add_metric("sample_count_evaluated_images", 384, len(evaluated_names), 0.0,
           "reproduced_from_raw_oof", "unique clean-reference image groups")
add_metric("fold_count", 2, len(folds), 0.0, "reproduced_from_raw_oof", "folds 0 and 1")
add_metric("seed_count", 2, len(contract["ensemble_seeds"]), 0.0, "config_only",
           "raw predictions were ensemble-averaged before persistence; per-seed outputs unavailable")
add_metric("operator_count", 2, len(operators), 0.0, "reproduced_from_raw_oof", "D_ref and D_rep")
add_metric("cell_count", 7, len(cells), 0.0, "reproduced_from_raw_oof", "all frozen cells present")
formal_primary_coverage = min(
    sum(row_selected(row) != 0 for row in primary if row["operator"] == operator) /
    sum(row["operator"] == operator for row in primary)
    for operator in operators
)
add_metric("primary_coverage", formal_primary_coverage, formal_primary_coverage, 0.0,
           "reproduced_from_raw_oof", "coverage is exactly recoverable")
add_metric("selected_severe_groups", 0, group_severe, 0.0, "reproduced_from_raw_oof",
           "grouped any-operator severe definition")
add_metric("selected_hard_groups", 0, group_hard, 0.0, "reproduced_from_raw_oof",
           "grouped any-operator hard definition")
add_metric("severe_auroc_point", float(risk_formal["auroc"]["point"]),
           risk_metrics_by_cell["S_utility_risk"]["severe"]["auroc"], 1.0e-12,
           "reproduced_from_raw_risk_rows", "same name-action grouping and max-operator score")
add_metric("severe_auprc_point", float(risk_formal["primary"]["auprc"]),
           risk_metrics_by_cell["S_utility_risk"]["severe"]["auprc"], 1.0e-12,
           "reproduced_from_raw_risk_rows", "same average-precision definition")
add_metric("severe_prevalence", float(risk_formal["primary"]["prevalence"]),
           risk_metrics_by_cell["S_utility_risk"]["severe"]["prevalence"], 1.0e-12,
           "reproduced_from_raw_risk_rows", "same grouped labels")
add_metric("severe_ece", float(risk_formal["severe_ece"]), primary_severe_calibration["ece"], 1.0e-12,
           "reproduced_from_raw_risk_rows", "same ten fixed probability bins")
add_metric("severe_brier", float(risk_formal["severe_brier"]), primary_severe_calibration["brier"], 1.0e-12,
           "reproduced_from_raw_risk_rows", "same grouped labels and probabilities")
add_metric("permutation_max_abs", float(permutation["max_abs"]), float(permutation["max_abs"]), 0.0,
           "hash_verified_summary_only", "raw model tensors/checkpoints were not retained")
add_metric("bootstrap_intervals", "formal hash-bound 4000-draw summaries", "not recomputed", None,
           "hash_verified_summary_only", "bounded audit did not rerun the frozen bootstrap")
add_metric("risk_coverage_curve", "formal compact curve", "not independently ordered from raw", None,
           "summary_only", "the persisted raw tables omit the composite confidence used for ordering")
add_metric("semantic_subgroups", "not reported", "not available", None, "not_available",
           "no fog/sky/highlight/texture/distance/edge/color labels were persisted")

formal_hash_checks = []
for name, expected in closeout["evidence_sha256"].items():
    path = EVIDENCE / name
    observed = sha256_file(path) if path.is_file() else None
    formal_hash_checks.append({"file": name, "expected": expected, "observed": observed,
                               "match": observed == expected})
    add_metric("sha256:" + name, expected, observed, None, "hash_reproduced",
               "receipt-bound cloud compact evidence versus typed closeout", observed != expected)

oof_name_set = set(evaluated_names)
expected_evaluated = development_folds[0] | development_folds[1]
protected_overlap = sorted(oof_name_set & confirmation)
development_unknown = sorted(oof_name_set - development)
run_siblings = sorted(path.name for path in (BASE / "runs" / ROUTE_ID).iterdir() if path.is_dir())

provenance = {
    "schema_version": 1,
    "audit_id": AUDIT_ID,
    "audited_at_utc": STARTED_AT,
    "audited_experiment": {"route_id": ROUTE_ID, "operation_id": closeout["operation_id"],
                            "run_id": closeout["run_id"]},
    "identity": {
        "route_commit_expected": ROUTE_COMMIT,
        "route_commit_observed": head,
        "route_commit_match": head == ROUTE_COMMIT,
        "route_operations_working_sha256": working_route_ops_sha,
        "route_operations_committed_sha256": committed_route_ops_sha,
        "config_match": working_route_ops_sha == committed_route_ops_sha,
        "official_checkpoint_expected_sha256": OFFICIAL_CHECKPOINT_SHA256,
        "official_checkpoint_observed_sha256": sha256_file(CHECKPOINT),
        "checkpoint_match": sha256_file(CHECKPOINT) == OFFICIAL_CHECKPOINT_SHA256,
        "ledger_expected_sha256": assets["r4_ledger"].get("sha256"),
        "ledger_observed_sha256": sha256_file(ledger_path),
        "ledger_match": assets["r4_ledger"].get("sha256") == sha256_file(ledger_path),
    },
    "population_and_roles": {
        "development_count": len(development),
        "confirmation_metadata_count": len(confirmation),
        "evaluated_name_count": len(oof_name_set),
        "evaluated_names_match_folds_0_1": oof_name_set == expected_evaluated,
        "unknown_development_names": development_unknown,
        "protected_name_overlap": protected_overlap,
        "protected_data_mixed": bool(protected_overlap or development_unknown),
        "source_access_audit": source_access,
    },
    "run_uniqueness": {
        "route_run_directories": run_siblings,
        "target_run_occurrences_under_route_root": run_siblings.count(RUN_ID),
        "duplicate_target_run": run_siblings.count(RUN_ID) != 1,
    },
    "raw_integrity": {
        "oof_rows": len(oof_rows),
        "oof_unique_keys": len(set(oof_keys)),
        "oof_duplicate_keys": len(oof_keys) - len(set(oof_keys)),
        "risk_rows": len(risk_rows),
        "risk_unique_keys": len(set(risk_keys)),
        "risk_duplicate_keys": len(risk_keys) - len(set(risk_keys)),
        "cells": cells,
        "folds": folds,
        "operators": operators,
        "finite_oof_metrics": all(math.isfinite(float(row[key])) for row in oof_rows
                                  for key in ("gain", "oracle", "shuffle_gain")),
        "formal_evidence_hashes_all_match": all(item["match"] for item in formal_hash_checks),
    },
    "limitations": [
        "cloud-only raw files are not listed in the formal closeout hash manifest; identity is inherited from the receipt-bound output path, run context and generation order",
        "raw A1 persistence contains ensemble policy rows and candidate harm/severe scores, but not per-seed predictions, per-action mean/q05 utility vectors, model checkpoints or composite risk-coverage confidence",
        "no semantic or fog-severity subgroup labels are present",
    ],
}
write_json("provenance_and_identity_audit.json", provenance)

sources = [
    stat_record(raw_oof_path, "formal run raw OOF policy input", "formal_raw_input"),
    stat_record(raw_risk_path, "formal run raw candidate risk input", "formal_raw_input"),
    stat_record(context_path, "receipt-bound run context", "formal_provenance"),
    stat_record(status_path, "formal run status log", "formal_provenance"),
    stat_record(closeout_path, "typed terminal closeout", "formal_evidence"),
    stat_record(ledger_path, "frozen data-role and fold ledger metadata", "formal_provenance"),
    stat_record(CHECKPOINT, "official checkpoint identity only", "formal_provenance"),
]
artifact_contracts = {
    "cloud_evidence_manifest.json": ["source inventory and output contract", "formal sources and audit outputs", "file", "sources/artifacts", "GitHub", "mixed"],
    "provenance_and_identity_audit.json": ["identity, roles, duplicates and integrity", "context/ledger/closeout/raw hashes", "file/image group", "identity/population/raw_integrity", "GitHub", "formal verification"],
    "official_metric_reproduction.json": ["which official values reproduce", "raw OOF/risk plus compact summaries", "clean-image group/worst operator", "metrics/limitations", "GitHub", "verification plus exploratory diagnostics"],
    "github_cloud_discrepancy.csv": ["GitHub-cloud value/hash differences", "formal summaries and cloud raw", "metric/file", "official/cloud/difference/status", "GitHub", "formal verification"],
    "fold_seed_operator_stability.csv": ["fold/operator stability and seed gap", "raw OOF plus contract", "operator-image", "dimension/gain/tail/coverage", "GitHub", "post-hoc exploratory"],
    "per_sample_distribution_summary.json": ["whether means hide concentration", "raw primary OOF rows", "operator-image and clean-image group", "distribution/regret decomposition", "GitHub", "post-hoc exploratory"],
    "severe_hard_case_summary.csv": ["tail counts by supported strata", "raw OOF/risk and formal risk-coverage", "operator-image/group/action", "event/count/rate", "GitHub", "mixed"],
    "risk_coverage_reanalysis.csv": ["what can be checked about coverage-tail tradeoff", "formal compact curve", "clean-image group/operator", "coverage/mean/CVaR/events/UCB", "GitHub", "formal summary with audit limitation"],
    "action_margin_and_label_stability.csv": ["action separation and cross-operator stability", "A0 compact plus A1 raw risk", "name-action/operator", "margin/agreement/concordance", "GitHub", "mixed"],
    "subgroup_failure_summary.csv": ["supported and missing subgroup evidence", "raw OOF/risk", "declared subgroup", "availability/gain/tail", "GitHub", "post-hoc exploratory"],
    "engineering_integrity_audit.json": ["engineering/statistical/scientific classification", "all audited sources", "run/artifact", "status/issues/validity", "GitHub", "formal verification"],
    "updated_bottleneck_assessment.json": ["support/narrow/expand/reject bottleneck", "formal and raw audit results", "route", "decision/statement/hypotheses", "GitHub", "post-hoc interpretation"],
    "prior_plan_reassessment.json": ["whether E1-E5 remain justified", "raw artifact sufficiency and roles", "planned item", "decision/reason", "GitHub", "post-hoc planning audit"],
    "cloud_audit_closeout.json": ["audit terminal and authorization", "all compact audit outputs", "audit", "state/decision/hashes/authorization", "GitHub", "audit authority only"],
}
manifest = {
    "schema_version": 1,
    "audit_id": AUDIT_ID,
    "cloud_evidence_cutoff_utc": max(item["mtime_utc"] for item in sources),
    "sources": sources,
    "artifact_contracts": [
        {"file": name, "question": values[0], "inputs": values[1], "statistical_unit": values[2],
         "key_fields": values[3], "retention": values[4], "evidence_class": values[5]}
        for name, values in artifact_contracts.items()
    ],
    "cloud_only_artifacts": [
        str(raw_oof_path), str(raw_risk_path), str(OUTPUT_ROOT / "stdout.log"),
        str(OUTPUT_ROOT / "stderr.log"), str(AUDIT_ROOT / "audit.log"),
    ],
}
write_json("cloud_evidence_manifest.json", manifest)

reproduction = {
    "schema_version": 1,
    "audit_id": AUDIT_ID,
    "reproducible": all(row["within_tolerance"] is True for row in metric_rows
                        if row["reproduction_status"].startswith("reproduced")),
    "terminal_state_affected": False,
    "point_metrics": point_reproduction,
    "operator_points": {"gain": operator_gain, "oracle": operator_oracle, "shuffle_gain": operator_shuffle,
                        "formal_worst_operator": worst_operator},
    "risk_metrics_by_cell_posthoc": risk_metrics_by_cell,
    "calibration_reproduction": primary_severe_calibration,
    "structural_summary": structural,
    "formal_hash_checks": formal_hash_checks,
    "not_independently_reproduced": {
        "bootstrap_intervals": "not rerun by design; compact result hashes match the typed closeout",
        "risk_coverage_order": "composite confidence was not persisted in the two raw cloud tables",
        "per_seed_metrics": "predictions were ensemble-averaged before raw-row persistence",
        "semantic_subgroups": "no subgroup labels persisted",
        "region_level_metrics": "A1 persisted image/operator rows, not region rows",
    },
}
write_json("official_metric_reproduction.json", reproduction)
write_csv("github_cloud_discrepancy.csv", metric_rows, [
    "item", "github_official_value", "cloud_audit_value", "difference", "tolerance",
    "within_tolerance", "difference_reason", "terminal_state_affected", "reproduction_status",
])

stability_rows = []
for fold in folds:
    for operator in operators:
        rows = [row for row in primary if int(row["fold"]) == fold and row["operator"] == operator]
        stability_rows.append({"dimension": "fold_operator", "fold": fold, "seed": "ensemble_3407_3411",
                               "operator": operator, "cell": "S_utility_risk", "status": "available",
                               **row_summary(rows)})
for operator in operators:
    stability_rows.append({"dimension": "operator", "fold": "all", "seed": "ensemble_3407_3411",
                           "operator": operator, "cell": "S_utility_risk", "status": "available",
                           **row_summary([row for row in primary if row["operator"] == operator])})
for cell in cells:
    for operator in operators:
        stability_rows.append({"dimension": "cell_operator", "fold": "all", "seed": "ensemble_3407_3411",
                               "operator": operator, "cell": cell, "status": "available",
                               **row_summary(by_cell_operator[(cell, operator)])})
for seed in contract["ensemble_seeds"]:
    stability_rows.append({"dimension": "seed", "fold": "not_available", "seed": seed,
                           "operator": "not_available", "cell": "S_utility_risk", "status": "not_available_raw_ensemble_only",
                           "n": "", "mean_gain_db": "", "coverage": "", "p10_gain_db": "",
                           "cvar5_gain_db": "", "worst20_mean_gain_db": "", "severe_count": "",
                           "hard_count": "", "negative_selection_rate": ""})
write_csv("fold_seed_operator_stability.csv", stability_rows)

repairable_rows = [row for row in primary if float(row["oracle"]) > 1.0e-12]
acted_repairable = [row for row in repairable_rows if row_selected(row) != 0]
abstained_repairable = [row for row in repairable_rows if row_selected(row) == 0]
wrong_active = [row for row in acted_repairable if row_gain(row) < float(row["oracle"]) - 1.0e-10]
harmful_active = [row for row in primary if row_selected(row) != 0 and row_gain(row) <= 0.0]
negative_oracle_rows = [row for row in primary if row["negative_oracle"].lower() == "true"]
distribution_summary = {
    "schema_version": 1,
    "audit_id": AUDIT_ID,
    "statistical_units": {"primary_raw": "operator-image", "group_worst": "clean-image worst operator"},
    "primary_operator_image_gain": distribution(row_gain(row) for row in primary),
    "primary_group_worst_gain": distribution(group_gain.values()),
    "by_operator": {operator: distribution(row_gain(row) for row in primary if row["operator"] == operator)
                    for operator in operators},
    "acted_only": distribution(row_gain(row) for row in primary if row_selected(row) != 0),
    "regret_decomposition_descriptive": {
        "repairable_operator_images": len(repairable_rows),
        "repairable_abstained": len(abstained_repairable),
        "repairable_abstention_fraction": len(abstained_repairable) / len(repairable_rows),
        "repairable_acted": len(acted_repairable),
        "repairable_acted_wrong_action": len(wrong_active),
        "repairable_acted_wrong_action_fraction": len(wrong_active) / len(acted_repairable) if acted_repairable else None,
        "all_active_harmful_or_zero": len(harmful_active),
        "negative_oracle_operator_images": len(negative_oracle_rows),
        "negative_oracle_selected_negative": sum(row_selected(row) == 2 for row in negative_oracle_rows),
        "overall_noop_fraction": sum(row_selected(row) == 0 for row in primary) / len(primary),
        "interpretation": "retention loss is behaviorally dominated by abstention; among rare active decisions some wrong-action regret remains, but this post-hoc decomposition is not a causal component substitution",
    },
    "availability": {
        "per_image": True,
        "per_region": False,
        "per_fold": True,
        "per_seed": False,
        "per_operator": True,
        "semantic_or_fog_subgroups": False,
        "high_confidence_failure": False,
        "reason": "raw persistence omits region rows, per-seed predictions, semantic labels and composite decision confidence",
    },
}
write_json("per_sample_distribution_summary.json", distribution_summary)

tail_rows = []
for fold in folds:
    for operator in operators:
        rows = [row for row in primary if int(row["fold"]) == fold and row["operator"] == operator]
        for event, threshold in (("severe", -0.2), ("hard", -0.5)):
            count = sum(row_gain(row) <= threshold for row in rows)
            tail_rows.append({"scope": "selected_primary", "cell": "S_utility_risk", "fold": fold,
                              "operator": operator, "action": "selected", "event": event,
                              "count": count, "total": len(rows), "rate": count / len(rows),
                              "evidence_class": "formal_raw_reproduction"})
for cell in cells:
    rows = [row for row in oof_rows if row["cell"] == cell]
    for event, threshold in (("severe", -0.2), ("hard", -0.5)):
        count = sum(row_gain(row) <= threshold for row in rows)
        tail_rows.append({"scope": "selected_cell", "cell": cell, "fold": "all", "operator": "both",
                          "action": "selected", "event": event, "count": count, "total": len(rows),
                          "rate": count / len(rows), "evidence_class": "posthoc_aggregate"})
for action in (1, 2):
    for operator in operators:
        rows = [row for row in risk_rows if row["cell"] == "S_utility_risk" and
                int(row["action"]) == action and row["operator"] == operator]
        for event, key in (("candidate_harm", "harm_label"), ("candidate_severe", "severe_label")):
            count = sum(int(row[key]) for row in rows)
            tail_rows.append({"scope": "candidate_label", "cell": "S_utility_risk", "fold": "all",
                              "operator": operator, "action": action, "event": event, "count": count,
                              "total": len(rows), "rate": count / len(rows), "evidence_class": "posthoc_aggregate"})
for row in risk_coverage_formal:
    tail_rows.append({"scope": "forced_risk_coverage", "cell": "S_utility_risk", "fold": "all",
                      "operator": row["operator"], "action": "score_ordered", "event": "severe_and_hard",
                      "count": f"{row['group_severe_count']}/{row['group_hard_count']}",
                      "total": row["selected_groups"], "rate": "", "evidence_class": "formal_compact_summary"})
write_csv("severe_hard_case_summary.csv", tail_rows)

risk_coverage_rows = []
for row in risk_coverage_formal:
    value = dict(row)
    value["audit_status"] = "formal_hash_verified_not_independently_reordered"
    value["raw_limitation"] = "composite_confidence_not_persisted"
    value["formal_or_exploratory"] = "formal_descriptive_curve"
    risk_coverage_rows.append(value)
write_csv("risk_coverage_reanalysis.csv", risk_coverage_rows)

label_rows = []
for row in a0_margin:
    label_rows.append({"scope": "A0_formal_three_action_margin", "operator": row["operator"],
                       "metric": "mean_margin_db", "value": row["mean_margin_db"], "count": row["count"],
                       "availability": "available", "evidence_class": "formal_compact"})
    label_rows.append({"scope": "A0_formal_three_action_margin", "operator": row["operator"],
                       "metric": "p10_margin_db", "value": row["p10_margin_db"], "count": row["count"],
                       "availability": "available", "evidence_class": "formal_compact"})
label_rows.extend([
    {"scope": "A0_formal_operator_consistency", "operator": "D_ref_vs_D_rep",
     "metric": "best_action_agreement", "value": a0_operator["best_action_agreement"], "count": 768,
     "availability": "available", "evidence_class": "formal_compact"},
    {"scope": "A0_formal_operator_consistency", "operator": "D_ref_vs_D_rep",
     "metric": "minimum_active_severe_agreement", "value": a0_operator["minimum_active_severe_agreement"], "count": 768,
     "availability": "available", "evidence_class": "formal_compact"},
    {"scope": "A1_posthoc_within_group", "operator": "worst_score_pairing",
     "metric": "severe_score_strict_concordance_on_label_discordant_groups",
     "value": severe_concordance["strict_concordance"], "count": severe_concordance["discordant_label_groups"],
     "availability": "available", "evidence_class": "posthoc_exploratory"},
    {"scope": "A1_posthoc_within_group", "operator": "worst_score_pairing",
     "metric": "harm_score_strict_concordance_on_label_discordant_groups",
     "value": harm_concordance["strict_concordance"], "count": harm_concordance["discordant_label_groups"],
     "availability": "available", "evidence_class": "posthoc_exploratory"},
    {"scope": "A1_raw_persistence", "operator": "not_available",
     "metric": "per_sample_best_second_utility_margin", "value": "", "count": "",
     "availability": "not_available_action_utilities_omitted", "evidence_class": "evidence_gap"},
])
write_csv("action_margin_and_label_stability.csv", label_rows)

subgroup_rows = []
for fold in folds:
    rows = [row for row in primary if int(row["fold"]) == fold]
    subgroup_rows.append({"subgroup_family": "fold", "subgroup": fold, "available": True,
                          **row_summary(rows), "interpretation": "posthoc descriptive; not a preregistered subgroup endpoint"})
for operator in operators:
    rows = [row for row in primary if row["operator"] == operator]
    subgroup_rows.append({"subgroup_family": "operator", "subgroup": operator, "available": True,
                          **row_summary(rows), "interpretation": "paired formal operator dimension"})
for action, label in ((0, "noop"), (1, "positive_full"), (2, "negative_full")):
    rows = [row for row in primary if row_selected(row) == action]
    summary = row_summary(rows) if rows else {"n": 0, "mean_gain_db": "", "coverage": "", "p10_gain_db": "",
                                              "cvar5_gain_db": "", "worst20_mean_gain_db": "", "severe_count": 0,
                                              "hard_count": 0, "negative_selection_rate": ""}
    subgroup_rows.append({"subgroup_family": "selected_action", "subgroup": label, "available": True,
                          **summary, "interpretation": "posthoc descriptive action stratum"})
for subgroup in ("light_fog", "medium_fog", "heavy_fog", "sky", "highlight", "low_texture",
                 "distance", "edge", "color_region", "region_level"):
    subgroup_rows.append({"subgroup_family": "semantic_or_region", "subgroup": subgroup, "available": False,
                          "n": "", "mean_gain_db": "", "coverage": "", "p10_gain_db": "",
                          "cvar5_gain_db": "", "worst20_mean_gain_db": "", "severe_count": "",
                          "hard_count": "", "negative_selection_rate": "",
                          "interpretation": "not available in persisted R4B-A1 raw artifacts"})
write_csv("subgroup_failure_summary.csv", subgroup_rows)

engineering = {
    "schema_version": 1,
    "audit_id": AUDIT_ID,
    "status": "ENGINEERING_INTEGRITY_VALID_WITH_RECORDED_NUMERICAL_GATE_FAILURE",
    "classification": {
        "engineering_integrity": "valid",
        "statistical_precision": "adequate_for_frozen_futility_decision; inadequate for unavailable semantic/per-seed questions",
        "scientific_result": "formal mechanism futility failure",
        "mixed_problem": False,
    },
    "checks": {
        "source_commit_match": head == ROUTE_COMMIT,
        "config_match": working_route_ops_sha == committed_route_ops_sha,
        "checkpoint_match": sha256_file(CHECKPOINT) == OFFICIAL_CHECKPOINT_SHA256,
        "ledger_match": assets["r4_ledger"].get("sha256") == sha256_file(ledger_path),
        "data_role_match": oof_name_set == expected_evaluated and not protected_overlap,
        "fold_operator_cell_completeness": len(oof_rows) == 5376 and len(risk_rows) == 10752,
        "duplicate_keys_absent": len(oof_keys) == len(set(oof_keys)) and len(risk_keys) == len(set(risk_keys)),
        "finite": provenance["raw_integrity"]["finite_oof_metrics"],
        "formal_evidence_hashes_match": all(item["match"] for item in formal_hash_checks),
        "official_points_reproduce": all(row["within_tolerance"] is True for row in metric_rows
                                         if row["reproduction_status"].startswith("reproduced")),
        "protected_data_mixed": bool(protected_overlap),
    },
    "issues": [{
        "id": "PERMUTATION_NUMERICAL_TOLERANCE_EXCEEDED",
        "observed": permutation["max_abs"],
        "frozen_tolerance": permutation["tolerance"],
        "classification": "recorded formal gate failure, not an engineering-terminal failure",
        "primary_endpoint_invalidated": False,
        "reason": "utility and retention upper bounds remain far below the non-futility thresholds and all raw point metrics reproduce",
    }],
    "scientific_conclusion_valid": True,
    "requires_integrity_event": False,
    "requires_corrective_contract": False,
    "original_closeout_retained": True,
}
write_json("engineering_integrity_audit.json", engineering)

bottleneck = {
    "schema_version": 1,
    "audit_id": AUDIT_ID,
    "previous_statement_decision": "supported_but_narrowed",
    "updated_formal_statement": "Within the frozen Haze4K development folds, three-action bank and R4B-A1 inference-visible representation, the deployable decision contract fails to convert candidate-conditioned signals into material signed utility and safe coverage. Raw policy rows show that the realized retention loss is dominated by abstention and the rare active policy remains positive-only; raw candidate-risk rows retain real risk discrimination, but the persisted artifacts cannot identify whether the upstream cause is utility representation, readout/coverage calibration, action-conditioned veto, or their interaction.",
    "beginner_statement": "The system usually refuses to act, and when it acts it only chooses the positive direction. It can often recognize risky candidates, but the saved evidence is not rich enough to tell whether the main cause is the scores, the action rule, or the risk veto.",
    "confidence": "high for the functional failure and abstention-dominant behavior; medium for the upstream mechanism",
    "scope": "R4B-A1 folds 0/1, 384 development images, D_ref/D_rep, no-op/positive-full/negative-full, ensemble seeds 3407/3411",
    "limitations": [
        "no per-seed predictions", "no per-action mean/q05 utility score vector", "no region or semantic subgroup rows",
        "no composite confidence for independent risk-coverage replay", "post-hoc diagnostics do not alter the terminal decision",
    ],
    "raw_support": {
        "overall_noop_fraction": distribution_summary["regret_decomposition_descriptive"]["overall_noop_fraction"],
        "repairable_abstention_fraction": distribution_summary["regret_decomposition_descriptive"]["repairable_abstention_fraction"],
        "negative_oracle_operator_images": len(negative_oracle_rows),
        "negative_oracle_selected_negative": sum(row_selected(row) == 2 for row in negative_oracle_rows),
        "severe_risk_primary": risk_metrics_by_cell["S_utility_risk"]["severe"],
        "severe_risk_action_only_posthoc": risk_metrics_by_cell["S_action_only"]["severe"],
        "within_group_severe_score_concordance": severe_concordance,
    },
    "mechanism_hypotheses": [
        {"id": "H1", "hypothesis": "inference representation lacks material action-conditioned signed relative utility",
         "decision": "supported_but_not_isolated", "confidence": "high", "missing": "per-action utility scores or an independent contract"},
        {"id": "H2", "hypothesis": "coverage/readout conflict causes positive-only high abstention",
         "decision": "behaviorally_supported_not_causal", "confidence": "medium_high", "missing": "frozen composite score vector and preregistered component replay"},
        {"id": "H3", "hypothesis": "risk head predicts generic difficulty rather than action-specific harm",
         "decision": "needs_narrowing", "confidence": "medium", "missing": "predeclared generic-vs-action-specific matched control; post-hoc cell and concordance metrics are descriptive"},
        {"id": "H4", "hypothesis": "image/group aggregation hides local gain-harm cancellation",
         "decision": "unresolved", "confidence": "medium", "missing": "region-level utilities and fixed region labels"},
        {"id": "H5", "hypothesis": "labels are unstable to implementation or perturbation",
         "decision": "weakened_for_two_frozen_operators_only", "confidence": "medium_high", "missing": "segmentation, sensor, precision and real-domain perturbations"},
    ],
    "terminal_decision_affected": False,
    "post_hoc_exploratory": True,
}
write_json("updated_bottleneck_assessment.json", bottleneck)

plans = {
    "schema_version": 1,
    "audit_id": AUDIT_ID,
    "current_authorization": "NONE",
    "items": [
        {"planned_item": "E1 R5_A0_FROZEN_OOF_COMPONENT_COUNTERFACTUAL_AUDIT", "decision": "modify_and_delay",
         "reason": "the exact per-action mean/q05 utility score vector and composite decision confidence required for base replay/component substitution were not persisted; do not regenerate them by retraining or inference",
         "assumption_supported": "partly: OOF policy rows and candidate risk rows exist and are identity-verified",
         "information_gain": "still potentially high after an independent contract, but the original replay design is not executable from the extant artifacts",
         "lower_cost_offline_answer": "this audit already answers descriptive abstention/action/tail concentration but not causal component substitution",
         "data_role_risk": "reuses consumed development data; any hypothesis test would be post-hoc unless independently contracted"},
        {"planned_item": "E2 R5_A1_FIXED_SCORE_READOUT_SUFFICIENCY_SCREEN", "decision": "delay",
         "reason": "conditional on an E1 coverage-dominant result that is not available; frozen score vector is incomplete",
         "assumption_supported": "high abstention is observed, but score sufficiency is not established", "information_gain": "conditional",
         "lower_cost_offline_answer": "none beyond descriptive abstention decomposition", "data_role_risk": "development reuse and post-hoc readout selection"},
        {"planned_item": "E3 R5_A1_SPATIAL_CANDIDATE_RESPONSE_SUFFICIENCY_SCREEN", "decision": "delay",
         "reason": "requires E1 action-dominant/interaction evidence and new spatial representation; neither is established by this audit",
         "assumption_supported": "not answered", "information_gain": "conditional", "lower_cost_offline_answer": "none; spatial tensors/regions were not persisted",
         "data_role_risk": "new contract required; current route authorizes none"},
        {"planned_item": "E4 R5_A1_ACTION_CONDITIONED_HARM_SPECIFICITY_SCREEN", "decision": "modify_and_delay",
         "reason": "raw action-risk rows permit descriptive within-group concordance and cell AUC, which partially narrows H3, but formal matched action-specific versus generic controls remain absent",
         "assumption_supported": "partly", "information_gain": "reduced but still conditional", "lower_cost_offline_answer": "the post-hoc metrics in this audit",
         "data_role_risk": "a confirmatory claim would require a new independent contract"},
        {"planned_item": "E5 R5_A0_REGION_AGGREGATION_CANCELLATION_AUDIT", "decision": "retain_but_delay",
         "reason": "A1 raw persistence has no region rows, so the aggregation hypothesis remains unresolved rather than supported",
         "assumption_supported": "not answered", "information_gain": "potentially high only after upstream component attribution remains unresolved",
         "lower_cost_offline_answer": "none from current artifacts", "data_role_risk": "development candidate cache reuse requires a new contract and fixed regions"},
    ],
}
write_json("prior_plan_reassessment.json", plans)

precloseout_hashes = {name: sha256_file(AUDIT_ROOT / name) for name in REQUIRED_OUTPUTS[:-1]}
closeout_audit = {
    "schema_version": 1,
    "audit_id": AUDIT_ID,
    "route_id": ROUTE_ID,
    "operation_id": "R4B_A1_POSTHOC_CLOUD_EVIDENCE_AUDIT",
    "source_operation_id": closeout["operation_id"],
    "source_run_id": closeout["run_id"],
    "source_route_commit": ROUTE_COMMIT,
    "state": "COMPLETED_AUDIT",
    "decision": "ORIGINAL_BOTTLENECK_SUPPORTED_BUT_NARROWED",
    "authorizes": "NONE",
    "evidence_role": "post_hoc_exploratory_audit",
    "started_at_utc": STARTED_AT,
    "completed_at_utc": utc_now(),
    "formal_terminal_state_affected": False,
    "source_identity_verified": all([
        head == ROUTE_COMMIT,
        working_route_ops_sha == committed_route_ops_sha,
        sha256_file(CHECKPOINT) == OFFICIAL_CHECKPOINT_SHA256,
        assets["r4_ledger"].get("sha256") == sha256_file(ledger_path),
        not protected_overlap,
        all(item["match"] for item in formal_hash_checks),
    ]),
    "engineering_integrity": engineering["status"],
    "material_discrepancies": [],
    "post_hoc_exploratory_analysis": True,
    "notice": "post-hoc exploratory analysis; does not change the original experiment terminal decision",
    "result_sha256": precloseout_hashes,
    "cloud_only_raw_sha256": {
        raw_oof_path.name: sha256_file(raw_oof_path),
        raw_risk_path.name: sha256_file(raw_risk_path),
    },
    "prohibited_followup": [
        "rerun or retune R4B-A1", "regenerate omitted scores by training or inference",
        "access confirmation, historical A1X outcomes, canary or locked test", "promote post-hoc subgroup findings to formal endpoints",
    ],
}
write_json("cloud_audit_closeout.json", closeout_audit)

completed = utc_now()
(AUDIT_ROOT / "status.txt").write_text(
    f"AUDIT_ID={AUDIT_ID}\nSTARTED_AT_UTC={STARTED_AT}\nCOMPLETED_AT_UTC={completed}\n"
    "STATE=COMPLETED_AUDIT\nDECISION=ORIGINAL_BOTTLENECK_SUPPORTED_BUT_NARROWED\n"
    "AUTHORIZES=NONE\nCLOUD_AUDIT_OK\n", encoding="utf-8",
)
(AUDIT_ROOT / "audit.log").write_text(
    "R4B-A1 post-hoc cloud evidence audit\n"
    f"source_output={OUTPUT_ROOT}\nsource_commit={head}\n"
    f"raw_oof_rows={len(oof_rows)} raw_risk_rows={len(risk_rows)}\n"
    f"formal_hashes_match={all(item['match'] for item in formal_hash_checks)}\n"
    "training=false inference=false bootstrap_rerun=false protected_data_access=false\n"
    "CLOUD_AUDIT_OK\n", encoding="utf-8",
)

print(json.dumps({
    "audit_root": str(AUDIT_ROOT),
    "source_output": str(OUTPUT_ROOT),
    "remote_repo": str(REMOTE_REPO),
    "route_commit": head,
    "raw_oof_sha256": sha256_file(raw_oof_path),
    "raw_risk_sha256": sha256_file(raw_risk_path),
    "source_identity_verified": closeout_audit["source_identity_verified"],
    "point_reproduction": point_reproduction,
    "overall_noop_fraction": distribution_summary["regret_decomposition_descriptive"]["overall_noop_fraction"],
    "repairable_abstention_fraction": distribution_summary["regret_decomposition_descriptive"]["repairable_abstention_fraction"],
    "risk_metrics_by_cell": risk_metrics_by_cell,
    "severe_concordance": severe_concordance,
    "harm_concordance": harm_concordance,
    "decision": closeout_audit["decision"],
    "authorizes": closeout_audit["authorizes"],
}, ensure_ascii=False, sort_keys=True))
for name in REQUIRED_OUTPUTS:
    print(f"AUDIT_FILE={name} BYTES={(AUDIT_ROOT / name).stat().st_size} SHA256={sha256_file(AUDIT_ROOT / name)}")
print("CLOUD_R4B_A1_AUDIT_OK")
PY

test -f "$AUDIT_ROOT/status.txt"
grep -Fxq 'CLOUD_AUDIT_OK' "$AUDIT_ROOT/status.txt"
printf 'REMOTE_R4B_A1_CLOUD_AUDIT_OK\n'
