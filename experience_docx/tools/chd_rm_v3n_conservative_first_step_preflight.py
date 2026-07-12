#!/usr/bin/env python3
"""v3n-A0 conservative first-step label preflight.

Reads the v3m-A1 cloud-only block table and tests one fixed target-semantics
rule: default to alpha=0.125, and allow only a single alpha=0.25 escalation
when direct_step_energy exceeds the 99th percentile of train-fold negatives
(`oracle_alpha <= 0.125`). This is label-only. It does not train, tune a
policy family, replay images, touch canary, or touch locked test.
"""

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


ACTION_VALUES = (0.0, 0.125, 0.25, 0.5, 1.0)
OPERATORS = ("D_ref", "D_rep")
DECISION_PASS = "V3N_A0_CONSERVATIVE_LABEL_PREFLIGHT_PASS_AUTHORIZE_A1_REPLAY_SMOKE_ONLY"
DECISION_FAIL = "V3N_A0_CONSERVATIVE_LABEL_PREFLIGHT_FAIL_STOP_NO_REPLAY"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_float(value):
    text = str(value).strip()
    if text.lower() == "nan":
        return float("nan")
    return float(text)


def action_index(alpha):
    value = parse_float(alpha)
    best = min(range(len(ACTION_VALUES)), key=lambda idx: abs(ACTION_VALUES[idx] - value))
    if abs(ACTION_VALUES[best] - value) > 1e-9:
        raise ValueError(f"unexpected oracle alpha: {alpha}")
    return best


def quantile(sorted_values, q):
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = (len(sorted_values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_values[lo])
    weight = pos - lo
    return float(sorted_values[lo] * (1.0 - weight) + sorted_values[hi] * weight)


def write_rows(path, rows):
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_negative_scores_by_fold(block_rows):
    negative_scores = defaultdict(list)
    row_counts = defaultdict(int)
    with Path(block_rows).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            operator = row["operator_label"]
            if operator not in OPERATORS:
                continue
            fold = int(row["fold"])
            oracle_idx = action_index(row["oracle_alpha"])
            score = parse_float(row["direct_step_energy"])
            if not math.isfinite(score):
                raise ValueError(f"non-finite direct_step_energy for {operator} fold {fold}")
            row_counts[(operator, fold)] += 1
            if oracle_idx <= 1:
                negative_scores[(operator, fold)].append(score)
    for key, values in negative_scores.items():
        values.sort()
    return negative_scores, row_counts


def build_thresholds(negative_scores, quantile_value):
    thresholds = {}
    for operator in OPERATORS:
        for holdout_fold in range(5):
            train_values = []
            for fold in range(5):
                if fold == holdout_fold:
                    continue
                train_values.extend(negative_scores[(operator, fold)])
            train_values.sort()
            thresholds[(operator, holdout_fold)] = {
                "threshold": quantile(train_values, quantile_value),
                "train_negative_count": len(train_values),
            }
    return thresholds


def empty_counts():
    return {
        "heldout_total_blocks": 0,
        "heldout_positive_blocks": 0,
        "heldout_negative_blocks": 0,
        "selected_blocks": 0,
        "selected_positive_blocks": 0,
        "selected_negative_blocks": 0,
        "selected_oracle_action_0": 0,
        "selected_oracle_action_1": 0,
        "selected_oracle_action_2": 0,
        "selected_oracle_action_3": 0,
        "selected_oracle_action_4": 0,
    }


def evaluate(block_rows, thresholds):
    counts = defaultdict(empty_counts)
    with Path(block_rows).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            operator = row["operator_label"]
            if operator not in OPERATORS:
                continue
            fold = int(row["fold"])
            key = (operator, fold)
            oracle_idx = action_index(row["oracle_alpha"])
            score = parse_float(row["direct_step_energy"])
            selected = score > thresholds[key]["threshold"]
            counts[key]["heldout_total_blocks"] += 1
            if oracle_idx >= 2:
                counts[key]["heldout_positive_blocks"] += 1
            else:
                counts[key]["heldout_negative_blocks"] += 1
            if selected:
                counts[key]["selected_blocks"] += 1
                counts[key][f"selected_oracle_action_{oracle_idx}"] += 1
                if oracle_idx >= 2:
                    counts[key]["selected_positive_blocks"] += 1
                else:
                    counts[key]["selected_negative_blocks"] += 1
    return counts


def add_rates(row):
    selected = row["selected_blocks"]
    total = row["heldout_total_blocks"]
    positives = row["heldout_positive_blocks"]
    negatives = row["heldout_negative_blocks"]
    row["selected_coverage"] = selected / total if total else float("nan")
    row["positive_recall"] = row["selected_positive_blocks"] / positives if positives else float("nan")
    row["negative_false_rate"] = row["selected_negative_blocks"] / negatives if negatives else float("nan")
    row["selected_precision"] = row["selected_positive_blocks"] / selected if selected else float("nan")
    row["selected_match_alpha_0p25_rate"] = row["selected_oracle_action_2"] / selected if selected else float("nan")
    row["selected_over_escalate_rate"] = (
        (row["selected_oracle_action_0"] + row["selected_oracle_action_1"]) / selected if selected else float("nan")
    )
    row["selected_under_escalate_rate"] = (
        (row["selected_oracle_action_3"] + row["selected_oracle_action_4"]) / selected if selected else float("nan")
    )
    return row


def summarize_operator(operator, fold_rows, args):
    summary = empty_counts()
    summary["operator_label"] = operator
    summary["fold_count"] = len(fold_rows)
    for row in fold_rows:
        for key, value in row.items():
            if key in summary and isinstance(value, int):
                summary[key] += value
    add_rates(summary)
    summary["max_fold_negative_false_rate"] = max(row["negative_false_rate"] for row in fold_rows)
    summary["min_fold_selected_coverage"] = min(row["selected_coverage"] for row in fold_rows)
    summary["min_fold_selected_precision"] = min(row["selected_precision"] for row in fold_rows)
    summary["min_fold_positive_recall"] = min(row["positive_recall"] for row in fold_rows)
    summary["operator_gate_pass"] = (
        summary["negative_false_rate"] <= args.max_negative_false_rate
        and summary["max_fold_negative_false_rate"] <= args.max_fold_negative_false_rate_per_fold
        and summary["selected_coverage"] >= args.min_selected_coverage
        and summary["min_fold_selected_coverage"] >= args.min_selected_coverage_per_fold
        and summary["selected_precision"] >= args.min_selected_precision
        and summary["positive_recall"] >= args.min_positive_recall
    )
    return summary


def write_closeout(path, summary):
    lines = []
    for row in summary["operator_summaries"]:
        lines.append(
            "| `{operator_label}` | `{selected_coverage:.7f}` | `{positive_recall:.7f}` | "
            "`{negative_false_rate:.7f}` | `{selected_precision:.7f}` | "
            "`{selected_over_escalate_rate:.7f}` | `{operator_gate_pass}` |".format(**row)
        )
    text = f"""# v3n A0 Conservative First-Step Label Preflight

Decision: `{summary['decision']}`.

This is a label-only diagnostic over the v3m-A1 cloud block table. It uses a
fixed rule: default to `alpha=0.125`, and allow only `alpha=0.25` when
`direct_step_energy` exceeds the 99th percentile of train-fold negative blocks.
It does not train, tune thresholds, replay images, use route-confirm, touch
canary, or touch locked test.

| Operator | Selected coverage | Positive recall | Negative false rate | Selected precision | Selected over-escalate | Pass |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
{chr(10).join(lines)}

No policy utility is claimed by this label-only phase. A pass authorizes only a
separate A1 32-image replay-smoke preflight; a fail stops this v3n rule.
"""
    Path(path).write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block_rows", required=True, type=Path)
    parser.add_argument("--evid", required=True, type=Path)
    parser.add_argument("--negative_quantile", type=float, default=0.99)
    parser.add_argument("--max_negative_false_rate", type=float, default=0.0125)
    parser.add_argument("--max_negative_false_rate_per_fold", type=float, default=0.02)
    parser.add_argument("--min_selected_coverage", type=float, default=0.005)
    parser.add_argument("--min_selected_coverage_per_fold", type=float, default=0.0025)
    parser.add_argument("--min_selected_precision", type=float, default=0.60)
    parser.add_argument("--min_positive_recall", type=float, default=0.01)
    args = parser.parse_args()

    args.evid.mkdir(parents=True, exist_ok=True)
    negative_scores, row_counts = load_negative_scores_by_fold(args.block_rows)
    thresholds = build_thresholds(negative_scores, args.negative_quantile)
    counts = evaluate(args.block_rows, thresholds)

    fold_rows = []
    for operator in OPERATORS:
        for fold in range(5):
            key = (operator, fold)
            row = {
                "operator_label": operator,
                "holdout_fold": fold,
                "threshold_direct_step_energy": thresholds[key]["threshold"],
                "train_negative_count": thresholds[key]["train_negative_count"],
                **counts[key],
            }
            add_rates(row)
            fold_rows.append(row)

    operator_summaries = []
    for operator in OPERATORS:
        operator_summaries.append(
            summarize_operator(operator, [row for row in fold_rows if row["operator_label"] == operator], args)
        )

    route_pass = all(row["operator_gate_pass"] for row in operator_summaries)
    summary = {
        "decision": DECISION_PASS if route_pass else DECISION_FAIL,
        "label_only": True,
        "policy_replay_used": False,
        "training_used": False,
        "threshold_family_search_used": False,
        "route_confirm_used": False,
        "canary_touched": False,
        "locked_test_touched": False,
        "inputs": {
            "block_rows": str(args.block_rows),
            "block_rows_sha256": sha256_file(args.block_rows),
        },
        "contract": {
            "score": "direct_step_energy",
            "default_alpha": 0.125,
            "selected_alpha": 0.25,
            "negative_definition": "oracle_alpha <= 0.125",
            "positive_definition": "oracle_alpha >= 0.25",
            "negative_quantile": args.negative_quantile,
            "max_negative_false_rate": args.max_negative_false_rate,
            "max_negative_false_rate_per_fold": args.max_negative_false_rate_per_fold,
            "min_selected_coverage": args.min_selected_coverage,
            "min_selected_coverage_per_fold": args.min_selected_coverage_per_fold,
            "min_selected_precision": args.min_selected_precision,
            "min_positive_recall": args.min_positive_recall,
        },
        "operator_summaries": operator_summaries,
    }

    write_rows(args.evid / "v3n_a0_conservative_first_step_fold_summary.csv", fold_rows)
    write_rows(args.evid / "v3n_a0_conservative_first_step_operator_summary.csv", operator_summaries)
    write_json(args.evid / "v3n_a0_conservative_first_step_summary.json", summary)
    write_closeout(args.evid / "v3n_a0_conservative_first_step_closeout.md", summary)
    print("V3N_A0_CONSERVATIVE_FIRST_STEP_PREFLIGHT_OK")


if __name__ == "__main__":
    main()
