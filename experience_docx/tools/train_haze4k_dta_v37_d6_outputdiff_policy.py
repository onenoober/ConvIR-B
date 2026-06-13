#!/usr/bin/env python3
"""D6 deployable output-difference/quality policy audit for DTA-v3.7."""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

TOOL_DIR = Path(__file__).resolve().parent
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

from train_haze4k_dta_v37_d3_high_positive_policy import (  # noqa: E402
    BANKS,
    DIAGNOSTIC_GT,
    FDF_FEATURES,
    POLICIES,
    STRICT_GATES,
    TUA_FEATURES,
    VARIANTS,
    aggregate as aggregate_highpos,
    finite_float,
    make_actions as make_base_actions,
    read_csv,
    select_policy,
    write_csv,
)
from train_haze4k_dta_v37_d5_targeted_intervention_policy import (  # noqa: E402
    SCORE_MODES,
    TARGETS,
    aggregate as aggregate_targeted,
    select_targeted,
)


DEPLOYABLE_BASE = [
    "input_brightness_mean",
    "input_texture_mean",
    "airlight_fallback_mean",
    "depth_mean",
    "depth_std",
]
OUTPUT_PREFIXES = ("od_", "outq_", "oq_", "diffgrp_")
OUTPUT_DISAGREE_KEYS = [
    "od_action_res_abs_mean",
    "od_action_res_abs_p95",
    "od_action_luma_abs_mean",
    "od_action_edge_mean",
    "od_action_concentration_top10",
    "oq_luma_mean_delta_vs_a0",
    "oq_contrast_p95_p05_delta_vs_a0",
    "oq_sat_mean_delta_vs_a0",
    "oq_edge_mean_delta_vs_a0",
    "outq_luma_mean",
    "outq_edge_mean",
    "outq_sat_mean",
]


def action_key(row: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(row.get("image_id")),
        str(row.get("fold")),
        str(row.get("seed")),
        str(row.get("variant")),
        f"{finite_float(row.get('alpha')):.2f}",
    )


def is_numeric_column(rows: list[dict[str, Any]], col: str) -> bool:
    values = [finite_float(row.get(col), float("nan")) for row in rows[: min(len(rows), 2000)]]
    finite = [value for value in values if math.isfinite(value)]
    return len(finite) >= max(10, len(values) // 20) and len({round(value, 10) for value in finite}) >= 2


def join_outputdiff_features(actions: list[dict[str, Any]], output_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output_cols = [col for col in output_rows[0] if col.startswith(OUTPUT_PREFIXES)]
    feature_by_key = {action_key(row): {col: row.get(col, 0.0) for col in output_cols} for row in output_rows}
    joined: list[dict[str, Any]] = []
    missing: list[tuple[str, str, str, str, str]] = []
    for row in actions:
        out = dict(row)
        if out.get("variant") == "A0":
            for col in output_cols:
                out[col] = 0.0
        else:
            key = action_key(out)
            features = feature_by_key.get(key)
            if features is None:
                missing.append(key)
                for col in output_cols:
                    out[col] = 0.0
            else:
                out.update(features)
        joined.append(out)
    if missing:
        examples = ", ".join(map(str, missing[:8]))
        raise ValueError(f"Missing D6 output-diff features for {len(missing)} action rows; examples={examples}")
    return joined


def add_disagreement_features(actions: list[dict[str, Any]]) -> None:
    by_group: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in actions:
        by_group[(str(row["image_id"]), str(row["fold"]), str(row["seed"]), f"{finite_float(row.get('alpha')):.2f}")].append(row)
    for rows in by_group.values():
        non_a0 = [row for row in rows if row.get("variant") != "A0"]
        if not non_a0:
            continue
        for key in OUTPUT_DISAGREE_KEYS:
            vals = [finite_float(row.get(key)) for row in non_a0]
            mean_val = float(np.mean(vals))
            range_val = float(np.max(vals) - np.min(vals))
            max_val = float(np.max(vals))
            for row in rows:
                value = finite_float(row.get(key))
                row[f"diffgrp_{key}_minus_mean"] = value - mean_val if row.get("variant") != "A0" else 0.0
                row[f"diffgrp_{key}_range"] = range_val
                row[f"diffgrp_{key}_is_group_max"] = 1.0 if row.get("variant") != "A0" and abs(value - max_val) <= 1e-12 else 0.0


def feature_groups(columns: set[str], rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    def keep(cols: list[str]) -> list[str]:
        unique = []
        for col in cols:
            if col in columns and col not in unique and is_numeric_column(rows, col):
                unique.append(col)
        return unique

    output_cols = sorted(col for col in columns if col.startswith(OUTPUT_PREFIXES))
    groups = {
        "Q_input_proxy": DEPLOYABLE_BASE,
        "TQAU_action_all": DEPLOYABLE_BASE + TUA_FEATURES + FDF_FEATURES,
        "outputdiff_only": list(output_cols),
        "outputdiff_plus_Q": DEPLOYABLE_BASE + list(output_cols),
        "deployable_TQAU_outputdiff_all": DEPLOYABLE_BASE + TUA_FEATURES + FDF_FEATURES + list(output_cols),
        "diagnostic_trans_TQAU_outputdiff": DEPLOYABLE_BASE + TUA_FEATURES + FDF_FEATURES + DIAGNOSTIC_GT + list(output_cols),
    }
    return {name: keep(cols) for name, cols in groups.items()}


def run_highpos(actions: list[dict[str, Any]], groups: dict[str, list[str]]) -> list[dict[str, Any]]:
    aggregates: list[dict[str, Any]] = []
    for group_name, cols in groups.items():
        if not cols:
            continue
        for bank_name in BANKS:
            for policy in POLICIES:
                selected = select_policy(actions, cols, bank_name, policy)
                row = aggregate_highpos(selected, group_name, bank_name, policy["name"])
                row.update({"phase": "D6_outputdiff_highpos", "selection_mode": policy["name"]})
                aggregates.append(row)
    return aggregates


def run_targeted(actions: list[dict[str, Any]], groups: dict[str, list[str]]) -> list[dict[str, Any]]:
    aggregates: list[dict[str, Any]] = []
    for group_name, cols in groups.items():
        if not cols:
            continue
        for bank_name in BANKS:
            for score_mode in SCORE_MODES:
                for target in TARGETS:
                    selected = select_targeted(actions, cols, bank_name, score_mode, target)
                    row = aggregate_targeted(selected, group_name, bank_name, score_mode["name"], target)
                    row.update({"phase": "D6_outputdiff_targeted", "policy_name": score_mode["name"], "selection_mode": score_mode["name"]})
                    aggregates.append(row)
    return aggregates


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--single_actions_csv", required=True, type=Path)
    ap.add_argument("--feature_action_table_csv", required=True, type=Path)
    ap.add_argument("--outputdiff_features_csv", required=True, type=Path)
    ap.add_argument("--output_dir", required=True, type=Path)
    ap.add_argument("--output_prefix", default="v37_d6_outputdiff")
    ap.add_argument("--include_run_substring", default="quick5full")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    actions = make_base_actions(read_csv(args.single_actions_csv), read_csv(args.feature_action_table_csv), args.include_run_substring)
    output_rows = read_csv(args.outputdiff_features_csv)
    actions = join_outputdiff_features(actions, output_rows)
    add_disagreement_features(actions)
    groups = feature_groups(set().union(*(row.keys() for row in actions)), actions)

    aggregates = run_highpos(actions, groups) + run_targeted(actions, groups)
    aggregates.sort(key=lambda row: (bool(row["strict_gate_pass"]), finite_float(row["score"])), reverse=True)
    strict_rows = [row for row in aggregates if row["strict_gate_pass"]]
    aggregate_path = args.output_dir / f"{args.output_prefix}_policy_aggregate.csv"
    nested_path = args.output_dir / f"{args.output_prefix}_policy_nested_report.csv"
    summary_path = args.output_dir / f"{args.output_prefix}_summary.json"
    feature_path = args.output_dir / f"{args.output_prefix}_feature_groups.json"
    write_csv(aggregate_path, aggregates)
    write_csv(nested_path, aggregates)
    feature_path.write_text(json.dumps(groups, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "route": "DTA-v3.7 U-TQS-Mix",
        "phase": "D6_outputdiff_quality_deployable_policy",
        "rows": len(actions),
        "outputdiff_rows": len(output_rows),
        "image_groups": len({(row["image_id"], row["fold"], row["seed"]) for row in actions}),
        "aggregate_csv": str(aggregate_path),
        "nested_report_csv": str(nested_path),
        "feature_groups_json": str(feature_path),
        "feature_group_counts": {name: len(cols) for name, cols in groups.items()},
        "strict_pass_count": len(strict_rows),
        "best_row": aggregates[0] if aggregates else {},
        "decision": "D6_OUTPUTDIFF_POLICY_STRICT_PASS" if strict_rows else "D6_OUTPUTDIFF_POLICY_STRICT_FAIL",
        "locked_test_touched": False,
        "strict_gates": STRICT_GATES,
        "leakage_note": "deployable_TQAU_outputdiff_all excludes trans_gt; diagnostic_trans_TQAU_outputdiff is not promotion-deployable.",
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"DTA_V3_7_D6_OUTPUTDIFF_POLICY_OK rows={len(actions)} outputdiff_rows={len(output_rows)} "
        f"aggregate={len(aggregates)} strict_pass={len(strict_rows)} decision={summary['decision']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
