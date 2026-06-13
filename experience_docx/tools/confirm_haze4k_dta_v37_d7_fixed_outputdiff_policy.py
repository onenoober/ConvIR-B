#!/usr/bin/env python3
"""D7 fixed-policy confirmation for the DTA-v3.7 D6 output-diff pass."""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

TOOL_DIR = Path(__file__).resolve().parent
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

from train_haze4k_dta_v37_d3_high_positive_policy import (  # noqa: E402
    STRICT_GATES,
    finite_float,
    gate_checks,
    read_csv,
    summarize,
    write_csv,
)
from train_haze4k_dta_v37_d5_targeted_intervention_policy import (  # noqa: E402
    SCORE_MODES,
    aggregate as aggregate_targeted,
    select_targeted,
)
from train_haze4k_dta_v37_d6_outputdiff_policy import (  # noqa: E402
    add_disagreement_features,
    feature_groups,
    join_outputdiff_features,
    make_base_actions,
)


DEFAULT_POLICIES = [
    {
        "policy_id": "primary_outputdiff_plus_Q_micro_shrink_pred_gain_t100",
        "role": "primary",
        "feature_group": "outputdiff_plus_Q",
        "action_bank": "micro_shrink",
        "score_mode": "pred_gain",
        "target_intervention": 1.0,
    },
    {
        "policy_id": "backup_outputdiff_only_micro_shrink_pred_gain_t100",
        "role": "tail_safety_backup",
        "feature_group": "outputdiff_only",
        "action_bank": "micro_shrink",
        "score_mode": "pred_gain",
        "target_intervention": 1.0,
    },
]


METRIC_KEYS = [
    "mean_dPSNR",
    "hard_bottom25_dPSNR",
    "easy_top25_dPSNR",
    "dSSIM",
    "positive_ratio",
    "worst_per_600",
    "max_outer_worst_per_600",
    "true_vs_zero",
    "true_vs_shuffle",
    "true_vs_normal",
    "intervention_rate",
    "mean_alpha",
]


def score_mode_by_name(name: str) -> dict[str, float]:
    for mode in SCORE_MODES:
        if mode["name"] == name:
            return mode
    raise KeyError(f"Unknown score mode: {name}")


def d6_match_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("feature_group")),
        str(row.get("action_bank")),
        str(row.get("score_mode", row.get("policy_name"))),
        f"{finite_float(row.get('target_intervention')):.2f}",
    )


def compare_to_d6(policy_row: dict[str, Any], d6_rows: list[dict[str, Any]], tolerance: float) -> dict[str, Any]:
    matches = [row for row in d6_rows if d6_match_key(row) == d6_match_key(policy_row)]
    if not matches:
        return {"d6_match_found": False, "d6_consistency_pass": False, "d6_metric_diffs": {}}
    d6 = matches[0]
    diffs: dict[str, float] = {}
    ok = True
    for key in METRIC_KEYS:
        diff = finite_float(policy_row.get(key), float("nan")) - finite_float(d6.get(key), float("nan"))
        diffs[key] = diff
        if not math.isfinite(diff) or abs(diff) > tolerance:
            ok = False
    return {
        "d6_match_found": True,
        "d6_consistency_pass": ok,
        "d6_metric_diffs": diffs,
    }


def outer_rows(selected: list[dict[str, Any]], policy: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_outer: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        by_outer[(str(row["fold"]), str(row["seed"]))].append(row)
    for (fold, seed), group in sorted(by_outer.items()):
        metrics = summarize(group)
        metrics["max_outer_worst_per_600"] = metrics["worst_per_600"]
        checks = gate_checks(metrics)
        metrics.update(
            {
                "policy_id": policy["policy_id"],
                "role": policy["role"],
                "fold": fold,
                "seed": seed,
                "feature_group": policy["feature_group"],
                "action_bank": policy["action_bank"],
                "score_mode": policy["score_mode"],
                "target_intervention": policy["target_intervention"],
                "strict_gate_checks": checks,
                "strict_gate_pass": all(checks.values()),
                "intervention_rate": sum(row.get("variant") != "A0" for row in group) / len(group),
            }
        )
        rows.append(metrics)
    return rows


def annotate_selected(selected: list[dict[str, Any]], policy: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in selected:
        annotated = dict(row)
        annotated.update(
            {
                "fixed_policy_id": policy["policy_id"],
                "fixed_policy_role": policy["role"],
                "fixed_feature_group": policy["feature_group"],
                "fixed_action_bank": policy["action_bank"],
                "fixed_score_mode": policy["score_mode"],
                "fixed_target_intervention": policy["target_intervention"],
            }
        )
        out.append(annotated)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--single_actions_csv", required=True, type=Path)
    ap.add_argument("--feature_action_table_csv", required=True, type=Path)
    ap.add_argument("--outputdiff_features_csv", required=True, type=Path)
    ap.add_argument("--d6_aggregate_csv", required=True, type=Path)
    ap.add_argument("--output_dir", required=True, type=Path)
    ap.add_argument("--output_prefix", default="v37_d7_fixed_outputdiff")
    ap.add_argument("--include_run_substring", default="quick5full")
    ap.add_argument("--consistency_tolerance", default=1e-9, type=float)
    ap.add_argument(
        "--policy_ids",
        default="",
        help="Optional comma-separated fixed policy ids to run; default runs all D7 fixed policies.",
    )
    ap.add_argument(
        "--skip_d6_consistency",
        action="store_true",
        help="For broader formal confirmations, gate on strict metrics without exact D6 aggregate matching.",
    )
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    actions = make_base_actions(read_csv(args.single_actions_csv), read_csv(args.feature_action_table_csv), args.include_run_substring)
    output_rows = read_csv(args.outputdiff_features_csv)
    actions = join_outputdiff_features(actions, output_rows)
    add_disagreement_features(actions)
    groups = feature_groups(set().union(*(row.keys() for row in actions)), actions)
    d6_rows = read_csv(args.d6_aggregate_csv)

    fixed_aggregate: list[dict[str, Any]] = []
    fixed_outer: list[dict[str, Any]] = []
    selected_all: list[dict[str, Any]] = []
    missing: list[str] = []

    allowed_policy_ids = {item.strip() for item in args.policy_ids.split(",") if item.strip()}
    fixed_policies = [policy for policy in DEFAULT_POLICIES if not allowed_policy_ids or policy["policy_id"] in allowed_policy_ids]
    if allowed_policy_ids and len(fixed_policies) != len(allowed_policy_ids):
        found = {policy["policy_id"] for policy in fixed_policies}
        missing_policy_ids = sorted(allowed_policy_ids - found)
        raise SystemExit(f"Unknown fixed policy id(s): {missing_policy_ids}")

    for policy in fixed_policies:
        cols = groups.get(policy["feature_group"], [])
        if not cols:
            missing.append(f"{policy['policy_id']}:empty_feature_group")
            continue
        selected = select_targeted(
            actions,
            cols,
            policy["action_bank"],
            score_mode_by_name(policy["score_mode"]),
            float(policy["target_intervention"]),
        )
        row = aggregate_targeted(
            selected,
            policy["feature_group"],
            policy["action_bank"],
            policy["score_mode"],
            float(policy["target_intervention"]),
        )
        row.update(
            {
                "policy_id": policy["policy_id"],
                "role": policy["role"],
                "phase": "D7_fixed_outputdiff_confirmation",
                "fixed_policy": True,
                "feature_count": len(cols),
                "locked_test_touched": False,
                "raw_d1_full_5x3_run": False,
            }
        )
        if args.skip_d6_consistency:
            row.update({"d6_match_found": False, "d6_consistency_pass": True, "d6_metric_diffs": {}})
        else:
            row.update(compare_to_d6(row, d6_rows, args.consistency_tolerance))
        fixed_aggregate.append(row)
        fixed_outer.extend(outer_rows(selected, policy))
        selected_all.extend(annotate_selected(selected, policy))

    primary = next((row for row in fixed_aggregate if row.get("role") == "primary"), None)
    backup = next((row for row in fixed_aggregate if row.get("role") == "tail_safety_backup"), None)
    primary_pass = bool(primary and primary.get("strict_gate_pass") and primary.get("d6_consistency_pass"))
    backup_pass = bool(backup and backup.get("strict_gate_pass") and backup.get("d6_consistency_pass"))
    decision = "D7_FIXED_OUTPUTDIFF_CONFIRM_PASS" if (primary_pass or backup_pass) and not missing else "D7_FIXED_OUTPUTDIFF_CONFIRM_FAIL"
    sealed = primary if primary_pass else backup if backup_pass else {}

    aggregate_path = args.output_dir / f"{args.output_prefix}_policy_aggregate.csv"
    outer_path = args.output_dir / f"{args.output_prefix}_per_outer_report.csv"
    selected_path = args.output_dir / f"{args.output_prefix}_selected_actions.csv"
    summary_path = args.output_dir / f"{args.output_prefix}_summary.json"
    fixed_config_path = args.output_dir / f"{args.output_prefix}_fixed_policy_config.json"
    write_csv(aggregate_path, fixed_aggregate)
    write_csv(outer_path, fixed_outer)
    write_csv(selected_path, selected_all)
    fixed_config_path.write_text(json.dumps(fixed_policies, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "route": "DTA-v3.7 U-TQS-Mix",
        "phase": "D7_fixed_outputdiff_confirmation",
        "decision": decision,
        "requested_policy_ids": sorted(allowed_policy_ids),
        "skip_d6_consistency": args.skip_d6_consistency,
        "fixed_policy_count": len(fixed_aggregate),
        "strict_consistent_count": sum(bool(row.get("strict_gate_pass") and row.get("d6_consistency_pass")) for row in fixed_aggregate),
        "primary_pass": primary_pass,
        "backup_pass": backup_pass,
        "sealed_policy_candidate": sealed,
        "missing": missing,
        "rows": len(actions),
        "outputdiff_rows": len(output_rows),
        "image_groups": len({(row["image_id"], row["fold"], row["seed"]) for row in actions}),
        "aggregate_csv": str(aggregate_path),
        "per_outer_csv": str(outer_path),
        "selected_actions_csv": str(selected_path),
        "fixed_policy_config_json": str(fixed_config_path),
        "locked_test_touched": False,
        "raw_d1_full_5x3_run": False,
        "strict_gates": STRICT_GATES,
        "note": "D7 freezes D6 strict rows and does not perform grid/policy selection.",
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"DTA_V3_7_D7_FIXED_OUTPUTDIFF_CONFIRM_OK policies={len(fixed_aggregate)} "
        f"strict_consistent={summary['strict_consistent_count']} decision={decision}",
        flush=True,
    )


if __name__ == "__main__":
    main()
