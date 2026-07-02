#!/usr/bin/env python3
"""D5 targeted-intervention policy audit for DTA-v3.7 D1/D3 actions."""
from __future__ import annotations

import argparse
import json
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
    POLICIES,
    VARIANTS,
    feature_groups,
    finite_float,
    fit_ridge,
    pred_ridge,
    read_csv,
    make_actions,
    design_matrix,
    summarize,
    gate_checks,
    write_csv,
    STRICT_GATES,
)

TARGETS = [0.35, 0.50, 0.60, 0.65, 0.70, 0.75, 0.85, 1.00]
SCORE_MODES = [
    {"name": "pred_gain", "pos": 0.0, "strong": 0.0, "severe": 0.0, "ssim": 0.0, "alpha": 0.0},
    {"name": "pred_highpos", "pos": 0.10, "strong": 0.10, "severe": 1.0, "ssim": 100.0, "alpha": 0.01},
    {"name": "pred_positive_forced", "pos": 0.20, "strong": 0.05, "severe": 1.5, "ssim": 150.0, "alpha": 0.02},
]


def select_targeted(actions: list[dict[str, Any]], cols: list[str], bank_name: str, score_mode: dict[str, float], target: float) -> list[dict[str, Any]]:
    outer_keys = sorted({(str(row["fold"]), str(row["seed"])) for row in actions})
    selected: list[dict[str, Any]] = []
    all_cols = cols + ["alpha_float", "alpha_ge_050", "alpha_is_full"] + [f"variant_is_{v}" for v in VARIANTS]
    bank = BANKS[bank_name]
    for outer in outer_keys:
        train = [r for r in actions if (str(r["fold"]), str(r["seed"])) != outer and round(finite_float(r.get("alpha")), 2) in bank]
        test = [r for r in actions if (str(r["fold"]), str(r["seed"])) == outer and round(finite_float(r.get("alpha")), 2) in bank]
        X = design_matrix(train, all_cols)
        y = np.asarray([finite_float(r.get("dPSNR")) for r in train], dtype=np.float64)
        gain = fit_ridge(X, y, alpha=3.0)
        pos = fit_ridge(X, (y > 0).astype(float), alpha=3.0)
        strong = fit_ridge(X, (y <= -0.05).astype(float), alpha=3.0)
        severe = fit_ridge(X, (y <= -0.20).astype(float), alpha=3.0)
        ssim_bad = fit_ridge(X, np.asarray([finite_float(r.get("dSSIM")) < -0.000005 for r in train], dtype=float), alpha=3.0)
        Xt = design_matrix(test, all_cols)
        pred_gain = pred_ridge(gain, Xt)
        pred_pos = np.clip(pred_ridge(pos, Xt), 0, 1)
        pred_strong = np.clip(pred_ridge(strong, Xt), 0, 1)
        pred_severe = np.clip(pred_ridge(severe, Xt), 0, 1)
        pred_ssim_bad = np.clip(pred_ridge(ssim_bad, Xt), 0, 1)
        by_image: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for idx, row in enumerate(test):
            out = dict(row)
            score = (
                pred_gain[idx]
                + score_mode["pos"] * pred_pos[idx]
                - score_mode["strong"] * pred_strong[idx]
                - score_mode["severe"] * pred_severe[idx]
                - score_mode["ssim"] * pred_ssim_bad[idx]
                + score_mode["alpha"] * finite_float(row.get("alpha"))
            )
            if row.get("variant") == "A0":
                score = -1e9
            out.update({
                "pred_gain": float(pred_gain[idx]), "pred_pos_prob": float(pred_pos[idx]),
                "pred_strong_prob": float(pred_strong[idx]), "pred_severe_prob": float(pred_severe[idx]),
                "pred_ssim_bad_prob": float(pred_ssim_bad[idx]), "policy_score": float(score),
            })
            by_image[(str(row["image_id"]), str(row["fold"]), str(row["seed"]))].append(out)
        best_non_a0 = []
        a0_rows = {}
        for key, cands in by_image.items():
            a0_rows[key] = [r for r in cands if r.get("variant") == "A0"][0]
            non_a0 = [r for r in cands if r.get("variant") != "A0"]
            best_non_a0.append((key, max(non_a0, key=lambda r: (finite_float(r["policy_score"]), finite_float(r["pred_gain"]), finite_float(r["alpha"])))) )
        best_non_a0.sort(key=lambda item: finite_float(item[1]["policy_score"]), reverse=True)
        take_n = int(round(target * len(best_non_a0)))
        take = {key for key, _ in best_non_a0[:take_n]}
        for key, row in best_non_a0:
            selected.append(row if key in take else a0_rows[key])
    return selected


def aggregate(rows: list[dict[str, Any]], feature_group: str, bank_name: str, score_mode: str, target: float) -> dict[str, Any]:
    metrics = summarize(rows)
    outer = defaultdict(list)
    for row in rows:
        outer[(str(row["fold"]), str(row["seed"]))].append(row)
    metrics["max_outer_worst_per_600"] = max(finite_float(summarize(v).get("worst_per_600")) for v in outer.values())
    checks = gate_checks(metrics)
    metrics.update({
        "feature_group": feature_group,
        "action_bank": bank_name,
        "score_mode": score_mode,
        "target_intervention": target,
        "model_type": "nested_numpy_ridge_targeted_intervention",
        "outer_groups": len(outer),
        "strict_gate_checks": checks,
        "strict_gate_pass": all(checks.values()),
        "intervention_rate": sum(r.get("variant") != "A0" for r in rows) / len(rows),
        "mean_alpha": float(np.mean([finite_float(r.get("alpha")) for r in rows])),
        "score": finite_float(metrics.get("mean_dPSNR")) + 0.5 * finite_float(metrics.get("hard_bottom25_dPSNR")) + 0.05 * finite_float(metrics.get("positive_ratio")) - 0.005 * finite_float(metrics.get("worst_per_600")),
    })
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--single_actions_csv", required=True, type=Path)
    ap.add_argument("--feature_action_table_csv", required=True, type=Path)
    ap.add_argument("--output_dir", required=True, type=Path)
    ap.add_argument("--output_prefix", default="v37_d5_targeted")
    ap.add_argument("--include_run_substring", default="quick5full")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    actions = make_actions(read_csv(args.single_actions_csv), read_csv(args.feature_action_table_csv), args.include_run_substring)
    groups = feature_groups(set().union(*(r.keys() for r in actions)))
    selected_all: list[dict[str, Any]] = []
    aggregates: list[dict[str, Any]] = []
    for group_name, cols in groups.items():
        for bank_name in BANKS:
            for score_mode in SCORE_MODES:
                for target in TARGETS:
                    selected = select_targeted(actions, cols, bank_name, score_mode, target)
                    for row in selected:
                        row.update({"feature_group": group_name, "action_bank": bank_name, "score_mode": score_mode["name"], "target_intervention": target})
                    selected_all.extend(selected)
                    aggregates.append(aggregate(selected, group_name, bank_name, score_mode["name"], target))
    aggregates.sort(key=lambda r: (bool(r["strict_gate_pass"]), finite_float(r["score"])), reverse=True)
    aggregate_path = args.output_dir / f"{args.output_prefix}_policy_aggregate.csv"
    nested_path = args.output_dir / f"{args.output_prefix}_policy_nested_report.csv"
    selected_path = args.output_dir / f"{args.output_prefix}_policy_selected_actions.csv"
    summary_path = args.output_dir / f"{args.output_prefix}_summary.json"
    write_csv(aggregate_path, aggregates)
    write_csv(nested_path, aggregates)
    write_csv(selected_path, selected_all)
    strict_rows = [r for r in aggregates if r["strict_gate_pass"]]
    summary = {
        "route": "DTA-v3.7 U-TQS-Mix",
        "phase": "D5_targeted_intervention_policy",
        "rows": len(actions),
        "image_groups": len({(r["image_id"], r["fold"], r["seed"]) for r in actions}),
        "aggregate_csv": str(aggregate_path),
        "selected_actions_csv": str(selected_path),
        "strict_pass_count": len(strict_rows),
        "best_row": aggregates[0] if aggregates else {},
        "decision": "D5_TARGETED_INTERVENTION_POLICY_STRICT_PASS" if strict_rows else "D5_TARGETED_INTERVENTION_POLICY_STRICT_FAIL",
        "locked_test_touched": False,
        "strict_gates": STRICT_GATES,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"DTA_V3_7_D5_TARGETED_INTERVENTION_POLICY_OK rows={len(actions)} aggregate={len(aggregates)} strict_pass={len(strict_rows)} decision={summary['decision']}")


if __name__ == "__main__":
    main()
