#!/usr/bin/env python3
"""Train nested deployable high-positive policies on D3 actual TAU actions.

This version intentionally depends only on the stdlib and NumPy because the
cloud ConvIR runtime does not include pandas/sklearn.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

STRICT_GATES = {
    "coverage_min": 0.95,
    "mean_dpsnr_min": 0.055,
    "hard_bottom25_min": 0.040,
    "dssim_min": -0.000005,
    "positive_ratio_min": 0.630,
    "true_vs_zero_min": 0.040,
    "true_vs_shuffle_min": 0.035,
    "true_vs_normal_min": 0.030,
    "worst_per_600_max": 48.0,
    "max_outer_worst_per_600_max": 60.0,
}
DEPLOYABLE_BASE = ["input_brightness_mean", "input_texture_mean", "airlight_fallback_mean", "depth_mean", "depth_std"]
TUA_FEATURES = ["dta_t_pred_mean", "dta_t_pred_std", "dta_t_uncertainty_mean", "dta_t_uncertainty_std", "dta_airlight_pred_mean", "dta_airlight_uncertainty_mean"]
FDF_FEATURES = [
    "dta_stage2_gate_mean", "dta_stage2_gate_max", "dta_stage2_conf_mean", "dta_stage2_conf_min",
    "dta_stage2_gamma_abs_mean", "dta_stage2_beta_abs_mean", "dta_stage2_delta_abs_mean",
    "dta_stage3_gate_mean", "dta_stage3_gate_max", "dta_stage3_conf_mean", "dta_stage3_conf_min",
    "dta_stage3_gamma_abs_mean", "dta_stage3_beta_abs_mean", "dta_stage3_delta_abs_mean",
    "dta_stage2_feature_gate_mean", "dta_stage2_feature_gate_max", "dta_stage2_feature_delta_abs_mean", "dta_stage2_feature_action_abs_mean",
    "dta_stage3_feature_gate_mean", "dta_stage3_feature_gate_max", "dta_stage3_feature_delta_abs_mean", "dta_stage3_feature_action_abs_mean",
    "dta_final_feature_gate_mean", "dta_final_feature_gate_max", "dta_final_feature_delta_abs_mean", "dta_final_feature_action_abs_mean",
]
DIAGNOSTIC_GT = ["trans_gt_mean", "trans_gt_p10", "trans_gt_p90"]
VARIANTS = ["A0", "u1_tau_l1_s004_g025_a006", "u2_tau_l3_s004_g015_a006", "u3_tau_l2_s002_g025_a006"]
BANKS = {"full": {0.0, 1.0}, "shrink": {0.0, 0.25, 0.50, 0.75, 1.0}, "micro_shrink": {0.0, 0.10, 0.25, 0.50, 0.75, 1.0}}
POLICIES = [
    {"name": "gain", "pos": 0.00, "strong": 0.20, "severe": 2.0, "ssim": 150.0, "alpha": 0.000, "min_score": 0.000},
    {"name": "highpos_v1", "pos": 0.06, "strong": 0.20, "severe": 2.0, "ssim": 200.0, "alpha": 0.005, "min_score": 0.000},
    {"name": "highpos_v2", "pos": 0.10, "strong": 0.15, "severe": 2.5, "ssim": 250.0, "alpha": 0.010, "min_score": 0.000},
    {"name": "positive_breakthrough", "pos": 0.14, "strong": 0.10, "severe": 3.0, "ssim": 250.0, "alpha": 0.015, "min_score": -0.005},
    {"name": "tail_guarded_highpos", "pos": 0.10, "strong": 0.35, "severe": 4.0, "ssim": 300.0, "alpha": 0.010, "min_score": 0.000},
]
TARGET_KEYS = {"cand_PSNR", "dPSNR", "dSSIM", "zero_delta_psnr", "shuffle_delta_psnr", "normal_delta_psnr", "failure_group", "A0_PSNR"}
ID_KEYS = {"image_id", "fold", "seed", "variant", "run_id"}


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(row.get(key), sort_keys=True) if isinstance(row.get(key), (dict, list)) else row.get(key, "") for key in keys})


def percentile(values: list[float], pct: float) -> float:
    vals = sorted(v for v in values if math.isfinite(v))
    if not vals:
        return float("nan")
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * pct / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    return vals[lo] if lo == hi else vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    deltas = [finite_float(row.get("dPSNR")) for row in rows]
    ssim_deltas = [finite_float(row.get("dSSIM")) for row in rows]
    a0_psnr = [finite_float(row.get("A0_PSNR")) for row in rows]
    sorted_idx = sorted(range(len(rows)), key=lambda idx: a0_psnr[idx])
    bucket_count = max(1, len(rows) // 4)
    hard_idx = sorted_idx[:bucket_count]
    easy_idx = sorted_idx[-bucket_count:]
    strong_cut = percentile(a0_psnr, 75.0)
    strong_idx = [idx for idx, value in enumerate(a0_psnr) if value >= strong_cut]
    n = len(rows)
    return {
        "count": n,
        "coverage": 1.0,
        "mean_dPSNR": float(np.mean(deltas)),
        "hard_bottom25_dPSNR": float(np.mean([deltas[idx] for idx in hard_idx])),
        "easy_top25_dPSNR": float(np.mean([deltas[idx] for idx in easy_idx])),
        "dSSIM": float(np.mean(ssim_deltas)),
        "positive_ratio": sum(delta > 0 for delta in deltas) / n,
        "strong_reference_cut_psnr": strong_cut,
        "strong_reference_regression_count": sum(deltas[idx] <= -0.05 for idx in strong_idx),
        "strong_count_le_-0.05": sum(delta <= -0.05 for delta in deltas),
        "strong_per_600": sum(delta <= -0.05 for delta in deltas) * 600.0 / n,
        "worst_count_le_-0.20": sum(delta <= -0.20 for delta in deltas),
        "worst_per_600": sum(delta <= -0.20 for delta in deltas) * 600.0 / n,
        "true_vs_zero": float(np.mean([finite_float(r.get("dPSNR")) - finite_float(r.get("zero_delta_psnr")) for r in rows])),
        "true_vs_shuffle": float(np.mean([finite_float(r.get("dPSNR")) - finite_float(r.get("shuffle_delta_psnr")) for r in rows])),
        "true_vs_normal": float(np.mean([finite_float(r.get("dPSNR")) - finite_float(r.get("normal_delta_psnr")) for r in rows])),
    }


def gate_checks(metrics: dict[str, Any]) -> dict[str, bool]:
    return {
        "coverage": finite_float(metrics.get("coverage")) >= STRICT_GATES["coverage_min"],
        "mean": finite_float(metrics.get("mean_dPSNR"), -1e9) >= STRICT_GATES["mean_dpsnr_min"],
        "hard": finite_float(metrics.get("hard_bottom25_dPSNR"), -1e9) >= STRICT_GATES["hard_bottom25_min"],
        "dssim": finite_float(metrics.get("dSSIM"), -1e9) >= STRICT_GATES["dssim_min"],
        "positive_ratio": finite_float(metrics.get("positive_ratio")) >= STRICT_GATES["positive_ratio_min"],
        "true_vs_zero": finite_float(metrics.get("true_vs_zero"), -1e9) >= STRICT_GATES["true_vs_zero_min"],
        "true_vs_shuffle": finite_float(metrics.get("true_vs_shuffle"), -1e9) >= STRICT_GATES["true_vs_shuffle_min"],
        "true_vs_normal": finite_float(metrics.get("true_vs_normal"), -1e9) >= STRICT_GATES["true_vs_normal_min"],
        "worst": finite_float(metrics.get("worst_per_600"), 1e9) <= STRICT_GATES["worst_per_600_max"],
        "max_outer_worst": finite_float(metrics.get("max_outer_worst_per_600"), 1e9) <= STRICT_GATES["max_outer_worst_per_600_max"],
    }


def feature_groups(columns: set[str]) -> dict[str, list[str]]:
    groups = {
        "Q_input_proxy": DEPLOYABLE_BASE,
        "T_pred": DEPLOYABLE_BASE + TUA_FEATURES,
        "FDF_action_stats": DEPLOYABLE_BASE + FDF_FEATURES,
        "deployable_TQAU_action_all": DEPLOYABLE_BASE + TUA_FEATURES + FDF_FEATURES,
        "diagnostic_with_trans_gt": DEPLOYABLE_BASE + TUA_FEATURES + FDF_FEATURES + DIAGNOSTIC_GT,
    }
    return {name: [col for col in cols if col in columns] for name, cols in groups.items()}


def make_actions(single_rows: list[dict[str, Any]], feature_rows: list[dict[str, Any]], include_run_substring: str) -> list[dict[str, Any]]:
    feat_by_key: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    image_feat: dict[tuple[str, str, str], dict[str, Any]] = {}
    feature_cols = [k for k in feature_rows[0] if k not in TARGET_KEYS and k not in ID_KEYS]
    for row in feature_rows:
        if include_run_substring not in str(row.get("run_id", "")):
            continue
        key = (str(row["image_id"]), str(row["fold"]), str(row["seed"]), str(row["variant"]))
        feat_by_key.setdefault(key, {col: row.get(col, 0.0) for col in feature_cols})
        ikey = key[:3]
        image_feat.setdefault(ikey, {col: row.get(col, 0.0) for col in DEPLOYABLE_BASE + DIAGNOSTIC_GT if col in feature_cols})
    actions: list[dict[str, Any]] = []
    image_seen: dict[tuple[str, str, str], str] = {}
    for row in single_rows:
        row = dict(row)
        row["fold"], row["seed"] = str(row["fold"]), str(row["seed"])
        row["alpha"] = finite_float(row.get("alpha"))
        key = (str(row["image_id"]), row["fold"], row["seed"], str(row["variant"]))
        if key not in feat_by_key:
            raise ValueError(f"Missing feature row for {key}")
        row.update(feat_by_key[key])
        actions.append(row)
        image_seen[key[:3]] = row.get("A0_PSNR", "0")
    for ikey, a0_psnr in image_seen.items():
        row = {
            "image_id": ikey[0], "fold": ikey[1], "seed": ikey[2], "action": "A0@0", "variant": "A0", "alpha": 0.0,
            "A0_PSNR": a0_psnr, "dPSNR": 0.0, "dSSIM": 0.0,
            "zero_delta_psnr": 0.0, "shuffle_delta_psnr": 0.0, "normal_delta_psnr": 0.0,
        }
        row.update(image_feat.get(ikey, {}))
        for col in feature_cols:
            row.setdefault(col, 0.0)
        actions.append(row)
    for row in actions:
        row["alpha_float"] = finite_float(row.get("alpha"))
        row["alpha_ge_050"] = 1.0 if row["alpha_float"] >= 0.5 else 0.0
        row["alpha_is_full"] = 1.0 if row["alpha_float"] >= 0.999 else 0.0
        for variant in VARIANTS:
            row[f"variant_is_{variant}"] = 1.0 if row.get("variant") == variant else 0.0
    return actions


def design_matrix(rows: list[dict[str, Any]], cols: list[str]) -> np.ndarray:
    return np.asarray([[finite_float(row.get(col)) for col in cols] for row in rows], dtype=np.float64)


def fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float = 3.0) -> dict[str, Any]:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-8] = 1.0
    Xs = (X - mean) / std
    Xb = np.concatenate([np.ones((Xs.shape[0], 1)), Xs], axis=1)
    reg = np.eye(Xb.shape[1]) * alpha
    reg[0, 0] = 0.0
    beta = np.linalg.pinv(Xb.T @ Xb + reg) @ Xb.T @ y
    return {"mean": mean, "std": std, "beta": beta}


def pred_ridge(model: dict[str, Any], X: np.ndarray) -> np.ndarray:
    Xs = (X - model["mean"]) / model["std"]
    Xb = np.concatenate([np.ones((Xs.shape[0], 1)), Xs], axis=1)
    return Xb @ model["beta"]


def select_policy(actions: list[dict[str, Any]], cols: list[str], bank_name: str, policy: dict[str, Any]) -> list[dict[str, Any]]:
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
                + policy["pos"] * pred_pos[idx]
                - policy["strong"] * pred_strong[idx]
                - policy["severe"] * pred_severe[idx]
                - policy["ssim"] * pred_ssim_bad[idx]
                + policy["alpha"] * finite_float(row.get("alpha"))
            )
            if row.get("variant") == "A0":
                score = 0.0
            out.update({
                "pred_gain": float(pred_gain[idx]), "pred_pos_prob": float(pred_pos[idx]),
                "pred_strong_prob": float(pred_strong[idx]), "pred_severe_prob": float(pred_severe[idx]),
                "pred_ssim_bad_prob": float(pred_ssim_bad[idx]), "policy_score": float(score),
            })
            by_image[(str(row["image_id"]), str(row["fold"]), str(row["seed"]))].append(out)
        for cands in by_image.values():
            non_a0 = [r for r in cands if r.get("variant") != "A0"]
            viable = [r for r in non_a0 if finite_float(r.get("policy_score")) >= policy["min_score"]]
            chosen = max(viable, key=lambda r: (finite_float(r["policy_score"]), finite_float(r["pred_gain"]), finite_float(r["alpha"]))) if viable else [r for r in cands if r.get("variant") == "A0"][0]
            selected.append(chosen)
    return selected


def aggregate(rows: list[dict[str, Any]], feature_group: str, bank_name: str, policy_name: str) -> dict[str, Any]:
    metrics = summarize(rows)
    outer = defaultdict(list)
    for row in rows:
        outer[(str(row["fold"]), str(row["seed"]))].append(row)
    metrics["max_outer_worst_per_600"] = max(finite_float(summarize(v).get("worst_per_600")) for v in outer.values())
    checks = gate_checks(metrics)
    metrics.update({
        "feature_group": feature_group,
        "action_bank": bank_name,
        "policy_name": policy_name,
        "model_type": "nested_numpy_ridge_high_positive",
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
    ap.add_argument("--output_prefix", default="v37_d4_highpos")
    ap.add_argument("--include_run_substring", default="quick5full")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    actions = make_actions(read_csv(args.single_actions_csv), read_csv(args.feature_action_table_csv), args.include_run_substring)
    groups = feature_groups(set().union(*(r.keys() for r in actions)))
    selected_all: list[dict[str, Any]] = []
    aggregates: list[dict[str, Any]] = []
    for group_name, cols in groups.items():
        for bank_name in BANKS:
            for policy in POLICIES:
                selected = select_policy(actions, cols, bank_name, policy)
                for row in selected:
                    row.update({"feature_group": group_name, "action_bank": bank_name, "policy_name": policy["name"]})
                selected_all.extend(selected)
                aggregates.append(aggregate(selected, group_name, bank_name, policy["name"]))
    aggregates.sort(key=lambda r: (bool(r["strict_gate_pass"]), finite_float(r["score"])), reverse=True)
    selected_path = args.output_dir / f"{args.output_prefix}_policy_selected_actions.csv"
    aggregate_path = args.output_dir / f"{args.output_prefix}_policy_aggregate.csv"
    nested_path = args.output_dir / f"{args.output_prefix}_policy_nested_report.csv"
    summary_path = args.output_dir / f"{args.output_prefix}_summary.json"
    write_csv(selected_path, selected_all)
    write_csv(aggregate_path, aggregates)
    write_csv(nested_path, aggregates)
    strict_rows = [r for r in aggregates if r["strict_gate_pass"]]
    summary = {
        "route": "DTA-v3.7 U-TQS-Mix",
        "phase": "D4_d3_high_positive_deployable_policy",
        "rows": len(actions),
        "image_groups": len({(r["image_id"], r["fold"], r["seed"]) for r in actions}),
        "aggregate_csv": str(aggregate_path),
        "selected_actions_csv": str(selected_path),
        "strict_pass_count": len(strict_rows),
        "best_row": aggregates[0] if aggregates else {},
        "decision": "D4_HIGH_POSITIVE_POLICY_STRICT_PASS" if strict_rows else "D4_HIGH_POSITIVE_POLICY_STRICT_FAIL",
        "locked_test_touched": False,
        "strict_gates": STRICT_GATES,
        "leakage_note": "deployable groups exclude trans_gt; diagnostic_with_trans_gt is not promotion-deployable.",
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"DTA_V3_7_D4_HIGH_POSITIVE_POLICY_OK rows={len(actions)} aggregate={len(aggregates)} strict_pass={len(strict_rows)} decision={summary['decision']}")


if __name__ == "__main__":
    main()
