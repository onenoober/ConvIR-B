#!/usr/bin/env python3
"""Train nested deployable high-positive policies on D3 actual TAU actions."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore", category=ConvergenceWarning)

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

DEPLOYABLE_BASE = [
    "input_brightness_mean",
    "input_texture_mean",
    "airlight_fallback_mean",
    "depth_mean",
    "depth_std",
]
TUA_FEATURES = [
    "dta_t_pred_mean",
    "dta_t_pred_std",
    "dta_t_uncertainty_mean",
    "dta_t_uncertainty_std",
    "dta_airlight_pred_mean",
    "dta_airlight_uncertainty_mean",
]
FDF_FEATURES = [
    "dta_stage2_gate_mean",
    "dta_stage2_gate_max",
    "dta_stage2_conf_mean",
    "dta_stage2_conf_min",
    "dta_stage2_gamma_abs_mean",
    "dta_stage2_beta_abs_mean",
    "dta_stage2_delta_abs_mean",
    "dta_stage3_gate_mean",
    "dta_stage3_gate_max",
    "dta_stage3_conf_mean",
    "dta_stage3_conf_min",
    "dta_stage3_gamma_abs_mean",
    "dta_stage3_beta_abs_mean",
    "dta_stage3_delta_abs_mean",
    "dta_stage2_feature_gate_mean",
    "dta_stage2_feature_gate_max",
    "dta_stage2_feature_delta_abs_mean",
    "dta_stage2_feature_action_abs_mean",
    "dta_stage3_feature_gate_mean",
    "dta_stage3_feature_gate_max",
    "dta_stage3_feature_delta_abs_mean",
    "dta_stage3_feature_action_abs_mean",
    "dta_final_feature_gate_mean",
    "dta_final_feature_gate_max",
    "dta_final_feature_delta_abs_mean",
    "dta_final_feature_action_abs_mean",
]
DIAGNOSTIC_GT = ["trans_gt_mean", "trans_gt_p10", "trans_gt_p90"]
VARIANTS = ["A0", "u1_tau_l1_s004_g025_a006", "u2_tau_l3_s004_g015_a006", "u3_tau_l2_s002_g025_a006"]
BANKS = {
    "full": {0.0, 1.0},
    "shrink": {0.0, 0.25, 0.50, 0.75, 1.0},
    "micro_shrink": {0.0, 0.10, 0.25, 0.50, 0.75, 1.0},
}
POLICIES = [
    {"name": "gain", "pos": 0.00, "strong": 0.20, "severe": 2.0, "ssim": 150.0, "alpha": 0.000, "min_score": 0.000},
    {"name": "highpos_v1", "pos": 0.06, "strong": 0.20, "severe": 2.0, "ssim": 200.0, "alpha": 0.005, "min_score": 0.000},
    {"name": "highpos_v2", "pos": 0.10, "strong": 0.15, "severe": 2.5, "ssim": 250.0, "alpha": 0.010, "min_score": 0.000},
    {"name": "positive_breakthrough", "pos": 0.14, "strong": 0.10, "severe": 3.0, "ssim": 250.0, "alpha": 0.015, "min_score": -0.005},
    {"name": "tail_guarded_highpos", "pos": 0.10, "strong": 0.35, "severe": 4.0, "ssim": 300.0, "alpha": 0.010, "min_score": 0.000},
]


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def percentile(values: list[float], pct: float) -> float:
    vals = sorted(v for v in values if math.isfinite(v))
    if not vals:
        return float("nan")
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * pct / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return vals[lo]
    return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(row.get(key), sort_keys=True) if isinstance(row.get(key), (dict, list)) else row.get(key, "")
                for key in keys
            })


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
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

    def mean_at(indices: list[int]) -> float:
        return float(np.mean([deltas[idx] for idx in indices])) if indices else float("nan")

    def surplus(control_key: str) -> float:
        return float(np.mean([finite_float(row.get("dPSNR")) - finite_float(row.get(control_key)) for row in rows]))

    worst_count = sum(delta <= -0.20 for delta in deltas)
    strong_count = sum(delta <= -0.05 for delta in deltas)
    return {
        "count": n,
        "coverage": 1.0,
        "mean_dPSNR": float(np.mean(deltas)),
        "hard_bottom25_dPSNR": mean_at(hard_idx),
        "easy_top25_dPSNR": mean_at(easy_idx),
        "dSSIM": float(np.mean(ssim_deltas)),
        "positive_ratio": sum(delta > 0 for delta in deltas) / n,
        "strong_reference_cut_psnr": strong_cut,
        "strong_reference_regression_count": sum(deltas[idx] <= -0.05 for idx in strong_idx),
        "strong_count_le_-0.05": strong_count,
        "strong_per_600": strong_count * 600.0 / n,
        "worst_count_le_-0.20": worst_count,
        "worst_per_600": worst_count * 600.0 / n,
        "true_vs_zero": surplus("zero_delta_psnr"),
        "true_vs_shuffle": surplus("shuffle_delta_psnr"),
        "true_vs_normal": surplus("normal_delta_psnr"),
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


def fit_binary(X: np.ndarray, y: np.ndarray):
    if len(set(y.tolist())) < 2:
        return None, float(y[0]) if len(y) else 0.0
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0))
    model.fit(X, y)
    return model, None


def predict_binary(model_info, X: np.ndarray) -> np.ndarray:
    model, constant = model_info
    if model is None:
        return np.full(X.shape[0], float(constant))
    return model.predict_proba(X)[:, 1]


def feature_groups(columns: set[str]) -> dict[str, list[str]]:
    groups = {
        "Q_input_proxy": DEPLOYABLE_BASE,
        "T_pred": DEPLOYABLE_BASE + TUA_FEATURES,
        "FDF_action_stats": DEPLOYABLE_BASE + FDF_FEATURES,
        "deployable_TQAU_action_all": DEPLOYABLE_BASE + TUA_FEATURES + FDF_FEATURES,
        "diagnostic_with_trans_gt": DEPLOYABLE_BASE + TUA_FEATURES + FDF_FEATURES + DIAGNOSTIC_GT,
    }
    return {name: [col for col in cols if col in columns] for name, cols in groups.items()}


def build_action_frame(single_actions: pd.DataFrame, feature_table: pd.DataFrame, include_run_substring: str) -> pd.DataFrame:
    for col in ["fold", "seed"]:
        single_actions[col] = single_actions[col].astype(str)
        feature_table[col] = feature_table[col].astype(str)
    single_actions["alpha"] = single_actions["alpha"].astype(float)
    feature_table = feature_table[feature_table["run_id"].astype(str).str.contains(include_run_substring, regex=False)].copy()
    key = ["image_id", "fold", "seed", "variant"]
    feature_table = feature_table.drop_duplicates(key)
    exclude = {
        "run_id", "cand_PSNR", "dPSNR", "dSSIM", "zero_delta_psnr", "shuffle_delta_psnr", "normal_delta_psnr",
        "failure_group", "A0_PSNR", "image_id", "fold", "seed", "variant",
    }
    feature_cols = [c for c in feature_table.columns if c not in exclude]
    merged = single_actions.merge(feature_table[key + feature_cols], on=key, how="left")
    if merged[feature_cols].isna().all(axis=1).any():
        missing = merged.loc[merged[feature_cols].isna().all(axis=1), key].head(5).to_dict("records")
        raise ValueError(f"Missing feature rows for examples: {missing}")

    image_cols = [c for c in DEPLOYABLE_BASE + DIAGNOSTIC_GT if c in feature_cols]
    image_features = feature_table.groupby(["image_id", "fold", "seed"], as_index=False)[image_cols].mean()
    base_rows = []
    for _, row in single_actions[["image_id", "fold", "seed", "A0_PSNR"]].drop_duplicates().iterrows():
        base_rows.append({
            "image_id": row["image_id"],
            "fold": row["fold"],
            "seed": row["seed"],
            "action": "A0@0",
            "variant": "A0",
            "alpha": 0.0,
            "A0_PSNR": row["A0_PSNR"],
            "dPSNR": 0.0,
            "dSSIM": 0.0,
            "zero_delta_psnr": 0.0,
            "shuffle_delta_psnr": 0.0,
            "normal_delta_psnr": 0.0,
        })
    a0 = pd.DataFrame(base_rows).merge(image_features, on=["image_id", "fold", "seed"], how="left")
    for col in feature_cols:
        if col not in a0.columns:
            a0[col] = 0.0
    merged = pd.concat([merged, a0[merged.columns]], ignore_index=True)
    for variant in VARIANTS:
        merged[f"variant_is_{variant}"] = (merged["variant"] == variant).astype(float)
    merged["alpha_float"] = merged["alpha"].astype(float)
    merged["alpha_ge_050"] = (merged["alpha_float"] >= 0.50).astype(float)
    merged["alpha_is_full"] = (merged["alpha_float"] >= 0.999).astype(float)
    return merged.fillna(0.0)


def select_policy_rows(df: pd.DataFrame, feature_cols: list[str], bank_name: str, policy: dict[str, float]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    group_cols = ["fold", "seed"]
    outer_keys = sorted(df[group_cols].drop_duplicates().itertuples(index=False, name=None))
    all_feature_cols = feature_cols + ["alpha_float", "alpha_ge_050", "alpha_is_full"] + [f"variant_is_{v}" for v in VARIANTS]
    for outer_key in outer_keys:
        test_mask = (df["fold"] == outer_key[0]) & (df["seed"] == outer_key[1])
        train = df.loc[~test_mask].copy()
        test = df.loc[test_mask].copy()
        train = train[train["alpha_float"].round(2).isin(BANKS[bank_name])]
        test = test[test["alpha_float"].round(2).isin(BANKS[bank_name])]
        X_train = train[all_feature_cols].astype(float).to_numpy()
        y_gain = train["dPSNR"].astype(float).to_numpy()
        ridge = make_pipeline(StandardScaler(), Ridge(alpha=3.0))
        ridge.fit(X_train, y_gain)
        pos_model = fit_binary(X_train, (y_gain > 0.0).astype(int))
        strong_model = fit_binary(X_train, (y_gain <= -0.05).astype(int))
        severe_model = fit_binary(X_train, (y_gain <= -0.20).astype(int))
        ssim_model = fit_binary(X_train, (train["dSSIM"].astype(float).to_numpy() < -0.000005).astype(int))

        X_test = test[all_feature_cols].astype(float).to_numpy()
        test = test.copy()
        test["pred_gain"] = ridge.predict(X_test)
        test["pred_pos_prob"] = predict_binary(pos_model, X_test)
        test["pred_strong_prob"] = predict_binary(strong_model, X_test)
        test["pred_severe_prob"] = predict_binary(severe_model, X_test)
        test["pred_ssim_bad_prob"] = predict_binary(ssim_model, X_test)
        test["policy_score"] = (
            test["pred_gain"]
            + float(policy["pos"]) * test["pred_pos_prob"]
            - float(policy["strong"]) * test["pred_strong_prob"]
            - float(policy["severe"]) * test["pred_severe_prob"]
            - float(policy["ssim"]) * test["pred_ssim_bad_prob"]
            + float(policy["alpha"]) * test["alpha_float"]
        )
        test.loc[test["variant"] == "A0", "policy_score"] = 0.0
        for _, cand in test.groupby(["image_id", "fold", "seed"], sort=False):
            non_a0 = cand[cand["variant"] != "A0"]
            if len(non_a0) and non_a0["policy_score"].max() >= float(policy["min_score"]):
                chosen = non_a0.sort_values(["policy_score", "pred_gain", "alpha_float"], ascending=False).iloc[0]
            else:
                chosen = cand[cand["variant"] == "A0"].iloc[0]
            out = chosen.to_dict()
            out["outer_fold"] = outer_key[0]
            out["outer_seed"] = outer_key[1]
            selected.append(out)
    return selected


def aggregate(selected_rows: list[dict[str, Any]], feature_group: str, bank_name: str, policy_name: str) -> dict[str, Any]:
    metrics = summarize(selected_rows)
    by_outer: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in selected_rows:
        by_outer[(str(row.get("fold")), str(row.get("seed")))].append(row)
    outer_worst = [finite_float(summarize(rows).get("worst_per_600")) for rows in by_outer.values()]
    metrics["max_outer_worst_per_600"] = max(outer_worst) if outer_worst else metrics.get("worst_per_600")
    checks = gate_checks(metrics)
    metrics.update({
        "feature_group": feature_group,
        "action_bank": bank_name,
        "policy_name": policy_name,
        "model_type": "nested_ridge_logistic_high_positive",
        "outer_groups": len(by_outer),
        "strict_gate_checks": checks,
        "strict_gate_pass": all(checks.values()),
        "intervention_rate": sum(str(row.get("variant")) != "A0" for row in selected_rows) / len(selected_rows),
        "positive_action_rate": sum(finite_float(row.get("dPSNR")) > 0 for row in selected_rows) / len(selected_rows),
        "mean_alpha": float(np.mean([finite_float(row.get("alpha")) for row in selected_rows])),
        "score": finite_float(metrics.get("mean_dPSNR")) + 0.5 * finite_float(metrics.get("hard_bottom25_dPSNR")) + 0.05 * finite_float(metrics.get("positive_ratio")) - 0.005 * finite_float(metrics.get("worst_per_600")),
    })
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--single_actions_csv", required=True, type=Path)
    parser.add_argument("--feature_action_table_csv", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--output_prefix", default="v37_d3_highpos")
    parser.add_argument("--include_run_substring", default="quick5full")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    single = pd.read_csv(args.single_actions_csv)
    features = pd.read_csv(args.feature_action_table_csv)
    df = build_action_frame(single, features, args.include_run_substring)
    groups = feature_groups(set(df.columns))

    selected_all: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    for group_name, cols in groups.items():
        for bank_name in BANKS:
            for policy in POLICIES:
                selected = select_policy_rows(df, cols, bank_name, policy)
                for row in selected:
                    row["feature_group"] = group_name
                    row["action_bank"] = bank_name
                    row["policy_name"] = policy["name"]
                selected_all.extend(selected)
                aggregate_rows.append(aggregate(selected, group_name, bank_name, policy["name"]))

    aggregate_rows.sort(key=lambda row: (bool(row["strict_gate_pass"]), finite_float(row["score"])), reverse=True)
    selected_path = args.output_dir / f"{args.output_prefix}_policy_selected_actions.csv"
    aggregate_path = args.output_dir / f"{args.output_prefix}_policy_aggregate.csv"
    nested_path = args.output_dir / f"{args.output_prefix}_policy_nested_report.csv"
    summary_path = args.output_dir / f"{args.output_prefix}_summary.json"
    write_csv(selected_path, selected_all)
    write_csv(aggregate_path, aggregate_rows)
    write_csv(nested_path, aggregate_rows)
    strict_rows = [row for row in aggregate_rows if row["strict_gate_pass"]]
    summary = {
        "route": "DTA-v3.7 U-TQS-Mix",
        "phase": "D4_d3_high_positive_deployable_policy",
        "single_actions_csv": str(args.single_actions_csv),
        "feature_action_table_csv": str(args.feature_action_table_csv),
        "output_prefix": args.output_prefix,
        "rows": int(len(df)),
        "image_groups": int(df[["image_id", "fold", "seed"]].drop_duplicates().shape[0]),
        "aggregate_csv": str(aggregate_path),
        "selected_actions_csv": str(selected_path),
        "strict_pass_count": len(strict_rows),
        "best_row": aggregate_rows[0] if aggregate_rows else {},
        "decision": "D4_HIGH_POSITIVE_POLICY_STRICT_PASS" if strict_rows else "D4_HIGH_POSITIVE_POLICY_STRICT_FAIL",
        "locked_test_touched": False,
        "strict_gates": STRICT_GATES,
        "leakage_note": "deployable groups exclude trans_gt; diagnostic_with_trans_gt is reported separately and not promotion-deployable.",
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "DTA_V3_7_D4_HIGH_POSITIVE_POLICY_OK "
        f"rows={len(df)} aggregate={len(aggregate_rows)} strict_pass={len(strict_rows)} "
        f"decision={summary['decision']}"
    )


if __name__ == "__main__":
    main()
