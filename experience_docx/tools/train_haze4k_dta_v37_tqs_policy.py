#!/usr/bin/env python3
"""Train nested table-only TQS gain-risk policies for DTA-v3.7.

This is Phase B: no image-model training and no locked-test access. It turns the
v3.6 formal OOF action table into a soft action-bank policy problem and evaluates
nested deployable gain-risk predictors on outer folds.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from select_haze4k_dta_v37_u_tqs_mix_phase_a import (
    STRICT_GATES,
    VARIANT_LABEL,
    VARIANT_ORDER,
    finite_float,
    gate_checks,
    read_csv_rows,
    summarize_policy,
    write_csv,
)

ALPHAS = {
    "l2_fdf_lite_s002_g025_bm2": [0.25, 0.50, 1.00],
    "l3_fdf_lite_s004_g015_bm2": [0.25, 0.50, 0.75, 1.00],
    "l1_fdf_lite_s004_g025_bm2": [0.25, 0.50, 0.75, 1.00],
}

FEATURE_GROUPS = {
    "Q_input_proxy": ("input_",),
    "D_depth": ("depth_",),
    "T_pred": ("dta_t_pred_",),
    "A_airlight_proxy": ("airlight_fallback", "airlight_proxy"),
    "U_uncertainty_conf": ("dta_t_uncertainty", "dta_stage2_conf", "dta_stage3_conf"),
    "FDF_action_stats": ("dta_stage", "dta_final", "dta_depth_mask", "dta_depth_delta", "dta_j_phys"),
    "deployable_TQAU_action_all": ("input_", "depth_", "airlight_fallback", "airlight_proxy", "dta_"),
    "diagnostic_with_trans_gt": ("input_", "depth_", "airlight_fallback", "airlight_proxy", "dta_", "trans_gt_"),
}

NON_FEATURE_PREFIXES = ("trans_gt_",)
ACTION_META = [
    "action_alpha",
    "action_is_a0",
    "action_is_l1",
    "action_is_l2",
    "action_is_l3",
    "action_strength_rank",
]


def key_for(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row.get("image_id")), str(row.get("fold")), str(row.get("seed")))


def utility_target(dpsnr: float, dssim: float) -> float:
    severe = max(0.0, -0.20 - dpsnr)
    strong = max(0.0, -0.05 - dpsnr)
    ssim_bad = max(0.0, -0.000005 - dssim)
    return dpsnr + 0.15 * max(dpsnr, 0.0) - 3.0 * severe - 0.6 * strong - 120.0 * ssim_bad


def base_feature_columns(rows: list[dict[str, Any]], group: str) -> list[str]:
    prefixes = FEATURE_GROUPS[group]
    cols = sorted({k for row in rows for k in row if any(k.startswith(p) for p in prefixes)})
    if not group.startswith("diagnostic"):
        cols = [c for c in cols if not c.startswith(NON_FEATURE_PREFIXES)]
    numeric = []
    for col in cols:
        vals = [finite_float(row.get(col)) for row in rows]
        finite = [v for v in vals if math.isfinite(v)]
        if len(finite) >= max(20, len(rows) // 50) and len({round(v, 12) for v in finite}) >= 2:
            numeric.append(col)
    return numeric


def rows_by_key(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, dict[str, Any]]]:
    out: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        variant = str(row.get("variant"))
        if variant in VARIANT_ORDER:
            out[key_for(row)][variant] = row
    return out


def build_action_candidates(rows: list[dict[str, Any]], group: str) -> tuple[list[dict[str, Any]], list[str]]:
    feature_cols = base_feature_columns(rows, group)
    keyed = rows_by_key(rows)
    candidates: list[dict[str, Any]] = []
    for key, by_variant in keyed.items():
        if not by_variant:
            continue
        sample = next(iter(by_variant.values()))
        base = {
            "image_id": sample.get("image_id"),
            "fold": sample.get("fold"),
            "seed": sample.get("seed"),
            "A0_PSNR": sample.get("A0_PSNR"),
            "variant": "A0",
            "action_label": "A0@0",
            "action_alpha": 0.0,
            "action_is_a0": 1.0,
            "action_is_l1": 0.0,
            "action_is_l2": 0.0,
            "action_is_l3": 0.0,
            "action_strength_rank": 0.0,
            "dPSNR": 0.0,
            "dSSIM": 0.0,
            "zero_delta_psnr": 0.0,
            "shuffle_delta_psnr": 0.0,
            "normal_delta_psnr": 0.0,
            "utility_target": 0.0,
            "positive_target": 0.0,
            "good_target": 0.0,
            "severe_target": 0.0,
            "strong_target": 0.0,
            "ssim_bad_target": 0.0,
        }
        for col in feature_cols:
            base[col] = sample.get(col)
        candidates.append(base)
        for variant in VARIANT_ORDER:
            row = by_variant.get(variant)
            if row is None:
                continue
            for alpha in ALPHAS[variant]:
                dpsnr = alpha * finite_float(row.get("dPSNR"), 0.0)
                dssim = alpha * finite_float(row.get("dSSIM"), 0.0)
                cand = {
                    "image_id": row.get("image_id"),
                    "fold": row.get("fold"),
                    "seed": row.get("seed"),
                    "A0_PSNR": row.get("A0_PSNR"),
                    "variant": variant,
                    "action_label": f"{VARIANT_LABEL.get(variant, variant)}@{alpha:g}",
                    "action_alpha": alpha,
                    "action_is_a0": 0.0,
                    "action_is_l1": 1.0 if variant == "l1_fdf_lite_s004_g025_bm2" else 0.0,
                    "action_is_l2": 1.0 if variant == "l2_fdf_lite_s002_g025_bm2" else 0.0,
                    "action_is_l3": 1.0 if variant == "l3_fdf_lite_s004_g015_bm2" else 0.0,
                    "action_strength_rank": alpha * (1.0 if variant != "l2_fdf_lite_s002_g025_bm2" else 0.75),
                    "dPSNR": dpsnr,
                    "dSSIM": dssim,
                    "zero_delta_psnr": alpha * finite_float(row.get("zero_delta_psnr"), 0.0),
                    "shuffle_delta_psnr": alpha * finite_float(row.get("shuffle_delta_psnr"), 0.0),
                    "normal_delta_psnr": alpha * finite_float(row.get("normal_delta_psnr"), 0.0),
                    "utility_target": utility_target(dpsnr, dssim),
                    "positive_target": 1.0 if dpsnr > 0.0 else 0.0,
                    "good_target": 1.0 if dpsnr > 0.02 and dssim >= -0.000005 else 0.0,
                    "severe_target": 1.0 if dpsnr <= -0.20 else 0.0,
                    "strong_target": 1.0 if dpsnr <= -0.05 else 0.0,
                    "ssim_bad_target": 1.0 if dssim < -0.000005 else 0.0,
                }
                for col in feature_cols:
                    value = finite_float(row.get(col))
                    # Scale action-amplitude features by alpha to make shrink visible.
                    if col.startswith("dta_") and any(token in col for token in ("action", "delta", "gamma", "beta")):
                        value *= alpha
                    cand[col] = value
                candidates.append(cand)
    return candidates, feature_cols + ACTION_META


class RidgeModel:
    def __init__(self, features: list[str], l2: float = 1.0) -> None:
        self.features = features
        self.l2 = l2
        self.medians: dict[str, float] = {}
        self.means: dict[str, float] = {}
        self.stds: dict[str, float] = {}
        self.weights: np.ndarray | None = None

    def _fit_scaler(self, rows: list[dict[str, Any]]) -> None:
        for feature in self.features:
            vals = np.array([finite_float(row.get(feature)) for row in rows], dtype=float)
            finite = vals[np.isfinite(vals)]
            median = float(np.median(finite)) if finite.size else 0.0
            vals = np.where(np.isfinite(vals), vals, median)
            mean = float(np.mean(vals)) if vals.size else 0.0
            std = float(np.std(vals)) if vals.size else 1.0
            self.medians[feature] = median
            self.means[feature] = mean
            self.stds[feature] = std if std >= 1e-9 else 1.0

    def _x(self, rows: list[dict[str, Any]]) -> np.ndarray:
        cols = []
        for feature in self.features:
            vals = np.array([finite_float(row.get(feature)) for row in rows], dtype=float)
            vals = np.where(np.isfinite(vals), vals, self.medians.get(feature, 0.0))
            vals = (vals - self.means.get(feature, 0.0)) / self.stds.get(feature, 1.0)
            cols.append(vals)
        if not cols:
            return np.ones((len(rows), 1), dtype=float)
        x = np.vstack(cols).T
        return np.hstack([np.ones((len(rows), 1), dtype=float), x])

    def fit(self, rows: list[dict[str, Any]], target: str) -> None:
        self._fit_scaler(rows)
        x = self._x(rows)
        y = np.array([finite_float(row.get(target), 0.0) for row in rows], dtype=float)
        reg = self.l2 * np.eye(x.shape[1], dtype=float)
        reg[0, 0] = 0.0
        try:
            self.weights = np.linalg.solve(x.T @ x + reg, x.T @ y)
        except np.linalg.LinAlgError:
            self.weights = np.linalg.pinv(x.T @ x + reg) @ x.T @ y

    def predict(self, rows: list[dict[str, Any]]) -> np.ndarray:
        if self.weights is None:
            return np.zeros(len(rows), dtype=float)
        return self._x(rows) @ self.weights


def fit_bundle(rows: list[dict[str, Any]], features: list[str]) -> dict[str, RidgeModel]:
    targets = ["utility_target", "dPSNR", "dSSIM", "positive_target", "severe_target", "strong_target", "ssim_bad_target"]
    bundle = {}
    for target in targets:
        model = RidgeModel(features, l2=2.0 if target.endswith("target") else 1.0)
        model.fit(rows, target)
        bundle[target] = model
    return bundle


def predict_bundle(bundle: dict[str, RidgeModel], rows: list[dict[str, Any]]) -> list[dict[str, float]]:
    preds = {name: model.predict(rows) for name, model in bundle.items()}
    out = []
    for idx in range(len(rows)):
        pred_dpsnr = float(preds["dPSNR"][idx])
        pred_dssim = float(preds["dSSIM"][idx])
        pred_pos = float(np.clip(preds["positive_target"][idx], 0.0, 1.0))
        pred_sev = float(np.clip(preds["severe_target"][idx], 0.0, 1.0))
        pred_strong = float(np.clip(preds["strong_target"][idx], 0.0, 1.0))
        pred_ssim_bad = float(np.clip(preds["ssim_bad_target"][idx], 0.0, 1.0))
        pred_utility = float(preds["utility_target"][idx]) + 0.05 * pred_pos - 0.25 * pred_sev - 0.08 * pred_strong - 0.03 * pred_ssim_bad
        out.append({
            "pred_utility": pred_utility,
            "pred_dPSNR": pred_dpsnr,
            "pred_dSSIM": pred_dssim,
            "pred_positive": pred_pos,
            "pred_severe": pred_sev,
            "pred_strong": pred_strong,
            "pred_ssim_bad": pred_ssim_bad,
        })
    return out


def choose_policy(candidates: list[dict[str, Any]], preds: list[dict[str, float]], risk_max: float, utility_min: float) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, str], list[tuple[dict[str, Any], dict[str, float]]]] = defaultdict(list)
    for row, pred in zip(candidates, preds):
        by_key[key_for(row)].append((row, pred))
    chosen = []
    for items in by_key.values():
        valid = []
        fallback = None
        for row, pred in items:
            if row.get("variant") == "A0":
                fallback = (row, pred)
            if row.get("variant") == "A0" or (pred["pred_severe"] <= risk_max and pred["pred_utility"] >= utility_min):
                valid.append((row, pred))
        if not valid and fallback is not None:
            valid = [fallback]
        valid.sort(key=lambda item: (item[1]["pred_utility"], item[1]["pred_dPSNR"], -finite_float(item[0].get("action_alpha"), 0.0)), reverse=True)
        row, pred = valid[0]
        chosen.append({**row, **pred})
    return chosen


def policy_score(metrics: dict[str, Any]) -> float:
    return (
        finite_float(metrics.get("mean_dPSNR"), -1.0)
        + 0.25 * finite_float(metrics.get("hard_bottom25_dPSNR"), -1.0)
        + 0.20 * finite_float(metrics.get("positive_ratio"), 0.0)
        + 0.10 * finite_float(metrics.get("true_vs_zero"), -1.0)
        - 0.0015 * finite_float(metrics.get("worst_per_600"), 999.0)
        - 0.0004 * finite_float(metrics.get("strong_per_600"), 999.0)
    )


def calibrate_thresholds(candidates: list[dict[str, Any]], preds: list[dict[str, float]]) -> dict[str, Any]:
    risk_grid = [0.00, 0.002, 0.005, 0.010, 0.020, 0.050, 0.100, 0.200, 1.000]
    utility_grid = [-0.050, -0.020, -0.005, 0.0, 0.005, 0.020, 0.050]
    rows = []
    for risk_max in risk_grid:
        for utility_min in utility_grid:
            chosen = choose_policy(candidates, preds, risk_max, utility_min)
            metrics = summarize_policy(chosen)
            metrics["max_outer_worst_per_600"] = metrics.get("worst_per_600")
            checks = gate_checks(metrics)
            metrics.update({
                "risk_max": risk_max,
                "utility_min": utility_min,
                "strict_gate_pass": all(checks.values()),
                "strict_gate_checks": checks,
                "score": policy_score(metrics),
            })
            rows.append(metrics)
    rows.sort(key=lambda row: (bool(row.get("strict_gate_pass")), finite_float(row.get("score"), -999.0)), reverse=True)
    return rows[0]


def aggregate_reports(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["feature_group"]), str(row["model_type"]))].append(row)
    out = []
    metric_keys = [
        "coverage",
        "mean_dPSNR",
        "hard_bottom25_dPSNR",
        "easy_top25_dPSNR",
        "dSSIM",
        "positive_ratio",
        "worst_per_600",
        "strong_per_600",
        "true_vs_zero",
        "true_vs_shuffle",
        "true_vs_normal",
        "intervention_rate",
    ]
    for (group, model), items in sorted(grouped.items()):
        row = {"feature_group": group, "model_type": model, "outer_folds": len(items)}
        for key in metric_keys:
            row[key] = statistics.mean(finite_float(item.get(key), 0.0) for item in items)
        row["max_outer_worst_per_600"] = max(finite_float(item.get("worst_per_600"), 0.0) for item in items)
        checks = gate_checks(row)
        row["strict_gate_pass"] = all(checks.values())
        row["strict_gate_checks"] = checks
        row["score"] = policy_score(row)
        out.append(row)
    return sorted(out, key=lambda row: (bool(row.get("strict_gate_pass")), finite_float(row.get("score"), -999.0)), reverse=True)


def nested_train_eval(rows: list[dict[str, Any]], groups: list[str]) -> dict[str, Any]:
    all_reports = []
    action_rows = []
    folds = sorted({str(row.get("fold")) for row in rows})
    for group in groups:
        candidates, features = build_action_candidates(rows, group)
        for outer_fold in folds:
            eval_cands = [row for row in candidates if str(row.get("fold")) == outer_fold]
            train_pool = [row for row in candidates if str(row.get("fold")) != outer_fold]
            train_folds = sorted({str(row.get("fold")) for row in train_pool})
            calib_fold = train_folds[0]
            model_rows = [row for row in train_pool if str(row.get("fold")) != calib_fold]
            calib_rows = [row for row in train_pool if str(row.get("fold")) == calib_fold]
            bundle = fit_bundle(model_rows, features)
            calib_preds = predict_bundle(bundle, calib_rows)
            best = calibrate_thresholds(calib_rows, calib_preds)
            eval_preds = predict_bundle(bundle, eval_cands)
            chosen = choose_policy(eval_cands, eval_preds, finite_float(best.get("risk_max"), 1.0), finite_float(best.get("utility_min"), -1.0))
            metrics = summarize_policy(chosen)
            checks = gate_checks(metrics)
            action_counter = Counter(str(row.get("action_label")) for row in chosen)
            intervention = sum(str(row.get("variant")) != "A0" for row in chosen)
            report = {
                "feature_group": group,
                "model_type": "ridge_tqs",
                "outer_fold": outer_fold,
                "calibration_fold": calib_fold,
                "feature_count": len(features),
                "model_rows": len(model_rows),
                "calibration_rows": len(calib_rows),
                "eval_candidates": len(eval_cands),
                "eval_images": len(chosen),
                "risk_max": best.get("risk_max"),
                "utility_min": best.get("utility_min"),
                "calibration_strict_pass": best.get("strict_gate_pass"),
                "intervention_rate": intervention / len(chosen) if chosen else float("nan"),
                "chosen_action_counts": dict(action_counter),
                "strict_gate_pass": all(checks.values()),
                "strict_gate_checks": checks,
                **metrics,
            }
            all_reports.append(report)
            for row in chosen:
                action_rows.append({
                    "feature_group": group,
                    "model_type": "ridge_tqs",
                    "outer_fold": outer_fold,
                    "image_id": row.get("image_id"),
                    "fold": row.get("fold"),
                    "seed": row.get("seed"),
                    "chosen_action": row.get("action_label"),
                    "chosen_variant": row.get("variant"),
                    "chosen_alpha": row.get("action_alpha"),
                    "dPSNR": row.get("dPSNR"),
                    "dSSIM": row.get("dSSIM"),
                    "pred_utility": row.get("pred_utility"),
                    "pred_dPSNR": row.get("pred_dPSNR"),
                    "pred_positive": row.get("pred_positive"),
                    "pred_severe": row.get("pred_severe"),
                    "pred_strong": row.get("pred_strong"),
                })
    aggregate = aggregate_reports(all_reports)
    return {"outer_reports": all_reports, "aggregate": aggregate, "actions": action_rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_action_table", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--feature_groups", default="Q_input_proxy,T_pred,U_uncertainty_conf,FDF_action_stats,deployable_TQAU_action_all,diagnostic_with_trans_gt")
    args = parser.parse_args()

    rows = read_csv_rows(args.input_action_table)
    groups = [item.strip() for item in args.feature_groups.split(",") if item.strip()]
    result = nested_train_eval(rows, groups)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "v37_tqs_policy_nested_report.csv", result["outer_reports"])
    write_csv(out / "v37_tqs_policy_aggregate.csv", result["aggregate"])
    write_csv(out / "v37_tqs_policy_action_table.csv", result["actions"])
    write_csv(out / "v37_tqs_feature_group_ablation.csv", result["aggregate"])
    summary = {
        "route": "DTA-v3.7 U-TQS-Mix",
        "phase": "B_table_only_tqs_gain_risk_predictor",
        "rows": len(rows),
        "feature_groups": groups,
        "strict_gates": STRICT_GATES,
        "best_aggregate": result["aggregate"][0] if result["aggregate"] else None,
        "strict_pass_count": sum(1 for row in result["aggregate"] if row.get("strict_gate_pass")),
        "decision": "PHASE_B_TABLE_POLICY_STRICT_PASS" if any(row.get("strict_gate_pass") for row in result["aggregate"]) else "PHASE_B_TABLE_POLICY_STRICT_FAIL_NEEDS_FEATURE_ENRICHMENT_OR_REAL_BLEND",
        "leakage_note": "diagnostic_with_trans_gt is not deployable; deployable groups exclude trans_gt by construction.",
    }
    (out / "v37_tqs_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "DTA_V3_7_TQS_PHASE_B_OK "
        f"rows={len(rows)} groups={len(groups)} strict_pass={summary['strict_pass_count']} decision={summary['decision']}"
    )


if __name__ == "__main__":
    main()
