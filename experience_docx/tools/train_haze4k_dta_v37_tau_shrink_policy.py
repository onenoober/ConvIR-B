#!/usr/bin/env python3
"""Train a staged DTA-v3.7 TAU soft-shrink policy on D1 screen evidence.

This is a train-derived D2 diagnostic. It consumes the D1 TAU action table,
builds an A0-preserving action bank with alpha-shrunk candidates, and evaluates
a nested deployable gain-risk policy. The locked Haze4K test is not used.
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
    finite_float,
    gate_checks,
    read_csv_rows,
    summarize_policy,
    write_csv,
)


DEFAULT_VARIANTS = (
    "u1_tau_l1_s004_g025_a006",
    "u2_tau_l3_s004_g015_a006",
    "u3_tau_l2_s002_g025_a006",
)

VARIANT_LABEL = {
    "u1_tau_l1_s004_g025_a006": "U1_TAU_high_gain",
    "u2_tau_l3_s004_g015_a006": "U2_TAU_balanced",
    "u3_tau_l2_s002_g025_a006": "U3_TAU_tail_safe",
}

VARIANT_RANK = {
    "u1_tau_l1_s004_g025_a006": 1.00,
    "u2_tau_l3_s004_g015_a006": 0.80,
    "u3_tau_l2_s002_g025_a006": 0.60,
}

ACTION_BANKS = {
    "full": [1.0],
    "shrink": [0.25, 0.50, 0.75, 1.0],
    "micro_shrink": [0.10, 0.25, 0.50, 0.75, 1.0],
    "tiny_micro_shrink": [0.05, 0.10, 0.25, 0.50, 0.75, 1.0],
}

FEATURE_GROUPS = {
    "Q_input_proxy": ("input_",),
    "Q_enriched_quality": ("input_", "q_", "dark_", "edge_", "texture_", "sky_", "highlight_", "color_"),
    "T_pred": ("dta_t_pred_",),
    "A_airlight": ("airlight_fallback", "dta_airlight_pred_"),
    "U_uncertainty_conf": ("dta_t_uncertainty", "dta_airlight_uncertainty", "dta_stage2_conf", "dta_stage3_conf"),
    "TAU_core": ("dta_t_pred_", "dta_t_uncertainty", "dta_airlight_pred_", "dta_airlight_uncertainty"),
    "FDF_action_stats": ("dta_stage", "dta_final", "dta_depth_mask", "dta_depth_delta", "dta_j_phys"),
    "deployable_TQAU_action_all": (
        "input_",
        "q_",
        "dark_",
        "edge_",
        "texture_",
        "sky_",
        "highlight_",
        "color_",
        "depth_",
        "airlight_fallback",
        "dta_",
    ),
    "diagnostic_with_trans_gt": (
        "input_",
        "q_",
        "dark_",
        "edge_",
        "texture_",
        "sky_",
        "highlight_",
        "color_",
        "depth_",
        "airlight_fallback",
        "dta_",
        "trans_gt_",
    ),
}

NON_DEPLOYABLE_PREFIXES = ("trans_gt_",)

ACTION_META = [
    "action_alpha",
    "action_is_a0",
    "action_is_u1",
    "action_is_u2",
    "action_is_u3",
    "action_strength_rank",
]


def parse_csv_set(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def key_for(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row.get("image_id")), str(row.get("fold")), str(row.get("seed")))


def row_group(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("fold")), str(row.get("seed")))


def filter_rows(
    rows: list[dict[str, Any]],
    variants: set[str],
    folds: set[str],
    seeds: set[str],
) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if str(row.get("variant")) in variants
        and str(row.get("fold")) in folds
        and str(row.get("seed")) in seeds
    ]


def utility_target(dpsnr: float, dssim: float) -> float:
    severe = max(0.0, -0.20 - dpsnr)
    strong = max(0.0, -0.05 - dpsnr)
    ssim_bad = max(0.0, -0.000005 - dssim)
    return dpsnr + 0.18 * max(dpsnr, 0.0) - 3.5 * severe - 0.7 * strong - 140.0 * ssim_bad


def policy_score(metrics: dict[str, Any]) -> float:
    return (
        finite_float(metrics.get("mean_dPSNR"), -1.0)
        + 0.25 * finite_float(metrics.get("hard_bottom25_dPSNR"), -1.0)
        + 0.20 * finite_float(metrics.get("positive_ratio"), 0.0)
        + 0.10 * finite_float(metrics.get("true_vs_zero"), -1.0)
        - 0.0015 * finite_float(metrics.get("worst_per_600"), 999.0)
        - 0.0004 * finite_float(metrics.get("strong_per_600"), 999.0)
    )


def summarize_with_outer(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = summarize_policy(rows)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row_group(row)].append(row)
    worsts = [finite_float(summarize_policy(items).get("worst_per_600"), 0.0) for items in grouped.values()]
    metrics["fold_seed_groups"] = len(grouped)
    metrics["max_outer_worst_per_600"] = max(worsts) if worsts else finite_float(metrics.get("worst_per_600"), 0.0)
    checks = gate_checks(metrics)
    metrics["strict_gate_pass"] = all(checks.values())
    metrics["strict_gate_checks"] = checks
    metrics["score"] = policy_score(metrics)
    return metrics


def load_and_join_features(action_table: Path, feature_table: Path | None) -> list[dict[str, Any]]:
    rows = read_csv_rows(action_table)
    if feature_table and feature_table.exists():
        feature_rows = {str(row["image_id"]): row for row in read_csv_rows(feature_table)}
        for row in rows:
            row.update({k: v for k, v in feature_rows.get(str(row.get("image_id")), {}).items() if k != "image_id"})
    return rows


def feature_columns(rows: list[dict[str, Any]], group: str) -> list[str]:
    prefixes = FEATURE_GROUPS[group]
    cols = sorted({key for row in rows for key in row if any(key.startswith(prefix) for prefix in prefixes)})
    if not group.startswith("diagnostic"):
        cols = [col for col in cols if not col.startswith(NON_DEPLOYABLE_PREFIXES)]
    numeric = []
    for col in cols:
        vals = [finite_float(row.get(col)) for row in rows]
        finite = [value for value in vals if math.isfinite(value)]
        if len(finite) >= max(20, len(rows) // 80) and len({round(value, 12) for value in finite}) >= 2:
            numeric.append(col)
    return numeric


def rows_by_key(rows: list[dict[str, Any]], variants: set[str]) -> dict[tuple[str, str, str], dict[str, dict[str, Any]]]:
    out: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        variant = str(row.get("variant"))
        if variant in variants:
            out[key_for(row)][variant] = row
    return out


def make_candidate(
    row: dict[str, Any],
    feature_cols: list[str],
    variant: str,
    alpha: float,
) -> dict[str, Any]:
    dpsnr = alpha * finite_float(row.get("dPSNR"), 0.0)
    dssim = alpha * finite_float(row.get("dSSIM"), 0.0)
    cand: dict[str, Any] = {
        "image_id": row.get("image_id"),
        "fold": row.get("fold"),
        "seed": row.get("seed"),
        "A0_PSNR": row.get("A0_PSNR"),
        "variant": variant,
        "action_label": f"{VARIANT_LABEL.get(variant, variant)}@{alpha:g}",
        "action_alpha": alpha,
        "action_is_a0": 0.0,
        "action_is_u1": 1.0 if variant == "u1_tau_l1_s004_g025_a006" else 0.0,
        "action_is_u2": 1.0 if variant == "u2_tau_l3_s004_g015_a006" else 0.0,
        "action_is_u3": 1.0 if variant == "u3_tau_l2_s002_g025_a006" else 0.0,
        "action_strength_rank": alpha * VARIANT_RANK.get(variant, 0.5),
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
        if col.startswith("dta_") and any(token in col for token in ("action", "delta", "gamma", "beta")):
            value *= alpha
        cand[col] = value
    return cand


def build_action_candidates(
    rows: list[dict[str, Any]],
    group: str,
    variants: set[str],
    alphas: list[float],
) -> tuple[list[dict[str, Any]], list[str]]:
    feature_cols = feature_columns(rows, group)
    keyed = rows_by_key(rows, variants)
    candidates: list[dict[str, Any]] = []
    for _, by_variant in keyed.items():
        if not by_variant:
            continue
        sample = next(iter(by_variant.values()))
        base: dict[str, Any] = {
            "image_id": sample.get("image_id"),
            "fold": sample.get("fold"),
            "seed": sample.get("seed"),
            "A0_PSNR": sample.get("A0_PSNR"),
            "variant": "A0",
            "action_label": "A0@0",
            "action_alpha": 0.0,
            "action_is_a0": 1.0,
            "action_is_u1": 0.0,
            "action_is_u2": 0.0,
            "action_is_u3": 0.0,
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
        for variant in sorted(variants):
            row = by_variant.get(variant)
            if row is None:
                continue
            for alpha in alphas:
                candidates.append(make_candidate(row, feature_cols, variant, alpha))
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
        pred_pos = float(np.clip(preds["positive_target"][idx], 0.0, 1.0))
        pred_sev = float(np.clip(preds["severe_target"][idx], 0.0, 1.0))
        pred_strong = float(np.clip(preds["strong_target"][idx], 0.0, 1.0))
        pred_ssim_bad = float(np.clip(preds["ssim_bad_target"][idx], 0.0, 1.0))
        pred_utility = (
            float(preds["utility_target"][idx])
            + 0.05 * pred_pos
            - 0.25 * pred_sev
            - 0.08 * pred_strong
            - 0.03 * pred_ssim_bad
        )
        out.append({
            "pred_utility": pred_utility,
            "pred_dPSNR": float(preds["dPSNR"][idx]),
            "pred_dSSIM": float(preds["dSSIM"][idx]),
            "pred_positive": pred_pos,
            "pred_severe": pred_sev,
            "pred_strong": pred_strong,
            "pred_ssim_bad": pred_ssim_bad,
        })
    return out


def choose_policy(
    candidates: list[dict[str, Any]],
    preds: list[dict[str, float]],
    risk_max: float,
    strong_max: float,
    utility_min: float,
) -> list[dict[str, Any]]:
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
            if row.get("variant") == "A0" or (
                pred["pred_severe"] <= risk_max
                and pred["pred_strong"] <= strong_max
                and pred["pred_utility"] >= utility_min
            ):
                valid.append((row, pred))
        if not valid and fallback is not None:
            valid = [fallback]
        valid.sort(
            key=lambda item: (
                item[1]["pred_utility"],
                item[1]["pred_dPSNR"],
                -finite_float(item[0].get("action_alpha"), 0.0),
            ),
            reverse=True,
        )
        row, pred = valid[0]
        chosen.append({**row, **pred})
    return chosen


def calibrate_thresholds(candidates: list[dict[str, Any]], preds: list[dict[str, float]]) -> dict[str, Any]:
    risk_grid = [0.00, 0.002, 0.005, 0.010, 0.020, 0.050, 0.100, 0.200, 1.000]
    strong_grid = [0.00, 0.010, 0.030, 0.050, 0.100, 0.200, 1.000]
    utility_grid = [-0.050, -0.020, -0.005, 0.0, 0.005, 0.020, 0.050]
    rows = []
    for risk_max in risk_grid:
        for strong_max in strong_grid:
            for utility_min in utility_grid:
                chosen = choose_policy(candidates, preds, risk_max, strong_max, utility_min)
                metrics = summarize_with_outer(chosen)
                metrics.update({
                    "risk_max": risk_max,
                    "strong_max": strong_max,
                    "utility_min": utility_min,
                })
                rows.append(metrics)
    rows.sort(key=lambda row: (bool(row.get("strict_gate_pass")), finite_float(row.get("score"), -999.0)), reverse=True)
    return rows[0]


def oracle_rows(candidates: list[dict[str, Any]], utility_mode: str) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_key[key_for(row)].append(row)
    chosen = []
    for items in by_key.values():
        if utility_mode == "max_dpsnr":
            items.sort(key=lambda row: (finite_float(row.get("dPSNR"), -999.0), finite_float(row.get("dSSIM"), -999.0)), reverse=True)
        elif utility_mode == "tail_averse":
            items.sort(key=lambda row: (finite_float(row.get("utility_target"), -999.0), finite_float(row.get("dPSNR"), -999.0)), reverse=True)
        elif utility_mode == "positive_guard":
            safe = [row for row in items if finite_float(row.get("dPSNR"), 0.0) >= -0.02 and finite_float(row.get("dSSIM"), 0.0) >= -0.000005]
            items = safe if safe else [row for row in items if row.get("variant") == "A0"]
            items.sort(key=lambda row: finite_float(row.get("dPSNR"), -999.0), reverse=True)
        else:
            raise ValueError(f"unknown utility_mode={utility_mode}")
        chosen.append(items[0])
    return chosen


def action_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "chosen_action_counts": dict(Counter(str(row.get("action_label")) for row in rows)),
        "chosen_variant_counts": dict(Counter(str(row.get("variant")) for row in rows)),
        "intervention_rate": sum(str(row.get("variant")) != "A0" for row in rows) / len(rows) if rows else float("nan"),
        "mean_chosen_alpha": statistics.mean(finite_float(row.get("action_alpha"), 0.0) for row in rows) if rows else float("nan"),
    }


def run_oracle_grid(rows: list[dict[str, Any]], variants: set[str]) -> list[dict[str, Any]]:
    grid = []
    for bank_name, alphas in ACTION_BANKS.items():
        candidates, _ = build_action_candidates(rows, "deployable_TQAU_action_all", variants, alphas)
        for mode in ("max_dpsnr", "tail_averse", "positive_guard"):
            chosen = oracle_rows(candidates, mode)
            metrics = summarize_with_outer(chosen)
            metrics.update({
                "bank_name": bank_name,
                "utility_mode": mode,
                **action_counts(chosen),
            })
            grid.append(metrics)
    return sorted(grid, key=lambda row: (bool(row.get("strict_gate_pass")), finite_float(row.get("score"), -999.0)), reverse=True)


def aggregate_reports(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["feature_group"]), str(row["model_type"]), str(row["action_bank"]))].append(row)
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
    for (group, model, bank), items in sorted(grouped.items()):
        row = {
            "feature_group": group,
            "model_type": model,
            "action_bank": bank,
            "outer_folds": len(items),
        }
        for key in metric_keys:
            row[key] = statistics.mean(finite_float(item.get(key), 0.0) for item in items)
        row["max_outer_worst_per_600"] = max(finite_float(item.get("worst_per_600"), 0.0) for item in items)
        checks = gate_checks(row)
        row["strict_gate_pass"] = all(checks.values())
        row["strict_gate_checks"] = checks
        row["score"] = policy_score(row)
        out.append(row)
    return sorted(out, key=lambda row: (bool(row.get("strict_gate_pass")), finite_float(row.get("score"), -999.0)), reverse=True)


def nested_train_eval(
    rows: list[dict[str, Any]],
    groups: list[str],
    variants: set[str],
    action_bank: str,
) -> dict[str, Any]:
    alphas = ACTION_BANKS[action_bank]
    all_reports = []
    action_rows = []
    folds = sorted({str(row.get("fold")) for row in rows})
    for group in groups:
        candidates, features = build_action_candidates(rows, group, variants, alphas)
        for outer_fold in folds:
            eval_cands = [row for row in candidates if str(row.get("fold")) == outer_fold]
            train_pool = [row for row in candidates if str(row.get("fold")) != outer_fold]
            train_seeds = sorted({str(row.get("seed")) for row in train_pool})
            if len(train_seeds) >= 2:
                calibration_seed = train_seeds[-1]
                model_rows = [row for row in train_pool if str(row.get("seed")) != calibration_seed]
                calib_rows = [row for row in train_pool if str(row.get("seed")) == calibration_seed]
            else:
                calibration_seed = train_seeds[0] if train_seeds else "none"
                split = max(1, int(0.7 * len(train_pool)))
                model_rows = train_pool[:split]
                calib_rows = train_pool[split:] or train_pool[:]
            bundle = fit_bundle(model_rows, features)
            calib_preds = predict_bundle(bundle, calib_rows)
            best = calibrate_thresholds(calib_rows, calib_preds)
            eval_preds = predict_bundle(bundle, eval_cands)
            chosen = choose_policy(
                eval_cands,
                eval_preds,
                finite_float(best.get("risk_max"), 1.0),
                finite_float(best.get("strong_max"), 1.0),
                finite_float(best.get("utility_min"), -1.0),
            )
            metrics = summarize_with_outer(chosen)
            counts = action_counts(chosen)
            report = {
                "feature_group": group,
                "model_type": "ridge_tqs_tau_shrink",
                "action_bank": action_bank,
                "outer_fold": outer_fold,
                "calibration_seed": calibration_seed,
                "feature_count": len(features),
                "model_rows": len(model_rows),
                "calibration_rows": len(calib_rows),
                "eval_candidates": len(eval_cands),
                "eval_images": len(chosen),
                "risk_max": best.get("risk_max"),
                "strong_max": best.get("strong_max"),
                "utility_min": best.get("utility_min"),
                "calibration_strict_pass": best.get("strict_gate_pass"),
                **counts,
                **metrics,
            }
            all_reports.append(report)
            for row in chosen:
                action_rows.append({
                    "feature_group": group,
                    "model_type": "ridge_tqs_tau_shrink",
                    "action_bank": action_bank,
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
    parser.add_argument("--image_feature_table", type=Path, default=None)
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--folds", default="0,1")
    parser.add_argument("--seeds", default="3407,3411")
    parser.add_argument("--feature_groups", default="Q_input_proxy,Q_enriched_quality,T_pred,TAU_core,U_uncertainty_conf,FDF_action_stats,deployable_TQAU_action_all,diagnostic_with_trans_gt")
    parser.add_argument("--action_bank", default="micro_shrink", choices=sorted(ACTION_BANKS))
    parser.add_argument("--output_prefix", default="v37_tau_shrink")
    args = parser.parse_args()

    variants = parse_csv_set(args.variants)
    rows = load_and_join_features(args.input_action_table, args.image_feature_table)
    rows = filter_rows(rows, variants, parse_csv_set(args.folds), parse_csv_set(args.seeds))
    groups = [item.strip() for item in args.feature_groups.split(",") if item.strip()]
    oracle_grid = run_oracle_grid(rows, variants)
    result = nested_train_eval(rows, groups, variants, args.action_bank)

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    prefix = args.output_prefix
    write_csv(out / f"{prefix}_oracle_grid.csv", oracle_grid)
    write_csv(out / f"{prefix}_policy_nested_report.csv", result["outer_reports"])
    write_csv(out / f"{prefix}_policy_aggregate.csv", result["aggregate"])
    write_csv(out / f"{prefix}_policy_action_table.csv", result["actions"])
    summary = {
        "route": "DTA-v3.7 U-TQS-Mix",
        "phase": "D2_tau_shrink_policy_stage",
        "rows": len(rows),
        "variants": sorted(variants),
        "folds": sorted(parse_csv_set(args.folds)),
        "seeds": sorted(parse_csv_set(args.seeds)),
        "feature_groups": groups,
        "action_bank": args.action_bank,
        "image_feature_table": str(args.image_feature_table) if args.image_feature_table else None,
        "output_prefix": args.output_prefix,
        "strict_gates": STRICT_GATES,
        "best_oracle": oracle_grid[0] if oracle_grid else None,
        "best_aggregate": result["aggregate"][0] if result["aggregate"] else None,
        "oracle_strict_pass_count": sum(1 for row in oracle_grid if row.get("strict_gate_pass")),
        "policy_strict_pass_count": sum(1 for row in result["aggregate"] if row.get("strict_gate_pass")),
        "decision": "D2_TAU_SHRINK_POLICY_STRICT_PASS" if any(row.get("strict_gate_pass") for row in result["aggregate"]) else "D2_TAU_SHRINK_POLICY_STRICT_FAIL",
        "locked_test_touched": False,
        "leakage_note": "diagnostic_with_trans_gt is not deployable; deployable groups exclude trans_gt by construction.",
        "proxy_note": "D2 table shrink uses alpha-scaled D1 deltas; a passing row still needs real rendered blend verification before formal claims.",
    }
    (out / f"{prefix}_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "DTA_V3_7_D2_TAU_SHRINK_POLICY_OK "
        f"rows={len(rows)} oracle_strict={summary['oracle_strict_pass_count']} "
        f"policy_strict={summary['policy_strict_pass_count']} decision={summary['decision']}"
    )


if __name__ == "__main__":
    main()
