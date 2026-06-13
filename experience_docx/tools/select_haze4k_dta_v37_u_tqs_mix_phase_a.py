#!/usr/bin/env python3
"""DTA-v3.7 U-TQS-Mix Phase A table-only diagnostics.

This tool is intentionally no-training and train-derived only. It consumes the
formal DTA-v3.6 OOF action table and asks whether a soft action bank / shrink
mixture has enough oracle headroom before any new image-model training starts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

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

VARIANT_ORDER = [
    "l2_fdf_lite_s002_g025_bm2",
    "l3_fdf_lite_s004_g015_bm2",
    "l1_fdf_lite_s004_g025_bm2",
]

VARIANT_LABEL = {
    "l2_fdf_lite_s002_g025_bm2": "L2_tail_safe",
    "l3_fdf_lite_s004_g015_bm2": "L3_balanced",
    "l1_fdf_lite_s004_g025_bm2": "L1_high_gain",
}

NON_FEATURE_COLUMNS = {
    "image_id",
    "fold",
    "seed",
    "variant",
    "run_id",
    "failure_group",
    "A0_PSNR",
    "cand_PSNR",
    "dPSNR",
    "dSSIM",
    "zero_delta_psnr",
    "shuffle_delta_psnr",
    "normal_delta_psnr",
}

LEAKY_COLUMNS = {
    "A0_PSNR",
    "cand_PSNR",
    "dPSNR",
    "dSSIM",
    "zero_delta_psnr",
    "shuffle_delta_psnr",
    "normal_delta_psnr",
}

TARGETS = {
    "positive_action": lambda row: finite_float(row.get("dPSNR")) > 0.0,
    "good_action_gt_002": lambda row: finite_float(row.get("dPSNR")) > 0.02 and finite_float(row.get("dSSIM")) >= -0.000005,
    "strong_regression_le_m005": lambda row: finite_float(row.get("dPSNR")) <= -0.05,
    "severe_regression_le_m020": lambda row: finite_float(row.get("dPSNR")) <= -0.20,
    "ssim_regression": lambda row: finite_float(row.get("dSSIM")) < -0.000005,
}


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def safe_mean(values: Iterable[float], default: float = float("nan")) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return statistics.mean(vals) if vals else default


def percentile(values: Iterable[float], pct: float) -> float:
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


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            out: dict[str, Any] = {}
            for key, value in row.items():
                if key in {"image_id", "fold", "seed", "variant", "run_id", "failure_group", "candidate", "selector_type", "feature_group", "error_type", "accept"}:
                    out[key] = value
                else:
                    out[key] = finite_float(value)
            rows.append(out)
    return rows


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


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def key_for(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row.get("image_id")), str(row.get("fold")), str(row.get("seed")))


def summarize_policy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    deltas = [finite_float(row.get("dPSNR"), 0.0) for row in rows]
    ssim_deltas = [finite_float(row.get("dSSIM"), 0.0) for row in rows]
    original_psnr = [finite_float(row.get("A0_PSNR"), 0.0) for row in rows]
    sorted_idx = sorted(range(len(rows)), key=lambda idx: original_psnr[idx])
    bucket_count = max(1, len(rows) // 4)
    hard_idx = sorted_idx[:bucket_count]
    easy_idx = sorted_idx[-bucket_count:]
    strong_cut = percentile(original_psnr, 75.0)
    strong_idx = [idx for idx, psnr in enumerate(original_psnr) if psnr >= strong_cut]
    worst_count = sum(delta <= -0.20 for delta in deltas)
    strong_count = sum(delta <= -0.05 for delta in deltas)
    n = len(rows)

    def mean_at(indices: list[int]) -> float:
        return statistics.mean(deltas[idx] for idx in indices) if indices else float("nan")

    def surplus(control_key: str) -> float:
        return statistics.mean(
            finite_float(row.get("dPSNR"), 0.0) - finite_float(row.get(control_key), 0.0)
            for row in rows
        )

    return {
        "count": n,
        "coverage": 1.0,
        "mean_dPSNR": statistics.mean(deltas),
        "hard_bottom25_dPSNR": mean_at(hard_idx),
        "easy_top25_dPSNR": mean_at(easy_idx),
        "dSSIM": statistics.mean(ssim_deltas),
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
    max_outer = finite_float(metrics.get("max_outer_worst_per_600"), finite_float(metrics.get("worst_per_600"), 1e9))
    return {
        "coverage": finite_float(metrics.get("coverage"), 0.0) >= STRICT_GATES["coverage_min"],
        "mean": finite_float(metrics.get("mean_dPSNR"), -1e9) >= STRICT_GATES["mean_dpsnr_min"],
        "hard": finite_float(metrics.get("hard_bottom25_dPSNR"), -1e9) >= STRICT_GATES["hard_bottom25_min"],
        "dssim": finite_float(metrics.get("dSSIM"), -1e9) >= STRICT_GATES["dssim_min"],
        "positive_ratio": finite_float(metrics.get("positive_ratio"), 0.0) >= STRICT_GATES["positive_ratio_min"],
        "true_vs_zero": finite_float(metrics.get("true_vs_zero"), -1e9) >= STRICT_GATES["true_vs_zero_min"],
        "true_vs_shuffle": finite_float(metrics.get("true_vs_shuffle"), -1e9) >= STRICT_GATES["true_vs_shuffle_min"],
        "true_vs_normal": finite_float(metrics.get("true_vs_normal"), -1e9) >= STRICT_GATES["true_vs_normal_min"],
        "worst": finite_float(metrics.get("worst_per_600"), 1e9) <= STRICT_GATES["worst_per_600_max"],
        "max_outer_worst": max_outer <= STRICT_GATES["max_outer_worst_per_600_max"],
    }


def variant_rows(rows: list[dict[str, Any]], variant: str) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("variant") == variant or row.get("candidate") == variant]


def positive_loss_budget_report(
    action_rows: list[dict[str, Any]],
    selector_rows: list[dict[str, Any]],
    strict_positive_ratio: float,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    selector_by_candidate = {candidate: variant_rows(selector_rows, candidate) for candidate in VARIANT_ORDER}
    for variant in VARIANT_ORDER:
        rows = variant_rows(action_rows, variant)
        if not rows:
            continue
        n = len(rows)
        full_pos_count = sum(finite_float(row.get("dPSNR"), 0.0) > 0 for row in rows)
        full_pos_ratio = full_pos_count / n
        allowed_loss_count = max(0.0, (full_pos_ratio - strict_positive_ratio) * n)
        allowed_loss_per_600 = allowed_loss_count * 600.0 / n
        full_metrics = summarize_policy(rows)
        row = {
            "candidate": variant,
            "candidate_label": VARIANT_LABEL.get(variant, variant),
            "count": n,
            "full_positive_ratio": full_pos_ratio,
            "strict_positive_ratio_min": strict_positive_ratio,
            "allowed_positive_loss_count": allowed_loss_count,
            "allowed_positive_loss_per_600": allowed_loss_per_600,
            "full_mean_dPSNR": full_metrics.get("mean_dPSNR"),
            "full_hard_bottom25_dPSNR": full_metrics.get("hard_bottom25_dPSNR"),
            "full_worst_per_600": full_metrics.get("worst_per_600"),
        }
        sel = selector_by_candidate.get(variant, [])
        if sel:
            accepted = [as_bool(item.get("accept")) for item in sel]
            sel_deltas = [finite_float(item.get("dPSNR"), 0.0) if keep else 0.0 for item, keep in zip(sel, accepted)]
            selector_positive_ratio = sum(delta > 0 for delta in sel_deltas) / len(sel_deltas)
            false_reject_positive = sum((not keep) and finite_float(item.get("dPSNR"), 0.0) > 0 for item, keep in zip(sel, accepted))
            false_reject_good = sum((not keep) and finite_float(item.get("dPSNR"), 0.0) > 0.02 for item, keep in zip(sel, accepted))
            positive_loss_ratio = full_pos_ratio - selector_positive_ratio
            positive_loss_count = positive_loss_ratio * n
            row.update({
                "selector_positive_ratio": selector_positive_ratio,
                "selector_coverage": sum(accepted) / len(accepted),
                "false_reject_positive_count": false_reject_positive,
                "false_reject_good_count": false_reject_good,
                "positive_loss_count": positive_loss_count,
                "positive_loss_per_600": positive_loss_count * 600.0 / n,
                "positive_loss_budget_multiple": (positive_loss_count / allowed_loss_count) if allowed_loss_count > 0 else float("inf"),
                "budget_status": "over_budget" if positive_loss_count > allowed_loss_count + 1e-9 else "within_budget",
            })
        out.append(row)
    return out


def taxonomy_report(selector_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not selector_rows:
        return []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in selector_rows:
        candidate = str(row.get("candidate") or row.get("variant"))
        accept = as_bool(row.get("accept"))
        dpsnr = finite_float(row.get("dPSNR"), 0.0)
        dssim = finite_float(row.get("dSSIM"), 0.0)
        if accept and dpsnr <= -0.20:
            err = "false_accept_severe"
        elif accept and dpsnr <= -0.05:
            err = "false_accept_strong"
        elif (not accept) and dpsnr > 0.02 and dssim >= -0.000005:
            err = "false_reject_good"
        elif (not accept) and dpsnr > 0.0:
            err = "false_reject_positive_weak_or_ssim_risk"
        elif (not accept) and dpsnr <= -0.05:
            err = "true_reject_bad"
        elif accept and dpsnr > 0.02:
            err = "true_accept_good"
        else:
            err = "neutral"
        grouped[(candidate, err)].append(row)
    out: list[dict[str, Any]] = []
    for (candidate, err), rows in sorted(grouped.items()):
        n_cand = sum(1 for row in selector_rows if str(row.get("candidate") or row.get("variant")) == candidate)
        out.append({
            "candidate": candidate,
            "candidate_label": VARIANT_LABEL.get(candidate, candidate),
            "taxonomy": err,
            "count": len(rows),
            "per_600": len(rows) * 600.0 / max(1, n_cand),
            "mean_dPSNR": safe_mean(finite_float(row.get("dPSNR")) for row in rows),
            "mean_dSSIM": safe_mean(finite_float(row.get("dSSIM")) for row in rows),
            "mean_input_brightness": safe_mean(finite_float(row.get("input_brightness_mean")) for row in rows),
            "mean_input_texture": safe_mean(finite_float(row.get("input_texture_mean")) for row in rows),
            "mean_depth": safe_mean(finite_float(row.get("depth_mean")) for row in rows),
            "mean_t_uncertainty": safe_mean(finite_float(row.get("dta_t_uncertainty_mean")) for row in rows),
            "mean_final_action_abs": safe_mean(finite_float(row.get("dta_final_feature_action_abs_mean")) for row in rows),
        })
    return out


def make_keyed_bank(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, dict[str, Any]]]:
    bank: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        variant = str(row.get("variant"))
        if variant in VARIANT_ORDER:
            bank[key_for(row)][variant] = row
    return bank


def action_from(row: dict[str, Any], alpha: float, variant: str) -> dict[str, Any]:
    return {
        "image_id": row.get("image_id"),
        "fold": row.get("fold"),
        "seed": row.get("seed"),
        "A0_PSNR": row.get("A0_PSNR"),
        "dPSNR": alpha * finite_float(row.get("dPSNR"), 0.0),
        "dSSIM": alpha * finite_float(row.get("dSSIM"), 0.0),
        "zero_delta_psnr": alpha * finite_float(row.get("zero_delta_psnr"), 0.0),
        "shuffle_delta_psnr": alpha * finite_float(row.get("shuffle_delta_psnr"), 0.0),
        "normal_delta_psnr": alpha * finite_float(row.get("normal_delta_psnr"), 0.0),
        "chosen_action": f"{VARIANT_LABEL.get(variant, variant)}@{alpha:g}",
        "chosen_variant": variant,
        "chosen_alpha": alpha,
    }


def a0_action(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "image_id": sample.get("image_id"),
        "fold": sample.get("fold"),
        "seed": sample.get("seed"),
        "A0_PSNR": sample.get("A0_PSNR"),
        "dPSNR": 0.0,
        "dSSIM": 0.0,
        "zero_delta_psnr": 0.0,
        "shuffle_delta_psnr": 0.0,
        "normal_delta_psnr": 0.0,
        "chosen_action": "A0@0",
        "chosen_variant": "A0",
        "chosen_alpha": 0.0,
    }


def oracle_utility(action: dict[str, Any], mode: str) -> float:
    dpsnr = finite_float(action.get("dPSNR"), 0.0)
    dssim = finite_float(action.get("dSSIM"), 0.0)
    severe_penalty = max(0.0, -0.20 - dpsnr)
    ssim_penalty = max(0.0, -0.000005 - dssim)
    if mode == "max_dpsnr":
        return dpsnr
    if mode == "tail_averse":
        return dpsnr + 0.15 * max(dpsnr, 0.0) - 4.0 * severe_penalty - 200.0 * ssim_penalty
    if mode == "ssim_guarded":
        return dpsnr if dssim >= -0.000005 else dpsnr - 0.10
    raise ValueError(f"Unknown utility mode: {mode}")


def soft_action_bank_oracle_grid(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keyed = make_keyed_bank(rows)
    specs = [
        ("A0_L3_full", ["l3_fdf_lite_s004_g015_bm2"], [1.0], True),
        ("A0_L3_shrink", ["l3_fdf_lite_s004_g015_bm2"], [0.25, 0.50, 0.75, 1.0], True),
        ("A0_L2_L3_L1_full", VARIANT_ORDER, [1.0], True),
        ("A0_L2_L3_L1_shrink", VARIANT_ORDER, [0.25, 0.50, 0.75, 1.0], True),
        ("A0_L2_L3_L1_micro_shrink", VARIANT_ORDER, [0.10, 0.25, 0.50, 0.75, 1.0], True),
        ("forced_L2_L3_L1_shrink_no_A0", VARIANT_ORDER, [0.25, 0.50, 0.75, 1.0], False),
    ]
    out: list[dict[str, Any]] = []
    for bank_name, variants, alphas, include_a0 in specs:
        for utility_mode in ("max_dpsnr", "tail_averse", "ssim_guarded"):
            chosen: list[dict[str, Any]] = []
            missing = 0
            for by_variant in keyed.values():
                sample = next(iter(by_variant.values()))
                actions: list[dict[str, Any]] = [a0_action(sample)] if include_a0 else []
                for variant in variants:
                    row = by_variant.get(variant)
                    if row is None:
                        continue
                    for alpha in alphas:
                        actions.append(action_from(row, alpha, variant))
                if not actions:
                    missing += 1
                    continue
                actions.sort(key=lambda item: (oracle_utility(item, utility_mode), -finite_float(item.get("chosen_alpha"), 0.0)), reverse=True)
                chosen.append(actions[0])
            metrics = summarize_policy(chosen)
            fold_worst = []
            for fold in sorted({str(row.get("fold")) for row in chosen}):
                fold_rows = [row for row in chosen if str(row.get("fold")) == fold]
                if fold_rows:
                    fold_worst.append(finite_float(summarize_policy(fold_rows).get("worst_per_600"), 0.0))
            metrics["max_outer_worst_per_600"] = max(fold_worst) if fold_worst else metrics.get("worst_per_600")
            checks = gate_checks(metrics)
            action_counts = Counter(str(row.get("chosen_action")) for row in chosen)
            variant_counts = Counter(str(row.get("chosen_variant")) for row in chosen)
            intervention_count = sum(str(row.get("chosen_variant")) != "A0" for row in chosen)
            metrics.update({
                "bank_name": bank_name,
                "utility_mode": utility_mode,
                "include_a0": include_a0,
                "alphas": ";".join(f"{a:g}" for a in alphas),
                "variants": ";".join(variants),
                "missing_key_count": missing,
                "intervention_rate": intervention_count / len(chosen) if chosen else float("nan"),
                "mean_chosen_alpha": safe_mean(finite_float(row.get("chosen_alpha")) for row in chosen),
                "chosen_action_counts": dict(action_counts),
                "chosen_variant_counts": dict(variant_counts),
                "strict_gate_pass": all(checks.values()),
                "strict_gate_checks": checks,
                "proxy_note": "metric-linear alpha proxy; real blended-image PSNR must be verified after Phase A",
            })
            out.append(metrics)
    return sorted(out, key=lambda row: (bool(row.get("strict_gate_pass")), finite_float(row.get("mean_dPSNR"), -1e9)), reverse=True)


def numeric_feature_columns(rows: list[dict[str, Any]]) -> list[str]:
    keys = sorted({key for row in rows for key in row if key not in NON_FEATURE_COLUMNS})
    out = []
    for key in keys:
        vals = [finite_float(row.get(key)) for row in rows]
        finite_vals = [value for value in vals if math.isfinite(value)]
        if len(finite_vals) >= max(20, len(rows) // 50) and len({round(value, 12) for value in finite_vals}) >= 2:
            out.append(key)
    return out


def rank_auc(y_true: list[int], scores: list[float]) -> float:
    pairs = [(score, label) for score, label in zip(scores, y_true) if math.isfinite(score)]
    pos = sum(label for _, label in pairs)
    neg = len(pairs) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    ordered = sorted(enumerate(pairs), key=lambda item: item[1][0])
    rank_sum = 0.0
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][1][0] == ordered[i][1][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            if ordered[k][1][1] == 1:
                rank_sum += avg_rank
        i = j
    return (rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)


def feature_groups(features: list[str]) -> dict[str, list[str]]:
    def pref(prefixes: tuple[str, ...]) -> list[str]:
        return [feature for feature in features if feature.startswith(prefixes)]

    q_input = pref(("input_",))
    depth = pref(("depth_",))
    trans_pred = pref(("dta_t_pred_",))
    uncertainty = [feature for feature in features if "uncertainty" in feature or feature.endswith("_conf_mean") or feature.endswith("_conf_min")]
    airlight = pref(("airlight_fallback", "airlight_proxy", "airlight_pred", "airlight_gt"))
    action = [feature for feature in features if feature.startswith("dta_") and feature not in set(trans_pred + uncertainty)]
    trans_gt = pref(("trans_gt_",))
    nr_iqa = [feature for feature in features if any(token in feature.lower() for token in ("iqa", "quality", "natural", "maniqa", "musiq", "clip_iqa", "contrast", "color"))]
    deployable = [feature for feature in features if not feature.startswith("trans_gt_") and feature not in LEAKY_COLUMNS]
    return {
        "Q_input_proxy": q_input,
        "D_depth": depth,
        "T_pred": trans_pred,
        "A_airlight_proxy": airlight,
        "U_uncertainty_conf": uncertainty,
        "FDF_action_stats": action,
        "NR_IQA_quality_if_present": nr_iqa,
        "deployable_TQAU_action_all": sorted(set(deployable)),
        "diagnostic_T_gt": trans_gt,
        "diagnostic_all_with_T_gt": sorted(set(deployable + trans_gt)),
    }


def feature_ablation_auc_report(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    features = numeric_feature_columns(rows)
    groups = feature_groups(features)
    out: list[dict[str, Any]] = []
    for variant in VARIANT_ORDER:
        vrows = variant_rows(rows, variant)
        if not vrows:
            continue
        for target_name, target_fn in TARGETS.items():
            y = [1 if target_fn(row) else 0 for row in vrows]
            pos = sum(y)
            neg = len(y) - pos
            for group_name, group_features in groups.items():
                auc_rows = []
                for feature in group_features:
                    auc = rank_auc(y, [finite_float(row.get(feature)) for row in vrows])
                    if math.isfinite(auc):
                        auc_rows.append((max(auc, 1.0 - auc), auc, feature))
                if not auc_rows:
                    out.append({
                        "candidate": variant,
                        "candidate_label": VARIANT_LABEL.get(variant, variant),
                        "target": target_name,
                        "feature_group": group_name,
                        "target_positive_count": pos,
                        "target_negative_count": neg,
                        "feature_count": 0,
                    })
                    continue
                auc_rows.sort(reverse=True)
                top_abs, top_auc, top_feature = auc_rows[0]
                out.append({
                    "candidate": variant,
                    "candidate_label": VARIANT_LABEL.get(variant, variant),
                    "target": target_name,
                    "feature_group": group_name,
                    "target_positive_count": pos,
                    "target_negative_count": neg,
                    "feature_count": len(auc_rows),
                    "best_feature": top_feature,
                    "best_auc": top_auc,
                    "best_auc_oriented": top_abs,
                    "mean_oriented_auc_top5": safe_mean(item[0] for item in auc_rows[:5]),
                    "top5_features": ";".join(f"{feature}:{auc:.4f}" for _, auc, feature in auc_rows[:5]),
                    "deployable": not group_name.startswith("diagnostic"),
                })
    return out


def scan_data_dir(data_dir: Path, max_entries: int = 20000) -> dict[str, Any]:
    if not data_dir or not data_dir.exists():
        return {"exists": False, "path": str(data_dir) if data_dir else ""}
    token_counts = Counter()
    likely_dirs: dict[str, list[str]] = defaultdict(list)
    file_count = 0
    dir_count = 0
    tokens = {
        "transmission": ("trans", "transmission"),
        "airlight": ("air", "airlight", "atmos", "atmospheric"),
        "depth": ("depth",),
        "hazy": ("hazy", "input"),
        "clean": ("clean", "gt", "target"),
    }
    for root, dirs, files in os.walk(data_dir):
        dir_count += 1
        lower_root = root.lower()
        for label, pats in tokens.items():
            if any(pat in lower_root for pat in pats) and len(likely_dirs[label]) < 12:
                likely_dirs[label].append(root)
        for name in files:
            file_count += 1
            lower = name.lower()
            for label, pats in tokens.items():
                if any(pat in lower or pat in lower_root for pat in pats):
                    token_counts[label] += 1
            if file_count >= max_entries:
                break
        if file_count >= max_entries:
            break
    return {
        "exists": True,
        "path": str(data_dir),
        "scanned_file_cap": max_entries,
        "scanned_file_count": file_count,
        "scanned_dir_count": dir_count,
        "token_file_counts": dict(token_counts),
        "likely_dirs": {key: vals for key, vals in likely_dirs.items()},
    }


def preflight_payload(rows: list[dict[str, Any]], data_dir: Path | None) -> dict[str, Any]:
    columns = sorted({key for row in rows for key in row})
    def cols_with(*needles: str) -> list[str]:
        return [col for col in columns if any(needle in col.lower() for needle in needles)]

    groups = {
        "transmission_gt_table": [col for col in columns if col.startswith("trans_gt_")],
        "transmission_pred_table": [col for col in columns if col.startswith("dta_t_pred_")],
        "transmission_uncertainty_table": [col for col in columns if "t_uncertainty" in col],
        "airlight_gt_table": [col for col in columns if col.startswith("airlight_gt") or col.startswith("A_gt")],
        "airlight_proxy_table": [col for col in columns if col.startswith("airlight_fallback") or col.startswith("airlight_proxy")],
        "quality_proxy_table": [col for col in columns if col.startswith("input_") or any(token in col.lower() for token in ("quality", "iqa", "natural", "contrast", "color"))],
        "candidate_action_table": [col for col in columns if col.startswith("dta_")],
        "nr_iqa_table": cols_with("maniqa", "musiq", "clip_iqa", "iqa", "naturalness"),
    }
    availability = {key: bool(value) for key, value in groups.items()}
    recommendations = []
    if not availability["airlight_gt_table"]:
        recommendations.append("No explicit airlight GT columns in the action table; use Haze4K dataset preflight or construct train-derived A proxy before supervised A_head training.")
    if not availability["nr_iqa_table"]:
        recommendations.append("No NR-IQA/MANIQA/MUSIQ/CLIP-IQA columns yet; Phase B should add cached deployable quality features, not locked-test feedback.")
    if availability["transmission_gt_table"] and availability["transmission_pred_table"]:
        recommendations.append("Transmission GT and t_pred table signals are present enough for table-only T consistency diagnostics.")
    return {
        "route": "DTA-v3.7 U-TQS-Mix",
        "phase": "A_table_only_no_new_training",
        "rows": len(rows),
        "columns": len(columns),
        "feature_groups": groups,
        "availability": availability,
        "data_dir_scan": scan_data_dir(data_dir) if data_dir else {"provided": False},
        "strict_gates": STRICT_GATES,
        "leakage_policy": "trans_gt and PSNR/SSIM delta columns are diagnostic/oracle only; deployable policy features must exclude them.",
        "recommendations": recommendations,
    }


def make_summary(args: argparse.Namespace, outputs: dict[str, Any]) -> dict[str, Any]:
    soft_rows = outputs["soft"]
    strict_soft = [row for row in soft_rows if row.get("strict_gate_pass") and finite_float(row.get("coverage"), 0.0) >= STRICT_GATES["coverage_min"]]
    best_soft = soft_rows[0] if soft_rows else {}
    return {
        "route": "DTA-v3.7 U-TQS-Mix",
        "status": "PHASE_A_TABLE_ONLY_COMPLETE",
        "input_action_table": str(args.input_action_table),
        "v36_selector_error_table": str(args.v36_selector_error_table) if args.v36_selector_error_table else None,
        "data_dir": str(args.data_dir) if args.data_dir else None,
        "strict_gates": STRICT_GATES,
        "soft_oracle_strict_pass_count": len(strict_soft),
        "best_soft_oracle_row": best_soft,
        "phase_a_gate": "PASS_SOFT_ORACLE_HEADROOM" if strict_soft else "FAIL_SOFT_ORACLE_HEADROOM",
        "next_if_pass": "Run Phase B TQS deployable gain-risk predictor and real blended-output verification on train-derived folds; do not tune on locked test.",
        "next_if_fail": "Do not train v3.7 policy; candidate action family needs stronger bounded candidates first.",
        "proxy_warning": "Soft alpha metrics are table-only linear-delta proxies; they are decisive for route triage but not final image-quality proof.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_action_table", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--v36_selector_error_table", type=Path, default=None)
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--strict_positive_ratio", type=float, default=0.630)
    args = parser.parse_args()

    rows = read_csv_rows(args.input_action_table)
    selector_rows = read_csv_rows(args.v36_selector_error_table) if args.v36_selector_error_table and args.v36_selector_error_table.exists() else []
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    budget = positive_loss_budget_report(rows, selector_rows, args.strict_positive_ratio)
    soft = soft_action_bank_oracle_grid(rows)
    taxonomy = taxonomy_report(selector_rows)
    auc = feature_ablation_auc_report(rows)
    preflight = preflight_payload(rows, args.data_dir)

    write_csv(out / "v37_positive_loss_budget_report.csv", budget)
    write_csv(out / "v37_soft_action_bank_oracle_grid.csv", soft)
    write_csv(out / "v37_false_reject_false_accept_taxonomy.csv", taxonomy)
    write_csv(out / "v37_feature_ablation_auc_report.csv", auc)
    (out / "v37_tA_quality_uncertainty_preflight.json").write_text(json.dumps(preflight, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = make_summary(args, {"budget": budget, "soft": soft, "taxonomy": taxonomy, "auc": auc, "preflight": preflight})
    (out / "v37_phase_a_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "DTA_V3_7_U_TQS_MIX_PHASE_A_OK "
        f"rows={len(rows)} soft_rows={len(soft)} strict_soft={summary['soft_oracle_strict_pass_count']} "
        f"gate={summary['phase_a_gate']}"
    )


if __name__ == "__main__":
    main()
