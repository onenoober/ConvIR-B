#!/usr/bin/env python3
"""Analyze v1.7 full-train A0/UDP risk-controlled expert mixture.

Inputs are the v1.7 full-train feature table. Outputs are train-derived text
evidence only: alpha-grid oracle, OOF gain/risk predictability, risk-coverage
policy curves, fold stability, calibration curves, and train-heldout
confirmation. Locked Haze4K test is not touched.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


ROUTE_ID = "haze4k_v17_rc_expert_mix_20260605"
ALPHAS = [1.0, 0.75, 0.5, 0.25]
ALPHA_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
GAIN_THRESHOLD_GRID = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
RISK_THRESHOLD_GRID = [0.02, 0.035, 0.05, 0.075, 0.10, 0.15, 0.20]
OOD_QUANTILE_GRID = [0.90, 0.95, 0.98, 1.00]
LEAKY_PREFIXES = ("alpha_", "oracle_")
LEAKY_COLUMNS = {
    "split",
    "name",
    "bucket",
    "v17_bucket",
    "fold",
    "a0_psnr",
    "a0_ssim",
    "udpnet_psnr",
    "udpnet_ssim",
    "delta_psnr",
    "delta_ssim",
}


def alpha_tag(alpha: float) -> str:
    return f"{alpha:.2f}".replace(".", "p")


def to_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(float(value)) else default
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return default
    try:
        out = float(text)
    except ValueError:
        return default
    return out if math.isfinite(out) else default


def mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * pct / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        raw_rows = list(csv.DictReader(handle))
    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        row: dict[str, Any] = {}
        for key, value in raw.items():
            fvalue = to_float(value)
            row[key] = fvalue if fvalue is not None else value
        rows.append(row)
    return rows


def add_v17_buckets(rows: list[dict[str, Any]]) -> None:
    a0_values = [float(row["a0_psnr"]) for row in rows]
    hard_cut = percentile(a0_values, 25)
    easy_cut = percentile(a0_values, 75)
    assert hard_cut is not None and easy_cut is not None
    for row in rows:
        psnr = float(row["a0_psnr"])
        if psnr <= hard_cut:
            row["v17_bucket"] = "hard_bottom25_by_a0_fulltrain"
        elif psnr >= easy_cut:
            row["v17_bucket"] = "easy_top25_by_a0_fulltrain"
        else:
            row["v17_bucket"] = "mid_by_a0_fulltrain"


def numeric_feature_names(rows: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    if not rows:
        return names
    for key in rows[0]:
        if key in LEAKY_COLUMNS or key.startswith(LEAKY_PREFIXES):
            continue
        if key.startswith("_"):
            continue
        if key.startswith("filename_param"):
            continue
        values = [to_float(row.get(key)) for row in rows]
        finite = [value for value in values if value is not None]
        if finite:
            names.append(key)
    return names


def alpha_delta(row: dict[str, Any], alpha: float) -> float:
    if alpha == 0.0:
        return 0.0
    return float(row[f"alpha_{alpha_tag(alpha)}_delta_psnr"])


def alpha_ssim_delta(row: dict[str, Any], alpha: float) -> float:
    if alpha == 0.0:
        return 0.0
    return float(row[f"alpha_{alpha_tag(alpha)}_delta_ssim"])


def labels_for_alpha(row: dict[str, Any], alpha: float) -> tuple[int, int]:
    delta = alpha_delta(row, alpha)
    ssim_delta = alpha_ssim_delta(row, alpha)
    gain = int(delta >= 0.10)
    risk = int(
        delta <= -0.20
        or ssim_delta <= -0.001
        or (row.get("v17_bucket") == "easy_top25_by_a0_fulltrain" and delta < 0)
    )
    return gain, risk


def summary_for_decisions(rows: list[dict[str, Any]], decisions: list[float], label: str) -> dict[str, Any]:
    deltas = [alpha_delta(row, alpha) for row, alpha in zip(rows, decisions)]
    ssim_deltas = [alpha_ssim_delta(row, alpha) for row, alpha in zip(rows, decisions)]
    hard = [(row, alpha) for row, alpha in zip(rows, decisions) if row.get("v17_bucket") == "hard_bottom25_by_a0_fulltrain"]
    easy = [(row, alpha) for row, alpha in zip(rows, decisions) if row.get("v17_bucket") == "easy_top25_by_a0_fulltrain"]
    strong = easy
    strong_reg = [1 for row, alpha in strong if alpha_delta(row, alpha) <= -0.05]
    worst = [1 for row, alpha in zip(rows, decisions) if alpha_delta(row, alpha) <= -0.20]
    tail_n = max(1, len(deltas) // 10)
    ordered = sorted(deltas)
    chosen = [alpha for alpha in decisions if alpha > 0]
    return {
        "label": label,
        "count": len(rows),
        "coverage": len(chosen) / max(1, len(rows)),
        "alpha_1p00_count": sum(alpha == 1.0 for alpha in decisions),
        "alpha_0p75_count": sum(alpha == 0.75 for alpha in decisions),
        "alpha_0p50_count": sum(alpha == 0.5 for alpha in decisions),
        "alpha_0p25_count": sum(alpha == 0.25 for alpha in decisions),
        "alpha_0p00_count": sum(alpha == 0.0 for alpha in decisions),
        "mean_delta": mean(deltas),
        "median_delta": statistics.median(deltas) if deltas else None,
        "p5_delta": percentile(deltas, 5),
        "p95_delta": percentile(deltas, 95),
        "hard_bottom25_delta": mean([alpha_delta(row, alpha) for row, alpha in hard]),
        "easy_top25_delta": mean([alpha_delta(row, alpha) for row, alpha in easy]),
        "worst10pct_delta": mean(ordered[:tail_n]),
        "best10pct_delta": mean(ordered[-tail_n:]),
        "mean_ssim_delta": mean(ssim_deltas),
        "positive_ratio": sum(delta > 0 for delta in deltas) / max(1, len(deltas)),
        "worst_regression_ratio": len(worst) / max(1, len(rows)),
        "strong_regression_ratio": len(strong_reg) / max(1, len(strong)),
    }


def metric_or(summary: dict[str, Any], key: str, default: float) -> float:
    value = summary.get(key)
    return default if value is None else float(value)


def add_gate_flags(summary: dict[str, Any], prefix: str) -> dict[str, Any]:
    if prefix == "oof":
        checks = {
            "mean_delta_ge_0p25": metric_or(summary, "mean_delta", -999) >= 0.25,
            "hard_bottom25_delta_ge_0p55": metric_or(summary, "hard_bottom25_delta", -999) >= 0.55,
            "easy_top25_delta_ge_0p03": metric_or(summary, "easy_top25_delta", -999) >= 0.03,
            "mean_ssim_delta_ge_0": metric_or(summary, "mean_ssim_delta", -999) >= 0,
            "worst_ratio_le_0p035": metric_or(summary, "worst_regression_ratio", 999) <= 0.035,
            "strong_ratio_le_0p08": metric_or(summary, "strong_regression_ratio", 999) <= 0.08,
        }
    elif prefix == "heldout":
        checks = {
            "mean_delta_ge_0p18": metric_or(summary, "mean_delta", -999) >= 0.18,
            "hard_bottom25_delta_ge_0p35": metric_or(summary, "hard_bottom25_delta", -999) >= 0.35,
            "easy_top25_delta_ge_neg0p02": metric_or(summary, "easy_top25_delta", -999) >= -0.02,
            "worst_ratio_le_0p04": metric_or(summary, "worst_regression_ratio", 999) <= 0.04,
            "strong_ratio_le_0p10": metric_or(summary, "strong_regression_ratio", 999) <= 0.10,
        }
    else:
        checks = {
            "mean_delta_ge_0p10": metric_or(summary, "mean_delta", -999) >= 0.10,
            "hard_bottom25_delta_ge_0p25": metric_or(summary, "hard_bottom25_delta", -999) >= 0.25,
            "easy_top25_delta_ge_neg0p10": metric_or(summary, "easy_top25_delta", -999) >= -0.10,
            "worst_ratio_le_0p10": metric_or(summary, "worst_regression_ratio", 999) <= 0.10,
            "strong_ratio_le_0p30": metric_or(summary, "strong_regression_ratio", 999) <= 0.30,
        }
    summary[f"{prefix}_gate_checks"] = checks
    summary[f"{prefix}_gate_pass"] = all(checks.values())
    return summary


def oracle_alpha_grid(rows: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    fixed = {}
    for alpha in ALPHA_GRID:
        fixed[alpha_tag(alpha)] = add_gate_flags(
            summary_for_decisions(rows, [alpha] * len(rows), f"fixed_alpha_{alpha_tag(alpha)}"),
            "utility",
        )
    oracle_decisions = []
    oracle_rows = []
    for row in rows:
        candidates = [(alpha, alpha_delta(row, alpha), alpha_ssim_delta(row, alpha)) for alpha in ALPHA_GRID]
        alpha, delta, ssim_delta = max(candidates, key=lambda item: item[1])
        if delta < 0:
            alpha = 0.0
            delta = 0.0
            ssim_delta = 0.0
        oracle_decisions.append(alpha)
        oracle_rows.append(
            {
                "split": row["split"],
                "name": row["name"],
                "v17_bucket": row["v17_bucket"],
                "oracle_alpha": alpha,
                "oracle_delta_psnr": delta,
                "oracle_delta_ssim": ssim_delta,
                "a0_psnr": row["a0_psnr"],
                "udp_delta_psnr": row["delta_psnr"],
            }
        )
    oracle_summary = add_gate_flags(summary_for_decisions(rows, oracle_decisions, "oracle_best_alpha"), "utility")
    payload = {
        "route_id": ROUTE_ID,
        "stage": "oracle switch-mix alpha grid",
        "status": "ORACLE_ALPHA_GRID_COMPLETE",
        "locked_test_touched": False,
        "alpha_grid": ALPHA_GRID,
        "fixed_alpha_summaries": fixed,
        "oracle_best_alpha_summary": oracle_summary,
        "note": "Oracle uses GT to choose the best alpha per image; this is an upper bound, not a deployable router.",
    }
    write_json(output_dir / "v17_oracle_switch_mix_alpha_grid.json", payload)
    write_csv(output_dir / "v17_oracle_switch_mix_alpha_grid_per_image.csv", oracle_rows)
    return payload


def stratum_for_row(row: dict[str, Any]) -> str:
    def q(value: float, cuts: list[float]) -> int:
        bucket = 0
        for cut in cuts:
            if value > cut:
                bucket += 1
        return bucket

    # Cuts are attached later by assign_folds.
    a0_bucket = row.get("_a0_q", 0)
    udp_bucket = row.get("_udp_q", 0)
    scene = str(row.get("name", "")).split("_")[0]
    try:
        scene_chunk = int(scene) // 250
    except ValueError:
        scene_chunk = 0
    return f"a{a0_bucket}_u{udp_bucket}_s{scene_chunk}_{row.get('v17_bucket')}"


def assign_folds(rows: list[dict[str, Any]], folds: int, seed: int) -> None:
    import random

    a0_cuts = [percentile([float(row["a0_psnr"]) for row in rows], pct) for pct in (20, 40, 60, 80)]
    udp_cuts = [percentile([float(row["delta_psnr"]) for row in rows], pct) for pct in (20, 40, 60, 80)]
    a0_cuts = [float(x) for x in a0_cuts if x is not None]
    udp_cuts = [float(x) for x in udp_cuts if x is not None]

    def q(value: float, cuts: list[float]) -> int:
        bucket = 0
        for cut in cuts:
            if value > cut:
                bucket += 1
        return bucket

    strata: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        row["_a0_q"] = q(float(row["a0_psnr"]), a0_cuts)
        row["_udp_q"] = q(float(row["delta_psnr"]), udp_cuts)
        strata.setdefault(stratum_for_row(row), []).append(row)
    rng = random.Random(seed)
    counts = [0] * folds
    for stratum, items in sorted(strata.items()):
        rng.shuffle(items)
        for item in items:
            fold = min(range(folds), key=lambda idx: counts[idx])
            item["fold"] = fold
            counts[fold] += 1


def feature_matrix(rows: list[dict[str, Any]], features: list[str]):
    import numpy as np

    return np.asarray(
        [
            [float(to_float(row.get(feature), 0.0) or 0.0) for feature in features]
            for row in rows
        ],
        dtype="float64",
    )


def fit_model_predict(train_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]], features: list[str], alpha: float, label_kind: str):
    import numpy as np
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    x_train = feature_matrix(train_rows, features)
    x_test = feature_matrix(test_rows, features)
    labels = []
    for row in train_rows:
        gain, risk = labels_for_alpha(row, alpha)
        labels.append(gain if label_kind == "gain" else risk)
    y = np.asarray(labels, dtype="int64")
    if len(set(labels)) < 2:
        prob = float(labels[0]) if labels else 0.0
        return [prob] * len(test_rows)
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=3407,
                ),
            ),
        ]
    )
    model.fit(x_train, y)
    return model.predict_proba(x_test)[:, 1].tolist()


def fit_all_head_probabilities(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    features: list[str],
) -> dict[tuple[float, str], list[float]]:
    probs: dict[tuple[float, str], list[float]] = {}
    for alpha in ALPHAS:
        for kind in ("gain", "risk"):
            probs[(alpha, kind)] = fit_model_predict(train_rows, test_rows, features, alpha, kind)
    return probs


def ood_distances(train_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]], features: list[str]) -> list[float]:
    import numpy as np
    from sklearn.impute import SimpleImputer
    from sklearn.neighbors import NearestNeighbors
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    x_train = feature_matrix(train_rows, features)
    x_test = feature_matrix(test_rows, features)
    transform = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    x_train_t = transform.fit_transform(x_train)
    x_test_t = transform.transform(x_test)
    if len(x_train_t) == 0:
        return [999.0] * len(test_rows)
    nn = NearestNeighbors(n_neighbors=min(5, len(x_train_t)))
    nn.fit(x_train_t)
    distances, _idx = nn.kneighbors(x_test_t)
    return np.mean(distances, axis=1).tolist()


def oof_predictions(rows: list[dict[str, Any]], features: list[str], folds: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pred_rows: list[dict[str, Any]] = []
    for fold in range(folds):
        train = [row for row in rows if int(row["fold"]) != fold]
        test = [row for row in rows if int(row["fold"]) == fold]
        fold_ood = ood_distances(train, test, features)
        fold_probs = fit_all_head_probabilities(train, test, features)
        for idx, row in enumerate(test):
            out = {
                "split": row["split"],
                "name": row["name"],
                "fold": fold,
                "v17_bucket": row["v17_bucket"],
                "a0_psnr": row["a0_psnr"],
                "udp_delta_psnr": row["delta_psnr"],
                "ood_distance": fold_ood[idx],
            }
            for alpha in ALPHAS:
                tag = alpha_tag(alpha)
                gain, risk = labels_for_alpha(row, alpha)
                out[f"alpha_{tag}_gain_prob"] = fold_probs[(alpha, "gain")][idx]
                out[f"alpha_{tag}_risk_prob"] = fold_probs[(alpha, "risk")][idx]
                out[f"alpha_{tag}_gain_label"] = gain
                out[f"alpha_{tag}_risk_label"] = risk
                out[f"alpha_{tag}_delta_psnr"] = alpha_delta(row, alpha)
                out[f"alpha_{tag}_delta_ssim"] = alpha_ssim_delta(row, alpha)
            pred_rows.append(out)
    meta = {
        "feature_count": len(features),
        "features": features,
        "fold_count": folds,
        "model": "SimpleImputer + StandardScaler + balanced LogisticRegression(C=1.0)",
        "ood_distance": "mean 5-nearest-neighbor distance in standardized deployable feature space",
    }
    return pred_rows, meta


def metric_auc_rows(pred_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    rows = []
    for alpha in ALPHAS:
        tag = alpha_tag(alpha)
        for kind in ("gain", "risk"):
            y = [int(row[f"alpha_{tag}_{kind}_label"]) for row in pred_rows]
            p = [float(row[f"alpha_{tag}_{kind}_prob"]) for row in pred_rows]
            if len(set(y)) < 2:
                auc = None
                ap = None
            else:
                auc = float(roc_auc_score(y, p))
                ap = float(average_precision_score(y, p))
            rows.append(
                {
                    "alpha": alpha,
                    "head": kind,
                    "count": len(y),
                    "positive_count": sum(y),
                    "positive_ratio": sum(y) / max(1, len(y)),
                    "roc_auc": auc,
                    "pr_auc": ap,
                    "brier": float(brier_score_loss(y, p)),
                }
            )
    return rows


def calibration_rows(pred_rows: list[dict[str, Any]], bins: int = 10) -> list[dict[str, Any]]:
    rows = []
    for alpha in ALPHAS:
        tag = alpha_tag(alpha)
        items = sorted(
            [
                (float(row[f"alpha_{tag}_risk_prob"]), int(row[f"alpha_{tag}_risk_label"]))
                for row in pred_rows
            ],
            key=lambda item: item[0],
        )
        if not items:
            continue
        for bin_id in range(bins):
            lo = bin_id / bins
            hi = (bin_id + 1) / bins
            bucket = [(p, y) for p, y in items if (p >= lo and (p < hi or bin_id == bins - 1))]
            if not bucket:
                rows.append({"alpha": alpha, "bin": bin_id, "count": 0, "predicted_risk": "", "observed_risk": ""})
            else:
                rows.append(
                    {
                        "alpha": alpha,
                        "bin": bin_id,
                        "count": len(bucket),
                        "predicted_risk": mean([p for p, _y in bucket]),
                        "observed_risk": mean([float(y) for _p, y in bucket]),
                    }
                )
    return rows


def decisions_from_policy(pred_rows: list[dict[str, Any]], tau_gain: float, tau_risk: float, tau_ood: float) -> list[float]:
    decisions = []
    for row in pred_rows:
        chosen = 0.0
        if float(row["ood_distance"]) <= tau_ood:
            for alpha in ALPHAS:
                tag = alpha_tag(alpha)
                if (
                    float(row[f"alpha_{tag}_gain_prob"]) >= tau_gain
                    and float(row[f"alpha_{tag}_risk_prob"]) <= tau_risk
                ):
                    chosen = alpha
                    break
        decisions.append(chosen)
    return decisions


def policy_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        not bool(row.get("oof_gate_pass")),
        -float(row.get("mean_delta") or -999),
        -float(row.get("hard_bottom25_delta") or -999),
        float(row.get("worst_regression_ratio") if row.get("worst_regression_ratio") is not None else 999),
        float(row.get("strong_regression_ratio") if row.get("strong_regression_ratio") is not None else 999),
        -float(row.get("coverage") or 0),
    )


def search_policy(rows: list[dict[str, Any]], pred_rows: list[dict[str, Any]], label: str) -> tuple[list[dict[str, Any]], dict[str, Any], list[float]]:
    row_by_name = {str(row["name"]): row for row in rows}
    ordered_rows = [row_by_name[str(pred["name"])] for pred in pred_rows]
    ood_values = [float(row["ood_distance"]) for row in pred_rows]
    policy_rows = []
    for tau_gain in GAIN_THRESHOLD_GRID:
        for tau_risk in RISK_THRESHOLD_GRID:
            for q in OOD_QUANTILE_GRID:
                tau_ood = percentile(ood_values, 100 * q)
                assert tau_ood is not None
                decisions = decisions_from_policy(pred_rows, tau_gain, tau_risk, tau_ood)
                summary = add_gate_flags(summary_for_decisions(ordered_rows, decisions, label), "oof")
                policy_rows.append(
                    {
                        "label": label,
                        "tau_gain": tau_gain,
                        "tau_risk": tau_risk,
                        "tau_ood_quantile": q,
                        "tau_ood": tau_ood,
                        **{k: v for k, v in summary.items() if k not in {"label", "oof_gate_checks"}},
                        "oof_gate_pass": summary["oof_gate_pass"],
                    }
                )
    policy_rows = sorted(policy_rows, key=policy_sort_key)
    selected = policy_rows[0]
    decisions = decisions_from_policy(pred_rows, float(selected["tau_gain"]), float(selected["tau_risk"]), float(selected["tau_ood"]))
    return policy_rows, selected, decisions


def rows_in_prediction_order(rows: list[dict[str, Any]], pred_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    row_by_name = {str(row["name"]): row for row in rows}
    return [row_by_name[str(pred["name"])] for pred in pred_rows]


def stability_by_fold(rows: list[dict[str, Any]], pred_rows: list[dict[str, Any]], selected: dict[str, Any], folds: int) -> list[dict[str, Any]]:
    by_name = {str(row["name"]): row for row in rows}
    out = []
    for fold in range(folds):
        fold_preds = [row for row in pred_rows if int(row["fold"]) == fold]
        fold_rows = [by_name[str(row["name"])] for row in fold_preds]
        decisions = decisions_from_policy(
            fold_preds,
            float(selected["tau_gain"]),
            float(selected["tau_risk"]),
            float(selected["tau_ood"]),
        )
        summary = add_gate_flags(summary_for_decisions(fold_rows, decisions, f"fold_{fold}"), "utility")
        out.append(
            {
                "fold": fold,
                "tau_gain": selected["tau_gain"],
                "tau_risk": selected["tau_risk"],
                "tau_ood": selected["tau_ood"],
                **{k: v for k, v in summary.items() if k not in {"label", "utility_gate_checks"}},
                "utility_gate_pass": summary["utility_gate_pass"],
            }
        )
    return out


def fit_predict_holdout(
    train_rows: list[dict[str, Any]],
    holdout_rows: list[dict[str, Any]],
    features: list[str],
    selected: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[float], dict[str, Any]]:
    pred_rows = []
    distances = ood_distances(train_rows, holdout_rows, features)
    holdout_probs = fit_all_head_probabilities(train_rows, holdout_rows, features)
    for idx, row in enumerate(holdout_rows):
        out = {
            "split": row["split"],
            "name": row["name"],
            "fold": "trainheldout",
            "v17_bucket": row["v17_bucket"],
            "a0_psnr": row["a0_psnr"],
            "udp_delta_psnr": row["delta_psnr"],
            "ood_distance": distances[idx],
        }
        for alpha in ALPHAS:
            tag = alpha_tag(alpha)
            gain, risk = labels_for_alpha(row, alpha)
            out[f"alpha_{tag}_gain_prob"] = holdout_probs[(alpha, "gain")][idx]
            out[f"alpha_{tag}_risk_prob"] = holdout_probs[(alpha, "risk")][idx]
            out[f"alpha_{tag}_gain_label"] = gain
            out[f"alpha_{tag}_risk_label"] = risk
            out[f"alpha_{tag}_delta_psnr"] = alpha_delta(row, alpha)
            out[f"alpha_{tag}_delta_ssim"] = alpha_ssim_delta(row, alpha)
        pred_rows.append(out)
    decisions = decisions_from_policy(
        pred_rows,
        float(selected["tau_gain"]),
        float(selected["tau_risk"]),
        float(selected["tau_ood"]),
    )
    summary = add_gate_flags(summary_for_decisions(holdout_rows, decisions, "trainheldout_confirm"), "heldout")
    payload = {
        "status": "TRAIN_HELDOUT_CONFIRM_COMPLETE",
        "locked_test_touched": False,
        "train_count": len(train_rows),
        "holdout_count": len(holdout_rows),
        "train_splits": sorted({str(row["split"]) for row in train_rows}),
        "holdout_splits": sorted({str(row["split"]) for row in holdout_rows}),
        "selected_policy_from_train_oof": selected,
        "holdout_summary": summary,
        "locked_test_allowed": bool(summary["heldout_gate_pass"]),
        "warning": "Locked test still requires route-card approval and immutable one-shot command even if this flag is true.",
    }
    return pred_rows, decisions, payload


def decision_per_image(rows: list[dict[str, Any]], pred_rows: list[dict[str, Any]], decisions: list[float]) -> list[dict[str, Any]]:
    by_name = {str(row["name"]): row for row in rows}
    out = []
    for pred, alpha in zip(pred_rows, decisions):
        row = by_name[str(pred["name"])]
        out.append(
            {
                "split": row["split"],
                "name": row["name"],
                "fold": pred.get("fold"),
                "v17_bucket": row["v17_bucket"],
                "chosen_alpha": alpha,
                "selected_delta_psnr": alpha_delta(row, alpha),
                "selected_delta_ssim": alpha_ssim_delta(row, alpha),
                "udp_delta_psnr": row["delta_psnr"],
                "a0_psnr": row["a0_psnr"],
                "ood_distance": pred.get("ood_distance"),
            }
        )
    return out


def write_readme(output_dir: Path) -> None:
    text = """# Haze4K v1.7 Risk-Controlled Expert Mix Analysis

Status: read `v17_analysis_status.json`.

Primary files:

- `v17_oracle_switch_mix_alpha_grid.json`: fixed alpha and GT-oracle alpha
  upper-bound summaries.
- `v17_oof_gain_risk_predictability.csv`: OOF gain/risk head ROC/PR/Brier
  diagnostics for alpha `1.0/0.75/0.5/0.25`.
- `v17_oof_risk_coverage_curves.csv`: searched risk-controlled alpha policy
  curves over gain, risk, and OOD thresholds.
- `v17_policy_stability_by_fold.csv`: selected policy metrics by OOF fold.
- `v17_calibration_curve_bad_risk.csv`: risk probability calibration bins.
- `v17_trainheldout_confirm_summary.json`: train_inner fit/OOF selection and
  val_regular+val_hard holdout confirmation.
- `v17_oof_policy_per_image.csv` and `v17_trainheldout_policy_per_image.csv`:
  selected alpha and per-image deltas.

Locked Haze4K test touched: no.

Interpretation: oracle files are upper bounds. Only OOF plus train-derived
heldout confirmation can authorize a later immutable locked-test command.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--train_splits", nargs="+", default=["train_inner"])
    parser.add_argument("--holdout_splits", nargs="+", default=["val_regular", "val_hard"])
    args = parser.parse_args()

    feature_csv = Path(args.feature_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(feature_csv)
    if not rows:
        raise RuntimeError(f"No rows read from {feature_csv}")
    add_v17_buckets(rows)
    assign_folds(rows, args.folds, args.seed)
    features = numeric_feature_names(rows)

    oracle_payload = oracle_alpha_grid(rows, output_dir)
    pred_rows, pred_meta = oof_predictions(rows, features, args.folds)
    predictability = metric_auc_rows(pred_rows)
    calibration = calibration_rows(pred_rows)
    policy_rows, selected, decisions = search_policy(rows, pred_rows, "fulltrain_oof_policy")
    fulltrain_ordered_rows = rows_in_prediction_order(rows, pred_rows)
    stability = stability_by_fold(rows, pred_rows, selected, args.folds)

    write_csv(output_dir / "v17_oof_gain_risk_predictability.csv", predictability)
    write_csv(output_dir / "v17_calibration_curve_bad_risk.csv", calibration)
    write_csv(output_dir / "v17_oof_risk_coverage_curves.csv", policy_rows)
    write_csv(output_dir / "v17_policy_stability_by_fold.csv", stability)
    write_csv(output_dir / "v17_oof_policy_per_image.csv", decision_per_image(rows, pred_rows, decisions))

    train_rows = [row for row in rows if str(row["split"]) in set(args.train_splits)]
    holdout_rows = [row for row in rows if str(row["split"]) in set(args.holdout_splits)]
    if not train_rows or not holdout_rows:
        raise RuntimeError(
            f"Train/holdout split mismatch: train={len(train_rows)} holdout={len(holdout_rows)}"
        )
    train_pred_rows, _train_meta = oof_predictions(train_rows, features, args.folds)
    train_policy_rows, train_selected, _train_decisions = search_policy(
        train_rows,
        train_pred_rows,
        "train_inner_oof_policy",
    )
    holdout_pred_rows, holdout_decisions, heldout_payload = fit_predict_holdout(
        train_rows,
        holdout_rows,
        features,
        train_selected,
    )
    write_csv(output_dir / "v17_traininner_oof_risk_coverage_curves.csv", train_policy_rows)
    write_csv(
        output_dir / "v17_trainheldout_policy_per_image.csv",
        decision_per_image(holdout_rows, holdout_pred_rows, holdout_decisions),
    )
    write_json(output_dir / "v17_trainheldout_confirm_summary.json", heldout_payload)

    status = {
        "route_id": ROUTE_ID,
        "status": "V17_RISK_CONTROLLED_MIX_ANALYSIS_COMPLETE",
        "locked_test_touched": False,
        "feature_csv": str(feature_csv),
        "row_count": len(rows),
        "feature_count": len(features),
        "prediction_meta": pred_meta,
        "oracle": oracle_payload,
        "selected_fulltrain_oof_policy": selected,
        "fulltrain_oof_summary": add_gate_flags(
            summary_for_decisions(fulltrain_ordered_rows, decisions, "selected_fulltrain_oof_policy"),
            "oof",
        ),
        "fold_utility_pass_count": sum(bool(row.get("utility_gate_pass")) for row in stability),
        "trainheldout": heldout_payload,
        "locked_test_allowed_by_v17_internal_contract": bool(
            add_gate_flags(summary_for_decisions(fulltrain_ordered_rows, decisions, "selected_fulltrain_oof_policy"), "oof")["oof_gate_pass"]
            and heldout_payload["holdout_summary"]["heldout_gate_pass"]
        ),
    }
    write_json(output_dir / "v17_analysis_status.json", status)
    write_readme(output_dir)
    print(
        "V17_RISK_CONTROLLED_MIX_ANALYSIS_OK "
        f"rows={len(rows)} features={len(features)} output_dir={output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
