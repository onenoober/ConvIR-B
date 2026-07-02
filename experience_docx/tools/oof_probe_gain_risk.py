#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from nopost_common import write_csv, write_json


TARGETS = ["benefit_label", "severe_risk_label"]
GROUP_PREFIXES = {
    "hazy_only": ("hazy_",),
    "internal_only": ("final_", "res1_", "res2_", "scm2_", "scm4_"),
    "all": ("hazy_", "final_", "res1_", "res2_", "scm2_", "scm4_"),
}


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def feature_columns(rows: list[dict[str, Any]], group: str) -> list[str]:
    prefixes = GROUP_PREFIXES[group]
    return [key for key in rows[0] if key.startswith(prefixes)]


def make_xy(rows: list[dict[str, Any]], cols: list[str], target: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray([[float(row[c]) for c in cols] for row in rows], dtype=np.float64)
    y = np.asarray([int(row[target]) for row in rows], dtype=np.float64)
    folds = np.asarray([int(row.get("oof_fold", idx % 5)) for idx, row in enumerate(rows)], dtype=np.int64)
    return x, y, folds


def standardize(train_x: np.ndarray, val_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    return (train_x - mean) / std, (val_x - mean) / std


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def fit_logistic(x: np.ndarray, y: np.ndarray, steps: int = 1200, lr: float = 0.05, l2: float = 1e-4) -> np.ndarray:
    xb = np.concatenate([np.ones((x.shape[0], 1)), x], axis=1)
    w = np.zeros(xb.shape[1], dtype=np.float64)
    pos = max(1.0, y.sum())
    neg = max(1.0, len(y) - y.sum())
    weights = np.where(y > 0.5, len(y) / (2.0 * pos), len(y) / (2.0 * neg))
    for _ in range(steps):
        p = sigmoid(xb @ w)
        grad = (xb.T @ ((p - y) * weights)) / len(y)
        grad[1:] += l2 * w[1:]
        w -= lr * grad
    return w


def predict_logistic(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    xb = np.concatenate([np.ones((x.shape[0], 1)), x], axis=1)
    return sigmoid(xb @ w)


def auc_score(y: np.ndarray, score: np.ndarray) -> float | None:
    y = y.astype(np.int64)
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(score)
    ranks = np.empty(len(score), dtype=np.float64)
    i = 0
    while i < len(score):
        j = i
        while j + 1 < len(score) and score[order[j + 1]] == score[order[i]]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    rank_sum_pos = ranks[y == 1].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def ece_score(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float | None:
    if len(y) == 0:
        return None
    total = len(y)
    ece = 0.0
    for b in range(bins):
        lo = b / bins
        hi = (b + 1) / bins
        if b == bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        if not mask.any():
            continue
        ece += mask.mean() * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(ece)


def run_oof(rows: list[dict[str, Any]], group: str, target: str) -> dict[str, Any]:
    cols = feature_columns(rows, group)
    x, y, folds = make_xy(rows, cols, target)
    pred = np.full(len(rows), np.nan, dtype=np.float64)
    fold_reports = []
    for fold in sorted(set(int(v) for v in folds)):
        train_mask = folds != fold
        val_mask = folds == fold
        if len(set(y[train_mask].astype(int))) < 2:
            fold_reports.append({"fold": fold, "status": "skip_one_class_train"})
            pred[val_mask] = y[train_mask].mean() if train_mask.any() else 0.5
            continue
        train_x, val_x = standardize(x[train_mask], x[val_mask])
        w = fit_logistic(train_x, y[train_mask])
        pred[val_mask] = predict_logistic(w, val_x)
        fold_auc = auc_score(y[val_mask], pred[val_mask])
        fold_reports.append(
            {
                "fold": fold,
                "status": "ok",
                "val_count": int(val_mask.sum()),
                "val_positive": int(y[val_mask].sum()),
                "auc": fold_auc,
            }
        )
    auc = auc_score(y, pred)
    ece = ece_score(y, pred)
    return {
        "group": group,
        "target": target,
        "feature_count": len(cols),
        "count": len(rows),
        "positive": int(y.sum()),
        "positive_rate": float(y.mean()),
        "auc": auc,
        "ece": ece,
        "pred": pred,
        "folds": fold_reports,
    }


def nullable_float(value: float | None) -> float:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return float("nan")
    return float(value)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-table", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--benefit-auc-gate", type=float, default=0.70)
    ap.add_argument("--risk-auc-gate", type=float, default=0.70)
    ap.add_argument("--internal-margin", type=float, default=0.00)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.feature_table)
    reports = []
    prediction_rows = []
    predictions: dict[tuple[str, str], np.ndarray] = {}

    for target in TARGETS:
        for group in ("hazy_only", "internal_only", "all"):
            report = run_oof(rows, group, target)
            predictions[(target, group)] = report.pop("pred")
            reports.append(report)

    for idx, row in enumerate(rows):
        out = {
            "name": row["name"],
            "source_split": row.get("source_split", ""),
            "oof_fold": row.get("oof_fold", ""),
            "WD0375_dPSNR": row.get("WD0375_dPSNR", ""),
            "benefit_label": row.get("benefit_label", ""),
            "severe_risk_label": row.get("severe_risk_label", ""),
        }
        for target in TARGETS:
            for group in ("hazy_only", "internal_only", "all"):
                out[f"{target}_{group}_pred"] = float(predictions[(target, group)][idx])
        prediction_rows.append(out)

    def get_auc(target: str, group: str) -> float:
        for report in reports:
            if report["target"] == target and report["group"] == group:
                return nullable_float(report["auc"])
        return float("nan")

    benefit_all = get_auc("benefit_label", "all")
    risk_all = get_auc("severe_risk_label", "all")
    benefit_internal = get_auc("benefit_label", "internal_only")
    benefit_hazy = get_auc("benefit_label", "hazy_only")
    risk_internal = get_auc("severe_risk_label", "internal_only")
    risk_hazy = get_auc("severe_risk_label", "hazy_only")

    pass_gate = (
        benefit_all >= args.benefit_auc_gate
        and risk_all >= args.risk_auc_gate
        and benefit_internal >= benefit_hazy + args.internal_margin
        and risk_internal >= risk_hazy + args.internal_margin
    )
    decision = "N1_MECHANISM_PASS_CONTINUE_N2_N3" if pass_gate else "N1_MECHANISM_FAIL_STOP_BEFORE_TRAINING"

    write_csv(args.out_dir / "v213_n1_oof_gain_risk_probe.csv", prediction_rows)
    write_csv(args.out_dir / "v213_n1_feature_ablation_report.csv", reports)
    calibration = {
        "reports": reports,
        "gates": {
            "benefit_auc_gate": args.benefit_auc_gate,
            "risk_auc_gate": args.risk_auc_gate,
            "internal_margin": args.internal_margin,
        },
        "key_metrics": {
            "benefit_all_auc": benefit_all,
            "risk_all_auc": risk_all,
            "benefit_internal_auc": benefit_internal,
            "benefit_hazy_auc": benefit_hazy,
            "risk_internal_auc": risk_internal,
            "risk_hazy_auc": risk_hazy,
        },
        "decision": decision,
        "pass": pass_gate,
        "locked_test_touched": False,
    }
    write_json(args.out_dir / "v213_n1_calibration_report.json", calibration)
    (args.out_dir / "v213_n1_decision.md").write_text(
        "# v2.13 N1 Feature Separability Probe\n\n"
        f"Decision: `{decision}`\n\n"
        f"- benefit all-feature AUC: `{benefit_all:.6f}`\n"
        f"- severe-risk all-feature AUC: `{risk_all:.6f}`\n"
        f"- benefit internal/hazy AUC: `{benefit_internal:.6f}` / `{benefit_hazy:.6f}`\n"
        f"- risk internal/hazy AUC: `{risk_internal:.6f}` / `{risk_hazy:.6f}`\n"
        "- locked test touched: `false`\n",
        encoding="utf-8",
    )
    print("N1_PROBE_PASS" if pass_gate else "N1_PROBE_FAIL")
    print(json.dumps(calibration, indent=2, sort_keys=True))
    if not pass_gate:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
