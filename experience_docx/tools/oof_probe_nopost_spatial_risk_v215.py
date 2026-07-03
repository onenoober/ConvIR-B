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


RISK_THRESHOLDS = (-0.10, -0.20, -0.30)
GROUP_ORDER = (
    "B0_hazy_runtime_v214",
    "B1_internal_scalar_v214",
    "B2_all_scalar_v214",
    "B3_internal_spatial",
    "B4_hazy_plus_internal_spatial",
    "B5_internal_sensitivity",
    "B6_spatial_plus_sensitivity",
    "B7_all_runtime_spatial_sensitivity",
)
PRIMARY_LABEL = "risk_m0p2"


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_markdown(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def key_name(row: dict[str, Any]) -> str:
    return row["name"]


def rows_by_name(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {key_name(row): row for row in rows}


def numeric_columns(rows: list[dict[str, Any]], prefixes: tuple[str, ...]) -> list[str]:
    if not rows:
        return []
    out: list[str] = []
    for key in rows[0]:
        if not key.startswith(prefixes):
            continue
        try:
            float(rows[0][key])
        except (TypeError, ValueError):
            continue
        out.append(key)
    return out


def runtime_cols(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    hazy = [
        c
        for c in numeric_columns(rows, ("hazy_",))
        if c
        not in {
            "hazy_PSNR",
        }
    ]
    internal = numeric_columns(rows, ("final_", "res1_", "res2_", "scm2_", "scm4_"))
    return {
        "B0_hazy_runtime_v214": hazy,
        "B1_internal_scalar_v214": internal,
        "B2_all_scalar_v214": hazy + internal,
    }


def merged_rows(
    scalar_rows: list[dict[str, Any]],
    spatial_rows: list[dict[str, Any]],
    fam_rows: list[dict[str, Any]],
    skip_rows: list[dict[str, Any]],
    jitter_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    spatial = rows_by_name(spatial_rows)
    fam = rows_by_name(fam_rows)
    skip = rows_by_name(skip_rows)
    jitter = rows_by_name(jitter_rows)
    merged: list[dict[str, Any]] = []
    for row in scalar_rows:
        name = key_name(row)
        out = dict(row)
        for source in (spatial, fam, skip, jitter):
            for key, value in source[name].items():
                if key not in out:
                    out[key] = value
        merged.append(out)
    return merged


def feature_groups(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    base = runtime_cols(rows)
    hazy = base["B0_hazy_runtime_v214"]
    scalar = base["B1_internal_scalar_v214"]
    spatial_hazy = numeric_columns(rows, ("sp_hazy_",))
    spatial_internal = numeric_columns(rows, ("sp_final_", "sp_res1_", "sp_res2_", "sp_scm2_", "sp_scm4_"))
    sensitivity = numeric_columns(rows, ("sens_",))
    groups = dict(base)
    groups.update(
        {
            "B3_internal_spatial": spatial_internal,
            "B4_hazy_plus_internal_spatial": hazy + spatial_hazy + spatial_internal,
            "B5_internal_sensitivity": sensitivity,
            "B6_spatial_plus_sensitivity": spatial_internal + sensitivity,
            "B7_all_runtime_spatial_sensitivity": hazy + scalar + spatial_hazy + spatial_internal + sensitivity,
        }
    )
    return groups


def labels(rows: list[dict[str, Any]], risk_threshold: float) -> np.ndarray:
    dpsnr = np.asarray([float(row["WD0375_dPSNR"]) for row in rows], dtype=np.float64)
    return (dpsnr <= risk_threshold).astype(np.float64)


def folds_for(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([int(row.get("oof_fold", idx % 5)) for idx, row in enumerate(rows)], dtype=np.int64)


def make_x(rows: list[dict[str, Any]], cols: list[str]) -> np.ndarray:
    return np.asarray([[float(row[col]) for col in cols] for row in rows], dtype=np.float64)


def standardize(train_x: np.ndarray, val_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    return (train_x - mean) / std, (val_x - mean) / std


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def fit_logistic(
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    steps: int,
    lr: float,
    l2: float,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    xb = np.concatenate([np.ones((x.shape[0], 1)), x], axis=1)
    w = rng.normal(0.0, 1e-4, size=xb.shape[1]).astype(np.float64)
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
        ranks[order[i : j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    rank_sum_pos = ranks[y == 1].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(y: np.ndarray, score: np.ndarray) -> float | None:
    y = y.astype(np.int64)
    n_pos = int(y.sum())
    if n_pos == 0:
        return None
    order = np.argsort(-score)
    y_sorted = y[order]
    precision = np.cumsum(y_sorted) / (np.arange(len(y_sorted)) + 1)
    return float((precision * y_sorted).sum() / n_pos)


def ece_score(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    ece = 0.0
    for b in range(bins):
        lo = b / bins
        hi = (b + 1) / bins
        mask = (p >= lo) & (p <= hi) if b == bins - 1 else (p >= lo) & (p < hi)
        if mask.any():
            ece += mask.mean() * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(ece)


def topk_stats(y: np.ndarray, score: np.ndarray, ks: tuple[int, ...] = (50, 100, 200)) -> dict[str, float]:
    out: dict[str, float] = {}
    total_positive = int(y.sum())
    base_rate = float(y.mean()) if len(y) else float("nan")
    order = np.argsort(-score)
    for k in ks:
        kk = min(k, len(y))
        hits = int(y[order[:kk]].sum())
        precision = hits / kk if kk else float("nan")
        out[f"top{k}_positive"] = float(hits)
        out[f"precision_at_{k}"] = precision
        out[f"recall_at_{k}"] = hits / total_positive if total_positive else float("nan")
        out[f"top{k}_enrichment"] = precision / base_rate if base_rate > 0 else float("nan")
    return out


def pr_curve_points(y: np.ndarray, score: np.ndarray, max_points: int = 200) -> list[dict[str, float]]:
    order = np.argsort(-score)
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    total_pos = max(1.0, float(y.sum()))
    precision = tp / np.maximum(1.0, tp + fp)
    recall = tp / total_pos
    n = len(y)
    if n > max_points:
        idx = sorted(set(int(round(v)) for v in np.linspace(0, n - 1, max_points)))
    else:
        idx = list(range(n))
    return [{"rank": int(i + 1), "precision": float(precision[i]), "recall": float(recall[i])} for i in idx]


def run_oof(
    rows: list[dict[str, Any]],
    cols: list[str],
    risk_threshold: float,
    seed: int,
    steps: int,
    lr: float,
    l2: float,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    y = labels(rows, risk_threshold)
    folds = folds_for(rows)
    x = make_x(rows, cols)
    pred = np.full(len(rows), np.nan, dtype=np.float64)
    fold_reports: list[dict[str, Any]] = []
    for fold in sorted(set(int(v) for v in folds)):
        train_mask = folds != fold
        val_mask = folds == fold
        if len(set(y[train_mask].astype(int))) < 2:
            baseline = float(y[train_mask].mean()) if train_mask.any() else 0.5
            pred[val_mask] = baseline
        else:
            train_x, val_x = standardize(x[train_mask], x[val_mask])
            w = fit_logistic(train_x, y[train_mask], seed=seed + fold, steps=steps, lr=lr, l2=l2)
            pred[val_mask] = predict_logistic(w, val_x)
        stats = topk_stats(y[val_mask], pred[val_mask], ks=(50, 100, 200))
        fold_reports.append(
            {
                "fold": fold,
                "seed": seed,
                "val_count": int(val_mask.sum()),
                "val_positive": int(y[val_mask].sum()),
                "roc_auc": auc_score(y[val_mask], pred[val_mask]),
                "pr_auc": average_precision(y[val_mask], pred[val_mask]),
                **stats,
            }
        )
    return pred, fold_reports


def aggregate_metric(rows: list[dict[str, Any]], score: np.ndarray, risk_threshold: float) -> dict[str, Any]:
    y = labels(rows, risk_threshold)
    out: dict[str, Any] = {
        "count": len(rows),
        "positive": int(y.sum()),
        "positive_rate": float(y.mean()),
        "roc_auc": auc_score(y, score),
        "pr_auc": average_precision(y, score),
        "ece": ece_score(y, score),
    }
    out.update(topk_stats(y, score))
    return out


def baseline_scores(rows: list[dict[str, Any]], group: str) -> np.ndarray:
    column = {
        "B0_hazy_runtime_v214": "severe_risk_label_hazy_runtime_pred",
        "B1_internal_scalar_v214": "severe_risk_label_internal_only_pred",
        "B2_all_scalar_v214": "severe_risk_label_all_runtime_pred",
    }[group]
    return np.asarray([float(row[column]) for row in rows], dtype=np.float64)


def bootstrap_delta(
    y: np.ndarray,
    candidate: np.ndarray,
    baseline: np.ndarray,
    *,
    seed: int,
    iterations: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    pr_deltas: list[float] = []
    top100_deltas: list[float] = []
    for _ in range(iterations):
        idx = rng.integers(0, len(y), size=len(y))
        cand_ap = average_precision(y[idx], candidate[idx])
        base_ap = average_precision(y[idx], baseline[idx])
        if cand_ap is None or base_ap is None:
            continue
        cand_top = topk_stats(y[idx], candidate[idx], ks=(100,))["top100_enrichment"]
        base_top = topk_stats(y[idx], baseline[idx], ks=(100,))["top100_enrichment"]
        pr_deltas.append(cand_ap - base_ap)
        top100_deltas.append(cand_top - base_top)
    def summarize(values: list[float]) -> dict[str, Any]:
        arr = np.asarray(values, dtype=np.float64)
        return {
            "n": int(len(arr)),
            "mean": float(arr.mean()) if len(arr) else None,
            "p05": float(np.quantile(arr, 0.05)) if len(arr) else None,
            "p50": float(np.quantile(arr, 0.50)) if len(arr) else None,
            "p95": float(np.quantile(arr, 0.95)) if len(arr) else None,
            "prob_delta_lt_0": float((arr < 0).mean()) if len(arr) else None,
        }
    return {"pr_auc_delta": summarize(pr_deltas), "top100_enrichment_delta": summarize(top100_deltas)}


def top100_decomposition(rows: list[dict[str, Any]], out_dir: Path) -> dict[str, Any]:
    y = labels(rows, -0.20)
    hazy = baseline_scores(rows, "B0_hazy_runtime_v214")
    all_runtime = baseline_scores(rows, "B2_all_scalar_v214")
    top_hazy = set(np.argsort(-hazy)[:100].tolist())
    top_all = set(np.argsort(-all_runtime)[:100].tolist())
    overlap = top_hazy.intersection(top_all)
    lost = sorted(top_hazy - top_all)
    gained = sorted(top_all - top_hazy)

    def case_row(idx: int) -> dict[str, Any]:
        row = rows[idx]
        return {
            "name": row["name"],
            "source_split": row.get("source_split", ""),
            "oof_fold": row.get("oof_fold", ""),
            "WD0375_dPSNR": row.get("WD0375_dPSNR", ""),
            "severe_risk_label": int(y[idx]),
            "hazy_runtime_score": float(hazy[idx]),
            "all_runtime_score": float(all_runtime[idx]),
            "score_delta_all_minus_hazy": float(all_runtime[idx] - hazy[idx]),
        }

    lost_rows = [case_row(i) for i in lost if y[i] > 0.5]
    gained_false_rows = [case_row(i) for i in gained if y[i] < 0.5]
    compare_rows = []
    for i in sorted(top_hazy.union(top_all)):
        item = case_row(i)
        item["in_hazy_top100"] = int(i in top_hazy)
        item["in_all_runtime_top100"] = int(i in top_all)
        compare_rows.append(item)
    swap_rows = []
    for i, row in enumerate(rows):
        item = case_row(i)
        item["hazy_rank"] = int(np.where(np.argsort(-hazy) == i)[0][0] + 1)
        item["all_runtime_rank"] = int(np.where(np.argsort(-all_runtime) == i)[0][0] + 1)
        item["rank_delta_all_minus_hazy"] = item["all_runtime_rank"] - item["hazy_rank"]
        swap_rows.append(item)
    write_csv(out_dir / "v215_s1_top100_hazy_vs_all_runtime.csv", compare_rows)
    write_csv(out_dir / "v215_s1_lost_severe_cases.csv", lost_rows)
    write_csv(out_dir / "v215_s1_gained_false_positive_cases.csv", gained_false_rows)
    write_csv(out_dir / "v215_s1_score_swap_analysis.csv", swap_rows)
    summary = {
        "hazy_top100_severe": int(y[list(top_hazy)].sum()),
        "all_runtime_top100_severe": int(y[list(top_all)].sum()),
        "top100_overlap": len(overlap),
        "lost_from_hazy_top100": len(lost),
        "gained_into_all_top100": len(gained),
        "lost_severe_count": len(lost_rows),
        "gained_false_positive_count": len(gained_false_rows),
    }
    write_markdown(
        out_dir / "v215_s1_top100_overlap_report.md",
        [
            "# v2.15 S1 Top-100 Failure Decomposition",
            "",
            f"- hazy-runtime top100 severe count: `{summary['hazy_top100_severe']}`",
            f"- all-runtime top100 severe count: `{summary['all_runtime_top100_severe']}`",
            f"- top100 overlap: `{summary['top100_overlap']}`",
            f"- lost severe cases: `{summary['lost_severe_count']}`",
            f"- gained false-positive cases: `{summary['gained_false_positive_count']}`",
            "",
            "Interpretation: all-runtime loses the v2.14 severe-risk top-tail gate when it admits fewer true severe cases than hazy-runtime.",
        ],
    )
    write_markdown(
        out_dir / "v215_s1_decision.md",
        [
            "# v2.15 S1 Decision",
            "",
            "Decision: `S1_DECOMPOSED_TOPTAIL_FAILURE_CONTINUE_S2_S3_S4`",
            "",
            "S1 is diagnostic-only and authorizes spatial/internal evidence audit, not training.",
        ],
    )
    return summary


def run_all_probes(
    rows: list[dict[str, Any]],
    groups: dict[str, list[str]],
    seeds: list[int],
    steps: int,
    lr: float,
    l2: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, np.ndarray], dict[str, list[dict[str, Any]]]]:
    metrics: list[dict[str, Any]] = []
    fold_seed_rows: list[dict[str, Any]] = []
    primary_predictions: dict[str, np.ndarray] = {}
    pr_curves: dict[str, list[dict[str, Any]]] = {}
    for threshold in RISK_THRESHOLDS:
        threshold_key = str(threshold).replace("-", "m").replace(".", "p")
        y = labels(rows, threshold)
        for group in GROUP_ORDER:
            if group in ("B0_hazy_runtime_v214", "B1_internal_scalar_v214", "B2_all_scalar_v214"):
                score = baseline_scores(rows, group)
                fold_reports = []
            else:
                seed_preds = []
                fold_reports = []
                for seed in seeds:
                    pred, fold_report = run_oof(rows, groups[group], threshold, seed, steps, lr, l2)
                    seed_preds.append(pred)
                    for item in fold_report:
                        fold_seed_rows.append(
                            {
                                "risk_threshold": threshold,
                                "group": group,
                                **item,
                            }
                        )
                    fold_reports.extend(fold_report)
                score = np.mean(np.stack(seed_preds, axis=0), axis=0)
            key = f"{threshold_key}:{group}"
            if threshold == -0.20:
                primary_predictions[group] = score
                pr_curves[group] = pr_curve_points(y, score)
            metric = {
                "risk_threshold": threshold,
                "threshold_key": threshold_key,
                "group": group,
                "feature_count": len(groups.get(group, [])),
                **aggregate_metric(rows, score, threshold),
            }
            metrics.append(metric)
    return metrics, fold_seed_rows, primary_predictions, pr_curves


def topk_rows_from_metrics(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        for k in (50, 100, 200):
            rows.append(
                {
                    "risk_threshold": metric["risk_threshold"],
                    "group": metric["group"],
                    "top_k": k,
                    "positive": metric["positive"],
                    "top_k_positive": metric[f"top{k}_positive"],
                    "precision": metric[f"precision_at_{k}"],
                    "recall": metric[f"recall_at_{k}"],
                    "enrichment": metric[f"top{k}_enrichment"],
                }
            )
    return rows


def overlap_rows(predictions: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    top_sets = {group: set(np.argsort(-score)[:100].tolist()) for group, score in predictions.items()}
    for left in GROUP_ORDER:
        if left not in top_sets:
            continue
        for right in GROUP_ORDER:
            if right not in top_sets or right <= left:
                continue
            inter = len(top_sets[left].intersection(top_sets[right]))
            out.append({"left": left, "right": right, "top100_overlap": inter, "jaccard": inter / (200 - inter)})
    return out


def decision(
    metrics: list[dict[str, Any]],
    fold_seed_rows: list[dict[str, Any]],
    bootstrap: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[str, dict[str, Any]]:
    def metric(group: str, threshold: float, key: str) -> float:
        for row in metrics:
            if row["group"] == group and abs(float(row["risk_threshold"]) - threshold) < 1e-9:
                return float(row[key])
        return float("nan")

    baseline = "B0_hazy_runtime_v214"
    candidates = [g for g in GROUP_ORDER if g not in ("B0_hazy_runtime_v214", "B1_internal_scalar_v214", "B2_all_scalar_v214")]
    best_group = max(candidates, key=lambda g: (metric(g, -0.20, "pr_auc"), metric(g, -0.20, "top100_enrichment")))
    pr_delta = metric(best_group, -0.20, "pr_auc") - metric(baseline, -0.20, "pr_auc")
    top100_delta = metric(best_group, -0.20, "top100_enrichment") - metric(baseline, -0.20, "top100_enrichment")
    top100_count_delta = metric(best_group, -0.20, "top100_positive") - metric(baseline, -0.20, "top100_positive")
    top50_delta = metric(best_group, -0.20, "top50_enrichment") - metric(baseline, -0.20, "top50_enrichment")
    boot_pr_p05 = bootstrap.get(best_group, {}).get("pr_auc_delta", {}).get("p05")
    thresholds_better = 0
    for threshold in RISK_THRESHOLDS:
        if (
            metric(best_group, threshold, "pr_auc") > metric(baseline, threshold, "pr_auc")
            or metric(best_group, threshold, "top100_enrichment") > metric(baseline, threshold, "top100_enrichment")
        ):
            thresholds_better += 1
    internal_spatial_gain = metric("B3_internal_spatial", -0.20, "pr_auc") - metric("B1_internal_scalar_v214", -0.20, "pr_auc")
    internal_sensitivity_gain = metric("B5_internal_sensitivity", -0.20, "pr_auc") - metric("B1_internal_scalar_v214", -0.20, "pr_auc")
    stable_units = 0
    total_units = 0
    baseline_top100 = metric(baseline, -0.20, "top100_enrichment")
    for row in fold_seed_rows:
        if row["group"] == best_group and abs(float(row["risk_threshold"]) + 0.20) < 1e-9:
            total_units += 1
            if float(row["top100_enrichment"]) >= baseline_top100:
                stable_units += 1
    gate_pass = (
        pr_delta >= args.pr_auc_delta_gate
        and boot_pr_p05 is not None
        and boot_pr_p05 >= 0
        and (top100_delta >= args.top100_enrichment_delta_gate or top100_count_delta >= 2)
        and top50_delta >= -args.top50_allowed_drop
        and thresholds_better >= 2
        and (metric(best_group, -0.20, "pr_auc") > metric(baseline, -0.20, "pr_auc") or metric(best_group, -0.20, "top100_enrichment") > metric(baseline, -0.20, "top100_enrichment"))
        and (internal_spatial_gain > 0 or internal_sensitivity_gain > 0)
        and stable_units >= 12
    )
    if gate_pass:
        label = "N1S_SPATIAL_INTERNAL_RISK_PASS_ALLOW_V216_DESIGN_REVIEW"
    elif internal_spatial_gain > 0 or internal_sensitivity_gain > 0:
        label = "N1S_PARTIAL_INTERNAL_SIGNAL_NO_TRAINING"
    else:
        label = "N1S_INTERNAL_EVIDENCE_FAIL_NO_TRAINING"
    detail = {
        "decision": label,
        "pass": gate_pass,
        "best_group": best_group,
        "baseline": baseline,
        "pr_delta": pr_delta,
        "top100_enrichment_delta": top100_delta,
        "top100_count_delta": top100_count_delta,
        "top50_enrichment_delta": top50_delta,
        "bootstrap_pr_p05": boot_pr_p05,
        "thresholds_better": thresholds_better,
        "internal_spatial_gain_vs_scalar_pr": internal_spatial_gain,
        "internal_sensitivity_gain_vs_scalar_pr": internal_sensitivity_gain,
        "stable_units": stable_units,
        "total_units": total_units,
        "locked_test_touched": False,
        "training_launched": False,
    }
    return label, detail


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v214-feature-table", type=Path, required=True)
    ap.add_argument("--v214-predictions", type=Path, required=True)
    ap.add_argument("--spatial-table", type=Path, required=True)
    ap.add_argument("--fam-table", type=Path, required=True)
    ap.add_argument("--skip-table", type=Path, required=True)
    ap.add_argument("--jitter-table", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seeds", default="3407,3411,2026")
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--bootstrap-iterations", type=int, default=1000)
    ap.add_argument("--pr-auc-delta-gate", type=float, default=0.015)
    ap.add_argument("--top100-enrichment-delta-gate", type=float, default=0.75)
    ap.add_argument("--top50-allowed-drop", type=float, default=0.25)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    scalar_rows = read_rows(args.v214_feature_table)
    pred_rows = read_rows(args.v214_predictions)
    pred_by_name = rows_by_name(pred_rows)
    for row in scalar_rows:
        row.update({k: v for k, v in pred_by_name[row["name"]].items() if k.endswith("_pred")})
    rows = merged_rows(
        scalar_rows,
        read_rows(args.spatial_table),
        read_rows(args.fam_table),
        read_rows(args.skip_table),
        read_rows(args.jitter_table),
    )
    groups = feature_groups(rows)
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    protocol = {
        "rows": len(rows),
        "primary_severe_label": "WD0375_dPSNR <= -0.2",
        "sensitivity_labels": ["WD0375_dPSNR <= -0.1", "WD0375_dPSNR <= -0.3"],
        "folds": "image-level grouped oof_fold from v2.13/v2.14 table",
        "repeat_seeds": seeds,
        "locked_touched": False,
        "training_launched": False,
        "groups": groups,
    }
    write_json(args.out_dir / "v215_s0_fold_manifest.json", protocol)
    write_markdown(
        args.out_dir / "v215_s0_protocol.md",
        [
            "# v2.15 S0 Protocol Lock",
            "",
            f"- rows: `{len(rows)}`",
            "- primary severe label: `WD0375_dPSNR <= -0.2`",
            "- sensitivity labels: `-0.1`, `-0.3`",
            "- folds: image-level grouped 5-fold from `oof_fold`",
            f"- repeat seeds: `{','.join(str(s) for s in seeds)}`",
            "- locked touched: `false`",
            "- training launched: `false`",
        ],
    )
    write_markdown(
        args.out_dir / "v215_s0_forbidden_feature_contract.md",
        [
            "# v2.15 Forbidden Feature Contract",
            "",
            "- no NoPost adapter training",
            "- no N3/N4/N5/N6/N7",
            "- no locked Haze4K",
            "- no `A0_output - hazy`",
            "- no `teacher_output - A0_output`",
            "- no `expert_output - anchor_output`",
            "- no RGB output-level correction",
        ],
    )
    (args.out_dir / "v215_s0_no_training_no_locked_status.txt").write_text(
        "locked_touched=false\ntraining_launched=false\n", encoding="utf-8"
    )

    s1_summary = top100_decomposition(rows, args.out_dir)
    metrics, fold_seed_rows, primary_predictions, curves = run_all_probes(
        rows, groups, seeds, args.steps, args.lr, args.l2
    )
    write_csv(args.out_dir / "v215_s4_oof_metrics.csv", metrics)
    write_csv(args.out_dir / "v215_s4_topk_enrichment.csv", topk_rows_from_metrics(metrics))
    write_csv(args.out_dir / "v215_s4_fold_seed_stability.csv", fold_seed_rows)
    write_csv(args.out_dir / "v215_s4_topk_overlap.csv", overlap_rows(primary_predictions))
    write_json(args.out_dir / "v215_s4_pr_curves.json", curves)
    write_csv(args.out_dir / "v215_s2_spatial_ablation_report.csv", [m for m in metrics if m["group"] in ("B1_internal_scalar_v214", "B3_internal_spatial", "B4_hazy_plus_internal_spatial")])
    write_csv(args.out_dir / "v215_s3_sensitivity_ablation_report.csv", [m for m in metrics if m["group"] in ("B1_internal_scalar_v214", "B5_internal_sensitivity", "B6_spatial_plus_sensitivity")])
    write_csv(args.out_dir / "v215_s2_spatial_label_sensitivity.csv", [m for m in metrics if m["group"] in ("B0_hazy_runtime_v214", "B3_internal_spatial", "B4_hazy_plus_internal_spatial")])

    prediction_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        out = {
            "name": row["name"],
            "source_split": row.get("source_split", ""),
            "oof_fold": row.get("oof_fold", ""),
            "WD0375_dPSNR": row.get("WD0375_dPSNR", ""),
            "severe_risk_label": int(labels([row], -0.20)[0]),
        }
        for group, score in primary_predictions.items():
            out[f"{group}_pred"] = float(score[idx])
        prediction_rows.append(out)
    write_csv(args.out_dir / "v215_s4_oof_predictions.csv", prediction_rows)

    y = labels(rows, -0.20)
    baseline = primary_predictions["B0_hazy_runtime_v214"]
    bootstrap: dict[str, Any] = {}
    for group, score in primary_predictions.items():
        if group == "B0_hazy_runtime_v214":
            continue
        bootstrap[group] = bootstrap_delta(
            y,
            score,
            baseline,
            seed=3407 + len(bootstrap),
            iterations=args.bootstrap_iterations,
        )
    write_json(args.out_dir / "v215_s4_bootstrap_delta.json", bootstrap)
    calibration = {
        group: {
            "ece": aggregate_metric(rows, score, -0.20)["ece"],
            "mean_score": float(score.mean()),
            "positive_mean_score": float(score[y > 0.5].mean()) if y.sum() else None,
            "negative_mean_score": float(score[y < 0.5].mean()) if (y < 0.5).any() else None,
        }
        for group, score in primary_predictions.items()
    }
    write_json(args.out_dir / "v215_s4_calibration_report.json", calibration)

    label, detail = decision(metrics, fold_seed_rows, bootstrap, args)
    detail["s1_summary"] = s1_summary
    write_json(args.out_dir / "v215_n1s_closeout.json", detail)
    write_markdown(
        args.out_dir / "v215_n1s_decision.md",
        [
            "# v2.15 N1S Spatial/Internal Risk Audit Decision",
            "",
            f"Decision: `{label}`",
            "",
            f"- best candidate: `{detail['best_group']}`",
            f"- baseline: `{detail['baseline']}`",
            f"- primary PR-AUC delta: `{detail['pr_delta']:.6f}`",
            f"- primary top100 enrichment delta: `{detail['top100_enrichment_delta']:.6f}`",
            f"- primary top100 severe-count delta: `{detail['top100_count_delta']:.0f}`",
            f"- top50 enrichment delta: `{detail['top50_enrichment_delta']:.6f}`",
            f"- bootstrap PR-AUC p05: `{detail['bootstrap_pr_p05']}`",
            f"- thresholds better: `{detail['thresholds_better']}` / `3`",
            f"- stable fold-seed units: `{detail['stable_units']}` / `{detail['total_units']}`",
            "- locked test touched: `false`",
            "- training launched: `false`",
            "",
            "No training is authorized unless this decision is a pass and a separate v2.16 design review is opened.",
        ],
    )
    print("V215_N1S_PASS" if detail["pass"] else "V215_N1S_FAIL")
    print(json.dumps(detail, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
