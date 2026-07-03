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


TARGETS = ("benefit_label", "severe_risk_label")
FORBIDDEN_RUNTIME_COLUMNS = {
    "hazy_PSNR",
    "A0_PSNR",
    "A0_SSIM",
    "WD0375_PSNR",
    "WD0375_SSIM",
    "WD0375_dPSNR",
    "WD0375_dSSIM",
}
FORBIDDEN_TOKENS = ("GT", "gt_", "teacher", "Teacher", "PSNR", "SSIM", "dPSNR", "dSSIM")
PRIMARY_GROUP_ORDER = (
    "hazy_leakcheck",
    "hazy_runtime",
    "internal_final",
    "internal_res",
    "internal_scm",
    "internal_only",
    "all_runtime",
    "all_with_leak",
)
SENSITIVITY_GROUPS = ("hazy_runtime", "internal_only", "all_runtime", "all_with_leak")


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def unique(seq: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in seq:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def has_forbidden_token(column: str) -> bool:
    return column in FORBIDDEN_RUNTIME_COLUMNS or any(token in column for token in FORBIDDEN_TOKENS)


def cols_with_prefix(columns: list[str], prefixes: tuple[str, ...]) -> list[str]:
    return [col for col in columns if col.startswith(prefixes)]


def build_feature_groups(rows: list[dict[str, Any]]) -> tuple[dict[str, list[str]], dict[str, Any]]:
    if not rows:
        raise ValueError("empty feature table")
    columns = list(rows[0].keys())
    hazy_all = cols_with_prefix(columns, ("hazy_",))
    leak = [col for col in ("hazy_PSNR",) if col in columns]
    hazy_runtime = [col for col in hazy_all if not has_forbidden_token(col)]
    internal_final = cols_with_prefix(columns, ("final_",))
    internal_res = cols_with_prefix(columns, ("res1_", "res2_"))
    internal_scm = cols_with_prefix(columns, ("scm2_", "scm4_"))
    internal_only = unique(internal_final + internal_res + internal_scm)
    groups = {
        "hazy_leakcheck": leak,
        "hazy_runtime": hazy_runtime,
        "internal_final": internal_final,
        "internal_res": internal_res,
        "internal_scm": internal_scm,
        "internal_only": internal_only,
        "all_runtime": unique(hazy_runtime + internal_only),
        "all_with_leak": unique(hazy_runtime + leak + internal_only),
    }
    forbidden_encountered = [
        col
        for col in columns
        if col in FORBIDDEN_RUNTIME_COLUMNS or any(token in col for token in ("GT", "teacher", "Teacher"))
    ]
    manifest = {
        "all_columns": columns,
        "forbidden_runtime_columns": sorted(FORBIDDEN_RUNTIME_COLUMNS),
        "forbidden_columns_encountered": sorted(forbidden_encountered),
        "forbidden_columns_excluded_from_runtime_groups": sorted(
            set(forbidden_encountered).intersection(FORBIDDEN_RUNTIME_COLUMNS)
        ),
        "leakcheck_columns": leak,
        "groups": groups,
        "locked_test_touched": False,
        "feature_table_rows": len(rows),
    }
    return groups, manifest


def make_y(rows: list[dict[str, Any]], target: str, benefit_threshold: float, risk_threshold: float) -> np.ndarray:
    dpsnr = np.asarray([float(row["WD0375_dPSNR"]) for row in rows], dtype=np.float64)
    if target == "benefit_label":
        return (dpsnr >= benefit_threshold).astype(np.float64)
    if target == "severe_risk_label":
        return (dpsnr <= risk_threshold).astype(np.float64)
    raise ValueError(target)


def make_x(rows: list[dict[str, Any]], cols: list[str]) -> np.ndarray:
    return np.asarray([[float(row[col]) for col in cols] for row in rows], dtype=np.float64)


def make_folds(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([int(row.get("oof_fold", idx % 5)) for idx, row in enumerate(rows)], dtype=np.int64)


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
    steps: int,
    lr: float,
    l2: float,
) -> np.ndarray:
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


def average_precision(y: np.ndarray, score: np.ndarray) -> float | None:
    y = y.astype(np.int64)
    n_pos = int(y.sum())
    if n_pos == 0:
        return None
    order = np.argsort(-score)
    y_sorted = y[order]
    precision = np.cumsum(y_sorted) / (np.arange(len(y_sorted)) + 1)
    return float((precision * y_sorted).sum() / n_pos)


def ece_score(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float | None:
    if len(y) == 0:
        return None
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


def nullable_float(value: float | None) -> float:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return float("nan")
    return float(value)


def run_oof(
    rows: list[dict[str, Any]],
    cols: list[str],
    target: str,
    benefit_threshold: float,
    risk_threshold: float,
    *,
    steps: int,
    lr: float,
    l2: float,
) -> tuple[dict[str, Any], np.ndarray]:
    y = make_y(rows, target, benefit_threshold, risk_threshold)
    folds = make_folds(rows)
    if not cols:
        pred = np.full(len(rows), float(y.mean()) if len(rows) else 0.5, dtype=np.float64)
        return (
            {
                "target": target,
                "threshold": benefit_threshold if target == "benefit_label" else risk_threshold,
                "feature_count": 0,
                "count": len(rows),
                "positive": int(y.sum()),
                "positive_rate": float(y.mean()) if len(y) else float("nan"),
                "roc_auc": None,
                "pr_auc": None,
                "ece": ece_score(y, pred),
                "status": "skip_no_features",
                "folds": [],
            },
            pred,
        )
    x = make_x(rows, cols)
    pred = np.full(len(rows), np.nan, dtype=np.float64)
    fold_reports = []
    for fold in sorted(set(int(v) for v in folds)):
        train_mask = folds != fold
        val_mask = folds == fold
        if len(set(y[train_mask].astype(int))) < 2:
            baseline = float(y[train_mask].mean()) if train_mask.any() else 0.5
            pred[val_mask] = baseline
            fold_reports.append(
                {
                    "fold": fold,
                    "status": "skip_one_class_train",
                    "val_count": int(val_mask.sum()),
                    "val_positive": int(y[val_mask].sum()),
                }
            )
            continue
        train_x, val_x = standardize(x[train_mask], x[val_mask])
        w = fit_logistic(train_x, y[train_mask], steps=steps, lr=lr, l2=l2)
        pred[val_mask] = predict_logistic(w, val_x)
        fold_reports.append(
            {
                "fold": fold,
                "status": "ok",
                "val_count": int(val_mask.sum()),
                "val_positive": int(y[val_mask].sum()),
                "auc": auc_score(y[val_mask], pred[val_mask]),
                "pr_auc": average_precision(y[val_mask], pred[val_mask]),
            }
        )
    report = {
        "target": target,
        "threshold": benefit_threshold if target == "benefit_label" else risk_threshold,
        "feature_count": len(cols),
        "count": len(rows),
        "positive": int(y.sum()),
        "positive_rate": float(y.mean()) if len(y) else float("nan"),
        "roc_auc": auc_score(y, pred),
        "pr_auc": average_precision(y, pred),
        "ece": ece_score(y, pred),
        "status": "ok",
        "folds": fold_reports,
    }
    return report, pred


def topk_enrichment(
    rows: list[dict[str, Any]],
    predictions: dict[tuple[str, str], np.ndarray],
    groups: tuple[str, ...],
    ks: list[int],
    risk_threshold: float,
) -> list[dict[str, Any]]:
    y = make_y(rows, "severe_risk_label", benefit_threshold=0.0, risk_threshold=risk_threshold)
    total_positive = int(y.sum())
    base_rate = float(y.mean()) if len(y) else float("nan")
    out: list[dict[str, Any]] = []
    for group in groups:
        pred = predictions.get(("severe_risk_label", group))
        if pred is None:
            continue
        order = np.argsort(-pred)
        for k0 in ks:
            k = min(k0, len(rows))
            top = order[:k]
            hits = int(y[top].sum())
            top_rate = hits / k if k else float("nan")
            out.append(
                {
                    "target": "severe_risk_label",
                    "risk_threshold": risk_threshold,
                    "group": group,
                    "top_k": k,
                    "total_positive": total_positive,
                    "base_rate": base_rate,
                    "top_k_positive": hits,
                    "top_k_rate": top_rate,
                    "capture_rate": hits / total_positive if total_positive else float("nan"),
                    "enrichment": top_rate / base_rate if base_rate > 0 else float("nan"),
                }
            )
    return out


def bootstrap_delta_auc(
    y: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    *,
    iterations: int,
    seed: int,
    margin: float,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    n = len(y)
    for _ in range(iterations):
        idx = rng.integers(0, n, size=n)
        auc_left = auc_score(y[idx], left[idx])
        auc_right = auc_score(y[idx], right[idx])
        if auc_left is None or auc_right is None:
            continue
        deltas.append(float(auc_left - auc_right))
    if not deltas:
        return {
            "n": 0,
            "mean": None,
            "p05": None,
            "p50": None,
            "p95": None,
            "prob_delta_lt_0": None,
            "prob_delta_lt_minus_margin": None,
        }
    arr = np.asarray(deltas, dtype=np.float64)
    return {
        "n": int(len(arr)),
        "mean": float(arr.mean()),
        "p05": float(np.quantile(arr, 0.05)),
        "p50": float(np.quantile(arr, 0.50)),
        "p95": float(np.quantile(arr, 0.95)),
        "prob_delta_lt_0": float((arr < 0).mean()),
        "prob_delta_lt_minus_margin": float((arr < -margin).mean()),
    }


def metric_lookup(reports: list[dict[str, Any]], target: str, group: str, key: str) -> float:
    for report in reports:
        if report["target"] == target and report["group"] == group:
            return nullable_float(report.get(key))
    return float("nan")


def write_leakage_report(path: Path, manifest: dict[str, Any], reports: list[dict[str, Any]]) -> None:
    lines = [
        "# v2.14 N1R Leakage Report",
        "",
        "This replay separates GT/teacher-derived columns from runtime-available features.",
        "",
        "## Excluded Runtime Columns",
        "",
    ]
    excluded = manifest["forbidden_columns_excluded_from_runtime_groups"]
    if excluded:
        lines.extend([f"- `{col}`" for col in excluded])
    else:
        lines.append("- none encountered")
    lines.extend(["", "## Leakcheck Metrics", ""])
    for target in TARGETS:
        leak_auc = metric_lookup(reports, target, "hazy_leakcheck", "roc_auc")
        hazy_auc = metric_lookup(reports, target, "hazy_runtime", "roc_auc")
        all_runtime_auc = metric_lookup(reports, target, "all_runtime", "roc_auc")
        all_with_leak_auc = metric_lookup(reports, target, "all_with_leak", "roc_auc")
        lines.extend(
            [
                f"- `{target}` leakcheck `hazy_PSNR` ROC-AUC: `{leak_auc:.6f}`",
                f"- `{target}` hazy-runtime ROC-AUC: `{hazy_auc:.6f}`",
                f"- `{target}` all-runtime ROC-AUC: `{all_runtime_auc:.6f}`",
                f"- `{target}` all-with-leak ROC-AUC: `{all_with_leak_auc:.6f}`",
            ]
        )
    lines.extend(
        [
            "",
            "Conclusion: `hazy_PSNR` is treated only as an oracle-leak sentinel. "
            "It is excluded from `hazy_runtime`, `all_runtime`, and all pass/fail gates.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_decision(
    path: Path,
    reports: list[dict[str, Any]],
    topk_rows: list[dict[str, Any]],
    bootstrap: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[str, bool]:
    benefit_all_auc = metric_lookup(reports, "benefit_label", "all_runtime", "roc_auc")
    risk_all_auc = metric_lookup(reports, "severe_risk_label", "all_runtime", "roc_auc")
    risk_internal_auc = metric_lookup(reports, "severe_risk_label", "internal_only", "roc_auc")
    risk_hazy_auc = metric_lookup(reports, "severe_risk_label", "hazy_runtime", "roc_auc")
    benefit_internal_auc = metric_lookup(reports, "benefit_label", "internal_only", "roc_auc")
    benefit_hazy_auc = metric_lookup(reports, "benefit_label", "hazy_runtime", "roc_auc")
    risk_all_ap = metric_lookup(reports, "severe_risk_label", "all_runtime", "pr_auc")
    risk_hazy_ap = metric_lookup(reports, "severe_risk_label", "hazy_runtime", "pr_auc")
    risk_ap_delta = risk_all_ap - risk_hazy_ap

    def enrich(group: str, k: int) -> float:
        for row in topk_rows:
            if row["group"] == group and int(row["top_k"]) == k:
                return nullable_float(row.get("enrichment"))
        return float("nan")

    risk_topk_delta = enrich("all_runtime", args.decision_top_k) - enrich("hazy_runtime", args.decision_top_k)

    boot_bad = []
    for target in TARGETS:
        key = f"{target}:internal_only-minus-hazy_runtime"
        item = bootstrap.get(key, {})
        p95 = item.get("p95")
        if p95 is not None and p95 < -args.internal_margin:
            boot_bad.append({"target": target, "p95": p95})

    gate_pass = (
        benefit_all_auc >= args.benefit_auc_gate
        and risk_all_auc >= args.risk_auc_gate
        and risk_internal_auc >= risk_hazy_auc - args.internal_margin
        and (risk_ap_delta > 0 or risk_topk_delta > 0)
        and not boot_bad
    )

    leak_risk_auc = metric_lookup(reports, "severe_risk_label", "hazy_leakcheck", "roc_auc")
    leak_all_auc = metric_lookup(reports, "severe_risk_label", "all_with_leak", "roc_auc")
    leak_strong = (leak_risk_auc > risk_hazy_auc + 0.05) or (leak_all_auc > risk_all_auc + 0.05)
    if gate_pass:
        decision = "N1R_RUNTIME_EVIDENCE_PASS_ALLOW_N3_DESIGN_REVIEW"
        recommendation = "Do not launch training in this replay step; proceed to explicit N3 design review."
    elif leak_strong:
        decision = "N1R_RUNTIME_EVIDENCE_FAIL_LEAK_DOMINANT_RECOMMEND_N1S_NO_TRAINING"
        recommendation = "Run an N1S spatial/internal expansion diagnostic before any adapter training."
    else:
        decision = "N1R_RUNTIME_EVIDENCE_FAIL_INSUFFICIENT_NO_TRAINING"
        recommendation = "Current runtime-valid evidence is insufficient; do not train this route."

    lines = [
        "# v2.14 N1R Runtime-Valid Evidence Decision",
        "",
        f"Decision: `{decision}`",
        "",
        "Locked Haze4K test touched: `false`",
        "",
        "## Primary Gates",
        "",
        f"- benefit all-runtime ROC-AUC: `{benefit_all_auc:.6f}` (gate >= `{args.benefit_auc_gate}`)",
        f"- severe-risk all-runtime ROC-AUC: `{risk_all_auc:.6f}` (gate >= `{args.risk_auc_gate}`)",
        f"- benefit internal/runtime-hazy ROC-AUC: `{benefit_internal_auc:.6f}` / `{benefit_hazy_auc:.6f}`",
        f"- severe-risk internal/runtime-hazy ROC-AUC: `{risk_internal_auc:.6f}` / `{risk_hazy_auc:.6f}`",
        f"- severe-risk all-runtime minus runtime-hazy PR-AUC: `{risk_ap_delta:.6f}`",
        f"- severe-risk all-runtime minus runtime-hazy top-{args.decision_top_k} enrichment: `{risk_topk_delta:.6f}`",
        f"- bootstrap worse-than-margin findings: `{len(boot_bad)}`",
        "",
        "## Recommendation",
        "",
        recommendation,
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return decision, gate_pass


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-table", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--benefit-threshold", type=float, default=0.05)
    ap.add_argument("--risk-threshold", type=float, default=-0.20)
    ap.add_argument("--benefit-sensitivity", default="0.05,0.10,0.20")
    ap.add_argument("--risk-sensitivity", default="-0.10,-0.20,-0.30")
    ap.add_argument("--benefit-auc-gate", type=float, default=0.70)
    ap.add_argument("--risk-auc-gate", type=float, default=0.70)
    ap.add_argument("--internal-margin", type=float, default=0.01)
    ap.add_argument("--top-k", default="20,50,100,200")
    ap.add_argument("--decision-top-k", type=int, default=100)
    ap.add_argument("--bootstrap-iterations", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--fail-on-gate-fail", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.feature_table)
    groups, manifest = build_feature_groups(rows)
    manifest.update(
        {
            "feature_table": str(args.feature_table),
            "primary_labels": {
                "benefit_label": f"WD0375_dPSNR >= {args.benefit_threshold}",
                "severe_risk_label": f"WD0375_dPSNR <= {args.risk_threshold}",
            },
            "gates": {
                "benefit_auc_gate": args.benefit_auc_gate,
                "risk_auc_gate": args.risk_auc_gate,
                "internal_margin": args.internal_margin,
                "decision_top_k": args.decision_top_k,
            },
        }
    )
    write_json(args.out_dir / "v214_n1r_runtime_feature_manifest.json", manifest)

    reports: list[dict[str, Any]] = []
    predictions: dict[tuple[str, str], np.ndarray] = {}
    for target in TARGETS:
        for group in PRIMARY_GROUP_ORDER:
            report, pred = run_oof(
                rows,
                groups[group],
                target,
                args.benefit_threshold,
                args.risk_threshold,
                steps=args.steps,
                lr=args.lr,
                l2=args.l2,
            )
            report["group"] = group
            predictions[(target, group)] = pred
            reports.append(report)

    metrics_rows = []
    for report in reports:
        row = {k: v for k, v in report.items() if k != "folds"}
        row["folds_json"] = json.dumps(report["folds"], sort_keys=True)
        metrics_rows.append(row)
    write_csv(args.out_dir / "v214_n1r_oof_metrics.csv", metrics_rows)

    prediction_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        out = {
            "name": row.get("name", ""),
            "source_split": row.get("source_split", ""),
            "oof_fold": row.get("oof_fold", ""),
            "WD0375_dPSNR": row.get("WD0375_dPSNR", ""),
            "benefit_label": int(make_y([row], "benefit_label", args.benefit_threshold, args.risk_threshold)[0]),
            "severe_risk_label": int(make_y([row], "severe_risk_label", args.benefit_threshold, args.risk_threshold)[0]),
        }
        for target in TARGETS:
            for group in PRIMARY_GROUP_ORDER:
                out[f"{target}_{group}_pred"] = float(predictions[(target, group)][idx])
        prediction_rows.append(out)
    write_csv(args.out_dir / "v214_n1r_oof_predictions.csv", prediction_rows)

    topk_rows = topk_enrichment(
        rows,
        predictions,
        PRIMARY_GROUP_ORDER,
        parse_int_list(args.top_k),
        args.risk_threshold,
    )
    write_csv(args.out_dir / "v214_n1r_topk_risk_enrichment.csv", topk_rows)

    ablation_rows = [
        {k: v for k, v in report.items() if k != "folds"}
        for report in reports
        if report["group"] in ("internal_final", "internal_res", "internal_scm", "internal_only", "hazy_runtime", "all_runtime")
    ]
    write_csv(args.out_dir / "v214_n1r_internal_block_ablation.csv", ablation_rows)

    sensitivity_rows: list[dict[str, Any]] = []
    for threshold in parse_float_list(args.benefit_sensitivity):
        for group in SENSITIVITY_GROUPS:
            report, _ = run_oof(
                rows,
                groups[group],
                "benefit_label",
                threshold,
                args.risk_threshold,
                steps=args.steps,
                lr=args.lr,
                l2=args.l2,
            )
            report["group"] = group
            report["target_kind"] = "benefit"
            sensitivity_rows.append({k: v for k, v in report.items() if k != "folds"})
    for threshold in parse_float_list(args.risk_sensitivity):
        for group in SENSITIVITY_GROUPS:
            report, _ = run_oof(
                rows,
                groups[group],
                "severe_risk_label",
                args.benefit_threshold,
                threshold,
                steps=args.steps,
                lr=args.lr,
                l2=args.l2,
            )
            report["group"] = group
            report["target_kind"] = "severe_risk"
            sensitivity_rows.append({k: v for k, v in report.items() if k != "folds"})
    write_csv(args.out_dir / "v214_n1r_label_sensitivity.csv", sensitivity_rows)

    bootstrap: dict[str, Any] = {}
    comparisons = (
        ("internal_only", "hazy_runtime"),
        ("all_runtime", "hazy_runtime"),
        ("all_with_leak", "all_runtime"),
        ("all_with_leak", "hazy_runtime"),
    )
    for target in TARGETS:
        y = make_y(rows, target, args.benefit_threshold, args.risk_threshold)
        for left, right in comparisons:
            key = f"{target}:{left}-minus-{right}"
            bootstrap[key] = bootstrap_delta_auc(
                y,
                predictions[(target, left)],
                predictions[(target, right)],
                iterations=args.bootstrap_iterations,
                seed=args.seed + len(bootstrap),
                margin=args.internal_margin,
            )
    bootstrap["settings"] = {
        "iterations": args.bootstrap_iterations,
        "seed": args.seed,
        "margin": args.internal_margin,
    }
    write_json(args.out_dir / "v214_n1r_delta_auc_bootstrap.json", bootstrap)

    write_leakage_report(args.out_dir / "v214_n1r_leakage_report.md", manifest, reports)
    decision, gate_pass = write_decision(
        args.out_dir / "v214_n1r_decision.md",
        reports,
        topk_rows,
        bootstrap,
        args,
    )

    closeout = {
        "decision": decision,
        "pass": gate_pass,
        "locked_test_touched": False,
        "primary_metrics": {
            "benefit_all_runtime_auc": metric_lookup(reports, "benefit_label", "all_runtime", "roc_auc"),
            "risk_all_runtime_auc": metric_lookup(reports, "severe_risk_label", "all_runtime", "roc_auc"),
            "benefit_internal_auc": metric_lookup(reports, "benefit_label", "internal_only", "roc_auc"),
            "benefit_hazy_runtime_auc": metric_lookup(reports, "benefit_label", "hazy_runtime", "roc_auc"),
            "risk_internal_auc": metric_lookup(reports, "severe_risk_label", "internal_only", "roc_auc"),
            "risk_hazy_runtime_auc": metric_lookup(reports, "severe_risk_label", "hazy_runtime", "roc_auc"),
        },
    }
    write_json(args.out_dir / "v214_n1r_closeout.json", closeout)
    print("V214_N1R_GATE_PASS" if gate_pass else "V214_N1R_GATE_FAIL")
    print(json.dumps(closeout, indent=2, sort_keys=True))
    if args.fail_on_gate_fail and not gate_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
