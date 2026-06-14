#!/usr/bin/env python3
"""C2b leakage-safe multi-rule router screen from C2 output-diff features.

This script consumes the C2 per-image feature CSV produced on convir-4090 and
searches transparent one- and two-condition abstaining policies with 5-fold OOF
replay. It writes text/CSV/JSON evidence only; no images or tensors are read.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np


POLICY_FEATURES = [
    "input_mean",
    "input_std",
    "input_grad_mean",
    "input_dark_mean",
    "depth_mean",
    "depth_std",
    "depth_grad_mean",
    "a0_mean",
    "a0_std",
    "a0_grad_mean",
    "a0_saturation_high",
    "a0_saturation_low",
    "udp_mean",
    "udp_std",
    "udp_grad_mean",
    "diff_signed_mean",
    "diff_abs_mean",
    "diff_abs_std",
    "diff_abs_p50",
    "diff_abs_p90",
    "diff_abs_p95",
    "diff_abs_max",
    "diff_grad_mean",
    "diff_to_a0_ratio",
    "a0_udp_psnr",
]

STRICT_GATE = {
    "mean_dPSNR": 0.12,
    "hard_bottom25_dPSNR": 0.20,
    "easy_top25_dPSNR": -0.02,
    "dSSIM": 0.0,
    "positive_ratio": 0.65,
    "severe_loss_per_600": 48.0,
}

ABSTENTION_GATE = {
    "mean_dPSNR": 0.12,
    "hard_bottom25_dPSNR": 0.20,
    "easy_top25_dPSNR": -0.02,
    "dSSIM": 0.0,
    "selected_precision": 0.65,
    "nonnegative_ratio": 0.90,
    "severe_loss_per_600": 48.0,
    "coverage": 0.10,
}

QUANTILES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.33, 0.40, 0.50, 0.60, 0.67, 0.75, 0.80, 0.85, 0.90, 0.95]


def fnum(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        out = float(value)
        return out if math.isfinite(out) else default
    except (TypeError, ValueError):
        return default


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fold_id(image_id: str, folds: int = 5) -> int:
    digest = hashlib.sha1(image_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % folds


def strict_gate_pass(row: dict[str, Any]) -> bool:
    return (
        fnum(row.get("mean_dPSNR")) >= STRICT_GATE["mean_dPSNR"]
        and fnum(row.get("hard_bottom25_dPSNR")) >= STRICT_GATE["hard_bottom25_dPSNR"]
        and fnum(row.get("easy_top25_dPSNR")) >= STRICT_GATE["easy_top25_dPSNR"]
        and fnum(row.get("dSSIM")) >= STRICT_GATE["dSSIM"]
        and fnum(row.get("positive_ratio")) >= STRICT_GATE["positive_ratio"]
        and fnum(row.get("severe_loss_per_600")) <= STRICT_GATE["severe_loss_per_600"]
    )


def abstention_gate_pass(row: dict[str, Any]) -> bool:
    return (
        fnum(row.get("mean_dPSNR")) >= ABSTENTION_GATE["mean_dPSNR"]
        and fnum(row.get("hard_bottom25_dPSNR")) >= ABSTENTION_GATE["hard_bottom25_dPSNR"]
        and fnum(row.get("easy_top25_dPSNR")) >= ABSTENTION_GATE["easy_top25_dPSNR"]
        and fnum(row.get("dSSIM")) >= ABSTENTION_GATE["dSSIM"]
        and fnum(row.get("selected_precision")) >= ABSTENTION_GATE["selected_precision"]
        and fnum(row.get("nonnegative_ratio")) >= ABSTENTION_GATE["nonnegative_ratio"]
        and fnum(row.get("severe_loss_per_600")) <= ABSTENTION_GATE["severe_loss_per_600"]
        and fnum(row.get("coverage")) >= ABSTENTION_GATE["coverage"]
    )


def score(row: dict[str, Any]) -> float:
    easy = fnum(row.get("easy_top25_dPSNR"))
    dssim = fnum(row.get("dSSIM"))
    selected_nonnegative = fnum(row.get("selected_nonnegative_ratio"), 1.0)
    severe = fnum(row.get("severe_loss_per_600"))
    penalty = 0.0
    if easy < ABSTENTION_GATE["easy_top25_dPSNR"]:
        penalty += 2.0 * (ABSTENTION_GATE["easy_top25_dPSNR"] - easy)
    if dssim < 0.0:
        penalty += 10.0 * (0.0 - dssim)
    if selected_nonnegative < 0.90:
        penalty += 0.5 * (0.90 - selected_nonnegative)
    return (
        fnum(row.get("mean_dPSNR"))
        + 0.25 * fnum(row.get("hard_bottom25_dPSNR"))
        + 0.15 * easy
        + 0.05 * fnum(row.get("selected_precision"))
        + 0.02 * fnum(row.get("coverage"))
        - 0.002 * severe
        - penalty
    )


class Table:
    def __init__(self, rows: list[dict[str, Any]], features: list[str]):
        self.rows = rows
        self.names = np.array([str(row["name"]) for row in rows])
        self.dpsnr = np.array([fnum(row["dPSNR"]) for row in rows], dtype=np.float64)
        self.dssim = np.array([fnum(row["dSSIM"]) for row in rows], dtype=np.float64)
        self.a0_psnr = np.array([fnum(row["A0_PSNR"]) for row in rows], dtype=np.float64)
        self.features = {feature: np.array([fnum(row[feature]) for row in rows], dtype=np.float64) for feature in features}

    def subset(self, mask: np.ndarray) -> "Table":
        idx = np.asarray(mask, dtype=bool)
        rows = [row for row, keep in zip(self.rows, idx, strict=False) if keep]
        return Table(rows, list(self.features.keys()))

    def mask_for_policy(self, policy_id: str) -> np.ndarray:
        if policy_id == "a0_anchor":
            return np.zeros(len(self.rows), dtype=bool)
        if policy_id == "all_fulludp":
            return np.ones(len(self.rows), dtype=bool)
        mask = np.ones(len(self.rows), dtype=bool)
        for condition in policy_id.split("_AND_"):
            if "_le_" in condition:
                feature, threshold = condition.rsplit("_le_", 1)
                mask &= self.features[feature] <= float(threshold)
            elif "_ge_" in condition:
                feature, threshold = condition.rsplit("_ge_", 1)
                mask &= self.features[feature] >= float(threshold)
            else:
                return np.zeros(len(self.rows), dtype=bool)
        return mask


def summarize_mask(table: Table, mask: np.ndarray) -> dict[str, Any]:
    count = len(table.rows)
    if count == 0:
        return {
            "count": 0,
            "selected_count": 0,
            "coverage": 0.0,
            "mean_dPSNR": 0.0,
            "hard_bottom25_dPSNR": 0.0,
            "easy_top25_dPSNR": 0.0,
            "dSSIM": 0.0,
            "positive_ratio": 0.0,
            "nonnegative_ratio": 0.0,
            "severe_loss_count": 0,
            "severe_loss_per_600": 0.0,
            "strong_loss_count": 0,
            "strong_loss_per_600": 0.0,
            "selected_precision": 0.0,
            "selected_nonnegative_ratio": 1.0,
            "selected_severe_count": 0,
        }
    selected = np.asarray(mask, dtype=bool)
    selected_count = int(selected.sum())
    deltas = np.where(selected, table.dpsnr, 0.0)
    ssims = np.where(selected, table.dssim, 0.0)
    order = np.argsort(table.a0_psnr)
    bucket = max(1, count // 4)
    severe = int(np.sum(deltas <= -0.20))
    strong = int(np.sum(deltas <= -0.05))
    selected_deltas = table.dpsnr[selected]
    selected_positive = int(np.sum(selected_deltas > 0.0)) if selected_count else 0
    selected_nonnegative = int(np.sum(selected_deltas >= 0.0)) if selected_count else 0
    selected_severe = int(np.sum(selected_deltas <= -0.20)) if selected_count else 0
    return {
        "count": count,
        "selected_count": selected_count,
        "coverage": selected_count / count,
        "mean_dPSNR": float(np.mean(deltas)),
        "hard_bottom25_dPSNR": float(np.mean(deltas[order[:bucket]])),
        "easy_top25_dPSNR": float(np.mean(deltas[order[-bucket:]])),
        "dSSIM": float(np.mean(ssims)),
        "positive_ratio": float(np.mean(deltas > 0.0)),
        "nonnegative_ratio": float(np.mean(deltas >= 0.0)),
        "severe_loss_count": severe,
        "severe_loss_per_600": severe / count * 600.0,
        "strong_loss_count": strong,
        "strong_loss_per_600": strong / count * 600.0,
        "selected_precision": selected_positive / selected_count if selected_count else 0.0,
        "selected_nonnegative_ratio": selected_nonnegative / selected_count if selected_count else 1.0,
        "selected_severe_count": selected_severe,
    }


def condition_masks(table: Table, features: list[str], quantiles: list[float]) -> list[tuple[str, str, np.ndarray]]:
    out: list[tuple[str, str, np.ndarray]] = []
    for feature in features:
        vals = table.features[feature]
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            continue
        thresholds = sorted({float(np.quantile(finite, q)) for q in quantiles})
        for threshold in thresholds:
            out.append((feature, f"{feature}_le_{threshold:.8g}", vals <= threshold))
            out.append((feature, f"{feature}_ge_{threshold:.8g}", vals >= threshold))
    return out


def add_policy(rows: list[dict[str, Any]], table: Table, policy_id: str, mask: np.ndarray, complexity: int) -> None:
    rec = {"policy_id": policy_id, "complexity": complexity}
    rec.update(summarize_mask(table, mask))
    rec["strict_gate_pass"] = strict_gate_pass(rec)
    rec["abstention_gate_pass"] = abstention_gate_pass(rec)
    rec["score"] = score(rec)
    rows.append(rec)


def policy_grid(table: Table, features: list[str], quantiles: list[float], top_k: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n = len(table.rows)
    add_policy(rows, table, "a0_anchor", np.zeros(n, dtype=bool), 0)
    add_policy(rows, table, "all_fulludp", np.ones(n, dtype=bool), 0)

    conditions = condition_masks(table, features, quantiles)
    for _feature, policy_id, mask in conditions:
        if mask.mean() >= 0.08:
            add_policy(rows, table, policy_id, mask, 1)

    for i, (feature_i, policy_i, mask_i) in enumerate(conditions):
        for feature_j, policy_j, mask_j in conditions[i + 1 :]:
            if feature_i == feature_j:
                continue
            mask = mask_i & mask_j
            coverage = float(mask.mean())
            if coverage < 0.08 or coverage > 0.80:
                continue
            add_policy(rows, table, f"{policy_i}_AND_{policy_j}", mask, 2)

    rows.sort(key=lambda row: (bool(row["strict_gate_pass"]), bool(row["abstention_gate_pass"]), fnum(row["score"])), reverse=True)
    keep: list[dict[str, Any]] = []
    for row in rows:
        if len(keep) < top_k or row["strict_gate_pass"] or row["abstention_gate_pass"]:
            keep.append(row)
    return keep


def choose_policy(table: Table, features: list[str], quantiles: list[float]) -> dict[str, Any]:
    grid = policy_grid(table, features, quantiles, top_k=50)
    strict = [row for row in grid if row["strict_gate_pass"]]
    if strict:
        return strict[0]
    abstention = [row for row in grid if row["abstention_gate_pass"]]
    if abstention:
        return abstention[0]
    return grid[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_rows", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--top_k", type=int, default=500)
    args = parser.parse_args()

    rows = read_csv(args.feature_rows)
    table = Table(rows, POLICY_FEATURES)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    policies = policy_grid(table, POLICY_FEATURES, QUANTILES, top_k=args.top_k)
    fields = [
        "policy_id",
        "complexity",
        "count",
        "selected_count",
        "coverage",
        "mean_dPSNR",
        "hard_bottom25_dPSNR",
        "easy_top25_dPSNR",
        "dSSIM",
        "positive_ratio",
        "nonnegative_ratio",
        "severe_loss_count",
        "severe_loss_per_600",
        "strong_loss_count",
        "strong_loss_per_600",
        "selected_precision",
        "selected_nonnegative_ratio",
        "selected_severe_count",
        "strict_gate_pass",
        "abstention_gate_pass",
        "score",
    ]
    write_csv(args.out_dir / "v20_c2b_multirule_policy_grid.csv", policies, fields)

    fold_rows: list[dict[str, Any]] = []
    selected_oof: set[str] = set()
    fold_ids = np.array([fold_id(str(name)) for name in table.names], dtype=np.int64)
    for fold in range(5):
        train_table = table.subset(fold_ids != fold)
        heldout_table = table.subset(fold_ids == fold)
        chosen = choose_policy(train_table, POLICY_FEATURES, QUANTILES)
        heldout_mask = heldout_table.mask_for_policy(str(chosen["policy_id"]))
        eval_rec: dict[str, Any] = {
            "fold": fold,
            "train_policy_id": chosen["policy_id"],
            "train_complexity": chosen.get("complexity", ""),
            "train_strict_gate_pass": chosen["strict_gate_pass"],
            "train_abstention_gate_pass": chosen["abstention_gate_pass"],
        }
        eval_rec.update(summarize_mask(heldout_table, heldout_mask))
        eval_rec["strict_gate_pass"] = strict_gate_pass(eval_rec)
        eval_rec["abstention_gate_pass"] = abstention_gate_pass(eval_rec)
        eval_rec["score"] = score(eval_rec)
        fold_rows.append(eval_rec)
        for name, selected in zip(heldout_table.names, heldout_mask, strict=False):
            if bool(selected):
                selected_oof.add(str(name))
    write_csv(
        args.out_dir / "v20_c2b_oof_fold_metrics.csv",
        fold_rows,
        ["fold", "train_policy_id", "train_complexity", "train_strict_gate_pass", "train_abstention_gate_pass"] + fields[2:],
    )

    oof_mask = np.array([str(name) in selected_oof for name in table.names], dtype=bool)
    oof_summary = summarize_mask(table, oof_mask)
    oof_summary["strict_gate_pass"] = strict_gate_pass(oof_summary)
    oof_summary["abstention_gate_pass"] = abstention_gate_pass(oof_summary)
    oof_summary["score"] = score(oof_summary)

    strict_pass = [row for row in policies if row["strict_gate_pass"]]
    abstention_pass = [row for row in policies if row["abstention_gate_pass"]]
    if oof_summary["strict_gate_pass"]:
        decision = "C2B_MULTIRULE_STRICT_SCREEN_PASS_START_C3_SHIFTED"
    elif oof_summary["abstention_gate_pass"]:
        decision = "C2B_MULTIRULE_ABSTENTION_SCREEN_PASS_START_C3_SHIFTED"
    elif abstention_pass:
        decision = "C2B_MULTIRULE_IN_SAMPLE_ONLY_FAIL_OOF"
    else:
        decision = "C2B_MULTIRULE_ROUTER_SCREEN_FAIL_REASSESS_FEATURES_OR_EXPERT"

    summary = {
        "route": "Haze4K-v2.0 StrongExpert-GainMix",
        "phase": "C2b Multi-Rule OutputDiff Router Screen",
        "locked_test_touched": False,
        "feature_rows": str(args.feature_rows),
        "rows": len(rows),
        "policy_features": POLICY_FEATURES,
        "quantiles": QUANTILES,
        "strict_gate": STRICT_GATE,
        "abstention_gate": ABSTENTION_GATE,
        "best_policy": policies[0] if policies else None,
        "best_strict_policy": strict_pass[0] if strict_pass else None,
        "best_abstention_policy": abstention_pass[0] if abstention_pass else None,
        "oof_summary": oof_summary,
        "fold_rows": fold_rows,
        "decision": decision,
    }
    (args.out_dir / "v20_c2b_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Haze4K v2.0 C2b Multi-Rule OutputDiff Router Screen",
        "",
        f"Decision: `{decision}`",
        "",
        "C2b reuses the C2 output-difference feature CSV and searches transparent one- and two-condition deployable policies.",
        "No raw images/tensors were read or written, and locked test data was not touched.",
        "",
        "## Best In-Sample Policy",
        "",
    ]
    for key in fields:
        if policies and key in policies[0]:
            lines.append(f"- `{key}`: `{policies[0][key]}`")
    lines.extend(["", "## OOF Replay", ""])
    for key, value in oof_summary.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- C3 shifted validation is authorized only if the OOF screen passes.",
            "- If OOF fails, do not touch locked test; improve features, expert compatibility, or router class first.",
        ]
    )
    (args.out_dir / "v20_c2b_decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"V20_C2B_MULTIRULE_ROUTER_OK decision={decision} rows={len(rows)} out={args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
