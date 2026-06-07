#!/usr/bin/env python3
"""v1.9 teacher-delta predictability ceiling analysis.

This table-only tool consumes the v1.7 full-train A0/UDP feature table and
tests whether UDP/alpha teacher gains can be predicted from deployable
pre-router features. It also reports the stronger post-expert ceiling as a
diagnostic contrast. Locked Haze4K test is not touched.
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


ALPHAS = [0.25, 0.50, 0.75, 1.00]
COVERAGE_TARGETS = [0.05, 0.10, 0.15, 0.20, 0.30]
LEAKY_COLUMNS = {
    "split",
    "name",
    "bucket",
    "v17_bucket",
    "v19_bucket",
    "fold",
    "a0_psnr",
    "a0_ssim",
    "udpnet_psnr",
    "udpnet_ssim",
    "delta_psnr",
    "delta_ssim",
    "oracle_best_alpha",
    "oracle_best_alpha_delta_psnr",
    "oracle_best_alpha_delta_ssim",
}
LEAKY_PREFIXES = ("alpha_", "oracle_", "v18_alpha_", "v19_alpha_")


def alpha_tag(alpha: float) -> str:
    return f"{alpha:.2f}".replace(".", "p")


def to_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        out = float(value)
        return out if math.isfinite(out) else default
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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fold_for_name(name: str, folds: int = 5) -> int:
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % folds


def add_buckets_and_folds(rows: list[dict[str, Any]]) -> None:
    values = [float(row["a0_psnr"]) for row in rows]
    hard_cut = percentile(values, 25)
    easy_cut = percentile(values, 75)
    assert hard_cut is not None and easy_cut is not None
    for row in rows:
        psnr = float(row["a0_psnr"])
        if psnr <= hard_cut:
            row["v19_bucket"] = "hard_bottom25_by_a0_fulltrain"
        elif psnr >= easy_cut:
            row["v19_bucket"] = "easy_top25_by_a0_fulltrain"
        else:
            row["v19_bucket"] = "mid_by_a0_fulltrain"
        row["fold"] = fold_for_name(str(row["name"]))
        for alpha in ALPHAS:
            tag = alpha_tag(alpha)
            delta = float(row[f"alpha_{tag}_delta_psnr"])
            ssim_delta = float(row[f"alpha_{tag}_delta_ssim"])
            row[f"v19_alpha_{tag}_gain"] = int(delta >= 0.10)
            row[f"v19_alpha_{tag}_risk"] = int(
                delta <= -0.20
                or ssim_delta <= -0.001
                or (row["v19_bucket"] == "easy_top25_by_a0_fulltrain" and delta < 0)
            )


def feature_group(key: str) -> str:
    if key.startswith(("input_", "dark_channel", "bright_channel", "depth_", "filename_param")):
        return "pre_router"
    if key.startswith("udp_a0_"):
        return "post_expert"
    return "other"


def numeric_features(rows: list[dict[str, Any]], group: str) -> list[str]:
    names: list[str] = []
    if not rows:
        return names
    for key in rows[0]:
        if key in LEAKY_COLUMNS or key.startswith(LEAKY_PREFIXES) or key.startswith("_"):
            continue
        values = [to_float(row.get(key)) for row in rows]
        finite = [value for value in values if value is not None]
        if len(finite) < max(10, len(rows) // 20):
            continue
        key_group = feature_group(key)
        if group in {"pre_router", "post_expert"} and key_group != group:
            continue
        if group == "all_safe" and key_group not in {"pre_router", "post_expert", "other"}:
            continue
        names.append(key)
    return names


def auc_for_feature(rows: list[dict[str, Any]], feature: str, label_key: str) -> tuple[float | None, int]:
    pairs = []
    for row in rows:
        value = to_float(row.get(feature))
        label = int(row.get(label_key, 0))
        if value is not None:
            pairs.append((value, label))
    positives = sum(label for _value, label in pairs)
    negatives = len(pairs) - positives
    if positives == 0 or negatives == 0:
        return None, positives
    pairs.sort(key=lambda item: item[0])
    rank_sum = 0.0
    i = 0
    rank = 1
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (rank + rank + (j - i) - 1) / 2.0
        rank_sum += sum(label for _value, label in pairs[i:j]) * avg_rank
        rank += j - i
        i = j
    auc = (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)
    return auc, positives


def alpha_delta(row: dict[str, Any], alpha: float) -> float:
    if alpha == 0.0:
        return 0.0
    return float(row[f"alpha_{alpha_tag(alpha)}_delta_psnr"])


def alpha_ssim_delta(row: dict[str, Any], alpha: float) -> float:
    if alpha == 0.0:
        return 0.0
    return float(row[f"alpha_{alpha_tag(alpha)}_delta_ssim"])


def summarize_decisions(rows: list[dict[str, Any]], decisions: list[float], label: str) -> dict[str, Any]:
    deltas = [alpha_delta(row, alpha) for row, alpha in zip(rows, decisions)]
    ssim_deltas = [alpha_ssim_delta(row, alpha) for row, alpha in zip(rows, decisions)]
    hard = [(row, alpha) for row, alpha in zip(rows, decisions) if row["v19_bucket"].startswith("hard")]
    easy = [(row, alpha) for row, alpha in zip(rows, decisions) if row["v19_bucket"].startswith("easy")]
    worst = [delta for delta in deltas if delta <= -0.20]
    strong_reg = [alpha_delta(row, alpha) for row, alpha in easy if alpha_delta(row, alpha) <= -0.05]
    tail_n = max(1, len(deltas) // 10)
    ordered = sorted(deltas)
    chosen = [alpha for alpha in decisions if alpha > 0]
    return {
        "label": label,
        "count": len(rows),
        "coverage": len(chosen) / max(1, len(rows)),
        "mean_delta": mean(deltas),
        "median_delta": statistics.median(deltas) if deltas else None,
        "hard_bottom25_delta": mean([alpha_delta(row, alpha) for row, alpha in hard]),
        "easy_top25_delta": mean([alpha_delta(row, alpha) for row, alpha in easy]),
        "worst10pct_delta": mean(ordered[:tail_n]),
        "mean_ssim_delta": mean(ssim_deltas),
        "positive_ratio": sum(delta > 0 for delta in deltas) / max(1, len(deltas)),
        "worst_regression_ratio": len(worst) / max(1, len(rows)),
        "strong_regression_ratio": len(strong_reg) / max(1, len(easy)),
        "alpha_1p00_count": sum(alpha == 1.0 for alpha in decisions),
        "alpha_0p75_count": sum(alpha == 0.75 for alpha in decisions),
        "alpha_0p50_count": sum(alpha == 0.50 for alpha in decisions),
        "alpha_0p25_count": sum(alpha == 0.25 for alpha in decisions),
        "alpha_0p00_count": sum(alpha == 0.0 for alpha in decisions),
    }


def score_summary(summary: dict[str, Any]) -> float:
    return (
        float(summary.get("mean_delta") or -999)
        + 0.9 * float(summary.get("hard_bottom25_delta") or -999)
        + 0.35 * float(summary.get("easy_top25_delta") or -999)
        + 20.0 * min(float(summary.get("mean_ssim_delta") or 0.0), 0.01)
        - 1.4 * float(summary.get("worst_regression_ratio") or 0.0)
        - 0.8 * float(summary.get("strong_regression_ratio") or 0.0)
    )


def threshold_for(values: list[float], direction: str, coverage: float) -> float:
    q = 100.0 * (1.0 - coverage if direction == "high" else coverage)
    value = percentile(values, q)
    return 0.0 if value is None else float(value)


def decisions_for_policy(rows: list[dict[str, Any]], policy: dict[str, Any]) -> list[float]:
    feature = str(policy["feature"])
    direction = str(policy["direction"])
    threshold = float(policy["threshold"])
    alpha = float(policy["alpha"])
    out = []
    for row in rows:
        value = to_float(row.get(feature))
        if value is None:
            out.append(0.0)
            continue
        accept = value >= threshold if direction == "high" else value <= threshold
        out.append(alpha if accept else 0.0)
    return out


def fit_best_single_feature_policy(train_rows: list[dict[str, Any]], group: str, alpha: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tag = alpha_tag(alpha)
    features = numeric_features(train_rows, group)
    auc_rows: list[dict[str, Any]] = []
    policy_rows: list[dict[str, Any]] = []
    for feature in features:
        gain_auc, gain_pos = auc_for_feature(train_rows, feature, f"v19_alpha_{tag}_gain")
        risk_auc, risk_pos = auc_for_feature(train_rows, feature, f"v19_alpha_{tag}_risk")
        if gain_auc is None:
            continue
        gain_direction = "high" if gain_auc >= 0.5 else "low"
        risk_direction = "high" if risk_auc is not None and risk_auc >= 0.5 else "low"
        auc_rows.append(
            {
                "feature_group": group,
                "alpha": alpha,
                "feature": feature,
                "gain_auc": gain_auc,
                "gain_direction": gain_direction,
                "gain_positive_count": gain_pos,
                "risk_auc": risk_auc,
                "risk_direction": risk_direction,
                "risk_positive_count": risk_pos,
            }
        )
        finite_values = [float(row[feature]) for row in train_rows if to_float(row.get(feature)) is not None]
        for coverage in COVERAGE_TARGETS:
            threshold = threshold_for(finite_values, gain_direction, coverage)
            policy = {
                "feature_group": group,
                "alpha": alpha,
                "feature": feature,
                "direction": gain_direction,
                "threshold": threshold,
                "coverage_target": coverage,
            }
            summary = summarize_decisions(train_rows, decisions_for_policy(train_rows, policy), "train_fit")
            policy_rows.append({**policy, **summary, "score": score_summary(summary)})
    if not policy_rows:
        fallback = {
            "feature_group": group,
            "alpha": alpha,
            "feature": "__none__",
            "direction": "high",
            "threshold": 0.0,
            "coverage_target": 0.0,
        }
        return auc_rows, fallback
    best = max(policy_rows, key=lambda row: float(row["score"]))
    return auc_rows, best


def run_oof(rows: list[dict[str, Any]], group: str, alpha: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    per_image: list[dict[str, Any]] = []
    fold_policies: list[dict[str, Any]] = []
    all_auc_rows: list[dict[str, Any]] = []
    for fold in range(5):
        train_rows = [row for row in rows if int(row["fold"]) != fold]
        valid_rows = [row for row in rows if int(row["fold"]) == fold]
        auc_rows, policy = fit_best_single_feature_policy(train_rows, group, alpha)
        all_auc_rows.extend({**item, "fold": fold} for item in auc_rows)
        decisions = decisions_for_policy(valid_rows, policy)
        fold_summary = summarize_decisions(valid_rows, decisions, f"fold_{fold}")
        fold_policies.append({**policy, **fold_summary, "fold": fold, "score": score_summary(fold_summary)})
        for row, decision in zip(valid_rows, decisions):
            per_image.append(
                {
                    "split": row["split"],
                    "name": row["name"],
                    "fold": fold,
                    "feature_group": group,
                    "alpha": alpha,
                    "selected_alpha": decision,
                    "selected_delta_psnr": alpha_delta(row, decision),
                    "selected_delta_ssim": alpha_ssim_delta(row, decision),
                    "a0_psnr": row["a0_psnr"],
                    "bucket": row["v19_bucket"],
                }
            )
    per_image.sort(key=lambda item: str(item["name"]))
    summary_rows = [
        {
            "split": split,
            **summarize_decisions(
                [row for row in rows if row["split"] == split],
                [
                    float(next(item["selected_alpha"] for item in per_image if item["name"] == row["name"]))
                    for row in rows
                    if row["split"] == split
                ],
                f"oof_{group}_alpha_{alpha_tag(alpha)}_{split}",
            ),
        }
        for split in sorted({str(row["split"]) for row in rows})
    ]
    combined_rows = rows
    combined_decisions = [
        float(next(item["selected_alpha"] for item in per_image if item["name"] == row["name"]))
        for row in combined_rows
    ]
    combined_summary = summarize_decisions(combined_rows, combined_decisions, f"oof_{group}_alpha_{alpha_tag(alpha)}")
    combined_summary["split_summaries"] = summary_rows
    return all_auc_rows, fold_policies, combined_summary


def run_heldout(rows: list[dict[str, Any]], group: str, alpha: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    train_rows = [row for row in rows if str(row["split"]) == "train_inner"]
    heldout_rows = [row for row in rows if str(row["split"]) in {"val_regular", "val_hard"}]
    _auc_rows, policy = fit_best_single_feature_policy(train_rows, group, alpha)
    decisions = decisions_for_policy(heldout_rows, policy)
    per_image = []
    for row, decision in zip(heldout_rows, decisions):
        per_image.append(
            {
                "split": row["split"],
                "name": row["name"],
                "feature_group": group,
                "alpha": alpha,
                "selected_alpha": decision,
                "selected_delta_psnr": alpha_delta(row, decision),
                "selected_delta_ssim": alpha_ssim_delta(row, decision),
                "a0_psnr": row["a0_psnr"],
                "bucket": row["v19_bucket"],
            }
        )
    summary = summarize_decisions(heldout_rows, decisions, f"heldout_{group}_alpha_{alpha_tag(alpha)}")
    return {**policy, **summary, "score": score_summary(summary)}, per_image


def add_gate(summary: dict[str, Any], prefix: str) -> dict[str, Any]:
    checks = {
        f"{prefix}_mean_delta_ge_0p18": float(summary.get("mean_delta") or -999) >= 0.18,
        f"{prefix}_hard_bottom25_ge_0p30": float(summary.get("hard_bottom25_delta") or -999) >= 0.30,
        f"{prefix}_easy_top25_ge_neg0p02": float(summary.get("easy_top25_delta") or -999) >= -0.02,
        f"{prefix}_ssim_ge_0": float(summary.get("mean_ssim_delta") or -999) >= 0.0,
        f"{prefix}_worst_ratio_le_0p05": float(summary.get("worst_regression_ratio") or 999) <= 0.05,
        f"{prefix}_strong_ratio_le_0p10": float(summary.get("strong_regression_ratio") or 999) <= 0.10,
    }
    summary[f"{prefix}_gate_checks"] = checks
    summary[f"{prefix}_gate_pass"] = all(checks.values())
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    rows = read_rows(Path(args.feature_csv))
    add_buckets_and_folds(rows)
    output_dir = Path(args.output_dir)

    all_auc_rows: list[dict[str, Any]] = []
    all_policy_rows: list[dict[str, Any]] = []
    heldout_rows_all: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for group in ("pre_router", "post_expert"):
        for alpha in ALPHAS:
            auc_rows, fold_policies, oof_summary = run_oof(rows, group, alpha)
            heldout_summary, heldout_rows = run_heldout(rows, group, alpha)
            all_auc_rows.extend(auc_rows)
            all_policy_rows.extend(fold_policies)
            heldout_rows_all.extend(heldout_rows)
            summaries.append(add_gate(oof_summary, "oof"))
            summaries.append(add_gate(heldout_summary, "heldout"))

    best_pre_oof = max(
        [item for item in summaries if str(item["label"]).startswith("oof_pre_router")],
        key=score_summary,
    )
    best_pre_heldout = max(
        [item for item in summaries if str(item["label"]).startswith("heldout_pre_router")],
        key=score_summary,
    )
    best_post_oof = max(
        [item for item in summaries if str(item["label"]).startswith("oof_post_expert")],
        key=score_summary,
    )
    best_post_heldout = max(
        [item for item in summaries if str(item["label"]).startswith("heldout_post_expert")],
        key=score_summary,
    )
    decision = "PRE_ROUTER_PREDICTABILITY_GATE_PASS"
    if not best_pre_oof["oof_gate_pass"] or not best_pre_heldout["heldout_gate_pass"]:
        decision = "PRE_ROUTER_PREDICTABILITY_GATE_FAIL_ROUTE_NEEDS_INTERNAL_OR_PATCH_POLICY"

    write_csv(output_dir / "v19_teacher_delta_feature_auc.csv", all_auc_rows)
    write_csv(output_dir / "v19_teacher_delta_oof_fold_policies.csv", all_policy_rows)
    write_csv(output_dir / "v19_teacher_delta_heldout_per_image.csv", heldout_rows_all)
    payload = {
        "route": "ConvIR-Dehaze-v1.9-ConditionalTeacherGuided",
        "stage": "teacher delta predictability ceiling",
        "locked_test_touched": False,
        "feature_csv": args.feature_csv,
        "count": len(rows),
        "summaries": summaries,
        "best_pre_router_oof": best_pre_oof,
        "best_pre_router_heldout": best_pre_heldout,
        "best_post_expert_oof": best_post_oof,
        "best_post_expert_heldout": best_post_heldout,
        "decision": decision,
    }
    write_json(output_dir / "v19_teacher_delta_predictability_summary.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
