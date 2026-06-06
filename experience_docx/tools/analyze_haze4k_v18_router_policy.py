#!/usr/bin/env python3
"""Table-only v1.8 router policy analysis for the A0/UDP expert bank.

This script consumes the v1.7 3000-row A0/UDP feature table and produces
train-derived router evidence. It does not run inference, train a model, or
touch locked Haze4K test data.
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
COVERAGE_TARGETS = [0.05, 0.10, 0.20, 0.30]
LEAKY_COLUMNS = {
    "split",
    "name",
    "bucket",
    "v17_bucket",
    "v18_bucket",
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
LEAKY_PREFIXES = ("alpha_", "oracle_")
DERIVED_LABEL_PREFIXES = ("v18_alpha_",)


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


def fold_for_name(name: str, folds: int = 5) -> int:
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % folds


def add_fulltrain_buckets(rows: list[dict[str, Any]]) -> None:
    values = [float(row["a0_psnr"]) for row in rows]
    hard_cut = percentile(values, 25)
    easy_cut = percentile(values, 75)
    assert hard_cut is not None and easy_cut is not None
    for row in rows:
        psnr = float(row["a0_psnr"])
        if psnr <= hard_cut:
            row["v18_bucket"] = "hard_bottom25_by_a0_fulltrain"
        elif psnr >= easy_cut:
            row["v18_bucket"] = "easy_top25_by_a0_fulltrain"
        else:
            row["v18_bucket"] = "mid_by_a0_fulltrain"
        row["fold"] = fold_for_name(str(row["name"]))


def feature_group(key: str) -> str:
    if key.startswith(("input_", "dark_channel", "bright_channel", "depth_", "filename_param")):
        return "cheap_pre"
    if key.startswith("udp_a0_"):
        return "post_expert"
    return "other"


def numeric_feature_names(rows: list[dict[str, Any]], group: str) -> list[str]:
    names: list[str] = []
    for key in rows[0]:
        if key in LEAKY_COLUMNS or key.startswith(LEAKY_PREFIXES) or key.startswith(DERIVED_LABEL_PREFIXES):
            continue
        if key.startswith("_"):
            continue
        values = [to_float(row.get(key)) for row in rows]
        finite = [value for value in values if value is not None]
        if len(finite) < max(10, len(rows) // 20):
            continue
        key_group = feature_group(key)
        if group == "cheap_pre" and key_group != "cheap_pre":
            continue
        if group == "post_expert" and key_group != "post_expert":
            continue
        if group == "all" and key_group not in {"cheap_pre", "post_expert", "other"}:
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


def add_alpha_labels(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        for alpha in ALPHAS:
            tag = alpha_tag(alpha)
            delta = float(row[f"alpha_{tag}_delta_psnr"])
            ssim_delta = float(row[f"alpha_{tag}_delta_ssim"])
            row[f"v18_alpha_{tag}_gain"] = int(delta >= 0.10)
            row[f"v18_alpha_{tag}_risk"] = int(
                delta <= -0.20
                or ssim_delta <= -0.001
                or (row.get("v18_bucket") == "easy_top25_by_a0_fulltrain" and delta < 0)
            )


def select_features(train_rows: list[dict[str, Any]], group: str, alpha: float) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    tag = alpha_tag(alpha)
    names = numeric_feature_names(train_rows, group)
    auc_rows: list[dict[str, Any]] = []
    for name in names:
        gain_auc, gain_pos = auc_for_feature(train_rows, name, f"v18_alpha_{tag}_gain")
        risk_auc, risk_pos = auc_for_feature(train_rows, name, f"v18_alpha_{tag}_risk")
        gain_strength = 0.0 if gain_auc is None else abs(gain_auc - 0.5)
        risk_strength = 0.0 if risk_auc is None else abs(risk_auc - 0.5)
        auc_rows.append(
            {
                "feature_group": group,
                "alpha": alpha,
                "feature": name,
                "gain_auc": gain_auc,
                "gain_positive_count": gain_pos,
                "gain_direction": "high" if gain_auc is not None and gain_auc >= 0.5 else "low",
                "gain_strength": gain_strength,
                "risk_auc": risk_auc,
                "risk_positive_count": risk_pos,
                "risk_direction": "high" if risk_auc is not None and risk_auc >= 0.5 else "low",
                "risk_strength": risk_strength,
            }
        )
    gain_rank = sorted(auc_rows, key=lambda row: row["gain_strength"], reverse=True)
    risk_rank = sorted(auc_rows, key=lambda row: row["risk_strength"], reverse=True)
    gain_features = [str(row["feature"]) for row in gain_rank[:3] if row["gain_strength"] > 0]
    risk_features = [str(row["feature"]) for row in risk_rank[:3] if row["risk_strength"] > 0]
    return auc_rows, gain_features, risk_features


def stats_for_features(rows: list[dict[str, Any]], features: list[str]) -> dict[str, tuple[float, float]]:
    stats = {}
    for feature in features:
        values = [float(row[feature]) for row in rows if to_float(row.get(feature)) is not None]
        mu = statistics.mean(values) if values else 0.0
        sigma = statistics.pstdev(values) if len(values) > 1 else 1.0
        if sigma <= 1e-12:
            sigma = 1.0
        stats[feature] = (mu, sigma)
    return stats


def oriented_z(row: dict[str, Any], feature: str, stats: dict[str, tuple[float, float]], direction: str) -> float:
    value = to_float(row.get(feature), 0.0)
    assert value is not None
    mu, sigma = stats[feature]
    z = (value - mu) / sigma
    return z if direction == "high" else -z


def feature_direction(auc_rows: list[dict[str, Any]], feature: str, kind: str) -> str:
    for row in auc_rows:
        if row["feature"] == feature:
            return str(row[f"{kind}_direction"])
    return "high"


def score_rows(
    rows: list[dict[str, Any]],
    auc_rows: list[dict[str, Any]],
    gain_features: list[str],
    risk_features: list[str],
    train_stats: dict[str, tuple[float, float]],
    risk_weight: float,
) -> list[float]:
    scores = []
    for row in rows:
        gain_terms = [
            oriented_z(row, feature, train_stats, feature_direction(auc_rows, feature, "gain"))
            for feature in gain_features
        ]
        risk_terms = [
            oriented_z(row, feature, train_stats, feature_direction(auc_rows, feature, "risk"))
            for feature in risk_features
        ]
        gain_score = mean(gain_terms) or 0.0
        risk_score = mean(risk_terms) or 0.0
        scores.append(gain_score - risk_weight * risk_score)
    return scores


def cutoff_for_coverage(scores: list[float], coverage: float) -> float:
    if not scores:
        return float("inf")
    ordered = sorted(scores, reverse=True)
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * coverage) - 1))
    return ordered[index]


def alpha_delta(row: dict[str, Any], alpha: float) -> float:
    if alpha <= 0:
        return 0.0
    return float(row[f"alpha_{alpha_tag(alpha)}_delta_psnr"])


def alpha_ssim_delta(row: dict[str, Any], alpha: float) -> float:
    if alpha <= 0:
        return 0.0
    return float(row[f"alpha_{alpha_tag(alpha)}_delta_ssim"])


def summarize_decisions(rows: list[dict[str, Any]], decisions: list[float], label: str) -> dict[str, Any]:
    deltas = [alpha_delta(row, alpha) for row, alpha in zip(rows, decisions)]
    ssim_deltas = [alpha_ssim_delta(row, alpha) for row, alpha in zip(rows, decisions)]
    hard = [(row, alpha) for row, alpha in zip(rows, decisions) if row.get("v18_bucket") == "hard_bottom25_by_a0_fulltrain"]
    easy = [(row, alpha) for row, alpha in zip(rows, decisions) if row.get("v18_bucket") == "easy_top25_by_a0_fulltrain"]
    strong = easy
    tail_n = max(1, len(deltas) // 10)
    ordered = sorted(deltas)
    strong_reg = [1 for row, alpha in strong if alpha_delta(row, alpha) <= -0.05]
    worst = [1 for row, alpha in zip(rows, decisions) if alpha_delta(row, alpha) <= -0.20]
    return {
        "label": label,
        "count": len(rows),
        "coverage": sum(alpha > 0 for alpha in decisions) / max(1, len(rows)),
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


def gate_flags(summary: dict[str, Any], stage: str) -> dict[str, bool]:
    if stage == "oof":
        return {
            "mean_delta_ge_0p20": float(summary.get("mean_delta") or -999) >= 0.20,
            "hard_bottom25_delta_ge_0p35": float(summary.get("hard_bottom25_delta") or -999) >= 0.35,
            "easy_top25_delta_ge_0": float(summary.get("easy_top25_delta") or -999) >= 0.0,
            "mean_ssim_delta_ge_0": float(summary.get("mean_ssim_delta") or -999) >= 0.0,
            "worst_ratio_le_0p04": float(summary.get("worst_regression_ratio") or 999) <= 0.04,
            "strong_ratio_le_0p08": float(summary.get("strong_regression_ratio") or 999) <= 0.08,
        }
    return {
        "mean_delta_ge_0p15": float(summary.get("mean_delta") or -999) >= 0.15,
        "hard_bottom25_delta_ge_0p25": float(summary.get("hard_bottom25_delta") or -999) >= 0.25,
        "easy_top25_delta_ge_neg0p02": float(summary.get("easy_top25_delta") or -999) >= -0.02,
        "worst_ratio_le_0p05": float(summary.get("worst_regression_ratio") or 999) <= 0.05,
        "strong_ratio_le_0p10": float(summary.get("strong_regression_ratio") or 999) <= 0.10,
    }


def train_policy(
    train_rows: list[dict[str, Any]],
    group: str,
    alpha: float,
    coverage: float,
    risk_weight: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[float]]:
    auc_rows, gain_features, risk_features = select_features(train_rows, group, alpha)
    features = sorted(set(gain_features + risk_features))
    train_stats = stats_for_features(train_rows, features)
    train_scores = score_rows(train_rows, auc_rows, gain_features, risk_features, train_stats, risk_weight)
    cutoff = cutoff_for_coverage(train_scores, coverage)
    policy = {
        "feature_group": group,
        "alpha": alpha,
        "coverage_target": coverage,
        "risk_weight": risk_weight,
        "cutoff": cutoff,
        "gain_features": gain_features,
        "risk_features": risk_features,
        "feature_stats": train_stats,
    }
    return policy, auc_rows, train_scores


def apply_policy(rows: list[dict[str, Any]], policy: dict[str, Any], auc_rows: list[dict[str, Any]]) -> list[float]:
    features = sorted(set(policy["gain_features"] + policy["risk_features"]))
    stats = {key: tuple(value) for key, value in policy["feature_stats"].items()}
    scores = score_rows(rows, auc_rows, policy["gain_features"], policy["risk_features"], stats, policy["risk_weight"])
    return [float(policy["alpha"]) if score >= float(policy["cutoff"]) else 0.0 for score in scores]


def policy_per_image(rows: list[dict[str, Any]], decisions: list[float], label: str) -> list[dict[str, Any]]:
    output = []
    for row, alpha in zip(rows, decisions):
        output.append(
            {
                "label": label,
                "split": row["split"],
                "name": row["name"],
                "fold": row.get("fold"),
                "bucket": row.get("v18_bucket"),
                "selected_alpha": alpha,
                "delta_psnr": alpha_delta(row, alpha),
                "delta_ssim": alpha_ssim_delta(row, alpha),
                "a0_psnr": row["a0_psnr"],
            }
        )
    return output


def run_oof(rows: list[dict[str, Any]], group: str, alpha: float, coverage: float, risk_weight: float) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    train_rows = [row for row in rows if row["split"] == "train_inner"]
    oof_rows: list[dict[str, Any]] = []
    oof_decisions: list[float] = []
    auc_all: list[dict[str, Any]] = []
    fold_summaries = []
    for fold in range(5):
        fold_train = [row for row in train_rows if row["fold"] != fold]
        fold_eval = [row for row in train_rows if row["fold"] == fold]
        policy, auc_rows, _scores = train_policy(fold_train, group, alpha, coverage, risk_weight)
        decisions = apply_policy(fold_eval, policy, auc_rows)
        fold_summary = summarize_decisions(fold_eval, decisions, f"fold_{fold}")
        fold_summary["fold"] = fold
        fold_summary["gate_checks"] = gate_flags(fold_summary, "heldout")
        fold_summary["gate_pass"] = all(fold_summary["gate_checks"].values())
        fold_summaries.append(fold_summary)
        oof_rows.extend(fold_eval)
        oof_decisions.extend(decisions)
        auc_all.extend(auc_rows)
    summary = summarize_decisions(oof_rows, oof_decisions, "train_inner_5fold_oof")
    summary["gate_checks"] = gate_flags(summary, "oof")
    summary["gate_pass"] = all(summary["gate_checks"].values())
    summary["fold_gate_pass_count"] = sum(1 for item in fold_summaries if item["gate_pass"])
    summary["fold_summaries"] = fold_summaries
    return summary, policy_per_image(oof_rows, oof_decisions, "oof"), auc_all


def run_heldout(rows: list[dict[str, Any]], group: str, alpha: float, coverage: float, risk_weight: float) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    train_rows = [row for row in rows if row["split"] == "train_inner"]
    heldout_rows = [row for row in rows if row["split"] in {"val_regular", "val_hard"}]
    policy, auc_rows, _scores = train_policy(train_rows, group, alpha, coverage, risk_weight)
    decisions = apply_policy(heldout_rows, policy, auc_rows)
    summary = summarize_decisions(heldout_rows, decisions, "train_inner_fit_heldout_confirm")
    summary["gate_checks"] = gate_flags(summary, "heldout")
    summary["gate_pass"] = all(summary["gate_checks"].values())
    return summary, policy_per_image(heldout_rows, decisions, "heldout"), policy, auc_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--feature_groups", nargs="+", default=["cheap_pre", "post_expert", "all"])
    parser.add_argument("--risk_weights", nargs="+", type=float, default=[0.5, 1.0, 1.5, 2.0])
    args = parser.parse_args()

    feature_csv = Path(args.feature_csv)
    output_dir = Path(args.output_dir)
    rows = read_rows(feature_csv)
    add_fulltrain_buckets(rows)
    add_alpha_labels(rows)

    grid_rows = []
    auc_rows_all = []
    best = None
    best_bundle = None
    for group in args.feature_groups:
        for alpha in ALPHAS:
            for coverage in COVERAGE_TARGETS:
                for risk_weight in args.risk_weights:
                    oof_summary, oof_per_image, oof_auc = run_oof(rows, group, alpha, coverage, risk_weight)
                    heldout_summary, heldout_per_image, policy, heldout_auc = run_heldout(
                        rows, group, alpha, coverage, risk_weight
                    )
                    score = (
                        float(oof_summary.get("mean_delta") or -999)
                        + float(oof_summary.get("hard_bottom25_delta") or -999)
                        + 0.5 * float(heldout_summary.get("mean_delta") or -999)
                        + 0.5 * float(heldout_summary.get("hard_bottom25_delta") or -999)
                        - 3.0 * max(0.0, float(oof_summary.get("worst_regression_ratio") or 0.0) - 0.04)
                        - 3.0 * max(0.0, float(heldout_summary.get("worst_regression_ratio") or 0.0) - 0.05)
                    )
                    row = {
                        "feature_group": group,
                        "alpha": alpha,
                        "coverage_target": coverage,
                        "risk_weight": risk_weight,
                        "score": score,
                        "oof_gate_pass": oof_summary["gate_pass"],
                        "heldout_gate_pass": heldout_summary["gate_pass"],
                        "oof_mean_delta": oof_summary["mean_delta"],
                        "oof_hard_bottom25_delta": oof_summary["hard_bottom25_delta"],
                        "oof_easy_top25_delta": oof_summary["easy_top25_delta"],
                        "oof_worst_ratio": oof_summary["worst_regression_ratio"],
                        "oof_strong_ratio": oof_summary["strong_regression_ratio"],
                        "heldout_mean_delta": heldout_summary["mean_delta"],
                        "heldout_hard_bottom25_delta": heldout_summary["hard_bottom25_delta"],
                        "heldout_easy_top25_delta": heldout_summary["easy_top25_delta"],
                        "heldout_worst_ratio": heldout_summary["worst_regression_ratio"],
                        "heldout_strong_ratio": heldout_summary["strong_regression_ratio"],
                        "gain_features": ",".join(policy["gain_features"]),
                        "risk_features": ",".join(policy["risk_features"]),
                    }
                    grid_rows.append(row)
                    auc_rows_all.extend(oof_auc)
                    auc_rows_all.extend(heldout_auc)
                    if best is None or score > best["score"]:
                        best = row
                        best_bundle = {
                            "policy": policy,
                            "oof_summary": oof_summary,
                            "heldout_summary": heldout_summary,
                            "oof_per_image": oof_per_image,
                            "heldout_per_image": heldout_per_image,
                            "auc_rows": heldout_auc,
                        }

    assert best is not None and best_bundle is not None
    grid_rows.sort(key=lambda row: row["score"], reverse=True)
    unique_auc = {}
    for row in auc_rows_all:
        key = (row["feature_group"], row["alpha"], row["feature"])
        unique_auc[key] = row
    auc_rows = sorted(unique_auc.values(), key=lambda row: (row["feature_group"], row["alpha"], row["feature"]))

    write_csv(output_dir / "v18_router_policy_grid.csv", grid_rows)
    write_csv(output_dir / "v18_router_feature_auc.csv", auc_rows)
    write_csv(output_dir / "v18_router_oof_policy_per_image.csv", best_bundle["oof_per_image"])
    write_csv(output_dir / "v18_router_heldout_policy_per_image.csv", best_bundle["heldout_per_image"])
    status = {
        "route": "ConvIR-Dehaze-v1.8-ExecutionQueue",
        "stage": "table-only A0/UDP router policy analysis",
        "status": "COMPLETED_TABLE_ANALYSIS",
        "locked_test_touched": False,
        "feature_csv": str(feature_csv),
        "row_count": len(rows),
        "grid_count": len(grid_rows),
        "best_policy_grid_row": best,
        "best_policy": best_bundle["policy"],
        "oof_summary": best_bundle["oof_summary"],
        "heldout_summary": best_bundle["heldout_summary"],
        "decision": (
            "ROUTER_GATE_PASS_LOCKED_STILL_BLOCKED_PENDING_ROUTE_CARD"
            if best_bundle["oof_summary"]["gate_pass"] and best_bundle["heldout_summary"]["gate_pass"]
            else "ROUTER_GATE_FAIL_CONTINUE_OTHER_V18_EXPERIMENTS"
        ),
        "outputs": {
            "policy_grid": "v18_router_policy_grid.csv",
            "feature_auc": "v18_router_feature_auc.csv",
            "oof_per_image": "v18_router_oof_policy_per_image.csv",
            "heldout_per_image": "v18_router_heldout_policy_per_image.csv",
        },
    }
    write_json(output_dir / "v18_router_best_policy_summary.json", status)
    write_json(output_dir / "v18_analysis_status.json", status)
    print(json.dumps(status, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
