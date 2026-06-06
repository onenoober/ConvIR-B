#!/usr/bin/env python3
"""Analyze v1.8 domain-adaptation evidence from train-derived Haze4K tables.

This tool is table-only. It does not train, run inference, or touch the locked
Haze4K test split. It has two jobs:

1. record whether any real-haze/domain-adaptation data is available; and
2. run an internal Haze4K domain-conditioned A0/UDP alpha policy diagnostic.
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


ALPHAS = [0.0, 0.25, 0.50, 0.75, 1.00]
IMG_EXT = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
STRATEGIES = [
    "sky_tertiles",
    "luma_sky_2x2",
    "bright_low_sat_2x2",
    "edge_sky_2x2",
    "filename_haze_2x2",
]


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


def alpha_tag(alpha: float) -> str:
    return f"{alpha:.2f}".replace(".", "p")


def alpha_delta(row: dict[str, Any], alpha: float) -> float:
    if alpha <= 0:
        return 0.0
    return float(row[f"alpha_{alpha_tag(alpha)}_delta_psnr"])


def alpha_ssim_delta(row: dict[str, Any], alpha: float) -> float:
    if alpha <= 0:
        return 0.0
    return float(row[f"alpha_{alpha_tag(alpha)}_delta_ssim"])


def fold_for_name(name: str, folds: int = 5) -> int:
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % folds


def add_buckets(rows: list[dict[str, Any]]) -> None:
    values = [float(row["a0_psnr"]) for row in rows]
    hard_cut = percentile(values, 25)
    easy_cut = percentile(values, 75)
    assert hard_cut is not None and easy_cut is not None
    for row in rows:
        psnr = float(row["a0_psnr"])
        if psnr <= hard_cut:
            row["v18_domain_bucket"] = "hard_bottom25_by_a0_fulltrain"
        elif psnr >= easy_cut:
            row["v18_domain_bucket"] = "easy_top25_by_a0_fulltrain"
        else:
            row["v18_domain_bucket"] = "mid_by_a0_fulltrain"
        row["fold"] = fold_for_name(str(row["name"]))


def merge_rows(feature_rows: list[dict[str, Any]], domain_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    domain_by_name = {str(row["name"]): row for row in domain_rows}
    merged = []
    for row in feature_rows:
        out = dict(row)
        domain = domain_by_name.get(str(row["name"]), {})
        for key, value in domain.items():
            if key in {"split", "name", "a0_psnr", "udp_delta_psnr", "bucket"}:
                continue
            out[f"domain_{key}"] = value
        merged.append(out)
    return merged


def count_images(path: Path) -> int:
    if not path.is_dir():
        return 0
    count = 0
    for item in path.rglob("*"):
        if item.is_file() and item.suffix.lower() in IMG_EXT:
            count += 1
    return count


def inventory_real_domain(candidates: list[str]) -> tuple[list[dict[str, Any]], str]:
    rows = []
    usable = []
    for item in candidates:
        path = Path(item)
        image_count = count_images(path)
        child_counts = {}
        if path.is_dir():
            for child in path.iterdir():
                if child.is_dir():
                    child_counts[child.name] = count_images(child)
        row = {
            "path": str(path),
            "exists": path.exists(),
            "is_dir": path.is_dir(),
            "image_count_recursive": image_count,
            "child_image_counts": child_counts,
        }
        rows.append(row)
        if image_count > 0:
            usable.append(row)
    decision = (
        "REAL_DOMAIN_DATA_AVAILABLE_AUDIT_ONLY"
        if usable
        else "REAL_DOMAIN_DATA_BLOCKED_NO_CANDIDATE_DATA"
    )
    return rows, decision


def thresholds(train_rows: list[dict[str, Any]]) -> dict[str, float]:
    keys = [
        "domain_sky_proxy_ratio",
        "domain_luma_mean",
        "domain_saturation_mean",
        "domain_gradient_mean",
        "filename_param_1",
        "filename_param_2",
    ]
    out: dict[str, float] = {}
    for key in keys:
        values = [float(row[key]) for row in train_rows if to_float(row.get(key)) is not None]
        out[f"{key}_p33"] = percentile(values, 33.333) or 0.0
        out[f"{key}_p50"] = percentile(values, 50) or 0.0
        out[f"{key}_p66"] = percentile(values, 66.667) or 0.0
    return out


def group_for(row: dict[str, Any], strategy: str, cuts: dict[str, float]) -> str:
    sky = float(row.get("domain_sky_proxy_ratio") or 0.0)
    luma = float(row.get("domain_luma_mean") or 0.0)
    sat = float(row.get("domain_saturation_mean") or 0.0)
    edge = float(row.get("domain_gradient_mean") or 0.0)
    haze1 = float(row.get("filename_param_1") or 0.0)
    haze2 = float(row.get("filename_param_2") or 0.0)
    if strategy == "sky_tertiles":
        if sky <= cuts["domain_sky_proxy_ratio_p33"]:
            return "sky_low"
        if sky >= cuts["domain_sky_proxy_ratio_p66"]:
            return "sky_high"
        return "sky_mid"
    if strategy == "luma_sky_2x2":
        return f"luma_{'high' if luma >= cuts['domain_luma_mean_p50'] else 'low'}__sky_{'high' if sky >= cuts['domain_sky_proxy_ratio_p50'] else 'low'}"
    if strategy == "bright_low_sat_2x2":
        return f"luma_{'high' if luma >= cuts['domain_luma_mean_p50'] else 'low'}__sat_{'low' if sat <= cuts['domain_saturation_mean_p50'] else 'high'}"
    if strategy == "edge_sky_2x2":
        return f"edge_{'high' if edge >= cuts['domain_gradient_mean_p50'] else 'low'}__sky_{'high' if sky >= cuts['domain_sky_proxy_ratio_p50'] else 'low'}"
    if strategy == "filename_haze_2x2":
        return f"haze1_{'high' if haze1 >= cuts['filename_param_1_p50'] else 'low'}__haze2_{'high' if haze2 >= cuts['filename_param_2_p50'] else 'low'}"
    raise ValueError(f"unknown strategy: {strategy}")


def summarize_decisions(rows: list[dict[str, Any]], decisions: list[float], label: str) -> dict[str, Any]:
    deltas = [alpha_delta(row, alpha) for row, alpha in zip(rows, decisions)]
    ssim_deltas = [alpha_ssim_delta(row, alpha) for row, alpha in zip(rows, decisions)]
    hard = [
        (row, alpha)
        for row, alpha in zip(rows, decisions)
        if row.get("v18_domain_bucket") == "hard_bottom25_by_a0_fulltrain"
    ]
    easy = [
        (row, alpha)
        for row, alpha in zip(rows, decisions)
        if row.get("v18_domain_bucket") == "easy_top25_by_a0_fulltrain"
    ]
    worst = [1 for row, alpha in zip(rows, decisions) if alpha_delta(row, alpha) <= -0.20]
    strong = [1 for row, alpha in easy if alpha_delta(row, alpha) <= -0.05]
    ordered = sorted(deltas)
    tail_n = max(1, len(deltas) // 10)
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
        "strong_regression_ratio": len(strong) / max(1, len(easy)),
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


def alpha_score(rows: list[dict[str, Any]], alpha: float) -> float:
    summary = summarize_decisions(rows, [alpha for _row in rows], f"alpha_{alpha_tag(alpha)}")
    mean_delta = float(summary.get("mean_delta") or -999)
    hard_delta = float(summary.get("hard_bottom25_delta") or 0.0)
    easy_delta = float(summary.get("easy_top25_delta") or 0.0)
    ssim_delta = float(summary.get("mean_ssim_delta") or 0.0)
    worst = float(summary.get("worst_regression_ratio") or 0.0)
    strong = float(summary.get("strong_regression_ratio") or 0.0)
    if easy_delta < -0.10 or worst > 0.10 or strong > 0.25:
        return -999.0 + mean_delta
    return mean_delta + 0.45 * hard_delta + 0.25 * easy_delta + 20.0 * min(ssim_delta, 0.01) - 2.0 * worst - strong


def train_domain_policy(train_rows: list[dict[str, Any]], strategy: str, min_group_size: int) -> dict[str, Any]:
    cuts = thresholds(train_rows)
    group_rows: dict[str, list[dict[str, Any]]] = {}
    for row in train_rows:
        group = group_for(row, strategy, cuts)
        group_rows.setdefault(group, []).append(row)
    global_alpha = max(ALPHAS, key=lambda alpha: alpha_score(train_rows, alpha))
    group_alpha = {}
    group_scores = {}
    for group, rows in group_rows.items():
        if len(rows) < min_group_size:
            alpha = global_alpha
        else:
            alpha = max(ALPHAS, key=lambda item: alpha_score(rows, item))
        group_alpha[group] = alpha
        group_scores[group] = {f"alpha_{alpha_tag(item)}": alpha_score(rows, item) for item in ALPHAS}
    return {
        "strategy": strategy,
        "thresholds": cuts,
        "global_alpha": global_alpha,
        "group_alpha": group_alpha,
        "group_scores": group_scores,
        "min_group_size": min_group_size,
    }


def apply_domain_policy(rows: list[dict[str, Any]], policy: dict[str, Any]) -> tuple[list[float], list[str]]:
    decisions = []
    groups = []
    for row in rows:
        group = group_for(row, str(policy["strategy"]), {key: float(value) for key, value in policy["thresholds"].items()})
        groups.append(group)
        decisions.append(float(policy["group_alpha"].get(group, policy["global_alpha"])))
    return decisions, groups


def per_image_rows(rows: list[dict[str, Any]], decisions: list[float], groups: list[str], label: str) -> list[dict[str, Any]]:
    out = []
    for row, alpha, group in zip(rows, decisions, groups):
        out.append(
            {
                "label": label,
                "split": row["split"],
                "name": row["name"],
                "fold": row.get("fold"),
                "domain_group": group,
                "bucket": row.get("v18_domain_bucket"),
                "selected_alpha": alpha,
                "delta_psnr": alpha_delta(row, alpha),
                "delta_ssim": alpha_ssim_delta(row, alpha),
                "a0_psnr": row["a0_psnr"],
                "domain_luma_mean": row.get("domain_luma_mean"),
                "domain_sky_proxy_ratio": row.get("domain_sky_proxy_ratio"),
                "domain_gradient_mean": row.get("domain_gradient_mean"),
            }
        )
    return out


def split_group_summary(rows: list[dict[str, Any]], strategy: str, cuts: dict[str, float]) -> list[dict[str, Any]]:
    summaries = []
    for split in sorted(set(str(row["split"]) for row in rows)):
        split_rows = [row for row in rows if row["split"] == split]
        groups = sorted(set(group_for(row, strategy, cuts) for row in split_rows))
        for group in groups:
            group_rows = [row for row in split_rows if group_for(row, strategy, cuts) == group]
            summaries.append(
                {
                    "strategy": strategy,
                    "split": split,
                    "domain_group": group,
                    "count": len(group_rows),
                    "a0_psnr_mean": mean([float(row["a0_psnr"]) for row in group_rows]),
                    "udp_delta_psnr_mean": mean([float(row["delta_psnr"]) for row in group_rows]),
                    "oracle_best_alpha_delta_psnr_mean": mean(
                        [float(row["oracle_best_alpha_delta_psnr"]) for row in group_rows]
                    ),
                    "luma_mean": mean([float(row.get("domain_luma_mean") or 0.0) for row in group_rows]),
                    "sky_proxy_ratio_mean": mean(
                        [float(row.get("domain_sky_proxy_ratio") or 0.0) for row in group_rows]
                    ),
                    "gradient_mean": mean([float(row.get("domain_gradient_mean") or 0.0) for row in group_rows]),
                }
            )
    return summaries


def run_oof(rows: list[dict[str, Any]], strategy: str, min_group_size: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    train_rows = [row for row in rows if row["split"] == "train_inner"]
    eval_rows: list[dict[str, Any]] = []
    decisions: list[float] = []
    groups: list[str] = []
    fold_summaries = []
    for fold in range(5):
        fold_train = [row for row in train_rows if row["fold"] != fold]
        fold_eval = [row for row in train_rows if row["fold"] == fold]
        policy = train_domain_policy(fold_train, strategy, min_group_size)
        fold_decisions, fold_groups = apply_domain_policy(fold_eval, policy)
        fold_summary = summarize_decisions(fold_eval, fold_decisions, f"fold_{fold}")
        fold_summary["fold"] = fold
        fold_summary["gate_checks"] = gate_flags(fold_summary, "heldout")
        fold_summary["gate_pass"] = all(fold_summary["gate_checks"].values())
        fold_summaries.append(fold_summary)
        eval_rows.extend(fold_eval)
        decisions.extend(fold_decisions)
        groups.extend(fold_groups)
    summary = summarize_decisions(eval_rows, decisions, "train_inner_5fold_oof")
    summary["gate_checks"] = gate_flags(summary, "oof")
    summary["gate_pass"] = all(summary["gate_checks"].values())
    summary["fold_gate_pass_count"] = sum(1 for item in fold_summaries if item["gate_pass"])
    summary["fold_summaries"] = fold_summaries
    return summary, per_image_rows(eval_rows, decisions, groups, "oof")


def run_heldout(rows: list[dict[str, Any]], strategy: str, min_group_size: int) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    train_rows = [row for row in rows if row["split"] == "train_inner"]
    heldout_rows = [row for row in rows if row["split"] in {"val_regular", "val_hard"}]
    policy = train_domain_policy(train_rows, strategy, min_group_size)
    decisions, groups = apply_domain_policy(heldout_rows, policy)
    summary = summarize_decisions(heldout_rows, decisions, "train_inner_fit_heldout_confirm")
    summary["gate_checks"] = gate_flags(summary, "heldout")
    summary["gate_pass"] = all(summary["gate_checks"].values())
    group_summary = split_group_summary(rows, strategy, {key: float(value) for key, value in policy["thresholds"].items()})
    return summary, per_image_rows(heldout_rows, decisions, groups, "heldout"), policy, group_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_csv", required=True)
    parser.add_argument("--domain_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--real_data_candidates", nargs="*", default=[])
    parser.add_argument("--min_group_sizes", nargs="+", type=int, default=[40, 80, 120])
    args = parser.parse_args()

    feature_rows = read_rows(Path(args.feature_csv))
    domain_rows = read_rows(Path(args.domain_csv))
    rows = merge_rows(feature_rows, domain_rows)
    add_buckets(rows)

    inventory_rows, real_decision = inventory_real_domain(args.real_data_candidates)
    grid_rows = []
    all_group_rows = []
    best = None
    best_bundle = None
    for strategy in STRATEGIES:
        for min_group_size in args.min_group_sizes:
            oof_summary, oof_per_image = run_oof(rows, strategy, min_group_size)
            heldout_summary, heldout_per_image, policy, group_rows = run_heldout(rows, strategy, min_group_size)
            score = (
                float(oof_summary.get("mean_delta") or -999)
                + float(oof_summary.get("hard_bottom25_delta") or -999)
                + 0.5 * float(heldout_summary.get("mean_delta") or -999)
                + 0.5 * float(heldout_summary.get("hard_bottom25_delta") or -999)
                - 3.0 * max(0.0, float(oof_summary.get("worst_regression_ratio") or 0.0) - 0.04)
                - 3.0 * max(0.0, float(heldout_summary.get("worst_regression_ratio") or 0.0) - 0.05)
            )
            row = {
                "strategy": strategy,
                "min_group_size": min_group_size,
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
            }
            grid_rows.append(row)
            all_group_rows.extend(group_rows)
            if best is None or score > best["score"]:
                best = row
                best_bundle = {
                    "policy": policy,
                    "oof_summary": oof_summary,
                    "heldout_summary": heldout_summary,
                    "oof_per_image": oof_per_image,
                    "heldout_per_image": heldout_per_image,
                    "group_rows": group_rows,
                }

    assert best is not None and best_bundle is not None
    output_dir = Path(args.output_dir)
    grid_rows.sort(key=lambda row: row["score"], reverse=True)
    write_csv(output_dir / "v18_domain_adaptation_policy_grid.csv", grid_rows)
    write_csv(output_dir / "v18_domain_adaptation_group_summary.csv", all_group_rows)
    write_csv(output_dir / "v18_domain_adaptation_oof_per_image.csv", best_bundle["oof_per_image"])
    write_csv(output_dir / "v18_domain_adaptation_heldout_per_image.csv", best_bundle["heldout_per_image"])
    status = {
        "route": "ConvIR-Dehaze-v1.8-ExecutionQueue",
        "stage": "domain adaptation table diagnostic",
        "status": "COMPLETED_DOMAIN_ADAPTATION_TABLE_ANALYSIS",
        "locked_test_touched": False,
        "feature_csv": args.feature_csv,
        "domain_csv": args.domain_csv,
        "row_count": len(rows),
        "real_domain_inventory": inventory_rows,
        "real_domain_decision": real_decision,
        "grid_count": len(grid_rows),
        "best_policy_grid_row": best,
        "best_policy": best_bundle["policy"],
        "oof_summary": best_bundle["oof_summary"],
        "heldout_summary": best_bundle["heldout_summary"],
        "decision": (
            "DOMAIN_POLICY_GATE_PASS_REAL_DATA_STILL_SEPARATE"
            if best_bundle["oof_summary"]["gate_pass"] and best_bundle["heldout_summary"]["gate_pass"]
            else "DOMAIN_POLICY_GATE_FAIL_CONTINUE_V18_QUEUE"
        ),
        "outputs": {
            "policy_grid": "v18_domain_adaptation_policy_grid.csv",
            "group_summary": "v18_domain_adaptation_group_summary.csv",
            "oof_per_image": "v18_domain_adaptation_oof_per_image.csv",
            "heldout_per_image": "v18_domain_adaptation_heldout_per_image.csv",
        },
    }
    write_json(output_dir / "v18_domain_adaptation_summary.json", status)
    write_json(output_dir / "v18_domain_adaptation_status.json", status)
    print(json.dumps(status, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
