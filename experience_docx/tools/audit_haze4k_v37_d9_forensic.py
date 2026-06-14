#!/usr/bin/env python3
"""DTA-v3.7 D9 failure forensics without post-test tuning."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


NUMERIC_FEATURES = [
    "A0_PSNR",
    "dPSNR",
    "dSSIM",
    "input_brightness_mean",
    "input_texture_mean",
    "airlight_fallback_mean",
    "depth_mean",
    "depth_std",
    "zero_delta_psnr",
    "shuffle_delta_psnr",
    "normal_delta_psnr",
    "od_action_res_abs_mean",
    "od_action_luma_abs_mean",
    "od_action_edge_mean",
    "od_sky_ratio",
    "od_highbright_ratio",
    "od_lowtex_ratio",
    "outq_luma_mean",
    "outq_edge_mean",
    "outq_sat_mean",
    "pred_gain",
    "pred_pos_prob",
    "pred_strong_prob",
    "pred_severe_prob",
    "policy_score",
]


def fnum(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def quantile(values: list[float], q: float) -> float:
    vals = sorted(v for v in values if math.isfinite(v))
    if not vals:
        return float("nan")
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    frac = pos - lo
    return vals[lo] * (1 - frac) + vals[hi] * frac


def summarize(rows: list[dict[str, str]]) -> dict[str, Any]:
    count = len(rows)
    if not rows:
        return {
            "count": 0,
            "mean_dPSNR": "",
            "hard_bottom25_dPSNR": "",
            "easy_top25_dPSNR": "",
            "dSSIM": "",
            "positive_ratio": "",
            "worst_count_le_-0.20": "",
            "worst_per_600": "",
            "strong_count_le_-0.05": "",
            "strong_per_600": "",
        }
    deltas = [fnum(r.get("dPSNR")) for r in rows]
    ssims = [fnum(r.get("dSSIM")) for r in rows]
    a0s = [fnum(r.get("A0_PSNR")) for r in rows]
    order = sorted(range(count), key=lambda i: a0s[i])
    k = max(1, count // 4)
    strong_cut = quantile(a0s, 0.75)
    strong = sum(1 for d, a in zip(deltas, a0s, strict=False) if a >= strong_cut and d <= -0.05)
    worst = sum(1 for d in deltas if d <= -0.20)
    return {
        "count": count,
        "mean_dPSNR": mean(deltas),
        "hard_bottom25_dPSNR": mean([deltas[i] for i in order[:k]]),
        "easy_top25_dPSNR": mean([deltas[i] for i in order[-k:]]),
        "dSSIM": mean(ssims),
        "positive_ratio": sum(1 for d in deltas if d > 0.0) / count,
        "worst_count_le_-0.20": worst,
        "worst_per_600": worst / count * 600.0,
        "strong_count_le_-0.05": strong,
        "strong_per_600": strong / count * 600.0,
    }


def numeric_buckets(rows: list[dict[str, str]], column: str) -> list[tuple[str, list[dict[str, str]]]]:
    vals = [fnum(r.get(column), float("nan")) for r in rows]
    vals = [v for v in vals if math.isfinite(v)]
    if not vals:
        return []
    q1, q2, q3 = quantile(vals, 0.25), quantile(vals, 0.50), quantile(vals, 0.75)
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        value = fnum(row.get(column), float("nan"))
        if not math.isfinite(value):
            label = "missing"
        elif value <= q1:
            label = "q1_low"
        elif value <= q2:
            label = "q2"
        elif value <= q3:
            label = "q3"
        else:
            label = "q4_high"
        groups[label].append(row)
    return sorted(groups.items())


def severity_bucket(delta: float) -> str:
    if delta <= -0.20:
        return "severe_loss_le_-0.20"
    if delta <= -0.05:
        return "strong_loss_le_-0.05"
    if delta < 0:
        return "mild_loss"
    if delta < 0.05:
        return "small_gain_0_to_0.05"
    if delta < 0.20:
        return "gain_0.05_to_0.20"
    return "large_gain_ge_0.20"


def feature_stats(rows: list[dict[str, str]], column: str) -> dict[str, float]:
    values = [fnum(r.get(column), float("nan")) for r in rows]
    values = [v for v in values if math.isfinite(v)]
    if not values:
        return {"mean": float("nan"), "std": float("nan")}
    return {"mean": mean(values), "std": pstdev(values) if len(values) > 1 else 0.0}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dta-evidence-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=80)
    args = parser.parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    d9_rows = read_csv(args.dta_evidence_dir / "v37_d9_locked_fixed_policy_selected_actions.csv")
    d8_rows = read_csv(args.dta_evidence_dir / "v37_d8_fixed_formal_selected_actions.csv")

    bucket_rows: list[dict[str, Any]] = []

    def add_bucket(feature: str, bucket: str, rows: list[dict[str, str]], source: str) -> None:
        row = {"source": source, "feature": feature, "bucket": bucket}
        row.update(summarize(rows))
        bucket_rows.append(row)

    for source, rows in [("D9_locked", d9_rows), ("D8_train_derived", d8_rows)]:
        by_variant: dict[str, list[dict[str, str]]] = defaultdict(list)
        by_action: dict[str, list[dict[str, str]]] = defaultdict(list)
        by_severity: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            by_variant[row.get("variant", "")].append(row)
            by_action[row.get("action", "")].append(row)
            by_severity[severity_bucket(fnum(row.get("dPSNR")))].append(row)
        for value, scoped in sorted(by_variant.items()):
            add_bucket("variant", value, scoped, source)
        for value, scoped in sorted(by_action.items()):
            add_bucket("action", value, scoped, source)
        for value, scoped in sorted(by_severity.items()):
            add_bucket("dpsnr_severity", value, scoped, source)
        for column in [
            "A0_PSNR",
            "input_brightness_mean",
            "input_texture_mean",
            "depth_mean",
            "airlight_fallback_mean",
            "od_action_luma_abs_mean",
            "od_sky_ratio",
            "pred_gain",
            "pred_severe_prob",
        ]:
            for value, scoped in numeric_buckets(rows, column):
                add_bucket(column, value, scoped, source)

    bucket_fields = [
        "source",
        "feature",
        "bucket",
        "count",
        "mean_dPSNR",
        "hard_bottom25_dPSNR",
        "easy_top25_dPSNR",
        "dSSIM",
        "positive_ratio",
        "worst_count_le_-0.20",
        "worst_per_600",
        "strong_count_le_-0.05",
        "strong_per_600",
    ]
    write_csv(out_dir / "v37_d9_forensic_bucket_summary.csv", bucket_rows, bucket_fields)

    top_regressions = sorted(d9_rows, key=lambda r: fnum(r.get("dPSNR")))[: args.top_k]
    top_fields = [
        "image_id",
        "fold",
        "seed",
        "variant",
        "action",
        "alpha",
        "A0_PSNR",
        "dPSNR",
        "dSSIM",
        "input_brightness_mean",
        "input_texture_mean",
        "depth_mean",
        "airlight_fallback_mean",
        "od_action_luma_abs_mean",
        "od_sky_ratio",
        "pred_gain",
        "pred_severe_prob",
        "policy_score",
    ]
    write_csv(out_dir / "v37_d9_forensic_top_regressions.csv", top_regressions, top_fields)

    drift_rows: list[dict[str, Any]] = []
    for column in NUMERIC_FEATURES:
        d8_stat = feature_stats(d8_rows, column)
        d9_stat = feature_stats(d9_rows, column)
        denom = d8_stat["std"] if math.isfinite(d8_stat["std"]) and d8_stat["std"] > 1e-12 else 1.0
        drift_rows.append(
            {
                "feature": column,
                "d8_mean": d8_stat["mean"],
                "d8_std": d8_stat["std"],
                "d9_mean": d9_stat["mean"],
                "d9_std": d9_stat["std"],
                "mean_delta_d9_minus_d8": d9_stat["mean"] - d8_stat["mean"],
                "std_units_vs_d8": (d9_stat["mean"] - d8_stat["mean"]) / denom,
            }
        )
    write_csv(
        out_dir / "v37_d9_forensic_feature_drift.csv",
        drift_rows,
        ["feature", "d8_mean", "d8_std", "d9_mean", "d9_std", "mean_delta_d9_minus_d8", "std_units_vs_d8"],
    )

    summary = {
        "route": "DTA-v3.7 U-TQS-Mix",
        "phase": "D9 locked failure forensic",
        "locked_test_touched": True,
        "post_test_tuning_allowed": False,
        "d9_rows": len(d9_rows),
        "d8_rows": len(d8_rows),
        "d9_summary": summarize(d9_rows),
        "d8_summary": summarize(d8_rows),
        "decision": "D9_FORENSIC_COMPLETE_NO_TUNING",
        "top_regression_min_dPSNR": fnum(top_regressions[0].get("dPSNR")) if top_regressions else None,
    }
    with (out_dir / "v37_d9_forensic_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    lines = [
        "# DTA-v3.7 D9 Forensic Audit",
        "",
        "Decision: `D9_FORENSIC_COMPLETE_NO_TUNING`",
        "",
        "This audit describes where the sealed policy failed. It is not a tuning input for DTA-v3.7 thresholds, features, actions, or checkpoints.",
        "",
        "## Summary",
        "",
    ]
    for key, value in summary["d9_summary"].items():
        lines.append(f"- D9 `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `v37_d9_forensic_bucket_summary.csv`",
            "- `v37_d9_forensic_top_regressions.csv`",
            "- `v37_d9_forensic_feature_drift.csv`",
            "- `v37_d9_forensic_summary.json`",
        ]
    )
    (out_dir / "v37_d9_forensic_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"V37_D9_FORENSIC_OK rows={len(d9_rows)} out={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
