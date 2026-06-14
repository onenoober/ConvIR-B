#!/usr/bin/env python3
"""C1 risk/correctability map for the available A0/FullUDP evidence."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Callable


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


def parse_name_params(name: str) -> tuple[float, float]:
    stem = Path(name).stem
    parts = stem.split("_")
    if len(parts) >= 3:
        return fnum(parts[1], float("nan")), fnum(parts[2], float("nan"))
    return float("nan"), float("nan")


def enrich(row: dict[str, str]) -> dict[str, Any]:
    p1, p2 = parse_name_params(row.get("name", ""))
    delta = fnum(row.get("delta_psnr"))
    return {
        "image_id": row.get("name", ""),
        "split": row.get("split", ""),
        "bucket": row.get("bucket", ""),
        "A0_PSNR": fnum(row.get("a0_psnr")),
        "FullUDP_PSNR": fnum(row.get("udpnet_psnr")),
        "dPSNR": delta,
        "dSSIM": fnum(row.get("delta_ssim")),
        "name_param_1": p1,
        "name_param_2": p2,
        "target_positive": int(delta > 0.0),
        "target_high_gain_ge_0.20": int(delta >= 0.20),
        "target_severe_loss_le_-0.20": int(delta <= -0.20),
        "target_loss_lt_0": int(delta < 0.0),
    }


def summarize(records: list[dict[str, Any]], *, policy: bool = False) -> dict[str, Any]:
    count = len(records)
    if count == 0:
        return {
            "count": 0,
            "coverage": 0,
            "mean_dPSNR": "",
            "hard_bottom25_dPSNR": "",
            "easy_top25_dPSNR": "",
            "dSSIM": "",
            "positive_ratio": "",
            "nonnegative_ratio": "",
            "severe_loss_count": "",
            "severe_loss_per_600": "",
            "strong_loss_count": "",
            "strong_loss_per_600": "",
        }
    deltas = [float(r["dPSNR"]) for r in records]
    ssims = [float(r["dSSIM"]) for r in records]
    a0s = [float(r["A0_PSNR"]) for r in records]
    order = sorted(range(count), key=lambda i: a0s[i])
    k = max(1, count // 4)
    severe = sum(1 for d in deltas if d <= -0.20)
    strong = sum(1 for d in deltas if d <= -0.05)
    return {
        "count": count,
        "coverage": sum(1 for r in records if r.get("chosen_fulludp", True)) / count if policy else 1.0,
        "mean_dPSNR": mean(deltas),
        "hard_bottom25_dPSNR": mean([deltas[i] for i in order[:k]]),
        "easy_top25_dPSNR": mean([deltas[i] for i in order[-k:]]),
        "dSSIM": mean(ssims),
        "positive_ratio": sum(1 for d in deltas if d > 0.0) / count,
        "nonnegative_ratio": sum(1 for d in deltas if d >= 0.0) / count,
        "severe_loss_count": severe,
        "severe_loss_per_600": severe / count * 600.0,
        "strong_loss_count": strong,
        "strong_loss_per_600": strong / count * 600.0,
    }


def auc_score(values: list[float], labels: list[int]) -> float:
    pairs = [(v, y) for v, y in zip(values, labels, strict=False) if math.isfinite(v)]
    if not pairs:
        return float("nan")
    pos = sum(y for _, y in pairs)
    neg = len(pairs) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    pairs.sort(key=lambda x: x[0])
    rank_sum = 0.0
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        rank_sum += avg_rank * sum(y for _, y in pairs[i:j])
        i = j
    return (rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)


def q_buckets(rows: list[dict[str, Any]], feature: str) -> dict[str, list[dict[str, Any]]]:
    vals = [float(r[feature]) for r in rows if math.isfinite(float(r[feature]))]
    if not vals:
        return {}
    q1, q2, q3 = quantile(vals, 0.25), quantile(vals, 0.50), quantile(vals, 0.75)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        value = float(row[feature])
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
    return groups


def policy_records(rows: list[dict[str, Any]], predicate: Callable[[dict[str, Any]], bool]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        choose = predicate(row)
        dpsnr = float(row["dPSNR"]) if choose else 0.0
        dssim = float(row["dSSIM"]) if choose else 0.0
        clone = dict(row)
        clone["chosen_fulludp"] = choose
        clone["dPSNR"] = dpsnr
        clone["dSSIM"] = dssim
        out.append(clone)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fulludp-eval-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [enrich(r) for r in read_csv(args.fulludp_eval_dir / "udpnet_convir_bucket_compare.csv")]

    bin_rows: list[dict[str, Any]] = []
    for feature in ["split", "bucket"]:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[str(row[feature])].append(row)
        for value, scoped in sorted(groups.items()):
            rec = {"feature": feature, "bucket": value}
            rec.update(summarize(scoped))
            bin_rows.append(rec)
    for feature in ["A0_PSNR", "name_param_1", "name_param_2"]:
        for value, scoped in sorted(q_buckets(rows, feature).items()):
            rec = {"feature": feature, "bucket": value}
            rec.update(summarize(scoped))
            bin_rows.append(rec)
    bin_fields = [
        "feature",
        "bucket",
        "count",
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
    ]
    write_csv(out_dir / "v20_c1_strong_expert_gain_risk_bins.csv", bin_rows, bin_fields)

    feature_rows: list[dict[str, Any]] = []
    continuous = ["A0_PSNR", "name_param_1", "name_param_2"]
    targets = ["target_positive", "target_high_gain_ge_0.20", "target_severe_loss_le_-0.20", "target_loss_lt_0"]
    for feature in continuous:
        values = [float(r[feature]) for r in rows]
        for target in targets:
            labels = [int(r[target]) for r in rows]
            auc = auc_score(values, labels)
            feature_rows.append(
                {
                    "feature": feature,
                    "target": target,
                    "auc_raw": auc,
                    "auc_best_orientation": max(auc, 1.0 - auc) if math.isfinite(auc) else "",
                    "direction": "higher" if math.isfinite(auc) and auc >= 0.5 else "lower",
                    "mean": mean([v for v in values if math.isfinite(v)]),
                    "std": pstdev([v for v in values if math.isfinite(v)]),
                }
            )
    write_csv(out_dir / "v20_c1_feature_auc.csv", feature_rows, ["feature", "target", "auc_raw", "auc_best_orientation", "direction", "mean", "std"])

    policy_rows: list[dict[str, Any]] = []

    def add_policy(name: str, predicate: Callable[[dict[str, Any]], bool]) -> None:
        scoped = policy_records(rows, predicate)
        rec = {"policy_id": name}
        rec.update(summarize(scoped, policy=True))
        rec["screen_gate_pass"] = (
            fnum(rec["mean_dPSNR"]) >= 0.12
            and fnum(rec["hard_bottom25_dPSNR"]) >= 0.20
            and fnum(rec["easy_top25_dPSNR"]) >= -0.02
            and fnum(rec["dSSIM"]) >= 0.0
            and fnum(rec["severe_loss_per_600"]) <= 48.0
        )
        rec["score"] = fnum(rec["mean_dPSNR"]) + 0.25 * fnum(rec["hard_bottom25_dPSNR"]) - 0.002 * fnum(rec["severe_loss_per_600"])
        policy_rows.append(rec)

    add_policy("all_fulludp", lambda r: True)
    add_policy("a0_anchor", lambda r: False)
    for split in sorted({r["split"] for r in rows}):
        add_policy(f"split_eq_{split}", lambda r, split=split: r["split"] == split)
    for bucket in sorted({r["bucket"] for r in rows}):
        add_policy(f"bucket_eq_{bucket}", lambda r, bucket=bucket: r["bucket"] == bucket)
    for feature in continuous:
        vals = [float(r[feature]) for r in rows if math.isfinite(float(r[feature]))]
        thresholds = sorted({quantile(vals, q) for q in [0.10, 0.20, 0.25, 0.33, 0.50, 0.67, 0.75, 0.80, 0.90]})
        for threshold in thresholds:
            add_policy(f"{feature}_le_{threshold:.6g}", lambda r, feature=feature, threshold=threshold: float(r[feature]) <= threshold)
            add_policy(f"{feature}_ge_{threshold:.6g}", lambda r, feature=feature, threshold=threshold: float(r[feature]) >= threshold)
            add_policy(
                f"val_hard_and_{feature}_le_{threshold:.6g}",
                lambda r, feature=feature, threshold=threshold: r["split"] == "val_hard" and float(r[feature]) <= threshold,
            )
            add_policy(
                f"val_hard_and_{feature}_ge_{threshold:.6g}",
                lambda r, feature=feature, threshold=threshold: r["split"] == "val_hard" and float(r[feature]) >= threshold,
            )

    policy_rows.sort(key=lambda r: (bool(r["screen_gate_pass"]), fnum(r["score"])), reverse=True)
    policy_fields = ["policy_id"] + bin_fields[2:] + ["screen_gate_pass", "score"]
    write_csv(out_dir / "v20_c1_simple_policy_grid.csv", policy_rows, policy_fields)

    safe = [r for r in policy_rows if r["screen_gate_pass"]]
    best = policy_rows[0] if policy_rows else {}
    best_safe = safe[0] if safe else None
    max_auc = max([fnum(r["auc_best_orientation"], float("nan")) for r in feature_rows if r["target"] in {"target_positive", "target_severe_loss_le_-0.20"} and r["auc_best_orientation"] != ""], default=float("nan"))
    if best_safe:
        decision = "C1_SIMPLE_FEATURE_POLICY_SCREEN_PASS_START_C2_TABLE_ROUTER"
    elif fnum(best.get("mean_dPSNR")) >= 0.12 and fnum(best.get("hard_bottom25_dPSNR")) >= 0.20:
        decision = "C1_GAIN_EXISTS_TAIL_RISK_NEEDS_OUTPUTDIFF_FEATURES"
    else:
        decision = "C1_SIMPLE_FEATURES_INSUFFICIENT_REACQUIRE_STRONG_EXPERT_OUTPUTS"

    summary = {
        "route": "Haze4K-v2.0 StrongExpert-GainMix",
        "phase": "C1 Strong Expert Risk/Correctability Map",
        "locked_test_touched": False,
        "rows": len(rows),
        "global_fulludp": summarize(rows),
        "max_simple_auc_positive_or_risk": max_auc,
        "best_policy": best,
        "best_safe_policy": best_safe,
        "decision": decision,
    }
    with (out_dir / "v20_c1_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    lines = [
        "# Haze4K v2.0 C1 Strong Expert Risk/Correctability Map",
        "",
        f"Decision: `{decision}`",
        "",
        "This phase uses existing internal-validation A0/FullUDP endpoint evidence only. Locked test data was not touched.",
        "",
        "## Best Simple Policy",
        "",
    ]
    for key, value in best.items():
        lines.append(f"- `{key}`: `{value}`")
    if best_safe:
        lines.extend(["", "## Best Screen-Passing Simple Policy", ""])
        for key, value in best_safe.items():
            lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- C1 is a risk-map and separability audit, not a final deployable router.",
            "- If no simple policy passes the screen gate, the next efficient step is to reacquire or render strong-expert outputs on `convir-4090` and compute FullUDP-A0 output-difference features before C2.",
        ]
    )
    (out_dir / "v20_c1_decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"V20_C1_RISK_MAP_OK decision={decision} out={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
