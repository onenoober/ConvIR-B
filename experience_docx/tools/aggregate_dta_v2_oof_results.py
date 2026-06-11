#!/usr/bin/env python3
"""Aggregate DTA-v2 OOF comparison and t_pred audit artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import statistics
from pathlib import Path
from typing import Any


RUN_RE = re.compile(
    r"^dta_v2_(?P<stage>[^_]+)_(?P<scope>adapter_only|adapter_neighbors)_"
    r"(?P<mode>[^_]+)_seed(?P<seed>\d+)_f(?P<fold>\d+)_compare$"
)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    k = (len(ordered) - 1) * pct / 100.0
    lower = math.floor(k)
    upper = math.ceil(k)
    if lower == upper:
        return ordered[int(k)]
    return ordered[lower] * (upper - k) + ordered[upper] * (k - lower)


def finite_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [
        {
            "name": row.get("name", ""),
            "original_psnr": finite_float(row.get("original_psnr")),
            "delta_psnr": finite_float(row.get("delta_psnr")),
            "delta_ssim": finite_float(row.get("delta_ssim")),
        }
        for row in rows
    ]
    usable = [
        row
        for row in usable
        if math.isfinite(row["original_psnr"])
        and math.isfinite(row["delta_psnr"])
        and math.isfinite(row["delta_ssim"])
    ]
    if not usable:
        return {"common_count": 0}
    deltas = [row["delta_psnr"] for row in usable]
    ssim_deltas = [row["delta_ssim"] for row in usable]
    sorted_by_original = sorted(usable, key=lambda row: row["original_psnr"])
    bucket_count = max(1, len(usable) // 4)
    hard = sorted_by_original[:bucket_count]
    easy = sorted_by_original[-bucket_count:]
    sorted_deltas = sorted(deltas)
    tail_count = max(1, len(deltas) // 10)
    strong_cut = percentile([row["original_psnr"] for row in usable], 75)
    strong = [row for row in usable if row["original_psnr"] >= strong_cut]
    strong_regressions = [row for row in strong if row["delta_psnr"] <= -0.05]
    worst_regressions = [row for row in usable if row["delta_psnr"] <= -0.20]
    return {
        "common_count": len(usable),
        "mean_psnr_delta": statistics.mean(deltas),
        "median_psnr_delta": statistics.median(deltas),
        "p5_psnr_delta": percentile(deltas, 5),
        "p95_psnr_delta": percentile(deltas, 95),
        "hard_bottom25_psnr_delta": statistics.mean(row["delta_psnr"] for row in hard),
        "easy_top25_psnr_delta": statistics.mean(row["delta_psnr"] for row in easy),
        "worst10pct_mean_psnr_delta": statistics.mean(sorted_deltas[:tail_count]),
        "best10pct_mean_psnr_delta": statistics.mean(sorted_deltas[-tail_count:]),
        "worst10img_mean_psnr_delta": statistics.mean(sorted_deltas[:10]),
        "best10img_mean_psnr_delta": statistics.mean(sorted_deltas[-10:]),
        "mean_ssim_delta": statistics.mean(ssim_deltas),
        "positive_ratio": sum(delta > 0 for delta in deltas) / len(deltas),
        "strong_reference_cut_psnr": strong_cut,
        "strong_reference_count": len(strong),
        "strong_regression_count_delta_le_-0.05": len(strong_regressions),
        "worst_regression_count_delta_le_-0.20": len(worst_regressions),
    }


def bootstrap_ci(values: list[float], *, iterations: int, seed: int) -> dict[str, float | int]:
    values = [float(v) for v in values if math.isfinite(float(v))]
    if not values:
        return {"count": 0, "mean": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")}
    rng = random.Random(seed)
    means = []
    for _ in range(iterations):
        total = 0.0
        for _ in values:
            total += values[rng.randrange(len(values))]
        means.append(total / len(values))
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "ci95_low": percentile(means, 2.5),
        "ci95_high": percentile(means, 97.5),
        "iterations": iterations,
    }


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def wilcoxon_signed_rank_approx(values: list[float]) -> dict[str, float | int]:
    nonzero = [float(v) for v in values if math.isfinite(float(v)) and abs(float(v)) > 1e-12]
    n = len(nonzero)
    if n == 0:
        return {"n": 0, "w_plus": 0.0, "w_minus": 0.0, "z": float("nan"), "p_two_sided": float("nan")}
    indexed = sorted(enumerate(nonzero), key=lambda item: abs(item[1]))
    ranks = [0.0] * n
    start = 0
    while start < n:
        end = start + 1
        while end < n and abs(indexed[end][1]) == abs(indexed[start][1]):
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        for idx in range(start, end):
            ranks[indexed[idx][0]] = avg_rank
        start = end
    w_plus = sum(rank for rank, value in zip(ranks, nonzero) if value > 0)
    w_minus = sum(rank for rank, value in zip(ranks, nonzero) if value < 0)
    expected = n * (n + 1) / 4.0
    variance = n * (n + 1) * (2 * n + 1) / 24.0
    z = (w_plus - expected) / math.sqrt(variance) if variance > 0 else float("nan")
    p = 2.0 * min(normal_cdf(z), 1.0 - normal_cdf(z)) if math.isfinite(z) else float("nan")
    return {"n": n, "w_plus": w_plus, "w_minus": w_minus, "z": z, "p_two_sided": max(0.0, min(1.0, p))}


def collect_run(compare_dir: Path) -> dict[str, Any] | None:
    match = RUN_RE.match(compare_dir.name)
    if not match:
        return None
    meta = match.groupdict()
    run_id = compare_dir.name[len("dta_v2_") : -len("_compare")]
    compare_jsons = sorted(compare_dir.glob("scout_eval_compare_*.json"))
    per_image_csvs = sorted(compare_dir.glob("scout_eval_per_image_*.csv"))
    if not compare_jsons or not per_image_csvs:
        return None
    tpred_dir = compare_dir.parent / f"dta_v2_{run_id}_tpred"
    tpred_jsons = sorted(tpred_dir.glob("dta_v2_tpred_quality_*.json"))
    comparison = read_json(compare_jsons[0]).get("comparison", {})
    rows = read_rows(per_image_csvs[0])
    tpred_metrics = read_json(tpred_jsons[0]).get("metrics", {}) if tpred_jsons else {}
    return {
        **meta,
        "fold": int(meta["fold"]),
        "seed": int(meta["seed"]),
        "run_id": run_id,
        "compare_json": str(compare_jsons[0]),
        "per_image_csv": str(per_image_csvs[0]),
        "tpred_json": str(tpred_jsons[0]) if tpred_jsons else "",
        "comparison": comparison,
        "tpred_metrics": tpred_metrics,
        "rows": rows,
    }


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    evidence_root = Path(args.evidence_root)
    runs = []
    for compare_dir in sorted(evidence_root.glob("dta_v2_*_compare")):
        run = collect_run(compare_dir)
        if run is None:
            continue
        if args.stage and run["stage"] != args.stage:
            continue
        if args.scope and run["scope"] != args.scope:
            continue
        if args.seed is not None and run["seed"] != args.seed:
            continue
        runs.append(run)

    per_run_rows = []
    groups: dict[tuple[str, str, str, int], list[dict[str, Any]]] = {}
    for run in runs:
        key = (run["stage"], run["scope"], run["mode"], run["seed"])
        groups.setdefault(key, []).append(run)
        row = {
            "stage": run["stage"],
            "scope": run["scope"],
            "mode": run["mode"],
            "seed": run["seed"],
            "fold": run["fold"],
            "run_id": run["run_id"],
        }
        for metric, value in run["comparison"].items():
            row[metric] = value
        for metric in (
            "t_l1_mean",
            "spearman_tpred_tgt_mean",
            "spearman_depth_neglogt_mean",
            "stage2_gate_mean_mean",
            "stage3_gate_mean_mean",
            "confidence_mean_mean",
        ):
            row[metric] = run["tpred_metrics"].get(metric)
        per_run_rows.append(row)

    group_rows = []
    bootstrap_report = {}
    for key, group_runs in sorted(groups.items()):
        stage, scope, mode, seed = key
        combined_rows: list[dict[str, Any]] = []
        folds = []
        for run in sorted(group_runs, key=lambda item: item["fold"]):
            folds.append(run["fold"])
            combined_rows.extend(run["rows"])
        summary = summarize_rows(combined_rows)
        deltas = [finite_float(row.get("delta_psnr")) for row in combined_rows]
        bootstrap = bootstrap_ci(deltas, iterations=args.bootstrap_iterations, seed=args.bootstrap_seed + seed)
        wilcoxon = wilcoxon_signed_rank_approx(deltas)
        bootstrap_report[f"{stage}_{scope}_{mode}_seed{seed}"] = {
            "folds": folds,
            "bootstrap_mean_psnr_delta": bootstrap,
            "wilcoxon_signed_rank_psnr_delta": wilcoxon,
        }
        row = {
            "stage": stage,
            "scope": scope,
            "mode": mode,
            "seed": seed,
            "folds": ",".join(str(fold) for fold in folds),
            "fold_count": len(folds),
        }
        row.update(summary)
        for metric in (
            "t_l1_mean",
            "spearman_tpred_tgt_mean",
            "spearman_depth_neglogt_mean",
            "stage2_gate_mean_mean",
            "stage3_gate_mean_mean",
            "confidence_mean_mean",
        ):
            vals = [finite_float(run["tpred_metrics"].get(metric)) for run in group_runs]
            vals = [val for val in vals if math.isfinite(val)]
            row[metric] = statistics.mean(vals) if vals else float("nan")
        row["bootstrap_ci95_low"] = bootstrap["ci95_low"]
        row["bootstrap_ci95_high"] = bootstrap["ci95_high"]
        row["wilcoxon_p_two_sided"] = wilcoxon["p_two_sided"]
        group_rows.append(row)

    output_dir = Path(args.output_dir) if args.output_dir else evidence_root
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_prefix
    write_csv(output_dir / f"{prefix}_per_run.csv", per_run_rows)
    write_csv(output_dir / f"{prefix}_by_mode.csv", group_rows)
    report = {
        "evidence_root": str(evidence_root),
        "run_count": len(runs),
        "filters": {"stage": args.stage, "scope": args.scope, "seed": args.seed},
        "per_run": per_run_rows,
        "by_mode": group_rows,
        "bootstrap_wilcoxon": bootstrap_report,
    }
    (output_dir / f"{prefix}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output_dir / f"{prefix}_bootstrap_wilcoxon_report.json").write_text(
        json.dumps(bootstrap_report, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence_root", required=True)
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--output_prefix", default="dta_v2_oof_aggregate")
    parser.add_argument("--stage", default="")
    parser.add_argument("--scope", default="")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--bootstrap_iterations", type=int, default=2000)
    parser.add_argument("--bootstrap_seed", type=int, default=9409)
    args = parser.parse_args()
    report = aggregate(args)
    print(
        "DTA_V2_OOF_AGGREGATE_OK "
        f"runs={report['run_count']} output={Path(args.output_dir or args.evidence_root) / args.output_prefix}",
        flush=True,
    )


if __name__ == "__main__":
    main()
