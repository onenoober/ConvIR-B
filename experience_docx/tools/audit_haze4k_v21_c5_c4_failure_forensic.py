#!/usr/bin/env python3
"""C5 forensic audit for the v2.0 C4 formal gap.

This audit is intentionally non-tuning: it replays the sealed C2d/C4
train-derived policy family from text evidence, decomposes the hard/positive
shortfall, and writes small CSV/JSON/Markdown evidence. It does not touch locked
Haze4K test data and does not emit thresholds or new policies.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from audit_haze4k_v20_c2b_multirule_router import fnum, write_csv
from audit_haze4k_v20_c3_shifted_validation import alpha_key, choose_threshold, mask_for_threshold
from audit_haze4k_v20_c4_formal_5x3 import SEEDS, seeded_fold_id


TARGET_POSITIVE_RATIO = 0.70
TARGET_HARD_DPSNR = 0.30
BASELINE_ALPHA = 0.25


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def available_alphas(rows: list[dict[str, Any]]) -> list[float]:
    if not rows:
        return []
    out: list[float] = []
    for field in rows[0]:
        if field.startswith("dPSNR_a"):
            key = field.removeprefix("dPSNR_")
            try:
                out.append(float(key[1:].replace("p", ".")))
            except ValueError:
                continue
    return sorted(set(out))


def parse_name_params(name: str) -> tuple[float, float]:
    stem = Path(name).stem
    parts = stem.split("_")
    airlight = float(parts[1]) if len(parts) > 1 else 1.0
    beta = float(parts[2]) if len(parts) > 2 else 1.0
    return airlight, beta


def rank_bins(values: list[float], bins: int = 4) -> list[str]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    labels = [""] * len(values)
    for rank, idx in enumerate(order):
        bin_idx = min(bins - 1, int(rank * bins / max(1, len(values))))
        labels[idx] = f"q{bin_idx + 1}"
    return labels


def build_group_labels(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    airlight: list[float] = []
    beta: list[float] = []
    for row in rows:
        a, b = parse_name_params(str(row["name"]))
        airlight.append(a)
        beta.append(b)
    return {
        "split": [str(row["split"]) for row in rows],
        "airlight_q4": rank_bins(airlight),
        "beta_haze_q4": rank_bins(beta),
        "depth_mean_q4": rank_bins([fnum(row["depth_mean"]) for row in rows]),
        "low_texture_input_grad_q4": rank_bins([fnum(row["input_grad_mean"]) for row in rows]),
        "input_dark_q4": rank_bins([fnum(row["input_dark_mean"]) for row in rows]),
        "diff_abs_q4": rank_bins([fnum(row["diff_abs_mean"]) for row in rows]),
        "a0_psnr_q4": rank_bins([fnum(row["A0_PSNR"]) for row in rows]),
    }


def replay_seed(rows: list[dict[str, Any]], seed: int) -> tuple[np.ndarray, list[dict[str, Any]]]:
    selected = np.zeros(len(rows), dtype=bool)
    fold_ids = np.array([seeded_fold_id(str(row["name"]), seed) for row in rows], dtype=np.int64)
    fold_records: list[dict[str, Any]] = []
    for fold in range(5):
        heldout_mask = fold_ids == fold
        train_rows = [row for row, keep in zip(rows, ~heldout_mask, strict=False) if keep]
        heldout_rows = [row for row, keep in zip(rows, heldout_mask, strict=False) if keep]
        chosen = choose_threshold(train_rows)
        heldout_selected = mask_for_threshold(heldout_rows, float(chosen["threshold"]))
        selected[heldout_mask] = heldout_selected
        fold_records.append(
            {
                "seed": seed,
                "fold": fold,
                "train_policy_id": chosen["policy_id"],
                "train_threshold": chosen["threshold"],
                "train_strict_gate_pass": chosen["strict_gate_pass"],
                "train_count": len(train_rows),
                "heldout_count": len(heldout_rows),
                "heldout_selected_count": int(heldout_selected.sum()),
            }
        )
    return selected, fold_records


def alpha_delta(row: dict[str, Any], alpha: float, metric: str = "dPSNR") -> float:
    return fnum(row.get(f"{metric}_{alpha_key(alpha)}"))


def best_alpha(row: dict[str, Any], alphas: list[float]) -> tuple[float, float, float]:
    best = (0.0, 0.0, 0.0)
    for alpha in alphas:
        dpsnr = alpha_delta(row, alpha, "dPSNR")
        dssim = alpha_delta(row, alpha, "dSSIM")
        if dpsnr > best[1]:
            best = (alpha, dpsnr, dssim)
    return best


def summarize_seed(rows: list[dict[str, Any]], selected: np.ndarray, alphas: list[float], seed: int) -> dict[str, Any]:
    count = len(rows)
    base_key = alpha_key(BASELINE_ALPHA)
    deltas = np.array([alpha_delta(row, BASELINE_ALPHA) if sel else 0.0 for row, sel in zip(rows, selected, strict=False)], dtype=np.float64)
    hard_count = max(1, count // 4)
    order = np.argsort([fnum(row["A0_PSNR"]) for row in rows])
    hard = order[:hard_count]
    selected_deltas = np.array([alpha_delta(row, BASELINE_ALPHA) for row, sel in zip(rows, selected, strict=False) if sel], dtype=np.float64)
    best_any = np.array([max(0.0, best_alpha(row, alphas)[1]) for row in rows], dtype=np.float64)
    return {
        "seed": seed,
        "count": count,
        "target_positive_count": int(math.ceil(TARGET_POSITIVE_RATIO * count)),
        "current_positive_count": int(np.sum(deltas > 0.0)),
        "positive_deficit_count": max(0, int(math.ceil(TARGET_POSITIVE_RATIO * count)) - int(np.sum(deltas > 0.0))),
        "selected_count": int(selected.sum()),
        "selected_positive_count": int(np.sum(selected_deltas > 0.0)) if selected_deltas.size else 0,
        "selected_negative_count": int(np.sum(selected_deltas <= 0.0)) if selected_deltas.size else 0,
        "selected_severe_count": int(np.sum(selected_deltas <= -0.20)) if selected_deltas.size else 0,
        "false_negative_any_alpha_positive_count": int(np.sum((~selected) & (best_any > 0.0))),
        "false_negative_alpha025_positive_count": int(np.sum((~selected) & (np.array([alpha_delta(row, BASELINE_ALPHA) for row in rows]) > 0.0))),
        "mean_dPSNR": float(np.mean(deltas)),
        "hard_bottom25_dPSNR": float(np.mean(deltas[hard])),
        "hard_target_gap": TARGET_HARD_DPSNR - float(np.mean(deltas[hard])),
        "hard_oracle_best_alpha_dPSNR": float(np.mean(best_any[hard])),
        "hard_oracle_gap_after_best_alpha": TARGET_HARD_DPSNR - float(np.mean(best_any[hard])),
        "baseline_alpha_column": f"dPSNR_{base_key}",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha_rows", type=Path, required=True)
    parser.add_argument("--c4_summary", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_csv(args.alpha_rows)
    c4_summary = json.loads(args.c4_summary.read_text(encoding="utf-8"))
    alphas = available_alphas(rows)
    if BASELINE_ALPHA not in alphas:
        raise ValueError(f"baseline alpha {BASELINE_ALPHA} missing from {args.alpha_rows}")

    seed_masks: dict[int, np.ndarray] = {}
    fold_records: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        selected, folds = replay_seed(rows, seed)
        seed_masks[seed] = selected
        fold_records.extend(folds)
        seed_rows.append(summarize_seed(rows, selected, alphas, seed))

    selected_vote = np.zeros(len(rows), dtype=np.float64)
    negative_vote = np.zeros(len(rows), dtype=np.float64)
    severe_vote = np.zeros(len(rows), dtype=np.float64)
    current_delta_sum = np.zeros(len(rows), dtype=np.float64)
    for seed, selected in seed_masks.items():
        _ = seed
        for idx, (row, sel) in enumerate(zip(rows, selected, strict=False)):
            delta = alpha_delta(row, BASELINE_ALPHA) if sel else 0.0
            current_delta_sum[idx] += delta
            if sel:
                selected_vote[idx] += 1.0
                negative_vote[idx] += float(delta <= 0.0)
                severe_vote[idx] += float(delta <= -0.20)
    current_delta_mean = current_delta_sum / max(1, len(SEEDS))
    best_alpha_rows: list[dict[str, Any]] = []
    best_any = np.zeros(len(rows), dtype=np.float64)
    best_alpha_values = np.zeros(len(rows), dtype=np.float64)
    for idx, row in enumerate(rows):
        b_alpha, b_dpsnr, b_dssim = best_alpha(row, alphas)
        best_any[idx] = max(0.0, b_dpsnr)
        best_alpha_values[idx] = b_alpha
        hard_rank = int(np.argsort([fnum(r["A0_PSNR"]) for r in rows]).tolist().index(idx))
        if hard_rank < max(1, len(rows) // 4):
            rec = {
                "name": row["name"],
                "split": row["split"],
                "A0_PSNR": row["A0_PSNR"],
                "selected_seed_count": int(selected_vote[idx]),
                "selected_negative_seed_count": int(negative_vote[idx]),
                "current_mean_dPSNR": current_delta_mean[idx],
                "best_alpha": b_alpha,
                "best_alpha_dPSNR": b_dpsnr,
                "best_alpha_dSSIM": b_dssim,
                "safe_high_alpha_exists": any(alpha >= 0.375 and alpha_delta(row, alpha) > 0.0 and alpha_delta(row, alpha, "dSSIM") >= 0.0 for alpha in alphas),
            }
            for alpha in alphas:
                rec[f"dPSNR_{alpha_key(alpha)}"] = alpha_delta(row, alpha)
                rec[f"dSSIM_{alpha_key(alpha)}"] = alpha_delta(row, alpha, "dSSIM")
            best_alpha_rows.append(rec)

    group_labels = build_group_labels(rows)
    bin_rows: list[dict[str, Any]] = []
    false_negative_any = (selected_vote == 0) & (best_any > 0.0)
    for dimension, labels in group_labels.items():
        for label in sorted(set(labels)):
            idx = np.array([lab == label for lab in labels], dtype=bool)
            if not idx.any():
                continue
            bin_rows.append(
                {
                    "dimension": dimension,
                    "bin": label,
                    "count": int(idx.sum()),
                    "selected_vote_count": int(selected_vote[idx].sum()),
                    "selected_negative_vote_count": int(negative_vote[idx].sum()),
                    "selected_severe_vote_count": int(severe_vote[idx].sum()),
                    "false_negative_any_alpha_count": int(false_negative_any[idx].sum()),
                    "current_mean_dPSNR": float(current_delta_mean[idx].mean()),
                    "best_alpha_oracle_mean_dPSNR": float(best_any[idx].mean()),
                    "mean_selected_seed_fraction": float((selected_vote[idx] / max(1, len(SEEDS))).mean()),
                }
            )

    selected_negative_rows: list[dict[str, Any]] = []
    feature_fields = [
        "input_mean",
        "input_grad_mean",
        "input_dark_mean",
        "depth_mean",
        "a0_mean",
        "diff_signed_mean",
        "diff_abs_mean",
        "diff_abs_p90",
        "diff_to_a0_ratio",
        "a0_udp_psnr",
    ]
    for idx, row in enumerate(rows):
        if negative_vote[idx] <= 0:
            continue
        b_alpha, b_dpsnr, b_dssim = best_alpha(row, alphas)
        rec = {
            "name": row["name"],
            "split": row["split"],
            "A0_PSNR": row["A0_PSNR"],
            "selected_seed_count": int(selected_vote[idx]),
            "selected_negative_seed_count": int(negative_vote[idx]),
            "selected_severe_seed_count": int(severe_vote[idx]),
            "current_mean_dPSNR": current_delta_mean[idx],
            "dPSNR_a0p25": alpha_delta(row, BASELINE_ALPHA),
            "best_alpha": b_alpha,
            "best_alpha_dPSNR": b_dpsnr,
            "best_alpha_dSSIM": b_dssim,
        }
        for field in feature_fields:
            rec[field] = row.get(field, "")
        selected_negative_rows.append(rec)
    selected_negative_rows.sort(key=lambda rec: (fnum(rec["current_mean_dPSNR"]), -int(rec["selected_negative_seed_count"])))

    write_csv(args.out_dir / "v21_c5_positive_deficit_report.csv", seed_rows, list(seed_rows[0].keys()))
    write_csv(args.out_dir / "v21_c5_c4_fold_replay.csv", fold_records, list(fold_records[0].keys()))
    hard_fields = list(best_alpha_rows[0].keys()) if best_alpha_rows else []
    write_csv(args.out_dir / "v21_c5_hard_bottom25_alpha_oracle.csv", best_alpha_rows, hard_fields)
    write_csv(args.out_dir / "v21_c5_false_positive_false_negative_bins.csv", bin_rows, list(bin_rows[0].keys()))
    write_csv(args.out_dir / "v21_c5_selected_negative_visual_proxy.csv", selected_negative_rows[:120], list(selected_negative_rows[0].keys()) if selected_negative_rows else [])

    aggregate = {
        "route": "Haze4K-v2.1 SEG-Mix",
        "phase": "C5 C4 Failure Forensic",
        "locked_test_touched": False,
        "policy_replayed": "C2d/C4 alpha=0.25 diff_signed_mean train-fold threshold",
        "input_alpha_rows": str(args.alpha_rows),
        "input_c4_summary": str(args.c4_summary),
        "available_alphas": alphas,
        "c4_decision": c4_summary.get("decision"),
        "c4_aggregate": c4_summary.get("aggregate"),
        "seed_rows": seed_rows,
        "hard_bottom25_safe_high_alpha_count": int(sum(1 for row in best_alpha_rows if row.get("safe_high_alpha_exists"))),
        "hard_bottom25_count": len(best_alpha_rows),
        "selected_negative_rows_written": min(120, len(selected_negative_rows)),
        "decision": "C5_FORENSIC_COMPLETE_NO_POLICY_TUNING_START_C6_C7",
    }
    (args.out_dir / "v21_c5_summary.json").write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    mean_positive_deficit = float(np.mean([fnum(row["positive_deficit_count"]) for row in seed_rows]))
    mean_hard_gap = float(np.mean([fnum(row["hard_target_gap"]) for row in seed_rows]))
    lines = [
        "# Haze4K v2.1 C5 C4 Failure Forensic",
        "",
        "Decision: `C5_FORENSIC_COMPLETE_NO_POLICY_TUNING_START_C6_C7`",
        "",
        "C5 only replays the sealed train-derived C2d/C4 family and decomposes the C4 gap. It does not select a new policy and does not touch locked test data.",
        "",
        "## Gap Summary",
        "",
        f"- Mean positive-count deficit to 0.70: `{mean_positive_deficit:.3f}` images per seeded replay.",
        f"- Mean hard-bottom25 gap to +0.30 dB: `{mean_hard_gap:.6f}` dB.",
        f"- Hard-bottom25 rows with at least one safe high-alpha candidate in existing alpha grid: `{aggregate['hard_bottom25_safe_high_alpha_count']}/{aggregate['hard_bottom25_count']}`.",
        f"- Selected-negative proxy rows written: `{aggregate['selected_negative_rows_written']}`.",
        "",
        "## Outputs",
        "",
        "- `v21_c5_positive_deficit_report.csv`",
        "- `v21_c5_false_positive_false_negative_bins.csv`",
        "- `v21_c5_hard_bottom25_alpha_oracle.csv`",
        "- `v21_c5_selected_negative_visual_proxy.csv`",
        "- `v21_c5_summary.json`",
        "",
        "## Interpretation Guardrail",
        "",
        "C5 may motivate C6/C7 experiment design, but its replay is not a tuning source for locked data. Locked and distillation remain blocked.",
    ]
    (args.out_dir / "v21_c5_c4_gap_decomposition.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"V21_C5_FORENSIC_OK decision={aggregate['decision']} rows={len(rows)} out={args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
