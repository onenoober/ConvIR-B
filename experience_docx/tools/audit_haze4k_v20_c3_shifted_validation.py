#!/usr/bin/env python3
"""C3 train-only shifted validation for the C2d alpha-shrink policy family.

The C2d OOF folds converged on a simple family: alpha=0.25 and a deployable
FullUDP-A0 output-difference threshold. C3 stress-tests that family by choosing
the threshold on all-but-one train-derived bin and replaying it on the held-out
bin. It uses C2d text metrics only and does not touch locked data.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from audit_haze4k_v20_c2b_multirule_router import (  # noqa: E402
    POLICY_FEATURES,
    QUANTILES,
    STRICT_GATE,
    Table,
    fnum,
    score,
    strict_gate_pass,
    summarize_mask,
    write_csv,
)


ALPHA = 0.25
FEATURE = "diff_signed_mean"
DIRECTION = "le"


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def alpha_key(alpha: float) -> str:
    return f"a{str(alpha).replace('.', 'p')}"


def alpha_rows(rows: list[dict[str, Any]], alpha: float) -> list[dict[str, Any]]:
    key = alpha_key(alpha)
    out: list[dict[str, Any]] = []
    for row in rows:
        clone = dict(row)
        clone["dPSNR"] = row[f"dPSNR_{key}"]
        clone["dSSIM"] = row[f"dSSIM_{key}"]
        clone["alpha"] = alpha
        out.append(clone)
    return out


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
    values = {
        "split": [str(row["split"]) for row in rows],
        "airlight_q4": rank_bins(airlight),
        "beta_haze_q4": rank_bins(beta),
        "depth_mean_q4": rank_bins([fnum(row["depth_mean"]) for row in rows]),
        "input_grad_lowtexture_q4": rank_bins([fnum(row["input_grad_mean"]) for row in rows]),
        "input_dark_q4": rank_bins([fnum(row["input_dark_mean"]) for row in rows]),
        "diff_abs_q4": rank_bins([fnum(row["diff_abs_mean"]) for row in rows]),
        "a0_psnr_stress_q4": rank_bins([fnum(row["A0_PSNR"]) for row in rows]),
    }
    return values


def candidate_thresholds(rows: list[dict[str, Any]]) -> list[float]:
    vals = np.array([fnum(row[FEATURE]) for row in rows], dtype=np.float64)
    return sorted({float(np.quantile(vals, q)) for q in QUANTILES})


def mask_for_threshold(rows: list[dict[str, Any]], threshold: float) -> np.ndarray:
    vals = np.array([fnum(row[FEATURE]) for row in rows], dtype=np.float64)
    if DIRECTION == "le":
        return vals <= threshold
    return vals >= threshold


def choose_threshold(train_rows: list[dict[str, Any]]) -> dict[str, Any]:
    ar = alpha_rows(train_rows, ALPHA)
    table = Table(ar, POLICY_FEATURES)
    candidates: list[dict[str, Any]] = []
    for threshold in candidate_thresholds(train_rows):
        rec = {
            "policy_id": f"alpha_{alpha_key(ALPHA)}__{FEATURE}_{DIRECTION}_{threshold:.8g}",
            "alpha": ALPHA,
            "feature": FEATURE,
            "direction": DIRECTION,
            "threshold": threshold,
        }
        rec.update(summarize_mask(table, mask_for_threshold(train_rows, threshold)))
        rec["strict_gate_pass"] = strict_gate_pass(rec)
        rec["score"] = score(rec)
        candidates.append(rec)
    candidates.sort(key=lambda row: (bool(row["strict_gate_pass"]), fnum(row["score"])), reverse=True)
    return candidates[0]


def summarize_selected(rows: list[dict[str, Any]], selected: np.ndarray) -> dict[str, Any]:
    ar = alpha_rows(rows, ALPHA)
    table = Table(ar, POLICY_FEATURES)
    rec = summarize_mask(table, selected)
    rec["strict_gate_pass"] = strict_gate_pass(rec)
    rec["score"] = score(rec)
    return rec


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha_rows", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_csv(args.alpha_rows)
    group_labels = build_group_labels(rows)
    bin_rows: list[dict[str, Any]] = []
    dim_rows: list[dict[str, Any]] = []

    for dimension, labels in group_labels.items():
        selected_all = np.zeros(len(rows), dtype=bool)
        unique_labels = sorted(set(labels))
        for label in unique_labels:
            holdout_idx = np.array([lab == label for lab in labels], dtype=bool)
            train_rows = [row for row, keep in zip(rows, ~holdout_idx, strict=False) if keep]
            holdout_rows = [row for row, keep in zip(rows, holdout_idx, strict=False) if keep]
            chosen = choose_threshold(train_rows)
            holdout_selected = mask_for_threshold(holdout_rows, float(chosen["threshold"]))
            selected_all[holdout_idx] = holdout_selected
            rec: dict[str, Any] = {
                "dimension": dimension,
                "heldout_bin": label,
                "heldout_count": len(holdout_rows),
                "train_count": len(train_rows),
                "train_policy_id": chosen["policy_id"],
                "train_threshold": chosen["threshold"],
                "train_strict_gate_pass": chosen["strict_gate_pass"],
            }
            rec.update(summarize_selected(holdout_rows, holdout_selected))
            rec["bin_safety_pass"] = (
                fnum(rec["mean_dPSNR"]) >= -0.02
                and fnum(rec["dSSIM"]) >= -0.001
                and fnum(rec["severe_loss_per_600"]) <= 96.0
            )
            bin_rows.append(rec)
        dim_rec: dict[str, Any] = {
            "dimension": dimension,
            "bin_count": len(unique_labels),
        }
        dim_rec.update(summarize_selected(rows, selected_all))
        dim_bins = [row for row in bin_rows if row["dimension"] == dimension]
        dim_rec["min_bin_mean_dPSNR"] = min(fnum(row["mean_dPSNR"]) for row in dim_bins)
        dim_rec["max_bin_severe_per_600"] = max(fnum(row["severe_loss_per_600"]) for row in dim_bins)
        dim_rec["bin_safety_pass_count"] = sum(1 for row in dim_bins if row["bin_safety_pass"])
        dim_rec["dimension_shift_pass"] = (
            bool(dim_rec["strict_gate_pass"])
            and fnum(dim_rec["min_bin_mean_dPSNR"]) >= -0.02
            and fnum(dim_rec["max_bin_severe_per_600"]) <= 96.0
            and int(dim_rec["bin_safety_pass_count"]) == len(unique_labels)
        )
        dim_rows.append(dim_rec)

    bin_fields = [
        "dimension",
        "heldout_bin",
        "heldout_count",
        "train_count",
        "train_policy_id",
        "train_threshold",
        "train_strict_gate_pass",
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
        "score",
        "bin_safety_pass",
    ]
    dim_fields = [
        "dimension",
        "bin_count",
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
        "score",
        "min_bin_mean_dPSNR",
        "max_bin_severe_per_600",
        "bin_safety_pass_count",
        "dimension_shift_pass",
    ]
    write_csv(args.out_dir / "v20_c3_shifted_bin_metrics.csv", bin_rows, bin_fields)
    write_csv(args.out_dir / "v20_c3_shifted_dimension_summary.csv", dim_rows, dim_fields)

    all_dimensions_pass = all(bool(row["dimension_shift_pass"]) for row in dim_rows)
    if all_dimensions_pass:
        decision = "C3_SHIFTED_VALIDATION_PASS_START_FORMAL_5X3"
    else:
        decision = "C3_SHIFTED_VALIDATION_FAIL_REASSESS_ROUTER_FEATURES"

    summary = {
        "route": "Haze4K-v2.0 StrongExpert-GainMix",
        "phase": "C3 Train-Only Shifted Validation",
        "locked_test_touched": False,
        "input_alpha_rows": str(args.alpha_rows),
        "policy_family": {
            "alpha": ALPHA,
            "feature": FEATURE,
            "direction": DIRECTION,
            "threshold_selection": "train-only quantile grid per held-out bin",
        },
        "strict_gate": STRICT_GATE,
        "dimension_rows": dim_rows,
        "bin_rows": bin_rows,
        "decision": decision,
    }
    (args.out_dir / "v20_c3_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Haze4K v2.0 C3 Train-Only Shifted Validation",
        "",
        f"Decision: `{decision}`",
        "",
        "C3 validates the C2d alpha-shrink policy family by holding out train-derived bins and selecting only the scalar output-diff threshold on the remaining bins.",
        "The locked Haze4K test remains untouched.",
        "",
        "## Dimension Summary",
        "",
        "| Dimension | Pass | Mean | Hard | Easy | dSSIM | Severe/600 | Min Bin Mean | Max Bin Severe/600 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in dim_rows:
        lines.append(
            f"| `{row['dimension']}` | `{row['dimension_shift_pass']}` | "
            f"`{fnum(row['mean_dPSNR']):.6f}` | `{fnum(row['hard_bottom25_dPSNR']):.6f}` | "
            f"`{fnum(row['easy_top25_dPSNR']):.6f}` | `{fnum(row['dSSIM']):.8f}` | "
            f"`{fnum(row['severe_loss_per_600']):.1f}` | `{fnum(row['min_bin_mean_dPSNR']):.6f}` | "
            f"`{fnum(row['max_bin_severe_per_600']):.1f}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Formal 5x3 replay is authorized only if every dimension passes.",
            "- Locked test remains blocked until formal 5x3 also passes and the final policy is sealed.",
        ]
    )
    (args.out_dir / "v20_c3_decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"V20_C3_SHIFTED_VALIDATION_OK decision={decision} dimensions={len(dim_rows)} out={args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
