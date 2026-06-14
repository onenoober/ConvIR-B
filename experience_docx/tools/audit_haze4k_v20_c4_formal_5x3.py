#!/usr/bin/env python3
"""C4 formal 5x3 train-derived replay for the C2d alpha-shrink policy family."""

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

from audit_haze4k_v20_c2b_multirule_router import POLICY_FEATURES, Table, fnum, score, strict_gate_pass, summarize_mask, write_csv
from audit_haze4k_v20_c3_shifted_validation import ALPHA, FEATURE, alpha_key, alpha_rows, choose_threshold, mask_for_threshold


SEEDS = [3407, 3411, 2026]

SCREEN_GATE = {
    "mean_dPSNR": 0.12,
    "hard_bottom25_dPSNR": 0.20,
    "easy_top25_dPSNR": -0.02,
    "dSSIM": 0.0,
    "positive_ratio": 0.65,
    "severe_loss_per_600": 48.0,
}

STRONG_FORMAL_GATE = {
    "mean_dPSNR": 0.20,
    "hard_bottom25_dPSNR": 0.30,
    "easy_top25_dPSNR": 0.0,
    "dSSIM": 0.0,
    "positive_ratio": 0.70,
    "severe_loss_per_600": 48.0,
    "max_seed_severe_loss_per_600": 60.0,
}


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def seeded_fold_id(image_id: str, seed: int, folds: int = 5) -> int:
    digest = hashlib.sha1(f"{seed}:{image_id}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % folds


def screen_gate_pass(row: dict[str, Any]) -> bool:
    return (
        fnum(row.get("mean_dPSNR")) >= SCREEN_GATE["mean_dPSNR"]
        and fnum(row.get("hard_bottom25_dPSNR")) >= SCREEN_GATE["hard_bottom25_dPSNR"]
        and fnum(row.get("easy_top25_dPSNR")) >= SCREEN_GATE["easy_top25_dPSNR"]
        and fnum(row.get("dSSIM")) >= SCREEN_GATE["dSSIM"]
        and fnum(row.get("positive_ratio")) >= SCREEN_GATE["positive_ratio"]
        and fnum(row.get("severe_loss_per_600")) <= SCREEN_GATE["severe_loss_per_600"]
    )


def strong_gate_pass(row: dict[str, Any], max_seed_severe: float) -> bool:
    return (
        fnum(row.get("mean_dPSNR")) >= STRONG_FORMAL_GATE["mean_dPSNR"]
        and fnum(row.get("hard_bottom25_dPSNR")) >= STRONG_FORMAL_GATE["hard_bottom25_dPSNR"]
        and fnum(row.get("easy_top25_dPSNR")) >= STRONG_FORMAL_GATE["easy_top25_dPSNR"]
        and fnum(row.get("dSSIM")) >= STRONG_FORMAL_GATE["dSSIM"]
        and fnum(row.get("positive_ratio")) >= STRONG_FORMAL_GATE["positive_ratio"]
        and fnum(row.get("severe_loss_per_600")) <= STRONG_FORMAL_GATE["severe_loss_per_600"]
        and max_seed_severe <= STRONG_FORMAL_GATE["max_seed_severe_loss_per_600"]
    )


def summarize_selected(rows: list[dict[str, Any]], selected: np.ndarray) -> dict[str, Any]:
    table = Table(alpha_rows(rows, ALPHA), POLICY_FEATURES)
    rec = summarize_mask(table, selected)
    rec["screen_gate_pass"] = screen_gate_pass(rec)
    rec["strict_gate_pass"] = strict_gate_pass(rec)
    rec["score"] = score(rec)
    return rec


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.pstdev(values)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha_rows", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_csv(args.alpha_rows)
    fold_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []

    for seed in SEEDS:
        selected_seed = np.zeros(len(rows), dtype=bool)
        fold_ids = np.array([seeded_fold_id(str(row["name"]), seed) for row in rows], dtype=np.int64)
        for fold in range(5):
            heldout_mask = fold_ids == fold
            train_rows = [row for row, keep in zip(rows, ~heldout_mask, strict=False) if keep]
            heldout_rows = [row for row, keep in zip(rows, heldout_mask, strict=False) if keep]
            chosen = choose_threshold(train_rows)
            heldout_selected = mask_for_threshold(heldout_rows, float(chosen["threshold"]))
            selected_seed[heldout_mask] = heldout_selected
            rec: dict[str, Any] = {
                "seed": seed,
                "fold": fold,
                "train_policy_id": chosen["policy_id"],
                "train_threshold": chosen["threshold"],
                "train_strict_gate_pass": chosen["strict_gate_pass"],
                "train_count": len(train_rows),
                "heldout_count": len(heldout_rows),
            }
            rec.update(summarize_selected(heldout_rows, heldout_selected))
            fold_rows.append(rec)
        seed_rec: dict[str, Any] = {"seed": seed}
        seed_rec.update(summarize_selected(rows, selected_seed))
        seed_rows.append(seed_rec)

    metrics = ["mean_dPSNR", "hard_bottom25_dPSNR", "easy_top25_dPSNR", "dSSIM", "positive_ratio", "nonnegative_ratio", "severe_loss_per_600", "selected_precision"]
    aggregate: dict[str, Any] = {"seed_count": len(seed_rows), "fold_count": len(fold_rows)}
    for metric in metrics:
        vals = [fnum(row[metric]) for row in seed_rows]
        mean_val, std_val = mean_std(vals)
        aggregate[f"{metric}_mean"] = mean_val
        aggregate[f"{metric}_std"] = std_val
    aggregate["max_seed_severe_loss_per_600"] = max(fnum(row["severe_loss_per_600"]) for row in seed_rows)
    aggregate["min_seed_hard_bottom25_dPSNR"] = min(fnum(row["hard_bottom25_dPSNR"]) for row in seed_rows)
    aggregate["min_seed_easy_top25_dPSNR"] = min(fnum(row["easy_top25_dPSNR"]) for row in seed_rows)
    aggregate["screen_gate_all_seeds_pass"] = all(bool(row["screen_gate_pass"]) for row in seed_rows)
    aggregate["strong_formal_gate_pass"] = strong_gate_pass(
        {
            "mean_dPSNR": aggregate["mean_dPSNR_mean"],
            "hard_bottom25_dPSNR": aggregate["hard_bottom25_dPSNR_mean"],
            "easy_top25_dPSNR": aggregate["easy_top25_dPSNR_mean"],
            "dSSIM": aggregate["dSSIM_mean"],
            "positive_ratio": aggregate["positive_ratio_mean"],
            "severe_loss_per_600": aggregate["severe_loss_per_600_mean"],
        },
        fnum(aggregate["max_seed_severe_loss_per_600"]),
    )

    if aggregate["strong_formal_gate_pass"] and aggregate["screen_gate_all_seeds_pass"]:
        decision = "C4_FORMAL_5X3_STRONG_PASS_AUTHORIZE_LOCKED_ONE_SHOT"
    elif aggregate["screen_gate_all_seeds_pass"]:
        decision = "C4_FORMAL_5X3_SCREEN_PASS_STRONG_TARGET_FAIL_NO_LOCKED"
    else:
        decision = "C4_FORMAL_5X3_FAIL_NO_LOCKED"

    fold_fields = [
        "seed",
        "fold",
        "train_policy_id",
        "train_threshold",
        "train_strict_gate_pass",
        "train_count",
        "heldout_count",
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
        "screen_gate_pass",
        "strict_gate_pass",
        "score",
    ]
    seed_fields = ["seed"] + fold_fields[7:]
    write_csv(args.out_dir / "v20_c4_formal_5x3_fold_metrics.csv", fold_rows, fold_fields)
    write_csv(args.out_dir / "v20_c4_formal_5x3_seed_summary.csv", seed_rows, seed_fields)

    summary = {
        "route": "Haze4K-v2.0 StrongExpert-GainMix",
        "phase": "C4 Formal 5x3 Train-Derived Replay",
        "locked_test_touched": False,
        "input_alpha_rows": str(args.alpha_rows),
        "policy_family": {
            "alpha": ALPHA,
            "feature": FEATURE,
            "direction": "le",
            "threshold_selection": "train-fold quantile grid",
        },
        "seeds": SEEDS,
        "screen_gate": SCREEN_GATE,
        "strong_formal_gate": STRONG_FORMAL_GATE,
        "fold_rows": fold_rows,
        "seed_rows": seed_rows,
        "aggregate": aggregate,
        "decision": decision,
    }
    (args.out_dir / "v20_c4_formal_5x3_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Haze4K v2.0 C4 Formal 5x3 Train-Derived Replay",
        "",
        f"Decision: `{decision}`",
        "",
        "C4 replays the C2d/C3 policy family over 5 folds x 3 seeded fold assignments. Locked test data was not touched.",
        "",
        "## Aggregate",
        "",
    ]
    for key, value in aggregate.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Seed Summary", ""])
    for row in seed_rows:
        lines.append(
            f"- seed `{row['seed']}`: mean `{row['mean_dPSNR']}`, hard `{row['hard_bottom25_dPSNR']}`, "
            f"easy `{row['easy_top25_dPSNR']}`, positive `{row['positive_ratio']}`, severe `{row['severe_loss_per_600']}`, "
            f"screen `{row['screen_gate_pass']}`"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Locked one-shot is authorized only by the strong formal gate.",
            "- If only the screen gate passes, continue train-derived router/expert work and do not touch locked test.",
        ]
    )
    (args.out_dir / "v20_c4_formal_5x3_decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"V20_C4_FORMAL_5X3_OK decision={decision} seeds={len(SEEDS)} out={args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
