#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


SEVERE = -0.20
STRONG_REG = -0.05


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def mean(values: list[float]) -> float:
    return float(statistics.mean(values)) if values else float("nan")


def median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else float("nan")


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * pct / 100.0
    lo = int(pos)
    hi = min(len(ordered) - 1, lo + 1)
    return float(ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo))


def cvar(values: list[float], pct: float = 5.0) -> float:
    if not values:
        return float("nan")
    k = max(1, int(round(len(values) * pct / 100.0)))
    return mean(sorted(values)[:k])


def summarize(rows: list[dict[str, Any]], checkpoint: str) -> dict[str, Any]:
    vals = [float(r["dPSNR"]) for r in rows]
    a0 = [float(r["a0_psnr"]) for r in rows]
    hard_cut = percentile(a0, 25)
    strong_cut = percentile(a0, 75)
    hard = [float(r["dPSNR"]) for r in rows if float(r["a0_psnr"]) <= hard_cut]
    easy = [float(r["dPSNR"]) for r in rows if float(r["a0_psnr"]) >= strong_cut]
    strong = [float(r["dPSNR"]) for r in rows if float(r["a0_psnr"]) >= strong_cut]
    return {
        "checkpoint": checkpoint,
        "count": len(rows),
        "mean_dPSNR": mean(vals),
        "median_dPSNR": median(vals),
        "p01_dPSNR": percentile(vals, 1),
        "p05_dPSNR": percentile(vals, 5),
        "CVaR5_dPSNR": cvar(vals, 5),
        "hard_bottom25_dPSNR": mean(hard),
        "easy_top25_dPSNR": mean(easy),
        "severe_count": sum(v <= SEVERE for v in vals),
        "severe_rate": sum(v <= SEVERE for v in vals) / len(vals) if vals else float("nan"),
        "positive_ratio": sum(v > 0 for v in vals) / len(vals) if vals else float("nan"),
        "strong_reference_count": len(strong),
        "strong_reference_regressions": sum(v <= STRONG_REG for v in strong),
        "mean_delta_final_l1_vs_a0": mean([float(r["delta_final_l1_vs_a0"]) for r in rows]),
        "mean_delta_lowband_l1_vs_a0": mean([float(r["delta_lowband_l1_vs_a0"]) for r in rows]),
        "mean_preserve_l1_to_a0": mean([float(r["preserve_l1_to_a0"]) for r in rows]),
        "mean_image_action_l1": mean([float(r["image_action_l1"]) for r in rows]),
        "budget_activation_rate": sum(float(r["budget_hinge"]) > 0 for r in rows) / len(rows) if rows else float("nan"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--r1-dir", type=Path, required=True)
    ap.add_argument("--r2-dir", type=Path, required=True)
    ap.add_argument("--v216-train-history", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    r2_closeout = json.loads((args.r2_dir / "v217_r2_closeout.json").read_text(encoding="utf-8"))
    loss_rows_raw = read_csv(args.r1_dir / "v217_r1_loss_delta_vs_identity.csv")
    loss_rows: list[dict[str, Any]] = []
    for row in loss_rows_raw:
        loss_rows.append({k: (float(v) if k not in {"checkpoint", "name"} and v != "" else v) for k, v in row.items()})

    checkpoints = sorted({str(r["checkpoint"]) for r in loss_rows})
    summary_rows = [summarize([r for r in loss_rows if str(r["checkpoint"]) == ckpt], ckpt) for ckpt in checkpoints]
    write_csv(args.out_dir / "v217_r3_objective_vs_tail_report.csv", summary_rows)
    write_csv(args.out_dir / "v217_r3_per_image_loss_terms.csv", loss_rows)

    cvar_rows = [
        {
            "checkpoint": row["checkpoint"],
            "mean_dPSNR": row["mean_dPSNR"],
            "p01_dPSNR": row["p01_dPSNR"],
            "p05_dPSNR": row["p05_dPSNR"],
            "CVaR5_dPSNR": row["CVaR5_dPSNR"],
            "severe_count": row["severe_count"],
            "severe_rate": row["severe_rate"],
            "strong_reference_regressions": row["strong_reference_regressions"],
        }
        for row in summary_rows
    ]
    write_csv(args.out_dir / "v217_r3_cvar_tail_metrics.csv", cvar_rows)

    train_history = json.loads(args.v216_train_history.read_text(encoding="utf-8")) if args.v216_train_history.is_file() else {}
    history_rows = train_history.get("history", [])
    budget_rows = [
        {
            "source": "train_history_epoch",
            "epoch": row.get("epoch", ""),
            "logged_budget": row.get("budget", ""),
            "budget_active": int(float(row.get("budget", 0.0)) > 0.0),
        }
        for row in history_rows
    ]
    for row in summary_rows:
        budget_rows.append(
            {
                "source": "val_checkpoint",
                "checkpoint": row["checkpoint"],
                "budget_activation_rate": row["budget_activation_rate"],
                "mean_image_action_l1": row["mean_image_action_l1"],
            }
        )
    write_csv(args.out_dir / "v217_r3_budget_activation_report.csv", budget_rows)

    preserve_rows = []
    for ckpt in checkpoints:
        rows = [r for r in loss_rows if str(r["checkpoint"]) == ckpt]
        a0 = [float(r["a0_psnr"]) for r in rows]
        hard_cut = percentile(a0, 25)
        strong_cut = percentile(a0, 75)
        groups = {
            "hard_bottom25": [r for r in rows if float(r["a0_psnr"]) <= hard_cut],
            "easy_top25": [r for r in rows if float(r["a0_psnr"]) >= strong_cut],
            "strong_reference": [r for r in rows if float(r["a0_psnr"]) >= strong_cut],
        }
        for group, subset in groups.items():
            preserve_rows.append(
                {
                    "checkpoint": ckpt,
                    "group": group,
                    "count": len(subset),
                    "mean_dPSNR": mean([float(r["dPSNR"]) for r in subset]),
                    "mean_preserve_l1_to_a0": mean([float(r["preserve_l1_to_a0"]) for r in subset]),
                    "regressions_le_-0p05": sum(float(r["dPSNR"]) <= STRONG_REG for r in subset),
                    "severe_le_-0p20": sum(float(r["dPSNR"]) <= SEVERE for r in subset),
                }
            )
    write_csv(args.out_dir / "v217_r3_preserve_mask_report.csv", preserve_rows)

    model5 = next((r for r in summary_rows if r["checkpoint"] == "model_5"), None)
    any_internal_pass = bool(r2_closeout.get("o1_pass") or r2_closeout.get("o2_pass") or r2_closeout.get("o3_pass"))
    if not any_internal_pass:
        decision = "R3_SKIPPED_R2_INTERNAL_FEATURE_ORACLE_FAILED"
    elif model5 and model5["mean_delta_final_l1_vs_a0"] < 0 and model5["severe_count"] > 12:
        decision = "R3_AVERAGE_OBJECTIVE_IMPROVES_BUT_TAIL_FAILS_REQUIRE_TAIL_AWARE_OBJECTIVE"
    elif model5 and model5["severe_count"] <= 12 and model5["p05_dPSNR"] >= -0.20:
        decision = "R3_TAIL_OBJECTIVE_AUDIT_PASS"
    else:
        decision = "R3_TAIL_OBJECTIVE_INCONCLUSIVE_REQUIRE_REVIEW"

    write_text(
        args.out_dir / "v217_r3_decision.md",
        "\n".join(
            [
                "# v2.17 R3 Tail-Objective Audit Decision",
                "",
                f"Decision: `{decision}`",
                "",
                f"- R2 decision: `{r2_closeout.get('decision')}`",
                f"- Any internal feature oracle pass: `{any_internal_pass}`",
                f"- model_5 mean delta final L1 vs A0: `{model5['mean_delta_final_l1_vs_a0'] if model5 else 'NA'}`",
                f"- model_5 p05 dPSNR: `{model5['p05_dPSNR'] if model5 else 'NA'}`",
                f"- model_5 severe count: `{model5['severe_count'] if model5 else 'NA'}`",
                "",
                "Interpretation:",
                "",
                "- Average reconstruction movement is not enough for the next trainable route.",
                "- Any WLDB-B training must include explicit p05/CVaR/severe preservation terms and an action budget that actually activates.",
                "- Locked Haze4K test remains untouched.",
            ]
        ),
    )
    write_json(
        args.out_dir / "v217_r3_closeout.json",
        {
            "decision": decision,
            "r2_decision": r2_closeout.get("decision"),
            "summary": summary_rows,
            "locked_test_touched": False,
            "training_launched": False,
        },
    )
    print("V217_R3_OBJECTIVE_TAIL_AUDIT_OK", decision, flush=True)


if __name__ == "__main__":
    main()
