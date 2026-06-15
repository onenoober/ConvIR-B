#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from c13_common import summarize_rows, write_csv, write_json


SCREEN_GATE = {
    "mean_dPSNR": 0.50,
    "hard_bottom25_dPSNR": 0.70,
    "easy_top25_dPSNR": 0.30,
    "positive_ratio": 0.80,
    "severe_loss_per_600": 72.0,
    "dSSIM": 0.0,
}

GROUP_GATE = {
    "mean_dPSNR": 0.0,
    "positive_ratio": 0.65,
    "severe_loss_per_600": 96.0,
}


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def pass_screen(summary: dict[str, Any]) -> bool:
    return (
        summary["mean_dPSNR"] >= SCREEN_GATE["mean_dPSNR"]
        and summary["hard_bottom25_dPSNR"] >= SCREEN_GATE["hard_bottom25_dPSNR"]
        and summary["easy_top25_dPSNR"] >= SCREEN_GATE["easy_top25_dPSNR"]
        and summary["positive_ratio"] >= SCREEN_GATE["positive_ratio"]
        and summary["severe_loss_per_600"] <= SCREEN_GATE["severe_loss_per_600"]
        and summary["dSSIM"] >= SCREEN_GATE["dSSIM"]
    )


def pass_group(summary: dict[str, Any]) -> bool:
    return (
        summary["mean_dPSNR"] >= GROUP_GATE["mean_dPSNR"]
        and summary["positive_ratio"] >= GROUP_GATE["positive_ratio"]
        and summary["severe_loss_per_600"] <= GROUP_GATE["severe_loss_per_600"]
    )


def quantile_bucket(rows: list[dict[str, Any]], key: str, bucket_count: int = 4) -> list[tuple[str, list[dict[str, Any]]]]:
    rows = [row for row in rows if row.get(key, "") not in ("", None)]
    if not rows:
        return []
    ordered = sorted(rows, key=lambda row: float(row[key]))
    out = []
    n = len(ordered)
    for idx in range(bucket_count):
        start = round(n * idx / bucket_count)
        end = round(n * (idx + 1) / bucket_count)
        part = ordered[start:end]
        if part:
            out.append((f"{key}_q{idx + 1}", part))
    return out


def group_rows(per_image_rows: list[dict[str, Any]], teacher_rows: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    merged = []
    for row in per_image_rows:
        rec = dict(row)
        teacher = teacher_rows.get(row["name"], {})
        rec["teacher_dPSNR"] = teacher.get("WD0375_dPSNR", "")
        rec["teacher_margin"] = teacher.get("WD0375_dPSNR", "")
        rec["teacher_residual_abs_proxy"] = teacher.get("WD0375_dPSNR", "")
        merged.append(rec)
    group_specs = []
    for key in ("A0_PSNR", "teacher_margin", "teacher_residual_abs_proxy"):
        group_specs.extend(quantile_bucket(merged, key, 4))
    group_specs.extend(
        [
            ("split_train_core", [row for row in merged if row.get("split") == "train_core"]),
            ("split_val", [row for row in merged if row.get("split") == "val"]),
        ]
    )
    out = []
    for label, rows in group_specs:
        if not rows:
            continue
        summary = summarize_rows(rows)
        summary.update({"group": label, "group_pass": pass_group(summary)})
        out.append(summary)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--teacher-metrics", type=Path, required=True)
    args = ap.parse_args()

    teacher_rows = {}
    if args.teacher_metrics.is_file():
        for row in read_csv(args.teacher_metrics):
            teacher_rows[row["name"]] = row

    leaderboard = []
    group_table = []
    for summary_path in sorted(args.out_dir.glob("v25_c13_eval_*_summary.json")):
        summary = read_json(summary_path)
        variant = summary["variant"]
        checkpoint = summary["checkpoint"]
        prefix = f"v25_c13_eval_{variant}_{checkpoint}"
        per_image_path = args.out_dir / f"{prefix}_per_image.csv"
        if not per_image_path.is_file():
            continue
        row = dict(summary)
        row["summary_json"] = summary_path.name
        row["per_image_csv"] = per_image_path.name
        row["screen_gate_pass"] = pass_screen(summary)
        leaderboard.append(row)
        for group in group_rows(read_csv(per_image_path), teacher_rows):
            group.update({"variant": variant, "checkpoint": checkpoint})
            group_table.append(group)

    leaderboard.sort(
        key=lambda row: (
            bool(row.get("screen_gate_pass")),
            float(row.get("mean_dPSNR", -999)),
            -float(row.get("severe_loss_per_600", 999)),
        ),
        reverse=True,
    )
    top = leaderboard[0] if leaderboard else None
    decision = "C13_SCREEN_PASS_GROUP_REVIEW" if top and top.get("screen_gate_pass") else "C13_SCREEN_FAIL_STOP_OR_REDESIGN"
    if top and top.get("screen_gate_pass") and group_table:
        top_groups = [row for row in group_table if row["variant"] == top["variant"] and row["checkpoint"] == top["checkpoint"]]
        if not all(row["group_pass"] for row in top_groups):
            decision = "C13_SCREEN_PASS_GROUP_MIN_FAIL_REVIEW"
    write_csv(args.out_dir / "v25_c13_screen_leaderboard.csv", leaderboard)
    write_csv(args.out_dir / "v25_c13_group_min_table.csv", group_table)
    write_json(
        args.out_dir / "v25_c13_screen_decision.json",
        {
            "route": "Haze4K v2.5 C13 A0-frozen residual distillation",
            "locked_test_touched": False,
            "locked_per_image_read": False,
            "screen_gate": SCREEN_GATE,
            "group_gate": GROUP_GATE,
            "decision": decision,
            "top": top,
            "candidate_count": len(leaderboard),
        },
    )
    (args.out_dir / "v25_c13_decision.md").write_text(
        "# Haze4K v2.5 C13 Decision\n\n"
        f"Decision: `{decision}`\n\n"
        f"Candidate count: `{len(leaderboard)}`\n\n"
        f"Top: `{json.dumps(top, sort_keys=True) if top else None}`\n",
        encoding="utf-8",
    )
    print("C13_SCREEN_SUMMARY_OK", json.dumps({"decision": decision, "candidate_count": len(leaderboard)}, sort_keys=True))


if __name__ == "__main__":
    main()
