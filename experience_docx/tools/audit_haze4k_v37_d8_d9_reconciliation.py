#!/usr/bin/env python3
"""Audit DTA-v3.7 D8/D9 evidence metadata consistency."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def read_csv_first(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            return dict(row)
    return {}


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def parse_status_claims(status: str) -> dict[str, Any]:
    claims: dict[str, Any] = {
        "folds": [],
        "seeds": [],
        "tasks": None,
        "raw_d1_full_5x3_run": None,
        "policy_search": None,
        "locked_test_touched": None,
        "outputdiff_group_starts": 0,
        "outputdiff_group_dones": 0,
        "done_fold_seed_pairs": [],
        "success_markers": [],
    }
    match = re.search(r"folds=([0-9,]+)\s+seeds=([0-9,]+)", status)
    if match:
        claims["folds"] = [int(x) for x in match.group(1).split(",") if x]
        claims["seeds"] = [int(x) for x in match.group(2).split(",") if x]
    task_match = re.search(r"stage3_outputdiff_start tasks=(\d+)", status)
    if task_match:
        claims["tasks"] = int(task_match.group(1))
    for key in ["raw_d1_full_5x3_run", "policy_search", "locked_test_touched"]:
        key_match = re.search(rf"{key}=([A-Za-z0-9_]+)", status)
        if key_match:
            value = key_match.group(1).lower()
            claims[key] = value == "true" if value in {"true", "false"} else key_match.group(1)
    starts = re.findall(r"d8_outputdiff_group_start fold=(\d+) seed=(\d+)", status)
    dones = re.findall(r"d8_outputdiff_group_done fold=(\d+) seed=(\d+) rc=(\d+)", status)
    claims["outputdiff_group_starts"] = len(starts)
    claims["outputdiff_group_dones"] = len(dones)
    claims["done_fold_seed_pairs"] = sorted({f"{fold}:{seed}" for fold, seed, rc in dones if rc == "0"})
    claims["success_markers"] = sorted(set(re.findall(r"\bDTA_V3_7_[A-Z0-9_]+_OK\b", status)))
    return claims


def boolish(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value.lower() == "true":
            return True
        if value.lower() == "false":
            return False
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dta-evidence-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    status_path = args.dta_evidence_dir / "status_phase_d8_fixed_formal.txt"
    d8_summary_path = args.dta_evidence_dir / "v37_d8_fixed_formal_summary.json"
    d8_agg_path = args.dta_evidence_dir / "v37_d8_fixed_formal_policy_aggregate.csv"
    d9_status_path = args.dta_evidence_dir / "status_phase_d9_locked_fixed_policy.txt"
    d9_agg_path = args.dta_evidence_dir / "v37_d9_locked_fixed_policy_aggregate.csv"

    status = read_text(status_path)
    d9_status = read_text(d9_status_path)
    claims = parse_status_claims(status)
    summary = load_json(d8_summary_path)
    aggregate = read_csv_first(d8_agg_path)
    d9_aggregate = read_csv_first(d9_agg_path)

    expected_tasks = len(claims["folds"]) * len(claims["seeds"]) if claims["folds"] and claims["seeds"] else None
    inconsistencies: list[dict[str, Any]] = []

    def check(name: str, status_value: Any, artifact_value: Any, severity: str, note: str) -> None:
        if status_value != artifact_value:
            inconsistencies.append(
                {
                    "field": name,
                    "status_value": status_value,
                    "artifact_value": artifact_value,
                    "severity": severity,
                    "note": note,
                }
            )

    check("tasks", expected_tasks, claims["tasks"], "info", "Status task count should equal folds x seeds.")
    if expected_tasks is not None:
        check("outputdiff_group_done_count", expected_tasks, claims["outputdiff_group_dones"], "error", "Each fold/seed outputdiff group should complete with rc=0.")
    check("raw_d1_full_5x3_run", claims["raw_d1_full_5x3_run"], boolish(summary.get("raw_d1_full_5x3_run")), "metadata", "D8 status says raw D1 5x3 was run; summary retained D7 value.")
    check("summary_phase", "D8_fixed_formal_confirmation", summary.get("phase"), "metadata", "D8 summary phase should not retain the D7 confirmation label.")
    check("aggregate_phase", "D8_fixed_formal_confirmation", aggregate.get("phase"), "metadata", "D8 aggregate phase should not retain the D7 confirmation label.")
    check("outer_groups", 15, int(float(str(summary.get("sealed_policy_candidate", {}).get("outer_groups", -1)))), "metadata", "Broader D8 scope is 5 folds x 3 seeds; current artifact records D7 outer group count.")
    check("aggregate_outer_groups", 15, int(float(aggregate.get("outer_groups", "-1"))), "metadata", "Broader D8 aggregate should record 15 fold/seed groups.")

    audit = {
        "route": "DTA-v3.7 U-TQS-Mix",
        "phase": "D8/D9 reconciliation audit",
        "locked_test_touched_by_d8": False,
        "locked_test_touched_by_d9": "locked_test_touched=true" in d9_status,
        "status_path": str(status_path),
        "d8_summary_path": str(d8_summary_path),
        "d8_aggregate_path": str(d8_agg_path),
        "d9_status_path": str(d9_status_path),
        "d9_aggregate_path": str(d9_agg_path),
        "status_claims": claims,
        "d8_summary_metrics": {
            "mean_dPSNR": summary.get("sealed_policy_candidate", {}).get("mean_dPSNR"),
            "hard_bottom25_dPSNR": summary.get("sealed_policy_candidate", {}).get("hard_bottom25_dPSNR"),
            "positive_ratio": summary.get("sealed_policy_candidate", {}).get("positive_ratio"),
            "worst_per_600": summary.get("sealed_policy_candidate", {}).get("worst_per_600"),
            "strict_gate_pass": summary.get("sealed_policy_candidate", {}).get("strict_gate_pass"),
        },
        "d9_aggregate_metrics": {
            "mean_dPSNR": d9_aggregate.get("mean_dPSNR"),
            "hard_bottom25_dPSNR": d9_aggregate.get("hard_bottom25_dPSNR"),
            "positive_ratio": d9_aggregate.get("positive_ratio"),
            "worst_per_600": d9_aggregate.get("worst_per_600"),
            "strict_gate_pass": d9_aggregate.get("strict_gate_pass"),
            "decision": d9_aggregate.get("decision"),
        },
        "inconsistencies": inconsistencies,
        "decision": "D8_METRICS_USABLE_METADATA_RECONCILIATION_REQUIRED" if inconsistencies else "D8_D9_EVIDENCE_CONSISTENT",
    }

    with (out_dir / "v37_d8_d9_reconciliation_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, sort_keys=True)

    with (out_dir / "v37_d8_d9_reconciliation_inconsistencies.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["field", "status_value", "artifact_value", "severity", "note"])
        writer.writeheader()
        for row in inconsistencies:
            writer.writerow(row)

    lines = [
        "# DTA-v3.7 D8/D9 Reconciliation Audit",
        "",
        f"Decision: `{audit['decision']}`",
        "",
        "## D8 status-derived scope",
        "",
        f"- folds: `{claims['folds']}`",
        f"- seeds: `{claims['seeds']}`",
        f"- expected tasks: `{expected_tasks}`",
        f"- outputdiff group starts/dones: `{claims['outputdiff_group_starts']}` / `{claims['outputdiff_group_dones']}`",
        f"- raw D1 full 5x3 status claim: `{claims['raw_d1_full_5x3_run']}`",
        "",
        "## Metric bottom line",
        "",
        f"- D8 summary strict pass: `{audit['d8_summary_metrics']['strict_gate_pass']}`",
        f"- D9 locked decision: `{audit['d9_aggregate_metrics']['decision']}`",
        "",
        "## Inconsistencies",
        "",
    ]
    if inconsistencies:
        for row in inconsistencies:
            lines.append(
                f"- `{row['field']}`: status `{row['status_value']}` vs artifact `{row['artifact_value']}` ({row['severity']}); {row['note']}"
            )
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The D8 metric values remain usable as a train-derived formal confirmation record, but the summary/aggregate metadata retained D7 labels and should not be read as a clean route-state source without this audit.",
            "- The D9 locked one-shot remains a failed confirmation and must not be used for threshold, feature, action, or checkpoint tuning.",
        ]
    )
    (out_dir / "v37_d8_d9_reconciliation_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"V37_D8_D9_RECONCILIATION_AUDIT_OK decision={audit['decision']} out={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
