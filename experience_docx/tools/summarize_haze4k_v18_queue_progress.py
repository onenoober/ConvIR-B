#!/usr/bin/env python3
"""Summarize the Haze4K v1.8 execution queue state.

This tool is read-only with respect to model runtime artifacts. It converts
status markers, per-seed selection JSON, compare JSON presence, and repair-log
state into machine-readable progress evidence while the queue is running and
after post-queue repair has finished.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CHECKPOINT_LABELS = ["model_5", "model_10", "model_15", "model_20", "Best", "Final"]
SPLITS = ["val_regular", "val_hard"]


def read_text(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def iso_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def parse_status(status_text: str) -> dict[str, Any]:
    train_started: set[str] = set()
    train_done: dict[str, int] = {}
    eval_done: dict[str, list[int]] = {}
    selection_done: dict[str, int] = {}
    step_done: dict[str, int] = {}
    raw_state = "UNKNOWN"

    for line in status_text.splitlines():
        if line.startswith("state="):
            raw_state = line.split("=", 1)[1].strip()

        match = re.search(r"seed_train_start seed=(\d+)", line)
        if match:
            train_started.add(match.group(1))

        match = re.search(r"seed_train_done seed=(\d+) rc=(-?\d+)", line)
        if match:
            train_done[match.group(1)] = int(match.group(2))

        match = re.search(r"step_done name=seed_(\d+)_.+?_eval rc=(-?\d+)", line)
        if match:
            eval_done.setdefault(match.group(1), []).append(int(match.group(2)))

        match = re.search(r"repair_step_done name=repair_seed_(\d+)_.+?_eval rc=(-?\d+)", line)
        if match:
            eval_done.setdefault(match.group(1), []).append(int(match.group(2)))

        match = re.search(
            r"(?:step_done|repair_step_done) name=(?:repair_)?seed_(\d+)_multimetric_checkpoint_selection rc=(-?\d+)",
            line,
        )
        if match:
            selection_done[match.group(1)] = int(match.group(2))

        match = re.search(r"step_done name=([A-Za-z0-9_]+) rc=(-?\d+)", line)
        if match:
            step_done[match.group(1)] = int(match.group(2))

    return {
        "raw_state": raw_state,
        "train_started": sorted(train_started, key=int),
        "train_done": train_done,
        "eval_done": eval_done,
        "selection_done": selection_done,
        "step_done": step_done,
    }


def expected_compare_path(eval_dir: Path, seed: str, label: str, split: str) -> Path:
    label_lower = label.lower()
    return eval_dir / f"scout_eval_compare_v18_seed{seed}_{label_lower}_{split}_vs_a0.json"


def parse_latest_train_log(seed_dir: Path) -> dict[str, Any]:
    train_logs = sorted(seed_dir.glob("train_*.log"), key=lambda path: path.stat().st_mtime)
    if not train_logs:
        return {
            "train_log": None,
            "latest_train_epoch": None,
            "latest_train_iter": None,
            "latest_val_psnr": None,
            "latest_dpga_dpfm1_scale": None,
            "latest_dpga_dpfm1_eff": None,
        }

    text = read_text(train_logs[-1])
    latest_epoch = None
    latest_iter = None
    latest_psnr = None
    latest_dpfm1_scale = None
    latest_dpfm1_eff = None

    for match in re.finditer(r"Epoch:\s*(\d+)\s+Iter:\s*(\d+)/", text):
        latest_epoch = int(match.group(1))
        latest_iter = int(match.group(2))

    for match in re.finditer(r"(\d+)\s+epoch\s*\n\s*Average PSNR\s+([0-9.]+)\s+dB", text):
        latest_epoch = int(match.group(1))
        latest_psnr = float(match.group(2))

    for match in re.finditer(
        r"DPGA_STATS Epoch:\s*(\d+).*?DPGA_dpfm1_scale=([-+0-9.eE]+)\s+DPGA_dpfm1_eff=([-+0-9.eE]+)",
        text,
    ):
        latest_epoch = int(match.group(1))
        latest_dpfm1_scale = float(match.group(2))
        latest_dpfm1_eff = float(match.group(3))

    return {
        "train_log": str(train_logs[-1]),
        "latest_train_epoch": latest_epoch,
        "latest_train_iter": latest_iter,
        "latest_val_psnr": latest_psnr,
        "latest_dpga_dpfm1_scale": latest_dpfm1_scale,
        "latest_dpga_dpfm1_eff": latest_dpfm1_eff,
    }


def summarize_seed(evid: Path, seed: str, parsed_status: dict[str, Any]) -> dict[str, Any]:
    seed_dir = evid / f"seed_{seed}"
    eval_dir = seed_dir / "eval_regular_hard"
    selection_json = seed_dir / f"v18_seed{seed}_multimetric_checkpoint_selection.json"
    selection = read_json(selection_json)
    train_log_state = parse_latest_train_log(seed_dir)

    compare_present = 0
    compare_missing: list[str] = []
    for label in CHECKPOINT_LABELS:
        for split in SPLITS:
            path = expected_compare_path(eval_dir, seed, label, split)
            if path.is_file():
                compare_present += 1
            else:
                compare_missing.append(f"{label}:{split}")

    train_rc = parsed_status["train_done"].get(seed)
    selection_rc = parsed_status["selection_done"].get(seed)
    eval_rcs = parsed_status["eval_done"].get(seed, [])
    eval_rc0_count = sum(1 for rc in eval_rcs if rc == 0)
    eval_rc_nonzero_count = sum(1 for rc in eval_rcs if rc != 0)
    has_selection = selection is not None
    raw_selected_label = selection.get("selected_checkpoint_label") if selection else None
    raw_decision = selection.get("decision") if selection else None
    selection_has_missing = False
    if selection:
        selection_has_missing = any(
            row.get("decision") == "MISSING_COMPARE_JSON" for row in selection.get("rows", [])
        )
    expected_count = len(CHECKPOINT_LABELS) * len(SPLITS)
    selection_valid_for_decision = bool(
        has_selection and compare_present == expected_count and not selection_has_missing
    )
    selected_label = raw_selected_label if selection_valid_for_decision else None
    if selection_valid_for_decision:
        decision = raw_decision
    elif eval_rc_nonzero_count > 0 and compare_present < expected_count:
        decision = "ENGINEERING_REPAIR_PENDING_NOT_SCIENTIFIC_RESULT"
    elif has_selection:
        decision = "PENDING_REPAIR_NOT_SCIENTIFIC_RESULT"
    else:
        decision = None

    if train_rc is None and seed in parsed_status["train_started"]:
        seed_state = "TRAIN_RUNNING_OR_INTERRUPTED"
    elif train_rc is None:
        seed_state = "PENDING"
    elif train_rc != 0:
        seed_state = "TRAIN_FAILED_ENGINEERING"
    elif eval_rc_nonzero_count > 0 and compare_present < len(CHECKPOINT_LABELS) * len(SPLITS):
        seed_state = "EVAL_FAILED_ENGINEERING_REPAIR_PENDING"
    elif compare_present < len(CHECKPOINT_LABELS) * len(SPLITS):
        seed_state = "EVAL_INCOMPLETE_OR_REPAIR_PENDING"
    elif not has_selection:
        seed_state = "SELECTION_PENDING"
    elif selection_has_missing:
        seed_state = "SELECTION_NEEDS_REPAIR"
    else:
        seed_state = "SEED_EVIDENCE_COMPLETE"

    return {
        "seed": seed,
        "state": seed_state,
        "train_started": seed in parsed_status["train_started"],
        "train_rc": train_rc,
        "eval_done_count_from_status": len(eval_rcs),
        "eval_rc0_count_from_status": eval_rc0_count,
        "eval_rc_nonzero_count_from_status": eval_rc_nonzero_count,
        "eval_engineering_failure_pending_repair": (
            eval_rc_nonzero_count > 0 and compare_present < expected_count
        ),
        "compare_json_present_count": compare_present,
        "compare_json_expected_count": expected_count,
        "compare_missing": compare_missing,
        "selection_json": str(selection_json),
        "selection_present": has_selection,
        "selection_rc": selection_rc,
        "selection_valid_for_decision": selection_valid_for_decision,
        "selected_checkpoint_label": selected_label,
        "selection_decision": decision,
        "raw_selected_checkpoint_label": raw_selected_label,
        "raw_selection_decision": raw_decision,
        "selection_has_missing_compare_json": selection_has_missing,
        **train_log_state,
    }


def repair_state(evid: Path, status_text: str) -> dict[str, Any]:
    repair_log = evid / "v18_eval_repair" / "repair_v18_missing_eval_and_aggregate.log"
    text = read_text(repair_log)
    combined_text = "\n".join(part for part in (text, status_text) if part)
    if not combined_text:
        return {"state": "NOT_STARTED", "log": str(repair_log)}
    lines = [line for line in combined_text.splitlines() if line.strip()]
    if "repair_done name=v18_missing_eval_and_aggregate" in combined_text or "V18_EVAL_REPAIR_OK" in combined_text:
        state = "DONE"
    elif "main_queue_inactive_start_repair" in combined_text:
        state = "RUNNING_REPAIR"
    elif "main_queue_still_active" in combined_text:
        state = "WAITING_FOR_MAIN_QUEUE"
    else:
        state = "STARTED"
    return {
        "state": state,
        "log": str(repair_log),
        "last_lines": lines[-10:],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence_root", required=True)
    parser.add_argument("--seeds", nargs="+", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()

    evid = Path(args.evidence_root)
    status_text = read_text(evid / "status.txt")
    parsed = parse_status(status_text)
    seed_rows = [summarize_seed(evid, str(seed), parsed) for seed in args.seeds]
    repair = repair_state(evid, status_text)

    step_done = parsed["step_done"]
    completed_states = {"SEED_EVIDENCE_COMPLETE"}
    all_seed_evidence_complete = all(row["state"] in completed_states for row in seed_rows)
    derived_status_state = (
        "COMPLETED_QUEUE_AND_REPAIR_DONE"
        if all_seed_evidence_complete and repair["state"] == "DONE"
        else parsed["raw_state"]
    )
    payload = {
        "stage": "v1.8 in-flight queue progress summary",
        "generated_at": iso_now(),
        "locked_test_touched": False,
        "evidence_root": str(evid),
        "status_state": parsed["raw_state"],
        "derived_status_state": derived_status_state,
        "queue_steps": {
            "q1_router_policy_fixed_rc": step_done.get("v18_router_policy_fixed_table_analysis"),
            "q2_domain_data_preflight_rc": step_done.get("v18_domain_data_preflight"),
            "q5_domain_adaptation_rc": step_done.get("v18_domain_adaptation_q5"),
        },
        "repair": repair,
        "seed_count_expected": len(args.seeds),
        "seed_count_started": sum(1 for row in seed_rows if row["train_started"]),
        "seed_count_train_done": sum(1 for row in seed_rows if row["train_rc"] == 0),
        "seed_count_evidence_complete": sum(1 for row in seed_rows if row["state"] in completed_states),
        "seed_count_eval_engineering_failures": sum(
            1 for row in seed_rows if row["state"] == "EVAL_FAILED_ENGINEERING_REPAIR_PENDING"
        ),
        "seed_count_eval_incomplete": sum(
            1 for row in seed_rows if row["state"] == "EVAL_INCOMPLETE_OR_REPAIR_PENDING"
        ),
        "seed_rows": seed_rows,
        "queue_progress_state": (
            "COMPLETED_QUEUE_AND_REPAIR_DONE"
            if all_seed_evidence_complete and repair["state"] == "DONE"
            else "ALL_SEED_EVIDENCE_COMPLETE"
            if all_seed_evidence_complete
            else "QUEUE_OR_REPAIR_IN_PROGRESS"
        ),
    }

    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    fields = [
        "seed",
        "state",
        "train_started",
        "train_rc",
        "eval_done_count_from_status",
        "eval_rc0_count_from_status",
        "eval_rc_nonzero_count_from_status",
        "eval_engineering_failure_pending_repair",
        "compare_json_present_count",
        "compare_json_expected_count",
        "selection_present",
        "selection_rc",
        "selection_valid_for_decision",
        "selected_checkpoint_label",
        "selection_decision",
        "raw_selected_checkpoint_label",
        "raw_selection_decision",
        "selection_has_missing_compare_json",
        "latest_train_epoch",
        "latest_train_iter",
        "latest_val_psnr",
        "latest_dpga_dpfm1_scale",
        "latest_dpga_dpfm1_eff",
        "compare_missing",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in seed_rows:
            out = dict(row)
            out["compare_missing"] = ";".join(row["compare_missing"])
            writer.writerow({field: out.get(field, "") for field in fields})

    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
