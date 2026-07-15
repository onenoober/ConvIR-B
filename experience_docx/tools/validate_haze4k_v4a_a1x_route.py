#!/usr/bin/env python3
"""Machine validator for the frozen A1X R2 local setup package."""

import argparse
import ast
import hashlib
import json
from pathlib import Path


ROUTE_ID = "haze4k_v5_chd_rm_v4a_a1x_exact_half_deployable_accessibility_20260715"
SOURCE_COMMIT = "3b4da35440c8c26a7d1bcaf1daf342e11d9a3898"
ROUTE_CARD_SHA256 = "c723e59a3cb06de63b1ae4a72eabcd64ca0a3d08ba1346f3a53e0a52cce452da"


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    card = repo / "experience_docx/experiment_cards/2026-07-15-haze4k-v5-v4a-a1x-exact-half-deployable-accessibility.md"
    handoff = repo / "experience_docx/experiment_logs" / ROUTE_ID / "v4a_a1x_r3_design_handoff.json"
    head = repo / "Dehazing/ITS/models/A1XAccess.py"
    runner = repo / "experience_docx/tools/chd_rm_v4a_a1x_exact_half_accessibility.py"
    for path in (card, handoff, head, runner):
        require(path.is_file(), f"missing required file: {path}")
        ast.parse(path.read_text(encoding="utf-8")) if path.suffix == ".py" else None
    require(file_sha256(card) == ROUTE_CARD_SHA256, "frozen route-card hash mismatch")
    typed = json.loads(handoff.read_text(encoding="utf-8"))
    require(typed["route_id"] == ROUTE_ID, "handoff route_id mismatch")
    require(typed["source_commit"] == SOURCE_COMMIT, "handoff source_commit mismatch")
    require(typed["authorizes"] == "R2_LOCAL_ROUTE_SETUP_ONLY", "handoff authorization mismatch")
    source = head.read_text(encoding="utf-8")
    for token in ("input_channels = 15", "24", "48", "96", "nn.init.zeros_", "torch.tanh"):
        require(token in source, f"diagnostic contract token missing: {token}")
    runner_source = runner.read_text(encoding="utf-8")
    require("A1X_FORMAL_CONFIRMATION_ONLY" in runner_source, "runner authorization guard missing")
    payload = {
        "route_id": ROUTE_ID,
        "source_commit": SOURCE_COMMIT,
        "route_card_sha256": file_sha256(card),
        "handoff_sha256": file_sha256(handoff),
        "head_sha256": file_sha256(head),
        "runner_sha256": file_sha256(runner),
        "a1x_data_touched": False,
        "canary_touched": False,
        "locked_test_touched": False,
        "cloud_command": False,
        "validator": "PASS",
    }
    Path(args.output_json).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("A1X_ROUTE_STATIC_VALIDATOR_OK")


if __name__ == "__main__":
    main()
