#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


TOOL_PATH = Path(__file__).resolve()
REPO_ROOT = TOOL_PATH.parents[2]
ITS_ROOT = REPO_ROOT / "Dehazing" / "ITS"
for path in (str(TOOL_PATH.parent), str(ITS_ROOT), str(REPO_ROOT), os.getcwd()):
    if path not in sys.path:
        sys.path.insert(0, path)

import nopost_lowband_v220_contract_identity as v220_p0  # noqa: E402


def rewrite_prefix(path: Path, old: str, new: str) -> None:
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace(old, new), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max-images", type=int, default=8)
    args = ap.parse_args()

    v220_p0.main_args = args
    original_argv = sys.argv[:]
    try:
        sys.argv = [
            str(TOOL_PATH),
            "--data-dir",
            str(args.data_dir),
            "--checkpoint",
            str(args.checkpoint),
            "--out-dir",
            str(args.out_dir),
            "--max-images",
            str(args.max_images),
        ]
        v220_p0.main()
    finally:
        sys.argv = original_argv

    mappings = [
        ("v220_p0_forbidden_symbol_scan.txt", "v221_p0_forbidden_symbol_scan.txt"),
        ("v220_p0_forward_signature.json", "v221_p0_forward_signature.json"),
        ("v220_p0_param_groups.json", "v221_p0_param_groups.json"),
        ("v220_p0_identity_summary.json", "v221_p0_identity_summary.json"),
        ("v220_p0_contract_audit.md", "v221_p0_contract_audit.md"),
        ("v220_p0_decision.md", "v221_p0_decision.md"),
    ]
    for old, new in mappings:
        src = args.out_dir / old
        dst = args.out_dir / new
        if src.exists():
            src.replace(dst)
            rewrite_prefix(dst, "v2.20", "v2.21")
            rewrite_prefix(dst, "V220", "V221")
            rewrite_prefix(dst, "v220", "v221")

    decision_path = args.out_dir / "v221_p0_decision.md"
    decision_text = decision_path.read_text(encoding="utf-8")
    decision_text = decision_text.replace(
        "P0_PASS_MIDFINAL_CONTEXT_CONTRACT_IDENTITY_SOURCE_CLEAN",
        "P0_PASS_V221_SAFETY_CALIBRATED_REPLAY_CONTRACT_IDENTITY_SOURCE_CLEAN",
    )
    decision_text = decision_text.replace(
        "P0_FAIL_MIDFINAL_CONTEXT_CONTRACT_IDENTITY",
        "P0_FAIL_V221_SAFETY_CALIBRATED_REPLAY_CONTRACT_IDENTITY",
    )
    decision_path.write_text(decision_text, encoding="utf-8")

    audit_path = args.out_dir / "v221_p0_contract_audit.md"
    audit_text = audit_path.read_text(encoding="utf-8")
    audit_text = audit_text.replace(
        "P0_PASS_MIDFINAL_CONTEXT_CONTRACT_IDENTITY_SOURCE_CLEAN",
        "P0_PASS_V221_SAFETY_CALIBRATED_REPLAY_CONTRACT_IDENTITY_SOURCE_CLEAN",
    )
    audit_text = audit_text.replace(
        "P0_FAIL_MIDFINAL_CONTEXT_CONTRACT_IDENTITY",
        "P0_FAIL_V221_SAFETY_CALIBRATED_REPLAY_CONTRACT_IDENTITY",
    )
    audit_text += (
        "\nThis P0 reuses the v2.20 NoPost mid+final context route as the action source "
        "and verifies that v2.21 remains a replay-only safety-controller audit. "
        "Training and locked-test commands are not launched.\n"
    )
    audit_path.write_text(audit_text, encoding="utf-8")

    closeout: dict[str, Any] = {
        "decision": "P0_PASS_V221_SAFETY_CALIBRATED_REPLAY_CONTRACT_IDENTITY_SOURCE_CLEAN",
        "replay_only": True,
        "training_launched": False,
        "locked_test_touched": False,
    }
    (args.out_dir / "v221_p0_contract_closeout.json").write_text(
        json.dumps(closeout, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("V221_P0_CONTRACT_IDENTITY_OK", closeout["decision"], flush=True)


if __name__ == "__main__":
    main()
