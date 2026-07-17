#!/usr/bin/env python3
"""Migrate signed convir-ops v4 state to the v4.3 failure-review contract."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import time
from pathlib import Path

import convir_ops_mcp as ops


def read_verified(path: Path, token: str) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not hmac.compare_digest(token, ops.sign(value.get("payload"))):
        raise ops.ToolError(f"state integrity failed: {path.name}")
    return value


def closeout_summary(repo: Path, context: dict) -> dict | None:
    path = (
        repo / "experience_docx" / "experiment_logs" / context["route_id"]
        / context["closeout_filename"]
    )
    if not path.is_file():
        return None
    raw = path.read_bytes()
    if len(raw) > ops.MAX_CLOSEOUT_BYTES:
        raise ops.ToolError(f"archived closeout is oversized: {path}")
    marker = (
        f"CONVIR_OPS_CLOSEOUT_SHA256={hashlib.sha256(raw).hexdigest()}\n"
        "CONVIR_OPS_CLOSEOUT_BEGIN\n"
        + raw.decode("utf-8")
        + "\nCONVIR_OPS_CLOSEOUT_END\n"
    )
    return ops.parse_closeout(context, marker)


def migrate(state_dir: Path, repo: Path, *, apply: bool, now: int) -> dict:
    state_dir = state_dir.resolve()
    repo = repo.resolve()
    if not state_dir.is_dir() or not (state_dir / "hmac.key").is_file():
        raise ops.ToolError("state directory is unavailable")
    if not (repo / ".git").exists():
        raise ops.ToolError("repository is not a Git worktree")
    ops.STATE_DIR = state_dir

    expired_plans = []
    retained_plans = []
    for path in sorted(state_dir.glob("plan-*.json")):
        token = path.stem.removeprefix("plan-")
        value = read_verified(path, token)
        expires_at = value.get("payload", {}).get("expires_at")
        if not isinstance(expires_at, int):
            raise ops.ToolError(f"plan expiry is invalid: {path.name}")
        if expires_at < now:
            expired_plans.append(path.name)
            if apply:
                path.unlink()
        else:
            retained_plans.append(path.name)

    migrated_engineering = []
    migrated_scientific = []
    unchanged_receipts = []
    for path in sorted(state_dir.glob("receipt-*.json")):
        token = path.stem.removeprefix("receipt-")
        value = read_verified(path, token)
        context = value.get("payload", {}).get("context")
        if not isinstance(context, dict):
            raise ops.ToolError(f"receipt context is invalid: {path.name}")
        summary = closeout_summary(repo, context)
        if summary is None:
            unchanged_receipts.append(path.name)
            continue
        terminal_state = summary["terminal_tuple"]["state"]
        if terminal_state == "FAILED_ENGINEERING":
            migrated_engineering.append(path.name)
            target_state = "ENGINEERING_ARCHIVE_AUTHORIZED"
            resolution = "archive"
        else:
            migrated_scientific.append(path.name)
            target_state = "CLOSEOUT_VALIDATED"
            resolution = None
        if apply:
            with ops.locked_record("receipt", token) as record:
                record["terminal_closeout"] = summary
                record["finish_closed"] = target_state
                record["engineering_failure_resolution"] = resolution
                record["v43_migrated_at"] = now

    return {
        "schema_version": 1,
        "applied": apply,
        "state_dir": str(state_dir),
        "repo": str(repo),
        "migration_time_unix": now,
        "expired_plans": expired_plans,
        "retained_plans": retained_plans,
        "migrated_engineering_receipts": migrated_engineering,
        "migrated_scientific_receipts": migrated_scientific,
        "unchanged_receipts": unchanged_receipts,
        "receipt_records_deleted": 0,
        "hmac_key_deleted": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = migrate(
        args.state_dir, args.repo, apply=args.apply, now=int(time.time()),
    )
    raw = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.write_text(raw, encoding="utf-8")
    print(raw, end="")
    print("CONVIR_OPS_V43_STATE_MIGRATION_OK")


if __name__ == "__main__":
    main()
