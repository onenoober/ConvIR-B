#!/usr/bin/env python3
"""Prepare one same-contract operation amendment from an authorized closeout."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import convir_ops_mcp as ops
from route_runtime_contract import (
    GENERIC_RUNNER_RELPATH,
    ContractError,
    runtime_spec_relpath,
    validate_runtime_spec,
)


GIT = "/usr/bin/git"
REQUEST_FIELDS = {"schema_version", "operation_id", "operation", "runtime_spec"}
OPERATION_FIELDS = {
    "mode", "require_gpu", "output_id", "closeout_filename",
    "allowed_terminal_tuples", "workspace_policy", "output_policy",
    "monitor_profile", "heartbeat_timeout_seconds", "min_free_gpu_mib",
    "max_gpu_utilization_pct",
}
SPEC_FIELDS = {
    "entrypoint_relpath", "asset_manifest_relpath", "timeout_seconds",
    "expected_wall_seconds", "total_units", "evidence_role", "resume_policy",
    "protected_data_permissions", "environment", "evidence_files",
}
GENERIC_ENGINEERING_TERMINAL = {
    "state": "FAILED_ENGINEERING", "decision": None, "authorizes": "NONE",
}


class AmendmentError(RuntimeError):
    pass


def git(repo: Path, *args: str) -> bytes:
    completed = subprocess.run(
        [GIT, *args], cwd=repo, capture_output=True, timeout=60, check=False,
    )
    if completed.returncode:
        detail = (completed.stdout + completed.stderr).decode(errors="replace").strip()
        raise AmendmentError(f"git {' '.join(args)} failed: {detail[:4096]}")
    return completed.stdout


def committed_json(repo: Path, relpath: str) -> dict[str, Any]:
    try:
        value = json.loads(git(repo, "show", f"HEAD:{relpath}"))
    except json.JSONDecodeError as exc:
        raise AmendmentError(f"committed JSON is invalid: {relpath}: {exc}") from exc
    if not isinstance(value, dict):
        raise AmendmentError(f"committed JSON must be an object: {relpath}")
    return value


def load_request(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AmendmentError(f"cannot read amendment request: {exc}") from exc
    if not isinstance(value, dict) or set(value) != REQUEST_FIELDS:
        raise AmendmentError("amendment request has an invalid top-level contract")
    if value["schema_version"] != 1:
        raise AmendmentError("amendment request must use schema 1")
    if not isinstance(value["operation"], dict) or set(value["operation"]) != OPERATION_FIELDS:
        raise AmendmentError("amendment operation has an invalid field contract")
    if not isinstance(value["runtime_spec"], dict) or set(value["runtime_spec"]) != SPEC_FIELDS:
        raise AmendmentError("amendment runtime_spec has an invalid field contract")
    return value


def normalized_operation(value: dict[str, Any], prior_relpath: str,
                         prior_tuple: dict[str, Any]) -> dict[str, Any]:
    operation = dict(value)
    operation_id = prior_tuple["authorizes"]
    ops.require_token(operation["mode"], "mode")
    ops.require_bool(operation["require_gpu"], "require_gpu")
    ops.require_token(operation["output_id"], "output_id")
    closeout = operation["closeout_filename"]
    if not isinstance(closeout, str) or not re.fullmatch(
        r"[A-Za-z0-9_.-]+_closeout\.json", closeout,
    ):
        raise AmendmentError("closeout_filename must end with _closeout.json")
    terminals = ops.require_terminal_tuples(operation["allowed_terminal_tuples"])
    if GENERIC_ENGINEERING_TERMINAL not in terminals:
        raise AmendmentError("allowed_terminal_tuples must include engineering failure")
    workspace = ops.require_enum(
        operation["workspace_policy"], "workspace_policy",
        {"fresh_route", "exact_continuation"},
    )
    output_policy = ops.require_enum(
        operation["output_policy"], "output_policy", {"new", "exact_resume"},
    )
    monitor = ops.require_enum(
        operation["monitor_profile"], "monitor_profile", set(ops.MONITOR_PROFILES),
    )
    heartbeat = ops.require_int(
        operation["heartbeat_timeout_seconds"], "heartbeat_timeout_seconds", 30, 86400,
    )
    min_free = ops.require_int(operation["min_free_gpu_mib"], "min_free_gpu_mib", 0, 1048576)
    max_util = ops.require_int(
        operation["max_gpu_utilization_pct"], "max_gpu_utilization_pct", 0, 100,
    )
    if operation["require_gpu"] and min_free < 1:
        raise AmendmentError("GPU operation requires positive min_free_gpu_mib")
    if not operation["require_gpu"] and (min_free != 0 or max_util != 100):
        raise AmendmentError("CPU operation requires 0 MiB and 100% thresholds")
    return {
        "runner_relpath": GENERIC_RUNNER_RELPATH,
        "mode": operation["mode"],
        "require_gpu": operation["require_gpu"],
        "output_id": operation["output_id"],
        "closeout_filename": closeout,
        "prior_closeout_relpath": prior_relpath,
        "prior_terminal_tuple": prior_tuple,
        "allowed_terminal_tuples": terminals,
        "workspace_policy": workspace,
        "output_policy": output_policy,
        "monitor_profile": monitor,
        "heartbeat_timeout_seconds": heartbeat,
        "min_free_gpu_mib": min_free,
        "max_gpu_utilization_pct": max_util,
    }


def build_amendment(manifest: dict[str, Any], closeout: dict[str, Any],
                    request: dict[str, Any], prior_relpath: str) \
        -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(manifest, dict) or set(manifest) != {
        "schema_version", "route_id", "rules_commit", "route_card_relpath", "operations",
    }:
        raise AmendmentError("current route manifest has an invalid contract")
    if not isinstance(manifest["operations"], dict) or not manifest["operations"]:
        raise AmendmentError("current route manifest has no operations")
    operation_id = ops.require_token(request["operation_id"], "operation_id")
    route_id = ops.require_token(manifest["route_id"], "route_id")
    if manifest["schema_version"] != ops.SCHEMA_VERSION:
        raise AmendmentError(f"route manifest must use schema {ops.SCHEMA_VERSION}")
    ops.require_sha(manifest["rules_commit"], "rules_commit", ops.SHA40)
    ops.require_relpath(
        manifest["route_card_relpath"], "route_card_relpath", ".md",
        prefix="experience_docx/experiment_cards/",
    )
    if operation_id in manifest["operations"]:
        raise AmendmentError("requested operation already exists in the current manifest")
    if closeout.get("route_id") != route_id:
        raise AmendmentError("prior closeout route_id does not match the route manifest")
    prior_tuple = {key: closeout.get(key) for key in ("state", "decision", "authorizes")}
    if prior_tuple["state"] != "COMPLETED_GATE_PASS" \
            or prior_tuple["authorizes"] != operation_id \
            or not isinstance(prior_tuple["decision"], str):
        raise AmendmentError("prior closeout does not authorize the requested operation")
    ops.require_terminal_tuple(prior_tuple, "prior_terminal_tuple")
    operation = normalized_operation(request["operation"], prior_relpath, prior_tuple)
    for existing in manifest["operations"].values():
        if not isinstance(existing, dict):
            continue
        if existing.get("output_id") == operation["output_id"]:
            raise AmendmentError("new operation reuses an existing output_id")
        if existing.get("closeout_filename") == operation["closeout_filename"]:
            raise AmendmentError("new operation reuses an existing closeout filename")
    candidate_manifest = {
        "schema_version": manifest["schema_version"],
        "route_id": route_id,
        "rules_commit": manifest["rules_commit"],
        "route_card_relpath": manifest["route_card_relpath"],
        "operations": {operation_id: operation},
    }
    candidate_spec = {
        "schema_version": 1,
        "route_id": route_id,
        "operation_id": operation_id,
        **request["runtime_spec"],
    }
    try:
        candidate_spec = validate_runtime_spec(
            candidate_spec, candidate_manifest, operation_id,
        )
    except ContractError as exc:
        raise AmendmentError(str(exc)) from exc
    return candidate_manifest, candidate_spec


def atomic_write(path: Path, value: dict[str, Any], *, replace: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise AmendmentError(f"refusing to replace existing file: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare(repo: Path, request_path: Path, prior_relpath: str, *, apply: bool) -> dict[str, Any]:
    try:
        request_path.relative_to(repo)
    except ValueError:
        pass
    else:
        raise AmendmentError("amendment request must stay outside the route worktree")
    request = load_request(request_path)
    prior_relpath = ops.require_relpath(
        prior_relpath, "prior_closeout_relpath", ".json",
        prefix="experience_docx/experiment_logs/",
    )
    manifest_path = repo / ops.ROUTE_OPERATIONS_RELPATH
    if subprocess.run([GIT, "diff", "--quiet", "--", ops.ROUTE_OPERATIONS_RELPATH], cwd=repo).returncode \
            or subprocess.run([GIT, "diff", "--cached", "--quiet", "--", ops.ROUTE_OPERATIONS_RELPATH], cwd=repo).returncode:
        raise AmendmentError("route_operations.json already has local changes")
    manifest = committed_json(repo, ops.ROUTE_OPERATIONS_RELPATH)
    closeout = committed_json(repo, prior_relpath)
    candidate_manifest, candidate_spec = build_amendment(
        manifest, closeout, request, prior_relpath,
    )
    operation_id = request["operation_id"]
    spec_relpath = runtime_spec_relpath(operation_id)
    spec_path = repo / spec_relpath
    if apply and spec_path.exists():
        raise AmendmentError(f"runtime spec already exists: {spec_relpath}")
    if apply:
        atomic_write(spec_path, candidate_spec, replace=False)
        atomic_write(manifest_path, candidate_manifest, replace=True)
    return {
        "schema_version": 1,
        "status": "NEXT_OPERATION_APPLIED" if apply else "NEXT_OPERATION_READY",
        "route_id": candidate_manifest["route_id"],
        "operation_id": operation_id,
        "prior_closeout_relpath": prior_relpath,
        "prior_terminal_tuple": candidate_manifest["operations"][operation_id]["prior_terminal_tuple"],
        "manifest_relpath": ops.ROUTE_OPERATIONS_RELPATH,
        "runtime_spec_relpath": spec_relpath,
        "preserved_fields": ["route_id", "rules_commit", "route_card_relpath"],
        "next_actions": [
            "complete_route_card_entrypoint_assets",
            "stage_complete_bundle_once",
            "validate_route_ready_once",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--prior-closeout", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    try:
        report = prepare(
            args.repo.resolve(), args.request.resolve(), args.prior_closeout,
            apply=args.apply,
        )
    except (AmendmentError, ops.ToolError, ContractError) as exc:
        print(f"NEXT_OPERATION_ERROR {exc}")
        raise SystemExit(1)
    print(json.dumps(report, sort_keys=True))
    print(f"{report['status']} operation_id={report['operation_id']}")


if __name__ == "__main__":
    main()
