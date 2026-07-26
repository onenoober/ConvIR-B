#!/usr/bin/env python3
"""Small route-program API for context-only paths and atomic result writes."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from route_runtime_contract import (
    ASSET_ACCESS_ROLES,
    CONTEXT_SCHEMA_VERSION,
    EVIDENCE_ROLES,
    RESUME_POLICIES,
    SHA40,
    SHA256,
    ContractError,
    require_int,
    require_relpath,
    require_token,
    safe_join,
)


LEGACY_CONTEXT_FIELDS = {
    "schema_version", "phase", "route_id", "operation_id", "run_id",
    "route_commit", "runner_sha256", "entrypoint_relpath", "remote_repo",
    "run_root", "output_path", "phase_output_path", "result_path", "status_path",
    "heartbeat_path", "device", "total_units", "evidence_role",
    "resume_policy", "protected_data_permissions", "assets",
}
CONTEXT_FIELDS = LEGACY_CONTEXT_FIELDS | {"engineering_contract"}
MAX_RESULT_BYTES = 32 * 1024


@dataclass(frozen=True)
class RouteAsset:
    id: str
    kind: str
    path: Path
    access_role: str
    contract_access: bool
    sha256: str | None = None
    commit: str | None = None


@dataclass(frozen=True)
class RouteContext:
    phase: str
    route_id: str
    operation_id: str
    run_id: str
    route_commit: str
    runner_sha256: str
    entrypoint_relpath: str
    remote_repo: Path
    run_root: Path
    output_path: Path
    phase_output_path: Path
    result_path: Path
    status_path: Path
    heartbeat_path: Path
    device: str
    total_units: int
    evidence_role: str
    resume_policy: str
    protected_data_permissions: dict[str, bool]
    assets: dict[str, RouteAsset]
    engineering_contract: dict[str, Any]


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"write-once JSON already exists: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(f"write-once JSON already exists: {path}") from exc
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _validate_result_size(value: dict[str, Any]) -> None:
    try:
        raw = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ContractError("route result is not JSON serializable") from exc
    if len(raw) > MAX_RESULT_BYTES:
        raise ContractError("route result exceeds the fixed size limit")


def _absolute_path(value: Any, name: str) -> Path:
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ContractError(f"{name} must be an absolute path")
    return Path(value).resolve()


def _load_assets(value: Any) -> dict[str, RouteAsset]:
    if not isinstance(value, list) or len(value) > 128:
        raise ContractError("route context assets must be a list with at most 128 entries")
    result: dict[str, RouteAsset] = {}
    for index, item in enumerate(value):
        name = f"assets[{index}]"
        if not isinstance(item, dict) or item.get("kind") not in {
            "file", "directory", "git_checkout",
        }:
            raise ContractError(f"{name} has an invalid contract")
        kind = item["kind"]
        common = {"id", "kind", "path", "access_role", "contract_access"}
        expected = {
            "file": common | {"sha256"},
            "directory": common,
            "git_checkout": common | {"commit"},
        }[kind]
        if set(item) != expected:
            raise ContractError(f"{name} has an invalid {kind} field contract")
        identifier = require_token(item["id"], f"{name}.id")
        if identifier in result:
            raise ContractError("route context asset ids must be unique")
        asset_path = _absolute_path(item["path"], f"{name}.path")
        access_role = item["access_role"]
        contract_access = item["contract_access"]
        if access_role not in ASSET_ACCESS_ROLES or not isinstance(contract_access, bool):
            raise ContractError(f"{name} access contract is invalid")
        digest = item.get("sha256")
        commit = item.get("commit")
        if digest is not None and (not isinstance(digest, str) or not SHA256.fullmatch(digest)):
            raise ContractError(f"{name}.sha256 is invalid")
        if commit is not None and (not isinstance(commit, str) or not SHA40.fullmatch(commit)):
            raise ContractError(f"{name}.commit is invalid")
        result[identifier] = RouteAsset(
            identifier, kind, asset_path, access_role, contract_access, digest, commit,
        )
    return result


def load_context(path: Path, expected_phase: str) -> RouteContext:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or frozenset(value) not in {
        frozenset(LEGACY_CONTEXT_FIELDS), frozenset(CONTEXT_FIELDS),
    }:
        raise ContractError("route context has an invalid field contract")
    if value["schema_version"] != CONTEXT_SCHEMA_VERSION:
        raise ContractError(f"route context must use schema {CONTEXT_SCHEMA_VERSION}")
    if value["phase"] != expected_phase or expected_phase not in {"contract", "run"}:
        raise ContractError("route context phase mismatch")
    route_id = require_token(value["route_id"], "route_id")
    operation_id = require_token(value["operation_id"], "operation_id")
    run_id = require_token(value["run_id"], "run_id")
    route_commit = value["route_commit"]
    runner_sha256 = value["runner_sha256"]
    if not isinstance(route_commit, str) or not SHA40.fullmatch(route_commit):
        raise ContractError("route context route_commit is invalid")
    if not isinstance(runner_sha256, str) or not SHA256.fullmatch(runner_sha256):
        raise ContractError("route context runner_sha256 is invalid")
    entrypoint = require_relpath(
        value["entrypoint_relpath"], "entrypoint_relpath",
        prefix="experience_docx/tools/", suffix=".py",
    )
    remote_repo = _absolute_path(value["remote_repo"], "remote_repo")
    run_root = _absolute_path(value["run_root"], "run_root")
    output_path = _absolute_path(value["output_path"], "output_path")
    phase_output = _absolute_path(value["phase_output_path"], "phase_output_path")
    result_path = _absolute_path(value["result_path"], "result_path")
    status_path = _absolute_path(value["status_path"], "status_path")
    heartbeat_path = _absolute_path(value["heartbeat_path"], "heartbeat_path")
    if output_path != (run_root / run_id).resolve():
        raise ContractError("output_path must equal run_root/run_id")
    phase_directory = "contract" if expected_phase == "contract" else "workload"
    if phase_output != (output_path / phase_directory).resolve():
        raise ContractError("phase_output_path does not match the phase")
    if result_path != (phase_output / f"{expected_phase}_result.json").resolve():
        raise ContractError("result_path does not match the phase result contract")
    if status_path != (output_path / "status.txt").resolve() \
            or heartbeat_path != (output_path / "heartbeat.json").resolve():
        raise ContractError("status or heartbeat path does not match output_path")
    if path.resolve() != (output_path / "control" / f"{expected_phase}_context.json").resolve():
        raise ContractError("context file is outside the fixed control path")
    permissions = value["protected_data_permissions"]
    if not isinstance(permissions, dict) or set(permissions) != {
        "allow_confirmation", "allow_canary", "allow_locked_test",
    } or not all(isinstance(item, bool) for item in permissions.values()):
        raise ContractError("route context protected-data permissions are invalid")
    device = value["device"]
    if device not in {"cpu", "cuda"}:
        raise ContractError("route context device is invalid for the phase")
    evidence_role = value["evidence_role"]
    if evidence_role not in EVIDENCE_ROLES:
        raise ContractError("route context evidence_role is invalid")
    resume_policy = value["resume_policy"]
    if resume_policy not in RESUME_POLICIES:
        raise ContractError("route context resume_policy is invalid")
    total_units = require_int(value["total_units"], "total_units", 0, 10_000_000)
    assets = _load_assets(value["assets"])
    engineering = value.get("engineering_contract", {
        "mode": "cpu_exact", "legacy_implicit_contract": True,
    })
    if not isinstance(engineering, dict) or engineering.get("mode") not in {
        "metadata_only", "cpu_exact", "cpu_reference_equivalent",
        "gpu_synthetic_no_data",
    }:
        raise ContractError("route context engineering_contract is invalid")
    if expected_phase == "contract" \
            and engineering["mode"] != "gpu_synthetic_no_data" and device != "cpu":
        raise ContractError("only gpu_synthetic_no_data may use CUDA in contract phase")
    return RouteContext(
        phase=expected_phase,
        route_id=route_id,
        operation_id=operation_id,
        run_id=run_id,
        route_commit=route_commit,
        runner_sha256=runner_sha256,
        entrypoint_relpath=entrypoint,
        remote_repo=remote_repo,
        run_root=run_root,
        output_path=output_path,
        phase_output_path=phase_output,
        result_path=result_path,
        status_path=status_path,
        heartbeat_path=heartbeat_path,
        device=device,
        total_units=total_units,
        evidence_role=evidence_role,
        resume_policy=resume_policy,
        protected_data_permissions=permissions,
        assets=assets,
        engineering_contract=engineering,
    )


def prepare_phase_output(context: RouteContext) -> None:
    if context.phase_output_path.exists():
        raise FileExistsError(f"phase output already exists: {context.phase_output_path}")
    context.phase_output_path.mkdir(parents=False)


def output_file(context: RouteContext, relpath: str) -> Path:
    return safe_join(context.phase_output_path, relpath)


def asset_path(context: RouteContext, asset_id: str, *, kind: str | None = None) -> Path:
    identifier = require_token(asset_id, "asset_id")
    if identifier not in context.assets:
        raise ContractError(f"required route asset is unavailable: {identifier}")
    asset = context.assets[identifier]
    if kind is not None and asset.kind != kind:
        raise ContractError(f"route asset {identifier} is not a {kind}")
    return asset.path


def write_workload_progress(
    context: RouteContext, *, completed_units: int, stage: str,
) -> None:
    """Append one generic machine-readable progress milestone."""
    if context.phase != "run":
        raise ContractError("workload progress requires run context")
    completed = require_int(
        completed_units, "completed_units", 0, context.total_units,
    )
    stage = require_token(stage, "stage")
    value = {
        "phase": "workload",
        "event": "workload_progress",
        "stage": stage,
        "completed_units": completed,
        "total_units": context.total_units,
    }
    line = json.dumps(value, sort_keys=True, separators=(",", ":"))
    with context.status_path.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
        stream.flush()
    print(line, flush=True)


def write_contract_progress(
    context: RouteContext, *, completed_iterations: int,
    total_iterations: int, stage: str,
) -> None:
    """Append one bounded control-only contract progress milestone."""
    if context.phase != "contract":
        raise ContractError("contract progress requires contract context")
    total = require_int(total_iterations, "total_iterations", 1, 10_000_000)
    completed = require_int(
        completed_iterations, "completed_iterations", 0, total,
    )
    value = {
        "phase": "contract",
        "event": "contract_progress",
        "stage": require_token(stage, "stage"),
        "completed_iterations": completed,
        "total_iterations": total,
    }
    line = json.dumps(value, sort_keys=True, separators=(",", ":"))
    with context.status_path.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
        stream.flush()
    print(line, flush=True)


def write_contract_result(context: RouteContext, *, checks: dict[str, bool],
                          engineering: dict[str, Any] | None = None) -> None:
    if context.phase != "contract" or not checks \
            or not all(isinstance(key, str) and require_token(key, "check")
                       and isinstance(value, bool) for key, value in checks.items()):
        raise ContractError("contract result checks are invalid")
    value = {
        "schema_version": 1,
        "route_id": context.route_id,
        "operation_id": context.operation_id,
        "phase": "contract",
        "ok": all(checks.values()),
        "checks": dict(sorted(checks.items())),
        "output_contract_checked": True,
        "finalizer_contract_checked": True,
        "confirmation_images_targets_outcomes_touched": False,
        "canary_touched": False,
        "locked_test_touched": False,
    }
    if not context.engineering_contract.get("legacy_implicit_contract"):
        expected = {
            "mode", "device", "fixture", "production_path_exercised",
            "protected_data_touched", "scientific_output_created",
            "scientific_training_occurred",
        }
        cost_contract = context.engineering_contract.get("cost_contract")
        if cost_contract is not None:
            expected.add("cost")
        if not isinstance(engineering, dict) or set(engineering) != expected:
            raise ContractError("schema-2 contract requires complete engineering evidence")
        if engineering["mode"] != context.engineering_contract["mode"] \
                or engineering["device"] != context.device:
            raise ContractError("engineering evidence mode/device mismatch")
        if any(engineering[key] is not False for key in (
            "protected_data_touched", "scientific_output_created",
            "scientific_training_occurred",
        )):
            raise ContractError("engineering contract cannot touch data, train, or publish science")
        fixture = engineering["fixture"]
        if fixture is not None:
            fields = {"batch", "channels", "height", "width"}
            if not isinstance(fixture, dict) or set(fixture) != fields \
                    or not all(isinstance(item, int) and not isinstance(item, bool) and item > 0
                               for item in fixture.values()):
                raise ContractError("engineering fixture is invalid")
        if cost_contract is not None:
            cost = engineering["cost"]
            cost_fields = {
                "observed_iterations", "observed_wall_seconds",
                "observed_peak_memory_mib",
            }
            if not isinstance(cost, dict) or set(cost) != cost_fields:
                raise ContractError("engineering cost evidence has an invalid field contract")
            expected_iterations = (
                cost_contract["formal_iterations"]
                if cost_contract["strategy"] == "same_scale_probe"
                else cost_contract["probe_iterations"]
            )
            if require_int(
                cost["observed_iterations"], "observed_iterations", 1, 100_000_000,
            ) != expected_iterations:
                raise ContractError("engineering cost evidence iteration count mismatch")
            wall = cost["observed_wall_seconds"]
            peak = cost["observed_peak_memory_mib"]
            if not isinstance(wall, (int, float)) or isinstance(wall, bool) or wall < 0:
                raise ContractError("observed_wall_seconds is invalid")
            if not isinstance(peak, (int, float)) or isinstance(peak, bool) or peak < 0:
                raise ContractError("observed_peak_memory_mib is invalid")
        value["engineering"] = engineering
    _validate_result_size(value)
    atomic_json(context.result_path, value)


def write_run_result(
    context: RouteContext,
    *,
    state: str,
    decision: str | None,
    authorizes: str,
    details: dict[str, Any] | None = None,
    confirmation_images_targets_outcomes_touched: bool = False,
    canary_touched: bool = False,
    locked_test_touched: bool = False,
) -> None:
    if context.phase != "run":
        raise ContractError("run result requires run context")
    require_token(state, "state")
    if decision is not None:
        require_token(decision, "decision")
    require_token(authorizes, "authorizes")
    if details is not None and not isinstance(details, dict):
        raise ContractError("run result details must be an object")
    touched = {
        "confirmation_images_targets_outcomes_touched": confirmation_images_targets_outcomes_touched,
        "canary_touched": canary_touched,
        "locked_test_touched": locked_test_touched,
    }
    if not all(isinstance(item, bool) for item in touched.values()):
        raise ContractError("protected-data touched fields must be boolean")
    value = {
        "schema_version": 1,
        "route_id": context.route_id,
        "operation_id": context.operation_id,
        "phase": "run",
        "state": state,
        "decision": decision,
        "authorizes": authorizes,
        "details": details or {},
        **touched,
    }
    _validate_result_size(value)
    atomic_json(context.result_path, value)
