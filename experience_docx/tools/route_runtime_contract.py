#!/usr/bin/env python3
"""Shared declarative contract for route-ready validation and cloud lifecycle."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


SPEC_SCHEMA_VERSION = 1
ASSET_SCHEMA_VERSION = 1
CONTEXT_SCHEMA_VERSION = 1
GENERIC_RUNNER_RELPATH = "experience_docx/tools/run_route_operation.sh"
RUNTIME_SPEC_DIRECTORY = "experience_docx/route_runtime_specs"
RUNTIME_BUNDLE_RELPATHS = (
    GENERIC_RUNNER_RELPATH,
    "experience_docx/tools/route_lifecycle.py",
    "experience_docx/tools/route_program_api.py",
    "experience_docx/tools/route_runtime_contract.py",
    "experience_docx/tools/run_telemetry.py",
)
EVIDENCE_ROLES = {
    "engineering_debug", "development_screening", "confirmation", "sealed_final",
}
RESUME_POLICIES = {"none", "complete_units"}
ALLOWED_EVIDENCE_SUFFIXES = {".json", ".csv", ".md", ".txt"}
SAFE_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
SAFE_RELPATH = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_./-]{0,511}$")
SAFE_ENV_KEY = re.compile(r"^CONVIR_ROUTE_[A-Z0-9_]{1,96}$")
SHA40 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
ASSET_PATH_TOKENS = ("{REMOTE_REPO}", "{RUN_ROOT}", "{OUTPUT_PATH}")
ASSET_ACCESS_ROLES = {
    "unrestricted", "engineering_debug", "development_screening",
    "confirmation", "canary", "sealed_final",
}


class ContractError(ValueError):
    pass


def canonical_digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def require_token(value: Any, name: str) -> str:
    if not isinstance(value, str) or not SAFE_TOKEN.fullmatch(value):
        raise ContractError(f"{name} must be a safe token")
    return value


def require_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ContractError(f"{name} must be boolean")
    return value


def require_int(value: Any, name: str, minimum: int, maximum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not minimum <= value <= maximum:
        raise ContractError(f"{name} must be in [{minimum}, {maximum}]")
    return value


def require_relpath(value: Any, name: str, *, prefix: str | None = None,
                    suffix: str | None = None) -> str:
    if not isinstance(value, str) or not SAFE_RELPATH.fullmatch(value) \
            or value.startswith("/") or ".." in Path(value).parts or "//" in value:
        raise ContractError(f"{name} must be a safe repository-relative path")
    if prefix is not None and not value.startswith(prefix):
        raise ContractError(f"{name} must start with {prefix}")
    if suffix is not None and not value.endswith(suffix):
        raise ContractError(f"{name} must end with {suffix}")
    return value


def safe_join(root: Path, relpath: str) -> Path:
    candidate = (root / relpath).absolute()
    try:
        candidate.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ContractError(f"path escapes root: {relpath}") from exc
    return candidate


def require_asset_path(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 1024 \
            or any(ord(character) < 32 for character in value):
        raise ContractError(f"{name} is invalid")
    tokens = [token for token in ASSET_PATH_TOKENS if token in value]
    if any(character in value for character in "{}"):
        if len(tokens) != 1 or not value.startswith(tokens[0]) \
                or any(token in value[len(tokens[0]):] for token in ASSET_PATH_TOKENS):
            raise ContractError(f"{name} has an invalid path token")
        suffix = value[len(tokens[0]):]
        if suffix and not suffix.startswith("/"):
            raise ContractError(f"{name} token must be followed by an absolute suffix")
        if ".." in Path(suffix).parts:
            raise ContractError(f"{name} cannot escape its token root")
    elif not Path(value).is_absolute():
        raise ContractError(f"{name} must be absolute or use one supported root token")
    return value


def runtime_spec_relpath(operation_id: str) -> str:
    return f"{RUNTIME_SPEC_DIRECTORY}/{require_token(operation_id, 'operation_id')}.json"


def _validate_permissions(value: Any) -> dict[str, bool]:
    expected = {"allow_confirmation", "allow_canary", "allow_locked_test"}
    if not isinstance(value, dict) or set(value) != expected:
        raise ContractError("protected_data_permissions has an invalid contract")
    return {key: require_bool(value[key], f"protected_data_permissions.{key}") for key in sorted(expected)}


def _validate_evidence_files(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not 1 <= len(value) <= 32:
        raise ContractError("evidence_files must contain 1-32 entries")
    result = []
    destinations = set()
    for index, item in enumerate(value):
        name = f"evidence_files[{index}]"
        expected = {"source_relpath", "destination_filename", "required", "max_bytes"}
        if not isinstance(item, dict) or set(item) != expected:
            raise ContractError(f"{name} has an invalid field contract")
        source = require_relpath(item["source_relpath"], f"{name}.source_relpath")
        if not source.startswith("workload/"):
            raise ContractError(f"{name}.source_relpath must be under workload/")
        destination = item["destination_filename"]
        if not isinstance(destination, str) or not SAFE_TOKEN.fullmatch(destination) \
                or Path(destination).name != destination:
            raise ContractError(f"{name}.destination_filename must be a filename")
        if Path(destination).suffix.lower() not in ALLOWED_EVIDENCE_SUFFIXES:
            raise ContractError(f"{name}.destination_filename has a forbidden suffix")
        if destination in destinations:
            raise ContractError("evidence destination filenames must be unique")
        destinations.add(destination)
        result.append({
            "source_relpath": source,
            "destination_filename": destination,
            "required": require_bool(item["required"], f"{name}.required"),
            "max_bytes": require_int(item["max_bytes"], f"{name}.max_bytes", 1, 1024 * 1024),
        })
    return result


def validate_runtime_spec(value: Any, manifest: dict[str, Any], operation_id: str) -> dict[str, Any]:
    expected = {
        "schema_version", "route_id", "operation_id", "entrypoint_relpath",
        "asset_manifest_relpath", "timeout_seconds", "expected_wall_seconds",
        "total_units", "evidence_role", "resume_policy", "protected_data_permissions",
        "environment", "evidence_files",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ContractError("runtime spec has an invalid top-level contract")
    if value["schema_version"] != SPEC_SCHEMA_VERSION:
        raise ContractError(f"runtime spec must use schema {SPEC_SCHEMA_VERSION}")
    if not isinstance(manifest, dict) or operation_id not in manifest.get("operations", {}):
        raise ContractError("runtime spec operation is absent from route manifest")
    operation = manifest["operations"][operation_id]
    route_id = require_token(value["route_id"], "route_id")
    if route_id != manifest.get("route_id"):
        raise ContractError("runtime spec route_id does not match route manifest")
    if require_token(value["operation_id"], "operation_id") != operation_id:
        raise ContractError("runtime spec operation_id mismatch")
    if operation.get("runner_relpath") != GENERIC_RUNNER_RELPATH:
        raise ContractError("route-ready operations must use the generic runner")
    entrypoint = require_relpath(
        value["entrypoint_relpath"], "entrypoint_relpath",
        prefix="experience_docx/tools/", suffix=".py",
    )
    if entrypoint in RUNTIME_BUNDLE_RELPATHS:
        raise ContractError("entrypoint must be route-specific")
    asset_path = value["asset_manifest_relpath"]
    if asset_path is not None:
        asset_path = require_relpath(
            asset_path, "asset_manifest_relpath",
            prefix="experience_docx/route_assets/", suffix=".json",
        )
    timeout = require_int(value["timeout_seconds"], "timeout_seconds", 1, 7 * 24 * 3600)
    expected_wall = require_int(
        value["expected_wall_seconds"], "expected_wall_seconds", 1, timeout,
    )
    role = value["evidence_role"]
    if role not in EVIDENCE_ROLES:
        raise ContractError(f"evidence_role must be one of {sorted(EVIDENCE_ROLES)}")
    resume = value["resume_policy"]
    if resume not in RESUME_POLICIES:
        raise ContractError(f"resume_policy must be one of {sorted(RESUME_POLICIES)}")
    if operation.get("output_policy") != "new":
        raise ContractError("resume_policy and operation output_policy disagree")
    permissions = _validate_permissions(value["protected_data_permissions"])
    if permissions["allow_locked_test"] and role != "sealed_final":
        raise ContractError("locked-test access requires sealed_final evidence role")
    if permissions["allow_confirmation"] and role not in {"confirmation", "sealed_final"}:
        raise ContractError("confirmation access requires confirmation or sealed_final role")
    if resume == "complete_units" and asset_path is None:
        raise ContractError("complete_units recovery requires a typed asset manifest")
    environment = value["environment"]
    if not isinstance(environment, dict) or len(environment) > 32:
        raise ContractError("environment must be an object with at most 32 entries")
    normalized_environment = {}
    for key, item in sorted(environment.items()):
        if not SAFE_ENV_KEY.fullmatch(key):
            raise ContractError(f"environment key is not allowed: {key}")
        if not isinstance(item, str) or len(item) > 512 \
                or any(ord(character) < 32 for character in item):
            raise ContractError(f"environment value is invalid: {key}")
        normalized_environment[key] = item
    return {
        "schema_version": SPEC_SCHEMA_VERSION,
        "route_id": route_id,
        "operation_id": operation_id,
        "entrypoint_relpath": entrypoint,
        "asset_manifest_relpath": asset_path,
        "timeout_seconds": timeout,
        "expected_wall_seconds": expected_wall,
        "total_units": require_int(value["total_units"], "total_units", 0, 10_000_000),
        "evidence_role": role,
        "resume_policy": resume,
        "protected_data_permissions": permissions,
        "environment": normalized_environment,
        "evidence_files": _validate_evidence_files(value["evidence_files"]),
    }


def validate_asset_manifest(value: Any, spec: dict[str, Any]) -> dict[str, Any]:
    expected = {"schema_version", "route_id", "operation_id", "assets"}
    if not isinstance(value, dict) or set(value) != expected:
        raise ContractError("asset manifest has an invalid top-level contract")
    if value["schema_version"] != ASSET_SCHEMA_VERSION:
        raise ContractError(f"asset manifest must use schema {ASSET_SCHEMA_VERSION}")
    if value["route_id"] != spec["route_id"] or value["operation_id"] != spec["operation_id"]:
        raise ContractError("asset manifest identity mismatch")
    assets = value["assets"]
    if not isinstance(assets, list) or not 1 <= len(assets) <= 128:
        raise ContractError("assets must contain 1-128 entries")
    result = []
    identifiers = set()
    for index, item in enumerate(assets):
        name = f"assets[{index}]"
        if not isinstance(item, dict):
            raise ContractError(f"{name} must be an object")
        kind = item.get("kind")
        common = {"id", "kind", "path", "access_role", "contract_access"}
        required = {
            "file": common | {"sha256"},
            "directory": common,
            "git_checkout": common | {"commit", "require_clean"},
        }
        if kind not in required or set(item) != required[kind]:
            raise ContractError(f"{name} has an invalid {kind!r} contract")
        identifier = require_token(item["id"], f"{name}.id")
        if identifier in identifiers:
            raise ContractError("asset ids must be unique")
        identifiers.add(identifier)
        path = require_asset_path(item["path"], f"{name}.path")
        access_role = item["access_role"]
        if access_role not in ASSET_ACCESS_ROLES:
            raise ContractError(f"{name}.access_role is invalid")
        contract_access = require_bool(item["contract_access"], f"{name}.contract_access")
        if contract_access and access_role in {"confirmation", "canary", "sealed_final"}:
            raise ContractError(f"{name} cannot expose protected data to contract phase")
        permissions = spec["protected_data_permissions"]
        required_permission = {
            "confirmation": "allow_confirmation",
            "canary": "allow_canary",
            "sealed_final": "allow_locked_test",
        }.get(access_role)
        if required_permission is not None and not permissions[required_permission]:
            raise ContractError(f"{name} is not permitted by the runtime spec")
        normalized = {
            "id": identifier, "kind": kind, "path": path,
            "access_role": access_role, "contract_access": contract_access,
        }
        if kind == "file":
            if not isinstance(item["sha256"], str) or not SHA256.fullmatch(item["sha256"]):
                raise ContractError(f"{name}.sha256 is invalid")
            normalized["sha256"] = item["sha256"]
        elif kind == "git_checkout":
            if not isinstance(item["commit"], str) or not SHA40.fullmatch(item["commit"]):
                raise ContractError(f"{name}.commit is invalid")
            normalized.update({
                "commit": item["commit"],
                "require_clean": require_bool(item["require_clean"], f"{name}.require_clean"),
            })
        result.append(normalized)
    return {
        "schema_version": ASSET_SCHEMA_VERSION,
        "route_id": spec["route_id"],
        "operation_id": spec["operation_id"],
        "assets": result,
    }
