#!/usr/bin/env python3
"""Compact experiment-assistant contract and durable record primitives."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import PurePosixPath
from typing import Any


CONTRACT_SCHEMA_VERSION = 1
ATTEMPT_SCHEMA_VERSION = 1
ARCHIVE_SCHEMA_VERSION = 1
MAX_AUTOMATIC_REPAIRS = 2

SAFE_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
SHA40 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
ENV_KEY = re.compile(r"^[A-Z_][A-Z0-9_]{0,95}$")
RESERVED_ENV_KEYS = {
    "CONVIR_EXPERIMENT_CONTRACT", "CONVIR_EXPERIMENT_OUTPUT", "LD_PRELOAD",
    "PYTHONHOME", "PYTHONPATH",
}

DATA_ROLES = {
    "training", "validation", "test", "confirmation", "canary", "locked_test",
}
PROTECTED_DATA_ROLES = {"confirmation", "canary", "locked_test"}
METRIC_DIRECTIONS = {"higher", "lower"}
ATTEMPT_STATES = {
    "PREPARED", "RUNNING", "COMPLETED_PASS", "COMPLETED_FAIL",
    "COMPLETED_INCONCLUSIVE", "FAILED_ENGINEERING", "CANCELLED", "UNKNOWN",
}
RESULT_STATES = {
    "COMPLETED_PASS", "COMPLETED_FAIL", "COMPLETED_INCONCLUSIVE",
}
PUBLIC_TOOL_NAMES = (
    "experiment_start", "experiment_status", "experiment_cancel",
    "experiment_repair", "experiment_get", "experiment_search",
)


class ContractError(ValueError):
    """Raised only when an issue would make execution or its result ambiguous."""


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ) + "\n").encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _require_dict(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{name} must be an object")
    return value


def _require_list(value: Any, name: str, *, minimum: int = 0) -> list[Any]:
    if not isinstance(value, list) or len(value) < minimum:
        raise ContractError(f"{name} must contain at least {minimum} item(s)")
    return value


def _require_token(value: Any, name: str) -> str:
    if not isinstance(value, str) or not SAFE_TOKEN.fullmatch(value):
        raise ContractError(f"{name} must be a safe token")
    return value


def _require_text(value: Any, name: str, *, minimum: int = 1,
                  maximum: int = 4096) -> str:
    if not isinstance(value, str):
        raise ContractError(f"{name} must be text")
    normalized = value.strip()
    if not minimum <= len(normalized) <= maximum:
        raise ContractError(f"{name} must contain {minimum}-{maximum} characters")
    return normalized


def _require_int(value: Any, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) \
            or not minimum <= value <= maximum:
        raise ContractError(f"{name} must be in [{minimum}, {maximum}]")
    return value


def _require_number(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{name} must be numeric")
    normalized = float(value)
    if not normalized == normalized or normalized in {float("inf"), float("-inf")}:
        raise ContractError(f"{name} must be finite")
    if positive and normalized <= 0:
        raise ContractError(f"{name} must be positive")
    return normalized


def _require_relpath(value: Any, name: str) -> str:
    text = _require_text(value, name, maximum=512)
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts or text.startswith("./"):
        raise ContractError(f"{name} must be a safe repository-relative path")
    return text


def _warn_unknown(value: dict[str, Any], allowed: set[str], name: str,
                  warnings: list[str]) -> None:
    for key in sorted(set(value) - allowed):
        warnings.append(f"{name}.{key} is not used by schema 1")


def _normalize_entrypoint(value: Any, warnings: list[str]) -> dict[str, Any]:
    item = _require_dict(value, "entrypoint")
    _warn_unknown(item, {"relpath", "argv", "environment"}, "entrypoint", warnings)
    argv = item.get("argv", [])
    if not isinstance(argv, list) or len(argv) > 256 \
            or any(not isinstance(arg, str) or len(arg) > 4096 for arg in argv):
        raise ContractError("entrypoint.argv must contain at most 256 literal arguments")
    environment = item.get("environment", {})
    if not isinstance(environment, dict) or len(environment) > 128:
        raise ContractError("entrypoint.environment must be an object with at most 128 keys")
    normalized_environment = {}
    for key, raw in sorted(environment.items()):
        if not isinstance(key, str) or not ENV_KEY.fullmatch(key):
            raise ContractError("entrypoint.environment contains an invalid key")
        if key in RESERVED_ENV_KEYS or key.startswith("CONVIR_EXPERIMENT_ASSISTANT_"):
            raise ContractError(f"entrypoint.environment.{key} is lifecycle-owned")
        if not isinstance(raw, str) or len(raw) > 4096:
            raise ContractError(f"entrypoint.environment.{key} must be bounded text")
        normalized_environment[key] = raw
    return {
        "relpath": _require_relpath(item.get("relpath"), "entrypoint.relpath"),
        "argv": list(argv),
        "environment": normalized_environment,
    }


def _normalize_datasets(value: Any, warnings: list[str]) -> list[dict[str, str]]:
    items = _require_list(value, "datasets", minimum=1)
    if len(items) > 64:
        raise ContractError("datasets may contain at most 64 entries")
    normalized = []
    seen: dict[str, str] = {}
    for index, raw in enumerate(items):
        item = _require_dict(raw, f"datasets[{index}]")
        _warn_unknown(item, {"id", "role"}, f"datasets[{index}]", warnings)
        identifier = _require_token(item.get("id"), f"datasets[{index}].id")
        role = item.get("role")
        if role not in DATA_ROLES:
            raise ContractError(
                f"datasets[{index}].role must be one of {sorted(DATA_ROLES)}"
            )
        previous = seen.get(identifier)
        if previous is not None and previous != role:
            raise ContractError(
                f"dataset {identifier} cannot be used as both {previous} and {role}"
            )
        if previous is not None:
            warnings.append(f"duplicate dataset declaration ignored: {identifier}/{role}")
            continue
        seen[identifier] = role
        normalized.append({"id": identifier, "role": role})
    return sorted(normalized, key=lambda item: (item["role"], item["id"]))


def _normalize_budget(value: Any, warnings: list[str]) -> dict[str, Any]:
    item = _require_dict(value, "budget")
    _warn_unknown(item, {"max_wall_seconds", "parameters"}, "budget", warnings)
    parameters = item.get("parameters", {})
    if not isinstance(parameters, dict) or len(parameters) > 128:
        raise ContractError("budget.parameters must be an object with at most 128 keys")
    try:
        canonical_json_bytes(parameters)
    except (TypeError, ValueError) as exc:
        raise ContractError("budget.parameters must contain JSON values") from exc
    return {
        "max_wall_seconds": _require_int(
            item.get("max_wall_seconds"), "budget.max_wall_seconds", 1, 30 * 24 * 3600,
        ),
        "parameters": parameters,
    }


def _normalize_metric(value: Any, warnings: list[str]) -> dict[str, Any]:
    item = _require_dict(value, "evaluation.primary_metric")
    _warn_unknown(
        item, {"id", "direction", "threshold"}, "evaluation.primary_metric", warnings,
    )
    direction = item.get("direction")
    if direction not in METRIC_DIRECTIONS:
        raise ContractError(
            f"evaluation.primary_metric.direction must be one of {sorted(METRIC_DIRECTIONS)}"
        )
    threshold = item.get("threshold")
    return {
        "id": _require_token(item.get("id"), "evaluation.primary_metric.id"),
        "direction": direction,
        "threshold": None if threshold is None else _require_number(
            threshold, "evaluation.primary_metric.threshold",
        ),
    }


def _normalize_precision(value: Any, warnings: list[str]) -> dict[str, Any] | None:
    if value is None:
        return None
    item = _require_dict(value, "evaluation.precision")
    _warn_unknown(
        item,
        {"target_half_width", "available_independent_units", "required_independent_units"},
        "evaluation.precision", warnings,
    )
    return {
        "target_half_width": _require_number(
            item.get("target_half_width"), "evaluation.precision.target_half_width",
            positive=True,
        ),
        "available_independent_units": _require_int(
            item.get("available_independent_units"),
            "evaluation.precision.available_independent_units", 1, 100_000_000,
        ),
        "required_independent_units": _require_int(
            item.get("required_independent_units"),
            "evaluation.precision.required_independent_units", 1, 100_000_000,
        ),
    }


def _normalize_evaluation(value: Any, warnings: list[str]) -> dict[str, Any]:
    item = _require_dict(value, "evaluation")
    _warn_unknown(
        item, {"primary_metric", "result_files", "precision"}, "evaluation", warnings,
    )
    files = _require_list(item.get("result_files"), "evaluation.result_files", minimum=1)
    if len(files) > 64:
        raise ContractError("evaluation.result_files may contain at most 64 entries")
    result_files = []
    for index, path in enumerate(files):
        normalized = _require_relpath(path, f"evaluation.result_files[{index}]")
        if normalized in result_files:
            warnings.append(f"duplicate result file ignored: {normalized}")
            continue
        result_files.append(normalized)
    return {
        "primary_metric": _normalize_metric(item.get("primary_metric"), warnings),
        "result_files": result_files,
        "precision": _normalize_precision(item.get("precision"), warnings),
    }


def validate_contract(value: Any) -> dict[str, Any]:
    """Normalize one short human-authored contract and report nonblocking warnings."""
    source = _require_dict(value, "contract")
    if source.get("schema_version") != CONTRACT_SCHEMA_VERSION:
        raise ContractError(f"contract.schema_version must be {CONTRACT_SCHEMA_VERSION}")
    warnings: list[str] = []
    _warn_unknown(source, {
        "schema_version", "experiment_id", "objective", "entrypoint", "datasets",
        "budget", "evaluation", "protected_access", "notes", "metadata",
    }, "contract", warnings)
    protected = source.get("protected_access", [])
    if not isinstance(protected, list) or any(role not in PROTECTED_DATA_ROLES for role in protected):
        raise ContractError(
            f"protected_access must contain only {sorted(PROTECTED_DATA_ROLES)}"
        )
    metadata = source.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ContractError("metadata must be an object")
    try:
        canonical_json_bytes(metadata)
    except (TypeError, ValueError) as exc:
        raise ContractError("metadata must contain JSON values") from exc
    notes = source.get("notes")
    normalized = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "experiment_id": _require_token(source.get("experiment_id"), "experiment_id"),
        "objective": _require_text(source.get("objective"), "objective", minimum=8),
        "entrypoint": _normalize_entrypoint(source.get("entrypoint"), warnings),
        "datasets": _normalize_datasets(source.get("datasets"), warnings),
        "budget": _normalize_budget(source.get("budget"), warnings),
        "evaluation": _normalize_evaluation(source.get("evaluation"), warnings),
        "protected_access": sorted(set(protected)),
        "notes": None if notes is None else _require_text(notes, "notes", maximum=8192),
        "metadata": metadata,
    }
    return {
        "contract": normalized,
        "contract_sha256": canonical_sha256(normalized),
        "warnings": warnings,
    }


def required_capabilities(contract: dict[str, Any]) -> list[str]:
    capabilities = {
        "content_addressed_source_snapshot", "lifecycle", "automatic_result_archive",
        "experiment_record_read",
    }
    if contract["protected_access"]:
        capabilities.add("explicit_protected_data_access")
    if contract["evaluation"]["precision"] is not None:
        capabilities.add("declared_precision_gate")
    return sorted(capabilities)


def assess_launch(validated: dict[str, Any], available_capabilities: set[str]) -> dict[str, Any]:
    """Return only blockers that affect this experiment and separate warnings."""
    contract = validated["contract"]
    blockers = []
    warnings = list(validated.get("warnings", []))
    roles = {item["role"] for item in contract["datasets"]}
    missing_permissions = sorted((roles & PROTECTED_DATA_ROLES) - set(contract["protected_access"]))
    if missing_permissions:
        blockers.append(
            "protected data requires explicit access in this contract: "
            + ", ".join(missing_permissions)
        )
    precision = contract["evaluation"]["precision"]
    if precision is not None and precision["available_independent_units"] \
            < precision["required_independent_units"]:
        blockers.append(
            "declared precision is infeasible: available_independent_units "
            f"{precision['available_independent_units']} < required_independent_units "
            f"{precision['required_independent_units']}"
        )
    missing_capabilities = sorted(
        set(required_capabilities(contract)) - set(available_capabilities)
    )
    if missing_capabilities:
        blockers.append(
            "server lacks capabilities required by this experiment: "
            + ", ".join(missing_capabilities)
        )
    if contract["evaluation"]["primary_metric"]["threshold"] is None:
        warnings.append("no decision threshold declared; result will be descriptive")
    if precision is None:
        warnings.append("no precision claim declared; evidence scope will be recorded as observed")
    return {
        "ok": not blockers,
        "blockers": blockers,
        "warnings": warnings,
        "required_capabilities": required_capabilities(contract),
    }


def scientific_kernel(contract: dict[str, Any]) -> dict[str, Any]:
    """Fields whose change alters the experiment question or evidence meaning."""
    return {
        "objective": contract["objective"],
        "datasets": contract["datasets"],
        "primary_metric": contract["evaluation"]["primary_metric"],
        "precision": contract["evaluation"]["precision"],
    }


def classify_contract_revision(original: dict[str, Any], revised: dict[str, Any]) -> dict[str, Any]:
    changed = []
    before = scientific_kernel(original)
    after = scientific_kernel(revised)
    for key in before:
        if before[key] != after[key]:
            changed.append(key)
    warnings = []
    if original["budget"] != revised["budget"]:
        warnings.append("training budget changed within the same experiment and will be recorded")
    if original["entrypoint"] != revised["entrypoint"]:
        warnings.append("entrypoint changed within the same experiment and requires a new source snapshot")
    if original["protected_access"] != revised["protected_access"]:
        warnings.append("protected-data access changed and must be explicitly recorded for the next attempt")
    return {
        "same_experiment": not changed,
        "new_experiment_reasons": changed,
        "warnings": warnings,
    }


def validate_source_snapshot(value: Any, *, require_recoverable: bool) -> dict[str, Any]:
    item = _require_dict(value, "source_snapshot")
    sha256 = item.get("sha256")
    if not isinstance(sha256, str) or not SHA256.fullmatch(sha256):
        raise ContractError("source_snapshot.sha256 must be a SHA-256 digest")
    storage = item.get("storage")
    if storage not in {"cloud_full", "hash_only"}:
        raise ContractError("source_snapshot.storage must be cloud_full or hash_only")
    if require_recoverable and storage != "cloud_full":
        raise ContractError("a result-bearing attempt requires a recoverable cloud_full source snapshot")
    base_commit = item.get("base_commit")
    if base_commit is not None and (
            not isinstance(base_commit, str) or not SHA40.fullmatch(base_commit)):
        raise ContractError("source_snapshot.base_commit must be a Git commit when present")
    diff_sha256 = item.get("diff_sha256")
    if diff_sha256 is not None and (
            not isinstance(diff_sha256, str) or not SHA256.fullmatch(diff_sha256)):
        raise ContractError("source_snapshot.diff_sha256 must be a SHA-256 digest when present")
    return {
        "sha256": sha256,
        "storage": storage,
        "base_commit": base_commit,
        "diff_sha256": diff_sha256,
    }


def authorize_attempt(previous_attempts: list[dict[str, Any]], *, automatic_repair: bool,
                      operator_confirmed: bool = False) -> dict[str, Any]:
    if previous_attempts:
        last_state = previous_attempts[-1].get("state")
        if last_state in {"PREPARED", "RUNNING", "UNKNOWN"}:
            return {
                "ok": False,
                "blocker": f"previous attempt state {last_state} forbids another launch",
                "automatic_repairs_used": sum(
                    bool(item.get("automatic_repair")) for item in previous_attempts
                ),
            }
    used = sum(bool(item.get("automatic_repair")) for item in previous_attempts)
    if automatic_repair and used >= MAX_AUTOMATIC_REPAIRS and not operator_confirmed:
        return {
            "ok": False,
            "blocker": "two automatic repairs have been used; operator confirmation is required",
            "automatic_repairs_used": used,
        }
    return {"ok": True, "blocker": None, "automatic_repairs_used": used}


def validate_attempt(value: Any) -> dict[str, Any]:
    item = _require_dict(value, "attempt")
    required = {
        "schema_version", "experiment_id", "attempt_number", "contract_sha256",
        "source_snapshot", "budget", "state", "automatic_repair", "started_at",
        "ended_at", "error_summary", "result", "cloud_run_ref",
    }
    if set(item) != required or item.get("schema_version") != ATTEMPT_SCHEMA_VERSION:
        raise ContractError("attempt has an invalid field contract")
    state = item.get("state")
    if state not in ATTEMPT_STATES:
        raise ContractError(f"attempt.state must be one of {sorted(ATTEMPT_STATES)}")
    digest = item.get("contract_sha256")
    if not isinstance(digest, str) or not SHA256.fullmatch(digest):
        raise ContractError("attempt.contract_sha256 must be a SHA-256 digest")
    automatic = item.get("automatic_repair")
    if not isinstance(automatic, bool):
        raise ContractError("attempt.automatic_repair must be boolean")
    result = item.get("result")
    if state in RESULT_STATES and not isinstance(result, dict):
        raise ContractError("a completed result state requires attempt.result")
    if state not in RESULT_STATES and result is not None:
        raise ContractError("a non-result attempt cannot publish attempt.result")
    error = item.get("error_summary")
    if state == "FAILED_ENGINEERING" and not isinstance(error, str):
        raise ContractError("FAILED_ENGINEERING requires an error_summary")
    if error is not None:
        error = _require_text(error, "attempt.error_summary", maximum=4096)
    snapshot = validate_source_snapshot(
        item.get("source_snapshot"), require_recoverable=state in RESULT_STATES,
    )
    return {
        "schema_version": ATTEMPT_SCHEMA_VERSION,
        "experiment_id": _require_token(item.get("experiment_id"), "attempt.experiment_id"),
        "attempt_number": _require_int(
            item.get("attempt_number"), "attempt.attempt_number", 1, 10_000,
        ),
        "contract_sha256": digest,
        "source_snapshot": snapshot,
        "budget": _normalize_budget(item.get("budget"), []),
        "state": state,
        "automatic_repair": automatic,
        "started_at": _require_text(item.get("started_at"), "attempt.started_at", maximum=64),
        "ended_at": None if item.get("ended_at") is None else _require_text(
            item["ended_at"], "attempt.ended_at", maximum=64,
        ),
        "error_summary": error,
        "result": result,
        "cloud_run_ref": _require_text(
            item.get("cloud_run_ref"), "attempt.cloud_run_ref", maximum=512,
        ),
    }


def should_archive_attempt(attempt: dict[str, Any]) -> bool:
    return attempt.get("state") in RESULT_STATES and isinstance(attempt.get("result"), dict)


def build_archive_record(validated_contract: dict[str, Any], attempts: list[dict[str, Any]],
                         *, recorded_at: str) -> dict[str, Any]:
    if not attempts:
        raise ContractError("archive requires at least one attempt")
    normalized_attempts = [validate_attempt(item) for item in attempts]
    experiment_id = validated_contract["contract"]["experiment_id"]
    if any(item["experiment_id"] != experiment_id for item in normalized_attempts):
        raise ContractError("archive attempts do not match the experiment id")
    if any(item["attempt_number"] != index for index, item in enumerate(normalized_attempts, 1)):
        raise ContractError("archive attempt numbers must be contiguous from one")
    final = normalized_attempts[-1]
    if not should_archive_attempt(final):
        raise ContractError("only a completed result-bearing attempt is automatically archived")
    if final["contract_sha256"] != validated_contract["contract_sha256"]:
        raise ContractError("final attempt contract identity mismatch")
    record = {
        "schema_version": ARCHIVE_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "recorded_at": _require_text(recorded_at, "recorded_at", maximum=64),
        "contract": validated_contract["contract"],
        "contract_sha256": validated_contract["contract_sha256"],
        "attempts": normalized_attempts,
        "terminal": {
            "state": final["state"],
            "attempt_number": final["attempt_number"],
        },
        "result": final["result"],
        "source_snapshot": final["source_snapshot"],
        "cloud_run_ref": final["cloud_run_ref"],
    }
    return {"record": record, "record_sha256": canonical_sha256(record)}


PUBLIC_TOOL_SCHEMAS = {
    "experiment_start": {
        "required": ["local_repo", "contract"],
        "properties": {"local_repo": "string", "contract": "object"},
    },
    "experiment_status": {
        "required": ["experiment_id"],
        "properties": {"experiment_id": "string"},
    },
    "experiment_cancel": {
        "required": ["experiment_id"],
        "properties": {"experiment_id": "string"},
    },
    "experiment_repair": {
        "required": ["experiment_id"],
        "properties": {
            "experiment_id": "string", "contract": "object",
            "operator_confirmed": "boolean",
        },
    },
    "experiment_get": {
        "required": ["experiment_id"],
        "properties": {"experiment_id": "string", "view": "summary|full"},
    },
    "experiment_search": {
        "required": [],
        "properties": {
            "query": "string", "states": "array", "limit": "integer",
            "compare_experiment_ids": "array",
        },
    },
}
