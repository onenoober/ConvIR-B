#!/usr/bin/env python3
"""Compile one human-authored experiment spec into deterministic route files.

The compiler removes duplicated identities and paths. It does not invent or
select scientific questions, variables, data roles, controls, gates, terminal
decisions, models, assets, or thresholds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable

import convir_ops_mcp as ops
import research_program_contract as program_contract
import scientific_contract as science_contract
from route_runtime_contract import (
    ContractError,
    MODEL_CAPABILITY_DIRECTORY,
    PRECISION_CERTIFICATE_DIRECTORY,
    RUNTIME_SPEC_DIRECTORY,
    capability_input_contract_sha256,
    repository_asset_identity_errors,
    validate_asset_manifest,
    validate_model_capability,
    validate_precision_certificate,
    validate_runtime_spec,
)


class ExperimentSpecError(RuntimeError):
    pass


SPEC_DIRECTORY = "experience_docx/experiment_specs"
PROGRAM_DIRECTORY = "experience_docx/research_programs"
SCIENTIFIC_DIRECTORY = "experience_docx/scientific_contracts"
ASSET_DIRECTORY = "experience_docx/route_assets"
CARD_DIRECTORY = "experience_docx/experiment_cards"
MANIFEST_RELPATH = "experience_docx/route_operations.json"
SHA40 = __import__("re").compile(r"^[0-9a-f]{40}$")
SOURCE_FIELDS = {
    "schema_version", "route_id", "rules_commit", "title", "rationale",
    "first_operation", "program_contract_relpath", "operations",
}
OPERATION_SOURCE_FIELDS = {
    "operation", "program_authorization", "scientific_contract", "runtime",
    "assets", "capability", "precision",
}
SCIENTIFIC_SOURCE_FIELDS_V1 = {
    "question", "population", "intervention", "primary_estimand", "controls",
    "uncertainty", "gates", "competing_explanation", "terminal_mapping",
    "disabled_actions",
}
SCIENTIFIC_SOURCE_FIELDS_V2 = {
    "question", "population", "intervention", "primary_estimand", "controls",
    "uncertainty", "gates", "competing_explanation", "decision_table",
    "disabled_actions",
}
SOURCE_OPERATION_FIELDS_V2 = {
    "runner_relpath", "mode", "require_gpu", "output_id",
    "closeout_filename", "prior_closeout_relpath", "prior_terminal_tuple",
    "workspace_policy", "output_policy", "monitor_profile",
    "heartbeat_timeout_seconds", "min_free_gpu_mib", "max_gpu_utilization_pct",
}
GENERIC_ENGINEERING_TERMINAL = {
    "state": "FAILED_ENGINEERING", "decision": None, "authorizes": "NONE",
}
RUNTIME_SOURCE_FIELDS = {
    "entrypoint_relpath", "timeout_seconds", "expected_wall_seconds", "total_units",
    "evidence_role", "resume_policy", "protected_data_permissions", "environment",
    "evidence_files", "engineering_contract", "precision_contract",
}
ENGINEERING_SOURCE_FIELDS = {"mode", "max_seconds", "cost_contract"}
PRECISION_CONTRACT_SOURCE_FIELDS = {"mode", "rationale"}


def json_bytes(value: Any) -> bytes:
    return (json.dumps(
        value, sort_keys=True, indent=2, ensure_ascii=False,
    ) + "\n").encode("utf-8")


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _object(value: Any, fields: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ExperimentSpecError(f"{name} must contain exactly {sorted(fields)}")
    return value


def _token(value: Any, name: str) -> str:
    try:
        return ops.require_token(value, name)
    except ops.ToolError as exc:
        raise ExperimentSpecError(str(exc)) from exc


def _text(value: Any, name: str, minimum: int, maximum: int) -> str:
    if not isinstance(value, str) or not minimum <= len(value.strip()) <= maximum:
        raise ExperimentSpecError(f"{name} must contain {minimum}-{maximum} characters")
    return value.strip()


def _relpath(value: Any, name: str, directory: str) -> str:
    if not isinstance(value, str):
        raise ExperimentSpecError(f"{name} must be a repository-relative JSON path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not value.startswith(directory + "/") \
            or not value.endswith(".json"):
        raise ExperimentSpecError(f"{name} must stay below {directory}/ and end in .json")
    return value


def expected_spec_relpath(route_id: str) -> str:
    return f"{SPEC_DIRECTORY}/{_token(route_id, 'route_id')}.json"


def _scientific_path(route_id: str, operation_id: str) -> str:
    return f"{SCIENTIFIC_DIRECTORY}/{route_id}__{operation_id}.json"


def _capability_path(route_id: str, operation_id: str) -> str:
    return f"{MODEL_CAPABILITY_DIRECTORY}/{route_id}__{operation_id}.json"


def _precision_path(route_id: str, operation_id: str) -> str:
    return f"{PRECISION_CERTIFICATE_DIRECTORY}/{route_id}__{operation_id}.json"


def _route_invariant(claim: dict[str, Any]) -> bytes:
    fields = {
        key: claim[key] for key in (
            "program_id", "family_id", "mechanism_type", "adjacent_sequence",
            "orthogonal_changes", "reopen_evidence",
        )
    }
    return program_contract.canonical_bytes(fields)


def _route_card(spec: dict[str, Any]) -> bytes:
    text = "\n".join([
        f"# {spec['title']}",
        "",
        "Status: PLANNED",
        "",
        f"- Route id: {spec['route_id']}",
        f"- First operation: {spec['first_operation']}",
        f"- Program contract: {spec['program_contract_relpath']}",
        f"- Experiment spec: {expected_spec_relpath(spec['route_id'])}",
        f"- Scientific contracts: {SCIENTIFIC_DIRECTORY}/",
        "",
        "## Scientific rationale",
        "",
        spec["rationale"],
        "",
    ])
    raw = text.encode("utf-8")
    if len(raw) > 8192:
        raise ExperimentSpecError("generated route card exceeds 8 KiB")
    return raw


def _scientific_source_fields(spec_schema: int) -> set[str]:
    return (
        SCIENTIFIC_SOURCE_FIELDS_V2
        if spec_schema == 2 else SCIENTIFIC_SOURCE_FIELDS_V1
    )


def _compile_operation_v2(
    value: Any, scientific: dict[str, Any], name: str,
) -> dict[str, Any]:
    operation = _object(value, SOURCE_OPERATION_FIELDS_V2, name)
    return {
        **operation,
        "allowed_terminal_tuples": [
            *science_contract.scientific_terminal_tuples(scientific),
            dict(GENERIC_ENGINEERING_TERMINAL),
        ],
    }


def _lint_error(path: str, code: str, message: Any) -> dict[str, str]:
    safe_message = " ".join(str(message).split())[:1024]
    return {"path": path, "code": code, "message": safe_message}


def _lint_code(message: str) -> str:
    lowered = message.lower()
    if "role" in lowered and ("differ" in lowered or "requires" in lowered):
        return "ROLE_ALIGNMENT"
    if "permission" in lowered:
        return "PERMISSION_ALIGNMENT"
    if "field contract" in lowered or "contain exactly" in lowered:
        return "INVALID_FIELDS"
    if "schema" in lowered:
        return "INVALID_SCHEMA"
    if "path" in lowered or "relpath" in lowered:
        return "INVALID_PATH"
    if "token" in lowered or "safe token" in lowered:
        return "INVALID_TOKEN"
    if "precision" in lowered:
        return "PRECISION_CONTRACT_INVALID"
    if "cost" in lowered or "engineering" in lowered:
        return "ENGINEERING_CONTRACT_INVALID"
    return "CONTRACT_INVALID"


def _append_lint(errors: list[dict[str, str]], path: str, exc: Any,
                 code: str | None = None) -> None:
    errors.append(_lint_error(path, code or _lint_code(str(exc)), exc))


def _source_text_hygiene_errors(raw: bytes, path: str) -> list[dict[str, str]]:
    errors = []
    if raw.startswith(b"\xef\xbb\xbf"):
        errors.append(_lint_error(path, "SOURCE_TEXT_BOM", "UTF-8 BOM is forbidden"))
    if b"\r" in raw:
        errors.append(_lint_error(path, "SOURCE_TEXT_NEWLINE", "use LF newlines only"))
    if not raw.endswith(b"\n"):
        errors.append(_lint_error(
            path, "SOURCE_TEXT_FINAL_NEWLINE", "exactly one final newline is required",
        ))
    elif raw.endswith(b"\n\n"):
        errors.append(_lint_error(
            path, "SOURCE_TEXT_TRAILING_BLANK_LINE",
            "trailing blank lines are forbidden",
        ))
    if any(line.endswith((b" ", b"\t")) for line in raw.splitlines()):
        errors.append(_lint_error(
            path, "SOURCE_TEXT_TRAILING_WHITESPACE",
            "trailing spaces or tabs are forbidden",
        ))
    return errors


def _capability_with_derived_input_identity(
    value: Any, *, spec_schema: int, runtime: dict[str, Any],
) -> Any:
    """Fill only the mechanically determined schema-2 input identity."""
    if spec_schema != 2 or not isinstance(value, dict):
        return value
    identity = value.get("reuse_identity")
    if not isinstance(identity, dict) or "input_contract_sha256" in identity:
        return value
    required = {"contract_mode", "minimum_fixture", "compatibility_imports"}
    if not required <= set(value):
        return value
    return {
        **value,
        "reuse_identity": {
            **identity,
            "input_contract_sha256": capability_input_contract_sha256(
                contract_mode=value["contract_mode"],
                minimum_fixture=value["minimum_fixture"],
                compatibility_imports=value["compatibility_imports"],
                cost_contract=runtime["engineering_contract"].get("cost_contract"),
            ),
        },
    }


def _lint_operation_components(
    *, errors: list[dict[str, str]], route_id: str, operation_id: str,
    item: dict[str, Any], effective_program: dict[str, Any] | None,
    evidence_exists: Callable[[str], bool] | None,
    read_repo_file: Callable[[str], bytes] | None, spec_schema: int,
) -> None:
    """Aggregate independent component errors for one source operation."""
    prefix = f"operations.{operation_id}"
    operation_source = item["operation"]
    operation = operation_source
    if not isinstance(operation_source, dict):
        _append_lint(errors, f"{prefix}.operation", "must be an object", "INVALID_TYPE")
        operation = None

    claim = item["program_authorization"]
    if effective_program is not None:
        try:
            program_contract.validate_route_authorization(
                effective_program, claim, evidence_exists=evidence_exists,
            )
        except (program_contract.ProgramContractError, KeyError, TypeError, ValueError) as exc:
            _append_lint(errors, f"{prefix}.program_authorization", exc, "PROGRAM_AUTHORIZATION_INVALID")

    scientific = None
    try:
        scientific_source = _object(
            item["scientific_contract"], _scientific_source_fields(spec_schema),
            f"{prefix}.scientific_contract",
        )
        scientific_value = {
            "schema_version": spec_schema, "route_id": route_id,
            "operation_id": operation_id, **scientific_source,
        }
        if spec_schema == 2:
            scientific = science_contract.validate_scientific_contract_v2(
                scientific_value, route_id, operation_id,
            )
            if operation is not None:
                operation = _compile_operation_v2(
                    operation_source, scientific, f"{prefix}.operation",
                )
        elif operation is not None:
            scientific = ops.validate_scientific_contract(
                scientific_value,
                route_id, operation_id, operation,
            )
    except (
        ExperimentSpecError, ops.ToolError, science_contract.ScientificContractError,
        KeyError, TypeError, ValueError,
    ) as exc:
        _append_lint(errors, f"{prefix}.scientific_contract", exc, "SCIENTIFIC_CONTRACT_INVALID")
        if spec_schema == 2:
            operation = None

    runtime_source = None
    engineering = None
    precision_contract_source = None
    try:
        runtime_source = _object(
            item["runtime"], RUNTIME_SOURCE_FIELDS, f"{prefix}.runtime",
        )
    except (ExperimentSpecError, KeyError, TypeError, ValueError) as exc:
        _append_lint(errors, f"{prefix}.runtime", exc, "RUNTIME_CONTRACT_INVALID")
    if runtime_source is not None:
        try:
            engineering = _object(
                runtime_source["engineering_contract"], ENGINEERING_SOURCE_FIELDS,
                f"{prefix}.runtime.engineering_contract",
            )
            if engineering["mode"] != "metadata_only" and engineering["cost_contract"] is None:
                raise ExperimentSpecError("cost_contract is required for non-metadata authoring")
        except (ExperimentSpecError, KeyError, TypeError, ValueError) as exc:
            _append_lint(
                errors, f"{prefix}.runtime.engineering_contract", exc,
                "ENGINEERING_CONTRACT_INVALID",
            )
            engineering = None
        try:
            precision_contract_source = _object(
                runtime_source["precision_contract"], PRECISION_CONTRACT_SOURCE_FIELDS,
                f"{prefix}.runtime.precision_contract",
            )
        except (ExperimentSpecError, KeyError, TypeError, ValueError) as exc:
            _append_lint(
                errors, f"{prefix}.runtime.precision_contract", exc,
                "PRECISION_CONTRACT_INVALID",
            )
            precision_contract_source = None

    validated_runtime = None
    asset = None
    precision = None
    if operation is not None and runtime_source is not None \
            and engineering is not None and precision_contract_source is not None:
        asset_path = None if item["assets"] is None \
            else f"{ASSET_DIRECTORY}/{route_id}__{operation_id}.json"
        capability_path = None if item["capability"] is None \
            else _capability_path(route_id, operation_id)
        precision_path = None if item["precision"] is None \
            else _precision_path(route_id, operation_id)
        runtime = {
            "schema_version": 2, "route_id": route_id, "operation_id": operation_id,
            **{
                key: value for key, value in runtime_source.items()
                if key not in {"engineering_contract", "precision_contract"}
            },
            "asset_manifest_relpath": asset_path,
            "engineering_contract": {
                **engineering, "capability_profile_relpath": capability_path,
            },
            "precision_contract": {
                **precision_contract_source, "certificate_relpath": precision_path,
            },
        }
        try:
            validated_runtime = validate_runtime_spec(
                runtime, {"route_id": route_id, "operations": {operation_id: operation}},
                operation_id,
            )
        except (ContractError, KeyError, TypeError, ValueError) as exc:
            _append_lint(errors, f"{prefix}.runtime", exc, "RUNTIME_CONTRACT_INVALID")

    if validated_runtime is not None:
        if item["assets"] is not None:
            try:
                asset = validate_asset_manifest({
                    "schema_version": spec_schema, "route_id": route_id,
                    "operation_id": operation_id, "assets": item["assets"],
                }, validated_runtime)
                if read_repo_file is not None:
                    for message in repository_asset_identity_errors(asset, read_repo_file):
                        _append_lint(
                            errors, f"{prefix}.assets", message,
                            "REPO_ASSET_IDENTITY_MISMATCH",
                        )
            except (ContractError, KeyError, TypeError, ValueError) as exc:
                _append_lint(errors, f"{prefix}.assets", exc, "ASSET_CONTRACT_INVALID")
        if item["capability"] is not None and (item["assets"] is None or asset is not None):
            try:
                capability_source = _capability_with_derived_input_identity(
                    item["capability"], spec_schema=spec_schema,
                    runtime=validated_runtime,
                )
                validate_model_capability(
                    {"schema_version": spec_schema, **capability_source},
                    validated_runtime, asset,
                )
            except (ContractError, KeyError, TypeError, ValueError) as exc:
                _append_lint(errors, f"{prefix}.capability", exc, "CAPABILITY_CONTRACT_INVALID")
        if item["precision"] is not None:
            try:
                precision = validate_precision_certificate(
                    {"schema_version": spec_schema, **item["precision"]},
                    validated_runtime, scientific,
                )
            except (ContractError, KeyError, TypeError, ValueError) as exc:
                _append_lint(errors, f"{prefix}.precision", exc, "PRECISION_CONTRACT_INVALID")
        if scientific is not None:
            try:
                ops.validate_contract_runtime_alignment(
                    scientific, validated_runtime, precision,
                )
            except (ops.ToolError, KeyError, TypeError, ValueError) as exc:
                _append_lint(errors, f"{prefix}.alignment", exc)
        if isinstance(claim, dict) and scientific is not None:
            try:
                permissions = claim["protected_permissions"]
                if claim["evidence_role"] != validated_runtime["evidence_role"] \
                        or claim["evidence_role"] != scientific["population"]["evidence_role"]:
                    raise ExperimentSpecError("program/scientific/runtime roles differ")
                if permissions != validated_runtime["protected_data_permissions"] \
                        or permissions != {
                            key: scientific["population"][key] for key in permissions
                        }:
                    raise ExperimentSpecError("program/scientific/runtime permissions differ")
            except (ExperimentSpecError, KeyError, TypeError, ValueError) as exc:
                _append_lint(errors, f"{prefix}.alignment", exc)


def lint_bundle(*, spec_relpath: str, spec_raw: bytes, program_raw: bytes,
                evidence_exists: Callable[[str], bool] | None = None,
                read_repo_file: Callable[[str], bytes] | None = None) -> dict[str, Any]:
    """Return stable, aggregate authoring diagnostics without writing files."""
    errors: list[dict[str, str]] = []
    try:
        source = json.loads(spec_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {
            "status": "EXPERIMENT_SPEC_INVALID",
            "errors": [_lint_error("experiment_spec", "INVALID_JSON", exc)],
        }
    try:
        program_source = json.loads(program_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {
            "status": "EXPERIMENT_SPEC_INVALID",
            "errors": [_lint_error("program_contract", "INVALID_JSON", exc)],
        }
    errors.extend(_source_text_hygiene_errors(spec_raw, "experiment_spec.source_text"))
    errors.extend(_source_text_hygiene_errors(program_raw, "program_contract.source_text"))
    if not isinstance(source, dict):
        return {
            "status": "EXPERIMENT_SPEC_INVALID",
            "errors": [_lint_error("experiment_spec", "INVALID_TYPE", "must be an object")],
        }
    missing = sorted(SOURCE_FIELDS - set(source))
    unexpected = sorted(set(source) - SOURCE_FIELDS)
    if missing:
        errors.append(_lint_error("experiment_spec", "MISSING_FIELDS", f"missing fields: {missing}"))
    if unexpected:
        errors.append(_lint_error("experiment_spec", "UNEXPECTED_FIELDS", f"unexpected fields: {unexpected}"))
    checks = (
        ("schema_version", lambda: source.get("schema_version") in {1, 2}
         or (_ for _ in ()).throw(ExperimentSpecError("must equal 1 or 2"))),
        ("route_id", lambda: _token(source.get("route_id"), "route_id")),
        ("rules_commit", lambda: isinstance(source.get("rules_commit"), str)
         and SHA40.fullmatch(source["rules_commit"])
         or (_ for _ in ()).throw(ExperimentSpecError("must be a 40-character Git SHA"))),
        ("title", lambda: _text(source.get("title"), "title", 4, 160)),
        ("rationale", lambda: _text(source.get("rationale"), "rationale", 16, 4096)),
        ("first_operation", lambda: _token(source.get("first_operation"), "first_operation")),
        ("program_contract_relpath", lambda: _relpath(
            source.get("program_contract_relpath"), "program_contract_relpath", PROGRAM_DIRECTORY,
        )),
    )
    for path, check in checks:
        try:
            check()
        except (ExperimentSpecError, TypeError) as exc:
            errors.append(_lint_error(path, _lint_code(str(exc)), exc))
    route_id = source.get("route_id")
    if isinstance(route_id, str) and ops.SAFE_TOKEN.fullmatch(route_id):
        try:
            if spec_relpath != expected_spec_relpath(route_id):
                raise ExperimentSpecError("path does not match route_id")
        except ExperimentSpecError as exc:
            errors.append(_lint_error("experiment_spec_relpath", "IDENTITY_MISMATCH", exc))
    try:
        effective_program = program_contract.validate_program_contract(
            program_source, evidence_exists=evidence_exists,
        )
        program_ok = True
    except program_contract.ProgramContractError as exc:
        program_ok = False
        effective_program = None
        errors.append(_lint_error("program_contract", "PROGRAM_CONTRACT_INVALID", exc))
    operations = source.get("operations")
    if not isinstance(operations, dict) or not operations:
        errors.append(_lint_error("operations", "INVALID_TYPE", "must be a non-empty object"))
        operations = {}
    first = source.get("first_operation")
    if operations and first not in operations:
        errors.append(_lint_error("first_operation", "IDENTITY_MISMATCH", "is absent from operations"))
    structurally_valid_operations = []
    for operation_id, item in operations.items():
        path = f"operations.{operation_id}"
        try:
            _token(operation_id, "operation_id")
        except ExperimentSpecError as exc:
            errors.append(_lint_error(path, "INVALID_TOKEN", exc))
            continue
        if not isinstance(item, dict):
            errors.append(_lint_error(path, "INVALID_TYPE", "must be an object"))
            continue
        item_missing = sorted(OPERATION_SOURCE_FIELDS - set(item))
        item_unexpected = sorted(set(item) - OPERATION_SOURCE_FIELDS)
        if item_missing:
            errors.append(_lint_error(path, "MISSING_FIELDS", f"missing fields: {item_missing}"))
        if item_unexpected:
            errors.append(_lint_error(path, "UNEXPECTED_FIELDS", f"unexpected fields: {item_unexpected}"))
        if not item_missing and not item_unexpected:
            structurally_valid_operations.append(operation_id)
    top_ready = not any(
        item["path"] in {
            "experiment_spec", "schema_version", "route_id", "rules_commit",
            "title", "rationale", "program_contract_relpath",
            "experiment_spec_relpath",
        }
        for item in errors
    )
    if top_ready:
        for operation_id in structurally_valid_operations:
            _lint_operation_components(
                errors=errors, route_id=route_id, operation_id=operation_id,
                item=operations[operation_id],
                effective_program=effective_program if program_ok else None,
                evidence_exists=evidence_exists,
                read_repo_file=read_repo_file,
                spec_schema=source["schema_version"],
            )
        if not errors and len(structurally_valid_operations) == len(operations):
            try:
                compile_bundle(
                    spec_relpath=spec_relpath, spec_raw=spec_raw,
                    program_raw=program_raw, evidence_exists=evidence_exists,
                )
            except (ExperimentSpecError, KeyError, TypeError, ValueError) as exc:
                errors.append(_lint_error("experiment_spec", _lint_code(str(exc)), exc))
    unique = {
        (item["path"], item["code"], item["message"]): item for item in errors
    }
    ordered = sorted(unique.values(), key=lambda item: (item["path"], item["code"], item["message"]))
    return {
        "status": "EXPERIMENT_SPEC_INVALID" if ordered else "EXPERIMENT_SPEC_VALID",
        "errors": ordered,
    }


def compile_bundle(*, spec_relpath: str, spec_raw: bytes, program_raw: bytes,
                   evidence_exists: Callable[[str], bool] | None = None) -> dict[str, bytes]:
    """Return every generated file as repository-relative path -> exact bytes."""
    try:
        source = json.loads(spec_raw)
        program_source = json.loads(program_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExperimentSpecError(f"source JSON is invalid: {exc}") from exc
    spec = _object(source, SOURCE_FIELDS, "experiment spec")
    if spec["schema_version"] not in {1, 2}:
        raise ExperimentSpecError("experiment spec schema_version must be 1 or 2")
    spec_schema = spec["schema_version"]
    route_id = _token(spec["route_id"], "route_id")
    if spec_relpath != expected_spec_relpath(route_id):
        raise ExperimentSpecError("experiment spec path does not match route_id")
    if not isinstance(spec["rules_commit"], str) or not SHA40.fullmatch(spec["rules_commit"]):
        raise ExperimentSpecError("rules_commit must be a 40-character Git SHA")
    _text(spec["title"], "title", 4, 160)
    _text(spec["rationale"], "rationale", 16, 4096)
    first = _token(spec["first_operation"], "first_operation")
    program_relpath = _relpath(
        spec["program_contract_relpath"], "program_contract_relpath", PROGRAM_DIRECTORY,
    )
    try:
        effective_program = program_contract.validate_program_contract(
            program_source, evidence_exists=evidence_exists,
        )
    except program_contract.ProgramContractError as exc:
        raise ExperimentSpecError(f"program contract is invalid: {exc}") from exc
    operations = spec["operations"]
    if not isinstance(operations, dict) or not operations:
        raise ExperimentSpecError("operations must be a non-empty object")
    if first not in operations:
        raise ExperimentSpecError("first_operation is absent from operations")
    manifest_operations = {}
    scientific_paths = {}
    generated: dict[str, bytes] = {}
    route_invariant = None
    for operation_id, source_operation in operations.items():
        _token(operation_id, "operation_id")
        item = _object(source_operation, OPERATION_SOURCE_FIELDS, f"operations.{operation_id}")
        operation_source = item["operation"]
        if not isinstance(operation_source, dict):
            raise ExperimentSpecError(f"operations.{operation_id}.operation must be an object")
        claim = item["program_authorization"]
        try:
            program_contract.validate_route_authorization(
                effective_program, claim, evidence_exists=evidence_exists,
            )
        except program_contract.ProgramContractError as exc:
            raise ExperimentSpecError(
                f"operations.{operation_id}.program_authorization is invalid: {exc}"
            ) from exc
        invariant = _route_invariant(claim)
        if route_invariant is None:
            route_invariant = invariant
        elif invariant != route_invariant:
            raise ExperimentSpecError("all operations must share one route-level mechanism claim")
        scientific_source = _object(
            item["scientific_contract"], _scientific_source_fields(spec_schema),
            f"operations.{operation_id}.scientific_contract",
        )
        scientific = {
            "schema_version": spec_schema, "route_id": route_id,
            "operation_id": operation_id,
            **scientific_source,
        }
        if spec_schema == 2:
            try:
                scientific = science_contract.validate_scientific_contract_v2(
                    scientific, route_id, operation_id,
                )
            except science_contract.ScientificContractError as exc:
                raise ExperimentSpecError(
                    f"operations.{operation_id} scientific contract is invalid: {exc}"
                ) from exc
            operation = _compile_operation_v2(
                operation_source, scientific, f"operations.{operation_id}.operation",
            )
        else:
            operation = operation_source
        manifest_operations[operation_id] = operation
        scientific_path = _scientific_path(route_id, operation_id)
        scientific_paths[operation_id] = scientific_path
        runtime_source = _object(
            item["runtime"], RUNTIME_SOURCE_FIELDS, f"operations.{operation_id}.runtime",
        )
        engineering = _object(
            runtime_source["engineering_contract"], ENGINEERING_SOURCE_FIELDS,
                               f"operations.{operation_id}.runtime.engineering_contract")
        if engineering["mode"] != "metadata_only" and engineering["cost_contract"] is None:
            raise ExperimentSpecError(
                f"operations.{operation_id}.runtime.engineering_contract.cost_contract is required"
            )
        precision_contract_source = _object(
            runtime_source["precision_contract"], PRECISION_CONTRACT_SOURCE_FIELDS,
            f"operations.{operation_id}.runtime.precision_contract",
        )
        asset_path = None if item["assets"] is None \
            else f"{ASSET_DIRECTORY}/{route_id}__{operation_id}.json"
        capability_path = None if item["capability"] is None else _capability_path(route_id, operation_id)
        precision_path = None if item["precision"] is None else _precision_path(route_id, operation_id)
        runtime = {
            "schema_version": 2, "route_id": route_id, "operation_id": operation_id,
            **{key: value for key, value in runtime_source.items()
               if key not in {"engineering_contract", "precision_contract"}},
            "asset_manifest_relpath": asset_path,
            "engineering_contract": {
                **engineering, "capability_profile_relpath": capability_path,
            },
            "precision_contract": {
                **precision_contract_source, "certificate_relpath": precision_path,
            },
        }
        manifest_preview = {"route_id": route_id, "operations": manifest_operations}
        try:
            validated_runtime = validate_runtime_spec(runtime, manifest_preview, operation_id)
            asset = None
            if item["assets"] is not None:
                asset = validate_asset_manifest({
                    "schema_version": spec_schema, "route_id": route_id,
                    "operation_id": operation_id, "assets": item["assets"],
                }, validated_runtime)
                generated[asset_path] = json_bytes(asset)
            if item["capability"] is not None:
                capability_source = _capability_with_derived_input_identity(
                    item["capability"], spec_schema=spec_schema,
                    runtime=validated_runtime,
                )
                capability = validate_model_capability(
                    {"schema_version": spec_schema, **capability_source},
                    validated_runtime, asset,
                )
                generated[capability_path] = json_bytes(capability)
            precision = None
            if item["precision"] is not None:
                precision = validate_precision_certificate(
                    {"schema_version": spec_schema, **item["precision"]},
                    validated_runtime, scientific,
                )
                generated[precision_path] = json_bytes({
                    key: value for key, value in precision.items()
                    if key != "computed_required_groups"
                })
            validated_scientific = ops.validate_scientific_contract(
                scientific, route_id, operation_id, operation,
            )
            ops.validate_contract_runtime_alignment(
                validated_scientific, validated_runtime, precision,
            )
        except (ContractError, ops.ToolError) as exc:
            raise ExperimentSpecError(f"operations.{operation_id} generated contract is invalid: {exc}") from exc
        permissions = claim["protected_permissions"]
        if claim["evidence_role"] != runtime["evidence_role"] \
                or claim["evidence_role"] != scientific["population"]["evidence_role"]:
            raise ExperimentSpecError(f"operations.{operation_id} program/scientific/runtime roles differ")
        if permissions != runtime["protected_data_permissions"] or permissions != {
            key: scientific["population"][key] for key in permissions
        }:
            raise ExperimentSpecError(f"operations.{operation_id} protected permissions differ")
        generated[scientific_path] = json_bytes(validated_scientific)
        generated[f"{RUNTIME_SPEC_DIRECTORY}/{operation_id}.json"] = json_bytes(runtime)
    manifest = {
        "schema_version": 6,
        "route_id": route_id,
        "rules_commit": spec["rules_commit"],
        "route_card_relpath": f"{CARD_DIRECTORY}/{route_id}.md",
        "scientific_contract_relpaths": scientific_paths,
        "program_contract_relpath": program_relpath,
        "program_contract_sha256": sha256(program_raw),
        "experiment_spec_relpath": spec_relpath,
        "experiment_spec_sha256": sha256(spec_raw),
        "operations": manifest_operations,
    }
    if len(json_bytes(manifest)) > ops.MAX_MANIFEST_BYTES:
        raise ExperimentSpecError("generated route manifest exceeds MCP size limit")
    generated[f"{CARD_DIRECTORY}/{route_id}.md"] = _route_card(spec)
    generated[MANIFEST_RELPATH] = json_bytes(manifest)
    return dict(sorted(generated.items()))


def compare_bundle(bundle: dict[str, bytes], read_bytes: Callable[[str], bytes]) -> list[str]:
    mismatches = []
    for relpath, expected in bundle.items():
        try:
            observed = read_bytes(relpath)
        except (FileNotFoundError, KeyError):
            mismatches.append(f"missing generated file: {relpath}")
            continue
        if observed != expected:
            mismatches.append(f"generated file drift: {relpath}")
    return mismatches


def compile_from_repo(repo: Path, spec_relpath: str) -> dict[str, bytes]:
    repo = repo.resolve()
    spec_path = repo / spec_relpath
    spec_raw = spec_path.read_bytes()
    source = json.loads(spec_raw)
    program_relpath = source.get("program_contract_relpath")
    if not isinstance(program_relpath, str):
        raise ExperimentSpecError("program_contract_relpath is missing")
    return compile_bundle(
        spec_relpath=spec_relpath,
        spec_raw=spec_raw,
        program_raw=(repo / program_relpath).read_bytes(),
        evidence_exists=lambda relpath: (repo / relpath).is_file(),
    )


def lint_from_repo(repo: Path, spec_relpath: str) -> dict[str, Any]:
    repo = repo.resolve()
    try:
        spec_raw = (repo / spec_relpath).read_bytes()
    except OSError as exc:
        return {
            "status": "EXPERIMENT_SPEC_INVALID",
            "errors": [_lint_error("experiment_spec", "READ_FAILED", exc)],
        }
    try:
        source = json.loads(spec_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {
            "status": "EXPERIMENT_SPEC_INVALID",
            "errors": [_lint_error("experiment_spec", "INVALID_JSON", exc)],
        }
    program_relpath = source.get("program_contract_relpath") if isinstance(source, dict) else None
    if not isinstance(program_relpath, str):
        program_raw = b"null\n"
    else:
        try:
            program_raw = (repo / program_relpath).read_bytes()
        except OSError as exc:
            return {
                "status": "EXPERIMENT_SPEC_INVALID",
                "errors": [_lint_error("program_contract_relpath", "READ_FAILED", exc)],
            }
    return lint_bundle(
        spec_relpath=spec_relpath, spec_raw=spec_raw, program_raw=program_raw,
        evidence_exists=lambda relpath: (repo / relpath).is_file(),
        read_repo_file=lambda relpath: (repo / relpath).read_bytes(),
    )


def write_bundle_atomic(repo: Path, bundle: dict[str, bytes]) -> None:
    """Install a complete derived bundle and restore prior bytes on failure."""
    repo = repo.resolve()
    previous: dict[str, bytes | None] = {}
    installed: list[str] = []
    with tempfile.TemporaryDirectory(prefix=".experiment-spec-finalize-", dir=repo) as raw_tmp:
        temporary = Path(raw_tmp)
        staged = temporary / "staged"
        rollback = temporary / "rollback"
        for relpath, raw in bundle.items():
            candidate = staged / relpath
            candidate.parent.mkdir(parents=True, exist_ok=True)
            candidate.write_bytes(raw)
            destination = repo / relpath
            previous[relpath] = destination.read_bytes() if destination.is_file() else None
        try:
            for relpath in bundle:
                destination = repo / relpath
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(staged / relpath, destination)
                installed.append(relpath)
        except OSError as exc:
            rollback_errors = []
            for relpath in reversed(installed):
                destination = repo / relpath
                prior = previous[relpath]
                try:
                    if prior is None:
                        destination.unlink(missing_ok=True)
                    else:
                        restored = rollback / relpath
                        restored.parent.mkdir(parents=True, exist_ok=True)
                        restored.write_bytes(prior)
                        os.replace(restored, destination)
                except OSError as rollback_exc:
                    rollback_errors.append(f"{relpath}: {rollback_exc}")
            if rollback_errors:
                raise ExperimentSpecError(
                    "bundle install and rollback failed: " + "; ".join(rollback_errors)
                ) from exc
            raise ExperimentSpecError(f"bundle install failed and was rolled back: {exc}") from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--spec", required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true")
    action.add_argument("--write", action="store_true")
    action.add_argument("--lint-all", action="store_true")
    action.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    lint = lint_from_repo(args.repo, args.spec)
    if args.lint_all or args.write or args.finalize:
        if lint["errors"]:
            print(json.dumps(lint, sort_keys=True))
            raise SystemExit(2)
        if args.lint_all:
            print(json.dumps(lint, sort_keys=True))
            return
    bundle = compile_from_repo(args.repo, args.spec)
    if args.check:
        mismatches = compare_bundle(
            bundle, lambda relpath: (args.repo.resolve() / relpath).read_bytes(),
        )
        if mismatches:
            raise SystemExit("; ".join(mismatches))
        print(json.dumps({"status": "EXPERIMENT_SPEC_BUNDLE_OK", "files": len(bundle)}))
        return
    write_bundle_atomic(args.repo, bundle)
    status = (
        "EXPERIMENT_SPEC_BUNDLE_FINALIZED"
        if args.finalize else "EXPERIMENT_SPEC_BUNDLE_WRITTEN"
    )
    print(json.dumps({"status": status, "files": len(bundle)}))


if __name__ == "__main__":
    main()
