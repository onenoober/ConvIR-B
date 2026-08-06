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
import re
import stat
import subprocess
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
    RUNTIME_BUNDLE_RELPATHS,
    RUNTIME_SPEC_DIRECTORY,
    SAFE_ENV_KEY,
    capability_input_contract_sha256,
    require_asset_path,
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
DEFAULT_AUTHORITATIVE_MAIN = "refs/remotes/github/main"
DATASET_ASSET_REGISTRY_RELPATH = "experience_docx/DATASET_ASSET_REGISTRY.json"
AUTHORING_RECEIPT_SCHEMA = 1
SHA40 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
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
SCIENTIFIC_SOURCE_FIELDS_V3 = {
    *SCIENTIFIC_SOURCE_FIELDS_V2, "research_update_binding",
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
DATASET_REGISTRY_REFERENCE_FIELDS = {
    "id", "registry_id", "access_role", "contract_access",
}
DATASET_REGISTRY_FIELDS = {
    "schema_version", "registry_id", "scope", "access_role_policy",
    "verification_source", "assets",
}
DATASET_REGISTRY_SOURCE_FIELDS = {
    "route_id", "run_id", "terminal_state", "terminal_record_sha256",
    "closeout_path", "closeout_sha256", "summary_path", "summary_sha256",
}
DATASET_REGISTRY_ASSET_FIELDS = {"kind", "path", "verification"}
DATASET_REGISTRY_VERIFICATIONS = {
    "parent_of_verified_layout", "verified_clear_root", "verified_haze_root",
}


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


def _safe_repo_relpath(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ExperimentSpecError(f"{name} must be a repository-relative path")
    path = Path(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ExperimentSpecError(f"{name} must be a safe repository-relative path")
    return path.as_posix()


def validate_dataset_asset_registry(raw: bytes) -> dict[str, dict[str, str]]:
    """Validate the location-only dataset registry from authoritative main."""
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExperimentSpecError(f"dataset asset registry is invalid JSON: {exc}") from exc
    registry = _object(value, DATASET_REGISTRY_FIELDS, "dataset asset registry")
    if registry["schema_version"] != 1:
        raise ExperimentSpecError("dataset asset registry schema_version must equal 1")
    _token(registry["registry_id"], "dataset asset registry id")
    if registry["scope"] != "location_only_no_scientific_authorization" \
            or registry["access_role_policy"] != "assigned_by_each_route_contract":
        raise ExperimentSpecError("dataset asset registry authority boundary is invalid")
    source = _object(
        registry["verification_source"], DATASET_REGISTRY_SOURCE_FIELDS,
        "dataset asset registry verification_source",
    )
    _token(source["route_id"], "dataset asset registry source route_id")
    _token(source["run_id"], "dataset asset registry source run_id")
    if source["terminal_state"] != "COMPLETED_GATE_PASS":
        raise ExperimentSpecError("dataset asset registry source must be a gate PASS")
    for field in ("terminal_record_sha256", "closeout_sha256", "summary_sha256"):
        if not isinstance(source[field], str) or not SHA256.fullmatch(source[field]):
            raise ExperimentSpecError(f"dataset asset registry {field} is invalid")
    for field in ("closeout_path", "summary_path"):
        path = _safe_repo_relpath(source[field], f"dataset asset registry {field}")
        expected_prefix = f"experience_docx/experiment_logs/{source['route_id']}/"
        if not path.startswith(expected_prefix) or not path.endswith(".json"):
            raise ExperimentSpecError(f"dataset asset registry {field} is invalid")
    assets = registry["assets"]
    if not isinstance(assets, dict) or not 1 <= len(assets) <= 256:
        raise ExperimentSpecError("dataset asset registry assets must be a non-empty object")
    result: dict[str, dict[str, str]] = {}
    paths = set()
    for registry_id, item in assets.items():
        _token(registry_id, "dataset registry asset id")
        item = _object(
            item, DATASET_REGISTRY_ASSET_FIELDS,
            f"dataset asset registry assets.{registry_id}",
        )
        if item["kind"] != "directory":
            raise ExperimentSpecError("dataset asset registry currently accepts directories only")
        try:
            path = require_asset_path(
                item["path"], f"dataset asset registry assets.{registry_id}.path",
            )
        except ContractError as exc:
            raise ExperimentSpecError(str(exc)) from exc
        if "{" in path or not Path(path).is_absolute():
            raise ExperimentSpecError("registered dataset paths must be absolute")
        if path in paths:
            raise ExperimentSpecError("dataset asset registry paths must be unique")
        paths.add(path)
        if item["verification"] not in DATASET_REGISTRY_VERIFICATIONS:
            raise ExperimentSpecError("dataset asset registry verification is invalid")
        result[registry_id] = {
            "kind": item["kind"], "path": path,
            "verification": item["verification"],
        }
    return result


def registered_asset_environment_key(asset_id: str) -> str:
    identifier = _token(asset_id, "registered asset id")
    suffix = re.sub(r"[^A-Za-z0-9]", "_", identifier).upper()
    key = f"CONVIR_ROUTE_ASSET_{suffix}"
    if not SAFE_ENV_KEY.fullmatch(key):
        raise ExperimentSpecError(f"registered asset id cannot form a runtime key: {asset_id}")
    return key


def expand_registered_assets(
    assets: Any, *, spec_schema: int,
    read_authoritative_file: Callable[[str], bytes] | None,
) -> tuple[Any, dict[str, str]]:
    """Expand schema-3 registry references and derive workload environment paths."""
    if not isinstance(assets, list) or not any(
        isinstance(item, dict) and "registry_id" in item for item in assets
    ):
        return assets, {}
    if spec_schema != 3 or read_authoritative_file is None:
        raise ExperimentSpecError(
            "dataset registry references require schema-3 authoritative compilation"
        )
    try:
        registry_raw = read_authoritative_file(DATASET_ASSET_REGISTRY_RELPATH)
    except (OSError, KeyError, RuntimeError) as exc:
        raise ExperimentSpecError(f"dataset asset registry is unavailable: {exc}") from exc
    registry = validate_dataset_asset_registry(registry_raw)
    expanded = []
    environment: dict[str, str] = {}
    for index, item in enumerate(assets):
        if not isinstance(item, dict) or "registry_id" not in item:
            expanded.append(item)
            continue
        reference = _object(
            item, DATASET_REGISTRY_REFERENCE_FIELDS,
            f"assets[{index}] dataset registry reference",
        )
        asset_id = _token(reference["id"], f"assets[{index}].id")
        registry_id = _token(reference["registry_id"], f"assets[{index}].registry_id")
        if registry_id not in registry:
            raise ExperimentSpecError(f"unknown dataset registry id: {registry_id}")
        registered = registry[registry_id]
        expanded.append({
            "id": asset_id,
            "kind": registered["kind"],
            "path": registered["path"],
            "access_role": reference["access_role"],
            "contract_access": reference["contract_access"],
        })
        key = registered_asset_environment_key(asset_id)
        if key in environment:
            raise ExperimentSpecError(f"registered asset environment key collision: {key}")
        environment[key] = registered["path"]
    return expanded, environment


def runtime_with_registered_asset_environment(
    runtime: dict[str, Any], environment: dict[str, str], *, name: str,
) -> dict[str, Any]:
    if not environment:
        return runtime
    existing = runtime.get("environment")
    if not isinstance(existing, dict):
        raise ExperimentSpecError(f"{name}.environment must be an object")
    conflicts = sorted(set(existing) & set(environment))
    if conflicts:
        raise ExperimentSpecError(
            f"{name}.environment uses compiler-owned registered asset keys: {conflicts}"
        )
    return {**runtime, "environment": {**existing, **environment}}


def _repo_member(repo: Path, relpath: str, name: str, *, must_be_file: bool = False) -> Path:
    repo = repo.resolve(strict=True)
    relpath = _safe_repo_relpath(relpath, name)
    current = repo
    parts = Path(relpath).parts
    for index, part in enumerate(parts):
        current = current / part
        if not os.path.lexists(current):
            if must_be_file or index < len(parts) - 1:
                raise ExperimentSpecError(f"{name} is unavailable: {relpath}")
            break
        observed = current.lstat()
        if stat.S_ISLNK(observed.st_mode):
            raise ExperimentSpecError(f"{name} cannot contain symlinks: {relpath}")
        if index < len(parts) - 1 and not stat.S_ISDIR(observed.st_mode):
            raise ExperimentSpecError(f"{name} parent is not a directory: {relpath}")
    try:
        current.resolve(strict=must_be_file).relative_to(repo)
    except (OSError, ValueError) as exc:
        raise ExperimentSpecError(f"{name} escapes the repository: {relpath}") from exc
    if must_be_file:
        observed = current.lstat()
        if not stat.S_ISREG(observed.st_mode):
            raise ExperimentSpecError(f"{name} is not a regular file: {relpath}")
    return current


def _repo_read_bytes(repo: Path, relpath: str, name: str) -> bytes:
    return _repo_member(repo, relpath, name, must_be_file=True).read_bytes()


def _repo_file_exists(repo: Path, relpath: str) -> bool:
    try:
        _repo_member(repo, relpath, "evidence path", must_be_file=True)
    except ExperimentSpecError:
        return False
    return True


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    try:
        completed = subprocess.run(
            ["/usr/bin/git", *args], cwd=repo, capture_output=True,
            timeout=30, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ExperimentSpecError(f"git {' '.join(args)} could not complete: {exc}") from exc
    if check and completed.returncode:
        detail = (completed.stdout + completed.stderr).decode(errors="replace").strip()[:2048]
        raise ExperimentSpecError(f"git {' '.join(args)} failed: {detail}")
    return completed


def resolve_fresh_authoritative_main(repo: Path, authoritative_main: str) -> str:
    """Resolve a remote-tracking ref only when it equals the live remote head."""
    match = __import__("re").fullmatch(
        r"refs/remotes/([A-Za-z0-9][A-Za-z0-9_.-]*)/([A-Za-z0-9][A-Za-z0-9_./-]*)",
        authoritative_main if isinstance(authoritative_main, str) else "",
    )
    if match is None or ".." in match.group(2).split("/"):
        raise ExperimentSpecError(
            "authoritative main must be an exact remote-tracking ref"
        )
    resolved = _git(
        repo, "rev-parse", "--verify", f"{authoritative_main}^{{commit}}",
    ).stdout.decode("ascii").strip()
    remote, branch = match.groups()
    observed = _git(
        repo, "ls-remote", "--exit-code", remote, f"refs/heads/{branch}",
    ).stdout.decode("ascii", errors="replace").split()
    if len(observed) != 2 or not SHA40.fullmatch(observed[0]) \
            or observed[1] != f"refs/heads/{branch}":
        raise ExperimentSpecError("authoritative remote main identity is malformed")
    if resolved != observed[0]:
        raise ExperimentSpecError(
            "authoritative remote-tracking main is stale; refresh it once before authoring"
        )
    return resolved


def _authoritative_evidence_resolver(
    repo: Path, rules_commit: Any, authoritative_main: str,
) -> tuple[str, Callable[[str], bool], Callable[[str], bytes]]:
    """Resolve archived authorization evidence from the current GitHub main."""
    if not isinstance(rules_commit, str) or not SHA40.fullmatch(rules_commit):
        raise ExperimentSpecError("rules_commit must be an exact 40-character commit")
    resolved = resolve_fresh_authoritative_main(repo, authoritative_main)
    _git(repo, "cat-file", "-e", f"{rules_commit}^{{commit}}")
    cache: dict[str, bool] = {}

    def exists(relpath: str) -> bool:
        try:
            relpath = _safe_repo_relpath(relpath, "authoritative evidence path")
        except ExperimentSpecError:
            return False
        if relpath not in cache:
            completed = _git(
                repo, "cat-file", "-t", f"{resolved}:{relpath}", check=False,
            )
            cache[relpath] = (
                completed.returncode == 0
                and completed.stdout.decode("ascii", errors="replace").strip() == "blob"
            )
        return cache[relpath]

    def read(relpath: str) -> bytes:
        relpath = _safe_repo_relpath(relpath, "authoritative evidence path")
        completed = _git(
            repo, "show", f"{resolved}:{relpath}", check=False,
        )
        if completed.returncode:
            raise ExperimentSpecError(
                f"authoritative evidence file is unavailable: {relpath}"
            )
        return completed.stdout

    return resolved, exists, read


def research_snapshot_commit(source: Any) -> str | None:
    """Return the one frozen research snapshot used by a schema-3 source."""
    if not isinstance(source, dict) or source.get("schema_version") != 3:
        return None
    operations = source.get("operations")
    if not isinstance(operations, dict) or not operations:
        raise ExperimentSpecError(
            "schema-3 experiment spec must contain operations before snapshot binding"
        )
    snapshots = set()
    for operation_id, operation in operations.items():
        if not isinstance(operation, dict):
            raise ExperimentSpecError(
                f"operations.{operation_id} must be an object before snapshot binding"
            )
        scientific = operation.get("scientific_contract")
        binding = scientific.get("research_update_binding") \
            if isinstance(scientific, dict) else None
        snapshot = binding.get("snapshot_commit") if isinstance(binding, dict) else None
        if not isinstance(snapshot, str) or not SHA40.fullmatch(snapshot):
            raise ExperimentSpecError(
                f"operations.{operation_id} has no valid research snapshot binding"
            )
        snapshots.add(snapshot)
    if len(snapshots) != 1:
        raise ExperimentSpecError(
            "all schema-3 operations must bind one research snapshot commit"
        )
    return next(iter(snapshots))


def canonical_runtime_bundle(repo: Path, authoritative_commit: str) -> dict[str, bytes]:
    """Read the complete runnable closure from one exact authoritative commit."""
    if not isinstance(authoritative_commit, str) or not SHA40.fullmatch(authoritative_commit):
        raise ExperimentSpecError("authoritative runtime commit is invalid")
    result = {}
    for relpath in RUNTIME_BUNDLE_RELPATHS:
        relpath = _safe_repo_relpath(relpath, "canonical runtime path")
        completed = _git(
            repo, "show", f"{authoritative_commit}:{relpath}", check=False,
        )
        if completed.returncode:
            raise ExperimentSpecError(
                f"authoritative main is missing canonical runtime file: {relpath}"
            )
        result[relpath] = completed.stdout
    return result


def _authoring_receipt_path(repo: Path, route_id: str) -> Path:
    relative = f"convir/authoring-receipts/{_token(route_id, 'route_id')}.json"
    raw = _git(repo, "rev-parse", "--path-format=absolute", "--git-path", relative).stdout
    return Path(raw.decode("utf-8").strip())


def _bundle_identity(bundle: dict[str, bytes]) -> tuple[list[dict[str, Any]], str]:
    files = [
        {"relpath": relpath, "sha256": sha256(raw), "size_bytes": len(raw)}
        for relpath, raw in sorted(bundle.items())
    ]
    return files, sha256(json_bytes(files))


def build_authoring_receipt(
    *, spec_relpath: str, spec_raw: bytes, program_relpath: str,
    program_raw: bytes, bundle: dict[str, bytes], authoritative_main_commit: str,
) -> dict[str, Any]:
    source = json.loads(spec_raw)
    files, bundle_sha = _bundle_identity(bundle)
    return {
        "schema_version": AUTHORING_RECEIPT_SCHEMA,
        "status": "EXPERIMENT_SPEC_BUNDLE_FINALIZED",
        "route_id": source["route_id"],
        "authoritative_main_commit": authoritative_main_commit,
        "experiment_spec": {
            "relpath": spec_relpath, "sha256": sha256(spec_raw),
        },
        "program_contract": {
            "relpath": program_relpath, "sha256": sha256(program_raw),
        },
        "generated_files": files,
        "bundle_sha256": bundle_sha,
        "allowed_next_action": "stage_complete_bundle_then_route_ready_once",
    }


def write_private_receipt_atomic(repo: Path, route_id: str, receipt: dict[str, Any]) -> Path:
    destination = _authoring_receipt_path(repo, route_id)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix=f".{destination.name}.", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(json_bytes(receipt))
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _ensure_repo_parent(repo: Path, relpath: str, name: str) -> Path:
    repo = repo.resolve(strict=True)
    relpath = _safe_repo_relpath(relpath, name)
    current = repo
    for part in Path(relpath).parts[:-1]:
        current = current / part
        try:
            current.mkdir()
        except FileExistsError:
            pass
        observed = current.lstat()
        if stat.S_ISLNK(observed.st_mode) or not stat.S_ISDIR(observed.st_mode):
            raise ExperimentSpecError(f"{name} parent cannot be a symlink or file: {relpath}")
    destination = current / Path(relpath).name
    if os.path.lexists(destination) and stat.S_ISLNK(destination.lstat().st_mode):
        raise ExperimentSpecError(f"{name} cannot overwrite a symlink: {relpath}")
    try:
        current.resolve(strict=True).relative_to(repo)
    except ValueError as exc:
        raise ExperimentSpecError(f"{name} escapes the repository: {relpath}") from exc
    return destination


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
    return {
        1: SCIENTIFIC_SOURCE_FIELDS_V1,
        2: SCIENTIFIC_SOURCE_FIELDS_V2,
        3: SCIENTIFIC_SOURCE_FIELDS_V3,
    }[spec_schema]


def _supporting_contract_schema(spec_schema: int) -> int:
    return 2 if spec_schema in {2, 3} else 1


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
    if _supporting_contract_schema(spec_schema) != 2 or not isinstance(value, dict):
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
    authoritative_snapshot_commit: str | None,
    read_authoritative_file: Callable[[str], bytes] | None,
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
        if spec_schema in {2, 3}:
            if spec_schema == 2:
                scientific = science_contract.validate_scientific_contract_v2(
                    scientific_value, route_id, operation_id,
                )
            else:
                scientific = science_contract.validate_scientific_contract_v3(
                    scientific_value, route_id, operation_id,
                    expected_snapshot_commit=authoritative_snapshot_commit,
                    read_evidence_file=read_authoritative_file,
                    require_current_design=True,
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
        if spec_schema in {2, 3}:
            operation = None

    expanded_assets = item["assets"]
    registered_environment: dict[str, str] = {}
    try:
        expanded_assets, registered_environment = expand_registered_assets(
            item["assets"], spec_schema=spec_schema,
            read_authoritative_file=read_authoritative_file,
        )
    except (ExperimentSpecError, KeyError, TypeError, ValueError) as exc:
        expanded_assets = None
        _append_lint(
            errors, f"{prefix}.assets", exc, "DATASET_ASSET_REGISTRY_INVALID",
        )

    runtime_source = None
    engineering = None
    precision_contract_source = None
    try:
        runtime_source = _object(
            item["runtime"], RUNTIME_SOURCE_FIELDS, f"{prefix}.runtime",
        )
        runtime_source = runtime_with_registered_asset_environment(
            runtime_source, registered_environment, name=f"{prefix}.runtime",
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
        supporting_schema = _supporting_contract_schema(spec_schema)
        if item["assets"] is not None and expanded_assets is not None:
            try:
                asset = validate_asset_manifest({
                    "schema_version": supporting_schema, "route_id": route_id,
                    "operation_id": operation_id, "assets": expanded_assets,
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
                    {"schema_version": supporting_schema, **capability_source},
                    validated_runtime, asset,
                )
            except (ContractError, KeyError, TypeError, ValueError) as exc:
                _append_lint(errors, f"{prefix}.capability", exc, "CAPABILITY_CONTRACT_INVALID")
        if item["precision"] is not None:
            try:
                precision = validate_precision_certificate(
                    {"schema_version": supporting_schema, **item["precision"]},
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
                read_repo_file: Callable[[str], bytes] | None = None,
                authoritative_snapshot_commit: str | None = None,
                read_authoritative_file: Callable[[str], bytes] | None = None,
                return_bundle: bool = False) -> dict[str, Any]:
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
        ("schema_version", lambda: source.get("schema_version") in {1, 2, 3}
         or (_ for _ in ()).throw(ExperimentSpecError("must equal 1, 2, or 3"))),
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
        program_contract.validate_current_amendment_encoding(program_source)
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
    compiled_bundle = None
    if top_ready:
        for operation_id in structurally_valid_operations:
            _lint_operation_components(
                errors=errors, route_id=route_id, operation_id=operation_id,
                item=operations[operation_id],
                effective_program=effective_program if program_ok else None,
                evidence_exists=evidence_exists,
                read_repo_file=read_repo_file,
                spec_schema=source["schema_version"],
                authoritative_snapshot_commit=authoritative_snapshot_commit,
                read_authoritative_file=read_authoritative_file,
            )
        if not errors and len(structurally_valid_operations) == len(operations):
            try:
                compiled_bundle = compile_bundle(
                    spec_relpath=spec_relpath, spec_raw=spec_raw,
                    program_raw=program_raw, evidence_exists=evidence_exists,
                    authoritative_snapshot_commit=authoritative_snapshot_commit,
                    read_authoritative_file=read_authoritative_file,
                )
            except (ExperimentSpecError, KeyError, TypeError, ValueError) as exc:
                errors.append(_lint_error("experiment_spec", _lint_code(str(exc)), exc))
    unique = {
        (item["path"], item["code"], item["message"]): item for item in errors
    }
    ordered = sorted(unique.values(), key=lambda item: (item["path"], item["code"], item["message"]))
    result = {
        "status": "EXPERIMENT_SPEC_INVALID" if ordered else "EXPERIMENT_SPEC_VALID",
        "errors": ordered,
    }
    if return_bundle and not ordered:
        result["_bundle"] = compiled_bundle
    return result


def compile_bundle(*, spec_relpath: str, spec_raw: bytes, program_raw: bytes,
                   evidence_exists: Callable[[str], bool] | None = None,
                   authoritative_snapshot_commit: str | None = None,
                   read_authoritative_file: Callable[[str], bytes] | None = None) \
        -> dict[str, bytes]:
    """Return every generated file as repository-relative path -> exact bytes."""
    try:
        source = json.loads(spec_raw)
        program_source = json.loads(program_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExperimentSpecError(f"source JSON is invalid: {exc}") from exc
    spec = _object(source, SOURCE_FIELDS, "experiment spec")
    if spec["schema_version"] not in {1, 2, 3}:
        raise ExperimentSpecError("experiment spec schema_version must be 1, 2, or 3")
    spec_schema = spec["schema_version"]
    frozen_research_snapshot = research_snapshot_commit(spec)
    if spec_schema == 3 and (
        authoritative_snapshot_commit != frozen_research_snapshot
        or read_authoritative_file is None or evidence_exists is None
    ):
        raise ExperimentSpecError(
            "schema-3 compilation requires its exact authoritative research snapshot"
        )
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
        program_contract.validate_current_amendment_encoding(program_source)
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
    route_research_update = None
    archived_program_ids = None
    for operation_id, source_operation in operations.items():
        _token(operation_id, "operation_id")
        item = _object(source_operation, OPERATION_SOURCE_FIELDS, f"operations.{operation_id}")
        operation_source = item["operation"]
        if not isinstance(operation_source, dict):
            raise ExperimentSpecError(f"operations.{operation_id}.operation must be an object")
        claim = item["program_authorization"]
        try:
            validated_claim = program_contract.validate_route_authorization(
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
        if spec_schema in {2, 3}:
            try:
                if spec_schema == 2:
                    scientific = science_contract.validate_scientific_contract_v2(
                        scientific, route_id, operation_id,
                    )
                else:
                    scientific = science_contract.validate_scientific_contract_v3(
                        scientific, route_id, operation_id,
                        expected_snapshot_commit=authoritative_snapshot_commit,
                        read_evidence_file=read_authoritative_file,
                        require_current_design=True,
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
        if spec_schema == 3:
            research_update = scientific["research_update_binding"]
            if route_research_update is None:
                route_research_update = research_update
            elif research_update != route_research_update:
                raise ExperimentSpecError(
                    "all schema-3 operations must share one research update binding"
                )
            if research_update["trigger_type"] == "program_foundation":
                if validated_claim["mechanism_type"] != "adjacent" \
                        or any(
                            family["attempts_used"] != 0
                            for family in effective_program["route_families"].values()
                        ):
                    raise ExperimentSpecError(
                        "program_foundation requires the first adjacent route of an unused program"
                    )
                if archived_program_ids is None:
                    if evidence_exists(science_contract.TERMINAL_INDEX_RELPATH):
                        archived_program_ids = set(
                            science_contract.archived_terminal_program_ids(
                                read_authoritative_file,
                            ).values()
                        )
                    else:
                        archived_program_ids = set()
                if effective_program["program_id"] in archived_program_ids:
                    raise ExperimentSpecError(
                        "program_foundation program_id already exists in archived terminal evidence"
                    )
        manifest_operations[operation_id] = operation
        scientific_path = _scientific_path(route_id, operation_id)
        scientific_paths[operation_id] = scientific_path
        expanded_assets, registered_environment = expand_registered_assets(
            item["assets"], spec_schema=spec_schema,
            read_authoritative_file=read_authoritative_file,
        )
        runtime_source = _object(
            item["runtime"], RUNTIME_SOURCE_FIELDS, f"operations.{operation_id}.runtime",
        )
        runtime_source = runtime_with_registered_asset_environment(
            runtime_source, registered_environment,
            name=f"operations.{operation_id}.runtime",
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
            supporting_schema = _supporting_contract_schema(spec_schema)
            asset = None
            if item["assets"] is not None:
                asset = validate_asset_manifest({
                    "schema_version": supporting_schema, "route_id": route_id,
                    "operation_id": operation_id, "assets": expanded_assets,
                }, validated_runtime)
                generated[asset_path] = json_bytes(asset)
            if item["capability"] is not None:
                capability_source = _capability_with_derived_input_identity(
                    item["capability"], spec_schema=spec_schema,
                    runtime=validated_runtime,
                )
                capability = validate_model_capability(
                    {"schema_version": supporting_schema, **capability_source},
                    validated_runtime, asset,
                )
                generated[capability_path] = json_bytes(capability)
            precision = None
            if item["precision"] is not None:
                precision = validate_precision_certificate(
                    {"schema_version": supporting_schema, **item["precision"]},
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


def _source_inputs_from_repo(
    repo: Path, spec_relpath: str, authoritative_main: str,
) -> tuple[
    str, bytes, str, bytes, str, Callable[[str], bool], Callable[[str], bytes],
]:
    repo = repo.resolve(strict=True)
    spec_relpath = _relpath(spec_relpath, "experiment spec", SPEC_DIRECTORY)
    spec_raw = _repo_read_bytes(repo, spec_relpath, "experiment spec")
    source = json.loads(spec_raw)
    program_relpath = source.get("program_contract_relpath")
    if not isinstance(program_relpath, str):
        raise ExperimentSpecError("program_contract_relpath is missing")
    program_relpath = _relpath(
        program_relpath, "program_contract_relpath", PROGRAM_DIRECTORY,
    )
    authoritative_commit, evidence_exists, read_authoritative_file = \
        _authoritative_evidence_resolver(
        repo, source.get("rules_commit"), authoritative_main,
    )
    return (
        spec_relpath, spec_raw, program_relpath,
        _repo_read_bytes(repo, program_relpath, "program contract"),
        authoritative_commit, evidence_exists, read_authoritative_file,
    )


def compile_from_repo(
    repo: Path, spec_relpath: str,
    authoritative_main: str = DEFAULT_AUTHORITATIVE_MAIN,
) -> dict[str, bytes]:
    (
        spec_relpath, spec_raw, _, program_raw, authoritative_commit,
        evidence_exists, read_authoritative_file,
    ) = _source_inputs_from_repo(repo, spec_relpath, authoritative_main)
    return compile_bundle(
        spec_relpath=spec_relpath,
        spec_raw=spec_raw,
        program_raw=program_raw,
        evidence_exists=evidence_exists,
        authoritative_snapshot_commit=authoritative_commit,
        read_authoritative_file=read_authoritative_file,
    )


def lint_from_repo(
    repo: Path, spec_relpath: str, *, require_current_schema: bool = False,
    authoritative_main: str = DEFAULT_AUTHORITATIVE_MAIN,
    return_bundle: bool = False,
) -> dict[str, Any]:
    repo = repo.resolve(strict=True)
    try:
        spec_relpath = _relpath(spec_relpath, "experiment spec", SPEC_DIRECTORY)
        spec_raw = _repo_read_bytes(repo, spec_relpath, "experiment spec")
    except (OSError, ExperimentSpecError) as exc:
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
            program_relpath = _relpath(
                program_relpath, "program_contract_relpath", PROGRAM_DIRECTORY,
            )
            program_raw = _repo_read_bytes(repo, program_relpath, "program contract")
        except (OSError, ExperimentSpecError) as exc:
            return {
                "status": "EXPERIMENT_SPEC_INVALID",
                "errors": [_lint_error("program_contract_relpath", "READ_FAILED", exc)],
            }
    try:
        authoritative_commit, evidence_exists, read_authoritative_file = \
            _authoritative_evidence_resolver(
            repo, source.get("rules_commit"), authoritative_main,
        )
        result = lint_bundle(
            spec_relpath=spec_relpath, spec_raw=spec_raw, program_raw=program_raw,
            evidence_exists=evidence_exists,
            read_repo_file=lambda relpath: _repo_read_bytes(
                repo, relpath, "route repository asset",
            ),
            authoritative_snapshot_commit=authoritative_commit,
            read_authoritative_file=read_authoritative_file,
            return_bundle=return_bundle,
        )
        if return_bundle and not result["errors"]:
            result["_authoring_inputs"] = {
                "spec_relpath": spec_relpath,
                "spec_raw": spec_raw,
                "program_relpath": program_relpath,
                "program_raw": program_raw,
                "authoritative_main_commit": authoritative_commit,
            }
    except (OSError, ExperimentSpecError) as exc:
        result = {
            "status": "EXPERIMENT_SPEC_INVALID",
            "errors": [_lint_error(
                "rules_commit", "AUTHORITATIVE_MAIN_IDENTITY_INVALID", exc,
            )],
        }
    if require_current_schema and (
        not isinstance(source, dict) or source.get("schema_version") != 3
    ):
        errors = [
            *result["errors"],
            _lint_error(
                "schema_version", "CURRENT_SCHEMA_REQUIRED",
                "new experiment authoring requires schema_version 3; "
                "schema_version 1/2 is historical read-only compatibility",
            ),
        ]
        result = {
            "status": "EXPERIMENT_SPEC_INVALID",
            "errors": sorted(
                {
                    (item["path"], item["code"], item["message"]): item
                    for item in errors
                }.values(),
                key=lambda item: (item["path"], item["code"], item["message"]),
            ),
        }
    return result


def write_bundle_atomic(repo: Path, bundle: dict[str, bytes]) -> None:
    """Install a complete derived bundle and restore prior bytes on failure."""
    repo = repo.resolve(strict=True)
    previous: dict[str, bytes | None] = {}
    installed: list[str] = []
    with tempfile.TemporaryDirectory(prefix=".experiment-spec-finalize-", dir=repo) as raw_tmp:
        temporary = Path(raw_tmp)
        staged = temporary / "staged"
        rollback = temporary / "rollback"
        for relpath, raw in bundle.items():
            relpath = _safe_repo_relpath(relpath, "generated bundle path")
            candidate = staged / relpath
            candidate.parent.mkdir(parents=True, exist_ok=True)
            candidate.write_bytes(raw)
            destination = _ensure_repo_parent(repo, relpath, "generated bundle path")
            previous[relpath] = destination.read_bytes() if destination.is_file() else None
        try:
            for relpath in bundle:
                relpath = _safe_repo_relpath(relpath, "generated bundle path")
                destination = _ensure_repo_parent(repo, relpath, "generated bundle path")
                os.replace(staged / relpath, destination)
                installed.append(relpath)
        except OSError as exc:
            rollback_errors = []
            for relpath in reversed(installed):
                destination = _ensure_repo_parent(repo, relpath, "generated bundle path")
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
    lint = lint_from_repo(
        args.repo, args.spec,
        require_current_schema=args.write or args.finalize,
        authoritative_main=DEFAULT_AUTHORITATIVE_MAIN,
        return_bundle=not args.lint_all,
    )
    if lint["errors"]:
        print(json.dumps(lint, sort_keys=True))
        raise SystemExit(2)
    if args.lint_all:
        print(json.dumps(lint, sort_keys=True))
        return
    bundle = lint.pop("_bundle")
    inputs = lint.pop("_authoring_inputs")
    if args.check:
        mismatches = compare_bundle(
            bundle,
            lambda relpath: _repo_read_bytes(
                args.repo.resolve(strict=True), relpath, "generated bundle path",
            ),
        )
        if mismatches:
            raise SystemExit("; ".join(mismatches))
        print(json.dumps({"status": "EXPERIMENT_SPEC_BUNDLE_OK", "files": len(bundle)}))
        return
    installed_bundle = bundle
    if args.finalize:
        installed_bundle = {
            **bundle,
            **canonical_runtime_bundle(
                args.repo.resolve(strict=True),
                inputs["authoritative_main_commit"],
            ),
        }
    write_bundle_atomic(args.repo, installed_bundle)
    status = (
        "EXPERIMENT_SPEC_BUNDLE_FINALIZED"
        if args.finalize else "EXPERIMENT_SPEC_BUNDLE_WRITTEN"
    )
    report: dict[str, Any] = {"status": status, "files": len(installed_bundle)}
    if args.finalize:
        receipt = build_authoring_receipt(bundle=installed_bundle, **inputs)
        write_private_receipt_atomic(args.repo.resolve(strict=True), receipt["route_id"], receipt)
        report["receipt"] = receipt
        report["receipt_sha256"] = sha256(json_bytes(receipt))
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
