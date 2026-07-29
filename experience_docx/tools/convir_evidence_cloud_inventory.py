#!/usr/bin/env python3
"""Identity-bound, transport-free cloud evidence inventory core."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import time
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

import convir_evidence_catalog as catalog
import convirctl


REMOTE_BASE = "/sda/home/wangyuxin/ConvIR-B"
REMOTE_RUNS = f"{REMOTE_BASE}/runs"
MANIFEST_SOURCE_PATH = "experience_docx/route_operations.json"
RUNTIME_SPEC_PREFIX = "experience_docx/route_runtime_specs/"
MAX_GITHUB_FILE_BYTES = 1024 * 1024
MAX_GITHUB_BOUND_BYTES = 16 * 1024 * 1024
MAX_LIFECYCLE_IDENTITY_BYTES = 64 * 1024
MAX_SCAN_ENTRIES = 25_000
MAX_SCAN_DEPTH = 16
MAX_RELATIVE_PATH_BYTES = 16 * 1024 * 1024
MAX_SCAN_SECONDS = 60
MAX_QUERY_ENTRIES = 100
MAX_QUERY_VALUE_BYTES = 8 * 1024
CURSOR_OPERATION = "evidence-cloud-inventory-query"
SAFE_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
SHA40 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
RECONCILIATION_STATES = (
    "MATCHED",
    "GITHUB_ONLY",
    "CLOUD_ONLY",
    "IDENTITY_CONFLICT",
    "CLOUD_UNAVAILABLE",
    "NOT_INVENTORIED",
)
RAW_INVENTORY_EVIDENCE_ROLES = {"engineering_debug", "development_screening"}
PROTECTED_TOUCH_FIELDS = (
    "confirmation_images_targets_outcomes_touched",
    "canary_touched",
    "locked_test_touched",
)


class InventoryError(RuntimeError):
    def __init__(self, message: str, *, state: str = "INVENTORY_INVALID",
                 exit_code: int = 3):
        super().__init__(message)
        self.state = state
        self.exit_code = exit_code


class _ScanStopped(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _require_sha(value: Any, pattern: re.Pattern[str], name: str, *,
                 state: str = "ARGUMENTS_INVALID", exit_code: int = 2) -> str:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise InventoryError(
            f"{name} has an invalid SHA identity",
            state=state,
            exit_code=exit_code,
        )
    return value


def _require_token(value: Any, name: str) -> str:
    if not isinstance(value, str) or not SAFE_TOKEN.fullmatch(value):
        raise InventoryError(f"{name} is not a safe token", state="IDENTITY_CONFLICT")
    return value


def _require_relpath(value: Any, name: str, *,
                     state: str = "IDENTITY_CONFLICT") -> str:
    if not isinstance(value, str) or not value or any(char in value for char in "\x00\n\r"):
        raise InventoryError(f"{name} is not safe relative path text", state=state)
    normalized = value.replace("\\", "/")
    candidate = PurePosixPath(normalized)
    if candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts):
        raise InventoryError(f"{name} is not a safe relative path", state=state)
    return candidate.as_posix()


def _git_blob(repo: Path, commit: str, relpath: str, *, maximum: int,
              expected_bytes: int | None = None,
              expected_sha256: str | None = None) -> bytes:
    relpath = _require_relpath(relpath, "GitHub blob path")
    try:
        oid = catalog.git_text(repo, "rev-parse", "--verify", f"{commit}:{relpath}")
        if not SHA40.fullmatch(oid) or catalog.git_text(repo, "cat-file", "-t", oid) != "blob":
            raise InventoryError("GitHub evidence object is not a blob")
        size = int(catalog.git_text(repo, "cat-file", "-s", oid))
    except (ValueError, catalog.CatalogError) as exc:
        raise InventoryError(
            f"GitHub evidence blob is unavailable: {relpath}",
            state="IDENTITY_CONFLICT",
        ) from exc
    if size < 0 or size > maximum:
        raise InventoryError(
            f"GitHub evidence blob exceeds its bound: {relpath}",
            state="IDENTITY_CONFLICT",
        )
    if expected_bytes is not None and size != expected_bytes:
        raise InventoryError(
            f"GitHub evidence byte identity differs: {relpath}",
            state="IDENTITY_CONFLICT",
        )
    raw = catalog.git_bytes(repo, ["cat-file", "blob", oid], limit=maximum)
    if len(raw) != size:
        raise InventoryError(
            f"GitHub evidence size changed during read: {relpath}",
            state="IDENTITY_CONFLICT",
        )
    if expected_sha256 is not None \
            and hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise InventoryError(
            f"GitHub evidence SHA-256 differs: {relpath}",
            state="IDENTITY_CONFLICT",
        )
    return raw


def _json_object(raw: bytes, name: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InventoryError(
            f"{name} is not valid UTF-8 JSON", state="IDENTITY_CONFLICT"
        ) from exc
    if not isinstance(value, dict):
        raise InventoryError(f"{name} must be a JSON object", state="IDENTITY_CONFLICT")
    return value


def _not_inventoried_binding(snapshot_commit: str, terminal_record_sha256: str,
                             reason: str) -> dict[str, Any]:
    return {
        "eligible": False,
        "state": "NOT_INVENTORIED",
        "reason": reason,
        "snapshot_commit": snapshot_commit,
        "terminal_record_sha256": terminal_record_sha256,
        "scientific_completeness": "not_assessed",
    }


def derive_run_root(route_id: str, output_id: str) -> str:
    route_id = _require_token(route_id, "route_id")
    output_id = _require_token(output_id, "output_id")
    return f"{REMOTE_RUNS}/{route_id}/{output_id}"


def prepare_terminal_binding(repo_value: str | Path, snapshot_commit: str,
                             catalog_sha256: str,
                             terminal_record_sha256: str) -> dict[str, Any]:
    """Resolve and fully verify one schema-2 terminal from an immutable snapshot."""
    snapshot_commit = _require_sha(snapshot_commit, SHA40, "snapshot_commit")
    catalog_sha256 = _require_sha(catalog_sha256, SHA256, "catalog_sha256")
    terminal_record_sha256 = _require_sha(
        terminal_record_sha256, SHA256, "terminal_record_sha256"
    )
    try:
        loaded = catalog.load_catalog(repo_value, snapshot_commit)
    except catalog.CatalogError as exc:
        state = "ARGUMENTS_INVALID" if exc.state == "ARGUMENTS_INVALID" \
            else "IDENTITY_CONFLICT"
        raise InventoryError(str(exc), state=state, exit_code=exc.exit_code) from exc
    if loaded["catalog_sha256"] != catalog_sha256:
        raise InventoryError("catalog identity differs", state="IDENTITY_CONFLICT")
    try:
        repo, records, index_identity = catalog.load_terminal_records(
            repo_value, snapshot_commit
        )
    except catalog.CatalogError as exc:
        state = "ARGUMENTS_INVALID" if exc.state == "ARGUMENTS_INVALID" \
            else "IDENTITY_CONFLICT"
        raise InventoryError(str(exc), state=state, exit_code=exc.exit_code) from exc
    selected = [
        record for record in records
        if record["record_sha256"] == terminal_record_sha256
    ]
    if not selected:
        return _not_inventoried_binding(
            snapshot_commit, terminal_record_sha256,
            "terminal_record_not_in_index",
        )
    if len(selected) != 1:
        raise InventoryError("terminal record identity is ambiguous")
    record = selected[0]
    if record["schema_version"] != 2:
        return _not_inventoried_binding(
            snapshot_commit, terminal_record_sha256,
            "terminal_schema_not_sha_bound",
        )

    route_id = _require_token(record["route_id"], "route_id")
    operation_id = _require_token(record["operation_id"], "operation_id")
    run_id = _require_token(record["run_id"], "run_id")
    route_commit = _require_sha(
        record["route_commit"], SHA40, "route_commit", state="IDENTITY_CONFLICT",
        exit_code=3,
    )
    total_github_bytes = 0

    def read_github_blob(relpath: str, *, expected_bytes: int | None = None,
                         expected_sha256: str | None = None) -> bytes:
        nonlocal total_github_bytes
        remaining = MAX_GITHUB_BOUND_BYTES - total_github_bytes
        if remaining < 1 or (
            expected_bytes is not None and expected_bytes > remaining
        ):
            raise InventoryError(
                "terminal GitHub evidence exceeds its aggregate bound",
                state="IDENTITY_CONFLICT",
            )
        raw = _git_blob(
            repo, snapshot_commit, relpath,
            maximum=min(MAX_GITHUB_FILE_BYTES, remaining),
            expected_bytes=expected_bytes,
            expected_sha256=expected_sha256,
        )
        total_github_bytes += len(raw)
        return raw

    evidence_root = PurePosixPath(record["closeout_path"]).parent.as_posix()
    evidence_prefix = f"{evidence_root}/"
    archived_paths = [
        record["closeout_path"], record["conclusion_path"],
        *(item["path"] for item in record["contract_bundle"]),
        *(item["path"] for item in record["result_files"]),
    ]
    if any(not path.startswith(evidence_prefix) for path in archived_paths):
        raise InventoryError(
            "terminal archive paths span evidence directories",
            state="IDENTITY_CONFLICT",
        )

    contract_raw = read_github_blob(
        record["contract_path"],
        expected_sha256=record["contract_sha256"],
    )
    closeout_raw = read_github_blob(
        record["closeout_path"],
        expected_sha256=record["closeout_sha256"],
    )
    conclusion_raw = read_github_blob(
        record["conclusion_path"],
        expected_sha256=record["conclusion_sha256"],
    )

    bundle_payloads: dict[str, bytes] = {}
    for item in record["contract_bundle"]:
        raw = read_github_blob(
            item["path"],
            expected_bytes=item["bytes"],
            expected_sha256=item["sha256"],
        )
        source_path = _require_relpath(
            item["source_path"], "contract_bundle[].source_path"
        )
        if source_path in bundle_payloads:
            raise InventoryError(
                "terminal contract bundle source paths are ambiguous",
                state="IDENTITY_CONFLICT",
            )
        bundle_payloads[source_path] = raw
    for item in record["result_files"]:
        read_github_blob(
            item["path"],
            expected_bytes=item["bytes"],
            expected_sha256=item["sha256"],
        )

    manifest_raw = bundle_payloads.get(MANIFEST_SOURCE_PATH)
    runtime_source_path = f"{RUNTIME_SPEC_PREFIX}{operation_id}.json"
    runtime_raw = bundle_payloads.get(runtime_source_path)
    if manifest_raw is None or runtime_raw is None:
        return _not_inventoried_binding(
            snapshot_commit, terminal_record_sha256,
            "archived_manifest_or_runtime_spec_missing",
        )
    manifest = _json_object(manifest_raw, "archived manifest")
    runtime = _json_object(runtime_raw, "archived runtime spec")
    conclusion = _json_object(conclusion_raw, "archived conclusion")
    if manifest.get("schema_version") != 6 or runtime.get("schema_version") != 2:
        return _not_inventoried_binding(
            snapshot_commit, terminal_record_sha256,
            "archived_contract_schema_not_supported",
        )
    if manifest.get("route_id") != route_id:
        raise InventoryError("manifest route_id differs", state="IDENTITY_CONFLICT")
    operations = manifest.get("operations")
    if not isinstance(operations, dict):
        raise InventoryError(
            "manifest operations are invalid", state="IDENTITY_CONFLICT"
        )
    operation = operations.get(operation_id)
    if not isinstance(operation, dict):
        raise InventoryError("manifest operation is absent", state="IDENTITY_CONFLICT")
    output_id = _require_token(operation.get("output_id"), "output_id")
    if output_id != run_id:
        raise InventoryError(
            "manifest output_id differs from terminal run_id",
            state="IDENTITY_CONFLICT",
        )
    closeout_name = PurePosixPath(record["closeout_path"]).name
    if operation.get("closeout_filename") != closeout_name:
        raise InventoryError(
            "manifest closeout filename differs from terminal record",
            state="IDENTITY_CONFLICT",
        )
    if runtime.get("route_id") != route_id \
            or runtime.get("operation_id") != operation_id:
        raise InventoryError("runtime spec identity differs", state="IDENTITY_CONFLICT")

    closeout = _json_object(closeout_raw, "archived closeout")
    if closeout.get("schema_version") != 2 or conclusion.get("schema_version") != 1:
        raise InventoryError(
            "closeout or conclusion schema differs", state="IDENTITY_CONFLICT"
        )
    exact_closeout = {
        "route_id": route_id,
        "operation_id": operation_id,
        "run_id": run_id,
        "route_commit": route_commit,
        "state": record["state"],
        "decision": record["decision"],
        "authorizes": record["authorizes"],
    }
    for key, expected in exact_closeout.items():
        if closeout.get(key) != expected:
            raise InventoryError(
                f"closeout {key} differs from terminal record",
                state="IDENTITY_CONFLICT",
            )
    for key in ("route_id", "operation_id", "run_id", "state", "decision", "authorizes"):
        if conclusion.get(key) != exact_closeout[key]:
            raise InventoryError(
                f"conclusion {key} differs from terminal record",
                state="IDENTITY_CONFLICT",
            )
    runner_sha256 = _require_sha(
        closeout.get("runner_sha256"), SHA256, "closeout.runner_sha256",
        state="IDENTITY_CONFLICT", exit_code=3,
    )
    if closeout.get("evidence_role") != runtime.get("evidence_role"):
        raise InventoryError(
            "closeout evidence role differs from runtime spec",
            state="IDENTITY_CONFLICT",
        )

    permissions = runtime.get("protected_data_permissions")
    permission_names = {
        "allow_confirmation", "allow_canary", "allow_locked_test"
    }
    if not isinstance(permissions, dict) or set(permissions) != permission_names \
            or any(value is not True and value is not False for value in permissions.values()):
        raise InventoryError(
            "runtime protected-data permissions are invalid",
            state="IDENTITY_CONFLICT",
        )
    evidence_role = runtime.get("evidence_role")
    if not isinstance(evidence_role, str) or not evidence_role:
        raise InventoryError("runtime evidence role is invalid", state="IDENTITY_CONFLICT")

    protected_touches = {}
    for field in PROTECTED_TOUCH_FIELDS:
        value = closeout.get(field)
        if value is not True and value is not False:
            raise InventoryError(
                f"closeout {field} is invalid", state="IDENTITY_CONFLICT"
            )
        protected_touches[field] = value

    result_by_name: dict[str, dict[str, Any]] = {}
    for item in record["result_files"]:
        name = PurePosixPath(item["path"]).name
        if name in result_by_name:
            raise InventoryError(
                "terminal result filenames are ambiguous",
                state="IDENTITY_CONFLICT",
            )
        result_by_name[name] = item
    evidence_sha256 = closeout.get("evidence_sha256")
    if not isinstance(evidence_sha256, dict):
        raise InventoryError(
            "closeout evidence SHA manifest is invalid", state="IDENTITY_CONFLICT"
        )
    expected_closeout_hashes = {
        PurePosixPath(item["path"]).name: item["sha256"]
        for item in record["result_files"]
    }
    if evidence_sha256 != expected_closeout_hashes:
        raise InventoryError(
            "closeout evidence SHA manifest differs from terminal results",
            state="IDENTITY_CONFLICT",
        )
    expected_evidence = []
    optional_evidence = []
    mapped_names = set()
    mapped_sources = set()
    declared_names = set()
    declarations = runtime.get("evidence_files")
    if not isinstance(declarations, list) or not declarations:
        raise InventoryError(
            "runtime spec declares no compact evidence", state="IDENTITY_CONFLICT"
        )
    for index, item in enumerate(declarations):
        if not isinstance(item, dict):
            raise InventoryError(
                f"runtime evidence_files[{index}] is invalid",
                state="IDENTITY_CONFLICT",
            )
        source_relpath = _require_relpath(
            item.get("source_relpath"), f"evidence_files[{index}].source_relpath"
        )
        destination = item.get("destination_filename")
        if not isinstance(destination, str) \
                or PurePosixPath(destination).name != destination:
            raise InventoryError(
                f"evidence_files[{index}].destination_filename is invalid",
                state="IDENTITY_CONFLICT",
            )
        if source_relpath in mapped_sources or destination in declared_names:
            raise InventoryError(
                "runtime evidence mapping is ambiguous",
                state="IDENTITY_CONFLICT",
            )
        mapped_sources.add(source_relpath)
        declared_names.add(destination)
        required = item.get("required")
        maximum = item.get("max_bytes")
        if required is not True and required is not False:
            raise InventoryError(
                f"evidence_files[{index}].required is invalid",
                state="IDENTITY_CONFLICT",
            )
        if not isinstance(maximum, int) or isinstance(maximum, bool) \
                or not 1 <= maximum <= MAX_GITHUB_FILE_BYTES:
            raise InventoryError(
                f"evidence_files[{index}].max_bytes is invalid",
                state="IDENTITY_CONFLICT",
            )
        result_record = result_by_name.get(destination)
        if required and result_record is None:
            raise InventoryError(
                f"required runtime evidence is absent from terminal results: {destination}",
                state="IDENTITY_CONFLICT",
            )
        if result_record is None:
            optional_evidence.append({
                "source_relpath": source_relpath,
                "destination_filename": destination,
                "max_bytes": maximum,
                "required": False,
            })
            continue
        if result_record["bytes"] > maximum \
                or evidence_sha256.get(destination) != result_record["sha256"]:
            raise InventoryError(
                f"runtime/closeout/result evidence identity differs: {destination}",
                state="IDENTITY_CONFLICT",
            )
        mapped_names.add(destination)
        expected_evidence.append({
            "source_relpath": source_relpath,
            "destination_filename": destination,
            "github_path": result_record["path"],
            "bytes": result_record["bytes"],
            "sha256": result_record["sha256"],
            "max_bytes": maximum,
            "required": required,
        })
    if not expected_evidence:
        raise InventoryError(
            "terminal has no source-mapped compact evidence",
            state="IDENTITY_CONFLICT",
        )

    unmapped_results = [
        {
            "github_path": item["path"],
            "destination_filename": PurePosixPath(item["path"]).name,
            "bytes": item["bytes"],
            "sha256": item["sha256"],
        }
        for item in record["result_files"]
        if PurePosixPath(item["path"]).name not in mapped_names
    ]
    raw_inventory_authorized = (
        evidence_role in RAW_INVENTORY_EVIDENCE_ROLES
        and not any(permissions.values())
        and not any(protected_touches.values())
    )
    return {
        "eligible": True,
        "state": "TERMINAL_BINDING_VERIFIED",
        "snapshot_commit": snapshot_commit,
        "catalog_sha256": catalog_sha256,
        "terminal_index_sha256": index_identity["sha256"],
        "terminal_record_sha256": terminal_record_sha256,
        "route_id": route_id,
        "operation_id": operation_id,
        "run_id": run_id,
        "output_id": output_id,
        "route_commit": route_commit,
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "runtime_spec_sha256": hashlib.sha256(runtime_raw).hexdigest(),
        "closeout_sha256": record["closeout_sha256"],
        "runner_sha256": runner_sha256,
        "closeout_filename": closeout_name,
        "evidence_role": evidence_role,
        "protected_data_permissions": dict(sorted(permissions.items())),
        "protected_data_touched": dict(sorted(protected_touches.items())),
        "raw_inventory_authorized": raw_inventory_authorized,
        "raw_inventory_exclusion_reason": (
            None if raw_inventory_authorized
            else "protected_or_unknown_role_permission_or_touch"
        ),
        "run_root": derive_run_root(route_id, output_id),
        "expected_lifecycle_identity": {
            "schema_version": 1,
            "route_id": route_id,
            "operation_id": operation_id,
            "run_id": run_id,
            "route_commit": route_commit,
            "runner_sha256": runner_sha256,
        },
        "expected_evidence": sorted(
            expected_evidence, key=lambda value: value["source_relpath"]
        ),
        "optional_evidence": sorted(
            optional_evidence, key=lambda value: value["source_relpath"]
        ),
        "unmapped_results": sorted(
            unmapped_results, key=lambda value: value["github_path"]
        ),
        "github_bound_bytes": total_github_bytes,
        "github_result_count": len(record["result_files"]),
        "github_contract_bundle_count": len(record["contract_bundle"]),
        "scientific_completeness": "not_assessed",
    }


def _limits(value: dict[str, int] | None) -> dict[str, int]:
    result = {
        "max_entries": MAX_SCAN_ENTRIES,
        "max_depth": MAX_SCAN_DEPTH,
        "max_relative_path_bytes": MAX_RELATIVE_PATH_BYTES,
        "max_seconds": MAX_SCAN_SECONDS,
    }
    if value is not None:
        if not isinstance(value, dict) or set(value) - set(result):
            raise InventoryError("scan limits have an invalid field contract")
        result.update(value)
    for key, item in result.items():
        if not isinstance(item, int) or isinstance(item, bool) or item < 1:
            raise InventoryError(f"scan limit {key} must be a positive integer")
    return result


def _safe_path_text(relative: str) -> bytes:
    if any(ord(char) < 32 or ord(char) == 127 for char in relative):
        raise _ScanStopped("PATH_CONTROL_CHARACTER")
    try:
        return relative.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise _ScanStopped("PATH_NOT_UTF8") from exc


def _path_chain_has_symlink(path: Path) -> bool:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            return False
        if stat.S_ISLNK(metadata.st_mode):
            return True
    return False


def _same_file_identity(path: Path, expected: os.stat_result) -> bool:
    try:
        observed = path.lstat()
    except OSError:
        return False
    return (
        observed.st_dev == expected.st_dev
        and observed.st_ino == expected.st_ino
        and stat.S_IFMT(observed.st_mode) == stat.S_IFMT(expected.st_mode)
        and observed.st_size == expected.st_size
        and observed.st_mtime_ns == expected.st_mtime_ns
    )


def _walk_metadata(root: Path, limits: dict[str, int]) -> dict[str, Any]:
    started = time.monotonic()
    files: dict[str, dict[str, Any]] = {}
    directories = 0
    total_bytes = 0
    entry_count = 0
    relative_path_bytes = 0
    issues = []
    stack: list[tuple[Path, str, int]] = [(root, "", 0)]
    complete = True
    try:
        while stack:
            if time.monotonic() - started > limits["max_seconds"]:
                raise _ScanStopped("SCAN_TIME_LIMIT")
            directory, prefix, parent_depth = stack.pop()
            try:
                with os.scandir(directory) as iterator:
                    children = list(iterator)
            except OSError as exc:
                raise _ScanStopped(f"DIRECTORY_UNREADABLE:{type(exc).__name__}") from exc
            children.sort(key=lambda item: item.name)
            next_directories = []
            for child in children:
                relative = f"{prefix}/{child.name}" if prefix else child.name
                encoded = _safe_path_text(relative)
                depth = parent_depth + 1
                entry_count += 1
                relative_path_bytes += len(encoded)
                if entry_count > limits["max_entries"]:
                    raise _ScanStopped("ENTRY_LIMIT")
                if relative_path_bytes > limits["max_relative_path_bytes"]:
                    raise _ScanStopped("PATH_BYTE_LIMIT")
                if depth > limits["max_depth"]:
                    raise _ScanStopped("DEPTH_LIMIT")
                try:
                    metadata = child.stat(follow_symlinks=False)
                except OSError as exc:
                    raise _ScanStopped(f"LSTAT_FAILED:{type(exc).__name__}") from exc
                mode = metadata.st_mode
                if stat.S_ISLNK(mode):
                    files[relative] = {
                        "file_type": "symlink", "bytes": 0,
                        "device": metadata.st_dev, "inode": metadata.st_ino,
                    }
                    raise _ScanStopped("SYMLINK_PRESENT")
                if stat.S_ISDIR(mode):
                    directories += 1
                    if depth >= limits["max_depth"]:
                        try:
                            with os.scandir(child.path) as iterator:
                                has_children = next(iterator, None) is not None
                        except OSError as exc:
                            raise _ScanStopped(
                                f"DIRECTORY_UNREADABLE:{type(exc).__name__}"
                            ) from exc
                        if has_children:
                            raise _ScanStopped("DEPTH_LIMIT")
                    else:
                        next_directories.append((Path(child.path), relative, depth))
                elif stat.S_ISREG(mode):
                    total_bytes += metadata.st_size
                    files[relative] = {
                        "file_type": "file", "bytes": metadata.st_size,
                        "device": metadata.st_dev, "inode": metadata.st_ino,
                    }
                else:
                    files[relative] = {
                        "file_type": "special", "bytes": 0,
                        "device": metadata.st_dev, "inode": metadata.st_ino,
                    }
                    raise _ScanStopped("SPECIAL_FILE_PRESENT")
            stack.extend(reversed(next_directories))
    except _ScanStopped as exc:
        complete = False
        issues.append(str(exc))
    return {
        "complete": complete,
        "issues": issues,
        "files": files,
        "entry_count": entry_count,
        "file_count": len(files),
        "directory_count": directories,
        "total_file_bytes": total_bytes,
        "relative_path_bytes": relative_path_bytes,
        "elapsed_seconds": round(time.monotonic() - started, 6),
    }


def _read_nofollow(path: Path, *, maximum: int,
                   expected_metadata: dict[str, Any] | None = None) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise InventoryError(
            f"bounded file read failed: {type(exc).__name__}",
            state="IDENTITY_CONFLICT",
        ) from exc
    try:
        observed = os.fstat(descriptor)
        if not stat.S_ISREG(observed.st_mode) or observed.st_size > maximum:
            raise InventoryError("bounded file read contract failed", state="IDENTITY_CONFLICT")
        if expected_metadata is not None and (
            observed.st_dev != expected_metadata["device"]
            or observed.st_ino != expected_metadata["inode"]
            or observed.st_size != expected_metadata["bytes"]
        ):
            raise InventoryError("file identity changed during scan", state="IDENTITY_CONFLICT")
        chunks = []
        remaining = maximum + 1
        while remaining:
            block = os.read(descriptor, min(1024 * 1024, remaining))
            if not block:
                break
            chunks.append(block)
            remaining -= len(block)
        raw = b"".join(chunks)
        if len(raw) > maximum or len(raw) != observed.st_size:
            raise InventoryError("bounded file size changed", state="IDENTITY_CONFLICT")
        return raw
    finally:
        os.close(descriptor)


def _expected_entry(item: dict[str, Any], state: str,
                    cloud: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "scope": "raw_output",
        "relative_path": item["source_relpath"],
        "artifact_class": "formal_compact_evidence",
        "extension": PurePosixPath(item["source_relpath"]).suffix.lower(),
        "file_type": None if cloud is None else cloud.get("file_type"),
        "bytes": None if cloud is None else cloud.get("bytes"),
        "github_path": item["github_path"],
        "github_bytes": item["bytes"],
        "github_sha256": item["sha256"],
        "cloud_sha256": None if cloud is None else cloud.get("sha256"),
        "identity_basis": (
            "github_sha256" if cloud is None else cloud.get("identity_basis", "metadata_only")
        ),
        "reconciliation_state": state,
        "policy_assessment": "required_formal_evidence",
    }


def _archive_only_entry(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "scope": "github_archive",
        "relative_path": item["destination_filename"],
        "artifact_class": "unmapped_formal_result",
        "extension": PurePosixPath(item["destination_filename"]).suffix.lower(),
        "file_type": None,
        "bytes": None,
        "github_path": item["github_path"],
        "github_bytes": item["bytes"],
        "github_sha256": item["sha256"],
        "cloud_sha256": None,
        "identity_basis": "github_sha256",
        "reconciliation_state": "NOT_INVENTORIED",
        "policy_assessment": "github_result_without_runtime_source_mapping",
    }


def _optional_entry(item: dict[str, Any], state: str,
                    cloud: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "scope": "raw_output",
        "relative_path": item["source_relpath"],
        "artifact_class": "optional_compact_evidence",
        "extension": PurePosixPath(item["source_relpath"]).suffix.lower(),
        "file_type": None if cloud is None else cloud.get("file_type"),
        "bytes": None if cloud is None else cloud.get("bytes"),
        "github_path": None,
        "github_bytes": None,
        "github_sha256": None,
        "cloud_sha256": None,
        "identity_basis": "metadata_only" if cloud is not None else "declaration_only",
        "reconciliation_state": state,
        "policy_assessment": "optional_runtime_evidence_without_archive",
    }


def _inventory(binding: dict[str, Any], *, state: str,
               discovery_completeness: str, entries: list[dict[str, Any]],
               scan: dict[str, Any] | None, limits: dict[str, int],
               issues: list[str]) -> dict[str, Any]:
    entries = sorted(
        entries,
        key=lambda item: (
            item["scope"], item["relative_path"], item["reconciliation_state"]
        ),
    )
    reconciliation = Counter(item["reconciliation_state"] for item in entries)
    policy = Counter(item["policy_assessment"] for item in entries)
    artifact = Counter(item["artifact_class"] for item in entries)
    extensions = Counter(item["extension"] or "<none>" for item in entries)
    identity_basis = Counter(item["identity_basis"] for item in entries)
    identity = {
        key: binding.get(key) for key in (
            "snapshot_commit", "catalog_sha256", "terminal_index_sha256",
            "terminal_record_sha256", "route_id", "operation_id", "run_id",
            "output_id", "route_commit", "manifest_sha256",
            "runtime_spec_sha256", "closeout_sha256", "runner_sha256",
        )
    }
    stable_scan = {
        key: (scan or {}).get(key, 0) for key in (
            "entry_count", "file_count", "directory_count", "total_file_bytes",
            "relative_path_bytes",
        )
    }
    digest_value = {
        "schema_version": 1,
        "identity": identity,
        "state": state,
        "discovery_completeness": discovery_completeness,
        "limits": limits,
        "scan": stable_scan,
        "issues": sorted(set(issues)),
        "entries": entries,
    }
    is_error = state in {"CLOUD_UNAVAILABLE", "IDENTITY_CONFLICT"}
    return {
        "schema_version": 1,
        "ok": not is_error,
        "operation": "cloud-inventory",
        "state": state,
        "exit_code": 3 if is_error else 0,
        "identity": identity,
        "run_root": binding.get("run_root"),
        "scope": "exact_terminal_raw_output",
        "limits": limits,
        "discovery_completeness": discovery_completeness,
        "scientific_completeness": "not_assessed",
        "issues": sorted(set(issues)),
        "scan": {
            **stable_scan,
            "elapsed_seconds": (scan or {}).get("elapsed_seconds", 0),
        },
        "reconciliation_counts": {
            name: reconciliation[name] for name in RECONCILIATION_STATES
        },
        "policy_counts": dict(sorted(policy.items())),
        "artifact_class_counts": dict(sorted(artifact.items())),
        "extension_counts": dict(sorted(extensions.items())),
        "identity_basis_counts": dict(sorted(identity_basis.items())),
        "inventory_sha256": canonical_sha256(digest_value),
        "entry_count": len(entries),
        "_entries": entries,
    }


def _scan_exact_run_root(binding: dict[str, Any], root_value: str | Path, *,
                         cloud_available: bool = True,
                         active_session: bool = False,
                         limits: dict[str, int] | None = None) -> dict[str, Any]:
    """Scan an adapter-owned root. This private core performs no transport."""
    limits = _limits(limits)
    if not isinstance(binding, dict) or binding.get("eligible") is not True:
        return _inventory(
            binding if isinstance(binding, dict) else {},
            state="NOT_INVENTORIED", discovery_completeness="not_inventoried",
            entries=[], scan=None, limits=limits,
            issues=[
                binding.get("reason", "terminal_binding_not_eligible")
                if isinstance(binding, dict) else "terminal_binding_not_eligible"
            ],
        )
    archive_only = [_archive_only_entry(item) for item in binding["unmapped_results"]]

    def scoped_entries(expected_state: str,
                       optional_state: str | None = None) -> list[dict[str, Any]]:
        optional_state = expected_state if optional_state is None else optional_state
        return [
            _expected_entry(item, expected_state)
            for item in binding["expected_evidence"]
        ] + [
            _optional_entry(item, optional_state)
            for item in binding["optional_evidence"]
        ] + archive_only

    if not binding["raw_inventory_authorized"]:
        return _inventory(
            binding, state="NOT_INVENTORIED",
            discovery_completeness="not_inventoried",
            entries=scoped_entries("NOT_INVENTORIED"), scan=None, limits=limits,
            issues=[binding["raw_inventory_exclusion_reason"]],
        )
    if not cloud_available:
        return _inventory(
            binding, state="CLOUD_UNAVAILABLE",
            discovery_completeness="unavailable",
            entries=scoped_entries("CLOUD_UNAVAILABLE"),
            scan=None, limits=limits, issues=["CLOUD_UNAVAILABLE"],
        )
    if active_session:
        return _inventory(
            binding, state="NOT_INVENTORIED",
            discovery_completeness="not_inventoried",
            entries=scoped_entries("NOT_INVENTORIED"),
            scan=None, limits=limits, issues=["ACTIVE_SESSION"],
        )

    root = Path(root_value)
    if not root.is_absolute():
        raise InventoryError(
            "internal scan root must be absolute", state="ARGUMENTS_INVALID",
            exit_code=2,
        )
    try:
        path_chain_has_symlink = _path_chain_has_symlink(root)
    except OSError as exc:
        return _inventory(
            binding, state="CLOUD_UNAVAILABLE",
            discovery_completeness="unavailable",
            entries=scoped_entries("CLOUD_UNAVAILABLE"),
            scan=None, limits=limits,
            issues=[f"PATH_CHAIN_LSTAT_FAILED:{type(exc).__name__}"],
        )
    if path_chain_has_symlink:
        return _inventory(
            binding, state="IDENTITY_CONFLICT",
            discovery_completeness="partial",
            entries=scoped_entries("IDENTITY_CONFLICT"),
            scan=None, limits=limits, issues=["PATH_COMPONENT_SYMLINK"],
        )
    try:
        root_metadata = root.lstat()
    except FileNotFoundError:
        return _inventory(
            binding, state="INVENTORY_READY",
            discovery_completeness="complete",
            entries=scoped_entries("GITHUB_ONLY", "NOT_INVENTORIED"),
            scan=None, limits=limits, issues=[],
        )
    except OSError as exc:
        return _inventory(
            binding, state="CLOUD_UNAVAILABLE",
            discovery_completeness="unavailable",
            entries=scoped_entries("CLOUD_UNAVAILABLE"),
            scan=None, limits=limits,
            issues=[f"ROOT_LSTAT_FAILED:{type(exc).__name__}"],
        )
    if not stat.S_ISDIR(root_metadata.st_mode):
        return _inventory(
            binding, state="IDENTITY_CONFLICT",
            discovery_completeness="partial",
            entries=scoped_entries("IDENTITY_CONFLICT"),
            scan=None, limits=limits, issues=["ROOT_TYPE_INVALID"],
        )

    control_path = root / "control"
    identity_path = control_path / "lifecycle_identity.json"
    try:
        identity_chain_has_symlink = _path_chain_has_symlink(identity_path)
    except OSError as exc:
        return _inventory(
            binding, state="CLOUD_UNAVAILABLE",
            discovery_completeness="unavailable",
            entries=scoped_entries("CLOUD_UNAVAILABLE"),
            scan=None, limits=limits,
            issues=[f"IDENTITY_PATH_LSTAT_FAILED:{type(exc).__name__}"],
        )
    if identity_chain_has_symlink:
        return _inventory(
            binding, state="IDENTITY_CONFLICT",
            discovery_completeness="partial",
            entries=scoped_entries("IDENTITY_CONFLICT"),
            scan=None, limits=limits, issues=["PATH_COMPONENT_SYMLINK"],
        )
    try:
        control_metadata = control_path.lstat()
        if stat.S_ISLNK(control_metadata.st_mode) \
                or not stat.S_ISDIR(control_metadata.st_mode):
            raise InventoryError(
                "control directory type differs", state="IDENTITY_CONFLICT"
            )
        identity_metadata = identity_path.lstat()
        if stat.S_ISLNK(identity_metadata.st_mode) \
                or not stat.S_ISREG(identity_metadata.st_mode):
            raise InventoryError("lifecycle identity type differs", state="IDENTITY_CONFLICT")
        identity_raw = _read_nofollow(
            identity_path, maximum=MAX_LIFECYCLE_IDENTITY_BYTES,
            expected_metadata={
                "device": identity_metadata.st_dev,
                "inode": identity_metadata.st_ino,
                "bytes": identity_metadata.st_size,
            },
        )
        observed_identity = _json_object(identity_raw, "lifecycle identity")
        if observed_identity != binding["expected_lifecycle_identity"]:
            raise InventoryError("lifecycle identity differs", state="IDENTITY_CONFLICT")
    except (FileNotFoundError, OSError, InventoryError) as exc:
        return _inventory(
            binding, state="IDENTITY_CONFLICT",
            discovery_completeness="partial",
            entries=scoped_entries("IDENTITY_CONFLICT"),
            scan=None, limits=limits,
            issues=[f"LIFECYCLE_IDENTITY_INVALID:{type(exc).__name__}"],
        )

    scan = _walk_metadata(root, limits)
    complete = scan["complete"]
    files = scan["files"]
    entries = []
    expected_paths = set()
    identity_conflict = any(
        issue in {"SYMLINK_PRESENT", "SPECIAL_FILE_PRESENT"}
        for issue in scan["issues"]
    )
    try:
        identity_stable = _read_nofollow(
            identity_path, maximum=MAX_LIFECYCLE_IDENTITY_BYTES,
            expected_metadata={
                "device": identity_metadata.st_dev,
                "inode": identity_metadata.st_ino,
                "bytes": identity_metadata.st_size,
            },
        ) == identity_raw
    except InventoryError:
        identity_stable = False
    if not _same_file_identity(root, root_metadata) \
            or not _same_file_identity(control_path, control_metadata) \
            or not _same_file_identity(identity_path, identity_metadata) \
            or not identity_stable:
        complete = False
        identity_conflict = True
        scan["complete"] = False
        scan["issues"].append("SCAN_ROOT_IDENTITY_CHANGED")
    for item in binding["expected_evidence"]:
        relative = item["source_relpath"]
        expected_paths.add(relative)
        metadata = files.get(relative)
        if metadata is None:
            entries.append(_expected_entry(
                item, "GITHUB_ONLY" if complete else "NOT_INVENTORIED"
            ))
            continue
        if metadata["file_type"] != "file":
            entries.append(_expected_entry(item, "IDENTITY_CONFLICT", metadata))
            identity_conflict = True
            continue
        try:
            raw = _read_nofollow(
                root / PurePosixPath(relative), maximum=item["max_bytes"],
                expected_metadata=metadata,
            )
            observed_sha = hashlib.sha256(raw).hexdigest()
            cloud = {
                **metadata,
                "sha256": observed_sha,
                "identity_basis": "sha256",
            }
            if len(raw) == item["bytes"] and observed_sha == item["sha256"]:
                entries.append(_expected_entry(item, "MATCHED", cloud))
            else:
                entries.append(_expected_entry(item, "IDENTITY_CONFLICT", cloud))
                identity_conflict = True
        except InventoryError:
            entries.append(_expected_entry(item, "IDENTITY_CONFLICT", metadata))
            identity_conflict = True

    for item in binding["optional_evidence"]:
        relative = item["source_relpath"]
        expected_paths.add(relative)
        metadata = files.get(relative)
        if metadata is None:
            entries.append(_optional_entry(item, "NOT_INVENTORIED"))
        elif metadata["file_type"] != "file":
            entries.append(_optional_entry(item, "IDENTITY_CONFLICT", metadata))
            identity_conflict = True
        else:
            entries.append(_optional_entry(
                item, "CLOUD_ONLY" if complete else "NOT_INVENTORIED", metadata
            ))

    for relative, metadata in sorted(files.items()):
        if relative in expected_paths:
            continue
        policy = (
            "expected_control_identity"
            if relative == "control/lifecycle_identity.json"
            else "expected_cloud_only_raw_artifact"
        )
        entry_state = (
            "IDENTITY_CONFLICT"
            if metadata["file_type"] in {"symlink", "special"}
            else ("CLOUD_ONLY" if complete else "NOT_INVENTORIED")
        )
        entries.append({
            "scope": "raw_output",
            "relative_path": relative,
            "artifact_class": (
                "control_identity" if policy == "expected_control_identity"
                else "raw_artifact"
            ),
            "extension": PurePosixPath(relative).suffix.lower(),
            "file_type": metadata["file_type"],
            "bytes": metadata["bytes"],
            "github_path": None,
            "github_bytes": None,
            "github_sha256": None,
            "cloud_sha256": (
                hashlib.sha256(identity_raw).hexdigest()
                if relative == "control/lifecycle_identity.json" else None
            ),
            "identity_basis": (
                "verified_control_identity"
                if relative == "control/lifecycle_identity.json" else "metadata_only"
            ),
            "reconciliation_state": entry_state,
            "policy_assessment": policy,
        })
    entries.extend(archive_only)
    state = "IDENTITY_CONFLICT" if identity_conflict else (
        "INVENTORY_READY" if complete else "INVENTORY_INCOMPLETE"
    )
    return _inventory(
        binding, state=state,
        discovery_completeness="complete" if complete else "partial",
        entries=entries, scan=scan, limits=limits, issues=scan["issues"],
    )


def inventory_summary(inventory: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in inventory.items() if key != "_entries"}


def inventory_query(inventory: dict[str, Any], *, inventory_sha256: str,
                    reconciliation_states: list[str] | None = None,
                    terms: list[str] | None = None, cursor: str | None = None,
                    limit: int = 20) -> dict[str, Any]:
    inventory_sha256 = _require_sha(
        inventory_sha256, SHA256, "inventory_sha256"
    )
    if inventory.get("inventory_sha256") != inventory_sha256:
        raise InventoryError(
            "inventory identity differs", state="INVENTORY_DRIFT", exit_code=3
        )
    states = list(RECONCILIATION_STATES) \
        if reconciliation_states is None else reconciliation_states
    if not isinstance(states, list) or any(state not in RECONCILIATION_STATES for state in states):
        raise InventoryError(
            "reconciliation_states is invalid", state="ARGUMENTS_INVALID", exit_code=2
        )
    states = sorted(set(states))
    terms = [] if terms is None else terms
    if not isinstance(terms, list) or len(terms) > 8 \
            or any(not isinstance(term, str) or not term or len(term) > 128 for term in terms):
        raise InventoryError("terms is invalid", state="ARGUMENTS_INVALID", exit_code=2)
    terms = [term.casefold() for term in terms]
    if not isinstance(limit, int) or isinstance(limit, bool) \
            or not 1 <= limit <= MAX_QUERY_ENTRIES:
        raise InventoryError("limit must be in [1, 100]", state="ARGUMENTS_INVALID", exit_code=2)
    candidates = []
    for entry in inventory.get("_entries", []):
        if entry["reconciliation_state"] not in states:
            continue
        searchable = canonical_bytes(entry).decode("utf-8").casefold()
        if all(term in searchable for term in terms):
            candidates.append(entry)
    identity = inventory["identity"]
    query_sha256 = canonical_sha256({
        "snapshot_commit": identity["snapshot_commit"],
        "terminal_record_sha256": identity["terminal_record_sha256"],
        "manifest_sha256": identity["manifest_sha256"],
        "runtime_spec_sha256": identity["runtime_spec_sha256"],
        "inventory_sha256": inventory_sha256,
        "reconciliation_states": states,
        "terms": terms,
    })
    offset = 0
    if cursor is not None:
        try:
            decoded = convirctl.decode_repo_cursor(
                cursor, CURSOR_OPERATION, query_sha256
            )
        except convirctl.ControlError as exc:
            raise InventoryError(
                str(exc), state=exc.state, exit_code=exc.exit_code
            ) from exc
        if decoded["commit"] != identity["snapshot_commit"] \
                or decoded["object_id"] != inventory_sha256[:40]:
            raise InventoryError(
                "inventory cursor identity differs",
                state="REPO_CURSOR_IDENTITY_MISMATCH",
            )
        offset = decoded["position"]
    if offset > len(candidates) or (offset == len(candidates) and offset != 0):
        raise InventoryError("inventory cursor is outside results", state="REPO_CURSOR_INVALID")
    selected = candidates[offset:offset + limit]
    while True:
        end = offset + len(selected)
        complete = end == len(candidates)
        next_cursor = None if complete else convirctl.encode_repo_cursor(
            CURSOR_OPERATION,
            identity["snapshot_commit"],
            query_sha256,
            end,
            inventory_sha256[:40],
        )
        value = {
            "schema_version": 1,
            "ok": True,
            "operation": "cloud-inventory-query",
            "state": "INVENTORY_ENTRIES_OK",
            "exit_code": 0,
            "snapshot_commit": identity["snapshot_commit"],
            "terminal_record_sha256": identity["terminal_record_sha256"],
            "inventory_sha256": inventory_sha256,
            "query_sha256": query_sha256,
            "reconciliation_states": states,
            "terms": terms,
            "offset": offset,
            "returned_count": len(selected),
            "total_count": len(candidates),
            "entries": selected,
            "page_sha256": canonical_sha256(selected),
            "complete": complete,
            "has_more": not complete,
            "next_cursor": next_cursor,
            "discovery_completeness": inventory["discovery_completeness"],
            "scientific_completeness": "not_assessed",
        }
        if len(canonical_bytes(value)) <= MAX_QUERY_VALUE_BYTES:
            return value
        if len(selected) <= 1:
            raise InventoryError(
                "one inventory entry exceeds the query response budget",
                state="ENTRY_TOO_LARGE",
            )
        selected.pop()
