#!/usr/bin/env python3
"""Identity-bound, transport-free cloud evidence inventory core."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any


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
MAX_TEXT_FILE_BYTES = 1024 * 1024
MAX_TEXT_PAGE_BYTES = 8 * 1024
MAX_RAW_ARTIFACT_MANIFEST_BYTES = 16 * 1024 * 1024
MAX_REMOTE_REQUEST_BYTES = 4 * 1024 * 1024
MAX_REMOTE_RESPONSE_BYTES = 64 * 1024
REMOTE_TMUX = "/usr/bin/tmux"
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
TEXT_READ_SUFFIXES = {".json", ".jsonl", ".csv", ".md", ".txt"}
TEXT_READ_ARTIFACT_CLASSES = {
    "formal_compact_evidence", "optional_compact_evidence", "raw_artifact",
}
RAW_ARTIFACT_RECEIPT_SUFFIX = "_raw_artifact_receipt.json"
RAW_ARTIFACT_MANIFEST_RELPATH = "control/raw_artifact_manifest.jsonl"
RAW_ARTIFACT_SCOPE_ROOTS = ("contract", "workload")
RAW_ARTIFACT_EXCLUDED_PATHS = (
    "control", "heartbeat.json", "runtime.log", "status.txt",
)
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
    def __init__(self, issue: str, *, state: str | None = None):
        super().__init__(issue)
        self.state = state


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
    import convir_evidence_catalog as catalog

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
        message = (
            "terminal GitHub evidence exceeds its aggregate bound"
            if maximum < MAX_GITHUB_FILE_BYTES
            else f"GitHub evidence blob exceeds its bound: {relpath}"
        )
        raise InventoryError(
            message,
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


def _raw_artifact_receipt_filename(closeout_filename: str) -> str:
    suffix = "_closeout.json"
    if not isinstance(closeout_filename, str) or not closeout_filename.endswith(suffix):
        raise InventoryError(
            "closeout filename cannot derive raw artifact receipt",
            state="IDENTITY_CONFLICT",
        )
    return closeout_filename[:-len(suffix)] + RAW_ARTIFACT_RECEIPT_SUFFIX


def _parse_raw_artifact_manifest(
    raw: bytes, receipt: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    if (
        len(raw) > MAX_RAW_ARTIFACT_MANIFEST_BYTES
        or hashlib.sha256(raw).hexdigest() != receipt["manifest_sha256"]
    ):
        raise InventoryError(
            "raw artifact manifest SHA-256 differs", state="IDENTITY_CONFLICT"
        )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise InventoryError(
            "raw artifact manifest is not UTF-8", state="IDENTITY_CONFLICT"
        ) from exc
    records: dict[str, dict[str, Any]] = {}
    previous = None
    category_counts = Counter()
    total_bytes = 0
    lines = text.splitlines()
    if any(not line for line in lines):
        raise InventoryError(
            "raw artifact manifest contains a blank line", state="IDENTITY_CONFLICT"
        )
    for number, line in enumerate(lines, start=1):
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise InventoryError(
                f"raw artifact manifest line {number} is invalid JSON",
                state="IDENTITY_CONFLICT",
            ) from exc
        expected = {
            "schema_version", "relative_path", "artifact_class", "bytes", "sha256",
        }
        if (
            not isinstance(item, dict) or set(item) != expected
            or item["schema_version"] != 2
        ):
            raise InventoryError(
                f"raw artifact manifest line {number} has an invalid schema",
                state="IDENTITY_CONFLICT",
            )
        relative = _require_relpath(item["relative_path"], "raw manifest relative_path")
        root = PurePosixPath(relative).parts[0]
        if (
            root not in RAW_ARTIFACT_SCOPE_ROOTS
            or item["artifact_class"] != f"{root}_output"
        ):
            raise InventoryError(
                f"raw artifact manifest line {number} is outside its stable scope",
                state="IDENTITY_CONFLICT",
            )
        size = item["bytes"]
        if (
            not isinstance(size, int) or isinstance(size, bool) or size < 0
            or not isinstance(item["sha256"], str)
            or not SHA256.fullmatch(item["sha256"])
        ):
            raise InventoryError(
                f"raw artifact manifest line {number} has an invalid identity",
                state="IDENTITY_CONFLICT",
            )
        if relative in records or (previous is not None and relative <= previous):
            raise InventoryError(
                "raw artifact manifest paths are duplicate or unsorted",
                state="IDENTITY_CONFLICT",
            )
        records[relative] = item
        previous = relative
        category_counts[item["artifact_class"]] += 1
        total_bytes += size
    expected_categories = {
        f"{root}_output": receipt["category_counts"][f"{root}_output"]
        for root in RAW_ARTIFACT_SCOPE_ROOTS
    }
    if (
        len(records) != receipt["entry_count"]
        or total_bytes != receipt["total_bytes"]
        or dict(category_counts) != {
            key: value for key, value in expected_categories.items() if value
        }
    ):
        raise InventoryError(
            "raw artifact manifest summary differs from its receipt",
            state="IDENTITY_CONFLICT",
        )
    return records


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
    import convir_evidence_catalog as catalog
    import convir_ops_mcp
    import prepare_terminal_archive
    import route_runtime_contract

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
    evidence_root = f"experience_docx/experiment_logs/{route_id}"
    closeout_path = _require_relpath(record["closeout_path"], "closeout_path")
    conclusion_path = _require_relpath(record["conclusion_path"], "conclusion_path")
    contract_path = _require_relpath(record["contract_path"], "contract_path")
    if not contract_path.startswith("experience_docx/experiment_cards/") \
            or not contract_path.endswith(".md"):
        raise InventoryError(
            "terminal contract path is not canonical",
            state="IDENTITY_CONFLICT",
        )
    if PurePosixPath(closeout_path).parent.as_posix() != evidence_root \
            or PurePosixPath(conclusion_path).parent.as_posix() != evidence_root:
        raise InventoryError(
            "terminal closeout or conclusion is outside its canonical evidence root",
            state="IDENTITY_CONFLICT",
        )
    for item in record["result_files"]:
        result_path = _require_relpath(item["path"], "result_files[].path")
        if PurePosixPath(result_path).parent.as_posix() != evidence_root:
            raise InventoryError(
                "terminal result is outside its canonical evidence root",
                state="IDENTITY_CONFLICT",
            )
    runtime_source_path = f"{RUNTIME_SPEC_PREFIX}{operation_id}.json"
    launch_contract_root = f"{evidence_root}/launch_contract/{operation_id}"
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

    contract_raw = read_github_blob(
        contract_path,
        expected_sha256=record["contract_sha256"],
    )
    closeout_raw = read_github_blob(
        closeout_path,
        expected_sha256=record["closeout_sha256"],
    )
    conclusion_raw = read_github_blob(
        conclusion_path,
        expected_sha256=record["conclusion_sha256"],
    )

    bundle_payloads: dict[str, bytes] = {}
    bundle_archive_paths: dict[str, str] = {}
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
        bundle_archive_paths[source_path] = _require_relpath(
            item["path"], "contract_bundle[].path"
        )
    result_payloads: dict[str, bytes] = {}
    for item in record["result_files"]:
        raw = read_github_blob(
            item["path"],
            expected_bytes=item["bytes"],
            expected_sha256=item["sha256"],
        )
        name = PurePosixPath(item["path"]).name
        if name in result_payloads:
            raise InventoryError(
                "terminal result filenames are ambiguous",
                state="IDENTITY_CONFLICT",
            )
        result_payloads[name] = raw

    manifest_raw = bundle_payloads.get(MANIFEST_SOURCE_PATH)
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
    if manifest.get("route_card_relpath") != contract_path:
        raise InventoryError(
            "manifest route card differs from terminal contract path",
            state="IDENTITY_CONFLICT",
        )
    operations = manifest.get("operations")
    if not isinstance(operations, dict):
        raise InventoryError(
            "manifest operations are invalid", state="IDENTITY_CONFLICT"
        )
    operation = operations.get(operation_id)
    if not isinstance(operation, dict):
        raise InventoryError("manifest operation is absent", state="IDENTITY_CONFLICT")
    output_id = _require_token(operation.get("output_id"), "output_id")
    mode = _require_token(operation.get("mode"), "mode")
    if output_id != run_id:
        raise InventoryError(
            "manifest output_id differs from terminal run_id",
            state="IDENTITY_CONFLICT",
        )
    closeout_name = PurePosixPath(closeout_path).name
    if operation.get("closeout_filename") != closeout_name:
        raise InventoryError(
            "manifest closeout filename differs from terminal record",
            state="IDENTITY_CONFLICT",
        )
    terminal_tuple = {
        "state": record["state"],
        "decision": record["decision"],
        "authorizes": record["authorizes"],
    }
    if terminal_tuple not in operation.get("allowed_terminal_tuples", []):
        raise InventoryError(
            "terminal tuple is not authorized by the archived manifest",
            state="IDENTITY_CONFLICT",
        )
    try:
        runtime = route_runtime_contract.validate_runtime_spec(
            runtime, manifest, operation_id
        )
    except route_runtime_contract.ContractError as exc:
        raise InventoryError(
            f"archived runtime spec is invalid: {exc}",
            state="IDENTITY_CONFLICT",
        ) from exc
    scientific_relpaths = manifest.get("scientific_contract_relpaths")
    if not isinstance(scientific_relpaths, dict):
        raise InventoryError(
            "manifest scientific contract paths are invalid",
            state="IDENTITY_CONFLICT",
        )
    canonical_bundle = {
        "manifest.json": MANIFEST_SOURCE_PATH,
        "route_note.md": manifest.get("route_card_relpath"),
        "experiment_spec.json": manifest.get("experiment_spec_relpath"),
        "program_contract.json": manifest.get("program_contract_relpath"),
        "scientific_contract.json": scientific_relpaths.get(operation_id),
        "runtime_spec.json": runtime_source_path,
        "asset_manifest.json": runtime.get("asset_manifest_relpath"),
        "capability_profile.json": runtime["engineering_contract"].get(
            "capability_profile_relpath"
        ),
        "precision_certificate.json": runtime["precision_contract"].get(
            "certificate_relpath"
        ),
    }
    required_bundle_names = {
        "manifest.json", "route_note.md", "experiment_spec.json",
        "program_contract.json", "scientific_contract.json", "runtime_spec.json",
    }
    if any(
        not isinstance(canonical_bundle[name], str)
        for name in required_bundle_names
    ):
        raise InventoryError(
            "canonical launch contract bundle is incomplete",
            state="IDENTITY_CONFLICT",
        )
    expected_bundle_paths: dict[str, str] = {}
    for archive_name, source_path in canonical_bundle.items():
        if source_path is None:
            continue
        source_path = _require_relpath(
            source_path, f"canonical bundle source {archive_name}"
        )
        if source_path in expected_bundle_paths:
            raise InventoryError(
                "canonical launch contract sources are ambiguous",
                state="IDENTITY_CONFLICT",
            )
        expected_bundle_paths[source_path] = \
            f"{launch_contract_root}/{archive_name}"
    if bundle_archive_paths != expected_bundle_paths:
        raise InventoryError(
            "terminal contract bundle paths are not canonical",
            state="IDENTITY_CONFLICT",
        )
    if bundle_payloads.get(contract_path) != contract_raw:
        raise InventoryError(
            "archived route note differs from the terminal contract",
            state="IDENTITY_CONFLICT",
        )

    closeout = _json_object(closeout_raw, "archived closeout")
    if closeout.get("schema_version") != 2:
        raise InventoryError(
            "closeout schema differs", state="IDENTITY_CONFLICT"
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
    try:
        prepare_terminal_archive.validate_conclusion(
            conclusion_raw, conclusion_path, closeout
        )
    except prepare_terminal_archive.TerminalArchiveError as exc:
        raise InventoryError(
            f"archived conclusion is invalid: {exc}",
            state="IDENTITY_CONFLICT",
        ) from exc
    conclusion_schema_version = conclusion.get("schema_version")
    if conclusion_schema_version is not None and (
        type(conclusion_schema_version) is not int
        or conclusion_schema_version not in {1, 2}
    ):
        raise InventoryError(
            "conclusion schema is unsupported", state="IDENTITY_CONFLICT"
        )
    conclusion_schema_state = {
        None: "LEGACY_UNVERSIONED",
        1: "LEGACY_V1",
        2: "CURRENT_V2",
    }[conclusion_schema_version]
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
    raw_receipt_name = _raw_artifact_receipt_filename(closeout_name)
    raw_receipt_candidates = sorted(
        name for name in result_payloads if name.endswith(RAW_ARTIFACT_RECEIPT_SUFFIX)
    )
    if raw_receipt_candidates and raw_receipt_candidates != [raw_receipt_name]:
        raise InventoryError(
            "raw artifact receipt filename is ambiguous or noncanonical",
            state="IDENTITY_CONFLICT",
        )
    raw_artifact_receipt = None
    raw_artifact_receipt_record = result_by_name.get(raw_receipt_name)
    if raw_artifact_receipt_record is not None:
        try:
            raw_artifact_receipt = prepare_terminal_archive.validate_raw_artifact_receipt(
                result_payloads[raw_receipt_name], raw_artifact_receipt_record["path"],
                closeout, closeout_name,
            )
        except prepare_terminal_archive.TerminalArchiveError as exc:
            raise InventoryError(
                f"raw artifact receipt is invalid: {exc}",
                state="IDENTITY_CONFLICT",
            ) from exc
    expected_evidence = []
    optional_evidence = []
    mapped_names = set()
    declarations = runtime.get("evidence_files")
    for item in declarations:
        source_relpath = item["source_relpath"]
        destination = item["destination_filename"]
        required = item["required"]
        maximum = item["max_bytes"]
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
    if raw_artifact_receipt_record is not None:
        mapped_names.add(raw_receipt_name)
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
        "mode": mode,
        "session": convir_ops_mcp.derive_session(
            route_id, mode, route_commit, output_id
        ),
        "route_commit": route_commit,
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "runtime_spec_sha256": hashlib.sha256(runtime_raw).hexdigest(),
        "closeout_sha256": record["closeout_sha256"],
        "terminal_schema_version": record["schema_version"],
        "closeout_schema_version": closeout["schema_version"],
        "conclusion_schema_version": conclusion_schema_version,
        "conclusion_schema_state": conclusion_schema_state,
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
        "raw_terminal_seal": (
            "manifest_pending_cloud_verification"
            if raw_artifact_receipt is not None else "legacy_unsealed"
        ),
        "raw_artifact_receipt": raw_artifact_receipt,
        "raw_artifact_receipt_github_path": (
            None if raw_artifact_receipt_record is None
            else raw_artifact_receipt_record["path"]
        ),
        "raw_artifact_receipt_sha256": (
            None if raw_artifact_receipt_record is None
            else raw_artifact_receipt_record["sha256"]
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
            expected_evidence,
            key=lambda value: (
                value["source_relpath"], value["destination_filename"]
            ),
        ),
        "optional_evidence": sorted(
            optional_evidence,
            key=lambda value: (
                value["source_relpath"], value["destination_filename"]
            ),
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
    hard = {
        "max_entries": MAX_SCAN_ENTRIES,
        "max_depth": MAX_SCAN_DEPTH,
        "max_relative_path_bytes": MAX_RELATIVE_PATH_BYTES,
        "max_seconds": MAX_SCAN_SECONDS,
    }
    result = dict(hard)
    if value is not None:
        if not isinstance(value, dict) or set(value) - set(result):
            raise InventoryError(
                "scan limits have an invalid field contract",
                state="ARGUMENTS_INVALID", exit_code=2,
            )
        for key, item in value.items():
            if not isinstance(item, int) or isinstance(item, bool) \
                    or not 1 <= item <= hard[key]:
                raise InventoryError(
                    f"scan limit {key} must be in [1, {hard[key]}]",
                    state="ARGUMENTS_INVALID", exit_code=2,
                )
        result.update(value)
    return result


def _safe_path_text(relative: str) -> bytes:
    if any(ord(char) < 32 or ord(char) == 127 for char in relative):
        raise _ScanStopped(
            "PATH_CONTROL_CHARACTER", state="IDENTITY_CONFLICT"
        )
    try:
        return relative.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise _ScanStopped("PATH_NOT_UTF8", state="IDENTITY_CONFLICT") from exc


def _directory_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )


def _file_flags() -> int:
    return os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)


def _file_type(mode: int) -> str:
    if stat.S_ISDIR(mode):
        return "directory"
    if stat.S_ISREG(mode):
        return "file"
    if stat.S_ISLNK(mode):
        return "symlink"
    return "special"


def _stat_metadata(observed: os.stat_result) -> dict[str, Any]:
    file_type = _file_type(observed.st_mode)
    return {
        "file_type": file_type,
        "bytes": observed.st_size if file_type == "file" else 0,
        "device": observed.st_dev,
        "inode": observed.st_ino,
        "mode_type": stat.S_IFMT(observed.st_mode),
        "stat_size": observed.st_size,
        "mtime_ns": observed.st_mtime_ns,
        "ctime_ns": observed.st_ctime_ns,
    }


def _metadata_matches(expected: dict[str, Any], observed: os.stat_result, *,
                      stable: bool) -> bool:
    actual = _stat_metadata(observed)
    fields = ["device", "inode", "mode_type"]
    if stable:
        fields.extend(["stat_size", "mtime_ns", "ctime_ns"])
    elif expected.get("file_type") == "file":
        fields.append("stat_size")
    return all(expected.get(field) == actual[field] for field in fields)


def _access_error(message: str, exc: OSError) -> InventoryError:
    identity_errnos = {
        errno.ENOENT, errno.ENOTDIR, errno.ELOOP, errno.EISDIR,
    }
    state = "IDENTITY_CONFLICT" if exc.errno in identity_errnos \
        else "CLOUD_UNAVAILABLE"
    return InventoryError(
        f"{message}: {type(exc).__name__}", state=state, exit_code=3
    )


def _fstat_or_error(descriptor: int, message: str) -> os.stat_result:
    try:
        return os.fstat(descriptor)
    except OSError as exc:
        raise _access_error(message, exc) from exc


def _open_directory_at(parent_fd: int, name: str,
                       expected: dict[str, Any]) -> int:
    try:
        descriptor = os.open(name, _directory_flags(), dir_fd=parent_fd)
    except OSError as exc:
        raise _access_error("directory open failed", exc) from exc
    try:
        observed = os.fstat(descriptor)
    except OSError as exc:
        os.close(descriptor)
        raise _access_error("directory identity read failed", exc) from exc
    if not stat.S_ISDIR(observed.st_mode) \
            or not _metadata_matches(expected, observed, stable=True):
        os.close(descriptor)
        raise InventoryError(
            "directory identity changed during inventory",
            state="IDENTITY_CONFLICT",
        )
    return descriptor


def _open_adapter_root(root: Path) -> tuple[int, dict[str, Any]] | None:
    if not root.is_absolute() or ".." in root.parts:
        raise InventoryError(
            "internal scan root must be an absolute normalized path",
            state="ARGUMENTS_INVALID", exit_code=2,
        )
    try:
        descriptor = os.open("/", _directory_flags())
    except OSError as exc:
        raise _access_error("filesystem root open failed", exc) from exc
    try:
        for part in root.parts[1:]:
            parent_before = _fstat_or_error(
                descriptor, "root parent identity read failed"
            )
            try:
                observed = os.stat(part, dir_fd=descriptor, follow_symlinks=False)
            except FileNotFoundError:
                parent_after = _fstat_or_error(
                    descriptor, "root parent identity reread failed"
                )
                parent_metadata = _stat_metadata(parent_before)
                if not _metadata_matches(parent_metadata, parent_after, stable=True):
                    raise InventoryError(
                        "root parent changed while proving absence",
                        state="IDENTITY_CONFLICT",
                    )
                os.close(descriptor)
                return None
            except OSError as exc:
                raise _access_error("root path stat failed", exc) from exc
            metadata = _stat_metadata(observed)
            if metadata["file_type"] != "directory":
                raise InventoryError(
                    "root path contains a non-directory component",
                    state="IDENTITY_CONFLICT",
                )
            child = _open_directory_at(descriptor, part, metadata)
            os.close(descriptor)
            descriptor = child
        return descriptor, _stat_metadata(_fstat_or_error(
            descriptor, "root identity read failed"
        ))
    except Exception:
        os.close(descriptor)
        raise


def _read_file_at(parent_fd: int, name: str, *, maximum: int,
                  expected: dict[str, Any]) -> bytes:
    try:
        descriptor = os.open(name, _file_flags(), dir_fd=parent_fd)
    except OSError as exc:
        raise _access_error("bounded file open failed", exc) from exc
    try:
        try:
            before = os.fstat(descriptor)
        except OSError as exc:
            raise _access_error("bounded file identity read failed", exc) from exc
        if not stat.S_ISREG(before.st_mode) or before.st_size > maximum \
                or not _metadata_matches(expected, before, stable=True):
            raise InventoryError(
                "bounded file identity differs", state="IDENTITY_CONFLICT"
            )
        chunks = []
        remaining = maximum + 1
        while remaining:
            try:
                block = os.read(descriptor, min(1024 * 1024, remaining))
            except OSError as exc:
                raise _access_error("bounded file read failed", exc) from exc
            if not block:
                break
            chunks.append(block)
            remaining -= len(block)
        raw = b"".join(chunks)
        try:
            after = os.fstat(descriptor)
        except OSError as exc:
            raise _access_error("bounded file identity reread failed", exc) from exc
        if len(raw) > maximum or len(raw) != before.st_size \
                or not _metadata_matches(_stat_metadata(before), after, stable=True):
            raise InventoryError(
                "bounded file changed during read", state="IDENTITY_CONFLICT"
            )
        return raw
    finally:
        os.close(descriptor)


def _read_relative_file(root_fd: int, relative: str, *, maximum: int,
                        expected: dict[str, Any],
                        nodes: dict[str, dict[str, Any]]) -> bytes:
    parts = PurePosixPath(relative).parts
    descriptor = os.dup(root_fd)
    try:
        prefix = []
        for part in parts[:-1]:
            prefix.append(part)
            directory = "/".join(prefix)
            metadata = nodes.get(directory)
            if metadata is None or metadata.get("file_type") != "directory":
                raise InventoryError(
                    "formal evidence ancestor identity differs",
                    state="IDENTITY_CONFLICT",
                )
            child = _open_directory_at(descriptor, part, metadata)
            os.close(descriptor)
            descriptor = child
        return _read_file_at(
            descriptor, parts[-1], maximum=maximum, expected=expected
        )
    finally:
        os.close(descriptor)


def _read_lifecycle_identity(root_fd: int) \
        -> tuple[bytes, dict[str, Any], dict[str, Any]]:
    try:
        control_stat = os.stat("control", dir_fd=root_fd, follow_symlinks=False)
    except OSError as exc:
        raise _access_error("control directory stat failed", exc) from exc
    control = _stat_metadata(control_stat)
    if control["file_type"] != "directory":
        raise InventoryError(
            "control directory type differs", state="IDENTITY_CONFLICT"
        )
    control_fd = _open_directory_at(root_fd, "control", control)
    try:
        try:
            identity_stat = os.stat(
                "lifecycle_identity.json", dir_fd=control_fd,
                follow_symlinks=False,
            )
        except OSError as exc:
            raise _access_error(
                "lifecycle identity stat failed", exc
            ) from exc
        identity = _stat_metadata(identity_stat)
        if identity["file_type"] != "file":
            raise InventoryError(
                "lifecycle identity type differs", state="IDENTITY_CONFLICT"
            )
        raw = _read_file_at(
            control_fd, "lifecycle_identity.json",
            maximum=MAX_LIFECYCLE_IDENTITY_BYTES, expected=identity,
        )
        return raw, control, identity
    finally:
        os.close(control_fd)


def _walk_metadata(root_fd: int, root_metadata: dict[str, Any],
                   limits: dict[str, int]) -> dict[str, Any]:
    started = time.monotonic()
    deadline = started + limits["max_seconds"]
    nodes: dict[str, dict[str, Any]] = {}
    directories = 0
    total_bytes = 0
    entry_count = 0
    relative_path_bytes = 0

    def stop_for_os(issue: str, exc: OSError) -> None:
        error = _access_error(issue, exc)
        raise _ScanStopped(str(error), state=error.state) from exc

    def check_deadline() -> None:
        if time.monotonic() > deadline:
            raise _ScanStopped("SCAN_TIME_LIMIT")

    def directory_has_children(descriptor: int,
                               expected: dict[str, Any]) -> bool:
        check_deadline()
        try:
            with os.scandir(descriptor) as iterator:
                child = next(iterator, None)
        except OSError as exc:
            stop_for_os("DIRECTORY_UNREADABLE", exc)
        after = os.fstat(descriptor)
        if not _metadata_matches(expected, after, stable=True):
            raise _ScanStopped(
                "DIRECTORY_CHANGED", state="IDENTITY_CONFLICT"
            )
        return child is not None

    def walk(descriptor: int, prefix: str, parent_depth: int,
             expected: dict[str, Any]) -> None:
        nonlocal directories, total_bytes, entry_count, relative_path_bytes
        check_deadline()
        if not _metadata_matches(expected, os.fstat(descriptor), stable=True):
            raise _ScanStopped(
                "DIRECTORY_CHANGED", state="IDENTITY_CONFLICT"
            )
        children: list[tuple[str, str, int, dict[str, Any]]] = []
        try:
            with os.scandir(descriptor) as iterator:
                while True:
                    check_deadline()
                    try:
                        child = next(iterator)
                    except StopIteration:
                        break
                    if entry_count >= limits["max_entries"]:
                        raise _ScanStopped("ENTRY_LIMIT")
                    relative = f"{prefix}/{child.name}" if prefix else child.name
                    encoded = _safe_path_text(relative)
                    if relative_path_bytes + len(encoded) \
                            > limits["max_relative_path_bytes"]:
                        raise _ScanStopped("PATH_BYTE_LIMIT")
                    depth = parent_depth + 1
                    if depth > limits["max_depth"]:
                        raise _ScanStopped("DEPTH_LIMIT")
                    try:
                        observed = child.stat(follow_symlinks=False)
                    except OSError as exc:
                        stop_for_os("LSTAT_FAILED", exc)
                    metadata = _stat_metadata(observed)
                    entry_count += 1
                    relative_path_bytes += len(encoded)
                    nodes[relative] = metadata
                    children.append((child.name, relative, depth, metadata))
                    if metadata["file_type"] == "file":
                        total_bytes += metadata["bytes"]
                    elif metadata["file_type"] == "directory":
                        directories += 1
                    elif metadata["file_type"] == "symlink":
                        raise _ScanStopped(
                            "SYMLINK_PRESENT", state="IDENTITY_CONFLICT"
                        )
                    else:
                        raise _ScanStopped(
                            "SPECIAL_FILE_PRESENT", state="IDENTITY_CONFLICT"
                        )
        except _ScanStopped:
            raise
        except OSError as exc:
            stop_for_os("DIRECTORY_UNREADABLE", exc)
        for name, relative, depth, metadata in sorted(children):
            if metadata["file_type"] != "directory":
                continue
            try:
                child_fd = _open_directory_at(descriptor, name, metadata)
            except InventoryError as exc:
                raise _ScanStopped(str(exc), state=exc.state) from exc
            try:
                if depth >= limits["max_depth"]:
                    if directory_has_children(child_fd, metadata):
                        raise _ScanStopped("DEPTH_LIMIT")
                else:
                    walk(child_fd, relative, depth, metadata)
            finally:
                os.close(child_fd)
        if not _metadata_matches(expected, os.fstat(descriptor), stable=True):
            raise _ScanStopped(
                "DIRECTORY_CHANGED", state="IDENTITY_CONFLICT"
            )

    complete = True
    issues = []
    failure_state = None
    descriptor = os.dup(root_fd)
    try:
        walk(descriptor, "", 0, root_metadata)
    except _ScanStopped as exc:
        complete = False
        issues.append(str(exc))
        failure_state = exc.state
    except OSError as exc:
        complete = False
        error = _access_error("SCAN_IO_FAILED", exc)
        issues.append(str(error))
        failure_state = error.state
    finally:
        os.close(descriptor)
    return {
        "complete": complete,
        "issues": issues,
        "failure_state": failure_state,
        "nodes": nodes,
        "entry_count": entry_count,
        "file_count": sum(
            item["file_type"] != "directory" for item in nodes.values()
        ),
        "directory_count": directories,
        "total_file_bytes": total_bytes,
        "relative_path_bytes": relative_path_bytes,
        "elapsed_seconds": round(time.monotonic() - started, 6),
    }


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
        "cloud_sha256": None if cloud is None else cloud.get("sha256"),
        "identity_basis": (
            "declaration_only" if cloud is None
            else cloud.get("identity_basis", "metadata_only")
        ),
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
            "output_id", "mode", "session", "route_commit", "manifest_sha256",
            "runtime_spec_sha256", "closeout_sha256", "runner_sha256",
            "raw_artifact_receipt_sha256",
        )
    }
    identity["raw_artifact_manifest_sha256"] = (
        binding.get("raw_artifact_receipt") or {}
    ).get("manifest_sha256")
    root_binding_enforced = binding.get("_root_binding_enforced") is True
    scope = "bound_run_root" if root_binding_enforced else "adapter_owned_root"
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
        "scope": scope,
        "root_binding_enforced": root_binding_enforced,
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
        "declared_run_root": binding.get("run_root"),
        "scope": scope,
        "root_binding_enforced": root_binding_enforced,
        "limits": limits,
        "discovery_completeness": discovery_completeness,
        "scientific_completeness": "not_assessed",
        "raw_terminal_seal": binding.get("raw_terminal_seal", "legacy_unsealed"),
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
        "_nodes": dict((scan or {}).get("nodes", {})),
    }


def _scan_adapter_root(binding: dict[str, Any], adapter_root: str | Path, *,
                       cloud_available: bool = True,
                       active_session: bool = False,
                       root_binding_enforced: bool = False,
                       limits: dict[str, int] | None = None) -> dict[str, Any]:
    """Scan a synthetic adapter-owned root without asserting production scope."""
    limits = _limits(limits)
    if isinstance(binding, dict):
        binding = dict(binding)
        binding["_root_binding_enforced"] = root_binding_enforced
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

    root = Path(adapter_root)
    try:
        opened = _open_adapter_root(root)
    except InventoryError as exc:
        if exc.state == "ARGUMENTS_INVALID":
            raise
        entry_state = (
            "IDENTITY_CONFLICT"
            if exc.state == "IDENTITY_CONFLICT" else "CLOUD_UNAVAILABLE"
        )
        return _inventory(
            binding, state=entry_state,
            discovery_completeness=(
                "partial" if entry_state == "IDENTITY_CONFLICT" else "unavailable"
            ),
            entries=scoped_entries(entry_state),
            scan=None, limits=limits,
            issues=[f"ROOT_OPEN_FAILED:{type(exc).__name__}"],
        )
    if opened is None:
        return _inventory(
            binding, state="INVENTORY_READY",
            discovery_completeness="complete",
            entries=scoped_entries("GITHUB_ONLY", "NOT_INVENTORIED"),
            scan=None, limits=limits, issues=[],
        )
    root_fd, root_metadata = opened
    try:
        try:
            identity_raw, control_metadata, identity_metadata = \
                _read_lifecycle_identity(root_fd)
            observed_identity = _json_object(identity_raw, "lifecycle identity")
            if observed_identity != binding["expected_lifecycle_identity"]:
                raise InventoryError(
                    "lifecycle identity differs", state="IDENTITY_CONFLICT"
                )
        except InventoryError as exc:
            entry_state = (
                "CLOUD_UNAVAILABLE"
                if exc.state == "CLOUD_UNAVAILABLE" else "IDENTITY_CONFLICT"
            )
            return _inventory(
                binding, state=entry_state,
                discovery_completeness=(
                    "unavailable" if entry_state == "CLOUD_UNAVAILABLE" else "partial"
                ),
                entries=scoped_entries(entry_state),
                scan=None, limits=limits,
                issues=[f"LIFECYCLE_IDENTITY_INVALID:{type(exc).__name__}"],
            )

        scan = _walk_metadata(root_fd, root_metadata, limits)
        complete = scan["complete"]
        nodes = scan["nodes"]
        entries = []
        expected_paths = set()
        identity_conflict = scan["failure_state"] == "IDENTITY_CONFLICT"
        cloud_unavailable = scan["failure_state"] == "CLOUD_UNAVAILABLE"
        try:
            stable_raw, stable_control, stable_identity = \
                _read_lifecycle_identity(root_fd)
            if stable_raw != identity_raw or stable_control != control_metadata \
                    or stable_identity != identity_metadata \
                    or not _metadata_matches(
                        root_metadata,
                        _fstat_or_error(root_fd, "root identity reread failed"),
                        stable=True,
                    ):
                raise InventoryError(
                    "root or lifecycle identity changed during inventory",
                    state="IDENTITY_CONFLICT",
                )
        except InventoryError as exc:
            complete = False
            scan["complete"] = False
            scan["issues"].append("SCAN_ROOT_IDENTITY_CHANGED")
            if exc.state == "CLOUD_UNAVAILABLE":
                cloud_unavailable = True
            else:
                identity_conflict = True

        def collision_for(relative: str) -> dict[str, Any] | None:
            parts = PurePosixPath(relative).parts
            prefix = []
            for part in parts[:-1]:
                prefix.append(part)
                ancestor = nodes.get("/".join(prefix))
                if ancestor is None:
                    return None
                if ancestor["file_type"] != "directory":
                    return ancestor
            exact = nodes.get(relative)
            if exact is not None and exact["file_type"] != "file":
                return exact
            return None

        raw_manifest_records: dict[str, dict[str, Any]] = {}
        raw_receipt = binding.get("raw_artifact_receipt")
        if raw_receipt is not None:
            try:
                manifest_relative = raw_receipt["manifest_relative_path"]
                manifest_metadata = nodes.get(manifest_relative)
                if (
                    manifest_metadata is None
                    or collision_for(manifest_relative) is not None
                ):
                    raise InventoryError(
                        "raw artifact manifest is missing or not a file",
                        state="IDENTITY_CONFLICT",
                    )
                manifest_raw = _read_relative_file(
                    root_fd, manifest_relative,
                    maximum=MAX_RAW_ARTIFACT_MANIFEST_BYTES,
                    expected=manifest_metadata, nodes=nodes,
                )
                raw_manifest_records = _parse_raw_artifact_manifest(
                    manifest_raw, raw_receipt,
                )
                stable_files = {
                    relative for relative, metadata in nodes.items()
                    if metadata["file_type"] == "file"
                    and PurePosixPath(relative).parts[0] in RAW_ARTIFACT_SCOPE_ROOTS
                }
                if stable_files != set(raw_manifest_records) or any(
                    nodes[relative]["bytes"] != item["bytes"]
                    for relative, item in raw_manifest_records.items()
                ):
                    raise InventoryError(
                        "stable cloud artifact paths or sizes differ from the manifest",
                        state="IDENTITY_CONFLICT",
                    )
                binding["raw_terminal_seal"] = "verified"
            except (InventoryError, KeyError, TypeError) as exc:
                raw_manifest_records = {}
                binding["raw_terminal_seal"] = "drifted"
                identity_conflict = True
                complete = False
                scan["complete"] = False
                scan["issues"].append(
                    f"RAW_ARTIFACT_MANIFEST_INVALID:{type(exc).__name__}"
                )

        missing_state = (
            "GITHUB_ONLY" if complete else
            "CLOUD_UNAVAILABLE" if cloud_unavailable else "NOT_INVENTORIED"
        )
        for item in binding["expected_evidence"]:
            relative = item["source_relpath"]
            expected_paths.add(relative)
            collision = collision_for(relative)
            if collision is not None:
                entries.append(_expected_entry(
                    item, "IDENTITY_CONFLICT", collision
                ))
                identity_conflict = True
                continue
            metadata = nodes.get(relative)
            if metadata is None:
                entries.append(_expected_entry(item, missing_state))
                continue
            try:
                raw = _read_relative_file(
                    root_fd, relative, maximum=item["max_bytes"],
                    expected=metadata, nodes=nodes,
                )
                observed_sha = hashlib.sha256(raw).hexdigest()
                cloud = {
                    **metadata,
                    "sha256": observed_sha,
                    "identity_basis": (
                        "github_sha256_and_terminal_manifest_sha256"
                        if relative in raw_manifest_records else "sha256"
                    ),
                }
                manifest_item = raw_manifest_records.get(relative)
                manifest_matches = manifest_item is None or (
                    manifest_item["bytes"] == len(raw)
                    and manifest_item["sha256"] == observed_sha
                )
                if (
                    len(raw) == item["bytes"]
                    and observed_sha == item["sha256"]
                    and manifest_matches
                ):
                    entries.append(_expected_entry(item, "MATCHED", cloud))
                else:
                    entries.append(_expected_entry(
                        item, "IDENTITY_CONFLICT", cloud
                    ))
                    identity_conflict = True
            except InventoryError as exc:
                entry_state = (
                    "CLOUD_UNAVAILABLE"
                    if exc.state == "CLOUD_UNAVAILABLE" else "IDENTITY_CONFLICT"
                )
                entries.append(_expected_entry(item, entry_state, metadata))
                complete = False
                if entry_state == "CLOUD_UNAVAILABLE":
                    cloud_unavailable = True
                else:
                    identity_conflict = True

        for item in binding["optional_evidence"]:
            relative = item["source_relpath"]
            expected_paths.add(relative)
            collision = collision_for(relative)
            if collision is not None:
                entries.append(_optional_entry(
                    item, "IDENTITY_CONFLICT", collision
                ))
                identity_conflict = True
                continue
            metadata = nodes.get(relative)
            if metadata is None:
                entries.append(_optional_entry(
                    item,
                    "CLOUD_UNAVAILABLE" if cloud_unavailable else "NOT_INVENTORIED",
                ))
            else:
                manifest_item = raw_manifest_records.get(relative)
                cloud = dict(metadata)
                if manifest_item is not None:
                    cloud.update({
                        "sha256": manifest_item["sha256"],
                        "identity_basis": "terminal_manifest_sha256",
                    })
                entries.append(_optional_entry(
                    item, "CLOUD_ONLY" if complete else "NOT_INVENTORIED",
                    cloud,
                ))

        for relative, metadata in sorted(nodes.items()):
            if metadata["file_type"] == "directory" or relative in expected_paths:
                continue
            policy = (
                "expected_control_identity"
                if relative == "control/lifecycle_identity.json"
                else "terminal_raw_artifact_manifest"
                if relative == RAW_ARTIFACT_MANIFEST_RELPATH
                else "terminal_manifest_bound_raw_artifact"
                if relative in raw_manifest_records
                else "expected_cloud_only_raw_artifact"
            )
            if metadata["file_type"] in {"symlink", "special"}:
                entry_state = "IDENTITY_CONFLICT"
            elif complete:
                entry_state = "CLOUD_ONLY"
            elif cloud_unavailable:
                entry_state = "CLOUD_UNAVAILABLE"
            else:
                entry_state = "NOT_INVENTORIED"
            entries.append({
                "scope": "raw_output",
                "relative_path": relative,
                "artifact_class": (
                    "control_identity" if policy == "expected_control_identity"
                    else "control_manifest"
                    if policy == "terminal_raw_artifact_manifest"
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
                ) if relative not in raw_manifest_records
                else raw_manifest_records[relative]["sha256"],
                "identity_basis": (
                    "verified_control_identity"
                    if relative == "control/lifecycle_identity.json"
                    else "terminal_manifest_sha256"
                    if relative in raw_manifest_records
                    else "terminal_manifest_identity"
                    if relative == RAW_ARTIFACT_MANIFEST_RELPATH
                    else "metadata_only"
                ),
                "reconciliation_state": entry_state,
                "policy_assessment": policy,
            })
        entries.extend(archive_only)
        state = (
            "IDENTITY_CONFLICT" if identity_conflict else
            "CLOUD_UNAVAILABLE" if cloud_unavailable else
            "INVENTORY_READY" if complete else "INVENTORY_INCOMPLETE"
        )
        discovery = (
            "partial" if identity_conflict else
            "unavailable" if cloud_unavailable else
            "complete" if complete else "partial"
        )
        return _inventory(
            binding, state=state, discovery_completeness=discovery,
            entries=entries, scan=scan, limits=limits, issues=scan["issues"],
        )
    finally:
        os.close(root_fd)


def inventory_summary(inventory: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in inventory.items()
        if not key.startswith("_")
    }


def normalize_query_arguments(
    reconciliation_states: list[str] | None,
    terms: list[str] | None,
    limit: int,
) -> tuple[list[str], list[str], int]:
    states = list(RECONCILIATION_STATES) \
        if reconciliation_states is None else reconciliation_states
    if not isinstance(states, list) \
            or any(state not in RECONCILIATION_STATES for state in states):
        raise InventoryError(
            "reconciliation_states is invalid", state="ARGUMENTS_INVALID", exit_code=2
        )
    states = sorted(set(states))
    terms = [] if terms is None else terms
    if not isinstance(terms, list) or len(terms) > 8 \
            or any(not isinstance(term, str) or not term or len(term) > 128 for term in terms):
        raise InventoryError("terms is invalid", state="ARGUMENTS_INVALID", exit_code=2)
    normalized_terms = [term.casefold() for term in terms]
    if not isinstance(limit, int) or isinstance(limit, bool) \
            or not 1 <= limit <= MAX_QUERY_ENTRIES:
        raise InventoryError("limit must be in [1, 100]", state="ARGUMENTS_INVALID", exit_code=2)
    return states, normalized_terms, limit


def inventory_query_sha256(
    identity: dict[str, Any], inventory_sha256: str,
    reconciliation_states: list[str], terms: list[str],
) -> str:
    return canonical_sha256({
        "snapshot_commit": identity["snapshot_commit"],
        "terminal_record_sha256": identity["terminal_record_sha256"],
        "manifest_sha256": identity["manifest_sha256"],
        "runtime_spec_sha256": identity["runtime_spec_sha256"],
        "inventory_sha256": inventory_sha256,
        "reconciliation_states": reconciliation_states,
        "terms": terms,
    })


def inventory_query_page(
    inventory: dict[str, Any], *, inventory_sha256: str,
    reconciliation_states: list[str] | None = None,
    terms: list[str] | None = None, offset: int = 0,
    limit: int = 20,
) -> dict[str, Any]:
    inventory_sha256 = _require_sha(
        inventory_sha256, SHA256, "inventory_sha256"
    )
    if inventory.get("inventory_sha256") != inventory_sha256:
        raise InventoryError(
            "inventory identity differs", state="INVENTORY_DRIFT", exit_code=3
        )
    states, terms, limit = normalize_query_arguments(
        reconciliation_states, terms, limit
    )
    if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
        raise InventoryError("offset is invalid", state="ARGUMENTS_INVALID", exit_code=2)
    candidates = []
    for entry in inventory.get("_entries", []):
        if entry["reconciliation_state"] not in states:
            continue
        searchable = canonical_bytes(entry).decode("utf-8").casefold()
        if all(term in searchable for term in terms):
            candidates.append(entry)
    identity = inventory["identity"]
    query_sha256 = inventory_query_sha256(
        identity, inventory_sha256, states, terms
    )
    if offset > len(candidates) or (offset == len(candidates) and offset != 0):
        raise InventoryError("inventory cursor is outside results", state="REPO_CURSOR_INVALID")
    selected = candidates[offset:offset + limit]
    while True:
        end = offset + len(selected)
        complete = end == len(candidates)
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
            "next_offset": None if complete else end,
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


def cloud_text_page(
    inventory: dict[str, Any], *, binding: dict[str, Any], adapter_root: str | Path,
    inventory_sha256: str, relative_path: str, offset: int = 0,
    page_bytes: int = MAX_TEXT_PAGE_BYTES,
    expected_file_sha256: str | None = None,
) -> dict[str, Any]:
    """Read one bounded UTF-8 page from an exact, fully inventoried cloud file."""
    inventory_sha256 = _require_sha(
        inventory_sha256, SHA256, "inventory_sha256"
    )
    if inventory.get("inventory_sha256") != inventory_sha256:
        raise InventoryError(
            "inventory identity differs", state="INVENTORY_DRIFT", exit_code=3
        )
    if inventory.get("state") != "INVENTORY_READY" \
            or inventory.get("discovery_completeness") != "complete" \
            or inventory.get("root_binding_enforced") is not True:
        raise InventoryError(
            "cloud text requires a complete inactive bound-run inventory",
            state="CLOUD_TEXT_NOT_INVENTORIED", exit_code=3,
        )
    if binding.get("raw_inventory_authorized") is not True:
        raise InventoryError(
            "cloud text scope is protected or not authorized",
            state="CLOUD_TEXT_PROTECTED_SCOPE", exit_code=3,
        )
    relative_path = _require_relpath(
        relative_path, "relative_path", state="ARGUMENTS_INVALID"
    )
    if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
        raise InventoryError(
            "offset is invalid", state="ARGUMENTS_INVALID", exit_code=2
        )
    if not isinstance(page_bytes, int) or isinstance(page_bytes, bool) \
            or not 4 <= page_bytes <= MAX_TEXT_PAGE_BYTES:
        raise InventoryError(
            f"page_bytes must be in [4, {MAX_TEXT_PAGE_BYTES}]",
            state="ARGUMENTS_INVALID", exit_code=2,
        )
    if expected_file_sha256 is not None:
        expected_file_sha256 = _require_sha(
            expected_file_sha256, SHA256, "expected_file_sha256"
        )

    matches = [
        item for item in inventory.get("_entries", [])
        if item.get("scope") == "raw_output"
        and item.get("relative_path") == relative_path
    ]
    if len(matches) != 1:
        raise InventoryError(
            "relative_path is not one exact inventory entry",
            state="CLOUD_TEXT_NOT_INVENTORIED", exit_code=3,
        )
    entry = matches[0]
    if entry.get("artifact_class") not in TEXT_READ_ARTIFACT_CLASSES \
            or entry.get("extension") not in TEXT_READ_SUFFIXES:
        raise InventoryError(
            "inventory entry is not an allowed text artifact",
            state="CLOUD_TEXT_UNSUPPORTED_FORMAT", exit_code=3,
        )
    allowed_states = {
        "formal_compact_evidence": {"MATCHED"},
        "optional_compact_evidence": {"CLOUD_ONLY"},
        "raw_artifact": {"CLOUD_ONLY"},
    }[entry["artifact_class"]]
    if entry.get("reconciliation_state") not in allowed_states \
            or entry.get("file_type") != "file":
        raise InventoryError(
            "inventory entry is not in a readable reconciled state",
            state="CLOUD_TEXT_NOT_INVENTORIED", exit_code=3,
        )
    size = entry.get("bytes")
    if not isinstance(size, int) or isinstance(size, bool) or size < 0 \
            or size > MAX_TEXT_FILE_BYTES:
        raise InventoryError(
            "cloud text file exceeds its bounded size",
            state="CLOUD_TEXT_FILE_TOO_LARGE", exit_code=3,
        )
    nodes = inventory.get("_nodes")
    metadata = nodes.get(relative_path) if isinstance(nodes, dict) else None
    if not isinstance(metadata, dict) or metadata.get("file_type") != "file" \
            or metadata.get("bytes") != size:
        raise InventoryError(
            "cloud text metadata identity differs",
            state="CLOUD_TEXT_IDENTITY_DRIFT", exit_code=3,
        )

    root = Path(adapter_root)
    if str(root) != binding.get("run_root"):
        raise InventoryError(
            "cloud text run root identity differs",
            state="CLOUD_TEXT_IDENTITY_DRIFT", exit_code=3,
        )
    opened = _open_adapter_root(root)
    if opened is None:
        raise InventoryError(
            "cloud text run root is unavailable",
            state="CLOUD_TEXT_IDENTITY_DRIFT", exit_code=3,
        )
    root_fd, root_metadata = opened
    try:
        identity_raw, control_metadata, identity_metadata = \
            _read_lifecycle_identity(root_fd)
        observed_identity = _json_object(identity_raw, "lifecycle identity")
        if observed_identity != binding.get("expected_lifecycle_identity"):
            raise InventoryError(
                "cloud text lifecycle identity differs",
                state="CLOUD_TEXT_IDENTITY_DRIFT", exit_code=3,
            )
        raw = _read_relative_file(
            root_fd, relative_path, maximum=MAX_TEXT_FILE_BYTES,
            expected=metadata, nodes=nodes,
        )
        stable_raw, stable_control, stable_identity = \
            _read_lifecycle_identity(root_fd)
        if stable_raw != identity_raw or stable_control != control_metadata \
                or stable_identity != identity_metadata \
                or not _metadata_matches(
                    root_metadata,
                    _fstat_or_error(root_fd, "cloud text root identity reread failed"),
                    stable=True,
                ):
            raise InventoryError(
                "cloud text root identity changed during read",
                state="CLOUD_TEXT_IDENTITY_DRIFT", exit_code=3,
            )
    finally:
        os.close(root_fd)

    try:
        raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise InventoryError(
            "cloud text file is not UTF-8",
            state="CLOUD_TEXT_NOT_UTF8", exit_code=3,
        ) from exc
    file_sha256 = hashlib.sha256(raw).hexdigest()
    if expected_file_sha256 is not None and file_sha256 != expected_file_sha256:
        raise InventoryError(
            "cloud text file SHA-256 changed",
            state="CLOUD_TEXT_IDENTITY_DRIFT", exit_code=3,
        )
    if entry["artifact_class"] == "formal_compact_evidence" \
            and file_sha256 != entry.get("github_sha256"):
        raise InventoryError(
            "formal cloud text differs from GitHub",
            state="CLOUD_TEXT_IDENTITY_DRIFT", exit_code=3,
        )
    if (
        entry["artifact_class"] in {"optional_compact_evidence", "raw_artifact"}
        and entry.get("identity_basis") == "terminal_manifest_sha256"
        and file_sha256 != entry.get("cloud_sha256")
    ):
        raise InventoryError(
            "cloud text differs from the terminal raw artifact manifest",
            state="CLOUD_TEXT_IDENTITY_DRIFT", exit_code=3,
        )
    if offset > len(raw) or (offset == len(raw) and len(raw) != 0):
        raise InventoryError(
            "cloud text cursor is outside the file",
            state="REPO_CURSOR_INVALID", exit_code=3,
        )
    try:
        raw[:offset].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise InventoryError(
            "cloud text cursor splits a UTF-8 character",
            state="REPO_CURSOR_INVALID", exit_code=3,
        ) from exc
    end = min(len(raw), offset + page_bytes)
    while end > offset:
        try:
            content = raw[offset:end].decode("utf-8")
            break
        except UnicodeDecodeError:
            end -= 1
    else:
        content = ""
    page_raw = raw[offset:end]
    terminal_page = end == len(raw)
    evidence_status = (
        "formal_sha_matched"
        if entry["artifact_class"] == "formal_compact_evidence"
        else "terminal_manifest_sha_matched"
        if entry.get("identity_basis") == "terminal_manifest_sha256"
        else "optional_declared"
        if entry["artifact_class"] == "optional_compact_evidence"
        else "unmapped_raw_text"
    )
    return {
        "schema_version": 2,
        "ok": True,
        "operation": "cloud-text-read",
        "state": "CLOUD_TEXT_PAGE_OK",
        "exit_code": 0,
        "snapshot_commit": binding["snapshot_commit"],
        "catalog_sha256": binding["catalog_sha256"],
        "terminal_record_sha256": binding["terminal_record_sha256"],
        "inventory_sha256": inventory_sha256,
        "relative_path": relative_path,
        "artifact_class": entry["artifact_class"],
        "policy_assessment": entry["policy_assessment"],
        "evidence_status": evidence_status,
        "reconciliation_state": entry["reconciliation_state"],
        "bytes": len(raw),
        "file_sha256": file_sha256,
        "identity_basis": "content_sha256",
        "encoding": "utf-8",
        "page_start_byte": offset,
        "page_end_byte": end,
        "page_bytes": len(page_raw),
        "page_sha256": hashlib.sha256(page_raw).hexdigest(),
        "content": content,
        "page_complete": True,
        "terminal_page": terminal_page,
        "complete": terminal_page,
        "has_more": not terminal_page,
        "scientific_completeness": "not_assessed",
    }


def inventory_query(inventory: dict[str, Any], *, inventory_sha256: str,
                    reconciliation_states: list[str] | None = None,
                    terms: list[str] | None = None, cursor: str | None = None,
                    limit: int = 20) -> dict[str, Any]:
    import convirctl

    states, normalized_terms, limit = normalize_query_arguments(
        reconciliation_states, terms, limit
    )
    identity = inventory["identity"]
    query_sha256 = inventory_query_sha256(
        identity, inventory_sha256, states, normalized_terms
    )
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
    value = inventory_query_page(
        inventory,
        inventory_sha256=inventory_sha256,
        reconciliation_states=states,
        terms=normalized_terms,
        offset=offset,
        limit=limit,
    )
    next_offset = value.pop("next_offset")
    value["next_cursor"] = None if next_offset is None else convirctl.encode_repo_cursor(
        CURSOR_OPERATION,
        identity["snapshot_commit"],
        value["query_sha256"],
        next_offset,
        inventory_sha256[:40],
    )
    return value


def _session_state(session: str) -> str:
    try:
        completed = subprocess.run(
            [REMOTE_TMUX, "has-session", "-t", session],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    if completed.returncode == 0:
        return "active"
    if completed.returncode == 1:
        return "inactive"
    return "unknown"


def _worker_failure(exc: Exception, operation: str) -> dict[str, Any]:
    if isinstance(exc, InventoryError):
        state = exc.state
        exit_code = exc.exit_code
    else:
        state = "REMOTE_STATE_UNKNOWN"
        exit_code = 70
    return {
        "ok": False,
        "operation": operation,
        "state": state,
        "exit_code": exit_code,
        "error": str(exc).encode("utf-8", errors="replace")[:2048].decode(
            "utf-8", errors="ignore"
        ),
        "scientific_completeness": "not_assessed",
    }


def remote_worker(request: dict[str, Any]) -> dict[str, Any]:
    expected_fields = {
        "schema_version", "operation", "binding", "adapter_root",
        "root_binding_enforced", "expected_session", "query",
    }
    if not isinstance(request, dict) or set(request) != expected_fields \
            or request.get("schema_version") not in {1, 2}:
        raise InventoryError(
            "remote worker request is invalid", state="ARGUMENTS_INVALID", exit_code=2
        )
    operation = request["operation"]
    if operation not in {"summary", "query", "text_read"}:
        raise InventoryError(
            "remote worker operation is invalid", state="ARGUMENTS_INVALID", exit_code=2
        )
    binding = request["binding"]
    if not isinstance(binding, dict) or binding.get("eligible") is not True \
            or binding.get("raw_inventory_authorized") is not True:
        raise InventoryError(
            "remote worker received an ineligible binding", state="IDENTITY_CONFLICT"
        )
    session = _require_token(request["expected_session"], "expected_session")
    if binding.get("session") != session:
        raise InventoryError("session identity differs", state="IDENTITY_CONFLICT")
    adapter_root = request["adapter_root"]
    if not isinstance(adapter_root, str):
        raise InventoryError(
            "adapter root is invalid", state="ARGUMENTS_INVALID", exit_code=2
        )
    root = Path(adapter_root)
    if not root.is_absolute() or ".." in root.parts:
        raise InventoryError(
            "adapter root is invalid", state="ARGUMENTS_INVALID", exit_code=2
        )
    root_binding_enforced = request["root_binding_enforced"]
    if root_binding_enforced is not True and root_binding_enforced is not False:
        raise InventoryError(
            "root binding flag is invalid", state="ARGUMENTS_INVALID", exit_code=2
        )
    if root_binding_enforced and adapter_root != binding.get("run_root"):
        raise InventoryError("run root identity differs", state="IDENTITY_CONFLICT")

    before = _session_state(session)
    if before == "active":
        result = _scan_adapter_root(
            binding, root, active_session=True,
            root_binding_enforced=root_binding_enforced,
        )
    elif before == "unknown":
        result = _scan_adapter_root(
            binding, root, cloud_available=False,
            root_binding_enforced=root_binding_enforced,
        )
    else:
        result = _scan_adapter_root(
            binding, root, root_binding_enforced=root_binding_enforced
        )
        after = _session_state(session)
        if after == "active":
            result = _scan_adapter_root(
                binding, root, active_session=True,
                root_binding_enforced=root_binding_enforced,
            )
        elif after == "unknown":
            result = _scan_adapter_root(
                binding, root, cloud_available=False,
                root_binding_enforced=root_binding_enforced,
            )

    if operation == "summary" or result.get("ok") is not True:
        return inventory_summary(result)
    query = request["query"]
    if operation == "text_read":
        if request.get("schema_version") != 2 or not isinstance(query, dict) \
                or set(query) != {
                    "inventory_sha256", "relative_path", "offset", "page_bytes",
                    "expected_file_sha256",
                }:
            raise InventoryError(
                "remote text-read request is invalid",
                state="ARGUMENTS_INVALID", exit_code=2,
            )
        return cloud_text_page(
            result,
            binding=binding,
            adapter_root=root,
            inventory_sha256=query["inventory_sha256"],
            relative_path=query["relative_path"],
            offset=query["offset"],
            page_bytes=query["page_bytes"],
            expected_file_sha256=query["expected_file_sha256"],
        )
    if not isinstance(query, dict) or set(query) != {
        "inventory_sha256", "reconciliation_states", "terms", "offset", "limit"
    }:
        raise InventoryError(
            "remote query request is invalid", state="ARGUMENTS_INVALID", exit_code=2
        )
    return inventory_query_page(
        result,
        inventory_sha256=query["inventory_sha256"],
        reconciliation_states=query["reconciliation_states"],
        terms=query["terms"],
        offset=query["offset"],
        limit=query["limit"],
    )


def remote_worker_main() -> None:
    source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    operation = "cloud-inventory-worker"
    try:
        raw = sys.stdin.buffer.read(MAX_REMOTE_REQUEST_BYTES + 1)
        if len(raw) > MAX_REMOTE_REQUEST_BYTES:
            raise InventoryError(
                "remote worker request exceeds its bound",
                state="ARGUMENTS_INVALID", exit_code=2,
            )
        request = _json_object(raw, "remote worker request")
        operation = f"cloud-inventory-{request.get('operation', 'worker')}"
        result = remote_worker(request)
    except Exception as exc:
        result = _worker_failure(exc, operation)
    response = {
        "worker_source_sha256": source_sha256,
        "result": result,
    }
    encoded = canonical_bytes(response) + b"\n"
    if len(encoded) > MAX_REMOTE_RESPONSE_BYTES:
        response = {
            "worker_source_sha256": source_sha256,
            "result": _worker_failure(
                InventoryError(
                    "remote worker response exceeds its bound",
                    state="RESPONSE_TOO_LARGE", exit_code=3,
                ),
                operation,
            ),
        }
        encoded = canonical_bytes(response) + b"\n"
    sys.stdout.buffer.write(encoded)
    sys.stdout.buffer.flush()


if __name__ == "__main__" and sys.argv[1:] == ["--remote-worker"]:
    remote_worker_main()
