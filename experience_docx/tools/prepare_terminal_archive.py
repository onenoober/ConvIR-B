#!/usr/bin/env python3
"""Prepare one minimal, complete terminal-science archive for GitHub main."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import subprocess
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

import convir_ops_mcp as ops
import capability_registry


GIT = "/usr/bin/git"
LOG_PREFIX = "experience_docx/experiment_logs/"
CARD_PREFIX = "experience_docx/experiment_cards/"
INDEX_PATH = "experience_docx/EXPERIMENT_TERMINAL_INDEX.jsonl"
ALLOWED_SUFFIXES = {".json", ".csv", ".md", ".txt", ".log", ".out", ".sh"}
FORBIDDEN_SUFFIXES = {
    ".pkl", ".pth", ".pt", ".ckpt", ".onnx", ".png", ".jpg", ".jpeg",
    ".bmp", ".gif", ".webp", ".npy", ".npz", ".mat", ".zip", ".tar",
    ".gz", ".7z", ".rar",
}
FORBIDDEN_NAME_TOKENS = {
    "cloud_only", "raw_prediction", "raw_feature", "raw_action", "per_sample",
}
MAX_EVIDENCE_FILES = 48
MAX_FILE_BYTES = 1024 * 1024
TOKEN = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
ARCHIVE_REMOTE = "github"
ARCHIVE_TARGET_REF = "main"
ARCHIVE_BASE_REF = "refs/remotes/github/main"


class TerminalArchiveError(RuntimeError):
    pass


def git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        [GIT, *args], cwd=repo, text=True, capture_output=True,
        timeout=60, check=False,
    )
    if completed.returncode:
        detail = (completed.stdout + completed.stderr).strip()[:4096]
        raise TerminalArchiveError(f"git {' '.join(args)} failed: {detail}")
    return completed.stdout.strip()


def safe_relative(value: str, *, prefix: str | None = None) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value:
        raise TerminalArchiveError(f"unsafe repository path: {value}")
    if prefix is not None and not value.startswith(prefix):
        raise TerminalArchiveError(f"path is outside {prefix}: {value}")
    return path


def checked_text(raw: bytes, relpath: str) -> str:
    empty_exclusion_asset = (
        not raw and PurePosixPath(relpath).name.endswith("_exclusions.txt")
    )
    if (not raw and not empty_exclusion_asset) or len(raw) > MAX_FILE_BYTES:
        raise TerminalArchiveError(f"file size is outside limits: {relpath}")
    if b"\0" in raw:
        raise TerminalArchiveError(f"binary content is forbidden: {relpath}")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TerminalArchiveError(f"text evidence is not UTF-8: {relpath}") from exc


def inspect_structured(raw: bytes, relpath: str) -> Any:
    text = checked_text(raw, relpath)
    suffix = PurePosixPath(relpath).suffix.lower()
    if suffix == ".json":
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise TerminalArchiveError(f"invalid JSON: {relpath}: {exc}") from exc
    if suffix == ".csv":
        rows = list(csv.reader(io.StringIO(text, newline="")))
        if not rows or not rows[0] or any(not item.strip() for item in rows[0]):
            raise TerminalArchiveError(f"CSV header is missing or empty: {relpath}")
        width = len(rows[0])
        if len(set(rows[0])) != width or width > 256:
            raise TerminalArchiveError(f"CSV header is duplicate or too wide: {relpath}")
        if any(len(row) != width for row in rows[1:]):
            raise TerminalArchiveError(f"CSV rows have inconsistent widths: {relpath}")
        return {"rows": len(rows), "columns": width}
    return None


def git_blob(repo: Path, commit: str, relpath: str) -> bytes:
    completed = subprocess.run(
        [GIT, "show", f"{commit}:{relpath}"], cwd=repo,
        capture_output=True, timeout=30, check=False,
    )
    if completed.returncode:
        raise TerminalArchiveError(f"tracked contract is missing at {commit}: {relpath}")
    return completed.stdout


def validate_contract(raw: bytes, relpath: str, route_id: str) -> None:
    parsed = inspect_structured(raw, relpath)
    if isinstance(parsed, dict):
        if parsed.get("route_id") != route_id:
            raise TerminalArchiveError(f"contract route_id mismatch: {relpath}")
        return
    text = raw.decode("utf-8")
    match = re.search(r"(?m)^- Route id:\s*(.+?)\s*$", text)
    observed = "" if match is None else match.group(1).strip()
    observed = observed.replace("\\`", "").replace("`", "").strip().rstrip(".")
    if observed != route_id:
        raise TerminalArchiveError(f"contract route identity is missing: {relpath}")


def required_runtime_evidence(
    repo: Path, route_commit: str, route_id: str, operation_id: str,
    closeout: dict[str, Any],
) -> set[str]:
    manifest_path = "experience_docx/route_operations.json"
    try:
        manifest = json.loads(git_blob(repo, route_commit, manifest_path))
    except json.JSONDecodeError as exc:
        raise TerminalArchiveError("invalid route operations manifest") from exc
    if manifest.get("route_id") != route_id:
        raise TerminalArchiveError("route operations route_id mismatch")
    operation = manifest.get("operations", {}).get(operation_id)
    if not isinstance(operation, dict):
        raise TerminalArchiveError("closeout operation is absent from route operations")
    terminal = {
        key: closeout.get(key) for key in ("state", "decision", "authorizes")
    }
    allowed = operation.get("allowed_terminal_tuples")
    if not isinstance(allowed, list) or terminal not in allowed:
        raise TerminalArchiveError("closeout terminal tuple is outside the frozen contract")
    spec_path = f"experience_docx/route_runtime_specs/{operation_id}.json"
    try:
        spec = json.loads(git_blob(repo, route_commit, spec_path))
    except json.JSONDecodeError as exc:
        raise TerminalArchiveError("invalid route runtime spec") from exc
    if spec.get("route_id") != route_id or spec.get("operation_id") != operation_id:
        raise TerminalArchiveError("runtime spec identity mismatch")
    if closeout.get("evidence_role") != spec.get("evidence_role"):
        raise TerminalArchiveError("closeout evidence role differs from the runtime contract")
    permissions = spec.get("protected_data_permissions")
    touched = {
        "allow_confirmation": closeout.get("confirmation_images_targets_outcomes_touched"),
        "allow_canary": closeout.get("canary_touched"),
        "allow_locked_test": closeout.get("locked_test_touched"),
    }
    if not isinstance(permissions, dict) or any(
        value is not True and value is not False for value in touched.values()
    ) or any(touched[key] and permissions.get(key) is not True for key in touched):
        raise TerminalArchiveError("closeout protected-data access violates the runtime contract")
    required = set()
    for item in spec.get("evidence_files", []):
        if not isinstance(item, dict):
            raise TerminalArchiveError("invalid runtime evidence declaration")
        filename = item.get("destination_filename")
        if item.get("required", False):
            if not isinstance(filename, str):
                raise TerminalArchiveError("required runtime evidence filename is missing")
            validate_evidence_name(filename)
            required.add(filename)
    if not required:
        raise TerminalArchiveError("scientific terminal runtime declares no required result evidence")
    return required


def launch_contract_bundle(
    repo: Path, route_commit: str, route_id: str, operation_id: str,
) -> tuple[dict[str, bytes], list[dict[str, Any]], dict[str, Any] | None]:
    manifest_path = "experience_docx/route_operations.json"
    manifest_raw = git_blob(repo, route_commit, manifest_path)
    try:
        manifest = json.loads(manifest_raw)
    except json.JSONDecodeError as exc:
        raise TerminalArchiveError("invalid route operations manifest") from exc
    if manifest.get("schema_version") != 6:
        return {}, [], None
    if manifest.get("route_id") != route_id:
        raise TerminalArchiveError("launch bundle route identity mismatch")
    operation = manifest.get("operations", {}).get(operation_id)
    if not isinstance(operation, dict):
        raise TerminalArchiveError("launch bundle operation is missing")
    prior_closeout = operation.get("prior_closeout_relpath")
    prior_terminal = operation.get("prior_terminal_tuple")
    if (prior_closeout is None) != (prior_terminal is None):
        raise TerminalArchiveError("launch bundle prior terminal binding is incomplete")
    if prior_closeout is not None:
        safe_relative(prior_closeout, prefix=LOG_PREFIX)
        if not isinstance(prior_terminal, dict) or set(prior_terminal) != {
            "state", "decision", "authorizes",
        }:
            raise TerminalArchiveError("launch bundle prior terminal tuple is invalid")
    spec_path = f"experience_docx/route_runtime_specs/{operation_id}.json"
    try:
        runtime = json.loads(git_blob(repo, route_commit, spec_path))
    except json.JSONDecodeError as exc:
        raise TerminalArchiveError("invalid route runtime spec") from exc
    paths = {
        "manifest.json": manifest_path,
        "route_note.md": manifest.get("route_card_relpath"),
        "experiment_spec.json": manifest.get("experiment_spec_relpath"),
        "program_contract.json": manifest.get("program_contract_relpath"),
        "scientific_contract.json": manifest.get(
            "scientific_contract_relpaths", {}
        ).get(operation_id),
        "runtime_spec.json": spec_path,
        "asset_manifest.json": runtime.get("asset_manifest_relpath"),
        "capability_profile.json": runtime.get(
            "engineering_contract", {}
        ).get("capability_profile_relpath"),
        "precision_certificate.json": runtime.get(
            "precision_contract", {}
        ).get("certificate_relpath"),
    }
    required_names = {
        "manifest.json", "route_note.md", "experiment_spec.json",
        "program_contract.json", "scientific_contract.json", "runtime_spec.json",
    }
    if any(not isinstance(paths[name], str) for name in required_names):
        raise TerminalArchiveError("canonical launch contract bundle is incomplete")
    payloads = {}
    records = []
    root = f"{LOG_PREFIX}{route_id}/launch_contract/{operation_id}"
    for archive_name, source_path in paths.items():
        if source_path is None:
            continue
        safe_relative(source_path)
        raw = git_blob(repo, route_commit, source_path)
        checked_text(raw, source_path)
        destination = f"{root}/{archive_name}"
        payloads[destination] = raw
        records.append({
            "path": destination,
            "source_path": source_path,
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        })
    return payloads, records, {
        "prior_closeout_path": prior_closeout,
        "prior_terminal_tuple": prior_terminal,
    }


def validate_conclusion(
    raw: bytes, relpath: str, closeout: dict[str, Any], *,
    required_schema_version: int | None = None,
) -> None:
    value = inspect_structured(raw, relpath)
    if required_schema_version is not None and (
        not isinstance(value, dict)
        or value.get("schema_version") != required_schema_version
    ):
        raise TerminalArchiveError(
            f"scientific conclusion schema_version must equal {required_schema_version}"
        )
    required = {
        "route_id", "operation_id", "run_id", "state", "decision", "authorizes",
        "primary_result", "gate_reasons", "competing_explanation", "limitations",
    }
    if not isinstance(value, dict) or not required.issubset(value):
        raise TerminalArchiveError(f"scientific conclusion is incomplete: {relpath}")
    for key in ("route_id", "operation_id", "run_id", "state", "decision", "authorizes"):
        if value.get(key) != closeout.get(key):
            raise TerminalArchiveError(f"scientific conclusion {key} mismatch")
    if not isinstance(value["primary_result"], str) or not value["primary_result"].strip():
        raise TerminalArchiveError("scientific conclusion primary_result is empty")
    if not isinstance(value["competing_explanation"], str) \
            or not value["competing_explanation"].strip():
        raise TerminalArchiveError("scientific conclusion competing_explanation is empty")
    for key in ("gate_reasons", "limitations"):
        if not isinstance(value[key], list) or not value[key] \
                or any(not isinstance(item, str) or not item.strip() for item in value[key]):
            raise TerminalArchiveError(f"scientific conclusion {key} is empty")


def validate_evidence_name(filename: str) -> None:
    path = PurePosixPath(filename)
    if path.name != filename or filename in {"", ".", ".."}:
        raise TerminalArchiveError(f"evidence filename must be top-level: {filename}")
    suffix = path.suffix.lower()
    if suffix in FORBIDDEN_SUFFIXES or suffix not in ALLOWED_SUFFIXES:
        raise TerminalArchiveError(f"forbidden evidence suffix: {filename}")
    lowered = filename.lower()
    if any(token in lowered for token in FORBIDDEN_NAME_TOKENS):
        raise TerminalArchiveError(f"raw/cloud-only evidence is forbidden: {filename}")


def capability_registry_record(
    closeout: dict[str, Any], closeout_path: PurePosixPath,
    payloads: dict[str, bytes],
) -> dict[str, Any] | None:
    metadata = closeout.get("capability_qualification")
    if metadata is None:
        return None
    if not isinstance(metadata, dict) or set(metadata) != {
        "qualification_id", "identity_sha256", "evidence_filename", "status",
        "scientific_authorization",
    }:
        raise TerminalArchiveError("capability qualification metadata is invalid")
    filename = metadata["evidence_filename"]
    if not isinstance(filename, str):
        raise TerminalArchiveError("capability qualification filename is invalid")
    validate_evidence_name(filename)
    relpath = f"{closeout_path.parent.as_posix()}/{filename}"
    raw = payloads.get(relpath)
    if raw is None:
        raise TerminalArchiveError("capability qualification evidence is missing")
    value = inspect_structured(raw, relpath)
    expected = {
        "schema_version", "qualification_id", "identity", "identity_sha256",
        "status", "contract_mode", "route_id", "operation_id", "run_id",
        "route_commit", "engineering_evidence", "scientific_authorization",
        "protected_data_touched",
    }
    if not isinstance(value, dict) or set(value) != expected \
            or value["schema_version"] != 1:
        raise TerminalArchiveError("capability qualification evidence is invalid")
    for key in ("route_id", "operation_id", "run_id", "route_commit"):
        if value[key] != closeout.get(key):
            raise TerminalArchiveError(f"capability qualification {key} mismatch")
    try:
        identity = capability_registry.validate_identity(value["identity"])
        identity_sha = capability_registry.identity_digest(identity)
    except capability_registry.CapabilityRegistryError as exc:
        raise TerminalArchiveError(str(exc)) from exc
    if value["identity_sha256"] != identity_sha \
            or metadata["identity_sha256"] != identity_sha \
            or value["qualification_id"] != metadata["qualification_id"] \
            or value["status"] != metadata["status"] \
            or value["status"] != "PASSED_ENGINEERING" \
            or value["scientific_authorization"] != "NONE" \
            or metadata["scientific_authorization"] != "NONE" \
            or value["protected_data_touched"] is not False \
            or not isinstance(value["engineering_evidence"], dict):
        raise TerminalArchiveError("capability qualification boundary is invalid")
    record = {
        "schema_version": 1,
        "qualification_id": value["qualification_id"],
        "identity": identity,
        "identity_sha256": identity_sha,
        "status": "PASSED_ENGINEERING",
        "contract_mode": value["contract_mode"],
        "evidence_relpath": relpath,
        "evidence_sha256": hashlib.sha256(raw).hexdigest(),
        "scientific_authorization": "NONE",
        "protected_data_touched": False,
    }
    try:
        return capability_registry.validate_record(record)
    except capability_registry.CapabilityRegistryError as exc:
        raise TerminalArchiveError(str(exc)) from exc


def audit_source(
    source_repo: Path,
    source_ref: str,
    route_id: str,
    closeout_relpath: str,
    contract_relpath: str,
    conclusion_relpath: str | None,
    receipt: str,
    *,
    existing_archive: bool = False,
    evidence_dir_override: Path | None = None,
    conclusion_dir_override: Path | None = None,
    expected_closeout_filename: str | None = None,
    expected_closeout_sha256: str | None = None,
) -> dict[str, Any]:
    source_repo = source_repo.resolve()
    if not TOKEN.fullmatch(route_id):
        raise TerminalArchiveError("route_id must be a safe token")
    if not SHA256.fullmatch(receipt):
        raise TerminalArchiveError("receipt must be a lowercase SHA-256 token")
    closeout_path = safe_relative(
        closeout_relpath, prefix=f"{LOG_PREFIX}{route_id}/",
    )
    if closeout_path.name != closeout_relpath.rsplit("/", 1)[-1] \
            or not closeout_path.name.endswith("_closeout.json"):
        raise TerminalArchiveError("closeout must be one top-level *_closeout.json")
    contract_path = safe_relative(contract_relpath, prefix=CARD_PREFIX)
    if contract_path.suffix.lower() not in {".md", ".json"}:
        raise TerminalArchiveError("contract must be a route-card Markdown or JSON file")
    if conclusion_relpath is None and not existing_archive:
        raise TerminalArchiveError("new terminal archives require one scientific conclusion JSON")
    if conclusion_relpath is not None:
        conclusion_path = safe_relative(
            conclusion_relpath, prefix=f"{LOG_PREFIX}{route_id}/",
        )
        if conclusion_path.name != conclusion_relpath.rsplit("/", 1)[-1] \
                or conclusion_path.suffix.lower() != ".json":
            raise TerminalArchiveError("conclusion must be one top-level JSON evidence file")

    source_commit = git(source_repo, "rev-parse", source_ref)
    contract_raw = git_blob(source_repo, source_commit, contract_relpath)
    validate_contract(contract_raw, contract_relpath, route_id)

    evidence_dir = evidence_dir_override or (source_repo / closeout_path.parent)
    conclusion_dir = conclusion_dir_override or evidence_dir
    closeout_file = evidence_dir / closeout_path.name
    if not closeout_file.is_file():
        raise TerminalArchiveError(f"closeout is missing from source worktree: {closeout_relpath}")
    closeout_raw = closeout_file.read_bytes()
    if (expected_closeout_filename is None) != (expected_closeout_sha256 is None):
        raise TerminalArchiveError("receipt closeout binding is incomplete")
    if expected_closeout_filename is not None:
        if closeout_path.name != expected_closeout_filename \
                or not isinstance(expected_closeout_sha256, str) \
                or not SHA256.fullmatch(expected_closeout_sha256) \
                or hashlib.sha256(closeout_raw).hexdigest() != expected_closeout_sha256:
            raise TerminalArchiveError("receipt-bound closeout identity mismatch")
    closeout = inspect_structured(closeout_raw, closeout_relpath)
    if not isinstance(closeout, dict):
        raise TerminalArchiveError("closeout must be a JSON object")
    if closeout.get("route_id") != route_id:
        raise TerminalArchiveError("closeout route_id mismatch")
    state = closeout.get("state")
    if not isinstance(state, str) or not state.startswith("COMPLETED_"):
        raise TerminalArchiveError("terminal fastpath accepts scientific/safety COMPLETED_* closeouts only")
    if not isinstance(closeout.get("decision"), str) or not closeout["decision"]:
        raise TerminalArchiveError("closeout decision is missing")
    if not isinstance(closeout.get("authorizes"), str):
        raise TerminalArchiveError("closeout authorizes is missing")
    route_commit = closeout.get("route_commit")
    if not isinstance(route_commit, str) or not re.fullmatch(r"[0-9a-f]{40}", route_commit):
        raise TerminalArchiveError("closeout route_commit is missing or invalid")
    if not existing_archive and source_commit != route_commit:
        raise TerminalArchiveError(
            f"source HEAD must equal launch route_commit: HEAD={source_commit} route={route_commit}"
        )
    evidence_sha = closeout.get("evidence_sha256")
    if not isinstance(evidence_sha, dict) or not evidence_sha:
        raise TerminalArchiveError(
            "scientific terminal archive requires closeout-bound result evidence; verdict-only archive is forbidden"
        )
    if len(evidence_sha) > MAX_EVIDENCE_FILES:
        raise TerminalArchiveError("too many compact evidence files")
    operation_id = closeout.get("operation_id")
    if not isinstance(operation_id, str) or not TOKEN.fullmatch(operation_id):
        raise TerminalArchiveError("closeout operation_id is missing or invalid")
    required_evidence = required_runtime_evidence(
        source_repo, route_commit, route_id, operation_id, closeout,
    )
    contract_payloads, contract_bundle, prior_terminal_record = launch_contract_bundle(
        source_repo, route_commit, route_id, operation_id,
    )
    if contract_bundle and closeout.get("schema_version") != 2:
        raise TerminalArchiveError(
            "schema-6 terminal closeout schema_version must equal 2"
        )
    missing_required = sorted(required_evidence - set(evidence_sha))
    if missing_required:
        raise TerminalArchiveError(
            "closeout omits required runtime evidence: " + ", ".join(missing_required)
        )

    files: list[dict[str, Any]] = []
    payloads: dict[str, bytes] = {
        contract_relpath: contract_raw,
        closeout_relpath: closeout_raw,
        **contract_payloads,
    }
    if conclusion_relpath is not None:
        conclusion_file = conclusion_dir / PurePosixPath(conclusion_relpath).name
        if not conclusion_file.is_file():
            raise TerminalArchiveError(
                f"scientific conclusion is missing: {conclusion_relpath}"
            )
        conclusion_raw = conclusion_file.read_bytes()
        validate_conclusion(
            conclusion_raw, conclusion_relpath, closeout,
            required_schema_version=2 if contract_bundle else None,
        )
        payloads[conclusion_relpath] = conclusion_raw
    for filename, expected_sha in sorted(evidence_sha.items()):
        if not isinstance(filename, str) or not isinstance(expected_sha, str):
            raise TerminalArchiveError("evidence_sha256 must map filenames to SHA-256 strings")
        validate_evidence_name(filename)
        if not SHA256.fullmatch(expected_sha):
            raise TerminalArchiveError(f"invalid evidence SHA-256: {filename}")
        source_file = evidence_dir / filename
        if not source_file.is_file():
            raise TerminalArchiveError(f"closeout-bound evidence is missing: {filename}")
        raw = source_file.read_bytes()
        inspect_structured(raw, filename)
        actual_sha = hashlib.sha256(raw).hexdigest()
        if actual_sha != expected_sha:
            raise TerminalArchiveError(
                f"closeout-bound evidence SHA-256 mismatch: {filename}"
            )
        relpath = f"{closeout_path.parent.as_posix()}/{filename}"
        payloads[relpath] = raw
        files.append({
            "path": relpath,
            "bytes": len(raw),
            "sha256": actual_sha,
        })
    registry_record = capability_registry_record(closeout, closeout_path, payloads)

    return {
        "schema_version": 2 if contract_bundle else 1,
        "status": "TERMINAL_SOURCE_AUDIT_OK",
        "route_id": route_id,
        "operation_id": closeout.get("operation_id"),
        "run_id": closeout.get("run_id"),
        "state": state,
        "decision": closeout["decision"],
        "authorizes": closeout["authorizes"],
        "receipt": receipt,
        "route_commit": route_commit,
        "source_commit": source_commit,
        "contract_path": contract_relpath,
        "closeout_path": closeout_relpath,
        "conclusion_path": conclusion_relpath,
        "result_files": files,
        "contract_bundle": contract_bundle,
        "prior_terminal_record": prior_terminal_record,
        "capability_registry_record": registry_record,
        "payloads": payloads,
        "checks": {
            "contract_bound": True,
            "terminal_tuple_complete": True,
            "all_closeout_bound_results_present": True,
            "all_runtime_required_results_present": True,
            "all_result_hashes_match": True,
            "single_scientific_conclusion_complete": (
                conclusion_relpath is not None or existing_archive
            ),
            "legacy_conclusion_waiver": existing_archive and conclusion_relpath is None,
            "compact_text_only": True,
            "verdict_only_archive_rejected": True,
            "full_launch_contract_bundle": bool(contract_bundle),
        },
    }


def fetch_receipt_evidence(receipt: str, destination: Path) -> dict[str, Any]:
    """Fetch only the MCP compact allowlist into an ephemeral directory."""
    try:
        context = ops.evidence_context({"receipt": receipt})
        records = ops.parse_evidence_manifest(
            ops.run_remote(
                ops.evidence_manifest_body(context), timeout=60, phase="evidence_manifest",
            )
        )
        if not records:
            raise TerminalArchiveError("receipt has no compact evidence")
        sources = [
            f"{ops.REMOTE_HOST}:{context['evidence_dir']}/{name}"
            for name in sorted(records)
        ]
        ops.run_local(["scp", *sources, str(destination)], timeout=300, phase="evidence_transfer")
        for name, record in records.items():
            path = destination / name
            if not path.is_file() or path.stat().st_size != record["bytes"] \
                    or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
                raise TerminalArchiveError(f"receipt evidence identity mismatch: {name}")
        return {
            "records": records,
            "closeout_filename": context["validated_closeout_filename"],
            "closeout_sha256": context["validated_closeout_sha256"],
        }
    except ops.ToolError as exc:
        raise TerminalArchiveError(f"receipt evidence fetch failed: {exc}") from exc


def index_record(audit: dict[str, Any]) -> dict[str, Any]:
    record = {
        "schema_version": 2 if audit.get("contract_bundle") else 1,
        "route_id": audit["route_id"],
        "operation_id": audit["operation_id"],
        "run_id": audit["run_id"],
        "state": audit["state"],
        "decision": audit["decision"],
        "authorizes": audit["authorizes"],
        "receipt": audit["receipt"],
        "route_commit": audit["route_commit"],
        "contract_path": audit["contract_path"],
        "closeout_path": audit["closeout_path"],
        "conclusion_path": audit["conclusion_path"],
        "result_paths": [item["path"] for item in audit["result_files"]],
    }
    if audit.get("contract_bundle"):
        record["contract_bundle"] = audit["contract_bundle"]
        record["prior_terminal_record"] = audit["prior_terminal_record"]
        record["result_files"] = audit["result_files"]
        payloads = audit["payloads"]
        record["contract_sha256"] = hashlib.sha256(
            payloads[audit["contract_path"]]
        ).hexdigest()
        record["closeout_sha256"] = hashlib.sha256(
            payloads[audit["closeout_path"]]
        ).hexdigest()
        record["conclusion_sha256"] = hashlib.sha256(
            payloads[audit["conclusion_path"]]
        ).hexdigest()
    return record


def read_index(raw: bytes) -> list[dict[str, Any]]:
    if not raw:
        return []
    records = []
    for number, line in enumerate(checked_text(raw, INDEX_PATH).splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TerminalArchiveError(f"invalid terminal index line {number}: {exc}") from exc
        if not isinstance(value, dict):
            raise TerminalArchiveError(f"terminal index line {number} is not an object")
        records.append(value)
    return records


def prepare_destination(
    destination_repo: Path,
    base_ref: str,
    audit: dict[str, Any],
) -> dict[str, Any]:
    if base_ref != ARCHIVE_BASE_REF:
        raise TerminalArchiveError(
            f"archive base_ref must be {ARCHIVE_BASE_REF}: {base_ref}"
        )
    destination_repo = destination_repo.resolve()
    if git(destination_repo, "status", "--porcelain"):
        raise TerminalArchiveError("destination archive worktree must be clean")
    head = git(destination_repo, "rev-parse", "HEAD")
    base = git(destination_repo, "rev-parse", base_ref)
    if head != base:
        raise TerminalArchiveError(
            f"destination HEAD must equal current main: HEAD={head} main={base}"
        )

    payloads: dict[str, bytes] = audit["payloads"]
    planned: dict[str, bytes] = {}
    for relpath, raw in payloads.items():
        safe_relative(relpath)
        target = destination_repo / relpath
        if target.exists():
            if target.read_bytes() != raw:
                raise TerminalArchiveError(f"destination has conflicting evidence: {relpath}")
        else:
            planned[relpath] = raw

    index_file = destination_repo / INDEX_PATH
    current_index = index_file.read_bytes() if index_file.exists() else b""
    records = read_index(current_index)
    record = index_record(audit)
    key = (record["route_id"], record["operation_id"], record["run_id"])
    duplicate = [item for item in records if (
        item.get("route_id"), item.get("operation_id"), item.get("run_id")
    ) == key]
    if duplicate and duplicate != [record]:
        raise TerminalArchiveError("terminal index contains a conflicting identity")
    if not duplicate:
        records.append(record)
        index_raw = b"".join(
            (json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
            for item in records
        )
        planned[INDEX_PATH] = index_raw

    new_capability = audit.get("capability_registry_record")
    if new_capability is not None:
        registry_file = destination_repo / capability_registry.REGISTRY_RELPATH
        registry_raw = registry_file.read_bytes() if registry_file.exists() else b""
        try:
            existing_capabilities = capability_registry.load_records(
                registry_raw.decode("utf-8").splitlines(),
                evidence_exists=lambda relpath: (destination_repo / relpath).is_file(),
                read_evidence=lambda relpath: (destination_repo / relpath).read_bytes(),
            )
            reuse = capability_registry.lookup(
                existing_capabilities, new_capability["identity"],
            )
        except (
            UnicodeDecodeError, OSError, capability_registry.CapabilityRegistryError,
        ) as exc:
            raise TerminalArchiveError(f"capability registry is invalid: {exc}") from exc
        if reuse["engineering_reuse_authorized"] is False:
            if any(
                item["qualification_id"] == new_capability["qualification_id"]
                for item in existing_capabilities
            ):
                raise TerminalArchiveError("capability qualification_id conflicts")
            planned[capability_registry.REGISTRY_RELPATH] = (
                registry_raw + capability_registry.canonical_bytes(new_capability)
            )

    for relpath, raw in planned.items():
        target = destination_repo / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)

    if planned:
        git(destination_repo, "add", "--", *sorted(planned))
        git(destination_repo, "diff", "--cached", "--check")
        staged = git(destination_repo, "diff", "--cached", "--name-only").splitlines()
        if staged != sorted(planned):
            raise TerminalArchiveError("staged archive paths differ from the prepared bundle")
    else:
        staged = []

    return {
        "schema_version": 1,
        "status": "TERMINAL_ARCHIVE_PREPARED",
        "route_id": audit["route_id"],
        "base_commit": base,
        "staged_paths": staged,
        "preserved_result_files": len(audit["result_files"]),
        "manual_parse_steps": 0,
        "duplicative_document_updates": 0,
        "post_terminal_cleanup_steps": 0,
        "remaining_operator_steps": ["commit", "push"],
    }


def refresh_remote_target(
    destination_repo: Path, *, remote: str, target_ref: str,
) -> tuple[str, str]:
    remote_ref = f"refs/remotes/{remote}/{target_ref}"
    git(
        destination_repo, "fetch", remote,
        f"refs/heads/{target_ref}:{remote_ref}",
    )
    return remote_ref, git(destination_repo, "rev-parse", remote_ref)


def finalize_destination(
    destination_repo: Path,
    route_id: str,
    expected_paths: list[str],
    audit: dict[str, Any],
    *,
    base_ref: str = ARCHIVE_BASE_REF,
    remote: str = ARCHIVE_REMOTE,
    target_ref: str = ARCHIVE_TARGET_REF,
    recovery_attempted: bool = False,
) -> dict[str, Any]:
    if base_ref != ARCHIVE_BASE_REF:
        raise TerminalArchiveError(
            f"archive base_ref must be {ARCHIVE_BASE_REF}: {base_ref}"
        )
    if remote != ARCHIVE_REMOTE or target_ref != ARCHIVE_TARGET_REF:
        raise TerminalArchiveError("archive destination must be github/main")
    destination_repo = destination_repo.resolve()
    staged = git(destination_repo, "diff", "--cached", "--name-only").splitlines()
    if staged != expected_paths or not staged:
        raise TerminalArchiveError("finalize staged paths differ from prepared archive")
    if git(destination_repo, "diff", "--name-only"):
        raise TerminalArchiveError("unstaged tracked changes appeared before finalize")
    if git(destination_repo, "ls-files", "--others", "--exclude-standard"):
        raise TerminalArchiveError("untracked files appeared before finalize")
    base_commit = git(destination_repo, "rev-parse", base_ref)
    git(destination_repo, "commit", "-m", f"Archive terminal evidence for {route_id}")
    evidence_commit = git(destination_repo, "rev-parse", "HEAD")
    try:
        git(destination_repo, "push", remote, f"HEAD:{target_ref}")
    except TerminalArchiveError as push_error:
        if recovery_attempted:
            raise TerminalArchiveError(
                "ARCHIVE_CONCURRENT_MAIN_ADVANCE recovery push failed: "
                + str(push_error)
            ) from push_error
        remote_ref, remote_commit = refresh_remote_target(
            destination_repo, remote=remote, target_ref=target_ref,
        )
        if remote_commit == evidence_commit:
            return {
                "status": "TERMINAL_ARCHIVE_PUSHED",
                "evidence_commit": evidence_commit,
                "remote": remote,
                "target_ref": target_ref,
                "remote_commit": remote_commit,
                "push_verified_after_transport_error": True,
                "remaining_operator_steps": [],
            }
        if remote_commit == base_commit:
            raise push_error
        if git(destination_repo, "merge-base", base_commit, remote_commit) != base_commit:
            raise TerminalArchiveError(
                "ARCHIVE_CONCURRENT_MAIN_ADVANCE is not a fast-forward"
            ) from push_error
        changed_paths = git(
            destination_repo, "diff", "--name-only",
            f"{base_commit}..{evidence_commit}",
        ).splitlines()
        if changed_paths != expected_paths:
            raise TerminalArchiveError(
                "ARCHIVE_CONCURRENT_MAIN_ADVANCE local commit changed unexpected paths"
            ) from push_error
        git(destination_repo, "checkout", "--detach", remote_ref)
        recovered = prepare_destination(destination_repo, base_ref, audit)
        if not recovered["staged_paths"]:
            return {
                "status": "TERMINAL_ARCHIVE_ALREADY_PRESENT",
                "evidence_commit": remote_commit,
                "remote": remote,
                "target_ref": target_ref,
                "remote_commit": remote_commit,
                "concurrent_main_advance_recovered": True,
                "initial_evidence_commit": evidence_commit,
                "remaining_operator_steps": [],
            }
        final = finalize_destination(
            destination_repo, route_id, recovered["staged_paths"], audit,
            base_ref=base_ref, remote=remote, target_ref=target_ref,
            recovery_attempted=True,
        )
        final["concurrent_main_advance_recovered"] = True
        final["initial_evidence_commit"] = evidence_commit
        return final
    remote_line = git(
        destination_repo, "ls-remote", "--heads", remote, f"refs/heads/{target_ref}",
    )
    remote_commit = remote_line.split()[0] if remote_line else ""
    if remote_commit != evidence_commit:
        raise TerminalArchiveError(
            f"remote archive identity mismatch: local={evidence_commit} remote={remote_commit}"
        )
    return {
        "status": "TERMINAL_ARCHIVE_PUSHED",
        "evidence_commit": evidence_commit,
        "remote": remote,
        "target_ref": target_ref,
        "remote_commit": remote_commit,
        "remaining_operator_steps": [],
    }


def serializable(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != "payloads"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--source-ref", default="HEAD")
    parser.add_argument("--route-id", required=True)
    parser.add_argument("--closeout", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--conclusion")
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--existing-archive", action="store_true")
    parser.add_argument("--destination-repo", type=Path)
    parser.add_argument(
        "--base-ref", default=ARCHIVE_BASE_REF, choices=[ARCHIVE_BASE_REF],
    )
    parser.add_argument("--commit-and-push", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--local-evidence-only", action="store_true")
    parser.add_argument("--remote", default=ARCHIVE_REMOTE, choices=[ARCHIVE_REMOTE])
    parser.add_argument(
        "--target-ref", default=ARCHIVE_TARGET_REF, choices=[ARCHIVE_TARGET_REF],
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.existing_archive and not args.audit_only:
        parser.error("--existing-archive is audit-only")
    if args.prepare_only and args.commit_and_push:
        parser.error("--prepare-only conflicts with --commit-and-push")
    if args.local_evidence_only and not args.audit_only:
        parser.error("--local-evidence-only is audit-only")
    if not args.audit_only and args.destination_repo is None:
        parser.error("--destination-repo is required unless --audit-only is used")
    try:
        with tempfile.TemporaryDirectory(prefix="terminal-evidence-") as temporary:
            evidence_override = None
            receipt_binding = None
            if not args.local_evidence_only:
                evidence_override = Path(temporary)
                receipt_binding = fetch_receipt_evidence(args.receipt, evidence_override)
            audit = audit_source(
                args.source_repo, args.source_ref, args.route_id,
                args.closeout, args.contract, args.conclusion, args.receipt,
                existing_archive=args.existing_archive,
                evidence_dir_override=evidence_override,
                conclusion_dir_override=(
                    args.source_repo / PurePosixPath(args.closeout).parent
                ),
                expected_closeout_filename=(
                    receipt_binding["closeout_filename"] if receipt_binding else None
                ),
                expected_closeout_sha256=(
                    receipt_binding["closeout_sha256"] if receipt_binding else None
                ),
            )
        report = serializable(audit)
        if not args.audit_only:
            report["archive"] = prepare_destination(
                args.destination_repo, args.base_ref, audit,
            )
            if not args.prepare_only:
                report["archive"]["finalize"] = finalize_destination(
                    args.destination_repo,
                    args.route_id,
                    report["archive"]["staged_paths"],
                    audit,
                    base_ref=args.base_ref,
                    remote=args.remote,
                    target_ref=args.target_ref,
                )
                report["archive"]["remaining_operator_steps"] = []
            report["status"] = "TERMINAL_ARCHIVE_FASTPATH_OK"
    except TerminalArchiveError as exc:
        print(f"TERMINAL_ARCHIVE_ERROR {exc}")
        raise SystemExit(1)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
    print(json.dumps(report, sort_keys=True))
    print(f"TERMINAL_ARCHIVE_OK route_id={report['route_id']} status={report['status']}")


if __name__ == "__main__":
    main()
