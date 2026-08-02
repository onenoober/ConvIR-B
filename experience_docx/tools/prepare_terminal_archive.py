#!/usr/bin/env python3
"""Prepare one minimal, complete terminal-science archive for GitHub main."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import math
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
SHA40 = re.compile(r"^[0-9a-f]{40}$")
RAW_ARTIFACT_RECEIPT_SUFFIX = "_raw_artifact_receipt.json"
RAW_ARTIFACT_RECOVERY_SUFFIX = "_raw_artifact_receipt_recovery.json"
RAW_ARTIFACT_MANIFEST_RELPATH = "control/raw_artifact_manifest.jsonl"
RAW_ARTIFACT_SCOPE_ROOTS = ["contract", "workload"]
RAW_ARTIFACT_EXCLUDED_PATHS = [
    "control", "heartbeat.json", "runtime.log", "status.txt",
]
MAX_RAW_ARTIFACT_FILES = 25_000
MAX_RAW_ARTIFACT_MANIFEST_BYTES = 32 * 1024 * 1024
RAW_ARTIFACT_RECOVERY_MARKER = "CONVIR_RAW_ARTIFACT_RECOVERY_OK"
REVIEW_FACTS_SUFFIX = "_review_facts.json"
REVIEW_FACTS_RECOVERY_SUFFIX = "_review_facts_recovery.json"
REVIEW_FACTS_RECOVERY_TYPE = "legacy_unbound_gate_confidence_metadata_v1"
MAX_REVIEW_FACTS = 128
REVIEW_FACTS_RULES_FLOOR = "ef1f746859fba84bd76ae74525b05f918994909f"
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


def rules_require_review_facts(repo: Path, rules_commit: Any) -> bool:
    if not isinstance(rules_commit, str) or not SHA40.fullmatch(rules_commit):
        return False
    if subprocess.run(
        [GIT, "cat-file", "-e", f"{REVIEW_FACTS_RULES_FLOOR}^{{commit}}"],
        cwd=repo, capture_output=True, timeout=30, check=False,
    ).returncode:
        return False
    completed = subprocess.run(
        [GIT, "merge-base", "--is-ancestor", REVIEW_FACTS_RULES_FLOOR, rules_commit],
        cwd=repo, capture_output=True, timeout=30, check=False,
    )
    if completed.returncode not in {0, 1}:
        raise TerminalArchiveError("cannot resolve review-facts rules ancestry")
    return completed.returncode == 0


def safe_relative(value: str, *, prefix: str | None = None) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value:
        raise TerminalArchiveError(f"unsafe repository path: {value}")
    if prefix is not None and not value.startswith(prefix):
        raise TerminalArchiveError(f"path is outside {prefix}: {value}")
    return path


def canonical_conclusion_relpath(closeout_relpath: str) -> str:
    path = safe_relative(closeout_relpath)
    suffix = "_closeout.json"
    if not path.name.endswith(suffix):
        raise TerminalArchiveError("closeout must end with _closeout.json")
    return path.with_name(path.name[:-len(suffix)] + "_conclusion.json").as_posix()


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
    required_schema_version: int | tuple[int, ...] | None = None,
) -> dict[str, Any]:
    value = inspect_structured(raw, relpath)
    allowed_schema_versions = (
        required_schema_version
        if isinstance(required_schema_version, tuple)
        else (required_schema_version,)
    )
    if required_schema_version is not None and (
        not isinstance(value, dict)
        or value.get("schema_version") not in allowed_schema_versions
    ):
        raise TerminalArchiveError(
            "scientific conclusion schema_version must be one of "
            f"{allowed_schema_versions}"
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
    if value.get("schema_version") == 3:
        for key in ("primary_fact_ids", "gate_fact_ids"):
            selected = value.get(key)
            if not isinstance(selected, list) or not selected \
                    or len(selected) != len(set(selected)) \
                    or any(not isinstance(item, str) or not TOKEN.fullmatch(item) for item in selected):
                raise TerminalArchiveError(
                    f"schema-3 scientific conclusion {key} is invalid"
                )
    return value


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


def raw_artifact_receipt_filename(closeout_filename: str) -> str:
    suffix = "_closeout.json"
    if not isinstance(closeout_filename, str) or not closeout_filename.endswith(suffix):
        raise TerminalArchiveError("closeout filename cannot derive raw artifact receipt")
    return closeout_filename[:-len(suffix)] + RAW_ARTIFACT_RECEIPT_SUFFIX


def raw_artifact_recovery_filename(closeout_filename: str) -> str:
    suffix = "_closeout.json"
    if not isinstance(closeout_filename, str) or not closeout_filename.endswith(suffix):
        raise TerminalArchiveError("closeout filename cannot derive raw artifact recovery")
    return closeout_filename[:-len(suffix)] + RAW_ARTIFACT_RECOVERY_SUFFIX


def review_facts_filename(closeout_filename: str) -> str:
    suffix = "_closeout.json"
    if not isinstance(closeout_filename, str) or not closeout_filename.endswith(suffix):
        raise TerminalArchiveError("closeout filename cannot derive review facts")
    return closeout_filename[:-len(suffix)] + REVIEW_FACTS_SUFFIX


def review_facts_recovery_filename(closeout_filename: str) -> str:
    suffix = "_closeout.json"
    if not isinstance(closeout_filename, str) or not closeout_filename.endswith(suffix):
        raise TerminalArchiveError("closeout filename cannot derive review facts recovery")
    return closeout_filename[:-len(suffix)] + REVIEW_FACTS_RECOVERY_SUFFIX


def _json_primitive(value: Any, name: str) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise TerminalArchiveError(f"review fact {name} must be one finite JSON primitive")


def _json_pointer(value: Any, pointer: str, name: str) -> Any:
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise TerminalArchiveError(f"review fact {name} JSON Pointer is invalid")
    current = value
    for raw_token in pointer[1:].split("/"):
        token = raw_token.replace("~1", "/").replace("~0", "~")
        if "~" in raw_token.replace("~0", "").replace("~1", ""):
            raise TerminalArchiveError(f"review fact {name} JSON Pointer escape is invalid")
        if isinstance(current, dict):
            if token not in current:
                raise TerminalArchiveError(f"review fact {name} JSON Pointer is absent")
            current = current[token]
        elif isinstance(current, list):
            if not token.isdigit() or (len(token) > 1 and token.startswith("0")):
                raise TerminalArchiveError(f"review fact {name} array index is invalid")
            index = int(token)
            if index >= len(current):
                raise TerminalArchiveError(f"review fact {name} array index is absent")
            current = current[index]
        else:
            raise TerminalArchiveError(f"review fact {name} JSON Pointer crosses a scalar")
    return current


def validate_review_facts(
    raw: bytes, relpath: str, closeout: dict[str, Any], conclusion: dict[str, Any],
    payloads: dict[str, bytes], evidence_sha256: dict[str, str],
    closeout_filename: str,
) -> dict[str, Any]:
    value = inspect_structured(raw, relpath)
    expected_top = {"schema_version", "route_id", "operation_id", "run_id", "facts"}
    if not isinstance(value, dict) or set(value) != expected_top \
            or value["schema_version"] != 2:
        raise TerminalArchiveError("review facts top-level contract is invalid")
    if PurePosixPath(relpath).name != review_facts_filename(closeout_filename):
        raise TerminalArchiveError("review facts filename is not canonical")
    for key in ("route_id", "operation_id", "run_id"):
        if value[key] != closeout.get(key):
            raise TerminalArchiveError(f"review facts {key} mismatch")
    facts = value["facts"]
    if not isinstance(facts, list) or not 1 <= len(facts) <= MAX_REVIEW_FACTS:
        raise TerminalArchiveError("review facts must contain 1-128 entries")
    expected_fact = {
        "fact_id", "claim_id", "metric", "unit", "population", "grouping",
        "point", "ci_lower", "ci_upper", "confidence_level", "threshold",
        "threshold_operator", "gate_outcome", "source_filename", "source_sha256",
        "json_pointers",
    }
    pointer_fields = {
        "point", "ci_lower", "ci_upper", "confidence_level", "threshold",
        "gate_outcome",
    }
    ids = set()
    by_id = {}
    evidence_parent = PurePosixPath(relpath).parent.as_posix()
    for fact in facts:
        if not isinstance(fact, dict) or set(fact) != expected_fact:
            raise TerminalArchiveError("review fact field contract is invalid")
        for key in ("fact_id", "claim_id"):
            if not isinstance(fact[key], str) or not TOKEN.fullmatch(fact[key]):
                raise TerminalArchiveError(f"review fact {key} is invalid")
        if fact["fact_id"] in ids:
            raise TerminalArchiveError("review fact ids must be unique")
        ids.add(fact["fact_id"])
        by_id[fact["fact_id"]] = fact
        for key in ("metric", "unit", "population", "grouping"):
            if not isinstance(fact[key], str) or not 1 <= len(fact[key].strip()) <= 256:
                raise TerminalArchiveError(f"review fact {key} is invalid")
        pointers = fact["json_pointers"]
        if not isinstance(pointers, dict) or set(pointers) != pointer_fields:
            raise TerminalArchiveError("review fact JSON Pointer contract is invalid")
        for key in pointer_fields:
            declared = _json_primitive(fact[key], key)
            pointer = pointers[key]
            if (declared is None) != (pointer is None):
                raise TerminalArchiveError(f"review fact {key} pointer/value presence differs")
            if pointer is not None and not isinstance(pointer, str):
                raise TerminalArchiveError(f"review fact {key} pointer is invalid")
        for key in ("point", "ci_lower", "ci_upper", "confidence_level", "threshold"):
            if fact[key] is not None and (
                not isinstance(fact[key], (int, float)) or isinstance(fact[key], bool)
            ):
                raise TerminalArchiveError(f"review fact {key} must be numeric")
        if fact["gate_outcome"] is not None and (
            not isinstance(fact["gate_outcome"], str)
            or not TOKEN.fullmatch(fact["gate_outcome"])
        ):
            raise TerminalArchiveError("review fact gate_outcome is invalid")
        if fact["point"] is None and fact["gate_outcome"] is None:
            raise TerminalArchiveError("review fact requires a point or gate outcome")
        if (fact["ci_lower"] is None) != (fact["ci_upper"] is None) \
                or (fact["ci_lower"] is None) != (fact["confidence_level"] is None):
            raise TerminalArchiveError("review fact confidence interval is incomplete")
        if fact["ci_lower"] is not None and (
            fact["ci_lower"] > fact["ci_upper"]
            or not 0 < fact["confidence_level"] < 1
        ):
            raise TerminalArchiveError("review fact confidence interval is invalid")
        if fact["threshold"] is None:
            if fact["threshold_operator"] is not None:
                raise TerminalArchiveError("review fact threshold operator lacks a threshold")
        elif fact["threshold_operator"] not in {">", ">=", "<", "<=", "=="}:
            raise TerminalArchiveError("review fact threshold operator is invalid")
        source_name = fact["source_filename"]
        if not isinstance(source_name, str) or Path(source_name).name != source_name \
                or Path(source_name).suffix.lower() != ".json" \
                or source_name == PurePosixPath(relpath).name:
            raise TerminalArchiveError("review fact source filename is invalid")
        if fact["source_sha256"] != evidence_sha256.get(source_name) \
                or not isinstance(fact["source_sha256"], str) \
                or not SHA256.fullmatch(fact["source_sha256"]):
            raise TerminalArchiveError("review fact source SHA-256 is not closeout-bound")
        source_relpath = f"{evidence_parent}/{source_name}"
        source_raw = payloads.get(source_relpath)
        if source_raw is None or hashlib.sha256(source_raw).hexdigest() != fact["source_sha256"]:
            raise TerminalArchiveError("review fact source payload is absent or changed")
        source = inspect_structured(source_raw, source_relpath)
        for key in pointer_fields:
            if pointers[key] is None:
                continue
            observed = _json_primitive(_json_pointer(source, pointers[key], key), key)
            both_numeric = all(
                isinstance(item, (int, float)) and not isinstance(item, bool)
                for item in (observed, fact[key])
            )
            if observed != fact[key] or (
                not both_numeric and type(observed) is not type(fact[key])
            ):
                raise TerminalArchiveError(f"review fact {key} differs from its source")
    for key in ("primary_fact_ids", "gate_fact_ids"):
        selected = conclusion.get(key)
        if not isinstance(selected, list) or not selected \
                or len(selected) != len(set(selected)) \
                or any(item not in ids for item in selected):
            raise TerminalArchiveError(f"scientific conclusion {key} is invalid")
    if any(by_id[item]["gate_outcome"] is None for item in conclusion["gate_fact_ids"]):
        raise TerminalArchiveError("conclusion gate facts lack gate outcomes")
    if conclusion.get("schema_version") == 3:
        primary_facts = [by_id[item] for item in conclusion["primary_fact_ids"]]
        if not any(
            fact["point"] is not None
            and fact["json_pointers"]["point"] is not None
            for fact in primary_facts
        ):
            raise TerminalArchiveError(
                "schema-3 conclusion requires a source-bound primary point estimate"
            )
    return value


def build_review_facts_recovery(
    *, raw: bytes, relpath: str, closeout: dict[str, Any],
    conclusion: dict[str, Any], payloads: dict[str, bytes],
    evidence_sha256: dict[str, str], closeout_filename: str,
    closeout_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], str, bytes]:
    """Recover only the legacy unbound confidence metadata on pure gate facts."""
    if not isinstance(closeout_sha256, str) or not SHA256.fullmatch(closeout_sha256):
        raise TerminalArchiveError("review facts recovery closeout identity is invalid")
    original_filename = PurePosixPath(relpath).name
    if original_filename != review_facts_filename(closeout_filename):
        raise TerminalArchiveError("review facts recovery source filename is not canonical")
    original_sha256 = hashlib.sha256(raw).hexdigest()
    if evidence_sha256.get(original_filename) != original_sha256:
        raise TerminalArchiveError("review facts recovery source is not closeout-bound")
    original = inspect_structured(raw, relpath)
    if not isinstance(original, dict) or not isinstance(original.get("facts"), list):
        raise TerminalArchiveError("review facts are not eligible for recovery")

    recovered = copy.deepcopy(original)
    changes = []
    for fact in recovered["facts"]:
        if not isinstance(fact, dict):
            continue
        pointers = fact.get("json_pointers")
        confidence = fact.get("confidence_level")
        numeric_confidence = (
            isinstance(confidence, (int, float))
            and not isinstance(confidence, bool)
            and math.isfinite(confidence)
            and 0 < confidence < 1
        )
        pure_gate_with_unbound_confidence = (
            numeric_confidence
            and fact.get("point") is None
            and fact.get("ci_lower") is None
            and fact.get("ci_upper") is None
            and fact.get("threshold") is None
            and fact.get("threshold_operator") is None
            and fact.get("gate_outcome") is not None
            and isinstance(pointers, dict)
            and pointers.get("point") is None
            and pointers.get("ci_lower") is None
            and pointers.get("ci_upper") is None
            and pointers.get("confidence_level") is None
            and pointers.get("threshold") is None
            and pointers.get("gate_outcome") is not None
        )
        if pure_gate_with_unbound_confidence:
            changes.append({
                "fact_id": fact.get("fact_id"),
                "field": "confidence_level",
                "original_value": confidence,
                "recovered_value": None,
            })
            fact["confidence_level"] = None
    if not changes:
        raise TerminalArchiveError(
            "review facts do not exhibit the recoverable legacy confidence defect"
        )

    recovered_raw = (
        json.dumps(recovered, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    validated = validate_review_facts(
        recovered_raw, relpath, closeout, conclusion, payloads,
        evidence_sha256, closeout_filename,
    )
    source_bindings = sorted(
        {
            (fact["source_filename"], fact["source_sha256"])
            for fact in validated["facts"]
        }
    )
    proof = {
        "schema_version": 1,
        "status": "REVIEW_FACTS_RECOVERED",
        "recovery_type": REVIEW_FACTS_RECOVERY_TYPE,
        "route_id": closeout["route_id"],
        "operation_id": closeout["operation_id"],
        "run_id": closeout["run_id"],
        "route_commit": closeout["route_commit"],
        "closeout_filename": closeout_filename,
        "closeout_sha256": closeout_sha256,
        "original_review_facts_filename": original_filename,
        "original_review_facts_sha256": original_sha256,
        "recovered_review_facts_sha256": hashlib.sha256(recovered_raw).hexdigest(),
        "recovered_serialization": "canonical_json_sort_keys_compact_newline_v1",
        "source_bindings": [
            {"filename": filename, "sha256": sha256}
            for filename, sha256 in source_bindings
        ],
        "changes": changes,
        "recovered_review_facts": validated,
        "checks": {
            "closeout_receipt_bound": True,
            "original_review_facts_closeout_bound": True,
            "only_unbound_gate_confidence_removed": True,
            "recovered_review_facts_strictly_valid": True,
            "source_bindings_unchanged": True,
            "terminal_identity_unchanged": True,
        },
    }
    proof_raw = (
        json.dumps(proof, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    proof_filename = review_facts_recovery_filename(closeout_filename)
    proof_relpath = f"{PurePosixPath(relpath).parent.as_posix()}/{proof_filename}"
    return validated, proof, proof_relpath, proof_raw


def parse_raw_artifact_receipt(
    raw: bytes, relpath: str, closeout: dict[str, Any], closeout_filename: str,
) -> dict[str, Any]:
    value = inspect_structured(raw, relpath)
    expected_fields = {
        "schema_version", "route_id", "operation_id", "run_id", "route_commit",
        "manifest_relative_path", "manifest_sha256", "entry_count", "total_bytes",
        "category_counts", "scope_roots", "excluded_paths",
    }
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise TerminalArchiveError("raw artifact receipt field contract is invalid")
    expected_name = raw_artifact_receipt_filename(closeout_filename)
    if PurePosixPath(relpath).name != expected_name:
        raise TerminalArchiveError("raw artifact receipt filename is not canonical")
    if value["schema_version"] != 2:
        raise TerminalArchiveError("raw artifact receipt schema_version must equal 2")
    for key in ("route_id", "operation_id", "run_id", "route_commit"):
        if value[key] != closeout.get(key):
            raise TerminalArchiveError(f"raw artifact receipt {key} mismatch")
    if not SHA40.fullmatch(value["route_commit"]):
        raise TerminalArchiveError("raw artifact receipt route_commit is invalid")
    if (
        value["manifest_relative_path"] != RAW_ARTIFACT_MANIFEST_RELPATH
        or not SHA256.fullmatch(value["manifest_sha256"])
    ):
        raise TerminalArchiveError("raw artifact manifest identity is invalid")
    for key, maximum in (("entry_count", MAX_RAW_ARTIFACT_FILES), ("total_bytes", None)):
        item = value[key]
        if (
            not isinstance(item, int) or isinstance(item, bool) or item < 0
            or (maximum is not None and item > maximum)
        ):
            raise TerminalArchiveError(f"raw artifact receipt {key} is invalid")
    expected_categories = {"contract_output", "workload_output"}
    categories = value["category_counts"]
    if (
        not isinstance(categories, dict) or set(categories) != expected_categories
        or any(
            not isinstance(item, int) or isinstance(item, bool) or item < 0
            for item in categories.values()
        )
    ):
        raise TerminalArchiveError("raw artifact receipt category counts are invalid")
    if (
        value["scope_roots"] != RAW_ARTIFACT_SCOPE_ROOTS
        or value["excluded_paths"] != RAW_ARTIFACT_EXCLUDED_PATHS
    ):
        raise TerminalArchiveError("raw artifact receipt scope contract is invalid")
    return value


def validate_raw_artifact_receipt(
    raw: bytes, relpath: str, closeout: dict[str, Any], closeout_filename: str,
) -> dict[str, Any]:
    value = parse_raw_artifact_receipt(raw, relpath, closeout, closeout_filename)
    if sum(value["category_counts"].values()) != value["entry_count"]:
        raise TerminalArchiveError("raw artifact receipt category counts are invalid")
    return value


def raw_artifact_manifest_recovery_body(
    context: dict[str, Any], receipt_name: str, receipt_sha256: str,
    receipt: dict[str, Any],
) -> str:
    """Build one fixed, receipt-bound cloud verifier for the legacy class bug."""
    validate_evidence_name(receipt_name)
    if not SHA256.fullmatch(receipt_sha256):
        raise TerminalArchiveError("raw artifact receipt SHA-256 is invalid")
    manifest_sha = receipt.get("manifest_sha256")
    if not isinstance(manifest_sha, str) or not SHA256.fullmatch(manifest_sha):
        raise TerminalArchiveError("raw artifact manifest SHA-256 is invalid")
    closeout_name = context["validated_closeout_filename"]
    validate_evidence_name(closeout_name)
    closeout_sha = context["validated_closeout_sha256"]
    if not isinstance(closeout_sha, str) or not SHA256.fullmatch(closeout_sha):
        raise TerminalArchiveError("receipt-bound closeout SHA-256 is invalid")
    counts = receipt.get("category_counts")
    if not isinstance(counts, dict) or set(counts) != {
        "contract_output", "workload_output",
    }:
        raise TerminalArchiveError("raw artifact receipt categories are invalid")
    numeric = [
        receipt.get("entry_count"), receipt.get("total_bytes"),
        counts["contract_output"], counts["workload_output"],
    ]
    if any(not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in numeric):
        raise TerminalArchiveError("raw artifact receipt counts are invalid")
    output_id = context.get("output_id")
    output_value = context.get("output_path")
    run_root_value = context.get("run_root")
    if not isinstance(output_id, str) or not TOKEN.fullmatch(output_id) \
            or not isinstance(output_value, str) \
            or not isinstance(run_root_value, str):
        raise TerminalArchiveError("receipt output identity is invalid")
    output_path = PurePosixPath(output_value)
    run_root = PurePosixPath(run_root_value)
    if not output_path.is_absolute() or not run_root.is_absolute() \
            or output_path.parent != run_root or output_path.name != output_id:
        raise TerminalArchiveError("receipt output path is invalid")
    lines = [
        "set -euo pipefail",
        "export LC_ALL=C",
        f"EVIDENCE_DIR={ops.q(context['evidence_dir'])}",
        'test -d "$EVIDENCE_DIR"',
        'test ! -L "$EVIDENCE_DIR"',
        'test "$(readlink -f -- "$EVIDENCE_DIR")" = "$EVIDENCE_DIR"',
        f'VALIDATED_CLOSEOUT="$EVIDENCE_DIR/{closeout_name}"',
        'test -f "$VALIDATED_CLOSEOUT"',
        'test ! -L "$VALIDATED_CLOSEOUT"',
        'test "$(readlink -f -- "$VALIDATED_CLOSEOUT")" = "$VALIDATED_CLOSEOUT"',
        f'test "$(sha256sum "$VALIDATED_CLOSEOUT" | awk \'{{print $1}}\')" = {ops.q(closeout_sha)}',
        f'RECEIPT="$EVIDENCE_DIR/{receipt_name}"',
        'test -f "$RECEIPT"',
        'test ! -L "$RECEIPT"',
        'test "$(readlink -f -- "$RECEIPT")" = "$RECEIPT"',
        f'test "$(sha256sum "$RECEIPT" | awk \'{{print $1}}\')" = {ops.q(receipt_sha256)}',
        f"OUTPUT_PATH={ops.q(output_value)}",
        'test -d "$OUTPUT_PATH"',
        'test ! -L "$OUTPUT_PATH"',
        'test "$(readlink -f -- "$OUTPUT_PATH")" = "$OUTPUT_PATH"',
        'CONTROL="$OUTPUT_PATH/control"',
        'test -d "$CONTROL"',
        'test ! -L "$CONTROL"',
        'test "$(readlink -f -- "$CONTROL")" = "$CONTROL"',
        'MANIFEST="$CONTROL/raw_artifact_manifest.jsonl"',
        'test -f "$MANIFEST"',
        'test ! -L "$MANIFEST"',
        'test "$(readlink -f -- "$MANIFEST")" = "$MANIFEST"',
        f'test "$(wc -c < "$MANIFEST")" -le {MAX_RAW_ARTIFACT_MANIFEST_BYTES}',
        (
            f'{ops.q(ops.REMOTE_PYTHON)} - "$MANIFEST" {ops.q(manifest_sha)} '
            f'{receipt["entry_count"]} {receipt["total_bytes"]} '
            f'{counts["contract_output"]} {counts["workload_output"]} <<\'PY\''
        ),
        "import hashlib",
        "import json",
        "import re",
        "import sys",
        "from pathlib import Path, PurePosixPath",
        "path = Path(sys.argv[1])",
        "expected_sha = sys.argv[2]",
        "expected_entries, expected_bytes, expected_contract, expected_workload = map(int, sys.argv[3:])",
        "raw = path.read_bytes()",
        "if hashlib.sha256(raw).hexdigest() != expected_sha:",
        "    raise SystemExit('raw artifact manifest SHA-256 mismatch')",
        "if raw and not raw.endswith(b'\\n'):",
        "    raise SystemExit('raw artifact manifest is not newline terminated')",
        "expected_fields = {'schema_version', 'relative_path', 'artifact_class', 'bytes', 'sha256'}",
        "sha256 = re.compile(r'^[0-9a-f]{64}$')",
        "roots = {'contract', 'workload'}",
        "corrected = {'contract_output': 0, 'workload_output': 0}",
        "legacy = {'contract_output': 0, 'workload_output': 0}",
        "entry_count = 0",
        "total_bytes = 0",
        "nested_misclassified = 0",
        "previous = None",
        "for number, line in enumerate(raw.splitlines(), start=1):",
        "    try:",
        "        row = json.loads(line.decode('utf-8'))",
        "    except (UnicodeDecodeError, json.JSONDecodeError) as exc:",
        "        raise SystemExit(f'raw artifact manifest row {number} is invalid: {exc}')",
        "    if not isinstance(row, dict) or set(row) != expected_fields or row['schema_version'] != 2:",
        "        raise SystemExit(f'raw artifact manifest row {number} field contract is invalid')",
        "    relative = row['relative_path']",
        "    if not isinstance(relative, str):",
        "        raise SystemExit(f'raw artifact manifest row {number} path is invalid')",
        "    parsed = PurePosixPath(relative)",
        "    if parsed.is_absolute() or '..' in parsed.parts or len(parsed.parts) < 2 or parsed.parts[0] not in roots or parsed.as_posix() != relative:",
        "        raise SystemExit(f'raw artifact manifest row {number} path is unsafe')",
        "    if previous is not None and relative <= previous:",
        "        raise SystemExit('raw artifact manifest paths are duplicate or unsorted')",
        "    previous = relative",
        "    size = row['bytes']",
        "    if not isinstance(size, int) or isinstance(size, bool) or size < 0:",
        "        raise SystemExit(f'raw artifact manifest row {number} byte count is invalid')",
        "    if not isinstance(row['sha256'], str) or not sha256.fullmatch(row['sha256']):",
        "        raise SystemExit(f'raw artifact manifest row {number} SHA-256 is invalid')",
        "    legacy_class = f'{parsed.parent.as_posix()}_output'",
        "    corrected_class = f'{parsed.parts[0]}_output'",
        "    if row['artifact_class'] != legacy_class:",
        "        raise SystemExit(f'raw artifact manifest row {number} is not from the known legacy producer')",
        "    if legacy_class in legacy:",
        "        legacy[legacy_class] += 1",
        "    corrected[corrected_class] += 1",
        "    if legacy_class != corrected_class:",
        "        nested_misclassified += 1",
        "    entry_count += 1",
        "    total_bytes += size",
        "if entry_count != expected_entries or total_bytes != expected_bytes:",
        "    raise SystemExit('raw artifact manifest totals differ from the receipt')",
        "if legacy != {'contract_output': expected_contract, 'workload_output': expected_workload}:",
        "    raise SystemExit('raw artifact manifest legacy counts differ from the receipt')",
        "if expected_contract + expected_workload == expected_entries or nested_misclassified <= 0:",
        "    raise SystemExit('raw artifact receipt does not exhibit the known nested-class defect')",
        "if sum(corrected.values()) != entry_count or nested_misclassified != entry_count - sum(legacy.values()):",
        "    raise SystemExit('raw artifact recovered categories do not cover the manifest')",
        "print(json.dumps({",
        "    'schema_version': 1,",
        "    'manifest_sha256': expected_sha,",
        "    'entry_count': entry_count,",
        "    'total_bytes': total_bytes,",
        "    'original_category_counts': legacy,",
        "    'recovered_category_counts': corrected,",
        "    'misclassified_nested_entry_count': nested_misclassified,",
        "    'legacy_nested_class_pattern_exact': True,",
        "}, sort_keys=True, separators=(',', ':')))",
        "PY",
        f"echo {RAW_ARTIFACT_RECOVERY_MARKER}",
    ]
    return "\n".join(lines)


def parse_raw_artifact_manifest_summary(output: str) -> dict[str, Any]:
    lines = [
        line for line in output.splitlines()
        if line not in {RAW_ARTIFACT_RECOVERY_MARKER, "CONVIR_REMOTE_SCRIPT_OK", ""}
    ]
    if output.splitlines().count(RAW_ARTIFACT_RECOVERY_MARKER) != 1 or len(lines) != 1:
        raise TerminalArchiveError("raw artifact recovery output is malformed")
    try:
        value = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        raise TerminalArchiveError("raw artifact recovery summary is invalid JSON") from exc
    expected = {
        "schema_version", "manifest_sha256", "entry_count", "total_bytes",
        "original_category_counts", "recovered_category_counts",
        "misclassified_nested_entry_count", "legacy_nested_class_pattern_exact",
    }
    if not isinstance(value, dict) or set(value) != expected or value["schema_version"] != 1:
        raise TerminalArchiveError("raw artifact recovery summary field contract is invalid")
    return value


def fetch_raw_artifact_manifest_summary(
    context: dict[str, Any], receipt_name: str, receipt_sha256: str,
    receipt: dict[str, Any],
) -> dict[str, Any]:
    try:
        output = ops.run_remote(
            raw_artifact_manifest_recovery_body(
                context, receipt_name, receipt_sha256, receipt,
            ),
            timeout=120,
            phase="raw_artifact_recovery",
        )
    except ops.ToolError as exc:
        raise TerminalArchiveError(
            f"raw artifact manifest recovery verification failed: {exc}"
        ) from exc
    return parse_raw_artifact_manifest_summary(output)


def build_raw_artifact_recovery(
    *, summary: dict[str, Any], receipt: dict[str, Any], receipt_name: str,
    receipt_sha256: str, closeout: dict[str, Any], closeout_filename: str,
    closeout_sha256: str, evidence_parent: str,
) -> tuple[dict[str, Any], str, bytes]:
    expected_categories = {"contract_output", "workload_output"}
    for key in ("original_category_counts", "recovered_category_counts"):
        value = summary.get(key)
        if not isinstance(value, dict) or set(value) != expected_categories or any(
            not isinstance(item, int) or isinstance(item, bool) or item < 0
            for item in value.values()
        ):
            raise TerminalArchiveError("raw artifact recovery category counts are invalid")
    numeric = [
        summary.get("entry_count"), summary.get("total_bytes"),
        summary.get("misclassified_nested_entry_count"),
    ]
    if any(not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in numeric):
        raise TerminalArchiveError("raw artifact recovery totals are invalid")
    original = summary["original_category_counts"]
    recovered = summary["recovered_category_counts"]
    if (
        summary.get("schema_version") != 1
        or summary.get("manifest_sha256") != receipt["manifest_sha256"]
        or summary["entry_count"] != receipt["entry_count"]
        or summary["total_bytes"] != receipt["total_bytes"]
        or original != receipt["category_counts"]
        or sum(original.values()) >= receipt["entry_count"]
        or sum(recovered.values()) != receipt["entry_count"]
        or summary["misclassified_nested_entry_count"]
        != receipt["entry_count"] - sum(original.values())
        or summary.get("legacy_nested_class_pattern_exact") is not True
    ):
        raise TerminalArchiveError("raw artifact recovery does not prove the legacy defect")
    if not SHA256.fullmatch(receipt_sha256) or not SHA256.fullmatch(closeout_sha256):
        raise TerminalArchiveError("raw artifact recovery identity is invalid")
    proof = {
        "schema_version": 1,
        "status": "RAW_ARTIFACT_RECEIPT_RECOVERED",
        "recovery_type": "lifecycle_nested_artifact_class_v1",
        "route_id": closeout["route_id"],
        "operation_id": closeout["operation_id"],
        "run_id": closeout["run_id"],
        "route_commit": closeout["route_commit"],
        "closeout_filename": closeout_filename,
        "closeout_sha256": closeout_sha256,
        "receipt_filename": receipt_name,
        "receipt_sha256": receipt_sha256,
        "manifest_relative_path": receipt["manifest_relative_path"],
        "manifest_sha256": receipt["manifest_sha256"],
        "entry_count": receipt["entry_count"],
        "total_bytes": receipt["total_bytes"],
        "original_category_counts": original,
        "recovered_category_counts": recovered,
        "misclassified_nested_entry_count": summary["misclassified_nested_entry_count"],
        "checks": {
            "closeout_identity_receipt_bound": True,
            "receipt_identity_closeout_bound": True,
            "manifest_identity_receipt_bound": True,
            "cloud_manifest_sha256_verified": True,
            "legacy_nested_class_pattern_exact": True,
            "entry_count_matches": True,
            "total_bytes_matches": True,
            "recovered_categories_cover_manifest": True,
        },
    }
    raw = (
        json.dumps(proof, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    filename = raw_artifact_recovery_filename(closeout_filename)
    relpath = f"{evidence_parent}/{filename}"
    return proof, relpath, raw


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
    required_conclusion_schema_version: int | None = None,
    raw_artifact_manifest_summary: dict[str, Any] | None = None,
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
        conclusion_relpath = canonical_conclusion_relpath(closeout_relpath)
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
    try:
        launch_manifest = json.loads(git_blob(
            source_repo, route_commit, "experience_docx/route_operations.json",
        ))
    except json.JSONDecodeError as exc:
        raise TerminalArchiveError("invalid route operations manifest") from exc
    review_facts_required = bool(contract_bundle) and rules_require_review_facts(
        source_repo, launch_manifest.get("rules_commit"),
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
    raw_artifact_receipt = None
    raw_artifact_receipt_raw = None
    raw_artifact_receipt_relpath = None
    raw_artifact_receipt_sha256 = None
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
        conclusion_value = validate_conclusion(
            conclusion_raw, conclusion_relpath, closeout,
            required_schema_version=(
                required_conclusion_schema_version
                if required_conclusion_schema_version is not None
                else ((2, 3) if contract_bundle else None)
            ),
        )
        payloads[conclusion_relpath] = conclusion_raw
    else:
        conclusion_value = None
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
        if filename.endswith(RAW_ARTIFACT_RECEIPT_SUFFIX):
            if raw_artifact_receipt is not None:
                raise TerminalArchiveError("multiple raw artifact receipts are forbidden")
            raw_artifact_receipt = parse_raw_artifact_receipt(
                raw, relpath, closeout, closeout_path.name,
            )
            raw_artifact_receipt_raw = raw
            raw_artifact_receipt_relpath = relpath
            raw_artifact_receipt_sha256 = actual_sha
        payloads[relpath] = raw
        files.append({
            "path": relpath,
            "bytes": len(raw),
            "sha256": actual_sha,
        })
    raw_artifact_recovery = None
    if raw_artifact_receipt is not None:
        category_total = sum(raw_artifact_receipt["category_counts"].values())
        if category_total == raw_artifact_receipt["entry_count"]:
            if raw_artifact_manifest_summary is not None:
                raise TerminalArchiveError(
                    "raw artifact recovery was supplied for a valid receipt"
                )
            validate_raw_artifact_receipt(
                raw_artifact_receipt_raw, raw_artifact_receipt_relpath,
                closeout, closeout_path.name,
            )
        else:
            if raw_artifact_manifest_summary is None \
                    or expected_closeout_sha256 is None:
                raise TerminalArchiveError(
                    "raw artifact receipt category counts are invalid"
                )
            proof, proof_relpath, proof_raw = build_raw_artifact_recovery(
                summary=raw_artifact_manifest_summary,
                receipt=raw_artifact_receipt,
                receipt_name=PurePosixPath(raw_artifact_receipt_relpath).name,
                receipt_sha256=raw_artifact_receipt_sha256,
                closeout=closeout,
                closeout_filename=closeout_path.name,
                closeout_sha256=expected_closeout_sha256,
                evidence_parent=closeout_path.parent.as_posix(),
            )
            payloads[proof_relpath] = proof_raw
            raw_artifact_recovery = {
                "path": proof_relpath,
                "bytes": len(proof_raw),
                "sha256": hashlib.sha256(proof_raw).hexdigest(),
                "proof": proof,
            }
            files.append({
                key: raw_artifact_recovery[key] for key in ("path", "bytes", "sha256")
            })
    elif raw_artifact_manifest_summary is not None:
        raise TerminalArchiveError("raw artifact recovery lacks an original receipt")
    expected_review_facts = review_facts_filename(closeout_path.name)
    facts_candidates = sorted(
        name for name in evidence_sha if name.endswith(REVIEW_FACTS_SUFFIX)
    )
    if facts_candidates and facts_candidates != [expected_review_facts]:
        raise TerminalArchiveError("review facts filename is ambiguous or noncanonical")
    review_facts = None
    review_facts_recovery = None
    if expected_review_facts in evidence_sha:
        if conclusion_value is None:
            raise TerminalArchiveError("review facts require a scientific conclusion")
        facts_relpath = f"{closeout_path.parent.as_posix()}/{expected_review_facts}"
        try:
            review_facts = validate_review_facts(
                payloads[facts_relpath], facts_relpath, closeout, conclusion_value,
                payloads, evidence_sha, closeout_path.name,
            )
        except TerminalArchiveError as original_error:
            if expected_closeout_sha256 is None:
                raise
            try:
                review_facts, proof, proof_relpath, proof_raw = (
                    build_review_facts_recovery(
                        raw=payloads[facts_relpath], relpath=facts_relpath,
                        closeout=closeout, conclusion=conclusion_value,
                        payloads=payloads, evidence_sha256=evidence_sha,
                        closeout_filename=closeout_path.name,
                        closeout_sha256=expected_closeout_sha256,
                    )
                )
            except TerminalArchiveError:
                raise original_error
            payloads[proof_relpath] = proof_raw
            review_facts_recovery = {
                "path": proof_relpath,
                "bytes": len(proof_raw),
                "sha256": hashlib.sha256(proof_raw).hexdigest(),
                "proof": proof,
            }
            files.append({
                key: review_facts_recovery[key] for key in ("path", "bytes", "sha256")
            })
    elif review_facts_required:
        raise TerminalArchiveError("schema-2 terminal requires review facts")
    elif isinstance(conclusion_value, dict) and any(
        key in conclusion_value for key in ("primary_fact_ids", "gate_fact_ids")
    ):
        raise TerminalArchiveError("scientific conclusion references missing review facts")
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
        "raw_artifact_receipt": raw_artifact_receipt,
        "raw_artifact_recovery": raw_artifact_recovery,
        "review_facts": review_facts,
        "review_facts_recovery": review_facts_recovery,
        "payloads": payloads,
        "checks": {
            "contract_bound": True,
            "terminal_tuple_complete": True,
            "all_closeout_bound_results_present": True,
            "all_runtime_required_results_present": True,
            "all_result_hashes_match": True,
            "raw_artifact_receipt_valid": (
                raw_artifact_receipt is not None and raw_artifact_recovery is None
            ),
            "raw_artifact_receipt_recovered": raw_artifact_recovery is not None,
            "raw_artifact_receipt_legacy_unsealed": raw_artifact_receipt is None,
            "review_facts_valid": review_facts is not None,
            "review_facts_original_valid": (
                review_facts is not None and review_facts_recovery is None
            ),
            "review_facts_recovered": review_facts_recovery is not None,
            "review_facts_legacy_unbound": review_facts is None,
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
        archive_contract = context.get("archive_contract")
        conclusion_schema_version = None
        if context.get("route_manifest_schema_version") == 6:
            if not isinstance(archive_contract, dict) \
                    or archive_contract.get("conclusion_schema_version") != 3:
                raise TerminalArchiveError(
                    "receipt lacks the current scientific conclusion contract"
                )
            conclusion_schema_version = archive_contract["conclusion_schema_version"]
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
        raw_artifact_manifest_summary = None
        raw_receipt_name = raw_artifact_receipt_filename(
            context["validated_closeout_filename"]
        )
        if raw_receipt_name in records:
            try:
                raw_receipt = json.loads(
                    (destination / raw_receipt_name).read_text(encoding="utf-8")
                )
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise TerminalArchiveError(
                    "receipt-bound raw artifact receipt is unreadable"
                ) from exc
            categories = raw_receipt.get("category_counts") \
                if isinstance(raw_receipt, dict) else None
            count_values_valid = (
                isinstance(categories, dict)
                and set(categories) == {"contract_output", "workload_output"}
                and all(
                    isinstance(item, int) and not isinstance(item, bool) and item >= 0
                    for item in categories.values()
                )
                and isinstance(raw_receipt.get("entry_count"), int)
                and not isinstance(raw_receipt.get("entry_count"), bool)
                and raw_receipt["entry_count"] >= 0
            )
            if count_values_valid and sum(categories.values()) != raw_receipt["entry_count"]:
                raw_artifact_manifest_summary = fetch_raw_artifact_manifest_summary(
                    context,
                    raw_receipt_name,
                    records[raw_receipt_name]["sha256"],
                    raw_receipt,
                )
        return {
            "records": records,
            "closeout_filename": context["validated_closeout_filename"],
            "closeout_sha256": context["validated_closeout_sha256"],
            "conclusion_schema_version": conclusion_schema_version,
            "raw_artifact_manifest_summary": raw_artifact_manifest_summary,
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
        recovery = audit.get("review_facts_recovery")
        if recovery is not None:
            proof = recovery["proof"]
            evidence_parent = PurePosixPath(audit["closeout_path"]).parent.as_posix()
            record["review_facts_recovery"] = {
                "status": proof["status"],
                "recovery_type": proof["recovery_type"],
                "proof_path": recovery["path"],
                "proof_bytes": recovery["bytes"],
                "proof_sha256": recovery["sha256"],
                "original_path": (
                    f"{evidence_parent}/{proof['original_review_facts_filename']}"
                ),
                "original_sha256": proof["original_review_facts_sha256"],
                "recovered_review_facts_sha256": (
                    proof["recovered_review_facts_sha256"]
                ),
            }
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
    advanced_from = None
    if head != base:
        if git(destination_repo, "merge-base", head, base) != head:
            raise TerminalArchiveError(
                "destination HEAD is not a clean ancestor of current main: "
                f"HEAD={head} main={base}"
            )
        git(destination_repo, "switch", "--detach", base)
        if git(destination_repo, "rev-parse", "HEAD") != base \
                or git(destination_repo, "status", "--porcelain"):
            raise TerminalArchiveError("destination main auto-advance did not settle cleanly")
        advanced_from = head

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
        "destination_main_auto_advanced": advanced_from is not None,
        "destination_previous_head": advanced_from,
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
                required_conclusion_schema_version=(
                    receipt_binding["conclusion_schema_version"]
                    if receipt_binding else None
                ),
                raw_artifact_manifest_summary=(
                    receipt_binding.get("raw_artifact_manifest_summary")
                    if receipt_binding else None
                ),
            )
        report = serializable(audit)
        if not args.audit_only:
            refreshed_ref, _ = refresh_remote_target(
                args.destination_repo,
                remote=args.remote,
                target_ref=args.target_ref,
            )
            if refreshed_ref != args.base_ref:
                raise TerminalArchiveError("refreshed archive base ref is not canonical")
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
