#!/usr/bin/env python3
"""Read-only MCP facade for the commit-bound experiment evidence catalog."""

import base64
from collections import OrderedDict
import hashlib
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import convir_evidence_catalog as catalog
import convir_evidence_cloud_inventory as inventory
import convirctl
import legacy_backfill_registry as legacy


SERVER_NAME = "convir-evidence-review"
SERVER_VERSION = "2.2.0"
WORKSPACE_ROOT_ENV = "CONVIR_EVIDENCE_LOCAL_WORKSPACE_ROOT"
DEFAULT_WORKSPACE_ROOT = "/home/ubuntu/workspace"
TRUSTED_REMOTE_NAME = "github"
TRUSTED_REMOTE_URLS = (
    "git@github.com:onenoober/ConvIR-B.git",
    "https://github.com/onenoober/ConvIR-B.git",
)
TRUSTED_MAIN_REF = "refs/remotes/github/main"
MAX_JSONRPC_RESPONSE_BYTES = 32 * 1024
MAX_REQUEST_ID_BYTES = 128
MAX_TOOL_RESULT_BYTES = MAX_JSONRPC_RESPONSE_BYTES
MAX_REMOTE_SCRIPT_BYTES = 256 * 1024
MAX_REMOTE_CAPTURE_BYTES = 64 * 1024
MAX_LEGACY_REGISTRY_BYTES = 512 * 1024
MAX_RESEARCH_CONTEXT_FILE_BYTES = 128 * 1024
BUNDLE_CURSOR_OPERATION = "evidence-bundle-files"
CLOUD_TEXT_CURSOR_OPERATION = "evidence-cloud-text-read"
REMOTE_HOST = "convir-4090"
REMOTE_PYTHON = "/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python"
REMOTE_BASH = "/bin/bash"
REMOTE_TEMP_ROOT = "/sda/home/wangyuxin/ConvIR-B/runtime/convir-evidence-review"
SSH = "/usr/bin/ssh"
REMOTE_TIMEOUT_SECONDS = 90
SELF_PATH = Path(__file__).resolve()
SERVER_SOURCE_SHA256 = hashlib.sha256(SELF_PATH.read_bytes()).hexdigest()
CATALOG_SOURCE_SHA256 = hashlib.sha256(
    (SELF_PATH.parent / "convir_evidence_catalog.py").read_bytes()
).hexdigest()
TRANSPORT_SOURCE_SHA256 = hashlib.sha256(
    (SELF_PATH.parent / "convirctl.py").read_bytes()
).hexdigest()
INVENTORY_SOURCE_PATH = SELF_PATH.parent / "convir_evidence_cloud_inventory.py"
INVENTORY_SOURCE_BYTES = INVENTORY_SOURCE_PATH.read_bytes()
INVENTORY_SOURCE_SHA256 = hashlib.sha256(INVENTORY_SOURCE_BYTES).hexdigest()
LEGACY_REGISTRY_SOURCE_SHA256 = hashlib.sha256(
    (SELF_PATH.parent / "legacy_backfill_registry.py").read_bytes()
).hexdigest()
MAX_REQUEST_ID_PLACEHOLDER = "x" * (MAX_REQUEST_ID_BYTES - 2)
CATALOG_CACHE_ENTRIES = 8
_CATALOG_CACHE = OrderedDict()
_CATALOG_CACHE_LOCK = threading.RLock()


class ReviewError(RuntimeError):
    def __init__(self, message, *, state="ARGUMENTS_INVALID", exit_code=2):
        super().__init__(message)
        self.state = state
        self.exit_code = exit_code


def load_catalog_cached(repo, commit):
    """Reuse immutable commit-bound catalogs without creating review state."""
    key = (str(Path(repo).resolve()), commit)
    with _CATALOG_CACHE_LOCK:
        loaded = _CATALOG_CACHE.get(key)
        if loaded is not None:
            _CATALOG_CACHE.move_to_end(key)
            return loaded
    loaded = catalog.load_catalog(repo, commit)
    with _CATALOG_CACHE_LOCK:
        existing = _CATALOG_CACHE.get(key)
        if existing is not None:
            _CATALOG_CACHE.move_to_end(key)
            return existing
        _CATALOG_CACHE[key] = loaded
        while len(_CATALOG_CACHE) > CATALOG_CACHE_ENTRIES:
            _CATALOG_CACHE.popitem(last=False)
    return loaded


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def emit_envelope(value):
    raw = canonical_bytes(value) + b"\n"
    if len(raw) > MAX_JSONRPC_RESPONSE_BYTES:
        raw = canonical_bytes({
            "jsonrpc": "2.0",
            "id": None,
            "error": {
                "code": -32001,
                "message": "JSON-RPC response exceeded the 32 KiB transport budget",
            },
        }) + b"\n"
    sys.stdout.buffer.write(raw)
    sys.stdout.buffer.flush()


def bounded_error(exc):
    raw = str(exc).encode("utf-8", errors="replace")[:2048]
    return raw.decode("utf-8", errors="ignore")


def failure_value(operation, exc):
    if isinstance(exc, (
        ReviewError, catalog.CatalogError, convirctl.ControlError,
        inventory.InventoryError,
    )):
        state = exc.state
        exit_code = exc.exit_code
    elif isinstance(exc, OSError):
        state = "LOCAL_IO_FAILED"
        exit_code = 2
    else:
        state = "INTERNAL_REVIEW_ERROR"
        exit_code = 70
    return {
        "schema_version": 2,
        "ok": False,
        "operation": operation,
        "state": state,
        "exit_code": exit_code,
        "error": bounded_error(exc),
        "scientific_completeness": "not_assessed",
        "excluded_sources": ["route_branches", "cloud_runtime", "result_contents"],
    }


def workspace_root():
    raw = os.environ.get(WORKSPACE_ROOT_ENV, DEFAULT_WORKSPACE_ROOT)
    root = Path(raw)
    if not root.is_absolute():
        raise ReviewError(f"{WORKSPACE_ROOT_ENV} must be absolute")
    try:
        root = root.resolve(strict=True)
    except OSError as exc:
        raise ReviewError(f"workspace root is unavailable: {root}") from exc
    if not root.is_dir():
        raise ReviewError(f"workspace root is not a directory: {root}")
    return root


def validate_repo(value):
    if not isinstance(value, str) or not value:
        raise ReviewError("local_repo must be a non-empty absolute path")
    candidate = Path(value)
    if not candidate.is_absolute():
        raise ReviewError("local_repo must be absolute")
    try:
        repo = candidate.resolve(strict=True)
    except OSError as exc:
        raise ReviewError("local_repo is unavailable") from exc
    try:
        repo.relative_to(workspace_root())
    except ValueError as exc:
        raise ReviewError("local_repo is outside the trusted workspace root") from exc
    if not repo.is_dir():
        raise ReviewError("local_repo is not a directory")
    completed = convirctl.run_argv(
        [convirctl.GIT, "-C", repo, "rev-parse", "--show-toplevel"],
        timeout=30,
        env=convirctl.git_environment(30),
    )
    if completed.returncode:
        raise ReviewError("local_repo is not a Git worktree")
    top = catalog.strict_text(completed.stdout, "Git worktree root").strip()
    try:
        resolved_top = Path(top).resolve(strict=True)
    except OSError as exc:
        raise ReviewError("Git worktree root is unavailable") from exc
    if resolved_top != repo:
        raise ReviewError("local_repo must be the Git worktree root")
    return repo


def trusted_main_identity(repo):
    try:
        remote_url = catalog.git_text(repo, "remote", "get-url", TRUSTED_REMOTE_NAME)
    except catalog.CatalogError as exc:
        raise ReviewError(
            "trusted GitHub remote is unavailable",
            state="GITHUB_REMOTE_UNAVAILABLE",
            exit_code=3,
        ) from exc
    if remote_url not in TRUSTED_REMOTE_URLS:
        raise ReviewError(
            "github remote does not identify the trusted repository",
            state="GITHUB_REMOTE_UNTRUSTED",
            exit_code=3,
        )
    try:
        tip = catalog.git_text(
            repo, "rev-parse", "--verify", f"{TRUSTED_MAIN_REF}^{{commit}}"
        )
    except catalog.CatalogError as exc:
        raise ReviewError(
            "trusted GitHub main ref is unavailable",
            state="GITHUB_MAIN_UNAVAILABLE",
            exit_code=3,
        ) from exc
    if not catalog.SHA40.fullmatch(tip):
        raise ReviewError(
            "trusted GitHub main ref did not resolve to a commit",
            state="GITHUB_MAIN_INVALID",
            exit_code=3,
        )
    return remote_url, tip


def require_snapshot_commit(value):
    if not isinstance(value, str) or not catalog.SHA40.fullmatch(value):
        raise ReviewError("snapshot_commit must be 40 lowercase hexadecimal characters")
    return value


def require_main_history_commit(repo, value):
    commit = require_snapshot_commit(value)
    remote_url, tip = trusted_main_identity(repo)
    try:
        resolved = catalog.git_text(
            repo, "rev-parse", "--verify", f"{commit}^{{commit}}"
        )
    except catalog.CatalogError as exc:
        raise ReviewError(
            "snapshot_commit is unavailable",
            state="SNAPSHOT_INVALID",
            exit_code=3,
        ) from exc
    if resolved != commit:
        raise ReviewError(
            "snapshot_commit did not resolve exactly",
            state="SNAPSHOT_INVALID",
            exit_code=3,
        )
    completed = convirctl.run_argv(
        [convirctl.GIT, "-C", repo, "merge-base", "--is-ancestor", commit, tip],
        timeout=30,
        env=convirctl.git_environment(30),
    )
    if completed.returncode == 1:
        raise ReviewError(
            "snapshot_commit is outside trusted GitHub main history",
            state="SNAPSHOT_OUTSIDE_GITHUB_MAIN",
            exit_code=3,
        )
    if completed.returncode:
        raise ReviewError(
            "GitHub main ancestry check failed",
            state="GIT_READ_FAILED",
            exit_code=3,
        )
    return commit, remote_url, tip


def add_github_identity(value, remote_url, tip):
    value.update({
        "trusted_remote": TRUSTED_REMOTE_NAME,
        "trusted_remote_url": remote_url,
        "trusted_ref": TRUSTED_MAIN_REF,
        "trusted_ref_tip": tip,
        "ref_freshness": "not_assessed",
        "git_mutations_performed": False,
    })
    return value


def load_legacy_registry(repo, commit):
    """Load the commit-bound legacy registry when that snapshot contains it."""
    try:
        listing = catalog.git_bytes(
            repo, ["--literal-pathspecs", "ls-tree", "-z", "--name-only",
                   commit, "--", legacy.REGISTRY_PATH],
            limit=4096,
        )
    except catalog.CatalogError as exc:
        raise ReviewError(str(exc), state=exc.state, exit_code=exc.exit_code) from exc
    if listing == b"":
        return None
    if listing != legacy.REGISTRY_PATH.encode("utf-8") + b"\0":
        raise ReviewError(
            "legacy registry path resolution is ambiguous",
            state="LEGACY_REGISTRY_INVALID", exit_code=3,
        )
    try:
        raw = catalog.git_bytes(
            repo, ["show", f"{commit}:{legacy.REGISTRY_PATH}"],
            limit=MAX_LEGACY_REGISTRY_BYTES,
        )
    except catalog.CatalogError as exc:
        raise ReviewError(str(exc), state=exc.state, exit_code=exc.exit_code) from exc
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReviewError(
            "legacy backfill registry is not valid UTF-8 JSON",
            state="LEGACY_REGISTRY_INVALID", exit_code=3,
        ) from exc
    try:
        legacy.validate_registry(value)
    except legacy.RegistryError as exc:
        raise ReviewError(
            str(exc), state="LEGACY_REGISTRY_INVALID", exit_code=3
        ) from exc
    if value.get("generator_source_sha256") != LEGACY_REGISTRY_SOURCE_SHA256:
        raise ReviewError(
            "legacy registry generator identity differs",
            state="LEGACY_REGISTRY_IDENTITY_MISMATCH", exit_code=3,
        )
    base_commit = value.get("snapshot_commit")
    if not isinstance(base_commit, str) or not catalog.SHA40.fullmatch(base_commit):
        raise ReviewError("legacy registry snapshot is invalid", state="LEGACY_REGISTRY_INVALID", exit_code=3)
    completed = convirctl.run_argv(
        [convirctl.GIT, "-C", repo, "merge-base", "--is-ancestor", base_commit, commit],
        timeout=30, env=convirctl.git_environment(30),
    )
    if completed.returncode:
        raise ReviewError(
            "legacy registry snapshot is outside the selected history",
            state="LEGACY_REGISTRY_SNAPSHOT_MISMATCH", exit_code=3,
        )
    try:
        base_catalog = load_catalog_cached(repo, base_commit)
    except catalog.CatalogError as exc:
        raise ReviewError(str(exc), state=exc.state, exit_code=exc.exit_code) from exc
    source_tree = value.get("source_tree", {})
    source_index = value.get("terminal_index", {})
    catalog_tree = base_catalog["header"]["experiment_log_tree"]
    catalog_index = base_catalog["header"]["terminal_index"]
    if any(source_tree.get(key) != catalog_tree.get(key) for key in (
        "path", "tree_oid", "path_collection_sha256", "tracked_file_count",
    )) or any(source_index.get(key) != catalog_index.get(key) for key in (
        "path", "blob_oid", "bytes", "sha256",
    )):
        raise ReviewError(
            "legacy registry source identities differ from its frozen snapshot",
            state="LEGACY_REGISTRY_IDENTITY_MISMATCH", exit_code=3,
        )
    expected_unindexed = sorted(
        item["directory_path"] for item in base_catalog["entries"]
        if item.get("record_kind") == "evidence_directory"
        and item.get("index_coverage") == "UNINDEXED"
    )
    scope = value.get("scope", {})
    if scope.get("unindexed_directory_count") != len(expected_unindexed) \
            or scope.get("unindexed_directory_collection_sha256") \
            != catalog.canonical_sha256(expected_unindexed):
        raise ReviewError(
            "legacy registry unindexed scope differs from its frozen snapshot",
            state="LEGACY_REGISTRY_IDENTITY_MISMATCH", exit_code=3,
        )
    value["registry_source_sha256"] = LEGACY_REGISTRY_SOURCE_SHA256
    return value


def legacy_registry_summary(registry_value):
    if registry_value is None:
        return {"available": False, "state": "LEGACY_REGISTRY_NOT_PRESENT"}
    return {
        "available": True,
        "state": "LEGACY_REGISTRY_VERIFIED",
        "snapshot_commit": registry_value["snapshot_commit"],
        "registry_sha256": registry_value["registry_sha256"],
        "registry_source_sha256": registry_value["registry_source_sha256"],
        "partition": registry_value["partition"],
        "policy": registry_value["policy"],
        "excluded_content": registry_value["excluded_content"],
    }


def filter_review_candidates(loaded, registry_value):
    if registry_value is None:
        return loaded
    directory_records = {
        item["directory_path"]: item for item in registry_value["directories"]
    }
    loose_records = {item["path"]: item for item in registry_value["loose_files"]}
    entries = []
    changed = False
    for item in loaded["entries"]:
        path = item.get("directory_path") or item.get("file_path")
        legacy_record = directory_records.get(path)
        if item.get("record_kind") == "loose_file":
            if loose_records.get(path, {}).get("default_access") == "DO_NOT_READ":
                changed = True
                continue
            entries.append(item)
            continue
        if legacy_record is not None and \
                legacy_record.get("tracked_file_count") == item.get("tracked_file_count") \
                and legacy_record.get("default_access") == "DO_NOT_READ":
            changed = True
            continue
        if legacy_record is None or legacy_record.get("tracked_file_count") != item.get("tracked_file_count"):
            entries.append(item)
            continue
        annotated = dict(item)
        annotated["legacy_backfill"] = {
            key: legacy_record[key] for key in (
                "category", "default_access", "classification_reason",
            )
        }
        entries.append(annotated)
        changed = True
    if not changed:
        return loaded
    copy = dict(loaded)
    copy["entries"] = entries
    return copy


def mcp_result(value):
    value = dict(value)
    value["schema_version"] = 2
    serialized = canonical_bytes(value).decode("utf-8")
    return {
        "content": [{
            "type": "text",
            "text": serialized,
        }],
        "structuredContent": value,
        "isError": value.get("ok") is not True,
    }


def result_fits(value):
    envelope = {
        "jsonrpc": "2.0",
        "id": MAX_REQUEST_ID_PLACEHOLDER,
        "result": value,
    }
    return len(canonical_bytes(envelope)) + 1 <= MAX_JSONRPC_RESPONSE_BYTES


def bounded_mcp_result(value):
    result = mcp_result(value)
    if result_fits(result):
        return result
    fallback = failure_value(
        value.get("operation", "catalog"),
        ReviewError(
            "MCP result exceeded the 32 KiB transport budget",
            state="RESPONSE_TOO_LARGE",
            exit_code=3,
        ),
    )
    return mcp_result(fallback)


def tool_completeness_receipt(args):
    operation = "completeness-receipt"
    try:
        repo = validate_repo(args.get("local_repo"))
        requested_commit = args.get("snapshot_commit")
        if requested_commit is None:
            remote_url, commit = trusted_main_identity(repo)
            tip = commit
        else:
            commit, remote_url, tip = require_main_history_commit(
                repo, requested_commit
            )
        registry_value = load_legacy_registry(repo, commit)
        loaded = load_catalog_cached(repo, commit)
        value = catalog.completeness_receipt(loaded)
        value["legacy_backfill"] = legacy_registry_summary(registry_value)
        add_github_identity(value, remote_url, tip)
        return bounded_mcp_result(value)
    except Exception as exc:
        return bounded_mcp_result(failure_value(operation, exc))


def _bound_contract_json(repo, commit, record, source_prefix):
    matches = [
        item for item in record.get("contract_bundle", [])
        if item.get("source_path", "").startswith(source_prefix)
        and item.get("source_path", "").endswith(".json")
    ]
    if not matches:
        return None, None
    if len(matches) != 1:
        raise ReviewError(
            f"terminal launch bundle has ambiguous {source_prefix} binding",
            state="IDENTITY_CONFLICT", exit_code=3,
        )
    binding = matches[0]
    raw = catalog.git_bytes(
        repo, ["show", f"{commit}:{binding['path']}"],
        limit=MAX_RESEARCH_CONTEXT_FILE_BYTES,
    )
    if len(raw) != binding["bytes"] \
            or hashlib.sha256(raw).hexdigest() != binding["sha256"]:
        raise ReviewError(
            f"terminal launch contract identity differs: {binding['path']}",
            state="IDENTITY_CONFLICT", exit_code=3,
        )
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReviewError(
            f"terminal launch contract is invalid JSON: {binding['path']}",
            state="IDENTITY_CONFLICT", exit_code=3,
        ) from exc
    if not isinstance(value, dict):
        raise ReviewError(
            f"terminal launch contract is not an object: {binding['path']}",
            state="IDENTITY_CONFLICT", exit_code=3,
        )
    return binding, value


def _terminal_research_context(repo, commit, record, records_by_sha=None):
    if record.get("schema_version") != 2:
        return {
            "relationship_status": "not_modeled",
            "reason": "historical_terminal_schema_has_no_verified_launch_relationship",
        }
    try:
        spec_binding, spec = _bound_contract_json(
            repo, commit, record, "experience_docx/experiment_specs/"
        )
        program_binding, program = _bound_contract_json(
            repo, commit, record, "experience_docx/research_programs/"
        )
        if spec is None or program is None:
            return {
                "relationship_status": "not_modeled",
                "reason": "terminal_launch_bundle_has_no_program_and_spec_pair",
            }
        if spec.get("route_id") != record["route_id"] \
                or spec.get("program_contract_relpath") != program_binding["source_path"]:
            raise ReviewError(
                "terminal launch relationship identity differs",
                state="IDENTITY_CONFLICT", exit_code=3,
            )
        operations = spec.get("operations")
        if not isinstance(operations, dict):
            raise ReviewError(
                "terminal experiment spec operations are not an object",
                state="IDENTITY_CONFLICT", exit_code=3,
            )
        operation = operations.get(record["operation_id"])
        claim = operation.get("program_authorization") if isinstance(operation, dict) else None
        if not isinstance(claim, dict):
            return {
                "relationship_status": "not_modeled",
                "reason": "terminal_experiment_spec_has_no_program_authorization",
            }
        fields = ("program_id", "family_id", "stage_id", "mechanism_type")
        if any(not isinstance(claim.get(field), str) or not claim[field]
               for field in fields) \
                or program.get("program_id") != claim["program_id"]:
            raise ReviewError(
                "terminal program authorization relationship differs",
                state="IDENTITY_CONFLICT", exit_code=3,
            )
        research_update = {}
        if spec.get("schema_version") == 3:
            scientific = operation.get("scientific_contract")
            update = scientific.get("research_update_binding") \
                if isinstance(scientific, dict) else None
            required = {
                "snapshot_commit", "trigger_type", "trigger_terminals",
                "bottleneck_class",
                "bottleneck_statement", "literature_basis", "hypotheses",
                "design_selection",
            }
            if not isinstance(update, dict) or set(update) != required:
                raise ReviewError(
                    "terminal research update field contract differs",
                    state="IDENTITY_CONFLICT", exit_code=3,
                )
            research_snapshot = update["snapshot_commit"]
            if not isinstance(research_snapshot, str) \
                    or not catalog.SHA40.fullmatch(research_snapshot):
                raise ReviewError(
                    "terminal research snapshot identity differs",
                    state="IDENTITY_CONFLICT", exit_code=3,
                )
            catalog.git_text(
                repo, "merge-base", "--is-ancestor", research_snapshot, commit,
            )
            triggers = update["trigger_terminals"]
            trigger_type = update["trigger_type"]
            valid_triggers = (
                trigger_type == "post_terminal"
                and isinstance(triggers, list) and 1 <= len(triggers) <= 8
            ) or (
                trigger_type == "program_foundation"
                and isinstance(triggers, list) and len(triggers) == 0
            )
            if not valid_triggers:
                raise ReviewError(
                    "terminal research trigger contract differs",
                    state="IDENTITY_CONFLICT", exit_code=3,
                )
            normalized_triggers = []
            for trigger in triggers:
                if not isinstance(trigger, dict) \
                        or set(trigger) != {"route_id", "terminal_record_sha256"} \
                        or not isinstance(trigger["route_id"], str) \
                        or not trigger["route_id"] \
                        or not isinstance(trigger["terminal_record_sha256"], str) \
                        or not catalog.SHA256.fullmatch(
                            trigger["terminal_record_sha256"]
                        ):
                    raise ReviewError(
                        "terminal research trigger identity differs",
                        state="IDENTITY_CONFLICT", exit_code=3,
                    )
                prior = (records_by_sha or {}).get(
                    trigger["terminal_record_sha256"]
                )
                if records_by_sha is not None and (
                    prior is None or prior.get("route_id") != trigger["route_id"]
                ):
                    raise ReviewError(
                        "terminal research trigger is absent from the snapshot",
                        state="IDENTITY_CONFLICT", exit_code=3,
                    )
                normalized_triggers.append({
                    "route_id": trigger["route_id"],
                    "terminal_record_sha256": trigger["terminal_record_sha256"],
                })
            hypotheses = update["hypotheses"]
            literature = update["literature_basis"]
            design = update["design_selection"]
            if not isinstance(hypotheses, list) \
                    or any(not isinstance(item, dict)
                           or not isinstance(item.get("id"), str)
                           or not item["id"] for item in hypotheses) \
                    or not isinstance(literature, list) \
                    or any(not isinstance(item, dict)
                           or not isinstance(item.get("identifier"), str)
                           or not item["identifier"] for item in literature) \
                    or not isinstance(design, dict) \
                    or not isinstance(design.get("strategy"), str) \
                    or not isinstance(update["bottleneck_class"], str):
                raise ReviewError(
                    "terminal research update summary differs",
                    state="IDENTITY_CONFLICT", exit_code=3,
                )
            research_update = {
                "research_snapshot_commit": research_snapshot,
                "research_trigger_type": trigger_type,
                "trigger_terminals": normalized_triggers,
                "trigger_route_ids": sorted({
                    item["route_id"] for item in normalized_triggers
                }),
                "trigger_terminal_record_sha256s": sorted({
                    item["terminal_record_sha256"] for item in normalized_triggers
                }),
                "bottleneck_class": update["bottleneck_class"],
                "hypothesis_ids": [item["id"] for item in hypotheses],
                "literature_identifiers": [
                    item["identifier"] for item in literature
                ],
                "design_strategy": design["strategy"],
            }
        return {
            "relationship_status": "modeled",
            "context_scope": "terminal_bound_launch_contract",
            **{field: claim[field] for field in fields},
            **research_update,
            "source_terminal_record_sha256": record["record_sha256"],
            "source_bindings": {
                "experiment_spec": {
                    "source_path": spec_binding["source_path"],
                    "archive_path": spec_binding["path"],
                    "sha256": spec_binding["sha256"],
                },
                "program_contract": {
                    "source_path": program_binding["source_path"],
                    "archive_path": program_binding["path"],
                    "sha256": program_binding["sha256"],
                },
            },
        }
    except (ReviewError, catalog.CatalogError) as exc:
        return {
            "relationship_status": "identity_conflict",
            "reason": bounded_error(exc),
        }


def _catalog_with_research_context(repo, commit, loaded):
    _, records, _ = catalog.load_terminal_records(repo, commit)
    by_sha = {record["record_sha256"]: record for record in records}
    entries = []
    for entry in loaded["entries"]:
        routes = []
        for route in entry.get("routes", []):
            selected = [
                terminal for terminal in route.get("terminals", [])
                if terminal.get("operation_id") == route.get("selected_operation_id")
            ]
            record = by_sha.get(selected[0]["record_sha256"]) \
                if len(selected) == 1 else None
            context = (
                _terminal_research_context(repo, commit, record, by_sha)
                if record is not None else {
                    "relationship_status": "not_modeled",
                    "reason": "route_has_no_unique_selected_terminal",
                }
            )
            routes.append({**route, "research_context": context})
        entries.append({**entry, "routes": routes})
    enriched = {**loaded, "entries": entries}
    enriched["research_context_collection_sha256"] = catalog.canonical_sha256([
        {
            "route_id": route["route_id"],
            "research_context": route["research_context"],
        }
        for entry in entries for route in entry.get("routes", [])
    ])
    return enriched


def query_arguments(args):
    coverage = args.get("coverage", "all")
    if coverage not in {"indexed", "unindexed", "all"}:
        raise ReviewError("coverage must be indexed, unindexed, or all")
    terms = args.get("terms", [])
    if not isinstance(terms, list) or any(not isinstance(term, str) for term in terms):
        raise ReviewError("terms must be an array of strings")
    if len(terms) > 8:
        raise ReviewError("terms accepts at most 8 items")
    exact_filters = {}
    for name in (
        "route_ids", "program_ids", "family_ids", "stage_ids",
        "mechanism_types", "trigger_route_ids",
        "trigger_terminal_record_sha256s", "terminal_states", "decisions",
        "authorizes",
    ):
        values = args.get(name, [])
        if not isinstance(values, list) \
                or any(not isinstance(value, str) or not value for value in values) \
                or len(values) > catalog.MAX_EXACT_FILTER_ITEMS \
                or len(values) != len(set(values)):
            raise ReviewError(
                f"{name} must contain at most {catalog.MAX_EXACT_FILTER_ITEMS} "
                "unique non-empty strings"
            )
        if name == "trigger_terminal_record_sha256s" \
                and any(not catalog.SHA256.fullmatch(value) for value in values):
            raise ReviewError(
                "trigger_terminal_record_sha256s must contain SHA-256 identities"
            )
        exact_filters[name] = values
    cursor = args.get("cursor")
    if cursor is not None and not isinstance(cursor, str):
        raise ReviewError("cursor must be text")
    limit = args.get("limit", 20)
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 100:
        raise ReviewError("limit must be an integer in [1, 100]")
    return coverage, terms, exact_filters, cursor, limit


def tool_catalog_query(args):
    operation = "catalog-entries"
    try:
        repo = validate_repo(args.get("local_repo"))
        commit, remote_url, tip = require_main_history_commit(
            repo, args.get("snapshot_commit")
        )
        coverage, terms, exact_filters, cursor, requested_limit = query_arguments(args)
        registry_value = load_legacy_registry(repo, commit)
        loaded = filter_review_candidates(load_catalog_cached(repo, commit), registry_value)
        loaded = _catalog_with_research_context(repo, commit, loaded)
        effective_limit = requested_limit
        while True:
            value = catalog.entries_response(
                loaded,
                SimpleNamespace(
                    coverage=coverage,
                    term=terms,
                    **exact_filters,
                    cursor=cursor,
                    limit=effective_limit,
                ),
            )
            value["legacy_backfill"] = legacy_registry_summary(registry_value)
            value["research_context_collection_sha256"] = loaded[
                "research_context_collection_sha256"
            ]
            add_github_identity(value, remote_url, tip)
            result = mcp_result(value)
            if result_fits(result):
                return result
            returned = value.get("returned_count", 0)
            if returned <= 1:
                raise ReviewError(
                    "one catalog entry exceeds the MCP response budget",
                    state="ENTRY_TOO_LARGE",
                    exit_code=3,
                )
            effective_limit = min(effective_limit - 1, returned - 1)
    except Exception as exc:
        return bounded_mcp_result(failure_value(operation, exc))


def _git_blob_size(repo, commit, relpath):
    oid = catalog.git_text(repo, "rev-parse", "--verify", f"{commit}:{relpath}")
    if not catalog.SHA40.fullmatch(oid) \
            or catalog.git_text(repo, "cat-file", "-t", oid) != "blob":
        raise ReviewError(
            f"GitHub evidence is not one blob: {relpath}",
            state="IDENTITY_CONFLICT", exit_code=3,
        )
    try:
        size = int(catalog.git_text(repo, "cat-file", "-s", oid))
    except ValueError as exc:
        raise ReviewError(
            f"GitHub evidence size is invalid: {relpath}",
            state="IDENTITY_CONFLICT", exit_code=3,
        ) from exc
    if size < 0:
        raise ReviewError(
            f"GitHub evidence size is invalid: {relpath}",
            state="IDENTITY_CONFLICT", exit_code=3,
        )
    return size


def _bundle_arguments(args):
    cursor = args.get("cursor")
    if cursor is not None and not isinstance(cursor, str):
        raise ReviewError("cursor must be text")
    limit = args.get("limit", 20)
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 100:
        raise ReviewError("limit must be an integer in [1, 100]")
    return cursor, limit


def _prepare_legacy_evidence_bundle(
        repo, commit, record, terminal_record_sha256, catalog_sha256,
        remote_url, tip):
    registry_value = load_legacy_registry(repo, commit)
    if registry_value is None:
        raise ReviewError(
            "legacy backfill registry is unavailable for this snapshot",
            state="EVIDENCE_BUNDLE_LEGACY_REGISTRY_MISSING", exit_code=3,
        )
    matches = [
        item for item in registry_value["legacy_terminals"]
        if item.get("terminal_record_sha256") == terminal_record_sha256
    ]
    if len(matches) != 1 or matches[0].get("backfill_status") \
            != "LEGACY_HASH_BOUND_REVIEWABLE":
        raise ReviewError(
            "legacy terminal is not a unique hash-bound reviewable record",
            state="EVIDENCE_BUNDLE_LEGACY_NOT_REVIEWABLE", exit_code=3,
        )
    files = []
    for item in matches[0].get("files", []):
        if item.get("readable") is not True:
            raise ReviewError(
                "legacy bundle contains an unreadable file",
                state="EVIDENCE_BUNDLE_LEGACY_UNREADABLE", exit_code=3,
            )
        actual_oid = catalog.git_text(
            repo, "rev-parse", "--verify", f"{commit}:{item['path']}"
        )
        actual_size = int(catalog.git_text(repo, "cat-file", "-s", actual_oid))
        if actual_oid != item.get("blob_oid") or actual_size != item.get("bytes"):
            raise ReviewError(
                f"legacy bundle file identity differs: {item['path']}",
                state="IDENTITY_CONFLICT", exit_code=3,
            )
        raw = catalog.git_bytes(
            repo, ["cat-file", "blob", actual_oid],
            limit=legacy.MAX_FILE_BYTES + 1,
        )
        if hashlib.sha256(raw).hexdigest() != item.get("sha256"):
            raise ReviewError(
                f"legacy bundle file SHA-256 differs: {item['path']}",
                state="IDENTITY_CONFLICT", exit_code=3,
            )
        name = Path(item["path"]).name.lower()
        roles = ["legacy_text_evidence"]
        if name.endswith("_closeout.json") or name == "closeout.json":
            roles.append("typed_closeout_legacy")
        if "conclusion" in name:
            roles.append("scientific_conclusion_legacy")
        files.append({
            "path": item["path"], "bytes": item["bytes"],
            "sha256": item["sha256"], "blob_oid": item["blob_oid"],
            "roles": roles, "required": True, "terminal_bound": False,
            "evidence_status": "legacy_hash_bound", "read_priority": 40,
            "content_returned": False, "read_via": "convirctl_repo_show",
        })
    files.sort(key=lambda item: (item["read_priority"], item["path"]))
    lineage = [{
        "schema_version": 1, "operation_id": record["operation_id"],
        "run_id": record["run_id"], "state": record["state"],
        "decision": record["decision"], "authorizes": record["authorizes"],
        "route_commit": record["route_commit"], "record_sha256": record["record_sha256"],
    }]
    bundle_sha256 = catalog.canonical_sha256({
        "snapshot_commit": commit, "catalog_sha256": catalog_sha256,
        "terminal_record_sha256": terminal_record_sha256,
        "lineage": lineage, "files": files,
        "research_context": {
            "relationship_status": "not_modeled",
            "reason": "historical_terminal_schema_has_no_verified_launch_relationship",
        },
    })
    binding = {
        "snapshot_commit": commit,
        "catalog_sha256": catalog_sha256,
        "terminal_record_sha256": terminal_record_sha256,
        "evidence_role": "legacy_hash_bound",
        "raw_inventory_authorized": False,
        "raw_inventory_exclusion_reason": "legacy_schema1_has_no_raw_artifact_seal",
    }
    return {
        "repo": repo, "remote_url": remote_url, "tip": tip,
        "record": record, "resolution": "LEGACY_HASH_BOUND_REVIEWABLE",
        "lineage": lineage, "files": files, "bundle_sha256": bundle_sha256,
        "research_context": {
            "relationship_status": "not_modeled",
            "reason": "historical_terminal_schema_has_no_verified_launch_relationship",
        },
        "project_receipt": catalog.completeness_receipt(
            load_catalog_cached(repo, commit)
        ),
        "binding": binding,
        "legacy_registry": legacy_registry_summary(registry_value),
    }


def _prepare_evidence_bundle(args):
    repo = validate_repo(args.get("local_repo"))
    commit, remote_url, tip = require_main_history_commit(
        repo, args.get("snapshot_commit")
    )
    requested_catalog_sha256 = args.get("catalog_sha256")
    if not isinstance(requested_catalog_sha256, str) \
            or not inventory.SHA256.fullmatch(requested_catalog_sha256):
        raise ReviewError("catalog_sha256 has an invalid SHA identity")
    terminal_record_sha256 = args.get("terminal_record_sha256")
    if not isinstance(terminal_record_sha256, str) \
            or not inventory.SHA256.fullmatch(terminal_record_sha256):
        raise ReviewError("terminal_record_sha256 has an invalid SHA identity")
    loaded = load_catalog_cached(repo, commit)
    if loaded["catalog_sha256"] != requested_catalog_sha256:
        raise ReviewError(
            "catalog identity differs", state="IDENTITY_CONFLICT", exit_code=3
        )
    _, records, _ = catalog.load_terminal_records(repo, commit)
    matches = [
        record for record in records
        if record["record_sha256"] == terminal_record_sha256
    ]
    if len(matches) != 1:
        raise ReviewError(
            "terminal record is absent or ambiguous",
            state="EVIDENCE_BUNDLE_NOT_RESOLVED", exit_code=3,
        )
    record = matches[0]
    route_records = [
        item for item in records if item["route_id"] == record["route_id"]
    ]
    leaf, resolution = catalog.select_terminal_leaf(route_records)
    if leaf is None or leaf["record_sha256"] != terminal_record_sha256:
        raise ReviewError(
            "terminal record is not the unique selected route leaf",
            state="EVIDENCE_BUNDLE_NOT_SELECTED_LEAF", exit_code=3,
        )
    if record["schema_version"] == 1:
        return _prepare_legacy_evidence_bundle(
            repo, commit, record, terminal_record_sha256,
            requested_catalog_sha256, remote_url, tip,
        )
    if record["schema_version"] != 2:
        raise ReviewError(
            "terminal record uses an unsupported schema",
            state="EVIDENCE_BUNDLE_SCHEMA_UNSUPPORTED", exit_code=3,
        )
    binding = inventory.prepare_terminal_binding(
        repo, commit, requested_catalog_sha256, terminal_record_sha256
    )
    if binding.get("eligible") is not True:
        raise ReviewError(
            binding.get("reason", "terminal binding is not eligible"),
            state="EVIDENCE_BUNDLE_NOT_ELIGIBLE", exit_code=3,
        )

    files = {}

    def add_file(path, size, sha256, role, *, source_path=None,
                 required=True, priority=100, evidence_status="authoritative"):
        path = catalog.require_relpath(path, "bundle file path")
        item = files.get(path)
        if item is None:
            item = {
                "path": path,
                "bytes": size,
                "sha256": sha256,
                "roles": [],
                "source_paths": [],
                "required": bool(required),
                "terminal_bound": True,
                "evidence_status": evidence_status,
                "read_priority": priority,
                "content_returned": False,
                "read_via": "convirctl_repo_show",
            }
            files[path] = item
        elif item["bytes"] != size or item["sha256"] != sha256:
            raise ReviewError(
                f"bundle file identity conflicts: {path}",
                state="IDENTITY_CONFLICT", exit_code=3,
            )
        item["roles"] = sorted(set([*item["roles"], role]))
        if source_path is not None:
            source_path = catalog.require_relpath(
                source_path, "bundle source path"
            )
            item["source_paths"] = sorted(set([*item["source_paths"], source_path]))
        item["required"] = item["required"] or bool(required)
        item["read_priority"] = min(item["read_priority"], priority)

    add_file(
        record["contract_path"],
        _git_blob_size(repo, commit, record["contract_path"]),
        record["contract_sha256"], "route_contract", priority=60,
    )
    add_file(
        record["conclusion_path"],
        _git_blob_size(repo, commit, record["conclusion_path"]),
        record["conclusion_sha256"], "scientific_conclusion", priority=20,
    )
    add_file(
        record["closeout_path"],
        _git_blob_size(repo, commit, record["closeout_path"]),
        record["closeout_sha256"], "typed_closeout", priority=40,
    )
    contract_roles = {
        "manifest.json": ("route_manifest", 70),
        "route_note.md": ("route_note", 60),
        "experiment_spec.json": ("experiment_spec", 55),
        "program_contract.json": ("program_contract", 50),
        "scientific_contract.json": ("scientific_contract", 10),
        "runtime_spec.json": ("runtime_spec", 65),
        "asset_manifest.json": ("asset_manifest", 75),
        "capability_profile.json": ("capability_profile", 80),
        "precision_certificate.json": ("precision_certificate", 45),
    }
    for item in record["contract_bundle"]:
        archive_name = Path(item["path"]).name
        role, priority = contract_roles.get(
            archive_name, ("launch_contract", 90)
        )
        add_file(
            item["path"], item["bytes"], item["sha256"], role,
            source_path=item["source_path"], priority=priority,
        )
    expected_by_path = {
        item["github_path"]: item for item in binding["expected_evidence"]
    }
    unmapped_by_path = {
        item["github_path"]: item for item in binding["unmapped_results"]
    }
    review_facts_recovery = record.get("review_facts_recovery")
    for item in record["result_files"]:
        expected = expected_by_path.get(item["path"])
        unmapped = unmapped_by_path.get(item["path"])
        if review_facts_recovery is not None \
                and item["path"] == review_facts_recovery["proof_path"]:
            if expected is not None:
                raise ReviewError(
                    "review facts recovery proof cannot be runtime-declared",
                    state="IDENTITY_CONFLICT", exit_code=3,
                )
            add_file(
                item["path"], item["bytes"], item["sha256"],
                "review_facts_recovery", required=True, priority=21,
                evidence_status="identity_bound_review_facts_recovery",
            )
        elif expected is not None:
            is_review_facts = Path(item["path"]).name.endswith("_review_facts.json")
            add_file(
                item["path"], item["bytes"], item["sha256"],
                "review_facts" if is_review_facts else "formal_result",
                source_path=expected["source_relpath"],
                required=expected["required"],
                priority=22 if is_review_facts else 30,
                evidence_status=(
                    "source_pointer_verified_facts"
                    if is_review_facts else "runtime_declared"
                ),
            )
        elif item["path"] == binding.get("raw_artifact_receipt_github_path"):
            add_file(
                item["path"], item["bytes"], item["sha256"],
                "raw_artifact_receipt", required=True, priority=25,
                evidence_status="terminal_raw_artifact_seal",
            )
        elif Path(item["path"]).name.endswith("_review_facts.json"):
            add_file(
                item["path"], item["bytes"], item["sha256"],
                "review_facts", required=True, priority=22,
                evidence_status="source_pointer_verified_facts",
            )
        elif unmapped is not None:
            add_file(
                item["path"], item["bytes"], item["sha256"],
                "unmapped_formal_result", required=True, priority=35,
                evidence_status="terminal_bound_unmapped",
            )
        else:
            raise ReviewError(
                f"terminal result mapping is incomplete: {item['path']}",
                state="IDENTITY_CONFLICT", exit_code=3,
            )
    file_records = sorted(
        files.values(), key=lambda item: (item["read_priority"], item["path"])
    )
    lineage = [
        catalog.terminal_summary(item)
        for item in sorted(route_records, key=lambda value: value["index_line"])
    ]
    research_context = _terminal_research_context(
        repo, commit, record,
        {item["record_sha256"]: item for item in records},
    )
    bundle_sha256 = catalog.canonical_sha256({
        "snapshot_commit": commit,
        "catalog_sha256": requested_catalog_sha256,
        "terminal_record_sha256": terminal_record_sha256,
        "lineage": lineage,
        "files": file_records,
        "research_context": research_context,
    })
    project_receipt = catalog.completeness_receipt(loaded)
    return {
        "repo": repo,
        "remote_url": remote_url,
        "tip": tip,
        "record": record,
        "binding": binding,
        "resolution": resolution,
        "lineage": lineage,
        "research_context": research_context,
        "files": file_records,
        "bundle_sha256": bundle_sha256,
        "project_receipt": project_receipt,
    }


def _bundle_page(prepared, args, limit):
    record = prepared["record"]
    files = prepared["files"]
    cursor, _ = _bundle_arguments(args)
    query_sha256 = catalog.canonical_sha256({
        "snapshot_commit": prepared["binding"]["snapshot_commit"],
        "catalog_sha256": prepared["binding"]["catalog_sha256"],
        "terminal_record_sha256": record["record_sha256"],
        "bundle_sha256": prepared["bundle_sha256"],
    })
    offset = 0
    if cursor is not None:
        try:
            decoded = convirctl.decode_repo_cursor(
                cursor, BUNDLE_CURSOR_OPERATION, query_sha256
            )
        except convirctl.ControlError as exc:
            raise ReviewError(
                str(exc), state=exc.state, exit_code=exc.exit_code
            ) from exc
        if decoded["commit"] != prepared["binding"]["snapshot_commit"] \
                or decoded["object_id"] != record["record_sha256"][:40]:
            raise ReviewError(
                "bundle cursor identity differs",
                state="REPO_CURSOR_IDENTITY_MISMATCH", exit_code=3,
            )
        offset = decoded["position"]
    if offset > len(files) or (offset == len(files) and offset != 0):
        raise ReviewError(
            "bundle cursor is outside the file manifest",
            state="REPO_CURSOR_INVALID", exit_code=3,
        )
    selected = files[offset:offset + limit]
    end = offset + len(selected)
    terminal_page = end == len(files)
    binding = prepared["binding"]
    receipt = prepared["project_receipt"]
    value = {
        "schema_version": 2,
        "ok": True,
        "operation": "evidence-bundle",
        "state": "EVIDENCE_BUNDLE_OK",
        "exit_code": 0,
        "snapshot_commit": binding["snapshot_commit"],
        "catalog_sha256": binding["catalog_sha256"],
        "terminal_record_sha256": record["record_sha256"],
        "route_id": record["route_id"],
        "operation_id": record["operation_id"],
        "run_id": record["run_id"],
        "terminal_resolution": prepared["resolution"],
        "selected_leaf": True,
        "bundle_sha256": prepared["bundle_sha256"],
        "bundle_completeness": (
            "legacy_hash_bound"
            if prepared["resolution"] == "LEGACY_HASH_BOUND_REVIEWABLE"
            else "complete"
        ),
        "project_review_completeness": receipt["review_completeness"],
        "project_unresolved_counts": receipt["unresolved_counts"],
        "lineage": prepared["lineage"],
        "research_context": prepared["research_context"],
        "cloud_review": {
            "evidence_role": binding["evidence_role"],
            "raw_inventory_authorized": binding["raw_inventory_authorized"],
            "exclusion_reason": binding["raw_inventory_exclusion_reason"],
            "content_read": False,
        },
        "query_sha256": query_sha256,
        "offset": offset,
        "returned_count": len(selected),
        "total_count": len(files),
        "files": selected,
        "page_sha256": catalog.canonical_sha256(selected),
        "page_complete": True,
        "terminal_page": terminal_page,
        "complete": terminal_page,
        "has_more": not terminal_page,
        "next_cursor": None if terminal_page else convirctl.encode_repo_cursor(
            BUNDLE_CURSOR_OPERATION,
            binding["snapshot_commit"], query_sha256, end,
            record["record_sha256"][:40],
        ),
        "content_returned": False,
        "scientific_completeness": "not_assessed",
        "excluded_sources": [
            "route_branches", "cloud_binary_artifacts", "protected_data",
            "file_contents",
        ],
    }
    if prepared["resolution"] == "LEGACY_HASH_BOUND_REVIEWABLE":
        value.update({
            "schema2_upgrade": False,
            "legacy_limitations": [
                "post_hoc_git_hash_binding",
                "no_schema2_terminal_manifest",
                "no_raw_artifact_seal",
            ],
        })
    return value


def tool_evidence_bundle(args):
    operation = "evidence-bundle"
    try:
        prepared = _prepare_evidence_bundle(args)
        _, requested_limit = _bundle_arguments(args)
        effective_limit = requested_limit
        while True:
            value = _bundle_page(prepared, args, effective_limit)
            add_github_identity(
                value, prepared["remote_url"], prepared["tip"]
            )
            result = mcp_result(value)
            if result_fits(result):
                return result
            if value["returned_count"] <= 1:
                raise ReviewError(
                    "one bundle file record exceeds the MCP response budget",
                    state="ENTRY_TOO_LARGE", exit_code=3,
                )
            effective_limit = value["returned_count"] - 1
    except Exception as exc:
        return bounded_mcp_result(failure_value(operation, exc))


def _remote_script(request):
    source = base64.encodebytes(INVENTORY_SOURCE_BYTES).decode("ascii").rstrip()
    request_raw = canonical_bytes(request)
    encoded_request = base64.encodebytes(request_raw).decode("ascii").rstrip()
    script = f"""#!/usr/bin/env bash
set -euo pipefail
umask 077
temp_root={REMOTE_TEMP_ROOT!r}
/usr/bin/mkdir -p -- "$temp_root"
resolved_root=$(/usr/bin/readlink -f -- "$temp_root")
test "$resolved_root" = "$temp_root"
task_dir=$(/usr/bin/mktemp -d -- "$temp_root/request.XXXXXXXX")
resolved_task=$(/usr/bin/readlink -f -- "$task_dir")
case "$resolved_task" in
  "$temp_root"/request.*) ;;
  *) exit 71 ;;
esac
task_dir="$resolved_task"
cleanup() {{
  case "$task_dir" in
    "$temp_root"/request.*) /usr/bin/rm -rf -- "$task_dir" ;;
    *) exit 72 ;;
  esac
}}
trap cleanup EXIT
/usr/bin/base64 --decode > "$task_dir/worker.py" <<'__CONVIR_WORKER_SOURCE__'
{source}
__CONVIR_WORKER_SOURCE__
/usr/bin/base64 --decode > "$task_dir/request.json" <<'__CONVIR_WORKER_REQUEST__'
{encoded_request}
__CONVIR_WORKER_REQUEST__
{REMOTE_PYTHON} "$task_dir/worker.py" --remote-worker < "$task_dir/request.json"
"""
    encoded = script.encode("utf-8")
    if len(encoded) > MAX_REMOTE_SCRIPT_BYTES:
        raise ReviewError(
            "fixed remote inventory script exceeded its bound",
            state="RESPONSE_TOO_LARGE", exit_code=3,
        )
    return encoded


def _run_fixed_remote(request):
    script = _remote_script(request)
    argv = [
        SSH, "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=30",
        REMOTE_HOST, REMOTE_BASH, "-s", "--",
    ]
    try:
        process = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise ReviewError(
            "fixed cloud transport could not start",
            state="CLOUD_UNAVAILABLE", exit_code=3,
        ) from exc

    stdout = bytearray()
    stderr = bytearray()
    thread_errors = []
    store_limit = MAX_REMOTE_CAPTURE_BYTES + 1

    def drain(stream, target):
        try:
            while True:
                block = stream.read(8192)
                if not block:
                    break
                remaining = store_limit - len(target)
                if remaining > 0:
                    target.extend(block[:remaining])
        except OSError as exc:
            thread_errors.append(exc)
        finally:
            stream.close()

    def feed():
        try:
            process.stdin.write(script)
            process.stdin.flush()
        except BrokenPipeError:
            pass
        except OSError as exc:
            thread_errors.append(exc)
        finally:
            process.stdin.close()

    threads = [
        threading.Thread(target=drain, args=(process.stdout, stdout), daemon=True),
        threading.Thread(target=drain, args=(process.stderr, stderr), daemon=True),
        threading.Thread(target=feed, daemon=True),
    ]
    for thread in threads:
        thread.start()
    try:
        return_code = process.wait(timeout=REMOTE_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as exc:
        process.kill()
        process.wait()
        for thread in threads:
            thread.join(timeout=5)
        raise ReviewError(
            "fixed cloud inventory timed out; remote state is unknown",
            state="REMOTE_STATE_UNKNOWN", exit_code=3,
        ) from exc
    for thread in threads:
        thread.join(timeout=5)
    if any(thread.is_alive() for thread in threads) or thread_errors:
        raise ReviewError(
            "fixed cloud transport streams did not close cleanly",
            state="REMOTE_STATE_UNKNOWN", exit_code=3,
        )
    if len(stdout) > MAX_REMOTE_CAPTURE_BYTES or len(stderr) > MAX_REMOTE_CAPTURE_BYTES:
        raise ReviewError(
            "fixed cloud transport output exceeded its bound",
            state="RESPONSE_TOO_LARGE", exit_code=3,
        )
    if return_code:
        raise ReviewError(
            "fixed cloud inventory command failed",
            state="CLOUD_UNAVAILABLE", exit_code=3,
        )
    try:
        response = json.loads(bytes(stdout).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReviewError(
            "fixed cloud inventory returned an invalid response",
            state="REMOTE_STATE_UNKNOWN", exit_code=3,
        ) from exc
    if not isinstance(response, dict) or set(response) != {
        "worker_source_sha256", "result"
    } or response["worker_source_sha256"] != INVENTORY_SOURCE_SHA256 \
            or not isinstance(response["result"], dict):
        raise ReviewError(
            "fixed cloud worker identity or response differs",
            state="IDENTITY_CONFLICT", exit_code=3,
        )
    return response["result"]


def _binding_identity(binding):
    identity = {
        key: binding.get(key) for key in (
            "snapshot_commit", "catalog_sha256", "terminal_index_sha256",
            "terminal_record_sha256", "route_id", "operation_id", "run_id",
            "output_id", "mode", "session", "route_commit",
            "workload_source_commit", "manifest_sha256",
            "runtime_spec_sha256", "closeout_sha256", "runner_sha256",
            "raw_artifact_receipt_sha256",
        )
    }
    identity["raw_artifact_manifest_sha256"] = (
        binding.get("raw_artifact_receipt") or {}
    ).get("manifest_sha256")
    return identity


def _prepare_cloud_binding(args):
    repo = validate_repo(args.get("local_repo"))
    commit, remote_url, tip = require_main_history_commit(
        repo, args.get("snapshot_commit")
    )
    binding = inventory.prepare_terminal_binding(
        repo,
        commit,
        args.get("catalog_sha256"),
        args.get("terminal_record_sha256"),
    )
    return binding, remote_url, tip


def _execute_inventory(binding, query=None):
    root_binding = binding.get("eligible") is True
    if binding.get("eligible") is not True \
            or binding.get("raw_inventory_authorized") is not True:
        adapter_root = binding.get("run_root", "/")
        full = inventory._scan_adapter_root(
            binding,
            adapter_root,
            root_binding_enforced=root_binding,
        )
        if query is None or full.get("ok") is not True:
            return inventory.inventory_summary(full)
        return inventory.inventory_query_page(full, **query)
    request = {
        "schema_version": 1,
        "operation": "summary" if query is None else "query",
        "binding": binding,
        "adapter_root": binding["run_root"],
        "root_binding_enforced": True,
        "expected_session": binding["session"],
        "query": query,
    }
    result = _run_fixed_remote(request)
    if result.get("ok") is not True:
        return result
    if query is None:
        if result.get("identity") != _binding_identity(binding) \
                or result.get("declared_run_root") != binding["run_root"] \
                or result.get("root_binding_enforced") is not True:
            raise ReviewError(
                "cloud inventory summary identity differs",
                state="IDENTITY_CONFLICT", exit_code=3,
            )
    else:
        if result.get("snapshot_commit") != binding["snapshot_commit"] \
                or result.get("terminal_record_sha256") \
                != binding["terminal_record_sha256"] \
                or result.get("inventory_sha256") != query["inventory_sha256"]:
            raise ReviewError(
                "cloud inventory query identity differs",
                state="IDENTITY_CONFLICT", exit_code=3,
            )
    return result


def tool_cloud_inventory_summary(args):
    operation = "cloud-inventory-summary"
    try:
        binding, remote_url, tip = _prepare_cloud_binding(args)
        value = _execute_inventory(binding)
        add_github_identity(value, remote_url, tip)
        return bounded_mcp_result(value)
    except Exception as exc:
        return bounded_mcp_result(failure_value(operation, exc))


def _cloud_query_arguments(args, binding):
    inventory_sha256 = args.get("inventory_sha256")
    if not isinstance(inventory_sha256, str) \
            or not inventory.SHA256.fullmatch(inventory_sha256):
        raise ReviewError("inventory_sha256 has an invalid SHA identity")
    states, terms, limit = inventory.normalize_query_arguments(
        args.get("reconciliation_states"), args.get("terms"), args.get("limit", 20)
    )
    cursor = args.get("cursor")
    if cursor is not None and not isinstance(cursor, str):
        raise ReviewError("cursor must be text")
    identity = _binding_identity(binding)
    query_sha256 = inventory.inventory_query_sha256(
        identity, inventory_sha256, states, terms
    )
    offset = 0
    if cursor is not None:
        try:
            decoded = convirctl.decode_repo_cursor(
                cursor, inventory.CURSOR_OPERATION, query_sha256
            )
        except convirctl.ControlError as exc:
            raise ReviewError(
                str(exc), state=exc.state, exit_code=exc.exit_code
            ) from exc
        if decoded["commit"] != binding["snapshot_commit"] \
                or decoded["object_id"] != inventory_sha256[:40]:
            raise ReviewError(
                "inventory cursor identity differs",
                state="REPO_CURSOR_IDENTITY_MISMATCH", exit_code=3,
            )
        offset = decoded["position"]
    return {
        "inventory_sha256": inventory_sha256,
        "reconciliation_states": states,
        "terms": terms,
        "offset": offset,
        "limit": limit,
    }, query_sha256


def _bounded_cloud_query(value, binding, query_sha256, remote_url, tip):
    page = dict(value)
    entries = list(page.get("entries", []))
    if not isinstance(page.get("offset"), int) \
            or not isinstance(page.get("total_count"), int) \
            or page.get("query_sha256") != query_sha256:
        raise ReviewError(
            "cloud inventory page contract differs",
            state="IDENTITY_CONFLICT", exit_code=3,
        )
    while True:
        end = page["offset"] + len(entries)
        complete = end == page["total_count"]
        page.update({
            "entries": entries,
            "returned_count": len(entries),
            "page_sha256": inventory.canonical_sha256(entries),
            "complete": complete,
            "has_more": not complete,
            "next_cursor": None if complete else convirctl.encode_repo_cursor(
                inventory.CURSOR_OPERATION,
                binding["snapshot_commit"],
                query_sha256,
                end,
                page["inventory_sha256"][:40],
            ),
        })
        page.pop("next_offset", None)
        add_github_identity(page, remote_url, tip)
        result = mcp_result(page)
        if result_fits(result):
            return result
        if len(entries) <= 1:
            raise ReviewError(
                "one inventory entry exceeds the MCP response budget",
                state="ENTRY_TOO_LARGE", exit_code=3,
            )
        entries.pop()


def tool_cloud_inventory_query(args):
    operation = "cloud-inventory-query"
    try:
        binding, remote_url, tip = _prepare_cloud_binding(args)
        query, query_sha256 = _cloud_query_arguments(args, binding)
        value = _execute_inventory(binding, query)
        if value.get("ok") is not True:
            add_github_identity(value, remote_url, tip)
            return bounded_mcp_result(value)
        return _bounded_cloud_query(
            value, binding, query_sha256, remote_url, tip
        )
    except Exception as exc:
        return bounded_mcp_result(failure_value(operation, exc))


def _cloud_text_query_sha256(binding, inventory_sha256, relative_path,
                             file_sha256):
    return catalog.canonical_sha256({
        "snapshot_commit": binding["snapshot_commit"],
        "terminal_record_sha256": binding["terminal_record_sha256"],
        "inventory_sha256": inventory_sha256,
        "relative_path": relative_path,
        "file_sha256": file_sha256,
    })


def _cloud_text_arguments(args, binding):
    inventory_sha256 = args.get("inventory_sha256")
    if not isinstance(inventory_sha256, str) \
            or not inventory.SHA256.fullmatch(inventory_sha256):
        raise ReviewError("inventory_sha256 has an invalid SHA identity")
    try:
        relative_path = inventory._require_relpath(
            args.get("relative_path"), "relative_path", state="ARGUMENTS_INVALID"
        )
    except inventory.InventoryError as exc:
        raise ReviewError(str(exc), state=exc.state, exit_code=exc.exit_code) from exc
    page_bytes = args.get("page_bytes", inventory.MAX_TEXT_PAGE_BYTES)
    if not isinstance(page_bytes, int) or isinstance(page_bytes, bool) \
            or not 256 <= page_bytes <= inventory.MAX_TEXT_PAGE_BYTES:
        raise ReviewError(
            f"page_bytes must be in [256, {inventory.MAX_TEXT_PAGE_BYTES}]"
        )
    file_sha256 = args.get("file_sha256")
    if file_sha256 is not None and (
        not isinstance(file_sha256, str)
        or not inventory.SHA256.fullmatch(file_sha256)
    ):
        raise ReviewError("file_sha256 has an invalid SHA identity")
    cursor = args.get("cursor")
    if cursor is not None and not isinstance(cursor, str):
        raise ReviewError("cursor must be text")
    if cursor is not None and file_sha256 is None:
        raise ReviewError("file_sha256 is required with a continuation cursor")
    offset = 0
    if cursor is not None:
        query_sha256 = _cloud_text_query_sha256(
            binding, inventory_sha256, relative_path, file_sha256
        )
        try:
            decoded = convirctl.decode_repo_cursor(
                cursor, CLOUD_TEXT_CURSOR_OPERATION, query_sha256
            )
        except convirctl.ControlError as exc:
            raise ReviewError(
                str(exc), state=exc.state, exit_code=exc.exit_code
            ) from exc
        if decoded["commit"] != binding["snapshot_commit"] \
                or decoded["object_id"] != file_sha256[:40]:
            raise ReviewError(
                "cloud text cursor identity differs",
                state="REPO_CURSOR_IDENTITY_MISMATCH", exit_code=3,
            )
        offset = decoded["position"]
    return {
        "inventory_sha256": inventory_sha256,
        "relative_path": relative_path,
        "offset": offset,
        "page_bytes": page_bytes,
        "expected_file_sha256": file_sha256,
    }


def _execute_cloud_text(binding, query):
    if binding.get("eligible") is not True:
        raise ReviewError(
            binding.get("reason", "terminal binding is not eligible"),
            state="CLOUD_TEXT_NOT_INVENTORIED", exit_code=3,
        )
    if binding.get("raw_inventory_authorized") is not True:
        raise ReviewError(
            binding.get(
                "raw_inventory_exclusion_reason",
                "cloud text scope is protected or not authorized",
            ),
            state="CLOUD_TEXT_PROTECTED_SCOPE", exit_code=3,
        )
    request = {
        "schema_version": 2,
        "operation": "text_read",
        "binding": binding,
        "adapter_root": binding["run_root"],
        "root_binding_enforced": True,
        "expected_session": binding["session"],
        "query": query,
    }
    result = _run_fixed_remote(request)
    if result.get("ok") is not True:
        return result
    if result.get("schema_version") != 2 \
            or result.get("snapshot_commit") != binding["snapshot_commit"] \
            or result.get("catalog_sha256") != binding["catalog_sha256"] \
            or result.get("terminal_record_sha256") \
            != binding["terminal_record_sha256"] \
            or result.get("inventory_sha256") != query["inventory_sha256"] \
            or result.get("relative_path") != query["relative_path"]:
        raise ReviewError(
            "cloud text response identity differs",
            state="IDENTITY_CONFLICT", exit_code=3,
        )
    return result


def _bounded_cloud_text(value, binding, remote_url, tip):
    page = dict(value)
    content = page.get("content")
    file_sha256 = page.get("file_sha256")
    if not isinstance(content, str) \
            or not isinstance(file_sha256, str) \
            or not inventory.SHA256.fullmatch(file_sha256):
        raise ReviewError(
            "cloud text page contract differs",
            state="IDENTITY_CONFLICT", exit_code=3,
        )
    start = page.get("page_start_byte")
    total_bytes = page.get("bytes")
    if not isinstance(start, int) or not isinstance(total_bytes, int):
        raise ReviewError(
            "cloud text page offsets differ",
            state="IDENTITY_CONFLICT", exit_code=3,
        )
    query_sha256 = _cloud_text_query_sha256(
        binding, page["inventory_sha256"], page["relative_path"], file_sha256
    )
    while True:
        raw = content.encode("utf-8")
        end = start + len(raw)
        terminal_page = end == total_bytes
        page.update({
            "content": content,
            "query_sha256": query_sha256,
            "page_end_byte": end,
            "page_bytes": len(raw),
            "page_sha256": hashlib.sha256(raw).hexdigest(),
            "terminal_page": terminal_page,
            "complete": terminal_page,
            "has_more": not terminal_page,
            "next_cursor": None if terminal_page else convirctl.encode_repo_cursor(
                CLOUD_TEXT_CURSOR_OPERATION,
                binding["snapshot_commit"], query_sha256, end,
                file_sha256[:40],
            ),
            "file_read_completeness": (
                "complete" if terminal_page else "partial"
            ),
            "unread_remaining_bytes": total_bytes - end,
        })
        add_github_identity(page, remote_url, tip)
        result = mcp_result(page)
        if result_fits(result):
            return result
        if len(content) <= 1:
            raise ReviewError(
                "one cloud text character exceeds the MCP response budget",
                state="ENTRY_TOO_LARGE", exit_code=3,
            )
        content = content[:max(1, len(content) // 2)]


def tool_cloud_text_read(args):
    operation = "cloud-text-read"
    try:
        binding, remote_url, tip = _prepare_cloud_binding(args)
        query = _cloud_text_arguments(args, binding)
        value = _execute_cloud_text(binding, query)
        if value.get("ok") is not True:
            add_github_identity(value, remote_url, tip)
            return bounded_mcp_result(value)
        return _bounded_cloud_text(value, binding, remote_url, tip)
    except Exception as exc:
        return bounded_mcp_result(failure_value(operation, exc))


OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["ok", "operation", "state", "exit_code"],
    "properties": {
        "ok": {"type": "boolean"},
        "operation": {"type": "string"},
        "state": {"type": "string"},
        "exit_code": {"type": "integer"},
    },
    "additionalProperties": True,
}


TOOLS = {
    "convir_evidence_completeness_receipt": {
        "description": (
            "Return one compact schema-2 receipt that partitions the complete "
            "commit-bound GitHub catalog and reports every unresolved discovery "
            "class without reading result contents."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["local_repo"],
            "properties": {
                "local_repo": {"type": "string"},
                "snapshot_commit": {
                    "type": "string", "pattern": "^[0-9a-f]{40}$",
                },
            },
            "additionalProperties": False,
        },
        "outputSchema": OUTPUT_SCHEMA,
        "handler": tool_completeness_receipt,
    },
    "convir_evidence_catalog_query": {
        "description": (
            "Query one immutable GitHub evidence catalog through a bounded, "
            "identity-bound cursor without reading result contents."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["local_repo", "snapshot_commit"],
            "properties": {
                "local_repo": {"type": "string"},
                "snapshot_commit": {"type": "string", "pattern": "^[0-9a-f]{40}$"},
                "coverage": {
                    "enum": ["indexed", "unindexed", "all"],
                    "default": "all",
                },
                "terms": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 8,
                },
                "route_ids": {
                    "type": "array", "items": {"type": "string", "minLength": 1},
                    "maxItems": catalog.MAX_EXACT_FILTER_ITEMS,
                },
                "program_ids": {
                    "type": "array", "items": {"type": "string", "minLength": 1},
                    "maxItems": catalog.MAX_EXACT_FILTER_ITEMS,
                },
                "family_ids": {
                    "type": "array", "items": {"type": "string", "minLength": 1},
                    "maxItems": catalog.MAX_EXACT_FILTER_ITEMS,
                },
                "stage_ids": {
                    "type": "array", "items": {"type": "string", "minLength": 1},
                    "maxItems": catalog.MAX_EXACT_FILTER_ITEMS,
                },
                "mechanism_types": {
                    "type": "array", "items": {"type": "string", "minLength": 1},
                    "maxItems": catalog.MAX_EXACT_FILTER_ITEMS,
                },
                "trigger_route_ids": {
                    "type": "array", "items": {"type": "string", "minLength": 1},
                    "maxItems": catalog.MAX_EXACT_FILTER_ITEMS,
                },
                "trigger_terminal_record_sha256s": {
                    "type": "array",
                    "items": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                    "maxItems": catalog.MAX_EXACT_FILTER_ITEMS,
                },
                "terminal_states": {
                    "type": "array", "items": {"type": "string", "minLength": 1},
                    "maxItems": catalog.MAX_EXACT_FILTER_ITEMS,
                },
                "decisions": {
                    "type": "array", "items": {"type": "string", "minLength": 1},
                    "maxItems": catalog.MAX_EXACT_FILTER_ITEMS,
                },
                "authorizes": {
                    "type": "array", "items": {"type": "string", "minLength": 1},
                    "maxItems": catalog.MAX_EXACT_FILTER_ITEMS,
                },
                "cursor": {"type": "string"},
                "limit": {
                    "type": "integer", "minimum": 1, "maximum": 100,
                    "default": 20,
                },
            },
            "additionalProperties": False,
        },
        "outputSchema": OUTPUT_SCHEMA,
        "handler": tool_catalog_query,
    },
    "convir_evidence_bundle": {
        "description": (
            "Resolve one exact schema-2 terminal leaf or one unique registry-"
            "backed schema-1 terminal and page its SHA-bound GitHub evidence "
            "manifest "
            "without returning file contents."
        ),
        "inputSchema": {
            "type": "object",
            "required": [
                "local_repo", "snapshot_commit", "catalog_sha256",
                "terminal_record_sha256",
            ],
            "properties": {
                "local_repo": {"type": "string"},
                "snapshot_commit": {
                    "type": "string", "pattern": "^[0-9a-f]{40}$",
                },
                "catalog_sha256": {
                    "type": "string", "pattern": "^[0-9a-f]{64}$",
                },
                "terminal_record_sha256": {
                    "type": "string", "pattern": "^[0-9a-f]{64}$",
                },
                "cursor": {"type": "string"},
                "limit": {
                    "type": "integer", "minimum": 1, "maximum": 100,
                    "default": 20,
                },
            },
            "additionalProperties": False,
        },
        "outputSchema": OUTPUT_SCHEMA,
        "handler": tool_evidence_bundle,
    },
    "convir_evidence_cloud_inventory_summary": {
        "description": (
            "Verify one immutable schema-2 terminal binding, then return a "
            "bounded reconciliation summary for its fixed cloud run root."
        ),
        "inputSchema": {
            "type": "object",
            "required": [
                "local_repo", "snapshot_commit", "catalog_sha256",
                "terminal_record_sha256",
            ],
            "properties": {
                "local_repo": {"type": "string"},
                "snapshot_commit": {"type": "string", "pattern": "^[0-9a-f]{40}$"},
                "catalog_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                "terminal_record_sha256": {
                    "type": "string", "pattern": "^[0-9a-f]{64}$",
                },
            },
            "additionalProperties": False,
        },
        "outputSchema": OUTPUT_SCHEMA,
        "handler": tool_cloud_inventory_summary,
    },
    "convir_evidence_cloud_inventory_query": {
        "description": (
            "Rescan one immutable schema-2 terminal run root and return a bounded "
            "identity-bound page only when its inventory SHA-256 is unchanged."
        ),
        "inputSchema": {
            "type": "object",
            "required": [
                "local_repo", "snapshot_commit", "catalog_sha256",
                "terminal_record_sha256", "inventory_sha256",
            ],
            "properties": {
                "local_repo": {"type": "string"},
                "snapshot_commit": {"type": "string", "pattern": "^[0-9a-f]{40}$"},
                "catalog_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                "terminal_record_sha256": {
                    "type": "string", "pattern": "^[0-9a-f]{64}$",
                },
                "inventory_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                "reconciliation_states": {
                    "type": "array",
                    "items": {"enum": list(inventory.RECONCILIATION_STATES)},
                    "maxItems": len(inventory.RECONCILIATION_STATES),
                },
                "terms": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1, "maxLength": 128},
                    "maxItems": 8,
                },
                "cursor": {"type": "string"},
                "limit": {
                    "type": "integer", "minimum": 1, "maximum": 100,
                    "default": 20,
                },
            },
            "additionalProperties": False,
        },
        "outputSchema": OUTPUT_SCHEMA,
        "handler": tool_cloud_inventory_query,
    },
    "convir_evidence_cloud_text_read": {
        "description": (
            "Read one bounded UTF-8 page from an exact text file already found "
            "in a complete inactive unprotected cloud inventory; continuation "
            "binds the file SHA-256 and never accepts a host or run root."
        ),
        "inputSchema": {
            "type": "object",
            "required": [
                "local_repo", "snapshot_commit", "catalog_sha256",
                "terminal_record_sha256", "inventory_sha256", "relative_path",
            ],
            "properties": {
                "local_repo": {"type": "string"},
                "snapshot_commit": {
                    "type": "string", "pattern": "^[0-9a-f]{40}$",
                },
                "catalog_sha256": {
                    "type": "string", "pattern": "^[0-9a-f]{64}$",
                },
                "terminal_record_sha256": {
                    "type": "string", "pattern": "^[0-9a-f]{64}$",
                },
                "inventory_sha256": {
                    "type": "string", "pattern": "^[0-9a-f]{64}$",
                },
                "relative_path": {"type": "string", "minLength": 1},
                "file_sha256": {
                    "type": "string", "pattern": "^[0-9a-f]{64}$",
                },
                "cursor": {"type": "string"},
                "page_bytes": {
                    "type": "integer", "minimum": 256,
                    "maximum": inventory.MAX_TEXT_PAGE_BYTES,
                    "default": inventory.MAX_TEXT_PAGE_BYTES,
                },
            },
            "additionalProperties": False,
        },
        "outputSchema": OUTPUT_SCHEMA,
        "handler": tool_cloud_text_read,
    },
}


def handle(request):
    method = request.get("method")
    params = request.get("params") or {}
    if method == "initialize":
        return {
            "protocolVersion": params.get("protocolVersion", "2024-11-05"),
            "capabilities": {"tools": {}},
            "serverInfo": {
                "name": SERVER_NAME,
                "version": SERVER_VERSION,
                "sourceSha256": SERVER_SOURCE_SHA256,
                "catalogSourceSha256": CATALOG_SOURCE_SHA256,
                "inventorySourceSha256": INVENTORY_SOURCE_SHA256,
                "transportSourceSha256": TRANSPORT_SOURCE_SHA256,
            },
        }
    if method == "ping":
        return {}
    if method == "tools/list":
        return {"tools": [
            {
                "name": name,
                "description": item["description"],
                "inputSchema": item["inputSchema"],
                "outputSchema": item["outputSchema"],
            }
            for name, item in TOOLS.items()
        ]}
    if method == "tools/call":
        name = params.get("name")
        args = params.get("arguments")
        if name not in TOOLS or not isinstance(args, dict):
            raise ReviewError("invalid MCP tool call")
        schema = TOOLS[name]["inputSchema"]
        unknown = set(args) - set(schema["properties"])
        missing = set(schema["required"]) - set(args)
        if unknown or missing:
            raise ReviewError(
                f"tool argument mismatch unknown={sorted(unknown)} "
                f"missing={sorted(missing)}"
            )
        return TOOLS[name]["handler"](args)
    raise ReviewError(f"unsupported method: {method}")


def require_request_id(value):
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ReviewError("MCP request id must be a string or integer")
    if len(canonical_bytes(value)) > MAX_REQUEST_ID_BYTES:
        raise ReviewError("JSON-encoded MCP request id exceeds 128 bytes")
    return value


def main():
    for line in sys.stdin:
        request_id = None
        response_expected = False
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ReviewError("MCP request must be an object")
            response_expected = "id" in request and request["id"] is not None
            if response_expected:
                request_id = require_request_id(request["id"])
            result = handle(request)
            if response_expected:
                emit_envelope({"jsonrpc": "2.0", "id": request_id, "result": result})
        except Exception as exc:
            if response_expected:
                emit_envelope({
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32000, "message": bounded_error(exc)},
                })


if __name__ == "__main__":
    main()
