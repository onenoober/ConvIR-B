#!/usr/bin/env python3
"""Build a compact, commit-bound catalog of GitHub experiment evidence."""

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

import convirctl


INDEX_PATH = "experience_docx/EXPERIMENT_TERMINAL_INDEX.jsonl"
LOG_ROOT = "experience_docx/experiment_logs"
MAX_INDEX_BYTES = 1024 * 1024
MAX_TREE_BYTES = 16 * 1024 * 1024
MAX_RESPONSE_BYTES = 32 * 1024
MAX_PAGE_ENTRIES = 100
MAX_TERMS = 8
CURSOR_OPERATION = "evidence-catalog-entries"
SHA40 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")

COMMON_FIELDS = {
    "schema_version", "route_id", "operation_id", "run_id", "state",
    "decision", "authorizes", "receipt", "route_commit", "contract_path",
    "closeout_path", "conclusion_path", "result_paths",
}
SCHEMA2_FIELDS = {
    "contract_bundle", "prior_terminal_record", "result_files",
    "contract_sha256", "closeout_sha256", "conclusion_sha256",
}
SCHEMA2_OPTIONAL_FIELDS = {"review_facts_recovery"}
RESULT_FILE_RECORD_FIELDS = {"path", "bytes", "sha256"}
CONTRACT_FILE_RECORD_FIELDS = RESULT_FILE_RECORD_FIELDS | {"source_path"}
PRIOR_RECORD_FIELDS = {"prior_closeout_path", "prior_terminal_tuple"}
TERMINAL_TUPLE_FIELDS = {"state", "decision", "authorizes"}
REVIEW_FACTS_RECOVERY_FIELDS = {
    "status", "recovery_type", "proof_path", "proof_bytes", "proof_sha256",
    "original_path", "original_sha256", "recovered_review_facts_sha256",
}
REVIEW_FACTS_RECOVERY_TYPE = "legacy_unbound_gate_confidence_metadata_v1"


class CatalogError(RuntimeError):
    def __init__(self, message, *, state="CATALOG_INVALID", exit_code=3):
        super().__init__(message)
        self.state = state
        self.exit_code = exit_code


class JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        raise CatalogError(message, state="ARGUMENTS_INVALID", exit_code=2)


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def canonical_sha256(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def strict_text(raw, name):
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CatalogError(f"{name} is not UTF-8", state="SOURCE_NOT_UTF8") from exc


def require_text(value, name):
    if not isinstance(value, str) or not value or any(char in value for char in "\x00\n\r"):
        raise CatalogError(f"{name} must be non-empty single-line text")
    return value


def require_relpath(value, name):
    value = require_text(value, name).replace("\\", "/")
    candidate = PurePosixPath(value)
    if candidate.is_absolute() or any(part in {"", ".", ".."} for part in value.split("/")):
        raise CatalogError(f"{name} must be a safe relative path")
    return candidate.as_posix()


def require_sha(value, pattern, name):
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise CatalogError(f"{name} has an invalid SHA identity")
    return value


def require_decision(value, name):
    if value is not None:
        require_text(value, name)
    return value


def validate_file_records(records, name, unmodeled, expected_fields):
    if not isinstance(records, list) or not records:
        raise CatalogError(f"{name} must be a non-empty list")
    paths = []
    for index, record in enumerate(records):
        if not isinstance(record, dict) or not expected_fields.issubset(record):
            raise CatalogError(f"{name}[{index}] has an invalid field contract")
        extra = sorted(set(record) - expected_fields)
        unmodeled.extend(f"{name}[].{field}" for field in extra)
        path = require_relpath(record["path"], f"{name}[{index}].path")
        if "source_path" in expected_fields:
            require_relpath(record["source_path"], f"{name}[{index}].source_path")
        if not isinstance(record["bytes"], int) or isinstance(record["bytes"], bool) \
                or record["bytes"] < 0:
            raise CatalogError(f"{name}[{index}].bytes is invalid")
        require_sha(record["sha256"], SHA256, f"{name}[{index}].sha256")
        paths.append(path)
    if len(paths) != len(set(paths)):
        raise CatalogError(f"{name} contains duplicate paths")
    return paths


def _private_file_record(record, *, include_source_path=False):
    value = {
        "path": require_relpath(record["path"], "private file record path"),
        "bytes": record["bytes"],
        "sha256": record["sha256"],
    }
    if include_source_path:
        value["source_path"] = require_relpath(
            record["source_path"], "private file record source_path"
        )
    return value


def validate_prior_record(value, unmodeled):
    if not isinstance(value, dict) or not PRIOR_RECORD_FIELDS.issubset(value):
        raise CatalogError("prior_terminal_record has an invalid field contract")
    unmodeled.extend(
        f"prior_terminal_record.{field}"
        for field in sorted(set(value) - PRIOR_RECORD_FIELDS)
    )
    parent_path = value["prior_closeout_path"]
    parent_tuple = value["prior_terminal_tuple"]
    if parent_path is None:
        if parent_tuple is not None:
            raise CatalogError("a root prior_terminal_record must have a null tuple")
        return None, None
    parent_path = require_relpath(parent_path, "prior_closeout_path")
    if not isinstance(parent_tuple, dict) or not TERMINAL_TUPLE_FIELDS.issubset(parent_tuple):
        raise CatalogError("prior_terminal_tuple has an invalid field contract")
    unmodeled.extend(
        f"prior_terminal_tuple.{field}"
        for field in sorted(set(parent_tuple) - TERMINAL_TUPLE_FIELDS)
    )
    require_text(parent_tuple["state"], "prior_terminal_tuple.state")
    require_decision(parent_tuple["decision"], "prior_terminal_tuple.decision")
    require_text(parent_tuple["authorizes"], "prior_terminal_tuple.authorizes")
    return parent_path, {
        key: parent_tuple[key] for key in ("state", "decision", "authorizes")
    }


def validate_review_facts_recovery(value, result_files):
    if not isinstance(value, dict) or set(value) != REVIEW_FACTS_RECOVERY_FIELDS:
        raise CatalogError("review_facts_recovery has an invalid field contract")
    if value["status"] != "REVIEW_FACTS_RECOVERED" \
            or value["recovery_type"] != REVIEW_FACTS_RECOVERY_TYPE:
        raise CatalogError("review_facts_recovery type is invalid")
    proof_path = require_relpath(
        value["proof_path"], "review_facts_recovery.proof_path"
    )
    original_path = require_relpath(
        value["original_path"], "review_facts_recovery.original_path"
    )
    if proof_path == original_path:
        raise CatalogError("review_facts_recovery paths must differ")
    if not PurePosixPath(proof_path).name.endswith("_review_facts_recovery.json") \
            or not PurePosixPath(original_path).name.endswith("_review_facts.json") \
            or PurePosixPath(proof_path).parent != PurePosixPath(original_path).parent:
        raise CatalogError("review_facts_recovery paths are noncanonical")
    if not isinstance(value["proof_bytes"], int) or isinstance(value["proof_bytes"], bool) \
            or value["proof_bytes"] < 0:
        raise CatalogError("review_facts_recovery.proof_bytes is invalid")
    for field in (
        "proof_sha256", "original_sha256", "recovered_review_facts_sha256",
    ):
        require_sha(value[field], SHA256, f"review_facts_recovery.{field}")
    by_path = {item["path"]: item for item in result_files}
    proof = by_path.get(proof_path)
    original = by_path.get(original_path)
    if proof is None or original is None:
        raise CatalogError("review_facts_recovery files are absent from result_files")
    if proof["bytes"] != value["proof_bytes"] \
            or proof["sha256"] != value["proof_sha256"] \
            or original["sha256"] != value["original_sha256"]:
        raise CatalogError("review_facts_recovery result identity differs")
    return {
        field: value[field] for field in sorted(REVIEW_FACTS_RECOVERY_FIELDS)
    }


def normalize_terminal_record(value, line_number, raw_line):
    if not isinstance(value, dict):
        raise CatalogError(f"terminal index line {line_number} is not an object")
    schema = value.get("schema_version")
    if not isinstance(schema, int) or isinstance(schema, bool) or schema not in {1, 2}:
        raise CatalogError(f"terminal index line {line_number} uses an unknown schema")
    required = COMMON_FIELDS | (SCHEMA2_FIELDS if schema == 2 else set())
    missing = sorted(required - set(value))
    if missing:
        raise CatalogError(
            f"terminal index line {line_number} is missing fields: {','.join(missing)}"
        )
    optional = SCHEMA2_OPTIONAL_FIELDS if schema == 2 else set()
    unmodeled = sorted(set(value) - required - optional)
    route_id = require_text(value["route_id"], "route_id")
    operation_id = require_text(value["operation_id"], "operation_id")
    run_id = require_text(value["run_id"], "run_id")
    state = require_text(value["state"], "state")
    decision = require_decision(value["decision"], "decision")
    authorizes = require_text(value["authorizes"], "authorizes")
    receipt = require_sha(value["receipt"], SHA256, "receipt")
    route_commit = require_sha(value["route_commit"], SHA40, "route_commit")
    contract_path = require_relpath(value["contract_path"], "contract_path")
    closeout_path = require_relpath(value["closeout_path"], "closeout_path")
    conclusion_path = require_relpath(value["conclusion_path"], "conclusion_path")
    result_paths = value["result_paths"]
    if not isinstance(result_paths, list) or not result_paths:
        raise CatalogError("result_paths must be a non-empty list")
    result_paths = [
        require_relpath(path, f"result_paths[{index}]")
        for index, path in enumerate(result_paths)
    ]
    if len(result_paths) != len(set(result_paths)):
        raise CatalogError("result_paths contains duplicate paths")

    parent_path = None
    parent_tuple = None
    contract_bundle_paths = []
    binding_level = "path_only_legacy"
    review_facts_recovery = None
    if schema == 2:
        contract_bundle_paths = validate_file_records(
            value["contract_bundle"], "contract_bundle", unmodeled,
            CONTRACT_FILE_RECORD_FIELDS,
        )
        if not contract_bundle_paths:
            raise CatalogError("contract_bundle is empty")
        file_paths = validate_file_records(
            value["result_files"], "result_files", unmodeled,
            RESULT_FILE_RECORD_FIELDS,
        )
        if result_paths != file_paths:
            raise CatalogError("result_paths and result_files paths differ")
        parent_path, parent_tuple = validate_prior_record(
            value["prior_terminal_record"], unmodeled
        )
        for field in ("contract_sha256", "closeout_sha256", "conclusion_sha256"):
            require_sha(value[field], SHA256, field)
        if "review_facts_recovery" in value:
            review_facts_recovery = validate_review_facts_recovery(
                value["review_facts_recovery"], value["result_files"]
            )
        binding_level = "sha256_manifest"

    return {
        "index_line": line_number,
        "record_sha256": hashlib.sha256(raw_line).hexdigest(),
        "route_id": route_id,
        "operation_id": operation_id,
        "run_id": run_id,
        "state": state,
        "decision": decision,
        "authorizes": authorizes,
        "receipt": receipt,
        "route_commit": route_commit,
        "contract_path": contract_path,
        "closeout_path": closeout_path,
        "conclusion_path": conclusion_path,
        "result_paths": result_paths,
        "contract_bundle_paths": contract_bundle_paths,
        "result_count": len(result_paths),
        "schema_version": schema,
        "binding_level": binding_level,
        "prior_closeout_path": parent_path,
        "prior_terminal_tuple": parent_tuple,
        "unmodeled_fields": sorted(set(unmodeled)),
        # Private identity material for bounded follow-on verifiers. Public
        # catalog responses continue to use terminal_summary's allowlist.
        "contract_bundle": [
            _private_file_record(item, include_source_path=True)
            for item in value.get("contract_bundle", [])
        ],
        "result_files": [
            _private_file_record(item)
            for item in value.get("result_files", [])
        ],
        "contract_sha256": value.get("contract_sha256"),
        "closeout_sha256": value.get("closeout_sha256"),
        "conclusion_sha256": value.get("conclusion_sha256"),
        "review_facts_recovery": review_facts_recovery,
    }


def parse_terminal_index(raw):
    if not raw or not raw.endswith(b"\n"):
        raise CatalogError("terminal index must be non-empty canonical JSONL")
    records = []
    seen_identity = set()
    seen_closeout = set()
    seen_receipt = set()
    for line_number, raw_line in enumerate(raw.splitlines(), start=1):
        if not raw_line:
            raise CatalogError(f"terminal index line {line_number} is empty")
        try:
            value = json.loads(strict_text(raw_line, f"terminal index line {line_number}"))
        except json.JSONDecodeError as exc:
            raise CatalogError(f"terminal index line {line_number} is invalid JSON") from exc
        record = normalize_terminal_record(value, line_number, raw_line)
        identity = (record["route_id"], record["operation_id"], record["run_id"])
        if identity in seen_identity:
            raise CatalogError("terminal index contains a duplicate route/operation/run identity")
        if record["closeout_path"] in seen_closeout:
            raise CatalogError("terminal index contains a duplicate closeout path")
        if record["receipt"] in seen_receipt:
            raise CatalogError("terminal index contains a duplicate receipt")
        seen_identity.add(identity)
        seen_closeout.add(record["closeout_path"])
        seen_receipt.add(record["receipt"])
        records.append(record)
    return records


def select_terminal_leaf(records):
    if len(records) == 1 and records[0]["schema_version"] == 1:
        return records[0], "VALID_SINGLE"
    if any(record["schema_version"] != 2 for record in records):
        return None, "AMBIGUOUS_LEGACY"
    if len({record["operation_id"] for record in records}) != len(records) \
            or len({record["run_id"] for record in records}) != len(records):
        return None, "INVALID_CHAIN"
    by_closeout = {record["closeout_path"]: record for record in records}
    children = defaultdict(list)
    roots = []
    for record in records:
        parent_path = record["prior_closeout_path"]
        if parent_path is None:
            roots.append(record)
            continue
        parent = by_closeout.get(parent_path)
        if parent is None:
            return None, "INVALID_CHAIN"
        expected_tuple = {
            key: parent[key] for key in ("state", "decision", "authorizes")
        }
        if record["prior_terminal_tuple"] != expected_tuple:
            return None, "INVALID_CHAIN"
        children[parent_path].append(record)
    if len(roots) != 1 or any(len(items) != 1 for items in children.values()):
        return None, "INVALID_CHAIN"
    visited = set()
    cursor = roots[0]
    while True:
        if cursor["closeout_path"] in visited:
            return None, "INVALID_CHAIN"
        visited.add(cursor["closeout_path"])
        next_items = children.get(cursor["closeout_path"], [])
        if not next_items:
            break
        cursor = next_items[0]
    if len(visited) != len(records):
        return None, "INVALID_CHAIN"
    return cursor, "VALID_CHAIN"


def terminal_summary(record):
    value = {
        "index_line": record["index_line"],
        "schema_version": record["schema_version"],
        "operation_id": record["operation_id"],
        "run_id": record["run_id"],
        "state": record["state"],
        "decision": record["decision"],
        "authorizes": record["authorizes"],
        "route_commit": record["route_commit"],
        "contract_path": record["contract_path"],
        "closeout_path": record["closeout_path"],
        "conclusion_path": record["conclusion_path"],
        "result_count": record["result_count"],
        "binding_level": record["binding_level"],
        "prior_closeout_path": record["prior_closeout_path"],
        "record_sha256": record["record_sha256"],
        "unmodeled_fields": record["unmodeled_fields"],
    }
    if record.get("review_facts_recovery") is not None:
        value["review_facts_recovery_status"] = record[
            "review_facts_recovery"
        ]["status"]
    return value


def evidence_directory(closeout_path):
    prefix = f"{LOG_ROOT}/"
    if not closeout_path.startswith(prefix):
        raise CatalogError("closeout_path is outside the experiment log root")
    relative = closeout_path[len(prefix):]
    if "/" not in relative:
        raise CatalogError("closeout_path is not inside one evidence directory")
    return relative.split("/", 1)[0]


def marker_counts(paths):
    counts = {"closeout_named": 0, "conclusion_named": 0, "summary_named": 0}
    for path in paths:
        name = PurePosixPath(path).name.lower()
        if name == "closeout.json" or name.endswith("_closeout.json"):
            counts["closeout_named"] += 1
        if name.endswith(".json") and "conclusion" in name:
            counts["conclusion_named"] += 1
        if name.endswith((".json", ".csv")) and "summary" in name:
            counts["summary_named"] += 1
    return counts


def build_catalog(*, snapshot_commit, index_identity, tree_identity, records, paths):
    require_sha(snapshot_commit, SHA40, "snapshot_commit")
    group_paths = defaultdict(list)
    loose_files = []
    prefix = f"{LOG_ROOT}/"
    for path in sorted(paths):
        path = require_relpath(path, "experiment log path")
        if not path.startswith(prefix):
            raise CatalogError("experiment log listing escaped its root")
        relative = path[len(prefix):]
        if "/" not in relative:
            loose_files.append(relative)
            continue
        group_paths[relative.split("/", 1)[0]].append(path)

    path_set = set(paths)
    records_by_group = defaultdict(list)
    groups_by_route = defaultdict(set)
    for record in records:
        group = evidence_directory(record["closeout_path"])
        if group not in group_paths:
            raise CatalogError(
                f"terminal index references a missing evidence directory: {group}",
                state="INDEX_DIRECTORY_MISSING",
            )
        references = [
            record["closeout_path"], record["conclusion_path"],
            *record["result_paths"], *record["contract_bundle_paths"],
        ]
        outside_root = [path for path in references if not path.startswith(prefix)]
        if outside_root:
            raise CatalogError(
                f"terminal index reference is outside the experiment log root: "
                f"{outside_root[0]}",
                state="INDEX_REFERENCE_INVALID",
            )
        missing = [path for path in references if path not in path_set]
        if missing:
            raise CatalogError(
                f"terminal index reference is missing from the source tree: {missing[0]}",
                state="INDEX_REFERENCE_MISSING",
            )
        records_by_group[group].append(record)
        groups_by_route[record["route_id"]].add(group)
    split_routes = {
        route_id: groups for route_id, groups in groups_by_route.items()
        if len(groups) != 1
    }
    if split_routes:
        route_id = sorted(split_routes)[0]
        raise CatalogError(
            f"terminal route spans multiple evidence directories: {route_id}",
            state="ROUTE_DIRECTORY_AMBIGUOUS",
        )

    entries = []
    route_count = 0
    resolution_counts = Counter()
    schema_counts = Counter(record["schema_version"] for record in records)
    binding_counts = Counter(record["binding_level"] for record in records)
    unmodeled_record_count = sum(bool(record["unmodeled_fields"]) for record in records)
    for group in sorted(group_paths):
        group_records = records_by_group.get(group, [])
        base = {
            "record_kind": "evidence_directory",
            "directory_name": group,
            "directory_path": f"{LOG_ROOT}/{group}",
            "tracked_file_count": len(group_paths[group]),
            "marker_counts": marker_counts(group_paths[group]),
            "marker_basis": "git_path_names_only",
        }
        if not group_records:
            entries.append({
                **base,
                "index_coverage": "UNINDEXED",
                "terminal_assessment": "NOT_ASSESSED",
                "routes": [],
            })
            continue
        routes = []
        records_by_route = defaultdict(list)
        for record in group_records:
            records_by_route[record["route_id"]].append(record)
        for route_id in sorted(records_by_route):
            route_records = sorted(
                records_by_route[route_id], key=lambda item: item["index_line"]
            )
            leaf, resolution = select_terminal_leaf(route_records)
            resolution_counts[resolution] += 1
            route_count += 1
            routes.append({
                "route_id": route_id,
                "terminal_resolution": resolution,
                "terminal_record_count": len(route_records),
                "selected_operation_id": leaf["operation_id"] if leaf else None,
                "terminals": [terminal_summary(record) for record in route_records],
            })
        entries.append({
            **base,
            "index_coverage": "INDEXED",
            "terminal_assessment": (
                "INDEXED_RESOLVED"
                if all(route["selected_operation_id"] is not None for route in routes)
                else "INDEXED_UNRESOLVED"
            ),
            "routes": routes,
        })

    indexed_count = sum(entry["index_coverage"] == "INDEXED" for entry in entries)
    entries.extend({
        "record_kind": "loose_file",
        "file_name": name,
        "file_path": f"{LOG_ROOT}/{name}",
        "index_coverage": "UNINDEXED",
        "terminal_assessment": "NOT_ASSESSED",
        "marker_counts": marker_counts([name]),
        "marker_basis": "git_path_names_only",
        "routes": [],
    } for name in sorted(loose_files))
    header = {
        "record_kind": "catalog_header",
        "schema_version": 1,
        "snapshot_commit": snapshot_commit,
        "source_scope": ["github_terminal_index", "github_experiment_log_tree_names"],
        "excluded_sources": ["route_branches", "cloud_runtime"],
        "scientific_completeness": "not_assessed",
        "discovery_completeness": "complete",
        "terminal_index": {
            **index_identity,
            "record_count": len(records),
            "route_count": route_count,
            "schema_counts": {str(key): schema_counts[key] for key in sorted(schema_counts)},
            "binding_counts": dict(sorted(binding_counts.items())),
            "unmodeled_record_count": unmodeled_record_count,
            "terminal_resolution_counts": dict(sorted(resolution_counts.items())),
        },
        "experiment_log_tree": {
            **tree_identity,
            "tracked_file_count": len(paths),
            "catalog_entry_count": len(entries),
            "directory_count": len(group_paths),
            "indexed_directory_count": indexed_count,
            "unindexed_directory_count": len(group_paths) - indexed_count,
            "loose_file_count": len(loose_files),
        },
    }
    collection_sha256 = canonical_sha256(entries)
    catalog_sha256 = canonical_sha256({
        "header": header, "entries": entries, "loose_files": sorted(loose_files),
    })
    return {
        "header": header,
        "entries": entries,
        "loose_files": sorted(loose_files),
        "collection_sha256": collection_sha256,
        "catalog_sha256": catalog_sha256,
    }


def git_bytes(repo, args, *, limit):
    completed = convirctl.run_argv_limited(
        [convirctl.GIT, "-C", repo, *args], input_bytes=b"", timeout=60,
        capture_limit=limit, env=convirctl.git_environment(60),
    )
    if completed.returncode:
        error = completed.stderr.decode("utf-8", errors="replace").strip()
        raise CatalogError(error or "Git source read failed", state="GIT_READ_FAILED")
    if len(completed.stdout) > limit:
        raise CatalogError("Git source exceeds its bounded read limit", state="SOURCE_TOO_LARGE")
    return completed.stdout


def git_text(repo, *args):
    return strict_text(git_bytes(repo, list(args), limit=4096), "Git output").strip()


def load_terminal_records(repo_value, commit):
    repo = Path(repo_value)
    if not repo.is_absolute():
        raise CatalogError("repo must be absolute", state="ARGUMENTS_INVALID", exit_code=2)
    repo = repo.resolve()
    if not repo.is_dir():
        raise CatalogError("repo is not a directory", state="ARGUMENTS_INVALID", exit_code=2)
    require_sha(commit, SHA40, "commit")
    resolved = git_text(repo, "rev-parse", "--verify", f"{commit}^{{commit}}")
    if resolved != commit:
        raise CatalogError("commit did not resolve exactly", state="SNAPSHOT_INVALID")

    index_oid = git_text(repo, "rev-parse", "--verify", f"{commit}:{INDEX_PATH}")
    require_sha(index_oid, SHA40, "terminal index blob")
    if git_text(repo, "cat-file", "-t", index_oid) != "blob":
        raise CatalogError("terminal index is not a blob", state="SOURCE_INVALID")
    try:
        index_size = int(git_text(repo, "cat-file", "-s", index_oid))
    except ValueError as exc:
        raise CatalogError("terminal index size is invalid", state="SOURCE_INVALID") from exc
    if index_size > MAX_INDEX_BYTES:
        raise CatalogError("terminal index exceeds 1 MiB", state="SOURCE_TOO_LARGE")
    index_raw = git_bytes(repo, ["cat-file", "blob", index_oid], limit=MAX_INDEX_BYTES)
    if len(index_raw) != index_size:
        raise CatalogError("terminal index size changed", state="SOURCE_IDENTITY_MISMATCH")

    return repo, parse_terminal_index(index_raw), {
        "path": INDEX_PATH,
        "blob_oid": index_oid,
        "bytes": index_size,
        "sha256": hashlib.sha256(index_raw).hexdigest(),
    }


def load_catalog(repo_value, commit):
    repo, records, index_identity = load_terminal_records(repo_value, commit)

    tree_oid = git_text(repo, "rev-parse", "--verify", f"{commit}:{LOG_ROOT}")
    require_sha(tree_oid, SHA40, "experiment log tree")
    if git_text(repo, "cat-file", "-t", tree_oid) != "tree":
        raise CatalogError("experiment log root is not a tree", state="SOURCE_INVALID")
    tree_raw = git_bytes(
        repo,
        ["--literal-pathspecs", "ls-tree", "-r", "-z", "--name-only", commit,
         "--", LOG_ROOT],
        limit=MAX_TREE_BYTES,
    )
    if tree_raw and not tree_raw.endswith(b"\0"):
        raise CatalogError("experiment log tree returned an incomplete path")
    raw_paths = tree_raw.split(b"\0")[:-1] if tree_raw else []
    paths = [strict_text(path, "experiment log path") for path in raw_paths]
    if len(paths) != len(set(paths)):
        raise CatalogError("experiment log tree contains duplicate paths")

    return build_catalog(
        snapshot_commit=commit,
        index_identity=index_identity,
        tree_identity={
            "path": LOG_ROOT,
            "tree_oid": tree_oid,
            "path_collection_sha256": canonical_sha256(sorted(paths)),
        },
        records=records,
        paths=paths,
    )


def response(operation, state, *, ok=True, exit_code=0, **fields):
    return {
        "ok": ok,
        "operation": operation,
        "state": state,
        "exit_code": exit_code,
        **fields,
    }


def summary_response(catalog):
    value = response(
        "catalog-summary", "CATALOG_SUMMARY_OK",
        header=catalog["header"],
        catalog_sha256=catalog["catalog_sha256"],
        collection_sha256=catalog["collection_sha256"],
    )
    if len(canonical_bytes(value)) + 1 > MAX_RESPONSE_BYTES:
        raise CatalogError("catalog summary exceeds the response budget")
    return value


def completeness_receipt(catalog_value):
    """Return a compact receipt for GitHub catalog discovery completeness."""
    header = catalog_value.get("header")
    entries = catalog_value.get("entries")
    if not isinstance(header, dict) or not isinstance(entries, list):
        raise CatalogError("catalog has an invalid completeness input")

    tree = header.get("experiment_log_tree")
    terminal_index = header.get("terminal_index")
    if not isinstance(tree, dict) or not isinstance(terminal_index, dict):
        raise CatalogError("catalog source identities are missing")

    entry_counts = Counter()
    routes = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise CatalogError("catalog contains a non-object entry")
        coverage = entry.get("index_coverage")
        kind = entry.get("record_kind")
        if coverage not in {"INDEXED", "UNINDEXED"}:
            raise CatalogError("catalog entry has an unknown index partition")
        if kind not in {"evidence_directory", "loose_file"}:
            raise CatalogError("catalog entry has an unknown record kind")
        if coverage == "INDEXED" and kind != "evidence_directory":
            raise CatalogError("only evidence directories may be indexed")
        if coverage == "UNINDEXED" and (
            entry.get("routes") or entry.get("terminal_assessment") != "NOT_ASSESSED"
        ):
            raise CatalogError("unindexed catalog entry was implicitly assessed")
        entry_counts[(coverage, kind)] += 1
        routes.extend(entry.get("routes", []))

    indexed_entries = sum(
        count for (coverage, _), count in entry_counts.items()
        if coverage == "INDEXED"
    )
    unindexed_entries = len(entries) - indexed_entries
    indexed_directories = entry_counts[("INDEXED", "evidence_directory")]
    unindexed_directories = entry_counts[("UNINDEXED", "evidence_directory")]
    loose_files = entry_counts[("UNINDEXED", "loose_file")]
    expected_entry_counts = {
        "catalog_entry_count": len(entries),
        "directory_count": indexed_directories + unindexed_directories,
        "indexed_directory_count": indexed_directories,
        "unindexed_directory_count": unindexed_directories,
        "loose_file_count": loose_files,
    }
    if any(tree.get(key) != value for key, value in expected_entry_counts.items()):
        raise CatalogError("catalog entry partitions differ from the tree summary")

    terminals = []
    resolution_counts = Counter()
    for route in routes:
        if not isinstance(route, dict) or not isinstance(route.get("terminals"), list):
            raise CatalogError("catalog route has an invalid terminal summary")
        if route.get("terminal_record_count") != len(route["terminals"]):
            raise CatalogError("catalog route terminal count differs")
        terminals.extend(route["terminals"])
        resolution_counts[route.get("terminal_resolution")] += 1

    schema_counts = Counter(str(item.get("schema_version")) for item in terminals)
    binding_counts = Counter(item.get("binding_level") for item in terminals)
    unmodeled_record_count = sum(bool(item.get("unmodeled_fields")) for item in terminals)
    expected_terminal_values = {
        "record_count": len(terminals),
        "route_count": len(routes),
        "schema_counts": dict(sorted(schema_counts.items())),
        "binding_counts": dict(sorted(binding_counts.items())),
        "unmodeled_record_count": unmodeled_record_count,
        "terminal_resolution_counts": dict(sorted(resolution_counts.items())),
    }
    if any(terminal_index.get(key) != value for key, value in expected_terminal_values.items()):
        raise CatalogError("catalog terminal partitions differ from the index summary")

    unresolved_routes = [
        route for route in routes if route.get("selected_operation_id") is None
    ]
    ambiguous_legacy_routes = sum(
        route.get("terminal_resolution") == "AMBIGUOUS_LEGACY"
        for route in unresolved_routes
    )
    invalid_terminal_routes = sum(
        route.get("terminal_resolution") == "INVALID_CHAIN"
        for route in unresolved_routes
    )
    other_unresolved_routes = (
        len(unresolved_routes) - ambiguous_legacy_routes - invalid_terminal_routes
    )
    unresolved_counts = {
        "unclassified_unindexed_entries": unindexed_entries,
        "path_only_legacy_terminal_records": binding_counts["path_only_legacy"],
        "ambiguous_legacy_routes": ambiguous_legacy_routes,
        "invalid_terminal_routes": invalid_terminal_routes,
        "other_unresolved_terminal_routes": other_unresolved_routes,
        "unmodeled_terminal_records": unmodeled_record_count,
    }
    review_completeness = (
        "incomplete" if any(unresolved_counts.values()) else "complete"
    )

    terminal_identity = {
        key: terminal_index.get(key)
        for key in ("path", "blob_oid", "bytes", "sha256")
    }
    tree_identity = {
        key: tree.get(key)
        for key in (
            "path", "tree_oid", "path_collection_sha256", "tracked_file_count",
        )
    }
    value = response(
        "completeness-receipt", "PROJECT_GITHUB_COMPLETENESS_RECEIPT_OK",
        schema_version=2,
        snapshot_commit=header.get("snapshot_commit"),
        receipt_scope="github_project_catalog",
        source_scope=list(header.get("source_scope", [])),
        excluded_sources=sorted(set(
            [*header.get("excluded_sources", []), "result_contents"]
        )),
        source_identities={
            "catalog_sha256": catalog_value.get("catalog_sha256"),
            "collection_sha256": catalog_value.get("collection_sha256"),
            "terminal_index": terminal_identity,
            "experiment_log_tree": tree_identity,
        },
        entry_partition={
            "catalog_entries": len(entries),
            "indexed_entries": indexed_entries,
            "unindexed_entries": unindexed_entries,
            "evidence_directories": indexed_directories + unindexed_directories,
            "indexed_directories": indexed_directories,
            "unindexed_directories": unindexed_directories,
            "loose_files": loose_files,
            "partition_complete": indexed_entries + unindexed_entries == len(entries),
        },
        terminal_partition={
            "terminal_records": len(terminals),
            "routes": len(routes),
            "resolved_routes": len(routes) - len(unresolved_routes),
            "unresolved_routes": len(unresolved_routes),
            "schema_counts": dict(sorted(schema_counts.items())),
            "binding_counts": dict(sorted(binding_counts.items())),
            "terminal_resolution_counts": dict(sorted(resolution_counts.items())),
        },
        unresolved_counts=unresolved_counts,
        unresolved_reason_counts_are_nonexclusive=True,
        review_completeness=review_completeness,
        scientific_completeness="not_assessed",
        git_mutations_performed=False,
    )
    value["receipt_sha256"] = canonical_sha256(value)
    if len(canonical_bytes(value)) + 1 > MAX_RESPONSE_BYTES:
        raise CatalogError("completeness receipt exceeds the response budget")
    return value


def catalog_cursor_position(catalog, args, query_sha256):
    if args.cursor is None:
        return 0
    try:
        cursor = convirctl.decode_repo_cursor(
            args.cursor, CURSOR_OPERATION, query_sha256
        )
    except convirctl.ControlError as exc:
        raise CatalogError(
            str(exc), state=exc.state, exit_code=exc.exit_code
        ) from exc
    expected_commit = catalog["header"]["snapshot_commit"]
    expected_tree = catalog["header"]["experiment_log_tree"]["tree_oid"]
    if cursor["commit"] != expected_commit or cursor["object_id"] != expected_tree:
        raise CatalogError(
            "catalog cursor does not match this source snapshot",
            state="REPO_CURSOR_IDENTITY_MISMATCH",
        )
    return cursor["position"]


def entries_response(catalog, args):
    terms = []
    for term in args.term:
        require_text(term, "term")
        if len(term) > 128:
            raise CatalogError("term exceeds 128 characters", state="ARGUMENTS_INVALID", exit_code=2)
        terms.append(term.casefold())
    if len(terms) > MAX_TERMS:
        raise CatalogError("at most 8 terms are accepted", state="ARGUMENTS_INVALID", exit_code=2)
    candidates = []
    for entry in catalog["entries"]:
        if args.coverage != "all" and entry["index_coverage"].lower() != args.coverage:
            continue
        searchable = canonical_bytes(entry).decode("utf-8").casefold()
        if all(term in searchable for term in terms):
            candidates.append(entry)
    query_sha256 = canonical_sha256({
        "snapshot_commit": catalog["header"]["snapshot_commit"],
        "catalog_sha256": catalog["catalog_sha256"],
        "coverage": args.coverage,
        "terms": terms,
    })
    offset = catalog_cursor_position(catalog, args, query_sha256)
    if offset > len(candidates) or (offset == len(candidates) and offset != 0):
        raise CatalogError(
            "catalog cursor is outside the filtered catalog",
            state="REPO_CURSOR_INVALID",
        )
    selected = candidates[offset:offset + args.limit]
    while selected:
        end = offset + len(selected)
        complete = end == len(candidates)
        next_cursor = None if complete else convirctl.encode_repo_cursor(
            CURSOR_OPERATION,
            catalog["header"]["snapshot_commit"],
            query_sha256,
            end,
            catalog["header"]["experiment_log_tree"]["tree_oid"],
        )
        value = response(
            "catalog-entries", "CATALOG_ENTRIES_OK",
            snapshot_commit=catalog["header"]["snapshot_commit"],
            catalog_sha256=catalog["catalog_sha256"],
            collection_sha256=canonical_sha256(candidates),
            query_sha256=query_sha256,
            coverage=args.coverage,
            terms=terms,
            offset=offset,
            returned_count=len(selected),
            total_count=len(candidates),
            entries=selected,
            page_sha256=canonical_sha256(selected),
            page_complete=True,
            terminal_page=complete,
            complete=complete,
            has_more=not complete,
            next_cursor=next_cursor,
            scientific_completeness="not_assessed",
            excluded_sources=["route_branches", "cloud_runtime"],
        )
        if len(canonical_bytes(value)) + 1 <= MAX_RESPONSE_BYTES:
            return value
        selected.pop()
    if offset == len(candidates):
        return response(
            "catalog-entries", "CATALOG_ENTRIES_OK",
            snapshot_commit=catalog["header"]["snapshot_commit"],
            catalog_sha256=catalog["catalog_sha256"],
            collection_sha256=canonical_sha256(candidates),
            query_sha256=query_sha256,
            coverage=args.coverage,
            terms=terms,
            offset=offset,
            returned_count=0,
            total_count=len(candidates),
            entries=[],
            page_sha256=canonical_sha256([]),
            page_complete=True,
            terminal_page=True,
            complete=True,
            has_more=False,
            next_cursor=None,
            scientific_completeness="not_assessed",
            excluded_sources=["route_branches", "cloud_runtime"],
        )
    raise CatalogError("one catalog entry exceeds the response budget", state="ENTRY_TOO_LARGE")


def exact_commit(value):
    if not SHA40.fullmatch(value):
        raise argparse.ArgumentTypeError("commit must be 40 lowercase hex")
    return value


def bounded_limit(value):
    try:
        value = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("limit must be an integer") from exc
    if not 1 <= value <= MAX_PAGE_ENTRIES:
        raise argparse.ArgumentTypeError("limit must be in [1, 100]")
    return value


def parser():
    root = JsonArgumentParser(description=__doc__)
    root.add_argument("--repo", required=True)
    root.add_argument("--commit", required=True, type=exact_commit)
    commands = root.add_subparsers(dest="operation", required=True)
    commands.add_parser("summary")
    commands.add_parser("receipt")
    entries = commands.add_parser("entries")
    entries.add_argument(
        "--coverage", choices=("indexed", "unindexed", "all"), default="indexed"
    )
    entries.add_argument("--term", action="append", default=[])
    entries.add_argument("--cursor")
    entries.add_argument("--limit", type=bounded_limit, default=20)
    return root


def main(argv=None):
    raw_args = list(sys.argv[1:] if argv is None else argv)
    operation = "catalog"
    try:
        args = parser().parse_args(raw_args)
        operation = f"catalog-{args.operation}"
        catalog = load_catalog(args.repo, args.commit)
        if args.operation == "summary":
            value = summary_response(catalog)
        elif args.operation == "receipt":
            value = completeness_receipt(catalog)
        else:
            value = entries_response(catalog, args)
    except CatalogError as exc:
        value = response(
            operation, exc.state, ok=False, exit_code=exc.exit_code, error=str(exc)
        )
    except convirctl.ControlError as exc:
        value = response(
            operation, exc.state, ok=False, exit_code=exc.exit_code, error=str(exc)
        )
    except OSError as exc:
        value = response(
            operation, "LOCAL_IO_FAILED", ok=False, exit_code=2, error=str(exc)
        )
    except Exception as exc:
        value = response(
            operation, "INTERNAL_CATALOG_ERROR", ok=False, exit_code=70,
            error=f"{type(exc).__name__}: {exc}",
        )
    if len(canonical_bytes(value)) + 1 > MAX_RESPONSE_BYTES:
        value = response(
            operation, "RESPONSE_TOO_LARGE", ok=False, exit_code=3,
            error="catalog response exceeded the 32 KiB transport budget",
        )
    print(canonical_bytes(value).decode("utf-8"))
    return value["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
