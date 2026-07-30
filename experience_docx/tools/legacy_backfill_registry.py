#!/usr/bin/env python3
"""Build and validate a read-only legacy evidence backfill registry."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path


REGISTRY_SCHEMA_VERSION = 2
REGISTRY_PATH = "experience_docx/LEGACY_BACKFILL_REGISTRY.json"
LOG_ROOT = "experience_docx/experiment_logs"
SHA40 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
MAX_FILE_BYTES = 1024 * 1024
TEXT_SUFFIXES = {".csv", ".json", ".jsonl", ".md", ".txt"}

EXCLUDED_PREFIXES = (
    "cloud_py310_", "command_reliability_", "convir_ops_",
    "engineering_repair_", "experiment_control_",
    "generic_run_monitoring_", "highvalue_sync_",
    "p0_p1_research_governance_", "route_ready_",
    "science_fastpath_",
)
EXPERIMENT_PREFIXES = ("haze4k_", "sots-ots-")
TERMINAL_EXPERIMENT_PREFIXES = (
    "convir_only_", "haze4k-", "haze4k_", "nhhaze_", "reside-", "sots-ots-",
)


class RegistryError(RuntimeError):
    pass


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def canonical_sha256(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def git(repo, *args, check=True, limit=8 * 1024 * 1024):
    completed = subprocess.run(
        ["/usr/bin/git", "-C", str(repo), *args],
        capture_output=True, check=False,
    )
    if completed.returncode and check:
        raise RegistryError(completed.stderr.decode("utf-8", errors="replace").strip())
    if len(completed.stdout) > limit:
        raise RegistryError("git output exceeds registry bound")
    return completed.stdout


def require_repo(value):
    repo = Path(value).resolve()
    if not repo.is_absolute() or not repo.is_dir():
        raise RegistryError("repo must be an existing absolute directory")
    return repo


def require_sha(value, pattern, label):
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise RegistryError(f"{label} has an invalid identity")
    return value


def snapshot_paths(repo, commit):
    require_sha(commit, SHA40, "commit")
    resolved = git(repo, "rev-parse", "--verify", f"{commit}^{{commit}}").decode().strip()
    if resolved != commit:
        raise RegistryError("commit did not resolve exactly")
    tree_oid = git(repo, "rev-parse", "--verify", f"{commit}:{LOG_ROOT}").decode().strip()
    require_sha(tree_oid, SHA40, "experiment-log tree")
    if git(repo, "cat-file", "-t", tree_oid).decode().strip() != "tree":
        raise RegistryError("experiment-log root is not a tree")
    raw = git(repo, "--literal-pathspecs", "ls-tree", "-r", "-z", "--name-only", commit, "--", LOG_ROOT)
    if raw and not raw.endswith(b"\0"):
        raise RegistryError("incomplete experiment-log path listing")
    try:
        paths = sorted(item.decode("utf-8") for item in raw.split(b"\0") if item)
    except UnicodeDecodeError as exc:
        raise RegistryError("experiment-log path is not UTF-8") from exc
    if len(paths) != len(set(paths)):
        raise RegistryError("experiment-log path listing contains duplicates")
    return tree_oid, paths


def classify_directory(directory_name):
    if directory_name.startswith(EXCLUDED_PREFIXES):
        return "NON_EXPERIMENT_RECORD", "DO_NOT_READ", "environment_or_operations_or_mcp_or_governance_record"
    if directory_name.startswith(EXPERIMENT_PREFIXES):
        return "LEGACY_EXPERIMENT_CANDIDATE", "READ_ON_DEMAND", "historical_experiment_directory"
    return "UNRESOLVED", "DO_NOT_READ", "directory_class_not_proven"


def classify_terminal_route(route_id):
    if route_id.startswith(EXCLUDED_PREFIXES):
        return "NON_EXPERIMENT_RECORD", "DO_NOT_READ", "environment_or_operations_or_mcp_or_governance_record"
    if route_id.startswith(TERMINAL_EXPERIMENT_PREFIXES):
        return "LEGACY_EXPERIMENT_CANDIDATE", "READ_ON_DEMAND", "historical_terminal_route"
    return "UNRESOLVED", "DO_NOT_READ", "terminal_route_class_not_proven"


def grouped_paths(paths):
    grouped = {}
    loose = []
    prefix = LOG_ROOT + "/"
    for path in paths:
        if not path.startswith(prefix):
            raise RegistryError(f"path outside experiment-log root: {path}")
        rest = path[len(prefix):]
        if "/" not in rest:
            loose.append(path)
            continue
        directory, relative = rest.split("/", 1)
        if not directory or not relative:
            raise RegistryError(f"invalid experiment-log path: {path}")
        grouped.setdefault(directory, []).append((path, relative))
    return grouped, sorted(loose)


def file_identity(repo, commit, path):
    oid = git(repo, "rev-parse", "--verify", f"{commit}:{path}").decode().strip()
    require_sha(oid, SHA40, f"blob identity for {path}")
    if git(repo, "cat-file", "-t", oid).decode().strip() != "blob":
        raise RegistryError(f"not a blob: {path}")
    try:
        size = int(git(repo, "cat-file", "-s", oid).decode().strip())
    except ValueError as exc:
        raise RegistryError(f"invalid blob size: {path}") from exc
    value = {"path": path, "blob_oid": oid, "bytes": size, "sha256": None, "readable": False}
    if size > MAX_FILE_BYTES:
        value["reason"] = "file_exceeds_1MiB_bound"
        return value
    if Path(path).suffix.lower() not in TEXT_SUFFIXES:
        value["reason"] = "unsupported_text_suffix"
        return value
    raw = git(repo, "cat-file", "blob", oid, limit=MAX_FILE_BYTES + 1)
    if len(raw) != size:
        raise RegistryError(f"blob size changed: {path}")
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError:
        value["reason"] = "file_is_not_utf8"
        return value
    value.update({"sha256": hashlib.sha256(raw).hexdigest(), "readable": True})
    return value


def terminal_file_paths(record):
    paths = [
        record["contract_path"],
        record["closeout_path"],
        record["conclusion_path"],
        *record.get("result_paths", []),
    ]
    return sorted(set(paths))


def build_registry(repo, commit):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import convir_evidence_catalog as catalog

    tree_oid, paths = snapshot_paths(repo, commit)
    catalog_value = catalog.load_catalog(repo, commit)
    unindexed_directories = {
        item["directory_path"]
        for item in catalog_value["entries"]
        if item.get("record_kind") == "evidence_directory"
        and item.get("index_coverage") == "UNINDEXED"
    }
    grouped, loose = grouped_paths(paths)
    if not unindexed_directories.issubset({LOG_ROOT + "/" + name for name in grouped}):
        raise RegistryError("catalog unindexed directory is absent from source tree")

    directories = []
    for name in sorted(grouped):
        directory_path = LOG_ROOT + "/" + name
        if directory_path not in unindexed_directories:
            continue
        category, access, reason = classify_directory(name)
        files = []
        if access == "READ_ON_DEMAND":
            directory_oid = git(repo, "rev-parse", "--verify", f"{commit}:{directory_path}").decode().strip()
            require_sha(directory_oid, SHA40, f"directory tree identity for {directory_path}")
            if git(repo, "cat-file", "-t", directory_oid).decode().strip() != "tree":
                raise RegistryError(f"not a tree: {directory_path}")
            files = [{
                "path": directory_path,
                "tree_oid": directory_oid,
                "readable": False,
                "reason": "directory_tree_identity_only; list_and_read_files_on_demand",
            }]
        directories.append({
            "directory_path": directory_path,
            "directory_name": name,
            "category": category,
            "default_access": access,
            "classification_reason": reason,
            "tracked_file_count": len(grouped[name]),
            "files": files,
        })

    _, terminal_records, terminal_index = catalog.load_terminal_records(repo, commit)
    schema1_records = [record for record in terminal_records if record.get("schema_version") == 1]
    record_count_by_route = Counter(record["route_id"] for record in schema1_records)
    legacy_terminals = []
    for record in schema1_records:
        category, access, reason = classify_terminal_route(record["route_id"])
        item = {
            "route_id": record["route_id"],
            "operation_id": record["operation_id"],
            "run_id": record["run_id"],
            "terminal_record_sha256": record["record_sha256"],
            "schema2_upgrade": False,
        }
        if access != "READ_ON_DEMAND":
            item.update({
                "backfill_status": "EXCLUDED" if category == "NON_EXPERIMENT_RECORD" else "UNRESOLVED",
                "reason": reason,
                "files": [],
            })
        elif record_count_by_route[record["route_id"]] != 1:
            item.update({
                "backfill_status": "LEGACY_HASH_BOUND_UNSELECTED",
                "reason": "route_has_multiple_schema1_terminal_records",
                "files": [],
            })
        else:
            item.update({
                "backfill_status": "LEGACY_HASH_BOUND_REVIEWABLE",
                "limitations": [
                    "post_hoc_git_hash_binding",
                    "no_schema2_terminal_manifest",
                    "no_raw_artifact_seal",
                ],
                "files": [file_identity(repo, commit, path) for path in terminal_file_paths(record)],
            })
        legacy_terminals.append(item)

    status_counts = Counter(item["backfill_status"] for item in legacy_terminals)
    registry = {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "record_type": "legacy_backfill_registry",
        "generator_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "snapshot_commit": commit,
        "source_scope": ["github_unindexed_experiment_log_entries", "github_path_only_terminal_index"],
        "excluded_content": ["cloud_runtime", "route_branches", "protected_data"],
        "policy": {
            "NON_EXPERIMENT_RECORD": "DO_NOT_READ",
            "UNRESOLVED": "DO_NOT_READ",
            "LEGACY_EXPERIMENT_CANDIDATE": "READ_ON_DEMAND",
            "schema2_upgrade": False,
            "scientific_completeness": "not_assessed",
        },
        "source_tree": {
            "path": LOG_ROOT,
            "tree_oid": tree_oid,
            "path_collection_sha256": canonical_sha256(paths),
            "tracked_file_count": len(paths),
            "directory_count": len(grouped),
            "loose_file_count": len(loose),
        },
        "scope": {
            "unindexed_directory_count": len(unindexed_directories),
            "unindexed_directory_collection_sha256": canonical_sha256(sorted(unindexed_directories)),
        },
        "terminal_index": terminal_index,
        "partition": {
            "legacy_experiment_candidates": sum(item["category"] == "LEGACY_EXPERIMENT_CANDIDATE" for item in directories),
            "non_experiment_records": sum(item["category"] == "NON_EXPERIMENT_RECORD" for item in directories),
            "unresolved_directories": sum(item["category"] == "UNRESOLVED" for item in directories),
            "unresolved_loose_files": len(loose),
            "legacy_terminal_records": len(schema1_records),
            "legacy_hash_bound_reviewable": status_counts["LEGACY_HASH_BOUND_REVIEWABLE"],
            "legacy_hash_bound_unselected": status_counts["LEGACY_HASH_BOUND_UNSELECTED"],
            "legacy_terminal_unresolved": status_counts["UNRESOLVED"] + status_counts["EXCLUDED"],
        },
        "directories": directories,
        "loose_files": [{"path": path, "category": "UNRESOLVED", "default_access": "DO_NOT_READ"} for path in loose],
        "legacy_terminals": sorted(legacy_terminals, key=lambda item: item["terminal_record_sha256"]),
    }
    registry["registry_sha256"] = canonical_sha256(registry)
    return registry


def validate_registry(registry):
    if not isinstance(registry, dict) or registry.get("schema_version") != REGISTRY_SCHEMA_VERSION:
        raise RegistryError("registry schema version is invalid")
    if registry.get("record_type") != "legacy_backfill_registry":
        raise RegistryError("registry record type is invalid")
    require_sha(registry.get("generator_source_sha256"), SHA256, "registry generator source")
    require_sha(registry.get("snapshot_commit"), SHA40, "registry snapshot commit")
    expected = registry.get("registry_sha256")
    copy = dict(registry)
    copy.pop("registry_sha256", None)
    if expected != canonical_sha256(copy):
        raise RegistryError("registry SHA-256 does not match its content")

    source_tree = registry.get("source_tree")
    terminal_index = registry.get("terminal_index")
    if not isinstance(source_tree, dict) or not isinstance(terminal_index, dict):
        raise RegistryError("registry source identities are missing")
    require_sha(source_tree.get("tree_oid"), SHA40, "registry source tree")
    require_sha(source_tree.get("path_collection_sha256"), SHA256, "registry source path collection")
    require_sha(terminal_index.get("blob_oid"), SHA40, "registry terminal-index blob")
    require_sha(terminal_index.get("sha256"), SHA256, "registry terminal-index SHA-256")

    directories = registry.get("directories")
    if not isinstance(directories, list) or len({item.get("directory_path") for item in directories if isinstance(item, dict)}) != len(directories):
        raise RegistryError("registry directories are invalid or duplicated")
    for item in directories:
        if not isinstance(item, dict) or item.get("category") not in {
            "NON_EXPERIMENT_RECORD", "UNRESOLVED", "LEGACY_EXPERIMENT_CANDIDATE",
        }:
            raise RegistryError("registry directory category is invalid")
        if item.get("category") in {"NON_EXPERIMENT_RECORD", "UNRESOLVED"}:
            if item.get("default_access") != "DO_NOT_READ" or item.get("files"):
                raise RegistryError("excluded/unresolved directory is readable")
        elif item.get("default_access") != "READ_ON_DEMAND":
            raise RegistryError("experiment candidate is unexpectedly blocked")

    terminals = registry.get("legacy_terminals")
    if not isinstance(terminals, list) or len({item.get("terminal_record_sha256") for item in terminals if isinstance(item, dict)}) != len(terminals):
        raise RegistryError("legacy terminal registry is invalid or duplicated")
    statuses = Counter()
    for terminal in terminals:
        if not isinstance(terminal, dict):
            raise RegistryError("legacy terminal is not an object")
        require_sha(terminal.get("terminal_record_sha256"), SHA256, "legacy terminal record")
        if terminal.get("schema2_upgrade") is not False:
            raise RegistryError("legacy terminal is incorrectly upgraded")
        status = terminal.get("backfill_status")
        if status not in {
            "LEGACY_HASH_BOUND_REVIEWABLE", "LEGACY_HASH_BOUND_UNSELECTED", "UNRESOLVED", "EXCLUDED",
        }:
            raise RegistryError("legacy terminal has an invalid status")
        statuses[status] += 1
        files = terminal.get("files")
        if not isinstance(files, list):
            raise RegistryError("legacy terminal files are invalid")
        if status == "LEGACY_HASH_BOUND_REVIEWABLE" and not files:
            raise RegistryError("reviewable legacy terminal has no files")
        if status != "LEGACY_HASH_BOUND_REVIEWABLE" and files:
            raise RegistryError("unselected legacy terminal exposes files")
        for file in files:
            if not isinstance(file, dict):
                raise RegistryError("legacy terminal file is invalid")
            require_sha(file.get("blob_oid"), SHA40, "legacy terminal blob")
            if file.get("readable"):
                require_sha(file.get("sha256"), SHA256, "readable legacy terminal file")
            elif file.get("sha256") is not None:
                require_sha(file.get("sha256"), SHA256, "unreadable legacy terminal file")

    partition = registry.get("partition")
    if not isinstance(partition, dict):
        raise RegistryError("registry partition is missing")
    if partition.get("legacy_terminal_records") != len(terminals) \
            or partition.get("legacy_hash_bound_reviewable") != statuses["LEGACY_HASH_BOUND_REVIEWABLE"] \
            or partition.get("legacy_hash_bound_unselected") != statuses["LEGACY_HASH_BOUND_UNSELECTED"]:
        raise RegistryError("registry terminal partition differs from records")
    return True


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--output", default=REGISTRY_PATH)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    repo = require_repo(args.repo)
    if args.check:
        raw = git(repo, "show", f"{args.commit}:{args.output}")
        try:
            registry = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RegistryError("committed registry is not valid UTF-8 JSON") from exc
        validate_registry(registry)
        base_commit = registry["snapshot_commit"]
        if git(repo, "merge-base", "--is-ancestor", base_commit, args.commit, check=False):
            raise RegistryError("registry snapshot is not an ancestor of the requested commit")
        expected = build_registry(repo, base_commit)
        if raw != canonical_bytes(expected) + b"\n":
            raise RegistryError("committed registry differs from its frozen snapshot rebuild")
        print("LEGACY_BACKFILL_REGISTRY_OK")
        return 0

    registry = build_registry(repo, args.commit)
    validate_registry(registry)
    output = Path(args.output)
    if not output.is_absolute():
        output = repo / output
    output = output.resolve()
    try:
        output.relative_to(repo)
    except ValueError as exc:
        raise RegistryError("output must remain inside the repository") from exc
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_bytes(registry) + b"\n")
    print(json.dumps({"marker": "LEGACY_BACKFILL_REGISTRY_BUILT", "registry_sha256": registry["registry_sha256"], "partition": registry["partition"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RegistryError as exc:
        print(f"LEGACY_BACKFILL_REGISTRY_FAILED: {exc}", file=sys.stderr)
        raise SystemExit(2)
