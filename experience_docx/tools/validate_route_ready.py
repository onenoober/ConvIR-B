#!/usr/bin/env python3
"""One-pass staged-snapshot validator using the exact convir-ops parser."""

from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable

import convir_ops_mcp as ops
import capability_registry
import experiment_spec_compiler as spec_compiler
import validate_experiment_card as card_validator
from route_runtime_contract import (
    GENERIC_RUNNER_RELPATH,
    RUNTIME_BUNDLE_RELPATHS,
    ContractError,
    repository_asset_identity_errors,
    runtime_spec_relpath,
    validate_asset_manifest,
    validate_model_capability,
    validate_precision_certificate,
    validate_runtime_spec,
)


class ReadyError(RuntimeError):
    pass


GIT = "/usr/bin/git"
BASH = "/bin/bash"
REVIEW_FACTS_RULES_FLOOR = "ef1f746859fba84bd76ae74525b05f918994909f"
GENERIC_ENGINEERING_TERMINAL = {
    "state": "FAILED_ENGINEERING", "decision": None, "authorizes": "NONE",
}


def require_current_runnable_schema(
    manifest_schema: Any, experiment_spec: Any | None = None,
) -> None:
    if manifest_schema != 6:
        raise ReadyError(
            "historical manifest schema 4/5 is read-only; route-ready requires schema 6"
        )
    if experiment_spec is not None and (
        not isinstance(experiment_spec, dict)
        or experiment_spec.get("schema_version") != 3
    ):
        raise ReadyError(
            "historical experiment schema 1/2 is read-only; route-ready requires schema 3"
        )


def typed_scientific_contract(value: Any) -> bool:
    return isinstance(value, dict) and value.get("schema_version") in {2, 3}


def claim_published_name(owners: dict[str, str], name: str, owner: str) -> None:
    if name in owners:
        raise ReadyError(
            f"published filename collision: {name} is used by {owners[name]} and {owner}"
        )
    owners[name] = owner


def command(repo: Path, *args: str, input_text: str | None = None) -> str:
    completed = subprocess.run(
        [*args], cwd=repo, input=input_text, text=True, capture_output=True,
        timeout=60, check=False,
    )
    if completed.returncode:
        detail = (completed.stdout + completed.stderr).strip()[:4096]
        raise ReadyError(f"command failed: {' '.join(args)}: {detail}")
    return completed.stdout.strip()


def rules_require_review_facts(repo: Path, rules_commit: Any) -> bool:
    if not isinstance(rules_commit, str) or not ops.SHA40.fullmatch(rules_commit):
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
        raise ReadyError("cannot resolve review-facts rules ancestry")
    return completed.returncode == 0


def staged_snapshot(repo: Path) -> str:
    if subprocess.run([GIT, "diff", "--quiet"], cwd=repo).returncode:
        raise ReadyError("unstaged tracked changes exist; stage the complete bundle first")
    untracked = command(repo, GIT, "ls-files", "--others", "--exclude-standard")
    if untracked:
        raise ReadyError("untracked files exist; stage or remove them before route-ready")
    command(repo, GIT, "diff", "--cached", "--check")
    tree = command(repo, GIT, "write-tree")
    parent = command(repo, GIT, "rev-parse", "HEAD")
    environment = os.environ.copy()
    environment.setdefault("GIT_AUTHOR_NAME", "route-ready")
    environment.setdefault("GIT_AUTHOR_EMAIL", "route-ready@localhost")
    environment.setdefault("GIT_COMMITTER_NAME", "route-ready")
    environment.setdefault("GIT_COMMITTER_EMAIL", "route-ready@localhost")
    completed = subprocess.run(
        [GIT, "commit-tree", tree, "-p", parent], cwd=repo,
        input="route-ready staged snapshot\n", text=True, capture_output=True,
        timeout=30, env=environment, check=False,
    )
    if completed.returncode:
        raise ReadyError(f"cannot create staged snapshot: {completed.stderr.strip()}")
    return completed.stdout.strip()


def clean_head(repo: Path) -> str:
    if command(repo, GIT, "status", "--porcelain"):
        raise ReadyError("HEAD validation requires a clean worktree")
    return command(repo, GIT, "rev-parse", "HEAD")


def show(repo: Path, commit: str, relpath: str) -> bytes:
    completed = subprocess.run(
        [GIT, "show", f"{commit}:{relpath}"], cwd=repo,
        capture_output=True, timeout=30, check=False,
    )
    if completed.returncode:
        raise ReadyError(f"snapshot is missing {relpath}")
    return completed.stdout


def show_optional(repo: Path, commit: str, relpath: str) -> bytes | None:
    completed = subprocess.run(
        [GIT, "show", f"{commit}:{relpath}"], cwd=repo,
        capture_output=True, timeout=30, check=False,
    )
    return completed.stdout if completed.returncode == 0 else None


def check_python(raw: bytes, relpath: str) -> None:
    try:
        compile(raw.decode("utf-8"), relpath, "exec")
    except (SyntaxError, UnicodeDecodeError) as exc:
        raise ReadyError(f"Python syntax failed: {relpath}: {exc}") from exc


def check_bash(raw: bytes, relpath: str) -> None:
    completed = subprocess.run([BASH, "-n"], input=raw, capture_output=True, timeout=30)
    if completed.returncode:
        raise ReadyError(f"Bash syntax failed: {relpath}: {completed.stderr.decode(errors='replace')}")


ENGINEERING_EVIDENCE_FIELDS = {
    "mode", "device", "fixture", "production_path_exercised",
    "protected_data_touched", "scientific_output_created",
    "scientific_training_occurred",
}
ROUTE_CONTEXT_FIELDS = {
    "phase", "route_id", "operation_id", "run_id", "route_commit",
    "runner_sha256", "entrypoint_relpath", "remote_repo", "run_root",
    "output_path", "phase_output_path", "result_path", "status_path",
    "heartbeat_path", "device", "total_units", "evidence_role",
    "resume_policy", "protected_data_permissions", "assets",
    "engineering_contract",
}


def _inline_dict_fields(node: ast.AST, assignments: dict[str, list[ast.AST]]) -> set[str]:
    if isinstance(node, ast.Name):
        candidates = assignments.get(node.id, [])
        if len(candidates) != 1:
            raise ReadyError(
                "schema-2 engineering evidence must be one inline or single-assignment dict"
            )
        node = candidates[0]
    if not isinstance(node, ast.Dict) or any(key is None for key in node.keys):
        raise ReadyError(
            "schema-2 engineering evidence must be one inline or single-assignment dict"
        )
    fields = []
    for key in node.keys:
        if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
            raise ReadyError("schema-2 engineering evidence keys must be literal strings")
        fields.append(key.value)
    if len(fields) != len(set(fields)):
        raise ReadyError("schema-2 engineering evidence contains duplicate fields")
    return set(fields)


def _check_engineering_writer(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    require_cost_evidence: bool = False,
) -> None:
    assignments: dict[str, list[ast.AST]] = {}
    for node in ast.walk(function):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assignments.setdefault(target.id, []).append(node.value)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) \
                and node.value is not None:
            assignments.setdefault(node.target.id, []).append(node.value)
    writers = [
        node for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "write_contract_result"
    ]
    for writer in writers:
        if any(keyword.arg is None for keyword in writer.keywords):
            raise ReadyError("schema-2 write_contract_result cannot use **kwargs")
        engineering = [
            keyword.value for keyword in writer.keywords if keyword.arg == "engineering"
        ]
        if len(engineering) != 1:
            raise ReadyError("schema-2 write_contract_result requires engineering evidence")
        if isinstance(engineering[0], ast.Name):
            name = engineering[0].id
            for node in ast.walk(function):
                if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store) \
                        and isinstance(node.value, ast.Name) and node.value.id == name:
                    raise ReadyError("schema-2 engineering evidence cannot be mutated")
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                        and isinstance(node.func.value, ast.Name) \
                        and node.func.value.id == name:
                    raise ReadyError("schema-2 engineering evidence cannot be mutated")
        expected_fields = set(ENGINEERING_EVIDENCE_FIELDS)
        if require_cost_evidence:
            expected_fields.add("cost")
        fields = _inline_dict_fields(engineering[0], assignments)
        if fields != expected_fields:
            missing = sorted(expected_fields - fields)
            unknown = sorted(fields - expected_fields)
            raise ReadyError(
                f"schema-2 engineering evidence fields differ: missing={missing} unknown={unknown}"
            )


def _check_context_attributes(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> None:
    context_names = set()
    for node in ast.walk(function):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) \
                and isinstance(node.value.func, ast.Name) \
                and node.value.func.id == "load_context":
            context_names.update(
                target.id for target in node.targets if isinstance(target, ast.Name)
            )
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) \
                and isinstance(node.value, ast.Call) \
                and isinstance(node.value.func, ast.Name) \
                and node.value.func.id == "load_context":
            context_names.add(node.target.id)
    for node in ast.walk(function):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) \
                and node.value.id in context_names \
                and node.attr not in ROUTE_CONTEXT_FIELDS:
            raise ReadyError(f"unknown RouteContext field: {node.attr}")


def check_entrypoint(
    raw: bytes, relpath: str, *, require_engineering: bool = False,
    scientific_schema: int = 1, require_unit_ledger: bool = False,
    require_cost_evidence: bool = False,
) -> None:
    check_python(raw, relpath)
    tree = ast.parse(raw.decode("utf-8"), filename=relpath)
    strings = {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    functions = {
        node.name: node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    def call_name(node: ast.Call) -> str | None:
        return node.func.id if isinstance(node.func, ast.Name) else None

    run_writer = (
        "write_gate_result" if scientific_schema in {2, 3} else "write_run_result"
    )
    for phase, writer in (("contract", "write_contract_result"), ("run", run_writer)):
        function = functions.get(phase)
        if function is None:
            raise ReadyError(f"entrypoint must define {phase}(context_path)")
        arguments = function.args
        if len(arguments.args) != 1 or arguments.args[0].arg != "context_path" \
                or arguments.posonlyargs or arguments.kwonlyargs or arguments.vararg \
                or arguments.kwarg or arguments.defaults or arguments.kw_defaults:
            raise ReadyError(f"entrypoint {phase} must accept only context_path")
        calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)]
        if not any(call_name(node) == writer for node in calls):
            raise ReadyError(f"entrypoint {phase} must call {writer}")
        if not any(
            call_name(node) == "load_context" and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant) and node.args[1].value == phase
            for node in calls
        ):
            raise ReadyError(f"entrypoint {phase} must load its exact context phase")
        if phase == "contract" and require_engineering:
            _check_engineering_writer(
                function, require_cost_evidence=require_cost_evidence,
            )
        _check_context_attributes(function)
        if phase == "run" and scientific_schema in {2, 3} \
                and any(call_name(node) == "write_run_result" for node in calls):
            raise ReadyError("scientific schema 2/3 cannot call write_run_result")
        if phase == "run" and require_unit_ledger:
            observed = {call_name(node) for node in calls}
            if not {
                "load_completed_unit_ledger", "record_completed_unit",
            } <= observed:
                raise ReadyError(
                    "scientific schema-2/3 run must load and record the completed-unit ledger"
                )
    main_function = functions.get("main")
    main_calls = set() if main_function is None else {
        call_name(node) for node in ast.walk(main_function) if isinstance(node, ast.Call)
    }
    if main_function is None or not {"contract", "run"} <= main_calls \
            or "--context" not in strings:
        raise ReadyError("entrypoint main must dispatch contract/run with --context")


def independent_operation_errors(
    *, asset: dict[str, Any] | None,
    read_repo_file: Callable[[str], bytes], entrypoint_raw: bytes,
    entrypoint_relpath: str, require_engineering: bool,
    scientific_schema: int, require_unit_ledger: bool,
    require_cost_evidence: bool,
) -> list[str]:
    """Collect independent repository-asset and entrypoint authoring errors."""
    errors = []
    if asset is not None:
        errors.extend(repository_asset_identity_errors(asset, read_repo_file))
    try:
        check_entrypoint(
            entrypoint_raw, entrypoint_relpath,
            require_engineering=require_engineering,
            scientific_schema=scientific_schema,
            require_unit_ledger=require_unit_ledger,
            require_cost_evidence=require_cost_evidence,
        )
    except ReadyError as exc:
        errors.append(str(exc))
    return errors


def inspect_card(repo: Path, snapshot: str, relpath: str) -> tuple[list[str], str]:
    raw = show(repo, snapshot, relpath)
    with tempfile.TemporaryDirectory(prefix="route-ready-card-") as temporary:
        path = Path(temporary) / "card.md"
        path.write_bytes(raw)
        errors, digest = card_validator.validate(path, launch_ready=True)
    return errors, digest


def inspect_slim_card(repo: Path, snapshot: str, relpath: str, *, route_id: str,
                      contract_directory: str) -> tuple[list[str], str]:
    raw = show(repo, snapshot, relpath)
    errors = []
    if len(raw) > 8192:
        errors.append("canonical route note exceeds 8 KiB")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return ["canonical route note is not UTF-8"], __import__("hashlib").sha256(raw).hexdigest()
    if "\\`" in text:
        errors.append("escaped backticks are forbidden")
    if not __import__("re").search(r"(?m)^Status:\s*PLANNED\s*$", text):
        errors.append("canonical route note requires Status: PLANNED")
    if f"- Route id: {route_id}" not in text:
        errors.append("canonical route note route_id mismatch")
    if f"- Scientific contracts: {contract_directory}" not in text:
        errors.append("canonical route note contract-directory pointer mismatch")
    if not __import__("re").search(r"(?m)^## Scientific rationale\s*$", text):
        errors.append("canonical route note requires one Scientific rationale section")
    if any(token in text for token in ("<TBD>", "TODO", "【填写】")):
        errors.append("canonical route note contains an unresolved placeholder")
    return errors, __import__("hashlib").sha256(raw).hexdigest()


def authoring_errors(manifest: dict[str, Any], card_errors: list[str]) -> list[str]:
    """Collect frequent independent authoring mistakes before strict parsing."""
    errors = [f"route card: {error}" for error in card_errors]
    operations = manifest.get("operations", {})
    if not isinstance(operations, dict):
        return errors
    for operation_id, operation in operations.items():
        if not isinstance(operation, dict):
            continue
        profile = operation.get("monitor_profile")
        if not isinstance(profile, str) or profile not in ops.MONITOR_PROFILES:
            errors.append(
                f"{operation_id}: monitor_profile must be one of "
                f"{sorted(ops.MONITOR_PROFILES)}"
            )
        terminals = operation.get("allowed_terminal_tuples")
        if not isinstance(terminals, list) \
                or GENERIC_ENGINEERING_TERMINAL not in terminals:
            errors.append(
                f"{operation_id}: allowed_terminal_tuples must include "
                "FAILED_ENGINEERING / null / NONE"
            )
    return errors


def _private_receipt_path(repo: Path, route_id: str) -> Path:
    ops.require_token(route_id, "route_id")
    relative = f"convir/authoring-receipts/{route_id}.json"
    return Path(command(
        repo, GIT, "rev-parse", "--path-format=absolute", "--git-path", relative,
    ))


def _private_route_ready_path(repo: Path, route_id: str) -> Path:
    ops.require_token(route_id, "route_id")
    relative = f"convir/route-ready-receipts/{route_id}.json"
    return Path(command(
        repo, GIT, "rev-parse", "--path-format=absolute", "--git-path", relative,
    ))


def route_ready_request_identity(
    repo: Path, snapshot: str, current_main: str, selected: list[str] | None,
) -> tuple[dict[str, Any], str]:
    manifest_raw = show(repo, snapshot, ops.ROUTE_OPERATIONS_RELPATH)
    manifest = json.loads(manifest_raw)
    if not isinstance(manifest, dict):
        raise ReadyError("route manifest must be an object")
    route_id = ops.require_token(manifest.get("route_id"), "route_id")
    operations = manifest.get("operations")
    if not isinstance(operations, dict) or not operations:
        raise ReadyError("route manifest contains no operations")
    requested = list(dict.fromkeys(selected or list(operations)))
    identity = {
        "route_id": route_id,
        "branch": command(repo, GIT, "branch", "--show-current"),
        "head": command(repo, GIT, "rev-parse", "HEAD"),
        "tree": command(repo, GIT, "rev-parse", f"{snapshot}^{{tree}}"),
        "current_main": current_main,
        "requested_operations": requested,
        "manifest_sha256": __import__("hashlib").sha256(manifest_raw).hexdigest(),
    }
    digest = __import__("hashlib").sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return identity, digest


def load_cached_route_ready(
    repo: Path, identity: dict[str, Any], identity_sha: str,
) -> dict[str, Any] | None:
    path = _private_route_ready_path(repo, identity["route_id"])
    try:
        cached = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError):
        return None
    required = {
        "schema_version", "status", "identity", "identity_sha256", "report",
        "report_sha256", "allowed_next_action",
    }
    if not isinstance(cached, dict) or set(cached) != required \
            or cached["schema_version"] != 1 \
            or cached["status"] != "ROUTE_READY_COMPLETED" \
            or cached["identity"] != identity \
            or cached["identity_sha256"] != identity_sha \
            or cached["allowed_next_action"] != "commit_push_then_plan_once" \
            or not isinstance(cached["report"], dict) \
            or cached["report_sha256"] != spec_compiler.sha256(
                spec_compiler.json_bytes(cached["report"])
            ):
        return None
    report = cached["report"]
    if report.get("status") != "ROUTE_READY" \
            or report.get("route_id") != identity["route_id"] \
            or report.get("current_main") != identity["current_main"] \
            or report.get("requested_operations") != identity["requested_operations"]:
        return None
    return {**report, "cache_reused": True}


def write_route_ready_receipt_atomic(
    repo: Path, identity: dict[str, Any], identity_sha: str, report: dict[str, Any],
) -> None:
    path = _private_route_ready_path(repo, identity["route_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    receipt = {
        "schema_version": 1,
        "status": "ROUTE_READY_COMPLETED",
        "identity": identity,
        "identity_sha256": identity_sha,
        "report": report,
        "report_sha256": spec_compiler.sha256(spec_compiler.json_bytes(report)),
        "allowed_next_action": "commit_push_then_plan_once",
    }
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(spec_compiler.json_bytes(receipt))
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _safe_receipt_relpath(value: Any) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ReadyError("authoring receipt contains an invalid generated relpath")
    path = Path(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ReadyError("authoring receipt contains an unsafe generated relpath")
    return path.as_posix()


def _expected_compiler_paths(
    repo: Path, snapshot: str, manifest: dict[str, Any],
) -> set[str]:
    card_path = manifest.get("route_card_relpath")
    scientific_paths = manifest.get("scientific_contract_relpaths")
    if not isinstance(card_path, str) or not isinstance(scientific_paths, dict) \
            or set(scientific_paths) != set(manifest["operations"]) \
            or any(not isinstance(path, str) for path in scientific_paths.values()):
        raise ReadyError("manifest compiler path contract is invalid")
    expected = {
        ops.ROUTE_OPERATIONS_RELPATH,
        card_path,
        *scientific_paths.values(),
    }
    for operation_id in manifest["operations"]:
        spec_path = runtime_spec_relpath(operation_id)
        expected.add(spec_path)
        runtime = json.loads(show(repo, snapshot, spec_path))
        if not isinstance(runtime, dict) \
                or not isinstance(runtime.get("engineering_contract"), dict) \
                or not isinstance(runtime.get("precision_contract"), dict):
            raise ReadyError(f"runtime compiler path contract is invalid: {operation_id}")
        for relpath in (
            runtime.get("asset_manifest_relpath"),
            runtime["engineering_contract"].get("capability_profile_relpath"),
            runtime["precision_contract"].get("certificate_relpath"),
        ):
            if relpath is not None and not isinstance(relpath, str):
                raise ReadyError(f"runtime compiler relpath is invalid: {operation_id}")
            if isinstance(relpath, str):
                expected.add(relpath)
    return expected | set(RUNTIME_BUNDLE_RELPATHS)


def validate_authoring_receipt(
    repo: Path, snapshot: str, current_main: str, manifest: dict[str, Any],
) -> tuple[str, str, bool]:
    """Verify compiler identity without checking out archived main evidence."""
    route_id = manifest["route_id"]
    path = _private_receipt_path(repo, route_id)
    spec_path = manifest["experiment_spec_relpath"]
    spec_raw = show(repo, snapshot, spec_path)
    try:
        source_spec = json.loads(spec_raw)
        research_snapshot = spec_compiler.research_snapshot_commit(source_spec)
    except (json.JSONDecodeError, spec_compiler.ExperimentSpecError) as exc:
        raise ReadyError(f"experiment research snapshot is invalid: {exc}") from exc
    require_current_runnable_schema(manifest.get("schema_version"), source_spec)
    evidence_commit = research_snapshot or current_main
    if research_snapshot is not None:
        ancestry = subprocess.run(
            [GIT, "merge-base", "--is-ancestor", research_snapshot, current_main],
            cwd=repo, capture_output=True, timeout=30, check=False,
        )
        if ancestry.returncode != 0:
            raise ReadyError(
                "frozen research snapshot is not in current GitHub-main history"
            )
    recovered = False
    try:
        raw = path.read_bytes()
        receipt = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        program_path = manifest["program_contract_relpath"]
        program_raw = show(repo, snapshot, program_path)
        try:
            derived_bundle = spec_compiler.compile_bundle(
                spec_relpath=spec_path,
                spec_raw=spec_raw,
                program_raw=program_raw,
                evidence_exists=lambda relpath: show_optional(
                    repo, evidence_commit, relpath,
                ) is not None,
                authoritative_snapshot_commit=evidence_commit,
                read_authoritative_file=lambda relpath: show(
                    repo, evidence_commit, relpath,
                ),
            )
            bundle = {
                **derived_bundle,
                **spec_compiler.canonical_runtime_bundle(repo, current_main),
            }
        except (spec_compiler.ExperimentSpecError, KeyError, TypeError, ValueError) as rebuild_exc:
            raise ReadyError(
                "schema-6 authoring receipt is missing and deterministic recovery failed: "
                f"{rebuild_exc}"
            ) from rebuild_exc
        drift = spec_compiler.compare_bundle(
            bundle, lambda relpath: show(repo, snapshot, relpath),
        )
        if drift:
            raise ReadyError(
                "schema-6 authoring receipt is missing and generated bundle drifted: "
                + "; ".join(drift)
            ) from exc
        receipt = spec_compiler.build_authoring_receipt(
            spec_relpath=spec_path,
            spec_raw=spec_raw,
            program_relpath=program_path,
            program_raw=program_raw,
            bundle=bundle,
            authoritative_main_commit=evidence_commit,
        )
        spec_compiler.write_private_receipt_atomic(repo, route_id, receipt)
        raw = spec_compiler.json_bytes(receipt)
        recovered = True
    required = {
        "schema_version", "status", "route_id", "authoritative_main_commit",
        "experiment_spec", "program_contract", "generated_files", "bundle_sha256",
        "allowed_next_action",
    }
    if not isinstance(receipt, dict) or set(receipt) != required:
        raise ReadyError("authoring receipt field contract is invalid")
    if receipt["schema_version"] != spec_compiler.AUTHORING_RECEIPT_SCHEMA \
            or receipt["status"] != "EXPERIMENT_SPEC_BUNDLE_FINALIZED" \
            or receipt["route_id"] != route_id \
            or not isinstance(receipt["authoritative_main_commit"], str) \
            or not ops.SHA40.fullmatch(receipt["authoritative_main_commit"]) \
            or receipt["allowed_next_action"] != "stage_complete_bundle_then_route_ready_once":
        raise ReadyError("authoring receipt identity or phase is stale")
    if research_snapshot is not None \
            and receipt["authoritative_main_commit"] != research_snapshot:
        raise ReadyError("authoring receipt research snapshot identity mismatch")
    source_pairs = (
        ("experiment_spec", "experiment_spec_relpath", "experiment_spec_sha256"),
        ("program_contract", "program_contract_relpath", "program_contract_sha256"),
    )
    for receipt_key, path_key, digest_key in source_pairs:
        source = receipt[receipt_key]
        if not isinstance(source, dict) or set(source) != {"relpath", "sha256"} \
                or source["relpath"] != manifest[path_key] \
                or source["sha256"] != manifest[digest_key]:
            raise ReadyError(f"authoring receipt {receipt_key} identity mismatch")
        if __import__("hashlib").sha256(
            show(repo, snapshot, source["relpath"])
        ).hexdigest() != source["sha256"]:
            raise ReadyError(f"authoring receipt {receipt_key} bytes drifted after finalize")
    files = receipt["generated_files"]
    if not isinstance(files, list) or not files:
        raise ReadyError("authoring receipt generated_files must be non-empty")
    observed_paths = []
    for item in files:
        if not isinstance(item, dict) or set(item) != {"relpath", "sha256", "size_bytes"}:
            raise ReadyError("authoring receipt generated file record is invalid")
        relpath = _safe_receipt_relpath(item["relpath"])
        generated_raw = show(repo, snapshot, relpath)
        if item["size_bytes"] != len(generated_raw) or item["sha256"] != \
                __import__("hashlib").sha256(generated_raw).hexdigest():
            raise ReadyError(f"generated file drifted after finalize: {relpath}")
        observed_paths.append(relpath)
    if len(observed_paths) != len(set(observed_paths)):
        raise ReadyError("authoring receipt generated_files contains duplicates")
    expected_paths = _expected_compiler_paths(repo, snapshot, manifest)
    if set(observed_paths) != expected_paths:
        raise ReadyError(
            "authoring receipt generated path set mismatch: "
            f"missing={sorted(expected_paths - set(observed_paths))} "
            f"unexpected={sorted(set(observed_paths) - expected_paths)}"
        )
    if spec_compiler.sha256(spec_compiler.json_bytes(files)) != receipt["bundle_sha256"]:
        raise ReadyError("authoring receipt bundle_sha256 mismatch")
    return (
        receipt["bundle_sha256"],
        __import__("hashlib").sha256(raw).hexdigest(),
        recovered,
    )


def canonical_bundle_check(repo: Path, snapshot: str, current_main: str,
                           bootstrap: bool) -> tuple[dict[str, str], list[str]]:
    result = {}
    missing_from_main = []
    for relpath in RUNTIME_BUNDLE_RELPATHS:
        route_raw = show(repo, snapshot, relpath)
        if relpath.endswith(".py"):
            check_python(route_raw, relpath)
        elif relpath.endswith(".sh"):
            check_bash(route_raw, relpath)
        result[relpath] = __import__("hashlib").sha256(route_raw).hexdigest()
        main_raw = show_optional(repo, current_main, relpath)
        if main_raw is None:
            if not bootstrap:
                raise ReadyError(f"current main is missing runtime bundle file: {relpath}")
            missing_from_main.append(relpath)
        elif route_raw != main_raw:
            raise ReadyError(f"runtime bundle drift from current main: {relpath}")
    return result, missing_from_main


def validate_all(repo: Path, snapshot: str, current_main: str,
                 selected: list[str] | None, bootstrap: bool) -> dict[str, Any]:
    branch = command(repo, GIT, "branch", "--show-current")
    ops.require_branch(branch)
    raw = show(repo, snapshot, ops.ROUTE_OPERATIONS_RELPATH)
    if len(raw) > ops.MAX_MANIFEST_BYTES:
        raise ReadyError("route_operations.json exceeds MCP size limit")
    manifest = json.loads(raw)
    manifest_schema = manifest.get("schema_version") if isinstance(manifest, dict) else None
    require_current_runnable_schema(manifest_schema)
    operations = manifest.get("operations", {}) if isinstance(manifest, dict) else {}
    if not isinstance(operations, dict) or not operations:
        raise ReadyError("no operations selected")
    for operation_id, operation in operations.items():
        runtime_spec_relpath(operation_id)
        if not isinstance(operation, dict):
            raise ReadyError(f"operation is not an object: {operation_id}")
    requested = list(dict.fromkeys(selected or list(operations)))
    unknown = sorted(set(requested) - set(operations))
    if unknown:
        raise ReadyError(f"selected operations are absent from manifest: {unknown}")
    operation_ids = list(operations)
    if len({item.get("output_id") for item in operations.values()}) != len(operations):
        raise ReadyError("operation output_id values must be unique")
    if len({item.get("closeout_filename") for item in operations.values()}) != len(operations):
        raise ReadyError("operation closeout filenames must be unique")
    authoring_bundle_sha = None
    authoring_receipt_sha = None
    authoring_receipt_recovered = False
    if manifest_schema >= 6:
        (
            authoring_bundle_sha,
            authoring_receipt_sha,
            authoring_receipt_recovered,
        ) = validate_authoring_receipt(repo, snapshot, current_main, manifest)
    card_errors = []
    card_digest = None
    route_card_relpath = manifest.get("route_card_relpath")
    if isinstance(route_card_relpath, str):
        try:
            if manifest_schema >= 5:
                card_errors, card_digest = inspect_slim_card(
                    repo, snapshot, route_card_relpath,
                    route_id=manifest.get("route_id"),
                    contract_directory="experience_docx/scientific_contracts/",
                )
            else:
                card_errors, card_digest = inspect_card(
                    repo, snapshot, route_card_relpath,
                )
        except ReadyError as exc:
            card_errors = [str(exc)]
    else:
        card_errors = ["manifest route_card_relpath is missing or invalid"]
    common_errors = authoring_errors(manifest, card_errors)
    if common_errors:
        raise ReadyError("authoring errors: " + "; ".join(common_errors))
    if card_digest is None:
        raise ReadyError("route card inspection produced no digest")
    parsed_contexts = {
        requested[0]: ops.parse_manifest(
            manifest, branch, snapshot, current_main, str(repo), requested[0],
        )
    }
    bundle, bootstrap_missing = canonical_bundle_check(
        repo, snapshot, current_main, bootstrap,
    )
    reports = {}
    operation_validation_errors = []
    published_names: dict[str, str] = {}
    for operation_id, operation in operations.items():
        claim_published_name(
            published_names, operation["closeout_filename"], f"{operation_id} closeout",
        )
    for operation_id in operation_ids:
        try:
            context = parsed_contexts.get(operation_id) or ops.parse_manifest(
                manifest, branch, snapshot, current_main, str(repo), operation_id,
            )
            spec_path = runtime_spec_relpath(operation_id)
            spec = validate_runtime_spec(json.loads(show(repo, snapshot, spec_path)), manifest, operation_id)
            if manifest_schema >= 5 and spec["schema_version"] != 2:
                raise ReadyError(f"{operation_id}: canonical manifest requires runtime schema 2")
            entrypoint_raw = show(repo, snapshot, spec["entrypoint_relpath"])
            asset_digest = None
            asset = None
            if spec["asset_manifest_relpath"] is not None:
                asset = validate_asset_manifest(
                    json.loads(show(repo, snapshot, spec["asset_manifest_relpath"])), spec,
                )
                asset_digest = __import__("hashlib").sha256(
                    json.dumps(asset, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()
            capability_digest = None
            capability_reuse = None
            capability_path = spec["engineering_contract"]["capability_profile_relpath"]
            if capability_path is not None:
                capability_raw = show(repo, snapshot, capability_path)
                capability = validate_model_capability(
                    json.loads(capability_raw), spec, asset,
                )
                capability_digest = __import__("hashlib").sha256(
                    json.dumps(capability, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()
                if capability.get("schema_version") == 2:
                    registry_raw = show(
                        repo, current_main, capability_registry.REGISTRY_RELPATH,
                    )
                    try:
                        capability_reuse = capability_registry.lookup_lines(
                            registry_raw.decode("utf-8").splitlines(),
                            capability["reuse_identity"],
                            evidence_exists=lambda relpath: (
                                show_optional(repo, current_main, relpath) is not None
                            ),
                            read_evidence=lambda relpath: show(repo, current_main, relpath),
                        )
                    except (
                        UnicodeDecodeError, capability_registry.CapabilityRegistryError,
                    ) as exc:
                        raise ReadyError(f"capability registry is invalid: {exc}") from exc
            contract = None
            if manifest_schema >= 5:
                contract_path = manifest["scientific_contract_relpaths"][operation_id]
                contract = ops.validate_scientific_contract(
                    json.loads(show(repo, snapshot, contract_path)),
                    manifest["route_id"], operation_id, operation,
                )
            precision = None
            precision_digest = None
            precision_path = spec["precision_contract"]["certificate_relpath"]
            precision_feasible = None
            if precision_path is not None:
                precision_raw = show(repo, snapshot, precision_path)
                precision = validate_precision_certificate(
                    json.loads(precision_raw), spec, contract,
                )
                precision_feasible = precision["feasible"]
                precision_digest = __import__("hashlib").sha256(
                    json.dumps(precision, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()
            if contract is not None:
                ops.validate_contract_runtime_alignment(contract, spec, precision)
            independent_errors = independent_operation_errors(
                asset=asset,
                read_repo_file=lambda relpath: show(repo, snapshot, relpath),
                entrypoint_raw=entrypoint_raw,
                entrypoint_relpath=spec["entrypoint_relpath"],
                require_engineering=spec["schema_version"] >= 2,
                scientific_schema=(
                    contract["schema_version"] if contract is not None else 1
                ),
                require_unit_ledger=(
                    typed_scientific_contract(contract)
                    and spec["total_units"] > 0
                ),
                require_cost_evidence=(
                    spec["engineering_contract"].get("cost_contract") is not None
                ),
            )
            engineering = {"state": "FAILED_ENGINEERING", "decision": None, "authorizes": "NONE"}
            if engineering not in context["allowed_terminal_tuples"]:
                raise ReadyError(f"{operation_id} must allow the generic engineering closeout")
            destinations = {item["destination_filename"] for item in spec["evidence_files"]}
            if context["closeout_filename"] in destinations:
                raise ReadyError(f"{operation_id} closeout collides with evidence filename")
            closeout_suffix = "_closeout.json"
            if typed_scientific_contract(contract):
                if not context["closeout_filename"].endswith(closeout_suffix):
                    raise ReadyError(f"{operation_id} closeout cannot derive raw receipt")
                raw_receipt_name = (
                    context["closeout_filename"][:-len(closeout_suffix)]
                    + "_raw_artifact_receipt.json"
                )
                if raw_receipt_name in destinations:
                    raise ReadyError(
                        f"{operation_id} raw receipt collides with evidence filename"
                    )
                claim_published_name(
                    published_names, raw_receipt_name,
                    f"{operation_id} automatic raw receipt",
                )
            if typed_scientific_contract(contract) \
                    and rules_require_review_facts(repo, manifest.get("rules_commit")):
                facts_name = (
                    context["closeout_filename"][:-len(closeout_suffix)]
                    + "_review_facts.json"
                )
                if facts_name not in destinations:
                    raise ReadyError(
                        f"{operation_id} schema-2 route must declare {facts_name}"
                    )
            for destination in destinations:
                claim_published_name(
                    published_names, destination, f"{operation_id} evidence",
                )
            if operation_id in requested:
                evidence_prefix = f"experience_docx/experiment_logs/{manifest['route_id']}"
                automatic_names = {
                    context["closeout_filename"],
                    *(
                        [raw_receipt_name]
                        if typed_scientific_contract(contract)
                        else []
                    ),
                }
                for filename in destinations | automatic_names:
                    if show_optional(repo, snapshot, f"{evidence_prefix}/{filename}") is not None:
                        raise ReadyError(
                            f"{operation_id} would overwrite existing evidence: {filename}"
                        )
            if independent_errors:
                raise ReadyError("; ".join(independent_errors))
            reports[operation_id] = {
                "mode": context["mode"],
                "output_id": context["output_id"],
                "runner_sha256": context["runner_sha256"],
                "runtime_spec_digest": __import__("hashlib").sha256(
                    show(repo, snapshot, spec_path)
                ).hexdigest(),
                "entrypoint_sha256": __import__("hashlib").sha256(entrypoint_raw).hexdigest(),
                "asset_manifest_digest": asset_digest,
                "model_capability_digest": capability_digest,
                "capability_reuse": capability_reuse,
                "engineering_contract_mode": spec["engineering_contract"]["mode"],
                "precision_certificate_digest": precision_digest,
                "precision_mode": spec["precision_contract"]["mode"],
                "precision_feasible": precision_feasible,
                "canonical_scientific_contract": manifest_schema >= 5,
                "contract_phase_required": not (
                    isinstance(capability_reuse, dict)
                    and capability_reuse.get("engineering_reuse_authorized") is True
                ),
                "generic_failure_closeout": True,
            }
        except (ReadyError, ops.ToolError, ContractError, json.JSONDecodeError) as exc:
            operation_validation_errors.append(f"{operation_id}: {exc}")
    if operation_validation_errors:
        raise ReadyError(
            "operation validation errors: " + " | ".join(operation_validation_errors)
        )
    return {
        "schema_version": 1,
        "status": "ROUTE_READY",
        "snapshot_commit": snapshot,
        "current_main": current_main,
        "branch": branch,
        "route_id": manifest["route_id"],
        "requested_operations": requested,
        "route_card_sha256": card_digest,
        "authoring_bundle_sha256": authoring_bundle_sha,
        "authoring_receipt_sha256": authoring_receipt_sha,
        "authoring_receipt_recovered": authoring_receipt_recovered,
        "runtime_bundle_sha256": bundle,
        "bootstrap_missing_from_main": bootstrap_missing,
        "operations": reports,
        "checks": {
            "mcp_parser_shared": True,
            "compiler_authoring_receipt": manifest_schema < 6 or bool(authoring_receipt_sha),
            "staged_snapshot": True,
            "card_launch_ready": True,
            "runtime_bundle_canonical": not bootstrap_missing,
            "python_bash_syntax": True,
            "contract_and_run_interface": True,
            "asset_schema": True,
            "device_capability_contract": True,
            "precision_feasibility_contract": True,
            "output_and_evidence_names_unique": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--operation", action="append", dest="operations")
    parser.add_argument("--snapshot", choices=("staged", "HEAD"), default="staged")
    parser.add_argument(
        "--current-main", default=spec_compiler.DEFAULT_AUTHORITATIVE_MAIN,
        choices=(spec_compiler.DEFAULT_AUTHORITATIVE_MAIN,),
    )
    parser.add_argument("--bootstrap-runtime-bundle", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve()
    try:
        snapshot = staged_snapshot(repo) if args.snapshot == "staged" else clean_head(repo)
        current_main = spec_compiler.resolve_fresh_authoritative_main(
            repo, args.current_main,
        )
        identity, identity_sha = route_ready_request_identity(
            repo, snapshot, current_main, args.operations,
        )
        report = load_cached_route_ready(repo, identity, identity_sha)
        if report is None:
            report = validate_all(
                repo, snapshot, current_main, args.operations,
                args.bootstrap_runtime_bundle,
            )
            report["cache_reused"] = False
            write_route_ready_receipt_atomic(repo, identity, identity_sha, report)
    except (
        OSError, ReadyError, ops.ToolError, ContractError, json.JSONDecodeError,
        spec_compiler.ExperimentSpecError,
    ) as exc:
        print(f"ROUTE_READY_ERROR {exc}")
        raise SystemExit(1)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    print(
        f"ROUTE_READY_OK snapshot_commit={report['snapshot_commit']} "
        f"cache_reused={str(report['cache_reused']).lower()}"
    )


if __name__ == "__main__":
    main()
