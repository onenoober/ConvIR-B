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
from typing import Any

import convir_ops_mcp as ops
import validate_experiment_card as card_validator
from route_runtime_contract import (
    GENERIC_RUNNER_RELPATH,
    RUNTIME_BUNDLE_RELPATHS,
    ContractError,
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
GENERIC_ENGINEERING_TERMINAL = {
    "state": "FAILED_ENGINEERING", "decision": None, "authorizes": "NONE",
}


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


def check_entrypoint(raw: bytes, relpath: str) -> None:
    check_python(raw, relpath)
    tree = ast.parse(raw.decode("utf-8"), filename=relpath)
    strings = {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    functions = {
        node.name: node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    def call_name(node: ast.Call) -> str | None:
        return node.func.id if isinstance(node.func, ast.Name) else None

    for phase, writer in (("contract", "write_contract_result"),
                          ("run", "write_run_result")):
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
    main_function = functions.get("main")
    main_calls = set() if main_function is None else {
        call_name(node) for node in ast.walk(main_function) if isinstance(node, ast.Call)
    }
    if main_function is None or not {"contract", "run"} <= main_calls \
            or "--context" not in strings:
        raise ReadyError("entrypoint main must dispatch contract/run with --context")


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
    card_errors = []
    card_digest = None
    route_card_relpath = manifest.get("route_card_relpath")
    if isinstance(route_card_relpath, str):
        try:
            if manifest_schema == 5:
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
            if manifest_schema == 5 and spec["schema_version"] != 2:
                raise ReadyError(f"{operation_id}: canonical manifest requires runtime schema 2")
            entrypoint_raw = show(repo, snapshot, spec["entrypoint_relpath"])
            check_entrypoint(entrypoint_raw, spec["entrypoint_relpath"])
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
            capability_path = spec["engineering_contract"]["capability_profile_relpath"]
            if capability_path is not None:
                capability_raw = show(repo, snapshot, capability_path)
                capability = validate_model_capability(
                    json.loads(capability_raw), spec, asset,
                )
                capability_digest = __import__("hashlib").sha256(
                    json.dumps(capability, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()
            precision_digest = None
            precision_path = spec["precision_contract"]["certificate_relpath"]
            precision_feasible = None
            if precision_path is not None:
                precision_raw = show(repo, snapshot, precision_path)
                precision = validate_precision_certificate(json.loads(precision_raw), spec)
                precision_feasible = precision["feasible"]
                precision_digest = __import__("hashlib").sha256(
                    json.dumps(precision, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()
            if manifest_schema == 5:
                contract_path = manifest["scientific_contract_relpaths"][operation_id]
                contract = ops.validate_scientific_contract(
                    json.loads(show(repo, snapshot, contract_path)),
                    manifest["route_id"], operation_id, operation,
                )
                ops.validate_contract_runtime_alignment(contract, spec, precision)
            engineering = {"state": "FAILED_ENGINEERING", "decision": None, "authorizes": "NONE"}
            if engineering not in context["allowed_terminal_tuples"]:
                raise ReadyError(f"{operation_id} must allow the generic engineering closeout")
            destinations = {item["destination_filename"] for item in spec["evidence_files"]}
            if context["closeout_filename"] in destinations:
                raise ReadyError(f"{operation_id} closeout collides with evidence filename")
            for destination in destinations:
                claim_published_name(
                    published_names, destination, f"{operation_id} evidence",
                )
            if operation_id in requested:
                evidence_prefix = f"experience_docx/experiment_logs/{manifest['route_id']}"
                for filename in destinations | {context["closeout_filename"]}:
                    if show_optional(repo, snapshot, f"{evidence_prefix}/{filename}") is not None:
                        raise ReadyError(
                            f"{operation_id} would overwrite existing evidence: {filename}"
                        )
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
                "engineering_contract_mode": spec["engineering_contract"]["mode"],
                "precision_certificate_digest": precision_digest,
                "precision_mode": spec["precision_contract"]["mode"],
                "precision_feasible": precision_feasible,
                "canonical_scientific_contract": manifest_schema == 5,
                "contract_phase_required": True,
                "generic_failure_closeout": True,
            }
        except (ops.ToolError, ContractError, json.JSONDecodeError) as exc:
            raise ReadyError(f"{operation_id}: {exc}") from exc
    return {
        "schema_version": 1,
        "status": "ROUTE_READY",
        "snapshot_commit": snapshot,
        "current_main": current_main,
        "branch": branch,
        "route_id": manifest["route_id"],
        "requested_operations": requested,
        "route_card_sha256": card_digest,
        "runtime_bundle_sha256": bundle,
        "bootstrap_missing_from_main": bootstrap_missing,
        "operations": reports,
        "checks": {
            "mcp_parser_shared": True,
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
    parser.add_argument("--current-main", default="refs/remotes/github/main")
    parser.add_argument("--bootstrap-runtime-bundle", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve()
    try:
        snapshot = staged_snapshot(repo) if args.snapshot == "staged" else clean_head(repo)
        current_main = command(repo, GIT, "rev-parse", args.current_main)
        report = validate_all(
            repo, snapshot, current_main, args.operations,
            args.bootstrap_runtime_bundle,
        )
    except (ReadyError, ops.ToolError, ContractError, json.JSONDecodeError) as exc:
        print(f"ROUTE_READY_ERROR {exc}")
        raise SystemExit(1)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    print(f"ROUTE_READY_OK snapshot_commit={snapshot}")


if __name__ == "__main__":
    main()
