#!/usr/bin/env python3
"""Classify a staged engineering repair without reopening scientific scope."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import convir_ops_mcp as ops
import experiment_spec_compiler as compiler
from route_runtime_contract import runtime_spec_relpath


GIT = "/usr/bin/git"
REPAIR_NOTE_PREFIXES = (
    "- Same-contract engineering repair:",
    "- Same-contract workload repair:",
)


class RepairError(RuntimeError):
    pass


def git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        [GIT, *args], cwd=repo, text=True, capture_output=True,
        timeout=60, check=False,
    )
    if completed.returncode:
        detail = (completed.stdout + completed.stderr).strip()[:4096]
        raise RepairError(f"git {' '.join(args)} failed: {detail}")
    return completed.stdout.strip()


def git_porcelain(repo: Path) -> str:
    completed = subprocess.run(
        [GIT, "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=repo, text=True, capture_output=True, timeout=60, check=False,
    )
    if completed.returncode:
        detail = (completed.stdout + completed.stderr).strip()[:4096]
        raise RepairError(f"git status --porcelain=v1 failed: {detail}")
    return completed.stdout.rstrip("\n")


def staged_snapshot(repo: Path) -> str:
    if subprocess.run([GIT, "diff", "--quiet"], cwd=repo).returncode:
        raise RepairError("unstaged tracked changes exist")
    if git(repo, "ls-files", "--others", "--exclude-standard"):
        raise RepairError("untracked files exist")
    git(repo, "diff", "--cached", "--check")
    tree = git(repo, "write-tree")
    parent = git(repo, "rev-parse", "HEAD")
    completed = subprocess.run(
        [GIT, "commit-tree", tree, "-p", parent], cwd=repo,
        input="engineering repair staged snapshot\n", text=True,
        capture_output=True, timeout=30, check=False,
        env={**__import__("os").environ,
             "GIT_AUTHOR_NAME": "repair-gate", "GIT_AUTHOR_EMAIL": "repair-gate@localhost",
             "GIT_COMMITTER_NAME": "repair-gate", "GIT_COMMITTER_EMAIL": "repair-gate@localhost"},
    )
    if completed.returncode:
        raise RepairError(completed.stderr.strip())
    return completed.stdout.strip()


def worktree_candidate_snapshot(repo: Path, candidate_paths: list[str]) -> str:
    """Create an ephemeral candidate commit without touching the real Git index."""
    if not candidate_paths:
        raise RepairError("worktree-candidate requires at least one --candidate-path")
    normalized = []
    for value in candidate_paths:
        path = Path(value)
        if path.is_absolute() or ".." in path.parts or value in {"", "."}:
            raise RepairError(f"candidate path is not repository-relative: {value}")
        normalized.append(path.as_posix())
    if len(normalized) != len(set(normalized)):
        raise RepairError("candidate paths must be unique")
    if subprocess.run([GIT, "diff", "--cached", "--quiet"], cwd=repo).returncode:
        raise RepairError("real Git index must be clean before worktree-candidate classification")
    original_head = git(repo, "rev-parse", "HEAD")
    original_index_tree = git(repo, "write-tree")
    status = git_porcelain(repo)
    observed = set()
    for line in status.splitlines():
        if len(line) < 4 or " -> " in line[3:]:
            raise RepairError("renamed or unparseable worktree candidate path")
        observed.add(line[3:])
    requested = set(normalized)
    if observed != requested:
        raise RepairError(
            f"worktree changes must exactly match --candidate-path values: "
            f"unlisted={sorted(observed - requested)} absent={sorted(requested - observed)}"
        )
    with tempfile.TemporaryDirectory(prefix="engineering-repair-index-") as temporary:
        index = str(Path(temporary) / "index")
        environment = {**os.environ, "GIT_INDEX_FILE": index}

        def isolated(*args: str, input_text: str | None = None) -> str:
            completed = subprocess.run(
                [GIT, *args], cwd=repo, env=environment, input=input_text, text=True,
                capture_output=True, timeout=60, check=False,
            )
            if completed.returncode:
                detail = (completed.stdout + completed.stderr).strip()[:4096]
                raise RepairError(f"temporary-index git {' '.join(args)} failed: {detail}")
            return completed.stdout.strip()

        isolated("read-tree", "HEAD")
        isolated("add", "-A", "--", *normalized)
        isolated("diff", "--cached", "--check")
        tree = isolated("write-tree")
        parent = git(repo, "rev-parse", "HEAD")
        environment.update({
            "GIT_AUTHOR_NAME": "repair-gate",
            "GIT_AUTHOR_EMAIL": "repair-gate@localhost",
            "GIT_COMMITTER_NAME": "repair-gate",
            "GIT_COMMITTER_EMAIL": "repair-gate@localhost",
        })
        snapshot = isolated(
            "commit-tree", tree, "-p", parent,
            input_text="engineering repair worktree candidate\n",
        )
    if subprocess.run([GIT, "diff", "--cached", "--quiet"], cwd=repo).returncode:
        raise RepairError("worktree-candidate classification changed the real Git index")
    if git(repo, "rev-parse", "HEAD") != original_head \
            or git(repo, "write-tree") != original_index_tree \
            or git_porcelain(repo) != status:
        raise RepairError("worktree-candidate classification changed repository state")
    return snapshot


def show(repo: Path, commit: str, relpath: str) -> bytes:
    completed = subprocess.run(
        [GIT, "show", f"{commit}:{relpath}"], cwd=repo,
        capture_output=True, timeout=30, check=False,
    )
    if completed.returncode:
        raise RepairError(f"snapshot is missing {relpath}")
    return completed.stdout


def load_json(repo: Path, commit: str, relpath: str) -> dict[str, Any]:
    try:
        value = json.loads(show(repo, commit, relpath))
    except json.JSONDecodeError as exc:
        raise RepairError(f"invalid JSON: {relpath}: {exc}") from exc
    if not isinstance(value, dict):
        raise RepairError(f"JSON must be an object: {relpath}")
    return value


def show_optional(repo: Path, commit: str, relpath: str) -> bytes | None:
    completed = subprocess.run(
        [GIT, "show", f"{commit}:{relpath}"], cwd=repo,
        capture_output=True, timeout=30, check=False,
    )
    return completed.stdout if completed.returncode == 0 else None


def call_leaf(node: ast.AST) -> str:
    if isinstance(node, ast.Attribute):
        return node.attr
    return node.id if isinstance(node, ast.Name) else ast.dump(node, include_attributes=False)


class SafeRepairAst(ast.NodeTransformer):
    def visit_Import(self, node: ast.Import) -> None:
        return None

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        return None

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST | None:
        if node.name == "contract":
            return None
        return self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST | None:
        if node.name == "contract":
            return None
        return self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> ast.AST:
        node = self.generic_visit(node)
        leaf = call_leaf(node.func)
        node.func = ast.Name(id=leaf, ctx=ast.Load())
        return node


def normalized_entrypoint(raw: bytes) -> str:
    try:
        tree = ast.parse(raw.decode("utf-8"))
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise RepairError(f"entrypoint is not valid UTF-8 Python: {exc}") from exc
    normalized = SafeRepairAst().visit(tree)
    ast.fix_missing_locations(normalized)
    return ast.dump(normalized, include_attributes=False)


def normalize_card(raw: bytes, old_output: str, new_output: str) -> str:
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise RepairError("route card is not UTF-8") from exc
    kept: list[str] = []
    skip_one_adjacent_blank = False
    for index, line in enumerate(lines):
        if line.startswith(REPAIR_NOTE_PREFIXES):
            # A standard repair note is commonly inserted as its own Markdown
            # paragraph. Removing only the note would then leave two adjacent
            # blank lines where the original card had one. Absorb exactly the
            # following blank in that local pattern; do not collapse blank
            # lines elsewhere in the scientific card.
            skip_one_adjacent_blank = bool(
                kept
                and not kept[-1].strip()
                and index + 1 < len(lines)
                and not lines[index + 1].strip()
            )
            continue
        if skip_one_adjacent_blank:
            skip_one_adjacent_blank = False
            if not line.strip():
                continue
        kept.append(line.replace(new_output, old_output))
    return "\n".join(kept).strip()


def validate_asset_repair(base: dict[str, Any], candidate: dict[str, Any]) -> list[str]:
    if {key: base.get(key) for key in ("schema_version", "route_id", "operation_id")} != \
            {key: candidate.get(key) for key in ("schema_version", "route_id", "operation_id")}:
        raise RepairError("asset manifest identity changed")
    old = {item.get("id"): item for item in base.get("assets", []) if isinstance(item, dict)}
    new = {item.get("id"): item for item in candidate.get("assets", []) if isinstance(item, dict)}
    if not old or set(old) != set(new):
        raise RepairError("asset set changed")
    changed = []
    for asset_id in sorted(old):
        before, after = old[asset_id], new[asset_id]
        if before == after:
            continue
        if before.get("kind") == "directory":
            raise RepairError(f"directory/data asset changed: {asset_id}")
        before_without_path = {key: value for key, value in before.items() if key != "path"}
        after_without_path = {key: value for key, value in after.items() if key != "path"}
        if before_without_path != after_without_path or before.get("path") == after.get("path"):
            raise RepairError(f"asset identity or contract changed: {asset_id}")
        identity = before.get("sha256") if before.get("kind") == "file" else before.get("commit")
        if not isinstance(identity, str):
            raise RepairError(f"path repair lacks immutable identity: {asset_id}")
        changed.append(asset_id)
    return changed


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _entrypoint_asset(operation: dict[str, Any], entrypoint: str) -> dict[str, Any]:
    matches = [
        item for item in operation.get("assets", [])
        if isinstance(item, dict) and item.get("kind") == "file"
        and item.get("path") == f"{{REMOTE_REPO}}/{entrypoint}"
    ]
    if len(matches) != 1:
        raise RepairError("schema-6 source must bind exactly one entrypoint file asset")
    return matches[0]


def _normalize_schema6_source(source: dict[str, Any], operation_id: str,
                              old_output: str, entrypoint_sha: str) -> dict[str, Any]:
    value = json.loads(json.dumps(source))
    operation = value["operations"][operation_id]
    operation["operation"]["output_id"] = old_output
    entrypoint = operation["runtime"]["entrypoint_relpath"]
    asset = _entrypoint_asset(operation, entrypoint)
    asset["sha256"] = entrypoint_sha
    bound = [
        item for item in operation["capability"]["bound_assets"]
        if item.get("id") == asset["id"]
    ]
    if len(bound) != 1:
        raise RepairError("schema-6 capability must bind the entrypoint asset exactly once")
    bound[0]["identity"] = entrypoint_sha
    reuse_identity = operation["capability"].get("reuse_identity")
    if reuse_identity is not None:
        reuse_identity["code_path_sha256"] = entrypoint_sha
    return value


def validate_schema6_repair(repo: Path, base: str, snapshot: str, operation_id: str,
                            old_manifest: dict[str, Any], new_manifest: dict[str, Any],
                            old_output: str, new_output: str) -> dict[str, Any]:
    stable_manifest_fields = {
        "schema_version", "route_id", "rules_commit", "route_card_relpath",
        "scientific_contract_relpaths", "program_contract_relpath",
        "program_contract_sha256", "experiment_spec_relpath",
    }
    if {key: old_manifest.get(key) for key in stable_manifest_fields} != \
            {key: new_manifest.get(key) for key in stable_manifest_fields}:
        raise RepairError("schema-6 route/program/scientific identity changed")
    for path in old_manifest["scientific_contract_relpaths"].values():
        if show(repo, base, path) != show(repo, snapshot, path):
            raise RepairError("canonical scientific contract changed")
    program_path = old_manifest["program_contract_relpath"]
    program_raw = show(repo, base, program_path)
    if program_raw != show(repo, snapshot, program_path):
        raise RepairError("research program contract changed")
    spec_path = old_manifest["experiment_spec_relpath"]
    old_spec_raw = show(repo, base, spec_path)
    new_spec_raw = show(repo, snapshot, spec_path)
    old_source = json.loads(old_spec_raw)
    new_source = json.loads(new_spec_raw)
    old_entrypoint = old_source["operations"][operation_id]["runtime"]["entrypoint_relpath"]
    new_entrypoint = new_source["operations"][operation_id]["runtime"]["entrypoint_relpath"]
    if old_entrypoint != new_entrypoint:
        raise RepairError("schema-6 entrypoint path changed")
    old_entrypoint_raw = show(repo, base, old_entrypoint)
    new_entrypoint_raw = show(repo, snapshot, new_entrypoint)
    if old_entrypoint_raw == new_entrypoint_raw:
        raise RepairError("schema-6 repair did not change the entrypoint binding")
    if normalized_entrypoint(old_entrypoint_raw) != normalized_entrypoint(new_entrypoint_raw):
        raise RepairError("entrypoint algorithm/control-flow/constants changed")
    old_sha, new_sha = _sha256(old_entrypoint_raw), _sha256(new_entrypoint_raw)
    if _normalize_schema6_source(old_source, operation_id, old_output, old_sha) != \
            _normalize_schema6_source(new_source, operation_id, old_output, old_sha):
        raise RepairError(
            "experiment spec changed beyond output and synchronized entrypoint identity"
        )
    if new_manifest.get("experiment_spec_sha256") != _sha256(new_spec_raw):
        raise RepairError("schema-6 experiment spec identity is not synchronized")
    bundle = compiler.compile_bundle(
        spec_relpath=spec_path, spec_raw=new_spec_raw, program_raw=program_raw,
        evidence_exists=lambda path: show_optional(repo, snapshot, path) is not None,
    )
    mismatches = compiler.compare_bundle(
        bundle, lambda path: show(repo, snapshot, path),
    )
    if mismatches:
        raise RepairError("schema-6 generated bundle is not deterministic: " + "; ".join(mismatches))
    changed_paths = set(filter(None, git(repo, "diff", "--name-only", base, snapshot).splitlines()))
    allowed = set(bundle) | {spec_path, old_entrypoint}
    unexpected = sorted(changed_paths - allowed)
    if unexpected:
        raise RepairError(f"unexpected schema-6 repair paths: {unexpected}")
    return {
        "asset_path_repairs": [],
        "entrypoint_symbol_binding_only": True,
        "schema6_compiler_regeneration_verified": True,
    }


def validate(repo: Path, base: str, snapshot: str, operation_id: str) -> dict[str, Any]:
    base = git(repo, "rev-parse", f"{base}^{{commit}}")
    manifest_path = ops.ROUTE_OPERATIONS_RELPATH
    old_manifest = load_json(repo, base, manifest_path)
    new_manifest = load_json(repo, snapshot, manifest_path)
    if set(old_manifest) != set(new_manifest) or old_manifest.get("route_id") != new_manifest.get("route_id") \
            or old_manifest.get("rules_commit") != new_manifest.get("rules_commit") \
            or old_manifest.get("route_card_relpath") != new_manifest.get("route_card_relpath"):
        raise RepairError("route/rules/card identity changed")
    if set(old_manifest.get("operations", {})) != set(new_manifest.get("operations", {})) \
            or operation_id not in old_manifest.get("operations", {}):
        raise RepairError("operation set changed")
    old_operation = old_manifest["operations"][operation_id]
    new_operation = new_manifest["operations"][operation_id]
    old_output, new_output = old_operation.get("output_id"), new_operation.get("output_id")
    if not isinstance(old_output, str) or not isinstance(new_output, str) or old_output == new_output:
        raise RepairError("repair requires one new output identity")
    if {key: value for key, value in old_operation.items() if key != "output_id"} != \
            {key: value for key, value in new_operation.items() if key != "output_id"}:
        raise RepairError("operation contract changed beyond output identity")

    if old_manifest.get("schema_version") == 6 and new_manifest.get("schema_version") == 6:
        schema6 = validate_schema6_repair(
            repo, base, snapshot, operation_id, old_manifest, new_manifest,
            old_output, new_output,
        )
        return {
            "schema_version": 1,
            "status": "AUTO_REPAIR_ELIGIBLE",
            "base_commit": base,
            "snapshot_commit": snapshot,
            "operation_id": operation_id,
            "old_output_id": old_output,
            "new_output_id": new_output,
            **schema6,
            "scientific_contract_unchanged": True,
            "sensitive_review_required": False,
        }

    spec_path = runtime_spec_relpath(operation_id)
    if show(repo, base, spec_path) != show(repo, snapshot, spec_path):
        raise RepairError("runtime/data/permission/seed/budget contract changed")
    spec = load_json(repo, snapshot, spec_path)
    entrypoint = spec["entrypoint_relpath"]
    asset_path = spec.get("asset_manifest_relpath")
    asset_changes = []
    if asset_path:
        asset_changes = validate_asset_repair(
            load_json(repo, base, asset_path), load_json(repo, snapshot, asset_path),
        )

    old_entrypoint = show(repo, base, entrypoint)
    new_entrypoint = show(repo, snapshot, entrypoint)
    symbol_binding_only = old_entrypoint != new_entrypoint
    if symbol_binding_only and normalized_entrypoint(old_entrypoint) != normalized_entrypoint(new_entrypoint):
        raise RepairError("entrypoint algorithm/control-flow/constants changed")

    card_path = new_manifest["route_card_relpath"]
    if normalize_card(show(repo, base, card_path), old_output, old_output) != \
            normalize_card(show(repo, snapshot, card_path), old_output, new_output):
        raise RepairError("route card changed beyond output identity and repair note")

    allowed = {manifest_path, spec_path, entrypoint, card_path}
    if asset_path:
        allowed.add(asset_path)
    changed_paths = set(filter(None, git(repo, "diff", "--name-only", base, snapshot).splitlines()))
    unexpected = sorted(changed_paths - allowed)
    if unexpected:
        raise RepairError(f"unexpected repair paths: {unexpected}")
    return {
        "schema_version": 1,
        "status": "AUTO_REPAIR_ELIGIBLE",
        "base_commit": base,
        "snapshot_commit": snapshot,
        "operation_id": operation_id,
        "old_output_id": old_output,
        "new_output_id": new_output,
        "asset_path_repairs": asset_changes,
        "entrypoint_symbol_binding_only": symbol_binding_only,
        "scientific_contract_unchanged": True,
        "sensitive_review_required": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--base", required=True)
    parser.add_argument("--operation", required=True)
    parser.add_argument(
        "--snapshot", choices=("staged", "worktree-candidate", "HEAD"),
        default="staged",
    )
    parser.add_argument("--candidate-path", action="append", default=[])
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve()
    try:
        if args.snapshot == "staged":
            snapshot = staged_snapshot(repo)
        elif args.snapshot == "worktree-candidate":
            snapshot = worktree_candidate_snapshot(repo, args.candidate_path)
        else:
            snapshot = git(repo, "rev-parse", "HEAD")
        report = validate(repo, args.base, snapshot, args.operation)
    except (RepairError, KeyError, TypeError, ValueError) as exc:
        report = {
            "schema_version": 1, "status": "SENSITIVE_REPAIR_REVIEW_REQUIRED",
            "reason": str(exc), "sensitive_review_required": True,
        }
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, sort_keys=True))
        print(f"SENSITIVE_REPAIR_REVIEW_REQUIRED reason={exc}")
        raise SystemExit(2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    print(f"AUTO_REPAIR_ELIGIBLE snapshot_commit={snapshot}")


if __name__ == "__main__":
    main()
