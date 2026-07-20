#!/usr/bin/env python3
"""Classify a staged engineering repair without reopening scientific scope."""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
from pathlib import Path
from typing import Any

import convir_ops_mcp as ops
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
    parser.add_argument("--snapshot", choices=("staged", "HEAD"), default="staged")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve()
    try:
        snapshot = staged_snapshot(repo) if args.snapshot == "staged" else git(repo, "rev-parse", "HEAD")
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
