#!/usr/bin/env python3
"""Create or audit one clean, exact-base research worktree.

This tool has no delete, commit, push, or remote-command surface. It may create
only a previously absent worktree and branch under the configured workspace.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


class WorkspaceError(RuntimeError):
    pass


GIT = "/usr/bin/git"
DEFAULT_WORKSPACE_ROOT = Path("/home/ubuntu/workspace")
SHA40 = re.compile(r"^[0-9a-f]{40}$")
SAFE_BRANCH = re.compile(r"^codex/[A-Za-z0-9][A-Za-z0-9._/-]{0,191}$")
SAFE_REMOTE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def run(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        [GIT, "-C", str(repo), *args], text=True, capture_output=True,
        timeout=60, check=False,
    )
    if completed.returncode:
        detail = (completed.stderr or completed.stdout).strip()[:2048]
        raise WorkspaceError(f"git {' '.join(args)} failed: {detail}")
    return completed.stdout.strip()


def optional_run(repo: Path, *args: str) -> str | None:
    completed = subprocess.run(
        [GIT, "-C", str(repo), *args], text=True, capture_output=True,
        timeout=60, check=False,
    )
    if completed.returncode == 1:
        return None
    if completed.returncode:
        detail = (completed.stderr or completed.stdout).strip()[:2048]
        raise WorkspaceError(f"git {' '.join(args)} failed: {detail}")
    return completed.stdout.strip()


def safe_existing_repo(value: str, workspace_root: Path) -> Path:
    path = Path(value).resolve()
    try:
        path.relative_to(workspace_root.resolve())
    except ValueError as exc:
        raise WorkspaceError("source repo must stay under workspace_root") from exc
    if not path.is_dir():
        raise WorkspaceError("source repo is unavailable")
    run(path, "rev-parse", "--git-dir")
    return path


def safe_destination(value: str, workspace_root: Path) -> Path:
    path = Path(value).resolve()
    try:
        relative = path.relative_to(workspace_root.resolve())
    except ValueError as exc:
        raise WorkspaceError("destination must stay under workspace_root") from exc
    if len(relative.parts) != 1:
        raise WorkspaceError("destination must be a direct workspace child")
    return path


def require_branch(value: str) -> str:
    if not SAFE_BRANCH.fullmatch(value) or ".." in value or "//" in value:
        raise WorkspaceError("branch must be a safe codex/<name> branch")
    return value


def require_remote(value: str, name: str) -> str:
    if not SAFE_REMOTE.fullmatch(value):
        raise WorkspaceError(f"{name} must be a safe remote name")
    return value


def remote_names(repo: Path) -> set[str]:
    return set(run(repo, "remote").splitlines())


def audit_worktree(destination: Path, *, base_commit: str, branch: str,
                   project_remote: str) -> dict:
    head = run(destination, "rev-parse", "HEAD")
    actual_branch = run(destination, "branch", "--show-current")
    changed = run(destination, "status", "--porcelain").splitlines()
    push_remote = optional_run(destination, "config", "--get", f"branch.{branch}.pushRemote")
    role = optional_run(destination, "config", "--get", f"remote.{project_remote}.convirRole")
    mismatches = []
    if head != base_commit:
        mismatches.append("base_commit")
    if actual_branch != branch:
        mismatches.append("branch")
    if changed:
        mismatches.append("worktree_clean")
    if push_remote != project_remote or role != "project_evidence":
        mismatches.append("project_remote_role")
    return {
        "schema_version": 1,
        "status": "WORKTREE_READY" if not mismatches else "WORKTREE_IDENTITY_MISMATCH",
        "destination": str(destination),
        "base_commit": base_commit,
        "head": head,
        "branch": actual_branch,
        "clean": not changed,
        "project_remote": project_remote,
        "push_remote": push_remote,
        "project_remote_role": role,
        "mismatches": mismatches,
        "write_scope": "new_worktree_only",
        "git_push_performed": False,
    }


def create_worktree(*, source_repo: str, destination: str, base_commit: str,
                    branch: str, project_remote: str,
                    upstream_remote: str | None = None,
                    workspace_root: Path = DEFAULT_WORKSPACE_ROOT) -> dict:
    source = safe_existing_repo(source_repo, workspace_root)
    target = safe_destination(destination, workspace_root)
    if target.exists():
        raise WorkspaceError("destination already exists; refusing overwrite")
    if not SHA40.fullmatch(base_commit):
        raise WorkspaceError("base_commit must be 40 lowercase hex")
    branch = require_branch(branch)
    project_remote = require_remote(project_remote, "project_remote")
    upstream_remote = None if upstream_remote is None else require_remote(
        upstream_remote, "upstream_remote",
    )
    remotes = remote_names(source)
    if project_remote not in remotes:
        raise WorkspaceError("project_remote is absent")
    if upstream_remote is not None and upstream_remote not in remotes:
        raise WorkspaceError("upstream_remote is absent")
    remote_main = run(source, "rev-parse", "--verify", f"refs/remotes/{project_remote}/main")
    if remote_main != base_commit:
        raise WorkspaceError("base_commit does not match project remote main")
    project_url = run(source, "remote", "get-url", project_remote)
    upstream_url = None if upstream_remote is None else run(
        source, "remote", "get-url", upstream_remote,
    )
    remote_main_line = subprocess.run(
        [GIT, "ls-remote", project_url, "refs/heads/main"],
        text=True, capture_output=True, timeout=60, check=False,
    )
    fields = remote_main_line.stdout.split()
    if remote_main_line.returncode or len(fields) != 2 or fields[0] != base_commit \
            or fields[1] != "refs/heads/main":
        raise WorkspaceError("base_commit does not match live project remote main")
    branch_probe = subprocess.run(
        [GIT, "ls-remote", project_url, f"refs/heads/{branch}"],
        text=True, capture_output=True, timeout=60, check=False,
    )
    if branch_probe.returncode:
        raise WorkspaceError("project branch existence check failed")
    if branch_probe.stdout.strip():
        raise WorkspaceError("branch already exists on project remote")
    clone = subprocess.run(
        [
            GIT, "clone", "--quiet", "--no-checkout", "--dissociate",
            "--reference-if-able", str(source), "--origin", project_remote,
            project_url, str(target),
        ],
        text=True, capture_output=True, timeout=120, check=False,
    )
    if clone.returncode:
        raise WorkspaceError(f"isolated checkout creation failed: {clone.stderr.strip()[:2048]}")
    try:
        run(target, "checkout", "--quiet", "-b", branch, base_commit)
        run(target, "config", f"branch.{branch}.pushRemote", project_remote)
        run(target, "config", f"remote.{project_remote}.convirRole", "project_evidence")
        if upstream_remote is not None:
            run(target, "remote", "add", upstream_remote, upstream_url)
            run(target, "config", f"remote.{upstream_remote}.convirRole", "upstream_source")
        report = audit_worktree(
            target, base_commit=base_commit, branch=branch,
            project_remote=project_remote,
        )
        if report["mismatches"]:
            raise WorkspaceError(f"created worktree failed audit: {report['mismatches']}")
        report.update({
            "source_repo": str(source),
            "source_worktree_clean_required": False,
            "upstream_remote": upstream_remote,
        })
        return report
    except BaseException:
        # Do not auto-delete: a partially created checkout is reported for
        # explicit inspection rather than hidden by destructive recovery.
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-repo", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--base-commit", required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--project-remote", default="github")
    parser.add_argument("--upstream-remote")
    parser.add_argument("--workspace-root", type=Path, default=DEFAULT_WORKSPACE_ROOT)
    args = parser.parse_args()
    try:
        report = create_worktree(
            source_repo=args.source_repo, destination=args.destination,
            base_commit=args.base_commit, branch=args.branch,
            project_remote=args.project_remote,
            upstream_remote=args.upstream_remote,
            workspace_root=args.workspace_root,
        )
    except WorkspaceError as exc:
        print(json.dumps({
            "schema_version": 1, "status": "WORKTREE_REJECTED",
            "error": str(exc), "git_push_performed": False,
        }, sort_keys=True))
        raise SystemExit(2)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
