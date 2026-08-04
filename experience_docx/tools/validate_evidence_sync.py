#!/usr/bin/env python3
"""One-pass validator for a staged compact experiment-evidence sync."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any


GIT = "/usr/bin/git"
ALLOWED_SUFFIXES = {".json", ".csv", ".md", ".txt"}
FORBIDDEN_NAME_TOKENS = {
    "cloud_only", "raw_prediction", "raw_feature", "raw_action", "per_sample",
}
MAX_FILES = 64
MAX_FILE_BYTES = 1024 * 1024
INDEX_PATH = "experience_docx/EXPERIMENT_INDEX.md"
CARD_PREFIX = "experience_docx/experiment_cards/"
FAMILY_PREFIX = "experience_docx/family_summaries/"
LOG_PREFIX = "experience_docx/experiment_logs/"
ENGINEERING_FAILURE_PREFIX = "experience_docx/engineering_failures/"


class EvidenceSyncError(RuntimeError):
    pass


def git(repo: Path, *args: str, input_text: str | None = None) -> str:
    completed = subprocess.run(
        [GIT, *args], cwd=repo, input=input_text, text=True,
        capture_output=True, timeout=60, check=False,
    )
    if completed.returncode:
        detail = (completed.stdout + completed.stderr).strip()[:4096]
        raise EvidenceSyncError(f"git {' '.join(args)} failed: {detail}")
    return completed.stdout.strip()


def staged_snapshot(repo: Path, base_ref: str) -> tuple[str, str]:
    if subprocess.run([GIT, "diff", "--quiet"], cwd=repo).returncode:
        raise EvidenceSyncError("unstaged tracked changes exist")
    if git(repo, "ls-files", "--others", "--exclude-standard"):
        raise EvidenceSyncError("untracked files exist")
    if not git(repo, "diff", "--cached", "--name-only"):
        raise EvidenceSyncError("no staged evidence changes")
    git(repo, "diff", "--cached", "--check")
    head = git(repo, "rev-parse", "HEAD")
    base = git(repo, "rev-parse", base_ref)
    if head != base:
        raise EvidenceSyncError(
            f"evidence sync HEAD must equal base ref: HEAD={head} base={base}"
        )
    tree = git(repo, "write-tree")
    environment = os.environ.copy()
    environment.setdefault("GIT_AUTHOR_NAME", "evidence-sync")
    environment.setdefault("GIT_AUTHOR_EMAIL", "evidence-sync@localhost")
    environment.setdefault("GIT_COMMITTER_NAME", "evidence-sync")
    environment.setdefault("GIT_COMMITTER_EMAIL", "evidence-sync@localhost")
    completed = subprocess.run(
        [GIT, "commit-tree", tree, "-p", head], cwd=repo,
        input="validated evidence sync snapshot\n", text=True,
        capture_output=True, timeout=30, env=environment, check=False,
    )
    if completed.returncode:
        raise EvidenceSyncError(completed.stderr.strip())
    return completed.stdout.strip(), base


def staged_paths(repo: Path) -> list[str]:
    forbidden = subprocess.run(
        [GIT, "diff", "--cached", "--name-only", "-z", "--diff-filter=CDRTUXB"],
        cwd=repo, capture_output=True, timeout=30, check=False,
    )
    if forbidden.returncode or forbidden.stdout:
        raise EvidenceSyncError("deleted, copied, renamed, or unmerged files are forbidden")
    raw = subprocess.run(
        [GIT, "diff", "--cached", "--name-only", "-z", "--diff-filter=AM"], cwd=repo,
        capture_output=True, timeout=30, check=False,
    )
    if raw.returncode:
        raise EvidenceSyncError("cannot read staged paths")
    paths = [item.decode("utf-8", errors="strict")
             for item in raw.stdout.split(b"\0") if item]
    if not 1 <= len(paths) <= MAX_FILES:
        raise EvidenceSyncError(f"staged evidence must contain 1-{MAX_FILES} files")
    return paths


def show(repo: Path, snapshot: str, relpath: str) -> bytes:
    completed = subprocess.run(
        [GIT, "show", f"{snapshot}:{relpath}"], cwd=repo,
        capture_output=True, timeout=30, check=False,
    )
    if completed.returncode:
        raise EvidenceSyncError(f"snapshot is missing {relpath}")
    return completed.stdout


def _engineering_path_run_id(relpath: str, route_id: str) -> str:
    prefix = f"{ENGINEERING_FAILURE_PREFIX}{route_id}/"
    if not relpath.startswith(prefix):
        raise EvidenceSyncError(
            "engineering archives must stay below "
            f"{ENGINEERING_FAILURE_PREFIX}{{route_id}}/{{run_id}}/"
        )
    remainder = PurePosixPath(relpath[len(prefix):])
    if len(remainder.parts) != 2:
        raise EvidenceSyncError("engineering archive files must be top-level in one run directory")
    run_id, filename = remainder.parts
    if not 1 <= len(run_id) <= 128 or any(
        character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-"
        for character in run_id
    ):
        raise EvidenceSyncError("engineering archive run_id must be a safe token")
    if PurePosixPath(filename).suffix.lower() not in ALLOWED_SUFFIXES:
        raise EvidenceSyncError(f"forbidden evidence suffix: {relpath}")
    lowered = filename.lower()
    if any(token in lowered for token in FORBIDDEN_NAME_TOKENS):
        raise EvidenceSyncError(f"cloud-only/raw evidence is forbidden: {relpath}")
    return run_id


def classify_path(relpath: str, route_id: str, *, engineering_archive: bool) -> str:
    path = PurePosixPath(relpath)
    if path.is_absolute() or ".." in path.parts or "\\" in relpath:
        raise EvidenceSyncError(f"unsafe staged path: {relpath}")
    if engineering_archive:
        _engineering_path_run_id(relpath, route_id)
        return "engineering_evidence"
    route_prefix = f"{LOG_PREFIX}{route_id}/"
    if relpath.startswith(route_prefix):
        filename = relpath[len(route_prefix):]
        if not filename or PurePosixPath(filename).name != filename:
            raise EvidenceSyncError("route evidence files must be top-level")
        if PurePosixPath(filename).suffix.lower() not in ALLOWED_SUFFIXES:
            raise EvidenceSyncError(f"forbidden evidence suffix: {relpath}")
        lowered = filename.lower()
        if any(token in lowered for token in FORBIDDEN_NAME_TOKENS):
            raise EvidenceSyncError(f"cloud-only/raw evidence is forbidden: {relpath}")
        return "route_evidence"
    if relpath == INDEX_PATH:
        return "project_memory"
    if relpath.startswith(CARD_PREFIX) and path.suffix == ".md" \
            and path.parent == PurePosixPath(CARD_PREFIX.rstrip("/")):
        return "route_card"
    if relpath.startswith(FAMILY_PREFIX) and path.suffix == ".md" \
            and path.parent == PurePosixPath(FAMILY_PREFIX.rstrip("/")):
        return "project_memory"
    raise EvidenceSyncError(f"path is outside the compact evidence allowlist: {relpath}")


def inspect_json(raw: bytes, relpath: str, route_id: str,
                 engineering_archive: bool, run_id: str | None) -> tuple[Any, bool]:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceSyncError(f"invalid JSON: {relpath}: {exc}") from exc
    failed_engineering = False
    if isinstance(value, dict):
        recorded_route = value.get("route_id")
        if recorded_route is not None and recorded_route != route_id:
            raise EvidenceSyncError(f"route_id mismatch in {relpath}")
        recorded_run = value.get("run_id")
        if recorded_run is not None and run_id is not None and recorded_run != run_id:
            raise EvidenceSyncError(f"run_id mismatch in {relpath}")
        failed_engineering = value.get("state") == "FAILED_ENGINEERING"
        if failed_engineering and not engineering_archive:
            raise EvidenceSyncError(
                "FAILED_ENGINEERING evidence requires explicit --engineering-archive"
            )
    return value, failed_engineering


def inspect_csv(raw: bytes, relpath: str) -> int:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EvidenceSyncError(f"CSV is not UTF-8: {relpath}") from exc
    reader = csv.reader(io.StringIO(text, newline=""))
    rows = list(reader)
    if not rows or not rows[0] or any(not item.strip() for item in rows[0]):
        raise EvidenceSyncError(f"CSV header is missing or empty: {relpath}")
    if len(set(rows[0])) != len(rows[0]) or len(rows[0]) > 256:
        raise EvidenceSyncError(f"CSV header is duplicate or too wide: {relpath}")
    width = len(rows[0])
    if any(len(row) != width for row in rows[1:]):
        raise EvidenceSyncError(f"CSV rows have inconsistent widths: {relpath}")
    return len(rows)


def validate_staged(repo: Path, route_id: str, base_ref: str, *,
                    allow_project_memory_update: bool,
                    engineering_archive: bool) -> dict[str, Any]:
    if not 1 <= len(route_id) <= 128 or any(
        character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-"
        for character in route_id
    ):
        raise EvidenceSyncError("route_id must be a safe token")
    if engineering_archive and allow_project_memory_update:
        raise EvidenceSyncError(
            "engineering archive cannot update central index or family memory"
        )
    snapshot, base = staged_snapshot(repo, base_ref)
    paths = staged_paths(repo)
    reports = []
    route_evidence = 0
    cards = 0
    closeouts = 0
    failed_engineering = 0
    engineering_run_ids = set()
    for relpath in paths:
        role = classify_path(
            relpath, route_id, engineering_archive=engineering_archive,
        )
        run_id = None
        if role == "engineering_evidence":
            run_id = _engineering_path_run_id(relpath, route_id)
            engineering_run_ids.add(run_id)
        if role == "project_memory" and not allow_project_memory_update:
            raise EvidenceSyncError(
                "index/family updates require --allow-project-memory-update"
            )
        raw = show(repo, snapshot, relpath)
        if not 1 <= len(raw) <= MAX_FILE_BYTES:
            raise EvidenceSyncError(f"file size is outside limits: {relpath}")
        if b"\0" in raw:
            raise EvidenceSyncError(f"binary content is forbidden: {relpath}")
        suffix = PurePosixPath(relpath).suffix.lower()
        parsed_rows = None
        parsed_json = None
        if suffix == ".json":
            parsed_json, is_failed = inspect_json(
                raw, relpath, route_id, engineering_archive, run_id,
            )
            failed_engineering += int(is_failed)
        elif suffix == ".csv":
            parsed_rows = inspect_csv(raw, relpath)
        else:
            try:
                raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise EvidenceSyncError(f"text evidence is not UTF-8: {relpath}") from exc
        if role == "route_card":
            cards += 1
            if f"- Route id: {route_id}" not in raw.decode("utf-8"):
                raise EvidenceSyncError(f"route card identity mismatch: {relpath}")
        if role in {"route_evidence", "engineering_evidence"}:
            route_evidence += 1
            if relpath.endswith("_closeout.json"):
                if not isinstance(parsed_json, dict) or parsed_json.get("route_id") != route_id \
                        or not all(key in parsed_json for key in ("state", "decision", "authorizes")):
                    raise EvidenceSyncError(f"closeout identity/terminal tuple is incomplete: {relpath}")
                if engineering_archive and (
                    parsed_json.get("run_id") != run_id
                    or parsed_json.get("state") != "FAILED_ENGINEERING"
                    or parsed_json.get("decision") is not None
                    or parsed_json.get("authorizes") != "NONE"
                ):
                    raise EvidenceSyncError(
                        f"engineering closeout identity/terminal tuple is invalid: {relpath}"
                    )
                closeouts += 1
        reports.append({
            "path": relpath, "role": role, "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(), "csv_rows": parsed_rows,
        })
    if route_evidence < 1 or closeouts < 1:
        raise EvidenceSyncError("sync requires route evidence and one *_closeout.json")
    if cards > 1:
        raise EvidenceSyncError("sync may update at most one route card")
    if engineering_archive and failed_engineering < 1:
        raise EvidenceSyncError("--engineering-archive requires FAILED_ENGINEERING evidence")
    if engineering_archive and len(engineering_run_ids) != 1:
        raise EvidenceSyncError("engineering archive must contain exactly one run_id")
    if engineering_archive and closeouts != 1:
        raise EvidenceSyncError("engineering archive must contain exactly one closeout")
    return {
        "schema_version": 1,
        "status": "EVIDENCE_SYNC_READY",
        "route_id": route_id,
        "snapshot_commit": snapshot,
        "base_commit": base,
        "allow_project_memory_update": allow_project_memory_update,
        "engineering_archive": engineering_archive,
        "files": reports,
        "checks": {
            "staged_snapshot": True,
            "compact_text_only": True,
            "structured_files_parsed": True,
            "route_identity_bound": True,
            "engineering_failure_policy": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--route-id", required=True)
    parser.add_argument("--base-ref", default="refs/remotes/github/main")
    parser.add_argument("--allow-project-memory-update", action="store_true")
    parser.add_argument("--engineering-archive", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    try:
        report = validate_staged(
            args.repo.resolve(), args.route_id, args.base_ref,
            allow_project_memory_update=args.allow_project_memory_update,
            engineering_archive=args.engineering_archive,
        )
    except EvidenceSyncError as exc:
        print(f"EVIDENCE_SYNC_ERROR {exc}")
        raise SystemExit(1)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
    print(json.dumps(report, sort_keys=True))
    print(f"EVIDENCE_SYNC_OK snapshot_commit={report['snapshot_commit']}")


if __name__ == "__main__":
    main()
