#!/usr/bin/env python3
"""GitHub text-record archive and read store for completed experiments."""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterator

from experiment_assistant_contract import (
    RESULT_STATES,
    canonical_json_bytes,
    canonical_sha256,
    validate_archive_record,
)


PRODUCTION_REMOTE_URL = "git@github.com:onenoober/ConvIR-B.git"
PRODUCTION_BRANCH = "main"
RECORD_ROOT = Path("experience_docx/experiment_records")
INDEX_PATH = Path("experience_docx/EXPERIMENT_RECORD_INDEX.jsonl")
MAX_RECORD_BYTES = 1024 * 1024
MAX_RECORDS = 10_000
MAX_GIT_OUTPUT_BYTES = 64 * 1024


class ArchiveError(RuntimeError):
    """A compact GitHub record cannot be archived or read reliably."""


def _run_git(repo: Path | None, *args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    command = ["/usr/bin/git"]
    if repo is not None:
        command.extend(["-C", str(repo)])
    command.extend(args)
    try:
        result = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ArchiveError(f"Git archive command failed: {type(exc).__name__}") from exc
    if len(result.stdout) > MAX_GIT_OUTPUT_BYTES or len(result.stderr) > MAX_GIT_OUTPUT_BYTES:
        raise ArchiveError("Git archive command exceeded its bounded output")
    if check and result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace")[-2048:].strip()
        raise ArchiveError(f"Git archive command failed: {detail}")
    return result


def _write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_bounded_regular(path: Path, maximum: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ArchiveError(f"experiment record cannot be opened safely: {path.name}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > maximum:
            raise ArchiveError(f"experiment record is not a bounded regular file: {path.name}")
        chunks = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(256 * 1024, maximum + 1 - total))
            if not chunk:
                break
            total += len(chunk)
            if total > maximum:
                raise ArchiveError(f"experiment record exceeds its size limit: {path.name}")
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            or total != after.st_size
        ):
            raise ArchiveError(f"experiment record changed while reading: {path.name}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _read_record(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(_read_bounded_regular(path, MAX_RECORD_BYTES).decode("utf-8"))
        return validate_archive_record(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        if isinstance(exc, ArchiveError):
            raise
        raise ArchiveError(f"experiment record cannot be validated: {path.name}") from exc


def _index_entry(record: dict[str, Any]) -> dict[str, Any]:
    metric = record["result"]["primary_metric"]
    experiment_id = record["experiment_id"]
    return {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "objective": record["contract"]["objective"],
        "recorded_at": record["recorded_at"],
        "state": record["terminal"]["state"],
        "primary_metric": {"id": metric["id"], "value": metric["value"]},
        "record_path": (RECORD_ROOT / f"{experiment_id}.json").as_posix(),
        "record_sha256": canonical_sha256(record),
    }


def _validated_index(repo: Path) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    index_path = repo / INDEX_PATH
    if not index_path.exists():
        if (repo / RECORD_ROOT).exists() and any((repo / RECORD_ROOT).iterdir()):
            raise ArchiveError("experiment record index is missing")
        return []
    try:
        data = index_path.read_bytes()
    except OSError as exc:
        raise ArchiveError("experiment record index cannot be read") from exc
    if len(data) > MAX_RECORDS * 8192:
        raise ArchiveError("experiment record index exceeds its bounded size")
    entries = []
    seen = set()
    for line_number, line in enumerate(data.splitlines(), 1):
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ArchiveError(f"experiment record index line {line_number} is invalid") from exc
        required = {
            "schema_version", "experiment_id", "objective", "recorded_at", "state",
            "primary_metric", "record_path", "record_sha256",
        }
        if not isinstance(entry, dict) or set(entry) != required \
                or entry.get("schema_version") != 1:
            raise ArchiveError(f"experiment record index line {line_number} has invalid fields")
        experiment_id = entry.get("experiment_id")
        expected_path = (RECORD_ROOT / f"{experiment_id}.json").as_posix()
        if experiment_id in seen or entry.get("record_path") != expected_path \
                or entry.get("state") not in RESULT_STATES:
            raise ArchiveError(f"experiment record index line {line_number} has invalid identity")
        record = _read_record(repo / expected_path)
        if _index_entry(record) != entry:
            raise ArchiveError(f"experiment record index line {line_number} conflicts with record")
        seen.add(experiment_id)
        entries.append((entry, record))
    if len(entries) > MAX_RECORDS:
        raise ArchiveError("experiment record index exceeds its record limit")
    if [item[0]["experiment_id"] for item in entries] != sorted(seen):
        raise ArchiveError("experiment record index must be sorted by experiment_id")
    record_root = repo / RECORD_ROOT
    paths = [] if not record_root.exists() else sorted(record_root.iterdir())
    if any(
        path.is_symlink() or not path.is_file() or path.suffix != ".json"
        for path in paths
    ):
        raise ArchiveError("experiment record directory contains an unsafe entry")
    if {path.stem for path in paths} != seen:
        raise ArchiveError("experiment record directory and index are incomplete")
    return entries


class GitArchiveStore:
    def __init__(
        self,
        remote_url: str,
        temporary_root: Path,
        *,
        allow_test_remote: bool = False,
    ):
        if not isinstance(remote_url, str) or not remote_url or len(remote_url) > 4096:
            raise ArchiveError("archive remote must be bounded text")
        if allow_test_remote:
            local = Path(remote_url)
            if not local.is_absolute() or not local.exists():
                raise ArchiveError("test archive remote must be an existing absolute path")
        elif remote_url != PRODUCTION_REMOTE_URL:
            raise ArchiveError("archive remote is not the fixed production GitHub repository")
        self.remote_url = remote_url
        self.allow_test_remote = allow_test_remote
        self.temporary_root = Path(temporary_root).resolve()
        self.temporary_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self.temporary_root.is_symlink() or not self.temporary_root.is_dir():
            raise ArchiveError("archive temporary root must be a non-symlink directory")

    @contextlib.contextmanager
    def _clone(self) -> Iterator[Path]:
        directory = Path(tempfile.mkdtemp(prefix="github-record-", dir=self.temporary_root))
        repo = directory / "repo"
        try:
            _run_git(
                None, "clone", "--quiet", "--single-branch", "--branch",
                PRODUCTION_BRANCH, "--no-tags", self.remote_url, str(repo),
            )
            yield repo
        finally:
            resolved = directory.resolve()
            try:
                resolved.relative_to(self.temporary_root)
            except ValueError as exc:
                raise ArchiveError("refusing unsafe archive temporary cleanup") from exc
            shutil.rmtree(resolved)

    @staticmethod
    def _rebuild_index(repo: Path) -> None:
        record_root = repo / RECORD_ROOT
        record_root.mkdir(parents=True, exist_ok=True)
        paths = sorted(record_root.iterdir(), key=lambda path: path.name)
        if len(paths) > MAX_RECORDS:
            raise ArchiveError("experiment record directory exceeds its record limit")
        records = []
        seen = set()
        for path in paths:
            if path.is_symlink() or not path.is_file() or path.suffix != ".json":
                raise ArchiveError("experiment record directory contains an unsafe entry")
            record = _read_record(path)
            if record["experiment_id"] != path.stem or path.stem in seen:
                raise ArchiveError("experiment record path identity is invalid")
            seen.add(path.stem)
            records.append(record)
        data = b"".join(canonical_json_bytes(_index_entry(record)) for record in records)
        _write_bytes(repo / INDEX_PATH, data)

    def archive(self, record: dict[str, Any], expected_sha256: str) -> dict[str, Any]:
        normalized = validate_archive_record(record)
        record_sha = canonical_sha256(normalized)
        if record_sha != expected_sha256:
            raise ArchiveError("archive record SHA-256 does not match its cloud receipt")
        experiment_id = normalized["experiment_id"]
        record_bytes = canonical_json_bytes(normalized)
        last_error = None
        for attempt in range(2):
            with self._clone() as repo:
                target = repo / RECORD_ROOT / f"{experiment_id}.json"
                if target.exists():
                    existing = _read_record(target)
                    if canonical_json_bytes(existing) != record_bytes:
                        raise ArchiveError(
                            "GitHub already contains a different record for this experiment_id"
                        )
                else:
                    _write_bytes(target, record_bytes)
                self._rebuild_index(repo)
                _run_git(repo, "add", "--", target.relative_to(repo).as_posix(), INDEX_PATH.as_posix())
                _run_git(repo, "diff", "--cached", "--check")
                changed = _run_git(repo, "diff", "--cached", "--quiet", check=False)
                if changed.returncode == 0:
                    _validated_index(repo)
                    commit = _run_git(repo, "rev-parse", "HEAD").stdout.decode().strip()
                    return {
                        "state": "GITHUB_ARCHIVED",
                        "commit": commit,
                        "record_path": target.relative_to(repo).as_posix(),
                        "record_sha256": record_sha,
                        "idempotent": True,
                    }
                if changed.returncode != 1:
                    raise ArchiveError("Git staged-change inspection failed")
                _run_git(repo, "config", "user.name", "ConvIR Experiment Assistant")
                _run_git(repo, "config", "user.email", "experiment-assistant@invalid")
                _run_git(repo, "commit", "--quiet", "-m", f"experiment-record-{experiment_id}")
                commit = _run_git(repo, "rev-parse", "HEAD").stdout.decode().strip()
                pushed = _run_git(
                    repo, "push", "--quiet", "origin", f"HEAD:refs/heads/{PRODUCTION_BRANCH}",
                    check=False,
                )
                if pushed.returncode == 0:
                    _validated_index(repo)
                    return {
                        "state": "GITHUB_ARCHIVED",
                        "commit": commit,
                        "record_path": target.relative_to(repo).as_posix(),
                        "record_sha256": record_sha,
                        "idempotent": False,
                    }
                last_error = pushed.stderr.decode("utf-8", errors="replace")[-2048:].strip()
                if attempt == 0:
                    continue
        raise ArchiveError(f"GitHub archive push failed after one fresh retry: {last_error}")

    def records(self) -> list[dict[str, Any]]:
        with self._clone() as repo:
            return [record for _, record in _validated_index(repo)]

    def get(self, experiment_id: str) -> dict[str, Any] | None:
        records = self.records()
        matches = [record for record in records if record["experiment_id"] == experiment_id]
        if len(matches) > 1:
            raise ArchiveError("GitHub contains duplicate experiment records")
        return matches[0] if matches else None
