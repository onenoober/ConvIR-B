#!/usr/bin/env python3
"""Deterministic content-addressed snapshots for current experiment source."""

from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import subprocess
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any


SNAPSHOT_SCHEMA_VERSION = 1
MAX_FILES = 20_000
MAX_FILE_BYTES = 8 * 1024 * 1024
MAX_TOTAL_BYTES = 128 * 1024 * 1024
MAX_GIT_OUTPUT_BYTES = 8 * 1024 * 1024

ALLOWED_SUFFIXES = {
    "", ".c", ".cc", ".cfg", ".conf", ".cpp", ".cu", ".cuh",
    ".h", ".hpp", ".ini", ".json", ".md", ".py", ".sh", ".toml",
    ".txt", ".yaml", ".yml",
}
FORBIDDEN_SUFFIXES = {
    ".7z", ".avi", ".bin", ".bmp", ".ckpt", ".gif", ".gz", ".jpeg",
    ".jpg", ".mp4", ".npy", ".npz", ".onnx", ".pdf", ".pickle", ".pkl",
    ".png", ".pt", ".pth", ".safetensors", ".tar", ".tif", ".tiff",
    ".webp", ".weights", ".zip",
}
EXCLUDED_PREFIXES = (
    ".git/",
    "experience_docx/experiment_logs/",
    "experience_docx/engineering_failures/",
)
EXCLUDED_PARTS = {
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".venv",
    "artifacts", "checkpoints", "data", "dataset", "datasets", "logs",
    "outputs", "results", "runs", "venv", "wandb", "weights",
}
MANIFEST_NAME = ".experiment-assistant-snapshot.json"


class SnapshotError(RuntimeError):
    pass


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _run_git(repo: Path, *args: str) -> bytes:
    try:
        result = subprocess.run(
            ["/usr/bin/git", "-C", str(repo), *args],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=60, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SnapshotError(f"Git command failed before snapshot: {type(exc).__name__}") from exc
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace")[-1024:].strip()
        raise SnapshotError(f"Git command failed before snapshot: {detail}")
    if len(result.stdout) > MAX_GIT_OUTPUT_BYTES:
        raise SnapshotError("Git snapshot path list exceeds its bounded output")
    return result.stdout


def _safe_relpath(raw: str) -> str:
    if not raw or "\x00" in raw or "\\" in raw:
        raise SnapshotError("snapshot path is not a safe repository-relative path")
    path = PurePosixPath(raw)
    if path.is_absolute() or ".." in path.parts or raw.startswith("./"):
        raise SnapshotError(f"snapshot path is unsafe: {raw}")
    return path.as_posix()


def _is_excluded(relpath: str) -> bool:
    if any(relpath == prefix[:-1] or relpath.startswith(prefix) for prefix in EXCLUDED_PREFIXES):
        return True
    parts = PurePosixPath(relpath).parts
    return any(part in EXCLUDED_PARTS for part in parts)


def _candidate_paths(repo: Path) -> tuple[list[str], str]:
    root = _run_git(repo, "rev-parse", "--show-toplevel").decode().strip()
    if Path(root).resolve() != repo.resolve():
        raise SnapshotError("snapshot repository path is not its Git root")
    head = _run_git(repo, "rev-parse", "HEAD").decode().strip()
    if len(head) != 40 or any(char not in "0123456789abcdef" for char in head):
        raise SnapshotError("snapshot repository HEAD is invalid")
    raw = _run_git(repo, "ls-files", "-co", "--exclude-standard", "-z")
    decoded = raw.decode("utf-8", errors="strict")
    paths = sorted(set(_safe_relpath(item) for item in decoded.split("\0") if item))
    if len(paths) > MAX_FILES:
        raise SnapshotError(f"snapshot contains more than {MAX_FILES} candidate paths")
    return paths, head


def _read_regular_file(path: Path, relpath: str, expected: os.stat_result) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise SnapshotError(f"snapshot path cannot be opened safely: {relpath}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise SnapshotError(f"snapshot path must be a regular file: {relpath}")
        if (before.st_dev, before.st_ino) != (expected.st_dev, expected.st_ino):
            raise SnapshotError(f"snapshot path changed before reading: {relpath}")
        chunks = []
        size = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, MAX_FILE_BYTES + 1 - size))
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
            if size > MAX_FILE_BYTES:
                raise SnapshotError(
                    f"snapshot source file exceeds {MAX_FILE_BYTES} bytes: {relpath}"
                )
        after = os.fstat(descriptor)
        if (
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            or size != after.st_size
        ):
            raise SnapshotError(f"snapshot path changed while reading: {relpath}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def collect_snapshot(repo: Path) -> dict[str, Any]:
    repo = Path(repo).resolve()
    if not repo.is_dir():
        raise SnapshotError("snapshot repository does not exist")
    paths, head = _candidate_paths(repo)
    records = []
    payloads: dict[str, bytes] = {}
    skipped = []
    total_bytes = 0
    for relpath in paths:
        if _is_excluded(relpath):
            skipped.append({"path": relpath, "reason": "excluded_path"})
            continue
        suffix = PurePosixPath(relpath).suffix.lower()
        if suffix in FORBIDDEN_SUFFIXES or suffix not in ALLOWED_SUFFIXES:
            skipped.append({"path": relpath, "reason": "non_source_suffix"})
            continue
        path = repo / relpath
        try:
            path_stat = path.lstat()
        except OSError as exc:
            raise SnapshotError(f"snapshot path cannot be read: {relpath}") from exc
        if stat.S_ISLNK(path_stat.st_mode) or not stat.S_ISREG(path_stat.st_mode):
            raise SnapshotError(f"snapshot path must be a regular non-symlink file: {relpath}")
        resolved = path.resolve()
        try:
            resolved.relative_to(repo)
        except ValueError as exc:
            raise SnapshotError(f"snapshot path escapes repository: {relpath}") from exc
        if path_stat.st_size > MAX_FILE_BYTES:
            raise SnapshotError(f"snapshot source file exceeds {MAX_FILE_BYTES} bytes: {relpath}")
        data = _read_regular_file(path, relpath, path_stat)
        total_bytes += len(data)
        if total_bytes > MAX_TOTAL_BYTES:
            raise SnapshotError(f"snapshot source exceeds {MAX_TOTAL_BYTES} total bytes")
        executable = bool(path_stat.st_mode & stat.S_IXUSR)
        payloads[relpath] = data
        records.append({
            "path": relpath,
            "bytes": len(data),
            "sha256": sha256_bytes(data),
            "executable": executable,
        })
    if not records:
        raise SnapshotError("snapshot contains no source files")
    manifest = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "base_commit": head,
        "files": records,
        "file_count": len(records),
        "total_bytes": total_bytes,
        "skipped": skipped,
    }
    return {"manifest": manifest, "payloads": payloads}


def build_snapshot(repo: Path, destination: Path) -> dict[str, Any]:
    collected = collect_snapshot(repo)
    manifest = collected["manifest"]
    payloads = collected["payloads"]
    manifest_bytes = canonical_json_bytes(manifest)
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent,
    )
    os.close(file_descriptor)
    temporary = Path(temporary_name)
    try:
        with tarfile.open(temporary, mode="w", format=tarfile.GNU_FORMAT) as archive:
            for record in manifest["files"]:
                data = payloads[record["path"]]
                info = tarfile.TarInfo(record["path"])
                info.size = len(data)
                info.mode = 0o755 if record["executable"] else 0o644
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                info.mtime = 0
                archive.addfile(info, io.BytesIO(data))
            info = tarfile.TarInfo(MANIFEST_NAME)
            info.size = len(manifest_bytes)
            info.mode = 0o644
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            archive.addfile(info, io.BytesIO(manifest_bytes))
        archive_bytes = temporary.read_bytes()
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "sha256": sha256_bytes(archive_bytes),
        "storage": "cloud_full",
        "base_commit": manifest["base_commit"],
        "diff_sha256": sha256_bytes(manifest_bytes),
        "manifest_sha256": sha256_bytes(manifest_bytes),
        "file_count": manifest["file_count"],
        "total_bytes": manifest["total_bytes"],
        "archive_bytes": len(archive_bytes),
        "path": str(destination),
    }
