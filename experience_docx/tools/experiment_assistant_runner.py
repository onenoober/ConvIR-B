#!/usr/bin/env python3
"""Persistent, overwrite-safe lifecycle backend for the experiment assistant."""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
import signal
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

from experiment_assistant_contract import (
    RESULT_STATES,
    ContractError,
    assess_launch,
    authorize_attempt,
    build_archive_record,
    canonical_json_bytes,
    classify_contract_revision,
    validate_contract,
)
from experiment_assistant_snapshot import (
    MANIFEST_NAME,
    MAX_FILE_BYTES,
    MAX_TOTAL_BYTES,
    SnapshotError,
    build_snapshot,
    sha256_bytes,
)


STATE_SCHEMA_VERSION = 1
MAX_RESULT_FILE_BYTES = 1024 * 1024
MAX_RESULT_TOTAL_BYTES = 8 * 1024 * 1024
MAX_EXPERIMENTS = 10_000
MAX_SEARCH_LIMIT = 100
RUNTIME_ENABLE_VALUE = "cloud-candidate"

AVAILABLE_CAPABILITIES = {
    "automatic_result_archive",
    "content_addressed_source_snapshot",
    "declared_precision_gate",
    "experiment_record_read",
    "lifecycle",
}


class BackendError(RuntimeError):
    """A lifecycle action cannot proceed without ambiguity or data loss."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _sha256_file(path: Path, *, maximum: int | None = None) -> str:
    digest = hashlib.sha256()
    total = 0
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise BackendError(f"not a regular file: {path.name}")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if maximum is not None and total > maximum:
                raise BackendError(f"file exceeds {maximum} bytes: {path.name}")
            digest.update(chunk)
        after = os.fstat(descriptor)
        if (
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        ):
            raise BackendError(f"file changed while reading: {path.name}")
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _read_bounded_file(path: Path, *, maximum: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > maximum:
            raise BackendError(f"file is not a bounded regular file: {path.name}")
        chunks = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(256 * 1024, maximum + 1 - total))
            if not chunk:
                break
            total += len(chunk)
            if total > maximum:
                raise BackendError(f"file exceeds {maximum} bytes: {path.name}")
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            or total != after.st_size
        ):
            raise BackendError(f"file changed while reading: {path.name}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _ensure_directory(path: Path) -> None:
    try:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        current = path.lstat()
    except OSError as exc:
        raise BackendError(f"cannot create assistant directory: {path}") from exc
    if stat.S_ISLNK(current.st_mode) or not stat.S_ISDIR(current.st_mode):
        raise BackendError(f"assistant path must be a non-symlink directory: {path}")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write(path: Path, value: Any, *, replace: bool = True) -> None:
    _ensure_directory(path.parent)
    data = canonical_json_bytes(value)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            try:
                os.link(temporary, path)
            except FileExistsError:
                if path.read_bytes() != data:
                    raise BackendError(f"refusing to overwrite immutable record: {path.name}")
            temporary.unlink(missing_ok=True)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = path.read_bytes()
        value = json.loads(data)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BackendError(f"cannot reliably parse assistant state: {path}") from exc
    if not isinstance(value, dict):
        raise BackendError(f"assistant state must be an object: {path}")
    return value


def _process_start_ticks(pid: int) -> int | None:
    try:
        text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        fields = text[text.rfind(")") + 2:].split()
        return int(fields[19])
    except (OSError, ValueError, IndexError):
        return None


def _process_matches(pid: int, start_ticks: int, token: str) -> bool:
    if _process_start_ticks(pid) != start_ticks:
        return False
    try:
        argv = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    except OSError:
        return False
    decoded = [item.decode("utf-8", errors="replace") for item in argv if item]
    return token in decoded and "_worker" in decoded


def _safe_result_relpath(raw: str) -> str:
    path = PurePosixPath(raw)
    if (
        not raw or "\x00" in raw or "\\" in raw or path.is_absolute()
        or ".." in path.parts or raw.startswith("./")
    ):
        raise BackendError(f"unsafe result path: {raw}")
    return path.as_posix()


def _load_snapshot_manifest(archive_path: Path) -> dict[str, Any]:
    try:
        with tarfile.open(archive_path, "r") as archive:
            member = archive.getmember(MANIFEST_NAME)
            source = archive.extractfile(member)
            if source is None or member.size > MAX_FILE_BYTES:
                raise BackendError("snapshot manifest is unavailable")
            manifest = json.loads(source.read())
    except (OSError, KeyError, tarfile.TarError, json.JSONDecodeError) as exc:
        raise BackendError("snapshot manifest cannot be verified") from exc
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), list):
        raise BackendError("snapshot manifest has an invalid shape")
    return manifest


def _extract_snapshot(archive_path: Path, expected_sha256: str, destination: Path) -> None:
    if _sha256_file(archive_path, maximum=MAX_TOTAL_BYTES + 32 * 1024 * 1024) \
            != expected_sha256:
        raise BackendError("source snapshot identity mismatch")
    _ensure_directory(destination)
    manifest = _load_snapshot_manifest(archive_path)
    expected = {
        item.get("path"): item for item in manifest["files"] if isinstance(item, dict)
    }
    if len(expected) != len(manifest["files"]):
        raise BackendError("snapshot manifest contains duplicate or invalid paths")
    seen: set[str] = set()
    total = 0
    try:
        with tarfile.open(archive_path, "r") as archive:
            for member in archive.getmembers():
                if member.name == MANIFEST_NAME:
                    continue
                relpath = _safe_result_relpath(member.name)
                if relpath not in expected or relpath in seen or not member.isfile():
                    raise BackendError(f"snapshot member is not declared safely: {relpath}")
                source = archive.extractfile(member)
                if source is None:
                    raise BackendError(f"snapshot member is unreadable: {relpath}")
                data = source.read(MAX_FILE_BYTES + 1)
                if len(data) > MAX_FILE_BYTES or len(data) != member.size:
                    raise BackendError(f"snapshot member size is invalid: {relpath}")
                record = expected[relpath]
                if record.get("bytes") != len(data) or record.get("sha256") != sha256_bytes(data):
                    raise BackendError(f"snapshot member identity mismatch: {relpath}")
                total += len(data)
                if total > MAX_TOTAL_BYTES:
                    raise BackendError("snapshot extraction exceeds its total-byte limit")
                target = destination / relpath
                _ensure_directory(target.parent)
                descriptor = os.open(
                    target,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                    0o700 if record.get("executable") else 0o600,
                )
                try:
                    with os.fdopen(descriptor, "wb") as handle:
                        handle.write(data)
                        handle.flush()
                        os.fsync(handle.fileno())
                except BaseException:
                    target.unlink(missing_ok=True)
                    raise
                seen.add(relpath)
    except (OSError, tarfile.TarError) as exc:
        raise BackendError("source snapshot extraction failed") from exc
    if seen != set(expected):
        raise BackendError("source snapshot is incomplete")


class ExperimentBackend:
    def __init__(self, root: Path, *, runtime_enabled: bool = False):
        root = Path(root)
        if not root.is_absolute():
            raise BackendError("assistant state root must be absolute")
        self.root = root.resolve()
        self.runtime_enabled = runtime_enabled
        _ensure_directory(self.root)
        for name in ("experiments", "records", "snapshots", "tmp"):
            _ensure_directory(self.root / name)

    @classmethod
    def from_environment(cls) -> "ExperimentBackend":
        raw = os.environ.get("CONVIR_EXPERIMENT_ASSISTANT_ROOT")
        if not raw:
            raise BackendError("CONVIR_EXPERIMENT_ASSISTANT_ROOT is required")
        return cls(
            Path(raw),
            runtime_enabled=(
                os.environ.get("CONVIR_EXPERIMENT_ASSISTANT_RUNTIME")
                == RUNTIME_ENABLE_VALUE
            ),
        )

    def _require_runtime(self) -> None:
        if not self.runtime_enabled:
            raise BackendError(
                "phase-2 runtime is disabled; enable it only in the cloud acceptance environment"
            )

    def _experiment_dir(self, experiment_id: str, *, create: bool) -> Path:
        if not isinstance(experiment_id, str) or not experiment_id \
                or any(char not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-"
                       for char in experiment_id):
            raise BackendError("experiment_id must be a safe token")
        path = self.root / "experiments" / experiment_id
        if create:
            _ensure_directory(path)
        else:
            try:
                path_stat = path.lstat()
            except OSError as exc:
                raise BackendError("experiment does not exist") from exc
            if stat.S_ISLNK(path_stat.st_mode) or not stat.S_ISDIR(path_stat.st_mode):
                raise BackendError("experiment state path is unsafe")
        return path

    @contextlib.contextmanager
    def _locked(self, experiment_id: str, *, create: bool = False) -> Iterator[Path]:
        directory = self._experiment_dir(experiment_id, create=create)
        lock_path = directory / "state.lock"
        descriptor = os.open(
            lock_path,
            os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield directory
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _load_state(self, directory: Path, *, required: bool = True) -> dict[str, Any] | None:
        path = directory / "state.json"
        if not path.exists():
            if required:
                raise BackendError("experiment does not exist")
            return None
        state = _read_json(path)
        if state.get("schema_version") != STATE_SCHEMA_VERSION \
                or state.get("experiment_id") != directory.name \
                or not isinstance(state.get("attempts"), list):
            raise BackendError("experiment state identity is invalid")
        return state

    def _save_state(self, directory: Path, state: dict[str, Any]) -> None:
        state["updated_at"] = utc_now()
        _atomic_write(directory / "state.json", state)

    def _snapshot(self, repo: Path, entrypoint: str) -> dict[str, Any]:
        repo = Path(repo).resolve()
        if not repo.is_dir():
            raise BackendError("local_repo does not exist")
        entrypoint_path = repo / entrypoint
        try:
            entry_stat = entrypoint_path.lstat()
            entrypoint_path.resolve().relative_to(repo)
        except (OSError, ValueError) as exc:
            raise BackendError("entrypoint is not a safe file inside local_repo") from exc
        if stat.S_ISLNK(entry_stat.st_mode) or not stat.S_ISREG(entry_stat.st_mode) \
                or entrypoint_path.suffix != ".py":
            raise BackendError("entrypoint must be a regular repository-relative Python file")

        temporary = self.root / "tmp" / f"snapshot-{uuid.uuid4().hex}.tar"
        try:
            built = build_snapshot(repo, temporary)
        except SnapshotError as exc:
            raise BackendError(str(exc)) from exc
        try:
            manifest = _load_snapshot_manifest(temporary)
            if entrypoint not in {item.get("path") for item in manifest["files"]}:
                raise BackendError(
                    "entrypoint is ignored, excluded, or absent from the source snapshot"
                )
            final = self.root / "snapshots" / f"{built['sha256']}.tar"
            try:
                os.link(temporary, final)
                os.chmod(final, 0o400)
                _fsync_directory(final.parent)
            except FileExistsError:
                if _sha256_file(final) != built["sha256"]:
                    raise BackendError("existing content-addressed snapshot is corrupted")
        finally:
            temporary.unlink(missing_ok=True)
        return {
            "sha256": built["sha256"],
            "storage": "cloud_full",
            "base_commit": built["base_commit"],
            "diff_sha256": built["diff_sha256"],
        }

    def _reconcile_locked(self, directory: Path, state: dict[str, Any]) -> dict[str, Any]:
        if not state["attempts"]:
            return state
        attempt = state["attempts"][-1]
        if attempt.get("state") not in {"PREPARED", "RUNNING"}:
            return state
        active = state.get("active")
        if isinstance(active, dict) and _process_matches(
            active.get("pid", -1), active.get("start_ticks", -1), active.get("token", ""),
        ):
            return state
        attempt["state"] = "UNKNOWN"
        attempt["ended_at"] = utc_now()
        attempt["error_summary"] = (
            "lifecycle process identity is unavailable before a terminal record; "
            "the attempt cannot be relaunched automatically"
        )
        state["active"] = None
        state["archive"] = {"state": "NOT_ARCHIVED_UNKNOWN"}
        self._save_state(directory, state)
        return state

    def _public_state(self, state: dict[str, Any], *, full: bool = False) -> dict[str, Any]:
        attempts = state["attempts"]
        latest = attempts[-1] if attempts else None
        response: dict[str, Any] = {
            "experiment_id": state["experiment_id"],
            "state": None if latest is None else latest["state"],
            "attempt_count": len(attempts),
            "automatic_repairs_used": sum(
                bool(item.get("automatic_repair")) for item in attempts
            ),
            "archive": state.get("archive", {"state": "NONE"}),
            "warnings": state.get("warnings", []),
            "updated_at": state.get("updated_at"),
        }
        if latest is not None:
            response["latest_attempt"] = {
                key: latest.get(key) for key in (
                    "attempt_number", "state", "started_at", "ended_at",
                    "error_summary", "result", "source_snapshot", "budget",
                    "cloud_run_ref", "automatic_repair",
                )
            }
        if full:
            response["contract"] = state["validated_contract"]["contract"]
            response["contract_sha256"] = state["validated_contract"]["contract_sha256"]
            response["attempts"] = attempts
        return response

    def _launch_locked(
        self,
        directory: Path,
        state: dict[str, Any],
        validated: dict[str, Any],
        repo: Path,
        *,
        automatic_repair: bool,
    ) -> dict[str, Any]:
        snapshot = self._snapshot(repo, validated["contract"]["entrypoint"]["relpath"])
        number = len(state["attempts"]) + 1
        attempt_dir = directory / "attempts" / str(number)
        if attempt_dir.exists():
            raise BackendError("refusing to overwrite an existing attempt directory")
        _ensure_directory(attempt_dir)
        for name in ("control", "result"):
            _ensure_directory(attempt_dir / name)
        _atomic_write(
            attempt_dir / "control" / "contract.json",
            validated["contract"],
            replace=False,
        )
        attempt = {
            "schema_version": 1,
            "experiment_id": state["experiment_id"],
            "attempt_number": number,
            "contract_sha256": validated["contract_sha256"],
            "source_snapshot": snapshot,
            "budget": validated["contract"]["budget"],
            "state": "PREPARED",
            "automatic_repair": automatic_repair,
            "started_at": utc_now(),
            "ended_at": None,
            "error_summary": None,
            "result": None,
            "cloud_run_ref": f"experiments/{state['experiment_id']}/attempts/{number}",
        }
        state["validated_contract"] = validated
        state["repo_path"] = str(repo)
        state["attempts"].append(attempt)
        state["archive"] = {"state": "NONE"}
        state["active"] = None
        self._save_state(directory, state)

        token = uuid.uuid4().hex
        command = [
            sys.executable, str(Path(__file__).resolve()), "_worker",
            "--root", str(self.root), "--experiment-id", state["experiment_id"],
            "--attempt-number", str(number), "--token", token,
        ]
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
        start_ticks = _process_start_ticks(process.pid)
        if start_ticks is None:
            process.terminate()
            attempt["state"] = "UNKNOWN"
            attempt["ended_at"] = utc_now()
            attempt["error_summary"] = "worker process identity could not be established"
            self._save_state(directory, state)
            raise BackendError("worker launch state is unknown; do not start another attempt")
        state["active"] = {
            "pid": process.pid,
            "start_ticks": start_ticks,
            "token": token,
            "attempt_number": number,
        }
        attempt["state"] = "RUNNING"
        self._save_state(directory, state)
        _atomic_write(
            attempt_dir / "control" / "launch.json",
            {
                "experiment_id": state["experiment_id"],
                "attempt_number": number,
                "pid": process.pid,
                "start_ticks": start_ticks,
                "token": token,
            },
            replace=False,
        )
        return self._public_state(state)

    def start(self, local_repo: str, contract: Any) -> dict[str, Any]:
        self._require_runtime()
        try:
            validated = validate_contract(contract)
        except ContractError as exc:
            raise BackendError(str(exc)) from exc
        assessment = assess_launch(validated, AVAILABLE_CAPABILITIES)
        if not assessment["ok"]:
            raise BackendError("; ".join(assessment["blockers"]))
        experiment_id = validated["contract"]["experiment_id"]
        repo = Path(local_repo).resolve()
        with self._locked(experiment_id, create=True) as directory:
            existing = self._load_state(directory, required=False)
            if existing is not None:
                existing = self._reconcile_locked(directory, existing)
                latest = existing["attempts"][-1] if existing["attempts"] else None
                if latest and latest["state"] in RESULT_STATES:
                    raise BackendError(
                        "experiment already has a complete result; use a new experiment_id"
                    )
                if latest and latest["state"] == "FAILED_ENGINEERING":
                    raise BackendError("engineering failure must continue through experiment_repair")
                raise BackendError("experiment_id already exists and cannot be overwritten")
            state = {
                "schema_version": STATE_SCHEMA_VERSION,
                "experiment_id": experiment_id,
                "validated_contract": validated,
                "repo_path": str(repo),
                "attempts": [],
                "active": None,
                "archive": {"state": "NONE"},
                "warnings": assessment["warnings"] + [
                    "phase-2 archive is a cloud canonical record; GitHub transport is not enabled"
                ],
                "created_at": utc_now(),
                "updated_at": utc_now(),
            }
            return self._launch_locked(
                directory, state, validated, repo, automatic_repair=False,
            )

    def status(self, experiment_id: str) -> dict[str, Any]:
        with self._locked(experiment_id) as directory:
            state = self._load_state(directory)
            assert state is not None
            state = self._reconcile_locked(directory, state)
            return self._public_state(state)

    def repair(
        self,
        experiment_id: str,
        *,
        contract: Any | None = None,
        operator_confirmed: bool = False,
    ) -> dict[str, Any]:
        self._require_runtime()
        with self._locked(experiment_id) as directory:
            state = self._load_state(directory)
            assert state is not None
            state = self._reconcile_locked(directory, state)
            latest = state["attempts"][-1] if state["attempts"] else None
            if latest is None or latest["state"] != "FAILED_ENGINEERING":
                raise BackendError("only a FAILED_ENGINEERING attempt can be repaired")
            original = state["validated_contract"]
            try:
                revised = original if contract is None else validate_contract(contract)
            except ContractError as exc:
                raise BackendError(str(exc)) from exc
            if revised["contract"]["experiment_id"] != experiment_id:
                raise BackendError("repair contract must keep the same experiment_id")
            classification = classify_contract_revision(
                original["contract"], revised["contract"],
            )
            if not classification["same_experiment"]:
                raise BackendError(
                    "repair changes experiment meaning; use a new experiment_id: "
                    + ", ".join(classification["new_experiment_reasons"])
                )
            assessment = assess_launch(revised, AVAILABLE_CAPABILITIES)
            if not assessment["ok"]:
                raise BackendError("; ".join(assessment["blockers"]))
            authorization = authorize_attempt(
                state["attempts"], automatic_repair=True,
                operator_confirmed=operator_confirmed,
            )
            if not authorization["ok"]:
                raise BackendError(authorization["blocker"])
            state["warnings"] = sorted(set(
                state.get("warnings", []) + assessment["warnings"]
                + classification["warnings"]
            ))
            return self._launch_locked(
                directory,
                state,
                revised,
                Path(state["repo_path"]),
                automatic_repair=not operator_confirmed,
            )

    def cancel(self, experiment_id: str) -> dict[str, Any]:
        self._require_runtime()
        with self._locked(experiment_id) as directory:
            state = self._load_state(directory)
            assert state is not None
            state = self._reconcile_locked(directory, state)
            latest = state["attempts"][-1] if state["attempts"] else None
            if latest is None:
                raise BackendError("experiment has no attempt")
            if latest["state"] == "CANCELLED":
                return self._public_state(state)
            if latest["state"] != "RUNNING" or not isinstance(state.get("active"), dict):
                raise BackendError(f"experiment is not cancellable in state {latest['state']}")
            active = dict(state["active"])
            if not _process_matches(
                active["pid"], active["start_ticks"], active["token"],
            ):
                state = self._reconcile_locked(directory, state)
                return self._public_state(state)
            attempt_number = latest["attempt_number"]
            control = directory / "attempts" / str(attempt_number) / "control"
            cancel_path = control / "cancel.json"
            if not cancel_path.exists():
                _atomic_write(
                    cancel_path,
                    {
                        "experiment_id": experiment_id,
                        "attempt_number": attempt_number,
                        "requested_at": utc_now(),
                    },
                    replace=False,
                )
            if (control / "worker_ready.json").exists():
                os.kill(active["pid"], signal.SIGTERM)

        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            time.sleep(0.05)
            with self._locked(experiment_id) as directory:
                state = self._load_state(directory)
                assert state is not None
                if state["attempts"][-1]["state"] != "RUNNING":
                    return self._public_state(state)
                if not _process_matches(
                    active["pid"], active["start_ticks"], active["token"],
                ):
                    state = self._reconcile_locked(directory, state)
                    return self._public_state(state)
        return {
            **self.status(experiment_id),
            "cancellation_pending": True,
        }

    def get(self, experiment_id: str, *, view: str = "summary") -> dict[str, Any]:
        if view not in {"summary", "full"}:
            raise BackendError("view must be summary or full")
        with self._locked(experiment_id) as directory:
            state = self._load_state(directory)
            assert state is not None
            state = self._reconcile_locked(directory, state)
            return self._public_state(state, full=view == "full")

    def search(
        self,
        *,
        query: str | None = None,
        states: list[str] | None = None,
        limit: int = 20,
        compare_experiment_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_SEARCH_LIMIT:
            raise BackendError(f"limit must be in [1, {MAX_SEARCH_LIMIT}]")
        if query is not None and (not isinstance(query, str) or len(query) > 256):
            raise BackendError("query must be bounded text")
        if states is not None and (
            not isinstance(states, list) or any(not isinstance(item, str) for item in states)
        ):
            raise BackendError("states must be an array of strings")
        compare = compare_experiment_ids or []
        if not isinstance(compare, list) or len(compare) > 16:
            raise BackendError("compare_experiment_ids may contain at most 16 ids")
        experiments_root = self.root / "experiments"
        entries = sorted(experiments_root.iterdir(), key=lambda path: path.name)
        if len(entries) > MAX_EXPERIMENTS:
            raise BackendError("experiment catalog exceeds its bounded size")
        normalized_query = "" if query is None else query.casefold().strip()
        wanted_states = None if states is None else set(states)
        summaries = []
        for path in entries:
            try:
                path_stat = path.lstat()
            except OSError as exc:
                raise BackendError("experiment catalog changed while reading") from exc
            if stat.S_ISLNK(path_stat.st_mode) or not stat.S_ISDIR(path_stat.st_mode):
                raise BackendError("experiment catalog contains an unsafe entry")
            with self._locked(path.name) as directory:
                state = self._load_state(directory)
                assert state is not None
                state = self._reconcile_locked(directory, state)
                summary = self._public_state(state)
                objective = state["validated_contract"]["contract"]["objective"]
                if normalized_query and normalized_query not in (
                    f"{path.name} {objective}".casefold()
                ):
                    continue
                if wanted_states is not None and summary["state"] not in wanted_states:
                    continue
                summaries.append({
                    "experiment_id": path.name,
                    "objective": objective,
                    "state": summary["state"],
                    "attempt_count": summary["attempt_count"],
                    "primary_metric": (
                        (summary.get("latest_attempt") or {}).get("result") or {}
                    ).get("primary_metric"),
                    "updated_at": summary["updated_at"],
                })
            if len(summaries) > limit:
                break
        truncated = len(summaries) > limit
        summaries = summaries[:limit]
        comparison = []
        for experiment_id in compare:
            item = self.get(experiment_id, view="summary")
            latest = item.get("latest_attempt") or {}
            comparison.append({
                "experiment_id": experiment_id,
                "state": item["state"],
                "attempt_count": item["attempt_count"],
                "primary_metric": (latest.get("result") or {}).get("primary_metric"),
                "budget": latest.get("budget"),
                "source_snapshot_sha256": (
                    latest.get("source_snapshot") or {}
                ).get("sha256"),
            })
        return {
            "experiments": summaries,
            "count": len(summaries),
            "comparison": comparison,
            "complete": not truncated,
        }


def _collect_result(contract: dict[str, Any], output_root: Path) -> tuple[str, dict[str, Any]]:
    file_records = []
    total = 0
    primary_payload: dict[str, Any] | None = None
    for index, raw in enumerate(contract["evaluation"]["result_files"]):
        relpath = _safe_result_relpath(raw)
        path = output_root / relpath
        try:
            path_stat = path.lstat()
            path.resolve().relative_to(output_root.resolve())
        except (OSError, ValueError) as exc:
            raise BackendError(f"required result is missing or unsafe: {relpath}") from exc
        if stat.S_ISLNK(path_stat.st_mode) or not stat.S_ISREG(path_stat.st_mode):
            raise BackendError(f"required result must be a regular file: {relpath}")
        if path_stat.st_size > MAX_RESULT_FILE_BYTES:
            raise BackendError(f"required result exceeds its compact size limit: {relpath}")
        total += path_stat.st_size
        if total > MAX_RESULT_TOTAL_BYTES:
            raise BackendError("required results exceed their compact total size limit")
        data = _read_bounded_file(path, maximum=MAX_RESULT_FILE_BYTES)
        digest = sha256_bytes(data)
        file_records.append({"path": relpath, "bytes": len(data), "sha256": digest})
        if index == 0:
            try:
                payload = json.loads(data.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise BackendError("first required result must be valid UTF-8 JSON") from exc
            if not isinstance(payload, dict) or not isinstance(payload.get("primary_metric"), dict):
                raise BackendError("first required result must contain primary_metric")
            primary_payload = payload["primary_metric"]
    assert primary_payload is not None
    metric_contract = contract["evaluation"]["primary_metric"]
    value = primary_payload.get("value")
    if primary_payload.get("id") != metric_contract["id"]:
        raise BackendError("result primary_metric.id does not match the contract")
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(float(value)):
        raise BackendError("result primary_metric.value must be finite and numeric")
    normalized_value = float(value)
    threshold = metric_contract["threshold"]
    if threshold is None:
        state = "COMPLETED_INCONCLUSIVE"
        basis = "descriptive_no_threshold"
    else:
        passed = normalized_value >= threshold \
            if metric_contract["direction"] == "higher" else normalized_value <= threshold
        state = "COMPLETED_PASS" if passed else "COMPLETED_FAIL"
        basis = "frozen_threshold"
    return state, {
        "primary_metric": {
            "id": metric_contract["id"],
            "value": normalized_value,
            "direction": metric_contract["direction"],
            "threshold": threshold,
        },
        "decision_basis": basis,
        "files": file_records,
    }


def _bounded_log_tail(path: Path) -> str:
    try:
        data = path.read_bytes()[-2048:]
    except OSError:
        return "entrypoint failed without a readable runtime log"
    text = data.decode("utf-8", errors="replace").strip()
    return text[-2048:] or "entrypoint failed without diagnostic output"


def _worker_main(root: Path, experiment_id: str, attempt_number: int, token: str) -> int:
    backend = ExperimentBackend(root, runtime_enabled=True)
    directory = backend._experiment_dir(experiment_id, create=False)
    attempt_dir = directory / "attempts" / str(attempt_number)
    launch_path = attempt_dir / "control" / "launch.json"
    deadline = time.monotonic() + 10.0
    while not launch_path.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    if not launch_path.exists():
        return 2
    launch = _read_json(launch_path)
    if launch.get("token") != token or launch.get("pid") != os.getpid() \
            or launch.get("start_ticks") != _process_start_ticks(os.getpid()):
        return 3
    with backend._locked(experiment_id) as locked_dir:
        state = backend._load_state(locked_dir)
        assert state is not None
        active = state.get("active")
        if not isinstance(active, dict) or active.get("token") != token \
                or active.get("attempt_number") != attempt_number \
                or state["attempts"][-1].get("state") != "RUNNING":
            return 4
        validated = state["validated_contract"]
        attempt = dict(state["attempts"][-1])

    cancelled = False
    child: subprocess.Popen[bytes] | None = None

    def request_cancel(_signum: int, _frame: Any) -> None:
        nonlocal cancelled
        cancelled = True
        if child is not None and child.poll() is None:
            child.terminate()

    signal.signal(signal.SIGTERM, request_cancel)
    signal.signal(signal.SIGINT, request_cancel)
    _atomic_write(
        attempt_dir / "control" / "worker_ready.json",
        {
            "experiment_id": experiment_id,
            "attempt_number": attempt_number,
            "pid": os.getpid(),
            "start_ticks": _process_start_ticks(os.getpid()),
            "token": token,
        },
        replace=False,
    )
    cancelled = (attempt_dir / "control" / "cancel.json").exists()
    terminal_state = "FAILED_ENGINEERING"
    result = None
    error_summary = None
    try:
        snapshot_path = backend.root / "snapshots" / f"{attempt['source_snapshot']['sha256']}.tar"
        workspace = attempt_dir / "workspace"
        _extract_snapshot(snapshot_path, attempt["source_snapshot"]["sha256"], workspace)
        output_root = attempt_dir / "result"
        contract = validated["contract"]
        entrypoint = workspace / contract["entrypoint"]["relpath"]
        log_path = attempt_dir / "runtime.log"
        environment = {
            key: value for key, value in os.environ.items()
            if key not in {"PYTHONHOME", "PYTHONPATH"}
            and not key.startswith("CONVIR_EXPERIMENT_")
        }
        environment.update(contract["entrypoint"]["environment"])
        environment["CONVIR_EXPERIMENT_OUTPUT"] = str(output_root)
        environment["CONVIR_EXPERIMENT_CONTRACT"] = str(
            attempt_dir / "control" / "contract.json"
        )
        with log_path.open("wb") as log:
            child = subprocess.Popen(
                [sys.executable, str(entrypoint), *contract["entrypoint"]["argv"]],
                cwd=workspace,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                close_fds=True,
            )
            end = time.monotonic() + contract["budget"]["max_wall_seconds"]
            terminate_at: float | None = None
            timed_out = False
            while child.poll() is None:
                now = time.monotonic()
                if (attempt_dir / "control" / "cancel.json").exists():
                    cancelled = True
                if cancelled and terminate_at is None:
                    child.terminate()
                    terminate_at = now + 5.0
                elif terminate_at is not None and now >= terminate_at:
                    child.kill()
                elif not cancelled and terminate_at is None and now >= end:
                    timed_out = True
                    child.terminate()
                    terminate_at = now + 5.0
                time.sleep(0.05)
            return_code = child.returncode
        if cancelled:
            terminal_state = "CANCELLED"
        elif timed_out:
            error_summary = "entrypoint exceeded the frozen wall-time budget"
        elif return_code != 0:
            error_summary = _bounded_log_tail(log_path)
        else:
            terminal_state, result = _collect_result(contract, output_root)
    except Exception as exc:
        error_summary = str(exc)[:4096] or type(exc).__name__
    if cancelled:
        terminal_state = "CANCELLED"
        result = None
        error_summary = None

    with backend._locked(experiment_id) as locked_dir:
        state = backend._load_state(locked_dir)
        assert state is not None
        current = state["attempts"][-1]
        active = state.get("active")
        if current.get("attempt_number") != attempt_number \
                or not isinstance(active, dict) or active.get("token") != token:
            return 5
        current["state"] = terminal_state
        current["ended_at"] = utc_now()
        current["result"] = result
        current["error_summary"] = error_summary
        state["active"] = None
        if terminal_state in RESULT_STATES:
            try:
                archive = build_archive_record(
                    state["validated_contract"], state["attempts"], recorded_at=utc_now(),
                )
                record_path = backend.root / "records" / f"{experiment_id}.json"
                _atomic_write(record_path, archive["record"], replace=False)
                state["archive"] = {
                    "state": "CLOUD_RECORDED_GITHUB_PENDING",
                    "record_sha256": archive["record_sha256"],
                    "record_ref": f"records/{experiment_id}.json",
                }
            except Exception as exc:
                current["state"] = "UNKNOWN"
                current["result"] = None
                current["error_summary"] = (
                    "result completed but its canonical record could not be preserved: "
                    + str(exc)[:2048]
                )
                state["archive"] = {"state": "NOT_ARCHIVED_UNKNOWN"}
        elif terminal_state == "FAILED_ENGINEERING":
            state["archive"] = {"state": "CLOUD_ONLY_ENGINEERING_FAILURE"}
        else:
            state["archive"] = {"state": "CLOUD_ONLY_CANCELLED"}
        backend._save_state(locked_dir, state)
    return 0


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    worker = subparsers.add_parser("_worker")
    worker.add_argument("--root", type=Path, required=True)
    worker.add_argument("--experiment-id", required=True)
    worker.add_argument("--attempt-number", type=int, required=True)
    worker.add_argument("--token", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    if args.command == "_worker":
        return _worker_main(args.root, args.experiment_id, args.attempt_number, args.token)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
