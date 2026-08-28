#!/usr/bin/env python3
"""Fixed local-to-cloud transport for the compact experiment assistant."""

from __future__ import annotations

import json
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any

from experiment_assistant_snapshot import MAX_TOTAL_BYTES, SnapshotError, build_snapshot


REMOTE_PROTOCOL_SCHEMA_VERSION = 1
REMOTE_HOST = "convir-4090"
REMOTE_BASE = "/sda/home/wangyuxin/ConvIR-B"
REMOTE_PYTHON = f"{REMOTE_BASE}/envs/convir-cu121/bin/python"
REMOTE_REPO = f"{REMOTE_BASE}/repos/ConvIR-B-mcp-main"
REMOTE_RUNNER = f"{REMOTE_REPO}/experience_docx/tools/experiment_assistant_runner.py"
REMOTE_STATE_ROOT = f"{REMOTE_BASE}/runtime/experiment-assistant-candidate"
REMOTE_DATASET_REGISTRY = f"{REMOTE_BASE}/runtime/experiment-assistant-datasets.json"
REMOTE_ARCHIVE = "git@github.com:onenoober/ConvIR-B.git"
SSH = "/usr/bin/ssh"
MAX_REMOTE_HEADER_BYTES = 256 * 1024
MAX_REMOTE_CAPTURE_BYTES = 64 * 1024
MAX_REMOTE_UPLOAD_BYTES = MAX_TOTAL_BYTES + 32 * 1024 * 1024
REMOTE_TIMEOUT_SECONDS = 180


class TransportError(RuntimeError):
    """The cloud request could not be completed with a reliable response."""


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ) + "\n").encode("utf-8")


class CloudExperimentClient:
    def __init__(
        self,
        *,
        remote_argv: list[str] | None = None,
        timeout: int = REMOTE_TIMEOUT_SECONDS,
    ):
        self.timeout = timeout
        self.remote_argv = list(remote_argv) if remote_argv is not None else [
            SSH,
            "-T",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=30",
            REMOTE_HOST,
            REMOTE_PYTHON,
            REMOTE_RUNNER,
            "_remote",
            "--root",
            REMOTE_STATE_ROOT,
            "--dataset-registry",
            REMOTE_DATASET_REGISTRY,
            "--archive-remote",
            REMOTE_ARCHIVE,
        ]

    def _request(
        self,
        operation: str,
        arguments: dict[str, Any],
        *,
        snapshot: dict[str, Any] | None = None,
        payload_path: Path | None = None,
    ) -> dict[str, Any]:
        payload_bytes = 0
        if payload_path is not None:
            payload_bytes = payload_path.stat().st_size
            if not 0 < payload_bytes <= MAX_REMOTE_UPLOAD_BYTES:
                raise TransportError("source snapshot exceeds the upload size limit")
        if (snapshot is None) != (payload_path is None):
            raise TransportError("source snapshot metadata and payload must be sent together")
        header = _canonical_bytes({
            "schema_version": REMOTE_PROTOCOL_SCHEMA_VERSION,
            "operation": operation,
            "arguments": arguments,
            "payload_bytes": payload_bytes,
            "source_snapshot": snapshot,
        })
        if len(header) > MAX_REMOTE_HEADER_BYTES:
            raise TransportError("remote request header exceeds its fixed size limit")
        try:
            process = subprocess.Popen(
                self.remote_argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except OSError as exc:
            raise TransportError("cloud transport could not start") from exc

        stdout = bytearray()
        stderr = bytearray()
        thread_errors: list[OSError] = []
        capture_limit = MAX_REMOTE_CAPTURE_BYTES + 1

        def drain(stream: Any, target: bytearray) -> None:
            try:
                while True:
                    chunk = stream.read(8192)
                    if not chunk:
                        break
                    remaining = capture_limit - len(target)
                    if remaining > 0:
                        target.extend(chunk[:remaining])
            except OSError as exc:
                thread_errors.append(exc)
            finally:
                stream.close()

        def feed() -> None:
            try:
                process.stdin.write(header)
                if payload_path is not None:
                    with payload_path.open("rb") as source:
                        while True:
                            chunk = source.read(1024 * 1024)
                            if not chunk:
                                break
                            process.stdin.write(chunk)
                process.stdin.flush()
            except BrokenPipeError:
                pass
            except OSError as exc:
                thread_errors.append(exc)
            finally:
                process.stdin.close()

        threads = [
            threading.Thread(target=drain, args=(process.stdout, stdout), daemon=True),
            threading.Thread(target=drain, args=(process.stderr, stderr), daemon=True),
            threading.Thread(target=feed, daemon=True),
        ]
        for thread in threads:
            thread.start()
        try:
            return_code = process.wait(timeout=self.timeout)
        except subprocess.TimeoutExpired as exc:
            process.kill()
            process.wait()
            for thread in threads:
                thread.join(timeout=5)
            raise TransportError(
                "cloud request timed out; experiment state is unknown and must be inspected"
            ) from exc
        for thread in threads:
            thread.join(timeout=5)
        if any(thread.is_alive() for thread in threads) or thread_errors:
            raise TransportError("cloud transport streams did not close reliably")
        if len(stdout) > MAX_REMOTE_CAPTURE_BYTES or len(stderr) > MAX_REMOTE_CAPTURE_BYTES:
            raise TransportError("cloud response exceeded its fixed capture limit")
        if return_code:
            detail = bytes(stdout + stderr).decode("utf-8", errors="replace").strip()
            raise TransportError(
                f"cloud transport failed before a reliable response: {detail[:4096]}"
            )
        try:
            response = json.loads(stdout)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TransportError("cloud response could not be parsed reliably") from exc
        if not isinstance(response, dict) \
                or response.get("schema_version") != REMOTE_PROTOCOL_SCHEMA_VERSION \
                or response.get("operation") != operation \
                or not isinstance(response.get("ok"), bool):
            raise TransportError("cloud response identity is invalid")
        if response["ok"]:
            result = response.get("result")
            if not isinstance(result, dict):
                raise TransportError("cloud response result is invalid")
            return result
        error = response.get("error")
        raise TransportError(
            error[:4096] if isinstance(error, str) else "cloud request failed"
        )

    def _snapshot_request(
        self,
        operation: str,
        local_repo: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        repo = Path(local_repo).resolve()
        with tempfile.TemporaryDirectory(prefix="experiment-assistant-upload-") as raw:
            archive = Path(raw) / "source.tar"
            try:
                built = build_snapshot(repo, archive)
            except SnapshotError as exc:
                raise TransportError(str(exc)) from exc
            metadata = {
                key: built[key]
                for key in ("sha256", "storage", "base_commit", "diff_sha256")
            }
            return self._request(
                operation,
                arguments,
                snapshot=metadata,
                payload_path=archive,
            )

    def start(self, local_repo: str, contract: Any) -> dict[str, Any]:
        return self._snapshot_request(
            "start", local_repo, {"local_repo": local_repo, "contract": contract},
        )

    def status(self, experiment_id: str) -> dict[str, Any]:
        return self._request("status", {"experiment_id": experiment_id})

    def cancel(self, experiment_id: str) -> dict[str, Any]:
        return self._request("cancel", {"experiment_id": experiment_id})

    def repair(
        self,
        experiment_id: str,
        *,
        contract: Any | None = None,
        operator_confirmed: bool = False,
    ) -> dict[str, Any]:
        context = self._request("repair_context", {"experiment_id": experiment_id})
        local_repo = context.get("local_repo")
        if not isinstance(local_repo, str):
            raise TransportError("cloud repair context did not identify the local repository")
        arguments = {
            "experiment_id": experiment_id,
            "local_repo": local_repo,
            "operator_confirmed": operator_confirmed,
        }
        if contract is not None:
            arguments["contract"] = contract
        return self._snapshot_request("repair", local_repo, arguments)

    def get(self, experiment_id: str, *, view: str = "summary") -> dict[str, Any]:
        return self._request("get", {"experiment_id": experiment_id, "view": view})

    def search(
        self,
        *,
        query: Any = None,
        states: Any = None,
        limit: Any = 20,
        compare_experiment_ids: Any = None,
    ) -> dict[str, Any]:
        arguments = {"limit": limit}
        if query is not None:
            arguments["query"] = query
        if states is not None:
            arguments["states"] = states
        if compare_experiment_ids is not None:
            arguments["compare_experiment_ids"] = compare_experiment_ids
        return self._request("search", arguments)
