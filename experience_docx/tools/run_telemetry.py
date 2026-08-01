#!/usr/bin/env python3
"""Fail-open, metadata-only telemetry for long-running route runners.

This module never reads experiment outputs, metrics, images, checkpoints, GPU
state, or logs.  The sidecar only observes the identity of its parent process
through ``/proc`` and atomically replaces one heartbeat JSON file.  Telemetry
failure is deliberately non-fatal to the parent workload.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path


SCHEMA_VERSION = 1
SAFE_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
MAX_STATUS_BYTES = 4096


def require_token(value: str, name: str) -> str:
    if not isinstance(value, str) or not SAFE_TOKEN.fullmatch(value):
        raise ValueError(f"{name} must be a safe token")
    return value


def require_progress(completed: int, total: int) -> tuple[int, int]:
    if completed < 0 or total < 0 or completed > total:
        raise ValueError("progress must satisfy 0 <= completed <= total")
    return completed, total


def telemetry_path(raw: str, expected_name: str) -> Path:
    path = Path(raw)
    if not path.is_absolute() or path.name != expected_name:
        raise ValueError(f"telemetry path must be an absolute {expected_name} path")
    if path.is_symlink():
        raise ValueError("telemetry destination cannot be a symbolic link")
    return path.absolute()


def process_start_ticks(pid: int) -> str | None:
    """Return Linux process start ticks without sending any signal."""
    if pid < 1:
        return None
    try:
        raw = (Path("/proc") / str(pid) / "stat").read_text(encoding="utf-8")
        tail = raw.rsplit(")", 1)[1].split()
        return tail[19]
    except (FileNotFoundError, IndexError, OSError):
        return None


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    if len(raw) > MAX_STATUS_BYTES:
        raise ValueError("telemetry payload exceeds the fixed size limit")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def append_event(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    if len(raw) > MAX_STATUS_BYTES:
        raise ValueError("status event exceeds the fixed size limit")
    descriptor = os.open(
        path, os.O_APPEND | os.O_CREAT | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0), 0o640,
    )
    try:
        os.write(descriptor, raw)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def payload(args: argparse.Namespace, sequence: int, parent_ticks: str | None = None) -> dict:
    completed, total = require_progress(args.completed, args.total)
    value = {
        "schema_version": SCHEMA_VERSION,
        "route_id": require_token(args.route_id, "route_id"),
        "run_id": require_token(args.run_id, "run_id"),
        "phase": require_token(args.phase, "phase"),
        "completed": completed,
        "total": total,
        "sequence": sequence,
        "timestamp_unix": time.time(),
        "writer_pid": os.getpid(),
        "source": "generic_run_telemetry",
    }
    if getattr(args, "parent_pid", None) is not None:
        value["parent_pid"] = args.parent_pid
        value["parent_start_ticks"] = parent_ticks
    return value


def run_pulse(args: argparse.Namespace) -> int:
    atomic_json(telemetry_path(args.heartbeat, "heartbeat.json"), payload(args, args.sequence))
    return 0


def run_event(args: argparse.Namespace) -> int:
    value = payload(args, args.sequence)
    value["event"] = require_token(args.event, "event")
    append_event(telemetry_path(args.status, "status.txt"), value)
    return 0


def run_sidecar(args: argparse.Namespace) -> int:
    if not 0.02 <= args.interval_seconds <= 86400:
        raise ValueError("interval_seconds must be between 0.02 and 86400")
    if args.max_pulses < 0:
        raise ValueError("max_pulses must be nonnegative")
    parent_ticks = process_start_ticks(args.parent_pid)
    if parent_ticks is None:
        return 0
    sequence = 0
    while process_start_ticks(args.parent_pid) == parent_ticks:
        sequence += 1
        atomic_json(
            telemetry_path(args.heartbeat, "heartbeat.json"),
            payload(args, sequence, parent_ticks),
        )
        if args.max_pulses and sequence >= args.max_pulses:
            break
        deadline = time.monotonic() + args.interval_seconds
        while time.monotonic() < deadline:
            if process_start_ticks(args.parent_pid) != parent_ticks:
                return 0
            time.sleep(min(0.25, max(0.0, deadline - time.monotonic())))
    return 0


def add_identity(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--route-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--completed", type=int, default=0)
    parser.add_argument("--total", type=int, default=0)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(dest="command", required=True)

    pulse = commands.add_parser("pulse")
    add_identity(pulse)
    pulse.add_argument("--heartbeat", required=True)
    pulse.add_argument("--sequence", type=int, default=1)
    pulse.set_defaults(handler=run_pulse)

    event = commands.add_parser("event")
    add_identity(event)
    event.add_argument("--status", required=True)
    event.add_argument("--event", required=True)
    event.add_argument("--sequence", type=int, default=0)
    event.set_defaults(handler=run_event)

    sidecar = commands.add_parser("sidecar")
    add_identity(sidecar)
    sidecar.add_argument("--heartbeat", required=True)
    sidecar.add_argument("--parent-pid", type=int, required=True)
    sidecar.add_argument("--interval-seconds", type=float, default=60.0)
    sidecar.add_argument("--max-pulses", type=int, default=0)
    sidecar.set_defaults(handler=run_sidecar)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except Exception as exc:  # telemetry must never stop the experiment runner
        print(
            f"RUN_TELEMETRY_DEGRADED command={args.command} "
            f"error={type(exc).__name__}:{exc}",
            file=sys.stderr, flush=True,
        )
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
