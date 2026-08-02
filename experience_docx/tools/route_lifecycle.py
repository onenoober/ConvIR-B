#!/usr/bin/env python3
"""Generic cloud lifecycle for declarative ConvIR-B route operations."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

import capability_registry
import scientific_contract as science_contract
from route_program_api import (
    atomic_json,
    load_completed_unit_ledger,
    load_context,
    validate_review_facts_value,
)
from route_runtime_contract import (
    ContractError,
    GENERIC_RUNNER_RELPATH,
    RUNTIME_SPEC_DIRECTORY,
    safe_join,
    validate_asset_manifest,
    validate_model_capability,
    validate_precision_certificate,
    validate_runtime_spec,
)


REMOTE_PYTHON = Path("/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python")
REMOTE_GIT = Path("/usr/bin/git")
REQUIRED_ENV = {
    "EXPECTED_ROUTE_COMMIT", "RUNNER_SHA256", "MODE", "REMOTE_REPO", "RUN_ROOT",
    "OUTPUT_PATH", "RUN_ID", "OUTPUT_ID", "GPU", "AUTHORITATIVE_MAIN_COMMIT",
}
MAX_CLOSEOUT_BYTES = 64 * 1024
RAW_ARTIFACT_MANIFEST_RELPATH = "control/raw_artifact_manifest.jsonl"
RAW_ARTIFACT_SCOPE_ROOTS = ("contract", "workload")
RAW_ARTIFACT_EXCLUDED_PATHS = (
    "control", "heartbeat.json", "runtime.log", "status.txt",
)
MAX_RAW_ARTIFACT_FILES = 25_000
MAX_RAW_ARTIFACT_PATH_BYTES = 16 * 1024 * 1024
VERIFIED_ASSETS: list[dict[str, Any]] = []
WORKLOAD_STARTED = False
ACTIVE_PROGRAM: subprocess.Popen | None = None
OPERATOR_CANCEL_REQUEST_PATH: Path | None = None


class LifecycleError(RuntimeError):
    def __init__(self, message: str, *, phase: str,
                 control_diagnostic: dict[str, Any] | None = None):
        super().__init__(message)
        self.phase = phase
        self.control_diagnostic = control_diagnostic


class OperatorCancelled(BaseException):
    def __init__(self, signum: int):
        super().__init__(f"operator cancellation signal {signum}")
        self.signum = signum


def operator_cancel_signal(signum: int, _frame: Any) -> None:
    request = OPERATOR_CANCEL_REQUEST_PATH
    if request is None or not request.is_file():
        raise LifecycleError(
            "termination signal lacks a receipt-bound operator request",
            phase="workload" if WORKLOAD_STARTED else "lifecycle",
        )
    raise OperatorCancelled(signum)


def safe_diagnostic_text(value: Any, maximum: int) -> str:
    text = str(value).replace("\x00", "")
    text = __import__("re").sub(
        r"(?i)\b(token|password|secret|api[_-]?key)\s*[:=]\s*\S+",
        r"\1=<redacted>", text,
    )
    text = __import__("re").sub(
        r"(?<![A-Za-z0-9_.-])(?:/sda/home|/home|/mnt)/[^\s:'\"]+",
        "<path>", text,
    )
    text = "\n".join(line.rstrip() for line in text.splitlines()[-20:])
    return text[-maximum:]


def diagnostic_log_tail(path: Path, maximum: int = 3072) -> str:
    """Read and sanitize only a bounded subprocess-log tail."""
    try:
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - 16 * 1024))
            raw = stream.read(16 * 1024)
    except OSError:
        return ""
    return safe_diagnostic_text(raw.decode("utf-8", errors="replace"), maximum)


def safe_control_diagnostic(value: Any) -> dict[str, Any]:
    """Keep only bounded, non-scientific lifecycle diagnostics."""
    if not isinstance(value, dict):
        return {}
    failed = value.get("failed_contract_checks")
    if not isinstance(failed, list):
        return {}
    result = []
    token = __import__("re").compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
    for item in failed[:32]:
        if isinstance(item, str) and token.fullmatch(item):
            result.append(item)
    return {"failed_contract_checks": sorted(set(result))} if result else {}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def raw_artifact_receipt_filename(closeout_filename: str) -> str:
    suffix = "_closeout.json"
    if not isinstance(closeout_filename, str) or not closeout_filename.endswith(suffix):
        raise LifecycleError(
            "closeout filename cannot derive raw artifact receipt", phase="evidence",
        )
    return closeout_filename[:-len(suffix)] + "_raw_artifact_receipt.json"


def _atomic_bytes(path: Path, raw: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _sha256_scanned_file(path: str, observed: os.stat_result) -> str:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise LifecycleError("raw artifact changed before hashing", phase="evidence") from exc
    try:
        current = os.fstat(descriptor)
        if not stat.S_ISREG(current.st_mode) or any(
            getattr(current, field) != getattr(observed, field)
            for field in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        ):
            raise LifecycleError("raw artifact changed before hashing", phase="evidence")
        digest = hashlib.sha256()
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        stable = os.fstat(descriptor)
        if any(
            getattr(stable, field) != getattr(current, field)
            for field in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        ):
            raise LifecycleError("raw artifact changed during hashing", phase="evidence")
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def build_raw_artifact_manifest(output: Path) -> tuple[bytes, list[dict[str, Any]]]:
    """Seal stable contract/workload files without including mutable control state."""
    records: list[dict[str, Any]] = []
    path_bytes = 0

    def walk(directory: Path, relative_root: str, artifact_class: str) -> None:
        nonlocal path_bytes
        try:
            with os.scandir(directory) as iterator:
                children = sorted(iterator, key=lambda item: item.name)
        except OSError as exc:
            raise LifecycleError(
                f"raw artifact scope is unreadable: {relative_root}", phase="evidence",
            ) from exc
        for child in children:
            relative = f"{relative_root}/{child.name}"
            try:
                encoded = relative.encode("utf-8", errors="strict")
                observed = child.stat(follow_symlinks=False)
            except (OSError, UnicodeEncodeError) as exc:
                raise LifecycleError(
                    f"raw artifact identity is unreadable: {relative}", phase="evidence",
                ) from exc
            path_bytes += len(encoded)
            if path_bytes > MAX_RAW_ARTIFACT_PATH_BYTES:
                raise LifecycleError("raw artifact path inventory exceeds its bound", phase="evidence")
            if stat.S_ISLNK(observed.st_mode):
                raise LifecycleError(
                    f"raw artifact cannot be a symlink: {relative}", phase="evidence",
                )
            if stat.S_ISDIR(observed.st_mode):
                walk(Path(child.path), relative, artifact_class)
                continue
            if not stat.S_ISREG(observed.st_mode):
                raise LifecycleError(
                    f"raw artifact must be a regular file: {relative}", phase="evidence",
                )
            if len(records) >= MAX_RAW_ARTIFACT_FILES:
                raise LifecycleError("raw artifact file inventory exceeds its bound", phase="evidence")
            records.append({
                "schema_version": 2,
                "relative_path": relative,
                "artifact_class": artifact_class,
                "bytes": observed.st_size,
                "sha256": _sha256_scanned_file(child.path, observed),
            })

    for root_name in RAW_ARTIFACT_SCOPE_ROOTS:
        root = output / root_name
        if root.is_symlink():
            raise LifecycleError(
                f"raw artifact scope cannot be a symlink: {root_name}", phase="evidence",
            )
        if root.exists():
            if not root.is_dir():
                raise LifecycleError(
                    f"raw artifact scope is not a directory: {root_name}", phase="evidence",
                )
            walk(root, root_name, f"{root_name}_output")
    records.sort(key=lambda item: item["relative_path"])
    raw = b"".join(
        (json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        for item in records
    )
    manifest_path = output / RAW_ARTIFACT_MANIFEST_RELPATH
    if manifest_path.exists():
        if manifest_path.is_symlink() or not manifest_path.is_file() \
                or manifest_path.read_bytes() != raw:
            raise LifecycleError(
                "existing raw artifact manifest differs from stable outputs",
                phase="evidence",
            )
    else:
        _atomic_bytes(manifest_path, raw)
    return raw, records


def stable_scope_inventory(output: Path) -> dict[str, str]:
    """Hash the bounded contract/workload scope for terminal-adapter isolation."""
    inventory: dict[str, str] = {}
    path_bytes = 0
    for root_name in RAW_ARTIFACT_SCOPE_ROOTS:
        root = output / root_name
        if not root.exists():
            continue
        if root.is_symlink() or not root.is_dir():
            raise LifecycleError(
                f"stable output scope is invalid: {root_name}", phase="finalize",
            )
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(output).as_posix()
            path_bytes += len(relative.encode("utf-8", errors="strict"))
            if path_bytes > MAX_RAW_ARTIFACT_PATH_BYTES:
                raise LifecycleError(
                    "stable output inventory exceeds its path bound", phase="finalize",
                )
            if path.is_symlink():
                raise LifecycleError(
                    f"stable output cannot be a symlink: {relative}", phase="finalize",
                )
            if path.is_dir():
                continue
            if not path.is_file():
                raise LifecycleError(
                    f"stable output must be a regular file: {relative}", phase="finalize",
                )
            if len(inventory) >= MAX_RAW_ARTIFACT_FILES:
                raise LifecycleError(
                    "stable output inventory exceeds its file bound", phase="finalize",
                )
            inventory[relative] = sha256(path)
    return inventory


def run_terminal_adapter(
    repo: Path, entrypoint: Path, run_context_path: Path, log_path: Path,
) -> None:
    """Invoke only the explicit adapter hook in an isolated Python process."""
    program = (
        "import importlib.util,sys;"
        "s=importlib.util.spec_from_file_location('convir_terminal_adapter',sys.argv[1]);"
        "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
        "f=getattr(m,'finalize_existing',None);"
        "assert callable(f),'finalize_existing adapter is unavailable';"
        "f(sys.argv[2])"
    )
    environment = os.environ.copy()
    environment["CONVIR_FINALIZATION_ONLY"] = "1"
    with log_path.open("ab") as log:
        completed = subprocess.run(
            [str(REMOTE_PYTHON), "-c", program, str(entrypoint), str(run_context_path)],
            cwd=repo, env=environment, stdout=log, stderr=subprocess.STDOUT,
            timeout=300, check=False,
        )
    if completed.returncode:
        tail = diagnostic_log_tail(log_path)
        detail = f"; program_tail={tail}" if tail else ""
        raise LifecycleError(
            f"terminal adapter failed rc={completed.returncode}{detail}", phase="finalize",
        )


def publish_raw_artifact_receipt(
    *, output: Path, evidence_root: Path, operation: dict[str, Any],
    env: dict[str, str], spec: dict[str, Any],
) -> tuple[str, str]:
    manifest_raw, records = build_raw_artifact_manifest(output)
    category_counts = {
        f"{root}_output": sum(
            item["artifact_class"] == f"{root}_output" for item in records
        )
        for root in RAW_ARTIFACT_SCOPE_ROOTS
    }
    if sum(category_counts.values()) != len(records):
        raise LifecycleError(
            "raw artifact category counts do not cover the manifest", phase="evidence",
        )
    receipt = {
        "schema_version": 2,
        "route_id": spec["route_id"],
        "operation_id": spec["operation_id"],
        "run_id": env["RUN_ID"],
        "route_commit": env["EXPECTED_ROUTE_COMMIT"],
        "manifest_relative_path": RAW_ARTIFACT_MANIFEST_RELPATH,
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "entry_count": len(records),
        "total_bytes": sum(item["bytes"] for item in records),
        "category_counts": category_counts,
        "scope_roots": list(RAW_ARTIFACT_SCOPE_ROOTS),
        "excluded_paths": list(RAW_ARTIFACT_EXCLUDED_PATHS),
    }
    filename = raw_artifact_receipt_filename(operation["closeout_filename"])
    destination = evidence_root / filename
    if destination.exists():
        raise LifecycleError("raw artifact receipt destination exists", phase="evidence")
    atomic_json(destination, receipt)
    return filename, sha256(destination)


def git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        [str(REMOTE_GIT), "-C", str(repo), *args], text=True, capture_output=True,
        timeout=30, check=False,
    )
    if completed.returncode:
        raise LifecycleError(f"git {' '.join(args)} failed", phase="identity_preflight")
    return completed.stdout.strip()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def require_environment() -> dict[str, str]:
    missing = sorted(key for key in REQUIRED_ENV if key not in os.environ)
    if missing:
        raise LifecycleError(f"missing lifecycle environment: {missing}", phase="environment")
    result = {key: os.environ[key] for key in REQUIRED_ENV}
    if not capability_registry.SHA40.fullmatch(result["AUTHORITATIVE_MAIN_COMMIT"]):
        raise LifecycleError(
            "AUTHORITATIVE_MAIN_COMMIT is invalid", phase="environment",
        )
    return result


def validate_lifecycle_paths(env: dict[str, str]) -> tuple[Path, Path, Path]:
    if env["OUTPUT_ID"] != env["RUN_ID"]:
        raise LifecycleError("OUTPUT_ID and RUN_ID disagree", phase="environment")
    repo = Path(env["REMOTE_REPO"])
    run_root = Path(env["RUN_ROOT"])
    output = Path(env["OUTPUT_PATH"])
    if not all(path.is_absolute() for path in (repo, run_root, output)):
        raise LifecycleError("lifecycle paths must be absolute", phase="environment")
    repo = repo.resolve()
    run_root = run_root.resolve()
    output = output.resolve()
    if output != (run_root / env["RUN_ID"]).resolve():
        raise LifecycleError("OUTPUT_PATH must equal RUN_ROOT/RUN_ID", phase="environment")
    return repo, run_root, output


def infer_operation(manifest: dict[str, Any], env: dict[str, str]) -> tuple[str, dict[str, Any]]:
    matches = [
        (operation_id, operation)
        for operation_id, operation in manifest.get("operations", {}).items()
        if operation.get("mode") == env["MODE"]
        and operation.get("output_id") == env["RUN_ID"]
        and operation.get("runner_relpath") == GENERIC_RUNNER_RELPATH
    ]
    if len(matches) != 1:
        raise LifecycleError("lifecycle operation identity is not unique", phase="manifest_preflight")
    return matches[0]


def resolve_asset_path(value: str, *, repo: Path, run_root: Path, output: Path) -> Path:
    replacements = {
        "{REMOTE_REPO}": str(repo),
        "{RUN_ROOT}": str(run_root),
        "{OUTPUT_PATH}": str(output),
    }
    result = value
    for token, replacement in replacements.items():
        result = result.replace(token, replacement)
    if "{" in result or "}" in result:
        raise LifecycleError(f"unresolved asset path token: {value}", phase="asset_preflight")
    path = Path(result)
    if not path.is_absolute():
        raise LifecycleError(f"asset path is not absolute: {value}", phase="asset_preflight")
    resolved = path.resolve()
    for token, root in (("{REMOTE_REPO}", repo), ("{RUN_ROOT}", run_root),
                        ("{OUTPUT_PATH}", output)):
        if value.startswith(token):
            try:
                resolved.relative_to(root.resolve())
            except ValueError as exc:
                raise LifecycleError(
                    f"asset path escapes {token}: {value}", phase="asset_preflight",
                ) from exc
    return path.absolute()


def verify_assets(asset_manifest: dict[str, Any] | None, *, repo: Path,
                  run_root: Path, output: Path, contract_only: bool = False) -> list[dict[str, Any]]:
    global VERIFIED_ASSETS
    VERIFIED_ASSETS = []
    if asset_manifest is None:
        return []
    observed = []
    selected = asset_manifest["assets"]
    if contract_only:
        selected = [item for item in selected if item["contract_access"]]
    for item in selected:
        path = resolve_asset_path(item["path"], repo=repo, run_root=run_root, output=output)
        if path.is_symlink():
            raise LifecycleError(f"asset cannot be a symlink: {item['id']}", phase="asset_preflight")
        if item["kind"] == "file":
            if not path.is_file() or sha256(path) != item["sha256"]:
                raise LifecycleError(f"file asset mismatch: {item['id']}", phase="asset_preflight")
            observed.append({
                "id": item["id"], "kind": "file", "path": str(path),
                "sha256": item["sha256"], "access_role": item["access_role"],
                "contract_access": item["contract_access"],
            })
        elif item["kind"] == "directory":
            if not path.is_dir():
                raise LifecycleError(f"directory asset missing: {item['id']}", phase="asset_preflight")
            observed.append({
                "id": item["id"], "kind": "directory", "path": str(path),
                "access_role": item["access_role"],
                "contract_access": item["contract_access"],
            })
        elif item["kind"] == "git_checkout":
            if not path.is_dir() or git(path, "rev-parse", "HEAD") != item["commit"]:
                raise LifecycleError(f"Git asset mismatch: {item['id']}", phase="asset_preflight")
            if item["require_clean"] and git(path, "status", "--porcelain"):
                raise LifecycleError(f"Git asset is dirty: {item['id']}", phase="asset_preflight")
            observed.append({
                "id": item["id"], "kind": "git_checkout", "path": str(path),
                "commit": item["commit"], "access_role": item["access_role"],
                "contract_access": item["contract_access"],
            })
        VERIFIED_ASSETS = verified_asset_identities(observed)
    return observed


def context_value(*, phase: str, env: dict[str, str], spec: dict[str, Any],
                  output: Path, status: Path, heartbeat: Path,
                  assets: list[dict[str, Any]]) -> dict[str, Any]:
    phase_output = output / ("contract" if phase == "contract" else "workload")
    context_assets = assets if phase == "run" else [
        item for item in assets if item["contract_access"]
    ]
    permissions = spec["protected_data_permissions"] if phase == "run" else {
        "allow_confirmation": False,
        "allow_canary": False,
        "allow_locked_test": False,
    }
    return {
        "schema_version": 1,
        "phase": phase,
        "route_id": spec["route_id"],
        "operation_id": spec["operation_id"],
        "run_id": env["RUN_ID"],
        "route_commit": env["EXPECTED_ROUTE_COMMIT"],
        "runner_sha256": env["RUNNER_SHA256"],
        "entrypoint_relpath": spec["entrypoint_relpath"],
        "remote_repo": env["REMOTE_REPO"],
        "run_root": env["RUN_ROOT"],
        "output_path": str(output),
        "phase_output_path": str(phase_output),
        "result_path": str(phase_output / f"{phase}_result.json"),
        "status_path": str(status),
        "heartbeat_path": str(heartbeat),
        "device": (
            "cuda" if phase == "contract"
            and spec["engineering_contract"]["mode"] == "gpu_synthetic_no_data"
            else ("cpu" if phase == "contract" else ("cuda" if env["GPU"] else "cpu"))
        ),
        "total_units": spec["total_units"],
        "evidence_role": spec["evidence_role"],
        "resume_policy": spec["resume_policy"],
        "protected_data_permissions": permissions,
        "assets": context_assets,
        "engineering_contract": spec["engineering_contract"],
    }


def telemetry(repo: Path, env: dict[str, str], status: Path, phase: str,
              event: str, completed: int, total: int) -> None:
    helper = repo / "experience_docx/tools/run_telemetry.py"
    try:
        subprocess.run([
            str(REMOTE_PYTHON), str(helper), "event", "--route-id", env["ROUTE_ID"],
            "--run-id", env["RUN_ID"], "--phase", phase, "--status", str(status),
            "--event", event, "--completed", str(completed), "--total", str(total),
        ], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
    except (OSError, subprocess.SubprocessError):
        pass


def start_sidecar(repo: Path, env: dict[str, str], heartbeat: Path,
                  total_units: int) -> subprocess.Popen | None:
    helper = repo / "experience_docx/tools/run_telemetry.py"
    try:
        return subprocess.Popen([
            str(REMOTE_PYTHON), str(helper), "sidecar", "--route-id", env["ROUTE_ID"],
            "--run-id", env["RUN_ID"], "--phase", "workload", "--heartbeat",
            str(heartbeat), "--parent-pid", str(os.getpid()), "--interval-seconds", "60",
            "--total", str(total_units),
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        return None


def run_program(*, phase: str, context_path: Path, entrypoint: Path,
                spec: dict[str, Any], env: dict[str, str], log_path: Path,
                timeout: int) -> int:
    global ACTIVE_PROGRAM
    command_env = os.environ.copy()
    command_env.update(spec["environment"])
    command_env["PYTHONUNBUFFERED"] = "1"
    command_env["CONVIR_CONTRACT_ONLY"] = "1" if phase == "contract" else "0"
    contract_gpu = (
        phase == "contract"
        and spec["engineering_contract"]["mode"] == "gpu_synthetic_no_data"
    )
    command_env["CUDA_VISIBLE_DEVICES"] = env["GPU"] if contract_gpu or phase == "run" else ""
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            [str(REMOTE_PYTHON), str(entrypoint), phase, "--context", str(context_path)],
            cwd=env["REMOTE_REPO"], env=command_env, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True,
        )
        ACTIVE_PROGRAM = process
        active_path = context_path.parent / "active_program.json"
        atomic_json(active_path, {
            "schema_version": 1,
            "route_id": spec["route_id"],
            "operation_id": spec["operation_id"],
            "run_id": env["RUN_ID"],
            "route_commit": env["EXPECTED_ROUTE_COMMIT"],
            "phase": phase,
            "pid": process.pid,
            "pgid": process.pid,
            "entrypoint": str(entrypoint),
            "context_path": str(context_path),
        })
        try:
            return process.wait(timeout=timeout)
        except OperatorCancelled:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=10)
            raise
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=30)
            return 124
        finally:
            ACTIVE_PROGRAM = None
            try:
                active_path.unlink()
            except FileNotFoundError:
                pass


def validate_contract_result(path: Path, spec: dict[str, Any]) -> dict[str, Any]:
    value = load_json(path)
    legacy_expected = {
        "schema_version", "route_id", "operation_id", "phase", "ok", "checks",
        "output_contract_checked", "finalizer_contract_checked",
        "confirmation_images_targets_outcomes_touched", "canary_touched", "locked_test_touched",
    }
    expected = legacy_expected if spec["engineering_contract"]["legacy_implicit_contract"] \
        else legacy_expected | {"engineering"}
    if not isinstance(value, dict) or set(value) != expected:
        raise LifecycleError("contract result has an invalid field contract", phase="contract")
    required_true = (
        value["schema_version"] == 1,
        value["route_id"] == spec["route_id"],
        value["operation_id"] == spec["operation_id"],
        value["phase"] == "contract",
        value["ok"] is True,
        value["output_contract_checked"] is True,
        value["finalizer_contract_checked"] is True,
        value["confirmation_images_targets_outcomes_touched"] is False,
        value["canary_touched"] is False,
        value["locked_test_touched"] is False,
        isinstance(value["checks"], dict) and bool(value["checks"])
        and all(item is True for item in value["checks"].values()),
    )
    if not all(required_true):
        failed = []
        if isinstance(value.get("checks"), dict):
            failed = [
                key for key, passed in value["checks"].items()
                if isinstance(key, str) and passed is False
            ]
        raise LifecycleError(
            "contract result did not pass", phase="contract",
            control_diagnostic={"failed_contract_checks": failed},
        )
    if "engineering" in expected:
        engineering = value["engineering"]
        capability = spec.get("_validated_capability_profile")
        if engineering.get("mode") != spec["engineering_contract"]["mode"]:
            raise LifecycleError("engineering mode mismatch", phase="contract")
        if engineering.get("device") not in {"cpu", "cuda"}:
            raise LifecycleError("engineering device is invalid", phase="contract")
        expected_device = (
            "cuda" if spec["engineering_contract"]["mode"] == "gpu_synthetic_no_data"
            else "cpu"
        )
        if engineering.get("device") != expected_device:
            raise LifecycleError("engineering device differs from the frozen mode", phase="contract")
        if engineering.get("protected_data_touched") is not False \
                or engineering.get("scientific_output_created") is not False \
                or engineering.get("scientific_training_occurred") is not False:
            raise LifecycleError("engineering contract crossed a scientific boundary", phase="contract")
        if spec["engineering_contract"]["mode"] != "metadata_only":
            fixture = engineering.get("fixture")
            minimum = capability.get("minimum_fixture") if isinstance(capability, dict) else None
            if not isinstance(fixture, dict) or not isinstance(minimum, dict) \
                    or any(fixture.get(key, 0) < minimum[key] for key in minimum):
                raise LifecycleError("engineering fixture is below the capability minimum", phase="contract")
            if engineering.get("production_path_exercised") is not True:
                raise LifecycleError("engineering contract did not exercise the production path", phase="contract")
        cost_contract = spec["engineering_contract"].get("cost_contract")
        if cost_contract is not None:
            cost = engineering.get("cost")
            if not isinstance(cost, dict) or set(cost) != {
                "observed_iterations", "observed_wall_seconds",
                "observed_peak_memory_mib",
            }:
                raise LifecycleError("engineering cost evidence is invalid", phase="contract")
            expected_iterations = (
                cost_contract["formal_iterations"]
                if cost_contract["strategy"] == "same_scale_probe"
                else cost_contract["probe_iterations"]
            )
            if cost.get("observed_iterations") != expected_iterations:
                raise LifecycleError("engineering cost iteration count mismatch", phase="contract")
            wall = cost.get("observed_wall_seconds")
            peak = cost.get("observed_peak_memory_mib")
            if not isinstance(wall, (int, float)) or isinstance(wall, bool) or wall < 0 \
                    or not isinstance(peak, (int, float)) or isinstance(peak, bool) or peak < 0:
                raise LifecycleError("engineering cost measurements are invalid", phase="contract")
            wall_limit = (
                cost_contract["max_wall_seconds"]
                if cost_contract["strategy"] == "same_scale_probe"
                else cost_contract["fixed_overhead_seconds"]
                + cost_contract["probe_iterations"]
                * cost_contract["max_seconds_per_iteration"]
            )
            if wall > wall_limit + 1e-9:
                raise LifecycleError("engineering cost wall-time bound failed", phase="contract")
            if peak > cost_contract["max_peak_memory_mib"]:
                raise LifecycleError("engineering cost memory bound failed", phase="contract")
    return value


def validate_run_result(
    path: Path, spec: dict[str, Any], operation: dict[str, Any],
    scientific: dict[str, Any] | None = None,
) -> dict[str, Any]:
    value = load_json(path)
    if scientific is not None and scientific.get("schema_version") == 2:
        expected = {
            "schema_version", "route_id", "operation_id", "phase",
            "gate_outcomes", "details",
            "confirmation_images_targets_outcomes_touched", "canary_touched",
            "locked_test_touched",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise LifecycleError(
                "schema-2 gate result has an invalid field contract", phase="finalize",
            )
        if value["schema_version"] != 2 or value["route_id"] != spec["route_id"] \
                or value["operation_id"] != spec["operation_id"] \
                or value["phase"] != "run":
            raise LifecycleError("gate result identity mismatch", phase="finalize")
        try:
            terminal = science_contract.evaluate_gate_outcomes(
                scientific, value["gate_outcomes"],
            )
        except science_contract.ScientificContractError as exc:
            raise LifecycleError(str(exc), phase="finalize") from exc
        if {key: terminal[key] for key in ("state", "decision", "authorizes")} \
                not in operation["allowed_terminal_tuples"]:
            raise LifecycleError(
                "derived gate terminal tuple is not allowed", phase="finalize",
            )
        value = {**value, **terminal}
    else:
        expected = {
            "schema_version", "route_id", "operation_id", "phase", "state", "decision",
            "authorizes", "details", "confirmation_images_targets_outcomes_touched",
            "canary_touched", "locked_test_touched",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise LifecycleError("run result has an invalid field contract", phase="finalize")
        if value["schema_version"] != 1 or value["route_id"] != spec["route_id"] \
                or value["operation_id"] != spec["operation_id"] or value["phase"] != "run":
            raise LifecycleError("run result identity mismatch", phase="finalize")
        terminal = {key: value[key] for key in ("state", "decision", "authorizes")}
        if terminal not in operation["allowed_terminal_tuples"]:
            raise LifecycleError("run result terminal tuple is not allowed", phase="finalize")
    permissions = spec["protected_data_permissions"]
    touched = {
        "allow_confirmation": value["confirmation_images_targets_outcomes_touched"],
        "allow_canary": value["canary_touched"],
        "allow_locked_test": value["locked_test_touched"],
    }
    if any(item is not False and item is not True for item in touched.values()):
        raise LifecycleError("run result touched flags are not boolean", phase="finalize")
    if any(touched[key] and not permissions[key] for key in touched):
        raise LifecycleError("run result violates protected-data permissions", phase="finalize")
    if not isinstance(value["details"], dict):
        raise LifecycleError("run result details must be an object", phase="finalize")
    return value


def copy_evidence(spec: dict[str, Any], output: Path, evidence_root: Path) -> dict[str, str]:
    if evidence_root.is_symlink():
        raise LifecycleError("evidence root cannot be a symlink", phase="evidence")
    planned: list[tuple[Path, Path]] = []
    for item in spec["evidence_files"]:
        source = safe_join(output, item["source_relpath"])
        destination = evidence_root / item["destination_filename"]
        if not source.is_file():
            if item["required"]:
                raise LifecycleError(f"required evidence missing: {item['source_relpath']}", phase="evidence")
            continue
        if source.is_symlink() or source.stat().st_size > item["max_bytes"]:
            raise LifecycleError(f"evidence contract failed: {item['source_relpath']}", phase="evidence")
        if destination.name.endswith("_review_facts.json"):
            try:
                validate_review_facts_value(
                    load_json(source), expected_filename=destination.name,
                )
            except (ContractError, json.JSONDecodeError, OSError) as exc:
                raise LifecycleError(
                    f"review facts contract failed: {item['source_relpath']}: {exc}",
                    phase="evidence",
                ) from exc
        if destination.exists():
            raise LifecycleError(f"evidence destination exists: {destination.name}", phase="evidence")
        planned.append((source, destination))
    prepared: list[tuple[Path, Path, str]] = []
    temporaries: list[Path] = []
    committed: list[Path] = []
    try:
        for source, destination in planned:
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{destination.name}.", suffix=".tmp", dir=evidence_root,
            )
            os.close(descriptor)
            temporary = Path(temporary_name)
            temporaries.append(temporary)
            shutil.copyfile(source, temporary)
            digest = sha256(temporary)
            prepared.append((temporary, destination, digest))
        for temporary, destination, _ in prepared:
            try:
                os.link(temporary, destination)
            except FileExistsError as exc:
                raise LifecycleError(
                    f"evidence destination raced: {destination.name}", phase="evidence",
                ) from exc
            committed.append(destination)
        return {destination.name: digest for _, destination, digest in prepared}
    except BaseException:
        for destination in committed:
            try:
                destination.unlink()
            except FileNotFoundError:
                pass
        raise
    finally:
        for temporary in temporaries:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def lifecycle_identity(env: dict[str, str], spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "route_id": spec["route_id"],
        "operation_id": spec["operation_id"],
        "run_id": env["RUN_ID"],
        "route_commit": env["EXPECTED_ROUTE_COMMIT"],
        "runner_sha256": env["RUNNER_SHA256"],
    }


def verified_asset_identities(assets: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    identities = []
    for item in assets or []:
        identity = {
            key: item[key] for key in (
                "id", "kind", "access_role", "contract_access", "sha256", "commit",
            ) if key in item
        }
        identities.append(identity)
    return sorted(identities, key=lambda item: item["id"])


def output_owned_by_run(output: Path, env: dict[str, str], spec: dict[str, Any]) -> bool:
    marker = output / "control/lifecycle_identity.json"
    try:
        return load_json(marker) == lifecycle_identity(env, spec)
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return False


def write_closeout(*, env: dict[str, str], spec: dict[str, Any], operation: dict[str, Any],
                   output: Path, evidence_root: Path, result: dict[str, Any],
                   evidence_sha256: dict[str, str], failure_phase: str | None = None,
                   returncode: int = 0,
                   verified_assets: list[dict[str, Any]] | None = None,
                   capability_reuse: dict[str, Any] | None = None,
                   capability_qualification: dict[str, Any] | None = None,
                   write_local_copy: bool = True) -> Path:
    asset_identities = verified_asset_identities(verified_assets)
    closeout = {
        "schema_version": 2 if "terminal_label" in result else 1,
        "route_id": spec["route_id"],
        "operation_id": spec["operation_id"],
        "run_id": env["RUN_ID"],
        "route_commit": env["EXPECTED_ROUTE_COMMIT"],
        "runner_sha256": env["RUNNER_SHA256"],
        "state": result["state"],
        "decision": result["decision"],
        "authorizes": result["authorizes"],
        "evidence_role": spec["evidence_role"],
        "confirmation_images_targets_outcomes_touched": result.get(
            "confirmation_images_targets_outcomes_touched", False
        ),
        "canary_touched": result.get("canary_touched", False),
        "locked_test_touched": result.get("locked_test_touched", False),
        "evidence_sha256": dict(sorted(evidence_sha256.items())),
        "verified_assets": asset_identities,
        "details": result.get("details", {}),
        "failure_phase": failure_phase,
        "returncode": returncode,
    }
    if capability_reuse is not None:
        closeout["capability_reuse"] = capability_reuse
    if capability_qualification is not None:
        closeout["capability_qualification"] = capability_qualification
    for key in (
        "terminal_label", "decision_rule_id", "gate_outcomes",
        "next_action_id", "family_effect",
    ):
        if key in result:
            closeout[key] = result[key]
    closeout_size = len(
        (json.dumps(closeout, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    if closeout_size > MAX_CLOSEOUT_BYTES:
        raise LifecycleError("closeout exceeds the MCP size limit", phase="finalize")
    destination = evidence_root / operation["closeout_filename"]
    if destination.exists():
        raise LifecycleError("closeout evidence destination exists", phase="evidence")
    atomic_json(destination, closeout)
    if write_local_copy:
        try:
            control = output / "control"
            control.mkdir(parents=True, exist_ok=True)
            atomic_json(control / operation["closeout_filename"], closeout)
        except (OSError, ValueError, TypeError):
            pass
    return destination


def resolve_capability_reuse(
    repo: Path, capability: dict[str, Any] | None, authoritative_commit: str,
) -> dict[str, Any] | None:
    if not isinstance(capability, dict) or capability.get("schema_version") != 2:
        return None
    def snapshot_blob(relpath: str, *, required: bool = True) -> bytes | None:
        completed = subprocess.run(
            [
                str(REMOTE_GIT), "-C", str(repo), "show",
                f"{authoritative_commit}:{relpath}",
            ],
            capture_output=True, timeout=30, check=False,
        )
        if completed.returncode:
            if required:
                raise LifecycleError(
                    f"authoritative main is missing {relpath}", phase="contract",
                )
            return None
        return completed.stdout
    try:
        registry_raw = snapshot_blob(capability_registry.REGISTRY_RELPATH)
        return capability_registry.lookup_lines(
            registry_raw.decode("utf-8").splitlines(),
            capability["reuse_identity"],
            evidence_exists=lambda relpath: snapshot_blob(
                relpath, required=False,
            ) is not None,
            read_evidence=lambda relpath: snapshot_blob(relpath),
        )
    except (OSError, UnicodeDecodeError, capability_registry.CapabilityRegistryError) as exc:
        raise LifecycleError(
            f"capability registry is invalid: {exc}", phase="contract",
        ) from exc


def observed_device_class(env: dict[str, str], capability: dict[str, Any]) -> str:
    expected = capability["reuse_identity"]["device_class"]
    if capability["contract_mode"] != "gpu_synthetic_no_data":
        observed = "cpu"
    else:
        probe_env = os.environ.copy()
        probe_env["CUDA_VISIBLE_DEVICES"] = env["GPU"]
        completed = subprocess.run(
            [
                str(REMOTE_PYTHON), "-c",
                "import torch; a,b=torch.cuda.get_device_capability(0); "
                "print(f'cuda_sm{a}{b}')",
            ],
            cwd=env["REMOTE_REPO"], env=probe_env, text=True,
            capture_output=True, timeout=60, check=False,
        )
        observed = completed.stdout.strip()
        if completed.returncode or not observed.startswith("cuda_sm"):
            raise LifecycleError("cannot verify CUDA device class", phase="contract")
    if observed != expected:
        raise LifecycleError(
            f"capability device class mismatch: expected={expected} observed={observed}",
            phase="contract",
        )
    return observed


def publish_capability_qualification(
    *, evidence_root: Path, operation: dict[str, Any], env: dict[str, str],
    spec: dict[str, Any], capability: dict[str, Any],
    contract_result: dict[str, Any],
) -> tuple[dict[str, Any], str, str]:
    identity = capability["reuse_identity"]
    identity_sha = capability_registry.identity_digest(identity)
    qualification_id = f"cap_{identity_sha[:24]}"
    filename = operation["closeout_filename"].replace(
        "_closeout.json", "_capability_qualification.json",
    )
    if filename == operation["closeout_filename"]:
        raise LifecycleError(
            "closeout filename cannot derive capability evidence", phase="evidence",
        )
    value = {
        "schema_version": 1,
        "qualification_id": qualification_id,
        "identity": identity,
        "identity_sha256": identity_sha,
        "status": "PASSED_ENGINEERING",
        "contract_mode": capability["contract_mode"],
        "route_id": spec["route_id"],
        "operation_id": spec["operation_id"],
        "run_id": env["RUN_ID"],
        "route_commit": env["EXPECTED_ROUTE_COMMIT"],
        "engineering_evidence": contract_result["engineering"],
        "scientific_authorization": "NONE",
        "protected_data_touched": False,
    }
    destination = evidence_root / filename
    atomic_json(destination, value)
    return ({
        "qualification_id": qualification_id,
        "identity_sha256": identity_sha,
        "evidence_filename": filename,
        "status": "PASSED_ENGINEERING",
        "scientific_authorization": "NONE",
    }, filename, sha256(destination))


def lifecycle() -> int:
    global VERIFIED_ASSETS, WORKLOAD_STARTED, OPERATOR_CANCEL_REQUEST_PATH
    VERIFIED_ASSETS = []
    WORKLOAD_STARTED = False
    OPERATOR_CANCEL_REQUEST_PATH = None
    env = require_environment()
    repo, run_root, output = validate_lifecycle_paths(env)
    manifest_path = repo / "experience_docx/route_operations.json"
    manifest = load_json(manifest_path)
    operation_id, operation = infer_operation(manifest, env)
    env["ROUTE_ID"] = manifest["route_id"]
    spec_path = repo / RUNTIME_SPEC_DIRECTORY / f"{operation_id}.json"
    spec = validate_runtime_spec(load_json(spec_path), manifest, operation_id)
    scientific = None
    scientific_path = manifest.get("scientific_contract_relpaths", {}).get(operation_id)
    if scientific_path is not None:
        scientific_value = load_json(repo / scientific_path)
        if scientific_value.get("schema_version") == 2:
            try:
                scientific = science_contract.validate_scientific_contract_v2(
                    scientific_value, spec["route_id"], operation_id,
                )
            except science_contract.ScientificContractError as exc:
                raise LifecycleError(str(exc), phase="contract") from exc
    entrypoint = repo / spec["entrypoint_relpath"]
    evidence_root = repo / "experience_docx/experiment_logs" / spec["route_id"]
    status = output / "status.txt"
    heartbeat = output / "heartbeat.json"
    runtime_log = output / "runtime.log"
    if git(repo, "rev-parse", "HEAD") != env["EXPECTED_ROUTE_COMMIT"]:
        raise LifecycleError("route commit mismatch", phase="identity_preflight")
    if git(repo, "rev-parse", "refs/convir-runtime/main") \
            != env["AUTHORITATIVE_MAIN_COMMIT"]:
        raise LifecycleError(
            "authoritative main runtime ref mismatch", phase="identity_preflight",
        )
    if git(repo, "status", "--porcelain"):
        raise LifecycleError("route workspace is dirty before launch", phase="identity_preflight")
    runner = repo / GENERIC_RUNNER_RELPATH
    if sha256(runner) != env["RUNNER_SHA256"]:
        raise LifecycleError("generic runner hash mismatch", phase="identity_preflight")
    if not entrypoint.is_file() or entrypoint.is_symlink():
        raise LifecycleError("route entrypoint is unavailable", phase="identity_preflight")
    if output.exists():
        raise LifecycleError("output path already exists", phase="output_preflight")
    evidence_root.mkdir(parents=True, exist_ok=True)
    if evidence_root.is_symlink():
        raise LifecycleError("evidence root cannot be a symlink", phase="output_preflight")
    if (evidence_root / operation["closeout_filename"]).exists():
        raise LifecycleError("closeout filename already exists", phase="output_preflight")
    for item in spec["evidence_files"]:
        if (evidence_root / item["destination_filename"]).exists():
            raise LifecycleError("evidence filename already exists", phase="output_preflight")
    if scientific is not None and scientific.get("schema_version") == 2:
        receipt_filename = raw_artifact_receipt_filename(
            operation["closeout_filename"]
        )
        if receipt_filename in {
            item["destination_filename"] for item in spec["evidence_files"]
        }:
            raise LifecycleError(
                "raw artifact receipt filename conflicts with runtime evidence",
                phase="output_preflight",
            )
        if (evidence_root / receipt_filename).exists():
            raise LifecycleError(
                "raw artifact receipt filename already exists", phase="output_preflight",
            )
    asset_manifest = None
    if spec["asset_manifest_relpath"] is not None:
        asset_manifest = validate_asset_manifest(
            load_json(repo / spec["asset_manifest_relpath"]), spec,
        )
    capability_path = spec["engineering_contract"]["capability_profile_relpath"]
    capability_reuse = None
    capability = None
    if capability_path is not None:
        spec["_validated_capability_profile"] = validate_model_capability(
            load_json(repo / capability_path), spec, asset_manifest,
        )
        capability = spec["_validated_capability_profile"]
        if capability.get("schema_version") == 2:
            observed_device_class(env, capability)
            capability_reuse = resolve_capability_reuse(
                repo, capability, env["AUTHORITATIVE_MAIN_COMMIT"],
            )
            qualification_filename = operation["closeout_filename"].replace(
                "_closeout.json", "_capability_qualification.json",
            )
            if (evidence_root / qualification_filename).exists():
                raise LifecycleError(
                    "capability qualification evidence already exists",
                    phase="output_preflight",
                )
    precision_path = spec["precision_contract"]["certificate_relpath"]
    if precision_path is not None:
        validate_precision_certificate(load_json(repo / precision_path), spec, scientific)
    strict_phased_assets = spec["schema_version"] >= 2
    assets = verify_assets(
        asset_manifest, repo=repo, run_root=run_root, output=output,
        contract_only=strict_phased_assets,
    )
    output.mkdir(parents=True)
    (output / "control").mkdir()
    OPERATOR_CANCEL_REQUEST_PATH = output / "control/operator_cancel_request.json"
    atomic_json(
        output / "control/lifecycle_identity.json",
        lifecycle_identity(env, spec),
    )
    contract_result = None
    if capability_reuse is not None \
            and capability_reuse["engineering_reuse_authorized"] is True:
        atomic_json(output / "control/capability_reuse.json", capability_reuse)
        telemetry(repo, env, status, "contract", "contract_exact_reuse", 1, 1)
    else:
        contract_context = context_value(
            phase="contract", env=env, spec=spec, output=output,
            status=status, heartbeat=heartbeat, assets=assets,
        )
        contract_context_path = output / "control/contract_context.json"
        atomic_json(contract_context_path, contract_context)
        telemetry(repo, env, status, "contract", "contract_start", 0, 1)
        rc = run_program(
            phase="contract", context_path=contract_context_path, entrypoint=entrypoint,
            spec=spec, env=env, log_path=runtime_log,
            timeout=min(
                spec["timeout_seconds"], spec["engineering_contract"]["max_seconds"],
            ),
        )
        if rc:
            tail = diagnostic_log_tail(runtime_log)
            detail = f"; program_tail={tail}" if tail else ""
            raise LifecycleError(f"contract program failed rc={rc}{detail}", phase="contract")
        contract_result = validate_contract_result(
            Path(contract_context["result_path"]), spec,
        )
        if (output / "workload").exists():
            raise LifecycleError("contract program created workload output", phase="contract")
        telemetry(repo, env, status, "contract", "contract_pass", 1, 1)
    if strict_phased_assets:
        assets = verify_assets(
            asset_manifest, repo=repo, run_root=run_root, output=output,
            contract_only=False,
        )
    run_context = context_value(
        phase="run", env=env, spec=spec, output=output,
        status=status, heartbeat=heartbeat, assets=assets,
    )
    run_context_path = output / "control/run_context.json"
    atomic_json(run_context_path, run_context)
    start_sidecar(repo, env, heartbeat, spec["total_units"])
    telemetry(repo, env, status, "workload", "workload_start", 0, spec["total_units"])
    WORKLOAD_STARTED = True
    rc = run_program(
        phase="run", context_path=run_context_path, entrypoint=entrypoint,
        spec=spec, env=env, log_path=runtime_log, timeout=spec["timeout_seconds"],
    )
    if rc:
        raise LifecycleError(f"run program failed rc={rc}", phase="workload")
    result = validate_run_result(
        Path(run_context["result_path"]), spec, operation, scientific,
    )
    if scientific is not None and scientific.get("schema_version") == 2 \
            and spec["total_units"] > 0:
        ledger = load_completed_unit_ledger(load_context(run_context_path, "run"))
        if len(ledger) != spec["total_units"]:
            raise LifecycleError(
                "completed-unit ledger does not cover total_units", phase="finalize",
            )
    evidence = copy_evidence(spec, output, evidence_root)
    if scientific is not None and scientific.get("schema_version") == 2:
        filename, digest = publish_raw_artifact_receipt(
            output=output, evidence_root=evidence_root, operation=operation,
            env=env, spec=spec,
        )
        evidence[filename] = digest
    capability_qualification = None
    if capability is not None and capability.get("schema_version") == 2 \
            and capability_reuse is not None \
            and capability_reuse["engineering_reuse_authorized"] is False:
        if contract_result is None:
            raise LifecycleError(
                "new capability qualification lacks contract evidence", phase="evidence",
            )
        capability_qualification, filename, digest = publish_capability_qualification(
            evidence_root=evidence_root, operation=operation, env=env, spec=spec,
            capability=capability, contract_result=contract_result,
        )
        evidence[filename] = digest
    telemetry(
        repo, env, status, "terminal", "workload_end",
        spec["total_units"], spec["total_units"],
    )
    print("GENERIC_ROUTE_OPERATION_OK", flush=True)
    write_closeout(
        env=env, spec=spec, operation=operation, output=output,
        evidence_root=evidence_root, result=result, evidence_sha256=evidence,
        verified_assets=assets, capability_reuse=capability_reuse,
        capability_qualification=capability_qualification,
    )
    return 0


def finalize_existing(source_commit: str) -> int:
    """Publish a completed run without rerunning its scientific workload."""
    global WORKLOAD_STARTED
    WORKLOAD_STARTED = True
    env = require_environment()
    if not capability_registry.SHA40.fullmatch(source_commit):
        raise LifecycleError("finalization source commit is invalid", phase="environment")
    repo, _, output = validate_lifecycle_paths(env)
    manifest = load_json(repo / "experience_docx/route_operations.json")
    operation_id, operation = infer_operation(manifest, env)
    env["ROUTE_ID"] = manifest["route_id"]
    spec = validate_runtime_spec(
        load_json(repo / RUNTIME_SPEC_DIRECTORY / f"{operation_id}.json"),
        manifest, operation_id,
    )
    scientific_path = manifest.get("scientific_contract_relpaths", {}).get(operation_id)
    scientific = None
    if scientific_path is not None:
        try:
            scientific = science_contract.validate_scientific_contract_v2(
                load_json(repo / scientific_path), spec["route_id"], operation_id,
            )
        except science_contract.ScientificContractError as exc:
            raise LifecycleError(str(exc), phase="contract") from exc
    if git(repo, "rev-parse", "HEAD") != env["EXPECTED_ROUTE_COMMIT"]:
        raise LifecycleError("finalization repair commit mismatch", phase="identity_preflight")
    if git(repo, "rev-parse", "refs/convir-runtime/main") \
            != env["AUTHORITATIVE_MAIN_COMMIT"]:
        raise LifecycleError(
            "authoritative main runtime ref mismatch", phase="identity_preflight",
        )
    runner = repo / GENERIC_RUNNER_RELPATH
    if not runner.is_file() or runner.is_symlink():
        raise LifecycleError("generic runner is unavailable", phase="identity_preflight")
    identity_path = output / "control/lifecycle_identity.json"
    identity = load_json(identity_path)
    expected_source_identity = {
        "schema_version": 1,
        "route_id": spec["route_id"],
        "operation_id": spec["operation_id"],
        "run_id": env["RUN_ID"],
        "route_commit": source_commit,
        "runner_sha256": env["RUNNER_SHA256"],
    }
    if identity != expected_source_identity:
        raise LifecycleError(
            "completed output is not bound to the finalization source commit",
            phase="identity_preflight",
        )
    run_context_path = output / "control/run_context.json"
    run_context = load_context(run_context_path, "run")
    if run_context.route_commit != source_commit \
            or run_context.route_id != spec["route_id"] \
            or run_context.operation_id != operation_id \
            or run_context.run_id != env["RUN_ID"]:
        raise LifecycleError("run context source identity mismatch", phase="finalize")
    result_path = Path(run_context.result_path)
    result = validate_run_result(result_path, spec, operation, scientific)
    if scientific is not None and spec["total_units"] > 0:
        ledger = load_completed_unit_ledger(run_context)
        if len(ledger) != spec["total_units"]:
            raise LifecycleError(
                "completed-unit ledger does not cover total_units", phase="finalize",
            )
    raw_manifest = output / RAW_ARTIFACT_MANIFEST_RELPATH
    if raw_manifest.exists():
        build_raw_artifact_manifest(output)
    before = stable_scope_inventory(output)
    review_fact_sources = {
        item["source_relpath"] for item in spec["evidence_files"]
        if item["destination_filename"].endswith("_review_facts.json")
    }
    adapter_used = env["EXPECTED_ROUTE_COMMIT"] != source_commit
    if adapter_used:
        if not review_fact_sources:
            raise LifecycleError(
                "finalization repair has no declared review-facts output",
                phase="finalize",
            )
        run_terminal_adapter(
            repo, repo / spec["entrypoint_relpath"], run_context_path,
            output / "runtime.log",
        )
    after = stable_scope_inventory(output)
    unexpected_paths = sorted(
        path for path in set(before) | set(after)
        if path not in review_fact_sources and before.get(path) != after.get(path)
    )
    created_outside_adapter = sorted(set(after) - set(before) - review_fact_sources)
    removed_outside_adapter = sorted(set(before) - set(after) - review_fact_sources)
    if unexpected_paths or created_outside_adapter or removed_outside_adapter:
        raise LifecycleError(
            "terminal adapter changed stable scientific outputs outside review facts",
            phase="finalize",
            control_diagnostic={
                "failed_contract_checks": ["terminal_adapter_output_isolation"],
            },
        )
    if adapter_used and raw_manifest.exists():
        raw_manifest.unlink()
    # Revalidate the frozen result and every ledger-bound output after the adapter.
    result = validate_run_result(result_path, spec, operation, scientific)
    if scientific is not None and spec["total_units"] > 0:
        ledger = load_completed_unit_ledger(run_context)
        if len(ledger) != spec["total_units"]:
            raise LifecycleError(
                "completed-unit ledger changed during finalization", phase="finalize",
            )
    evidence_root = repo / "experience_docx/experiment_logs" / spec["route_id"]
    evidence_root.mkdir(parents=True, exist_ok=True)
    if evidence_root.is_symlink():
        raise LifecycleError("evidence root cannot be a symlink", phase="evidence")
    closeout_path = evidence_root / operation["closeout_filename"]
    if closeout_path.exists():
        raise LifecycleError("finalization closeout destination exists", phase="evidence")
    evidence = copy_evidence(spec, output, evidence_root)
    if scientific is not None:
        filename, digest = publish_raw_artifact_receipt(
            output=output, evidence_root=evidence_root, operation=operation,
            env=env, spec=spec,
        )
        evidence[filename] = digest
    capability_reuse = None
    capability_qualification = None
    capability_path = spec["engineering_contract"]["capability_profile_relpath"]
    if capability_path is not None and not adapter_used:
        asset_manifest = None
        if spec["asset_manifest_relpath"] is not None:
            asset_manifest = validate_asset_manifest(
                load_json(repo / spec["asset_manifest_relpath"]), spec,
            )
        capability = validate_model_capability(
            load_json(repo / capability_path), spec, asset_manifest,
        )
        if capability.get("schema_version") == 2:
            capability_reuse = resolve_capability_reuse(
                repo, capability, env["AUTHORITATIVE_MAIN_COMMIT"],
            )
            if capability_reuse["engineering_reuse_authorized"] is False:
                contract_context_path = output / "control/contract_context.json"
                contract_context = load_context(contract_context_path, "contract")
                contract_result = validate_contract_result(
                    Path(contract_context.result_path), spec,
                )
                capability_qualification, filename, digest = (
                    publish_capability_qualification(
                        evidence_root=evidence_root, operation=operation, env=env,
                        spec=spec, capability=capability,
                        contract_result=contract_result,
                    )
                )
                evidence[filename] = digest
    result = json.loads(json.dumps(result))
    result["details"] = {
        **result.get("details", {}),
        "finalization_recovery": {
            "source_commit": source_commit,
            "finalization_commit": env["EXPECTED_ROUTE_COMMIT"],
            "adapter_used": adapter_used,
            "workload_reexecuted": False,
            "stable_output_isolation_verified": True,
            "new_capability_registration_authorized": not adapter_used,
        },
    }
    failed_closeout = output / "control/failed_engineering_closeout.json"
    verified_assets = []
    if failed_closeout.is_file():
        prior = load_json(failed_closeout)
        if isinstance(prior, dict) and isinstance(prior.get("verified_assets"), list):
            verified_assets = prior["verified_assets"]
    write_closeout(
        env=env, spec=spec, operation=operation, output=output,
        evidence_root=evidence_root, result=result, evidence_sha256=evidence,
        verified_assets=verified_assets, capability_reuse=capability_reuse,
        capability_qualification=capability_qualification,
    )
    print("GENERIC_ROUTE_FINALIZATION_REPAIR_OK", flush=True)
    return 0


def cancellation_progress(status: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "completed_units": 0, "total_units": 0, "stage": None,
    }

    def visit(value: Any, typed: bool = False) -> None:
        if isinstance(value, dict):
            envelope = any(
                isinstance(key, str)
                and __import__("re").fullmatch(r"[A-Z][A-Z0-9_]{0,63}_PROGRESS", key)
                and isinstance(item, dict)
                for key, item in value.items()
            )
            current_typed = typed or value.get("phase") in {
                "contract", "workload", "terminal",
            } or envelope
            completed = value.get("completed_units", value.get("completed"))
            total = value.get("total_units", value.get("total"))
            if current_typed and isinstance(completed, int) \
                    and completed >= result["completed_units"]:
                result["completed_units"] = completed
                result["total_units"] = (
                    total if isinstance(total, int) and total >= completed else 0
                )
            stage = value.get("stage")
            if current_typed and isinstance(stage, str) \
                    and __import__("re").fullmatch(
                        r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", stage,
                    ):
                result["stage"] = stage
            for item in value.values():
                visit(item, current_typed)
        elif isinstance(value, list):
            for item in value:
                visit(item, typed)

    try:
        lines = status.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return result
    for line in lines[-100:]:
        try:
            visit(json.loads(line))
        except (json.JSONDecodeError, TypeError):
            continue
    return result


def finalize_operator_cancellation(exc: OperatorCancelled) -> None:
    env = require_environment()
    repo, _, output = validate_lifecycle_paths(env)
    manifest = load_json(repo / "experience_docx/route_operations.json")
    operation_id, operation = infer_operation(manifest, env)
    env["ROUTE_ID"] = manifest["route_id"]
    spec = validate_runtime_spec(
        load_json(repo / RUNTIME_SPEC_DIRECTORY / f"{operation_id}.json"),
        manifest, operation_id,
    )
    request_path = output / "control/operator_cancel_request.json"
    request = load_json(request_path)
    expected = {
        "schema_version": 1,
        "route_id": spec["route_id"],
        "run_id": env["RUN_ID"],
        "route_commit": env["EXPECTED_ROUTE_COMMIT"],
        "runner_sha256": env["RUNNER_SHA256"],
        "action": "cancel",
    }
    if not isinstance(request, dict) or any(
            request.get(key) != value for key, value in expected.items()):
        raise LifecycleError(
            "operator cancellation request identity mismatch",
            phase="operator_cancel",
        )
    request_id = request.get("request_id")
    requested_at = request.get("requested_at_unix")
    if not isinstance(request_id, str) \
            or not __import__("re").fullmatch(r"[0-9a-f]{32}", request_id) \
            or not isinstance(requested_at, int):
        raise LifecycleError(
            "operator cancellation request is malformed",
            phase="operator_cancel",
        )
    evidence_root = repo / "experience_docx/experiment_logs" / spec["route_id"]
    evidence_root.mkdir(parents=True, exist_ok=True)
    closeout_path = evidence_root / operation["closeout_filename"]
    if closeout_path.exists():
        return
    progress = cancellation_progress(output / "status.txt")
    role = spec.get("evidence_role")
    result = {
        "state": "CANCELLED_BY_OPERATOR",
        "decision": None,
        "authorizes": "NONE",
        "confirmation_images_targets_outcomes_touched": (
            WORKLOAD_STARTED and role == "confirmation"
        ),
        "canary_touched": WORKLOAD_STARTED and role == "canary",
        "locked_test_touched": WORKLOAD_STARTED and role == "sealed_final",
        "details": {
            "request_id": request_id,
            "requested_at_unix": requested_at,
            **progress,
            "termination_mode": "graceful",
            "signal": exc.signum,
            "workload_started": WORKLOAD_STARTED,
            "scientific_result_interpretable": False,
            "partial_scientific_evidence_reuse_authorized": False,
        },
    }
    write_closeout(
        env=env, spec=spec, operation=operation, output=output,
        evidence_root=evidence_root, result=result, evidence_sha256={},
        failure_phase="operator_cancel", returncode=130,
        verified_assets=VERIFIED_ASSETS,
        write_local_copy=output_owned_by_run(output, env, spec),
    )


def main() -> None:
    signal.signal(signal.SIGTERM, operator_cancel_signal)
    signal.signal(signal.SIGINT, operator_cancel_signal)
    if sys.argv[1:] and (
            len(sys.argv) != 3 or sys.argv[1] != "--finalize-existing"):
        print(
            "usage: route_lifecycle.py [--finalize-existing SOURCE_COMMIT]",
            file=sys.stderr,
        )
        raise SystemExit(2)
    operation = (
        (lambda: finalize_existing(sys.argv[2]))
        if sys.argv[1:] else lifecycle
    )
    try:
        raise SystemExit(operation())
    except SystemExit:
        raise
    except OperatorCancelled as exc:
        try:
            finalize_operator_cancellation(exc)
        except BaseException as closeout_exc:
            print(
                f"GENERIC_ROUTE_CANCEL_CLOSEOUT_FAILED "
                f"{type(closeout_exc).__name__}: {closeout_exc}",
                file=sys.stderr,
            )
        print("GENERIC_ROUTE_OPERATION_CANCELLED", file=sys.stderr)
        raise SystemExit(130)
    except BaseException as exc:
        try:
            env = require_environment()
            repo, _, output = validate_lifecycle_paths(env)
            manifest = load_json(repo / "experience_docx/route_operations.json")
            operation_id, operation = infer_operation(manifest, env)
            env["ROUTE_ID"] = manifest["route_id"]
            spec = validate_runtime_spec(
                load_json(repo / RUNTIME_SPEC_DIRECTORY / f"{operation_id}.json"),
                manifest, operation_id,
            )
            owns_output = output_owned_by_run(output, env, spec)
            if not output.exists():
                try:
                    output.mkdir(parents=True)
                    (output / "control").mkdir()
                    atomic_json(
                        output / "control/lifecycle_identity.json",
                        lifecycle_identity(env, spec),
                    )
                    owns_output = True
                except FileExistsError:
                    owns_output = output_owned_by_run(output, env, spec)
            evidence_root = repo / "experience_docx/experiment_logs" / spec["route_id"]
            evidence_root.mkdir(parents=True, exist_ok=True)
            message = safe_diagnostic_text(" ".join(str(exc).split()), 2048)
            scientific_roles = {
                "development_screening", "confirmation", "canary", "sealed_final",
            }
            protected_roles = {"confirmation", "canary", "sealed_final"}
            observed_roles = {
                item.get("access_role") for item in VERIFIED_ASSETS if isinstance(item, dict)
            }
            current_role = spec.get("evidence_role") if isinstance(spec, dict) else None
            failure = {
                "state": "FAILED_ENGINEERING", "decision": None, "authorizes": "NONE",
                "details": {
                    "error_type": type(exc).__name__,
                    "error_message": message,
                    "traceback_tail": safe_diagnostic_text(
                        "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
                        4096,
                    ),
                    "workload_started": WORKLOAD_STARTED,
                    "scientific_data_touched": bool(
                        observed_roles & scientific_roles
                        or WORKLOAD_STARTED and current_role in scientific_roles
                    ),
                    "protected_data_touched": bool(
                        observed_roles & protected_roles
                        or WORKLOAD_STARTED and current_role in protected_roles
                    ),
                    **safe_control_diagnostic(
                        getattr(exc, "control_diagnostic", None)
                    ),
                },
            }
            if {key: failure[key] for key in ("state", "decision", "authorizes")} \
                    not in operation["allowed_terminal_tuples"]:
                raise LifecycleError("operation lacks generic engineering tuple", phase="failure_closeout")
            write_closeout(
                env=env, spec=spec, operation=operation, output=output,
                evidence_root=evidence_root, result=failure, evidence_sha256={},
                failure_phase=getattr(exc, "phase", "lifecycle"), returncode=1,
                verified_assets=VERIFIED_ASSETS,
                write_local_copy=owns_output,
            )
        except BaseException as closeout_exc:
            print(f"GENERIC_ROUTE_CLOSEOUT_FAILED {type(closeout_exc).__name__}: {closeout_exc}", file=sys.stderr)
        print(
            f"GENERIC_ROUTE_OPERATION_FAILED {type(exc).__name__}: "
            f"{safe_diagnostic_text(exc, 2048)}",
            file=sys.stderr,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
