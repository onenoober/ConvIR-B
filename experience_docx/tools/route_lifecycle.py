#!/usr/bin/env python3
"""Generic cloud lifecycle for declarative ConvIR-B route operations."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

from route_program_api import atomic_json
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
    "OUTPUT_PATH", "RUN_ID", "OUTPUT_ID", "GPU",
}
MAX_CLOSEOUT_BYTES = 64 * 1024
VERIFIED_ASSETS: list[dict[str, Any]] = []
WORKLOAD_STARTED = False


class LifecycleError(RuntimeError):
    def __init__(self, message: str, *, phase: str):
        super().__init__(message)
        self.phase = phase


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    return {key: os.environ[key] for key in REQUIRED_ENV}


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
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=30)
            return 124


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
        raise LifecycleError("contract result did not pass", phase="contract")
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
    return value


def validate_run_result(path: Path, spec: dict[str, Any], operation: dict[str, Any]) -> dict[str, Any]:
    value = load_json(path)
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
        if not source.is_file():
            if item["required"]:
                raise LifecycleError(f"required evidence missing: {item['source_relpath']}", phase="evidence")
            continue
        if source.is_symlink() or source.stat().st_size > item["max_bytes"]:
            raise LifecycleError(f"evidence contract failed: {item['source_relpath']}", phase="evidence")
        destination = evidence_root / item["destination_filename"]
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
                   write_local_copy: bool = True) -> Path:
    asset_identities = verified_asset_identities(verified_assets)
    closeout = {
        "schema_version": 1,
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


def lifecycle() -> int:
    global VERIFIED_ASSETS, WORKLOAD_STARTED
    VERIFIED_ASSETS = []
    WORKLOAD_STARTED = False
    env = require_environment()
    repo, run_root, output = validate_lifecycle_paths(env)
    manifest_path = repo / "experience_docx/route_operations.json"
    manifest = load_json(manifest_path)
    operation_id, operation = infer_operation(manifest, env)
    env["ROUTE_ID"] = manifest["route_id"]
    spec_path = repo / RUNTIME_SPEC_DIRECTORY / f"{operation_id}.json"
    spec = validate_runtime_spec(load_json(spec_path), manifest, operation_id)
    entrypoint = repo / spec["entrypoint_relpath"]
    evidence_root = repo / "experience_docx/experiment_logs" / spec["route_id"]
    status = output / "status.txt"
    heartbeat = output / "heartbeat.json"
    runtime_log = output / "runtime.log"
    if git(repo, "rev-parse", "HEAD") != env["EXPECTED_ROUTE_COMMIT"]:
        raise LifecycleError("route commit mismatch", phase="identity_preflight")
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
    asset_manifest = None
    if spec["asset_manifest_relpath"] is not None:
        asset_manifest = validate_asset_manifest(
            load_json(repo / spec["asset_manifest_relpath"]), spec,
        )
    capability_path = spec["engineering_contract"]["capability_profile_relpath"]
    if capability_path is not None:
        spec["_validated_capability_profile"] = validate_model_capability(
            load_json(repo / capability_path), spec, asset_manifest,
        )
    precision_path = spec["precision_contract"]["certificate_relpath"]
    if precision_path is not None:
        validate_precision_certificate(load_json(repo / precision_path), spec)
    strict_phased_assets = spec["schema_version"] >= 2
    assets = verify_assets(
        asset_manifest, repo=repo, run_root=run_root, output=output,
        contract_only=strict_phased_assets,
    )
    output.mkdir(parents=True)
    (output / "control").mkdir()
    atomic_json(
        output / "control/lifecycle_identity.json",
        lifecycle_identity(env, spec),
    )
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
        timeout=min(spec["timeout_seconds"], spec["engineering_contract"]["max_seconds"]),
    )
    if rc:
        tail = diagnostic_log_tail(runtime_log)
        detail = f"; program_tail={tail}" if tail else ""
        raise LifecycleError(f"contract program failed rc={rc}{detail}", phase="contract")
    validate_contract_result(Path(contract_context["result_path"]), spec)
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
    result = validate_run_result(Path(run_context["result_path"]), spec, operation)
    evidence = copy_evidence(spec, output, evidence_root)
    telemetry(
        repo, env, status, "terminal", "workload_end",
        spec["total_units"], spec["total_units"],
    )
    print("GENERIC_ROUTE_OPERATION_OK", flush=True)
    write_closeout(
        env=env, spec=spec, operation=operation, output=output,
        evidence_root=evidence_root, result=result, evidence_sha256=evidence,
        verified_assets=assets,
    )
    return 0


def main() -> None:
    try:
        raise SystemExit(lifecycle())
    except SystemExit:
        raise
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
