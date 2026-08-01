#!/usr/bin/env python3
"""Restricted six-tool MCP bridge for ConvIR-B cloud route operations."""

import fcntl
import hashlib
import hmac
import json
import os
import re
import shlex
import stat
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import capability_registry
import scientific_contract as science_contract
from route_runtime_contract import (
    ContractError as RuntimeContractError,
    runtime_spec_relpath,
    validate_asset_manifest,
    validate_model_capability,
    validate_precision_certificate,
    validate_runtime_spec,
)


SERVER_NAME = "convir-ops"
SERVER_VERSION = "5.6.0"
SERVER_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
SCHEMA_VERSION = 4
SUPPORTED_MANIFEST_SCHEMA_VERSIONS = {4, 5, 6}
REMOTE_HOST = "convir-4090"
REMOTE_BASE = "/sda/home/wangyuxin/ConvIR-B"
REMOTE_REPOS = f"{REMOTE_BASE}/repos"
REMOTE_RUNS = f"{REMOTE_BASE}/runs"
REMOTE_PYTHON = f"{REMOTE_BASE}/envs/convir-cu121/bin/python"
NVIDIA_SMI = "/usr/bin/nvidia-smi"
CLOUD_GIT_SEED = f"{REMOTE_REPOS}/ConvIR-B-official-arch-anchor"
GITHUB_URL = "git@github.com:onenoober/ConvIR-B.git"
SSH = "/usr/bin/ssh"
REMOTE_BASH = "/bin/bash"
REMOTE_TMUX = "/usr/bin/tmux"
MAX_REMOTE_SCRIPT_BYTES = 256 * 1024
MAX_REMOTE_CAPTURE_BYTES = 64 * 1024
ROUTE_OPERATIONS_RELPATH = "experience_docx/route_operations.json"
RULE_COMPATIBILITY_RELPATH = "experience_docx/RULE_COMPATIBILITY.json"
RULE_BUNDLE_RELPATHS = (
    "AGENTS.md",
    "experience_docx/SCIENCE_FASTPATH.md",
    "experience_docx/ROUTE_READY_FASTPATH.md",
    "experience_docx/EXPERIMENT_GOVERNANCE_PROTOCOL.md",
    "experience_docx/MODEL_RUN_OPERATIONS_PROTOCOL.md",
    "experience_docx/COMMAND_RELIABILITY_PROTOCOL.md",
    "experience_docx/CONVIR_OPS_MCP.md",
    "experience_docx/BRANCH_EXPERIMENT_SYNC_PROTOCOL.md",
)
LOCAL_WORKSPACE_ROOT = Path(
    os.environ.get("CONVIR_OPS_LOCAL_WORKSPACE_ROOT", "/home/ubuntu/workspace")
).resolve()
LOCAL_GIT_SEED = Path(
    os.environ.get(
        "CONVIR_OPS_LOCAL_GIT_SEED",
        str(Path(__file__).resolve().parents[2]),
    )
).resolve()
STATE_DIR = Path(
    os.environ.get("CONVIR_OPS_STATE_DIR", "~/.codex/convir-ops-v4")
).expanduser().resolve()
PLAN_TTL_SECONDS = 15 * 60
MAX_FINISH_WINDOWS = 64
MAX_OPERATOR_OBSERVATIONS = 256
OPERATOR_OBSERVATION_MIN_INTERVAL_SECONDS = 15
OPERATOR_CANCEL_GRACE_SECONDS = 30
OPERATOR_CANCEL_FORCE_SECONDS = 10
MAX_EVIDENCE_BYTES = 1024 * 1024
MAX_MANIFEST_BYTES = 16 * 1024
MAX_CLOSEOUT_BYTES = 64 * 1024
MAX_DIAGNOSTIC_TEXT_BYTES = 4096
GPU_SUMMARY_LIMIT = 8
GPU_PROBE_RETRY_DELAY_SECONDS = 2
CONCLUSION_SCHEMA_VERSION = 2
CONCLUSION_REQUIRED_FIELDS = (
    "schema_version", "route_id", "operation_id", "run_id", "state",
    "decision", "authorizes", "primary_result", "gate_reasons",
    "competing_explanation", "limitations", "primary_fact_ids",
    "gate_fact_ids",
)
SAFE_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
SAFE_BRANCH = re.compile(r"^codex/[A-Za-z0-9][A-Za-z0-9_.\-/]{0,191}$")
SHA40 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
ALLOWED_EVIDENCE_SUFFIXES = {".json", ".csv", ".md", ".txt"}
MONITOR_PROFILES = {
    "short": {"max_polls": 3, "interval_seconds": 10},
    "standard": {"max_polls": 4, "interval_seconds": 15},
}
OPERATOR_CANCEL_TERMINAL = {
    "state": "CANCELLED_BY_OPERATOR", "decision": None, "authorizes": "NONE",
}


class ToolError(RuntimeError):
    def __init__(self, message, *, failure_phase="unknown", failure_class="contract"):
        super().__init__(message)
        self.failure_phase = failure_phase
        self.failure_class = failure_class


def emit(value):
    sys.stdout.write(json.dumps(value, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def text_result(text, *, is_error=False, structured=None):
    if structured is not None:
        keys = (
            "operation_state", "status", "state", "marker", "route_id",
            "operation_id", "run_id", "decision", "authorizes", "ok",
            "failure_class", "failure_phase", "audit_digest",
            "plan_token", "receipt", "retry_after_seconds",
        )
        summary = {key: structured[key] for key in keys if key in structured}
        summary["structured_content_available"] = True
        text = json.dumps(summary, sort_keys=True, separators=(",", ":"))
    value = {"content": [{"type": "text", "text": text}], "isError": is_error}
    if structured is not None:
        value["structuredContent"] = structured
    return value


def canonical_digest(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def typed_result(ok, state, failure_class="none", *, observed=None, expected=None,
                 mismatches=None, next_actions=None, **extra):
    value = {
        "schema_version": SCHEMA_VERSION,
        "ok": ok,
        "operation_state": state,
        "failure_class": failure_class,
        "observed": observed or {},
        "expected": expected or {},
        "mismatches": [safe_diagnostic_text(item, 1024) for item in (mismatches or [])],
        "allowed_next_actions": next_actions or [],
    }
    value.update(extra)
    value["audit_digest"] = canonical_digest(value)
    return text_result(json.dumps(value, sort_keys=True), is_error=not ok, structured=value)


def typed_failure(state, failure_class, message, **kwargs):
    return typed_result(
        False, state, failure_class,
        mismatches=[safe_diagnostic_text(message, 1024)], **kwargs,
    )


def safe_diagnostic_text(value, maximum=MAX_DIAGNOSTIC_TEXT_BYTES):
    text = str(value).replace("\x00", "")
    text = re.sub(
        r"(?i)\b(token|password|secret|api[_-]?key)\s*[:=]\s*\S+",
        r"\1=<redacted>", text,
    )
    text = re.sub(
        r"(?<![A-Za-z0-9_.-])(?:/sda/home|/home|/mnt|[A-Za-z]:\\)[^\s:'\"]+",
        "<path>", text,
    )
    lines = [line.rstrip() for line in text.splitlines()[-20:]]
    return "\n".join(lines).encode("utf-8", errors="replace")[-maximum:].decode(
        "utf-8", errors="replace",
    )


def safe_status_summary(status):
    allowed = {
        "phase", "event", "state", "completed_units", "total_units",
        "completed", "total",
    }
    records = []
    for line in str(status).splitlines()[-20:]:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(value, dict):
            continue
        compact = {key: value[key] for key in allowed if key in value}
        if compact:
            records.append(compact)
    return safe_diagnostic_text(
        json.dumps(records[-8:], sort_keys=True, separators=(",", ":")), 2048,
    )


def failure_result(state, exc, default_phase):
    if isinstance(exc, ToolError):
        return typed_failure(
            state, exc.failure_class, str(exc), failure_phase=exc.failure_phase
        )
    return typed_failure(
        state, "command_infra", type(exc).__name__, failure_phase=default_phase
    )


def require_token(value, name):
    if not isinstance(value, str) or not SAFE_TOKEN.fullmatch(value):
        raise ToolError(f"{name} must be a safe token")
    return value


def require_branch(value):
    if not isinstance(value, str) or not SAFE_BRANCH.fullmatch(value):
        raise ToolError("branch must be codex/<safe-route-name>")
    return value


def require_sha(value, name, pattern):
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise ToolError(f"{name} has an invalid digest")
    return value


def require_relpath(value, name, suffix, prefix="experience_docx/"):
    if not isinstance(value, str) or value.startswith("/") or ".." in Path(value).parts:
        raise ToolError(f"{name} must be a safe repository-relative path")
    if not value.startswith(prefix) or not value.endswith(suffix):
        raise ToolError(f"{name} has an invalid path contract")
    return value


def require_bool(value, name):
    if not isinstance(value, bool):
        raise ToolError(f"{name} must be boolean")
    return value


def require_int(value, name, minimum, maximum):
    if not isinstance(value, int) or isinstance(value, bool) or not minimum <= value <= maximum:
        raise ToolError(f"{name} must be in [{minimum}, {maximum}]")
    return value


def require_enum(value, name, choices):
    if value not in choices:
        raise ToolError(f"{name} must be one of {sorted(choices)}")
    return value


def require_terminal_tuple(value, name, *, allow_null=False, allow_null_decision=False):
    if value is None and allow_null:
        return None
    if not isinstance(value, dict) or set(value) != {"state", "decision", "authorizes"}:
        raise ToolError(f"{name} must contain state, decision, authorizes")
    decision = value["decision"]
    if decision is None and not allow_null_decision:
        raise ToolError(f"{name}.decision cannot be null")
    return {
        "state": require_token(value["state"], f"{name}.state"),
        "decision": None if decision is None else require_token(decision, f"{name}.decision"),
        "authorizes": require_token(value["authorizes"], f"{name}.authorizes"),
    }


def require_terminal_tuples(value):
    if not isinstance(value, list) or not 1 <= len(value) <= 8:
        raise ToolError("allowed_terminal_tuples must contain 1-8 tuples")
    result = [
        require_terminal_tuple(item, "allowed_terminal_tuples", allow_null_decision=True)
        for item in value
    ]
    if len({canonical_digest(item) for item in result}) != len(result):
        raise ToolError("allowed_terminal_tuples contains duplicates")
    return result


def first_operation_from_card(text):
    match = re.search(r"(?m)^- First operation:\s*([^\s]+)\s*$", text)
    if not match:
        raise ToolError("route card must contain one exact First operation field")
    return require_token(match.group(1), "First operation")


def validate_scientific_contract(value, route_id, operation_id, operation):
    if isinstance(value, dict) and value.get("schema_version") == 2:
        try:
            contract = science_contract.validate_scientific_contract_v2(
                value, route_id, operation_id,
            )
            scientific = science_contract.scientific_terminal_tuples(contract)
        except science_contract.ScientificContractError as exc:
            raise ToolError(str(exc)) from exc
        engineering = {
            "state": "FAILED_ENGINEERING", "decision": None, "authorizes": "NONE",
        }
        allowed = require_terminal_tuples(operation["allowed_terminal_tuples"])
        expected = [*scientific, engineering]
        if {
            canonical_digest(item) for item in allowed
        } != {
            canonical_digest(item) for item in expected
        }:
            raise ToolError(
                "scientific schema 2 allowed terminal tuples must be derived exactly"
            )
        return contract
    expected = {
        "schema_version", "route_id", "operation_id", "question",
        "population", "intervention", "primary_estimand", "controls",
        "uncertainty", "gates", "competing_explanation",
        "terminal_mapping", "disabled_actions",
    }
    if not isinstance(value, dict) or set(value) != expected or value["schema_version"] != 1:
        raise ToolError("scientific contract has an invalid top-level contract")
    if value["route_id"] != route_id or value["operation_id"] != operation_id:
        raise ToolError("scientific contract identity mismatch")
    for key in ("question", "competing_explanation"):
        if not isinstance(value[key], str) or not 16 <= len(value[key]) <= 2048:
            raise ToolError(f"scientific contract {key} must contain 16-2048 characters")
    population = value["population"]
    population_fields = {
        "evidence_role", "grouping_unit", "independent_group_count",
        "allow_confirmation", "allow_canary", "allow_locked_test",
    }
    if not isinstance(population, dict) or set(population) != population_fields:
        raise ToolError("scientific contract population is invalid")
    role = require_enum(
        population["evidence_role"], "population.evidence_role",
        {"engineering_debug", "development_screening", "confirmation", "sealed_final"},
    )
    require_token(population["grouping_unit"], "population.grouping_unit")
    require_int(population["independent_group_count"], "population.independent_group_count", 0, 10_000_000)
    permissions = {
        key: require_bool(population[key], f"population.{key}")
        for key in ("allow_confirmation", "allow_canary", "allow_locked_test")
    }
    if permissions["allow_locked_test"] and role != "sealed_final":
        raise ToolError("scientific contract locked test requires sealed_final role")
    if permissions["allow_confirmation"] and role not in {"confirmation", "sealed_final"}:
        raise ToolError("scientific contract confirmation access requires confirmation/sealed role")
    intervention = value["intervention"]
    if not isinstance(intervention, dict) or set(intervention) != {
        "primary_variable", "reference", "matched_budget", "fixed_factors",
    }:
        raise ToolError("scientific contract intervention is invalid")
    for key in ("primary_variable", "reference", "matched_budget"):
        if not isinstance(intervention[key], str) or not intervention[key].strip():
            raise ToolError(f"scientific contract intervention.{key} is empty")
    fixed = intervention["fixed_factors"]
    if not isinstance(fixed, list) or not fixed \
            or any(not isinstance(item, str) or not item.strip() for item in fixed):
        raise ToolError("scientific contract fixed_factors is empty")
    estimand = value["primary_estimand"]
    if not isinstance(estimand, dict) or set(estimand) != {
        "id", "metric", "direction", "aggregation", "unit",
    }:
        raise ToolError("scientific contract primary_estimand is invalid")
    require_token(estimand["id"], "primary_estimand.id")
    require_enum(estimand["direction"], "primary_estimand.direction", {"higher", "lower"})
    for key in ("metric", "aggregation", "unit"):
        if not isinstance(estimand[key], str) or not estimand[key].strip():
            raise ToolError(f"primary_estimand.{key} is empty")
    controls = value["controls"]
    if not isinstance(controls, list) or not controls \
            or any(not isinstance(item, str) or not item.strip() for item in controls):
        raise ToolError("scientific contract requires at least one matched/negative control")
    uncertainty = value["uncertainty"]
    if not isinstance(uncertainty, dict) or set(uncertainty) != {
        "method", "confidence_level", "independent_unit",
    } or uncertainty["confidence_level"] != 0.95:
        raise ToolError("scientific contract uncertainty is invalid")
    for key in ("method", "independent_unit"):
        if not isinstance(uncertainty[key], str) or not uncertainty[key].strip():
            raise ToolError(f"scientific contract uncertainty.{key} is empty")
    gates = value["gates"]
    if not isinstance(gates, list) or not gates:
        raise ToolError("scientific contract requires at least one gate")
    gate_ids = set()
    for index, gate in enumerate(gates):
        fields = {"id", "type", "estimand", "direction", "threshold", "decision_role"}
        if not isinstance(gate, dict) or set(gate) != fields:
            raise ToolError(f"scientific contract gates[{index}] is invalid")
        identifier = require_token(gate["id"], f"gates[{index}].id")
        if identifier in gate_ids:
            raise ToolError("scientific contract gate ids must be unique")
        gate_ids.add(identifier)
        require_enum(gate["type"], f"gates[{index}].type", {
            "integrity", "materiality", "safety", "coverage", "precision",
        })
        require_enum(gate["direction"], f"gates[{index}].direction", {"min", "max", "equal"})
        require_enum(gate["decision_role"], f"gates[{index}].decision_role", {
            "decisive", "inconclusive_only", "descriptive",
        })
        threshold = gate["threshold"]
        if not isinstance(threshold, (int, float, bool, str)) \
                or isinstance(threshold, str) and not threshold.strip():
            raise ToolError(f"scientific contract gates[{index}].threshold is invalid")
        if not isinstance(gate["estimand"], str) or not gate["estimand"].strip():
            raise ToolError(f"scientific contract gates[{index}].estimand is empty")
    terminal_mapping = value["terminal_mapping"]
    if not isinstance(terminal_mapping, dict) or set(terminal_mapping) != {
        "pass", "fail", "inconclusive",
    }:
        raise ToolError("scientific contract terminal_mapping is invalid")
    allowed = require_terminal_tuples(operation["allowed_terminal_tuples"])
    for label in ("pass", "fail", "inconclusive"):
        terminal = require_terminal_tuple(terminal_mapping[label], f"terminal_mapping.{label}")
        if terminal not in allowed:
            raise ToolError(f"scientific contract {label} terminal is absent from allowed tuples")
        if role in {"engineering_debug", "development_screening"} \
                and terminal["authorizes"] in {
                    "PROMOTION", "DEPLOYMENT", "LOCKED_TEST", "SEALED_FINAL",
                }:
            raise ToolError("development evidence cannot directly authorize promotion/final use")
    disabled = value["disabled_actions"]
    if not isinstance(disabled, list) or not disabled \
            or any(not isinstance(item, str) or not item.strip() for item in disabled):
        raise ToolError("scientific contract disabled_actions is empty")
    return value


def validate_contract_runtime_alignment(contract, spec, precision=None):
    population = contract["population"]
    if population["evidence_role"] != spec["evidence_role"]:
        raise ToolError("scientific/runtime evidence roles differ")
    permissions = {
        "allow_confirmation": population["allow_confirmation"],
        "allow_canary": population["allow_canary"],
        "allow_locked_test": population["allow_locked_test"],
    }
    if permissions != spec["protected_data_permissions"]:
        raise ToolError("scientific/runtime protected permissions differ")
    if precision is not None:
        if precision.get("schema_version") == 2:
            if contract.get("schema_version") != 2:
                raise ToolError("precision schema 2 requires scientific schema 2")
            if precision["primary_estimand_id"] != contract["primary_estimand"]["id"]:
                raise ToolError("scientific/precision primary estimands differ")
            if precision["independent_unit"] != contract["primary_estimand"]["unit"]:
                raise ToolError("scientific/precision independent units differ")
        elif population["independent_group_count"] != precision["independent_groups_available"]:
            raise ToolError("scientific/precision independent group counts differ")


def validate_committed_operation_bundle(bare_repo, route_commit, manifest,
                                        operation_id, context):
    try:
        spec_path = runtime_spec_relpath(operation_id)
        spec_raw = git_show(bare_repo, route_commit, spec_path)
        spec = validate_runtime_spec(json.loads(spec_raw), manifest, operation_id)
        if context["route_manifest_schema_version"] >= 5 and spec["schema_version"] != 2:
            raise ToolError("canonical manifest requires runtime schema 2")
        asset = None
        if spec["asset_manifest_relpath"] is not None:
            asset = validate_asset_manifest(
                json.loads(git_show(bare_repo, route_commit, spec["asset_manifest_relpath"])),
                spec,
            )
        capability = None
        capability_reuse = None
        capability_path = spec["engineering_contract"]["capability_profile_relpath"]
        if capability_path is not None:
            capability = validate_model_capability(
                json.loads(git_show(bare_repo, route_commit, capability_path)), spec, asset,
            )
            if capability.get("schema_version") == 2:
                try:
                    records = capability_registry.load_records(
                        git_show(
                            bare_repo, route_commit, capability_registry.REGISTRY_RELPATH,
                        ).splitlines(),
                        evidence_exists=lambda relpath: bool(
                            git_show_bytes(bare_repo, route_commit, relpath)
                        ),
                        read_evidence=lambda relpath: git_show_bytes(
                            bare_repo, route_commit, relpath,
                        ),
                    )
                    capability_reuse = capability_registry.lookup(
                        records, capability["reuse_identity"],
                    )
                except capability_registry.CapabilityRegistryError as exc:
                    raise ToolError(f"capability registry is invalid: {exc}") from exc
        contract = None
        if context["route_manifest_schema_version"] >= 5:
            contract = validate_scientific_contract(
                json.loads(git_show(
                    bare_repo, route_commit, context["scientific_contract_relpath"],
                )),
                manifest["route_id"], operation_id, manifest["operations"][operation_id],
            )
        precision = None
        precision_path = spec["precision_contract"]["certificate_relpath"]
        if precision_path is not None:
            precision = validate_precision_certificate(
                json.loads(git_show(bare_repo, route_commit, precision_path)), spec,
                contract,
            )
        if contract is not None:
            validate_contract_runtime_alignment(contract, spec, precision)
        context.update({
            "runtime_spec_digest": hashlib.sha256(spec_raw.encode()).hexdigest(),
            "engineering_contract_mode": spec["engineering_contract"]["mode"],
            "engineering_max_seconds": spec["engineering_contract"]["max_seconds"],
            "expected_wall_seconds": spec["expected_wall_seconds"],
            "precision_mode": spec["precision_contract"]["mode"],
            "capability_profile_id": None if capability is None else capability["profile_id"],
            "capability_reuse": capability_reuse,
            "precision_certificate_id": None if precision is None else precision["certificate_id"],
        })
        return spec
    except (RuntimeContractError, json.JSONDecodeError) as exc:
        raise ToolError(f"committed operation bundle is invalid: {exc}") from exc


def q(value):
    return shlex.quote(str(value))


def derive_remote_repo(route_id, output_id):
    seed = f"{route_id}\0{output_id}".encode()
    digest = hashlib.sha256(seed).hexdigest()[:16]
    prefix = f"{route_id[:32]}-{output_id[:24]}"[:56]
    return f"{REMOTE_REPOS}/{prefix}-{digest}"


def derive_session(route_id, mode, commit, output_id):
    seed = f"{route_id}\0{mode}\0{commit}\0{output_id}".encode()
    digest = hashlib.sha256(seed).hexdigest()[:12]
    return f"convir-{route_id[:18]}-{mode[:10]}-{output_id[:10]}-{digest}"[:64]


def run_local(args, *, timeout, phase):
    try:
        result = subprocess.run(args, text=True, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise ToolError(
            f"{phase} timed out", failure_phase=phase, failure_class="command_infra"
        ) from exc
    if result.returncode:
        detail = (result.stdout + result.stderr).strip()[:4096]
        raise ToolError(
            f"{phase} failed rc={result.returncode}: {detail}",
            failure_phase=phase,
            failure_class="command_infra",
        )
    return result.stdout.strip()


def inspect_local(args, *, timeout=30):
    try:
        result = subprocess.run(args, text=True, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "returncode": None, "output": "timeout"}
    return {
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "output": (result.stdout + result.stderr).strip()[:4096],
    }


def run_remote(body, *, timeout=120, phase="remote_transport"):
    """Send an internally generated script over one fixed, bounded SSH channel."""
    if not isinstance(body, str) or "\x00" in body:
        raise ToolError(
            "remote body must be NUL-free text",
            failure_phase=phase,
            failure_class="contract",
        )
    script = (
        "#!/usr/bin/env bash\nset -euo pipefail\n" + body.rstrip("\n") + "\n"
    ).encode("utf-8")
    if len(script) > MAX_REMOTE_SCRIPT_BYTES:
        raise ToolError(
            "remote body exceeds the fixed size limit",
            failure_phase=phase,
            failure_class="contract",
        )
    connect_timeout = max(1, min(int(timeout), 30))
    argv = [
        SSH, "-T", "-o", "BatchMode=yes",
        "-o", f"ConnectTimeout={connect_timeout}",
        REMOTE_HOST, REMOTE_BASH, "-s", "--",
    ]
    try:
        process = subprocess.Popen(
            argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
    except OSError as exc:
        raise ToolError(
            f"{phase} could not start",
            failure_phase=phase,
            failure_class="command_infra",
        ) from exc

    stdout = bytearray()
    stderr = bytearray()
    thread_errors = []
    store_limit = MAX_REMOTE_CAPTURE_BYTES + 1

    def drain(stream, target):
        try:
            while True:
                block = stream.read(8192)
                if not block:
                    break
                remaining = store_limit - len(target)
                if remaining > 0:
                    target.extend(block[:remaining])
        except OSError as exc:
            thread_errors.append(exc)
        finally:
            stream.close()

    def feed():
        try:
            process.stdin.write(script)
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
        return_code = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        process.kill()
        process.wait()
        for thread in threads:
            thread.join(timeout=5)
        raise ToolError(
            f"{phase} timed out; remote state is unknown",
            failure_phase=phase,
            failure_class="command_infra",
        ) from exc
    for thread in threads:
        thread.join(timeout=5)
    if any(thread.is_alive() for thread in threads) or thread_errors:
        raise ToolError(
            f"{phase} streams did not close cleanly",
            failure_phase=phase,
            failure_class="command_infra",
        )
    if len(stdout) > MAX_REMOTE_CAPTURE_BYTES or len(stderr) > MAX_REMOTE_CAPTURE_BYTES:
        raise ToolError(
            f"{phase} output exceeded the fixed capture limit",
            failure_phase=phase,
            failure_class="command_infra",
        )
    stdout_text = bytes(stdout).decode("utf-8", errors="replace")
    stderr_text = bytes(stderr).decode("utf-8", errors="replace")
    if return_code:
        detail = (stdout_text + stderr_text).strip()[:4096]
        raise ToolError(
            f"{phase} failed rc={return_code}: {detail}",
            failure_phase=phase,
            failure_class="command_infra",
        )
    return stdout_text.strip()


def github_refs(refs):
    output = run_local(
        ["git", "ls-remote", GITHUB_URL, *refs], timeout=60, phase="github_ref_fetch"
    )
    result = {}
    for line in output.splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[1] in refs and SHA40.fullmatch(fields[0]):
            result[fields[1]] = fields[0]
    if set(result) != set(refs):
        raise ToolError(
            "required GitHub refs are missing",
            failure_phase="github_ref_fetch",
            failure_class="command_infra",
        )
    return result


def git_show(repo, commit, path):
    return run_local(
        ["git", "-C", repo, "show", f"{commit}:{path}"],
        timeout=30,
        phase="local_git_verify",
    )


def git_show_bytes(repo, commit, path):
    try:
        result = subprocess.run(
            ["git", "-C", repo, "show", f"{commit}:{path}"],
            capture_output=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired as exc:
        raise ToolError(
            "Git blob read timed out",
            failure_phase="local_git_verify",
            failure_class="command_infra",
        ) from exc
    if result.returncode:
        raise ToolError(
            "Git blob read failed",
            failure_phase="local_git_verify",
            failure_class="command_infra",
        )
    return result.stdout


def git_object_exists(repo, commit, path):
    result = subprocess.run(
        ["git", "-C", repo, "cat-file", "-e", f"{commit}:{path}"],
        capture_output=True, timeout=30, check=False,
    )
    return result.returncode == 0


def rule_bundle_digest(repo, commit):
    digest = hashlib.sha256()
    for path in RULE_BUNDLE_RELPATHS:
        raw = git_show_bytes(repo, commit, path)
        digest.update(path.encode() + b"\0" + raw + b"\0")
    return digest.hexdigest()


def rule_compatibility_profile(repo, commit):
    try:
        value = json.loads(git_show_bytes(repo, commit, RULE_COMPATIBILITY_RELPATH))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ToolError("rule compatibility profile is invalid JSON") from exc
    required = {
        "schema_version", "compatibility_id", "compatible_prior_rules_commits",
    }
    if not isinstance(value, dict) or set(value) != required \
            or value.get("schema_version") != 1:
        raise ToolError("rule compatibility profile has an invalid field contract")
    compatibility_id = value.get("compatibility_id")
    prior = value.get("compatible_prior_rules_commits")
    if not isinstance(compatibility_id, str) \
            or not SAFE_TOKEN.fullmatch(compatibility_id) \
            or not isinstance(prior, list) or len(prior) > 64 \
            or len(prior) != len(set(prior)) \
            or any(not isinstance(item, str) or not SHA40.fullmatch(item) for item in prior):
        raise ToolError("rule compatibility profile identity is invalid")
    return value


def require_rule_compatibility(
    repo, recorded_commit, current_commit, recorded_digest, current_digest,
    *, expected_compatibility_id=None,
):
    """Accept exact rules or one explicit current-main compatibility declaration."""
    profile = None
    if git_object_exists(repo, current_commit, RULE_COMPATIBILITY_RELPATH):
        profile = rule_compatibility_profile(repo, current_commit)
    if profile is not None:
        compatibility_id = profile["compatibility_id"]
        if expected_compatibility_id is not None \
                and compatibility_id != expected_compatibility_id:
            raise ToolError("rule compatibility identity changed")
        if recorded_digest == current_digest \
                or recorded_commit in profile["compatible_prior_rules_commits"]:
            return compatibility_id
        raise ToolError(
            "canonical rule bundle changed without an explicit compatibility declaration"
        )
    if recorded_digest == current_digest:
        if expected_compatibility_id is not None:
            raise ToolError("rule compatibility profile disappeared")
        return f"exact-{current_digest}"
    raise ToolError("canonical rule bundle changed; one compatibility review is required")


def blob_sha(repo, commit, path):
    value = run_local(
        ["git", "-C", repo, "rev-parse", f"{commit}:{path}"],
        timeout=30,
        phase="local_git_verify",
    )
    return require_sha(value, "blob", SHA40)


def prepare_seeded_bare(path):
    if not LOCAL_GIT_SEED.is_dir():
        raise ToolError(
            "local Git seed is unavailable",
            failure_phase="local_git_prepare",
            failure_class="command_infra",
        )
    run_local(
        ["git", "-C", str(LOCAL_GIT_SEED), "rev-parse", "--git-dir"],
        timeout=30,
        phase="local_git_prepare",
    )
    run_local(
        ["git", "clone", "--quiet", "--bare", "--shared", str(LOCAL_GIT_SEED), path],
        timeout=30,
        phase="local_git_prepare",
    )


def fetch_verified_refs(repo, branch_ref, expected_branch, expected_main):
    run_local(
        [
            "git", "-C", repo, "fetch", "--quiet", "--no-tags", "--depth=1", GITHUB_URL,
            f"+{branch_ref}:refs/convir-verify/route",
            "+refs/heads/main:refs/convir-verify/main",
        ],
        timeout=120,
        phase="local_git_fetch",
    )
    observed_branch = run_local(
        ["git", "-C", repo, "rev-parse", "refs/convir-verify/route"],
        timeout=30,
        phase="local_git_verify",
    )
    observed_main = run_local(
        ["git", "-C", repo, "rev-parse", "refs/convir-verify/main"],
        timeout=30,
        phase="local_git_verify",
    )
    if observed_branch != expected_branch or observed_main != expected_main:
        raise ToolError("fetched GitHub refs do not match ls-remote")


def ensure_commit(repo, commit):
    try:
        result = subprocess.run(
            ["git", "-C", repo, "cat-file", "-e", f"{commit}^{{commit}}"],
            capture_output=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired as exc:
        raise ToolError(
            "local Git commit check timed out",
            failure_phase="local_git_verify",
            failure_class="command_infra",
        ) from exc
    if result.returncode == 0:
        return
    run_local(
        ["git", "-C", repo, "fetch", "--quiet", "--no-tags", "--depth=1", GITHUB_URL, commit],
        timeout=120,
        phase="local_git_fetch",
    )
    run_local(
        ["git", "-C", repo, "cat-file", "-e", f"{commit}^{{commit}}"],
        timeout=30,
        phase="local_git_verify",
    )


def parse_manifest(value, branch, route_commit, current_main, bare_repo, operation_id):
    legacy_top = {
        "schema_version", "route_id", "rules_commit",
        "route_card_relpath", "operations",
    }
    current_top = legacy_top | {"scientific_contract_relpaths"}
    compiled_top = current_top | {
        "program_contract_relpath", "program_contract_sha256",
        "experiment_spec_relpath", "experiment_spec_sha256",
    }
    if not isinstance(value, dict) or value.get("schema_version") not in SUPPORTED_MANIFEST_SCHEMA_VERSIONS:
        raise ToolError("route operations manifest has an unsupported schema")
    manifest_schema = value["schema_version"]
    expected_top = {
        4: legacy_top, 5: current_top, 6: compiled_top,
    }[manifest_schema]
    if set(value) != expected_top:
        raise ToolError("route operations manifest has an invalid top-level contract")
    route_id = require_token(value["route_id"], "route_id")
    rules_commit = require_sha(value["rules_commit"], "rules_commit", SHA40)
    route_card = require_relpath(
        value["route_card_relpath"], "route_card_relpath", ".md",
        prefix="experience_docx/experiment_cards/",
    )
    route_card_blob = blob_sha(bare_repo, route_commit, route_card)
    scientific_contract = None
    scientific_contract_blob = None
    scientific_contract_value = None
    if manifest_schema >= 5:
        contract_paths = value["scientific_contract_relpaths"]
        if not isinstance(contract_paths, dict) or set(contract_paths) != set(value["operations"]):
            raise ToolError("scientific contract paths must map every operation exactly")
        scientific_contract = require_relpath(
            contract_paths[operation_id], "scientific_contract_relpaths operation", ".json",
            prefix="experience_docx/scientific_contracts/",
        )
        scientific_contract_blob = blob_sha(bare_repo, route_commit, scientific_contract)
        scientific_contract_value = json.loads(git_show(bare_repo, route_commit, scientific_contract))
    operations = value["operations"]
    if not isinstance(operations, dict) or not 1 <= len(operations) <= 8:
        raise ToolError("operations must contain 1-8 entries")
    if operation_id not in operations:
        raise ToolError("operation_id is absent from the manifest")
    operation = operations[operation_id]
    operation_fields = {
        "runner_relpath", "mode", "require_gpu", "output_id",
        "closeout_filename", "prior_closeout_relpath",
        "prior_terminal_tuple", "allowed_terminal_tuples", "workspace_policy",
        "output_policy", "monitor_profile", "heartbeat_timeout_seconds",
        "min_free_gpu_mib", "max_gpu_utilization_pct",
    }
    if not isinstance(operation, dict) or set(operation) != operation_fields:
        raise ToolError("selected operation has an invalid field contract")
    if manifest_schema >= 5:
        scientific_contract_value = validate_scientific_contract(
            scientific_contract_value, route_id, operation_id, operation,
        )
    runner = require_relpath(
        operation["runner_relpath"], "runner_relpath", ".sh",
        prefix="experience_docx/tools/run_",
    )
    mode = require_token(operation["mode"], "mode")
    require_gpu = require_bool(operation["require_gpu"], "require_gpu")
    prior_path = operation["prior_closeout_relpath"]
    prior_tuple = operation["prior_terminal_tuple"]
    if (prior_path is None) != (prior_tuple is None):
        raise ToolError("prior closeout path and tuple must both be null or both be set")
    if prior_path is not None:
        prior_path = require_relpath(prior_path, "prior_closeout_relpath", ".json")
        prior_tuple = require_terminal_tuple(prior_tuple, "prior_terminal_tuple")
        if (
            prior_tuple["state"] != "COMPLETED_GATE_PASS"
            or prior_tuple["authorizes"] != operation_id
        ):
            raise ToolError("prior closeout must authorize the selected operation id")
        prior = json.loads(git_show(bare_repo, route_commit, prior_path))
        actual = {key: prior.get(key) for key in prior_tuple}
        if prior.get("route_id") != route_id or actual != prior_tuple:
            raise ToolError("prior closeout does not match its sealed terminal tuple")
    elif manifest_schema == 4:
        if first_operation_from_card(git_show(bare_repo, route_commit, route_card)) != operation_id:
            raise ToolError("selected operation is not the frozen first operation")
    else:
        if not isinstance(scientific_contract_value, dict) \
                or scientific_contract_value.get("route_id") != route_id \
                or scientific_contract_value.get("operation_id") != operation_id:
            raise ToolError("scientific contract identity or first operation mismatch")
    if manifest_schema == 6:
        program_path = require_relpath(
            value["program_contract_relpath"], "program_contract_relpath", ".json",
            prefix="experience_docx/research_programs/",
        )
        spec_path = require_relpath(
            value["experiment_spec_relpath"], "experiment_spec_relpath", ".json",
            prefix="experience_docx/experiment_specs/",
        )
        program_raw = git_show_bytes(bare_repo, route_commit, program_path)
        spec_raw = git_show_bytes(bare_repo, route_commit, spec_path)
        try:
            source_spec = json.loads(spec_raw)
        except json.JSONDecodeError as exc:
            raise ToolError("experiment spec JSON is invalid") from exc
        if not isinstance(source_spec, dict) or source_spec.get("schema_version") != 2:
            raise ToolError(
                "runnable manifest schema 6 requires experiment spec schema 2"
            )
        if hashlib.sha256(program_raw).hexdigest() != require_sha(
                value["program_contract_sha256"], "program_contract_sha256", SHA256):
            raise ToolError("program contract SHA-256 mismatch")
        if hashlib.sha256(spec_raw).hexdigest() != require_sha(
                value["experiment_spec_sha256"], "experiment_spec_sha256", SHA256):
            raise ToolError("experiment spec SHA-256 mismatch")
        try:
            import experiment_spec_compiler as compiler
            bundle = compiler.compile_bundle(
                spec_relpath=spec_path,
                spec_raw=spec_raw,
                program_raw=program_raw,
                evidence_exists=lambda relpath: git_object_exists(
                    bare_repo, current_main, relpath,
                ),
            )
        except Exception as exc:
            if isinstance(exc, ToolError):
                raise
            raise ToolError(f"compiled experiment bundle is invalid: {exc}") from exc
        mismatches = [
            relpath for relpath, expected in bundle.items()
            if git_show_bytes(bare_repo, route_commit, relpath) != expected
        ]
        if mismatches:
            raise ToolError(f"compiled experiment bundle drift: {mismatches[:8]}")
    recorded_rules = rule_bundle_digest(bare_repo, rules_commit)
    current_rules = rule_bundle_digest(bare_repo, current_main)
    rules_compatibility_id = require_rule_compatibility(
        bare_repo, rules_commit, current_main, recorded_rules, current_rules,
    )
    runner_raw = git_show_bytes(bare_repo, route_commit, runner)
    runner_sha = hashlib.sha256(runner_raw).hexdigest()
    min_free = require_int(operation["min_free_gpu_mib"], "min_free_gpu_mib", 0, 1048576)
    max_util = require_int(operation["max_gpu_utilization_pct"], "max_gpu_utilization_pct", 0, 100)
    if require_gpu and min_free < 1:
        raise ToolError("GPU operations require positive min_free_gpu_mib")
    if not require_gpu and (min_free != 0 or max_util != 100):
        raise ToolError("non-GPU operations require 0 MiB and 100% thresholds")
    output_id = require_token(operation["output_id"], "output_id")
    closeout = operation["closeout_filename"]
    if not isinstance(closeout, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+_closeout\.json", closeout):
        raise ToolError("closeout_filename must end with _closeout.json")
    context = {
        "schema_version": SCHEMA_VERSION,
        "route_manifest_schema_version": manifest_schema,
        "branch": branch,
        "route_branch_commit": route_commit,
        "current_rules_commit": current_main,
        "route_id": route_id,
        "operation_id": operation_id,
        "remote_repo": derive_remote_repo(route_id, output_id),
        "run_root": f"{REMOTE_RUNS}/{route_id}",
        "route_card_relpath": route_card,
        "route_card_blob": route_card_blob,
        "scientific_contract_relpath": scientific_contract,
        "scientific_contract_blob": scientific_contract_blob,
        "scientific_contract_digest": (
            canonical_digest(scientific_contract_value)
            if scientific_contract_value is not None else None
        ),
        "rules_commit": rules_commit,
        "rules_bundle_digest": current_rules,
        "rules_compatibility_id": rules_compatibility_id,
        "runner_relpath": runner,
        "runner_sha256": runner_sha,
        "mode": mode,
        "require_gpu": require_gpu,
        "output_id": output_id,
        "output_path": f"{REMOTE_RUNS}/{route_id}/{output_id}",
        "closeout_filename": closeout,
        "closeout_path": f"{derive_remote_repo(route_id, output_id)}/experience_docx/experiment_logs/{route_id}/{closeout}",
        "prior_closeout_relpath": prior_path,
        "prior_terminal_tuple": prior_tuple,
        "allowed_terminal_tuples": require_terminal_tuples(operation["allowed_terminal_tuples"]),
        "workspace_policy": require_enum(
            operation["workspace_policy"], "workspace_policy",
            {"fresh_route", "exact_continuation"},
        ),
        "output_policy": require_enum(
            operation["output_policy"], "output_policy", {"new", "exact_resume"}
        ),
        "monitor_profile": require_enum(
            operation["monitor_profile"], "monitor_profile", set(MONITOR_PROFILES)
        ),
        "heartbeat_timeout_seconds": require_int(
            operation["heartbeat_timeout_seconds"], "heartbeat_timeout_seconds", 30, 86400
        ),
        "min_free_gpu_mib": min_free,
        "max_gpu_utilization_pct": max_util,
    }
    context["session"] = derive_session(route_id, mode, route_commit, output_id)
    return context


def load_operation(args):
    if args.get("schema_version") != SCHEMA_VERSION:
        raise ToolError(f"schema_version must be {SCHEMA_VERSION}")
    branch = require_branch(args.get("branch"))
    route_commit = require_sha(args.get("route_branch_commit"), "route_branch_commit", SHA40)
    operation_id = require_token(args.get("operation_id"), "operation_id")
    branch_ref = f"refs/heads/{branch}"
    refs = github_refs([branch_ref, "refs/heads/main"])
    if refs[branch_ref] != route_commit:
        raise ToolError("route branch HEAD does not match route_branch_commit")
    with tempfile.TemporaryDirectory(prefix="convir-ops-plan-") as temporary:
        bare_repo = str(Path(temporary) / "repo.git")
        prepare_seeded_bare(bare_repo)
        fetch_verified_refs(
            bare_repo, branch_ref, route_commit, refs["refs/heads/main"]
        )
        manifest_raw = git_show(bare_repo, route_commit, ROUTE_OPERATIONS_RELPATH)
        if len(manifest_raw.encode()) > MAX_MANIFEST_BYTES:
            raise ToolError("route_operations.json exceeds 16 KiB")
        manifest = json.loads(manifest_raw)
        if manifest.get("schema_version") not in SUPPORTED_MANIFEST_SCHEMA_VERSIONS:
            raise ToolError("route manifest schema is unsupported")
        rules_commit = manifest.get("rules_commit") if isinstance(manifest, dict) else None
        if isinstance(rules_commit, str) and SHA40.fullmatch(rules_commit):
            ensure_commit(bare_repo, rules_commit)
        context = parse_manifest(
            manifest, branch, route_commit, refs["refs/heads/main"], bare_repo, operation_id
        )
        validate_committed_operation_bundle(
            bare_repo, route_commit, manifest, operation_id, context,
        )
    return manifest, operation_id, context


def state_secret():
    ensure_state_directory()
    path = STATE_DIR / "hmac.key"
    try:
        fd = os.open(
            path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600,
        )
    except FileExistsError:
        fd = secure_state_file_descriptor(path, os.O_RDONLY, "MCP signing key")
        with os.fdopen(fd, "rb") as handle:
            value = handle.read(33)
    else:
        value = os.urandom(32)
        with os.fdopen(fd, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        fsync_directory(STATE_DIR)
    if len(value) != 32:
        raise ToolError("MCP signing key is invalid", failure_class="command_infra")
    return value


def sign(payload):
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hmac.new(state_secret(), raw, hashlib.sha256).hexdigest()


def ensure_state_directory():
    STATE_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        observed = STATE_DIR.lstat()
    except OSError as exc:
        raise ToolError("MCP state directory is unavailable", failure_class="command_infra") from exc
    if not stat.S_ISDIR(observed.st_mode) or stat.S_ISLNK(observed.st_mode) \
            or observed.st_uid != os.getuid() or stat.S_IMODE(observed.st_mode) != 0o700:
        raise ToolError(
            "MCP state directory must be an owned non-symlink 0700 directory",
            failure_class="command_infra",
        )


def fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def secure_state_file_descriptor(path, flags, name):
    ensure_state_directory()
    try:
        descriptor = os.open(path, flags | os.O_NOFOLLOW | os.O_CLOEXEC)
    except OSError as exc:
        raise ToolError(f"{name} is unavailable", failure_class="command_infra") from exc
    observed = os.fstat(descriptor)
    if not stat.S_ISREG(observed.st_mode) or observed.st_uid != os.getuid() \
            or stat.S_IMODE(observed.st_mode) != 0o600:
        os.close(descriptor)
        raise ToolError(
            f"{name} must be an owned non-symlink 0600 regular file",
            failure_class="command_infra",
        )
    return descriptor


def record_mac(value):
    protected = {key: item for key, item in value.items() if key != "record_mac"}
    raw = b"convir-ops-record-v2\0" + json.dumps(
        protected, sort_keys=True, separators=(",", ":"),
    ).encode()
    return hmac.new(state_secret(), raw, hashlib.sha256).hexdigest()


def seal_record(value):
    value["record_schema_version"] = 2
    value["record_mac"] = record_mac(value)


def verify_or_migrate_record(value):
    if value.get("record_schema_version") == 2:
        observed = value.get("record_mac")
        if not isinstance(observed, str) or not hmac.compare_digest(observed, record_mac(value)):
            raise ToolError("record state integrity check failed")
        revision = value.get("revision")
        if not isinstance(revision, int) or revision < 0:
            raise ToolError("record revision is invalid")
        return
    if "record_schema_version" in value or "record_mac" in value or "revision" in value:
        raise ToolError("record state integrity contract is invalid")
    value["revision"] = 0
    value["legacy_state_migrated_at"] = int(time.time())
    seal_record(value)


def record_path(kind, token):
    require_sha(token, kind, SHA256)
    return STATE_DIR / f"{kind}-{token}.json"


def write_new_record(kind, payload, extra):
    token = sign(payload)
    path = record_path(kind, token)
    ensure_state_directory()
    value = {"payload": payload, **extra, "revision": 0}
    seal_record(value)
    try:
        fd = os.open(
            path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600,
        )
    except FileExistsError as exc:
        raise ToolError(f"{kind} token collision", failure_class="command_infra") from exc
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, separators=(",", ":"))
        handle.flush()
        os.fsync(handle.fileno())
    fsync_directory(STATE_DIR)
    return token


@contextmanager
def locked_record(kind, token):
    path = record_path(kind, token)
    try:
        descriptor = secure_state_file_descriptor(path, os.O_RDWR, f"{kind} record")
    except ToolError as exc:
        raise ToolError(f"{kind} is unknown, expired, or insecure") from exc
    with os.fdopen(descriptor, "r+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            value = json.load(handle)
            if not hmac.compare_digest(token, sign(value.get("payload"))):
                raise ToolError(f"{kind} integrity check failed")
            verify_or_migrate_record(value)
            try:
                yield value
            finally:
                value["revision"] += 1
                seal_record(value)
                handle.seek(0)
                handle.truncate()
                json.dump(value, handle, sort_keys=True, separators=(",", ":"))
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def tool_plan_manifest(args):
    try:
        manifest, operation_id, context = load_operation(args)
        now = int(time.time())
        payload = {
            "context": context,
            "issued_at": now,
            "expires_at": now + PLAN_TTL_SECONDS,
            "nonce": uuid.uuid4().hex,
        }
        token = write_new_record("plan", payload, {"receipt": None})
        return typed_result(
            True,
            "PLAN_READY",
            observed={
                "operation_id": operation_id,
                "manifest_digest": canonical_digest(manifest),
                "route_id": context["route_id"],
                "remote_repo": context["remote_repo"],
                "output_path": context["output_path"],
                "session": context["session"],
                "rules_bundle_digest": context["rules_bundle_digest"],
            },
            expected={
                "route_commit": context["route_branch_commit"],
                "route_card_blob": context["route_card_blob"],
                "runner_sha256": context["runner_sha256"],
            },
            next_actions=["convir_route_start"],
            plan_token=token,
            plan_expires_at=payload["expires_at"],
        )
    except (json.JSONDecodeError, TypeError) as exc:
        return failure_result("PLAN_REJECTED", ToolError(str(exc)), "local_git_manifest")
    except Exception as exc:
        return failure_result("PLAN_REJECTED", exc, "github_ref_fetch")


def verify_live_context(context):
    refs = github_refs([f"refs/heads/{context['branch']}", "refs/heads/main"])
    if refs[f"refs/heads/{context['branch']}"] != context["route_branch_commit"]:
        raise ToolError("route branch advanced after planning")
    current = refs["refs/heads/main"]
    if current == context["current_rules_commit"]:
        return
    with tempfile.TemporaryDirectory(prefix="convir-ops-live-rules-") as temporary:
        repo = str(Path(temporary) / "repo.git")
        prepare_seeded_bare(repo)
        ensure_commit(repo, current)
        current_digest = rule_bundle_digest(repo, current)
        try:
            require_rule_compatibility(
                repo,
                context["current_rules_commit"],
                current,
                context["rules_bundle_digest"],
                current_digest,
                expected_compatibility_id=context.get("rules_compatibility_id"),
            )
        except ToolError as exc:
            raise ToolError(
                "canonical rules changed after planning; create one fresh plan"
            ) from exc


def gpu_probe_body(context, gpu_index=None):
    """Return a strict, bounded GPU query that preserves failure identity."""
    return "\n".join([
        f"NVIDIA_SMI={q(NVIDIA_SMI)}",
        f"MIN_FREE={int(context['min_free_gpu_mib'])}",
        f"MAX_UTIL={int(context['max_gpu_utilization_pct'])}",
        f"GPU_TARGET={q(gpu_index if gpu_index is not None else '')}",
        f"GPU_SUMMARY_LIMIT={GPU_SUMMARY_LIMIT}",
        'test -x "$NVIDIA_SMI" || { echo "CONVIR_OPS_GPU_QUERY_FAILED reason=binary_missing"; exit 76; }',
        'GPU_QUERY_OUT=$(mktemp)',
        'GPU_QUERY_ERR=$(mktemp)',
        "trap 'rm -f -- \"$GPU_QUERY_OUT\" \"$GPU_QUERY_ERR\"' EXIT",
        'LAST_KIND=unknown',
        'LAST_QUERY_RC=0',
        'LAST_QUERY_ERR_BYTES=0',
        'LAST_QUERY_ERR_SHA=none',
        'LAST_QUERY_ERR_TEXT=none',
        'LAST_PARSED=""',
        'for attempt in 1 2; do',
        '  : >"$GPU_QUERY_OUT"',
        '  : >"$GPU_QUERY_ERR"',
        '  GPU_ARGS=(--query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits)',
        '  test -z "$GPU_TARGET" || GPU_ARGS=(-i "$GPU_TARGET" "${GPU_ARGS[@]}")',
        '  LAST_QUERY_RC=0',
        '  "$NVIDIA_SMI" "${GPU_ARGS[@]}" >"$GPU_QUERY_OUT" 2>"$GPU_QUERY_ERR" || LAST_QUERY_RC=$?',
        '  if test "$LAST_QUERY_RC" -ne 0; then',
        '    LAST_KIND=query_failed',
        '    LAST_QUERY_ERR_BYTES=$(wc -c <"$GPU_QUERY_ERR")',
        "    LAST_QUERY_ERR_SHA=$(sha256sum \"$GPU_QUERY_ERR\" | awk '{print $1}')",
        '    LAST_QUERY_ERR_TEXT=$(<"$GPU_QUERY_ERR")',
        "    LAST_QUERY_ERR_TEXT=${LAST_QUERY_ERR_TEXT//$'\r'/ }",
        "    LAST_QUERY_ERR_TEXT=${LAST_QUERY_ERR_TEXT//$'\n'/ }",
        '    LAST_QUERY_ERR_TEXT=${LAST_QUERY_ERR_TEXT:0:512}',
        '  else',
        '    PARSE_RC=0',
        "    LAST_PARSED=$(awk -F, -v min=\"$MIN_FREE\" -v max=\"$MAX_UTIL\" -v target=\"$GPU_TARGET\" -v limit=\"$GPU_SUMMARY_LIMIT\" '",
        '      function trim(value) { gsub(/^[[:space:]]+|[[:space:]]+$/, "", value); return value }',
        '      {',
        '        if (NF != 3) { bad=1; next }',
        '        idx=trim($1); free=trim($2); util=trim($3)',
        '        if (idx !~ /^[0-9]+$/ || free !~ /^[0-9]+$/ || util !~ /^[0-9]+$/) { bad=1; next }',
        '        if (seen[idx]++ || util + 0 > 100 || (target != "" && idx != target)) { bad=1; next }',
        '        total++',
        '        if (shown < limit) {',
        '          summary = summary (shown ? ";" : "") idx ":" (free + 0) ":" (util + 0)',
        '          shown++',
        '        }',
        '        if (selected == "" && free + 0 >= min && util + 0 <= max) selected=idx',
        '      }',
        '      END {',
        '        if (bad || total == 0) exit 65',
        '        printf "CONVIR_OPS_GPU_SUMMARY rows=%d total=%d data=%s\\n", shown, total, summary',
        '        if (selected != "") printf "CONVIR_OPS_GPU_OK index=%s\\n", selected',
        '        else print "CONVIR_OPS_RESOURCE_WAIT_REQUIRED"',
        "      }' \"$GPU_QUERY_OUT\") || PARSE_RC=$?",
        '    if test "$PARSE_RC" -ne 0; then',
        '      LAST_KIND=unparseable',
        "    elif printf '%s\\n' \"$LAST_PARSED\" | grep -q '^CONVIR_OPS_GPU_OK index=[0-9]\\+$'; then",
        '      LAST_KIND=ok',
        '      break',
        '    else',
        '      LAST_KIND=resource_wait',
        '    fi',
        '  fi',
        f'  test "$attempt" = 2 || sleep {GPU_PROBE_RETRY_DELAY_SECONDS}',
        'done',
        'case "$LAST_KIND" in',
        "  ok) printf '%s\\n' \"$LAST_PARSED\" ;;",
        "  resource_wait) printf '%s\\n' \"$LAST_PARSED\"; exit 75 ;;",
        '  unparseable) echo "CONVIR_OPS_GPU_QUERY_UNPARSEABLE"; exit 77 ;;',
        "  query_failed) printf 'CONVIR_OPS_GPU_QUERY_FAILED rc=%s stderr_bytes=%s stderr_sha256=%s stderr_text=%s\\n' \"$LAST_QUERY_RC\" \"$LAST_QUERY_ERR_BYTES\" \"$LAST_QUERY_ERR_SHA\" \"$LAST_QUERY_ERR_TEXT\"; exit 76 ;;",
        '  *) echo "CONVIR_OPS_GPU_QUERY_FAILED reason=internal_state"; exit 76 ;;',
        'esac',
    ])


def parse_gpu_summary(output):
    summaries = re.findall(
        r"(?m)^CONVIR_OPS_GPU_SUMMARY rows=(\d+) total=(\d+) data=([0-9:;]*)$",
        output,
    )
    if len(summaries) != 1:
        raise ToolError(
            "GPU probe summary is missing, duplicated, or malformed",
            failure_phase="resource_preflight",
            failure_class="command_infra",
        )
    rows_text, total_text, data = summaries[0]
    rows, total = int(rows_text), int(total_text)
    records = []
    if data:
        for item in data.split(";"):
            fields = item.split(":")
            if len(fields) != 3 or any(not field.isdigit() for field in fields):
                raise ToolError(
                    "GPU summary record is malformed",
                    failure_phase="resource_preflight", failure_class="command_infra",
                )
            records.append({
                "index": int(fields[0]), "free_mib": int(fields[1]),
                "utilization_pct": int(fields[2]),
            })
    if rows != len(records) or not 1 <= rows <= GPU_SUMMARY_LIMIT or total < rows:
        raise ToolError(
            "GPU summary cardinality is invalid",
            failure_phase="resource_preflight", failure_class="command_infra",
        )
    if len({item["index"] for item in records}) != len(records) or any(
            item["utilization_pct"] > 100 for item in records):
        raise ToolError(
            "GPU summary values are invalid",
            failure_phase="resource_preflight", failure_class="command_infra",
        )
    return {
        "rows": records, "total_gpu_count": total,
        "summary_truncated": total > rows,
    }


def parse_gpu(output):
    summary = parse_gpu_summary(output)
    matches = re.findall(r"(?m)^CONVIR_OPS_GPU_OK index=(\d+)$", output)
    if len(matches) != 1:
        raise ToolError(
            "GPU probe success marker is missing, duplicated, or malformed",
            failure_phase="resource_preflight", failure_class="command_infra",
        )
    selected = int(matches[0])
    if not summary["summary_truncated"] and selected not in {
            item["index"] for item in summary["rows"]}:
        raise ToolError(
            "selected GPU is absent from the complete summary",
            failure_phase="resource_preflight", failure_class="command_infra",
        )
    summary["index"] = selected
    return summary


def gpu_probe_failure(exc):
    detail = str(exc)
    observed = {"runner_started": False}
    if "CONVIR_OPS_RESOURCE_WAIT_REQUIRED" in detail:
        try:
            observed["gpu_summary"] = parse_gpu_summary(detail)
        except ToolError:
            pass
        return typed_failure(
            "RESOURCE_WAIT_REQUIRED", "command_infra",
            "no GPU currently satisfies the frozen resource gate",
            observed=observed, expected={"runner_started": False},
            next_actions=["convir_route_start"], retry_after_seconds=30,
            failure_phase="resource_preflight",
        )
    if "CONVIR_OPS_GPU_QUERY_UNPARSEABLE" in detail:
        message = "nvidia-smi output did not satisfy the strict GPU metrics contract"
    elif "CONVIR_OPS_GPU_QUERY_FAILED" in detail:
        marker = re.search(r"CONVIR_OPS_GPU_QUERY_FAILED[^\r\n]*", detail)
        message = marker.group(0) if marker else "nvidia-smi query failed"
    else:
        message = safe_diagnostic_text(detail, 1024)
    return typed_failure(
        "GPU_RESOURCE_PROBE_FAILED", "command_infra", message,
        observed=observed, expected={"runner_started": False},
        next_actions=["engineering_review_once"], failure_phase="resource_preflight",
    )


def atomic_start_body(context, gpu_index):
    lines = []
    if gpu_index is not None:
        lines.extend(gpu_probe_body(context, gpu_index).splitlines())
    lines.extend([
        f"REMOTE_REPO={q(context['remote_repo'])}",
        f"GITHUB_URL={q(GITHUB_URL)}",
        f"GIT_SEED={q(CLOUD_GIT_SEED)}",
        f"BRANCH={q(context['branch'])}",
        f"EXPECTED_COMMIT={q(context['route_branch_commit'])}",
        f"RUNNER={q(context['runner_relpath'])}",
        f"EXPECTED_RUNNER_SHA={q(context['runner_sha256'])}",
        f"RUN_ROOT={q(context['run_root'])}",
        f"OUTPUT_PATH={q(context['output_path'])}",
        f"CLOSEOUT={q(context['closeout_path'])}",
        f"SESSION={q(context['session'])}",
        f"TMUX={q(REMOTE_TMUX)}",
        f"WORKSPACE_POLICY={q(context['workspace_policy'])}",
        f"OUTPUT_POLICY={q(context['output_policy'])}",
        f"GPU_INDEX={q(gpu_index if gpu_index is not None else '')}",
        'FRESH_CREATED=0',
        'cleanup_fresh() { rc=$?; if test "$FRESH_CREATED" = 1; then rm -rf -- "$REMOTE_REPO"; echo CONVIR_OPS_FRESH_WORKSPACE_CLEANED; fi; exit "$rc"; }',
        'trap cleanup_fresh ERR',
        'test "$(git ls-remote "$GITHUB_URL" "refs/heads/$BRANCH" | awk \'NR==1 {print $1}\')" = "$EXPECTED_COMMIT"',
        'if test "$WORKSPACE_POLICY" = fresh_route; then',
        '  test ! -e "$REMOTE_REPO"',
        '  FRESH_CREATED=1',
        '  test -d "$GIT_SEED/.git"',
        '  git clone --quiet --shared --no-checkout "$GIT_SEED" "$REMOTE_REPO"',
        '  git -C "$REMOTE_REPO" remote rename origin seed',
        '  git -C "$REMOTE_REPO" remote add github "$GITHUB_URL"',
        '  git -C "$REMOTE_REPO" fetch --quiet --no-tags --depth=1 github "+refs/heads/$BRANCH:refs/remotes/github/$BRANCH"',
        '  git -C "$REMOTE_REPO" checkout --quiet -b "$BRANCH" "$EXPECTED_COMMIT"',
        'else',
        '  test -d "$REMOTE_REPO/.git"',
        '  test -z "$(git -C "$REMOTE_REPO" status --porcelain)"',
        '  test "$(git -C "$REMOTE_REPO" branch --show-current)" = "$BRANCH"',
        '  git -C "$REMOTE_REPO" fetch --quiet github "+refs/heads/$BRANCH:refs/remotes/github/$BRANCH"',
        '  git -C "$REMOTE_REPO" merge --quiet --ff-only "$EXPECTED_COMMIT"',
        'fi',
        'test "$(git -C "$REMOTE_REPO" rev-parse HEAD)" = "$EXPECTED_COMMIT"',
        'test -z "$(git -C "$REMOTE_REPO" status --porcelain)"',
        f'test -x {q(REMOTE_PYTHON)}',
        'test -f "$REMOTE_REPO/$RUNNER"',
        'RUNNER_SHA=$(sha256sum "$REMOTE_REPO/$RUNNER" | awk \'{print $1}\')',
        'test "$RUNNER_SHA" = "$EXPECTED_RUNNER_SHA"',
        '"$TMUX" has-session -t "$SESSION" 2>/dev/null && { echo CONVIR_OPS_SESSION_CONFLICT; exit 73; } || true',
        'if test "$OUTPUT_POLICY" = new; then',
        '  test ! -e "$OUTPUT_PATH"',
        '  test ! -e "$CLOSEOUT"',
        'else',
        '  test -d "$OUTPUT_PATH"',
        '  test ! -e "$CLOSEOUT"',
        'fi',
    ])
    lines.extend([
        f'"$TMUX" new-session -d -s "$SESSION" env EXPECTED_ROUTE_COMMIT="$EXPECTED_COMMIT" RUNNER_SHA256="$EXPECTED_RUNNER_SHA" MODE={q(context["mode"])} REMOTE_REPO="$REMOTE_REPO" RUN_ROOT="$RUN_ROOT" OUTPUT_PATH="$OUTPUT_PATH" RUN_ID={q(context["output_id"])} OUTPUT_ID={q(context["output_id"])} GPU="$GPU_INDEX" bash "$REMOTE_REPO/$RUNNER"',
        'trap - ERR',
        'echo "CONVIR_OPS_LAUNCH_OK session=$SESSION gpu=\${GPU_INDEX:-none}"',
    ])
    return "\n".join(lines)


def unknown_start_inspection_body(context):
    return "\n".join([
        f"REMOTE_REPO={q(context['remote_repo'])}",
        f"EXPECTED_COMMIT={q(context['route_branch_commit'])}",
        f"EXPECTED_BRANCH={q(context['branch'])}",
        f"RUNNER={q(context['runner_relpath'])}",
        f"EXPECTED_RUNNER_SHA={q(context['runner_sha256'])}",
        f"OUTPUT_PATH={q(context['output_path'])}",
        f"CLOSEOUT={q(context['closeout_path'])}",
        f"SESSION={q(context['session'])}",
        f"TMUX={q(REMOTE_TMUX)}",
        f"ROUTE_ID={q(context['route_id'])}",
        f"RUN_ID={q(context['output_id'])}",
        'repo=absent; runner=absent; dirty=-1',
        'if test -e "$REMOTE_REPO"; then',
        '  if test -d "$REMOTE_REPO/.git"; then',
        '    head=$(git -C "$REMOTE_REPO" rev-parse HEAD 2>/dev/null || true)',
        '    branch=$(git -C "$REMOTE_REPO" branch --show-current 2>/dev/null || true)',
        '    if test "$head" = "$EXPECTED_COMMIT" && test "$branch" = "$EXPECTED_BRANCH"; then repo=exact; else repo=mismatch; fi',
        '    dirty=$(git -C "$REMOTE_REPO" status --porcelain 2>/dev/null | wc -l)',
        '  else repo=partial; fi',
        'fi',
        'if test -f "$REMOTE_REPO/$RUNNER"; then',
        '  runner_sha=$(sha256sum "$REMOTE_REPO/$RUNNER" | awk \'{print $1}\')',
        '  if test "$runner_sha" = "$EXPECTED_RUNNER_SHA"; then runner=exact; else runner=mismatch; fi',
        'fi',
        'active=false; "$TMUX" has-session -t "$SESSION" 2>/dev/null && active=true || true',
        'output=absent; test ! -d "$OUTPUT_PATH" || output=present',
        'identity_path="$OUTPUT_PATH/control/lifecycle_identity.json"',
        f'json_states=$({q(REMOTE_PYTHON)} - "$identity_path" "$CLOSEOUT" "$ROUTE_ID" "$RUN_ID" "$EXPECTED_COMMIT" "$EXPECTED_RUNNER_SHA" <<\'PY\'',
        'import json, pathlib, sys',
        'identity_path, closeout_path, route_id, run_id, commit, runner = sys.argv[1:]',
        'expected = {"route_id": route_id, "run_id": run_id, "route_commit": commit, "runner_sha256": runner}',
        'def inspect(path):',
        '    candidate = pathlib.Path(path)',
        '    if not candidate.is_file():',
        '        return "absent"',
        '    try:',
        '        value = json.loads(candidate.read_text(encoding="utf-8"))',
        '    except Exception:',
        '        return "invalid"',
        '    return "valid" if isinstance(value, dict) and all(value.get(key) == item for key, item in expected.items()) else "invalid"',
        'print(inspect(identity_path), inspect(closeout_path))',
        'PY',
        ')',
        'read -r identity closeout <<<"$json_states"',
        'printf "CONVIR_OPS_START_INSPECTION repo=%s runner=%s active=%s output=%s identity=%s closeout=%s dirty=%s\\n" "$repo" "$runner" "$active" "$output" "$identity" "$closeout" "$dirty"',
    ])


def parse_unknown_start_inspection(output):
    match = re.search(
        r"(?m)^CONVIR_OPS_START_INSPECTION "
        r"repo=(absent|exact|mismatch|partial) "
        r"runner=(absent|exact|mismatch) active=(true|false) "
        r"output=(absent|present) identity=(absent|valid|invalid) "
        r"closeout=(absent|valid|invalid) dirty=(-?\d+)$",
        output,
    )
    if not match:
        raise ToolError(
            "unknown-start inspection marker is missing",
            failure_phase="start_recovery", failure_class="command_infra",
        )
    return {
        "repo": match.group(1),
        "runner": match.group(2),
        "active": match.group(3) == "true",
        "output": match.group(4),
        "identity": match.group(5),
        "closeout": match.group(6),
        "dirty_entries": int(match.group(7)),
    }


def abandoned_start_cleanup_body(context):
    return "\n".join([
        f"REMOTE_REPO={q(context['remote_repo'])}",
        f"EXPECTED_COMMIT={q(context['route_branch_commit'])}",
        f"EXPECTED_BRANCH={q(context['branch'])}",
        f"RUNNER={q(context['runner_relpath'])}",
        f"EXPECTED_RUNNER_SHA={q(context['runner_sha256'])}",
        f"OUTPUT_PATH={q(context['output_path'])}",
        f"CLOSEOUT={q(context['closeout_path'])}",
        f"SESSION={q(context['session'])}",
        f"TMUX={q(REMOTE_TMUX)}",
        f"REPO_ROOT={q(REMOTE_REPOS)}",
        'case "$REMOTE_REPO" in "$REPO_ROOT"/*) ;; *) exit 91 ;; esac',
        'test -d "$REMOTE_REPO/.git"',
        'test "$(git -C "$REMOTE_REPO" rev-parse HEAD)" = "$EXPECTED_COMMIT"',
        'test "$(git -C "$REMOTE_REPO" branch --show-current)" = "$EXPECTED_BRANCH"',
        'test -z "$(git -C "$REMOTE_REPO" status --porcelain)"',
        'test "$(sha256sum "$REMOTE_REPO/$RUNNER" | awk \'{print $1}\')" = "$EXPECTED_RUNNER_SHA"',
        '"$TMUX" has-session -t "$SESSION" 2>/dev/null && exit 92 || true',
        'test ! -e "$OUTPUT_PATH"',
        'test ! -e "$CLOSEOUT"',
        'rm -rf -- "$REMOTE_REPO"',
        'test ! -e "$REMOTE_REPO"',
        'echo CONVIR_OPS_ABANDONED_START_CLEANUP_OK',
    ])


def issue_receipt(context, gpu_index, launch_output):
    payload = {
        "context": context,
        "gpu_index": gpu_index,
        "launch_digest": hashlib.sha256(launch_output.encode()).hexdigest(),
        "issued_at": int(time.time()),
    }
    return write_new_record(
        "receipt", payload,
        {
            "launched": True, "finish_calls": 0, "finish_closed": None,
            "monitor_stale_count": 0, "terminal_closeout": None,
            "engineering_failure_resolution": None, "workload_verified": False,
            "finish_not_before_unix": 0, "pending_finish_response": None,
            "operator_observation_calls": 0,
            "operator_observation_not_before_unix": 0,
            "operator_observation_cache": None,
            "operator_terminal_detected": False,
            "operator_cancel_attempts": 0,
            "operator_cancel_request_id": None,
            "operator_cancel_state": None,
        },
    )


def recover_unknown_start(record):
    payload = record["payload"]
    context = payload["context"]
    if record.get("recovery_attempted"):
        return typed_failure(
            "START_STATE_UNKNOWN", "command_infra",
            "the single unknown-start recovery inspection was already consumed",
            observed={"remote_repo": context["remote_repo"], "runner_started": "unknown"},
            next_actions=["engineering_review_once"], failure_phase="start_recovery",
        )
    record["recovery_attempted"] = True
    try:
        inspection_output = run_remote(
            unknown_start_inspection_body(context), timeout=30, phase="start_recovery"
        )
        observed = parse_unknown_start_inspection(inspection_output)
    except ToolError as exc:
        return typed_failure(
            "START_STATE_UNKNOWN", "command_infra", str(exc),
            observed={"remote_repo": context["remote_repo"], "runner_started": "unknown"},
            next_actions=["engineering_review_once"], failure_phase="start_recovery",
        )
    launch_proven = (
        observed["repo"] == "exact"
        and observed["runner"] == "exact"
        and (
            observed["active"]
            or (
                observed["output"] == "present"
                and observed["identity"] != "invalid"
            )
            or observed["closeout"] == "valid"
        )
    )
    if launch_proven:
        receipt = issue_receipt(
            context, record.get("gpu_index"), inspection_output,
        )
        record["receipt"] = receipt
        return tool_finish({"receipt": receipt})
    no_runtime_signal = (
        not observed["active"]
        and observed["output"] == "absent"
        and observed["closeout"] == "absent"
    )
    if observed["repo"] == "absent" and no_runtime_signal:
        record["attempted"] = False
        record.pop("recovery_attempted", None)
        return typed_failure(
            "START_RETRY_READY", "command_infra",
            "inspection proved that launch did not create a workspace or output",
            observed=observed, expected={"runner_started": False},
            next_actions=["convir_route_start"], failure_phase="start_recovery",
        )
    if (
        context["workspace_policy"] == "fresh_route"
        and observed["repo"] == "exact"
        and observed["runner"] == "exact"
        and observed["dirty_entries"] == 0
        and no_runtime_signal
    ):
        try:
            cleanup_output = run_remote(
                abandoned_start_cleanup_body(context), timeout=30,
                phase="start_recovery_cleanup",
            )
            if cleanup_output.splitlines().count("CONVIR_OPS_ABANDONED_START_CLEANUP_OK") != 1:
                raise ToolError(
                    "abandoned-start cleanup marker is missing",
                    failure_phase="start_recovery_cleanup", failure_class="command_infra",
                )
        except ToolError as exc:
            return typed_failure(
                "START_STATE_UNKNOWN", "command_infra", str(exc),
                observed=observed, next_actions=["engineering_review_once"],
                failure_phase="start_recovery_cleanup",
            )
        record["attempted"] = False
        record.pop("recovery_attempted", None)
        return typed_failure(
            "START_RETRY_READY", "command_infra",
            "an exact clean abandoned workspace was removed before any runner output",
            observed={**observed, "workspace_cleanup": "completed"},
            expected={"runner_started": False}, next_actions=["convir_route_start"],
            failure_phase="start_recovery_cleanup",
        )
    return typed_failure(
        "START_STATE_UNKNOWN", "command_infra",
        "inspection could not prove a launch or a safe retry state",
        observed=observed, next_actions=["engineering_review_once"],
        failure_phase="start_recovery",
    )


def tool_start(args):
    token = args.get("plan_token")
    try:
        with locked_record("plan", token) as record:
            payload = record["payload"]
            if record.get("receipt"):
                return typed_result(
                    True, "LAUNCH_IDEMPOTENT",
                    observed={"receipt": record["receipt"]},
                    next_actions=["convir_route_finish"],
                    receipt=record["receipt"],
                )
            if record.get("attempted"):
                return recover_unknown_start(record)
            if time.time() > payload["expires_at"]:
                raise ToolError("plan has expired")
            context = payload["context"]
            verify_live_context(context)
            gpu_index = None
            if context["require_gpu"]:
                try:
                    gpu_probe = parse_gpu(
                        run_remote(gpu_probe_body(context), timeout=30, phase="resource_preflight")
                    )
                    gpu_index = gpu_probe["index"]
                    record["gpu_summary"] = gpu_probe
                except ToolError as exc:
                    return gpu_probe_failure(exc)
            record["gpu_index"] = gpu_index
            record["attempted"] = True
            try:
                output = run_remote(
                    atomic_start_body(context, gpu_index), timeout=150, phase="launch_command"
                )
            except ToolError as exc:
                if any(marker in str(exc) for marker in (
                        "CONVIR_OPS_RESOURCE_WAIT_REQUIRED",
                        "CONVIR_OPS_GPU_QUERY_FAILED",
                        "CONVIR_OPS_GPU_QUERY_UNPARSEABLE",
                )):
                    record["attempted"] = False
                    return gpu_probe_failure(exc)
                return typed_failure(
                    "START_STATE_UNKNOWN", "command_infra", str(exc),
                    observed={"remote_repo": context["remote_repo"], "runner_started": "unknown"},
                    next_actions=["convir_route_start"],
                    failure_phase="launch_command",
                )
            receipt = issue_receipt(context, gpu_index, output)
            record["receipt"] = receipt
            # A created process is not evidence that the scientific workload is
            # healthy. Spend one bounded observation window before reporting
            # the start result so preflight and unit-zero failures are visible.
            return tool_finish({"receipt": receipt})
    except Exception as exc:
        return failure_result("START_REJECTED", exc, "launch_command")


def scientific_archive_contract(context, terminal_closeout, finish_closed):
    """Return the operator-authored conclusion contract for new scientific routes."""
    terminal = terminal_closeout.get("terminal_tuple", {})
    state = terminal.get("state") if isinstance(terminal, dict) else None
    if finish_closed != "CLOSEOUT_VALIDATED" \
            or context.get("route_manifest_schema_version") != 6 \
            or not isinstance(state, str) or not state.startswith("COMPLETED_"):
        return None
    closeout_name = context["closeout_filename"]
    if not closeout_name.endswith("_closeout.json"):
        raise ToolError("validated closeout filename is not canonical")
    evidence_prefix = f"experience_docx/experiment_logs/{context['route_id']}"
    conclusion_name = closeout_name[:-len("_closeout.json")] + "_conclusion.json"
    return {
        "contract_path": context["route_card_relpath"],
        "closeout_path": f"{evidence_prefix}/{closeout_name}",
        "conclusion_path": f"{evidence_prefix}/{conclusion_name}",
        "conclusion_schema_version": CONCLUSION_SCHEMA_VERSION,
        "required_conclusion_fields": list(CONCLUSION_REQUIRED_FIELDS),
    }


def evidence_context(args):
    token = args.get("receipt")
    with locked_record("receipt", token) as record:
        if not record.get("launched"):
            raise ToolError("receipt has no successful launch")
        closed = record.get("finish_closed")
        if closed == "ENGINEERING_REVIEW_REQUIRED":
            raise ToolError(
                "engineering failure requires an explicit repair-or-archive decision before evidence access",
                failure_phase="engineering_review", failure_class="engineering_runtime",
            )
        if closed in {"ENGINEERING_REPAIR_AUTHORIZED", "ENGINEERING_AUTO_REPAIR_AUTHORIZED"}:
            raise ToolError(
                "engineering repair was selected; failed-run evidence remains cloud-only unless archive is separately chosen",
                failure_phase="engineering_review", failure_class="engineering_runtime",
            )
        if closed not in {"CLOSEOUT_VALIDATED", "ENGINEERING_ARCHIVE_AUTHORIZED"}:
            raise ToolError(
                "evidence access requires a validated terminal closeout",
                failure_phase="evidence_manifest", failure_class="contract",
            )
        terminal_closeout = record.get("terminal_closeout")
        if not isinstance(terminal_closeout, dict) \
                or terminal_closeout.get("closeout_filename") \
                != record["payload"]["context"]["closeout_filename"] \
                or not isinstance(terminal_closeout.get("closeout_sha256"), str) \
                or not SHA256.fullmatch(terminal_closeout["closeout_sha256"]):
            raise ToolError(
                "receipt lacks an exact validated closeout binding",
                failure_phase="evidence_manifest", failure_class="evidence",
            )
        context = record["payload"]["context"]
        archive_contract = scientific_archive_contract(
            context, terminal_closeout, closed,
        )
    result = {
        **context,
        "evidence_dir": f"{context['remote_repo']}/experience_docx/experiment_logs/{context['route_id']}",
        "validated_closeout_filename": terminal_closeout["closeout_filename"],
        "validated_closeout_sha256": terminal_closeout["closeout_sha256"],
    }
    if archive_contract is not None:
        result["archive_contract"] = archive_contract
    return result


def begin_finish(token):
    with locked_record("receipt", token) as record:
        if not record.get("launched"):
            raise ToolError("receipt has no successful launch")
        if record.get("finish_closed"):
            raise ToolError(f"finish is closed: {record['finish_closed']}")
        now = int(time.time())
        not_before = record.get("finish_not_before_unix", 0)
        cached = record.get("pending_finish_response")
        terminal_detected = record.get("operator_terminal_detected") is True
        if not terminal_detected and isinstance(not_before, int) \
                and now < not_before and isinstance(cached, dict):
            return None, cached
        calls = record.get("finish_calls", 0)
        if not isinstance(calls, int) or calls < 0:
            raise ToolError("receipt finish counter is invalid", failure_class="command_infra")
        if calls >= MAX_FINISH_WINDOWS:
            record["finish_closed"] = "OBSERVATION_BUDGET_EXHAUSTED"
            raise ToolError("finish observation budget is exhausted")
        record["finish_calls"] = calls + 1
        record["finish_not_before_unix"] = 0
        record["pending_finish_response"] = None
        record["operator_terminal_detected"] = False
        context = dict(record["payload"]["context"])
        context["_receipt_issued_at"] = int(record["payload"]["issued_at"])
        context["_monitor_stale_count"] = int(record.get("monitor_stale_count", 0))
        return context, None


def cache_finish_response(token, response, not_before):
    with locked_record("receipt", token) as record:
        record["finish_not_before_unix"] = int(not_before)
        record["pending_finish_response"] = response


def close_finish(token, state):
    with locked_record("receipt", token) as record:
        record["finish_closed"] = require_token(state, "finish_closed")


def close_scientific_finish(token, closeout):
    with locked_record("receipt", token) as record:
        record["terminal_closeout"] = closeout
        record["finish_closed"] = "CLOSEOUT_VALIDATED"


def validated_scientific_result(token, closeout, observed, *, manifest=None):
    close_scientific_finish(token, closeout)
    with locked_record("receipt", token) as record:
        context = record["payload"]["context"]
        archive_contract = scientific_archive_contract(
            context, closeout, "CLOSEOUT_VALIDATED",
        )
    extra = {}
    next_actions = ["scientific_review_or_archive"]
    if archive_contract is not None:
        next_actions = [
            "convir_evidence_list",
            "convir_evidence_fetch",
            "author_scientific_conclusion",
            "prepare_terminal_archive",
        ]
        extra.update({
            "archive_contract": archive_contract,
            "archive_ready": False,
            "required_action_sequence": next_actions,
        })
    if manifest is not None:
        extra["manifest"] = manifest
    return typed_result(
        True, "CLOSEOUT_VALIDATED", observed=observed,
        next_actions=next_actions, receipt=token, **extra,
    )


def authorize_engineering_auto_repair(token, closeout):
    with locked_record("receipt", token) as record:
        record["terminal_closeout"] = closeout
        record["engineering_failure_resolution"] = "repair"
        record["finish_closed"] = "ENGINEERING_AUTO_REPAIR_AUTHORIZED"


def resolve_engineering_failure(token, resolution):
    resolution = require_enum(
        resolution, "engineering_failure_resolution", {"repair", "archive", "discard"},
    )
    with locked_record("receipt", token) as record:
        if not record.get("launched"):
            raise ToolError("receipt has no successful launch")
        migrated_archive_repair = (
            resolution == "repair"
            and record.get("finish_closed") == "ENGINEERING_ARCHIVE_AUTHORIZED"
            and record.get("engineering_failure_resolution") == "archive"
            and isinstance(record.get("v43_migrated_at"), int)
        )
        current_state = record.get("finish_closed")
        if current_state == "ENGINEERING_AUTO_REPAIR_AUTHORIZED" and resolution == "repair":
            return typed_result(
                True, "ENGINEERING_AUTO_REPAIR_AUTHORIZED",
                observed={"closeout": record.get("terminal_closeout"), "resolution": "repair"},
                next_actions=["inspect_failure_once", "prepare_one_same_contract_engineering_repair"],
                archive_authorized=False, relaunch_authorized=False,
            )
        if current_state not in {
            "ENGINEERING_REVIEW_REQUIRED", "ENGINEERING_AUTO_REPAIR_AUTHORIZED",
        } and not migrated_archive_repair:
            raise ToolError("receipt is not awaiting an engineering failure decision")
        closeout = record.get("terminal_closeout")
        if not isinstance(closeout, dict) or closeout.get("terminal_tuple", {}).get("state") != "FAILED_ENGINEERING":
            raise ToolError("receipt has no validated engineering closeout", failure_class="evidence")
        if resolution == "discard":
            diagnostic = closeout.get("engineering_diagnostic")
            if not isinstance(diagnostic, dict):
                raise ToolError("engineering diagnostic is unavailable", failure_class="evidence")
            if diagnostic.get("scientific_data_touched") is not False \
                    or diagnostic.get("protected_data_touched") is not False:
                raise ToolError(
                    "discard requires verified absence of scientific and protected data access",
                    failure_phase="engineering_discard", failure_class="evidence",
                )
            context = record["payload"]["context"]
            validate_discard_context(context)
            output = run_remote(
                engineering_discard_body(context, closeout), timeout=60,
                phase="engineering_discard",
            )
            if output.splitlines().count("CONVIR_OPS_ENGINEERING_DISCARD_OK") != 1:
                raise ToolError(
                    "engineering discard marker is missing",
                    failure_phase="engineering_discard", failure_class="command_infra",
                )
            record["engineering_failure_resolution"] = "discard"
            record["finish_closed"] = "ENGINEERING_DISCARDED"
            return typed_result(
                True, "ENGINEERING_DISCARDED",
                observed={
                    "resolution": "discard", "receipt_bound": True,
                    "deleted": ["remote_route_workspace", "operation_output"],
                    "postcheck": {
                        "remote_route_workspace_absent": True,
                        "operation_output_absent": True,
                    },
                },
                next_actions=["record_discard_audit_only"],
                archive_authorized=False, relaunch_authorized=False,
            )
        record["engineering_failure_resolution"] = resolution
        if resolution == "repair":
            if migrated_archive_repair:
                record["v431_migrated_archive_reopened_at"] = int(time.time())
            record["finish_closed"] = "ENGINEERING_REPAIR_AUTHORIZED"
            return typed_result(
                True, "ENGINEERING_REPAIR_AUTHORIZED",
                observed={
                    "closeout": closeout,
                    "resolution": resolution,
                    "migrated_archive_reopened": migrated_archive_repair,
                },
                next_actions=["prepare_one_same_contract_engineering_repair"],
                archive_authorized=False, relaunch_authorized=False,
            )
        record["finish_closed"] = "ENGINEERING_ARCHIVE_AUTHORIZED"
        return typed_result(
            True, "ENGINEERING_ARCHIVE_AUTHORIZED",
            observed={"closeout": closeout, "resolution": resolution},
            next_actions=["convir_evidence_list", "convir_evidence_fetch", "archive_compact_failure_evidence"],
            archive_authorized=True, relaunch_authorized=False,
        )


def validate_discard_context(context):
    remote_repo = Path(context["remote_repo"])
    run_root = Path(context["run_root"])
    output_path = Path(context["output_path"])
    if remote_repo.parent != Path(REMOTE_REPOS) or remote_repo == Path(CLOUD_GIT_SEED):
        raise ToolError("discard remote workspace is outside its dedicated root")
    if run_root.parent != Path(REMOTE_RUNS) or output_path.parent != run_root:
        raise ToolError("discard output is outside its receipt-bound run root")
    if remote_repo.name in {"ConvIR-B-official-arch-anchor", "main"}:
        raise ToolError("discard cannot target a shared or anchor workspace")
    expected_repo = derive_remote_repo(context["route_id"], context["output_id"])
    expected_run = f"{REMOTE_RUNS}/{context['route_id']}"
    if str(remote_repo) != expected_repo or str(run_root) != expected_run \
            or str(output_path) != f"{expected_run}/{context['output_id']}":
        raise ToolError("discard paths do not match the receipt identity")
    expected_closeout = (
        remote_repo / "experience_docx" / "experiment_logs" /
        context["route_id"] / context["closeout_filename"]
    )
    if Path(context["closeout_path"]) != expected_closeout:
        raise ToolError("discard closeout path does not match the receipt identity")


def engineering_discard_body(context, closeout):
    validate_discard_context(context)
    identity = closeout["identity"]
    return "\n".join([
        f"REMOTE_REPO={q(context['remote_repo'])}",
        f"OUTPUT_PATH={q(context['output_path'])}",
        f"RUN_ROOT={q(context['run_root'])}",
        f"REPO_ROOT={q(REMOTE_REPOS)}",
        f"RUNS_ROOT={q(REMOTE_RUNS)}",
        f"SESSION={q(context['session'])}",
        f"CLOSEOUT={q(context['closeout_path'])}",
        f"EXPECTED_COMMIT={q(identity['route_commit'])}",
        f"EXPECTED_CLOSEOUT_SHA={q(closeout['closeout_sha256'])}",
        f"EXPECTED_ROUTE={q(identity['route_id'])}",
        f"EXPECTED_RUN={q(identity['run_id'])}",
        'test "$(dirname "$REMOTE_REPO")" = "$REPO_ROOT"',
        'test "$(dirname "$RUN_ROOT")" = "$RUNS_ROOT"',
        'test "$(dirname "$OUTPUT_PATH")" = "$RUN_ROOT"',
        'test "$REMOTE_REPO" != "$REPO_ROOT/ConvIR-B-official-arch-anchor"',
        'tmux has-session -t "$SESSION" 2>/dev/null && exit 92 || true',
        'test -d "$REMOTE_REPO/.git"',
        'test "$(git -C "$REMOTE_REPO" rev-parse HEAD)" = "$EXPECTED_COMMIT"',
        'test -f "$CLOSEOUT"',
        'test "$(sha256sum "$CLOSEOUT" | awk \'{print $1}\')" = "$EXPECTED_CLOSEOUT_SHA"',
        'if test -e "$OUTPUT_PATH"; then',
        f'  {q(REMOTE_PYTHON)} - "$OUTPUT_PATH/control/lifecycle_identity.json" "$EXPECTED_ROUTE" "$EXPECTED_RUN" "$EXPECTED_COMMIT" <<\'PY\'',
        'import json, sys',
        'value = json.load(open(sys.argv[1], encoding="utf-8"))',
        'assert value["route_id"] == sys.argv[2]',
        'assert value["run_id"] == sys.argv[3]',
        'assert value["route_commit"] == sys.argv[4]',
        'PY',
        '  rm -rf -- "$OUTPUT_PATH"',
        'fi',
        'rm -rf -- "$REMOTE_REPO"',
        'rmdir "$RUN_ROOT" 2>/dev/null || true',
        'test ! -e "$OUTPUT_PATH"',
        'test ! -e "$REMOTE_REPO"',
        'echo CONVIR_OPS_ENGINEERING_DISCARD_OK',
    ])


def record_stale_observation(token):
    """Record a bounded warning without closing later closeout validation."""
    with locked_record("receipt", token) as record:
        count = record.get("monitor_stale_count", 0)
        if not isinstance(count, int) or count < 0:
            raise ToolError("receipt stale counter is invalid", failure_class="command_infra")
        record["monitor_stale_count"] = count + 1
        return count + 1


def monitor_body(context, profile):
    status = f"{context['output_path']}/status.txt"
    heartbeat = f"{context['output_path']}/heartbeat.json"
    return "\n".join([
        f"SESSION={q(context['session'])}",
        f"STATUS={q(status)}",
        f"HEARTBEAT={q(heartbeat)}",
        f"CLOSEOUT={q(context['closeout_path'])}",
        f"MAX_POLLS={profile['max_polls']}",
        f"INTERVAL={profile['interval_seconds']}",
        f"STALE={int(context['heartbeat_timeout_seconds'])}",
        f"LAUNCHED_AT={int(context.get('_receipt_issued_at', int(time.time())))}",
        'active=false; terminal=false; stale=false; heartbeat_age=-1; heartbeat_source=launch; n=0',
        'for n in $(seq 1 "$MAX_POLLS"); do',
        '  active=false; tmux has-session -t "$SESSION" 2>/dev/null && active=true',
        '  test ! -f "$CLOSEOUT" || { terminal=true; break; }',
        '  if test -f "$HEARTBEAT"; then heartbeat_source=heartbeat; heartbeat_age=$(( $(date +%s) - $(stat -c %Y "$HEARTBEAT") ));',
        '  elif test -f "$STATUS"; then heartbeat_source=status; heartbeat_age=$(( $(date +%s) - $(stat -c %Y "$STATUS") ));',
        '  else heartbeat_source=launch; heartbeat_age=$(( $(date +%s) - LAUNCHED_AT )); fi',
        '  if test "$active" = true && test "$heartbeat_age" -ge "$STALE"; then stale=true; break; fi',
        '  test "$active" = true || break',
        '  test "$n" = "$MAX_POLLS" || sleep "$INTERVAL"',
        'done',
        'echo "CONVIR_OPS_MONITOR polls=$n active=$active terminal=$terminal stale=$stale heartbeat_age=$heartbeat_age heartbeat_source=$heartbeat_source"',
        'echo CONVIR_OPS_STATUS_BEGIN',
        'test ! -f "$STATUS" || tail -n 20 "$STATUS"',
        'echo CONVIR_OPS_STATUS_END',
        'if test -f "$CLOSEOUT"; then',
        f'  {q(REMOTE_PYTHON)} - "$CLOSEOUT" <<\'PY\'',
        'import hashlib, json, sys',
        'raw = open(sys.argv[1], "rb").read(65537)',
        'assert len(raw) <= 65536',
        'value = json.loads(raw)',
        'print("CONVIR_OPS_CLOSEOUT_SHA256=" + hashlib.sha256(raw).hexdigest())',
        'print("CONVIR_OPS_CLOSEOUT_BEGIN")',
        'print(json.dumps(value, sort_keys=True, separators=(",", ":")))',
        'print("CONVIR_OPS_CLOSEOUT_END")',
        'PY',
        'fi',
    ])


def parse_monitor(output):
    meta = re.search(
        r"(?m)^CONVIR_OPS_MONITOR polls=(\d+) active=(true|false) terminal=(true|false) stale=(true|false) heartbeat_age=(-?\d+)(?: heartbeat_source=(heartbeat|status|launch))?$",
        output,
    )
    begin = output.find("CONVIR_OPS_STATUS_BEGIN")
    end = output.find("CONVIR_OPS_STATUS_END")
    if not meta or begin < 0 or end < begin:
        raise ToolError("monitor markers are missing", failure_phase="monitor", failure_class="command_infra")
    return {
        "poll_count": int(meta.group(1)), "active": meta.group(2) == "true",
        "terminal": meta.group(3) == "true", "stale": meta.group(4) == "true",
        "heartbeat_age_seconds": int(meta.group(5)),
        "heartbeat_source": meta.group(6) or "legacy_status",
        "status": output[begin + len("CONVIR_OPS_STATUS_BEGIN"):end].strip()[:4096],
    }


def workload_progress(status):
    """Return the strongest machine-readable workload count and capacity."""
    best_completed = 0
    best_total = 0

    def visit(value, workload=False):
        nonlocal best_completed, best_total
        if isinstance(value, dict):
            phase = value.get("phase")
            event = value.get("event")
            progress_envelope = any(
                isinstance(key, str)
                and re.fullmatch(r"[A-Z][A-Z0-9_]{0,63}_PROGRESS", key)
                and isinstance(item, dict)
                for key, item in value.items()
            )
            in_workload = workload or phase == "workload" or event in {
                "workload_start", "workload_progress", "workload_pass",
            } or progress_envelope
            completed = value.get("completed_units", value.get("completed"))
            total = value.get("total_units", value.get("total"))
            valid_total = (
                total if isinstance(total, int) and total >= completed else 0
            ) if isinstance(completed, int) else 0
            if in_workload and isinstance(completed, int) and completed >= 0 \
                    and (completed > best_completed or (
                        completed == best_completed and valid_total > best_total
                    )):
                best_completed = completed
                best_total = valid_total
            for item in value.values():
                visit(item, in_workload)
        elif isinstance(value, list):
            for item in value:
                visit(item, workload)

    for line in status.splitlines():
        try:
            visit(json.loads(line))
        except (json.JSONDecodeError, TypeError):
            continue
    return {"completed_units": best_completed, "total_units": best_total}


def contract_progress(status):
    """Return the strongest bounded contract progress milestone."""
    best = {"stage": None, "completed_iterations": 0, "total_iterations": 0}
    for line in status.splitlines():
        try:
            value = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(value, dict) or value.get("phase") != "contract" \
                or value.get("event") != "contract_progress" or set(value) != {
                    "phase", "event", "stage", "completed_iterations", "total_iterations",
                }:
            continue
        completed = value.get("completed_iterations")
        total = value.get("total_iterations")
        stage = value.get("stage")
        if isinstance(completed, int) and isinstance(total, int) \
                and 0 <= completed <= total and total > 0 \
                and isinstance(stage, str) and SAFE_TOKEN.fullmatch(stage) \
                and completed >= best["completed_iterations"]:
            best = {
                "stage": stage, "completed_iterations": completed,
                "total_iterations": total,
            }
    return best


def progress_stage(status):
    """Extract only a typed, token-safe control stage from status telemetry."""
    best = {
        "workload": {"seen": False, "completed": -1, "stage": None},
        "other": {"seen": False, "completed": -1, "stage": None},
    }

    def visit(value, phase_hint=None):
        if isinstance(value, dict):
            envelope = any(
                isinstance(key, str)
                and re.fullmatch(r"[A-Z][A-Z0-9_]{0,63}_PROGRESS", key)
                and isinstance(item, dict)
                for key, item in value.items()
            )
            phase = value.get("phase")
            event = value.get("event")
            workload = phase_hint == "workload" or phase == "workload" \
                or event in {
                    "workload_start", "workload_progress", "workload_pass",
                    "workload_end",
                } or envelope
            typed = workload or phase_hint == "other" \
                or phase in {"contract", "terminal"} \
                or event == "contract_progress"
            current = "workload" if workload else "other"
            if typed:
                best[current]["seen"] = True
            stage = value.get("stage")
            completed = value.get(
                "completed_units", value.get(
                    "completed", value.get("completed_iterations", 0),
                ),
            )
            if typed and isinstance(stage, str) \
                    and SAFE_TOKEN.fullmatch(stage) \
                    and isinstance(completed, int) \
                    and completed >= best[current]["completed"]:
                best[current].update(completed=completed, stage=stage)
            for item in value.values():
                visit(item, current if typed else phase_hint)
        elif isinstance(value, list):
            for item in value:
                visit(item, phase_hint)

    for line in status.splitlines():
        try:
            visit(json.loads(line))
        except (json.JSONDecodeError, TypeError):
            continue
    if best["workload"]["seen"]:
        return best["workload"]["stage"] or "workload"
    return best["other"]["stage"]


def validate_operator_context(context):
    remote_repo = Path(context["remote_repo"])
    run_root = Path(context["run_root"])
    output_path = Path(context["output_path"])
    closeout_path = Path(context["closeout_path"])
    if remote_repo.parent != Path(REMOTE_REPOS) or remote_repo == Path(CLOUD_GIT_SEED):
        raise ToolError("operator control remote workspace is outside its dedicated root")
    if run_root.parent != Path(REMOTE_RUNS) or output_path.parent != run_root:
        raise ToolError("operator control output is outside its receipt-bound run root")
    expected_repo = derive_remote_repo(context["route_id"], context["output_id"])
    if str(remote_repo) != expected_repo \
            or str(run_root) != f"{REMOTE_RUNS}/{context['route_id']}" \
            or str(output_path) != f"{run_root}/{context['output_id']}":
        raise ToolError("operator control paths do not match the receipt identity")
    expected_closeout = (
        remote_repo / "experience_docx" / "experiment_logs" /
        context["route_id"] / context["closeout_filename"]
    )
    if closeout_path != expected_closeout:
        raise ToolError("operator control closeout path does not match the receipt identity")
    expected_session = derive_session(
        context["route_id"], context["mode"],
        context["route_branch_commit"], context["output_id"],
    )
    if context["session"] != expected_session:
        raise ToolError("operator control session does not match the receipt identity")


def operator_observation_body(context):
    validate_operator_context(context)
    status = f"{context['output_path']}/status.txt"
    heartbeat = f"{context['output_path']}/heartbeat.json"
    identity = f"{context['output_path']}/control/lifecycle_identity.json"
    return "\n".join([
        f"REMOTE_REPO={q(context['remote_repo'])}",
        f"EXPECTED_COMMIT={q(context['route_branch_commit'])}",
        f"RUNNER={q(context['runner_relpath'])}",
        f"EXPECTED_RUNNER_SHA={q(context['runner_sha256'])}",
        f"ROUTE_ID={q(context['route_id'])}",
        f"RUN_ID={q(context['output_id'])}",
        f"OUTPUT_PATH={q(context['output_path'])}",
        f"IDENTITY={q(identity)}",
        f"STATUS={q(status)}",
        f"HEARTBEAT={q(heartbeat)}",
        f"CLOSEOUT={q(context['closeout_path'])}",
        f"SESSION={q(context['session'])}",
        f"TMUX={q(REMOTE_TMUX)}",
        f"LAUNCHED_AT={int(context.get('_receipt_issued_at', int(time.time())))}",
        'test -d "$REMOTE_REPO/.git"',
        'test "$(git -C "$REMOTE_REPO" rev-parse HEAD)" = "$EXPECTED_COMMIT"',
        'test "$(sha256sum "$REMOTE_REPO/$RUNNER" | awk \'{print $1}\')" = "$EXPECTED_RUNNER_SHA"',
        'if test -e "$OUTPUT_PATH"; then',
        '  test -f "$IDENTITY"',
        f'  {q(REMOTE_PYTHON)} - "$IDENTITY" "$ROUTE_ID" "$RUN_ID" "$EXPECTED_COMMIT" "$EXPECTED_RUNNER_SHA" <<\'PY\'',
        'import json, sys',
        'value = json.load(open(sys.argv[1], encoding="utf-8"))',
        'expected = {"route_id": sys.argv[2], "run_id": sys.argv[3], "route_commit": sys.argv[4], "runner_sha256": sys.argv[5]}',
        'assert all(value.get(key) == item for key, item in expected.items())',
        'PY',
        'fi',
        'snapshot_at=$(date +%s)',
        'active=false; "$TMUX" has-session -t "$SESSION" 2>/dev/null && active=true || true',
        'terminal=false; test ! -f "$CLOSEOUT" || terminal=true',
        'if test -f "$HEARTBEAT"; then heartbeat_source=heartbeat; heartbeat_age=$(( snapshot_at - $(stat -c %Y "$HEARTBEAT") ));',
        'elif test -f "$STATUS"; then heartbeat_source=status; heartbeat_age=$(( snapshot_at - $(stat -c %Y "$STATUS") ));',
        'else heartbeat_source=launch; heartbeat_age=$(( snapshot_at - LAUNCHED_AT )); fi',
        'echo "CONVIR_OPS_OPERATOR_OBSERVATION snapshot_at=$snapshot_at active=$active terminal=$terminal heartbeat_age=$heartbeat_age heartbeat_source=$heartbeat_source"',
        'echo CONVIR_OPS_STATUS_BEGIN',
        'test ! -f "$STATUS" || tail -n 20 "$STATUS"',
        'echo CONVIR_OPS_STATUS_END',
    ])


def parse_operator_observation(output):
    meta = re.search(
        r"(?m)^CONVIR_OPS_OPERATOR_OBSERVATION snapshot_at=(\d+) active=(true|false) terminal=(true|false) heartbeat_age=(-?\d+) heartbeat_source=(heartbeat|status|launch)$",
        output,
    )
    begin = output.find("CONVIR_OPS_STATUS_BEGIN")
    end = output.find("CONVIR_OPS_STATUS_END")
    if not meta or begin < 0 or end < begin:
        raise ToolError(
            "operator observation markers are missing",
            failure_phase="operator_observation", failure_class="command_infra",
        )
    return {
        "snapshot_at_unix": int(meta.group(1)),
        "active": meta.group(2) == "true",
        "terminal": meta.group(3) == "true",
        "heartbeat_age_seconds": int(meta.group(4)),
        "heartbeat_source": meta.group(5),
        "status": output[
            begin + len("CONVIR_OPS_STATUS_BEGIN"):end
        ].strip()[:4096],
    }


def begin_operator_observation(token):
    with locked_record("receipt", token) as record:
        if not record.get("launched"):
            raise ToolError("receipt has no successful launch")
        if record.get("finish_closed"):
            raise ToolError(f"finish is closed: {record['finish_closed']}")
        now = int(time.time())
        not_before = record.get("operator_observation_not_before_unix", 0)
        cached = record.get("operator_observation_cache")
        if isinstance(not_before, int) and now < not_before \
                and isinstance(cached, dict):
            return None, dict(cached)
        calls = record.get("operator_observation_calls", 0)
        if not isinstance(calls, int) or calls < 0:
            raise ToolError(
                "operator observation counter is invalid",
                failure_class="command_infra",
            )
        if calls >= MAX_OPERATOR_OBSERVATIONS:
            raise ToolError(
                "operator observation budget is exhausted; formal finish and cancellation remain available",
                failure_phase="operator_observation", failure_class="contract",
            )
        record["operator_observation_calls"] = calls + 1
        context = dict(record["payload"]["context"])
        context["_receipt_issued_at"] = int(record["payload"]["issued_at"])
        return context, None


def cache_operator_observation(token, snapshot):
    with locked_record("receipt", token) as record:
        record["operator_observation_not_before_unix"] = (
            int(snapshot["snapshot_at_unix"])
            + OPERATOR_OBSERVATION_MIN_INTERVAL_SECONDS
        )
        record["operator_observation_cache"] = snapshot


def mark_operator_terminal_detected(token, snapshot):
    with locked_record("receipt", token) as record:
        record["operator_terminal_detected"] = True
        record["finish_not_before_unix"] = 0
        record["pending_finish_response"] = None
        record["operator_observation_cache"] = snapshot


def operator_progress_result(token, snapshot, *, cached):
    progress = snapshot["workload_progress"]
    observed = {
        "stage": snapshot.get("stage"),
        "completed_units": progress["completed_units"],
        "total_units": progress["total_units"],
        "active": snapshot["active"],
        "heartbeat_age_seconds": snapshot["heartbeat_age_seconds"],
        "heartbeat_source": snapshot["heartbeat_source"],
        "snapshot_at_unix": snapshot["snapshot_at_unix"],
        "cached": cached,
        "current_health_claimed": not cached,
    }
    if snapshot.get("stale") is True:
        return typed_failure(
            "PROGRESS_STALE", "command_infra",
            "heartbeat exceeded the sealed limit",
            observed=observed,
            next_actions=["convir_route_finish_progress_only", "engineering_review_once"],
            failure_phase="operator_observation", receipt_remains_open=True,
            receipt=token, result_blind=True, cached=cached,
            snapshot_at_unix=snapshot["snapshot_at_unix"],
        )
    state = "PROGRESS_REFRESH_CACHED" if cached else "PROGRESS_REFRESHED"
    return typed_result(
        True, state, observed=observed,
        next_actions=["convir_route_finish_progress_only", "convir_route_finish_cancel"],
        receipt=token, result_blind=True, cached=cached,
        snapshot_at_unix=snapshot["snapshot_at_unix"],
        retry_after_seconds=max(
            0,
            snapshot["snapshot_at_unix"]
            + OPERATOR_OBSERVATION_MIN_INTERVAL_SECONDS - int(time.time()),
        ),
    )


def observe_operator_progress(token):
    context, cached = begin_operator_observation(token)
    if cached is not None:
        return operator_progress_result(token, cached, cached=True)
    output = run_remote(
        operator_observation_body(context), timeout=30,
        phase="operator_observation",
    )
    observed = parse_operator_observation(output)
    status = observed.pop("status")
    progress = workload_progress(status)
    snapshot = {
        **observed,
        "stage": progress_stage(status),
        "workload_progress": progress,
        "stale": observed["heartbeat_age_seconds"]
        >= int(context["heartbeat_timeout_seconds"]),
    }
    if snapshot["terminal"]:
        mark_operator_terminal_detected(token, snapshot)
        return typed_result(
            True, "TERMINAL_DETECTED",
            observed={
                key: snapshot[key] for key in (
                    "active", "snapshot_at_unix", "heartbeat_age_seconds",
                    "heartbeat_source",
                )
            },
            next_actions=["convir_route_finish"], receipt=token,
            result_blind=True, cached=False,
            snapshot_at_unix=snapshot["snapshot_at_unix"],
        )
    if not snapshot["active"]:
        close_finish(token, "CLOSEOUT_MISSING")
        return typed_failure(
            "CLOSEOUT_MISSING", "evidence", "session ended without closeout",
            observed={
                "active": False, "terminal": False,
                "snapshot_at_unix": snapshot["snapshot_at_unix"],
                "cached": False,
            },
            next_actions=["engineering_review_once"],
            failure_phase="closeout", receipt=token, result_blind=True,
        )
    cache_operator_observation(token, snapshot)
    return operator_progress_result(token, snapshot, cached=False)


def record_workload_verified(token):
    with locked_record("receipt", token) as record:
        record["workload_verified"] = True


def parse_closeout(context, output):
    begin = output.find("CONVIR_OPS_CLOSEOUT_BEGIN")
    end = output.find("CONVIR_OPS_CLOSEOUT_END")
    if begin < 0 or end < begin:
        return None
    value = json.loads(output[begin + len("CONVIR_OPS_CLOSEOUT_BEGIN"):end].strip())
    expected_identity = {
        "route_id": context["route_id"], "run_id": context["output_id"],
        "route_commit": context["route_branch_commit"], "runner_sha256": context["runner_sha256"],
    }
    if not isinstance(value, dict) or {key: value.get(key) for key in expected_identity} != expected_identity:
        raise ToolError("closeout provenance mismatch", failure_class="evidence")
    terminal = {key: value.get(key) for key in ("state", "decision", "authorizes")}
    if terminal not in context["allowed_terminal_tuples"] \
            and terminal != OPERATOR_CANCEL_TERMINAL:
        raise ToolError("closeout terminal tuple is not allowed", failure_class="evidence")
    match = re.search(r"(?m)^CONVIR_OPS_CLOSEOUT_SHA256=([0-9a-f]{64})$", output)
    if not match:
        raise ToolError("closeout SHA-256 is missing", failure_class="evidence")
    result = {
        "identity": expected_identity, "terminal_tuple": terminal,
        "closeout_sha256": match.group(1), "closeout_filename": context["closeout_filename"],
    }
    if terminal["state"] == "FAILED_ENGINEERING":
        details = value.get("details") if isinstance(value.get("details"), dict) else {}
        error_type = details.get("error_type")
        error_message = details.get("error_message")
        verified_assets = value.get("verified_assets")
        safe_assets = []
        if isinstance(verified_assets, list):
            for item in verified_assets[:64]:
                if not isinstance(item, dict):
                    continue
                safe_assets.append({
                    key: item[key] for key in (
                        "id", "kind", "access_role", "contract_access", "sha256", "commit",
                    ) if key in item
                })
        traceback_tail = details.get("traceback_tail")
        workload_started = details.get("workload_started")
        scientific_touched = details.get("scientific_data_touched")
        protected_touched = details.get("protected_data_touched")
        failed_checks = details.get("failed_contract_checks")
        if not isinstance(failed_checks, list):
            failed_checks = []
        failed_checks = [
            item for item in failed_checks[:32]
            if isinstance(item, str) and SAFE_TOKEN.fullmatch(item)
        ]
        result["engineering_diagnostic"] = {
            "failure_phase": value.get("failure_phase") if isinstance(value.get("failure_phase"), str) else None,
            "returncode": value.get("returncode") if isinstance(value.get("returncode"), int) else None,
            "exception_type": safe_diagnostic_text(error_type, 128) if isinstance(error_type, str) else None,
            "exception_message": safe_diagnostic_text(error_message, 2048) if isinstance(error_message, str) else None,
            "traceback_tail": safe_diagnostic_text(traceback_tail, 4096) if isinstance(traceback_tail, str) else None,
            "last_status": None,
            "verified_assets": safe_assets,
            "workload_started": workload_started if isinstance(workload_started, bool) else None,
            "scientific_data_touched": scientific_touched if isinstance(scientific_touched, bool) else None,
            "protected_data_touched": protected_touched if isinstance(protected_touched, bool) else None,
            "failed_contract_checks": sorted(set(failed_checks)),
            "suggested_repair_class": engineering_failure_class(value.get("failure_phase")),
        }
    elif terminal == OPERATOR_CANCEL_TERMINAL:
        details = value.get("details") if isinstance(value.get("details"), dict) else {}
        result["operator_cancellation"] = {
            "request_id": details.get("request_id")
            if isinstance(details.get("request_id"), str) else None,
            "requested_at_unix": details.get("requested_at_unix")
            if isinstance(details.get("requested_at_unix"), int) else None,
            "completed_units": details.get("completed_units")
            if isinstance(details.get("completed_units"), int) else 0,
            "total_units": details.get("total_units")
            if isinstance(details.get("total_units"), int) else 0,
            "stage": details.get("stage")
            if isinstance(details.get("stage"), str)
            and SAFE_TOKEN.fullmatch(details["stage"]) else None,
            "termination_mode": details.get("termination_mode")
            if details.get("termination_mode") in {"graceful", "forced", "control_finalize"}
            else None,
            "scientific_result_interpretable": False,
        }
    return result


def engineering_failure_class(phase):
    if phase in {
        "environment", "identity_preflight", "manifest_preflight",
        "asset_preflight", "output_preflight", "resource_preflight",
    }:
        return "preflight_resource"
    if phase in {"evidence", "finalize", "failure_closeout", "closeout"}:
        return "evidence_closeout"
    return "engineering_runtime"


def operator_cancel_body(context, request_id):
    validate_operator_context(context)
    identity = f"{context['output_path']}/control/lifecycle_identity.json"
    request = f"{context['output_path']}/control/operator_cancel_request.json"
    targets = f"{context['output_path']}/control/operator_cancel_targets.json"
    status = f"{context['output_path']}/status.txt"
    lifecycle = f"{context['remote_repo']}/experience_docx/tools/route_lifecycle.py"
    return "\n".join([
        f"REMOTE_REPO={q(context['remote_repo'])}",
        f"RUN_ROOT={q(context['run_root'])}",
        f"OUTPUT_PATH={q(context['output_path'])}",
        f"CLOSEOUT={q(context['closeout_path'])}",
        f"IDENTITY={q(identity)}",
        f"REQUEST={q(request)}",
        f"TARGETS={q(targets)}",
        f"STATUS={q(status)}",
        f"SESSION={q(context['session'])}",
        f"TMUX={q(REMOTE_TMUX)}",
        f"ROUTE_ID={q(context['route_id'])}",
        f"RUN_ID={q(context['output_id'])}",
        f"MODE={q(context.get('operation_id', context['mode']))}",
        f"EXPECTED_COMMIT={q(context['route_branch_commit'])}",
        f"RUNNER={q(context['runner_relpath'])}",
        f"EXPECTED_RUNNER_SHA={q(context['runner_sha256'])}",
        f"LIFECYCLE={q(lifecycle)}",
        f"REQUEST_ID={q(request_id)}",
        f"GRACE={OPERATOR_CANCEL_GRACE_SECONDS}",
        f"FORCE_WAIT={OPERATOR_CANCEL_FORCE_SECONDS}",
        'test -d "$REMOTE_REPO/.git"',
        'test "$(dirname "$REMOTE_REPO")" = ' + q(REMOTE_REPOS),
        'test "$(dirname "$RUN_ROOT")" = ' + q(REMOTE_RUNS),
        'test "$(dirname "$OUTPUT_PATH")" = "$RUN_ROOT"',
        'test "$(git -C "$REMOTE_REPO" rev-parse HEAD)" = "$EXPECTED_COMMIT"',
        'test -f "$REMOTE_REPO/$RUNNER"',
        'test "$(sha256sum "$REMOTE_REPO/$RUNNER" | awk \'{print $1}\')" = "$EXPECTED_RUNNER_SHA"',
        'test -f "$IDENTITY"',
        'active=false; terminal=false; termination_mode=none; snapshot_at=$(date +%s)',
        'if test -f "$CLOSEOUT"; then',
        '  terminal=true; termination_mode=already_terminal',
        'elif "$TMUX" has-session -t "$SESSION" 2>/dev/null; then',
        '  active=true',
        '  pane_pid=$("$TMUX" display-message -p -t "$SESSION" "#{pane_pid}")',
        f'  {q(REMOTE_PYTHON)} - "$REMOTE_REPO" "$OUTPUT_PATH" "$IDENTITY" "$REQUEST" "$TARGETS" "$ROUTE_ID" "$RUN_ID" "$EXPECTED_COMMIT" "$EXPECTED_RUNNER_SHA" "$RUNNER" "$SESSION" "$pane_pid" "$REQUEST_ID" {q(REMOTE_PYTHON)} "$LIFECYCLE" <<\'PY\'',
        'import json, os, signal, sys, time',
        'from pathlib import Path',
        'repo, output, identity_path, request_path, targets_path = map(Path, sys.argv[1:6])',
        'route_id, run_id, commit, runner_sha, runner_rel, session = sys.argv[6:12]',
        'pane_pid, request_id, remote_python, lifecycle = int(sys.argv[12]), sys.argv[13], sys.argv[14], sys.argv[15]',
        'def atomic(path, value):',
        '    path.parent.mkdir(parents=True, exist_ok=True)',
        '    temp = path.with_name(path.name + ".tmp." + request_id)',
        '    raw = (json.dumps(value, indent=2, sort_keys=True) + "\\n").encode()',
        '    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)',
        '    try:',
        '        with os.fdopen(fd, "wb") as stream: stream.write(raw); stream.flush(); os.fsync(stream.fileno())',
        '        os.replace(temp, path)',
        '    finally:',
        '        try: temp.unlink()',
        '        except FileNotFoundError: pass',
        'def proc(pid):',
        '    root = Path("/proc") / str(pid)',
        '    stat = (root / "stat").read_text().split()',
        '    cmd = (root / "cmdline").read_bytes().rstrip(b"\\0").decode().split("\\0")',
        '    env = {}',
        '    for item in (root / "environ").read_bytes().split(b"\\0"):',
        '        if b"=" in item:',
        '            key, value = item.split(b"=", 1); env[key.decode()] = value.decode()',
        '    return {"pid": pid, "ppid": int(stat[3]), "pgid": int(stat[4]), "sid": int(stat[5]), "start_ticks": int(stat[21]), "cmdline": cmd, "env": env, "uid": root.stat().st_uid}',
        'identity = json.loads(identity_path.read_text())',
        'expected = {"route_id": route_id, "run_id": run_id, "route_commit": commit, "runner_sha256": runner_sha}',
        'assert all(identity.get(key) == value for key, value in expected.items())',
        'pane = proc(pane_pid)',
        'assert pane["uid"] == os.getuid()',
        'assert pane["cmdline"] == [remote_python, lifecycle]',
        'for key, value in {"RUN_ID": run_id, "OUTPUT_PATH": str(output), "EXPECTED_ROUTE_COMMIT": commit, "RUNNER_SHA256": runner_sha}.items(): assert pane["env"].get(key) == value',
        'request_value = {"schema_version": 1, "request_id": request_id, **expected, "session": session, "requested_at_unix": int(time.time()), "action": "cancel"}',
        'if request_path.exists():',
        '    prior = json.loads(request_path.read_text())',
        '    assert all(prior.get(key) == value for key, value in request_value.items() if key != "requested_at_unix")',
        '    request_value = prior',
        'else: atomic(request_path, request_value)',
        'children = []',
        'child_file = Path("/proc") / str(pane_pid) / "task" / str(pane_pid) / "children"',
        'for token in child_file.read_text().split() if child_file.exists() else []:',
        '    try: child = proc(int(token))',
        '    except FileNotFoundError: continue',
        '    assert child["ppid"] == pane_pid and child["uid"] == os.getuid()',
        '    assert child["cmdline"] and child["cmdline"][0] == remote_python',
        '    assert any(arg.startswith(str(repo) + "/") or arg.startswith(str(output) + "/") for arg in child["cmdline"][1:])',
        '    for key, value in {"RUN_ID": run_id, "EXPECTED_ROUTE_COMMIT": commit}.items(): assert child["env"].get(key) == value',
        '    children.append({key: child[key] for key in ("pid", "pgid", "sid", "start_ticks", "cmdline")})',
        'target_value = {"schema_version": 1, "request_id": request_id, "session": session, "lifecycle": {key: pane[key] for key in ("pid", "start_ticks", "cmdline")}, "children": children}',
        'if targets_path.exists():',
        '    prior = json.loads(targets_path.read_text())',
        '    assert prior.get("request_id") == request_id and prior.get("session") == session',
        '    assert prior.get("lifecycle") == target_value["lifecycle"]',
        'else: atomic(targets_path, target_value)',
        'os.kill(pane_pid, signal.SIGTERM)',
        'PY',
        '  n=0',
        '  for n in $(seq 1 "$GRACE"); do',
        '    test ! -f "$CLOSEOUT" || break',
        '    "$TMUX" has-session -t "$SESSION" 2>/dev/null || break',
        '    sleep 1',
        '  done',
        '  if test -f "$CLOSEOUT"; then',
        '    terminal=true; active=false; termination_mode=graceful',
        '  else',
        f'    {q(REMOTE_PYTHON)} - "$REMOTE_REPO" "$OUTPUT_PATH" "$CLOSEOUT" "$IDENTITY" "$REQUEST" "$TARGETS" "$STATUS" "$ROUTE_ID" "$RUN_ID" "$MODE" "$EXPECTED_COMMIT" "$EXPECTED_RUNNER_SHA" "$SESSION" "$REQUEST_ID" "$FORCE_WAIT" {q(REMOTE_PYTHON)} "$LIFECYCLE" <<\'PY\'',
        'import hashlib, json, os, re, signal, subprocess, sys, time',
        'from pathlib import Path',
        'repo, output, closeout, identity_path, request_path, targets_path, status_path = map(Path, sys.argv[1:8])',
        'route_id, run_id, mode, commit, runner_sha, session, request_id = sys.argv[8:15]',
        'force_wait, remote_python, lifecycle = int(sys.argv[15]), sys.argv[16], sys.argv[17]',
        'request = json.loads(request_path.read_text()); targets = json.loads(targets_path.read_text())',
        'assert request["request_id"] == request_id and targets["request_id"] == request_id',
        'identity = json.loads(identity_path.read_text())',
        'assert identity.get("route_id") == route_id and identity.get("run_id") == run_id and identity.get("route_commit") == commit and identity.get("runner_sha256") == runner_sha',
        'def current(item):',
        '    root = Path("/proc") / str(item["pid"])',
        '    try:',
        '        stat = (root / "stat").read_text().split()',
        '        cmd = (root / "cmdline").read_bytes().rstrip(b"\\0").decode().split("\\0")',
        '        return root.stat().st_uid == os.getuid() and int(stat[21]) == item["start_ticks"] and cmd == item["cmdline"]',
        '    except FileNotFoundError: return False',
        'escalated = False',
        'for child in targets["children"]:',
        '    if not current(child): continue',
        '    escalated = True',
        '    if child["pgid"] == child["pid"] and child["sid"] == child["pid"]: os.killpg(child["pgid"], signal.SIGTERM)',
        '    else: os.kill(child["pid"], signal.SIGTERM)',
        'lifecycle_target = targets["lifecycle"]',
        'if current(lifecycle_target): escalated = True; os.kill(lifecycle_target["pid"], signal.SIGTERM)',
        'deadline = time.time() + force_wait',
        'while time.time() < deadline and any(current(item) for item in [lifecycle_target, *targets["children"]]): time.sleep(0.25)',
        'for child in targets["children"]:',
        '    if not current(child): continue',
        '    if child["pgid"] == child["pid"] and child["sid"] == child["pid"]: os.killpg(child["pgid"], signal.SIGKILL)',
        '    else: os.kill(child["pid"], signal.SIGKILL)',
        'if current(lifecycle_target): os.kill(lifecycle_target["pid"], signal.SIGKILL)',
        'time.sleep(1)',
        'assert not any(current(item) for item in [lifecycle_target, *targets["children"]])',
        'if closeout.exists(): raise SystemExit(0)',
        'completed = total = 0; stage = None',
        'def visit(value, typed=False):',
        '    global completed, total, stage',
        '    if isinstance(value, dict):',
        '        envelope = any(isinstance(k, str) and re.fullmatch(r"[A-Z][A-Z0-9_]{0,63}_PROGRESS", k) and isinstance(v, dict) for k, v in value.items())',
        '        typed = typed or value.get("phase") in {"contract", "workload", "terminal"} or envelope',
        '        c = value.get("completed_units", value.get("completed")); t = value.get("total_units", value.get("total"))',
        '        if typed and isinstance(c, int) and c >= completed: completed = c; total = t if isinstance(t, int) and t >= c else 0',
        '        s = value.get("stage")',
        '        if typed and isinstance(s, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", s): stage = s',
        '        for item in value.values(): visit(item, typed)',
        '    elif isinstance(value, list):',
        '        for item in value: visit(item, typed)',
        'if status_path.exists():',
        '    for line in status_path.read_text(errors="replace").splitlines()[-100:]:',
        '        try: visit(json.loads(line))',
        '        except (json.JSONDecodeError, TypeError): pass',
        'closeout_value = {"schema_version": 1, "route_id": route_id, "operation_id": mode, "run_id": run_id, "route_commit": commit, "runner_sha256": runner_sha, "state": "CANCELLED_BY_OPERATOR", "decision": None, "authorizes": "NONE", "evidence_role": "operator_control", "confirmation_images_targets_outcomes_touched": None, "canary_touched": None, "locked_test_touched": None, "evidence_sha256": {}, "verified_assets": [], "details": {"request_id": request_id, "requested_at_unix": request["requested_at_unix"], "completed_units": completed, "total_units": total, "stage": stage, "termination_mode": "forced" if escalated else "control_finalize", "scientific_result_interpretable": False, "partial_scientific_evidence_reuse_authorized": False}, "failure_phase": "operator_cancel", "returncode": 130}',
        'raw = (json.dumps(closeout_value, indent=2, sort_keys=True) + "\\n").encode()',
        'temp = closeout.with_name(closeout.name + ".tmp." + request_id)',
        'fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)',
        'with os.fdopen(fd, "wb") as stream: stream.write(raw); stream.flush(); os.fsync(stream.fileno())',
        'try: os.link(temp, closeout)',
        'except FileExistsError: pass',
        'finally: temp.unlink(missing_ok=True)',
        'local = output / "control" / closeout.name',
        'if not local.exists():',
        '    try: os.link(closeout, local)',
        '    except FileExistsError: pass',
        'PY',
        '    terminal=true; active=false; termination_mode=forced',
        '  fi',
        'fi',
        'snapshot_at=$(date +%s)',
        'echo "CONVIR_OPS_CANCEL snapshot_at=$snapshot_at active=$active terminal=$terminal mode=$termination_mode"',
        'if test -f "$CLOSEOUT"; then',
        f'  {q(REMOTE_PYTHON)} - "$CLOSEOUT" <<\'PY\'',
        'import hashlib, json, sys',
        'raw = open(sys.argv[1], "rb").read(65537)',
        'assert len(raw) <= 65536',
        'value = json.loads(raw)',
        'print("CONVIR_OPS_CLOSEOUT_SHA256=" + hashlib.sha256(raw).hexdigest())',
        'print("CONVIR_OPS_CLOSEOUT_BEGIN")',
        'print(json.dumps(value, sort_keys=True, separators=(",", ":")))',
        'print("CONVIR_OPS_CLOSEOUT_END")',
        'PY',
        'fi',
    ])


def parse_operator_cancel(output):
    match = re.search(
        r"(?m)^CONVIR_OPS_CANCEL snapshot_at=(\d+) active=(true|false) terminal=(true|false) mode=(already_terminal|graceful|forced|none)$",
        output,
    )
    if not match:
        raise ToolError(
            "operator cancellation markers are missing",
            failure_phase="operator_cancel", failure_class="command_infra",
        )
    return {
        "snapshot_at_unix": int(match.group(1)),
        "active": match.group(2) == "true",
        "terminal": match.group(3) == "true",
        "termination_mode": match.group(4),
    }


def begin_operator_cancel(token):
    with locked_record("receipt", token) as record:
        if not record.get("launched"):
            raise ToolError("receipt has no successful launch")
        if record.get("finish_closed") == "CANCELLED_BY_OPERATOR" \
                and isinstance(record.get("terminal_closeout"), dict):
            return None, record["terminal_closeout"], record.get("operator_cancel_request_id")
        if record.get("finish_closed"):
            raise ToolError(f"finish is closed: {record['finish_closed']}")
        attempts = record.get("operator_cancel_attempts", 0)
        if not isinstance(attempts, int) or attempts < 0:
            raise ToolError("operator cancellation counter is invalid", failure_class="command_infra")
        if attempts >= 2:
            raise ToolError(
                "operator cancellation recovery budget is exhausted",
                failure_phase="operator_cancel", failure_class="command_infra",
            )
        request_id = record.get("operator_cancel_request_id")
        if request_id is None:
            request_id = uuid.uuid4().hex
            record["operator_cancel_request_id"] = request_id
        if not isinstance(request_id, str) or not re.fullmatch(r"[0-9a-f]{32}", request_id):
            raise ToolError("operator cancellation request identity is invalid", failure_class="command_infra")
        record["operator_cancel_attempts"] = attempts + 1
        record["operator_cancel_state"] = "REQUESTED"
        context = dict(record["payload"]["context"])
        context["_receipt_issued_at"] = int(record["payload"]["issued_at"])
        return context, None, request_id


def record_operator_cancel_unknown(token):
    with locked_record("receipt", token) as record:
        record["operator_cancel_state"] = "STATE_UNKNOWN"


def close_operator_cancel(token, closeout):
    with locked_record("receipt", token) as record:
        record["terminal_closeout"] = closeout
        record["finish_closed"] = "CANCELLED_BY_OPERATOR"
        record["operator_cancel_state"] = "CANCELLED"
        record["finish_not_before_unix"] = 0
        record["pending_finish_response"] = None


def operator_cancel_result(token, closeout, request_id, *, cached=False):
    cancellation = closeout.get("operator_cancellation", {})
    return typed_result(
        True, "CANCELLED_BY_OPERATOR",
        observed={
            "request_id": cancellation.get("request_id") or request_id,
            "requested_at_unix": cancellation.get("requested_at_unix"),
            "completed_units": cancellation.get("completed_units", 0),
            "total_units": cancellation.get("total_units", 0),
            "stage": cancellation.get("stage"),
            "termination_mode": cancellation.get("termination_mode"),
            "cached": cached,
            "scientific_result_interpretable": False,
        },
        next_actions=["preserve_cancellation_audit", "design_new_run_if_needed"],
        receipt=token, archive_authorized=False, relaunch_authorized=False,
        scientific_authorization="NONE", cached=cached,
    )


def cancel_operator_route(token):
    context, cached_closeout, request_id = begin_operator_cancel(token)
    if cached_closeout is not None:
        return operator_cancel_result(
            token, cached_closeout, request_id, cached=True,
        )
    try:
        output = run_remote(
            operator_cancel_body(context, request_id),
            timeout=OPERATOR_CANCEL_GRACE_SECONDS + OPERATOR_CANCEL_FORCE_SECONDS + 30,
            phase="operator_cancel",
        )
    except ToolError:
        record_operator_cancel_unknown(token)
        raise
    observed = parse_operator_cancel(output)
    closeout = parse_closeout(context, output)
    if closeout is None:
        if not observed["active"] and not observed["terminal"]:
            close_finish(token, "CLOSEOUT_MISSING")
            return typed_failure(
                "CLOSEOUT_MISSING", "evidence",
                "session ended before cancellation and has no closeout",
                observed={**observed, "request_id": request_id},
                next_actions=["engineering_review_once"],
                failure_phase="closeout", receipt=token,
            )
        raise ToolError(
            "operator cancellation did not produce a closeout",
            failure_phase="operator_cancel", failure_class="evidence",
        )
    terminal = closeout["terminal_tuple"]
    if terminal == OPERATOR_CANCEL_TERMINAL:
        cancellation = closeout.get("operator_cancellation", {})
        if cancellation.get("request_id") != request_id:
            raise ToolError(
                "operator cancellation closeout request identity mismatch",
                failure_phase="operator_cancel", failure_class="evidence",
            )
        close_operator_cancel(token, closeout)
        return operator_cancel_result(token, closeout, request_id)
    if terminal["state"] == "FAILED_ENGINEERING":
        closeout["engineering_diagnostic"]["last_status"] = None
        authorize_engineering_auto_repair(token, closeout)
        failure_phase = closeout["engineering_diagnostic"]["failure_phase"]
        return typed_failure(
            "ENGINEERING_AUTO_REPAIR_AUTHORIZED",
            engineering_failure_class(failure_phase),
            "the route reached an engineering terminal before cancellation took effect",
            observed={"closeout": closeout, **observed},
            next_actions=["inspect_failure_once", "prepare_one_same_contract_engineering_repair"],
            failure_phase=failure_phase, archive_authorized=False,
            relaunch_authorized=False, receipt=token,
        )
    return validated_scientific_result(
        token, closeout, {"closeout": closeout, **observed},
    )


def tool_finish(args):
    token = args.get("receipt")
    try:
        resolution = args.get("engineering_failure_resolution")
        operator_action = args.get("operator_action", "observe")
        observation_mode = args.get("observation_mode", "sealed")
        if operator_action not in {"observe", "cancel"}:
            raise ToolError("operator_action must be observe or cancel")
        if observation_mode not in {"sealed", "progress_only"}:
            raise ToolError("observation_mode must be sealed or progress_only")
        if resolution is not None and (
                operator_action != "observe" or observation_mode != "sealed"):
            raise ToolError(
                "engineering failure resolution cannot be combined with operator control"
            )
        if resolution is not None:
            return resolve_engineering_failure(token, resolution)
        if operator_action == "cancel":
            if observation_mode != "sealed":
                raise ToolError("cancellation cannot be combined with progress_only")
            return cancel_operator_route(token)
        if observation_mode == "progress_only":
            return observe_operator_progress(token)
        context, cached = begin_finish(token)
        if cached is not None:
            return cached
        profile = MONITOR_PROFILES[context["monitor_profile"]]
        output = run_remote(
            monitor_body(context, profile),
            timeout=profile["max_polls"] * profile["interval_seconds"] + 20,
            phase="monitor",
        )
        monitor = parse_monitor(output)
        if monitor["stale"]:
            stale_count = record_stale_observation(token)
            return typed_failure(
                "MONITOR_STALE", "command_infra", "heartbeat exceeded the sealed limit",
                observed={**monitor, "receipt_stale_observations": stale_count},
                next_actions=["wait_until_expected_end_then_convir_route_finish", "engineering_review_once"],
                failure_phase="monitor", receipt_remains_open=True,
            )
        closeout = parse_closeout(context, output)
        if closeout:
            if closeout["terminal_tuple"]["state"] == "FAILED_ENGINEERING":
                closeout["engineering_diagnostic"]["last_status"] = safe_status_summary(
                    monitor["status"],
                )
                authorize_engineering_auto_repair(token, closeout)
                failure_phase = closeout["engineering_diagnostic"]["failure_phase"]
                return typed_failure(
                    "ENGINEERING_AUTO_REPAIR_AUTHORIZED", engineering_failure_class(failure_phase),
                    "engineering failure was detected before a healthy workload claim; one same-contract repair is authorized automatically, while sensitive changes still require review",
                    observed={"monitor": monitor, "closeout": closeout},
                    next_actions=["inspect_failure_once", "prepare_one_same_contract_engineering_repair"],
                    failure_phase=failure_phase,
                    archive_authorized=False, relaunch_authorized=False, receipt=token,
                )
            return validated_scientific_result(
                token, closeout, {"monitor": monitor, "closeout": closeout},
                manifest={
                    "closeout_filename": closeout["closeout_filename"],
                    "closeout_sha256": closeout["closeout_sha256"],
                },
            )
        if not monitor["active"]:
            close_finish(token, "CLOSEOUT_MISSING")
            return typed_failure(
                "CLOSEOUT_MISSING", "evidence", "session ended without closeout",
                observed=monitor, next_actions=["engineering_review_once"], failure_phase="closeout",
            )
        progress = workload_progress(monitor["status"])
        if progress["completed_units"] > 0:
            record_workload_verified(token)
            now = int(time.time())
            expected_end = now + max(30, context.get("expected_wall_seconds", 30))
            retry_after = expected_end - now
            result = typed_result(
                True, "RUNNING_VERIFIED",
                observed={**monitor, "workload_progress": progress},
                next_actions=["wait_until_expected_end_then_convir_route_finish"],
                receipt=token, workload_verified=True,
                retry_after_seconds=retry_after,
                not_before_unix=now + retry_after,
                expected_phase_end_unix=expected_end,
            )
            cache_finish_response(token, result, now + retry_after)
            return result
        now = int(time.time())
        retry_after = 30
        expected_phase_end = max(
            now + retry_after,
            context["_receipt_issued_at"]
            + context.get("engineering_max_seconds", retry_after),
        )
        result = typed_result(
            True, "LAUNCHED_PENDING_VERIFICATION",
            observed={
                **monitor, "workload_progress": progress,
                "contract_progress": contract_progress(monitor["status"]),
            },
            next_actions=["convir_route_finish_after_startup_interval"],
            receipt=token, workload_verified=False,
            retry_after_seconds=retry_after, not_before_unix=now + retry_after,
            expected_phase_end_unix=expected_phase_end,
        )
        cache_finish_response(token, result, now + retry_after)
        return result
    except (json.JSONDecodeError, TypeError) as exc:
        return failure_result("FINISH_INVALID", ToolError(str(exc), failure_class="evidence"), "closeout")
    except Exception as exc:
        return failure_result("FINISH_REJECTED", exc, "monitor")


def validate_evidence_name(name):
    if not isinstance(name, str) or Path(name).name != name:
        raise ToolError("evidence filename must be top-level")
    if not SAFE_TOKEN.fullmatch(Path(name).stem):
        raise ToolError("evidence filename stem is invalid")
    if Path(name).suffix.lower() not in ALLOWED_EVIDENCE_SUFFIXES or "cloud_only" in name.lower():
        raise ToolError("evidence file is not compact-text eligible")
    return name


def evidence_manifest_body(context, names=None):
    closeout_name = validate_evidence_name(context["validated_closeout_filename"])
    closeout_sha = require_sha(
        context["validated_closeout_sha256"], "validated_closeout_sha256", SHA256,
    )
    lines = [
        "export LC_ALL=C",
        f"EVIDENCE_DIR={q(context['evidence_dir'])}",
        'test -d "$EVIDENCE_DIR"',
        'test ! -L "$EVIDENCE_DIR"',
        'test "$(readlink -f -- "$EVIDENCE_DIR")" = "$EVIDENCE_DIR"',
        f'VALIDATED_CLOSEOUT="$EVIDENCE_DIR/{closeout_name}"',
        'test -f "$VALIDATED_CLOSEOUT"',
        'test ! -L "$VALIDATED_CLOSEOUT"',
        'test "$(readlink -f -- "$VALIDATED_CLOSEOUT")" = "$VALIDATED_CLOSEOUT"',
        f'test "$(sha256sum "$VALIDATED_CLOSEOUT" | awk \'{{print $1}}\')" = {q(closeout_sha)}',
    ]
    if names is None:
        lines.extend([
            'shopt -s nullglob',
            'for path in "$EVIDENCE_DIR"/*; do',
            '  test -f "$path" || continue',
            '  test ! -L "$path" || continue',
            '  test "$(readlink -f -- "$path")" = "$path" || continue',
            '  name=$(basename "$path")',
            '  [[ "$name" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\.(json|csv|md|txt)$ ]] || continue',
            '  case "$name" in *cloud_only*|*CLOUD_ONLY*) continue ;; esac',
            '  size=$(wc -c < "$path")',
            f'  test "$size" -le {MAX_EVIDENCE_BYTES} || continue',
            '  read -r digest _ < <(sha256sum "$path")',
            '  printf "%s\\t%s\\t%s\\n" "$name" "$size" "$digest"',
            'done',
        ])
    else:
        for name in names:
            lines.extend([
                f'path="$EVIDENCE_DIR/{name}"',
                'test -f "$path"',
                'test ! -L "$path"',
                'test "$(readlink -f -- "$path")" = "$path"',
                'size=$(wc -c < "$path")',
                f'test "$size" -le {MAX_EVIDENCE_BYTES}',
                'read -r digest _ < <(sha256sum "$path")',
                f'printf "%s\\t%s\\t%s\\n" {q(name)} "$size" "$digest"',
            ])
    lines.append("echo CONVIR_OPS_EVIDENCE_MANIFEST_OK")
    return "\n".join(lines)


def parse_evidence_manifest(output):
    if output.splitlines().count("CONVIR_OPS_EVIDENCE_MANIFEST_OK") != 1:
        raise ToolError(
            "evidence manifest marker is missing or duplicated",
            failure_phase="evidence_manifest", failure_class="command_infra",
        )
    result = {}
    for line in output.splitlines():
        if line in {"CONVIR_OPS_EVIDENCE_MANIFEST_OK", "CONVIR_REMOTE_SCRIPT_OK", ""}:
            continue
        fields = line.split("\t")
        if len(fields) != 3 or not fields[1].isdigit() or not SHA256.fullmatch(fields[2]):
            raise ToolError(
                "malformed evidence manifest record",
                failure_phase="evidence_manifest", failure_class="command_infra",
            )
        name = validate_evidence_name(fields[0])
        if name in result:
            raise ToolError("duplicate evidence record", failure_class="command_infra")
        result[name] = {"bytes": int(fields[1]), "sha256": fields[2]}
    return result


def tool_evidence_manifest(args):
    try:
        context = evidence_context(args)
        records = parse_evidence_manifest(
            run_remote(evidence_manifest_body(context), timeout=60, phase="evidence_manifest")
        )
        value = {
            "route_id": context["route_id"],
            "files": [{"name": name, **record} for name, record in sorted(records.items())],
            "marker": "CONVIR_OPS_EVIDENCE_MANIFEST_OK",
        }
        if "archive_contract" in context:
            value["archive_contract"] = context["archive_contract"]
        return text_result(json.dumps(value, indent=2), structured=value)
    except Exception as exc:
        return failure_result("EVIDENCE_MANIFEST_FAILED", exc, "evidence_manifest")


def validate_local_repo(value):
    if not isinstance(value, str) or not value.startswith("/"):
        raise ToolError("local_repo must be an absolute WSL path")
    path = Path(value).resolve()
    try:
        path.relative_to(LOCAL_WORKSPACE_ROOT)
    except ValueError as exc:
        raise ToolError("local_repo must stay under the workspace root") from exc
    if not (path / ".git").exists():
        raise ToolError("local_repo must be a Git worktree")
    top = run_local(
        ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
        timeout=30, phase="local_repo_identity",
    )
    if Path(top).resolve() != path:
        raise ToolError("local_repo must be the exact Git worktree root")
    return path


def ensure_real_directory_chain(root, relative):
    root = Path(root).resolve(strict=True)
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ToolError("destination directory is outside the repository")
    current = root
    for part in relative.parts:
        current = current / part
        try:
            current.mkdir(mode=0o700)
        except FileExistsError:
            pass
        try:
            observed = current.lstat()
        except OSError as exc:
            raise ToolError("destination directory is unavailable") from exc
        if stat.S_ISLNK(observed.st_mode) or not stat.S_ISDIR(observed.st_mode):
            raise ToolError("destination directory cannot contain symlinks")
    try:
        current.resolve(strict=True).relative_to(root)
    except ValueError as exc:
        raise ToolError("destination directory escaped the repository") from exc
    return current


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _terminal_tuple_from_record(record):
    return {
        key: record.get(key) for key in ("state", "decision", "authorizes")
    }


def _select_terminal_leaf(records):
    if len(records) == 1 and records[0].get("schema_version") != 2:
        return records[0]
    if any(record.get("schema_version") != 2 for record in records):
        return None
    by_closeout = {}
    operation_ids = set()
    run_ids = set()
    for record in records:
        closeout_path = record.get("closeout_path")
        operation_id = record.get("operation_id")
        run_id = record.get("run_id")
        binding = record.get("prior_terminal_record")
        if not isinstance(closeout_path, str) or closeout_path in by_closeout \
                or not isinstance(operation_id, str) or operation_id in operation_ids \
                or not isinstance(run_id, str) or run_id in run_ids \
                or not isinstance(binding, dict) or set(binding) != {
                    "prior_closeout_path", "prior_terminal_tuple",
                }:
            return None
        by_closeout[closeout_path] = record
        operation_ids.add(operation_id)
        run_ids.add(run_id)
    referenced = set()
    roots = []
    for record in records:
        binding = record["prior_terminal_record"]
        parent_path = binding["prior_closeout_path"]
        prior_tuple = binding["prior_terminal_tuple"]
        if parent_path is None:
            if prior_tuple is not None:
                return None
            roots.append(record)
            continue
        parent = by_closeout.get(parent_path)
        if parent is None or prior_tuple != _terminal_tuple_from_record(parent):
            return None
        referenced.add(parent_path)
    leaves = [
        record for path, record in by_closeout.items() if path not in referenced
    ]
    if len(roots) != 1 or len(leaves) != 1:
        return None
    visited = set()
    cursor = leaves[0]
    while cursor is not None:
        path = cursor["closeout_path"]
        if path in visited:
            return None
        visited.add(path)
        parent_path = cursor["prior_terminal_record"]["prior_closeout_path"]
        cursor = None if parent_path is None else by_closeout.get(parent_path)
    return leaves[0] if len(visited) == len(records) else None


def _snapshot_blob(repo, ref, relpath):
    if not isinstance(relpath, str) or not relpath:
        return None
    completed = subprocess.run(
        ["git", "-C", str(repo), "show", f"{ref}:{relpath}"],
        capture_output=True, timeout=30, check=False,
    )
    return completed.stdout if completed.returncode == 0 else None


def worktree_route_identity(repo, route_id, ref):
    """Bind the requested route id to the committed route manifest when present."""
    raw = _snapshot_blob(repo, ref, ROUTE_OPERATIONS_RELPATH)
    if raw is None:
        return {
            "status": "ROUTE_ID_UNRESOLVED_NO_MANIFEST",
            "requested_route_id": route_id,
            "manifest_route_id": None,
            "route_id_confirmed": False,
        }
    try:
        manifest = json.loads(raw)
    except json.JSONDecodeError:
        return {
            "status": "ROUTE_ID_UNRESOLVED_INVALID_MANIFEST",
            "requested_route_id": route_id,
            "manifest_route_id": None,
            "route_id_confirmed": False,
        }
    manifest_route_id = manifest.get("route_id") if isinstance(manifest, dict) else None
    if manifest_route_id == route_id:
        return {
            "status": "ROUTE_ID_CONFIRMED_BY_HEAD_MANIFEST",
            "requested_route_id": route_id,
            "manifest_route_id": manifest_route_id,
            "route_id_confirmed": True,
        }
    return {
        "status": "ROUTE_ID_MISMATCH_WITH_HEAD_MANIFEST",
        "requested_route_id": route_id,
        "manifest_route_id": manifest_route_id,
        "route_id_confirmed": False,
    }


def authoritative_snapshot(repo, route_id, ref, *, route_id_confirmed=False):
    """Return a terminal-record pointer without reading result contents."""
    if not route_id_confirmed:
        return {
            "status": "ROUTE_ID_UNRESOLVED",
            "route_id": route_id,
            "route_id_confirmed": False,
        }
    index_path = "experience_docx/EXPERIMENT_TERMINAL_INDEX.jsonl"
    completed = subprocess.run(
        ["git", "-C", str(repo), "show", f"{ref}:{index_path}"],
        capture_output=True, timeout=30, check=False,
    )
    if completed.returncode:
        return {"status": "TERMINAL_INDEX_UNAVAILABLE", "route_id": route_id}
    matches = []
    record_hashes = {}
    try:
        for raw in completed.stdout.decode("utf-8").splitlines():
            if not raw.strip():
                continue
            item = json.loads(raw)
            if isinstance(item, dict) and item.get("route_id") == route_id:
                matches.append(item)
                record_hashes[id(item)] = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {"status": "TERMINAL_INDEX_INVALID", "route_id": route_id}
    if not matches:
        return {
            "status": (
                "NO_TERMINAL_RECORD" if route_id_confirmed
                else "ROUTE_ID_UNRESOLVED"
            ),
            "route_id": route_id,
            "route_id_confirmed": route_id_confirmed,
        }
    record = _select_terminal_leaf(matches)
    if record is None:
        return {
            "status": "TERMINAL_RECORD_AMBIGUOUS",
            "route_id": route_id,
            "terminal_record_count": len(matches),
            "operation_ids": sorted({
                str(item.get("operation_id")) for item in matches
            }),
        }
    result_files = record.get("result_files", [])
    contract_bundle = record.get("contract_bundle", [])
    return {
        "status": "AUTHORITATIVE_SNAPSHOT_OK",
        "route_id": route_id,
        "route_id_confirmed": True,
        "terminal_index_path": index_path,
        "terminal_record_sha256": record_hashes[id(record)],
        "operation_id": record.get("operation_id"),
        "run_id": record.get("run_id"),
        "contract_path": record.get("contract_path"),
        "closeout_path": record.get("closeout_path"),
        "conclusion_path": record.get("conclusion_path"),
        "result_path_count": len(
            result_files if record.get("schema_version") == 2
            else record.get("result_paths", [])
        ),
        "contract_bundle_file_count": len(contract_bundle) if isinstance(contract_bundle, list) else 0,
        "scientific_authorization": "NOT_DERIVED",
    }


def tool_evidence_fetch(args):
    try:
        context = evidence_context(args)
        local_repo = validate_local_repo(args.get("local_repo"))
        files = args.get("files")
        if not isinstance(files, list) or not 1 <= len(files) <= 32:
            raise ToolError("files must contain 1-32 names")
        names = [validate_evidence_name(name) for name in files]
        if len(names) != len(set(names)):
            raise ToolError("files contains duplicates")
        records = parse_evidence_manifest(
            run_remote(evidence_manifest_body(context, names), timeout=60, phase="evidence_manifest")
        )
        if set(records) != set(names):
            raise ToolError("evidence allowlist did not match exactly", failure_class="command_infra")
        destination_dir = ensure_real_directory_chain(
            local_repo,
            Path("experience_docx") / "experiment_logs" / context["route_id"],
        )
        fetched, verified, pending = [], [], []
        for name in names:
            destination = destination_dir / name
            if os.path.lexists(destination):
                observed = destination.lstat()
                if stat.S_ISLNK(observed.st_mode) or not stat.S_ISREG(observed.st_mode):
                    raise ToolError(f"refusing non-regular destination {name}")
                if sha256_file(destination) != records[name]["sha256"]:
                    raise ToolError(f"refusing to overwrite mismatched {name}")
                verified.append(name)
            else:
                pending.append(name)
        if pending:
            with tempfile.TemporaryDirectory(prefix=".convir-evidence-", dir=destination_dir) as stage:
                sources = [f"{REMOTE_HOST}:{context['evidence_dir']}/{name}" for name in pending]
                run_local(["scp", *sources, stage], timeout=300, phase="evidence_transfer")
                for name in pending:
                    source = Path(stage) / name
                    try:
                        observed = source.lstat()
                    except OSError as exc:
                        raise ToolError(f"downloaded evidence is missing: {name}") from exc
                    if stat.S_ISLNK(observed.st_mode) or not stat.S_ISREG(observed.st_mode) \
                            or source.resolve(strict=True).parent != Path(stage).resolve(strict=True) \
                            or sha256_file(source) != records[name]["sha256"]:
                        raise ToolError(f"downloaded evidence hash mismatch: {name}", failure_class="command_infra")
                    try:
                        os.link(source, destination_dir / name)
                    except FileExistsError as exc:
                        raise ToolError(f"refusing concurrent overwrite: {name}") from exc
                    fetched.append(name)
        value = {
            "fetched": fetched, "already_verified": verified,
            "destination": str(destination_dir), "git_mutations_performed": False,
        }
        if "archive_contract" in context:
            value["archive_contract"] = context["archive_contract"]
        return text_result(json.dumps(value, indent=2), structured=value)
    except Exception as exc:
        return failure_result("EVIDENCE_FETCH_FAILED", exc, "evidence_transfer")


def build_snapshot_phase_receipt(value):
    snapshot = value.get("authoritative_snapshot")
    snapshot_status = snapshot.get("status") if isinstance(snapshot, dict) else None
    route_identity = value.get("route_identity")
    route_id_confirmed = bool(
        isinstance(route_identity, dict) and route_identity.get("route_id_confirmed")
    )
    if not value.get("github_main_ref_fresh"):
        allowed_next_action = "refresh_github_main_once"
    elif snapshot_status == "AUTHORITATIVE_SNAPSHOT_OK" and route_id_confirmed:
        allowed_next_action = "read_authoritative_snapshot_references"
    else:
        allowed_next_action = "resolve_route_identity_or_snapshot"
    receipt = {
        "schema_version": 1,
        "phase": "SNAPSHOT_IDENTITY",
        "route_id": value.get("authoritative_snapshot", {}).get("route_id"),
        "branch": value.get("branch"),
        "head": value.get("head"),
        "github_main_commit": value.get("github_main_remote"),
        "github_main_ref_fresh": value.get("github_main_ref_fresh") is True,
        "worktree_clean": value.get("worktree_clean") is True,
        "authoritative_snapshot_status": snapshot_status,
        "route_id_confirmed": route_id_confirmed,
        "route_id_binding_status": (
            route_identity.get("status") if isinstance(route_identity, dict) else None
        ),
        "scientific_authorization": "NOT_DERIVED",
        "allowed_next_action": allowed_next_action,
    }
    return {**receipt, "receipt_sha256": canonical_digest(receipt)}


def tool_git_evidence_status(args):
    try:
        route_id = require_token(args.get("route_id"), "route_id")
        repo = validate_local_repo(args.get("local_repo"))
        prefix = ["git", "-C", str(repo)]
        branch = run_local([*prefix, "branch", "--show-current"], timeout=30, phase="git_status")
        head = run_local([*prefix, "rev-parse", "HEAD"], timeout=30, phase="git_status")
        local_main = run_local([*prefix, "rev-parse", "github/main"], timeout=30, phase="git_status")
        remote = run_local([*prefix, "ls-remote", "github", "refs/heads/main"], timeout=60, phase="github_ref_fetch").split()
        if len(remote) != 2 or not SHA40.fullmatch(remote[0]):
            raise ToolError("GitHub main is malformed", failure_class="command_infra")
        detail = args.get("detail", "summary")
        if detail not in {"summary", "route", "full"}:
            raise ToolError("detail must be summary, route, or full")
        status = run_local([*prefix, "status", "--short"], timeout=30, phase="git_status")
        changed_all = status.splitlines() if status else []
        route_prefix = f"experience_docx/experiment_logs/{route_id}/"
        route_changed = [line for line in changed_all if route_prefix in line]

        route_identity = worktree_route_identity(repo, route_id, head)
        snapshot = authoritative_snapshot(
            repo, route_id, "github/main",
            route_id_confirmed=route_identity["route_id_confirmed"],
        )
        value = {
            "local_repo": str(repo), "branch": branch, "head": head,
            "github_main_local": local_main, "github_main_remote": remote[0],
            "github_main_ref_fresh": local_main == remote[0],
            "worktree_clean": not status,
            "changed_path_count": len(changed_all),
            "route_evidence_change_count": len(route_changed),
            "route_identity": route_identity,
            "authoritative_snapshot": snapshot,
            "detail_level": detail,
            "git_mutations_performed": False,
        }
        if detail in {"route", "full"}:
            value["route_evidence_changes"] = route_changed[:100]
            value["route_evidence_changes_truncated"] = len(route_changed) > 100
        if detail == "full":
            value["changed_paths"] = changed_all[:100]
            value["changed_paths_truncated"] = len(changed_all) > 100
        value["phase_receipt"] = build_snapshot_phase_receipt(value)
        return text_result(json.dumps(value, indent=2), structured=value)
    except Exception as exc:
        return failure_result("GIT_STATUS_FAILED", exc, "git_status")


TOOLS = {
    "convir_route_plan": {
        "description": "Read and seal one legacy or canonical-contract route through the stable schema-v4 control protocol without contacting the cloud.",
        "inputSchema": {
            "type": "object",
            "required": ["schema_version", "branch", "route_branch_commit", "operation_id"],
            "properties": {
                "schema_version": {"const": 4}, "branch": {"type": "string"},
                "route_branch_commit": {"type": "string"}, "operation_id": {"type": "string"},
            },
            "additionalProperties": False,
        },
        "handler": tool_plan_manifest,
    },
    "convir_route_start": {
        "description": "Apply a reviewed plan once, then perform one bounded startup observation; return RUNNING_VERIFIED only after positive workload progress, otherwise expose pending verification or an early failure.",
        "inputSchema": {
            "type": "object", "required": ["plan_token"],
            "properties": {"plan_token": {"type": "string"}}, "additionalProperties": False,
        },
        "handler": tool_start,
    },
    "convir_route_finish": {
        "description": "Observe a sealed window, refresh result-blind progress, detect an early terminal, or perform receipt-bound operator cancellation; validate every closeout and keep scientific decisions explicit.",
        "inputSchema": {
            "type": "object", "required": ["receipt"],
            "properties": {
                "receipt": {"type": "string"},
                "engineering_failure_resolution": {"enum": ["repair", "archive", "discard"]},
                "operator_action": {
                    "enum": ["observe", "cancel"], "default": "observe",
                },
                "observation_mode": {
                    "enum": ["sealed", "progress_only"], "default": "sealed",
                },
            },
            "additionalProperties": False,
        },
        "handler": tool_finish,
    },
    "convir_evidence_list": {
        "description": "List compact top-level evidence from the workspace sealed by a launch receipt.",
        "inputSchema": {
            "type": "object", "required": ["receipt"],
            "properties": {"receipt": {"type": "string"}}, "additionalProperties": False,
        },
        "handler": tool_evidence_manifest,
    },
    "convir_evidence_fetch": {
        "description": "Fetch a compact evidence allowlist from the receipt-bound workspace with SHA-256 verification; never stage or push Git.",
        "inputSchema": {
            "type": "object", "required": ["receipt", "local_repo", "files"],
            "properties": {
                "receipt": {"type": "string"}, "local_repo": {"type": "string"},
                "files": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 32},
            },
            "additionalProperties": False,
        },
        "handler": tool_evidence_fetch,
    },
    "convir_git_status": {
        "description": "Read-only compact worktree, GitHub-main freshness, and authoritative route snapshot audit; defaults to a token-bounded summary and never mutates Git.",
        "inputSchema": {
            "type": "object", "required": ["route_id", "local_repo"],
            "properties": {
                "route_id": {"type": "string"}, "local_repo": {"type": "string"},
                "detail": {"enum": ["summary", "route", "full"], "default": "summary"},
            },
            "additionalProperties": False,
        },
        "handler": tool_git_evidence_status,
    },
}


def handle(request):
    method = request.get("method")
    params = request.get("params") or {}
    if method == "initialize":
        return {
            "protocolVersion": params.get("protocolVersion", "2024-11-05"),
            "capabilities": {"tools": {}},
            "serverInfo": {
                "name": SERVER_NAME, "version": SERVER_VERSION,
                "sourceSha256": SERVER_SOURCE_SHA256,
            },
        }
    if method == "ping":
        return {}
    if method == "tools/list":
        return {"tools": [{"name": name, "description": item["description"], "inputSchema": item["inputSchema"]} for name, item in TOOLS.items()]}
    if method == "tools/call":
        name = params.get("name")
        args = params.get("arguments")
        if name not in TOOLS or not isinstance(args, dict):
            raise ToolError("invalid MCP tool call")
        schema = TOOLS[name]["inputSchema"]
        unknown = set(args) - set(schema["properties"])
        missing = set(schema["required"]) - set(args)
        if unknown or missing:
            raise ToolError(f"tool argument mismatch unknown={sorted(unknown)} missing={sorted(missing)}")
        return TOOLS[name]["handler"](args)
    raise ToolError(f"unsupported method: {method}")


def main():
    for line in sys.stdin:
        request_id = None
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ToolError("MCP request must be an object")
            request_id = request.get("id")
            result = handle(request)
            if request_id is not None:
                emit({"jsonrpc": "2.0", "id": request_id, "result": result})
        except Exception as exc:
            if request_id is not None:
                emit({"jsonrpc": "2.0", "id": request_id, "error": {"code": -32000, "message": str(exc)}})


if __name__ == "__main__":
    main()
