#!/usr/bin/env python3
"""Restricted MCP operations bridge for the persistent convir-4090 host.

The server intentionally exposes route operations, not arbitrary SSH execution.
It runs locally under WSL and delegates PowerShell/WSL/SSH transport to the
tracked convir_remote_script.sh wrapper.
"""

import hashlib
import hmac
import json
import os
import re
import shlex
import subprocess
import sys
import fcntl
import time
import uuid
import tempfile
from contextlib import contextmanager
from pathlib import Path


SERVER_NAME = "convir-ops-v2"
SERVER_VERSION = "2.0.0"
REMOTE_HOST = "convir-4090"
REMOTE_BASE = "/sda/home/wangyuxin/ConvIR-B"
REMOTE_REPOS = f"{REMOTE_BASE}/repos"
REMOTE_RUNS = f"{REMOTE_BASE}/runs"
REMOTE_PYTHON = f"{REMOTE_BASE}/envs/convir-cu121/bin/python"
GITHUB_URL = "git@github.com:onenoober/ConvIR-B.git"
ROUTE_OPERATIONS_RELPATH = "experience_docx/route_operations.json"
LOCAL_WORKSPACE_ROOT = Path(os.environ.get("CONVIR_OPS_LOCAL_WORKSPACE_ROOT", "/home/ubuntu/workspace")).resolve()
MAX_EVIDENCE_BYTES = 1024 * 1024
SAFE_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
SAFE_BRANCH = re.compile(r"^codex/[A-Za-z0-9][A-Za-z0-9_.\-/]{0,191}$")
SHA40 = re.compile(r"^[0-9a-f]{40}$")
ALLOWED_EVIDENCE_SUFFIXES = {".json", ".csv", ".md", ".txt"}
SCHEMA_VERSION = 2
RECEIPT_TTL_SECONDS = 15 * 60
MONITOR_MAX_POLLS = 64
MONITOR_MAX_INTERVAL_SECONDS = 60
MAX_CLOSEOUT_BYTES = 64 * 1024
MAX_MONITOR_SECONDS = 480
START_RESOURCE_ATTEMPTS = 2
REMOTE_PREFLIGHT_TIMEOUT = 120
REMOTE_LAUNCH_TIMEOUT = 150
REMOTE_CLOSEOUT_TIMEOUT = 60
MONITOR_PROFILES = {
    "short": {"max_polls": 9, "interval_seconds": 15},
    "standard": {"max_polls": 21, "interval_seconds": 15},
    "long": {"max_polls": 33, "interval_seconds": 15},
}
RECEIPT_DIR = Path(os.environ.get("CONVIR_OPS_RECEIPT_DIR", "~/.codex/convir-ops/receipts")).expanduser().resolve()


class ToolError(RuntimeError):
    """Expected user-facing tool rejection with a typed operational phase."""

    def __init__(self, message, *, failure_phase=None, failure_class=None):
        super().__init__(message)
        self.failure_phase = failure_phase
        self.failure_class = failure_class


def emit(value):
    sys.stdout.write(json.dumps(value, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def mcp_error(request_id, code, message):
    emit({"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}})


def mcp_result(request_id, result):
    emit({"jsonrpc": "2.0", "id": request_id, "result": result})


def text_result(text, is_error=False, structured_content=None):
    result = {"content": [{"type": "text", "text": text}], "isError": is_error}
    if structured_content is not None:
        result["structuredContent"] = structured_content
    return result


def require_token(value, name):
    if not isinstance(value, str) or not SAFE_TOKEN.fullmatch(value):
        raise ToolError(f"{name} must contain only letters, digits, dot, underscore, or hyphen")
    return value


def require_mode(value):
    """A mode is an opaque bounded token; route runners own its semantics."""
    return require_token(value, "mode")


def derive_session(route_id, mode, route_commit, output_id):
    readable = f"convir-{route_id[:18]}-{mode[:12]}-{output_id[:12]}"
    digest = hashlib.sha256(f"{route_id}\0{mode}\0{route_commit}\0{output_id}".encode("utf-8")).hexdigest()[:16]
    return f"{readable[:47]}-{digest}"[:64]


def require_branch(value):
    if not isinstance(value, str) or not SAFE_BRANCH.fullmatch(value):
        raise ToolError("branch must be a codex/<safe-route-name> branch")
    return value


def require_commit(value):
    if not isinstance(value, str) or not SHA40.fullmatch(value):
        raise ToolError("expected_commit must be a 40-character lowercase Git SHA")
    return value


def require_runner(value):
    if not isinstance(value, str) or value.startswith("/") or ".." in Path(value).parts:
        raise ToolError("runner_relpath must be a safe relative path")
    path = Path(value)
    if path.suffix != ".sh" or not re.fullmatch(r"experience_docx/tools/run_[A-Za-z0-9_.-]+\.sh", str(path)):
        raise ToolError("runner_relpath must be an experience_docx/tools/*.sh route runner")
    return str(path)


def require_repo_relpath(value, name):
    if not isinstance(value, str) or value.startswith("/") or ".." in Path(value).parts:
        raise ToolError(f"{name} must be a safe repository-relative path")
    if not re.fullmatch(r"experience_docx/[A-Za-z0-9_.\-/]+\.json", value):
        raise ToolError(f"{name} must be an experience_docx/*.json path")
    return value


def require_closeout_filename(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+_closeout\.json", value):
        raise ToolError("closeout_filename must be a compact *_closeout.json filename")
    return value


def require_terminal_tuple(value, name):
    if not isinstance(value, dict) or set(value) != {"state", "decision", "authorizes"}:
        raise ToolError(f"{name} must contain exactly state, decision, and authorizes")
    return {key: require_token(value[key], f"{name}.{key}") for key in ("state", "decision", "authorizes")}


def require_terminal_tuples(value):
    if not isinstance(value, list) or not value or len(value) > 16:
        raise ToolError("allowed_terminal_tuples must contain 1-16 typed tuples")
    tuples = [require_terminal_tuple(item, "allowed_terminal_tuples entry") for item in value]
    if len({canonical_digest(item) for item in tuples}) != len(tuples):
        raise ToolError("allowed_terminal_tuples must not contain duplicates")
    return tuples


def require_bool(value, name, default):
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ToolError(f"{name} must be boolean")
    return value


def require_int(value, name, default, minimum, maximum):
    if value is None:
        return default
    if not isinstance(value, int) or isinstance(value, bool) or not minimum <= value <= maximum:
        raise ToolError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def require_enum(value, name, choices):
    if value not in choices:
        raise ToolError(f"{name} must be one of: {', '.join(sorted(choices))}")
    return value


def require_string_list(value, name, maximum=16):
    if not isinstance(value, list) or len(value) > maximum or any(not isinstance(item, str) or not SAFE_TOKEN.fullmatch(item) for item in value):
        raise ToolError(f"{name} must be a list of at most {maximum} safe tokens")
    if len(value) != len(set(value)):
        raise ToolError(f"{name} must not contain duplicates")
    return value


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def typed_result(ok, operation_state, failure_class="none", *, observed=None, expected=None, mismatches=None, allowed_next_actions=None, **extra):
    payload = {
        "schema_version": SCHEMA_VERSION,
        "ok": ok,
        "operation_state": operation_state,
        "failure_class": failure_class,
        "observed": observed or {},
        "expected": expected or {},
        "mismatches": mismatches or [],
        "allowed_next_actions": allowed_next_actions or [],
    }
    payload.update(extra)
    payload["audit_digest"] = canonical_digest({key: value for key, value in payload.items() if key != "audit_digest"})
    return text_result(json.dumps(payload, sort_keys=True), is_error=not ok, structured_content=payload)


def typed_failure(operation_state, failure_class, message, **kwargs):
    return typed_result(False, operation_state, failure_class, mismatches=[message], **kwargs)


def failure_phase_for_error(error):
    phase = getattr(error, "failure_phase", None)
    if phase:
        return phase
    message = str(error)
    marker = re.search(r"(?:^|\s)phase=([A-Za-z0-9_.-]+)", message)
    return marker.group(1) if marker else "unknown"


def failure_details(error):
    return {"failure_phase": failure_phase_for_error(error)}


def structured_payload(result):
    payload = result.get("structuredContent")
    if not isinstance(payload, dict):
        raise ToolError("internal typed result is missing structuredContent")
    return payload


def failure_class_for_error(error):
    explicit = getattr(error, "failure_class", None)
    if explicit:
        return explicit
    message = str(error).lower()
    if "conflict" in message or "must_not_exist" in message or "collision" in message:
        return "collision"
    if "timed out" in message or "remote" in message or "local operation" in message:
        return "command_infra"
    return "authorization"


def route_context(arguments):
    route_id = require_token(arguments.get("route_id"), "route_id")
    repo_name = require_token(arguments.get("repo_name"), "repo_name")
    return {
        "route_id": route_id,
        "repo_name": repo_name,
        "run_root": f"{REMOTE_RUNS}/{route_id}",
        "evidence_dir": f"{REMOTE_REPOS}/{repo_name}/experience_docx/experiment_logs/{route_id}",
    }


def stage_context(arguments):
    context = route_context(arguments)
    context.update({
        "branch": require_branch(arguments.get("branch")),
        "expected_commit": require_commit(arguments.get("expected_commit")),
        "runner_relpath": require_runner(arguments.get("runner_relpath")),
        "mode": require_token(arguments.get("mode"), "mode"),
        "session": require_token(arguments.get("session"), "session"),
        "require_gpu": require_bool(arguments.get("require_gpu"), "require_gpu", True),
    })
    return context


def authorization_context(arguments):
    if arguments.get("schema_version") != SCHEMA_VERSION:
        raise ToolError("schema_version must be 2")
    context = stage_context({
        "route_id": arguments.get("route_id"), "repo_name": arguments.get("repo_name"),
        "branch": arguments.get("branch"), "expected_commit": arguments.get("route_branch_commit"),
        "runner_relpath": arguments.get("runner_relpath"), "mode": arguments.get("mode"),
        "session": "convir-placeholder", "require_gpu": arguments.get("require_gpu"),
    })
    context["rules_commit"] = require_commit(arguments.get("rules_commit"))
    context["mode"] = require_mode(context["mode"])
    context["stage_state"] = require_token(arguments.get("stage_state"), "stage_state")
    context["decision"] = require_token(arguments.get("decision"), "decision")
    context["authorizes"] = require_token(arguments.get("authorizes"), "authorizes")
    context["locked_test_policy"] = require_enum(arguments.get("locked_test_policy"), "locked_test_policy", {"blocked", "explicitly_authorized"})
    context["forbidden_continuations"] = require_string_list(arguments.get("forbidden_continuations"), "forbidden_continuations")
    context["output_id"] = require_token(arguments.get("output_id"), "output_id")
    context["session"] = derive_session(context["route_id"], context["mode"], context["expected_commit"], context["output_id"])
    context["workspace_id"] = require_token(arguments.get("workspace_id"), "workspace_id")
    context["workspace_policy"] = require_enum(arguments.get("workspace_policy"), "workspace_policy", {"fresh_route", "exact_continuation"})
    workspace_digest = hashlib.sha256(
        f"{context['repo_name']}\0{context['route_id']}\0{context['workspace_id']}".encode("utf-8")
    ).hexdigest()[:16]
    workspace_prefix = f"{context['repo_name'][:20]}-{context['route_id'][:24]}-{context['workspace_id'][:20]}"[:64]
    context["remote_repo"] = f"{REMOTE_REPOS}/{workspace_prefix}-{workspace_digest}"
    if arguments.get("collision_policy") != "must_not_exist":
        raise ToolError("collision_policy must be must_not_exist")
    context["output_path"] = f"{context['run_root']}/{context['output_id']}"
    context["closeout_filename"] = require_closeout_filename(arguments.get("closeout_filename"))
    context["closeout_path"] = f"{context['remote_repo']}/experience_docx/experiment_logs/{context['route_id']}/{context['closeout_filename']}"
    context["authorization_relpath"] = require_repo_relpath(arguments.get("authorization_relpath"), "authorization_relpath")
    context["prior_terminal_tuple"] = require_terminal_tuple(arguments.get("prior_terminal_tuple"), "prior_terminal_tuple")
    context["allowed_terminal_tuples"] = require_terminal_tuples(arguments.get("allowed_terminal_tuples"))
    context["monitor_profile"] = require_enum(arguments.get("monitor_profile"), "monitor_profile", set(MONITOR_PROFILES))
    context["heartbeat_timeout_seconds"] = require_int(arguments.get("heartbeat_timeout_seconds"), "heartbeat_timeout_seconds", 300, 30, 86400)
    context["min_free_gpu_mib"] = require_int(arguments.get("min_free_gpu_mib"), "min_free_gpu_mib", 0, 0, 1048576)
    context["max_gpu_utilization_pct"] = require_int(arguments.get("max_gpu_utilization_pct"), "max_gpu_utilization_pct", 100, 0, 100)
    if context["require_gpu"] and context["min_free_gpu_mib"] < 1:
        raise ToolError("GPU operations require min_free_gpu_mib >= 1")
    if not context["require_gpu"] and (context["min_free_gpu_mib"] != 0 or context["max_gpu_utilization_pct"] != 100):
        raise ToolError("non-GPU operations require min_free_gpu_mib=0 and max_gpu_utilization_pct=100")
    return context


def receipt_payload(context, runner_sha, preflight_digest, gpu_index, now):
    return {
        "schema_version": SCHEMA_VERSION, "route_id": context["route_id"], "repo_name": context["repo_name"],
        "branch": context["branch"], "route_branch_commit": context["expected_commit"], "rules_commit": context["rules_commit"], "remote_repo": context["remote_repo"],
        "runner_relpath": context["runner_relpath"], "runner_sha256": runner_sha, "mode": context["mode"],
        "session": context["session"], "require_gpu": context["require_gpu"], "stage_state": context["stage_state"],
        "decision": context["decision"], "authorizes": context["authorizes"], "locked_test_policy": context["locked_test_policy"],
        "forbidden_continuations": context["forbidden_continuations"], "output_id": context["output_id"],
        "output_path": context["output_path"], "preflight_digest": preflight_digest, "issued_at": now,
        "expires_at": now + RECEIPT_TTL_SECONDS, "launch_nonce": uuid.uuid4().hex,
        "authorization_relpath": context["authorization_relpath"], "prior_terminal_tuple": context["prior_terminal_tuple"],
        "allowed_terminal_tuples": context["allowed_terminal_tuples"], "closeout_filename": context["closeout_filename"],
        "workspace_id": context["workspace_id"], "workspace_policy": context["workspace_policy"],
        "monitor_profile": context["monitor_profile"], "heartbeat_timeout_seconds": context["heartbeat_timeout_seconds"],
        "min_free_gpu_mib": context["min_free_gpu_mib"], "max_gpu_utilization_pct": context["max_gpu_utilization_pct"],
        "gpu_index": gpu_index,
    }


def receipt_secret():
    RECEIPT_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = RECEIPT_DIR / "receipt-hmac.key"
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        # The concurrent creator may not have flushed the new key yet.
        for _ in range(100):
            try:
                secret = path.read_bytes()
                if len(secret) >= 32:
                    break
            except FileNotFoundError:
                pass
            time.sleep(0.01)
        else:
            raise ToolError("receipt signing key is unavailable")
    else:
        with os.fdopen(fd, "wb") as handle:
            secret = os.urandom(32)
            handle.write(secret)
    if len(secret) < 32:
        raise ToolError("receipt signing key is invalid")
    return secret


def receipt_token(payload):
    body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hmac.new(receipt_secret(), body, hashlib.sha256).hexdigest()


def receipt_path(token):
    if not isinstance(token, str) or not re.fullmatch(r"[0-9a-f]{64}", token):
        raise ToolError("receipt is unknown or malformed")
    return RECEIPT_DIR / f"{token}.json"


def plan_path(token):
    if not isinstance(token, str) or not re.fullmatch(r"[0-9a-f]{64}", token):
        raise ToolError("plan_token is unknown or malformed")
    return RECEIPT_DIR / f"plan-{token}.json"


@contextmanager
def locked_record(path, missing_message):
    if not path.is_file():
        raise ToolError(missing_message)
    with path.open("r+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            record = json.load(handle)
            yield record
            handle.seek(0)
            handle.truncate()
            json.dump(record, handle, sort_keys=True, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def locked_receipt(token):
    with locked_record(receipt_path(token), "receipt is unknown or malformed") as record:
        yield record


def issue_receipt(context, preflight_output):
    match = re.search(r"(?m)^([0-9a-f]{64})\s+.+$", preflight_output)
    if not match:
        raise ToolError("preflight did not report the runner SHA-256")
    runner_sha = match.group(1)
    gpu_match = re.search(r"(?m)^CONVIR_OPS_GPU_OK index=(\d+)", preflight_output)
    gpu_index = int(gpu_match.group(1)) if gpu_match else None
    if context["require_gpu"] and gpu_index is None:
        raise ToolError("preflight did not seal a qualifying GPU")
    payload = receipt_payload(context, runner_sha, canonical_digest(preflight_output), gpu_index, int(time.time()))
    token = receipt_token(payload)
    RECEIPT_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    record = {"payload": payload, "launched": False, "launch_key": None, "launch_result": None}
    path = receipt_path(token)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise ToolError("receipt token collision") from exc
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(record, handle, sort_keys=True, separators=(",", ":"))
    return token, payload


def mark_receipt_launched(token, idempotency_key, launch_result):
    with locked_receipt(token) as record:
        payload = validate_receipt_record(token, record)
        if record.get("launched"):
            if record.get("launch_key") != idempotency_key:
                raise ToolError("receipt has been consumed by a different idempotency key")
            return payload
        record["launched"] = True
        record["launch_key"] = idempotency_key
        record["launch_result"] = launch_result
        return payload


def issue_plan(context, now):
    payload = {
        "schema_version": SCHEMA_VERSION, "context": context, "issued_at": now,
        "expires_at": now + RECEIPT_TTL_SECONDS, "plan_nonce": uuid.uuid4().hex,
    }
    token = receipt_token(payload)
    RECEIPT_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    record = {"payload": payload, "consumed": False, "receipt": None}
    path = plan_path(token)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise ToolError("plan token collision") from exc
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(record, handle, sort_keys=True, separators=(",", ":"))
    return token, payload


@contextmanager
def locked_plan(token):
    with locked_record(plan_path(token), "plan_token is unknown or malformed") as record:
        yield record


def validate_plan_record(token, record):
    payload = record.get("payload")
    if not isinstance(payload, dict) or not hmac.compare_digest(token, receipt_token(payload)):
        raise ToolError("plan token integrity validation failed")
    if time.time() > payload["expires_at"]:
        raise ToolError("plan token has expired")
    return payload


def resolve_receipt(arguments, *, require_unlaunched=False):
    token = arguments.get("receipt")
    return token, locked_receipt(token)


def validate_receipt_record(token, record, require_unlaunched=False):
    payload = record.get("payload")
    if not isinstance(payload, dict) or not hmac.compare_digest(token, receipt_token(payload)):
        raise ToolError("receipt integrity validation failed")
    # Expiry limits the launch authorization window. Once launched, the signed
    # receipt remains the authority for monitoring and closeout of long runs.
    if time.time() > payload["expires_at"] and not record.get("launched"):
        raise ToolError("receipt has expired")
    if require_unlaunched and record.get("launched"):
        raise ToolError("receipt has already been consumed")
    return payload


def q(value):
    return shlex.quote(str(value))


def helper_path():
    path = Path(__file__).with_name("convir_remote_script.sh")
    if not path.is_file():
        raise ToolError(f"tracked transport wrapper is missing: {path}")
    return path


def run_local(args, timeout):
    return _run_local(args, timeout, phase="local_command")


def _run_local(args, timeout, *, phase):
    started = time.monotonic()
    try:
        result = subprocess.run(args, text=True, capture_output=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        elapsed = round(time.monotonic() - started, 3)
        raise ToolError(
            f"phase={phase} command_class=local timeout_seconds={timeout} elapsed_seconds={elapsed}",
            failure_phase=phase, failure_class="command_infra",
        ) from exc
    if result.returncode != 0:
        output = (result.stdout + result.stderr).strip()
        detail = output[:8192] if output else f"local operation failed with rc={result.returncode}"
        raise ToolError(
            f"phase={phase} command_class=local rc={result.returncode}: {detail}",
            failure_phase=phase, failure_class="command_infra",
        )
    return result.stdout.strip()


def inspect_local(args, timeout=30):
    """Run a fixed local inspection command without treating a finding as a tool failure."""
    try:
        result = subprocess.run(args, text=True, capture_output=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return {"ok": False, "returncode": None, "output": f"timed out after {timeout}s"}
    output = (result.stdout + result.stderr).strip()
    return {"ok": result.returncode == 0, "returncode": result.returncode, "output": output[:8192]}


def run_remote_body(body, timeout=120, *, phase="remote_transport"):
    with tempfile.NamedTemporaryFile("w", suffix=".sh", prefix="convir-ops-", delete=False, encoding="utf-8") as handle:
        body_path = Path(handle.name)
        handle.write("#!/usr/bin/env bash\nset -euo pipefail\n")
        handle.write(body)
        handle.write("\n")
    try:
        try:
            return _run_local(["bash", str(helper_path()), str(body_path)], timeout, phase=phase)
        except ToolError as exc:
            message = str(exc)
            if "CONVIR_OPS_RESOURCE_UNAVAILABLE" in message:
                raise ToolError(message, failure_phase="resource_preflight", failure_class="command_infra") from exc
            if "CONVIR_OPS_SESSION_CONFLICT" in message:
                raise ToolError(message, failure_phase="workspace_prepare", failure_class="collision") from exc
            raise
    finally:
        body_path.unlink(missing_ok=True)


def github_ref_sha(ref):
    return github_ref_shas([ref])[ref]


def github_ref_shas(refs):
    """Resolve an exact set of GitHub refs in one network round trip."""
    if not refs or len(refs) != len(set(refs)):
        raise ToolError("GitHub refs must be a non-empty unique list")
    output = _run_local(["git", "ls-remote", GITHUB_URL, *refs], timeout=60, phase="github_ref_fetch")
    observed = {}
    for line in output.splitlines():
        fields = line.split()
        if len(fields) == 2 and SHA40.fullmatch(fields[0]) and fields[1] in refs:
            observed[fields[1]] = fields[0]
    if set(observed) != set(refs):
        missing = sorted(set(refs) - set(observed))
        raise ToolError(f"GitHub refs are missing or malformed: {', '.join(missing)}")
    return observed


def verify_github_context(context):
    branch_ref = f"refs/heads/{context['branch']}"
    refs = github_ref_shas([branch_ref, "refs/heads/main"])
    if refs[branch_ref] != context["expected_commit"]:
        raise ToolError("route branch head does not match expected_commit")
    if refs["refs/heads/main"] != context["rules_commit"]:
        raise ToolError("GitHub main does not match rules_commit")
    with tempfile.TemporaryDirectory(prefix="convir-ops-plan-") as temporary:
        bare_repo = str(Path(temporary) / "repo.git")
        _run_local(["git", "init", "--quiet", "--bare", bare_repo], timeout=30, phase="local_git_prepare")
        _run_local(["git", "-C", bare_repo, "fetch", "--quiet", "--depth=1", GITHUB_URL, context["expected_commit"]], timeout=120, phase="local_git_fetch")
        _run_local(["git", "-C", bare_repo, "cat-file", "-e", f"{context['expected_commit']}:{context['runner_relpath']}"], timeout=30, phase="local_git_verify")
    return {
        "branch": context["branch"], "route_commit": context["expected_commit"],
        "rules_commit": context["rules_commit"], "runner_relpath": context["runner_relpath"],
    }


def load_operation_manifest(arguments):
    if arguments.get("schema_version") != SCHEMA_VERSION:
        raise ToolError("schema_version must be 2")
    request = {
        "branch": require_branch(arguments.get("branch")),
        "route_branch_commit": require_commit(arguments.get("route_branch_commit")),
        "operation_id": require_token(arguments.get("operation_id"), "operation_id"),
    }
    branch_ref = f"refs/heads/{request['branch']}"
    refs = github_ref_shas([branch_ref, "refs/heads/main"])
    if refs[branch_ref] != request["route_branch_commit"]:
        raise ToolError("route branch head does not match route_branch_commit")
    with tempfile.TemporaryDirectory(prefix="convir-ops-plan-") as temporary:
        bare_repo = str(Path(temporary) / "repo.git")
        _run_local(["git", "init", "--quiet", "--bare", bare_repo], timeout=30, phase="local_git_prepare")
        _run_local(["git", "-C", bare_repo, "fetch", "--quiet", "--depth=1", GITHUB_URL, request["route_branch_commit"]], timeout=120, phase="local_git_fetch")
        manifest_raw = _run_local(["git", "-C", bare_repo, "show", f"{request['route_branch_commit']}:{ROUTE_OPERATIONS_RELPATH}"], timeout=30, phase="local_git_manifest")
        if len(manifest_raw.encode("utf-8")) > 65536:
            raise ToolError("route operations manifest exceeds 64 KiB")
        manifest = json.loads(manifest_raw)
        required_top = {"schema_version", "route_id", "repo_name", "workspace_id", "rules_commit", "operations"}
        if not isinstance(manifest, dict) or set(manifest) != required_top or manifest.get("schema_version") != SCHEMA_VERSION:
            raise ToolError("route operations manifest has an invalid top-level contract")
        repo_name = require_token(manifest.get("repo_name"), "route operations manifest repo_name")
        workspace_id = require_token(manifest.get("workspace_id"), "route operations manifest workspace_id")
        operations = manifest.get("operations")
        if not isinstance(operations, dict) or not operations or len(operations) > 32:
            raise ToolError("route operations manifest must contain 1-32 operations")
        if request["operation_id"] not in operations:
            raise ToolError("operation_id is absent from route operations manifest")
        operation = operations[request["operation_id"]]
        required_operation = {
            "runner_relpath", "mode", "require_gpu", "stage_state", "decision", "authorizes",
            "locked_test_policy", "forbidden_continuations", "output_id", "closeout_filename",
            "collision_policy", "authorization_relpath", "prior_terminal_tuple", "allowed_terminal_tuples",
            "workspace_policy", "monitor_profile", "heartbeat_timeout_seconds", "min_free_gpu_mib",
            "max_gpu_utilization_pct",
        }
        if not isinstance(operation, dict) or set(operation) != required_operation:
            raise ToolError("selected route operation has an invalid field contract")
        context = authorization_context({
            "schema_version": SCHEMA_VERSION, "route_id": manifest.get("route_id"),
            "repo_name": repo_name, "workspace_id": workspace_id, "branch": request["branch"],
            "route_branch_commit": request["route_branch_commit"], "rules_commit": manifest.get("rules_commit"),
            **operation,
        })
        _run_local(["git", "-C", bare_repo, "cat-file", "-e", f"{request['route_branch_commit']}:{context['runner_relpath']}"], timeout=30, phase="local_git_verify")
    if refs["refs/heads/main"] != context["rules_commit"]:
        raise ToolError("GitHub main does not match the manifest rules_commit")
    checks = {
        "branch": request["branch"], "route_commit": request["route_branch_commit"],
        "rules_commit": context["rules_commit"], "runner_relpath": context["runner_relpath"],
    }
    return request, manifest, context, checks


def tool_plan_manifest(arguments):
    try:
        request, manifest, context, checks = load_operation_manifest(arguments)
        plan_hash = canonical_digest(context)
        plan_token, plan_payload = issue_plan(context, int(time.time()))
        return typed_result(
            True, "PREPARE_PLANNED",
            observed={
                "manifest_digest": canonical_digest(manifest), "operation_id": request["operation_id"],
                "github_checks_digest": canonical_digest(checks), "remote_repo": context["remote_repo"],
                "session": context["session"], "output_id": context["output_id"],
            },
            expected={"plan_hash": plan_hash}, allowed_next_actions=["route_start_authorized"],
            plan_token=plan_token, plan_expires_at=plan_payload["expires_at"],
        )
    except ToolError as exc:
        return typed_failure("PREPARE_REJECTED", failure_class_for_error(exc), str(exc), **failure_details(exc))
    except (json.JSONDecodeError, TypeError):
        return typed_failure("PREPARE_REJECTED", "authorization", "route operations manifest is not valid compact JSON")
    except Exception as exc:
        return typed_failure("PREPARE_FAILED", "command_infra", type(exc).__name__, failure_phase="github_ref_fetch")


def preflight_body(context, include_gpu, create_clone=False, *, gpu_attempts=1, keep_fresh_trap=False):
    lines = [
        f"REMOTE_REPO={q(context['remote_repo'])}", f"GITHUB_URL={q(GITHUB_URL)}",
        f"RUN_ROOT={q(context['run_root'])}",
        f"BRANCH={q(context['branch'])}",
        f"EXPECTED_COMMIT={q(context['expected_commit'])}",
        f"RULES_COMMIT={q(context.get('rules_commit', 'missing'))}",
        f"RUNNER={q(context['runner_relpath'])}",
        f"SESSION={q(context['session'])}",
        f"PY={q(REMOTE_PYTHON)}",
        f"WORKSPACE_POLICY={q(context.get('workspace_policy', 'fresh_route'))}",
        'test "$(git ls-remote "$GITHUB_URL" "refs/heads/$BRANCH" | awk \'NR==1 {print $1}\')" = "$EXPECTED_COMMIT"' if create_clone else ':',
        'test "$(git ls-remote "$GITHUB_URL" refs/heads/main | awk \'NR==1 {print $1}\')" = "$RULES_COMMIT"' if create_clone else ':',
    ]
    if create_clone and context.get("workspace_policy") == "fresh_route":
        lines.extend([
            'FRESH_CREATED=0',
            'cleanup_fresh() { rc=$?; if [ "$FRESH_CREATED" = 1 ]; then rm -rf -- "$REMOTE_REPO"; echo CONVIR_OPS_FRESH_WORKSPACE_CLEANED; fi; exit "$rc"; }',
            'trap cleanup_fresh ERR',
            'test ! -e "$REMOTE_REPO"',
            'FRESH_CREATED=1',
            'git clone --origin github --no-checkout --single-branch --branch "$BRANCH" "$GITHUB_URL" "$REMOTE_REPO"',
            'git -C "$REMOTE_REPO" fetch --quiet github "+refs/heads/$BRANCH:refs/remotes/github/$BRANCH" "+refs/heads/main:refs/remotes/github/main"',
            'git -C "$REMOTE_REPO" checkout --quiet "$BRANCH"',
        ])
    elif create_clone:
        lines.extend([
            'test -d "$REMOTE_REPO/.git"',
            'test "$(git -C "$REMOTE_REPO" remote get-url github)" = "$GITHUB_URL"',
            'test -z "$(git -C "$REMOTE_REPO" status --porcelain)"',
            'test "$(git -C "$REMOTE_REPO" branch --show-current)" = "$BRANCH"',
            'CURRENT_COMMIT=$(git -C "$REMOTE_REPO" rev-parse HEAD)',
            'git -C "$REMOTE_REPO" fetch --quiet github "+refs/heads/$BRANCH:refs/remotes/github/$BRANCH" "+refs/heads/main:refs/remotes/github/main"',
            'git -C "$REMOTE_REPO" merge-base --is-ancestor "$CURRENT_COMMIT" "$EXPECTED_COMMIT"',
            'git -C "$REMOTE_REPO" merge --quiet --ff-only "$EXPECTED_COMMIT"',
        ])
    else:
        lines.extend([
            'test -d "$REMOTE_REPO/.git"',
            'test "$(git -C "$REMOTE_REPO" remote get-url github)" = "$GITHUB_URL"',
        ])
    lines.extend([
        'test "$(git -C "$REMOTE_REPO" branch --show-current)" = "$BRANCH"',
        'test "$(git -C "$REMOTE_REPO" rev-parse HEAD)" = "$EXPECTED_COMMIT"',
        'test "$(git -C "$REMOTE_REPO" rev-parse github/main)" = "$RULES_COMMIT"',
        'test -z "$(git -C "$REMOTE_REPO" status --porcelain)"',
        'test -x "$PY"',
        'test -f "$REMOTE_REPO/$RUNNER"',
        'RUNNER_SHA=$(sha256sum "$REMOTE_REPO/$RUNNER" | awk \'{print $1}\')',
        f'test "$RUNNER_SHA" = {q(context["runner_sha256"])}' if context.get("runner_sha256") else 'printf "%s  %s/%s\n" "$RUNNER_SHA" "$REMOTE_REPO" "$RUNNER"',
        f'test ! -e {q(context.get("output_path", context["run_root"] + "/unbound"))}',
        f'test ! -e {q(context.get("closeout_path", context["remote_repo"] + "/unbound-closeout"))}',
        'if tmux has-session -t "$SESSION" 2>/dev/null; then echo CONVIR_OPS_SESSION_CONFLICT; exit 1; fi',
        'echo CONVIR_OPS_SESSION_FREE',
    ])
    if "authorization_relpath" in context:
        authorization_check = (
            "import json,sys; p=sys.argv[1]; expected=json.loads(sys.argv[2]); "
            "raw=open(p,'rb').read(65537); assert len(raw)<=65536; value=json.loads(raw); "
            "assert value.get('route_id')==expected['route_id']; "
            "assert {k:value.get(k) for k in ('state','decision','authorizes')}==expected['prior_terminal_tuple']"
        )
        expected = {"route_id": context["route_id"], "prior_terminal_tuple": context["prior_terminal_tuple"]}
        lines.append(f"$PY -c {q(authorization_check)} {q(context['remote_repo'] + '/' + context['authorization_relpath'])} {q(json.dumps(expected, sort_keys=True))}")
    if include_gpu:
        lines.extend([
            f"MIN_FREE_GPU_MIB={int(context['min_free_gpu_mib'])}",
            f"MAX_GPU_UTIL={int(context['max_gpu_utilization_pct'])}",
        ])
        if context.get("gpu_index") is None:
            lines.extend([
                f"GPU_ATTEMPTS={int(gpu_attempts)}",
                'GPU_INDEX=""',
                'for attempt in $(seq 1 "$GPU_ATTEMPTS"); do',
                '  GPU_INDEX=$(nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | awk -F, -v min="$MIN_FREE_GPU_MIB" -v max="$MAX_GPU_UTIL" \'{gsub(/ /,"",$1); gsub(/ /,"",$2); gsub(/ /,"",$3); if ($2 >= min && $3 <= max) {print $1; exit}}\' || true)',
                '  if test -n "$GPU_INDEX"; then',
                '    GPU_OK=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | awk -F, -v min="$MIN_FREE_GPU_MIB" -v max="$MAX_GPU_UTIL" \'{gsub(/ /,"",$1); gsub(/ /,"",$2); if ($1 >= min && $2 <= max) print "yes"}\' || true)',
                '    test "$GPU_OK" = yes && break',
                '  fi',
                '  GPU_INDEX=""',
                '  test "$attempt" = "$GPU_ATTEMPTS" || sleep 2',
                'done',
                'if test -z "$GPU_INDEX"; then echo "CONVIR_OPS_RESOURCE_UNAVAILABLE phase=resource_preflight min_free_mib=$MIN_FREE_GPU_MIB max_util_pct=$MAX_GPU_UTIL attempts=$GPU_ATTEMPTS"; exit 75; fi',
            ])
        else:
            lines.extend([
                f"GPU_INDEX={int(context['gpu_index'])}",
                'GPU_OK=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | awk -F, -v min="$MIN_FREE_GPU_MIB" -v max="$MAX_GPU_UTIL" \'{gsub(/ /,"",$1); gsub(/ /,"",$2); if ($1 >= min && $2 <= max) print "yes"}\' || true)',
                'if test "$GPU_OK" != yes; then echo "CONVIR_OPS_RESOURCE_UNAVAILABLE phase=resource_preflight gpu_index=$GPU_INDEX min_free_mib=$MIN_FREE_GPU_MIB max_util_pct=$MAX_GPU_UTIL"; exit 75; fi',
            ])
        lines.append('echo "CONVIR_OPS_GPU_OK index=$GPU_INDEX min_free_mib=$MIN_FREE_GPU_MIB max_util_pct=$MAX_GPU_UTIL"')
    if create_clone and context.get("workspace_policy") == "fresh_route" and not keep_fresh_trap:
        lines.append('trap - ERR')
    lines.append('echo "CONVIR_OPS_PREFLIGHT_OK route=$RUN_ROOT mode=' + context["mode"] + '"')
    return "\n".join(lines)


def gpu_probe_body(context, attempts=START_RESOURCE_ATTEMPTS):
    """Probe the sealed GPU contract before reserving a fresh workspace."""
    minimum = int(context["min_free_gpu_mib"])
    maximum = int(context["max_gpu_utilization_pct"])
    return "\n".join([
        f"MIN_FREE_GPU_MIB={minimum}", f"MAX_GPU_UTIL={maximum}", f"GPU_ATTEMPTS={int(attempts)}",
        'GPU_INDEX=""',
        'for attempt in $(seq 1 "$GPU_ATTEMPTS"); do',
        '  GPU_INDEX=$(nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | awk -F, -v min="$MIN_FREE_GPU_MIB" -v max="$MAX_GPU_UTIL" \'{gsub(/ /,"",$1); gsub(/ /,"",$2); gsub(/ /,"",$3); if ($2 >= min && $3 <= max) {print $1; exit}}\' || true)',
        '  if test -n "$GPU_INDEX"; then',
        '    GPU_OK=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | awk -F, -v min="$MIN_FREE_GPU_MIB" -v max="$MAX_GPU_UTIL" \'{gsub(/ /,"",$1); gsub(/ /,"",$2); if ($1 >= min && $2 <= max) print "yes"}\' || true)',
        '    test "$GPU_OK" = yes && break',
        '  fi',
        '  GPU_INDEX=""',
        '  test "$attempt" = "$GPU_ATTEMPTS" || sleep 2',
        'done',
        'if test -z "$GPU_INDEX"; then echo "CONVIR_OPS_RESOURCE_UNAVAILABLE phase=resource_preflight min_free_mib=$MIN_FREE_GPU_MIB max_util_pct=$MAX_GPU_UTIL attempts=$GPU_ATTEMPTS"; exit 75; fi',
        'echo "CONVIR_OPS_GPU_OK index=$GPU_INDEX min_free_mib=$MIN_FREE_GPU_MIB max_util_pct=$MAX_GPU_UTIL"',
        'echo CONVIR_OPS_RESOURCE_OK',
    ])


def atomic_start_body(context):
    """Prepare and launch in one remote shell after the dynamic checks."""
    gpu_env = ' GPU="$GPU_INDEX"' if context["require_gpu"] else ""
    launch = (
        f'tmux new-session -d -s {q(context["session"])} env '
        f'EXPECTED_ROUTE_COMMIT={q(context["expected_commit"])} '
        'RUNNER_SHA256="$RUNNER_SHA" '
        f'MODE={q(context["mode"])} REMOTE_REPO={q(context["remote_repo"])} '
        f'RUN_ROOT={q(context["run_root"])} RUN_ID={q(context["output_id"])} '
        f'OUTPUT_ID={q(context["output_id"])}{gpu_env} '
        f'bash {q(context["remote_repo"] + "/" + context["runner_relpath"])}'
    )
    return "\n".join([
        preflight_body(
            context, context["require_gpu"], create_clone=True,
            gpu_attempts=START_RESOURCE_ATTEMPTS, keep_fresh_trap=True,
        ),
        launch,
        'trap - ERR',
        'echo "CONVIR_OPS_LAUNCH_OK session=$SESSION gpu=${GPU_INDEX:-none}"',
    ])


def tool_prepare_authorized(arguments):
    """Plan/apply authorization boundary. Apply is the only receipt producer."""
    try:
        context = authorization_context(arguments)
        phase = require_enum(arguments.get("phase"), "phase", {"plan", "apply"})
        plan = canonical_digest(context)
        if phase == "plan":
            output = verify_github_context(context)
            plan_token, plan_payload = issue_plan(context, int(time.time()))
            return typed_result(
                True, "PREPARE_PLANNED",
                observed={
                    "github_checks": output, "remote_repo": context["remote_repo"],
                    "session": context["session"], "mode": context["mode"], "output_id": context["output_id"],
                    "closeout_filename": context["closeout_filename"],
                },
                expected={"plan_hash": plan}, allowed_next_actions=["route_start_authorized"],
                plan_token=plan_token, plan_expires_at=plan_payload["expires_at"],
            )
        if arguments.get("plan_hash") != plan:
            return typed_failure("PREPARE_REJECTED", "authorization", "plan_hash does not bind this exact authorization tuple", expected={"plan_hash": plan})
        try:
            output = run_remote_body(
                preflight_body(context, context["require_gpu"], create_clone=True),
                timeout=REMOTE_PREFLIGHT_TIMEOUT, phase="workspace_prepare",
            )
        except Exception as exc:
            return typed_failure(
                "PREPARE_RECOVERY_REQUIRED", failure_class_for_error(exc), str(exc) or type(exc).__name__,
                observed={"remote_repo": context["remote_repo"], "fresh_workspace_cleanup": "trap_after_path_reservation"},
                allowed_next_actions=["route_prepare_authorized.apply"], **failure_details(exc),
            )
        receipt, payload = issue_receipt(context, output)
        return typed_result(True, "PREPARED", observed={"preflight": output}, expected={"authorization_tuple": context}, allowed_next_actions=["route_launch"], receipt=receipt, receipt_expires_at=payload["expires_at"], receipt_digest=canonical_digest(payload))
    except ToolError as exc:
        return typed_failure("PREPARE_REJECTED", failure_class_for_error(exc), str(exc), **failure_details(exc))
    except Exception as exc:
        return typed_failure("PREPARE_FAILED", "command_infra", type(exc).__name__, failure_phase="workspace_prepare")


def tool_receipt_launch(arguments):
    try:
        key = require_token(arguments.get("idempotency_key"), "idempotency_key")
        token, lock = resolve_receipt(arguments)
        with lock as record:
            payload = validate_receipt_record(token, record)
            if record["launched"]:
                if key == record["launch_key"]:
                    return typed_result(True, "LAUNCH_IDEMPOTENT", observed=record["launch_result"], expected={"receipt": token}, allowed_next_actions=["route_monitor", "route_closeout_validate"])
                return typed_failure("LAUNCH_REJECTED", "authorization", "receipt has been consumed by a different idempotency key")
            context = {
            "route_id": payload["route_id"], "repo_name": payload["repo_name"], "remote_repo": payload["remote_repo"],
            "run_root": f"{REMOTE_RUNS}/{payload['route_id']}", "branch": payload["branch"], "expected_commit": payload["route_branch_commit"],
            "runner_relpath": payload["runner_relpath"], "mode": payload["mode"], "session": payload["session"], "require_gpu": payload["require_gpu"],
            "output_path": payload["output_path"],
            "closeout_path": f"{payload['remote_repo']}/experience_docx/experiment_logs/{payload['route_id']}/{payload['closeout_filename']}",
            "rules_commit": payload["rules_commit"], "authorization_relpath": payload["authorization_relpath"], "prior_terminal_tuple": payload["prior_terminal_tuple"],
            "runner_sha256": payload["runner_sha256"], "min_free_gpu_mib": payload["min_free_gpu_mib"],
            "max_gpu_utilization_pct": payload["max_gpu_utilization_pct"], "gpu_index": payload["gpu_index"],
            }
            gpu_env = f" GPU={q(payload['gpu_index'])}" if payload["require_gpu"] else ""
            output = run_remote_body("\n".join([
            preflight_body(context, context["require_gpu"]),
            f"tmux new-session -d -s {q(context['session'])} env EXPECTED_ROUTE_COMMIT={q(context['expected_commit'])} RUNNER_SHA256={q(payload['runner_sha256'])} MODE={q(context['mode'])} REMOTE_REPO={q(context['remote_repo'])} RUN_ROOT={q(context['run_root'])} RUN_ID={q(payload['output_id'])} OUTPUT_ID={q(payload['output_id'])}{gpu_env} bash {q(context['remote_repo'] + '/' + context['runner_relpath'])}",
            'echo "CONVIR_OPS_LAUNCH_OK session=$SESSION"',
            ]), timeout=REMOTE_LAUNCH_TIMEOUT, phase="launch_command")
            record["launched"] = True
            record["launch_key"] = key
            record["launch_result"] = {"transport": output, "session": payload["session"], "output_id": payload["output_id"]}
            return typed_result(True, "LAUNCHED", observed=record["launch_result"], expected={"receipt": token}, allowed_next_actions=["route_monitor", "route_closeout_validate"])
    except ToolError as exc:
        return typed_failure("LAUNCH_REJECTED", failure_class_for_error(exc), str(exc), **failure_details(exc))
    except Exception as exc:
        return typed_failure("LAUNCH_FAILED", "command_infra", type(exc).__name__, allowed_next_actions=["route_prepare_authorized.apply"], **failure_details(exc))


def closeout_extractor():
    return "import hashlib,json,sys; raw=open(sys.argv[1],'rb').read(65537); assert len(raw)<=65536; value=json.loads(raw); print('CONVIR_OPS_CLOSEOUT_SHA256='+hashlib.sha256(raw).hexdigest()); print('CONVIR_OPS_CLOSEOUT_JSON_BEGIN'); print(json.dumps(value,sort_keys=True,separators=(',',':'))); print('CONVIR_OPS_CLOSEOUT_JSON_END')"


def monitor_body(payload, max_polls, interval, include_closeout=False):
    remote_run_root = f"{REMOTE_RUNS}/{payload['route_id']}"
    remote_closeout = f"{payload['remote_repo']}/experience_docx/experiment_logs/{payload['route_id']}/{payload['closeout_filename']}"
    lines = [
        f"RUN_ROOT={q(remote_run_root)}", f"SESSION={q(payload['session'])}", f"PY={q(REMOTE_PYTHON)}",
        f"TERMINALS={q(json.dumps(payload['allowed_terminal_tuples'], separators=(',', ':')))}",
        f"MAX_POLLS={max_polls}", f"INTERVAL={interval}", f"HEARTBEAT_TIMEOUT={int(payload['heartbeat_timeout_seconds'])}",
        "previous=''", "stale=false", "terminal=false", "monitor_tmp=$(mktemp -d)",
        "trap 'rm -rf -- \"$monitor_tmp\"' EXIT",
        "printf '%s' \"$TERMINALS\" > \"$monitor_tmp/allowed.json\"",
        "for n in $(seq 1 \"$MAX_POLLS\"); do",
        '  active=false; tmux has-session -t "$SESSION" 2>/dev/null && active=true',
        '  status=$(test -f "$RUN_ROOT/status.txt" && tail -n 16 "$RUN_ROOT/status.txt" || true)',
        '  heartbeat_age=-1',
        '  if test -f "$RUN_ROOT/status.txt"; then heartbeat_age=$(( $(date +%s) - $(stat -c %Y "$RUN_ROOT/status.txt") )); elif test "$active" = true; then session_created=$(tmux display-message -p -t "$SESSION" "#{session_created}" 2>/dev/null || true); test -z "$session_created" || heartbeat_age=$(( $(date +%s) - session_created )); fi',
        '  printf "%s" "$status" > "$monitor_tmp/status.txt"',
        '  if "$PY" - "$monitor_tmp/allowed.json" "$monitor_tmp/status.txt" <<\'PY\'',
        'import json, re, sys',
        'allowed = json.load(open(sys.argv[1], encoding="utf-8"))',
        'raw = open(sys.argv[2], encoding="utf-8", errors="replace").read()',
        'def present(key, value):',
        '    pattern = r"(?m)(?:^|[ \\t])" + re.escape(str(key)) + r"=" + re.escape(str(value)) + r"(?=$|[ \\t])"',
        '    return re.search(pattern, raw) is not None',
        'raise SystemExit(0 if any(all(present(key, value) for key, value in item.items()) for item in allowed) else 1)',
        'PY',
        '  then terminal=true; final_status="$status"; final_active="$active"; final_heartbeat_age="$heartbeat_age"; break; fi',
        '  final_status="$status"; final_active="$active"; final_heartbeat_age="$heartbeat_age"',
        '  test "$active" != true || test "$heartbeat_age" -lt 0 || test "$heartbeat_age" -lt "$HEARTBEAT_TIMEOUT" || { stale=true; break; }',
        '  test "$n" = "$MAX_POLLS" || sleep "$INTERVAL"', 'done',
        'printf "CONVIR_OPS_MONITOR_META polls=%s active=%s terminal=%s stale=%s heartbeat_age=%s\\n" "$n" "$final_active" "${terminal:-false}" "$stale" "$final_heartbeat_age"',
        'echo CONVIR_OPS_MONITOR_STATUS_BEGIN', 'printf "%s\\n" "$final_status"', 'echo CONVIR_OPS_MONITOR_STATUS_END',
    ]
    if include_closeout:
        lines.extend([
            f"CLOSEOUT={q(remote_closeout)}",
            'if { test "${terminal:-false}" = true || test "$final_active" = false; } && test -f "$CLOSEOUT"; then',
            f"  $PY -c {q(closeout_extractor())} \"$CLOSEOUT\"",
            'fi',
        ])
    return "\n".join(lines)


def parse_monitor_output(output):
    meta = re.search(r"(?m)^CONVIR_OPS_MONITOR_META polls=(\d+) active=(true|false) terminal=(true|false) stale=(true|false) heartbeat_age=(-?\d+)$", output)
    begin, end = "CONVIR_OPS_MONITOR_STATUS_BEGIN", "CONVIR_OPS_MONITOR_STATUS_END"
    start, finish = output.find(begin), output.find(end)
    if not meta or start < 0 or finish < 0 or finish <= start:
        raise ToolError("monitor wrapper markers are missing")
    return {
        "poll_count": int(meta.group(1)), "active": meta.group(2) == "true",
        "terminal": meta.group(3) == "true", "stale": meta.group(4) == "true",
        "heartbeat_age_seconds": int(meta.group(5)), "status": output[start + len(begin):finish].strip()[:4096],
    }


def tool_receipt_monitor(arguments):
    try:
        token, lock = resolve_receipt(arguments)
        with lock as record:
            payload = validate_receipt_record(token, record)
        max_polls = require_int(arguments.get("max_polls"), "max_polls", 1, 1, MONITOR_MAX_POLLS)
        interval = require_int(arguments.get("interval_seconds"), "interval_seconds", 0, 0, MONITOR_MAX_INTERVAL_SECONDS)
        max_polls = min(max_polls, max(1, MAX_MONITOR_SECONDS // max(1, interval + 1)))
        timeout = max_polls * max(1, interval) + 20
        observed = parse_monitor_output(
            run_remote_body(monitor_body(payload, max_polls, interval), timeout=timeout, phase="monitor")
        )
        state = "MONITOR_STALE" if observed["stale"] else ("MONITOR_TERMINAL" if observed["terminal"] else ("MONITOR_OBSERVED" if observed["active"] else "MONITOR_INACTIVE_CLOSEOUT_PENDING"))
        actions = ["engineering_review"] if observed["stale"] else (["route_finish"] if observed["terminal"] or not observed["active"] else ["route_finish"])
        return typed_result(True, state, observed=observed, expected={"receipt": canonical_digest(payload)}, allowed_next_actions=actions)
    except ToolError as exc:
        return typed_failure("MONITOR_REJECTED", failure_class_for_error(exc), str(exc), **failure_details(exc))
    except Exception as exc:
        return typed_failure("MONITOR_FAILED", "command_infra", type(exc).__name__, failure_phase="monitor")


def closeout_result_from_output(payload, output):
    begin, end = "CONVIR_OPS_CLOSEOUT_JSON_BEGIN", "CONVIR_OPS_CLOSEOUT_JSON_END"
    start, finish = output.find(begin), output.find(end)
    if start < 0 or finish < 0 or finish <= start:
        raise ToolError("closeout wrapper markers are missing")
    observed = json.loads(output[start + len(begin):finish].strip())
    if not isinstance(observed, dict):
        return typed_failure("CLOSEOUT_INVALID", "evaluation", "runner closeout must be a JSON object")
    expected_identity = {
        "route_id": payload["route_id"], "run_id": payload["output_id"],
        "route_commit": payload["route_branch_commit"], "runner_sha256": payload["runner_sha256"],
    }
    actual_identity = {key: observed.get(key) for key in expected_identity}
    if actual_identity != expected_identity:
        return typed_failure("CLOSEOUT_INVALID", "evidence", "runner closeout provenance mismatch", observed=actual_identity, expected=expected_identity)
    actual = {key: observed.get(key) for key in ("state", "decision", "authorizes")}
    if actual not in payload["allowed_terminal_tuples"]:
        return typed_failure("CLOSEOUT_INVALID", "evidence", "runner closeout tuple is not in the sealed allowed set", observed=actual, expected={"allowed_terminal_tuples": payload["allowed_terminal_tuples"]})
    sha_match = re.search(r"(?m)^CONVIR_OPS_CLOSEOUT_SHA256=([0-9a-f]{64})$", output)
    if not sha_match:
        raise ToolError("closeout raw SHA-256 is missing")
    manifest = {"closeout_filename": payload["closeout_filename"], "closeout_sha256": sha_match.group(1), "receipt_digest": canonical_digest(payload)}
    return typed_result(True, "CLOSEOUT_VALIDATED", observed={"identity": actual_identity, "terminal_tuple": actual}, expected={"identity": expected_identity, "allowed_terminal_tuples": payload["allowed_terminal_tuples"]}, allowed_next_actions=["human_review_archive_candidate"], manifest=manifest, archive_candidate=json.dumps({"route_id": payload["route_id"], "run_id": payload["output_id"], "terminal_tuple": actual, "manifest": manifest}, sort_keys=True))


def tool_closeout_validate(arguments):
    try:
        token, lock = resolve_receipt(arguments)
        with lock as record:
            payload = validate_receipt_record(token, record)
        filename = payload["closeout_filename"]
        remote_path = f"{payload['remote_repo']}/experience_docx/experiment_logs/{payload['route_id']}/{filename}"
        output = run_remote_body(
            f"test -f {q(remote_path)}\n{q(REMOTE_PYTHON)} -c {q(closeout_extractor())} {q(remote_path)}",
            timeout=REMOTE_CLOSEOUT_TIMEOUT, phase="closeout",
        )
        return closeout_result_from_output(payload, output)
    except ToolError as exc:
        return typed_failure("CLOSEOUT_REJECTED", failure_class_for_error(exc), str(exc), **failure_details(exc))
    except (json.JSONDecodeError, TypeError):
        return typed_failure("CLOSEOUT_INVALID", "evaluation", "runner closeout is not valid compact JSON")
    except Exception as exc:
        return typed_failure("CLOSEOUT_FAILED", "command_infra", type(exc).__name__, failure_phase="closeout")


def _tool_start_authorized(arguments):
    """Prepare and launch one exact route in a bounded, receipt-safe envelope."""
    token = arguments.get("plan_token")
    idempotency_key = token[:32] if isinstance(token, str) else token
    with locked_plan(token) as plan_record:
        plan_payload = validate_plan_record(token, plan_record)
        context = plan_payload["context"]
        plan_hash = canonical_digest(context)
        if plan_record.get("consumed"):
            receipt = plan_record.get("receipt")
            if not receipt:
                return typed_failure("START_REJECTED", "authorization", "consumed plan record is missing its receipt")
            return tool_receipt_launch({"receipt": receipt, "idempotency_key": idempotency_key})
        if context["require_gpu"]:
            try:
                run_remote_body(
                    gpu_probe_body(context),
                    timeout=30,
                    phase="resource_preflight",
                )
            except ToolError as exc:
                if failure_phase_for_error(exc) == "resource_preflight":
                    return typed_failure(
                        "RESOURCE_WAIT_REQUIRED", "command_infra", str(exc) or type(exc).__name__,
                        observed={
                            "remote_repo": context["remote_repo"],
                            "min_free_gpu_mib": context["min_free_gpu_mib"],
                            "max_gpu_utilization_pct": context["max_gpu_utilization_pct"],
                            "attempts": START_RESOURCE_ATTEMPTS,
                        }, expected={"runner_started": False},
                        allowed_next_actions=["route_start_authorized"],
                        retry_after_seconds=15, **failure_details(exc),
                    )
                return typed_failure(
                    "START_RECOVERY_REQUIRED", failure_class_for_error(exc), str(exc) or type(exc).__name__,
                    observed={"remote_repo": context["remote_repo"], "runner_started": False},
                    allowed_next_actions=["route_start_authorized"], **failure_details(exc),
                )
        try:
            output = run_remote_body(
                atomic_start_body(context),
                timeout=REMOTE_LAUNCH_TIMEOUT,
                phase="launch_command",
            )
        except ToolError as exc:
            if failure_phase_for_error(exc) == "resource_preflight":
                return typed_failure(
                    "RESOURCE_WAIT_REQUIRED", "command_infra", str(exc) or type(exc).__name__,
                    observed={"remote_repo": context["remote_repo"], "runner_started": False, "workspace_cleanup": "fresh_workspace_trap"},
                    expected={"runner_started": False}, allowed_next_actions=["route_start_authorized"],
                    retry_after_seconds=15, **failure_details(exc),
                )
            return typed_failure(
                "START_STATE_UNKNOWN" if failure_phase_for_error(exc) == "launch_command" else "START_RECOVERY_REQUIRED",
                failure_class_for_error(exc), str(exc) or type(exc).__name__,
                observed={"remote_repo": context["remote_repo"], "runner_started": "unknown"},
                expected={"runner_started": False},
                allowed_next_actions=["engineering_review"], **failure_details(exc),
            )
        receipt, receipt_payload_value = issue_receipt(context, output)
        launch_result = {
            "transport": output,
            "session": context["session"],
            "output_id": context["output_id"],
        }
        mark_receipt_launched(receipt, idempotency_key, launch_result)
        plan_record["consumed"] = True
        plan_record["receipt"] = receipt
        return typed_result(
            True, "LAUNCHED", observed={
                "session": context["session"], "output_id": context["output_id"],
                "remote_repo": context["remote_repo"], "launch_state": "LAUNCHED",
            }, expected={"receipt_digest": canonical_digest(receipt_payload_value), "plan_hash": plan_hash},
            allowed_next_actions=["route_finish"], receipt=receipt,
            receipt_expires_at=receipt_payload_value["expires_at"],
        )


def tool_start_authorized(arguments):
    try:
        return _tool_start_authorized(arguments)
    except ToolError as exc:
        return typed_failure("START_REJECTED", failure_class_for_error(exc), str(exc), **failure_details(exc))
    except Exception as exc:
        return typed_failure("START_FAILED", "command_infra", type(exc).__name__, failure_phase="launch_command")


def tool_finish(arguments):
    """Observe one sealed profile window and validate closeout in the same SSH call."""
    try:
        token, lock = resolve_receipt(arguments)
        with lock as record:
            payload = validate_receipt_record(token, record)
        profile = MONITOR_PROFILES[payload["monitor_profile"]]
        max_polls = min(profile["max_polls"], max(1, MAX_MONITOR_SECONDS // (profile["interval_seconds"] + 1)))
        timeout = max_polls * profile["interval_seconds"] + 20
        output = run_remote_body(
            monitor_body(payload, max_polls, profile["interval_seconds"], include_closeout=True),
            timeout=timeout,
            phase="monitor",
        )
        monitor = parse_monitor_output(output)
        if monitor["stale"]:
            return typed_failure(
                "MONITOR_STALE", "command_infra", "runner heartbeat exceeded the sealed timeout",
                observed=monitor, expected={"heartbeat_timeout_seconds": payload["heartbeat_timeout_seconds"]},
                allowed_next_actions=["engineering_review"],
            )
        if "CONVIR_OPS_CLOSEOUT_JSON_BEGIN" not in output:
            if not monitor["active"]:
                return typed_failure(
                    "CLOSEOUT_MISSING", "evaluation", "runner session ended without the sealed closeout",
                    observed=monitor,
                    expected={"closeout_filename": payload["closeout_filename"]},
                    allowed_next_actions=["engineering_review"],
                )
            return typed_result(
                True, "MONITOR_OBSERVED", observed=monitor,
                expected={"monitor_profile": payload["monitor_profile"], "receipt": canonical_digest(payload)},
                allowed_next_actions=["route_finish"],
            )
        closeout = structured_payload(closeout_result_from_output(payload, output))
        return typed_result(
            closeout["ok"], closeout["operation_state"], closeout["failure_class"],
            observed={"monitor": monitor, "closeout": closeout["observed"]},
            expected=closeout["expected"], mismatches=closeout["mismatches"],
            allowed_next_actions=closeout["allowed_next_actions"],
            **{key: closeout[key] for key in ("manifest", "archive_candidate") if key in closeout},
        )
    except ToolError as exc:
        return typed_failure("FINISH_REJECTED", failure_class_for_error(exc), str(exc), **failure_details(exc))
    except (json.JSONDecodeError, TypeError):
        return typed_failure("FINISH_INVALID", "evaluation", "runner closeout is not valid compact JSON")
    except Exception as exc:
        return typed_failure("FINISH_FAILED", "command_infra", type(exc).__name__, failure_phase="closeout")


def validate_evidence_file(name):
    require_token(name.rsplit(".", 1)[0] if "." in name else "", "evidence filename stem")
    suffix = Path(name).suffix.lower()
    if suffix not in ALLOWED_EVIDENCE_SUFFIXES or "cloud_only" in name.lower():
        raise ToolError("evidence files must be compact .json/.csv/.md/.txt and must not be cloud_only")
    return name


def manifest_body(context, files=None):
    lines = [
        "export LC_ALL=C",
        f"EVIDENCE_DIR={q(context['evidence_dir'])}",
        'if test ! -d "$EVIDENCE_DIR"; then echo CONVIR_OPS_EVIDENCE_UNAVAILABLE >&2; exit 66; fi',
    ]
    if files is None:
        lines.extend([
            'shopt -s nullglob',
            'for path in "$EVIDENCE_DIR"/*; do',
            '  name=$(basename "$path")',
            '  test -f "$path" || continue',
            '  [[ "$name" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\.(json|csv|md|txt)$ ]] || continue',
            '  case "${name,,}" in *cloud_only*) continue ;; esac',
            '  size=$(wc -c < "$path")',
            f'  if [ "$size" -le {MAX_EVIDENCE_BYTES} ]; then',
            '    read -r digest _ < <(sha256sum "$path")',
            '    test "${#digest}" -eq 64 || { echo CONVIR_OPS_EVIDENCE_INVALID_HASH >&2; exit 67; }',
            '    printf "%s\\t%s\\t%s\\n" "$name" "$size" "$digest"',
            '  fi',
            'done',
        ])
    else:
        for name in files:
            lines.extend([
                f"path=\"$EVIDENCE_DIR/{name}\"",
                'test -f "$path"',
                'size=$(wc -c < "$path")',
                f'test "$size" -le {MAX_EVIDENCE_BYTES}',
                'read -r digest _ < <(sha256sum "$path")',
                'test "${#digest}" -eq 64',
                f'printf "%s\\t%s\\t%s\\n" {q(name)} "$size" "$digest"',
            ])
    lines.append("echo CONVIR_OPS_EVIDENCE_MANIFEST_OK")
    return "\n".join(lines)


def parse_manifest(output):
    records = {}
    marker_count = sum(1 for line in output.splitlines() if line == "CONVIR_OPS_EVIDENCE_MANIFEST_OK")
    if marker_count != 1:
        raise ToolError(
            "phase=evidence_manifest evidence manifest marker is missing or duplicated",
            failure_phase="evidence_manifest", failure_class="command_infra",
        )
    for line in output.splitlines():
        fields = line.split("\t")
        if len(fields) == 3 and fields[1].isdigit() and re.fullmatch(r"[0-9a-f]{64}", fields[2]):
            if fields[0] in records:
                raise ToolError(
                    f"phase=evidence_manifest duplicate evidence record: {fields[0]}",
                    failure_phase="evidence_manifest", failure_class="command_infra",
                )
            records[fields[0]] = {"bytes": int(fields[1]), "sha256": fields[2]}
        elif line not in {"CONVIR_OPS_EVIDENCE_MANIFEST_OK", "CONVIR_REMOTE_SCRIPT_OK", ""}:
            raise ToolError(
                "phase=evidence_manifest malformed evidence record",
                failure_phase="evidence_manifest", failure_class="command_infra",
            )
    return records


def tool_evidence_manifest(arguments):
    try:
        context = route_context(arguments)
        output = run_remote_body(manifest_body(context), timeout=60, phase="evidence_manifest")
        records = parse_manifest(output)
        result = {
            "schema_version": SCHEMA_VERSION,
            "ok": True,
            "operation_state": "EVIDENCE_MANIFEST_READY",
            "failure_class": "none",
            "failure_phase": "evidence_manifest",
            "observed": {"files": records, "count": len(records)},
            "allowed_next_actions": ["evidence_review"],
        }
        result["audit_digest"] = canonical_digest(result)
        return text_result(json.dumps(result, sort_keys=True), structured_content=result)
    except ToolError as exc:
        return typed_failure("EVIDENCE_MANIFEST_FAILED", failure_class_for_error(exc), str(exc), **failure_details(exc))
    except Exception as exc:
        return typed_failure("EVIDENCE_MANIFEST_FAILED", "command_infra", type(exc).__name__, failure_phase="evidence_manifest")


def validate_local_repo(value):
    if not isinstance(value, str) or not value.startswith("/"):
        raise ToolError("local_repo must be an absolute WSL path")
    local_repo = Path(value).resolve()
    try:
        local_repo.relative_to(LOCAL_WORKSPACE_ROOT)
    except ValueError as exc:
        raise ToolError(f"local_repo must stay under {LOCAL_WORKSPACE_ROOT}") from exc
    if not (local_repo / ".git").exists():
        raise ToolError("local_repo must be an existing Git worktree")
    return local_repo


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tool_evidence_fetch(arguments):
    try:
        context = route_context(arguments)
        local_repo = validate_local_repo(arguments.get("local_repo"))
        files = arguments.get("files")
        if not isinstance(files, list) or not files or len(files) > 32:
            raise ToolError("files must be a non-empty list of at most 32 compact evidence filenames")
        names = [validate_evidence_file(item) for item in files]
        if len(set(names)) != len(names):
            raise ToolError("files must not contain duplicates")
        records = parse_manifest(run_remote_body(manifest_body(context, names), timeout=60, phase="evidence_manifest"))
        if set(records) != set(names):
            raise ToolError(
                "remote evidence manifest did not return the requested allowlist exactly",
                failure_phase="evidence_manifest", failure_class="command_infra",
            )
        destination_dir = local_repo / "experience_docx" / "experiment_logs" / context["route_id"]
        destination_dir.mkdir(parents=True, exist_ok=True)
        fetched = []
        skipped = []
        pending = []
        for name in names:
            destination = (destination_dir / name).resolve()
            try:
                destination.relative_to(local_repo)
            except ValueError as exc:
                raise ToolError("resolved evidence destination escaped local_repo") from exc
            expected = records[name]["sha256"]
            if destination.exists():
                if sha256_file(destination) != expected:
                    raise ToolError(f"refusing to overwrite hash-mismatched local evidence: {destination}")
                skipped.append(name)
                continue
            pending.append((name, destination, expected))
        if pending:
            transfer_timeout = min(
                300,
                max(90, 45 + sum(records[name]["bytes"] for name, _, _ in pending) // (256 * 1024)),
            )
            with tempfile.TemporaryDirectory(prefix=".convir-evidence-", dir=destination_dir) as staging:
                staging_dir = Path(staging)
                sources = [f"{REMOTE_HOST}:{context['evidence_dir']}/{name}" for name, _, _ in pending]
                _run_local(["scp", *sources, str(staging_dir)], transfer_timeout, phase="evidence_transfer")
                for name, _, expected in pending:
                    staged = staging_dir / name
                    if not staged.is_file() or sha256_file(staged) != expected:
                        raise ToolError(
                            f"downloaded evidence hash mismatch: {name}",
                            failure_phase="evidence_transfer", failure_class="command_infra",
                        )
                for name, destination, expected in pending:
                    try:
                        os.link(staging_dir / name, destination)
                    except FileExistsError as exc:
                        raise ToolError(f"refusing to overwrite newly created local evidence: {destination}") from exc
                    if sha256_file(destination) != expected:
                        destination.unlink(missing_ok=True)
                        raise ToolError(f"local evidence hash mismatch after transfer: {name}")
                    fetched.append(name)
        result = {"fetched": fetched, "already_verified": skipped, "destination": str(destination_dir), "transfer": "single_scp"}
        return text_result(json.dumps(result, indent=2), structured_content=result)
    except ToolError as exc:
        return typed_failure("EVIDENCE_FETCH_FAILED", failure_class_for_error(exc), str(exc), **failure_details(exc))
    except Exception as exc:
        return typed_failure("EVIDENCE_FETCH_FAILED", "command_infra", type(exc).__name__, failure_phase="evidence_transfer")


def tool_git_evidence_status(arguments):
    try:
        route_id = require_token(arguments.get("route_id"), "route_id")
        local_repo = validate_local_repo(arguments.get("local_repo"))
        prefix = ["git", "-C", str(local_repo)]
        branch = _run_local([*prefix, "branch", "--show-current"], timeout=30, phase="git_status")
        head = _run_local([*prefix, "rev-parse", "HEAD"], timeout=30, phase="git_status")
        local_main = _run_local([*prefix, "rev-parse", "--verify", "github/main"], timeout=30, phase="git_status")
        remote_main_raw = _run_local([*prefix, "ls-remote", "github", "refs/heads/main"], timeout=60, phase="github_ref_fetch")
        remote_main_fields = remote_main_raw.split()
        if len(remote_main_fields) != 2 or not SHA40.fullmatch(remote_main_fields[0]):
            raise ToolError("github main could not be resolved through git ls-remote", failure_phase="github_ref_fetch", failure_class="command_infra")
        remote_main = remote_main_fields[0]
        status = _run_local([*prefix, "status", "--short"], timeout=30, phase="git_status")
        diff_check = inspect_local([*prefix, "diff", "--check"], timeout=30)
        cached_diff_check = inspect_local([*prefix, "diff", "--cached", "--check"], timeout=30)
        ahead = behind = None
        if local_main == remote_main:
            counts = _run_local([*prefix, "rev-list", "--left-right", "--count", "HEAD...github/main"], timeout=30, phase="git_status").split()
            if len(counts) == 2 and all(item.isdigit() for item in counts):
                ahead, behind = map(int, counts)
        route_evidence_prefix = f"experience_docx/experiment_logs/{route_id}/"
        changed_paths = status.splitlines()[:100] if status else []
        route_evidence_changes = [line for line in changed_paths if route_evidence_prefix in line]
        return text_result(json.dumps({
            "local_repo": str(local_repo),
            "branch": branch,
            "head": head,
            "github_main_local": local_main,
            "github_main_remote": remote_main,
            "github_main_ref_fresh": local_main == remote_main,
            "ahead_of_github_main": ahead,
            "behind_github_main": behind,
            "worktree_clean": not status,
            "changed_paths": changed_paths,
            "route_evidence_changes": route_evidence_changes,
            "diff_check": diff_check,
            "cached_diff_check": cached_diff_check,
            "git_mutations_performed": False,
        }, indent=2))
    except ToolError as exc:
        return typed_failure("GIT_STATUS_FAILED", failure_class_for_error(exc), str(exc), **failure_details(exc))
    except Exception as exc:
        return typed_failure("GIT_STATUS_FAILED", "command_infra", type(exc).__name__, failure_phase="git_status")


TOOLS = {
    "convir_route_plan_manifest": {
        "description": "Read and seal one schema-v2 operation from the exact GitHub route commit without contacting the cloud.",
        "inputSchema": {
            "type": "object",
            "required": ["schema_version", "branch", "route_branch_commit", "operation_id"],
            "properties": {
                "schema_version": {"const": 2},
                "branch": {"type": "string"},
                "route_branch_commit": {"type": "string"},
                "operation_id": {"type": "string"},
            },
            "additionalProperties": False,
        },
        "handler": tool_plan_manifest,
    },
    "convir_route_start_authorized": {
        "description": "Apply a reviewed signed plan and idempotently start its exact receipt-bound runner.",
        "inputSchema": {
            "type": "object",
            "required": ["plan_token"],
            "properties": {"plan_token": {"type": "string"}},
            "additionalProperties": False,
        },
        "handler": tool_start_authorized,
    },
    "convir_route_finish": {
        "description": "Observe one sealed server-side window and validate terminal closeout provenance in the same remote call.",
        "inputSchema": {
            "type": "object",
            "required": ["receipt"],
            "properties": {"receipt": {"type": "string"}},
            "additionalProperties": False,
        },
        "handler": tool_finish,
    },
    "convir_evidence_manifest": {
        "description": "Read-only inventory of top-level compact evidence eligible for review. It excludes cloud_only files and files larger than 1 MiB.",
        "inputSchema": {
            "type": "object",
            "required": ["route_id", "repo_name"],
            "properties": {"route_id": {"type": "string"}, "repo_name": {"type": "string"}},
            "additionalProperties": False,
        },
        "handler": tool_evidence_manifest,
    },
    "convir_evidence_fetch": {
        "description": "Fetch an explicit allowlist of compact evidence into a local Git worktree after remote and local SHA-256 verification. It never stages, commits, or pushes Git.",
        "inputSchema": {
            "type": "object",
            "required": ["route_id", "repo_name", "local_repo", "files"],
            "properties": {"route_id": {"type": "string"}, "repo_name": {"type": "string"}, "local_repo": {"type": "string"}, "files": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 32}},
            "additionalProperties": False,
        },
        "handler": tool_evidence_fetch,
    },
    "convir_git_evidence_status": {
        "description": "Read-only local/GitHub audit for a named route evidence worktree. It compares the local github/main ref with git ls-remote, reports clean state and diff checks, and never fetches, stages, commits, or pushes.",
        "inputSchema": {
            "type": "object",
            "required": ["route_id", "local_repo"],
            "properties": {"route_id": {"type": "string"}, "local_repo": {"type": "string"}},
            "additionalProperties": False,
        },
        "handler": tool_git_evidence_status,
    },
}


def handle(request):
    request_id = request.get("id")
    method = request.get("method")
    params = request.get("params") or {}
    if method == "initialize":
        version = params.get("protocolVersion", "2024-11-05")
        return {"protocolVersion": version, "capabilities": {"tools": {}}, "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION}}
    if method == "tools/list":
        return {"tools": [{key: value[key] for key in ("description", "inputSchema")} | {"name": name} for name, value in TOOLS.items()]}
    if method == "tools/call":
        name = params.get("name")
        if name not in TOOLS:
            raise ToolError(f"unknown tool: {name}")
        arguments = params.get("arguments") or {}
        if not isinstance(arguments, dict):
            raise ToolError("tool arguments must be an object")
        schema = TOOLS[name]["inputSchema"]
        unknown = set(arguments) - set(schema["properties"])
        missing = set(schema.get("required", [])) - set(arguments)
        if unknown:
            raise ToolError("tool arguments contain forbidden fields: " + ", ".join(sorted(unknown)))
        if missing:
            raise ToolError("tool arguments are missing required fields: " + ", ".join(sorted(missing)))
        return TOOLS[name]["handler"](arguments)
    if method in {"notifications/initialized", "notifications/cancelled"}:
        return None
    if method == "ping":
        return {}
    raise ToolError(f"unsupported MCP method: {method}")


def main():
    for line in sys.stdin:
        request = None
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ToolError("MCP request must be an object")
            request_id = request.get("id")
            result = handle(request)
            if request_id is not None and result is not None:
                mcp_result(request_id, result)
        except ToolError as exc:
            if request is not None and request.get("id") is not None:
                mcp_result(request["id"], text_result(str(exc), is_error=True))
            else:
                print(f"{SERVER_NAME}: {exc}", file=sys.stderr)
        except Exception as exc:  # Never expose an arbitrary shell traceback to MCP clients.
            if request is not None and request.get("id") is not None:
                mcp_error(request["id"], -32603, f"{SERVER_NAME} internal error: {type(exc).__name__}")
            else:
                print(f"{SERVER_NAME}: internal error: {type(exc).__name__}", file=sys.stderr)


if __name__ == "__main__":
    main()
