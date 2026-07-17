#!/usr/bin/env python3
"""Restricted six-tool MCP bridge for ConvIR-B cloud route operations."""

import fcntl
import hashlib
import hmac
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path


SERVER_NAME = "convir-ops"
SERVER_VERSION = "4.2.0"
SERVER_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
SCHEMA_VERSION = 4
REMOTE_HOST = "convir-4090"
REMOTE_BASE = "/sda/home/wangyuxin/ConvIR-B"
REMOTE_REPOS = f"{REMOTE_BASE}/repos"
REMOTE_RUNS = f"{REMOTE_BASE}/runs"
REMOTE_PYTHON = f"{REMOTE_BASE}/envs/convir-cu121/bin/python"
CLOUD_GIT_SEED = f"{REMOTE_REPOS}/ConvIR-B-official-arch-anchor"
GITHUB_URL = "git@github.com:onenoober/ConvIR-B.git"
SSH = "/usr/bin/ssh"
REMOTE_BASH = "/bin/bash"
MAX_REMOTE_SCRIPT_BYTES = 256 * 1024
MAX_REMOTE_CAPTURE_BYTES = 64 * 1024
ROUTE_OPERATIONS_RELPATH = "experience_docx/route_operations.json"
RULE_BUNDLE_RELPATHS = (
    "AGENTS.md",
    "experience_docx/EXPERIMENT_GOVERNANCE_PROTOCOL.md",
    "experience_docx/MODEL_RUN_OPERATIONS_PROTOCOL.md",
    "experience_docx/CONVIR_OPS_MCP.md",
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
MAX_EVIDENCE_BYTES = 1024 * 1024
MAX_MANIFEST_BYTES = 16 * 1024
MAX_CLOSEOUT_BYTES = 64 * 1024
SAFE_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
SAFE_BRANCH = re.compile(r"^codex/[A-Za-z0-9][A-Za-z0-9_.\-/]{0,191}$")
SHA40 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
ALLOWED_EVIDENCE_SUFFIXES = {".json", ".csv", ".md", ".txt"}
MONITOR_PROFILES = {
    "short": {"max_polls": 3, "interval_seconds": 10},
    "standard": {"max_polls": 4, "interval_seconds": 15},
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
        "mismatches": mismatches or [],
        "allowed_next_actions": next_actions or [],
    }
    value.update(extra)
    value["audit_digest"] = canonical_digest(value)
    return text_result(json.dumps(value, sort_keys=True), is_error=not ok, structured=value)


def typed_failure(state, failure_class, message, **kwargs):
    return typed_result(False, state, failure_class, mismatches=[message], **kwargs)


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


def rule_bundle_digest(repo, commit):
    digest = hashlib.sha256()
    for path in RULE_BUNDLE_RELPATHS:
        raw = git_show_bytes(repo, commit, path)
        digest.update(path.encode() + b"\0" + raw + b"\0")
    return digest.hexdigest()


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
    expected_top = {
        "schema_version", "route_id", "rules_commit",
        "route_card_relpath", "operations",
    }
    if not isinstance(value, dict) or set(value) != expected_top:
        raise ToolError("route operations manifest has an invalid top-level contract")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ToolError(f"route operations manifest must use schema {SCHEMA_VERSION}")
    route_id = require_token(value["route_id"], "route_id")
    rules_commit = require_sha(value["rules_commit"], "rules_commit", SHA40)
    route_card = require_relpath(
        value["route_card_relpath"], "route_card_relpath", ".md",
        prefix="experience_docx/experiment_cards/",
    )
    route_card_blob = blob_sha(bare_repo, route_commit, route_card)
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
    elif first_operation_from_card(git_show(bare_repo, route_commit, route_card)) != operation_id:
        raise ToolError("selected operation is not the frozen first operation")
    recorded_rules = rule_bundle_digest(bare_repo, rules_commit)
    current_rules = rule_bundle_digest(bare_repo, current_main)
    if recorded_rules != current_rules:
        raise ToolError(
            "canonical rule bundle changed; one compatibility review is required"
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
        "branch": branch,
        "route_branch_commit": route_commit,
        "current_rules_commit": current_main,
        "route_id": route_id,
        "remote_repo": derive_remote_repo(route_id, output_id),
        "run_root": f"{REMOTE_RUNS}/{route_id}",
        "route_card_relpath": route_card,
        "route_card_blob": route_card_blob,
        "rules_commit": rules_commit,
        "rules_bundle_digest": recorded_rules,
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
        rules_commit = manifest.get("rules_commit") if isinstance(manifest, dict) else None
        if isinstance(rules_commit, str) and SHA40.fullmatch(rules_commit):
            ensure_commit(bare_repo, rules_commit)
        context = parse_manifest(
            manifest, branch, route_commit, refs["refs/heads/main"], bare_repo, operation_id
        )
    return manifest, operation_id, context


def state_secret():
    STATE_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = STATE_DIR / "hmac.key"
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        value = path.read_bytes()
    else:
        value = os.urandom(32)
        with os.fdopen(fd, "wb") as handle:
            handle.write(value)
    if len(value) != 32:
        raise ToolError("MCP signing key is invalid", failure_class="command_infra")
    return value


def sign(payload):
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hmac.new(state_secret(), raw, hashlib.sha256).hexdigest()


def record_path(kind, token):
    require_sha(token, kind, SHA256)
    return STATE_DIR / f"{kind}-{token}.json"


def write_new_record(kind, payload, extra):
    token = sign(payload)
    path = record_path(kind, token)
    STATE_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise ToolError(f"{kind} token collision", failure_class="command_infra") from exc
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump({"payload": payload, **extra}, handle, sort_keys=True, separators=(",", ":"))
    return token


@contextmanager
def locked_record(kind, token):
    path = record_path(kind, token)
    if not path.is_file():
        raise ToolError(f"{kind} is unknown or expired")
    with path.open("r+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            value = json.load(handle)
            if not hmac.compare_digest(token, sign(value.get("payload"))):
                raise ToolError(f"{kind} integrity check failed")
            try:
                yield value
            finally:
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
        if rule_bundle_digest(repo, current) != context["rules_bundle_digest"]:
            raise ToolError("canonical rules changed after planning; create one fresh plan")


def gpu_probe_body(context):
    return "\n".join([
        f"MIN_FREE={int(context['min_free_gpu_mib'])}",
        f"MAX_UTIL={int(context['max_gpu_utilization_pct'])}",
        'GPU_INDEX=""',
        'for attempt in 1 2; do',
        '  GPU_INDEX=$(nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | awk -F, -v min="$MIN_FREE" -v max="$MAX_UTIL" \'{gsub(/ /,"",$1); gsub(/ /,"",$2); gsub(/ /,"",$3); if ($2 >= min && $3 <= max) {print $1; exit}}\' || true)',
        '  test -n "$GPU_INDEX" && break',
        '  test "$attempt" = 2 || sleep 2',
        'done',
        'test -n "$GPU_INDEX" || { echo CONVIR_OPS_RESOURCE_WAIT_REQUIRED; exit 75; }',
        'echo "CONVIR_OPS_GPU_OK index=$GPU_INDEX"',
    ])


def parse_gpu(output):
    match = re.search(r"(?m)^CONVIR_OPS_GPU_OK index=(\d+)$", output)
    if not match:
        raise ToolError(
            "GPU probe returned no eligible device",
            failure_phase="resource_preflight",
            failure_class="command_infra",
        )
    return int(match.group(1))


def atomic_start_body(context, gpu_index):
    lines = [
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
        'tmux has-session -t "$SESSION" 2>/dev/null && { echo CONVIR_OPS_SESSION_CONFLICT; exit 73; } || true',
        'if test "$OUTPUT_POLICY" = new; then',
        '  test ! -e "$OUTPUT_PATH"',
        '  test ! -e "$CLOSEOUT"',
        'else',
        '  test -d "$OUTPUT_PATH"',
        '  test ! -e "$CLOSEOUT"',
        'fi',
    ]
    if gpu_index is not None:
        lines.extend([
            f"MIN_FREE={int(context['min_free_gpu_mib'])}",
            f"MAX_UTIL={int(context['max_gpu_utilization_pct'])}",
            'GPU_OK=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | awk -F, -v min="$MIN_FREE" -v max="$MAX_UTIL" \'{gsub(/ /,"",$1); gsub(/ /,"",$2); if ($1 >= min && $2 <= max) print "yes"}\')',
            'test "$GPU_OK" = yes || { echo CONVIR_OPS_RESOURCE_WAIT_REQUIRED; exit 75; }',
        ])
    lines.extend([
        f'tmux new-session -d -s "$SESSION" env EXPECTED_ROUTE_COMMIT="$EXPECTED_COMMIT" RUNNER_SHA256="$EXPECTED_RUNNER_SHA" MODE={q(context["mode"])} REMOTE_REPO="$REMOTE_REPO" RUN_ROOT="$RUN_ROOT" OUTPUT_PATH="$OUTPUT_PATH" RUN_ID={q(context["output_id"])} OUTPUT_ID={q(context["output_id"])} GPU="$GPU_INDEX" bash "$REMOTE_REPO/$RUNNER"',
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
        'active=false; tmux has-session -t "$SESSION" 2>/dev/null && active=true || true',
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
        f"REPO_ROOT={q(REMOTE_REPOS)}",
        'case "$REMOTE_REPO" in "$REPO_ROOT"/*) ;; *) exit 91 ;; esac',
        'test -d "$REMOTE_REPO/.git"',
        'test "$(git -C "$REMOTE_REPO" rev-parse HEAD)" = "$EXPECTED_COMMIT"',
        'test "$(git -C "$REMOTE_REPO" branch --show-current)" = "$EXPECTED_BRANCH"',
        'test -z "$(git -C "$REMOTE_REPO" status --porcelain)"',
        'test "$(sha256sum "$REMOTE_REPO/$RUNNER" | awk \'{print $1}\')" = "$EXPECTED_RUNNER_SHA"',
        'tmux has-session -t "$SESSION" 2>/dev/null && exit 92 || true',
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
            "monitor_stale_count": 0,
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
        return typed_result(
            True, "LAUNCH_RECOVERED",
            observed={**observed, "route_id": context["route_id"],
                      "session": context["session"], "output_path": context["output_path"]},
            expected={"runner_sha256": context["runner_sha256"]},
            next_actions=["convir_route_finish"], receipt=receipt,
        )
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
                    gpu_index = parse_gpu(
                        run_remote(gpu_probe_body(context), timeout=30, phase="resource_preflight")
                    )
                except ToolError as exc:
                    return typed_failure(
                        "RESOURCE_WAIT_REQUIRED", "command_infra", str(exc),
                        expected={"runner_started": False},
                        next_actions=["convir_route_start"],
                        retry_after_seconds=30,
                        failure_phase="resource_preflight",
                    )
            record["gpu_index"] = gpu_index
            record["attempted"] = True
            try:
                output = run_remote(
                    atomic_start_body(context, gpu_index), timeout=150, phase="launch_command"
                )
            except ToolError as exc:
                if "CONVIR_OPS_RESOURCE_WAIT_REQUIRED" in str(exc) and "CONVIR_OPS_FRESH_WORKSPACE_CLEANED" in str(exc):
                    record["attempted"] = False
                    return typed_failure(
                        "RESOURCE_WAIT_REQUIRED", "command_infra", "resource changed before launch",
                        expected={"runner_started": False},
                        next_actions=["convir_route_start"],
                        retry_after_seconds=30,
                        failure_phase="resource_preflight",
                    )
                return typed_failure(
                    "START_STATE_UNKNOWN", "command_infra", str(exc),
                    observed={"remote_repo": context["remote_repo"], "runner_started": "unknown"},
                    next_actions=["convir_route_start"],
                    failure_phase="launch_command",
                )
            receipt = issue_receipt(context, gpu_index, output)
            record["receipt"] = receipt
            return typed_result(
                True, "LAUNCHED",
                observed={
                    "route_id": context["route_id"],
                    "session": context["session"],
                    "output_path": context["output_path"],
                    "remote_repo": context["remote_repo"],
                    "gpu_index": gpu_index,
                },
                expected={"runner_sha256": context["runner_sha256"]},
                next_actions=["convir_route_finish"],
                receipt=receipt,
            )
    except Exception as exc:
        return failure_result("START_REJECTED", exc, "launch_command")


def receipt_context(token):
    with locked_record("receipt", token) as record:
        if not record.get("launched"):
            raise ToolError("receipt has no successful launch")
        return record["payload"]["context"]


def begin_finish(token):
    with locked_record("receipt", token) as record:
        if not record.get("launched"):
            raise ToolError("receipt has no successful launch")
        if record.get("finish_closed"):
            raise ToolError(f"finish is closed: {record['finish_closed']}")
        calls = record.get("finish_calls", 0)
        if not isinstance(calls, int) or calls < 0:
            raise ToolError("receipt finish counter is invalid", failure_class="command_infra")
        if calls >= MAX_FINISH_WINDOWS:
            record["finish_closed"] = "OBSERVATION_BUDGET_EXHAUSTED"
            raise ToolError("finish observation budget is exhausted")
        record["finish_calls"] = calls + 1
        context = dict(record["payload"]["context"])
        context["_receipt_issued_at"] = int(record["payload"]["issued_at"])
        context["_monitor_stale_count"] = int(record.get("monitor_stale_count", 0))
        return context


def close_finish(token, state):
    with locked_record("receipt", token) as record:
        record["finish_closed"] = require_token(state, "finish_closed")


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
    if terminal not in context["allowed_terminal_tuples"]:
        raise ToolError("closeout terminal tuple is not allowed", failure_class="evidence")
    match = re.search(r"(?m)^CONVIR_OPS_CLOSEOUT_SHA256=([0-9a-f]{64})$", output)
    if not match:
        raise ToolError("closeout SHA-256 is missing", failure_class="evidence")
    return {
        "identity": expected_identity, "terminal_tuple": terminal,
        "closeout_sha256": match.group(1), "closeout_filename": context["closeout_filename"],
    }


def tool_finish(args):
    token = args.get("receipt")
    try:
        context = begin_finish(token)
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
            close_finish(token, "CLOSEOUT_VALIDATED")
            return typed_result(
                True, "CLOSEOUT_VALIDATED",
                observed={"monitor": monitor, "closeout": closeout},
                next_actions=["scientific_review_or_archive"],
                manifest={"closeout_filename": closeout["closeout_filename"], "closeout_sha256": closeout["closeout_sha256"]},
            )
        if not monitor["active"]:
            close_finish(token, "CLOSEOUT_MISSING")
            return typed_failure(
                "CLOSEOUT_MISSING", "evidence", "session ended without closeout",
                observed=monitor, next_actions=["engineering_review_once"], failure_phase="closeout",
            )
        return typed_result(True, "MONITOR_OBSERVED", observed=monitor, next_actions=["convir_route_finish"])
    except (json.JSONDecodeError, TypeError) as exc:
        return failure_result("FINISH_INVALID", ToolError(str(exc), failure_class="evidence"), "closeout")
    except Exception as exc:
        return failure_result("FINISH_REJECTED", exc, "monitor")


def evidence_context(args):
    context = receipt_context(args.get("receipt"))
    return {
        **context,
        "evidence_dir": f"{context['remote_repo']}/experience_docx/experiment_logs/{context['route_id']}",
    }


def validate_evidence_name(name):
    if not isinstance(name, str) or Path(name).name != name:
        raise ToolError("evidence filename must be top-level")
    if not SAFE_TOKEN.fullmatch(Path(name).stem):
        raise ToolError("evidence filename stem is invalid")
    if Path(name).suffix.lower() not in ALLOWED_EVIDENCE_SUFFIXES or "cloud_only" in name.lower():
        raise ToolError("evidence file is not compact-text eligible")
    return name


def evidence_manifest_body(context, names=None):
    lines = [
        "export LC_ALL=C",
        f"EVIDENCE_DIR={q(context['evidence_dir'])}",
        'test -d "$EVIDENCE_DIR"',
    ]
    if names is None:
        lines.extend([
            'shopt -s nullglob',
            'for path in "$EVIDENCE_DIR"/*; do',
            '  test -f "$path" || continue',
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
    return path


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        destination_dir = local_repo / "experience_docx" / "experiment_logs" / context["route_id"]
        destination_dir.mkdir(parents=True, exist_ok=True)
        fetched, verified, pending = [], [], []
        for name in names:
            destination = destination_dir / name
            if destination.exists():
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
                    if not source.is_file() or sha256_file(source) != records[name]["sha256"]:
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
        return text_result(json.dumps(value, indent=2), structured=value)
    except Exception as exc:
        return failure_result("EVIDENCE_FETCH_FAILED", exc, "evidence_transfer")


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
        status = run_local([*prefix, "status", "--short"], timeout=30, phase="git_status")
        changed = status.splitlines()[:100] if status else []
        route_prefix = f"experience_docx/experiment_logs/{route_id}/"
        value = {
            "local_repo": str(repo), "branch": branch, "head": head,
            "github_main_local": local_main, "github_main_remote": remote[0],
            "github_main_ref_fresh": local_main == remote[0],
            "worktree_clean": not status, "changed_paths": changed,
            "route_evidence_changes": [line for line in changed if route_prefix in line],
            "diff_check": inspect_local([*prefix, "diff", "--check"]),
            "cached_diff_check": inspect_local([*prefix, "diff", "--cached", "--check"]),
            "git_mutations_performed": False,
        }
        return text_result(json.dumps(value, indent=2), structured=value)
    except Exception as exc:
        return failure_result("GIT_STATUS_FAILED", exc, "git_status")


TOOLS = {
    "convir_route_plan": {
        "description": "Read and seal one schema-v4 operation from the exact GitHub route commit without contacting the cloud.",
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
        "description": "Apply a reviewed plan once and return a receipt bound to its exact route, runner, output, and rules bundle.",
        "inputSchema": {
            "type": "object", "required": ["plan_token"],
            "properties": {"plan_token": {"type": "string"}}, "additionalProperties": False,
        },
        "handler": tool_start,
    },
    "convir_route_finish": {
        "description": "Observe one sealed window of at most 60 seconds and validate terminal closeout provenance.",
        "inputSchema": {
            "type": "object", "required": ["receipt"],
            "properties": {"receipt": {"type": "string"}}, "additionalProperties": False,
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
        "description": "Read-only worktree and GitHub-main freshness audit; never fetch, stage, commit, or push.",
        "inputSchema": {
            "type": "object", "required": ["route_id", "local_repo"],
            "properties": {"route_id": {"type": "string"}, "local_repo": {"type": "string"}},
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
