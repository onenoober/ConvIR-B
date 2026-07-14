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
import time
import uuid
import tempfile
from contextlib import contextmanager
from pathlib import Path


SERVER_NAME = "convir-ops"
SERVER_VERSION = "1.2.0"
REMOTE_HOST = "convir-4090"
REMOTE_BASE = "/sda/home/wangyuxin/ConvIR-B"
REMOTE_REPOS = f"{REMOTE_BASE}/repos"
REMOTE_RUNS = f"{REMOTE_BASE}/runs"
REMOTE_PYTHON = f"{REMOTE_BASE}/envs/convir-cu121/bin/python"
LOCAL_WORKSPACE_ROOT = Path(os.environ.get("CONVIR_OPS_LOCAL_WORKSPACE_ROOT", "/home/ubuntu/workspace")).resolve()
MAX_EVIDENCE_BYTES = 1024 * 1024
SAFE_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
SAFE_BRANCH = re.compile(r"^codex/[A-Za-z0-9][A-Za-z0-9_.\-/]{0,191}$")
SHA40 = re.compile(r"^[0-9a-f]{40}$")
ALLOWED_EVIDENCE_SUFFIXES = {".json", ".csv", ".md", ".txt"}
SCHEMA_VERSION = 2
RECEIPT_TTL_SECONDS = 15 * 60
MONITOR_MAX_POLLS = 20
MONITOR_MAX_INTERVAL_SECONDS = 30
MAX_CLOSEOUT_BYTES = 64 * 1024
MAX_MONITOR_SECONDS = 60
RECEIPT_DIR = Path(os.environ.get("CONVIR_OPS_RECEIPT_DIR", "~/.codex/convir-ops/receipts")).expanduser().resolve()


class ToolError(RuntimeError):
    """Expected user-facing tool rejection."""


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


def structured_payload(result):
    payload = result.get("structuredContent")
    if not isinstance(payload, dict):
        raise ToolError("internal typed result is missing structuredContent")
    return payload


def failure_class_for_error(error):
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
        "source_repo": f"{REMOTE_REPOS}/{repo_name}",
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
    workspace_digest = hashlib.sha256(
        f"{context['repo_name']}\0{context['route_id']}\0{context['expected_commit']}\0{context['mode']}\0{context['output_id']}".encode("utf-8")
    ).hexdigest()[:16]
    workspace_prefix = f"{context['repo_name'][:20]}-{context['route_id'][:24]}-{context['output_id'][:20]}"[:64]
    context["remote_repo"] = f"{REMOTE_REPOS}/{workspace_prefix}-{workspace_digest}"
    if arguments.get("collision_policy") != "must_not_exist":
        raise ToolError("collision_policy must be must_not_exist")
    context["output_path"] = f"{context['run_root']}/{context['output_id']}"
    context["closeout_filename"] = require_closeout_filename(arguments.get("closeout_filename"))
    context["closeout_path"] = f"{context['remote_repo']}/experience_docx/experiment_logs/{context['route_id']}/{context['closeout_filename']}"
    context["authorization_relpath"] = require_repo_relpath(arguments.get("authorization_relpath"), "authorization_relpath")
    context["prior_terminal_tuple"] = require_terminal_tuple(arguments.get("prior_terminal_tuple"), "prior_terminal_tuple")
    context["allowed_terminal_tuples"] = require_terminal_tuples(arguments.get("allowed_terminal_tuples"))
    return context


def receipt_payload(context, runner_sha, preflight_digest, now):
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
def locked_receipt(token):
    path = receipt_path(token)
    lock = path.with_suffix(".lock")
    deadline = time.monotonic() + 5
    while True:
        try:
            lock.mkdir()
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise ToolError("receipt record is busy")
            time.sleep(0.02)
    try:
        if not path.is_file():
            raise ToolError("receipt is unknown or malformed")
        record = json.loads(path.read_text(encoding="utf-8"))
        yield record
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, sort_keys=True, separators=(",", ":")), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        lock.rmdir()


def issue_receipt(context, preflight_output):
    match = re.search(r"(?m)^([0-9a-f]{64})\s+.+$", preflight_output)
    if not match:
        raise ToolError("preflight did not report the runner SHA-256")
    runner_sha = match.group(1)
    payload = receipt_payload(context, runner_sha, canonical_digest(preflight_output), int(time.time()))
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
    path = plan_path(token)
    lock = path.with_suffix(".lock")
    deadline = time.monotonic() + 5
    while True:
        try:
            lock.mkdir()
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise ToolError("plan record is busy")
            time.sleep(0.02)
    try:
        if not path.is_file():
            raise ToolError("plan_token is unknown or malformed")
        record = json.loads(path.read_text(encoding="utf-8"))
        yield record
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, sort_keys=True, separators=(",", ":")), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        lock.rmdir()


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
    try:
        result = subprocess.run(args, text=True, capture_output=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        raise ToolError(f"local operation timed out after {timeout}s") from exc
    output = (result.stdout + result.stderr).strip()
    if result.returncode != 0:
        raise ToolError(output or f"local operation failed with rc={result.returncode}")
    return output


def inspect_local(args, timeout=30):
    """Run a fixed local inspection command without treating a finding as a tool failure."""
    try:
        result = subprocess.run(args, text=True, capture_output=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return {"ok": False, "returncode": None, "output": f"timed out after {timeout}s"}
    output = (result.stdout + result.stderr).strip()
    return {"ok": result.returncode == 0, "returncode": result.returncode, "output": output[:8192]}


def run_remote_body(body, timeout=45):
    with tempfile.NamedTemporaryFile("w", suffix=".sh", prefix="convir-ops-", delete=False, encoding="utf-8") as handle:
        body_path = Path(handle.name)
        handle.write("#!/usr/bin/env bash\nset -euo pipefail\n")
        handle.write(body)
        handle.write("\n")
    try:
        return run_local(["bash", str(helper_path()), str(body_path)], timeout)
    finally:
        body_path.unlink(missing_ok=True)


def github_plan_body(context):
    """Read-only GitHub identity and tree checks for the plan phase."""
    return "\n".join([
        f"SOURCE_REPO={q(context['source_repo'])}", f"BRANCH={q(context['branch'])}",
        f"EXPECTED_COMMIT={q(context['expected_commit'])}", f"RULES_COMMIT={q(context['rules_commit'])}", f"RUNNER={q(context['runner_relpath'])}",
        'test -d "$SOURCE_REPO/.git"',
        'GITHUB_URL=$(git -C "$SOURCE_REPO" remote get-url github)',
        'test -n "$GITHUB_URL"',
        'test "$(git ls-remote "$GITHUB_URL" "refs/heads/$BRANCH" | awk \'NR==1 {print $1}\')" = "$EXPECTED_COMMIT"',
        'test "$(git ls-remote "$GITHUB_URL" refs/heads/main | awk \'NR==1 {print $1}\')" = "$RULES_COMMIT"',
        'git -C "$SOURCE_REPO" cat-file -e "$EXPECTED_COMMIT:$RUNNER"',
        'echo CONVIR_OPS_PLAN_GITHUB_OK',
    ])


def preflight_body(context, include_gpu, create_clone=False):
    lines = [
        f"REMOTE_REPO={q(context['remote_repo'])}", f"SOURCE_REPO={q(context.get('source_repo', context['remote_repo']))}",
        f"RUN_ROOT={q(context['run_root'])}",
        f"BRANCH={q(context['branch'])}",
        f"EXPECTED_COMMIT={q(context['expected_commit'])}",
        f"RULES_COMMIT={q(context.get('rules_commit', 'missing'))}",
        f"RUNNER={q(context['runner_relpath'])}",
        f"SESSION={q(context['session'])}",
        f"PY={q(REMOTE_PYTHON)}",
        'FRESH_CREATED=0' if create_clone else ':',
        'cleanup_fresh() { rc=$?; if [ "$FRESH_CREATED" = 1 ]; then rm -rf -- "$REMOTE_REPO"; echo CONVIR_OPS_FRESH_WORKSPACE_CLEANED; fi; exit "$rc"; }' if create_clone else ':',
        'trap cleanup_fresh ERR' if create_clone else ':',
        'test -d "$SOURCE_REPO/.git"',
        'GITHUB_URL=$(git -C "$SOURCE_REPO" remote get-url github)' if create_clone else ':',
        'test ! -e "$REMOTE_REPO"' if create_clone else 'test -d "$REMOTE_REPO/.git"',
        'FRESH_CREATED=1' if create_clone else ':',
        'git clone --origin github --no-checkout "$GITHUB_URL" "$REMOTE_REPO"' if create_clone else ':',
        'git -C "$REMOTE_REPO" fetch --quiet github "$BRANCH" main' if create_clone else ':',
        'git -C "$REMOTE_REPO" checkout --quiet "$BRANCH"',
        'test "$(git -C "$REMOTE_REPO" branch --show-current)" = "$BRANCH"',
        'test "$(git -C "$REMOTE_REPO" rev-parse HEAD)" = "$EXPECTED_COMMIT"',
        'git -C "$REMOTE_REPO" fetch --quiet github main',
        'test "$(git -C "$REMOTE_REPO" rev-parse github/main)" = "$RULES_COMMIT"',
        'test -z "$(git -C "$REMOTE_REPO" status --porcelain)"',
        'test -x "$PY"',
        'test -f "$REMOTE_REPO/$RUNNER"',
        'sha256sum "$REMOTE_REPO/$RUNNER"',
        f'test ! -e {q(context.get("output_path", context["run_root"] + "/unbound"))}',
        f'test ! -e {q(context.get("closeout_path", context["remote_repo"] + "/unbound-closeout"))}',
        'if tmux has-session -t "$SESSION" 2>/dev/null; then echo CONVIR_OPS_SESSION_CONFLICT; exit 1; fi',
        'echo CONVIR_OPS_SESSION_FREE',
    ]
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
        lines.append('nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader')
    if create_clone:
        lines.append('trap - ERR')
    lines.append('echo "CONVIR_OPS_PREFLIGHT_OK route=$RUN_ROOT mode=' + context["mode"] + '"')
    return "\n".join(lines)


def tool_prepare_authorized(arguments):
    """Plan/apply authorization boundary. Apply is the only receipt producer."""
    try:
        context = authorization_context(arguments)
        phase = require_enum(arguments.get("phase"), "phase", {"plan", "apply"})
        plan = canonical_digest(context)
        if phase == "plan":
            output = run_remote_body(github_plan_body(context), timeout=30)
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
            output = run_remote_body(preflight_body(context, context["require_gpu"], create_clone=True), timeout=45)
        except Exception as exc:
            return typed_failure("PREPARE_RECOVERY_REQUIRED", failure_class_for_error(exc), str(exc) or type(exc).__name__, observed={"remote_repo": context["remote_repo"], "fresh_workspace_cleanup": "trap_after_path_reservation"}, allowed_next_actions=["route_prepare_authorized.apply"])
        receipt, payload = issue_receipt(context, output)
        return typed_result(True, "PREPARED", observed={"preflight": output}, expected={"authorization_tuple": context}, allowed_next_actions=["route_launch"], receipt=receipt, receipt_expires_at=payload["expires_at"], receipt_digest=canonical_digest(payload))
    except ToolError as exc:
        return typed_failure("PREPARE_REJECTED", failure_class_for_error(exc), str(exc))
    except Exception as exc:
        return typed_failure("PREPARE_FAILED", "command_infra", type(exc).__name__)


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
            "source_repo": payload["remote_repo"],
            "run_root": f"{REMOTE_RUNS}/{payload['route_id']}", "branch": payload["branch"], "expected_commit": payload["route_branch_commit"],
            "runner_relpath": payload["runner_relpath"], "mode": payload["mode"], "session": payload["session"], "require_gpu": payload["require_gpu"],
            "output_path": payload["output_path"], "rules_commit": payload["rules_commit"], "authorization_relpath": payload["authorization_relpath"], "prior_terminal_tuple": payload["prior_terminal_tuple"],
            }
            output = run_remote_body("\n".join([
            preflight_body(context, context["require_gpu"]),
            f"tmux new-session -d -s {q(context['session'])} env EXPECTED_ROUTE_COMMIT={q(context['expected_commit'])} MODE={q(context['mode'])} REMOTE_REPO={q(context['remote_repo'])} RUN_ROOT={q(context['run_root'])} RUN_ID={q(payload['output_id'])} OUTPUT_ID={q(payload['output_id'])} bash {q(context['remote_repo'] + '/' + context['runner_relpath'])}",
            'echo "CONVIR_OPS_LAUNCH_OK session=$SESSION"',
            ]), timeout=60)
            record["launched"] = True
            record["launch_key"] = key
            record["launch_result"] = {"transport": output, "session": payload["session"], "output_id": payload["output_id"]}
            return typed_result(True, "LAUNCHED", observed=record["launch_result"], expected={"receipt": token}, allowed_next_actions=["route_monitor", "route_closeout_validate"])
    except ToolError as exc:
        return typed_failure("LAUNCH_REJECTED", "authorization", str(exc))
    except Exception as exc:
        return typed_failure("LAUNCH_FAILED", "command_infra", type(exc).__name__, allowed_next_actions=["route_prepare_authorized.apply"])


def tool_receipt_monitor(arguments):
    try:
        token, lock = resolve_receipt(arguments)
        with lock as record:
            payload = validate_receipt_record(token, record)
        mode = require_enum(arguments.get("monitor_mode", "poll"), "monitor_mode", {"poll", "until_change", "until_terminal"})
        max_polls = require_int(arguments.get("max_polls"), "max_polls", 1, 1, MONITOR_MAX_POLLS)
        interval = require_int(arguments.get("interval_seconds"), "interval_seconds", 0, 0, MONITOR_MAX_INTERVAL_SECONDS)
        if mode == "poll":
            max_polls = 1
        max_polls = min(max_polls, max(1, MAX_MONITOR_SECONDS // max(1, interval + 1)))
        remote_run_root = f"{REMOTE_RUNS}/{payload['route_id']}"
        body = "\n".join([
            f"RUN_ROOT={q(remote_run_root)}", f"SESSION={q(payload['session'])}", f"PY={q(REMOTE_PYTHON)}", f"TERMINALS={q(json.dumps(payload['allowed_terminal_tuples'], separators=(',', ':')))}",
            f"MAX_POLLS={max_polls}", f"INTERVAL={interval}", f"MODE={q(mode)}", "previous=''", "for n in $(seq 1 \"$MAX_POLLS\"); do",
            '  active=false; tmux has-session -t "$SESSION" 2>/dev/null && active=true',
            '  status=$(test -f "$RUN_ROOT/status.txt" && tail -n 16 "$RUN_ROOT/status.txt" || true)',
            '  terminal=$($PY -c "import json,re,sys; allowed=json.loads(sys.argv[1]); raw=sys.argv[2]; present=lambda k,v: re.search(r\"(?m)(?:^|[ \\t])\"+re.escape(str(k))+r\"=\"+re.escape(str(v))+r\"(?=$|[ \\t])\",raw) is not None; print(any(all(present(k,v) for k,v in item.items()) for item in allowed))" "$TERMINALS" "$status")',
            '  final_status="$status"; final_active="$active"; test "$terminal" = True && { terminal=true; break; }',
            '  test "$MODE" = poll && break', '  test "$MODE" = until_change && test "$status" != "$previous" && test -n "$previous" && break',
            '  previous=$status; test "$n" = "$MAX_POLLS" || sleep "$INTERVAL"', 'done',
            'printf "CONVIR_OPS_MONITOR_META polls=%s active=%s terminal=%s\\n" "$n" "$final_active" "${terminal:-false}"',
            'echo CONVIR_OPS_MONITOR_STATUS_BEGIN', 'printf "%s\\n" "$final_status"', 'echo CONVIR_OPS_MONITOR_STATUS_END',
        ])
        timeout = max(MAX_MONITOR_SECONDS + 5, max_polls * max(1, interval) + 10)
        output = run_remote_body(body, timeout=timeout)
        meta = re.search(r"(?m)^CONVIR_OPS_MONITOR_META polls=(\d+) active=(true|false) terminal=(true|false)$", output)
        begin, end = "CONVIR_OPS_MONITOR_STATUS_BEGIN", "CONVIR_OPS_MONITOR_STATUS_END"
        start, finish = output.find(begin), output.find(end)
        if not meta or start < 0 or finish < 0 or finish <= start:
            raise ToolError("monitor wrapper markers are missing")
        active = meta.group(2) == "true"
        terminal = meta.group(3) == "true"
        state = "MONITOR_TERMINAL" if terminal else ("MONITOR_OBSERVED" if active else "MONITOR_INACTIVE_CLOSEOUT_PENDING")
        summary = output[start + len(begin):finish].strip()[:4096]
        polls = int(meta.group(1))
        return typed_result(True, state, observed={"status": summary, "monitor_mode": mode, "active": active, "terminal": terminal, "poll_count": polls}, expected={"receipt": canonical_digest(payload)}, allowed_next_actions=["route_closeout_validate"] if terminal or not active else ["route_monitor"])
    except ToolError as exc:
        return typed_failure("MONITOR_REJECTED", "authorization", str(exc))
    except Exception as exc:
        return typed_failure("MONITOR_FAILED", "command_infra", type(exc).__name__)


def tool_closeout_validate(arguments):
    try:
        token, lock = resolve_receipt(arguments)
        with lock as record:
            payload = validate_receipt_record(token, record)
        expected = require_terminal_tuple(arguments.get("terminal_tuple"), "terminal_tuple")
        if expected not in payload["allowed_terminal_tuples"]:
            return typed_failure("CLOSEOUT_REJECTED", "authorization", "terminal tuple is not in the sealed allowed_terminal_tuples set", expected={"allowed_terminal_tuples": payload["allowed_terminal_tuples"]})
        filename = payload["closeout_filename"]
        remote_path = f"{payload['remote_repo']}/experience_docx/experiment_logs/{payload['route_id']}/{filename}"
        extractor = "import hashlib,json,sys; raw=open(sys.argv[1],'rb').read(65537); assert len(raw)<=65536; value=json.loads(raw); print('CONVIR_OPS_CLOSEOUT_SHA256='+hashlib.sha256(raw).hexdigest()); print('CONVIR_OPS_CLOSEOUT_JSON_BEGIN'); print(json.dumps(value,sort_keys=True,separators=(',',':'))); print('CONVIR_OPS_CLOSEOUT_JSON_END')"
        output = run_remote_body(f"test -f {q(remote_path)}\n{q(REMOTE_PYTHON)} -c {q(extractor)} {q(remote_path)}", timeout=30)
        begin, end = "CONVIR_OPS_CLOSEOUT_JSON_BEGIN", "CONVIR_OPS_CLOSEOUT_JSON_END"
        start, finish = output.find(begin), output.find(end)
        if start < 0 or finish < 0 or finish <= start:
            raise ToolError("closeout wrapper markers are missing")
        observed = json.loads(output[start + len(begin):finish].strip())
        if observed.get("route_id") != payload["route_id"]:
            return typed_failure("CLOSEOUT_INVALID", "evidence", "runner closeout route_id mismatch", observed={"route_id": observed.get("route_id")}, expected={"route_id": payload["route_id"]})
        actual = {key: observed.get(key) for key in expected}
        if actual != expected:
            return typed_failure("CLOSEOUT_INVALID", "evidence", "runner closeout tuple mismatch", observed=actual, expected=expected)
        sha_match = re.search(r"(?m)^CONVIR_OPS_CLOSEOUT_SHA256=([0-9a-f]{64})$", output)
        if not sha_match:
            raise ToolError("closeout raw SHA-256 is missing")
        manifest = {"closeout_filename": filename, "closeout_sha256": sha_match.group(1), "receipt_digest": canonical_digest(payload)}
        return typed_result(True, "CLOSEOUT_VALIDATED", observed=actual, expected=expected, allowed_next_actions=["human_review_archive_candidate"], manifest=manifest, archive_candidate=json.dumps({"route_id": payload["route_id"], "terminal_tuple": actual, "manifest": manifest}, sort_keys=True))
    except ToolError as exc:
        return typed_failure("CLOSEOUT_REJECTED", "authorization", str(exc))
    except (json.JSONDecodeError, TypeError):
        return typed_failure("CLOSEOUT_INVALID", "evaluation", "runner closeout is not valid compact JSON")
    except Exception as exc:
        return typed_failure("CLOSEOUT_FAILED", "command_infra", type(exc).__name__)


def tool_start_authorized(arguments):
    """Compose reviewed apply and receipt launch without exposing free-form commands."""
    token = arguments.get("plan_token")
    idempotency_key = require_token(arguments.get("idempotency_key"), "idempotency_key")
    with locked_plan(token) as plan_record:
        plan_payload = validate_plan_record(token, plan_record)
        context = plan_payload["context"]
        plan_hash = canonical_digest(context)
        if arguments.get("plan_hash") != plan_hash:
            return typed_failure("START_REJECTED", "authorization", "plan_hash does not bind the stored plan", expected={"plan_hash": plan_hash})
        if plan_record.get("consumed"):
            return typed_failure("START_REJECTED", "authorization", "plan token has already been consumed", observed={"receipt": plan_record.get("receipt")})
        try:
            output = run_remote_body(preflight_body(context, context["require_gpu"], create_clone=True), timeout=45)
        except Exception as exc:
            return typed_failure("PREPARE_RECOVERY_REQUIRED", failure_class_for_error(exc), str(exc) or type(exc).__name__, observed={"remote_repo": context["remote_repo"], "fresh_workspace_cleanup": "trap_after_path_reservation"}, allowed_next_actions=["route_prepare_authorized.plan"])
        receipt, receipt_payload_value = issue_receipt(context, output)
        plan_record["consumed"] = True
        plan_record["receipt"] = receipt
        launched_result = tool_receipt_launch({"receipt": receipt, "idempotency_key": idempotency_key})
        launched = structured_payload(launched_result)
        return typed_result(
            launched["ok"], launched["operation_state"], launched["failure_class"],
            observed={
                "session": context["session"], "output_id": context["output_id"],
                "remote_repo": context["remote_repo"], "launch_state": launched["operation_state"],
            },
            expected={"receipt_digest": canonical_digest(receipt_payload_value), "plan_hash": plan_hash},
            mismatches=launched["mismatches"], allowed_next_actions=launched["allowed_next_actions"],
            receipt=receipt, receipt_expires_at=receipt_payload_value["expires_at"],
        )


def tool_finish(arguments):
    """Observe a bounded interval and validate sealed closeout when inactive."""
    monitor_arguments = {key: arguments[key] for key in ("receipt", "monitor_mode", "max_polls", "interval_seconds") if key in arguments}
    monitor_arguments.setdefault("monitor_mode", "until_terminal")
    monitor_arguments.setdefault("max_polls", 15)
    monitor_arguments.setdefault("interval_seconds", 3)
    monitored_result = tool_receipt_monitor(monitor_arguments)
    monitored = structured_payload(monitored_result)
    if not monitored["ok"] or monitored["operation_state"] == "MONITOR_OBSERVED":
        return monitored_result
    closeout_result = tool_closeout_validate({"receipt": arguments.get("receipt"), "terminal_tuple": arguments.get("terminal_tuple")})
    closeout = structured_payload(closeout_result)
    monitor_summary = {
        key: monitored["observed"].get(key)
        for key in ("active", "terminal", "poll_count", "status")
    }
    return typed_result(
        closeout["ok"], closeout["operation_state"], closeout["failure_class"],
        observed={"monitor": monitor_summary, "terminal_tuple": closeout["observed"]},
        expected=closeout["expected"], mismatches=closeout["mismatches"],
        allowed_next_actions=closeout["allowed_next_actions"],
        **{key: closeout[key] for key in ("manifest", "archive_candidate") if key in closeout},
    )


def validate_evidence_file(name):
    require_token(name.rsplit(".", 1)[0] if "." in name else "", "evidence filename stem")
    suffix = Path(name).suffix.lower()
    if suffix not in ALLOWED_EVIDENCE_SUFFIXES or "cloud_only" in name.lower():
        raise ToolError("evidence files must be compact .json/.csv/.md/.txt and must not be cloud_only")
    return name


def manifest_body(context, files=None):
    lines = [f"EVIDENCE_DIR={q(context['evidence_dir'])}", 'test -d "$EVIDENCE_DIR"']
    if files is None:
        lines.extend([
            'shopt -s nullglob',
            'for path in "$EVIDENCE_DIR"/*; do',
            '  name=$(basename "$path")',
            '  case "$name" in *.json|*.csv|*.md|*.txt) ;; *) continue ;; esac',
            '  case "$name" in *cloud_only*) continue ;; esac',
            '  size=$(wc -c < "$path")',
            f'  if [ "$size" -le {MAX_EVIDENCE_BYTES} ]; then',
            '    read -r digest _ < <(sha256sum "$path")',
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
                f'printf "{name}\\t%s\\t%s\\n" "$size" "$digest"',
            ])
    lines.append("echo CONVIR_OPS_EVIDENCE_MANIFEST_OK")
    return "\n".join(lines)


def parse_manifest(output):
    records = {}
    for line in output.splitlines():
        fields = line.split("\t")
        if len(fields) == 3 and fields[1].isdigit() and re.fullmatch(r"[0-9a-f]{64}", fields[2]):
            records[fields[0]] = {"bytes": int(fields[1]), "sha256": fields[2]}
    return records


def tool_evidence_manifest(arguments):
    context = route_context(arguments)
    return text_result(run_remote_body(manifest_body(context), timeout=30))


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
    context = route_context(arguments)
    local_repo = validate_local_repo(arguments.get("local_repo"))
    files = arguments.get("files")
    if not isinstance(files, list) or not files or len(files) > 32:
        raise ToolError("files must be a non-empty list of at most 32 compact evidence filenames")
    names = [validate_evidence_file(item) for item in files]
    if len(set(names)) != len(names):
        raise ToolError("files must not contain duplicates")
    records = parse_manifest(run_remote_body(manifest_body(context, names), timeout=45))
    if set(records) != set(names):
        raise ToolError("remote evidence manifest did not return the requested allowlist exactly")
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
            180,
            max(60, 30 + sum(records[name]["bytes"] for name, _, _ in pending) // (256 * 1024)),
        )
        with tempfile.TemporaryDirectory(prefix=".convir-evidence-", dir=destination_dir) as staging:
            staging_dir = Path(staging)
            sources = [f"{REMOTE_HOST}:{context['evidence_dir']}/{name}" for name, _, _ in pending]
            run_local(["scp", *sources, str(staging_dir)], timeout=transfer_timeout)
            for name, _, expected in pending:
                staged = staging_dir / name
                if not staged.is_file() or sha256_file(staged) != expected:
                    raise ToolError(f"downloaded evidence hash mismatch: {name}")
            for name, destination, expected in pending:
                try:
                    os.link(staging_dir / name, destination)
                except FileExistsError as exc:
                    raise ToolError(f"refusing to overwrite newly created local evidence: {destination}") from exc
                if sha256_file(destination) != expected:
                    destination.unlink(missing_ok=True)
                    raise ToolError(f"local evidence hash mismatch after transfer: {name}")
                fetched.append(name)
    return text_result(json.dumps({"fetched": fetched, "already_verified": skipped, "destination": str(destination_dir), "transfer": "single_scp"}, indent=2))


def tool_git_evidence_status(arguments):
    route_id = require_token(arguments.get("route_id"), "route_id")
    local_repo = validate_local_repo(arguments.get("local_repo"))
    prefix = ["git", "-C", str(local_repo)]
    branch = run_local([*prefix, "branch", "--show-current"], timeout=15)
    head = run_local([*prefix, "rev-parse", "HEAD"], timeout=15)
    local_main = run_local([*prefix, "rev-parse", "--verify", "github/main"], timeout=15)
    remote_main_raw = run_local([*prefix, "ls-remote", "github", "refs/heads/main"], timeout=30)
    remote_main_fields = remote_main_raw.split()
    if len(remote_main_fields) != 2 or not SHA40.fullmatch(remote_main_fields[0]):
        raise ToolError("github main could not be resolved through git ls-remote")
    remote_main = remote_main_fields[0]
    status = run_local([*prefix, "status", "--short"], timeout=20)
    diff_check = inspect_local([*prefix, "diff", "--check"], timeout=20)
    cached_diff_check = inspect_local([*prefix, "diff", "--cached", "--check"], timeout=20)
    ahead = behind = None
    if local_main == remote_main:
        counts = run_local([*prefix, "rev-list", "--left-right", "--count", "HEAD...github/main"], timeout=20).split()
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


TOOLS = {
    "convir_route_prepare_authorized": {
        "description": "Two-phase plan/apply authorization for a fresh route workspace. Apply seals an exact authorization tuple, successful preflight, and a one-use launch receipt; it never launches a runner.",
        "inputSchema": {
            "type": "object",
            "required": ["schema_version", "phase", "route_id", "repo_name", "branch", "route_branch_commit", "rules_commit", "runner_relpath", "mode", "stage_state", "decision", "authorizes", "locked_test_policy", "forbidden_continuations", "output_id", "closeout_filename", "collision_policy", "authorization_relpath", "prior_terminal_tuple", "allowed_terminal_tuples"],
            "properties": {
                "schema_version": {"const": 2}, "phase": {"enum": ["plan", "apply"]}, "plan_hash": {"type": "string"}, "route_id": {"type": "string"}, "repo_name": {"type": "string"}, "branch": {"type": "string"}, "route_branch_commit": {"type": "string"}, "rules_commit": {"type": "string"}, "runner_relpath": {"type": "string"}, "mode": {"type": "string", "pattern": "^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$"}, "require_gpu": {"type": "boolean", "default": True}, "stage_state": {"type": "string"}, "decision": {"type": "string"}, "authorizes": {"type": "string"}, "locked_test_policy": {"enum": ["blocked", "explicitly_authorized"]}, "forbidden_continuations": {"type": "array", "items": {"type": "string"}}, "output_id": {"type": "string"}, "closeout_filename": {"type": "string", "pattern": "^[A-Za-z0-9_.-]+_closeout\\.json$"}, "collision_policy": {"const": "must_not_exist"}, "authorization_relpath": {"type": "string"}, "prior_terminal_tuple": {"type": "object"}, "allowed_terminal_tuples": {"type": "array", "minItems": 1, "maxItems": 16, "items": {"type": "object"}}
            },
            "additionalProperties": False,
        },
        "handler": tool_prepare_authorized,
    },
    "convir_route_launch": {
        "description": "Receipt-bound idempotent launch of the runner sealed by authorized preparation. It repeats dynamic preflight and accepts no command, path, or tuple fields.",
        "inputSchema": {
            "type": "object",
            "required": ["receipt", "idempotency_key"],
            "properties": {"receipt": {"type": "string"}, "idempotency_key": {"type": "string"}},
            "additionalProperties": False,
        },
        "handler": tool_receipt_launch,
    },
    "convir_route_monitor": {
        "description": "Receipt-bound bounded monitor. until_terminal polling occurs on the server and reports observed state without making a gate decision.",
        "inputSchema": {
            "type": "object",
            "required": ["receipt"],
            "properties": {"receipt": {"type": "string"}, "monitor_mode": {"enum": ["poll", "until_change", "until_terminal"], "default": "poll"}, "max_polls": {"type": "integer", "minimum": 1, "maximum": 20}, "interval_seconds": {"type": "integer", "minimum": 0, "maximum": 30}},
            "additionalProperties": False,
        },
        "handler": tool_receipt_monitor,
    },
    "convir_route_closeout_validate": {
        "description": "Receipt-bound typed closeout validation. It checks only the exact sealed terminal tuple and returns a checksum manifest plus an archive candidate; it never commits or pushes.",
        "inputSchema": {
            "type": "object",
            "required": ["receipt", "terminal_tuple"],
            "properties": {"receipt": {"type": "string"}, "terminal_tuple": {"type": "object", "required": ["state", "decision", "authorizes"], "additionalProperties": False, "properties": {"state": {"type": "string"}, "decision": {"type": "string"}, "authorizes": {"type": "string"}}}},
            "additionalProperties": False,
        },
        "handler": tool_closeout_validate,
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

TOOLS["convir_route_start_authorized"] = {
    "description": "Recommended normal-path composition: after a reviewed prepare plan, apply the exact plan hash and launch its receipt-bound runner in one call.",
    "inputSchema": {
        "type": "object", "required": ["plan_token", "plan_hash", "idempotency_key"],
        "properties": {
            "plan_token": {"type": "string"}, "plan_hash": {"type": "string"},
            "idempotency_key": {"type": "string"},
        },
        "additionalProperties": False,
    },
    "handler": tool_start_authorized,
}
TOOLS["convir_route_finish"] = {
    "description": "Recommended normal-path composition: bounded monitoring followed by sealed closeout validation only after the session is inactive or a sealed terminal tuple is observed.",
    "inputSchema": {
        "type": "object", "required": ["receipt", "terminal_tuple"],
        "properties": {
            "receipt": {"type": "string"},
            "terminal_tuple": {"type": "object", "required": ["state", "decision", "authorizes"]},
            "monitor_mode": {"enum": ["poll", "until_change", "until_terminal"], "default": "until_terminal"},
            "max_polls": {"type": "integer", "minimum": 1, "maximum": 20},
            "interval_seconds": {"type": "integer", "minimum": 0, "maximum": 30},
        },
        "additionalProperties": False,
    },
    "handler": tool_finish,
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
