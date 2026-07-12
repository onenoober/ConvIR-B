#!/usr/bin/env python3
"""Restricted MCP operations bridge for the persistent convir-4090 host.

The server intentionally exposes route operations, not arbitrary SSH execution.
It runs locally under WSL and delegates PowerShell/WSL/SSH transport to the
tracked convir_remote_script.sh wrapper.
"""

import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path


SERVER_NAME = "convir-ops"
SERVER_VERSION = "1.1.0"
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


class ToolError(RuntimeError):
    """Expected user-facing tool rejection."""


def emit(value):
    sys.stdout.write(json.dumps(value, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def mcp_error(request_id, code, message):
    emit({"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}})


def mcp_result(request_id, result):
    emit({"jsonrpc": "2.0", "id": request_id, "result": result})


def text_result(text, is_error=False):
    return {"content": [{"type": "text", "text": text}], "isError": is_error}


def require_token(value, name):
    if not isinstance(value, str) or not SAFE_TOKEN.fullmatch(value):
        raise ToolError(f"{name} must contain only letters, digits, dot, underscore, or hyphen")
    return value


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
    if path.suffix != ".sh" or not str(path).startswith("experience_docx/tools/"):
        raise ToolError("runner_relpath must be an experience_docx/tools/*.sh route runner")
    return str(path)


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


def route_context(arguments):
    route_id = require_token(arguments.get("route_id"), "route_id")
    repo_name = require_token(arguments.get("repo_name"), "repo_name")
    return {
        "route_id": route_id,
        "repo_name": repo_name,
        "remote_repo": f"{REMOTE_REPOS}/{repo_name}",
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


def preflight_body(context, include_gpu):
    lines = [
        f"REMOTE_REPO={q(context['remote_repo'])}",
        f"RUN_ROOT={q(context['run_root'])}",
        f"BRANCH={q(context['branch'])}",
        f"EXPECTED_COMMIT={q(context['expected_commit'])}",
        f"RUNNER={q(context['runner_relpath'])}",
        f"SESSION={q(context['session'])}",
        f"PY={q(REMOTE_PYTHON)}",
        'test "$(git -C "$REMOTE_REPO" branch --show-current)" = "$BRANCH"',
        'test "$(git -C "$REMOTE_REPO" rev-parse HEAD)" = "$EXPECTED_COMMIT"',
        'test -z "$(git -C "$REMOTE_REPO" status --porcelain)"',
        'test -x "$PY"',
        'test -f "$REMOTE_REPO/$RUNNER"',
        'if tmux has-session -t "$SESSION" 2>/dev/null; then echo CONVIR_OPS_SESSION_CONFLICT; exit 1; fi',
        'echo CONVIR_OPS_SESSION_FREE',
    ]
    if include_gpu:
        lines.append('nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader')
    lines.append('echo "CONVIR_OPS_PREFLIGHT_OK route=$RUN_ROOT mode=' + context["mode"] + '"')
    return "\n".join(lines)


def tool_preflight(arguments):
    context = stage_context(arguments)
    return text_result(run_remote_body(preflight_body(context, context["require_gpu"])))


def tool_launch(arguments):
    context = stage_context(arguments)
    lines = [preflight_body(context, context["require_gpu"]),
        f"tmux new-session -d -s {q(context['session'])} env EXPECTED_ROUTE_COMMIT={q(context['expected_commit'])} MODE={q(context['mode'])} REMOTE_REPO={q(context['remote_repo'])} RUN_ROOT={q(context['run_root'])} bash {q(context['remote_repo'] + '/' + context['runner_relpath'])}",
        'echo "CONVIR_OPS_LAUNCH_OK session=$SESSION"',
    ]
    return text_result(run_remote_body("\n".join(lines)))


def tool_monitor(arguments):
    context = route_context(arguments)
    session = require_token(arguments.get("session"), "session")
    tail_lines = require_int(arguments.get("tail_lines"), "tail_lines", 16, 1, 80)
    body = "\n".join([
        f"RUN_ROOT={q(context['run_root'])}",
        f"EVIDENCE_DIR={q(context['evidence_dir'])}",
        f"SESSION={q(session)}",
        'if tmux has-session -t "$SESSION" 2>/dev/null; then echo CONVIR_OPS_SESSION_ACTIVE; else echo CONVIR_OPS_SESSION_INACTIVE; fi',
        'if [ -f "$RUN_ROOT/status.txt" ]; then tail -n ' + str(tail_lines) + ' "$RUN_ROOT/status.txt"; else echo CONVIR_OPS_STATUS_MISSING; fi',
        'if [ -d "$EVIDENCE_DIR" ]; then find "$EVIDENCE_DIR" -maxdepth 1 -type f -name "*_closeout.json" -printf "%f\\n" | sort | tail -n 3; fi',
        'echo CONVIR_OPS_MONITOR_OK',
    ])
    return text_result(run_remote_body(body, timeout=30))


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
            f'  if [ "$size" -le {MAX_EVIDENCE_BYTES} ]; then printf "%s\\t%s\\t%s\\n" "$name" "$size" "$(sha256sum "$path" | awk \"{{print \\\$1}}\")"; fi',
            'done',
        ])
    else:
        for name in files:
            lines.extend([
                f"path=\"$EVIDENCE_DIR/{name}\"",
                'test -f "$path"',
                'size=$(wc -c < "$path")',
                f'test "$size" -le {MAX_EVIDENCE_BYTES}',
                f'printf "{name}\\t%s\\t%s\\n" "$size" "$(sha256sum \"$path\" | awk \"{{print \\\$1}}\")"',
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
    "convir_route_preflight": {
        "description": "Read-only generic cloud preflight for a tracked route runner. It validates route identity, clean Git state, tmux availability, runner presence, and optionally current GPU availability.",
        "inputSchema": {
            "type": "object",
            "required": ["route_id", "repo_name", "branch", "expected_commit", "runner_relpath", "mode", "session"],
            "properties": {
                "route_id": {"type": "string"}, "repo_name": {"type": "string"}, "branch": {"type": "string"},
                "expected_commit": {"type": "string"}, "runner_relpath": {"type": "string"}, "mode": {"type": "string"},
                "session": {"type": "string"}, "require_gpu": {"type": "boolean", "default": True},
            },
            "additionalProperties": False,
        },
        "handler": tool_preflight,
    },
    "convir_route_launch": {
        "description": "Launch a tracked route runner in a new tmux session after repeating the generic preflight. It does not accept arbitrary commands or output paths.",
        "inputSchema": {
            "type": "object",
            "required": ["route_id", "repo_name", "branch", "expected_commit", "runner_relpath", "mode", "session"],
            "properties": {
                "route_id": {"type": "string"}, "repo_name": {"type": "string"}, "branch": {"type": "string"},
                "expected_commit": {"type": "string"}, "runner_relpath": {"type": "string"}, "mode": {"type": "string"},
                "session": {"type": "string"}, "require_gpu": {"type": "boolean", "default": True},
            },
            "additionalProperties": False,
        },
        "handler": tool_launch,
    },
    "convir_route_monitor": {
        "description": "Read-only monitor for one route. It reports tmux state, a bounded status tail, and closeout filenames only.",
        "inputSchema": {
            "type": "object",
            "required": ["route_id", "repo_name", "session"],
            "properties": {"route_id": {"type": "string"}, "repo_name": {"type": "string"}, "session": {"type": "string"}, "tail_lines": {"type": "integer", "minimum": 1, "maximum": 80, "default": 16}},
            "additionalProperties": False,
        },
        "handler": tool_monitor,
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
        return TOOLS[name]["handler"](params.get("arguments") or {})
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
