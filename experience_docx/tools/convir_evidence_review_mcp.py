#!/usr/bin/env python3
"""Read-only MCP facade for the commit-bound experiment evidence catalog."""

import base64
import hashlib
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import convir_evidence_catalog as catalog
import convir_evidence_cloud_inventory as inventory
import convirctl


SERVER_NAME = "convir-evidence-review"
SERVER_VERSION = "1.1.0"
WORKSPACE_ROOT_ENV = "CONVIR_EVIDENCE_LOCAL_WORKSPACE_ROOT"
DEFAULT_WORKSPACE_ROOT = "/home/ubuntu/workspace"
TRUSTED_REMOTE_NAME = "github"
TRUSTED_REMOTE_URLS = (
    "git@github.com:onenoober/ConvIR-B.git",
    "https://github.com/onenoober/ConvIR-B.git",
)
TRUSTED_MAIN_REF = "refs/remotes/github/main"
MAX_JSONRPC_RESPONSE_BYTES = 32 * 1024
MAX_REQUEST_ID_BYTES = 128
MAX_TOOL_RESULT_BYTES = MAX_JSONRPC_RESPONSE_BYTES
MAX_REMOTE_SCRIPT_BYTES = 256 * 1024
MAX_REMOTE_CAPTURE_BYTES = 64 * 1024
REMOTE_HOST = "convir-4090"
REMOTE_PYTHON = "/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python"
REMOTE_BASH = "/bin/bash"
REMOTE_TEMP_ROOT = "/sda/home/wangyuxin/ConvIR-B/runtime/convir-evidence-review"
SSH = "/usr/bin/ssh"
REMOTE_TIMEOUT_SECONDS = 90
SELF_PATH = Path(__file__).resolve()
SERVER_SOURCE_SHA256 = hashlib.sha256(SELF_PATH.read_bytes()).hexdigest()
CATALOG_SOURCE_SHA256 = hashlib.sha256(
    (SELF_PATH.parent / "convir_evidence_catalog.py").read_bytes()
).hexdigest()
TRANSPORT_SOURCE_SHA256 = hashlib.sha256(
    (SELF_PATH.parent / "convirctl.py").read_bytes()
).hexdigest()
INVENTORY_SOURCE_PATH = SELF_PATH.parent / "convir_evidence_cloud_inventory.py"
INVENTORY_SOURCE_BYTES = INVENTORY_SOURCE_PATH.read_bytes()
INVENTORY_SOURCE_SHA256 = hashlib.sha256(INVENTORY_SOURCE_BYTES).hexdigest()
MAX_REQUEST_ID_PLACEHOLDER = "x" * (MAX_REQUEST_ID_BYTES - 2)


class ReviewError(RuntimeError):
    def __init__(self, message, *, state="ARGUMENTS_INVALID", exit_code=2):
        super().__init__(message)
        self.state = state
        self.exit_code = exit_code


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def emit_envelope(value):
    raw = canonical_bytes(value) + b"\n"
    if len(raw) > MAX_JSONRPC_RESPONSE_BYTES:
        raw = canonical_bytes({
            "jsonrpc": "2.0",
            "id": None,
            "error": {
                "code": -32001,
                "message": "JSON-RPC response exceeded the 32 KiB transport budget",
            },
        }) + b"\n"
    sys.stdout.buffer.write(raw)
    sys.stdout.buffer.flush()


def bounded_error(exc):
    raw = str(exc).encode("utf-8", errors="replace")[:2048]
    return raw.decode("utf-8", errors="ignore")


def failure_value(operation, exc):
    if isinstance(exc, (
        ReviewError, catalog.CatalogError, convirctl.ControlError,
        inventory.InventoryError,
    )):
        state = exc.state
        exit_code = exc.exit_code
    elif isinstance(exc, OSError):
        state = "LOCAL_IO_FAILED"
        exit_code = 2
    else:
        state = "INTERNAL_REVIEW_ERROR"
        exit_code = 70
    return {
        "ok": False,
        "operation": operation,
        "state": state,
        "exit_code": exit_code,
        "error": bounded_error(exc),
        "scientific_completeness": "not_assessed",
        "excluded_sources": ["route_branches", "cloud_runtime", "result_contents"],
    }


def workspace_root():
    raw = os.environ.get(WORKSPACE_ROOT_ENV, DEFAULT_WORKSPACE_ROOT)
    root = Path(raw)
    if not root.is_absolute():
        raise ReviewError(f"{WORKSPACE_ROOT_ENV} must be absolute")
    try:
        root = root.resolve(strict=True)
    except OSError as exc:
        raise ReviewError(f"workspace root is unavailable: {root}") from exc
    if not root.is_dir():
        raise ReviewError(f"workspace root is not a directory: {root}")
    return root


def validate_repo(value):
    if not isinstance(value, str) or not value:
        raise ReviewError("local_repo must be a non-empty absolute path")
    candidate = Path(value)
    if not candidate.is_absolute():
        raise ReviewError("local_repo must be absolute")
    try:
        repo = candidate.resolve(strict=True)
    except OSError as exc:
        raise ReviewError("local_repo is unavailable") from exc
    try:
        repo.relative_to(workspace_root())
    except ValueError as exc:
        raise ReviewError("local_repo is outside the trusted workspace root") from exc
    if not repo.is_dir():
        raise ReviewError("local_repo is not a directory")
    completed = convirctl.run_argv(
        [convirctl.GIT, "-C", repo, "rev-parse", "--show-toplevel"],
        timeout=30,
        env=convirctl.git_environment(30),
    )
    if completed.returncode:
        raise ReviewError("local_repo is not a Git worktree")
    top = catalog.strict_text(completed.stdout, "Git worktree root").strip()
    try:
        resolved_top = Path(top).resolve(strict=True)
    except OSError as exc:
        raise ReviewError("Git worktree root is unavailable") from exc
    if resolved_top != repo:
        raise ReviewError("local_repo must be the Git worktree root")
    return repo


def trusted_main_identity(repo):
    try:
        remote_url = catalog.git_text(repo, "remote", "get-url", TRUSTED_REMOTE_NAME)
    except catalog.CatalogError as exc:
        raise ReviewError(
            "trusted GitHub remote is unavailable",
            state="GITHUB_REMOTE_UNAVAILABLE",
            exit_code=3,
        ) from exc
    if remote_url not in TRUSTED_REMOTE_URLS:
        raise ReviewError(
            "github remote does not identify the trusted repository",
            state="GITHUB_REMOTE_UNTRUSTED",
            exit_code=3,
        )
    try:
        tip = catalog.git_text(
            repo, "rev-parse", "--verify", f"{TRUSTED_MAIN_REF}^{{commit}}"
        )
    except catalog.CatalogError as exc:
        raise ReviewError(
            "trusted GitHub main ref is unavailable",
            state="GITHUB_MAIN_UNAVAILABLE",
            exit_code=3,
        ) from exc
    if not catalog.SHA40.fullmatch(tip):
        raise ReviewError(
            "trusted GitHub main ref did not resolve to a commit",
            state="GITHUB_MAIN_INVALID",
            exit_code=3,
        )
    return remote_url, tip


def require_snapshot_commit(value):
    if not isinstance(value, str) or not catalog.SHA40.fullmatch(value):
        raise ReviewError("snapshot_commit must be 40 lowercase hexadecimal characters")
    return value


def require_main_history_commit(repo, value):
    commit = require_snapshot_commit(value)
    remote_url, tip = trusted_main_identity(repo)
    try:
        resolved = catalog.git_text(
            repo, "rev-parse", "--verify", f"{commit}^{{commit}}"
        )
    except catalog.CatalogError as exc:
        raise ReviewError(
            "snapshot_commit is unavailable",
            state="SNAPSHOT_INVALID",
            exit_code=3,
        ) from exc
    if resolved != commit:
        raise ReviewError(
            "snapshot_commit did not resolve exactly",
            state="SNAPSHOT_INVALID",
            exit_code=3,
        )
    completed = convirctl.run_argv(
        [convirctl.GIT, "-C", repo, "merge-base", "--is-ancestor", commit, tip],
        timeout=30,
        env=convirctl.git_environment(30),
    )
    if completed.returncode == 1:
        raise ReviewError(
            "snapshot_commit is outside trusted GitHub main history",
            state="SNAPSHOT_OUTSIDE_GITHUB_MAIN",
            exit_code=3,
        )
    if completed.returncode:
        raise ReviewError(
            "GitHub main ancestry check failed",
            state="GIT_READ_FAILED",
            exit_code=3,
        )
    return commit, remote_url, tip


def add_github_identity(value, remote_url, tip):
    value.update({
        "trusted_remote": TRUSTED_REMOTE_NAME,
        "trusted_remote_url": remote_url,
        "trusted_ref": TRUSTED_MAIN_REF,
        "trusted_ref_tip": tip,
        "ref_freshness": "not_assessed",
        "git_mutations_performed": False,
    })
    return value


def mcp_result(value):
    serialized = canonical_bytes(value).decode("utf-8")
    return {
        "content": [{
            "type": "text",
            "text": serialized,
        }],
        "structuredContent": value,
        "isError": value.get("ok") is not True,
    }


def result_fits(value):
    envelope = {
        "jsonrpc": "2.0",
        "id": MAX_REQUEST_ID_PLACEHOLDER,
        "result": value,
    }
    return len(canonical_bytes(envelope)) + 1 <= MAX_JSONRPC_RESPONSE_BYTES


def bounded_mcp_result(value):
    result = mcp_result(value)
    if result_fits(result):
        return result
    fallback = failure_value(
        value.get("operation", "catalog"),
        ReviewError(
            "MCP result exceeded the 32 KiB transport budget",
            state="RESPONSE_TOO_LARGE",
            exit_code=3,
        ),
    )
    return mcp_result(fallback)


def tool_catalog_summary(args):
    operation = "catalog-summary"
    try:
        repo = validate_repo(args.get("local_repo"))
        remote_url, commit = trusted_main_identity(repo)
        value = add_github_identity(
            catalog.summary_response(catalog.load_catalog(repo, commit)),
            remote_url,
            commit,
        )
        return bounded_mcp_result(value)
    except Exception as exc:
        return bounded_mcp_result(failure_value(operation, exc))


def query_arguments(args):
    coverage = args.get("coverage", "all")
    if coverage not in {"indexed", "unindexed", "all"}:
        raise ReviewError("coverage must be indexed, unindexed, or all")
    terms = args.get("terms", [])
    if not isinstance(terms, list) or any(not isinstance(term, str) for term in terms):
        raise ReviewError("terms must be an array of strings")
    if len(terms) > 8:
        raise ReviewError("terms accepts at most 8 items")
    cursor = args.get("cursor")
    if cursor is not None and not isinstance(cursor, str):
        raise ReviewError("cursor must be text")
    limit = args.get("limit", 20)
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 100:
        raise ReviewError("limit must be an integer in [1, 100]")
    return coverage, terms, cursor, limit


def tool_catalog_query(args):
    operation = "catalog-entries"
    try:
        repo = validate_repo(args.get("local_repo"))
        commit, remote_url, tip = require_main_history_commit(
            repo, args.get("snapshot_commit")
        )
        coverage, terms, cursor, requested_limit = query_arguments(args)
        loaded = catalog.load_catalog(repo, commit)
        effective_limit = requested_limit
        while True:
            value = catalog.entries_response(
                loaded,
                SimpleNamespace(
                    coverage=coverage,
                    term=terms,
                    cursor=cursor,
                    limit=effective_limit,
                ),
            )
            add_github_identity(value, remote_url, tip)
            result = mcp_result(value)
            if result_fits(result):
                return result
            returned = value.get("returned_count", 0)
            if returned <= 1:
                raise ReviewError(
                    "one catalog entry exceeds the MCP response budget",
                    state="ENTRY_TOO_LARGE",
                    exit_code=3,
                )
            effective_limit = min(effective_limit - 1, returned - 1)
    except Exception as exc:
        return bounded_mcp_result(failure_value(operation, exc))


def _remote_script(request):
    source = base64.encodebytes(INVENTORY_SOURCE_BYTES).decode("ascii").rstrip()
    request_raw = canonical_bytes(request)
    encoded_request = base64.encodebytes(request_raw).decode("ascii").rstrip()
    script = f"""#!/usr/bin/env bash
set -euo pipefail
umask 077
temp_root={REMOTE_TEMP_ROOT!r}
/usr/bin/mkdir -p -- "$temp_root"
resolved_root=$(/usr/bin/readlink -f -- "$temp_root")
test "$resolved_root" = "$temp_root"
task_dir=$(/usr/bin/mktemp -d -- "$temp_root/request.XXXXXXXX")
resolved_task=$(/usr/bin/readlink -f -- "$task_dir")
case "$resolved_task" in
  "$temp_root"/request.*) ;;
  *) exit 71 ;;
esac
task_dir="$resolved_task"
cleanup() {{
  case "$task_dir" in
    "$temp_root"/request.*) /usr/bin/rm -rf -- "$task_dir" ;;
    *) exit 72 ;;
  esac
}}
trap cleanup EXIT
/usr/bin/base64 --decode > "$task_dir/worker.py" <<'__CONVIR_WORKER_SOURCE__'
{source}
__CONVIR_WORKER_SOURCE__
/usr/bin/base64 --decode > "$task_dir/request.json" <<'__CONVIR_WORKER_REQUEST__'
{encoded_request}
__CONVIR_WORKER_REQUEST__
{REMOTE_PYTHON} "$task_dir/worker.py" --remote-worker < "$task_dir/request.json"
"""
    encoded = script.encode("utf-8")
    if len(encoded) > MAX_REMOTE_SCRIPT_BYTES:
        raise ReviewError(
            "fixed remote inventory script exceeded its bound",
            state="RESPONSE_TOO_LARGE", exit_code=3,
        )
    return encoded


def _run_fixed_remote(request):
    script = _remote_script(request)
    argv = [
        SSH, "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=30",
        REMOTE_HOST, REMOTE_BASH, "-s", "--",
    ]
    try:
        process = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise ReviewError(
            "fixed cloud transport could not start",
            state="CLOUD_UNAVAILABLE", exit_code=3,
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
        return_code = process.wait(timeout=REMOTE_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as exc:
        process.kill()
        process.wait()
        for thread in threads:
            thread.join(timeout=5)
        raise ReviewError(
            "fixed cloud inventory timed out; remote state is unknown",
            state="REMOTE_STATE_UNKNOWN", exit_code=3,
        ) from exc
    for thread in threads:
        thread.join(timeout=5)
    if any(thread.is_alive() for thread in threads) or thread_errors:
        raise ReviewError(
            "fixed cloud transport streams did not close cleanly",
            state="REMOTE_STATE_UNKNOWN", exit_code=3,
        )
    if len(stdout) > MAX_REMOTE_CAPTURE_BYTES or len(stderr) > MAX_REMOTE_CAPTURE_BYTES:
        raise ReviewError(
            "fixed cloud transport output exceeded its bound",
            state="RESPONSE_TOO_LARGE", exit_code=3,
        )
    if return_code:
        raise ReviewError(
            "fixed cloud inventory command failed",
            state="CLOUD_UNAVAILABLE", exit_code=3,
        )
    try:
        response = json.loads(bytes(stdout).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReviewError(
            "fixed cloud inventory returned an invalid response",
            state="REMOTE_STATE_UNKNOWN", exit_code=3,
        ) from exc
    if not isinstance(response, dict) or set(response) != {
        "worker_source_sha256", "result"
    } or response["worker_source_sha256"] != INVENTORY_SOURCE_SHA256 \
            or not isinstance(response["result"], dict):
        raise ReviewError(
            "fixed cloud worker identity or response differs",
            state="IDENTITY_CONFLICT", exit_code=3,
        )
    return response["result"]


def _binding_identity(binding):
    return {
        key: binding.get(key) for key in (
            "snapshot_commit", "catalog_sha256", "terminal_index_sha256",
            "terminal_record_sha256", "route_id", "operation_id", "run_id",
            "output_id", "mode", "session", "route_commit", "manifest_sha256",
            "runtime_spec_sha256", "closeout_sha256", "runner_sha256",
        )
    }


def _prepare_cloud_binding(args):
    repo = validate_repo(args.get("local_repo"))
    commit, remote_url, tip = require_main_history_commit(
        repo, args.get("snapshot_commit")
    )
    binding = inventory.prepare_terminal_binding(
        repo,
        commit,
        args.get("catalog_sha256"),
        args.get("terminal_record_sha256"),
    )
    return binding, remote_url, tip


def _execute_inventory(binding, query=None):
    root_binding = binding.get("eligible") is True
    if binding.get("eligible") is not True \
            or binding.get("raw_inventory_authorized") is not True:
        adapter_root = binding.get("run_root", "/")
        full = inventory._scan_adapter_root(
            binding,
            adapter_root,
            root_binding_enforced=root_binding,
        )
        if query is None or full.get("ok") is not True:
            return inventory.inventory_summary(full)
        return inventory.inventory_query_page(full, **query)
    request = {
        "schema_version": 1,
        "operation": "summary" if query is None else "query",
        "binding": binding,
        "adapter_root": binding["run_root"],
        "root_binding_enforced": True,
        "expected_session": binding["session"],
        "query": query,
    }
    result = _run_fixed_remote(request)
    if result.get("ok") is not True:
        return result
    if query is None:
        if result.get("identity") != _binding_identity(binding) \
                or result.get("declared_run_root") != binding["run_root"] \
                or result.get("root_binding_enforced") is not True:
            raise ReviewError(
                "cloud inventory summary identity differs",
                state="IDENTITY_CONFLICT", exit_code=3,
            )
    else:
        if result.get("snapshot_commit") != binding["snapshot_commit"] \
                or result.get("terminal_record_sha256") \
                != binding["terminal_record_sha256"] \
                or result.get("inventory_sha256") != query["inventory_sha256"]:
            raise ReviewError(
                "cloud inventory query identity differs",
                state="IDENTITY_CONFLICT", exit_code=3,
            )
    return result


def tool_cloud_inventory_summary(args):
    operation = "cloud-inventory-summary"
    try:
        binding, remote_url, tip = _prepare_cloud_binding(args)
        value = _execute_inventory(binding)
        add_github_identity(value, remote_url, tip)
        return bounded_mcp_result(value)
    except Exception as exc:
        return bounded_mcp_result(failure_value(operation, exc))


def _cloud_query_arguments(args, binding):
    inventory_sha256 = args.get("inventory_sha256")
    if not isinstance(inventory_sha256, str) \
            or not inventory.SHA256.fullmatch(inventory_sha256):
        raise ReviewError("inventory_sha256 has an invalid SHA identity")
    states, terms, limit = inventory.normalize_query_arguments(
        args.get("reconciliation_states"), args.get("terms"), args.get("limit", 20)
    )
    cursor = args.get("cursor")
    if cursor is not None and not isinstance(cursor, str):
        raise ReviewError("cursor must be text")
    identity = _binding_identity(binding)
    query_sha256 = inventory.inventory_query_sha256(
        identity, inventory_sha256, states, terms
    )
    offset = 0
    if cursor is not None:
        try:
            decoded = convirctl.decode_repo_cursor(
                cursor, inventory.CURSOR_OPERATION, query_sha256
            )
        except convirctl.ControlError as exc:
            raise ReviewError(
                str(exc), state=exc.state, exit_code=exc.exit_code
            ) from exc
        if decoded["commit"] != binding["snapshot_commit"] \
                or decoded["object_id"] != inventory_sha256[:40]:
            raise ReviewError(
                "inventory cursor identity differs",
                state="REPO_CURSOR_IDENTITY_MISMATCH", exit_code=3,
            )
        offset = decoded["position"]
    return {
        "inventory_sha256": inventory_sha256,
        "reconciliation_states": states,
        "terms": terms,
        "offset": offset,
        "limit": limit,
    }, query_sha256


def _bounded_cloud_query(value, binding, query_sha256, remote_url, tip):
    page = dict(value)
    entries = list(page.get("entries", []))
    if not isinstance(page.get("offset"), int) \
            or not isinstance(page.get("total_count"), int) \
            or page.get("query_sha256") != query_sha256:
        raise ReviewError(
            "cloud inventory page contract differs",
            state="IDENTITY_CONFLICT", exit_code=3,
        )
    while True:
        end = page["offset"] + len(entries)
        complete = end == page["total_count"]
        page.update({
            "entries": entries,
            "returned_count": len(entries),
            "page_sha256": inventory.canonical_sha256(entries),
            "complete": complete,
            "has_more": not complete,
            "next_cursor": None if complete else convirctl.encode_repo_cursor(
                inventory.CURSOR_OPERATION,
                binding["snapshot_commit"],
                query_sha256,
                end,
                page["inventory_sha256"][:40],
            ),
        })
        page.pop("next_offset", None)
        add_github_identity(page, remote_url, tip)
        result = mcp_result(page)
        if result_fits(result):
            return result
        if len(entries) <= 1:
            raise ReviewError(
                "one inventory entry exceeds the MCP response budget",
                state="ENTRY_TOO_LARGE", exit_code=3,
            )
        entries.pop()


def tool_cloud_inventory_query(args):
    operation = "cloud-inventory-query"
    try:
        binding, remote_url, tip = _prepare_cloud_binding(args)
        query, query_sha256 = _cloud_query_arguments(args, binding)
        value = _execute_inventory(binding, query)
        if value.get("ok") is not True:
            add_github_identity(value, remote_url, tip)
            return bounded_mcp_result(value)
        return _bounded_cloud_query(
            value, binding, query_sha256, remote_url, tip
        )
    except Exception as exc:
        return bounded_mcp_result(failure_value(operation, exc))


OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["ok", "operation", "state", "exit_code"],
    "properties": {
        "ok": {"type": "boolean"},
        "operation": {"type": "string"},
        "state": {"type": "string"},
        "exit_code": {"type": "integer"},
    },
    "additionalProperties": True,
}


TOOLS = {
    "convir_evidence_catalog_summary": {
        "description": (
            "Freeze trusted github/main to an immutable commit and return only "
            "the compact GitHub evidence-catalog identity and coverage summary."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["local_repo"],
            "properties": {
                "local_repo": {"type": "string"},
            },
            "additionalProperties": False,
        },
        "outputSchema": OUTPUT_SCHEMA,
        "handler": tool_catalog_summary,
    },
    "convir_evidence_catalog_query": {
        "description": (
            "Query one immutable GitHub evidence catalog through a bounded, "
            "identity-bound cursor without reading result contents."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["local_repo", "snapshot_commit"],
            "properties": {
                "local_repo": {"type": "string"},
                "snapshot_commit": {"type": "string", "pattern": "^[0-9a-f]{40}$"},
                "coverage": {
                    "enum": ["indexed", "unindexed", "all"],
                    "default": "all",
                },
                "terms": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 8,
                },
                "cursor": {"type": "string"},
                "limit": {
                    "type": "integer", "minimum": 1, "maximum": 100,
                    "default": 20,
                },
            },
            "additionalProperties": False,
        },
        "outputSchema": OUTPUT_SCHEMA,
        "handler": tool_catalog_query,
    },
    "convir_evidence_cloud_inventory_summary": {
        "description": (
            "Verify one immutable schema-2 terminal binding, then return a "
            "bounded reconciliation summary for its fixed cloud run root."
        ),
        "inputSchema": {
            "type": "object",
            "required": [
                "local_repo", "snapshot_commit", "catalog_sha256",
                "terminal_record_sha256",
            ],
            "properties": {
                "local_repo": {"type": "string"},
                "snapshot_commit": {"type": "string", "pattern": "^[0-9a-f]{40}$"},
                "catalog_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                "terminal_record_sha256": {
                    "type": "string", "pattern": "^[0-9a-f]{64}$",
                },
            },
            "additionalProperties": False,
        },
        "outputSchema": OUTPUT_SCHEMA,
        "handler": tool_cloud_inventory_summary,
    },
    "convir_evidence_cloud_inventory_query": {
        "description": (
            "Rescan one immutable schema-2 terminal run root and return a bounded "
            "identity-bound page only when its inventory SHA-256 is unchanged."
        ),
        "inputSchema": {
            "type": "object",
            "required": [
                "local_repo", "snapshot_commit", "catalog_sha256",
                "terminal_record_sha256", "inventory_sha256",
            ],
            "properties": {
                "local_repo": {"type": "string"},
                "snapshot_commit": {"type": "string", "pattern": "^[0-9a-f]{40}$"},
                "catalog_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                "terminal_record_sha256": {
                    "type": "string", "pattern": "^[0-9a-f]{64}$",
                },
                "inventory_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                "reconciliation_states": {
                    "type": "array",
                    "items": {"enum": list(inventory.RECONCILIATION_STATES)},
                    "maxItems": len(inventory.RECONCILIATION_STATES),
                },
                "terms": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1, "maxLength": 128},
                    "maxItems": 8,
                },
                "cursor": {"type": "string"},
                "limit": {
                    "type": "integer", "minimum": 1, "maximum": 100,
                    "default": 20,
                },
            },
            "additionalProperties": False,
        },
        "outputSchema": OUTPUT_SCHEMA,
        "handler": tool_cloud_inventory_query,
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
                "name": SERVER_NAME,
                "version": SERVER_VERSION,
                "sourceSha256": SERVER_SOURCE_SHA256,
                "catalogSourceSha256": CATALOG_SOURCE_SHA256,
                "inventorySourceSha256": INVENTORY_SOURCE_SHA256,
                "transportSourceSha256": TRANSPORT_SOURCE_SHA256,
            },
        }
    if method == "ping":
        return {}
    if method == "tools/list":
        return {"tools": [
            {
                "name": name,
                "description": item["description"],
                "inputSchema": item["inputSchema"],
                "outputSchema": item["outputSchema"],
            }
            for name, item in TOOLS.items()
        ]}
    if method == "tools/call":
        name = params.get("name")
        args = params.get("arguments")
        if name not in TOOLS or not isinstance(args, dict):
            raise ReviewError("invalid MCP tool call")
        schema = TOOLS[name]["inputSchema"]
        unknown = set(args) - set(schema["properties"])
        missing = set(schema["required"]) - set(args)
        if unknown or missing:
            raise ReviewError(
                f"tool argument mismatch unknown={sorted(unknown)} "
                f"missing={sorted(missing)}"
            )
        return TOOLS[name]["handler"](args)
    raise ReviewError(f"unsupported method: {method}")


def require_request_id(value):
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ReviewError("MCP request id must be a string or integer")
    if len(canonical_bytes(value)) > MAX_REQUEST_ID_BYTES:
        raise ReviewError("JSON-encoded MCP request id exceeds 128 bytes")
    return value


def main():
    for line in sys.stdin:
        request_id = None
        response_expected = False
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ReviewError("MCP request must be an object")
            response_expected = "id" in request and request["id"] is not None
            if response_expected:
                request_id = require_request_id(request["id"])
            result = handle(request)
            if response_expected:
                emit_envelope({"jsonrpc": "2.0", "id": request_id, "result": result})
        except Exception as exc:
            if response_expected:
                emit_envelope({
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32000, "message": bounded_error(exc)},
                })


if __name__ == "__main__":
    main()
