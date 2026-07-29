#!/usr/bin/env python3
"""Read-only MCP facade for the commit-bound experiment evidence catalog."""

import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import convir_evidence_catalog as catalog
import convirctl


SERVER_NAME = "convir-evidence-review"
SERVER_VERSION = "1.0.0"
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
MAX_TOOL_RESULT_BYTES = MAX_JSONRPC_RESPONSE_BYTES - 512
SELF_PATH = Path(__file__).resolve()
SERVER_SOURCE_SHA256 = hashlib.sha256(SELF_PATH.read_bytes()).hexdigest()
CATALOG_SOURCE_SHA256 = hashlib.sha256(
    (SELF_PATH.parent / "convir_evidence_catalog.py").read_bytes()
).hexdigest()
TRANSPORT_SOURCE_SHA256 = hashlib.sha256(
    (SELF_PATH.parent / "convirctl.py").read_bytes()
).hexdigest()


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
    if isinstance(exc, (ReviewError, catalog.CatalogError, convirctl.ControlError)):
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
    return len(canonical_bytes(value)) <= MAX_TOOL_RESULT_BYTES


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
