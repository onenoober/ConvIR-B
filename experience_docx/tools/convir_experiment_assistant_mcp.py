#!/usr/bin/env python3
"""Six-tool phase-2 stdio MCP for the compact experiment assistant."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from experiment_assistant_contract import PUBLIC_TOOL_NAMES
from experiment_assistant_runner import BASE_CAPABILITIES, BackendError, ExperimentBackend


SERVER_NAME = "convir-experiment-assistant"
SERVER_VERSION = "0.3.0-candidate"
PROTOCOL_SCHEMA_VERSION = 1
MAX_REQUEST_BYTES = 256 * 1024
MAX_RESPONSE_BYTES = 64 * 1024


def _object_schema(
    properties: dict[str, Any], required: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


TOOL_DEFINITIONS = [
    {
        "name": "experiment_start",
        "description": "Snapshot current code and start one new experiment attempt.",
        "inputSchema": _object_schema(
            {
                "local_repo": {"type": "string", "minLength": 1, "maxLength": 1024},
                "contract": {"type": "object"},
            },
            ["local_repo", "contract"],
        ),
    },
    {
        "name": "experiment_status",
        "description": "Read the current lifecycle state and latest compact result.",
        "inputSchema": _object_schema(
            {"experiment_id": {"type": "string", "minLength": 1, "maxLength": 128}},
            ["experiment_id"],
        ),
    },
    {
        "name": "experiment_cancel",
        "description": "Cancel the exact active attempt using stored process identity.",
        "inputSchema": _object_schema(
            {"experiment_id": {"type": "string", "minLength": 1, "maxLength": 128}},
            ["experiment_id"],
        ),
    },
    {
        "name": "experiment_repair",
        "description": "Start a bounded same-experiment repair after engineering failure.",
        "inputSchema": _object_schema(
            {
                "experiment_id": {"type": "string", "minLength": 1, "maxLength": 128},
                "contract": {"type": "object"},
                "operator_confirmed": {"type": "boolean", "default": False},
            },
            ["experiment_id"],
        ),
    },
    {
        "name": "experiment_get",
        "description": "Read one compact cloud experiment record and attempt history.",
        "inputSchema": _object_schema(
            {
                "experiment_id": {"type": "string", "minLength": 1, "maxLength": 128},
                "view": {"type": "string", "enum": ["summary", "full"], "default": "summary"},
            },
            ["experiment_id"],
        ),
    },
    {
        "name": "experiment_search",
        "description": "Search compact records or compare a bounded experiment-id list.",
        "inputSchema": _object_schema(
            {
                "query": {"type": "string", "maxLength": 256},
                "states": {"type": "array", "items": {"type": "string"}, "maxItems": 16},
                "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 20},
                "compare_experiment_ids": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1, "maxLength": 128},
                    "maxItems": 16,
                },
            }
        ),
    },
]


if tuple(item["name"] for item in TOOL_DEFINITIONS) != PUBLIC_TOOL_NAMES:
    raise RuntimeError("public MCP surface drifted from the compact contract")


def _source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _diagnostics() -> dict[str, Any]:
    capabilities = set(BASE_CAPABILITIES)
    if os.environ.get("CONVIR_EXPERIMENT_DATASET_REGISTRY"):
        capabilities.update({
            "dataset_registry_resolution", "explicit_protected_data_access",
        })
    if os.environ.get("CONVIR_EXPERIMENT_ARCHIVE_ENABLED") == "1":
        capabilities.add("automatic_result_archive")
    return {
        "server_name": SERVER_NAME,
        "server_version": SERVER_VERSION,
        "protocol_schema_version": PROTOCOL_SCHEMA_VERSION,
        "server_source_sha256": _source_sha256(),
        "capabilities": sorted(capabilities),
        "adoption_state": "PHASE_3_CANDIDATE_NOT_REGISTERED",
    }


def _backend(tool_name: str) -> ExperimentBackend:
    return ExperimentBackend.from_environment(
        load_dataset_registry=tool_name in {"experiment_start", "experiment_repair"},
        load_archive_store=tool_name != "experiment_cancel",
    )


def _require_arguments(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise BackendError("tool arguments must be an object")
    return value


def call_tool(name: str, arguments: Any) -> dict[str, Any]:
    args = _require_arguments(arguments)
    backend = _backend(name)
    if name == "experiment_start":
        if set(args) != {"local_repo", "contract"}:
            raise BackendError("experiment_start requires only local_repo and contract")
        if not isinstance(args["local_repo"], str):
            raise BackendError("local_repo must be text")
        result = backend.start(args["local_repo"], args["contract"])
    elif name == "experiment_status":
        if set(args) != {"experiment_id"}:
            raise BackendError("experiment_status requires only experiment_id")
        result = backend.status(args["experiment_id"])
    elif name == "experiment_cancel":
        if set(args) != {"experiment_id"}:
            raise BackendError("experiment_cancel requires only experiment_id")
        result = backend.cancel(args["experiment_id"])
    elif name == "experiment_repair":
        if not set(args) <= {"experiment_id", "contract", "operator_confirmed"} \
                or "experiment_id" not in args:
            raise BackendError("experiment_repair arguments are invalid")
        if not isinstance(args.get("operator_confirmed", False), bool):
            raise BackendError("operator_confirmed must be boolean")
        result = backend.repair(
            args["experiment_id"],
            contract=args.get("contract"),
            operator_confirmed=args.get("operator_confirmed", False),
        )
    elif name == "experiment_get":
        if not set(args) <= {"experiment_id", "view"} or "experiment_id" not in args:
            raise BackendError("experiment_get arguments are invalid")
        result = backend.get(args["experiment_id"], view=args.get("view", "summary"))
    elif name == "experiment_search":
        if not set(args) <= {"query", "states", "limit", "compare_experiment_ids"}:
            raise BackendError("experiment_search arguments are invalid")
        result = backend.search(
            query=args.get("query"),
            states=args.get("states"),
            limit=args.get("limit", 20),
            compare_experiment_ids=args.get("compare_experiment_ids"),
        )
    else:
        raise BackendError(f"unknown experiment-assistant tool: {name}")
    return {**_diagnostics(), **result}


def _tool_result(value: dict[str, Any], *, is_error: bool = False) -> dict[str, Any]:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return {
        "content": [{"type": "text", "text": text}],
        "structuredContent": value,
        "isError": is_error,
    }


def handle_request(request: dict[str, Any]) -> dict[str, Any] | None:
    method = request.get("method")
    request_id = request.get("id")
    if method == "notifications/initialized":
        return None
    if method == "initialize":
        params = request.get("params", {})
        if not isinstance(params, dict):
            raise BackendError("initialize params must be an object")
        result = {
            "protocolVersion": params.get("protocolVersion", "2024-11-05"),
            "capabilities": {"tools": {}},
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            "instructions": (
                "Phase-3 candidate: use six intent-level tools; runtime is cloud-only "
                "and the candidate is not registered as the project default."
            ),
        }
        return {"jsonrpc": "2.0", "id": request_id, "result": result}
    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {"tools": TOOL_DEFINITIONS},
        }
    if method == "tools/call":
        params = request.get("params")
        if not isinstance(params, dict):
            raise BackendError("tools/call params must be an object")
        try:
            value = call_tool(params.get("name"), params.get("arguments", {}))
            result = _tool_result(value)
        except (BackendError, OSError, TypeError, ValueError) as exc:
            result = _tool_result(
                {**_diagnostics(), "ok": False, "error": str(exc)[:4096]},
                is_error=True,
            )
        return {"jsonrpc": "2.0", "id": request_id, "result": result}
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": -32601, "message": "Method not found"},
    }


def main() -> int:
    for raw_line in sys.stdin.buffer:
        if len(raw_line) > MAX_REQUEST_BYTES:
            continue
        try:
            request = json.loads(raw_line)
            if not isinstance(request, dict):
                raise ValueError("request must be an object")
            response = handle_request(request)
            if response is None:
                continue
        except (json.JSONDecodeError, TypeError, ValueError, BackendError) as exc:
            response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32600, "message": str(exc)[:1024]},
            }
        encoded = canonical_response = (
            json.dumps(response, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
            + "\n"
        ).encode("utf-8")
        if len(canonical_response) > MAX_RESPONSE_BYTES:
            fallback = {
                "jsonrpc": "2.0",
                "id": response.get("id"),
                "error": {"code": -32603, "message": "bounded response size exceeded"},
            }
            encoded = (
                json.dumps(fallback, sort_keys=True, separators=(",", ":")) + "\n"
            ).encode("utf-8")
        sys.stdout.buffer.write(encoded)
        sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
