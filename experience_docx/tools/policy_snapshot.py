#!/usr/bin/env python3
"""Build and verify a compact deterministic index of authoritative rules."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Callable


class PolicySnapshotError(RuntimeError):
    pass


SNAPSHOT_RELPATH = "experience_docx/AI_POLICY_SNAPSHOT.json"
SHA40 = re.compile(r"^[0-9a-f]{40}$")
POLICY_SOURCES = (
    "AGENTS.md",
    "experience_docx/RULE_COMPATIBILITY.json",
    "experience_docx/SCIENCE_FASTPATH.md",
    "experience_docx/CONVIR_EVIDENCE_REVIEW.md",
    "experience_docx/ROUTE_READY_FASTPATH.md",
    "experience_docx/MODEL_RUN_OPERATIONS_PROTOCOL.md",
    "experience_docx/EXPERIMENT_GOVERNANCE_PROTOCOL.md",
    "experience_docx/COMMAND_RELIABILITY_PROTOCOL.md",
    "experience_docx/CONVIR_OPS_MCP.md",
    "experience_docx/BRANCH_EXPERIMENT_SYNC_PROTOCOL.md",
)
CHANGE_ROUTES = {
    "read_only_audit": [
        "AGENTS.md", "experience_docx/SCIENCE_FASTPATH.md",
        "experience_docx/CONVIR_EVIDENCE_REVIEW.md",
        "experience_docx/EXPERIMENT_GOVERNANCE_PROTOCOL.md",
    ],
    "route_authoring": [
        "AGENTS.md", "experience_docx/ROUTE_READY_FASTPATH.md",
        "experience_docx/EXPERIMENT_GOVERNANCE_PROTOCOL.md",
    ],
    "model_or_training": [
        "AGENTS.md", "experience_docx/MODEL_RUN_OPERATIONS_PROTOCOL.md",
        "experience_docx/EXPERIMENT_GOVERNANCE_PROTOCOL.md",
    ],
    "cloud_control": [
        "AGENTS.md", "experience_docx/COMMAND_RELIABILITY_PROTOCOL.md",
        "experience_docx/CONVIR_OPS_MCP.md",
    ],
    "evidence_archive": [
        "AGENTS.md", "experience_docx/BRANCH_EXPERIMENT_SYNC_PROTOCOL.md",
        "experience_docx/SCIENCE_FASTPATH.md",
    ],
    "governance_change": list(POLICY_SOURCES),
}
SOURCE_OF_TRUTH_ORDER = [
    {
        "step": "project_authority",
        "source": "github_main",
        "required_action": "establish_live_main_then_read_exact_project_snapshot",
        "owns": [
            "current_rules",
            "compact_terminal_evidence",
            "terminal_authorization",
        ],
    },
    {
        "step": "route_target",
        "source": "github_main_terminal_or_github_route_branch",
        "required_action": "bind_archived_route_from_terminal_index_or_new_route_from_manifest_and_program_lineage",
        "forbidden_uses": [
            "infer_metric_from_manifest",
            "infer_verdict_from_manifest",
            "infer_authorization_from_directory_or_chat",
        ],
    },
    {
        "step": "local_write_binding",
        "source": "local_route_worktree",
        "allowed_uses": [
            "bind_branch_head_route_id_before_write",
            "check_worktree_safety",
        ],
        "forbidden_uses": [
            "block_github_terminal_read_because_of_local_mismatch",
            "infer_metric",
            "infer_verdict",
            "infer_terminal",
            "infer_authorization",
            "infer_completed_workload",
        ],
    },
    {
        "step": "runtime_detail",
        "source": "convir_4090_or_receipt_bound_mcp",
        "required_action": "read_only_after_identity_and_github_binding",
        "owns": ["raw_runtime_state", "detailed_outputs"],
    },
]


def json_bytes(value) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode("utf-8")


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def build_snapshot(*, rules_commit: str, read_bytes: Callable[[str], bytes]) -> dict:
    if not isinstance(rules_commit, str) or not SHA40.fullmatch(rules_commit):
        raise PolicySnapshotError("rules_commit must be 40 lowercase hex")
    sources = {}
    bundle = hashlib.sha256()
    for relpath in POLICY_SOURCES:
        try:
            raw = read_bytes(relpath)
        except (FileNotFoundError, KeyError) as exc:
            raise PolicySnapshotError(f"policy source is missing: {relpath}") from exc
        if not isinstance(raw, bytes) or not raw:
            raise PolicySnapshotError(f"policy source is empty or non-bytes: {relpath}")
        digest = sha256(raw)
        sources[relpath] = {"sha256": digest, "bytes": len(raw)}
        bundle.update(relpath.encode() + b"\0" + raw + b"\0")
    routes = {
        change: {
            "read_full": paths,
            "snapshot_is_authority": False,
        }
        for change, paths in sorted(CHANGE_ROUTES.items())
    }
    return {
        "schema_version": 1,
        "rules_commit": rules_commit,
        "rules_bundle_sha256": bundle.hexdigest(),
        "snapshot_role": "deterministic_read_index_not_policy_authority",
        "default_runtime_host": "convir-4090",
        "local_runtime_allowed": False,
        "workflow": ["SNAPSHOT", "CONTRACT", "EXECUTE", "DECIDE", "ARCHIVE"],
        "source_of_truth_order": SOURCE_OF_TRUTH_ORDER,
        "route_mechanisms": ["adjacent", "orthogonal", "reopen"],
        "protected_roles_fail_closed": ["confirmation", "canary", "locked_test"],
        "full_rule_read_required_when": [
            "the change class route lists the file",
            "the snapshot hash differs from the repository source",
            "the request changes governance, protected data, or scientific authorization",
            "rules conflict or the change class is unknown",
        ],
        "change_routes": routes,
        "sources": sources,
    }


def verify_snapshot(value, *, read_bytes: Callable[[str], bytes]) -> dict:
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise PolicySnapshotError("snapshot schema is invalid")
    expected = build_snapshot(rules_commit=value.get("rules_commit"), read_bytes=read_bytes)
    if value != expected:
        raise PolicySnapshotError("policy snapshot drifted from authoritative sources")
    return expected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--rules-commit", required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true")
    action.add_argument("--check", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve()
    read = lambda relpath: (repo / relpath).read_bytes()
    expected = build_snapshot(rules_commit=args.rules_commit, read_bytes=read)
    path = repo / SNAPSHOT_RELPATH
    if args.write:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(json_bytes(expected))
        print(json.dumps({"status": "POLICY_SNAPSHOT_WRITTEN", "sha256": sha256(json_bytes(expected))}))
        return
    observed = json.loads(path.read_bytes())
    verify_snapshot(observed, read_bytes=read)
    print(json.dumps({"status": "POLICY_SNAPSHOT_OK", "sha256": sha256(path.read_bytes())}))


if __name__ == "__main__":
    main()
