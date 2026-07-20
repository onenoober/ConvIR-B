#!/usr/bin/env bash
set -euo pipefail

on_error() {
  rc=$?
  printf 'CONVIR_OPS_V5_RUNTIME_ACTIVATION_FAILED line=%s command=%q rc=%s\n' "$1" "$2" "$rc" >&2
  exit "$rc"
}
trap 'on_error "$LINENO" "$BASH_COMMAND"' ERR

branch=codex/convir-ops-v5-final-slim-20260720
control_plane_commit=31cc862f5107106dad8de266299b1bfea0b7a376
github=git@github.com:onenoober/ConvIR-B.git
seed=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-official-arch-anchor
python=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
evidence_root=/sda/home/wangyuxin/ConvIR-B/runs/convir_ops_v5_final_slim_acceptance_20260720
route_id=haze4k_v5_r16_s3_domain_matched_action_ceiling_20260720
work=$(mktemp -d /tmp/convir-ops-v5-runtime-activation.XXXXXX)
trap 'rm -rf -- "$work"' EXIT

printf 'CONVIR_OPS_V5_RUNTIME_ACTIVATION_STAGE=checkout\n'
git clone --quiet --shared --no-checkout "$seed" "$work/repo"
git -C "$work/repo" remote remove github >/dev/null 2>&1 || true
git -C "$work/repo" remote add github "$github"
git -C "$work/repo" fetch --quiet --no-tags github \
  "+refs/heads/$branch:refs/validation/candidate" \
  "+refs/heads/main:refs/remotes/github/main"
candidate=$(git -C "$work/repo" rev-parse refs/validation/candidate)
git -C "$work/repo" merge-base --is-ancestor "$control_plane_commit" "$candidate"
git -C "$work/repo" checkout --quiet --detach "$candidate"
test -z "$(git -C "$work/repo" status --porcelain)"
test "$(git -C "$work/repo" rev-parse "$candidate:experience_docx/tools/convir_ops_mcp.py")" = \
  "$(git -C "$work/repo" rev-parse "$control_plane_commit:experience_docx/tools/convir_ops_mcp.py")"

printf 'CONVIR_OPS_V5_RUNTIME_ACTIVATION_STAGE=stdio_handshake\n'
mkdir -p "$evidence_root"
PYTHONPATH="$work/repo/experience_docx/tools" "$python" - \
  "$work/repo" "$candidate" "$route_id" \
  "$evidence_root/runtime-activation-$candidate.json" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

repo, candidate, route_id, output = sys.argv[1:]
server = Path(repo) / "experience_docx/tools/convir_ops_mcp.py"
requests = [
    {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "final-slim-runtime-activation", "version": "1.0"},
        },
    },
    {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
    {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "convir_git_status",
            "arguments": {
                "local_repo": repo,
                "route_id": route_id,
                "detail": "summary",
            },
        },
    },
]
payload = "".join(json.dumps(item, separators=(",", ":")) + "\n" for item in requests)
completed = subprocess.run(
    [sys.executable, str(server)],
    input=payload,
    text=True,
    capture_output=True,
    timeout=30,
    check=True,
)
if completed.stderr.strip():
    raise AssertionError(f"unexpected server stderr: {completed.stderr[:1000]}")
responses = {}
for line in completed.stdout.splitlines():
    response = json.loads(line)
    responses[response["id"]] = response
assert set(responses) == {1, 2, 3}, responses
assert all("error" not in responses[index] for index in (1, 2, 3)), responses

initialize = responses[1]["result"]
tools = responses[2]["result"]["tools"]
call = responses[3]["result"]
structured = call["structuredContent"]
summary = call["content"][0]["text"]
expected_tools = {
    "convir_route_plan",
    "convir_route_start",
    "convir_route_finish",
    "convir_evidence_list",
    "convir_evidence_fetch",
    "convir_git_status",
}
assert initialize["serverInfo"]["version"] == "5.0.0", initialize
assert {item["name"] for item in tools} == expected_tools, tools
assert len(tools) == 6
assert len(summary.encode("utf-8")) <= 2048, len(summary.encode("utf-8"))
assert isinstance(structured["changed_path_count"], int)
assert isinstance(structured["route_evidence_change_count"], int)
assert isinstance(structured["authoritative_snapshot"], dict)
assert structured["authoritative_snapshot"], structured

result = {
    "schema_version": 1,
    "state": "ACTIVE_VERIFIED",
    "candidate_commit": candidate,
    "control_plane_commit": "31cc862f5107106dad8de266299b1bfea0b7a376",
    "server_version": initialize["serverInfo"]["version"],
    "server_source_sha256": initialize["serverInfo"]["sourceSha256"],
    "tool_count": len(tools),
    "tool_names": sorted(item["name"] for item in tools),
    "compact_snapshot_verified": True,
    "summary_bytes": len(summary.encode("utf-8")),
    "changed_path_count": structured["changed_path_count"],
    "route_evidence_change_count": structured["route_evidence_change_count"],
    "authoritative_snapshot": structured["authoritative_snapshot"],
    "historical_evidence_modified": False,
    "model_calls": 0,
    "gpu_access": 0,
    "dataset_access": 0,
    "checkpoint_access": 0,
    "confirmation_access": 0,
    "canary_access": 0,
    "locked_test_access": 0,
}
Path(output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print("CONVIR_OPS_V5_RUNTIME_ACTIVATION_JSON_BEGIN")
print(json.dumps(result, indent=2, sort_keys=True))
print("CONVIR_OPS_V5_RUNTIME_ACTIVATION_JSON_END")
PY

printf 'CONVIR_OPS_V5_RUNTIME_ACTIVATION_OK candidate=%s server_version=5.0.0 tools=6 protected_data_access=0\n' "$candidate"
