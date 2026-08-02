#!/usr/bin/env bash
set -euo pipefail

on_error() {
  rc=$?
  printf 'CONVIR_MAIN_FIRST_SNAPSHOT_CLOUD_FAILED line=%s command=%q rc=%s\n' \
    "$1" "$2" "$rc" >&2
  exit "$rc"
}
trap 'on_error "$LINENO" "$BASH_COMMAND"' ERR

branch=codex/convir-main-first-snapshot-v1
base=7866e061b8784d778d5b26fb1fead99dc65cbcc5
github=git@github.com:onenoober/ConvIR-B.git
seed=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-official-arch-anchor
python=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
work=$(mktemp -d /tmp/convir-main-first-snapshot.XXXXXX)
trap 'rm -rf -- "$work"' EXIT

git clone --quiet --shared --no-checkout "$seed" "$work/repo"
git -C "$work/repo" remote add github "$github"
git -C "$work/repo" fetch --quiet --no-tags github \
  "+refs/heads/$branch:refs/validation/candidate" \
  "+refs/heads/main:refs/remotes/github/main"
candidate=$(git -C "$work/repo" rev-parse refs/validation/candidate)
git -C "$work/repo" merge-base --is-ancestor "$base" "$candidate"
git -C "$work/repo" checkout --quiet --detach "$candidate"
test -z "$(git -C "$work/repo" status --porcelain)"
git -C "$work/repo" diff --check "$base" "$candidate"
git -C "$work/repo" diff --quiet "$base" "$candidate" -- \
  experience_docx/experiment_logs

expected=$'AGENTS.md\nexperience_docx/AI_POLICY_SNAPSHOT.json\nexperience_docx/COMMAND_RELIABILITY_PROTOCOL.md\nexperience_docx/CONVIR_EVIDENCE_REVIEW.md\nexperience_docx/CONVIR_OPS_MCP.md\nexperience_docx/ROUTE_READY_FASTPATH.md\nexperience_docx/RULE_COMPATIBILITY.json\nexperience_docx/SCIENCE_FASTPATH.md\nexperience_docx/tools/convir_ops_mcp.py\nexperience_docx/tools/policy_snapshot.py\nexperience_docx/tools/tests/test_convir_ops_mcp.py\nexperience_docx/tools/tests/test_convir_ops_v5_final_slim.py\nexperience_docx/tools/validate_convir_main_first_snapshot_cloud.sh'
actual=$(git -C "$work/repo" diff --name-only "$base" "$candidate")
test "$actual" = "$expected"

tools=$work/repo/experience_docx/tools
tests=$tools/tests
bash -n "$tools/validate_convir_main_first_snapshot_cloud.sh"
"$python" -m py_compile \
  "$tools/convir_ops_mcp.py" \
  "$tools/policy_snapshot.py" \
  "$tests/test_convir_ops_mcp.py" \
  "$tests/test_convir_ops_v5_final_slim.py"

rules_commit=$("$python" - "$work/repo/experience_docx/AI_POLICY_SNAPSHOT.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    print(json.load(handle)["rules_commit"])
PY
)
PYTHONPATH="$tools" "$python" "$tools/policy_snapshot.py" \
  --repo "$work/repo" --rules-commit "$rules_commit" --check >/dev/null

stdout=$work/unittest.stdout
stderr=$work/unittest.stderr
trap - ERR
set +e
PYTHONPATH="$tools:$tests" "$python" -m unittest \
  test_capability_registry \
  test_convir_evidence_cloud_inventory \
  test_convir_evidence_review_mcp \
  test_convir_ops_mcp \
  test_convir_ops_v5_final_slim \
  test_experiment_spec_compiler \
  test_policy_snapshot \
  test_prepare_terminal_archive \
  test_route_lifecycle \
  test_validate_engineering_repair \
  test_validate_route_ready >"$stdout" 2>"$stderr"
rc=$?
set -e
trap 'on_error "$LINENO" "$BASH_COMMAND"' ERR
if [[ $rc -ne 0 ]]; then
  tail -n 200 "$stdout" >&2 || true
  tail -n 200 "$stderr" >&2 || true
  exit "$rc"
fi
test_count=$("$python" - "$stderr" <<'PY'
import re
import sys
from pathlib import Path

matches = re.findall(r"Ran ([0-9]+) tests?", Path(sys.argv[1]).read_text())
assert matches
print(matches[-1])
PY
)
test "$test_count" -ge 245

CONVIR_OPS_LOCAL_WORKSPACE_ROOT="$work" PYTHONPATH="$tools" "$python" - \
  "$tools/convir_ops_mcp.py" "$work/repo" <<'PY'
import json
import os
import subprocess
import sys

server, repo = sys.argv[1:]
requests = [
    {
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05", "capabilities": {},
            "clientInfo": {"name": "main-first-acceptance", "version": "1.0"},
        },
    },
    {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
    {
        "jsonrpc": "2.0", "id": 3, "method": "tools/call",
        "params": {
            "name": "convir_git_status",
            "arguments": {"scope": "project", "local_repo": repo},
        },
    },
]
payload = "".join(json.dumps(item, separators=(",", ":")) + "\n" for item in requests)
completed = subprocess.run(
    [sys.executable, server], input=payload, text=True, capture_output=True,
    timeout=60, check=True, env=os.environ.copy(),
)
assert completed.stderr.strip() == "", completed.stderr
responses = {item["id"]: item for item in map(json.loads, completed.stdout.splitlines())}
assert set(responses) == {1, 2, 3}
assert responses[1]["result"]["serverInfo"]["version"] == "5.8.0"
expected = {
    "convir_route_plan", "convir_route_start", "convir_route_finish",
    "convir_evidence_list", "convir_evidence_fetch", "convir_git_status",
}
surface = responses[2]["result"]["tools"]
assert len(surface) == 6
assert {item["name"] for item in surface} == expected
status_tool = next(item for item in surface if item["name"] == "convir_git_status")
assert status_tool["inputSchema"]["properties"]["scope"] == {
    "enum": ["project", "route"], "default": "route",
}
result = responses[3]["result"]
assert result.get("isError") is False, result
value = result["structuredContent"]
assert value["scope"] == "project"
assert value["github_main_ref_fresh"] is True
assert value["local_worktree_assessed"] is False
assert value["authoritative_snapshot"]["status"] == "AUTHORITATIVE_PROJECT_SNAPSHOT_OK"
assert value["authoritative_read_binding"]["status"] == "BOUND"
assert value["local_write_binding"]["status"] == "NOT_APPLICABLE"
assert value["phase_receipt"]["scientific_authorization"] == "NOT_DERIVED"
PY

source_sha=$(sha256sum "$tools/convir_ops_mcp.py" | cut -d' ' -f1)
printf 'CONVIR_MAIN_FIRST_SNAPSHOT_CLOUD_OK candidate=%s tests=%s source_sha256=%s ops_version=5.8.0 tools=6 model_calls=0 gpu_access=0 protected_data_access=0 experiment_launches=0\n' \
  "$candidate" "$test_count" "$source_sha"
