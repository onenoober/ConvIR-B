#!/usr/bin/env bash
set -euo pipefail

on_error() {
  rc=$?
  printf 'CONVIR_OPS_GOVERNANCE_HARDENING_CLOUD_FAILED line=%s command=%q rc=%s\n' \
    "$1" "$2" "$rc" >&2
  exit "$rc"
}
trap 'on_error "$LINENO" "$BASH_COMMAND"' ERR

branch=codex/convir-ops-governance-hardening-v1
base=7024906a5441d055ab4dbe8a12311ae9bdfb9766
github=git@github.com:onenoober/ConvIR-B.git
seed=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-official-arch-anchor
python=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
work=$(mktemp -d /tmp/convir-ops-governance-hardening.XXXXXX)
trap 'rm -rf -- "$work"' EXIT

git clone --quiet --shared --no-checkout "$seed" "$work/repo"
git -C "$work/repo" fetch --quiet --no-tags "$github" \
  "+refs/heads/$branch:refs/validation/candidate"
candidate=$(git -C "$work/repo" rev-parse refs/validation/candidate)
git -C "$work/repo" merge-base --is-ancestor "$base" "$candidate"
git -C "$work/repo" checkout --quiet --detach "$candidate"
test -z "$(git -C "$work/repo" status --porcelain)"
git -C "$work/repo" diff --check "$base" "$candidate"
git -C "$work/repo" diff --quiet "$base" "$candidate" -- \
  experience_docx/experiment_logs

expected=$'AGENTS.md\nexperience_docx/AI_POLICY_SNAPSHOT.json\nexperience_docx/BRANCH_EXPERIMENT_SYNC_PROTOCOL.md\nexperience_docx/COMMAND_RELIABILITY_PROTOCOL.md\nexperience_docx/CONVIR_EVIDENCE_REVIEW.md\nexperience_docx/CONVIR_OPS_MCP.md\nexperience_docx/EXPERIMENT_GOVERNANCE_PROTOCOL.md\nexperience_docx/MODEL_RUN_OPERATIONS_PROTOCOL.md\nexperience_docx/ROUTE_READY_FASTPATH.md\nexperience_docx/RULE_COMPATIBILITY.json\nexperience_docx/SCIENCE_FASTPATH.md\nexperience_docx/tools/capability_registry.py\nexperience_docx/tools/convir_evidence_cloud_inventory.py\nexperience_docx/tools/convir_evidence_review_mcp.py\nexperience_docx/tools/convir_ops_mcp.py\nexperience_docx/tools/experiment_spec_compiler.py\nexperience_docx/tools/prepare_terminal_archive.py\nexperience_docx/tools/route_lifecycle.py\nexperience_docx/tools/tests/test_capability_registry.py\nexperience_docx/tools/tests/test_convir_evidence_cloud_inventory.py\nexperience_docx/tools/tests/test_convir_evidence_review_mcp.py\nexperience_docx/tools/tests/test_convir_ops_mcp.py\nexperience_docx/tools/tests/test_experiment_spec_compiler.py\nexperience_docx/tools/tests/test_prepare_terminal_archive.py\nexperience_docx/tools/tests/test_route_lifecycle.py\nexperience_docx/tools/tests/test_validate_engineering_repair.py\nexperience_docx/tools/tests/test_validate_route_ready.py\nexperience_docx/tools/validate_convir_ops_governance_hardening_cloud.sh\nexperience_docx/tools/validate_engineering_repair.py\nexperience_docx/tools/validate_route_ready.py\nexperience_docx/tools/validate_workflow_coordination_v2_cloud.sh\nexperience_docx/tools/validate_workflow_handoff_fastpath_fix_cloud.sh'
actual=$(git -C "$work/repo" diff --name-only "$base" "$candidate")
test "$actual" = "$expected"

tools=$work/repo/experience_docx/tools
tests=$tools/tests
bash -n "$tools/validate_convir_ops_governance_hardening_cloud.sh"
"$python" -m py_compile \
  "$tools/capability_registry.py" \
  "$tools/convir_evidence_cloud_inventory.py" \
  "$tools/convir_evidence_review_mcp.py" \
  "$tools/convir_ops_mcp.py" \
  "$tools/experiment_spec_compiler.py" \
  "$tools/prepare_terminal_archive.py" \
  "$tools/route_lifecycle.py" \
  "$tools/validate_engineering_repair.py" \
  "$tools/validate_route_ready.py"

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
  test_experiment_spec_compiler \
  test_validate_route_ready \
  test_convir_ops_mcp \
  test_route_lifecycle \
  test_validate_engineering_repair \
  test_prepare_terminal_archive >"$stdout" 2>"$stderr"
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
test "$test_count" -ge 235

PYTHONPATH="$tools" "$python" - \
  "$tools/convir_ops_mcp.py" "$tools/convir_evidence_review_mcp.py" <<'PY'
import json
import subprocess
import sys

server = sys.argv[1]
review_server = sys.argv[2]
requests = [
    {
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05", "capabilities": {},
            "clientInfo": {"name": "governance-hardening-acceptance", "version": "1.0"},
        },
    },
    {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
]
payload = "".join(json.dumps(item, separators=(",", ":")) + "\n" for item in requests)
completed = subprocess.run(
    [sys.executable, server], input=payload, text=True, capture_output=True,
    timeout=30, check=True,
)
assert completed.stderr.strip() == "", completed.stderr
responses = {item["id"]: item for item in map(json.loads, completed.stdout.splitlines())}
assert set(responses) == {1, 2}
assert responses[1]["result"]["serverInfo"]["version"] == "5.7.0"
expected = {
    "convir_route_plan", "convir_route_start", "convir_route_finish",
    "convir_evidence_list", "convir_evidence_fetch", "convir_git_status",
}
surface = responses[2]["result"]["tools"]
assert len(surface) == 6
assert {item["name"] for item in surface} == expected
fetch = next(item for item in surface if item["name"] == "convir_evidence_fetch")
assert fetch["inputSchema"]["properties"]["delivery"] == {
    "enum": ["inline", "materialize"], "default": "inline",
}
finish = next(item for item in surface if item["name"] == "convir_route_finish")
assert finish["inputSchema"]["properties"]["engineering_failure_resolution"]["enum"] == [
    "repair", "archive", "discard", "finalize",
]

completed = subprocess.run(
    [sys.executable, review_server], input=payload, text=True, capture_output=True,
    timeout=30, check=True,
)
assert completed.stderr.strip() == "", completed.stderr
responses = {item["id"]: item for item in map(json.loads, completed.stdout.splitlines())}
assert set(responses) == {1, 2}
assert responses[1]["result"]["serverInfo"]["version"] == "2.1.0"
review_expected = {
    "convir_evidence_completeness_receipt", "convir_evidence_catalog_query",
    "convir_evidence_bundle", "convir_evidence_cloud_inventory_summary",
    "convir_evidence_cloud_inventory_query", "convir_evidence_cloud_text_read",
}
review_surface = responses[2]["result"]["tools"]
assert len(review_surface) == 6
assert {item["name"] for item in review_surface} == review_expected
PY

source_sha=$(sha256sum "$tools/convir_ops_mcp.py" | cut -d' ' -f1)
printf 'CONVIR_OPS_GOVERNANCE_HARDENING_CLOUD_OK candidate=%s tests=%s source_sha256=%s ops_version=5.7.0 review_version=2.1.0 tools=6+6 model_calls=0 gpu_access=0 protected_data_access=0 experiment_launches=0\n' \
  "$candidate" "$test_count" "$source_sha"
