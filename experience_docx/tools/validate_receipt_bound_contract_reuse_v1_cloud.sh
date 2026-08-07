#!/usr/bin/env bash
set -euo pipefail

on_error() {
  rc=$?
  printf 'RECEIPT_BOUND_CONTRACT_REUSE_V1_CLOUD_FAILED line=%s command=%q rc=%s\n' \
    "$1" "$2" "$rc" >&2
  exit "$rc"
}
trap 'on_error "$LINENO" "$BASH_COMMAND"' ERR

branch=codex/receipt-bound-contract-reuse-v1
base=5a687608a7c58c81c4d8ee329443c78b8a7e1a04
github=git@github.com:onenoober/ConvIR-B.git
seed=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-official-arch-anchor
python=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
work=$(mktemp -d /tmp/convir-receipt-bound-contract-reuse.XXXXXX)
trap 'rm -rf -- "$work"' EXIT

printf 'RECEIPT_BOUND_CONTRACT_REUSE_V1_STAGE=checkout\n'
git clone --quiet --shared --no-checkout "$seed" "$work/repo"
git -C "$work/repo" fetch --quiet --no-tags "$github" \
  "+refs/heads/$branch:refs/validation/candidate"
candidate=$(git -C "$work/repo" rev-parse refs/validation/candidate)
git -C "$work/repo" cat-file -e "$base^{commit}"
git -C "$work/repo" merge-base --is-ancestor "$base" "$candidate"
git -C "$work/repo" checkout --quiet --detach "$candidate"
test -z "$(git -C "$work/repo" status --porcelain)"
git -C "$work/repo" diff --quiet "$base" "$candidate" -- experience_docx/experiment_logs
git -C "$work/repo" diff --check "$base" "$candidate"

expected=$'AGENTS.md\nexperience_docx/AI_POLICY_SNAPSHOT.json\nexperience_docx/COMMAND_RELIABILITY_PROTOCOL.md\nexperience_docx/CONVIR_OPS_MCP.md\nexperience_docx/EXPERIMENT_GOVERNANCE_PROTOCOL.md\nexperience_docx/MODEL_RUN_OPERATIONS_PROTOCOL.md\nexperience_docx/ROUTE_READY_FASTPATH.md\nexperience_docx/RULE_COMPATIBILITY.json\nexperience_docx/SCIENCE_FASTPATH.md\nexperience_docx/tools/convir_ops_mcp.py\nexperience_docx/tools/inspect_receipt_bound_contract_reuse_v1_cloud.sh\nexperience_docx/tools/route_lifecycle.py\nexperience_docx/tools/route_runtime_contract.py\nexperience_docx/tools/tests/test_convir_ops_mcp.py\nexperience_docx/tools/tests/test_route_lifecycle.py\nexperience_docx/tools/tests/test_validate_engineering_repair.py\nexperience_docx/tools/validate_engineering_repair.py\nexperience_docx/tools/validate_receipt_bound_contract_reuse_v1_cloud.sh'
actual=$(git -C "$work/repo" diff --name-only "$base" "$candidate")
test "$actual" = "$expected"

tools=$work/repo/experience_docx/tools
tests=$tools/tests

printf 'RECEIPT_BOUND_CONTRACT_REUSE_V1_STAGE=compile\n'
"$python" -m py_compile \
  "$tools/convir_ops_mcp.py" \
  "$tools/route_lifecycle.py" \
  "$tools/route_runtime_contract.py" \
  "$tools/validate_engineering_repair.py" \
  "$tests/test_convir_ops_mcp.py" \
  "$tests/test_policy_snapshot.py" \
  "$tests/test_route_lifecycle.py" \
  "$tests/test_route_runtime_contract.py" \
  "$tests/test_validate_engineering_repair.py"

printf 'RECEIPT_BOUND_CONTRACT_REUSE_V1_STAGE=policy_snapshot\n'
rules_commit=$("$python" - "$work/repo/experience_docx/AI_POLICY_SNAPSHOT.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    print(json.load(handle)["rules_commit"])
PY
)
PYTHONPATH="$tools" "$python" "$tools/policy_snapshot.py" \
  --repo "$work/repo" --rules-commit "$rules_commit" --check >/dev/null

printf 'RECEIPT_BOUND_CONTRACT_REUSE_V1_STAGE=targeted_regression\n'
PYTHONPATH="$tools:$tests" "$python" -m unittest -v \
  test_convir_ops_mcp \
  test_policy_snapshot \
  test_route_lifecycle \
  test_route_runtime_contract \
  test_validate_engineering_repair

printf 'RECEIPT_BOUND_CONTRACT_REUSE_V1_STAGE=full_regression\n'
stdout=$work/unittest.stdout
stderr=$work/unittest.stderr
trap - ERR
set +e
PYTHONPATH="$tools:$tests" "$python" -m unittest discover \
  -s "$tests" -p 'test_*.py' >"$stdout" 2>"$stderr"
rc=$?
set -e
trap 'on_error "$LINENO" "$BASH_COMMAND"' ERR
if [[ $rc -ne 0 ]]; then
  tail -n 200 "$stdout" >&2 || true
  tail -n 200 "$stderr" >&2 || true
  exit "$rc"
fi
test_count=$(sed -nE 's/^Ran ([0-9]+) tests?.*/\1/p' "$stderr" | tail -n 1)
[[ "$test_count" =~ ^[0-9]+$ ]]
test "$test_count" -ge 150

printf 'RECEIPT_BOUND_CONTRACT_REUSE_V1_STAGE=fresh_mcp_activation\n'
PYTHONPATH="$tools" "$python" - "$tools/convir_ops_mcp.py" "$candidate" "$test_count" <<'PY'
import json
import subprocess
import sys

server, candidate, test_count = sys.argv[1:]
requests = [
    {
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05", "capabilities": {},
            "clientInfo": {"name": "receipt-reuse-acceptance", "version": "1.0"},
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
initialize = responses[1]["result"]
listed = responses[2]["result"]["tools"]
expected = {
    "convir_route_plan", "convir_route_start", "convir_route_finish",
    "convir_evidence_list", "convir_evidence_fetch", "convir_git_status",
}
assert initialize["serverInfo"]["version"] == "5.13.0", initialize
assert len(listed) == 6 and {item["name"] for item in listed} == expected, listed

import convir_ops_mcp as ops
assert ops.SCHEMA_VERSION == 4
assert ops.SERVER_VERSION == "5.13.0"
assert len(ops.TOOLS) == 6
finish = ops.TOOLS["convir_route_finish"]["inputSchema"]
assert "reviewed_repair" in finish["properties"]["engineering_failure_resolution"]["enum"]
print(json.dumps({
    "candidate_commit": candidate,
    "tests_passed": int(test_count),
    "server_version": ops.SERVER_VERSION,
    "protocol_schema": ops.SCHEMA_VERSION,
    "tool_count": len(ops.TOOLS),
    "receipt_contract_reuse": "PASS",
    "chained_reuse": "PASS",
    "scientific_authorization": "NONE",
    "gpu_access": 0,
    "dataset_access": 0,
    "checkpoint_access": 0,
    "protected_data_access": 0,
    "experiment_launches": 0,
}, indent=2, sort_keys=True))
PY

printf 'RECEIPT_BOUND_CONTRACT_REUSE_V1_CLOUD_OK candidate=%s tests=%s tools=6 gpu_access=0 dataset_access=0 protected_data_access=0 experiment_launches=0\n' \
  "$candidate" "$test_count"
