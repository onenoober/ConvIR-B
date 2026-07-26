#!/usr/bin/env bash
set -euo pipefail

on_error() {
  rc=$?
  printf 'CONTROL_PLANE_EFFICIENCY_V1_CLOUD_FAILED line=%s command=%q rc=%s\n' "$1" "$2" "$rc" >&2
  exit "$rc"
}
trap 'on_error "$LINENO" "$BASH_COMMAND"' ERR

branch=codex/convir-control-plane-efficiency-v1
base=85a8c1a46bd1f3ced110bd2a050aec5ed6eeee7b
github=git@github.com:onenoober/ConvIR-B.git
seed=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-official-arch-anchor
python=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
work=$(mktemp -d /tmp/convir-control-plane-efficiency.XXXXXX)
trap 'rm -rf -- "$work"' EXIT

printf 'CONTROL_PLANE_EFFICIENCY_V1_STAGE=checkout\n'
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

tools=$work/repo/experience_docx/tools
tests=$tools/tests

printf 'CONTROL_PLANE_EFFICIENCY_V1_STAGE=compile\n'
"$python" -m py_compile \
  "$tools/convir_ops_mcp.py" \
  "$tools/experiment_spec_compiler.py" \
  "$tools/route_lifecycle.py" \
  "$tools/route_program_api.py" \
  "$tools/route_runtime_contract.py" \
  "$tests/test_convir_ops_mcp.py" \
  "$tests/test_experiment_spec_compiler.py" \
  "$tests/test_route_lifecycle.py" \
  "$tests/test_route_program_api.py" \
  "$tests/test_route_runtime_contract.py"

printf 'CONTROL_PLANE_EFFICIENCY_V1_STAGE=policy_snapshot\n'
rules_commit=$("$python" - "$work/repo/experience_docx/AI_POLICY_SNAPSHOT.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["rules_commit"])
PY
)
PYTHONPATH="$tools" "$python" "$tools/policy_snapshot.py" \
  --repo "$work/repo" --rules-commit "$rules_commit" --check >/dev/null

printf 'CONTROL_PLANE_EFFICIENCY_V1_STAGE=full_regression\n'
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

printf 'CONTROL_PLANE_EFFICIENCY_V1_STAGE=fresh_mcp_activation\n'
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
            "clientInfo": {"name": "control-plane-efficiency-acceptance", "version": "1.0"},
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
assert set(responses) == {1, 2}, responses
initialize = responses[1]["result"]
tools = responses[2]["result"]["tools"]
expected = {
    "convir_route_plan", "convir_route_start", "convir_route_finish",
    "convir_evidence_list", "convir_evidence_fetch", "convir_git_status",
}
assert initialize["serverInfo"]["version"] == "5.2.0", initialize
assert len(tools) == 6 and {item["name"] for item in tools} == expected, tools

import convir_ops_mcp as ops
assert ops.SCHEMA_VERSION == 4
assert ops.SUPPORTED_MANIFEST_SCHEMA_VERSIONS == {4, 5, 6}
assert ops.SERVER_VERSION == "5.2.0"
assert len(ops.TOOLS) == 6
print(json.dumps({
    "candidate_commit": candidate,
    "tests_passed": int(test_count),
    "server_version": ops.SERVER_VERSION,
    "protocol_schema": ops.SCHEMA_VERSION,
    "manifest_schemas": sorted(ops.SUPPORTED_MANIFEST_SCHEMA_VERSIONS),
    "tool_count": len(ops.TOOLS),
    "aggregate_lint": "PASS",
    "cost_contract": "PASS",
    "contract_progress_and_diagnostic": "PASS",
    "finish_throttle": "PASS",
    "historical_runtime_compatibility": "PASS",
    "experiment_logs_modified": False,
    "model_calls": 0,
    "gpu_access": 0,
    "dataset_access": 0,
    "checkpoint_access": 0,
    "protected_data_access": 0,
    "experiment_launches": 0,
}, indent=2, sort_keys=True))
PY

printf 'CONTROL_PLANE_EFFICIENCY_V1_CLOUD_OK candidate=%s tests=%s tools=6 model_calls=0 gpu_access=0 protected_data_access=0 experiment_launches=0\n' \
  "$candidate" "$test_count"
