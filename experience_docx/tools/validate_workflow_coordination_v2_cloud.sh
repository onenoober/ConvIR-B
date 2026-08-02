#!/usr/bin/env bash
set -euo pipefail

on_error() {
  rc=$?
  printf 'WORKFLOW_COORDINATION_V2_CLOUD_FAILED line=%s command=%q rc=%s\n' "$1" "$2" "$rc" >&2
  exit "$rc"
}
trap 'on_error "$LINENO" "$BASH_COMMAND"' ERR

branch=codex/convir-workflow-coordination-v1
base=d707c0ffc158df4a8cab38a74a41555e7e4af456
github=git@github.com:onenoober/ConvIR-B.git
seed=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-official-arch-anchor
python=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
work=$(mktemp -d /tmp/workflow-coordination-v2.XXXXXX)
trap 'rm -rf -- "$work"' EXIT

git clone --quiet --shared --no-checkout "$seed" "$work/repo"
git -C "$work/repo" fetch --quiet --no-tags "$github" \
  "+refs/heads/$branch:refs/validation/candidate"
candidate=$(git -C "$work/repo" rev-parse refs/validation/candidate)
git -C "$work/repo" cat-file -e "$base^{commit}"
git -C "$work/repo" merge-base --is-ancestor "$base" "$candidate"
git -C "$work/repo" checkout --quiet --detach "$candidate"
test -z "$(git -C "$work/repo" status --porcelain)"

tools="$work/repo/experience_docx/tools"
"$python" -m py_compile \
  "$tools/experiment_spec_compiler.py" \
  "$tools/validate_route_ready.py" \
  "$tools/convir_ops_mcp.py" \
  "$tools/convir_evidence_review_mcp.py" \
  "$tools/policy_snapshot.py" \
  "$tools/tests/test_experiment_spec_compiler.py" \
  "$tools/tests/test_validate_route_ready.py" \
  "$tools/tests/test_convir_ops_mcp.py" \
  "$tools/tests/test_convir_ops_v5_final_slim.py" \
  "$tools/tests/test_convir_evidence_review_mcp.py"

stdout="$work/unittest.stdout"
stderr="$work/unittest.stderr"
if PYTHONPATH="$tools:$tools/tests" "$python" -m unittest discover \
    -s "$tools/tests" -p 'test_*.py' >"$stdout" 2>"$stderr"; then
  rc=0
else
  rc=$?
fi
if [[ $rc -ne 0 ]]; then
  tail -n 200 "$stdout" >&2 || true
  tail -n 200 "$stderr" >&2 || true
  exit "$rc"
fi
tests=$(sed -nE 's/^Ran ([0-9]+) tests?.*/\1/p' "$stderr" | tail -n 1)
[[ $tests =~ ^[0-9]+$ ]]
test "$tests" -ge 300

policy_rules_commit=$("$python" -c \
  'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["rules_commit"])' \
  "$work/repo/experience_docx/AI_POLICY_SNAPSHOT.json")
[[ $policy_rules_commit =~ ^[0-9a-f]{40}$ ]]
"$python" "$tools/policy_snapshot.py" \
  --repo "$work/repo" --rules-commit "$policy_rules_commit" --check

PYTHONPATH="$tools" "$python" - "$tools" <<'PY'
import hashlib
import json
from pathlib import Path
import subprocess
import sys

tools = Path(sys.argv[1])

def activate(script, version, expected_tools):
    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": "2024-11-05", "capabilities": {},
            "clientInfo": {"name": "cloud-acceptance", "version": "1"},
        }},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
    ]
    completed = subprocess.run(
        [sys.executable, str(script)],
        input="".join(json.dumps(item) + "\n" for item in requests),
        text=True, capture_output=True, timeout=30, check=True,
    )
    responses = [json.loads(line) for line in completed.stdout.splitlines() if line]
    assert [item["id"] for item in responses] == [1, 2], responses
    info = responses[0]["result"]["serverInfo"]
    assert info["version"] == version, info
    assert info["sourceSha256"] == hashlib.sha256(script.read_bytes()).hexdigest(), info
    names = [item["name"] for item in responses[1]["result"]["tools"]]
    assert names == expected_tools, names

activate(tools / "convir_ops_mcp.py", "5.7.0", [
    "convir_route_plan", "convir_route_start", "convir_route_finish",
    "convir_evidence_list", "convir_evidence_fetch", "convir_git_status",
])
activate(tools / "convir_evidence_review_mcp.py", "2.1.0", [
    "convir_evidence_completeness_receipt",
    "convir_evidence_catalog_query",
    "convir_evidence_bundle",
    "convir_evidence_cloud_inventory_summary",
    "convir_evidence_cloud_inventory_query",
    "convir_evidence_cloud_text_read",
])
PY

git -C "$work/repo" diff --check "$base" "$candidate"
git -C "$work/repo" diff --quiet
printf 'WORKFLOW_COORDINATION_V2_CLOUD_OK candidate=%s tests=%s ops_version=5.7.0 review_version=2.1.0 tools=6+6 model_calls=0 gpu_access=0 protected_data_access=0\n' \
  "$candidate" "$tests"
