#!/usr/bin/env bash
set -euo pipefail

on_error() {
  rc=$?
  printf 'REVIEWED_REPAIR_RECEIPT_GPU_BINDING_CLOUD_FAILED line=%s command=%q rc=%s\n' \
    "$1" "$2" "$rc" >&2
  exit "$rc"
}
trap 'on_error "$LINENO" "$BASH_COMMAND"' ERR

branch=codex/reviewed-repair-receipt-gpu-binding-fix
base=fb44d5544205ff27ac368e71671d595ed95602c4
github=git@github.com:onenoober/ConvIR-B.git
seed=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-official-arch-anchor
python=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
work=$(mktemp -d /tmp/convir-reviewed-repair-gpu-binding.XXXXXX)
trap 'rm -rf -- "$work"' EXIT

printf 'REVIEWED_REPAIR_RECEIPT_GPU_BINDING_STAGE=checkout\n'
git clone --quiet --shared --no-checkout "$seed" "$work/repo"
git -C "$work/repo" fetch --quiet --no-tags "$github" \
  "+refs/heads/$branch:refs/validation/candidate"
candidate=$(git -C "$work/repo" rev-parse refs/validation/candidate)
git -C "$work/repo" cat-file -e "$base^{commit}"
git -C "$work/repo" merge-base --is-ancestor "$base" "$candidate"
git -C "$work/repo" checkout --quiet --detach "$candidate"
test -z "$(git -C "$work/repo" status --porcelain)"
git -C "$work/repo" diff --check "$base" "$candidate"

expected=$'experience_docx/tools/convir_ops_mcp.py\nexperience_docx/tools/tests/test_convir_ops_mcp.py\nexperience_docx/tools/validate_reviewed_repair_receipt_gpu_binding_cloud.sh'
actual=$(git -C "$work/repo" diff --name-only "$base" "$candidate")
test "$actual" = "$expected"

tools=$work/repo/experience_docx/tools
tests=$tools/tests

printf 'REVIEWED_REPAIR_RECEIPT_GPU_BINDING_STAGE=compile\n'
"$python" -m py_compile \
  "$tools/convir_ops_mcp.py" \
  "$tests/test_convir_ops_mcp.py"

printf 'REVIEWED_REPAIR_RECEIPT_GPU_BINDING_STAGE=targeted_regression\n'
PYTHONPATH="$tools:$tests" "$python" -m unittest -v \
  test_convir_ops_mcp.ConvirOpsV4Tests.test_receipt_gpu_index_reads_signed_launch_binding \
  test_convir_ops_mcp.ConvirOpsV4Tests.test_receipt_gpu_index_keeps_missing_binding_fail_closed \
  test_convir_ops_mcp.ConvirOpsV4Tests.test_receipt_gpu_index_rejects_malformed_signed_binding \
  test_convir_ops_mcp.ConvirOpsV4Tests.test_receipt_gpu_index_rejects_conflicting_bindings \
  test_convir_ops_mcp.ConvirOpsV4Tests.test_reviewed_workload_repair_reads_gpu_from_receipt_payload \
  test_convir_ops_mcp.ConvirOpsV4Tests.test_reviewed_workload_repair_rejects_missing_signed_gpu

printf 'REVIEWED_REPAIR_RECEIPT_GPU_BINDING_STAGE=full_regression\n'
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

printf 'REVIEWED_REPAIR_RECEIPT_GPU_BINDING_STAGE=fresh_mcp_activation\n'
PYTHONPATH="$tools" "$python" - "$tools/convir_ops_mcp.py" "$candidate" "$test_count" <<'PY'
import hashlib
import importlib.util
import json
import subprocess
import sys

server, candidate, test_count = sys.argv[1:]
requests = [
    {
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05", "capabilities": {},
            "clientInfo": {"name": "reviewed-repair-gpu-binding", "version": "1.0"},
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

source_sha = hashlib.sha256(open(server, "rb").read()).hexdigest()
print(json.dumps({
    "candidate_commit": candidate,
    "tests_passed": int(test_count),
    "server_version": initialize["serverInfo"]["version"],
    "server_source_sha256": source_sha,
    "tool_count": len(listed),
    "signed_receipt_gpu_binding": "PASS",
    "missing_or_conflicting_binding": "FAIL_CLOSED",
    "gpu_access": 0,
    "dataset_access": 0,
    "checkpoint_access": 0,
    "protected_data_access": 0,
    "experiment_launches": 0,
}, indent=2, sort_keys=True))
PY

printf 'REVIEWED_REPAIR_RECEIPT_GPU_BINDING_CLOUD_OK candidate=%s tests=%s tools=6 gpu_access=0 dataset_access=0 protected_data_access=0 experiment_launches=0\n' \
  "$candidate" "$test_count"
