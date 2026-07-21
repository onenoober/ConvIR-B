#!/usr/bin/env bash
set -euo pipefail

on_error() {
  rc=$?
  printf 'CONVIR_OPS_GPU_RESOURCE_PROBE_CLOUD_FAILED line=%s command=%q rc=%s\n' "$1" "$2" "$rc" >&2
  exit "$rc"
}
trap 'on_error "$LINENO" "$BASH_COMMAND"' ERR

branch=codex/convir-ops-gpu-resource-probe-fix-20260721
candidate=18f28ff9443de774125007909fbd8da2ccf37354
base=d5eef4977f11561a660f31e2bb4fb56035bb56bb
github=git@github.com:onenoober/ConvIR-B.git
seed=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-official-arch-anchor
python=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
work=$(mktemp -d /tmp/convir-ops-gpu-probe.XXXXXX)
trap 'rm -rf -- "$work"' EXIT

printf 'CONVIR_OPS_GPU_RESOURCE_PROBE_STAGE=checkout\n'
git clone --quiet --shared --no-checkout "$seed" "$work/repo"
git -C "$work/repo" fetch --quiet --no-tags "$github" \
  "+refs/heads/$branch:refs/validation/candidate"
test "$(git -C "$work/repo" rev-parse refs/validation/candidate)" = "$candidate"
git -C "$work/repo" merge-base --is-ancestor "$base" "$candidate"
git -C "$work/repo" checkout --quiet --detach "$candidate"
test -z "$(git -C "$work/repo" status --porcelain)"

tools="$work/repo/experience_docx/tools"
printf 'CONVIR_OPS_GPU_RESOURCE_PROBE_STAGE=compile\n'
"$python" -m py_compile \
  "$tools/convir_ops_mcp.py" \
  "$tools/tests/test_convir_ops_mcp.py"

printf 'CONVIR_OPS_GPU_RESOURCE_PROBE_STAGE=full_regression\n'
stdout="$work/unittest.stdout"
stderr="$work/unittest.stderr"
set +e
PYTHONPATH="$tools" "$python" -m unittest discover \
  -s "$tools/tests" -p 'test_*.py' >"$stdout" 2>"$stderr"
rc=$?
set -e
if [[ $rc -ne 0 ]]; then
  tail -n 200 "$stdout" >&2 || true
  tail -n 200 "$stderr" >&2 || true
  exit "$rc"
fi
tests=$(sed -nE 's/^Ran ([0-9]+) tests?.*/\1/p' "$stderr" | tail -n 1)
[[ $tests =~ ^[0-9]+$ ]]
test "$tests" -ge 120

printf 'CONVIR_OPS_GPU_RESOURCE_PROBE_STAGE=real_query\n'
PYTHONPATH="$tools" "$python" - <<'PY'
import json
import os
import subprocess

import convir_ops_mcp as ops

assert ops.SERVER_VERSION == "5.1.0"
assert ops.SCHEMA_VERSION == 4
assert ops.SUPPORTED_MANIFEST_SCHEMA_VERSIONS == {4, 5, 6}
assert len(ops.TOOLS) == 6
assert ops.NVIDIA_SMI == "/usr/bin/nvidia-smi"
assert os.access(ops.NVIDIA_SMI, os.X_OK)
context = {
    "min_free_gpu_mib": 4096,
    "max_gpu_utilization_pct": 90,
}
body = "set -euo pipefail\n" + ops.gpu_probe_body(context)
completed = subprocess.run(
    ["/bin/bash", "-c", body],
    text=True, capture_output=True, timeout=30, check=False,
)
if completed.returncode:
    print(completed.stdout, end="")
    print(completed.stderr, end="", file=__import__("sys").stderr)
    raise SystemExit(completed.returncode)
probe = ops.parse_gpu(completed.stdout)
selected = [row for row in probe["rows"] if row["index"] == probe["index"]]
assert len(selected) == 1
row = selected[0]
assert row["free_mib"] >= context["min_free_gpu_mib"]
assert row["utilization_pct"] <= context["max_gpu_utilization_pct"]
print("CONVIR_OPS_GPU_RESOURCE_PROBE_REAL_QUERY_OK " + json.dumps({
    "selected_gpu": row["index"],
    "free_mib": row["free_mib"],
    "utilization_pct": row["utilization_pct"],
    "total_gpu_count": probe["total_gpu_count"],
}, sort_keys=True, separators=(",", ":")))
PY

printf 'CONVIR_OPS_GPU_RESOURCE_PROBE_CLOUD_OK candidate=%s tests=%s tools=6 schema=4 manifests=4,5,6 gpu_query_only=1 gpu_workload=0 dataset_access=0 checkpoint_access=0 protected_data_access=0\n' \
  "$candidate" "$tests"
