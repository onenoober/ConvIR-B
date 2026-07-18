#!/usr/bin/env bash
set -euo pipefail

branch=${CONVIR_V432_BRANCH:-codex/route-flow-tools-v1-20260717}
github=git@github.com:onenoober/ConvIR-B.git
seed=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-official-arch-anchor
python=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
work=$(mktemp -d /tmp/convir-ops-v432.XXXXXX)
trap 'rm -rf -- "$work"' EXIT

git clone --quiet --shared --no-checkout "$seed" "$work/repo"
git -C "$work/repo" fetch --quiet --no-tags "$github" \
  "+refs/heads/$branch:refs/validation/candidate"
candidate=$(git -C "$work/repo" rev-parse refs/validation/candidate)
git -C "$work/repo" checkout --quiet --detach "$candidate"
test -z "$(git -C "$work/repo" status --porcelain)"

tools="$work/repo/experience_docx/tools"
"$python" -m py_compile \
  "$tools/convirctl.py" \
  "$tools/convir_ops_mcp.py" \
  "$tools/route_program_api.py" \
  "$tools/tests/test_convirctl.py" \
  "$tools/validate_engineering_repair.py" \
  "$tools/tests/test_convir_ops_mcp.py" \
  "$tools/tests/test_route_program_api.py" \
  "$tools/tests/test_validate_engineering_repair.py"

stdout="$work/unittest.stdout"
stderr="$work/unittest.stderr"
set +e
PYTHONPATH="$tools" "$python" -m unittest discover \
  -s "$tools/tests" -p 'test_*.py' >"$stdout" 2>"$stderr"
rc=$?
set -e
if [[ $rc -ne 0 ]]; then
  tail -n 160 "$stdout" >&2 || true
  tail -n 160 "$stderr" >&2 || true
  exit "$rc"
fi
tests=$(sed -nE 's/^Ran ([0-9]+) tests?.*/\1/p' "$stderr" | tail -n 1)
[[ $tests =~ ^[0-9]+$ ]]
test "$tests" -ge 111

probe="$work/probe.json"
"$python" - "$tools/convir_ops_mcp.py" "$probe" <<'PY'
import importlib.util, json, sys
path, output = sys.argv[1:]
spec = importlib.util.spec_from_file_location("ops", path)
ops = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ops)
assert ops.SERVER_VERSION == "4.3.2"
assert ops.SCHEMA_VERSION == 4
assert len(ops.TOOLS) == 6
assert set(ops.TOOLS) == {
    "convir_route_plan", "convir_route_start", "convir_route_finish",
    "convir_evidence_list", "convir_evidence_fetch", "convir_git_status",
}
progress = ops.workload_progress(
  '{"phase":"workload","event":"workload_start","completed":0,"total":10}\n'
  '{"R3_A0_PROGRESS":{"completed_units":4,"total_units":10}}\n'
  '{"R3_A1_PROGRESS":{"completed_units":8,"total_units":10}}\n'
)
assert progress == {"completed_units": 8, "total_units": 10}
assert ops.workload_progress(
    '{"message":"completed_units","completed_units":99,"total_units":100}\n'
) == {"completed_units": 0, "total_units": 0}
from route_program_api import write_workload_progress
assert callable(write_workload_progress)
json.dump({
    "server_version": ops.SERVER_VERSION,
    "schema_version": ops.SCHEMA_VERSION,
    "tool_count": len(ops.TOOLS),
    "startup_states": ["RUNNING_VERIFIED", "LAUNCHED_PENDING_VERIFICATION"],
    "engineering_state": "ENGINEERING_AUTO_REPAIR_AUTHORIZED",
}, open(output, "w"), indent=2)
PY

git -C "$work/repo" diff --check
test -z "$(git -C "$work/repo" status --porcelain)"
printf 'CONVIR_OPS_V432_CLOUD_OK candidate=%s tests=%s schema=4 tools=6 model_calls=0 gpu_access=0 protected_data_access=0\n' \
  "$candidate" "$tests"
