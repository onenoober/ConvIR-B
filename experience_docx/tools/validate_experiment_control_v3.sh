#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ! $1 =~ ^[0-9a-f]{40}$ ]]; then
  echo "usage: $0 <candidate-commit>" >&2
  exit 2
fi

candidate=$1
repo_url=git@github.com:onenoober/ConvIR-B.git
python=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
run_root="/tmp/convir-control-v3-${candidate:0:12}"
checkout="$run_root/repository"
status="$run_root/status.txt"
stdout="$run_root/stdout.log"
stderr="$run_root/stderr.log"

if [[ -e $run_root ]]; then
  echo "CONTROL_V3_FAILED existing_run_root=$run_root" >&2
  exit 1
fi

mkdir -p "$run_root"
printf 'state=RUNNING\ncandidate=%s\nmodel_calls=0\n' "$candidate" > "$status"

cleanup() {
  rc=$?
  rm -rf "$checkout"
  if [[ $rc -ne 0 ]]; then
    printf 'state=FAILED\ncandidate=%s\nmodel_calls=0\nexit_code=%s\n' "$candidate" "$rc" > "$status"
  fi
  exit "$rc"
}
trap cleanup EXIT

git clone --quiet --no-checkout --origin github "$repo_url" "$checkout"
git -C "$checkout" checkout --quiet --detach "$candidate"
test "$(git -C "$checkout" rev-parse HEAD)" = "$candidate"

"$python" -m py_compile \
  "$checkout/experience_docx/tools/convir_ops_mcp.py" \
  "$checkout/experience_docx/tools/validate_experiment_card.py" \
  "$checkout/experience_docx/tools/tests/test_convir_ops_mcp.py" \
  "$checkout/experience_docx/tools/tests/test_validate_experiment_card.py"

set +e
"$python" -m unittest discover -s "$checkout/experience_docx/tools/tests" -p 'test_*.py' \
  >"$stdout" 2>"$stderr"
rc=$?
set -e
if [[ $rc -ne 0 ]]; then
  tail -n 80 "$stdout" >&2 || true
  tail -n 80 "$stderr" >&2 || true
  exit "$rc"
fi

"$python" - "$checkout/experience_docx/tools/convir_ops_mcp.py" <<'PY'
import importlib.util
import sys
from pathlib import Path

path = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("convir_ops", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert module.SERVER_VERSION == "3.0.0"
assert module.SCHEMA_VERSION == 3
assert len(module.TOOLS) == 6
assert all(item["max_polls"] * item["interval_seconds"] <= 60 for item in module.MONITOR_PROFILES.values())
print("CONTROL_V3_CONTRACT_OK tools=6 model_calls=0")
PY

git -C "$checkout" diff --check
git -C "$checkout" diff --quiet
printf 'state=PASS\ncandidate=%s\nmodel_calls=0\ntests=14\ntools=6\n' "$candidate" > "$status"
echo "CONTROL_V3_OK candidate=$candidate run_root=$run_root model_calls=0"
