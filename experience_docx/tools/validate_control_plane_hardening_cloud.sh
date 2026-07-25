#!/usr/bin/env bash
set -euo pipefail

on_error() {
  rc=$?
  printf 'CONTROL_PLANE_HARDENING_CLOUD_FAILED line=%s command=%q rc=%s\n' \
    "$1" "$2" "$rc" >&2
  exit "$rc"
}
trap 'on_error "$LINENO" "$BASH_COMMAND"' ERR

branch=codex/control-plane-hardening-v1
github=git@github.com:onenoober/ConvIR-B.git
seed=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-official-arch-anchor
python=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
work=$(mktemp -d /tmp/control-plane-hardening.XXXXXX)
trap 'rm -rf -- "$work"' EXIT

git clone --quiet --shared --no-checkout "$seed" "$work/repo"
git -C "$work/repo" fetch --quiet --no-tags "$github" \
  "+refs/heads/$branch:refs/validation/candidate"
candidate=$(git -C "$work/repo" rev-parse refs/validation/candidate)
git -C "$work/repo" checkout --quiet --detach "$candidate"
test -z "$(git -C "$work/repo" status --porcelain)"

tools=$work/repo/experience_docx/tools
"$python" -m py_compile \
  "$tools/validate_route_ready.py" \
  "$tools/validate_engineering_repair.py" \
  "$tools/route_lifecycle.py" \
  "$tools/tests/test_validate_route_ready.py" \
  "$tools/tests/test_validate_engineering_repair.py" \
  "$tools/tests/test_route_lifecycle.py"

printf 'CONTROL_PLANE_HARDENING_STAGE=targeted_regression\n'
PYTHONPATH="$tools:$tools/tests" "$python" -m unittest -v \
  test_validate_route_ready \
  test_validate_engineering_repair \
  test_route_lifecycle

printf 'CONTROL_PLANE_HARDENING_STAGE=full_control_plane_regression\n'
stdout=$work/unittest.stdout
stderr=$work/unittest.stderr
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
test "$tests" -ge 150

printf 'CONTROL_PLANE_HARDENING_STAGE=surface_and_policy\n'
PYTHONPATH="$tools" "$python" - "$work/repo" <<'PY'
import json
import sys
from pathlib import Path

repo = Path(sys.argv[1])
sys.path.insert(0, str(repo / "experience_docx/tools"))
import convir_ops_mcp as ops
import policy_snapshot

assert ops.SCHEMA_VERSION == 4
assert ops.SUPPORTED_MANIFEST_SCHEMA_VERSIONS == {4, 5, 6}
assert len(ops.TOOLS) == 6
snapshot = json.loads((repo / policy_snapshot.SNAPSHOT_RELPATH).read_bytes())
policy_snapshot.verify_snapshot(
    snapshot, read_bytes=lambda path: (repo / path).read_bytes(),
)
PY

printf 'CONTROL_PLANE_HARDENING_CLOUD_OK candidate=%s tests=%s tools=6 schemas=4,5,6 gpu_access=0 dataset_access=0 protected_data_access=0\n' \
  "$candidate" "$tests"
