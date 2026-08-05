#!/bin/bash
set -euo pipefail

BRANCH=codex/engineering-diagnose-p1
REMOTE_URL=git@github.com:onenoober/ConvIR-B.git
BASE=/sda/home/wangyuxin/ConvIR-B
PYTHON=$BASE/envs/convir-cu121/bin/python
SEED=$BASE/repos/ConvIR-B-official-arch-anchor

REMOTE_LINE=$(/usr/bin/git ls-remote "$REMOTE_URL" "refs/heads/$BRANCH")
read -r CANDIDATE REMOTE_REF <<< "$REMOTE_LINE"
test "$REMOTE_REF" = "refs/heads/$BRANCH"
[[ "$CANDIDATE" =~ ^[0-9a-f]{40}$ ]]

RUN_ROOT=$BASE/runs/engineering-diagnose-p1/$CANDIDATE
REPO=$BASE/repos/engineering-diagnose-p1-$CANDIDATE
STATUS=$RUN_ROOT/status.txt
mkdir -p "$RUN_ROOT"
test ! -e "$STATUS"
printf 'state=PREPARING\ncommit=%s\n' "$CANDIDATE" > "$STATUS"

on_exit() {
  code=$?
  if test "$code" -ne 0; then
    printf 'state=FAILED_ENGINEERING\nexit_code=%s\n' "$code" >> "$STATUS"
  fi
}
trap on_exit EXIT

test -d "$SEED/.git" || test -f "$SEED/HEAD"
test ! -e "$REPO"
/usr/bin/git clone --quiet --no-checkout --reference-if-able "$SEED" "$REMOTE_URL" "$REPO"
/usr/bin/git -C "$REPO" checkout --quiet --detach "$CANDIDATE"
test -z "$(/usr/bin/git -C "$REPO" status --porcelain)"

export PYTHONPATH=$REPO/experience_docx/tools:$REPO/experience_docx/tools/tests
"$PYTHON" -m unittest \
  test_convir_ops_mcp \
  test_validate_route_ready \
  test_policy_snapshot

"$PYTHON" - "$REPO" <<'PY'
import pathlib
import sys

repo = pathlib.Path(sys.argv[1])
sys.path.insert(0, str(repo / "experience_docx" / "tools"))
import convir_ops_mcp as ops

assert ops.SERVER_VERSION == "5.11.0"
assert len(ops.TOOLS) == 6
finish = ops.TOOLS["convir_route_finish"]
resolutions = finish["inputSchema"]["properties"]["engineering_failure_resolution"]["enum"]
assert resolutions == ["diagnose", "repair", "archive", "discard", "finalize"]
PY

printf 'state=COMPLETED_GATE_PASS\ncommit=%s\nmarker=ENGINEERING_DIAGNOSE_P1_CLOUD_OK\n' \
  "$CANDIDATE" > "$STATUS"
printf 'ENGINEERING_DIAGNOSE_P1_CLOUD_OK candidate=%s tests=3 ops_version=5.11.0 tools=6 experiment_access=0 protected_data_access=0\n' \
  "$CANDIDATE"
