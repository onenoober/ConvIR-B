#!/bin/bash
set -euo pipefail

BRANCH=codex/p0-p1-research-governance-fastpath-20260721
REMOTE_URL=git@github.com:onenoober/ConvIR-B.git
BASE=/sda/home/wangyuxin/ConvIR-B
PYTHON=$BASE/envs/convir-cu121/bin/python
SEED=$BASE/repos/ConvIR-B-official-arch-anchor

REMOTE_LINE=$(/usr/bin/git ls-remote "$REMOTE_URL" "refs/heads/$BRANCH")
read -r REMOTE_COMMIT REMOTE_REF <<< "$REMOTE_LINE"
test "$REMOTE_REF" = "refs/heads/$BRANCH"
[[ "$REMOTE_COMMIT" =~ ^[0-9a-f]{40}$ ]]

RUN_ROOT=$BASE/runs/p0-p1-governance-validation/$REMOTE_COMMIT/spec-compiler
REPO=$BASE/repos/p0-p1-governance-validation-$REMOTE_COMMIT-spec
STATUS=$RUN_ROOT/status.txt
LOG=$RUN_ROOT/validation.log
mkdir -p "$RUN_ROOT"
if test -e "$STATUS"; then
  printf 'existing validation status: %s\n' "$(tr '\n' ' ' < "$STATUS")"
  exit 73
fi
printf 'state=PREPARING\nbranch=%s\ncommit=%s\n' "$BRANCH" "$REMOTE_COMMIT" > "$STATUS"

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
/usr/bin/git -C "$REPO" checkout --quiet --detach "$REMOTE_COMMIT"
test "$(/usr/bin/git -C "$REPO" rev-parse HEAD)" = "$REMOTE_COMMIT"
test -z "$(/usr/bin/git -C "$REPO" status --porcelain)"
printf 'state=RUNNING\nbranch=%s\ncommit=%s\n' "$BRANCH" "$REMOTE_COMMIT" > "$STATUS"

cd "$REPO"
"$PYTHON" -m unittest -v \
  experience_docx.tools.tests.test_experiment_spec_compiler \
  experience_docx.tools.tests.test_research_program_contract \
  experience_docx.tools.tests.test_convir_ops_v5_final_slim \
  experience_docx.tools.tests.test_validate_route_ready \
  experience_docx.tools.tests.test_route_runtime_contract 2>&1 | tee "$LOG"

grep -Fq 'test_schema6_plan_boundary_recompiles_and_rejects_derived_drift' "$LOG"
grep -Fq 'test_historical_manifest_schemas_remain_supported' "$LOG"
grep -Fq 'OK' "$LOG"
printf 'state=COMPLETED_GATE_PASS\nbranch=%s\ncommit=%s\nmarker=P0_SPEC_COMPILER_CLOUD_OK\n' \
  "$BRANCH" "$REMOTE_COMMIT" > "$STATUS"
echo P0_SPEC_COMPILER_CLOUD_OK
