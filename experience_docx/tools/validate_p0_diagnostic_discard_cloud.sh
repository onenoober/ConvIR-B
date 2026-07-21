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

RUN_ROOT=$BASE/runs/p0-p1-governance-validation/$REMOTE_COMMIT/diagnostic-discard
REPO=$BASE/repos/p0-p1-governance-validation-$REMOTE_COMMIT-diagnostic
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
PYTHONPATH="$REPO/experience_docx/tools:$REPO/experience_docx/tools/tests" \
"$PYTHON" -m unittest -v \
  experience_docx.tools.tests.test_convir_ops_mcp \
  experience_docx.tools.tests.test_convir_ops_v5_final_slim \
  experience_docx.tools.tests.test_route_lifecycle \
  experience_docx.tools.tests.test_migrate_convir_ops_v43_state 2>&1 | tee "$LOG"

grep -Fq 'test_receipt_bound_engineering_discard_requires_verified_no_data_touch' "$LOG"
grep -Fq 'test_scientific_terminal_cannot_be_discarded' "$LOG"
grep -Fq 'test_engineering_diagnostic_is_bounded_redacted_and_control_only' "$LOG"
grep -Fq 'test_engineering_archive_resolution_unlocks_compact_evidence' "$LOG"
grep -Fq 'OK' "$LOG"
printf 'state=COMPLETED_GATE_PASS\nbranch=%s\ncommit=%s\nmarker=P0_DIAGNOSTIC_DISCARD_CLOUD_OK\n' \
  "$BRANCH" "$REMOTE_COMMIT" > "$STATUS"
echo P0_DIAGNOSTIC_DISCARD_CLOUD_OK
