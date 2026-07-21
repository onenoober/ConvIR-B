#!/bin/bash
set -euo pipefail

BRANCH=codex/p0-p1-research-governance-fastpath-20260721
REMOTE_URL=git@github.com:onenoober/ConvIR-B.git
BASE=/sda/home/wangyuxin/ConvIR-B
PYTHON=$BASE/envs/convir-cu121/bin/python
SEED=$BASE/repos/ConvIR-B-official-arch-anchor

REMOTE_LINE=$(/usr/bin/git ls-remote "$REMOTE_URL" "refs/heads/$BRANCH")
read -r RULES_COMMIT REMOTE_REF <<< "$REMOTE_LINE"
test "$REMOTE_REF" = "refs/heads/$BRANCH"
[[ "$RULES_COMMIT" =~ ^[0-9a-f]{40}$ ]]

RUN_ROOT=$BASE/runs/p0-p1-governance-validation/$RULES_COMMIT/policy-snapshot
REPO=$BASE/repos/p0-p1-policy-snapshot-$RULES_COMMIT
STATUS=$RUN_ROOT/status.txt
SNAPSHOT=$RUN_ROOT/AI_POLICY_SNAPSHOT.json
mkdir -p "$RUN_ROOT"
if test -e "$STATUS"; then
  printf 'existing snapshot status: %s\n' "$(tr '\n' ' ' < "$STATUS")"
  exit 73
fi
printf 'state=PREPARING\ncommit=%s\n' "$RULES_COMMIT" > "$STATUS"

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
/usr/bin/git -C "$REPO" checkout --quiet --detach "$RULES_COMMIT"
test -z "$(/usr/bin/git -C "$REPO" status --porcelain)"
PYTHONPATH=$REPO/experience_docx/tools "$PYTHON" \
  $REPO/experience_docx/tools/policy_snapshot.py \
  --repo "$REPO" --rules-commit "$RULES_COMMIT" --write >/dev/null
cp "$REPO/experience_docx/AI_POLICY_SNAPSHOT.json" "$SNAPSHOT"
PYTHONPATH=$REPO/experience_docx/tools "$PYTHON" \
  $REPO/experience_docx/tools/policy_snapshot.py \
  --repo "$REPO" --rules-commit "$RULES_COMMIT" --check >/dev/null
printf 'state=COMPLETED_GATE_PASS\ncommit=%s\nmarker=POLICY_SNAPSHOT_CLOUD_OK\n' \
  "$RULES_COMMIT" > "$STATUS"
echo POLICY_SNAPSHOT_JSON_BEGIN
cat "$SNAPSHOT"
echo POLICY_SNAPSHOT_JSON_END
echo POLICY_SNAPSHOT_CLOUD_OK
