#!/usr/bin/env bash
set -euo pipefail

BRANCH=codex/generic-run-monitoring-20260716
EXPECTED_COMMIT=4bbd031546f031dcc350d7a039bdb28ecf6856e2
EXPECTED_RUNNER_SHA=21df0834d7de2f4102714decc5a5f4ce092274b633cdfdfd383011a83499ae06
GITHUB_URL=git@github.com:onenoober/ConvIR-B.git
REMOTE_REPO=/sda/home/wangyuxin/ConvIR-B/repos/generic-monitoring-candidate-4bbd03154
OUTPUT_PATH=/sda/home/wangyuxin/ConvIR-B/runs/generic_run_monitoring_validation_20260716/candidate-4bbd03154-r2
SESSION=convir-generic-monitor-candidate-4bbd03154
RUNNER=experience_docx/tools/run_generic_run_monitoring_validation.sh
CLOSEOUT="$REMOTE_REPO/experience_docx/experiment_logs/generic_run_monitoring_validation_20260716/generic_run_monitoring_validation_closeout.json"

created_repo=false
created_output=false
cleanup_prelaunch() {
  rc=$?
  if "$created_output"; then rm -rf -- "$OUTPUT_PATH"; fi
  if "$created_repo"; then rm -rf -- "$REMOTE_REPO"; fi
  exit "$rc"
}
trap cleanup_prelaunch ERR

test ! -e "$REMOTE_REPO"
test ! -e "$OUTPUT_PATH"
test ! -e "$CLOSEOUT"
tmux has-session -t "$SESSION" 2>/dev/null && { echo GENERIC_MONITOR_CANDIDATE_SESSION_CONFLICT; exit 73; } || true
created_repo=true
git clone --quiet --no-checkout --origin github --single-branch --branch "$BRANCH" "$GITHUB_URL" "$REMOTE_REPO"
git -C "$REMOTE_REPO" checkout --quiet --detach "$EXPECTED_COMMIT"
test "$(git -C "$REMOTE_REPO" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git -C "$REMOTE_REPO" status --porcelain)"
test "$(sha256sum "$REMOTE_REPO/$RUNNER" | awk '{print $1}')" = "$EXPECTED_RUNNER_SHA"
test -x /sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
mkdir -p "$OUTPUT_PATH"
created_output=true
tmux new-session -d -s "$SESSION" \
  "env EXPECTED_ROUTE_COMMIT='$EXPECTED_COMMIT' RUNNER_SHA256='$EXPECTED_RUNNER_SHA' MODE='synthetic_validation' REMOTE_REPO='$REMOTE_REPO' RUN_ROOT='/sda/home/wangyuxin/ConvIR-B/runs/generic_run_monitoring_validation_20260716' OUTPUT_PATH='$OUTPUT_PATH' RUN_ID='candidate-4bbd03154-r2' OUTPUT_ID='candidate-4bbd03154-r2' GPU='' bash '$REMOTE_REPO/$RUNNER' >>'$OUTPUT_PATH/launch.log' 2>&1"
trap - ERR
echo "GENERIC_MONITOR_CANDIDATE_LAUNCH_OK commit=$EXPECTED_COMMIT session=$SESSION output=$OUTPUT_PATH"
