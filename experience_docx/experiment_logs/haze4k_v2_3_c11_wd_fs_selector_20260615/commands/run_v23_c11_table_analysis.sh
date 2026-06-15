#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v23-c11-wd-fs-selector
PY=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
EVID="$REMOTE_ROOT/experience_docx/experiment_logs/haze4k_v2_3_c11_wd_fs_selector_20260615"
STATUS="$EVID/status_c11.txt"
LOG="$EVID/runtime_logs/v23_c11_table_analysis.log"

mkdir -p "$EVID/runtime_logs" "$EVID/commands"

{
  echo "state=RUNNING_AUDIT"
  echo "run_id=v23_c11_table_analysis"
  echo "start_time=$(date --iso-8601=seconds)"
  echo "locked_test_touched=false"
  echo "locked_per_image_read=false"
  echo "distillation=false"
  echo "remote_root=$REMOTE_ROOT"
  echo "python=$PY"
  echo "source_branch=$(git -C "$REMOTE_ROOT" branch --show-current)"
  echo "source_commit=$(git -C "$REMOTE_ROOT" rev-parse HEAD)"
} > "$STATUS"

set +e
"$PY" "$REMOTE_ROOT/experience_docx/tools/analyze_haze4k_v23_c11_wd_fs_selector.py" \
  --repo-root "$REMOTE_ROOT" \
  --out-dir "$EVID" \
  2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e

{
  echo "finish_time=$(date --iso-8601=seconds)"
  echo "exit_code=$rc"
  if [ "$rc" -eq 0 ]; then
    echo "state=COMPLETED_ANALYSIS"
    echo "C11_TABLE_ANALYSIS_OK"
  else
    echo "state=FAILED_COMMAND"
    echo "C11_TABLE_ANALYSIS_FAILED"
  fi
} >> "$STATUS"

exit "$rc"
