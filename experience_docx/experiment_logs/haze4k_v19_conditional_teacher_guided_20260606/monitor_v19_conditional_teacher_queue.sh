#!/usr/bin/env bash
set -euo pipefail

WORK=${WORK:-/root/autodl-tmp/workspace/ConvIR-B-v1-9-conditional-teacher}
EVID=${EVID:-$WORK/experience_docx/experiment_logs/haze4k_v19_conditional_teacher_guided_20260606}
STATUS=$EVID/status.txt

printf 'remote_time=%s\n' "$(date -Is)"
printf 'work=%s\n' "$WORK"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true
fi
if command -v tmux >/dev/null 2>&1; then
  if tmux has-session -t v19_cond_teacher 2>/dev/null; then
    printf 'v19_cond_teacher=ACTIVE\n'
  else
    printf 'v19_cond_teacher=NOT_ACTIVE\n'
  fi
  tmux ls 2>/dev/null || true
fi
if [ -f "$STATUS" ]; then
  printf '%s\n' '--- status tail ---'
  tail -n 100 "$STATUS"
else
  printf 'status_missing=%s\n' "$STATUS"
fi
printf '%s\n' '--- recent evidence ---'
find "$EVID" -maxdepth 3 -type f -printf '%TY-%Tm-%Td %TH:%TM %p\n' 2>/dev/null | sort | tail -n 80 || true
printf 'REMOTE_MONITOR_OK\n'
