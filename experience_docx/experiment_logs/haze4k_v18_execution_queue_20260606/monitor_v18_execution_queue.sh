#!/usr/bin/env bash
set -euo pipefail

WORK=${WORK:-/root/autodl-tmp/workspace/ConvIR-B-v1-8-execution-queue}
EVID=${EVID:-$WORK/experience_docx/experiment_logs/haze4k_v18_execution_queue_20260606}
STATUS=$EVID/status.txt

printf 'remote_time=%s\n' "$(date -Is)"
printf 'work=%s\n' "$WORK"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true
else
  printf 'nvidia_smi_missing\n'
fi
if command -v tmux >/dev/null 2>&1; then
  if tmux has-session -t v18_execution_queue 2>/dev/null; then
    printf 'v18_execution_queue=ACTIVE\n'
  else
    printf 'v18_execution_queue=NOT_ACTIVE\n'
  fi
  if tmux has-session -t v18_domain_adaptation_q5 2>/dev/null; then
    printf 'v18_domain_adaptation_q5=ACTIVE\n'
  else
    printf 'v18_domain_adaptation_q5=NOT_ACTIVE\n'
  fi
  if tmux has-session -t v18_eval_repair 2>/dev/null; then
    printf 'v18_eval_repair=ACTIVE\n'
  else
    printf 'v18_eval_repair=NOT_ACTIVE\n'
  fi
  tmux ls 2>/dev/null || true
else
  printf 'tmux_missing\n'
fi
if [ -f "$STATUS" ]; then
  printf '%s\n' '--- status tail ---'
  tail -n 80 "$STATUS"
else
  printf 'status_missing=%s\n' "$STATUS"
fi
if [ -f "$WORK/experience_docx/tools/summarize_haze4k_v18_queue_progress.py" ]; then
  PROGRESS_OUT=$EVID/v18_progress
  mkdir -p "$PROGRESS_OUT"
  printf '%s\n' '--- progress summary ---'
  /root/miniconda3/envs/convir-cu128/bin/python \
    "$WORK/experience_docx/tools/summarize_haze4k_v18_queue_progress.py" \
    --evidence_root "$EVID" \
    --seeds 3407 2026 929 123 777 1701 2222 3141 4242 5151 \
    --output_json "$PROGRESS_OUT/v18_progress_summary.json" \
    --output_csv "$PROGRESS_OUT/v18_progress_seeds.csv" || true
else
  printf 'progress_summary_tool_missing=%s\n' "$WORK/experience_docx/tools/summarize_haze4k_v18_queue_progress.py"
fi
printf '%s\n' '--- recent evidence ---'
find "$EVID" -maxdepth 3 -type f -printf '%TY-%Tm-%Td %TH:%TM %p\n' 2>/dev/null | sort | tail -n 80 || true
printf 'REMOTE_MONITOR_OK\n'
