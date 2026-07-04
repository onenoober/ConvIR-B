#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v2-20-nopost-midfinal-context-lowband-learnability
EVID="$REMOTE_ROOT/experience_docx/experiment_logs/haze4k_v2_20_nopost_midfinal_context_lowband_learnability_20260703"
STATUS="$EVID/status.txt"

printf 'remote_time=%s\n' "$(date --iso-8601=seconds)"
cd "$REMOTE_ROOT"
printf 'branch=%s\n' "$(git branch --show-current)"
printf 'commit=%s\n' "$(git rev-parse --short HEAD)"
printf 'workspace_status_lines=%s\n' "$(git status --short | wc -l)"

for s in v220_p0 v220_o3ctx; do
  if tmux has-session -t "$s" 2>/dev/null; then
    printf '%s=ACTIVE\n' "$s"
  else
    printf '%s=NOT_ACTIVE\n' "$s"
  fi
done

if [ -f "$STATUS" ]; then
  tail -60 "$STATUS"
else
  printf 'status_file=MISSING\n'
fi

find "$EVID" -maxdepth 1 -type f -printf '%TY-%Tm-%Td %TH:%TM %s %f\n' | sort | tail -40
printf 'REMOTE_MONITOR_OK\n'
