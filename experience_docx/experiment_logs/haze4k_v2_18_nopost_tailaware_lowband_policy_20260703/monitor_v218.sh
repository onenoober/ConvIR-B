#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v2-18-nopost-tailaware-lowband-policy
EVID="$REMOTE_ROOT/experience_docx/experiment_logs/haze4k_v2_18_nopost_tailaware_lowband_policy_20260703"
STATUS="$EVID/status.txt"

date --iso-8601=seconds
cd "$REMOTE_ROOT"
git branch --show-current
git rev-parse --short HEAD

for session in v218_p1_p2_p3 v218_p4; do
  if tmux has-session -t "$session" 2>/dev/null; then
    printf '%s=ACTIVE\n' "$session"
  else
    printf '%s=NOT_ACTIVE\n' "$session"
  fi
done

if [ -f "$STATUS" ]; then
  tail -n 80 "$STATUS"
else
  echo "STATUS_MISSING"
fi

find "$EVID" -maxdepth 1 -type f -printf '%TY-%Tm-%Td %TH:%TM %s %f\n' | sort | tail -n 80
echo "V218_MONITOR_OK"
