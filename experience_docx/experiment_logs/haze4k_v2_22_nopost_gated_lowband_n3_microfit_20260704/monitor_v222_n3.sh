#!/usr/bin/env bash
set -euo pipefail

BASE=/sda/home/wangyuxin/ConvIR-B
WORK=$BASE/repos/ConvIR-B-v2-22-nopost-gated-lowband-n3-microfit
EVID=$WORK/experience_docx/experiment_logs/haze4k_v2_22_nopost_gated_lowband_n3_microfit_20260704
STATUS=$EVID/status.txt
LOG=$EVID/v222_n3_microfit.log
SESSION=v222_n3_microfit

date --iso-8601=seconds
cd "$WORK"
echo "branch=$(git branch --show-current)"
echo "commit=$(git rev-parse --short HEAD)"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "${SESSION}=ACTIVE"
else
  echo "${SESSION}=NOT_ACTIVE"
fi
echo "--- status tail ---"
if [[ -f "$STATUS" ]]; then tail -40 "$STATUS"; else echo "STATUS_MISSING"; fi
echo "--- log tail ---"
if [[ -f "$LOG" ]]; then tail -80 "$LOG"; else echo "LOG_MISSING"; fi
echo "--- evidence files ---"
find "$EVID" -maxdepth 2 -type f -printf "%TY-%Tm-%Td %TH:%TM %s %p\n" | sort | tail -80
echo "REMOTE_MONITOR_OK"
