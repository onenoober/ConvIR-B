#!/usr/bin/env bash
set -euo pipefail

BASE=/sda/home/wangyuxin/ConvIR-B
WORK=$BASE/repos/ConvIR-B-v2-23-nopost-oof-gated-lowband-train
EVID=$WORK/experience_docx/experiment_logs/haze4k_v2_23_nopost_oof_gated_lowband_train_20260704
STATUS=$EVID/status.txt
LOG=$EVID/v223_oof_screen.log
SESSION=v223_oof_screen

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
