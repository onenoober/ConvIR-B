#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v2-25a-nopost-risk-softlabel-scale-distill
EVID=$REMOTE_ROOT/experience_docx/experiment_logs/haze4k_v2_25a_nopost_risk_softlabel_scale_distill_20260704
SESSION=v225a_risk

printf 'remote_time=%s\n' "$(date --iso-8601=seconds)"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  printf 'session=%s ACTIVE\n' "$SESSION"
else
  printf 'session=%s NOT_ACTIVE\n' "$SESSION"
fi
if [ -f "$EVID/status.txt" ]; then
  printf 'STATUS_TAIL\n'
  tail -30 "$EVID/status.txt"
else
  printf 'STATUS_MISSING\n'
fi
if [ -f "$EVID/v225a_risk_distill.log" ]; then
  printf 'LOG_TAIL\n'
  tail -60 "$EVID/v225a_risk_distill.log"
fi
printf 'ARTIFACTS\n'
find "$EVID" -maxdepth 2 -type f -printf '%P %s bytes\n' 2>/dev/null | sort | head -120 || true
printf 'REMOTE_MONITOR_OK\n'
