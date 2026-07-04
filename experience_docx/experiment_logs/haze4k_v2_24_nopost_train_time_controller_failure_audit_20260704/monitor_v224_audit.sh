#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v2-24-nopost-train-time-controller-failure-audit
EVID=$REMOTE_ROOT/experience_docx/experiment_logs/haze4k_v2_24_nopost_train_time_controller_failure_audit_20260704
SESSION=v224_audit

printf 'remote_time=%s\n' "$(date --iso-8601=seconds)"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  printf 'session=%s ACTIVE\n' "$SESSION"
else
  printf 'session=%s NOT_ACTIVE\n' "$SESSION"
fi
if [ -f "$EVID/status.txt" ]; then
  printf 'STATUS_TAIL\n'
  tail -20 "$EVID/status.txt"
else
  printf 'STATUS_MISSING\n'
fi
if [ -f "$EVID/v224_audit.log" ]; then
  printf 'LOG_TAIL\n'
  tail -40 "$EVID/v224_audit.log"
fi
printf 'ARTIFACTS\n'
find "$EVID" -maxdepth 1 -type f -printf '%f %s bytes\n' 2>/dev/null | sort || true
printf 'REMOTE_MONITOR_OK\n'
