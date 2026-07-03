#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v2-17-nopost-lowband-alignment-tail-audit
EVID="$REMOTE_ROOT/experience_docx/experiment_logs/haze4k_v2_17_nopost_lowband_alignment_tail_audit_20260703"
STATUS="$EVID/status.txt"

printf 'remote_time=%s\n' "$(date --iso-8601=seconds)"
for s in v217_r1 v217_r2 v217_r3; do
  if tmux has-session -t "$s" 2>/dev/null; then
    printf '%s=ACTIVE\n' "$s"
  else
    printf '%s=NOT_ACTIVE\n' "$s"
  fi
done

if [ -f "$STATUS" ]; then
  printf '%s\n' 'status_tail'
  tail -n 40 "$STATUS"
else
  printf 'status_missing=%s\n' "$STATUS"
fi

printf '%s\n' 'evidence_files'
find "$EVID" -maxdepth 1 -type f -printf '%TY-%Tm-%Td %TH:%TM %f\n' 2>/dev/null | sort | tail -n 40 || true

printf '%s\n' 'log_tails'
for log in "$EVID"/v217_r1_postmortem.log "$EVID"/v217_r2_capacity_ladder.log "$EVID"/v217_r3_objective_tail_audit.log; do
  if [ -f "$log" ]; then
    printf 'LOG=%s\n' "$(basename "$log")"
    tail -n 20 "$log"
  fi
done

printf 'REMOTE_MONITOR_OK\n'
