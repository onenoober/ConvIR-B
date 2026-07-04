#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v2-21-nopost-safety-calibrated-lowband-replay
EVID="$REMOTE_ROOT/experience_docx/experiment_logs/haze4k_v2_21_nopost_safety_calibrated_lowband_replay_20260704"
STATUS="$EVID/status.txt"

cd "$REMOTE_ROOT"
printf 'remote_time=%s\n' "$(date --iso-8601=seconds)"
printf 'branch=%s\n' "$(git branch --show-current)"
printf 'commit=%s\n' "$(git rev-parse --short HEAD)"
printf 'status_short_count=%s\n' "$(git status --short | wc -l)"
if tmux has-session -t v221_safety_replay 2>/dev/null; then
  printf '%s\n' 'tmux_v221_safety_replay=ACTIVE'
else
  printf '%s\n' 'tmux_v221_safety_replay=NOT_ACTIVE'
fi
if [ -f "$STATUS" ]; then
  printf '%s\n' 'status_tail_start'
  tail -40 "$STATUS"
  printf '%s\n' 'status_tail_end'
else
  printf '%s\n' 'status_missing'
fi
printf '%s\n' 'recent_evidence_start'
find "$EVID" -maxdepth 1 -type f -printf '%TY-%Tm-%Td %TH:%TM %f\n' | sort | tail -40
printf '%s\n' 'recent_evidence_end'
printf '%s\n' 'REMOTE_MONITOR_OK'
