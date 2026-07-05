#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v2-29-nopost-ilfrb-acs-safe-oof-action-bank-calibration
EVID=$REMOTE_ROOT/experience_docx/experiment_logs/haze4k_v2_29_nopost_ilfrb_acs_safe_oof_action_bank_calibration_20260705
SESSION=${SESSION:-v229_p2a}

echo "remote_time=$(date --iso-8601=seconds)"
cd "$REMOTE_ROOT"
echo "branch=$(git branch --show-current)"
echo "commit=$(git rev-parse --short HEAD)"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux_${SESSION}=ACTIVE"
else
  echo "tmux_${SESSION}=NOT_ACTIVE"
fi
echo "status_tail:"
tail -n 50 "$EVID/status.txt" || true
echo "recent_log_tail:"
tail -n 40 "$EVID/v229_p2a_diagnostics.log" || true
echo "evidence_files:"
find "$EVID" -maxdepth 1 -type f -printf '%f %s bytes\n' | sort
echo "REMOTE_MONITOR_OK"
