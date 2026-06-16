#!/usr/bin/env bash
set -euo pipefail
ROOT=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v28-nhhaze-official-weights
EVID=$ROOT/experience_docx/experiment_logs/haze4k_v2_9_nhhaze_official_test_alpha_grid_20260616
echo "remote_time=$(date --iso-8601=seconds)"
if tmux has-session -t v29_nhhaze_official_test_alpha_grid 2>/dev/null; then
  echo 'v29_nhhaze_official_test_alpha_grid=ACTIVE'
else
  echo 'v29_nhhaze_official_test_alpha_grid=NOT_ACTIVE'
fi
if [ -f "$EVID/status_v29_nhhaze_official_test_alpha_grid.txt" ]; then
  echo 'status_tail_begin'
  tail -n 80 "$EVID/status_v29_nhhaze_official_test_alpha_grid.txt"
  echo 'status_tail_end'
fi
if [ -f "$EVID/runtime_logs/v29_nhhaze_official_test_alpha_grid.log" ]; then
  echo 'log_tail_begin'
  tail -n 80 "$EVID/runtime_logs/v29_nhhaze_official_test_alpha_grid.log"
  echo 'log_tail_end'
fi
echo 'files_begin'
find "$EVID" -maxdepth 2 -type f -printf '%P\n' | sort
echo 'files_end'
echo 'REMOTE_MONITOR_OK'
