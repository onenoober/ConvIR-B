#!/usr/bin/env bash
set -euo pipefail

ROOT="/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v27-nhhaze-transfer"
EVID="$ROOT/experience_docx/experiment_logs/haze4k_v2_7_nhhaze_transfer_20260616"

echo "remote_time=$(date -Is)"
if tmux has-session -t v27_nhhaze_transfer 2>/dev/null; then
  echo "v27_nhhaze_transfer=ACTIVE"
else
  echo "v27_nhhaze_transfer=NOT_ACTIVE"
fi

if [ -f "$EVID/status_v27_nhhaze_transfer.txt" ]; then
  echo "main_status_tail_begin"
  tail -n 60 "$EVID/status_v27_nhhaze_transfer.txt"
  echo "main_status_tail_end"
else
  echo "main_status=MISSING"
fi

for status in "$EVID"/shards/status_v27_nhhaze_wdmamba_transfer_shard*.txt; do
  [ -f "$status" ] || continue
  echo "shard_status_begin $(basename "$status")"
  tail -n 12 "$status"
  echo "shard_status_end $(basename "$status")"
done

for log in "$EVID"/runtime_logs/v27_nhhaze_wdmamba_transfer_shard*.log; do
  [ -f "$log" ] || continue
  echo "shard_log_progress_begin $(basename "$log")"
  { grep -E 'progress|V27_NHHAZE_EVAL_SHARD_OK|Traceback|Error|FAILED|out of memory' "$log" || true; } | tail -n 12
  echo "shard_log_progress_end $(basename "$log")"
done

if [ -f "$EVID/v27_nhhaze_wdmamba_transfer_summary.json" ]; then
  echo "summary_exists=yes"
fi

echo "evidence_files_begin"
find "$EVID" -maxdepth 2 -type f | sed "s#^$EVID/##" | sort | tail -n 80
echo "evidence_files_end"
echo "REMOTE_MONITOR_OK"
