#!/usr/bin/env bash
set -euo pipefail

REMOTE_REPO=/sda/home/wangyuxin/ConvIR-B/repos/generic-monitoring-candidate-48203394e
OUTPUT_PATH=/sda/home/wangyuxin/ConvIR-B/runs/generic_run_monitoring_validation_20260716/candidate-48203394e-r4
SESSION=convir-generic-monitor-candidate-48203394e
EVIDENCE="$REMOTE_REPO/experience_docx/experiment_logs/generic_run_monitoring_validation_20260716"
CLOSEOUT="$EVIDENCE/generic_run_monitoring_validation_closeout.json"
SUMMARY="$EVIDENCE/generic_run_monitoring_validation_summary.json"

active=false
tmux has-session -t "$SESSION" 2>/dev/null && active=true || true
echo "GENERIC_MONITOR_CANDIDATE_STATE active=$active status_exists=$(test -f "$OUTPUT_PATH/status.txt" && echo true || echo false) heartbeat_exists=$(test -f "$OUTPUT_PATH/heartbeat.json" && echo true || echo false) closeout_exists=$(test -f "$CLOSEOUT" && echo true || echo false) summary_exists=$(test -f "$SUMMARY" && echo true || echo false)"
if test -f "$OUTPUT_PATH/status.txt"; then
  echo GENERIC_MONITOR_CANDIDATE_STATUS_BEGIN
  tail -n 12 "$OUTPUT_PATH/status.txt"
  echo GENERIC_MONITOR_CANDIDATE_STATUS_END
fi
for log in launch.log runtime.log; do
  if test -f "$OUTPUT_PATH/$log"; then
    echo "GENERIC_MONITOR_CANDIDATE_LOG_BEGIN name=$log"
    tail -n 80 "$OUTPUT_PATH/$log"
    echo "GENERIC_MONITOR_CANDIDATE_LOG_END name=$log"
  fi
done
if test -f "$CLOSEOUT"; then
  echo "GENERIC_MONITOR_CANDIDATE_CLOSEOUT_SHA256=$(sha256sum "$CLOSEOUT" | awk '{print $1}')"
  echo GENERIC_MONITOR_CANDIDATE_CLOSEOUT_BEGIN
  cat "$CLOSEOUT"
  echo GENERIC_MONITOR_CANDIDATE_CLOSEOUT_END
fi
if test -f "$SUMMARY"; then
  echo "GENERIC_MONITOR_CANDIDATE_SUMMARY_SHA256=$(sha256sum "$SUMMARY" | awk '{print $1}')"
  echo GENERIC_MONITOR_CANDIDATE_SUMMARY_BEGIN
  cat "$SUMMARY"
  echo GENERIC_MONITOR_CANDIDATE_SUMMARY_END
fi
echo GENERIC_MONITOR_CANDIDATE_INSPECTION_OK
