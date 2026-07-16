#!/usr/bin/env bash
set -euo pipefail

REMOTE_REPO=/sda/home/wangyuxin/ConvIR-B/repos/generic_run_monitoring_validatio-generic-monitor-validat-2676af9304b040c9
OUTPUT_PATH=/sda/home/wangyuxin/ConvIR-B/runs/generic_run_monitoring_validation_20260716/generic-monitor-validation-r1
CLOSEOUT=$REMOTE_REPO/experience_docx/experiment_logs/generic_run_monitoring_validation_20260716/generic_run_monitoring_validation_closeout.json
SESSION=convir-generic_run_monito-synthetic_-generic-mo-701e6d57725a

repo_exists=false
output_exists=false
closeout_exists=false
session_active=false
test ! -e "$REMOTE_REPO" || repo_exists=true
test ! -e "$OUTPUT_PATH" || output_exists=true
test ! -e "$CLOSEOUT" || closeout_exists=true
tmux has-session -t "$SESSION" 2>/dev/null && session_active=true || true

echo "GENERIC_MONITOR_START_UNKNOWN_STATE repo_exists=$repo_exists output_exists=$output_exists closeout_exists=$closeout_exists session_active=$session_active"
echo GENERIC_MONITOR_START_UNKNOWN_INSPECTION_OK
