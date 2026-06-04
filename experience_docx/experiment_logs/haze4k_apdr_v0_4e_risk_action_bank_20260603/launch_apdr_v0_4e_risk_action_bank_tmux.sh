#!/usr/bin/env bash
set -euo pipefail

SESSION=${SESSION:-apdr_v04e_risk_action_bank}
ROOT=${ROOT:-/root/autodl-tmp/workspace/ConvIR-B-apdr-v0-4b-mapping-triage}
LOG_DIR=${LOG_DIR:-$ROOT/experience_docx/experiment_logs/haze4k_apdr_v0_4e_risk_action_bank_20260603}
mkdir -p "$LOG_DIR"

tmux new-session -d -s "$SESSION" \
  "cd '$ROOT' && bash experience_docx/experiment_logs/haze4k_apdr_v0_4e_risk_action_bank_20260603/run_apdr_v0_4e_risk_action_bank_sigma3.sh; echo exit_code=\$? > '$LOG_DIR/tmux_exit_apdr_v04e_risk_action_bank_$(date +%Y%m%d).txt'"

echo "launched tmux session: $SESSION"
echo "tail: tail -f $LOG_DIR/status.txt"
