#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/root/autodl-tmp/workspace/ConvIR-B-apdr-v0-4b-derived-lowfield-basis}
SESSION=${SESSION:-apdr_v04b_gatec_t128_$(date +%Y%m%d_%H%M%S)}
LOG_DIR=${LOG_DIR:-$ROOT/experience_docx/experiment_logs/haze4k_apdr_v0_4b_basis_router_gatec_train128_minival_20260603}
RUN_SCRIPT="$ROOT/experience_docx/experiment_logs/haze4k_apdr_v0_4b_basis_router_gatec_train128_minival_20260603/run_apdr_v0_4b_basis_router_gatec_train128_minival_sigma3.sh"
EXIT_FILE="$LOG_DIR/tmux_exit_${SESSION}.txt"

mkdir -p "$LOG_DIR"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 2
fi

tmux new-session -d -s "$SESSION" "bash $RUN_SCRIPT; code=\$?; printf 'exit_code=%s\n' \"\$code\" > $EXIT_FILE; exit \$code"
echo "$SESSION"
