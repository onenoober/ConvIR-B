#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/root/autodl-tmp/workspace/ConvIR-B-apdr-v0-4a-low-field-only}
SESSION=${SESSION:-apdr_v04_sigma3_align_$(date +%Y%m%d_%H%M%S)}
LOG_DIR="$ROOT/experience_docx/experiment_logs/haze4k_apdr_v0_4_sigma3_alignment_20260603"
STATUS="$LOG_DIR/status.txt"

mkdir -p "$LOG_DIR"
cd "$ROOT"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required for the parallel launcher." >&2
  exit 2
fi

echo "launch_sigma3_alignment_parallel session=$SESSION root=$ROOT $(date --iso-8601=seconds)" | tee -a "$STATUS"

tmux new-session -d -s "$SESSION" -n freeparam \
  "cd '$ROOT' && bash experience_docx/experiment_logs/haze4k_apdr_v0_4_sigma3_alignment_20260603/run_apdr_v0_4_freeparam_low_sigma3_32.sh"
tmux new-window -t "$SESSION" -n correctability \
  "cd '$ROOT' && bash experience_docx/experiment_logs/haze4k_apdr_v0_4_sigma3_alignment_20260603/run_apdr_v0_4_correctability_traincalib_sigma3.sh"

echo "tmux_session=$SESSION" | tee -a "$STATUS"
tmux list-windows -t "$SESSION" | tee -a "$STATUS"
