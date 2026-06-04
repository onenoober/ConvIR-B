#!/usr/bin/env bash
set -euo pipefail

WORK=${WORK:-/root/autodl-tmp/workspace/ConvIR-B-dpga-tail-control}
LOG_DIR=$WORK/experience_docx/experiment_logs/haze4k_dpga_tail_control_20260604
DIAG_DIR=$LOG_DIR/runtime_diagnostics
MODULE_CSV=$DIAG_DIR/dpga_module_ablation_best_final.csv
SCALE_CSV=$DIAG_DIR/dpga_scale_sweep_best_final.csv
WATCH_LOG=$LOG_DIR/watch_and_launch_dpga_v1_1.log
STATUS=$LOG_DIR/status.txt
SLEEP_SECONDS=${SLEEP_SECONDS:-60}
MAX_WAIT_SECONDS=${MAX_WAIT_SECONDS:-21600}

mkdir -p "$LOG_DIR"
start_ts=$(date +%s)

{
  echo "watch_and_launch_start $(date --iso-8601=seconds)"
  echo "work=$WORK"
  echo "module_csv=$MODULE_CSV"
  echo "scale_csv=$SCALE_CSV"
  while true; do
    now=$(date +%s)
    elapsed=$((now - start_ts))
    if [[ -s "$MODULE_CSV" && -s "$SCALE_CSV" ]]; then
      echo "diagnostics_csv_ready elapsed=${elapsed}s $(date --iso-8601=seconds)"
      break
    fi
    if [[ $elapsed -ge $MAX_WAIT_SECONDS ]]; then
      echo "diagnostics_wait_timeout elapsed=${elapsed}s" >&2
      exit 5
    fi
    pid=""
    if [[ -f "$LOG_DIR/dpga_runtime_diagnostics.pid" ]]; then
      pid=$(cat "$LOG_DIR/dpga_runtime_diagnostics.pid" || true)
    fi
    if [[ -n "$pid" ]] && ! ps -p "$pid" >/dev/null 2>&1; then
      echo "diagnostics_pid_exited_without_csv pid=$pid elapsed=${elapsed}s" >&2
      find "$DIAG_DIR" -maxdepth 1 -type f -print 2>/dev/null || true
      exit 6
    fi
    echo "waiting_for_diagnostics elapsed=${elapsed}s pid=${pid:-unknown} $(date --iso-8601=seconds)"
    sleep "$SLEEP_SECONDS"
  done

  bash "$LOG_DIR/run_decide_dpga_v1_1_training.sh"
  bash "$LOG_DIR/run_dpga_v1_1_tail_control_train.sh"
  echo "watch_and_launch_done $(date --iso-8601=seconds)"
} 2>&1 | tee "$WATCH_LOG"

echo "watch_and_launch_log=$WATCH_LOG" | tee -a "$STATUS"
