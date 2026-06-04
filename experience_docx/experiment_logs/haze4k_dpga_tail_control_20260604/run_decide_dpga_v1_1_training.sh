#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-/root/miniconda3/envs/convir-cu128/bin/python}
WORK=${WORK:-/root/autodl-tmp/workspace/ConvIR-B-dpga-tail-control}
LOG_DIR=$WORK/experience_docx/experiment_logs/haze4k_dpga_tail_control_20260604
DIAG_DIR=$LOG_DIR/runtime_diagnostics
DECISION_DIR=$LOG_DIR/v1_1_decision
DECISION_JSON=$DECISION_DIR/dpga_v1_1_training_decision.json
DECISION_MD=$DECISION_DIR/dpga_v1_1_training_decision.md
LOG=$LOG_DIR/decide_dpga_v1_1_training.log
STATUS=$LOG_DIR/status.txt

mkdir -p "$DECISION_DIR"

{
  echo "decide_dpga_v1_1_training_start $(date --iso-8601=seconds)"
  echo "work=$WORK"
  echo "diagnostics=$DIAG_DIR"
  "$PY" "$WORK/experience_docx/tools/decide_haze4k_dpga_v1_1_training.py" \
    --diagnostics_dir "$DIAG_DIR" \
    --output_json "$DECISION_JSON" \
    --output_md "$DECISION_MD"
  echo "decide_dpga_v1_1_training_done $(date --iso-8601=seconds)"
} 2>&1 | tee "$LOG"

{
  echo "v1_1_decision_json=$DECISION_JSON"
  echo "v1_1_decision_md=$DECISION_MD"
} | tee -a "$STATUS"
