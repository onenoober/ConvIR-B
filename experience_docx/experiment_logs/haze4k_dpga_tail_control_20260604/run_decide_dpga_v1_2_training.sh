#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-/root/miniconda3/envs/convir-cu128/bin/python}
WORK=${WORK:-/root/autodl-tmp/workspace/ConvIR-B-dpga-tail-control}
LOG_DIR=$WORK/experience_docx/experiment_logs/haze4k_dpga_tail_control_20260604
ANALYSIS_JSON=$LOG_DIR/v1_1_failure_analysis/dpga_v1_1_val_inner_failure_analysis.json
PREV_DECISION_JSON=$LOG_DIR/v1_1_decision/dpga_v1_1_training_decision.json
OUT_DIR=$LOG_DIR/v1_2_decision
STATUS=$LOG_DIR/status.txt

mkdir -p "$OUT_DIR"
"$PY" "$WORK/experience_docx/tools/decide_haze4k_dpga_v1_2_training.py" \
  --failure_analysis_json "$ANALYSIS_JSON" \
  --previous_decision_json "$PREV_DECISION_JSON" \
  --output_json "$OUT_DIR/dpga_v1_2_training_decision.json" \
  --output_md "$OUT_DIR/dpga_v1_2_training_decision.md" \
  2>&1 | tee "$OUT_DIR/decide_dpga_v1_2_training.log"

{
  echo "v1_2_decision_json=$OUT_DIR/dpga_v1_2_training_decision.json"
  echo "v1_2_decision_md=$OUT_DIR/dpga_v1_2_training_decision.md"
} | tee -a "$STATUS"
