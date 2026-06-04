#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-/root/miniconda3/envs/convir-cu128/bin/python}
WORK=${WORK:-/root/autodl-tmp/workspace/ConvIR-B-dpga-tail-control}
LOG_DIR=$WORK/experience_docx/experiment_logs/haze4k_dpga_tail_control_20260604
EVAL_DIR=$LOG_DIR/v1_1_val_inner_eval
OUT_DIR=$LOG_DIR/v1_1_failure_analysis
STATUS=$LOG_DIR/status.txt

mkdir -p "$OUT_DIR"
"$PY" "$WORK/experience_docx/tools/analyze_haze4k_dpga_val_inner_failure.py" \
  --best_per_image_csv "$EVAL_DIR/scout_eval_per_image_v1_1_val_inner_best_vs_a0.csv" \
  --final_per_image_csv "$EVAL_DIR/scout_eval_per_image_v1_1_val_inner_final_vs_a0.csv" \
  --gate_json "$EVAL_DIR/gate_dpga_v1_1_val_inner.json" \
  --output_json "$OUT_DIR/dpga_v1_1_val_inner_failure_analysis.json" \
  --output_md "$OUT_DIR/dpga_v1_1_val_inner_failure_analysis.md" \
  --output_group_csv "$OUT_DIR/dpga_v1_1_val_inner_failure_groups.csv" \
  2>&1 | tee "$OUT_DIR/analyze_dpga_v1_1_val_failure.log"

{
  echo "v1_1_failure_analysis_json=$OUT_DIR/dpga_v1_1_val_inner_failure_analysis.json"
  echo "v1_1_failure_analysis_md=$OUT_DIR/dpga_v1_1_val_inner_failure_analysis.md"
  echo "v1_1_failure_groups_csv=$OUT_DIR/dpga_v1_1_val_inner_failure_groups.csv"
} | tee -a "$STATUS"
