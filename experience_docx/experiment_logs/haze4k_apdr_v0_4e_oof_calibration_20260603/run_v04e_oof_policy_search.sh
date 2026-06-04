#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/root/autodl-tmp/workspace/ConvIR-B-apdr-v0-4b-mapping-triage}
PY=${PY:-/root/miniconda3/envs/convir-cu128/bin/python}
LOG_DIR=${LOG_DIR:-$ROOT/experience_docx/experiment_logs/haze4k_apdr_v0_4e_oof_calibration_20260603}

cd "$ROOT"
"$PY" experience_docx/tools/analyze_haze4k_apdr_v0_4e_oof_policy_search.py \
  --per_image_csv "$LOG_DIR/v04e_oof_candidate_action_per_image_sigma3.csv" \
  --output_dir "$LOG_DIR" \
  --tag sigma3 \
  > "$LOG_DIR/v04e_oof_policy_search_sigma3.log" 2>&1
