#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/root/autodl-tmp/workspace/ConvIR-B-apdr-v0-3-shed-diagnostics}
PY=${PY:-/root/miniconda3/envs/convir-cu128/bin/python}
DATA_DIR=${DATA_DIR:-/root/autodl-tmp/workspace/Dehaze-Net/dataset/HAZE4K}
SOURCE_APDR_ROOT=${SOURCE_APDR_ROOT:-/root/autodl-tmp/workspace/ConvIR-B-apdr-v0-2rc-oracle-diagnostic}
SELECTOR=${SELECTOR:-$SOURCE_APDR_ROOT/experience_docx/experiment_logs/haze4k_apdr_v0_2rc_frozen_residual_20260603/selector_checkpoint_apdr_v0_2rc_frozen_selector_seed3407.pkl}
LOG_DIR="$ROOT/experience_docx/experiment_logs/haze4k_apdr_v0_3_residual_source_oracle_20260603"
STATUS="$LOG_DIR/status.txt"
TAG=${TAG:-apdr_v0_3_residual_source_oracle_seed3407}

mkdir -p "$LOG_DIR"
cd "$ROOT/Dehazing/ITS"

log_status() {
  echo "$* $(date --iso-8601=seconds)" | tee -a "$STATUS"
}

log_status "start_residual_source_oracle root=$ROOT selector=$SELECTOR tag=$TAG"

"$PY" "$ROOT/experience_docx/tools/oracle_haze4k_apdr_residual_source_ablation.py" \
  --data_dir "$DATA_DIR" \
  --selector_checkpoint "$SELECTOR" \
  --output_dir "$LOG_DIR" \
  --tag "$TAG" \
  --device cuda \
  --kernel_size 31 \
  --sigma 7.0 \
  --residual_max 0.04 \
  --progress_freq 100 \
  > "$LOG_DIR/oracle_residual_source_${TAG}.log" 2>&1

log_status "complete_residual_source_oracle $TAG"
