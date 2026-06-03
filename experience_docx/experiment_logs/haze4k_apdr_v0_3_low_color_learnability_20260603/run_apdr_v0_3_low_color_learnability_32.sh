#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/root/autodl-tmp/workspace/ConvIR-B-apdr-v0-3-shed-diagnostics}
PY=${PY:-/root/miniconda3/envs/convir-cu128/bin/python}
DATA_DIR=${DATA_DIR:-/root/autodl-tmp/workspace/Dehaze-Net/dataset/HAZE4K}
SOURCE_APDR_ROOT=${SOURCE_APDR_ROOT:-/root/autodl-tmp/workspace/ConvIR-B-apdr-v0-2rc-oracle-diagnostic}
SELECTOR=${SELECTOR:-$SOURCE_APDR_ROOT/experience_docx/experiment_logs/haze4k_apdr_v0_2rc_frozen_residual_20260603/selector_checkpoint_apdr_v0_2rc_frozen_selector_seed3407.pkl}
LOG_DIR="$ROOT/experience_docx/experiment_logs/haze4k_apdr_v0_3_low_color_learnability_20260603"
STATUS="$LOG_DIR/status.txt"
TAG=${TAG:-apdr_v0_3_low_color_learnability_32_seed3407}

mkdir -p "$LOG_DIR"
cd "$ROOT/Dehazing/ITS"

log_status() {
  echo "$* $(date --iso-8601=seconds)" | tee -a "$STATUS"
}

log_status "start_low_color_learnability root=$ROOT selector=$SELECTOR tag=$TAG"

if "$PY" "$ROOT/experience_docx/tools/overfit_haze4k_apdr_low_color_learnability.py" \
  --data_dir "$DATA_DIR" \
  --selector_checkpoint "$SELECTOR" \
  --output_dir "$LOG_DIR" \
  --tag "$TAG" \
  --num_images 32 \
  --steps 500 \
  --learning_rate 1e-3 \
  --grad_clip_norm 1.0 \
  --seed 3407 \
  --device cuda \
  --low_size 32 \
  --hidden 32 \
  --progress_freq 50 \
  > "$LOG_DIR/overfit_${TAG}.log" 2>&1; then
  log_status "gate_pass_low_color_learnability $TAG"
else
  log_status "gate_fail_low_color_learnability $TAG"
fi

log_status "complete_low_color_learnability $TAG"
