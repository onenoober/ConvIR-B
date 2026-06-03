#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/root/autodl-tmp/workspace/ConvIR-B-apdr-v0-3-shed-diagnostics}
PY=${PY:-/root/miniconda3/envs/convir-cu128/bin/python}
DATA_DIR=${DATA_DIR:-/root/autodl-tmp/workspace/Dehaze-Net/dataset/HAZE4K}
SOURCE_APDR_ROOT=${SOURCE_APDR_ROOT:-/root/autodl-tmp/workspace/ConvIR-B-apdr-v0-2rc-oracle-diagnostic}
SELECTOR=${SELECTOR:-$SOURCE_APDR_ROOT/experience_docx/experiment_logs/haze4k_apdr_v0_2rc_frozen_residual_20260603/selector_checkpoint_apdr_v0_2rc_frozen_selector_seed3407.pkl}
LOG_DIR="$ROOT/experience_docx/experiment_logs/haze4k_apdr_v0_3_crop_mask_mismatch_20260603"
STATUS="$LOG_DIR/status.txt"
TAG=${TAG:-apdr_v0_3_crop_mask_mismatch_128x4_seed3407}

mkdir -p "$LOG_DIR"
cd "$ROOT/Dehazing/ITS"

log_status() {
  echo "$* $(date --iso-8601=seconds)" | tee -a "$STATUS"
}

log_status "start_crop_mask_mismatch root=$ROOT selector=$SELECTOR tag=$TAG"

if "$PY" "$ROOT/experience_docx/tools/audit_haze4k_apdr_crop_mask_mismatch.py" \
  --data_dir "$DATA_DIR" \
  --selector_checkpoint "$SELECTOR" \
  --output_dir "$LOG_DIR" \
  --tag "$TAG" \
  --num_images 128 \
  --crops_per_image 4 \
  --crop_size 256 \
  --seed 3407 \
  --device cuda \
  --progress_freq 16 \
  > "$LOG_DIR/audit_${TAG}.log" 2>&1; then
  log_status "gate_pass_crop_mask_mismatch $TAG"
else
  log_status "gate_fail_crop_mask_mismatch $TAG"
fi

log_status "complete_crop_mask_mismatch $TAG"
