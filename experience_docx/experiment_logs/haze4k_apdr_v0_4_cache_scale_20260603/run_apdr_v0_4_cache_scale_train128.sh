#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/root/autodl-tmp/workspace/ConvIR-B-apdr-v0-4-cclf-diagnostics}
PY=${PY:-/root/miniconda3/envs/convir-cu128/bin/python}
DATA_DIR=${DATA_DIR:-/root/autodl-tmp/workspace/Dehaze-Net/dataset/HAZE4K}
SOURCE_APDR_ROOT=${SOURCE_APDR_ROOT:-/root/autodl-tmp/workspace/ConvIR-B-apdr-v0-2rc-oracle-diagnostic}
SELECTOR=${SELECTOR:-$SOURCE_APDR_ROOT/experience_docx/experiment_logs/haze4k_apdr_v0_2rc_frozen_residual_20260603/selector_checkpoint_apdr_v0_2rc_frozen_selector_seed3407.pkl}
LOG_DIR="$ROOT/experience_docx/experiment_logs/haze4k_apdr_v0_4_cache_scale_20260603"
STATUS="$LOG_DIR/status.txt"
TAG=${TAG:-apdr_v0_4_cache_scale_train128_seed3407}
CACHE_DIR=${CACHE_DIR:-$LOG_DIR/tensor_cache/train128}

mkdir -p "$LOG_DIR" "$CACHE_DIR"
cd "$ROOT/Dehazing/ITS"

log_status() {
  echo "$* $(date --iso-8601=seconds)" | tee -a "$STATUS"
}

log_status "start_cache_scale root=$ROOT selector=$SELECTOR tag=$TAG"

"$PY" "$ROOT/experience_docx/tools/audit_haze4k_apdr_v0_4_cache_and_scale.py" \
  --data_dir "$DATA_DIR" \
  --selector_checkpoint "$SELECTOR" \
  --output_dir "$LOG_DIR" \
  --cache_dir "$CACHE_DIR" \
  --tag "$TAG" \
  --split train \
  --max_images 128 \
  --write_cache 1 \
  --sigmas 3,5,7,11,15 \
  --crop_size 256 \
  --seed 3407 \
  --device cuda \
  --progress_freq 16 \
  > "$LOG_DIR/cache_scale_${TAG}.log" 2>&1

log_status "complete_cache_scale $TAG"
