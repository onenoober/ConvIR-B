#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/root/autodl-tmp/workspace/ConvIR-B-apdr-v0-4a-low-field-only}
PY=${PY:-/root/miniconda3/envs/convir-cu128/bin/python}
DATA_DIR=${DATA_DIR:-/root/autodl-tmp/workspace/Dehaze-Net/dataset/HAZE4K}
SOURCE_APDR_ROOT=${SOURCE_APDR_ROOT:-/root/autodl-tmp/workspace/ConvIR-B-apdr-v0-2rc-oracle-diagnostic}
SELECTOR=${SELECTOR:-$SOURCE_APDR_ROOT/experience_docx/experiment_logs/haze4k_apdr_v0_2rc_frozen_residual_20260603/selector_checkpoint_apdr_v0_2rc_frozen_selector_seed3407.pkl}
LOG_DIR="$ROOT/experience_docx/experiment_logs/haze4k_apdr_v0_4_sigma3_alignment_20260603"
STATUS="$LOG_DIR/status.txt"
TAG=${TAG:-apdr_v0_4_correctability_traincalib_sigma3_seed3407}

mkdir -p "$LOG_DIR"
cd "$ROOT/Dehazing/ITS"

log_status() {
  echo "$* $(date --iso-8601=seconds)" | tee -a "$STATUS"
}

log_status "start_correctability_traincalib_sigma3 root=$ROOT selector=$SELECTOR tag=$TAG"

if "$PY" "$ROOT/experience_docx/tools/audit_haze4k_apdr_v0_4_correctability_traincalib.py" \
  --data_dir "$DATA_DIR" \
  --selector_checkpoint "$SELECTOR" \
  --output_dir "$LOG_DIR" \
  --tag "$TAG" \
  --train_max_images 0 \
  --test_max_images 0 \
  --folds 5 \
  --seed 3407 \
  --device cuda \
  --kernel_size 31 \
  --sigma 3.0 \
  --steps 800 \
  --progress_freq 200 \
  > "$LOG_DIR/correctability_traincalib_${TAG}.log" 2>&1; then
  log_status "gate_pass_correctability_traincalib_sigma3 $TAG"
else
  log_status "gate_fail_correctability_traincalib_sigma3 $TAG"
fi

log_status "complete_correctability_traincalib_sigma3 $TAG"
