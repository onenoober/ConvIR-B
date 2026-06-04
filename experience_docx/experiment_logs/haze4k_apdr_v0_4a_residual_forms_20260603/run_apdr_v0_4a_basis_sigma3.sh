#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/root/autodl-tmp/workspace/ConvIR-B-apdr-v0-4a-low-field-only}
PY=${PY:-/root/miniconda3/envs/convir-cu128/bin/python}
DATA_DIR=${DATA_DIR:-/root/autodl-tmp/workspace/Dehaze-Net/dataset/HAZE4K}
SOURCE_APDR_ROOT=${SOURCE_APDR_ROOT:-/root/autodl-tmp/workspace/ConvIR-B-apdr-v0-2rc-oracle-diagnostic}
SELECTOR=${SELECTOR:-$SOURCE_APDR_ROOT/experience_docx/experiment_logs/haze4k_apdr_v0_2rc_frozen_residual_20260603/selector_checkpoint_apdr_v0_2rc_frozen_selector_seed3407.pkl}
SIGMA3_ROOT=${SIGMA3_ROOT:-$ROOT/experience_docx/experiment_logs/haze4k_apdr_v0_4_sigma3_alignment_20260603}
CORR_JSON=${CORR_JSON:-$SIGMA3_ROOT/correctability_traincalib_apdr_v0_4_correctability_traincalib_sigma3_seed3407.json}
CORR_TRAIN_CSV=${CORR_TRAIN_CSV:-$SIGMA3_ROOT/correctability_traincalib_train_oof_apdr_v0_4_correctability_traincalib_sigma3_seed3407.csv}
LOG_DIR="$ROOT/experience_docx/experiment_logs/haze4k_apdr_v0_4a_residual_forms_20260603"
STATUS="$LOG_DIR/status.txt"
TAG=${TAG:-apdr_v0_4a_basis_sigma3_seed3407}

mkdir -p "$LOG_DIR"
cd "$ROOT/Dehazing/ITS"

log_status() {
  echo "$* $(date --iso-8601=seconds)" | tee -a "$STATUS"
}

log_status "start_basis_sigma3 root=$ROOT selector=$SELECTOR tag=$TAG"

if "$PY" "$ROOT/experience_docx/tools/overfit_haze4k_apdr_v0_4a_lowfield.py" \
  --data_dir "$DATA_DIR" \
  --selector_checkpoint "$SELECTOR" \
  --correctability_json "$CORR_JSON" \
  --correctability_train_csv "$CORR_TRAIN_CSV" \
  --output_dir "$LOG_DIR" \
  --tag "$TAG" \
  --model_type basis \
  --num_images 32 \
  --steps 800 \
  --learning_rate 1e-3 \
  --weight_decay 1e-4 \
  --grad_clip_norm 1.0 \
  --seed 3407 \
  --device cuda \
  --hidden_channels 48 \
  --low_size 32 \
  --num_bases 16 \
  --kernel_size 31 \
  --sigma 3.0 \
  --crop_size 4096 \
  --lowpass_lambda 0.05 \
  --tv_lambda 0.001 \
  --progress_freq 50 \
  > "$LOG_DIR/lowfield_overfit32_${TAG}.log" 2>&1; then
  log_status "gate_pass_basis_sigma3 $TAG"
else
  log_status "gate_fail_basis_sigma3 $TAG"
fi

log_status "complete_basis_sigma3 $TAG"
