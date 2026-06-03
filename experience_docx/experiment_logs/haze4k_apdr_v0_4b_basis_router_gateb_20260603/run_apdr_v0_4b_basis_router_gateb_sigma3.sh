#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/root/autodl-tmp/workspace/ConvIR-B-apdr-v0-4b-derived-lowfield-basis}
PY=${PY:-/root/miniconda3/envs/convir-cu128/bin/python}
DATA_DIR=${DATA_DIR:-/root/autodl-tmp/workspace/Dehaze-Net/dataset/HAZE4K}
SOURCE_APDR_ROOT=${SOURCE_APDR_ROOT:-/root/autodl-tmp/workspace/ConvIR-B-apdr-v0-2rc-oracle-diagnostic}
SELECTOR=${SELECTOR:-$SOURCE_APDR_ROOT/experience_docx/experiment_logs/haze4k_apdr_v0_2rc_frozen_residual_20260603/selector_checkpoint_apdr_v0_2rc_frozen_selector_seed3407.pkl}
SIGMA3_ROOT=${SIGMA3_ROOT:-$ROOT/experience_docx/experiment_logs/haze4k_apdr_v0_4_sigma3_alignment_20260603}
CORR_JSON=${CORR_JSON:-$SIGMA3_ROOT/correctability_traincalib_apdr_v0_4_correctability_traincalib_sigma3_seed3407.json}
CORR_TRAIN_CSV=${CORR_TRAIN_CSV:-$SIGMA3_ROOT/correctability_traincalib_train_oof_apdr_v0_4_correctability_traincalib_sigma3_seed3407.csv}
LOG_DIR=${LOG_DIR:-$ROOT/experience_docx/experiment_logs/haze4k_apdr_v0_4b_basis_router_gateb_20260603}
STATUS="$LOG_DIR/status.txt"
TAG=${TAG:-apdr_v0_4b_basis_router_gateb_sigma3_seed3407}
STAGE_LABEL=${STAGE_LABEL:-APDR-v0.4B basis-only coefficient router Gate B}
ARTIFACT_PREFIX=${ARTIFACT_PREFIX:-basis_router_gateb}
STATUS_NAME=${STATUS_NAME:-v0_4b_basis_router_gateb}
LOG_FILE=${LOG_FILE:-$LOG_DIR/${ARTIFACT_PREFIX}_${TAG}.log}
BASIS_NUM_IMAGES=${BASIS_NUM_IMAGES:-0}
TRAIN_COUNT=${TRAIN_COUNT:-0}
EVAL_COUNT=${EVAL_COUNT:-128}
FIT_COUNT=${FIT_COUNT:-0}
LOW_SIZE=${LOW_SIZE:-32}
K_VALUES=${K_VALUES:-16,32}
STEPS=${STEPS:-2000}
PROGRESS_FREQ=${PROGRESS_FREQ:-100}

mkdir -p "$LOG_DIR"
cd "$ROOT/Dehazing/ITS"

log_status() {
  echo "$* $(date --iso-8601=seconds)" | tee -a "$STATUS"
}

log_status "start_${STATUS_NAME} root=$ROOT tag=$TAG basis_num_images=$BASIS_NUM_IMAGES train_count=$TRAIN_COUNT eval_count=$EVAL_COUNT fit_count=$FIT_COUNT low_size=$LOW_SIZE k_values=$K_VALUES steps=$STEPS"

if "$PY" "$ROOT/experience_docx/tools/overfit_haze4k_apdr_v0_4b_basis_router.py" \
  --data_dir "$DATA_DIR" \
  --selector_checkpoint "$SELECTOR" \
  --correctability_json "$CORR_JSON" \
  --correctability_train_csv "$CORR_TRAIN_CSV" \
  --output_dir "$LOG_DIR" \
  --tag "$TAG" \
  --stage_label "$STAGE_LABEL" \
  --artifact_prefix "$ARTIFACT_PREFIX" \
  --basis_num_images "$BASIS_NUM_IMAGES" \
  --train_count "$TRAIN_COUNT" \
  --eval_count "$EVAL_COUNT" \
  --fit_count "$FIT_COUNT" \
  --seed 3407 \
  --device cuda \
  --pca_device cpu \
  --residual_max 0.04 \
  --kernel_size 31 \
  --sigma 3.0 \
  --low_size "$LOW_SIZE" \
  --k_values "$K_VALUES" \
  --projection_ridge 1e-5 \
  --hidden_dim 64 \
  --steps "$STEPS" \
  --learning_rate 1e-3 \
  --weight_decay 1e-4 \
  --grad_clip_norm 1.0 \
  --field_beta 0.01 \
  --coeff_beta 0.1 \
  --coeff_lambda 0.2 \
  --tv_lambda 0.001 \
  --progress_freq "$PROGRESS_FREQ" \
  > "$LOG_FILE" 2>&1; then
  log_status "complete_${STATUS_NAME} $TAG"
else
  log_status "error_${STATUS_NAME} $TAG"
  exit 1
fi
