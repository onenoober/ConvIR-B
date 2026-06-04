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
LOG_DIR="$ROOT/experience_docx/experiment_logs/haze4k_apdr_v0_4b_derived_basis_20260603"
STATUS="$LOG_DIR/status.txt"
TAG=${TAG:-apdr_v0_4b_derived_basis_sigma3_seed3407}
NUM_IMAGES=${NUM_IMAGES:-0}
LOW_SIZES=${LOW_SIZES:-32,48}
K_VALUES=${K_VALUES:-4,8,16,32,48}
PCA_DEVICE=${PCA_DEVICE:-cpu}
PROGRESS_FREQ=${PROGRESS_FREQ:-100}

mkdir -p "$LOG_DIR"
cd "$ROOT/Dehazing/ITS"

log_status() {
  echo "$* $(date --iso-8601=seconds)" | tee -a "$STATUS"
}

log_status "start_v0_4b_derived_basis root=$ROOT tag=$TAG num_images=$NUM_IMAGES low_sizes=$LOW_SIZES k_values=$K_VALUES"

if "$PY" "$ROOT/experience_docx/tools/audit_haze4k_apdr_v0_4b_derived_basis.py" \
  --data_dir "$DATA_DIR" \
  --selector_checkpoint "$SELECTOR" \
  --correctability_json "$CORR_JSON" \
  --correctability_train_csv "$CORR_TRAIN_CSV" \
  --output_dir "$LOG_DIR" \
  --tag "$TAG" \
  --num_images "$NUM_IMAGES" \
  --seed 3407 \
  --device cuda \
  --pca_device "$PCA_DEVICE" \
  --residual_max 0.04 \
  --kernel_size 31 \
  --sigma 3.0 \
  --low_sizes "$LOW_SIZES" \
  --k_values "$K_VALUES" \
  --projection_ridge 1e-5 \
  --coeff_ridge 1e-3 \
  --router_ridge 1e-5 \
  --folds 5 \
  --router_probe_count 32 \
  --progress_freq "$PROGRESS_FREQ" \
  > "$LOG_DIR/derived_basis_${TAG}.log" 2>&1; then
  log_status "complete_v0_4b_derived_basis $TAG"
else
  log_status "error_v0_4b_derived_basis $TAG"
  exit 1
fi
