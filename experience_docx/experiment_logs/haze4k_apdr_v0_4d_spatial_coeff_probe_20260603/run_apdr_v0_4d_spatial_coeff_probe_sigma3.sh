#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/root/autodl-tmp/workspace/ConvIR-B-apdr-v0-4b-mapping-triage}
PY=${PY:-/root/miniconda3/envs/convir-cu128/bin/python}
DATA_DIR=${DATA_DIR:-/root/autodl-tmp/workspace/Dehaze-Net/dataset/HAZE4K}
SOURCE_APDR_ROOT=${SOURCE_APDR_ROOT:-/root/autodl-tmp/workspace/ConvIR-B-apdr-v0-2rc-oracle-diagnostic}
SELECTOR=${SELECTOR:-$SOURCE_APDR_ROOT/experience_docx/experiment_logs/haze4k_apdr_v0_2rc_frozen_residual_20260603/selector_checkpoint_apdr_v0_2rc_frozen_selector_seed3407.pkl}
SIGMA3_ROOT=${SIGMA3_ROOT:-$ROOT/experience_docx/experiment_logs/haze4k_apdr_v0_4_sigma3_alignment_20260603}
CORR_JSON=${CORR_JSON:-$SIGMA3_ROOT/correctability_traincalib_apdr_v0_4_correctability_traincalib_sigma3_seed3407.json}
CORR_TRAIN_CSV=${CORR_TRAIN_CSV:-$SIGMA3_ROOT/correctability_traincalib_train_oof_apdr_v0_4_correctability_traincalib_sigma3_seed3407.csv}
LOG_DIR=${LOG_DIR:-$ROOT/experience_docx/experiment_logs/haze4k_apdr_v0_4d_spatial_coeff_probe_20260603}
STATUS="$LOG_DIR/status.txt"
TAG=${TAG:-apdr_v0_4d_spatial_coeff_probe_sigma3_seed3407}
BASIS_NUM_IMAGES=${BASIS_NUM_IMAGES:-0}
TRAIN_COUNT=${TRAIN_COUNT:-128}
EVAL_COUNT=${EVAL_COUNT:-256}
LOW_SIZE=${LOW_SIZE:-32}
K_VALUES=${K_VALUES:-16,32}
RIDGE_VALUES=${RIDGE_VALUES:-0.01,0.1,1.0,10.0}
PLS_COMPONENTS=${PLS_COMPONENTS:-4,8,16}
KNN_VALUES=${KNN_VALUES:-5,9}
SPATIAL_GRID=${SPATIAL_GRID:-4}
SPATIAL_PROJ_CHANNELS=${SPATIAL_PROJ_CHANNELS:-8}
HOOK_PATHS=${HOOK_PATHS:-Encoder.0,Encoder.1,Encoder.2,Decoder.0,Decoder.1,Decoder.2,Convs.0,Convs.1,APDR_1.context}
PROGRESS_FREQ=${PROGRESS_FREQ:-100}

mkdir -p "$LOG_DIR"
cd "$ROOT/Dehazing/ITS"

log_status() {
  echo "$* $(date --iso-8601=seconds)" | tee -a "$STATUS"
}

log_status "start_v0_4d_spatial_coeff_probe root=$ROOT tag=$TAG basis_num_images=$BASIS_NUM_IMAGES train_count=$TRAIN_COUNT eval_count=$EVAL_COUNT low_size=$LOW_SIZE k_values=$K_VALUES grid=$SPATIAL_GRID proj=$SPATIAL_PROJ_CHANNELS"

if "$PY" "$ROOT/experience_docx/tools/audit_haze4k_apdr_v0_4d_spatial_coeff_probe.py" \
  --data_dir "$DATA_DIR" \
  --selector_checkpoint "$SELECTOR" \
  --correctability_json "$CORR_JSON" \
  --correctability_train_csv "$CORR_TRAIN_CSV" \
  --output_dir "$LOG_DIR" \
  --tag "$TAG" \
  --basis_num_images "$BASIS_NUM_IMAGES" \
  --train_count "$TRAIN_COUNT" \
  --eval_count "$EVAL_COUNT" \
  --seed 3407 \
  --device cuda \
  --pca_device cpu \
  --residual_max 0.04 \
  --kernel_size 31 \
  --sigma 3.0 \
  --low_size "$LOW_SIZE" \
  --k_values "$K_VALUES" \
  --projection_ridge 1e-5 \
  --ridge_values "$RIDGE_VALUES" \
  --pls_components "$PLS_COMPONENTS" \
  --knn_values "$KNN_VALUES" \
  --spatial_grid "$SPATIAL_GRID" \
  --spatial_proj_channels "$SPATIAL_PROJ_CHANNELS" \
  --hook_paths "$HOOK_PATHS" \
  --progress_freq "$PROGRESS_FREQ" \
  > "$LOG_DIR/spatial_coeff_probe_${TAG}.log" 2>&1; then
  log_status "complete_v0_4d_spatial_coeff_probe $TAG"
else
  log_status "error_v0_4d_spatial_coeff_probe $TAG"
  exit 1
fi
