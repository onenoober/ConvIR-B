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
LOG_DIR=${LOG_DIR:-$ROOT/experience_docx/experiment_logs/haze4k_apdr_v0_4b_mapping_triage_20260603}
STATUS="$LOG_DIR/status.txt"
TAG=${TAG:-apdr_v0_4b_mapping_triage_sigma3_seed3407}
BASIS_NUM_IMAGES=${BASIS_NUM_IMAGES:-0}
TRAIN_COUNT=${TRAIN_COUNT:-128}
EVAL_COUNT=${EVAL_COUNT:-256}
LOW_SIZE=${LOW_SIZE:-32}
K_VALUES=${K_VALUES:-8,16,32}
RIDGE_VALUES=${RIDGE_VALUES:-0.0001,0.001,0.01,0.1,1.0,10.0}
PLS_COMPONENTS=${PLS_COMPONENTS:-2,4,8,12,16}
KNN_VALUES=${KNN_VALUES:-1,3,5,9}
MLP_STEPS=${MLP_STEPS:-800}
PROGRESS_FREQ=${PROGRESS_FREQ:-100}

mkdir -p "$LOG_DIR"
cd "$ROOT/Dehazing/ITS"

log_status() {
  echo "$* $(date --iso-8601=seconds)" | tee -a "$STATUS"
}

log_status "start_v0_4b_mapping_triage root=$ROOT tag=$TAG basis_num_images=$BASIS_NUM_IMAGES train_count=$TRAIN_COUNT eval_count=$EVAL_COUNT low_size=$LOW_SIZE k_values=$K_VALUES"

if "$PY" "$ROOT/experience_docx/tools/audit_haze4k_apdr_v0_4b_mapping_triage.py" \
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
  --cv_ridge 1e-3 \
  --ridge_values "$RIDGE_VALUES" \
  --pls_components "$PLS_COMPONENTS" \
  --knn_values "$KNN_VALUES" \
  --folds 5 \
  --mlp_hidden_dim 32 \
  --mlp_steps "$MLP_STEPS" \
  --mlp_learning_rate 5e-4 \
  --mlp_weight_decay 1e-2 \
  --mlp_beta 0.1 \
  --mlp_patience 80 \
  --mlp_grad_clip_norm 1.0 \
  --progress_freq "$PROGRESS_FREQ" \
  > "$LOG_DIR/mapping_triage_${TAG}.log" 2>&1; then
  log_status "complete_v0_4b_mapping_triage $TAG"
else
  log_status "error_v0_4b_mapping_triage $TAG"
  exit 1
fi
