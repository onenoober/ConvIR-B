#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/root/autodl-tmp/workspace/ConvIR-B-apdr-v0-3-shed-diagnostics}
PY=${PY:-/root/miniconda3/envs/convir-cu128/bin/python}
DATA_DIR=${DATA_DIR:-/root/autodl-tmp/workspace/Dehaze-Net/dataset/HAZE4K}
SOURCE_APDR_ROOT=${SOURCE_APDR_ROOT:-/root/autodl-tmp/workspace/ConvIR-B-apdr-v0-2rc-oracle-diagnostic}
SELECTOR=${SELECTOR:-$SOURCE_APDR_ROOT/experience_docx/experiment_logs/haze4k_apdr_v0_2rc_frozen_residual_20260603/selector_checkpoint_apdr_v0_2rc_frozen_selector_seed3407.pkl}
LOG_DIR="$ROOT/experience_docx/experiment_logs/haze4k_apdr_v0_3_correctability_proxy_20260603"
STATUS="$LOG_DIR/status.txt"
TAG=${TAG:-apdr_v0_3_correctability_proxy_test_seed3407}

mkdir -p "$LOG_DIR"
cd "$ROOT/Dehazing/ITS"

log_status() {
  echo "$* $(date --iso-8601=seconds)" | tee -a "$STATUS"
}

log_status "start_correctability_proxy root=$ROOT selector=$SELECTOR tag=$TAG"

if "$PY" "$ROOT/experience_docx/tools/audit_haze4k_apdr_correctability_proxy.py" \
  --data_dir "$DATA_DIR" \
  --selector_checkpoint "$SELECTOR" \
  --output_dir "$LOG_DIR" \
  --tag "$TAG" \
  --split test \
  --train_fraction 0.70 \
  --seed 3407 \
  --device cuda \
  --kernel_size 31 \
  --sigma 7.0 \
  --positive_gain 0.10 \
  --negative_gain 0.01 \
  --hidden_dim 32 \
  --steps 800 \
  --learning_rate 1e-3 \
  --weight_decay 1e-3 \
  --progress_freq 100 \
  > "$LOG_DIR/audit_${TAG}.log" 2>&1; then
  log_status "gate_pass_correctability_proxy $TAG"
else
  log_status "gate_fail_correctability_proxy $TAG"
fi

log_status "complete_correctability_proxy $TAG"
