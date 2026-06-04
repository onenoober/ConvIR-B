#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/root/autodl-tmp/workspace/ConvIR-B-apdr-v0-2rc-oracle-diagnostic}
PY=${PY:-/root/miniconda3/envs/convir-cu128/bin/python}
DATA_DIR=${DATA_DIR:-/root/autodl-tmp/workspace/Dehaze-Net/dataset/HAZE4K}
PRETRAIN=${PRETRAIN:-/root/autodl-tmp/workspace/ConvIR-B/Dehazing/pretrained_models/haze4k-base.pkl}
LOG_DIR="$ROOT/experience_docx/experiment_logs/haze4k_apdr_v0_2rc_oracle_diagnostic_20260603"
STATUS="$LOG_DIR/status.txt"
TAG=${TAG:-apdr_v0_2rc_oracle_diagnostic_seed3407}

mkdir -p "$LOG_DIR"
cd "$ROOT/Dehazing/ITS"

log_status() {
  echo "$* $(date --iso-8601=seconds)" | tee -a "$STATUS"
}

log_status "start_oracle_diagnostic root=$ROOT tag=$TAG"

"$PY" "$ROOT/experience_docx/tools/preflight_haze4k_apdr_v0_2rc_budget.py" \
  --data_dir "$DATA_DIR" \
  --checkpoint "$PRETRAIN" \
  --output_dir "$LOG_DIR" \
  --tag "$TAG" \
  --device cuda \
  --seed 3407 \
  --global_epochs 5 \
  --spatial_epochs 3 \
  --global_batch_size 4 \
  --spatial_batch_size 8 \
  --global_resize 384 \
  --num_worker 8 \
  --global_learning_rate 2e-4 \
  --spatial_learning_rate 2e-4 \
  --global_train_batches_per_epoch 0 \
  --spatial_train_batches_per_epoch 0 \
  --calibration_images 0 \
  --budget_calibration_images 0 \
  --loss_eval_images 256 \
  --pixel_samples_per_image 2048 \
  --progress_freq 100 \
  --run_oracle_on_replay_fail \
  > "$LOG_DIR/oracle_diagnostic_${TAG}.log" 2>&1

log_status "complete_oracle_diagnostic tag=$TAG"
