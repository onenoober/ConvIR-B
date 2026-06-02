#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/root/autodl-tmp/workspace/ConvIR-B-apdr-convir-v0}
ITS_ROOT="$ROOT/Dehazing/ITS"
PY=${PY:-/root/miniconda3/envs/convir-cu128/bin/python}
DATA_DIR=${DATA_DIR:-/root/autodl-tmp/workspace/Dehaze-Net/dataset/HAZE4K}
PRETRAIN=${PRETRAIN:-/root/autodl-tmp/workspace/ConvIR-B/Dehazing/pretrained_models/haze4k-base.pkl}
LOG_DIR="$ROOT/experience_docx/experiment_logs/haze4k_apdr_v0_20260602"
STATUS="$LOG_DIR/status.txt"

MODEL_NAME=ConvIR-Haze4K-APDR-v0-adapter-only-stop20-seed3407-20260602
TAG=apdr_v0_stop20_seed3407_vs_a0

mkdir -p "$LOG_DIR"
cd "$ITS_ROOT"

log_status() {
  echo "$* $(date --iso-8601=seconds)" | tee -a "$STATUS"
}

log_status "start root=$ROOT"

"$PY" "$ROOT/experience_docx/tools/preflight_haze4k_apdr.py" \
  --data_dir "$DATA_DIR" \
  --checkpoint "$PRETRAIN" \
  --output "$LOG_DIR/preflight_apdr_v0.json" \
  --device cuda \
  --height 256 \
  --width 256 \
  --batch_size 1
log_status "preflight_pass"

BEST="$ITS_ROOT/results/$MODEL_NAME/Training-Results/Best.pkl"
if [[ -f "$BEST" ]]; then
  log_status "skip_existing_train $MODEL_NAME"
else
  log_status "train_start $MODEL_NAME"
  "$PY" main.py \
    --mode train \
    --model_name "$MODEL_NAME" \
    --data Haze4K \
    --data_dir "$DATA_DIR" \
    --version base \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --num_epoch 1000 \
    --stop_epoch 20 \
    --print_freq 50 \
    --num_worker 8 \
    --save_freq 5 \
    --valid_freq 1 \
    --mod_stats_freq 1 \
    --mod_stats_batches 64 \
    --seed 3407 \
    --init_model "$PRETRAIN" \
    --arch apdr \
    --apdr_train_scope apdr_only \
    --apdr_prior_mode rgb_haze \
    --apdr_residual_max 0.04 \
    --apdr_gate_max 0.5 \
    --apdr_gate_init 0.02 \
    --apdr_anchor_lambda 0.0 \
    --apdr_gate_lambda 0.0 \
    --apdr_residual_lambda 0.0 \
    > "$LOG_DIR/train_${MODEL_NAME}.log" 2>&1
  log_status "train_done $MODEL_NAME"
fi

if [[ ! -f "$BEST" ]]; then
  log_status "missing_best_checkpoint $BEST"
  exit 1
fi

log_status "eval_start $TAG"
"$PY" "$ROOT/experience_docx/tools/eval_haze4k_checkpoint_compare.py" \
  --data_dir "$DATA_DIR" \
  --original_checkpoint "$PRETRAIN" \
  --original_arch convir \
  --original_mode original \
  --original_name a0 \
  --candidate_checkpoint "$BEST" \
  --candidate_arch apdr \
  --candidate_mode original \
  --candidate_name apdr \
  --candidate_apdr_prior_mode rgb_haze \
  --candidate_apdr_residual_max 0.04 \
  --candidate_apdr_gate_max 0.5 \
  --candidate_apdr_gate_init 0.02 \
  --output_dir "$LOG_DIR" \
  --tag "$TAG"

"$PY" "$ROOT/experience_docx/tools/analyze_haze4k_delta_buckets.py" \
  --csv "$LOG_DIR/scout_eval_per_image_${TAG}.csv" \
  --candidate_name apdr \
  --output "$LOG_DIR/scout_eval_bucket_analysis_${TAG}.json"

if "$PY" "$ROOT/experience_docx/tools/gate_haze4k_apdr_stop20.py" \
  --compare_json "$LOG_DIR/scout_eval_compare_${TAG}.json" \
  --bucket_json "$LOG_DIR/scout_eval_bucket_analysis_${TAG}.json" \
  --output "$LOG_DIR/gate_${TAG}.json"; then
  log_status "gate_pass $TAG"
else
  log_status "gate_fail_stop $TAG"
  exit 0
fi

log_status "complete $TAG"
