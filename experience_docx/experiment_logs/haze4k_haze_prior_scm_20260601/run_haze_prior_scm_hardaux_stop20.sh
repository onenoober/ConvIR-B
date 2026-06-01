#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/workspace/ConvIR-B-haze-prior-scm
ITS_ROOT="$ROOT/Dehazing/ITS"
DATA_DIR=/root/autodl-tmp/workspace/Dehaze-Net/dataset/HAZE4K
PY=/root/miniconda3/envs/py310/bin/python
LOG_DIR="$ROOT/experience_docx/experiment_logs/haze4k_haze_prior_scm_20260601"

CONTROL_MODEL=ConvIR-Haze4K-original-hardaux-stop20-seed3407-20260601
CANDIDATE_MODEL=ConvIR-Haze4K-hazeprior-scm-hardaux-stop20-seed3407-20260601
CONTROL_DIR="$ITS_ROOT/results/$CONTROL_MODEL"
CANDIDATE_DIR="$ITS_ROOT/results/$CANDIDATE_MODEL"
CONTROL_BEST="$CONTROL_DIR/Training-Results/Best.pkl"
CONTROL_LAST="$CONTROL_DIR/Training-Results/Final.pkl"
CANDIDATE_BEST="$CANDIDATE_DIR/Training-Results/Best.pkl"
CANDIDATE_LAST="$CANDIDATE_DIR/Training-Results/Final.pkl"

prepare_run_dir() {
  local run_dir=$1
  if [ -f "$run_dir/Training-Results/Final.pkl" ]; then
    return
  fi
  if [ -d "$run_dir" ]; then
    local archived="${run_dir}.partial.$(date +%Y%m%d-%H%M%S)"
    echo "archiving partial run $run_dir -> $archived"
    mv "$run_dir" "$archived"
  fi
}

run_train() {
  local model_name=$1
  local scm_mode=$2
  local stats_freq=$3
  local log_name=$4
  local run_dir="$ITS_ROOT/results/$model_name"

  if [ -f "$run_dir/Training-Results/Final.pkl" ]; then
    echo "reuse completed $model_name"
    return
  fi

  prepare_run_dir "$run_dir"
  cd "$ITS_ROOT"
  "$PY" main.py \
    --mode train \
    --model_name "$model_name" \
    --data Haze4K \
    --data_dir "$DATA_DIR" \
    --version base \
    --fam_mode original \
    --scm_mode "$scm_mode" \
    --loss_mode hard_aux \
    --hard_aux_lambda 0.25 \
    --hard_aux_warmup_epochs 3 \
    --hard_aux_ramp_epochs 5 \
    --batch_size 8 \
    --learning_rate 4e-4 \
    --num_epoch 1000 \
    --stop_epoch 20 \
    --print_freq 50 \
    --num_worker 8 \
    --save_freq 5 \
    --valid_freq 1 \
    --seed 3407 \
    --scm_stats_freq "$stats_freq" \
    --scm_stats_batches 64 \
    > "$LOG_DIR/$log_name" 2>&1
}

mkdir -p "$LOG_DIR"

{
  echo "start $(date --iso-8601=seconds)"
  echo "host $(hostname)"
  echo "root $ROOT"
  echo "data_dir $DATA_DIR"
  "$PY" - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
PY

  echo "running real-batch preflight $(date --iso-8601=seconds)"
  cd "$ROOT"
  "$PY" experience_docx/tools/preflight_haze4k_haze_prior_scm.py \
    --data_dir "$DATA_DIR" \
    --batch_size 8 \
    --num_worker 0 \
    --loss_mode hard_aux \
    --hard_aux_lambda 0.25 \
    --output "$LOG_DIR/preflight_real_batch_seed3407.json"

  echo "running matched original-SCM hard_aux control $(date --iso-8601=seconds)"
  run_train "$CONTROL_MODEL" original 0 original_hardaux_train_stop20_seed3407.log
  echo "done matched control $(date --iso-8601=seconds)"

  echo "running haze-prior SCM hard_aux candidate $(date --iso-8601=seconds)"
  run_train "$CANDIDATE_MODEL" haze_prior 5 haze_prior_scm_hardaux_train_stop20_seed3407.log
  echo "done candidate $(date --iso-8601=seconds)"

  echo "running Best checkpoint compare $(date --iso-8601=seconds)"
  "$PY" "$ROOT/experience_docx/tools/eval_haze4k_checkpoint_compare.py" \
    --data_dir "$DATA_DIR" \
    --original_checkpoint "$CONTROL_BEST" \
    --original_mode original \
    --original_scm_mode original \
    --candidate_checkpoint "$CANDIDATE_BEST" \
    --candidate_mode original \
    --candidate_scm_mode haze_prior \
    --candidate_name haze_prior_scm_hardaux \
    --output_dir "$LOG_DIR" \
    --tag seed3407_stop20_best

  "$PY" "$ROOT/experience_docx/tools/analyze_haze4k_delta_buckets.py" \
    --csv "$LOG_DIR/scout_eval_per_image_seed3407_stop20_best.csv" \
    --candidate_name haze_prior_scm_hardaux \
    --output "$LOG_DIR/scout_eval_bucket_analysis_seed3407_stop20_best.json"

  echo "running Last checkpoint compare $(date --iso-8601=seconds)"
  "$PY" "$ROOT/experience_docx/tools/eval_haze4k_checkpoint_compare.py" \
    --data_dir "$DATA_DIR" \
    --original_checkpoint "$CONTROL_LAST" \
    --original_mode original \
    --original_scm_mode original \
    --candidate_checkpoint "$CANDIDATE_LAST" \
    --candidate_mode original \
    --candidate_scm_mode haze_prior \
    --candidate_name haze_prior_scm_hardaux \
    --output_dir "$LOG_DIR" \
    --tag seed3407_stop20_last

  "$PY" "$ROOT/experience_docx/tools/analyze_haze4k_delta_buckets.py" \
    --csv "$LOG_DIR/scout_eval_per_image_seed3407_stop20_last.csv" \
    --candidate_name haze_prior_scm_hardaux \
    --output "$LOG_DIR/scout_eval_bucket_analysis_seed3407_stop20_last.json"

  echo "complete $(date --iso-8601=seconds)"
} | tee "$LOG_DIR/status.txt"
