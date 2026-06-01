#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/workspace/ConvIR-B-hardfreq-loss
ITS_ROOT="$ROOT/Dehazing/ITS"
DATA_DIR=/root/autodl-tmp/workspace/Dehaze-Net/dataset/HAZE4K
PY=/root/miniconda3/envs/convir-cu128/bin/python
RUN_ROOT="$ITS_ROOT/results/ConvIR-Haze4K-hardfreq-loss-stop20-20260601"
LOG_DIR="$RUN_ROOT/logs"
CANDIDATE_MODEL=ConvIR-Haze4K-hardfreq-loss-stop20-seed3407-20260601
ORIGINAL_BEST=/root/autodl-tmp/workspace/ConvIR-B/Dehazing/ITS/results/ConvIR-Haze4K-original-stop20-seed3407-20260531/Training-Results/Best.pkl
CANDIDATE_BEST="$ITS_ROOT/results/$CANDIDATE_MODEL/Training-Results/Best.pkl"
CANDIDATE_LAST="$ITS_ROOT/results/$CANDIDATE_MODEL/Training-Results/Final.pkl"

mkdir -p "$LOG_DIR"
cd "$ITS_ROOT"

{
  echo "start $(date --iso-8601=seconds)"
  echo "root $ROOT"
  echo "python $PY"
  echo "data_dir $DATA_DIR"
  echo "git_head $(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "git_branch $(git -C "$ROOT" branch --show-current 2>/dev/null || echo unknown)"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true

  echo "running hardfreq preflight $(date --iso-8601=seconds)"
  "$PY" "$ROOT/experience_docx/tools/preflight_haze4k_hardfreq_loss.py" \
    --data_dir "$DATA_DIR" \
    --seed 3407 \
    --batch_size 8 \
    --hard_fft_lambda 0.02 \
    --output "$LOG_DIR/hardfreq_loss_preflight_seed3407.json"

  echo "running hardfreq train $(date --iso-8601=seconds)"
  "$PY" main.py \
    --mode train \
    --model_name "$CANDIDATE_MODEL" \
    --data Haze4K \
    --data_dir "$DATA_DIR" \
    --version base \
    --fam_mode original \
    --loss_mode hard_fft_boost \
    --hard_fft_lambda 0.02 \
    --batch_size 8 \
    --learning_rate 4e-4 \
    --num_epoch 1000 \
    --stop_epoch 20 \
    --print_freq 50 \
    --num_worker 8 \
    --save_freq 5 \
    --valid_freq 1 \
    --seed 3407 \
    > "$LOG_DIR/hardfreq_loss_train_stop20_seed3407.log" 2>&1
  echo "done hardfreq train rc=$? $(date --iso-8601=seconds)"

  echo "running hardfreq Best compare $(date --iso-8601=seconds)"
  "$PY" "$ROOT/experience_docx/tools/eval_haze4k_checkpoint_compare.py" \
    --data_dir "$DATA_DIR" \
    --original_checkpoint "$ORIGINAL_BEST" \
    --candidate_checkpoint "$CANDIDATE_BEST" \
    --candidate_mode original \
    --candidate_name hardfreq_loss \
    --output_dir "$LOG_DIR" \
    --tag seed3407_stop20_best

  "$PY" "$ROOT/experience_docx/tools/analyze_haze4k_delta_buckets.py" \
    --csv "$LOG_DIR/scout_eval_per_image_seed3407_stop20_best.csv" \
    --candidate_name hardfreq_loss \
    --output "$LOG_DIR/scout_eval_bucket_analysis_seed3407_stop20_best.json"

  echo "running hardfreq Last compare $(date --iso-8601=seconds)"
  "$PY" "$ROOT/experience_docx/tools/eval_haze4k_checkpoint_compare.py" \
    --data_dir "$DATA_DIR" \
    --original_checkpoint "$ORIGINAL_BEST" \
    --candidate_checkpoint "$CANDIDATE_LAST" \
    --candidate_mode original \
    --candidate_name hardfreq_loss \
    --output_dir "$LOG_DIR" \
    --tag seed3407_stop20_last

  "$PY" "$ROOT/experience_docx/tools/analyze_haze4k_delta_buckets.py" \
    --csv "$LOG_DIR/scout_eval_per_image_seed3407_stop20_last.csv" \
    --candidate_name hardfreq_loss \
    --output "$LOG_DIR/scout_eval_bucket_analysis_seed3407_stop20_last.json"

  echo "running hardfreq Best-vs-Last direct compare $(date --iso-8601=seconds)"
  "$PY" "$ROOT/experience_docx/tools/eval_haze4k_checkpoint_compare.py" \
    --data_dir "$DATA_DIR" \
    --original_checkpoint "$CANDIDATE_BEST" \
    --original_mode original \
    --original_name best \
    --candidate_checkpoint "$CANDIDATE_LAST" \
    --candidate_mode original \
    --candidate_name last \
    --output_dir "$LOG_DIR" \
    --tag seed3407_stop20_best_vs_last

  echo "complete $(date --iso-8601=seconds)"
} | tee "$LOG_DIR/status.txt"
