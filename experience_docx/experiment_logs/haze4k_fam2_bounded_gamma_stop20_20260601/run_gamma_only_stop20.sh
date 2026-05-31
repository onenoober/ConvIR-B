#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/workspace/ConvIR-B
ITS_ROOT="$ROOT/Dehazing/ITS"
DATA_DIR=/root/autodl-tmp/workspace/Dehaze-Net/dataset/HAZE4K
PY=/root/miniconda3/envs/convir-cu128/bin/python
RUN_ROOT="$ITS_ROOT/results/ConvIR-Haze4K-fam2-bounded-gamma-stop20-20260601"
LOG_DIR="$RUN_ROOT/logs"
CANDIDATE_MODEL=ConvIR-Haze4K-fam2_modres_gamma_bounded-stop20-seed3407-20260601
ORIGINAL_BEST="$ITS_ROOT/results/ConvIR-Haze4K-original-stop20-seed3407-20260531/Training-Results/Best.pkl"
CANDIDATE_BEST="$ITS_ROOT/results/$CANDIDATE_MODEL/Training-Results/Best.pkl"
CANDIDATE_LAST="$ITS_ROOT/results/$CANDIDATE_MODEL/Training-Results/Final.pkl"

mkdir -p "$LOG_DIR"
cd "$ITS_ROOT"

{
  echo "start $(date --iso-8601=seconds)"

  echo "running gamma-only preflight $(date --iso-8601=seconds)"
  "$PY" "$ROOT/experience_docx/tools/check_haze4k_fam_equivalence.py" \
    --candidate_mode fam2_modres_gamma_bounded \
    --seed 3407 \
    --output "$LOG_DIR/fam2_gamma_bounded_equivalence_seed3407.json"
  "$PY" "$ROOT/experience_docx/tools/preflight_haze4k_fam2.py" \
    --data_dir "$DATA_DIR" \
    --mode fam2_modres_gamma_bounded \
    --seed 3407 \
    --output "$LOG_DIR/fam2_gamma_bounded_real_batch_seed3407.json"

  echo "running gamma-only train $(date --iso-8601=seconds)"
  "$PY" main.py \
    --mode train \
    --model_name "$CANDIDATE_MODEL" \
    --data Haze4K \
    --data_dir "$DATA_DIR" \
    --version base \
    --fam_mode fam2_modres_gamma_bounded \
    --batch_size 8 \
    --learning_rate 4e-4 \
    --num_epoch 1000 \
    --stop_epoch 20 \
    --print_freq 50 \
    --num_worker 8 \
    --save_freq 5 \
    --valid_freq 1 \
    --seed 3407 \
    --mod_stats_freq 1 \
    --mod_stats_batches 64 \
    > "$LOG_DIR/fam2_modres_gamma_bounded_train_stop20_seed3407.log" 2>&1
  echo "done gamma-only train rc=$? $(date --iso-8601=seconds)"

  echo "running gamma-only Best compare $(date --iso-8601=seconds)"
  "$PY" "$ROOT/experience_docx/tools/eval_haze4k_checkpoint_compare.py" \
    --data_dir "$DATA_DIR" \
    --original_checkpoint "$ORIGINAL_BEST" \
    --candidate_checkpoint "$CANDIDATE_BEST" \
    --candidate_mode fam2_modres_gamma_bounded \
    --candidate_name fam2_modres_gamma_bounded \
    --output_dir "$LOG_DIR" \
    --tag seed3407_stop20_best

  "$PY" "$ROOT/experience_docx/tools/analyze_haze4k_delta_buckets.py" \
    --csv "$LOG_DIR/scout_eval_per_image_seed3407_stop20_best.csv" \
    --candidate_name fam2_modres_gamma_bounded \
    --output "$LOG_DIR/scout_eval_bucket_analysis_seed3407_stop20_best.json"

  "$PY" "$ROOT/experience_docx/tools/analyze_haze4k_modulation_buckets.py" \
    --data_dir "$DATA_DIR" \
    --checkpoint "$CANDIDATE_BEST" \
    --candidate_mode fam2_modres_gamma_bounded \
    --candidate_name fam2_modres_gamma_bounded \
    --compare_csv "$LOG_DIR/scout_eval_per_image_seed3407_stop20_best.csv" \
    --output_json "$LOG_DIR/modulation_bucket_analysis_seed3407_stop20_best.json" \
    --output_csv "$LOG_DIR/modulation_per_image_seed3407_stop20_best.csv"

  echo "running gamma-only Last compare $(date --iso-8601=seconds)"
  "$PY" "$ROOT/experience_docx/tools/eval_haze4k_checkpoint_compare.py" \
    --data_dir "$DATA_DIR" \
    --original_checkpoint "$ORIGINAL_BEST" \
    --candidate_checkpoint "$CANDIDATE_LAST" \
    --candidate_mode fam2_modres_gamma_bounded \
    --candidate_name fam2_modres_gamma_bounded \
    --output_dir "$LOG_DIR" \
    --tag seed3407_stop20_last

  "$PY" "$ROOT/experience_docx/tools/analyze_haze4k_delta_buckets.py" \
    --csv "$LOG_DIR/scout_eval_per_image_seed3407_stop20_last.csv" \
    --candidate_name fam2_modres_gamma_bounded \
    --output "$LOG_DIR/scout_eval_bucket_analysis_seed3407_stop20_last.json"

  "$PY" "$ROOT/experience_docx/tools/analyze_haze4k_modulation_buckets.py" \
    --data_dir "$DATA_DIR" \
    --checkpoint "$CANDIDATE_LAST" \
    --candidate_mode fam2_modres_gamma_bounded \
    --candidate_name fam2_modres_gamma_bounded \
    --compare_csv "$LOG_DIR/scout_eval_per_image_seed3407_stop20_last.csv" \
    --output_json "$LOG_DIR/modulation_bucket_analysis_seed3407_stop20_last.json" \
    --output_csv "$LOG_DIR/modulation_per_image_seed3407_stop20_last.csv"

  echo "running gamma-only Best-vs-Last direct compare $(date --iso-8601=seconds)"
  "$PY" "$ROOT/experience_docx/tools/eval_haze4k_checkpoint_compare.py" \
    --data_dir "$DATA_DIR" \
    --original_checkpoint "$CANDIDATE_BEST" \
    --original_mode fam2_modres_gamma_bounded \
    --original_name best \
    --candidate_checkpoint "$CANDIDATE_LAST" \
    --candidate_mode fam2_modres_gamma_bounded \
    --candidate_name last \
    --output_dir "$LOG_DIR" \
    --tag seed3407_stop20_best_vs_last

  echo "complete $(date --iso-8601=seconds)"
} | tee "$LOG_DIR/status.txt"
