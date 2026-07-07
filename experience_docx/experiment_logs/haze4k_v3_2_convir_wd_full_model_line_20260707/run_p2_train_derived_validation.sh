#!/usr/bin/env bash
set -euo pipefail

BASE=/sda/home/wangyuxin/ConvIR-B
WORK=$BASE/repos/ConvIR-B-v3-2-convir-wd
EVID=$WORK/experience_docx/experiment_logs/haze4k_v3_2_convir_wd_full_model_line_20260707
PY=$BASE/envs/convir-cu121/bin/python
DATA=$BASE/datasets/Haze4K/Haze4K
SPLIT_ROOT=$BASE/datasets/Haze4K/Haze4K_v32_p2_fold0_train480_val120
A0=$BASE/checkpoints/official/Haze4K/haze4k-base.pkl
V31_CSV=$BASE/repos/ConvIR-B-github-main/experience_docx/experiment_logs/haze4k_v3_1_full_model_candidate_bakeoff_20260707/v31_candidate_per_image_cloud_only.csv
STATUS=$EVID/status.txt
MODEL_NAME=ConvIR-Haze4K-v32-p2-wddecoder-seed3407-20260707
MODEL_DIR=$WORK/Dehazing/ITS/results/$MODEL_NAME
TRAIN_LOG=$EVID/p2_train_v32_wddecoder_seed3407.log
SPLIT_JSON=$EVID/v32_p2_split_summary.json
BEST_JSON=$EVID/v32_p2_eval_best_summary.json
FINAL_JSON=$EVID/v32_p2_eval_final_summary.json
BEST_CSV=$EVID/v32_p2_eval_best_per_image_cloud_only.csv
FINAL_CSV=$EVID/v32_p2_eval_final_per_image_cloud_only.csv

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

mkdir -p "$EVID"
{
  echo "p2_train_derived_validation_start haze4k_v3_2_convir_wd_full_model_line_20260707 $(date --iso-8601=seconds)"
  echo "work=$WORK"
  echo "python=$PY"
  echo "data=$DATA"
  echo "split_root=$SPLIT_ROOT"
  echo "a0=$A0"
  echo "v31_csv=$V31_CSV"
  cd "$WORK"
  echo "branch=$(git branch --show-current)"
  echo "commit=$(git rev-parse --short HEAD)"
} | tee -a "$STATUS"

cd "$WORK"
test -x "$PY"
test -d "$DATA/train/haze"
test -d "$DATA/train/gt"
test -f "$A0"
test -f "$V31_CSV"
if [[ -e "$MODEL_DIR" ]]; then
  echo "V32_P2_OUTPUT_EXISTS $MODEL_DIR" | tee -a "$STATUS"
  exit 1
fi

"$PY" experience_docx/tools/haze4k_v32_p2_make_split.py \
  --v31_csv "$V31_CSV" \
  --source_data_dir "$DATA" \
  --output_dir "$SPLIT_ROOT" \
  --summary_output "$SPLIT_JSON" \
  --val_fold 0

cd "$WORK/Dehazing/ITS"
set +e
PYTHONUNBUFFERED=1 "$PY" main.py \
  --model_name "$MODEL_NAME" \
  --arch convir_wd_lite \
  --convir_wd_train_scope wd_decoder \
  --convir_wd_decoder_learning_rate 0.00001 \
  --convir_wd_dwt_low_weight 0.05 \
  --convir_wd_dwt_high_weight 0.01 \
  --convir_wd_y_weight 0.05 \
  --init_model "$A0" \
  --mode train \
  --data Haze4K \
  --version base \
  --data_dir "$SPLIT_ROOT" \
  --batch_size 4 \
  --learning_rate 0.0002 \
  --weight_decay 0 \
  --num_epoch 20 \
  --stop_epoch 20 \
  --print_freq 30 \
  --num_worker 4 \
  --save_freq 5 \
  --valid_freq 5 \
  --mod_stats_freq 5 \
  --mod_stats_batches 32 \
  --grad_clip_norm 0.01 \
  --seed 3407 \
  > "$TRAIN_LOG" 2>&1
train_rc=$?
set -e
echo "p2_train_done rc=$train_rc haze4k_v3_2_convir_wd_full_model_line_20260707 $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [[ "$train_rc" -ne 0 ]]; then
  echo "V32_P2_TRAIN_FAILED" | tee -a "$STATUS"
  exit "$train_rc"
fi

cd "$WORK"
BEST_CKPT=$MODEL_DIR/Training-Results/Best.pkl
FINAL_CKPT=$MODEL_DIR/Training-Results/Final.pkl
test -f "$BEST_CKPT"
test -f "$FINAL_CKPT"

PYTHONUNBUFFERED=1 "$PY" experience_docx/tools/haze4k_v32_p2_eval_compare.py \
  --data_dir "$SPLIT_ROOT" \
  --original_checkpoint "$A0" \
  --candidate_checkpoint "$BEST_CKPT" \
  --candidate_arch convir_wd_lite \
  --candidate_name v32_p2_best \
  --output_summary "$BEST_JSON" \
  --output_per_image "$BEST_CSV" \
  --v31_per_image_csv "$V31_CSV" \
  > "$EVID/p2_eval_best_v32.log" 2>&1
echo "p2_eval_best_done haze4k_v3_2_convir_wd_full_model_line_20260707 $(date --iso-8601=seconds)" | tee -a "$STATUS"

PYTHONUNBUFFERED=1 "$PY" experience_docx/tools/haze4k_v32_p2_eval_compare.py \
  --data_dir "$SPLIT_ROOT" \
  --original_checkpoint "$A0" \
  --candidate_checkpoint "$FINAL_CKPT" \
  --candidate_arch convir_wd_lite \
  --candidate_name v32_p2_final \
  --output_summary "$FINAL_JSON" \
  --output_per_image "$FINAL_CSV" \
  --v31_per_image_csv "$V31_CSV" \
  > "$EVID/p2_eval_final_v32.log" 2>&1
echo "p2_eval_final_done haze4k_v3_2_convir_wd_full_model_line_20260707 $(date --iso-8601=seconds)" | tee -a "$STATUS"

"$PY" - <<PY | tee -a "$STATUS"
import json
best = json.load(open("$BEST_JSON"))
gate = best["gate"]
print("V32_P2_BEST_DECISION", gate["decision"])
print("V32_P2_BEST_CONTINUE_ALLOWED_TO_P3", gate["continue_allowed_to_p3_design"])
print("V32_P2_BEST_MEAN_DELTA", best["comparison"]["mean_psnr_delta"])
print("V32_P2_BEST_HARD_DELTA", best["comparison"]["hard_bottom25_psnr_delta"])
print("V32_P2_BEST_EASY_DELTA", best["comparison"]["easy_top25_psnr_delta"])
print("V32_P2_BEST_P05_DELTA", best["comparison"]["p05_psnr_delta"])
print("V32_P2_BEST_CVAR5_DELTA", best["comparison"]["cvar5_psnr_delta"])
PY

if "$PY" - <<PY
import json
gate = json.load(open("$BEST_JSON"))["gate"]
raise SystemExit(0 if gate["continue_allowed_to_p3_design"] else 1)
PY
then
  echo "V32_P2_TRAIN_DERIVED_VALIDATION_PASS_P3_DESIGN_ALLOWED_LOCKED_TEST_BLOCKED" | tee -a "$STATUS"
else
  echo "V32_P2_TRAIN_DERIVED_VALIDATION_FAIL_OR_NOT_COMPETITIVE_LOCKED_TEST_BLOCKED" | tee -a "$STATUS"
fi

echo "p2_train_derived_validation_done haze4k_v3_2_convir_wd_full_model_line_20260707 $(date --iso-8601=seconds)" | tee -a "$STATUS"
