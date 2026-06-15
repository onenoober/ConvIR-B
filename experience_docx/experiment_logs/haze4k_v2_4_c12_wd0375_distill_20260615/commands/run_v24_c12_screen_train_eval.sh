#!/usr/bin/env bash
set -euo pipefail
ROOT=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v24-c12-wd0375-distill
PY=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
EVID=$ROOT/experience_docx/experiment_logs/haze4k_v2_4_c12_wd0375_distill_20260615
DATA=/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K
CACHE=/sda/home/wangyuxin/ConvIR-B/runtime_cache/v24_c12_wd0375_teacher/train_core
RUNROOT=/sda/home/wangyuxin/ConvIR-B/runs/v24_c12_wd0375_distill_screen
STATUS=$EVID/status_c12_screen.txt
mkdir -p "$EVID/runtime_logs" "$RUNROOT"
if [ ! -f "$EVID/v24_c12_split_manifest.json" ]; then
  echo "MISSING_SPLIT_MANIFEST" | tee -a "$STATUS"
  exit 2
fi
cache_count=$(find "$CACHE" -maxdepth 1 -type f -name '*.png' | wc -l)
if [ "$cache_count" -lt 2400 ]; then
  echo "MISSING_TEACHER_CACHE count=$cache_count" | tee -a "$STATUS"
  exit 2
fi
{
  echo "state=RUNNING_TRAIN"
  echo "run_id=v24_c12_screen_train_eval"
  echo "start_time=$(date --iso-8601=seconds)"
  echo "locked_test_touched=false"
  echo "locked_per_image_read=false"
  echo "source_commit=$(git -C "$ROOT" rev-parse HEAD)"
  echo "teacher_cache_count=$cache_count"
} > "$STATUS"
variants=(
  c12_gt075_teacher025_lr1e-5:0.75:0.25:0.01:0
  c12_gt050_teacher050_lr1e-5:0.50:0.50:0.01:1
  c12_gt025_teacher075_lr1e-5:0.25:0.75:0.01:2
  c12_teacher100_lr1e-5:0.00:1.00:0.00:3
)
pids=()
for spec in "${variants[@]}"; do
  IFS=: read -r variant gt_w teacher_w fft_w gpu <<< "$spec"
  out="$RUNROOT/$variant"
  log="$EVID/runtime_logs/train_${variant}.log"
  if [ -d "$out/checkpoints" ]; then
    echo "REFUSE_EXISTING_OUTPUT variant=$variant out=$out" | tee -a "$STATUS"
    exit 3
  fi
  (
    export CUDA_VISIBLE_DEVICES=$gpu
    "$PY" "$ROOT/experience_docx/tools/train_haze4k_v24_c12_wd0375_student.py" \
      --variant "$variant" \
      --convir-dir "$ROOT/Dehazing/ITS" \
      --data-dir "$DATA" \
      --split-manifest "$EVID/v24_c12_split_manifest.json" \
      --teacher-dir "$CACHE" \
      --init-checkpoint /sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl \
      --out-dir "$out" \
      --seed 3407 \
      --epochs 5 \
      --batch-size 8 \
      --num-workers 4 \
      --crop-size 256 \
      --learning-rate 1e-5 \
      --gt-weight "$gt_w" \
      --teacher-weight "$teacher_w" \
      --fft-weight "$fft_w" \
      --print-freq 100
  ) > "$log" 2>&1 &
  pids+=("$!")
  echo "launched_train variant=$variant gpu=$gpu pid=${pids[-1]}" | tee -a "$STATUS"
done
train_rc=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then train_rc=1; fi
done
if [ "$train_rc" -ne 0 ]; then
  echo "state=FAILED_COMMAND" >> "$STATUS"
  echo "C12_SCREEN_TRAIN_FAILED" >> "$STATUS"
  exit "$train_rc"
fi
{
  echo "state=RUNNING_EVAL"
  echo "train_finish_time=$(date --iso-8601=seconds)"
  echo "C12_SCREEN_TRAIN_OK"
} >> "$STATUS"
eval_rc=0
for spec in "${variants[@]}"; do
  IFS=: read -r variant gt_w teacher_w fft_w gpu <<< "$spec"
  for epoch in 1 2 3 4 5; do
    ckpt="$RUNROOT/$variant/checkpoints/model_${epoch}.pkl"
    log="$EVID/runtime_logs/eval_${variant}_model_${epoch}.log"
    if [ ! -f "$ckpt" ]; then
      echo "MISSING_CKPT variant=$variant epoch=$epoch" | tee -a "$STATUS"
      eval_rc=1
      continue
    fi
    (
      export CUDA_VISIBLE_DEVICES=$gpu
      "$PY" "$ROOT/experience_docx/tools/eval_haze4k_v24_c12_student.py" \
        --variant "$variant" \
        --checkpoint-name "model_${epoch}" \
        --convir-dir "$ROOT/Dehazing/ITS" \
        --data-dir "$DATA" \
        --split-manifest "$EVID/v24_c12_split_manifest.json" \
        --a0-checkpoint /sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl \
        --student-checkpoint "$ckpt" \
        --out-dir "$EVID" \
        --print-freq 100
    ) > "$log" 2>&1 || eval_rc=1
    echo "eval_done variant=$variant epoch=$epoch rc=$eval_rc" | tee -a "$STATUS"
  done
done
if [ "$eval_rc" -ne 0 ]; then
  echo "state=FAILED_COMMAND" >> "$STATUS"
  echo "C12_SCREEN_EVAL_FAILED" >> "$STATUS"
  exit "$eval_rc"
fi
"$PY" "$ROOT/experience_docx/tools/summarize_haze4k_v24_c12_screen.py" --out-dir "$EVID" 2>&1 | tee "$EVID/runtime_logs/v24_c12_screen_summary.log"
{
  echo "finish_time=$(date --iso-8601=seconds)"
  echo "state=COMPLETED_SCREEN"
  echo "C12_SCREEN_TRAIN_EVAL_OK"
} >> "$STATUS"
