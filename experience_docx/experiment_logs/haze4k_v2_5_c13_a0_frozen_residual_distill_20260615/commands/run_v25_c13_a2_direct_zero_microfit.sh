#!/usr/bin/env bash
set -euo pipefail

ROOT=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v25-c13-a0-frozen-residual-distill
PY=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
EVID=$ROOT/experience_docx/experiment_logs/haze4k_v2_5_c13_a0_frozen_residual_distill_20260615
C12_EVID=$ROOT/experience_docx/experiment_logs/haze4k_v2_4_c12_wd0375_distill_20260615
DATA=/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K
A0=/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl
CACHE=/sda/home/wangyuxin/ConvIR-B/runtime_cache/v24_c12_wd0375_teacher/train_core
RUNROOT=/sda/home/wangyuxin/ConvIR-B/runs/v25_c13_a0_frozen_residual_distill
STATUS=$EVID/status_c13_a2_direct_zero_microfit.txt

mkdir -p "$EVID/runtime_logs" "$EVID/commands" "$RUNROOT"
{
  echo "state=PREFLIGHT_RUNNING"
  echo "run_id=v25_c13_a2_direct_zero_microfit"
  echo "start_time=$(date --iso-8601=seconds)"
  echo "locked_test_touched=false"
  echo "locked_per_image_read=false"
  echo "source_commit=$(git -C "$ROOT" rev-parse HEAD)"
  echo "source_branch=$(git -C "$ROOT" branch --show-current)"
  echo "design=direct residual with zero head, no gate bootstrap, clamped output"
} > "$STATUS"

for path in "$PY" "$DATA" "$A0" "$CACHE" "$EVID/v25_c13_split_manifest.json" "$C12_EVID/v24_c12_teacher_cache_metrics.csv"; do
  if [ ! -e "$path" ]; then
    echo "MISSING_REQUIRED_PATH path=$path" | tee -a "$STATUS"
    echo "state=PREFLIGHT_FAILED_ENGINEERING" >> "$STATUS"
    exit 2
  fi
done

{
  echo "state=C13_A2_MICROFIT_RUNNING"
  echo "microfit_start_time=$(date --iso-8601=seconds)"
} >> "$STATUS"

pids=()
specs=(
  c13a2_directzero16:16:1:rgb_wavelet:32:3:10
  c13a2_directzero64:64:2:rgb_wavelet:32:3:10
  c13a2_directzero256:256:3:rgb_wavelet:32:3:10
)
for spec in "${specs[@]}"; do
  IFS=: read -r variant max_images gpu feature width depth epochs <<< "$spec"
  out="$RUNROOT/$variant"
  if [ -e "$out" ]; then
    echo "REFUSE_EXISTING_OUTPUT variant=$variant out=$out" | tee -a "$STATUS"
    echo "state=PREFLIGHT_FAILED_ENGINEERING" >> "$STATUS"
    exit 3
  fi
  (
    export CUDA_VISIBLE_DEVICES=$gpu
    "$PY" "$ROOT/experience_docx/tools/train_haze4k_v25_c13_residual_adapter.py" \
      --variant "$variant" \
      --convir-dir "$ROOT/Dehazing/ITS" \
      --data-dir "$DATA" \
      --split-manifest "$EVID/v25_c13_split_manifest.json" \
      --teacher-dir "$CACHE" \
      --teacher-metrics "$C12_EVID/v24_c12_teacher_cache_metrics.csv" \
      --a0-checkpoint "$A0" \
      --out-dir "$out" \
      --feature-mode "$feature" \
      --adapter-width "$width" \
      --adapter-depth "$depth" \
      --bootstrap-scale 0.0 \
      --residual-mode direct \
      --residual-scale 1.0 \
      --head-init zero \
      --clamp-output \
      --seed 3407 \
      --epochs "$epochs" \
      --batch-size 1 \
      --num-workers 2 \
      --crop-size 0 \
      --max-images "$max_images" \
      --keep-partial-batch \
      --learning-rate 1e-4 \
      --weight-decay 0 \
      --teacher-margin 0.10 \
      --gt-weight 0.50 \
      --teacher-weight 0.50 \
      --preserve-weight 2.00 \
      --freq-weight 0.00 \
      --color-weight 0.05 \
      --tv-weight 0.02 \
      --raw-weight 0.001 \
      --grad-clip-norm 1.0 \
      --print-freq 20
  ) > "$EVID/runtime_logs/train_${variant}.log" 2>&1 &
  pids+=("$!")
  echo "launched_a2_microfit variant=$variant gpu=$gpu pid=${pids[-1]}" | tee -a "$STATUS"
done

rc=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then rc=1; fi
done
if [ "$rc" -ne 0 ]; then
  echo "C13_A2_MICROFIT_FAILED" | tee -a "$STATUS"
  echo "state=FAILED_COMMAND" >> "$STATUS"
  exit "$rc"
fi

for spec in "${specs[@]}"; do
  IFS=: read -r variant max_images gpu feature width depth epochs <<< "$spec"
  (
    export CUDA_VISIBLE_DEVICES=$gpu
    "$PY" "$ROOT/experience_docx/tools/eval_haze4k_v25_c13_residual_adapter.py" \
      --variant "$variant" \
      --checkpoint Best \
      --convir-dir "$ROOT/Dehazing/ITS" \
      --data-dir "$DATA" \
      --split-manifest "$EVID/v25_c13_split_manifest.json" \
      --a0-checkpoint "$A0" \
      --student-checkpoint "$RUNROOT/$variant/checkpoints/Best.pkl" \
      --out-dir "$EVID" \
      --feature-mode "$feature" \
      --adapter-width "$width" \
      --adapter-depth "$depth" \
      --bootstrap-scale 0.0 \
      --residual-mode direct \
      --residual-scale 1.0 \
      --head-init zero \
      --clamp-output \
      --max-train "$max_images" \
      --max-val 128
  ) > "$EVID/runtime_logs/eval_${variant}_Best.log" 2>&1
  echo "eval_a2_microfit_done variant=$variant" | tee -a "$STATUS"
done

{
  echo "finish_time=$(date --iso-8601=seconds)"
  echo "state=C13_A2_MICROFIT_DONE_REVIEW_BEFORE_B_SCREEN"
  echo "C13_A2_DIRECT_ZERO_MICROFIT_OK"
} >> "$STATUS"
