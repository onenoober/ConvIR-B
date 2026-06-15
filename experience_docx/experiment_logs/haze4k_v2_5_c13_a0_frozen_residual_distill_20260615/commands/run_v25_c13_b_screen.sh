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
STATUS=$EVID/status_c13_b_screen.txt

mkdir -p "$EVID/runtime_logs" "$RUNROOT"
{
  echo "state=PREFLIGHT_RUNNING"
  echo "run_id=v25_c13_b_screen"
  echo "start_time=$(date --iso-8601=seconds)"
  echo "locked_test_touched=false"
  echo "locked_per_image_read=false"
  echo "source_commit=$(git -C "$ROOT" rev-parse HEAD)"
} > "$STATUS"

for path in "$PY" "$DATA" "$A0" "$CACHE" "$EVID/v25_c13_split_manifest.json" "$C12_EVID/v24_c12_teacher_cache_metrics.csv"; do
  if [ ! -e "$path" ]; then
    echo "MISSING_REQUIRED_PATH path=$path" | tee -a "$STATUS"
    echo "state=PREFLIGHT_FAILED_ENGINEERING" >> "$STATUS"
    exit 2
  fi
done

if [ -f "$EVID/status_c13_0_audit_microfit.txt" ] && ! grep -q 'C13_0_AUDIT_MICROFIT_OK' "$EVID/status_c13_0_audit_microfit.txt"; then
  echo "C13_0_NOT_PASSED_REFUSE_B_SCREEN" | tee -a "$STATUS"
  echo "state=PREFLIGHT_FAILED_ENGINEERING" >> "$STATUS"
  exit 3
fi

{
  echo "state=C13_SCREEN_RUNNING"
  echo "screen_train_start_time=$(date --iso-8601=seconds)"
} >> "$STATUS"

specs=(
  c13b_rgb_residual:rgb:32:3:0:0.50:1.00:0.50:0.05:0.05:0.01
  c13b_rgb_wavelet_residual:rgb_wavelet:32:3:1:0.50:1.00:0.50:0.10:0.05:0.01
  c13b_rgb_wavelet_preserve_strong:rgb_wavelet:48:4:2:0.60:0.80:0.80:0.12:0.08:0.02
)

pids=()
for spec in "${specs[@]}"; do
  IFS=: read -r variant feature width depth gpu gt_w teacher_w preserve_w freq_w color_w tv_w <<< "$spec"
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
      --bootstrap-scale 0.01 \
      --seed 3407 \
      --epochs 5 \
      --batch-size 8 \
      --num-workers 4 \
      --crop-size 256 \
      --learning-rate 2e-4 \
      --teacher-margin 0.02 \
      --gt-weight "$gt_w" \
      --teacher-weight "$teacher_w" \
      --preserve-weight "$preserve_w" \
      --freq-weight "$freq_w" \
      --color-weight "$color_w" \
      --tv-weight "$tv_w" \
      --grad-clip-norm 1.0 \
      --print-freq 100
  ) > "$EVID/runtime_logs/train_${variant}.log" 2>&1 &
  pids+=("$!")
  echo "launched_screen_train variant=$variant gpu=$gpu pid=${pids[-1]}" | tee -a "$STATUS"
done

train_rc=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then train_rc=1; fi
done
if [ "$train_rc" -ne 0 ]; then
  echo "C13_B_SCREEN_TRAIN_FAILED" | tee -a "$STATUS"
  echo "state=FAILED_COMMAND" >> "$STATUS"
  exit "$train_rc"
fi

{
  echo "state=RUNNING_EVAL"
  echo "screen_eval_start_time=$(date --iso-8601=seconds)"
} >> "$STATUS"

for spec in "${specs[@]}"; do
  IFS=: read -r variant feature width depth gpu gt_w teacher_w preserve_w freq_w color_w tv_w <<< "$spec"
  for epoch in 1 2 3 4 5; do
    ckpt="$RUNROOT/$variant/checkpoints/model_${epoch}.pkl"
    if [ ! -f "$ckpt" ]; then
      echo "MISSING_CKPT variant=$variant epoch=$epoch" | tee -a "$STATUS"
      echo "state=FAILED_COMMAND" >> "$STATUS"
      exit 4
    fi
    (
      export CUDA_VISIBLE_DEVICES=$gpu
      "$PY" "$ROOT/experience_docx/tools/eval_haze4k_v25_c13_residual_adapter.py" \
        --variant "$variant" \
        --checkpoint "model_${epoch}" \
        --convir-dir "$ROOT/Dehazing/ITS" \
        --data-dir "$DATA" \
        --split-manifest "$EVID/v25_c13_split_manifest.json" \
        --a0-checkpoint "$A0" \
        --student-checkpoint "$ckpt" \
        --out-dir "$EVID" \
        --feature-mode "$feature" \
        --adapter-width "$width" \
        --adapter-depth "$depth" \
        --bootstrap-scale 0.01 \
        --max-train 0 \
        --max-val 0
    ) > "$EVID/runtime_logs/eval_${variant}_model_${epoch}.log" 2>&1
    echo "eval_screen_done variant=$variant epoch=$epoch" | tee -a "$STATUS"
  done
done

"$PY" "$ROOT/experience_docx/tools/summarize_haze4k_v25_c13_screen.py" \
  --out-dir "$EVID" \
  --teacher-metrics "$C12_EVID/v24_c12_teacher_cache_metrics.csv" \
  2>&1 | tee "$EVID/runtime_logs/v25_c13_screen_summary.log"

decision=$("$PY" - <<PY
import json
from pathlib import Path
p=Path("$EVID/v25_c13_screen_decision.json")
print(json.loads(p.read_text())["decision"])
PY
)
{
  echo "finish_time=$(date --iso-8601=seconds)"
  echo "state=$decision"
  echo "C13_B_SCREEN_OK"
} >> "$STATUS"
