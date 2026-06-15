#!/usr/bin/env bash
set -euo pipefail

ROOT=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v25-c13-a0-frozen-residual-distill
PY=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
EVID=$ROOT/experience_docx/experiment_logs/haze4k_v2_5_c13_a0_frozen_residual_distill_20260615
C12_EVID=$ROOT/experience_docx/experiment_logs/haze4k_v2_4_c12_wd0375_distill_20260615
DATA=/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K
A0=/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl
C12_RUN=/sda/home/wangyuxin/ConvIR-B/runs/v24_c12_wd0375_distill_screen/c12_gt075_teacher025_lr1e-5/checkpoints/model_1.pkl
CACHE=/sda/home/wangyuxin/ConvIR-B/runtime_cache/v24_c12_wd0375_teacher/train_core
RUNROOT=/sda/home/wangyuxin/ConvIR-B/runs/v25_c13_a0_frozen_residual_distill
STATUS=$EVID/status_c13_0_audit_microfit.txt

mkdir -p "$EVID/runtime_logs" "$EVID/commands" "$RUNROOT"
{
  echo "state=PREFLIGHT_RUNNING"
  echo "run_id=v25_c13_0_audit_microfit"
  echo "start_time=$(date --iso-8601=seconds)"
  echo "locked_test_touched=false"
  echo "locked_per_image_read=false"
  echo "source_commit=$(git -C "$ROOT" rev-parse HEAD)"
  echo "source_branch=$(git -C "$ROOT" branch --show-current)"
} > "$STATUS"

for path in "$PY" "$DATA" "$A0" "$C12_RUN" "$CACHE" "$C12_EVID/v24_c12_split_manifest.json" "$C12_EVID/v24_c12_teacher_cache_metrics.csv"; do
  if [ ! -e "$path" ]; then
    echo "MISSING_REQUIRED_PATH path=$path" | tee -a "$STATUS"
    echo "state=PREFLIGHT_FAILED_ENGINEERING" >> "$STATUS"
    exit 2
  fi
done

cache_count=$(find "$CACHE" -maxdepth 1 -type f -name '*.png' | wc -l)
if [ "$cache_count" -lt 2400 ]; then
  echo "MISSING_TEACHER_CACHE count=$cache_count" | tee -a "$STATUS"
  echo "state=PREFLIGHT_FAILED_ENGINEERING" >> "$STATUS"
  exit 2
fi
echo "teacher_cache_count=$cache_count" >> "$STATUS"

"$PY" "$ROOT/experience_docx/tools/make_haze4k_v25_c13_splits.py" \
  --data-dir "$DATA" \
  --c12-split-manifest "$C12_EVID/v24_c12_split_manifest.json" \
  --c12-teacher-metrics "$C12_EVID/v24_c12_teacher_cache_metrics.csv" \
  --out-dir "$EVID" \
  2>&1 | tee "$EVID/runtime_logs/v25_c13_split_manifest.log"

cat > "$EVID/v25_c13_0_source_manifest.json" <<MANIFEST
{
  "route": "Haze4K v2.5 C13 A0-frozen residual distillation",
  "branch": "$(git -C "$ROOT" branch --show-current)",
  "source_commit": "$(git -C "$ROOT" rev-parse HEAD)",
  "architecture_anchor": "github/codex/haze4k-official-arch-anchor",
  "anchor_commit": "2d529d4",
  "python": "$PY",
  "data_dir": "$DATA",
  "a0_checkpoint": "$A0",
  "c12_student_checkpoint": "$C12_RUN",
  "teacher_cache": "$CACHE",
  "locked_test_touched": false,
  "locked_per_image_read": false,
  "trainable_prefix": "C13_",
  "a0_frozen": true,
  "initial_output_equals_a0": true
}
MANIFEST

{
  echo "state=RUNNING_AUDIT"
  echo "audit_start_time=$(date --iso-8601=seconds)"
} >> "$STATUS"

(
  export CUDA_VISIBLE_DEVICES=0
  "$PY" "$ROOT/experience_docx/tools/audit_haze4k_v25_c13_c12_failure.py" \
    --repo-root "$ROOT" \
    --convir-dir "$ROOT/Dehazing/ITS" \
    --data-dir "$DATA" \
    --split-manifest "$EVID/v25_c13_split_manifest.json" \
    --teacher-metrics "$C12_EVID/v24_c12_teacher_cache_metrics.csv" \
    --a0-checkpoint "$A0" \
    --c12-student-checkpoint "$C12_RUN" \
    --out-dir "$EVID" \
    --feature-mode rgb_wavelet \
    --adapter-width 32 \
    --adapter-depth 3 \
    --bootstrap-scale 0.01 \
    --max-train 64 \
    --max-val 64
) 2>&1 | tee "$EVID/runtime_logs/v25_c13_0_c12_failure_audit.log"

if ! grep -q '"c13_model0_a0_parity_pass": true' "$EVID/v25_c13_0_c12_failure_audit.json"; then
  echo "C13_MODEL0_A0_PARITY_FAIL" | tee -a "$STATUS"
  echo "state=C13_AUDIT_FAILED_ENGINEERING" >> "$STATUS"
  exit 4
fi
echo "C13_AUDIT_OK" >> "$STATUS"

{
  echo "state=C13_MICROFIT_RUNNING"
  echo "microfit_start_time=$(date --iso-8601=seconds)"
} >> "$STATUS"

pids=()
specs=(
  c13a_microfit16:16:0:rgb_wavelet:32:3:8
  c13a_microfit64:64:1:rgb_wavelet:32:3:8
  c13a_microfit256:256:2:rgb_wavelet:32:3:8
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
      --bootstrap-scale 0.01 \
      --seed 3407 \
      --epochs "$epochs" \
      --batch-size 8 \
      --num-workers 4 \
      --crop-size 256 \
      --max-images "$max_images" \
      --keep-partial-batch \
      --learning-rate 2e-4 \
      --teacher-margin 0.02 \
      --gt-weight 0.50 \
      --teacher-weight 1.00 \
      --preserve-weight 0.50 \
      --freq-weight 0.10 \
      --color-weight 0.05 \
      --tv-weight 0.01 \
      --grad-clip-norm 1.0 \
      --print-freq 20
  ) > "$EVID/runtime_logs/train_${variant}.log" 2>&1 &
  pids+=("$!")
  echo "launched_microfit variant=$variant gpu=$gpu pid=${pids[-1]}" | tee -a "$STATUS"
done

rc=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then rc=1; fi
done
if [ "$rc" -ne 0 ]; then
  echo "C13_MICROFIT_FAILED" | tee -a "$STATUS"
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
      --bootstrap-scale 0.01 \
      --max-train "$max_images" \
      --max-val 128
  ) > "$EVID/runtime_logs/eval_${variant}_Best.log" 2>&1
  echo "eval_microfit_done variant=$variant" | tee -a "$STATUS"
done

{
  echo "finish_time=$(date --iso-8601=seconds)"
  echo "state=C13_MICROFIT_OK_START_B_SCREEN"
  echo "C13_0_AUDIT_MICROFIT_OK"
} >> "$STATUS"
