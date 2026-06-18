#!/usr/bin/env bash
set -euo pipefail

ROOT=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v25-c13-a0-frozen-residual-distill
PY=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
EVID=$ROOT/experience_docx/experiment_logs/haze4k_v2_5_c13_a0_frozen_residual_distill_20260615
C12_EVID=$ROOT/experience_docx/experiment_logs/haze4k_v2_4_c12_wd0375_distill_20260615
DATA=/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K
A0=/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl
RUNROOT=/sda/home/wangyuxin/ConvIR-B/runs/v25_c13_a0_frozen_residual_distill
C8_PER_IMAGE=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v22-c8-mini-expert-oracle/experience_docx/experiment_logs/haze4k_v2_2_c8_mini_expert_oracle_20260615/v22_c8_1_wdmamba_per_image.csv
STATUS=$EVID/status_c13_f_diagnostics.txt

mkdir -p "$EVID/runtime_logs" "$EVID/commands"
{
  echo "state=PREFLIGHT_RUNNING"
  echo "run_id=v25_c13_f_diagnostics"
  echo "start_time=$(date --iso-8601=seconds)"
  echo "locked_test_touched=false"
  echo "locked_per_image_read=false"
  echo "source_commit=$(git -C "$ROOT" rev-parse HEAD)"
  echo "source_branch=$(git -C "$ROOT" branch --show-current)"
  echo "diagnostic_scope=C13-F0 full600 replay plus F1/F2/F3 oracle taxonomy"
} > "$STATUS"

for path in \
  "$PY" \
  "$DATA" \
  "$A0" \
  "$EVID/v25_c13_split_manifest.json" \
  "$C12_EVID/v24_c12_teacher_cache_metrics.csv" \
  "$C8_PER_IMAGE" \
  "$RUNROOT/c13a2_directzero256/checkpoints/Best.pkl" \
  "$RUNROOT/c13a3_adaptive050/checkpoints/Best.pkl" \
  "$RUNROOT/c13a4_scale050/checkpoints/Best.pkl" \
  "$ROOT/experience_docx/tools/diagnose_haze4k_v25_c13_f.py"; do
  if [ ! -e "$path" ]; then
    echo "MISSING_REQUIRED_PATH path=$path" | tee -a "$STATUS"
    echo "state=PREFLIGHT_FAILED_ENGINEERING" >> "$STATUS"
    exit 2
  fi
done

for out in \
  "$EVID/v25_c13_f0_full600_leaderboard.csv" \
  "$EVID/v25_c13_f1_f2_oracle_summary.json" \
  "$EVID/v25_c13_f_diagnostic_decision.json"; do
  if [ -e "$out" ] && [ "${ALLOW_OVERWRITE:-0}" != "1" ]; then
    echo "REFUSE_EXISTING_OUTPUT path=$out" | tee -a "$STATUS"
    echo "state=PREFLIGHT_FAILED_ENGINEERING" >> "$STATUS"
    exit 3
  fi
done

{
  echo "state=RUNNING_AUDIT"
  echo "f_diagnostics_start=$(date --iso-8601=seconds)"
} >> "$STATUS"

"$PY" "$ROOT/experience_docx/tools/diagnose_haze4k_v25_c13_f.py" teacher-c8 \
  --split-manifest "$EVID/v25_c13_split_manifest.json" \
  --c8-per-image "$C8_PER_IMAGE" \
  --out-dir "$EVID" \
  > "$EVID/runtime_logs/v25_c13_f0_teacher_wd0375.log" 2>&1
echo "f0_teacher_done" | tee -a "$STATUS"

pids=()
launch_eval() {
  local tag=$1
  local gpu=$2
  local ckpt=$3
  shift 3
  (
    export CUDA_VISIBLE_DEVICES=$gpu
    "$PY" "$ROOT/experience_docx/tools/diagnose_haze4k_v25_c13_f.py" eval-c13 \
      --tag "$tag" \
      --checkpoint Best \
      --convir-dir "$ROOT/Dehazing/ITS" \
      --data-dir "$DATA" \
      --split-manifest "$EVID/v25_c13_split_manifest.json" \
      --a0-checkpoint "$A0" \
      --student-checkpoint "$ckpt" \
      --out-dir "$EVID" \
      "$@"
  ) > "$EVID/runtime_logs/v25_c13_f0_${tag}.log" 2>&1 &
  pids+=("$!")
  echo "launched_f0 tag=$tag gpu=$gpu pid=${pids[-1]}" | tee -a "$STATUS"
}

launch_eval c13a2_directzero256 2 "$RUNROOT/c13a2_directzero256/checkpoints/Best.pkl" \
  --feature-mode rgb_wavelet --adapter-width 32 --adapter-depth 3 --bootstrap-scale 0.0 \
  --residual-mode direct --residual-scale 1.0 --head-init zero --clamp-output

launch_eval c13a3_adaptive050 3 "$RUNROOT/c13a3_adaptive050/checkpoints/Best.pkl" \
  --feature-mode rgb_wavelet --adapter-width 32 --adapter-depth 3 --bootstrap-scale 0.0 \
  --residual-mode adaptive_scalar --residual-scale 1.0 --scale-init 0.50 --head-init zero --clamp-output

launch_eval c13a4_scale050 4 "$RUNROOT/c13a4_scale050/checkpoints/Best.pkl" \
  --feature-mode rgb_wavelet --adapter-width 32 --adapter-depth 3 --bootstrap-scale 0.0 \
  --residual-mode direct --residual-scale 0.50 --head-init zero --clamp-output

launch_eval a5_a4sweep_s025 5 "$RUNROOT/c13a4_scale050/checkpoints/Best.pkl" \
  --feature-mode rgb_wavelet --adapter-width 32 --adapter-depth 3 --bootstrap-scale 0.0 \
  --residual-mode direct --residual-scale 0.25 --head-init zero --clamp-output

launch_eval a5_a4sweep_s030 6 "$RUNROOT/c13a4_scale050/checkpoints/Best.pkl" \
  --feature-mode rgb_wavelet --adapter-width 32 --adapter-depth 3 --bootstrap-scale 0.0 \
  --residual-mode direct --residual-scale 0.30 --head-init zero --clamp-output

(
  export CUDA_VISIBLE_DEVICES=7
  "$PY" "$ROOT/experience_docx/tools/diagnose_haze4k_v25_c13_f.py" oracle \
    --convir-dir "$ROOT/Dehazing/ITS" \
    --data-dir "$DATA" \
    --split-manifest "$EVID/v25_c13_split_manifest.json" \
    --a0-checkpoint "$A0" \
    --student-checkpoint "$RUNROOT/c13a4_scale050/checkpoints/Best.pkl" \
    --c8-per-image "$C8_PER_IMAGE" \
    --out-dir "$EVID" \
    --feature-mode rgb_wavelet \
    --adapter-width 32 \
    --adapter-depth 3 \
    --actual-scale 0.50 \
    --patch-size 64
) > "$EVID/runtime_logs/v25_c13_f1_f2_f3_oracle.log" 2>&1 &
pids+=("$!")
echo "launched_f1_f2_f3_oracle gpu=7 pid=${pids[-1]}" | tee -a "$STATUS"

rc=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    rc=1
  fi
done
if [ "$rc" -ne 0 ]; then
  echo "C13_F_DIAGNOSTICS_FAILED" | tee -a "$STATUS"
  echo "state=FAILED_COMMAND" >> "$STATUS"
  exit "$rc"
fi

"$PY" "$ROOT/experience_docx/tools/diagnose_haze4k_v25_c13_f.py" aggregate \
  --out-dir "$EVID" \
  > "$EVID/runtime_logs/v25_c13_f_aggregate.log" 2>&1
echo "f_aggregate_done" | tee -a "$STATUS"

{
  echo "finish_time=$(date --iso-8601=seconds)"
  echo "state=C13_F_DIAGNOSTIC_COMPLETE_GATE_VS_RESIDUAL_DIRECTION_REVIEW"
  echo "C13_F_DIAGNOSTICS_OK"
} >> "$STATUS"
