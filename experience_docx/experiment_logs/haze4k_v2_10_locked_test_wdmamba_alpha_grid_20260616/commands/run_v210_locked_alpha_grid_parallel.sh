#!/usr/bin/env bash
set -euo pipefail
ROOT=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v22-c9-fixed-wdmamba-router-locked
EVID=$ROOT/experience_docx/experiment_logs/haze4k_v2_10_locked_test_wdmamba_alpha_grid_20260616
PY=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
TOOL=$EVID/commands/eval_v210_locked_alpha_grid.py
DATA=/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K
A0=/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl
WDM=/sda/home/wangyuxin/ConvIR-B/checkpoints/WDMamba_ckpts/haze4k_35.88.pth
WDM_REPO=/sda/home/wangyuxin/ConvIR-B/repos/external_experts/WDMamba
STATUS=$EVID/status_v210_locked_alpha_grid.txt
MAIN_LOG=$EVID/runtime_logs/v210_locked_alpha_grid_main.log
PREFIX=v210_haze4k_locked_wdmamba_alpha_grid
ALPHAS=(0 0.125 0.25 0.375 0.50 0.75 1.0)
mkdir -p "$EVID/runtime_logs" "$EVID/shards"
{
  echo "run_start v210_locked_alpha_grid_corrected $(date --iso-8601=seconds)"
  echo "state=RUNNING_EVAL"
  echo "locked_test_touched=true"
  echo "locked_policy=diagnostic_grid_only_no_alpha_selection"
  echo "metric_protocol=v2.2_locked_one_shot_compatible"
  echo "root=$ROOT"
  cd "$ROOT"
  if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "root_is_git=1"
    echo "branch=$(git branch --show-current)"
    echo "commit=$(git rev-parse --short HEAD)"
    echo "status_count=$(git status --short | wc -l)"
  else
    echo "root_is_git=0"
  fi
  echo "python=$PY"
  echo "tool=$TOOL"
  echo "tool_sha256=$(sha256sum "$TOOL" | awk '{print $1}')"
  echo "data=$DATA"
  echo "test_haze_count=$(find "$DATA/test/haze" -maxdepth 1 -type f -iname '*.png' | wc -l)"
  echo "test_gt_count=$(find "$DATA/test/gt" -maxdepth 1 -type f -iname '*.png' | wc -l)"
  echo "a0=$A0"
  echo "a0_sha256=$(sha256sum "$A0" | awk '{print $1}')"
  echo "wdmamba=$WDM"
  echo "wdmamba_sha256=$(sha256sum "$WDM" | awk '{print $1}')"
} | tee -a "$STATUS" | tee -a "$MAIN_LOG"

mapfile -t GPUS < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F, '{g=$1+0; mem=$2+0; util=$3+0; if (mem < 12000 && util < 80) print g}')
if [ "${#GPUS[@]}" -eq 0 ]; then
  echo "V210_NO_FREE_GPU" | tee -a "$STATUS" "$MAIN_LOG"
  exit 1
fi
SHARDS=${#GPUS[@]}
printf 'selected_gpus=%s\n' "${GPUS[*]}" | tee -a "$STATUS" "$MAIN_LOG"
printf 'shard_count=%s\n' "$SHARDS" | tee -a "$STATUS" "$MAIN_LOG"

declare -a PIDS=()
declare -a SHARD_CSVS=()
for shard in $(seq 0 $((SHARDS-1))); do
  gpu=${GPUS[$shard]}
  OUT=$EVID/shards/shard_${shard}
  LOG=$EVID/runtime_logs/shard_${shard}_gpu_${gpu}.log
  mkdir -p "$OUT"
  SHARD_CSVS+=("$OUT/${PREFIX}_shard${shard}_per_image.csv")
  echo "launch_shard shard=$shard gpu=$gpu $(date --iso-8601=seconds)" | tee -a "$STATUS" "$MAIN_LOG"
  (
    set -euo pipefail
    CUDA_VISIBLE_DEVICES=$gpu "$PY" "$TOOL" \
      --mode evaluate \
      --data-dir "$DATA" \
      --data-split test \
      --out-dir "$OUT" \
      --prefix "${PREFIX}_shard${shard}" \
      --convir-its-dir "$ROOT/Dehazing/ITS" \
      --a0-checkpoint "$A0" \
      --wdmamba-checkpoint "$WDM" \
      --wdmamba-repo "$WDM_REPO" \
      --alphas "${ALPHAS[@]}" \
      --shard-index "$shard" \
      --shard-count "$SHARDS" \
      --print-freq 10
  ) > "$LOG" 2>&1 &
  PIDS+=("$!")
done

fail=0
for idx in "${!PIDS[@]}"; do
  pid=${PIDS[$idx]}
  if wait "$pid"; then
    echo "shard_done shard=$idx rc=0 $(date --iso-8601=seconds)" | tee -a "$STATUS" "$MAIN_LOG"
  else
    rc=$?
    echo "shard_done shard=$idx rc=$rc $(date --iso-8601=seconds)" | tee -a "$STATUS" "$MAIN_LOG"
    fail=1
  fi
done
if [ "$fail" -ne 0 ]; then
  echo "V210_SHARD_FAILED" | tee -a "$STATUS" "$MAIN_LOG"
  exit 1
fi
"$PY" "$TOOL" \
  --mode aggregate \
  --out-dir "$EVID" \
  --prefix "$PREFIX" \
  --a0-checkpoint "$A0" \
  --wdmamba-checkpoint "$WDM" \
  --wdmamba-repo "$WDM_REPO" \
  --alphas "${ALPHAS[@]}" \
  --input-csvs "${SHARD_CSVS[@]}" \
  --expected-count 1000 2>&1 | tee -a "$MAIN_LOG"
echo "postprocess_done $(date --iso-8601=seconds)" | tee -a "$STATUS" "$MAIN_LOG"
echo "V210_LOCKED_ALPHA_GRID_OK" | tee -a "$STATUS" "$MAIN_LOG"
echo "state=COMPLETED_GATE_PASS" | tee -a "$STATUS" "$MAIN_LOG"
