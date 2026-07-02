#!/usr/bin/env bash
set -euo pipefail

BASE=${BASE:-/sda/home/wangyuxin/ConvIR-B}
WORK=${WORK:-$BASE/repos/ConvIR-B-dta-v3-7-u-tqs-mix-d3-realblend}
TAU_WORK=${TAU_WORK:-$BASE/repos/ConvIR-B-dta-v3-7-u-tqs-mix-phased1bundle}
PY=${PY:-$BASE/envs/convir-cu121/bin/python}
DATA=${DATA:-$BASE/datasets/Haze4K/Haze4K}
A0=${A0:-$BASE/checkpoints/official/Haze4K/haze4k-base.pkl}
DEPTH=${DEPTH:-$BASE/depth_cache/depth_anything_v2_small_hf}
EVID=$WORK/experience_docx/experiment_logs/haze4k_dta_v3_7_u_tqs_mix_20260613
STATUS=$EVID/status_phase_d3_tau_real_blend.txt
SPLIT_JSON=${SPLIT_JSON:-$WORK/experience_docx/experiment_logs/haze4k_dta_v3_6_hrcs_20260613/dta_v3_6_haze4k_oof_splits_seed3407.json}
ACTION_TABLE=${ACTION_TABLE:-$WORK/experience_docx/experiment_logs/haze4k_dta_v3_7_u_tqs_mix_20260613/v37_tau_oof_per_image_action_table.csv}
FOLDS_CSV=${FOLDS_CSV:-0,1}
SEEDS_CSV=${SEEDS_CSV:-3407,3411}
STAGE=${STAGE:-quick5full}
INCLUDE_RUN_SUBSTRING=${INCLUDE_RUN_SUBSTRING:-quick5full}
MAX_IMAGES=${MAX_IMAGES:-0}
GPU_LIST=${GPU_LIST:-}
MAX_PARALLEL=${MAX_PARALLEL:-}
MAX_GPUS=${MAX_GPUS:-0}
FREE_GPU_MAX_USED_MIB=${FREE_GPU_MAX_USED_MIB:-2500}
FREE_GPU_MAX_UTIL_PCT=${FREE_GPU_MAX_UTIL_PCT:-20}
DTA_V37_DYNAMIC_GPU=${DTA_V37_DYNAMIC_GPU:-1}
GPU_WAIT_SECONDS=${GPU_WAIT_SECONDS:-60}
GPU_LAUNCH_STAGGER_SECONDS=${GPU_LAUNCH_STAGGER_SECONDS:-5}

mkdir -p "$EVID"
{
  echo "dta_v3_7_phase_d3_tau_real_blend_start $(date --iso-8601=seconds)"
  echo "state=RUNNING_EVAL_D3_TAU_REAL_BLEND"
  echo "work=$WORK"
  echo "tau_work=$TAU_WORK"
  echo "python=$PY"
  echo "data=$DATA"
  echo "a0=$A0"
  echo "depth=$DEPTH"
  echo "split_json=$SPLIT_JSON"
  echo "action_table=$ACTION_TABLE"
  echo "folds=$FOLDS_CSV seeds=$SEEDS_CSV stage=$STAGE include_run_substring=$INCLUDE_RUN_SUBSTRING max_images=$MAX_IMAGES"
  echo "formal_full_5x3=false"
  echo "locked_test_touched=false"
  echo "real_blend_note=actual rendered D1 TAU quick5full tensor blends, train-root fold val only"
  echo "dynamic_gpu=$DTA_V37_DYNAMIC_GPU free_gpu_max_used_mib=$FREE_GPU_MAX_USED_MIB free_gpu_max_util_pct=$FREE_GPU_MAX_UTIL_PCT gpu_wait_seconds=$GPU_WAIT_SECONDS"
} | tee -a "$STATUS"

for p in "$WORK" "$TAU_WORK" "$PY" "$DATA" "$A0" "$DEPTH" "$SPLIT_JSON" "$ACTION_TABLE"; do
  if [[ ! -e "$p" ]]; then
    echo "DTA_V3_7_PHASE_D3_TAU_REAL_BLEND_MISSING_PATH $p" | tee -a "$STATUS"
    exit 3
  fi
done

cd "$WORK"
{ git branch --show-current; git rev-parse --short HEAD; git status --short; } | tee -a "$STATUS"


if [[ -z "$GPU_LIST" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    gpu_count=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
    if [[ "$gpu_count" -le 0 ]]; then gpu_count=1; fi
    GPU_LIST=$(seq -s, 0 $((gpu_count - 1)))
  else
    GPU_LIST=0
  fi
fi
IFS=',' read -r -a GPUS <<< "$GPU_LIST"
if [[ "$MAX_GPUS" -gt 0 && "${#GPUS[@]}" -gt "$MAX_GPUS" ]]; then
  GPUS=("${GPUS[@]:0:$MAX_GPUS}")
  GPU_LIST=$(IFS=,; echo "${GPUS[*]}")
fi
if [[ -z "$MAX_PARALLEL" ]]; then
  MAX_PARALLEL=${#GPUS[@]}
fi
if [[ "$MAX_PARALLEL" -lt 1 ]]; then MAX_PARALLEL=1; fi
if [[ "$MAX_PARALLEL" -gt "${#GPUS[@]}" ]]; then MAX_PARALLEL=${#GPUS[@]}; fi

echo "dta_v3_7_phase_d3_tau_real_blend_parallel gpu_candidates=$GPU_LIST max_parallel=$MAX_PARALLEL dynamic_gpu=$DTA_V37_DYNAMIC_GPU" | tee -a "$STATUS"

detect_free_gpus() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    printf '%s\n' "${GPUS[@]}"
    return
  fi
  local candidates=",$GPU_LIST,"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits |
    awk -F, -v max_used="$FREE_GPU_MAX_USED_MIB" -v max_util="$FREE_GPU_MAX_UTIL_PCT" -v candidates="$candidates" '
      {
        gsub(/ /, "", $1); gsub(/ /, "", $2); gsub(/ /, "", $3);
        gpu="," $1 ",";
        if (index(candidates, gpu) && $2 <= max_used && $3 <= max_util) print $1;
      }'
}

IFS=',' read -r -a FOLDS <<< "$FOLDS_CSV"
IFS=',' read -r -a SEEDS <<< "$SEEDS_CSV"
OUT_DIR=$EVID/phase_d3_tau_real_blend_groups
mkdir -p "$OUT_DIR"

TASKS=()
for seed in "${SEEDS[@]}"; do
  for fold in "${FOLDS[@]}"; do
    TASKS+=("$fold,$seed")
  done
done

task_idx=0
job_idx=0
fail=0
launch_one() {
  local gpu=$1
  local fold=$2
  local seed=$3
  job_idx=$((job_idx + 1))
  local tag=seed${seed}_f${fold}
  local log=$EVID/v37_phase_d3_tau_real_blend_${tag}.log
  echo "dta_v3_7_phase_d3_tau_real_blend_launch tag=$tag gpu=$gpu $(date --iso-8601=seconds)" | tee -a "$STATUS"
  (
    set -euo pipefail
    cd "$WORK"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$PY" \
      experience_docx/tools/eval_haze4k_dta_v37_tau_real_blend_oracle.py \
      --data_dir "$DATA" \
      --a0_checkpoint "$A0" \
      --checkpoint_root "$TAU_WORK" \
      --action_table_csv "$ACTION_TABLE" \
      --include_run_substring "$INCLUDE_RUN_SUBSTRING" \
      --depth_cache_dir "$DEPTH" \
      --split_json "$SPLIT_JSON" \
      --fold "$fold" \
      --seed "$seed" \
      --stage "$STAGE" \
      --output_dir "$OUT_DIR" \
      --max_images "$MAX_IMAGES"
  ) > "$log" 2>&1 &
}

while [[ "$task_idx" -lt "${#TASKS[@]}" ]]; do
  while [[ $(jobs -rp | wc -l | tr -d ' ') -ge "$MAX_PARALLEL" ]]; do
    if ! wait -n; then fail=1; fi
  done

  if [[ "$DTA_V37_DYNAMIC_GPU" == "1" ]]; then
    mapfile -t FREE_GPUS < <(detect_free_gpus)
    if [[ "${#FREE_GPUS[@]}" -eq 0 ]]; then
      if [[ $(jobs -rp | wc -l | tr -d ' ') -gt 0 ]]; then
        if ! wait -n; then fail=1; fi
      else
        echo "dta_v3_7_phase_d3_wait_free_gpu max_used_mib=$FREE_GPU_MAX_USED_MIB max_util_pct=$FREE_GPU_MAX_UTIL_PCT $(date --iso-8601=seconds)" | tee -a "$STATUS"
        sleep "$GPU_WAIT_SECONDS"
      fi
      continue
    fi
    gpu=${FREE_GPUS[0]}
  else
    gpu=${GPUS[$((job_idx % ${#GPUS[@]}))]}
  fi

  IFS=',' read -r fold seed <<< "${TASKS[$task_idx]}"
  task_idx=$((task_idx + 1))
  launch_one "$gpu" "$fold" "$seed"
  sleep "$GPU_LAUNCH_STAGGER_SECONDS"
done

while [[ $(jobs -rp | wc -l | tr -d ' ') -gt 0 ]]; do
  if ! wait -n; then fail=1; fi
done

if [[ "$fail" -ne 0 ]]; then
  echo "DTA_V3_7_PHASE_D3_TAU_REAL_BLEND_GROUP_FAILED $(date --iso-8601=seconds)" | tee -a "$STATUS"
  exit 1
fi

echo "dta_v3_7_phase_d3_tau_real_blend_group_jobs_done $(date --iso-8601=seconds)" | tee -a "$STATUS"
PYTHONUNBUFFERED=1 "$PY" experience_docx/tools/aggregate_haze4k_dta_v37_tau_real_blend_oracle.py \
  --input_glob "$OUT_DIR/v37_tau_real_blend_selected_seed*_f*.csv" \
  --output_dir "$EVID" \
  2>&1 | tee "$EVID/v37_phase_d3_tau_real_blend_aggregate.log"

echo "DTA_V3_7_PHASE_D3_TAU_REAL_BLEND_OK $(date --iso-8601=seconds)" | tee -a "$STATUS"
