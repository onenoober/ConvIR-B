#!/usr/bin/env bash
set -euo pipefail

BASE=${BASE:-/sda/home/wangyuxin/ConvIR-B}
WORK=${WORK:-$BASE/repos/ConvIR-B-dta-v3-7-u-tqs-mix-phasec1}
V36_WORK=${V36_WORK:-$BASE/repos/ConvIR-B-dta-v3-6-hrcs}
PY=${PY:-$BASE/envs/convir-cu121/bin/python}
DATA=${DATA:-$BASE/datasets/Haze4K/Haze4K}
A0=${A0:-$BASE/checkpoints/official/Haze4K/haze4k-base.pkl}
DEPTH=${DEPTH:-$BASE/depth_cache/depth_anything_v2_small_hf}
EVID=$WORK/experience_docx/experiment_logs/haze4k_dta_v3_7_u_tqs_mix_20260613
STATUS=$EVID/status_phase_c1_real_blend.txt
SPLIT_JSON=${SPLIT_JSON:-$WORK/experience_docx/experiment_logs/haze4k_dta_v3_6_hrcs_20260613/dta_v3_6_haze4k_oof_splits_seed3407.json}
FOLDS_CSV=${FOLDS_CSV:-0,1,2,3,4}
SEEDS_CSV=${SEEDS_CSV:-3407,3411,2026}
STAGE=${STAGE:-quick5full}
MAX_IMAGES=${MAX_IMAGES:-0}
GPU_LIST=${GPU_LIST:-}
MAX_PARALLEL=${MAX_PARALLEL:-}
MAX_GPUS=${MAX_GPUS:-0}
FREE_GPU_MAX_USED_MIB=${FREE_GPU_MAX_USED_MIB:-2500}

mkdir -p "$EVID"
{
  echo "dta_v3_7_phase_c1_real_blend_start $(date --iso-8601=seconds)"
  echo "state=RUNNING_EVAL"
  echo "work=$WORK"
  echo "v36_work=$V36_WORK"
  echo "python=$PY"
  echo "data=$DATA"
  echo "a0=$A0"
  echo "depth=$DEPTH"
  echo "split_json=$SPLIT_JSON"
  echo "folds=$FOLDS_CSV seeds=$SEEDS_CSV stage=$STAGE max_images=$MAX_IMAGES"
  echo "locked_test_touched=false"
  echo "real_blend_note=actual rendered tensor blends, train-root fold val only"
} | tee -a "$STATUS"

for p in "$WORK" "$V36_WORK" "$PY" "$DATA" "$A0" "$DEPTH" "$SPLIT_JSON"; do
  if [[ ! -e "$p" ]]; then
    echo "DTA_V3_7_PHASE_C1_MISSING_PATH $p" | tee -a "$STATUS"
    exit 3
  fi
done

cd "$WORK"
{ git branch --show-current; git rev-parse --short HEAD; git status --short; } | tee -a "$STATUS"

if [[ -z "$GPU_LIST" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_LIST=$(
      nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
      awk -F, -v max_used="$FREE_GPU_MAX_USED_MIB" '{gsub(/ /, "", $1); gsub(/ /, "", $2); if ($2 <= max_used) print $1}' |
      paste -sd, -
    )
    if [[ -z "$GPU_LIST" ]]; then
      gpu_count=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
      if [[ "$gpu_count" -le 0 ]]; then gpu_count=1; fi
      GPU_LIST=$(seq -s, 0 $((gpu_count - 1)))
    fi
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
echo "dta_v3_7_phase_c1_parallel gpu_list=$GPU_LIST max_parallel=$MAX_PARALLEL" | tee -a "$STATUS"

IFS=',' read -r -a FOLDS <<< "$FOLDS_CSV"
IFS=',' read -r -a SEEDS <<< "$SEEDS_CSV"
OUT_DIR=$EVID/phase_c1_real_blend_groups
mkdir -p "$OUT_DIR"

job_idx=0
fail=0
launch_one() {
  local fold=$1
  local seed=$2
  local gpu=${GPUS[$((job_idx % ${#GPUS[@]}))]}
  job_idx=$((job_idx + 1))
  local tag=seed${seed}_f${fold}
  local log=$EVID/v37_phase_c1_real_blend_${tag}.log
  echo "dta_v3_7_phase_c1_launch tag=$tag gpu=$gpu $(date --iso-8601=seconds)" | tee -a "$STATUS"
  (
    set -euo pipefail
    cd "$WORK"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$PY" \
      experience_docx/tools/eval_haze4k_dta_v37_real_blend_oracle.py \
      --data_dir "$DATA" \
      --a0_checkpoint "$A0" \
      --checkpoint_root "$V36_WORK" \
      --depth_cache_dir "$DEPTH" \
      --split_json "$SPLIT_JSON" \
      --fold "$fold" \
      --seed "$seed" \
      --stage "$STAGE" \
      --output_dir "$OUT_DIR" \
      --max_images "$MAX_IMAGES"
  ) > "$log" 2>&1 &
}

for seed in "${SEEDS[@]}"; do
  for fold in "${FOLDS[@]}"; do
    while [[ $(jobs -rp | wc -l | tr -d ' ') -ge "$MAX_PARALLEL" ]]; do
      if ! wait -n; then fail=1; fi
    done
    launch_one "$fold" "$seed"
  done
done

while [[ $(jobs -rp | wc -l | tr -d ' ') -gt 0 ]]; do
  if ! wait -n; then fail=1; fi
done

if [[ "$fail" -ne 0 ]]; then
  echo "DTA_V3_7_PHASE_C1_REAL_BLEND_GROUP_FAILED $(date --iso-8601=seconds)" | tee -a "$STATUS"
  exit 1
fi

echo "dta_v3_7_phase_c1_group_jobs_done $(date --iso-8601=seconds)" | tee -a "$STATUS"
PYTHONUNBUFFERED=1 "$PY" experience_docx/tools/aggregate_haze4k_dta_v37_real_blend_oracle.py \
  --input_glob "$OUT_DIR/v37_real_blend_selected_seed*_f*.csv" \
  --output_dir "$EVID" \
  2>&1 | tee "$EVID/v37_phase_c1_real_blend_aggregate.log"

echo "DTA_V3_7_PHASE_C1_REAL_BLEND_OK $(date --iso-8601=seconds)" | tee -a "$STATUS"
