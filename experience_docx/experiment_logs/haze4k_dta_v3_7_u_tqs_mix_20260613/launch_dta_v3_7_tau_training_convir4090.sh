#!/usr/bin/env bash
set -euo pipefail

BASE=${BASE:-/sda/home/wangyuxin/ConvIR-B}
WORK=${WORK:-$BASE/repos/ConvIR-B-dta-v3-7-u-tqs-mix-phased1}
PY=${PY:-$BASE/envs/convir-cu121/bin/python}
EVID=$WORK/experience_docx/experiment_logs/haze4k_dta_v3_7_u_tqs_mix_20260613
STATUS=$EVID/status_phase_d1_tau_training.txt
RUN_SCRIPT=$EVID/run_dta_v3_7_tau_candidate_convir4090.sh
STAGE=${STAGE:-quick5full}
VARIANTS_CSV=${VARIANTS_CSV:-u1_tau_l1_s004_g025_a006,u2_tau_l3_s004_g015_a006,u3_tau_l2_s002_g025_a006}
FOLDS_CSV=${FOLDS_CSV:-0,1}
SEEDS_CSV=${SEEDS_CSV:-3407,3411}
DTA_V37_STAGE_SCREEN_ONLY=${DTA_V37_STAGE_SCREEN_ONLY:-1}
DTA_V37_STAGE_SCREEN_VARIANTS=${DTA_V37_STAGE_SCREEN_VARIANTS:-$VARIANTS_CSV}
DTA_V37_STAGE_SCREEN_FOLDS=${DTA_V37_STAGE_SCREEN_FOLDS:-$FOLDS_CSV}
DTA_V37_STAGE_SCREEN_SEEDS=${DTA_V37_STAGE_SCREEN_SEEDS:-$SEEDS_CSV}
DTA_V37_RUN_BATCH_TAG=${DTA_V37_RUN_BATCH_TAG:-}
MAX_IMAGES=${MAX_IMAGES:-0}
FORCE=${FORCE:-0}
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
  echo "dta_v3_7_tau_training_queue_start stage=$STAGE $(date --iso-8601=seconds)"
  echo "state=RUNNING_TRAIN_DERIVED_INTEGRATED_TAU_FORMAL_QUEUE"
  echo "work=$WORK"
  echo "python=$PY"
  echo "variants=$VARIANTS_CSV folds=$FOLDS_CSV seeds=$SEEDS_CSV"
  echo "stage_screen_only=$DTA_V37_STAGE_SCREEN_ONLY screen_variants=$DTA_V37_STAGE_SCREEN_VARIANTS screen_folds=$DTA_V37_STAGE_SCREEN_FOLDS screen_seeds=$DTA_V37_STAGE_SCREEN_SEEDS"
  echo "run_batch_tag=$DTA_V37_RUN_BATCH_TAG"
  echo "formal_full_5x3_requires_explicit_screen_promotion=true"
  echo "max_images=$MAX_IMAGES force=$FORCE max_gpus=$MAX_GPUS"
  echo "dynamic_gpu=$DTA_V37_DYNAMIC_GPU free_gpu_max_used_mib=$FREE_GPU_MAX_USED_MIB free_gpu_max_util_pct=$FREE_GPU_MAX_UTIL_PCT gpu_wait_seconds=$GPU_WAIT_SECONDS gpu_launch_stagger_seconds=$GPU_LAUNCH_STAGGER_SECONDS"
  echo "locked_test_touched=false"
} | tee -a "$STATUS"

cd "$WORK"
{ git branch --show-current; git rev-parse --short HEAD; git status --short; } | tee -a "$STATUS"
if [[ ! -x "$RUN_SCRIPT" ]]; then
  echo "DTA_V3_7_TAU_RUN_SCRIPT_MISSING $RUN_SCRIPT" | tee -a "$STATUS"
  exit 3
fi

IFS=',' read -r -a VARIANTS <<< "$VARIANTS_CSV"
IFS=',' read -r -a FOLDS <<< "$FOLDS_CSV"
IFS=',' read -r -a SEEDS <<< "$SEEDS_CSV"
if [[ -z "$GPU_LIST" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    gpu_count=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
    if [[ "$gpu_count" -le 0 ]]; then gpu_count=1; fi
    if [[ "$DTA_V37_DYNAMIC_GPU" == "1" ]]; then
      GPU_LIST=$(seq -s, 0 $((gpu_count - 1)))
    else
      GPU_LIST=$(
        nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
        awk -F, -v max_used="$FREE_GPU_MAX_USED_MIB" '{gsub(/ /, "", $1); gsub(/ /, "", $2); if ($2 <= max_used) print $1}' |
        paste -sd, -
      )
      if [[ -z "$GPU_LIST" ]]; then
        GPU_LIST=$(seq -s, 0 $((gpu_count - 1)))
      fi
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

echo "dta_v3_7_tau_parallel gpu_candidates=$GPU_LIST max_parallel=$MAX_PARALLEL dynamic_gpu=$DTA_V37_DYNAMIC_GPU" | tee -a "$STATUS"

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

job_idx=0
fail=0
launch_one() {
  local gpu=$1
  shift
  local variant=$1
  local fold=$2
  local seed=$3
  job_idx=$((job_idx + 1))
  local tag=${variant}_seed${seed}_f${fold}_${STAGE}
  local log_tag=$tag
  if [[ -n "$DTA_V37_RUN_BATCH_TAG" ]]; then log_tag=${tag}_${DTA_V37_RUN_BATCH_TAG}; fi
  local log=$EVID/phase_d1_tau_queue_${log_tag}.log
  echo "dta_v3_7_tau_launch tag=$tag batch_tag=$DTA_V37_RUN_BATCH_TAG gpu=$gpu $(date --iso-8601=seconds)" | tee -a "$STATUS"
  (
    set -euo pipefail
    VARIANT="$variant" FOLD="$fold" SEED="$seed" STAGE="$STAGE" CUDA_VISIBLE_DEVICES="$gpu" \
      MAX_IMAGES="$MAX_IMAGES" FORCE="$FORCE" RUN_TRAIN_CONTROLS=1 USE_SPLIT=1 \
      DTA_V37_STAGE_SCREEN_ONLY="$DTA_V37_STAGE_SCREEN_ONLY" \
      DTA_V37_STAGE_SCREEN_VARIANTS="$DTA_V37_STAGE_SCREEN_VARIANTS" \
      DTA_V37_STAGE_SCREEN_FOLDS="$DTA_V37_STAGE_SCREEN_FOLDS" \
      DTA_V37_STAGE_SCREEN_SEEDS="$DTA_V37_STAGE_SCREEN_SEEDS" \
      DTA_V37_RUN_BATCH_TAG="$DTA_V37_RUN_BATCH_TAG" \
      "$RUN_SCRIPT"
  ) > "$log" 2>&1 &
}

TASKS=()
for variant in "${VARIANTS[@]}"; do
  for fold in "${FOLDS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      TASKS+=("$variant,$fold,$seed")
    done
  done
done

task_idx=0
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
        echo "dta_v3_7_tau_wait_free_gpu max_used_mib=$FREE_GPU_MAX_USED_MIB max_util_pct=$FREE_GPU_MAX_UTIL_PCT $(date --iso-8601=seconds)" | tee -a "$STATUS"
        sleep "$GPU_WAIT_SECONDS"
      fi
      continue
    fi
    gpu=${FREE_GPUS[0]}
    IFS=',' read -r variant fold seed <<< "${TASKS[$task_idx]}"
    task_idx=$((task_idx + 1))
    launch_one "$gpu" "$variant" "$fold" "$seed"
    sleep "$GPU_LAUNCH_STAGGER_SECONDS"
  else
    IFS=',' read -r variant fold seed <<< "${TASKS[$task_idx]}"
    task_idx=$((task_idx + 1))
    gpu=${GPUS[$((job_idx % ${#GPUS[@]}))]}
    launch_one "$gpu" "$variant" "$fold" "$seed"
  fi
done

while [[ $(jobs -rp | wc -l | tr -d ' ') -gt 0 ]]; do
  if ! wait -n; then fail=1; fi
done

if [[ "$fail" -ne 0 ]]; then
  echo "DTA_V3_7_TAU_QUEUE_FAILED $(date --iso-8601=seconds)" | tee -a "$STATUS"
  exit 1
fi

echo "dta_v3_7_tau_jobs_done $(date --iso-8601=seconds)" | tee -a "$STATUS"
PYTHONUNBUFFERED=1 "$PY" experience_docx/tools/select_haze4k_dta_v35_nested_calibration.py \
  --evidence_dir "$EVID" \
  --action_table_csv "$EVID/v37_tau_oof_per_image_action_table.csv" \
  --oracle_curve_csv "$EVID/v37_tau_oracle_risk_coverage_curve.csv" \
  --nested_report_json "$EVID/v37_tau_selector_nested_calibration_report.json" \
  --nested_report_csv "$EVID/v37_tau_selector_nested_calibration_report.csv" \
  --nested_selected_csv "$EVID/v37_tau_selector_nested_selected_images.csv" \
  --min_coverage 0.20 \
  --max_coverage 0.95 \
  2>&1 | tee "$EVID/dta_v3_7_tau_build_action_table.log"

echo "DTA_V3_7_TAU_QUEUE_OK $(date --iso-8601=seconds)" | tee -a "$STATUS"
