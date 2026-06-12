#!/usr/bin/env bash
set -euo pipefail

BASE=${BASE:-/sda/home/wangyuxin/ConvIR-B}
WORK=${WORK:-$BASE/repos/ConvIR-B-dta-v3-5-fdf-rcs-lite}
PY=${PY:-$BASE/envs/convir-cu121/bin/python}
EVID=$WORK/experience_docx/experiment_logs/haze4k_dta_v3_5_fdf_rcs_lite_20260612
STATUS=$EVID/status.txt
RUN_SCRIPT=$EVID/run_dta_v3_5_fdf_rcs_lite_convir4090.sh
STAGE=${STAGE:-quick5full}
VARIANTS_CSV=${VARIANTS_CSV:-l0_a0_sanity,l1_fdf_lite_s004_g025_bm2,l2_fdf_lite_s002_g025_bm2,l3_fdf_lite_s004_g015_bm2,l4_fdf_lite_tail_s004_g025_bm2,l5_res010_tail_s004_g025_bm2,l5_res015_tail_s004_g025_bm2}
FOLDS_CSV=${FOLDS_CSV:-0,1}
SEEDS_CSV=${SEEDS_CSV:-3407,3411}
MAX_IMAGES=${MAX_IMAGES:-0}
FORCE=${FORCE:-0}
GPU_LIST=${GPU_LIST:-}
MAX_PARALLEL=${MAX_PARALLEL:-}
FREE_GPU_MAX_USED_MIB=${FREE_GPU_MAX_USED_MIB:-2500}

mkdir -p "$EVID"
{
  echo "dta_v3_5_triage_queue_start stage=$STAGE $(date --iso-8601=seconds)"
  echo "work=$WORK"
  echo "python=$PY"
  echo "variants=$VARIANTS_CSV folds=$FOLDS_CSV seeds=$SEEDS_CSV"
  echo "max_images=$MAX_IMAGES force=$FORCE"
  echo "locked_test_touched=false"
} | tee -a "$STATUS"

cd "$WORK"
{ git branch --show-current; git rev-parse --short HEAD; git status --short; } | tee -a "$STATUS"
if [[ ! -x "$RUN_SCRIPT" ]]; then
  echo "DTA_V3_5_RUN_SCRIPT_MISSING $RUN_SCRIPT" | tee -a "$STATUS"
  exit 3
fi

IFS=',' read -r -a VARIANTS <<< "$VARIANTS_CSV"
IFS=',' read -r -a FOLDS <<< "$FOLDS_CSV"
IFS=',' read -r -a SEEDS <<< "$SEEDS_CSV"
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
if [[ -z "$MAX_PARALLEL" ]]; then
  MAX_PARALLEL=${#GPUS[@]}
fi
if [[ "$MAX_PARALLEL" -lt 1 ]]; then MAX_PARALLEL=1; fi

echo "dta_v3_5_triage_parallel gpu_list=$GPU_LIST max_parallel=$MAX_PARALLEL" | tee -a "$STATUS"

job_idx=0
fail=0
launch_one() {
  local variant=$1
  local fold=$2
  local seed=$3
  local gpu=${GPUS[$((job_idx % ${#GPUS[@]}))]}
  job_idx=$((job_idx + 1))
  local tag=${variant}_seed${seed}_f${fold}_${STAGE}
  local log=$EVID/queue_${tag}.log
  echo "dta_v3_5_queue_launch tag=$tag gpu=$gpu $(date --iso-8601=seconds)" | tee -a "$STATUS"
  (
    set -euo pipefail
    VARIANT="$variant" FOLD="$fold" SEED="$seed" STAGE="$STAGE" CUDA_VISIBLE_DEVICES="$gpu" \
      MAX_IMAGES="$MAX_IMAGES" FORCE="$FORCE" RUN_TRAIN_CONTROLS=1 RUN_TEST=0 USE_SPLIT=1 \
      "$RUN_SCRIPT"
  ) > "$log" 2>&1 &
}

for variant in "${VARIANTS[@]}"; do
  for fold in "${FOLDS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      if [[ "$variant" == "l0_a0_sanity" && "$seed" != "${SEEDS[0]}" ]]; then
        continue
      fi
      while [[ $(jobs -rp | wc -l | tr -d ' ') -ge "$MAX_PARALLEL" ]]; do
        if ! wait -n; then fail=1; fi
      done
      launch_one "$variant" "$fold" "$seed"
    done
  done
done

while [[ $(jobs -rp | wc -l | tr -d ' ') -gt 0 ]]; do
  if ! wait -n; then fail=1; fi
done

if [[ "$fail" -ne 0 ]]; then
  echo "DTA_V3_5_TRIAGE_QUEUE_FAILED $(date --iso-8601=seconds)" | tee -a "$STATUS"
  exit 1
fi

echo "dta_v3_5_triage_jobs_done $(date --iso-8601=seconds)" | tee -a "$STATUS"

expected_runs=$(( ${#FOLDS[@]} * ${#SEEDS[@]} ))
PYTHONUNBUFFERED=1 "$PY" experience_docx/tools/summarize_haze4k_dta_v35_triage.py \
  --evidence_dir "$EVID" \
  --output_json "$EVID/dta_v3_5_fdf_rcs_triage_summary.json" \
  --output_csv "$EVID/dta_v3_5_fdf_rcs_triage_summary.csv" \
  --variant_csv "$EVID/dta_v3_5_fdf_rcs_triage_variant_summary.csv" \
  --expected_runs_per_variant "$expected_runs" \
  2>&1 | tee "$EVID/dta_v3_5_triage_summary.log"

echo "dta_v3_5_triage_summary_done $(date --iso-8601=seconds)" | tee -a "$STATUS"

PYTHONUNBUFFERED=1 "$PY" experience_docx/tools/select_haze4k_dta_v35_nested_calibration.py \
  --evidence_dir "$EVID" \
  --action_table_csv "$EVID/v35_oof_per_image_action_table.csv" \
  --oracle_curve_csv "$EVID/v35_oracle_risk_coverage_curve.csv" \
  --nested_report_json "$EVID/v35_selector_nested_calibration_report.json" \
  --nested_report_csv "$EVID/v35_selector_nested_calibration_report.csv" \
  --nested_selected_csv "$EVID/v35_selector_nested_selected_images.csv" \
  --min_coverage 0.20 \
  --max_coverage 0.95 \
  2>&1 | tee "$EVID/dta_v3_5_nested_selector.log"

echo "DTA_V3_5_TRIAGE_QUEUE_OK $(date --iso-8601=seconds)" | tee -a "$STATUS"
