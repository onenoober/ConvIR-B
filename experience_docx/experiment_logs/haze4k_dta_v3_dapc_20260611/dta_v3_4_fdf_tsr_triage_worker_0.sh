#!/usr/bin/env bash
set -euo pipefail
EVID="/home/caozhiyang/ConvIR-B/repos/ConvIR-B-dta-v3-4-fdf-tsr-finetune/experience_docx/experiment_logs/haze4k_dta_v3_dapc_20260611"
TASKS="/home/caozhiyang/ConvIR-B/repos/ConvIR-B-dta-v3-4-fdf-tsr-finetune/experience_docx/experiment_logs/haze4k_dta_v3_dapc_20260611/dta_v3_4_fdf_tsr_triage_tasks.tsv"
RUN_SCRIPT="/home/caozhiyang/ConvIR-B/repos/ConvIR-B-dta-v3-4-fdf-tsr-finetune/experience_docx/experiment_logs/haze4k_dta_v3_dapc_20260611/run_dta_v3_4_fdf_tsr_convir5090.sh"
STATUS="/home/caozhiyang/ConvIR-B/repos/ConvIR-B-dta-v3-4-fdf-tsr-finetune/experience_docx/experiment_logs/haze4k_dta_v3_dapc_20260611/status.txt"
WORKER=0
WORKER_COUNT=2
GPU=5
STAGE="quick5full"
FORCE="0"
idx=0
while IFS=$'\t' read -r variant seed fold; do
  if (( idx % WORKER_COUNT == WORKER )); then
    echo "dta_v3_4_triage_task_start worker=$WORKER gpu=$GPU idx=$idx variant=$variant seed=$seed fold=$fold $(date --iso-8601=seconds)" | tee -a "$STATUS"
    CUDA_VISIBLE_DEVICES="$GPU" VARIANT="$variant" SEED="$seed" FOLD="$fold" STAGE="$STAGE" USE_SPLIT=1 RUN_TRAIN_CONTROLS=1 RUN_TEST=0 FORCE="$FORCE" bash "$RUN_SCRIPT"
    echo "dta_v3_4_triage_task_done worker=$WORKER gpu=$GPU idx=$idx variant=$variant seed=$seed fold=$fold $(date --iso-8601=seconds)" | tee -a "$STATUS"
  fi
  idx=$((idx + 1))
done < "$TASKS"
echo "DTA_V3_4_TRIAGE_WORKER_OK worker=$WORKER gpu=$GPU $(date --iso-8601=seconds)" | tee -a "$STATUS"
