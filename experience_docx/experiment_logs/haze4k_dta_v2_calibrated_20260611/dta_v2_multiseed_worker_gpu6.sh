#!/usr/bin/env bash
set -euo pipefail
ROOT=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-v2-calibrated
EVID=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-v2-calibrated/experience_docx/experiment_logs/haze4k_dta_v2_calibrated_20260611
SCRIPT=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-v2-calibrated/experience_docx/experiment_logs/haze4k_dta_v2_calibrated_20260611/run_dta_v2_train_eval_convir4090.sh
SPLIT_JSON=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-v2-calibrated/experience_docx/experiment_logs/haze4k_dta_v2_calibrated_20260611/dta_v2_haze4k_oof_splits_seed3407.json
STATUS=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-v2-calibrated/experience_docx/experiment_logs/haze4k_dta_v2_calibrated_20260611/status.txt
GPU=6
JOB_FILE=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-v2-calibrated/experience_docx/experiment_logs/haze4k_dta_v2_calibrated_20260611/dta_v2_oof20_adapter_only_multiseed_3411_3413_jobs.tsv
while IFS=$'\t' read -r seed fold mode job_gpu; do
  [[ "$job_gpu" == "$GPU" ]] || continue
  run_id="oof20_adapter_only_${mode}_seed${seed}_f${fold}"
  echo "multiseed_job_start run_id=$run_id gpu=$GPU \2026-06-11T13:08:28+08:00" | tee -a "$STATUS"
  set +e
  CUDA_VISIBLE_DEVICES="$GPU" FOLD_TAG="f${fold}" SPLIT_JSON="$SPLIT_JSON" TRAIN_SPLIT="fold${fold}_train" EVAL_SPLIT="fold${fold}_val" TRAIN_DEPTH_SPLIT=train EVAL_DEPTH_SPLIT=train bash "$SCRIPT" oof20 adapter_only "$mode" "$seed"
  rc=$?
  set -e
  echo "multiseed_job_done rc=$rc run_id=$run_id gpu=$GPU \2026-06-11T13:08:28+08:00" | tee -a "$STATUS"
  if [[ "$rc" -ne 0 ]]; then
    echo "DTA_V2_MULTI_SEED_JOB_FAILED run_id=$run_id gpu=$GPU" | tee -a "$STATUS"
    exit "$rc"
  fi
done < "$JOB_FILE"
echo "DTA_V2_MULTI_SEED_WORKER_OK gpu=$GPU \2026-06-11T13:08:28+08:00" | tee -a "$STATUS"
