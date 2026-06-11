#!/usr/bin/env bash
set -euo pipefail
ROOT=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-v2-calibrated
EVID=$ROOT/experience_docx/experiment_logs/haze4k_dta_v2_calibrated_20260611
SCRIPT=$EVID/run_dta_v2_train_eval_convir4090.sh
SPLIT_JSON=$EVID/dta_v2_haze4k_oof_splits_seed3407.json
STATUS=$EVID/status.txt
declare -a JOBS=(
  "1 invert 0"
  "1 normal 1"
  "1 shuffle 2"
  "1 zero 3"
  "2 invert 4"
  "2 normal 5"
  "2 shuffle 6"
  "2 zero 7"
)
for job in "${JOBS[@]}"; do
  read -r fold mode gpu <<<"$job"
  session="dta_v2_oof20_adapter_only_${mode}_f${fold}"
  run_id="oof20_adapter_only_${mode}_seed3407_f${fold}"
  echo "launch_start run_id=$run_id session=$session gpu=$gpu $(date --iso-8601=seconds)" | tee -a "$STATUS"
  tmux new-session -d -s "$session" "cd '$ROOT' && CUDA_VISIBLE_DEVICES=$gpu FOLD_TAG=f${fold} SPLIT_JSON='$SPLIT_JSON' TRAIN_SPLIT=fold${fold}_train EVAL_SPLIT=fold${fold}_val TRAIN_DEPTH_SPLIT=train EVAL_DEPTH_SPLIT=train bash '$SCRIPT' oof20 adapter_only '$mode' 3407"
  sleep 2
  tmux has-session -t "$session"
  echo "launch_ok run_id=$run_id session=$session gpu=$gpu $(date --iso-8601=seconds)" | tee -a "$STATUS"
done
echo "DTA_V2_OOF20_ADAPTER_ONLY_F1_F2_LAUNCH_OK $(date --iso-8601=seconds)" | tee -a "$STATUS"
