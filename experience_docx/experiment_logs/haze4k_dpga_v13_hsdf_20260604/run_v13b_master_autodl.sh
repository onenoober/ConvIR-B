#!/usr/bin/env bash
set -u

WORK=${WORK:-/root/autodl-tmp/workspace/ConvIR-B-v1-3-hsdf}
LOG_DIR=$WORK/experience_docx/experiment_logs/haze4k_dpga_v13_hsdf_20260604
MASTER_LOG=$LOG_DIR/v13b_master_autodl.log
STATUS=$LOG_DIR/status.txt

mkdir -p "$LOG_DIR"
{
  echo "v13b_master_start $(date --iso-8601=seconds)"
  echo "work=$WORK"
} | tee -a "$STATUS" "$MASTER_LOG"

cd "$WORK"

if [[ ! -f "$LOG_DIR/intenal_val/haze4k_dpga_v13_regular_hard_seed3407.json" ]]; then
  bash "$LOG_DIR/run_make_v13_hard_splits.sh"
  split_rc=$?
else
  split_rc=0
fi
echo "v13b_master_split_rc=$split_rc $(date --iso-8601=seconds)" | tee -a "$MASTER_LOG"
if [[ "$split_rc" -ne 0 ]]; then
  echo "v13b_master_stop split_failed" | tee -a "$STATUS" "$MASTER_LOG"
  exit "$split_rc"
fi

bash "$LOG_DIR/run_dpga_v13b_train.sh"
train_rc=$?
echo "v13b_master_train_rc=$train_rc $(date --iso-8601=seconds)" | tee -a "$MASTER_LOG"

if [[ "$train_rc" -eq 0 ]]; then
  bash "$LOG_DIR/run_eval_dpga_v13b_regular_hard.sh"
  eval_rc=$?
  echo "v13b_master_eval_rc=$eval_rc $(date --iso-8601=seconds)" | tee -a "$MASTER_LOG"
  bash "$LOG_DIR/run_dpga_v13b_runtime_ablation_val_regular.sh"
  runtime_rc=$?
  echo "v13b_master_runtime_rc=$runtime_rc $(date --iso-8601=seconds)" | tee -a "$MASTER_LOG"
else
  eval_rc=99
  runtime_rc=99
fi

{
  echo "v13b_master_done split=$split_rc train=$train_rc eval=$eval_rc runtime=$runtime_rc $(date --iso-8601=seconds)"
} | tee -a "$STATUS" "$MASTER_LOG"

if [[ "$split_rc" -ne 0 || "$train_rc" -ne 0 || "$eval_rc" -ne 0 || "$runtime_rc" -ne 0 ]]; then
  exit 1
fi
