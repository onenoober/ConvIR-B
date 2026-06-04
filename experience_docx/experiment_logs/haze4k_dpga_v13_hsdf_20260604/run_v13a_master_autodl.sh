#!/usr/bin/env bash
set -u

WORK=${WORK:-/root/autodl-tmp/workspace/ConvIR-B-v1-3-hsdf}
LOG_DIR=$WORK/experience_docx/experiment_logs/haze4k_dpga_v13_hsdf_20260604
MASTER_LOG=$LOG_DIR/v13a_master_autodl.log
STATUS=$LOG_DIR/status.txt

mkdir -p "$LOG_DIR"
{
  echo "v13a_master_start $(date --iso-8601=seconds)"
  echo "work=$WORK"
} | tee -a "$STATUS" "$MASTER_LOG"

cd "$WORK"

bash "$LOG_DIR/run_make_v13_hard_splits.sh"
split_rc=$?
echo "v13a_master_split_rc=$split_rc $(date --iso-8601=seconds)" | tee -a "$MASTER_LOG"
if [[ "$split_rc" -ne 0 ]]; then
  echo "v13a_master_stop split_failed" | tee -a "$STATUS" "$MASTER_LOG"
  exit "$split_rc"
fi

bash "$LOG_DIR/run_v13_intermediate_audits.sh" &
audit_pid=$!

bash "$LOG_DIR/run_dpga_v13a_train.sh"
train_rc=$?
echo "v13a_master_train_rc=$train_rc $(date --iso-8601=seconds)" | tee -a "$MASTER_LOG"

wait "$audit_pid"
audit_rc=$?
echo "v13a_master_audit_rc=$audit_rc $(date --iso-8601=seconds)" | tee -a "$MASTER_LOG"

if [[ "$train_rc" -eq 0 ]]; then
  bash "$LOG_DIR/run_eval_dpga_v13a_regular_hard.sh"
  eval_rc=$?
  echo "v13a_master_eval_rc=$eval_rc $(date --iso-8601=seconds)" | tee -a "$MASTER_LOG"
  bash "$LOG_DIR/run_dpga_v13_runtime_ablation_val_regular.sh"
  runtime_rc=$?
  echo "v13a_master_runtime_rc=$runtime_rc $(date --iso-8601=seconds)" | tee -a "$MASTER_LOG"
else
  eval_rc=99
  runtime_rc=99
fi

{
  echo "v13a_master_done split=$split_rc audit=$audit_rc train=$train_rc eval=$eval_rc runtime=$runtime_rc $(date --iso-8601=seconds)"
} | tee -a "$STATUS" "$MASTER_LOG"

if [[ "$split_rc" -ne 0 || "$audit_rc" -ne 0 || "$train_rc" -ne 0 || "$eval_rc" -ne 0 || "$runtime_rc" -ne 0 ]]; then
  exit 1
fi
