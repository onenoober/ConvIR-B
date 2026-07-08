#!/usr/bin/env bash
set -euo pipefail
FOLD=${1:?fold required}
GPU_ID=${2:?gpu required}
BASE=/sda/home/wangyuxin/ConvIR-B
WORK=$BASE/repos/ConvIR-B-haze4k-v4-8-train-derived-validation-reset-wt
ROUTE_ID=haze4k_v4_8_train_derived_validation_reset_20260708
EVID=$WORK/experience_docx/experiment_logs/$ROUTE_ID
PY=$BASE/envs/convir-cu121/bin/python
DATA=$BASE/datasets/Haze4K/Haze4K
A0=$BASE/checkpoints/official/Haze4K/haze4k-base.pkl
MODEL_NAME=ConvIR-Haze4K-v48-DCFSB-adapter4-fold${FOLD}-seed3407-bs7repair-20260708
CAND=$WORK/Dehazing/ITS/results/$MODEL_NAME/Training-Results/Final.pkl
SPLIT=$EVID/splits/fold_${FOLD}_val.txt
OUT=$EVID/oof_eval_bs7repair/fold_${FOLD}
STATUS=$OUT/status.txt
LOG=$OUT/eval_${MODEL_NAME}.log
mkdir -p "$OUT"
{
  echo "eval_start v48_fold${FOLD}_oof_bs7repair $(date --iso-8601=seconds)"
  echo "state=RUNNING_EVAL"
  echo "fold=$FOLD"
  echo "gpu=$GPU_ID"
  echo "work=$WORK"
  echo "branch=$(cd "$WORK" && git branch --show-current)"
  echo "commit=$(cd "$WORK" && git rev-parse HEAD)"
  echo "python=$PY"
  echo "data=$DATA"
  echo "a0=$A0"
  echo "candidate=$CAND"
  echo "split=$SPLIT"
  echo "out=$OUT"
  echo "locked_test_policy=train-derived fold val only; no test enumeration"
} | tee -a "$STATUS"
if [ ! -x "$PY" ]; then echo "V48_FOLD${FOLD}_OOF_EVAL_FAILED python_missing" | tee -a "$STATUS"; exit 2; fi
if [ ! -f "$A0" ]; then echo "V48_FOLD${FOLD}_OOF_EVAL_FAILED a0_missing" | tee -a "$STATUS"; exit 2; fi
if [ ! -f "$CAND" ]; then echo "V48_FOLD${FOLD}_OOF_EVAL_FAILED candidate_missing" | tee -a "$STATUS"; exit 2; fi
if [ ! -f "$SPLIT" ]; then echo "V48_FOLD${FOLD}_OOF_EVAL_FAILED split_missing" | tee -a "$STATUS"; exit 2; fi
if [ -f "$OUT/val_per_image_compact.csv" ]; then echo "V48_FOLD${FOLD}_OOF_EVAL_FAILED output_exists" | tee -a "$STATUS"; exit 3; fi
set +e
CUDA_VISIBLE_DEVICES=$GPU_ID TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONUNBUFFERED=1 "$PY" "$EVID/eval_v48_oof_fold.py" \
  --fold "$FOLD" \
  --work "$WORK" \
  --data-root "$DATA" \
  --split-file "$SPLIT" \
  --a0 "$A0" \
  --candidate "$CAND" \
  --output-dir "$OUT" \
  > "$LOG" 2>&1
rc=$?
set -e
if [ "$rc" -eq 0 ]; then
  echo "state=EVAL_DONE" | tee -a "$STATUS"
  echo "eval_done rc=0 v48_fold${FOLD}_oof_bs7repair $(date --iso-8601=seconds)" | tee -a "$STATUS"
  echo "V48_FOLD${FOLD}_OOF_EVAL_OK" | tee -a "$STATUS"
else
  echo "state=FAILED_EVAL" | tee -a "$STATUS"
  echo "eval_done rc=$rc v48_fold${FOLD}_oof_bs7repair $(date --iso-8601=seconds)" | tee -a "$STATUS"
  echo "V48_FOLD${FOLD}_OOF_EVAL_FAILED" | tee -a "$STATUS"
fi
exit "$rc"
