#!/usr/bin/env bash
set -euo pipefail
FOLD=${1:?fold required}
GPU_ID=${2:?gpu required}
BASE=/sda/home/wangyuxin/ConvIR-B
WORK=$BASE/repos/ConvIR-B-haze4k-v4-8-train-derived-validation-reset-wt
ITS=$WORK/Dehazing/ITS
ROUTE_ID=haze4k_v4_8_train_derived_validation_reset_20260708
EVID=$WORK/experience_docx/experiment_logs/$ROUTE_ID
FOLD_DIR=$EVID/folds_bs7repair/fold_${FOLD}
PY=$BASE/envs/convir-cu121/bin/python
DATA=$BASE/datasets/Haze4K/Haze4K
A0=$BASE/checkpoints/official/Haze4K/haze4k-base.pkl
TRAIN_SPLIT=$EVID/splits/fold_${FOLD}_train.txt
VAL_SPLIT=$EVID/splits/fold_${FOLD}_val.txt
MODEL_NAME=ConvIR-Haze4K-v48-DCFSB-adapter4-fold${FOLD}-seed3407-bs7repair-20260708
OUT_DIR=$ITS/results/$MODEL_NAME
STATUS=$FOLD_DIR/status.txt
TRAIN_LOG=$FOLD_DIR/train_${MODEL_NAME}.log
mkdir -p "$FOLD_DIR"
{
  echo "train_start v48_fold${FOLD}_bs7repair $(date --iso-8601=seconds)"
  echo "state=RUNNING_TRAIN"
  echo "repair=batch_size_7_all_folds_for_consistent_oof"
  echo "fold=$FOLD"
  echo "gpu=$GPU_ID"
  echo "work=$WORK"
  echo "branch=$(cd "$WORK" && git branch --show-current)"
  echo "commit=$(cd "$WORK" && git rev-parse HEAD)"
  echo "model_name=$MODEL_NAME"
  echo "out_dir=$OUT_DIR"
  echo "train_split=$TRAIN_SPLIT"
  echo "val_split=$VAL_SPLIT"
  echo "data=$DATA"
  echo "a0=$A0"
  echo "python=$PY"
  echo "batch_size=7"
  echo "locked_test_policy=train-derived folds only; valid_freq=999 > stop_epoch=4"
} | tee -a "$STATUS"
if [ ! -x "$PY" ]; then echo "V48_FOLD${FOLD}_BS7REPAIR_FAILED python_missing" | tee -a "$STATUS"; exit 2; fi
if [ ! -f "$TRAIN_SPLIT" ]; then echo "V48_FOLD${FOLD}_BS7REPAIR_FAILED split_missing" | tee -a "$STATUS"; exit 2; fi
if [ ! -f "$A0" ]; then echo "V48_FOLD${FOLD}_BS7REPAIR_FAILED a0_missing" | tee -a "$STATUS"; exit 2; fi
if [ -e "$OUT_DIR" ]; then echo "V48_FOLD${FOLD}_BS7REPAIR_FAILED output_exists" | tee -a "$STATUS"; exit 3; fi
cd "$ITS"
set +e
CUDA_VISIBLE_DEVICES=$GPU_ID TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONUNBUFFERED=1 "$PY" main.py \
  --model_name "$MODEL_NAME" \
  --data Haze4K \
  --version base \
  --fam_mode original \
  --arch dcfsb_bottleneck \
  --dcfsb_train_scope adapter_only \
  --mode train \
  --data_dir "$DATA" \
  --train_split_file "$TRAIN_SPLIT" \
  --batch_size 7 \
  --learning_rate 0.0001 \
  --weight_decay 0.0001 \
  --grad_clip_norm 0.001 \
  --num_epoch 1000 \
  --stop_epoch 4 \
  --print_freq 50 \
  --num_worker 8 \
  --save_freq 4 \
  --valid_freq 999 \
  --mod_stats_freq 0 \
  --mod_stats_batches 0 \
  --init_model "$A0" \
  --seed 3407 \
  > "$TRAIN_LOG" 2>&1
rc=$?
set -e
if [ "$rc" -eq 0 ]; then
  echo "state=TRAIN_DONE" | tee -a "$STATUS"
  echo "train_done rc=0 v48_fold${FOLD}_bs7repair $(date --iso-8601=seconds)" | tee -a "$STATUS"
  echo "V48_FOLD${FOLD}_BS7REPAIR_TRAIN_OK" | tee -a "$STATUS"
else
  echo "state=FAILED_TRAIN" | tee -a "$STATUS"
  echo "train_done rc=$rc v48_fold${FOLD}_bs7repair $(date --iso-8601=seconds)" | tee -a "$STATUS"
  echo "V48_FOLD${FOLD}_BS7REPAIR_TRAIN_FAILED" | tee -a "$STATUS"
fi
exit "$rc"
