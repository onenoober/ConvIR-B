#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v2-16-nopost-wavelet-lowband-decoder
EVID="$REMOTE_ROOT/experience_docx/experiment_logs/haze4k_v2_16_nopost_wavelet_lowband_decoder_20260703"
RUN_DIR="$EVID/wldb_a_seed3407"
PY=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
DATA=/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K
CKPT=/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl
SPLIT="$EVID/v216_t1_per_image_band_deltas.csv"
STATUS="$EVID/status.txt"
TRAIN_LOG="$RUN_DIR/v216_wldb_a_seed3407_train.log"
EVAL_LOG="$RUN_DIR/v216_wldb_a_seed3407_eval.log"

mkdir -p "$RUN_DIR"
cd "$REMOTE_ROOT"

{
  echo "wldb_a_state=RUNNING_TRAIN"
  echo "wldb_a_seed=3407"
  echo "locked_test_touched=false"
  echo "wldb_a_run_start $(date --iso-8601=seconds)"
} | tee -a "$STATUS"

test -x "$PY"
test -d "$DATA/train"
test -f "$CKPT"
test -f "$SPLIT"

set +e
"$PY" experience_docx/tools/train_nopost_wldb_a.py \
  --data-dir "$DATA" \
  --checkpoint "$CKPT" \
  --split-csv "$SPLIT" \
  --out-dir "$RUN_DIR" \
  --fold 0 \
  --seed 3407 \
  --epochs 20 \
  --batch-size 8 \
  --num-workers 4 \
  --crop-size 256 \
  --lr 2e-4 \
  --lowband-weight 0.25 \
  --preserve-weight 0.02 \
  --budget-weight 0.05 \
  --action-budget 0.025 \
  --save-freq 5 \
  --print-freq 50 \
  2>&1 | tee "$TRAIN_LOG"
train_rc=${PIPESTATUS[0]}
set -e

if [ "$train_rc" -ne 0 ]; then
  echo "wldb_a_train_done rc=$train_rc $(date --iso-8601=seconds)" | tee -a "$STATUS"
  echo "wldb_a_state=FAILED_TRAIN_COMMAND" | tee -a "$STATUS"
  exit "$train_rc"
fi

echo "wldb_a_state=RUNNING_EVAL" | tee -a "$STATUS"

set +e
"$PY" experience_docx/tools/eval_nopost_wldb_a.py \
  --data-dir "$DATA" \
  --official-checkpoint "$CKPT" \
  --split-csv "$SPLIT" \
  --out-dir "$RUN_DIR" \
  --fold 0 \
  --checkpoint model_5="$RUN_DIR/checkpoints/model_5.pkl" \
  --checkpoint model_10="$RUN_DIR/checkpoints/model_10.pkl" \
  --checkpoint model_15="$RUN_DIR/checkpoints/model_15.pkl" \
  --checkpoint model_20="$RUN_DIR/checkpoints/model_20.pkl" \
  --checkpoint Best="$RUN_DIR/checkpoints/Best.pkl" \
  --checkpoint Final="$RUN_DIR/checkpoints/Final.pkl" \
  2>&1 | tee "$EVAL_LOG"
eval_rc=${PIPESTATUS[0]}
set -e

{
  echo "wldb_a_eval_done rc=$eval_rc $(date --iso-8601=seconds)"
  if [ "$eval_rc" -eq 0 ]; then
    echo "wldb_a_state=COMPLETED_SCREEN"
    echo "V216_WLDB_A_SEED3407_RUN_OK"
  else
    echo "wldb_a_state=FAILED_EVAL_COMMAND"
    echo "V216_WLDB_A_SEED3407_RUN_FAILED"
  fi
} | tee -a "$STATUS"

exit "$eval_rc"
