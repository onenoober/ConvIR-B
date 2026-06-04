#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-/root/miniconda3/envs/convir-cu128/bin/python}
WORK=${WORK:-/root/autodl-tmp/workspace/ConvIR-B-dpga-tail-control}
DATA=${DATA:-/root/autodl-tmp/workspace/Dehaze-Net/dataset/HAZE4K}
LOG_DIR=$WORK/experience_docx/experiment_logs/haze4k_dpga_tail_control_20260604
OUT_DIR=$LOG_DIR/internal_val
SPLIT_JSON=$OUT_DIR/haze4k_train_inner_val_inner_seed3407.json
LOG=$LOG_DIR/make_internal_val_split.log
STATUS=$LOG_DIR/status.txt

mkdir -p "$OUT_DIR"

{
  echo "make_internal_val_split_start $(date --iso-8601=seconds)"
  echo "work=$WORK"
  echo "data=$DATA"
  echo "output=$SPLIT_JSON"
  "$PY" "$WORK/experience_docx/tools/make_haze4k_internal_val_split.py" \
    --data_dir "$DATA" \
    --output "$SPLIT_JSON" \
    --val_count 300 \
    --seed 3407
  echo "make_internal_val_split_done $(date --iso-8601=seconds)"
} 2>&1 | tee "$LOG"

echo "internal_val_split=$SPLIT_JSON" | tee -a "$STATUS"
