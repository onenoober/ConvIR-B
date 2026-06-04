#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-/root/miniconda3/envs/convir-cu128/bin/python}
WORK=${WORK:-/root/autodl-tmp/workspace/ConvIR-B-v1-3-hsdf}
DATA=${DATA:-/root/autodl-tmp/workspace/Dehaze-Net/dataset/HAZE4K}
DEPTH=${DEPTH:-/root/autodl-tmp/workspace/Dehaze-Net/experiment/HAZE4K/depth_cache/depth_anything_v2_small_hf}
A0=${A0:-/root/autodl-tmp/workspace/ConvIR-B/Dehazing/pretrained_models/haze4k-base.pkl}
LOG_DIR=$WORK/experience_docx/experiment_logs/haze4k_dpga_v13_hsdf_20260604
OUT=$LOG_DIR/intenal_val/haze4k_dpga_v13_regular_hard_seed3407.json
STATUS=$LOG_DIR/status.txt

mkdir -p "$LOG_DIR/intenal_val"
{
  echo "v13_split_start $(date --iso-8601=seconds)"
  echo "work=$WORK"
  echo "data=$DATA"
  echo "depth=$DEPTH"
  echo "a0=$A0"
} | tee -a "$STATUS"

cd "$WORK"
PYTHONUNBUFFERED=1 "$PY" experience_docx/tools/make_haze4k_dpga_v13_hard_splits.py \
  --its_dir Dehazing/ITS \
  --data_dir "$DATA" \
  --a0_checkpoint "$A0" \
  --depth_cache_dir "$DEPTH" \
  --depth_split train \
  --output "$OUT" \
  --seed 3407 \
  --val_regular_count 300 \
  --val_hard_count 300 \
  > "$LOG_DIR/make_v13_hard_splits.log" 2>&1

echo "v13_split_done rc=$? output=$OUT $(date --iso-8601=seconds)" | tee -a "$STATUS"
