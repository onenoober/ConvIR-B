#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-/root/miniconda3/envs/convir-cu128/bin/python}
WORK=${WORK:-/root/autodl-tmp/workspace/ConvIR-B-v1-4-udp-lite}
DATA=${DATA:-/root/autodl-tmp/workspace/Dehaze-Net/dataset/HAZE4K}
DEPTH=${DEPTH:-/root/autodl-tmp/workspace/Dehaze-Net/experiment/HAZE4K/depth_cache/depth_anything_v2_small_hf}
A0=${A0:-/root/autodl-tmp/workspace/ConvIR-B/Dehazing/pretrained_models/haze4k-base.pkl}
LOG_DIR=$WORK/experience_docx/experiment_logs/haze4k_udp_lite_v14_20260604
OUT=$LOG_DIR/preflight
STATUS=$LOG_DIR/status.txt

mkdir -p "$OUT"
{
  echo "v14_zero_init_start $(date --iso-8601=seconds)"
  echo "work=$WORK"
  echo "data=$DATA"
  echo "depth=$DEPTH"
  echo "a0=$A0"
} | tee -a "$STATUS"

cd "$WORK"
PYTHONUNBUFFERED=1 "$PY" experience_docx/tools/check_haze4k_udp_lite_zero_init_equivalence.py \
  --its_dir Dehazing/ITS \
  --checkpoint "$A0" \
  --data_dir "$DATA" \
  --depth_cache_dir "$DEPTH" \
  --depth_split test \
  --dpga_active_adapters dpfm \
  --dpga_udp_components all \
  --tolerance 1e-6 \
  --output_json "$OUT/v14_zero_init_equivalence.json" \
  > "$OUT/v14_zero_init_equivalence.log" 2>&1

echo "v14_zero_init_done rc=$? output=$OUT/v14_zero_init_equivalence.json $(date --iso-8601=seconds)" | tee -a "$STATUS"
