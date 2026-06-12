#!/usr/bin/env bash
set -euo pipefail
BASE=/home/caozhiyang/ConvIR-B
WORK=$BASE/repos/ConvIR-B-dta-v3-4-fdf-tsr-finetune
PY=$BASE/envs/convir-cu128/bin/python
DATA=$BASE/datasets/Haze4K/Haze4K
DEPTH=$BASE/depth_cache/depth_anything_v2_small_hf
EVID=$WORK/experience_docx/experiment_logs/haze4k_dta_v3_dapc_20260611
STATUS=$EVID/status.txt
LOG=$EVID/cache_depth_anything_v2_haze4k_convir5090.log
mkdir -p "$EVID" "$DEPTH"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
echo "depth_cache_start $(date --iso-8601=seconds) output=$DEPTH cuda=$CUDA_VISIBLE_DEVICES" | tee -a "$STATUS"
cd "$WORK"
set +e
PYTHONUNBUFFERED=1 "$PY" experience_docx/tools/cache_depth_anything_v2_haze4k.py \
  --data_dir "$DATA" \
  --output_dir "$DEPTH" \
  --splits train,test \
  --device cuda:0 \
  2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e
echo "depth_cache_done rc=$rc $(date --iso-8601=seconds)" | tee -a "$STATUS"
exit "$rc"
