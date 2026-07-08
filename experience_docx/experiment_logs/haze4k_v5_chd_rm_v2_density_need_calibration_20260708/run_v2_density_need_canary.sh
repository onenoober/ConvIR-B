#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT=/sda/home/wangyuxin/ConvIR-B
REPO=$REMOTE_ROOT/repos/ConvIR-B-haze4k-v5-v2-chd-rm-density-need-calibration
PY=$REMOTE_ROOT/envs/convir-cu121/bin/python
DATA=$REMOTE_ROOT/datasets/Haze4K/Haze4K
A0=$REMOTE_ROOT/checkpoints/official/Haze4K/haze4k-base.pkl
EVID=$REPO/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2_density_need_calibration_20260708
OUT=$EVID/canary
SPLIT=$REPO/experience_docx/experiment_logs/haze4k_v5_chd_rm_v1_data_baseline_lock_20260708/haze4k_internal_split_2400_600.json
STATUS=$EVID/status.txt
LOG=$EVID/v2_density_need_canary.log

mkdir -p "$OUT"
touch "$STATUS"

{
  echo "canary_start chdrm_v2_density_need $(date --iso-8601=seconds)"
  echo "branch=$(git -C "$REPO" branch --show-current)"
  echo "commit=$(git -C "$REPO" rev-parse HEAD)"
  echo "locked_test=not_used"
} | tee -a "$STATUS"

GPU_ID=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2,2n | head -n 1 | cut -d, -f1 | tr -d ' ')
export CUDA_VISIBLE_DEVICES="$GPU_ID"
echo "canary_cuda_visible_devices=$CUDA_VISIBLE_DEVICES" | tee -a "$STATUS"

set +e
PYTHONUNBUFFERED=1 "$PY" "$REPO/experience_docx/tools/run_chd_rm_v2_density_need_calibration.py" \
  --data_dir "$DATA" \
  --checkpoint "$A0" \
  --split_json "$SPLIT" \
  --output_dir "$OUT" \
  --epochs 1 \
  --batch_size 4 \
  --crop_size 256 \
  --metric_sample_size 16 \
  --threshold_limit 24 \
  --train_limit 32 \
  --val_limit 12 \
  --num_workers 0 \
  --progress_every 4 \
  2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e

echo "canary_done rc=$rc chdrm_v2_density_need $(date --iso-8601=seconds)" | tee -a "$STATUS"
exit "$rc"

