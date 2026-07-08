#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT=/sda/home/wangyuxin/ConvIR-B
REPO=$REMOTE_ROOT/repos/ConvIR-B-haze4k-v5-v2c-chd-rm-need-coverage-calibration
PY=$REMOTE_ROOT/envs/convir-cu121/bin/python
DATA=$REMOTE_ROOT/datasets/Haze4K/Haze4K
A0=$REMOTE_ROOT/checkpoints/official/Haze4K/haze4k-base.pkl
EVID=$REPO/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2c_need_coverage_calibration_20260709
V1E=$REPO/experience_docx/experiment_logs/haze4k_v5_chd_rm_v1_data_baseline_lock_20260708
V2E=$REPO/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2_density_need_calibration_20260708
V2BE=$REMOTE_ROOT/repos/ConvIR-B-haze4k-v5-v2b-chd-rm-need-calibration-repair/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2b_need_calibration_repair_20260708
SPLIT=$V1E/haze4k_internal_split_2400_600.json
V2_THRESH=$V2E/density_need_thresholds.json
V2B_THRESH=$V2BE/need_thresholds_v2b.json
V2B_ART=$V2BE/artifacts
STATUS=$EVID/status.txt
LOG=$EVID/v2c_need_coverage_calibration.log

mkdir -p "$EVID"
touch "$STATUS"

{
  echo "run_start chdrm_v2c_need_coverage_calibration $(date --iso-8601=seconds)"
  echo "repo=$REPO"
  echo "branch=$(git -C "$REPO" branch --show-current)"
  echo "commit=$(git -C "$REPO" rev-parse HEAD)"
  echo "python=$PY"
  echo "data=$DATA"
  echo "checkpoint=$A0"
  echo "split=$SPLIT"
  echo "v2b_artifacts=$V2B_ART"
  echo "locked_test=not_used"
} | tee -a "$STATUS"

GPU_ID=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2,2n | head -n 1 | cut -d, -f1 | tr -d ' ')
export CUDA_VISIBLE_DEVICES="$GPU_ID"
echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES" | tee -a "$STATUS"

set +e
PYTHONUNBUFFERED=1 "$PY" "$REPO/experience_docx/tools/run_chd_rm_v2c_need_coverage_calibration.py" \
  --data_dir "$DATA" \
  --checkpoint "$A0" \
  --split_json "$SPLIT" \
  --v2_thresholds "$V2_THRESH" \
  --v2b_thresholds "$V2B_THRESH" \
  --v2b_artifact_dir "$V2B_ART" \
  --output_dir "$EVID" \
  --metric_sample_size 64 \
  --progress_every 50 \
  2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e

echo "run_done rc=$rc chdrm_v2c_need_coverage_calibration $(date --iso-8601=seconds)" | tee -a "$STATUS"
exit "$rc"
