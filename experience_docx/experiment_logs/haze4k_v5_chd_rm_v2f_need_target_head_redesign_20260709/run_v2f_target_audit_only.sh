#!/usr/bin/env bash
set -euo pipefail

BASE=/sda/home/wangyuxin/ConvIR-B
WORK=$BASE/repos/ConvIR-B-haze4k-v5-v2f-chd-rm-need-target-head-redesign
PY=$BASE/envs/convir-cu121/bin/python
DATA=$BASE/datasets/Haze4K/Haze4K
A0=$BASE/checkpoints/official/Haze4K/haze4k-base.pkl
EVID=$WORK/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2f_need_target_head_redesign_20260709
STATUS=$EVID/status.txt
LOG=$EVID/v2f_target_audit_only.log

SPLIT=$BASE/repos/ConvIR-B-haze4k-v5-v2c-chd-rm-need-coverage-calibration/experience_docx/experiment_logs/haze4k_v5_chd_rm_v1_data_baseline_lock_20260708/haze4k_internal_split_2400_600.json
V2_THRESH=$BASE/repos/ConvIR-B-haze4k-v5-v2-chd-rm-density-need-calibration/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2_density_need_calibration_20260708/density_need_thresholds.json
V2B_THRESH=$BASE/repos/ConvIR-B-haze4k-v5-v2b-chd-rm-need-calibration-repair/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2b_need_calibration_repair_20260708/need_thresholds_v2b.json
D3=$BASE/repos/ConvIR-B-haze4k-v5-v2-chd-rm-density-need-calibration/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2_density_need_calibration_20260708/artifacts/d3_density_only_head.pt
D7C_TOPK=$BASE/repos/ConvIR-B-haze4k-v5-v2d-chd-rm-need-spatial-hard-negative/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2d_need_spatial_hard_negative_20260709/d7c_full/artifacts/d7c_mc_topk_hn_ordinal_head.pt
D7C_HN=$BASE/repos/ConvIR-B-haze4k-v5-v2d-chd-rm-need-spatial-hard-negative/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2d_need_spatial_hard_negative_20260709/d7c_full/artifacts/d7c_mc_hn_ordinal_head.pt

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

mkdir -p "$EVID"
{
  echo "v2f_target_audit_start $(date --iso-8601=seconds)"
  echo "work=$WORK"
  echo "python=$PY"
  echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
  echo "locked_haze4k_test_usage=none"
} | tee -a "$STATUS"

cd "$WORK"
SOURCE_COMMIT=$(git rev-parse HEAD)

set +e
PYTHONUNBUFFERED=1 "$PY" experience_docx/tools/run_chd_rm_v2f_need_target_head_redesign.py \
  --data_dir "$DATA" \
  --checkpoint "$A0" \
  --split_json "$SPLIT" \
  --v2_thresholds "$V2_THRESH" \
  --v2b_thresholds "$V2B_THRESH" \
  --density_artifact "$D3" \
  --d7c_topk_artifact "$D7C_TOPK" \
  --d7c_hn_artifact "$D7C_HN" \
  --output_dir "$EVID" \
  --source_commit "$SOURCE_COMMIT" \
  --map_grid 64 \
  --probe_grid 32 \
  --probe_epochs 5 \
  --probe_pixels_per_image_per_class 24 \
  2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e

echo "v2f_target_audit_done rc=$rc $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [[ "$rc" -eq 0 ]]; then
  echo "V2F_TARGET_AUDIT_ONLY_OK" | tee -a "$STATUS"
else
  echo "V2F_TARGET_AUDIT_ONLY_FAILED" | tee -a "$STATUS"
fi
exit "$rc"
