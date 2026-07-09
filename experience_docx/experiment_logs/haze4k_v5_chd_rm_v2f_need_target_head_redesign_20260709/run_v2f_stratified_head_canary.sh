#!/usr/bin/env bash
set -euo pipefail

BASE=/sda/home/wangyuxin/ConvIR-B
WORK=$BASE/repos/ConvIR-B-haze4k-v5-v2f-chd-rm-need-target-head-redesign
PY=$BASE/envs/convir-cu121/bin/python
DATA=$BASE/datasets/Haze4K/Haze4K
A0=$BASE/checkpoints/official/Haze4K/haze4k-base.pkl
EVID=$WORK/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2f_need_target_head_redesign_20260709
STATUS=$EVID/status.txt
LOG=$EVID/v2f_f4_stratified_head_canary.log

SPLIT=$BASE/repos/ConvIR-B-haze4k-v5-v2c-chd-rm-need-coverage-calibration/experience_docx/experiment_logs/haze4k_v5_chd_rm_v1_data_baseline_lock_20260708/haze4k_internal_split_2400_600.json
V2_THRESH=$BASE/repos/ConvIR-B-haze4k-v5-v2-chd-rm-density-need-calibration/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2_density_need_calibration_20260708/density_need_thresholds.json
V2B_THRESH=$BASE/repos/ConvIR-B-haze4k-v5-v2b-chd-rm-need-calibration-repair/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2b_need_calibration_repair_20260708/need_thresholds_v2b.json
D3=$BASE/repos/ConvIR-B-haze4k-v5-v2-chd-rm-density-need-calibration/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2_density_need_calibration_20260708/artifacts/d3_density_only_head.pt

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

mkdir -p "$EVID"
{
  echo "v2f_f4_canary_start $(date --iso-8601=seconds)"
  echo "work=$WORK"
  echo "python=$PY"
  echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
  echo "locked_haze4k_test_usage=none"
  echo "D2=not_run RARM=not_connected_or_trained v3=not_run"
} | tee -a "$STATUS"

cd "$WORK"
SOURCE_COMMIT=$(git rev-parse HEAD)

set +e
PYTHONUNBUFFERED=1 "$PY" experience_docx/tools/run_chd_rm_v2f_stratified_head_canary.py \
  --data_dir "$DATA" \
  --checkpoint "$A0" \
  --split_json "$SPLIT" \
  --v2_thresholds "$V2_THRESH" \
  --v2b_thresholds "$V2B_THRESH" \
  --density_artifact "$D3" \
  --output_dir "$EVID" \
  --source_commit "$SOURCE_COMMIT" \
  --epochs 6 \
  --batch_size 8 \
  --density_bins 5 \
  --metric_sample_size 64 \
  --fit_grid 64 \
  --variants f4_global_strat_control f4_cond_strat_core f4_cond_strat_ldhn f4_excess_strat_ldhn \
  2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e

echo "v2f_f4_canary_done rc=$rc $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [[ "$rc" -eq 0 ]]; then
  echo "V2F_F4_STRATIFIED_HEAD_CANARY_OK" | tee -a "$STATUS"
else
  echo "V2F_F4_STRATIFIED_HEAD_CANARY_FAILED" | tee -a "$STATUS"
fi
exit "$rc"
