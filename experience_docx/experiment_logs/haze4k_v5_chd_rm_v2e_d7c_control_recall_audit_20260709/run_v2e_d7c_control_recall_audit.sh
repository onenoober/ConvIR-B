#!/usr/bin/env bash
set -euo pipefail
REPO=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-haze4k-v5-v2e-chd-rm-d7c-control-recall-audit
PY=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
EVID=$REPO/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2e_d7c_control_recall_audit_20260709
LOG=$EVID/v2e_d7c_control_recall_audit.log
STATUS=$EVID/status.txt
mkdir -p "$EVID"
cd "$REPO"
echo "RUNNING_AUDIT start $(date --iso-8601=seconds) fastperm_pixel_sample=512" | tee "$STATUS"
set +e
CUDA_VISIBLE_DEVICES=1 "$PY" experience_docx/tools/run_chd_rm_v2e_d7c_control_recall_audit.py \
  --data_dir /sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K \
  --checkpoint /sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl \
  --split_json /sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-haze4k-v5-v2c-chd-rm-need-coverage-calibration/experience_docx/experiment_logs/haze4k_v5_chd_rm_v1_data_baseline_lock_20260708/haze4k_internal_split_2400_600.json \
  --v2_thresholds /sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-haze4k-v5-v2-chd-rm-density-need-calibration/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2_density_need_calibration_20260708/density_need_thresholds.json \
  --v2b_thresholds /sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-haze4k-v5-v2b-chd-rm-need-calibration-repair/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2b_need_calibration_repair_20260708/need_thresholds_v2b.json \
  --density_artifact /sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-haze4k-v5-v2-chd-rm-density-need-calibration/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2_density_need_calibration_20260708/artifacts/d3_density_only_head.pt \
  --d7c_topk_artifact /sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-haze4k-v5-v2d-chd-rm-need-spatial-hard-negative/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2d_need_spatial_hard_negative_20260709/d7c_full/artifacts/d7c_mc_topk_hn_ordinal_head.pt \
  --d7c_hn_artifact /sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-haze4k-v5-v2d-chd-rm-need-spatial-hard-negative/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2d_need_spatial_hard_negative_20260709/d7c_full/artifacts/d7c_mc_hn_ordinal_head.pt \
  --output_dir "$EVID" \
  --seed 3407 \
  --permutation_count 100 \
  --permutation_pixel_sample_per_image 512 \
  --control_epochs 6 \
  --batch_size 8 \
  --num_workers 4 \
  --progress_every 50 \
  --threshold_grid 121 \
  --metric_sample_size 64 \
  2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e
if [ "$rc" -eq 0 ]; then
  echo "RUN_DONE rc=0 $(date --iso-8601=seconds)" | tee -a "$STATUS"
  echo V2E_FULL_OK | tee -a "$STATUS"
else
  echo "RUN_FAILED rc=$rc $(date --iso-8601=seconds)" | tee -a "$STATUS"
  echo V2E_FULL_FAILED | tee -a "$STATUS"
fi
exit "$rc"
