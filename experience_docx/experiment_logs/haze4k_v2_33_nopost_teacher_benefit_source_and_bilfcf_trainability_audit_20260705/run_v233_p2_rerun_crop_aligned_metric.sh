#!/usr/bin/env bash
set -euo pipefail

WORK="/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-haze4k-v2-33-nopost-teacher-benefit-source-and-bilfcf-trainability-audit"
EVID="$WORK/experience_docx/experiment_logs/haze4k_v2_33_nopost_teacher_benefit_source_and_bilfcf_trainability_audit_20260705"
DATA="/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K"
A0="/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl"
PY="/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python"

cd "$WORK"

echo "p2_rerun_crop_aligned_metric_start $(date -Is)" >> "$EVID/status.txt"

for name in \
  v233_p2_loss_gradient_scale_closeout.json \
  v233_p2_loss_gradient_scale_sanity.csv \
  v233_p2_one_image_overfit_report.csv \
  v233_p2_sign_flip_control.csv
do
  if [ -f "$EVID/$name" ] && [ ! -f "$EVID/${name%.json}_rerun_evalmode_metric_mismatch.json" ] && [[ "$name" == *.json ]]; then
    cp "$EVID/$name" "$EVID/${name%.json}_rerun_evalmode_metric_mismatch.json"
  fi
  if [ -f "$EVID/$name" ] && [ ! -f "$EVID/${name%.csv}_rerun_evalmode_metric_mismatch.csv" ] && [[ "$name" == *.csv ]]; then
    cp "$EVID/$name" "$EVID/${name%.csv}_rerun_evalmode_metric_mismatch.csv"
  fi
done

PYTHONUNBUFFERED=1 "$PY" experience_docx/tools/run_haze4k_v233_nopost_audit.py \
  --phase p2 \
  --data_dir "$DATA" \
  --checkpoint "$A0" \
  --output_dir "$EVID" \
  --steps 24 \
  --hard_scan_count 32 \
  > "$EVID/v233_p2_loss_gradient_scale_sanity_rerun_crop_aligned.log" 2>&1

echo "p2_rerun_crop_aligned_metric_done rc=0 $(date -Is)" >> "$EVID/status.txt"
echo "V233_P2_RERUN_CROP_ALIGNED_OK" >> "$EVID/status.txt"
