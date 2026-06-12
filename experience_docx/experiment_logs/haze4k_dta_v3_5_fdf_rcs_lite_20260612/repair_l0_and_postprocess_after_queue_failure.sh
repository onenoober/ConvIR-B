#!/usr/bin/env bash
set -euo pipefail
BASE=/sda/home/wangyuxin/ConvIR-B
ROOT=$BASE/repos/ConvIR-B-dta-v3-5-fdf-rcs-lite
EVID=$ROOT/experience_docx/experiment_logs/haze4k_dta_v3_5_fdf_rcs_lite_20260612
STATUS=$EVID/status.txt
PY=$BASE/envs/convir-cu121/bin/python
RUN_SCRIPT=$EVID/run_dta_v3_5_fdf_rcs_lite_convir4090.sh
mkdir -p "$EVID"
cd "$ROOT"
echo "dta_v3_5_l0_repair_postprocess_start $(date --iso-8601=seconds)" | tee -a "$STATUS"
echo "repair_gpu_cap=max2_of_allowed5" | tee -a "$STATUS"
git fetch github codex/haze4k-dta-v3-5-fdf-rcs-lite
git merge --ff-only FETCH_HEAD
chmod +x "$RUN_SCRIPT" "$EVID/launch_dta_v3_5_fdf_rcs_lite_triage_convir4090.sh"
echo "repair_head=$(git rev-parse --short HEAD)" | tee -a "$STATUS"
for old in \
  "$EVID/queue_l0_a0_sanity_seed3407_f0_quick5full.log" \
  "$EVID/queue_l0_a0_sanity_seed3407_f1_quick5full.log" \
  "$EVID/dta_v3_5_v35_fdf_rcs_l0_a0_sanity_seed3407_f0_quick5full_fallback_train_invert_eval.log" \
  "$EVID/dta_v3_5_v35_fdf_rcs_l0_a0_sanity_seed3407_f1_quick5full_fallback_train_invert_eval.log"; do
  if [[ -f "$old" && ! -f "$old.pre_fix_failure" ]]; then
    cp -p "$old" "$old.pre_fix_failure"
  fi
done
run_l0() {
  local fold=$1
  local gpu=$2
  local log=$EVID/queue_l0_a0_sanity_seed3407_f${fold}_quick5full_repair.log
  echo "dta_v3_5_l0_repair_launch fold=$fold gpu=$gpu $(date --iso-8601=seconds)" | tee -a "$STATUS"
  (
    set -euo pipefail
    VARIANT=l0_a0_sanity FOLD="$fold" SEED=3407 STAGE=quick5full CUDA_VISIBLE_DEVICES="$gpu" \
      MAX_IMAGES=0 FORCE=1 RUN_TRAIN_CONTROLS=1 RUN_TEST=0 USE_SPLIT=1 "$RUN_SCRIPT"
  ) > "$log" 2>&1 &
}
run_l0 0 1
run_l0 1 6
fail=0
while [[ $(jobs -rp | wc -l | tr -d ' ') -gt 0 ]]; do
  if ! wait -n; then fail=1; fi
done
if [[ "$fail" -ne 0 ]]; then
  echo "DTA_V3_5_L0_REPAIR_FAILED $(date --iso-8601=seconds)" | tee -a "$STATUS"
  exit 1
fi
echo "dta_v3_5_l0_repair_done $(date --iso-8601=seconds)" | tee -a "$STATUS"
PYTHONUNBUFFERED=1 "$PY" experience_docx/tools/summarize_haze4k_dta_v35_triage.py \
  --evidence_dir "$EVID" \
  --output_json "$EVID/dta_v3_5_fdf_rcs_triage_summary.json" \
  --output_csv "$EVID/dta_v3_5_fdf_rcs_triage_summary.csv" \
  --variant_csv "$EVID/dta_v3_5_fdf_rcs_triage_variant_summary.csv" \
  --expected_runs_per_variant 4 \
  2>&1 | tee "$EVID/dta_v3_5_triage_summary_repair.log"
echo "dta_v3_5_triage_summary_done $(date --iso-8601=seconds)" | tee -a "$STATUS"
PYTHONUNBUFFERED=1 "$PY" experience_docx/tools/select_haze4k_dta_v35_nested_calibration.py \
  --evidence_dir "$EVID" \
  --action_table_csv "$EVID/v35_oof_per_image_action_table.csv" \
  --oracle_curve_csv "$EVID/v35_oracle_risk_coverage_curve.csv" \
  --nested_report_json "$EVID/v35_selector_nested_calibration_report.json" \
  --nested_report_csv "$EVID/v35_selector_nested_calibration_report.csv" \
  --nested_selected_csv "$EVID/v35_selector_nested_selected_images.csv" \
  --min_coverage 0.20 \
  --max_coverage 0.95 \
  2>&1 | tee "$EVID/dta_v3_5_nested_selector_repair.log"
echo "DTA_V3_5_L0_REPAIR_POSTPROCESS_OK $(date --iso-8601=seconds)" | tee -a "$STATUS"
