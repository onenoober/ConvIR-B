#!/usr/bin/env bash
set -euo pipefail

BASE=${BASE:-/sda/home/wangyuxin/ConvIR-B}
WORK=${WORK:-$BASE/repos/ConvIR-B-dta-v3-6-hrcs}
PY=${PY:-$BASE/envs/convir-cu121/bin/python}
EVID=$WORK/experience_docx/experiment_logs/haze4k_dta_v3_6_hrcs_20260613
STATUS=$EVID/status.txt
V35_EVID=${V35_EVID:-$BASE/repos/ConvIR-B-dta-v3-5-fdf-rcs-lite/experience_docx/experiment_logs/haze4k_dta_v3_5_fdf_rcs_lite_20260612}
ACTION_TABLE=${ACTION_TABLE:-$V35_EVID/v35_oof_per_image_action_table.csv}
VARIANTS=${VARIANTS:-l3_fdf_lite_s004_g015_bm2,l1_fdf_lite_s004_g025_bm2,l2_fdf_lite_s002_g025_bm2}
COVERAGE_GRID=${COVERAGE_GRID:-1.00,0.99,0.98,0.97,0.96,0.95,0.94,0.93,0.92,0.90}
SELECTOR_MODELS=${SELECTOR_MODELS:-logistic,gbdt}
FEATURE_GROUPS=${FEATURE_GROUPS:-input_only,input_depth,input_depth_action,deployable_all,diagnostic_with_trans_gt,diagnostic_with_cf_delta}

mkdir -p "$EVID"
{
  echo "dta_v3_6_hrcs_phase_a_start $(date --iso-8601=seconds)"
  echo "state=RUNNING_AUDIT"
  echo "work=$WORK"
  echo "python=$PY"
  echo "source_v35_evidence=$V35_EVID"
  echo "action_table=$ACTION_TABLE"
  echo "variants=$VARIANTS"
  echo "coverage_grid=$COVERAGE_GRID"
  echo "selector_models=$SELECTOR_MODELS"
  echo "feature_groups=$FEATURE_GROUPS"
  echo "locked_test_touched=false"
  echo "user_relaxed_locked_test_override_recorded=true"
} | tee -a "$STATUS"

for p in "$WORK" "$PY" "$ACTION_TABLE"; do
  if [[ ! -e "$p" ]]; then
    echo "DTA_V3_6_HRCS_MISSING_PATH $p" | tee -a "$STATUS"
    exit 3
  fi
done

cd "$WORK"
{ git branch --show-current; git rev-parse --short HEAD; git status --short; } | tee -a "$STATUS"

set +e
PYTHONUNBUFFERED=1 "$PY" experience_docx/tools/select_haze4k_dta_v36_hrcs.py \
  --input_action_table "$ACTION_TABLE" \
  --output_dir "$EVID" \
  --variants "$VARIANTS" \
  --selector_models "$SELECTOR_MODELS" \
  --feature_groups "$FEATURE_GROUPS" \
  --coverage_grid "$COVERAGE_GRID" \
  2>&1 | tee "$EVID/dta_v3_6_hrcs_phase_a.log"
rc=${PIPESTATUS[0]}
set -e
echo "dta_v3_6_hrcs_phase_a_done rc=$rc $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [[ "$rc" -ne 0 ]]; then
  echo "DTA_V3_6_HRCS_PHASE_A_FAILED $(date --iso-8601=seconds)" | tee -a "$STATUS"
  exit "$rc"
fi

echo "DTA_V3_6_HRCS_PHASE_A_OK $(date --iso-8601=seconds)" | tee -a "$STATUS"
