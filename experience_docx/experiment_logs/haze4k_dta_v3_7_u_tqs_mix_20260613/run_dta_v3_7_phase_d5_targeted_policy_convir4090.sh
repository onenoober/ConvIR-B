#!/usr/bin/env bash
set -euo pipefail
BASE=${BASE:-/sda/home/wangyuxin/ConvIR-B}
WORK=${WORK:-$BASE/repos/ConvIR-B-dta-v3-7-u-tqs-mix-d5-targeted}
PY=${PY:-$BASE/envs/convir-cu121/bin/python}
EVID=$WORK/experience_docx/experiment_logs/haze4k_dta_v3_7_u_tqs_mix_20260613
STATUS=$EVID/status_phase_d5_targeted_policy.txt
SINGLE_ACTIONS=${SINGLE_ACTIONS:-$EVID/v37_tau_real_blend_single_actions_all.csv}
FEATURE_TABLE=${FEATURE_TABLE:-$EVID/v37_tau_oof_per_image_action_table.csv}
OUTPUT_PREFIX=${OUTPUT_PREFIX:-v37_d5_targeted}
INCLUDE_RUN_SUBSTRING=${INCLUDE_RUN_SUBSTRING:-quick5full}
mkdir -p "$EVID"
{
  echo "dta_v3_7_phase_d5_targeted_policy_start $(date --iso-8601=seconds)"
  echo "state=RUNNING_AUDIT_D5_TARGETED_INTERVENTION_POLICY"
  echo "work=$WORK"
  echo "python=$PY"
  echo "single_actions=$SINGLE_ACTIONS"
  echo "feature_table=$FEATURE_TABLE"
  echo "output_prefix=$OUTPUT_PREFIX"
  echo "locked_test_touched=false"
  echo "scope=D1/D3 quick5full train-derived folds 0,1 seeds 3407,3411 only"
} | tee -a "$STATUS"
for p in "$WORK" "$PY" "$SINGLE_ACTIONS" "$FEATURE_TABLE"; do
  if [[ ! -e "$p" ]]; then echo "DTA_V3_7_PHASE_D5_TARGETED_MISSING_PATH $p" | tee -a "$STATUS"; exit 3; fi
done
cd "$WORK"
{ git branch --show-current; git rev-parse --short HEAD; git status --short; } | tee -a "$STATUS"
set +e
PYTHONUNBUFFERED=1 "$PY" experience_docx/tools/train_haze4k_dta_v37_d5_targeted_intervention_policy.py \
  --single_actions_csv "$SINGLE_ACTIONS" \
  --feature_action_table_csv "$FEATURE_TABLE" \
  --output_dir "$EVID" \
  --output_prefix "$OUTPUT_PREFIX" \
  --include_run_substring "$INCLUDE_RUN_SUBSTRING" \
  2>&1 | tee "$EVID/v37_phase_d5_targeted_policy.log"
rc=${PIPESTATUS[0]}
set -e
echo "dta_v3_7_phase_d5_targeted_policy_done rc=$rc $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [[ "$rc" -ne 0 ]]; then echo "DTA_V3_7_PHASE_D5_TARGETED_POLICY_FAILED $(date --iso-8601=seconds)" | tee -a "$STATUS"; exit "$rc"; fi
echo "DTA_V3_7_PHASE_D5_TARGETED_POLICY_OK $(date --iso-8601=seconds)" | tee -a "$STATUS"
