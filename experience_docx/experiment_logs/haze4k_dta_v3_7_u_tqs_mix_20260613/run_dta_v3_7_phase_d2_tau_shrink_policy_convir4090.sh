#!/usr/bin/env bash
set -euo pipefail

BASE=${BASE:-/sda/home/wangyuxin/ConvIR-B}
REPO=${REPO:-$BASE/repos/ConvIR-B-dta-v3-7-u-tqs-mix-d2-policy}
PY=${PY:-$BASE/envs/convir-cu121/bin/python}
EVID=$REPO/experience_docx/experiment_logs/haze4k_dta_v3_7_u_tqs_mix_20260613
STATUS=$EVID/status_phase_d2_tau_shrink_policy.txt
LOG=$EVID/v37_phase_d2_tau_shrink_policy.log
INPUT=${INPUT:-$EVID/v37_tau_oof_per_image_action_table.csv}
FEATURES=${FEATURES:-$EVID/v37_quality_image_features.csv}
VARIANTS=${VARIANTS:-u1_tau_l1_s004_g025_a006,u2_tau_l3_s004_g015_a006,u3_tau_l2_s002_g025_a006}
FOLDS=${FOLDS:-0,1}
SEEDS=${SEEDS:-3407,3411}
ACTION_BANK=${ACTION_BANK:-micro_shrink}
OUTPUT_PREFIX=${OUTPUT_PREFIX:-v37_tau_shrink}

mkdir -p "$EVID"
cd "$REPO"

if [ -f "$STATUS" ] && grep -q '^RUNNING' "$STATUS"; then
  echo "Existing Phase D2 status is RUNNING; refusing to overwrite active run." >&2
  exit 2
fi
if [ -f "$EVID/${OUTPUT_PREFIX}_summary.json" ] && [ "${ALLOW_RERUN:-0}" != "1" ]; then
  echo "Phase D2 summary already exists; set ALLOW_RERUN=1 to rerun." >&2
  exit 3
fi

{
  echo "RUNNING DTA-v3.7 Phase D2 TAU shrink policy $(date -Is)"
  echo "state=RUNNING_TRAIN_DERIVED_TABLE_POLICY"
  echo "host=$(hostname)"
  echo "repo=$REPO"
  echo "branch=$(git branch --show-current)"
  echo "commit=$(git rev-parse --short HEAD)"
  echo "python=$PY"
  echo "input=$INPUT"
  echo "features=$FEATURES"
  echo "variants=$VARIANTS folds=$FOLDS seeds=$SEEDS action_bank=$ACTION_BANK"
  echo "locked_test_touched=false"
  echo "proxy_note=alpha-scaled D1 table deltas; real rendered verification is required before formal claims"
} | tee "$STATUS"

"$PY" experience_docx/tools/train_haze4k_dta_v37_tau_shrink_policy.py \
  --input_action_table "$INPUT" \
  --image_feature_table "$FEATURES" \
  --output_dir "$EVID" \
  --variants "$VARIANTS" \
  --folds "$FOLDS" \
  --seeds "$SEEDS" \
  --action_bank "$ACTION_BANK" \
  --output_prefix "$OUTPUT_PREFIX" \
  2>&1 | tee "$LOG"

{
  echo "COMPLETED DTA_V3_7_D2_TAU_SHRINK_POLICY_OK $(date -Is)"
  echo "summary=$EVID/${OUTPUT_PREFIX}_summary.json"
  echo "locked_test_touched=false"
} | tee -a "$STATUS"
