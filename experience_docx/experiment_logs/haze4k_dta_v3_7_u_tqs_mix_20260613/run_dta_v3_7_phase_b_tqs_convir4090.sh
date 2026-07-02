#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-v3-7-u-tqs-mix}
PY=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
EVID=$REPO/experience_docx/experiment_logs/haze4k_dta_v3_7_u_tqs_mix_20260613
STATUS=$EVID/status_phase_b_tqs.txt
LOG=$EVID/v37_phase_b_tqs.log
INPUT=$REPO/experience_docx/experiment_logs/haze4k_dta_v3_6_hrcs_20260613/v36_formal_oof_per_image_action_table.csv

mkdir -p "$EVID"
cd "$REPO"

if [ -f "$STATUS" ] && grep -q '^RUNNING' "$STATUS"; then
  echo "Existing Phase B status is RUNNING; refusing to overwrite active run." >&2
  exit 2
fi
if [ -f "$EVID/v37_tqs_summary.json" ] && [ "${ALLOW_RERUN:-0}" != "1" ]; then
  echo "Phase B TQS summary already exists; set ALLOW_RERUN=1 to rerun." >&2
  exit 3
fi

{
  echo "RUNNING DTA-v3.7 Phase B TQS $(date -Is)"
  echo "host=$(hostname)"
  echo "repo=$REPO"
  echo "branch=$(git branch --show-current)"
  echo "commit=$(git rev-parse --short HEAD)"
  echo "python=$PY"
  echo "input=$INPUT"
} | tee "$STATUS"

"$PY" experience_docx/tools/train_haze4k_dta_v37_tqs_policy.py \
  --input_action_table "$INPUT" \
  --output_dir "$EVID" \
  2>&1 | tee "$LOG"

{
  echo "COMPLETED DTA_V3_7_TQS_PHASE_B_OK $(date -Is)"
  echo "summary=$EVID/v37_tqs_summary.json"
} | tee -a "$STATUS"

echo "DTA_V3_7_PHASE_B_TQS_RUN_SCRIPT_OK"
