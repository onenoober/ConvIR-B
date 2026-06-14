#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT=${REMOTE_ROOT:-/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v20-strongexpert-gainmix-c1}
EVID=${EVID:-$REMOTE_ROOT/experience_docx/experiment_logs/haze4k_v2_0_strongexpert_gainmix_20260614}
PY=${PY:-/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python}
FULLUDP_EVAL=${FULLUDP_EVAL:-/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-v3-7-u-tqs-mix-d8-formal-5541ca9/experience_docx/experiment_logs/haze4k_fulludp_v15_phase0_repro_20260605/phase0_official_eval}
STATUS=$EVID/status_c1.txt
LOG=$EVID/v20_c1_risk_map.log

mkdir -p "$EVID"
{
  echo "v20_c1_risk_map_start $(date --iso-8601=seconds)"
  echo "remote_root=$REMOTE_ROOT"
  echo "python=$PY"
  echo "fulludp_eval=$FULLUDP_EVAL"
  echo "locked_test_touched=false"
  if [ -d "$REMOTE_ROOT/.git" ]; then
    git -C "$REMOTE_ROOT" branch --show-current | sed 's/^/branch=/'
    git -C "$REMOTE_ROOT" rev-parse --short HEAD | sed 's/^/commit=/'
    git -C "$REMOTE_ROOT" status --short | sed -n '1,80p' | sed 's/^/git_status=/'
  fi
} | tee -a "$STATUS"

cd "$REMOTE_ROOT"
set +e
"$PY" experience_docx/tools/audit_haze4k_v20_c1_risk_map.py \
  --fulludp-eval-dir "$FULLUDP_EVAL" \
  --out-dir "$EVID" \
  2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e
echo "v20_c1_risk_map_done rc=$rc $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [ "$rc" -eq 0 ]; then
  echo "V20_C1_RISK_MAP_OK $(date --iso-8601=seconds)" | tee -a "$STATUS"
else
  echo "V20_C1_RISK_MAP_FAILED rc=$rc $(date --iso-8601=seconds)" | tee -a "$STATUS"
fi
exit "$rc"
