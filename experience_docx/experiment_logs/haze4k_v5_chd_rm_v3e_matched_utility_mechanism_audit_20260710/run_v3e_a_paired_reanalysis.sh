#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v3e-matched-utility-mechanism-audit}
OLD=${OLD:-/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v3d-rarm-adapter-only-preflight}
PY=${PY:-/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python}
ROUTE_ID=haze4k_v5_chd_rm_v3e_matched_utility_mechanism_audit_20260710
EVID="$ROOT/experience_docx/experiment_logs/$ROUTE_ID"
OLD_LOG="$OLD/experience_docx/experiment_logs/haze4k_v5_chd_rm_v3d_rarm_adapter_only_preflight_20260710"
STATUS="$EVID/status.txt"
LOG="$EVID/v3e_a_paired_reanalysis.log"

mkdir -p "$EVID"
echo "v3e_a_start $ROUTE_ID $(date --iso-8601=seconds)" | tee -a "$STATUS"
cd "$ROOT"
set +e
PYTHONUNBUFFERED=1 "$PY" experience_docx/tools/chd_rm_v3e_mechanism_audit.py paired \
  --old_logdir "$OLD_LOG" \
  --output_dir "$EVID" \
  --bootstrap 20000 \
  > "$LOG" 2>&1
rc=$?
set -e
echo "v3e_a_done rc=$rc $ROUTE_ID $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [ "$rc" -eq 0 ]; then
  echo "V3E_A_PAIRED_REANALYSIS_OK" | tee -a "$STATUS"
else
  echo "V3E_A_PAIRED_REANALYSIS_FAILED" | tee -a "$STATUS"
fi
exit "$rc"
