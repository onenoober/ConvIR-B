#!/usr/bin/env bash
set -euo pipefail

BASE=${BASE:-/sda/home/wangyuxin/ConvIR-B}
REMOTE_ROOT=${REMOTE_ROOT:-$BASE/repos/ConvIR-B-dta-v2-calibrated}
PY=${PY:-$BASE/envs/convir-cu121/bin/python}
DATA=${DATA:-$BASE/datasets/Haze4K/Haze4K}
DEPTH=${DEPTH:-$BASE/depth_cache/depth_anything_v2_small_hf}
EVID=$REMOTE_ROOT/experience_docx/experiment_logs/haze4k_dta_v2_calibrated_20260611
OUT=$EVID/dta_v2_depth_transmission_audit
STATUS=$EVID/status.txt
LOG=$EVID/dta_v2_depth_transmission_audit.log
mkdir -p "$OUT"
{
  echo "audit_start dta_v2 $(date --iso-8601=seconds)"
  echo "remote_root=$REMOTE_ROOT"
  echo "python=$PY"
  echo "data=$DATA"
  echo "depth=$DEPTH"
} | tee -a "$STATUS"
cd "$REMOTE_ROOT"
{
  git branch --show-current
  git rev-parse --short HEAD
  git status --short
} | tee -a "$STATUS"
set +e
PYTHONUNBUFFERED=1 "$PY" experience_docx/tools/audit_haze4k_depth_transmission.py \
  --data_dir "$DATA" \
  --depth_cache_dir "$DEPTH" \
  --splits train,test \
  --output_dir "$OUT" \
  --max_pixels_per_image 65536 \
  --seed 3407 \
  2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e
echo "audit_done rc=$rc dta_v2 $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [[ "$rc" -eq 0 ]]; then
  echo "DTA_V2_DEPTH_TRANSMISSION_AUDIT_OK" | tee -a "$STATUS"
else
  echo "DTA_V2_DEPTH_TRANSMISSION_AUDIT_FAILED" | tee -a "$STATUS"
fi
exit "$rc"
