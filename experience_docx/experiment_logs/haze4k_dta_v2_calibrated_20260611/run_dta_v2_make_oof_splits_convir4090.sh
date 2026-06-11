#!/usr/bin/env bash
set -euo pipefail
BASE=${BASE:-/sda/home/wangyuxin/ConvIR-B}
REMOTE_ROOT=${REMOTE_ROOT:-$BASE/repos/ConvIR-B-dta-v2-calibrated}
PY=${PY:-$BASE/envs/convir-cu121/bin/python}
DATA=${DATA:-$BASE/datasets/Haze4K/Haze4K}
EVID=$REMOTE_ROOT/experience_docx/experiment_logs/haze4k_dta_v2_calibrated_20260611
STATUS=$EVID/status.txt
LOG=$EVID/dta_v2_oof_splits.log
OUT=$EVID/dta_v2_haze4k_oof_splits_seed3407.json
mkdir -p "$EVID"
{
  echo "oof_splits_start dta_v2 $(date --iso-8601=seconds)"
  echo "remote_root=$REMOTE_ROOT"
  echo "data=$DATA"
  echo "output=$OUT"
} | tee -a "$STATUS"
cd "$REMOTE_ROOT"
set +e
PYTHONUNBUFFERED=1 "$PY" experience_docx/tools/make_haze4k_dta_oof_splits.py \
  --data_dir "$DATA" \
  --output "$OUT" \
  --folds 5 \
  --seed 3407 \
  2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e
echo "oof_splits_done rc=$rc dta_v2 $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [[ "$rc" -eq 0 ]]; then
  echo "DTA_V2_OOF_SPLITS_OK" | tee -a "$STATUS"
else
  echo "DTA_V2_OOF_SPLITS_FAILED" | tee -a "$STATUS"
fi
exit "$rc"
