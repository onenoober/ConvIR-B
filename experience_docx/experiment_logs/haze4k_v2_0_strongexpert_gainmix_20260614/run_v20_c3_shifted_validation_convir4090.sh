#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT=${REMOTE_ROOT:-/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v20-strongexpert-gainmix-c1}
EVID=${EVID:-$REMOTE_ROOT/experience_docx/experiment_logs/haze4k_v2_0_strongexpert_gainmix_20260614}
PY=${PY:-/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python}
ALPHA_ROWS=${ALPHA_ROWS:-$EVID/v20_c2d_alpha_feature_rows.csv}
STATUS=$EVID/status_c3.txt
LOG=$EVID/v20_c3_shifted_validation.log

mkdir -p "$EVID"
echo "===== v20_c3_shifted_validation_start $(date --iso-8601=seconds) =====" >> "$LOG"
{
  echo "v20_c3_shifted_validation_start $(date --iso-8601=seconds)"
  echo "remote_root=$REMOTE_ROOT"
  echo "python=$PY"
  echo "alpha_rows=$ALPHA_ROWS"
  echo "locked_test_touched=false"
  if [ -d "$REMOTE_ROOT/.git" ]; then
    git -C "$REMOTE_ROOT" branch --show-current | sed 's/^/branch=/'
    git -C "$REMOTE_ROOT" rev-parse --short HEAD | sed 's/^/commit=/'
    git -C "$REMOTE_ROOT" status --short | sed -n '1,180p' | sed 's/^/git_status=/'
  fi
  test -x "$PY" && echo "python_exists=true"
  test -f "$ALPHA_ROWS" && echo "alpha_rows_exists=true"
  if [ -f "$ALPHA_ROWS" ]; then
    wc -l "$ALPHA_ROWS" | sed 's/^/alpha_rows_wc=/'
    sha256sum "$ALPHA_ROWS" | sed 's/^/alpha_rows_sha256=/'
  fi
} | tee -a "$STATUS"

cd "$REMOTE_ROOT"
set +e
"$PY" experience_docx/tools/audit_haze4k_v20_c3_shifted_validation.py \
  --alpha_rows "$ALPHA_ROWS" \
  --out_dir "$EVID" \
  2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
set -e
echo "v20_c3_shifted_validation_done rc=$rc $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [ "$rc" -eq 0 ]; then
  echo "V20_C3_SHIFTED_VALIDATION_OK $(date --iso-8601=seconds)" | tee -a "$STATUS"
else
  echo "V20_C3_SHIFTED_VALIDATION_FAILED rc=$rc $(date --iso-8601=seconds)" | tee -a "$STATUS"
fi
exit "$rc"
