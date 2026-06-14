#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT=${REMOTE_ROOT:-/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v20-strongexpert-gainmix-c1}
EVID=${EVID:-$REMOTE_ROOT/experience_docx/experiment_logs/haze4k_v2_0_strongexpert_gainmix_20260614}
PY=${PY:-/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python}
FEATURE_ROWS=${FEATURE_ROWS:-$EVID/v20_c2_outputdiff_feature_rows.csv}
STATUS=$EVID/status_c2b.txt
LOG=$EVID/v20_c2b_multirule_router.log

mkdir -p "$EVID"
echo "===== v20_c2b_multirule_router_start $(date --iso-8601=seconds) =====" >> "$LOG"
{
  echo "v20_c2b_multirule_router_start $(date --iso-8601=seconds)"
  echo "remote_root=$REMOTE_ROOT"
  echo "python=$PY"
  echo "feature_rows=$FEATURE_ROWS"
  echo "locked_test_touched=false"
  if [ -d "$REMOTE_ROOT/.git" ]; then
    git -C "$REMOTE_ROOT" branch --show-current | sed 's/^/branch=/'
    git -C "$REMOTE_ROOT" rev-parse --short HEAD | sed 's/^/commit=/'
    git -C "$REMOTE_ROOT" status --short | sed -n '1,160p' | sed 's/^/git_status=/'
  fi
  test -x "$PY" && echo "python_exists=true"
  test -f "$FEATURE_ROWS" && echo "feature_rows_exists=true"
  if [ -f "$FEATURE_ROWS" ]; then
    wc -l "$FEATURE_ROWS" | sed 's/^/feature_rows_wc=/'
    sha256sum "$FEATURE_ROWS" | sed 's/^/feature_rows_sha256=/'
  fi
} | tee -a "$STATUS"

cd "$REMOTE_ROOT"
set +e
"$PY" experience_docx/tools/audit_haze4k_v20_c2b_multirule_router.py \
  --feature_rows "$FEATURE_ROWS" \
  --out_dir "$EVID" \
  --top_k 600 \
  2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
set -e
echo "v20_c2b_multirule_router_done rc=$rc $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [ "$rc" -eq 0 ]; then
  echo "V20_C2B_MULTIRULE_ROUTER_OK $(date --iso-8601=seconds)" | tee -a "$STATUS"
else
  echo "V20_C2B_MULTIRULE_ROUTER_FAILED rc=$rc $(date --iso-8601=seconds)" | tee -a "$STATUS"
fi
exit "$rc"
