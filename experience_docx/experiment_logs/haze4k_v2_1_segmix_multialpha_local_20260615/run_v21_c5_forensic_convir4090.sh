#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT=${REMOTE_ROOT:-/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v21-segmix-multialpha-local}
EVID=${EVID:-$REMOTE_ROOT/experience_docx/experiment_logs/haze4k_v2_1_segmix_multialpha_local_20260615}
PY=${PY:-/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python}
V20_EVID=${V20_EVID:-$REMOTE_ROOT/experience_docx/experiment_logs/haze4k_v2_0_strongexpert_gainmix_20260614}
ALPHA_ROWS=${ALPHA_ROWS:-$V20_EVID/v20_c2d_alpha_feature_rows.csv}
C4_SUMMARY=${C4_SUMMARY:-$V20_EVID/v20_c4_formal_5x3_summary.json}
STATUS=$EVID/status_c5.txt
LOG=$EVID/v21_c5_forensic.log

mkdir -p "$EVID"
echo "===== v21_c5_forensic_start $(date --iso-8601=seconds) =====" >> "$LOG"
{
  echo "v21_c5_forensic_start $(date --iso-8601=seconds)"
  echo "remote_root=$REMOTE_ROOT"
  echo "python=$PY"
  echo "alpha_rows=$ALPHA_ROWS"
  echo "c4_summary=$C4_SUMMARY"
  echo "locked_test_touched=false"
  if [ -d "$REMOTE_ROOT/.git" ]; then
    git -C "$REMOTE_ROOT" branch --show-current | sed 's/^/branch=/'
    git -C "$REMOTE_ROOT" rev-parse --short HEAD | sed 's/^/commit=/'
    git -C "$REMOTE_ROOT" status --short | sed -n '1,200p' | sed 's/^/git_status=/'
  fi
  test -x "$PY" && echo "python_exists=true"
  test -f "$ALPHA_ROWS" && echo "alpha_rows_exists=true"
  test -f "$C4_SUMMARY" && echo "c4_summary_exists=true"
  if [ -f "$ALPHA_ROWS" ]; then sha256sum "$ALPHA_ROWS" | sed 's/^/alpha_rows_sha256=/'; fi
  if [ -f "$C4_SUMMARY" ]; then sha256sum "$C4_SUMMARY" | sed 's/^/c4_summary_sha256=/'; fi
} | tee -a "$STATUS"

cd "$REMOTE_ROOT"
set +e
"$PY" experience_docx/tools/audit_haze4k_v21_c5_c4_failure_forensic.py \
  --alpha_rows "$ALPHA_ROWS" \
  --c4_summary "$C4_SUMMARY" \
  --out_dir "$EVID" \
  2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
set -e
echo "v21_c5_forensic_done rc=$rc $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [ "$rc" -eq 0 ]; then
  echo "V21_C5_FORENSIC_OK $(date --iso-8601=seconds)" | tee -a "$STATUS"
else
  echo "V21_C5_FORENSIC_FAILED rc=$rc $(date --iso-8601=seconds)" | tee -a "$STATUS"
fi
exit "$rc"
