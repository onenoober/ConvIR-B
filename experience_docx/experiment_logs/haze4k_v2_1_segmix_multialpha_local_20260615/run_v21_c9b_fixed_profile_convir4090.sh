#!/usr/bin/env bash
set -euo pipefail
REMOTE_ROOT=${REMOTE_ROOT:-/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v21-segmix-multialpha-local}
EVID=${EVID:-$REMOTE_ROOT/experience_docx/experiment_logs/haze4k_v2_1_segmix_multialpha_local_20260615}
PY=${PY:-/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python}
PROFILE_ROWS=${PROFILE_ROWS:-$EVID/v21_c7c_risk_profile_per_image.csv}
IMAGE_ROWS=${IMAGE_ROWS:-$EVID/v21_c7b_image_rows.csv}
IMAGE_FEATURE_ROWS=${IMAGE_FEATURE_ROWS:-$EVID/v21_c6_multialpha_feature_rows.csv}
FIXED_PROFILE=${FIXED_PROFILE:-riskcap36_no075}
STATUS=$EVID/status_c9b.txt
LOG=$EVID/v21_c9b_fixed_profile.log
mkdir -p "$EVID"
echo "===== v21_c9b_fixed_profile_start $(date --iso-8601=seconds) =====" >> "$LOG"
{
 echo "v21_c9b_fixed_profile_start $(date --iso-8601=seconds)"
 echo "remote_root=$REMOTE_ROOT"
 echo "python=$PY"
 echo "profile_rows=$PROFILE_ROWS"
 echo "image_rows=$IMAGE_ROWS"
 echo "image_feature_rows=$IMAGE_FEATURE_ROWS"
 echo "fixed_profile=$FIXED_PROFILE"
 echo "locked_test_touched=false"
 if [ -f "$REMOTE_ROOT/.codex_source_branch" ]; then sed 's/^/source_branch=/' "$REMOTE_ROOT/.codex_source_branch"; fi
 if [ -f "$REMOTE_ROOT/.codex_source_commit" ]; then sed 's/^/source_commit=/' "$REMOTE_ROOT/.codex_source_commit"; fi
 if [ -f "$REMOTE_ROOT/.codex_source_copy_time" ]; then sed 's/^/source_copy_time=/' "$REMOTE_ROOT/.codex_source_copy_time"; fi
 test -x "$PY" && echo "python_exists=true"
 test -f "$PROFILE_ROWS" && echo "profile_rows_exists=true"
 test -f "$IMAGE_ROWS" && echo "image_rows_exists=true"
 test -f "$IMAGE_FEATURE_ROWS" && echo "image_feature_rows_exists=true"
} | tee -a "$STATUS"
cd "$REMOTE_ROOT"
set +e
"$PY" experience_docx/tools/audit_haze4k_v21_c9b_fixed_profile_stress.py \
 --profile_per_image "$PROFILE_ROWS" \
 --image_rows "$IMAGE_ROWS" \
 --image_feature_rows "$IMAGE_FEATURE_ROWS" \
 --fixed_profile "$FIXED_PROFILE" \
 --out_dir "$EVID" \
 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
set -e
echo "v21_c9b_fixed_profile_done rc=$rc $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [ "$rc" -eq 0 ]; then echo "V21_C9B_FIXED_PROFILE_OK $(date --iso-8601=seconds)" | tee -a "$STATUS"; else echo "V21_C9B_FIXED_PROFILE_FAILED rc=$rc $(date --iso-8601=seconds)" | tee -a "$STATUS"; fi
exit "$rc"
