#!/usr/bin/env bash
set -euo pipefail
BASE=/sda/home/wangyuxin/ConvIR-B
WORK=$BASE/repos/ConvIR-B-haze4k-v2-33-nopost-teacher-benefit-source-and-bilfcf-trainability-audit
EVID=$WORK/experience_docx/experiment_logs/haze4k_v2_33_nopost_teacher_benefit_source_and_bilfcf_trainability_audit_20260705
PY=$BASE/envs/convir-cu121/bin/python
STATUS=$EVID/status.txt
mkdir -p "$EVID"
echo "closeout_start $(date --iso-8601=seconds)" | tee -a "$STATUS"
cd "$WORK"
set +e
PYTHONUNBUFFERED=1 "$PY" experience_docx/tools/run_haze4k_v233_nopost_audit.py --phase closeout --output_dir "$EVID" > "$EVID/v233_closeout.log" 2>&1
rc=$?
set -e
echo "closeout_done rc=$rc $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [[ "$rc" -eq 0 ]]; then
  echo "V233_CLOSEOUT_OK" | tee -a "$STATUS"
else
  echo "V233_CLOSEOUT_FAILED" | tee -a "$STATUS"
fi
exit "$rc"
