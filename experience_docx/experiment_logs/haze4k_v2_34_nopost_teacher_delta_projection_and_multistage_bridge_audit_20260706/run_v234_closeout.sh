#!/usr/bin/env bash
set -euo pipefail

BASE=/sda/home/wangyuxin/ConvIR-B
WORK=$BASE/repos/ConvIR-B-haze4k-v2-34-nopost-teacher-delta-projection-and-multistage-bridge-audit
EVID=$WORK/experience_docx/experiment_logs/haze4k_v2_34_nopost_teacher_delta_projection_and_multistage_bridge_audit_20260706
PY=$BASE/envs/convir-cu121/bin/python
STATUS=$EVID/status.txt

mkdir -p "$EVID"
echo "v234_closeout_start $(date --iso-8601=seconds)" | tee -a "$STATUS"
cd "$WORK"
set +e
PYTHONUNBUFFERED=1 "$PY" experience_docx/tools/run_haze4k_v234_nopost_projection_audit.py \
  --phase closeout \
  --output_dir "$EVID" \
  > "$EVID/v234_closeout.log" 2>&1
rc=$?
set -e
echo "v234_closeout_done rc=$rc $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [[ "$rc" -eq 0 ]]; then
  echo "V234_CLOSEOUT_OK" | tee -a "$STATUS"
else
  echo "V234_CLOSEOUT_FAILED" | tee -a "$STATUS"
fi
exit "$rc"

