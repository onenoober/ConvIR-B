set -euo pipefail
REMOTE_ROOT="/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-haze4k-v2-34-nopost-teacher-delta-projection-and-multistage-bridge-audit"
EVID="$REMOTE_ROOT/experience_docx/experiment_logs/haze4k_v2_34_nopost_teacher_delta_projection_and_multistage_bridge_audit_20260706"
PY="/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python"
LOG="$EVID/runtime_logs/v234_p0c_metric_contract_diag.log"
STATUS="$EVID/status.txt"

mkdir -p "$EVID/runtime_logs"
cd "$REMOTE_ROOT"
echo "v234_p0c_metric_contract_diag_start $(date +%Y-%m-%dT%H:%M:%S%z) locked=untouched" | tee -a "$STATUS"
set +e
"$PY" -u experience_docx/tools/v234_p0c_metric_contract_diag.py \
  --repo "$REMOTE_ROOT" \
  --out-dir "$EVID" \
  --limit 32 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e
echo "v234_p0c_metric_contract_diag_done rc=$rc $(date +%Y-%m-%dT%H:%M:%S%z)" | tee -a "$STATUS"
if [ "$rc" -eq 0 ]; then
  echo V234_P0C_COMMAND_OK
else
  echo V234_P0C_COMMAND_FAILED
fi
exit "$rc"
