#!/usr/bin/env bash
set -euo pipefail
REMOTE_ROOT="/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-haze4k-v2-35-fullimage-teacher-cache-context-contract-audit"
EVID="$REMOTE_ROOT/experience_docx/experiment_logs/haze4k_v2_35_fullimage_teacher_cache_context_contract_audit_20260706"
PY="/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python"
LOG="$EVID/runtime_logs/v235_p3_same_contract_substrate.log"
STATUS="$EVID/status.txt"
mkdir -p "$EVID/runtime_logs"
cd "$REMOTE_ROOT"
echo "v235_p3_start $(date --iso-8601=seconds) locked=untouched" | tee -a "$STATUS"
set +e
"$PY" -u experience_docx/tools/run_haze4k_v235_context_contract_audit.py p3-substrate \
  --out-dir "$EVID" \
  --p2-csv "$EVID/v235_p2_context_size_sweep_per_image.csv" \
  --p2-summary "$EVID/v235_p2_context_size_sweep_summary.json" 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e
echo "v235_p3_done rc=$rc $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [ "$rc" -eq 0 ]; then echo V235_P3_COMMAND_OK; else echo V235_P3_COMMAND_FAILED; fi
exit "$rc"
