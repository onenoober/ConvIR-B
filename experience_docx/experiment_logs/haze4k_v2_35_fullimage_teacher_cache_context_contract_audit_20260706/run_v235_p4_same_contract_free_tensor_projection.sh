#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
REMOTE_ROOT="/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-haze4k-v2-35-fullimage-teacher-cache-context-contract-audit"
EVID="$REMOTE_ROOT/experience_docx/experiment_logs/haze4k_v2_35_fullimage_teacher_cache_context_contract_audit_20260706"
PY="/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python"
P0C="/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-haze4k-v2-34-nopost-teacher-delta-projection-and-multistage-bridge-audit/experience_docx/experiment_logs/haze4k_v2_34_nopost_teacher_delta_projection_and_multistage_bridge_audit_20260706/v234_p0c_metric_contract_diagnostic.csv"
LOG="$EVID/runtime_logs/v235_p4_same_contract_free_tensor_projection.log"
STATUS="$EVID/status.txt"
mkdir -p "$EVID/runtime_logs"
cd "$REMOTE_ROOT"
echo "v235_p4_start $(date --iso-8601=seconds) locked=untouched cuda_visible=${CUDA_VISIBLE_DEVICES}" | tee -a "$STATUS"
set +e
"$PY" -u experience_docx/tools/run_haze4k_v235_context_contract_audit.py p4-projection \
  --out-dir "$EVID" \
  --p0c-csv "$P0C" \
  --p3-summary "$EVID/v235_p3_same_contract_positive_substrate_summary.json" \
  --device cuda \
  --learning-rate 0.003 \
  --grad-clip-norm 0.5 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e
echo "v235_p4_done rc=$rc $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [ "$rc" -eq 0 ]; then echo V235_P4_COMMAND_OK; else echo V235_P4_COMMAND_FAILED; fi
exit "$rc"
