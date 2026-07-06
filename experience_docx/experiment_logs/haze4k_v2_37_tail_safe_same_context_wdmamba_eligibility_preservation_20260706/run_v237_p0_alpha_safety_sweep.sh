#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT="/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-haze4k-v2-37-tail-safe-same-context-wdmamba-eligibility-preservation"
EVID="$REMOTE_ROOT/experience_docx/experiment_logs/haze4k_v2_37_tail_safe_same_context_wdmamba_eligibility_preservation_20260706"
PY="/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python"
V235_MANIFEST="/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-haze4k-v2-35-fullimage-teacher-cache-context-contract-audit/experience_docx/experiment_logs/haze4k_v2_35_fullimage_teacher_cache_context_contract_audit_20260706/v235_p1_fullimage_teacher_cache_manifest.csv"
FOLD_MANIFEST="$REMOTE_ROOT/experience_docx/experiment_logs/haze4k_v2_36_same_contract_wlfbridge_s4s6_generator_trainability_20260706/v236_p0_fold_split_manifest.csv"
LOG="$EVID/runtime_logs/v237_p0_alpha_safety_sweep.log"
STATUS="$EVID/status.txt"

mkdir -p "$EVID/runtime_logs"
cd "$REMOTE_ROOT"
echo "v237_p0_start $(date --iso-8601=seconds) locked=untouched" | tee -a "$STATUS"
set +e
"$PY" -u experience_docx/tools/run_haze4k_v237_tail_safe_wdmamba_audit.py p0-alpha-sweep \
  --out-dir "$EVID" \
  --v235-manifest "$V235_MANIFEST" \
  --fold-manifest "$FOLD_MANIFEST" 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e
echo "v237_p0_done rc=$rc $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [ "$rc" -eq 0 ]; then echo V237_P0_COMMAND_OK; else echo V237_P0_COMMAND_FAILED; fi
exit "$rc"
