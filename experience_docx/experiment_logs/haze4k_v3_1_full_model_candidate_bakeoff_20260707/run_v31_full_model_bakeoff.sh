#!/usr/bin/env bash
set -euo pipefail
ROOT=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-github-main
EVID="$ROOT/experience_docx/experiment_logs/haze4k_v3_1_full_model_candidate_bakeoff_20260707"
PY=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
STATUS="$EVID/status.txt"
LOG="$EVID/runtime_logs/v31_full_model_bakeoff.log"
mkdir -p "$EVID/runtime_logs"
cd "$ROOT"
echo "v31_preflight $(date --iso-8601=seconds) branch=$(git branch --show-current) head=$(git rev-parse --short HEAD) route=diagnostic_only locked=untouched canary80=blocked selector_alpha=blocked bridge_generator=blocked" | tee -a "$STATUS"
for f in \
  /sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-haze4k-v2-37-tail-safe-same-context-wdmamba-eligibility-preservation/experience_docx/experiment_logs/haze4k_v2_37_tail_safe_same_context_wdmamba_eligibility_preservation_20260706/v237_p0_alpha_safety_sweep_per_image.csv \
  /sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-haze4k-v2-39-convirl-same-family-teacher-contract-audit/experience_docx/experiment_logs/haze4k_v2_39_convirl_same_family_teacher_contract_audit_20260706/v239_p0_convirl_fullimage_teacher_sweep_per_image.csv \
  /sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-github-main/experience_docx/experiment_logs/haze4k_v2_2_c8_mini_expert_oracle_20260615/v22_c8_2_fsudp_full_per_image.csv; do
  if [ ! -f "$f" ]; then
    echo "v31_blocked_missing_source $f $(date --iso-8601=seconds)" | tee -a "$STATUS"
    echo V31_FULL_MODEL_BAKEOFF_FAILED
    exit 2
  fi
done
"$PY" -m py_compile "$EVID/v31_aggregate_full_model_bakeoff.py"
echo "v31_aggregate_start $(date --iso-8601=seconds)" | tee -a "$STATUS"
set +e
"$PY" -u "$EVID/v31_aggregate_full_model_bakeoff.py" 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e
echo "v31_aggregate_done rc=$rc $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [ "$rc" -eq 0 ]; then echo V31_FULL_MODEL_BAKEOFF_OK; else echo V31_FULL_MODEL_BAKEOFF_FAILED; fi
exit "$rc"
