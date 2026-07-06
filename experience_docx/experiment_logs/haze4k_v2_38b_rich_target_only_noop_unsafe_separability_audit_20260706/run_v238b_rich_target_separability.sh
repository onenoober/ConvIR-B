#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT="/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-haze4k-v2-38b-rich-target-only-noop-unsafe-separability-audit"
EVID="$REMOTE_ROOT/experience_docx/experiment_logs/haze4k_v2_38b_rich_target_only_noop_unsafe_separability_audit_20260706"
PY="/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python"
P0_CSV="/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-haze4k-v2-37-tail-safe-same-context-wdmamba-eligibility-preservation/experience_docx/experiment_logs/haze4k_v2_37_tail_safe_same_context_wdmamba_eligibility_preservation_20260706/v237_p0_alpha_safety_sweep_per_image.csv"
P4_FEATURES="/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-haze4k-v2-37-tail-safe-same-context-wdmamba-eligibility-preservation/experience_docx/experiment_logs/haze4k_v2_37_tail_safe_same_context_wdmamba_eligibility_preservation_20260706/v237_p4_target_only_eligibility_features.csv"
STATUS="$EVID/status.txt"
mkdir -p "$EVID/runtime_logs"
cd "$REMOTE_ROOT"

echo "v238b_preflight $(date --iso-8601=seconds) branch=$(git branch --show-current) head=$(git rev-parse --short HEAD) locked=untouched bridge=blocked generator=blocked" | tee -a "$STATUS"
"$PY" -m py_compile experience_docx/tools/run_haze4k_v238b_rich_noop_separability_audit.py
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
echo "v238b_start $(date --iso-8601=seconds) cuda_visible_devices=$CUDA_VISIBLE_DEVICES" | tee -a "$STATUS"
REUSE_ARGS=()
if [ -f "$EVID/v238b_p0_rich_target_feature_manifest.csv" ]; then
  REUSE_ARGS=(--reuse-feature-manifest)
  echo "v238b_reuse_feature_manifest $(date --iso-8601=seconds)" | tee -a "$STATUS"
fi
set +e
"$PY" -u experience_docx/tools/run_haze4k_v238b_rich_noop_separability_audit.py \
  --out-dir "$EVID" \
  --p0-csv "$P0_CSV" \
  --p4-features "$P4_FEATURES" \
  --device cuda \
  "${REUSE_ARGS[@]}" \
  2>&1 | tee "$EVID/runtime_logs/v238b_rich_target_separability.log"
rc=${PIPESTATUS[0]}
set -e
echo "v238b_done rc=$rc $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [ "$rc" -eq 0 ]; then echo V238B_COMMAND_OK; else echo V238B_COMMAND_FAILED; fi
exit "$rc"
