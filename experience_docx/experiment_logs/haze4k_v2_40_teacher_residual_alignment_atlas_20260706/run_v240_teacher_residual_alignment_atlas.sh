#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT="/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-haze4k-v2-40-teacher-residual-alignment-atlas"
EVID="$REMOTE_ROOT/experience_docx/experiment_logs/haze4k_v2_40_teacher_residual_alignment_atlas_20260706"
PY="/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python"
V237_P0="/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-haze4k-v2-37-tail-safe-same-context-wdmamba-eligibility-preservation/experience_docx/experiment_logs/haze4k_v2_37_tail_safe_same_context_wdmamba_eligibility_preservation_20260706/v237_p0_alpha_safety_sweep_per_image.csv"
V238_P0="/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-haze4k-v2-38-microalpha-same-context-wdmamba-safe-substrate-projection/experience_docx/experiment_logs/haze4k_v2_38_microalpha_same_context_wdmamba_safe_substrate_projection_20260706/v238_p0_microalpha_safety_sweep_per_image.csv"
V239_P0="/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-haze4k-v2-39-convirl-same-family-teacher-contract-audit/experience_docx/experiment_logs/haze4k_v2_39_convirl_same_family_teacher_contract_audit_20260706/v239_p0_convirl_fullimage_teacher_sweep_per_image.csv"
FEATURE_MANIFEST="/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-haze4k-v2-38b-rich-target-only-noop-unsafe-separability-audit/experience_docx/experiment_logs/haze4k_v2_38b_rich_target_only_noop_unsafe_separability_audit_20260706/v238b_p0_rich_target_feature_manifest.csv"
STATUS="$EVID/status.txt"
LOG="$EVID/runtime_logs/v240_teacher_residual_alignment_atlas.log"

mkdir -p "$EVID/runtime_logs"
cd "$REMOTE_ROOT"

echo "v240_preflight $(date --iso-8601=seconds) branch=$(git branch --show-current) head=$(git rev-parse --short HEAD) locked=untouched bridge=blocked generator=blocked canary80=blocked" | tee -a "$STATUS"
"$PY" -m py_compile experience_docx/tools/run_haze4k_v240_teacher_residual_alignment_atlas.py
for required in "$V237_P0" "$V238_P0" "$V239_P0" "$FEATURE_MANIFEST"; do
  if [ ! -f "$required" ]; then
    echo "v240_blocked_missing_required_file $required $(date --iso-8601=seconds)" | tee -a "$STATUS"
    echo V240_ATLAS_COMMAND_FAILED
    exit 2
  fi
done

echo "v240_atlas_start $(date --iso-8601=seconds)" | tee -a "$STATUS"
set +e
"$PY" -u experience_docx/tools/run_haze4k_v240_teacher_residual_alignment_atlas.py \
  --out-dir "$EVID" \
  --v237-p0-csv "$V237_P0" \
  --v238-p0-csv "$V238_P0" \
  --v239-p0-csv "$V239_P0" \
  --feature-manifest "$FEATURE_MANIFEST" \
  2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e
echo "v240_atlas_done rc=$rc $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [ "$rc" -eq 0 ]; then echo V240_ATLAS_COMMAND_OK; else echo V240_ATLAS_COMMAND_FAILED; fi
exit "$rc"
