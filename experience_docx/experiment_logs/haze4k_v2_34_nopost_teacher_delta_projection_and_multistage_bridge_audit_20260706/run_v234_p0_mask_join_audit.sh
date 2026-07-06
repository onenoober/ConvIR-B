#!/usr/bin/env bash
set -euo pipefail

BASE=/sda/home/wangyuxin/ConvIR-B
WORK=$BASE/repos/ConvIR-B-haze4k-v2-34-nopost-teacher-delta-projection-and-multistage-bridge-audit
EVID=$WORK/experience_docx/experiment_logs/haze4k_v2_34_nopost_teacher_delta_projection_and_multistage_bridge_audit_20260706
PY=$BASE/envs/convir-cu121/bin/python
DATA=$BASE/datasets/Haze4K/Haze4K
A0=$BASE/checkpoints/official/Haze4K/haze4k-base.pkl
WDMAMBA_REPO=$BASE/repos/external_experts/WDMamba
WDMAMBA_CKPT=$BASE/checkpoints/WDMamba_ckpts/haze4k_35.88.pth
WD_TABLE=$BASE/repos/ConvIR-B-v22-c8-mini-expert-oracle/experience_docx/experiment_logs/haze4k_v2_2_c8_mini_expert_oracle_20260615/v22_c8_1_wdmamba_full_per_image.csv
V233_P4=$BASE/repos/ConvIR-B-haze4k-v2-33-nopost-teacher-benefit-source-and-bilfcf-trainability-audit/experience_docx/experiment_logs/haze4k_v2_33_nopost_teacher_benefit_source_and_bilfcf_trainability_audit_20260705/v233_p4_teacher_benefit_masked_canary32_per_image.csv
STATUS=$EVID/status.txt

export CUDA_VISIBLE_DEVICES=0 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

mkdir -p "$EVID"
echo "v234_p0_start $(date --iso-8601=seconds)" | tee -a "$STATUS"
cd "$WORK"
set +e
PYTHONUNBUFFERED=1 "$PY" experience_docx/tools/run_haze4k_v234_nopost_projection_audit.py \
  --phase p0 \
  --data_dir "$DATA" \
  --checkpoint "$A0" \
  --output_dir "$EVID" \
  --wdmamba_repo "$WDMAMBA_REPO" \
  --wdmamba_checkpoint "$WDMAMBA_CKPT" \
  --wdmamba_table "$WD_TABLE" \
  --v233_p4_per_image "$V233_P4" \
  --teacher_alpha 0.5 \
  --sample_count 32 \
  --crop_size 256 \
  > "$EVID/v234_p0_mask_join_audit.log" 2>&1
rc=$?
set -e
echo "v234_p0_done rc=$rc $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [[ "$rc" -eq 0 ]]; then
  echo "V234_P0_MASK_JOIN_AUDIT_OK" | tee -a "$STATUS"
else
  echo "V234_P0_MASK_JOIN_AUDIT_FAILED" | tee -a "$STATUS"
fi
exit "$rc"

