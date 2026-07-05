#!/usr/bin/env bash
set -euo pipefail

BASE=/sda/home/wangyuxin/ConvIR-B
WORK=$BASE/repos/ConvIR-B-haze4k-v2-33-nopost-teacher-benefit-source-and-bilfcf-trainability-audit
EVID=$WORK/experience_docx/experiment_logs/haze4k_v2_33_nopost_teacher_benefit_source_and_bilfcf_trainability_audit_20260705
PY=$BASE/envs/convir-cu121/bin/python
DATA=$BASE/datasets/Haze4K/Haze4K
A0=$BASE/checkpoints/official/Haze4K/haze4k-base.pkl
WDMAMBA_REPO=$BASE/repos/external_experts/WDMamba
WDMAMBA_CKPT=$BASE/checkpoints/WDMamba_ckpts/haze4k_35.88.pth
STATUS=$EVID/status.txt

export CUDA_VISIBLE_DEVICES=0 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

mkdir -p "$EVID"
echo "p4_teacher_benefit_masked_canary32_start $(date --iso-8601=seconds)" | tee -a "$STATUS"
cd "$WORK"
set +e
PYTHONUNBUFFERED=1 "$PY" experience_docx/tools/run_haze4k_v233_nopost_audit.py \
  --phase p4 \
  --data_dir "$DATA" \
  --checkpoint "$A0" \
  --output_dir "$EVID" \
  --wdmamba_repo "$WDMAMBA_REPO" \
  --wdmamba_checkpoint "$WDMAMBA_CKPT" \
  --teacher_alpha 0.5 \
  --p4_sample_count 32 \
  --p4_batch_size 4 \
  --p4_mask_min_gain 0.05 \
  --steps 24 \
  > "$EVID/v233_p4_teacher_benefit_masked_canary32.log" 2>&1
rc=$?
set -e
echo "p4_teacher_benefit_masked_canary32_done rc=$rc $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [[ "$rc" -eq 0 ]]; then
  echo "V233_P4_TEACHER_BENEFIT_MASKED_CANARY32_OK" | tee -a "$STATUS"
else
  echo "V233_P4_TEACHER_BENEFIT_MASKED_CANARY32_FAILED" | tee -a "$STATUS"
fi
exit "$rc"
