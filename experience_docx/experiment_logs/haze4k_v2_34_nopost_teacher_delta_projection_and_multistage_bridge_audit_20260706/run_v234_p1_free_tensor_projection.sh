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
STATUS=$EVID/status.txt

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1} TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

mkdir -p "$EVID"
echo "v234_p1_start $(date --iso-8601=seconds)" | tee -a "$STATUS"
cd "$WORK"
set +e
PYTHONUNBUFFERED=1 "$PY" experience_docx/tools/run_haze4k_v234_nopost_projection_audit.py \
  --phase p1 \
  --data_dir "$DATA" \
  --checkpoint "$A0" \
  --output_dir "$EVID" \
  --wdmamba_repo "$WDMAMBA_REPO" \
  --wdmamba_checkpoint "$WDMAMBA_CKPT" \
  --teacher_alpha 0.5 \
  --sample_count 32 \
  --crop_size 256 \
  --projection_steps 32 \
  --learning_rate 0.03 \
  --energy_weight 0.0001 \
  > "$EVID/v234_p1_free_tensor_projection.log" 2>&1
rc=$?
set -e
echo "v234_p1_done rc=$rc $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [[ "$rc" -eq 0 ]]; then
  echo "V234_P1_FREE_TENSOR_PROJECTION_OK" | tee -a "$STATUS"
else
  echo "V234_P1_FREE_TENSOR_PROJECTION_FAILED" | tee -a "$STATUS"
fi
exit "$rc"
