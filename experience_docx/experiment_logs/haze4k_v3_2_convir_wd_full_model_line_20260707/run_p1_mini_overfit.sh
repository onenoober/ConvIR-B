#!/usr/bin/env bash
set -euo pipefail

BASE=/sda/home/wangyuxin/ConvIR-B
WORK=$BASE/repos/ConvIR-B-v3-2-convir-wd
EVID=$WORK/experience_docx/experiment_logs/haze4k_v3_2_convir_wd_full_model_line_20260707
PY=$BASE/envs/convir-cu121/bin/python
DATA=$BASE/datasets/Haze4K/Haze4K
A0=$BASE/checkpoints/official/Haze4K/haze4k-base.pkl
STATUS=$EVID/status.txt
LOG=$EVID/p1_mini_overfit_v32.log
JSON_OUT=$EVID/v32_p1_mini_overfit.json

export CUDA_VISIBLE_DEVICES=0
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

mkdir -p "$EVID"
{
  echo "p1_mini_overfit_start haze4k_v3_2_convir_wd_full_model_line_20260707 $(date --iso-8601=seconds)"
  echo "work=$WORK"
  echo "python=$PY"
  echo "data=$DATA"
  echo "a0=$A0"
  cd "$WORK"
  echo "branch=$(git branch --show-current)"
  echo "commit=$(git rev-parse --short HEAD)"
} | tee -a "$STATUS"

cd "$WORK"
set +e
PYTHONUNBUFFERED=1 "$PY" experience_docx/tools/haze4k_v32_convir_wd_mini_overfit.py \
  --data_dir "$DATA" \
  --checkpoint "$A0" \
  --output "$JSON_OUT" \
  --samples 8 \
  --crop_size 128 \
  --steps 120 \
  --batch_size 2 \
  --scope wd_decoder \
  --wd_lr 0.0002 \
  --decoder_lr 0.00001 \
  --grad_clip_norm 0.01 \
  --gate_loss_ratio 0.95 \
  > "$LOG" 2>&1
rc=$?
set -e

echo "p1_mini_overfit_done rc=$rc haze4k_v3_2_convir_wd_full_model_line_20260707 $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [[ "$rc" -eq 0 ]]; then
  echo "V32_P1_MINI_OVERFIT_OK" | tee -a "$STATUS"
else
  echo "V32_P1_MINI_OVERFIT_FAILED" | tee -a "$STATUS"
fi
exit "$rc"
