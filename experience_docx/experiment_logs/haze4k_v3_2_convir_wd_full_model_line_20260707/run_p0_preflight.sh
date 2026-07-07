#!/usr/bin/env bash
set -euo pipefail

BASE=/sda/home/wangyuxin/ConvIR-B
WORK=$BASE/repos/ConvIR-B-v3-2-convir-wd
EVID=$WORK/experience_docx/experiment_logs/haze4k_v3_2_convir_wd_full_model_line_20260707
PY=$BASE/envs/convir-cu121/bin/python
DATA=$BASE/datasets/Haze4K/Haze4K
A0=$BASE/checkpoints/official/Haze4K/haze4k-base.pkl
STATUS=$EVID/status.txt
LOG=$EVID/preflight_v32.log
JSON_OUT=$EVID/v32_p0_preflight.json

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

mkdir -p "$EVID"
{
  echo "p0_preflight_start haze4k_v3_2_convir_wd_full_model_line_20260707 $(date --iso-8601=seconds)"
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
PYTHONUNBUFFERED=1 "$PY" experience_docx/tools/haze4k_v32_convir_wd_preflight.py \
  --data_dir "$DATA" \
  --checkpoint "$A0" \
  --output "$JSON_OUT" \
  > "$LOG" 2>&1
rc=$?
set -e

echo "p0_preflight_done rc=$rc haze4k_v3_2_convir_wd_full_model_line_20260707 $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [[ "$rc" -eq 0 ]]; then
  echo "V32_P0_PREFLIGHT_OK" | tee -a "$STATUS"
else
  echo "V32_P0_PREFLIGHT_FAILED" | tee -a "$STATUS"
fi
exit "$rc"
