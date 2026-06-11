#!/usr/bin/env bash
set -euo pipefail
BASE=${BASE:-/sda/home/wangyuxin/ConvIR-B}
REMOTE_ROOT=${REMOTE_ROOT:-$BASE/repos/ConvIR-B-dta-v2-calibrated}
ITS=$REMOTE_ROOT/Dehazing/ITS
PY=${PY:-$BASE/envs/convir-cu121/bin/python}
DATA=${DATA:-$BASE/datasets/Haze4K/Haze4K}
A0=${A0:-$BASE/checkpoints/official/Haze4K/haze4k-base.pkl}
DEPTH=${DEPTH:-$BASE/depth_cache/depth_anything_v2_small_hf}
EVID=$REMOTE_ROOT/experience_docx/experiment_logs/haze4k_dta_v2_calibrated_20260611
STATUS=$EVID/status.txt
LOG=$EVID/dta_v2_preflight.log
JSON_OUT=$EVID/dta_v2_preflight.json
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
mkdir -p "$EVID"
{
  echo "preflight_start dta_v2 $(date --iso-8601=seconds)"
  echo "remote_root=$REMOTE_ROOT"
  echo "python=$PY"
  echo "data=$DATA"
  echo "checkpoint=$A0"
  echo "depth=$DEPTH"
  echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
} | tee -a "$STATUS"
cd "$REMOTE_ROOT"
{
  git branch --show-current
  git rev-parse --short HEAD
  git status --short
} | tee -a "$STATUS"
cd "$ITS"
set +e
PYTHONUNBUFFERED=1 "$PY" "$REMOTE_ROOT/experience_docx/tools/check_haze4k_dta_preflight.py" \
  --checkpoint "$A0" \
  --data_dir "$DATA" \
  --depth_cache_dir "$DEPTH" \
  --depth_split train \
  --output_json "$JSON_OUT" \
  --arch dta_v2 \
  --dta_variant v2 \
  --dta_prior_channels 32 \
  --dta_gate_bias -6.0 \
  --dta_gate_limit 0.06 \
  --dta_gamma_limit 0.12 \
  --dta_beta_limit 0.06 \
  --dta_confidence_floor 0.25 \
  --dta_confidence_local_scale 6.0 \
  --dta_output_residual_scale 0.03 \
  --use_trans_gt \
  --rank_weight 0.001 \
  --tv_weight 0.0001 \
  --trans_weight 0.02 \
  --phys_weight 0.005 \
  2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e
echo "preflight_done rc=$rc dta_v2 $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [[ "$rc" -eq 0 ]]; then
  echo "DTA_V2_PREFLIGHT_OK" | tee -a "$STATUS"
else
  echo "DTA_V2_PREFLIGHT_FAILED" | tee -a "$STATUS"
fi
exit "$rc"
