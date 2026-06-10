#!/usr/bin/env bash
set -euo pipefail

BASE=${BASE:-/sda/home/wangyuxin/ConvIR-B}
REMOTE_ROOT=${REMOTE_ROOT:-$BASE/repos/ConvIR-B-dta-lowgate}
ITS=$REMOTE_ROOT/Dehazing/ITS
EVID=$REMOTE_ROOT/experience_docx/experiment_logs/haze4k_dta_lowgate_20260610
PY=$BASE/envs/convir-cu121/bin/python
DATA=$BASE/datasets/Haze4K/Haze4K
A0=$BASE/checkpoints/official/Haze4K/haze4k-base.pkl
STATUS=$EVID/status.txt
LOG=$EVID/dta_lowgate_preflight.log
JSON_OUT=$EVID/dta_lowgate_preflight.json
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

find_depth_cache() {
  if [[ -n "${DEPTH:-}" && -d "${DEPTH:-}" ]]; then
    printf '%s\n' "$DEPTH"
    return 0
  fi
  local candidates=(
    "$BASE/depth_cache/depth_anything_v2_small_hf"
    "$BASE/caches/Haze4K/depth_anything_v2_small_hf"
    "$BASE/datasets/Haze4K/depth_anything_v2_small_hf"
    "$BASE/experiment/HAZE4K/depth_cache/depth_anything_v2_small_hf"
    "/root/autodl-tmp/workspace/Dehaze-Net/experiment/HAZE4K/depth_cache/depth_anything_v2_small_hf"
  )
  local item
  for item in "${candidates[@]}"; do
    if [[ -d "$item/train" && -d "$item/test" ]]; then
      printf '%s\n' "$item"
      return 0
    fi
  done
  return 1
}

mkdir -p "$EVID"
{
  echo "preflight_start haze4k_dta_lowgate $(date --iso-8601=seconds)"
  echo "remote_root=$REMOTE_ROOT"
  echo "python=$PY"
  echo "data=$DATA"
  echo "checkpoint=$A0"
  echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
} | tee -a "$STATUS"

if ! DEPTH_CACHE=$(find_depth_cache); then
  echo "FAILED_INFRA_MISSING_DEPTH_CACHE" | tee -a "$STATUS"
  exit 2
fi
echo "depth_cache=$DEPTH_CACHE" | tee -a "$STATUS"

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
  --depth_cache_dir "$DEPTH_CACHE" \
  --depth_split train \
  --output_json "$JSON_OUT" \
  --dta_gate_bias -7.0 \
  --dta_gate_limit 0.03 \
  --rank_weight 0.003 \
  --tv_weight 0.0003 \
  2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e
echo "preflight_done rc=$rc haze4k_dta_lowgate $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [[ "$rc" -eq 0 ]]; then
  echo "DTA_LOWGATE_PREFLIGHT_OK" | tee -a "$STATUS"
else
  echo "DTA_LOWGATE_PREFLIGHT_FAILED" | tee -a "$STATUS"
fi
exit "$rc"
