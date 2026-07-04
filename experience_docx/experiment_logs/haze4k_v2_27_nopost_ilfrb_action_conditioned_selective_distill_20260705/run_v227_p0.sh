#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v2-27-nopost-ilfrb-action-conditioned-selective-distill
EVID=$REMOTE_ROOT/experience_docx/experiment_logs/haze4k_v2_27_nopost_ilfrb_action_conditioned_selective_distill_20260705
PY=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
DATA=/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K
CKPT=/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl
SPLIT=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v2-17-nopost-lowband-alignment-tail-audit/experience_docx/experiment_logs/haze4k_v2_16_nopost_wavelet_lowband_decoder_20260703/v216_t1_per_image_band_deltas.csv
STATUS=$EVID/status.txt
LOG=$EVID/v227_p0_contract_identity.log

mkdir -p "$EVID"
cd "$REMOTE_ROOT"

if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_ID=${GPU_ID:-$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F, '{gsub(/ /,"",$1); gsub(/ /,"",$2); print $2" "$1}' | sort -n | awk 'NR==1{print $2}')}
  export CUDA_VISIBLE_DEVICES=$GPU_ID
else
  GPU_ID=cpu
fi

{
  echo "v227_p0_state=PREFLIGHT_RUNNING"
  echo "v227_p0_start $(date --iso-8601=seconds) gpu=$GPU_ID"
  echo "branch=$(git branch --show-current)"
  echo "commit=$(git rev-parse --short HEAD)"
  echo "remote_root=$REMOTE_ROOT"
  echo "python=$PY"
  echo "data=$DATA"
  echo "checkpoint=$CKPT"
  echo "split_csv=$SPLIT"
  echo "locked_test_touched=false"
  echo "training_launched=false"
} | tee -a "$STATUS"

test -x "$PY"
test -d "$DATA/train"
test -f "$CKPT"
test -f "$SPLIT"

set +e
"$PY" experience_docx/tools/nopost_v227_ilfrb_acs_diagnostics.py \
  --phases p0 \
  --data-dir "$DATA" \
  --checkpoint "$CKPT" \
  --split-csv "$SPLIT" \
  --out-dir "$EVID" \
  --max-images 20 \
  --p0-images 8 \
  2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e

state=PREFLIGHT_FAILED_ENGINEERING
marker=V227_P0_FAILED
if [ "$rc" -eq 0 ] && grep -q 'P0_PASS_ILFRB_ACS_CONTRACT_IDENTITY_SOURCE_CLEAN' "$EVID/v227_p0_decision.md"; then
  state=COMPLETED_GATE_PASS
  marker=V227_P0_OK
fi

{
  echo "v227_p0_done rc=$rc $(date --iso-8601=seconds)"
  echo "v227_p0_state=$state"
  echo "$marker"
} | tee -a "$STATUS"

exit "$rc"
