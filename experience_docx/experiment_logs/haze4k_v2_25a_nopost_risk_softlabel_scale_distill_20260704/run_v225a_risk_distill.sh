#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v2-25a-nopost-risk-softlabel-scale-distill
EVID=$REMOTE_ROOT/experience_docx/experiment_logs/haze4k_v2_25a_nopost_risk_softlabel_scale_distill_20260704
PY=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
DATA=/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K
CKPT=/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl
SPLIT=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v2-17-nopost-lowband-alignment-tail-audit/experience_docx/experiment_logs/haze4k_v2_16_nopost_wavelet_lowband_decoder_20260703/v216_t1_per_image_band_deltas.csv
V221=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v2-21-nopost-safety-calibrated-lowband-replay/experience_docx/experiment_logs/haze4k_v2_21_nopost_safety_calibrated_lowband_replay_20260704/v221_p1_safety_gated_replay_metrics.csv
STATUS=$EVID/status.txt
LOG=$EVID/v225a_risk_distill.log

mkdir -p "$EVID"
cd "$REMOTE_ROOT"
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_ID=${GPU_ID:-$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F, '{gsub(/ /,"",$1); gsub(/ /,"",$2); print $2" "$1}' | sort -n | awk 'NR==1{print $2}')}
  export CUDA_VISIBLE_DEVICES=$GPU_ID
else
  GPU_ID=cpu
fi
echo "v225a_script_start $(date --iso-8601=seconds) gpu=$GPU_ID" | tee -a "$STATUS"
set +e
"$PY" experience_docx/tools/nopost_lowband_v225a_risk_softlabel_distill.py \
  --data-dir "$DATA" \
  --checkpoint "$CKPT" \
  --split-csv "$SPLIT" \
  --v221-metrics-csv "$V221" \
  --out-dir "$EVID" \
  --folds 0,1,2 \
  --train-samples-per-fold 384 \
  --eval-samples-per-fold 160 \
  --epochs 4 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e
echo "v225a_script_done rc=$rc $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [ "$rc" -eq 0 ]; then echo V225A_SCRIPT_OK; else echo V225A_SCRIPT_FAILED; fi
exit "$rc"
