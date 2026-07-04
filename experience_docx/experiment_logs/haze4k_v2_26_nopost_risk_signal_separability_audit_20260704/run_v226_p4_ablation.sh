#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v2-26-nopost-risk-signal-separability-audit
EVID=$REMOTE_ROOT/experience_docx/experiment_logs/haze4k_v2_26_nopost_risk_signal_separability_audit_20260704
PY=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
DATA=/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K
CKPT=/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl
SPLIT=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v2-17-nopost-lowband-alignment-tail-audit/experience_docx/experiment_logs/haze4k_v2_16_nopost_wavelet_lowband_decoder_20260703/v216_t1_per_image_band_deltas.csv
V221=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v2-21-nopost-safety-calibrated-lowband-replay/experience_docx/experiment_logs/haze4k_v2_21_nopost_safety_calibrated_lowband_replay_20260704/v221_p1_safety_gated_replay_metrics.csv
V225A=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v2-25a-nopost-risk-softlabel-scale-distill/experience_docx/experiment_logs/haze4k_v2_25a_nopost_risk_softlabel_scale_distill_20260704
STATUS=$EVID/status.txt
LOG=$EVID/v226_p4_ablation.log

mkdir -p "$EVID"
cd "$REMOTE_ROOT"
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_ID=${GPU_ID:-$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F, '{gsub(/ /,"",$1); gsub(/ /,"",$2); print $2" "$1}' | sort -n | awk 'NR==1{print $2}')}
  export CUDA_VISIBLE_DEVICES=$GPU_ID
else
  GPU_ID=cpu
fi
echo "v226_p4_script_start $(date --iso-8601=seconds) gpu=$GPU_ID" | tee -a "$STATUS"
set +e
"$PY" experience_docx/tools/nopost_lowband_v226_risk_signal_audit.py \
  --phases p4 \
  --data-dir "$DATA" \
  --checkpoint "$CKPT" \
  --split-csv "$SPLIT" \
  --v221-metrics-csv "$V221" \
  --v225a-evid "$V225A" \
  --out-dir "$EVID" \
  --p4-train-samples 192 \
  --p4-eval-samples 96 \
  --p4-epochs 3 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e
echo "v226_p4_script_done rc=$rc $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [ "$rc" -eq 0 ]; then echo V226_P4_SCRIPT_OK; else echo V226_P4_SCRIPT_FAILED; fi
exit "$rc"
