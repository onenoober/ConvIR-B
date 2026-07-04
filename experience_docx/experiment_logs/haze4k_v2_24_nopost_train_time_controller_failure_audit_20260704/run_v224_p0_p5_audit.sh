#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v2-24-nopost-train-time-controller-failure-audit
EVID=$REMOTE_ROOT/experience_docx/experiment_logs/haze4k_v2_24_nopost_train_time_controller_failure_audit_20260704
PY=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
DATA=/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K
CKPT=/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl
V223_REPO=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v2-23-nopost-oof-gated-lowband-train
V223_EVID=$V223_REPO/experience_docx/experiment_logs/haze4k_v2_23_nopost_oof_gated_lowband_train_20260704
SPLIT=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v2-17-nopost-lowband-alignment-tail-audit/experience_docx/experiment_logs/haze4k_v2_16_nopost_wavelet_lowband_decoder_20260703/v216_t1_per_image_band_deltas.csv
V221=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v2-21-nopost-safety-calibrated-lowband-replay/experience_docx/experiment_logs/haze4k_v2_21_nopost_safety_calibrated_lowband_replay_20260704/v221_p1_safety_gated_replay_metrics.csv
V221_FACTOR=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v2-21-nopost-safety-calibrated-lowband-replay/experience_docx/experiment_logs/haze4k_v2_21_nopost_safety_calibrated_lowband_replay_20260704/v221_p1_factorial_action_gate_audit.csv
STATUS=$EVID/status.txt
LOG=$EVID/v224_audit.log

mkdir -p "$EVID"
cd "$REMOTE_ROOT"

if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_ID=${GPU_ID:-$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F, '{gsub(/ /,"",$1); gsub(/ /,"",$2); print $2" "$1}' | sort -n | awk 'NR==1{print $2}')}
  export CUDA_VISIBLE_DEVICES=$GPU_ID
else
  GPU_ID=cpu
fi

echo "v224_audit_script_start $(date --iso-8601=seconds) gpu=$GPU_ID" | tee -a "$STATUS"
set +e
"$PY" experience_docx/tools/nopost_lowband_v224_controller_failure_audit.py \
  --data-dir "$DATA" \
  --checkpoint "$CKPT" \
  --split-csv "$SPLIT" \
  --v221-metrics-csv "$V221" \
  --v221-factorial-csv "$V221_FACTOR" \
  --v223-repo "$V223_REPO" \
  --v223-evidence "$V223_EVID" \
  --out-dir "$EVID" \
  --folds 0,1,2 \
  --train-samples-per-fold 384 \
  --eval-samples-per-fold 160 \
  --p4-max-samples 36 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e
echo "v224_audit_script_done rc=$rc $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [ "$rc" -eq 0 ]; then
  echo V224_AUDIT_SCRIPT_OK
else
  echo V224_AUDIT_SCRIPT_FAILED
fi
exit "$rc"
