#!/usr/bin/env bash
set -euo pipefail

BASE=/sda/home/wangyuxin/ConvIR-B
WORK=$BASE/repos/ConvIR-B-v2-23-nopost-oof-gated-lowband-train
EVID=$WORK/experience_docx/experiment_logs/haze4k_v2_23_nopost_oof_gated_lowband_train_20260704
PY=$BASE/envs/convir-cu121/bin/python
DATA=$BASE/datasets/Haze4K/Haze4K
A0=$BASE/checkpoints/official/Haze4K/haze4k-base.pkl
SPLIT=$BASE/repos/ConvIR-B-v2-17-nopost-lowband-alignment-tail-audit/experience_docx/experiment_logs/haze4k_v2_16_nopost_wavelet_lowband_decoder_20260703/v216_t1_per_image_band_deltas.csv
V221=$BASE/repos/ConvIR-B-v2-21-nopost-safety-calibrated-lowband-replay/experience_docx/experiment_logs/haze4k_v2_21_nopost_safety_calibrated_lowband_replay_20260704/v221_p1_safety_gated_replay_metrics.csv
STATUS=$EVID/status.txt
LOG=$EVID/v223_oof_screen.log

mkdir -p "$EVID"
{
  echo "oof_screen_start $(date --iso-8601=seconds)"
  echo "work=$WORK"
  echo "branch=$(cd "$WORK" && git branch --show-current)"
  echo "commit=$(cd "$WORK" && git rev-parse --short HEAD)"
  echo "python=$PY"
  echo "data=$DATA"
  echo "checkpoint=$A0"
  echo "split_csv=$SPLIT"
  echo "v221_metrics_csv=$V221"
  echo "locked_test_touched=false"
} | tee -a "$STATUS"

cd "$WORK"
set +e
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
PYTHONUNBUFFERED=1 "$PY" experience_docx/tools/nopost_lowband_v223_oof_train.py \
  --data-dir "$DATA" \
  --checkpoint "$A0" \
  --split-csv "$SPLIT" \
  --v221-metrics-csv "$V221" \
  --out-dir "$EVID" \
  --folds 0,1,2 \
  --train-samples-per-fold 384 \
  --eval-samples-per-fold 160 \
  --epochs 2 \
  --crop-size 256 \
  --learning-rate 0.00008 \
  --weight-decay 0.0001 \
  --grad-clip-norm 0.001 \
  --risk-loss-weight 0.05 \
  --gate-mean-weight 0.0005 \
  --action-l1-weight 0.0001 \
  --seed 223 \
  2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e
echo "oof_screen_done rc=$rc $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [[ "$rc" -eq 0 ]]; then
  echo "V223_OOF_SCREEN_DONE" | tee -a "$STATUS"
else
  echo "V223_OOF_SCREEN_FAILED" | tee -a "$STATUS"
fi
exit "$rc"
