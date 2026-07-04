#!/usr/bin/env bash
set -euo pipefail

BASE=/sda/home/wangyuxin/ConvIR-B
WORK=$BASE/repos/ConvIR-B-v2-22-nopost-gated-lowband-n3-microfit
EVID=$WORK/experience_docx/experiment_logs/haze4k_v2_22_nopost_gated_lowband_n3_microfit_20260704
PY=$BASE/envs/convir-cu121/bin/python
DATA=$BASE/datasets/Haze4K/Haze4K
A0=$BASE/checkpoints/official/Haze4K/haze4k-base.pkl
V216=$BASE/repos/ConvIR-B-v2-17-nopost-lowband-alignment-tail-audit/experience_docx/experiment_logs/haze4k_v2_16_nopost_wavelet_lowband_decoder_20260703/v216_t1_per_image_band_deltas.csv
V221=$BASE/repos/ConvIR-B-v2-21-nopost-safety-calibrated-lowband-replay/experience_docx/experiment_logs/haze4k_v2_21_nopost_safety_calibrated_lowband_replay_20260704/v221_p1_safety_gated_replay_metrics.csv
STATUS=$EVID/status.txt
LOG=$EVID/v222_n3_microfit.log

mkdir -p "$EVID"
{
  echo "n3_run_start $(date --iso-8601=seconds)"
  echo "work=$WORK"
  echo "branch=$(cd "$WORK" && git branch --show-current)"
  echo "commit=$(cd "$WORK" && git rev-parse --short HEAD)"
  echo "python=$PY"
  echo "data=$DATA"
  echo "checkpoint=$A0"
  echo "split_csv=$V216"
  echo "v221_metrics_csv=$V221"
  echo "locked_test_touched=false"
} | tee -a "$STATUS"

cd "$WORK"
set +e
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
PYTHONUNBUFFERED=1 "$PY" experience_docx/tools/nopost_lowband_v222_n3_microfit.py \
  --data-dir "$DATA" \
  --checkpoint "$A0" \
  --split-csv "$V216" \
  --v221-metrics-csv "$V221" \
  --out-dir "$EVID" \
  --stages microfit16,microfit64,microfit256 \
  --crop-size 256 \
  --eval-samples 128 \
  --epochs16 4 \
  --epochs64 5 \
  --epochs256 6 \
  --learning-rate 0.0001 \
  --weight-decay 0.0001 \
  --grad-clip-norm 0.001 \
  --risk-loss-weight 0.05 \
  --gate-mean-weight 0.0005 \
  --action-l1-weight 0.0001 \
  --seed 222 \
  2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e
echo "n3_run_done rc=$rc $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [[ "$rc" -eq 0 ]]; then
  echo "V222_N3_MICROFIT_OK" | tee -a "$STATUS"
else
  echo "V222_N3_MICROFIT_FAILED" | tee -a "$STATUS"
fi
exit "$rc"
