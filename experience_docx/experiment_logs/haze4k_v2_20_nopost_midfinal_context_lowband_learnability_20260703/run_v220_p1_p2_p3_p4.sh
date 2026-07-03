#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v2-20-nopost-midfinal-context-lowband-learnability
EVID="$REMOTE_ROOT/experience_docx/experiment_logs/haze4k_v2_20_nopost_midfinal_context_lowband_learnability_20260703"
V216=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v2-17-nopost-lowband-alignment-tail-audit/experience_docx/experiment_logs/haze4k_v2_16_nopost_wavelet_lowband_decoder_20260703
PY=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
DATA=/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K
CKPT=/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl
STATUS="$EVID/status.txt"
LOG="$EVID/v220_p1_p2_p3_p4_o3_context_learnability_audit.log"

mkdir -p "$EVID"
cd "$REMOTE_ROOT"

{
  echo "v220_p1_p2_p3_p4_state=RUNNING_AUDIT"
  echo "v220_p1_p2_p3_p4_start $(date --iso-8601=seconds)"
  echo "branch=$(git branch --show-current)"
  echo "commit=$(git rev-parse --short HEAD)"
  echo "remote_root=$REMOTE_ROOT"
  echo "python=$PY"
  echo "data=$DATA"
  echo "checkpoint=$CKPT"
  echo "split_csv=$V216/v216_t1_per_image_band_deltas.csv"
  echo "locked_test_touched=false"
  echo "training_launched=false"
} | tee -a "$STATUS"

test -x "$PY"
test -d "$DATA/train"
test -f "$CKPT"
test -f "$V216/v216_t1_per_image_band_deltas.csv"
test -f "$EVID/v220_p0_decision.md"
grep -q 'P0_PASS_MIDFINAL_CONTEXT_CONTRACT_IDENTITY_SOURCE_CLEAN' "$EVID/v220_p0_decision.md"

set +e
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}" "$PY" experience_docx/tools/nopost_lowband_v220_o3_context_learnability_audit.py \
  --data-dir "$DATA" \
  --checkpoint "$CKPT" \
  --split-csv "$V216/v216_t1_per_image_band_deltas.csv" \
  --out-dir "$EVID" \
  --steps-o2 25 \
  --steps-o3 18 \
  --lr 0.08 \
  --delta-scale 0.50 \
  --final-grid 16 \
  --mid-grid 8 \
  --cnn-hidden 64 \
  --cnn-epochs 180 \
  --shuffle-epochs 100 \
  --classifier-epochs 400 \
  --batch-size 64 \
  --ridge-lambda 1.0 \
  --seed 220 \
  --print-freq 25 \
  2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e

state=FAILED_COMMAND
marker=V220_P1_P2_P3_P4_FAILED
if [ "$rc" -eq 0 ]; then
  if grep -q 'V220_P1B_PASS_REVIEW_N3_MICROFIT_ROUTE_CARD_NO_TRAINING_LAUNCHED' "$EVID/v220_p1_p2_p3_p4_closeout.json"; then
    state=COMPLETED_GATE_PASS
    marker=V220_P1B_GATE_PASS_REVIEW_ONLY
  else
    state=COMPLETED_GATE_FAIL
    marker=V220_NORMAL_GATE_PAUSE_NO_TRAINING
  fi
fi

{
  echo "v220_p1_p2_p3_p4_done rc=$rc $(date --iso-8601=seconds)"
  echo "v220_p1_p2_p3_p4_state=$state"
  echo "$marker"
} | tee -a "$STATUS"

exit "$rc"
