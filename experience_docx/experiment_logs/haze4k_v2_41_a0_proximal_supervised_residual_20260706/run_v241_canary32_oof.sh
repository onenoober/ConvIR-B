#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-haze4k-v2-41-a0-proximal-supervised-residual
EVID=$REMOTE_ROOT/experience_docx/experiment_logs/haze4k_v2_41_a0_proximal_supervised_residual_20260706
PY=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
DATA=/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K
CKPT=/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl
RUN_ID=v241_canary32_oof
OUT=$REMOTE_ROOT/Dehazing/ITS/results/ConvIR-Haze4K-v2.41-a0prox-canary32-oof-20260706
STATUS=$EVID/status.txt
LOG_DIR=$EVID/runtime_logs
LOG=$LOG_DIR/${RUN_ID}.log

mkdir -p "$EVID" "$LOG_DIR" "$OUT"
cd "$REMOTE_ROOT"

echo "${RUN_ID}_start $(date --iso-8601=seconds) branch=$(git branch --show-current) head=$(git rev-parse --short HEAD) locked=blocked canary80=blocked" | tee -a "$STATUS"

set +e
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1} "$PY" \
  experience_docx/tools/run_haze4k_v241_canary32_oof.py \
  --repo-root "$REMOTE_ROOT" \
  --data-dir "$DATA" \
  --checkpoint "$CKPT" \
  --evidence-root "$EVID" \
  --output-dir "$OUT" \
  --seed 24132 \
  --folds 5 \
  --train-size 32 \
  --val-size 32 \
  --epochs 10 \
  --batch-size 4 \
  --crop-size 256 \
  --learning-rate 4e-4 \
  --beta 0.05 \
  --lambda-anchor 0.2 \
  --lambda-hinge 2.0 \
  --lambda-cvar 1.0 \
  --hinge-margin 0.0 \
  --cvar-frac 0.2 \
  --grad-clip-norm 0.001 \
  --gate-mean-delta 0.15 \
  --gate-hard-delta 0.30 \
  --gate-easy-delta 0.00 \
  --gate-p05-delta -0.01 \
  --gate-cvar5-delta -0.02 \
  --gate-min-fold-pass 4 \
  --gate-easy-hard-energy-ratio 0.50 \
  2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e

echo "${RUN_ID}_done rc=$rc $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [ "$rc" -eq 0 ]; then
  echo "${RUN_ID}_OK" | tee -a "$STATUS"
else
  echo "${RUN_ID}_FAILED" | tee -a "$STATUS"
fi
exit "$rc"
