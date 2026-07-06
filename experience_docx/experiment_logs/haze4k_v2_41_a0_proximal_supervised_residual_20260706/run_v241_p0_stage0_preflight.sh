#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT="/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-haze4k-v2-41-a0-proximal-supervised-residual"
EVID="$REMOTE_ROOT/experience_docx/experiment_logs/haze4k_v2_41_a0_proximal_supervised_residual_20260706"
PY="/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python"
CKPT="/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl"
STATUS="$EVID/status.txt"
LOG="$EVID/runtime_logs/v241_p0_stage0_preflight.log"

mkdir -p "$EVID/runtime_logs"
cd "$REMOTE_ROOT"

echo "v241_p0_preflight_start $(date --iso-8601=seconds) branch=$(git branch --show-current) head=$(git rev-parse --short HEAD) locked=untouched canary=blocked" | tee -a "$STATUS"
"$PY" -m py_compile Dehazing/ITS/models/A0ProxResidualConvIR.py Dehazing/ITS/main.py experience_docx/tools/preflight_haze4k_v241_a0prox.py
if [ ! -f "$CKPT" ]; then
  echo "v241_p0_blocked_missing_checkpoint $CKPT $(date --iso-8601=seconds)" | tee -a "$STATUS"
  echo V241_P0_STAGE0_PREFLIGHT_FAILED
  exit 2
fi
set +e
"$PY" -u experience_docx/tools/preflight_haze4k_v241_a0prox.py \
  --out-dir "$EVID" \
  --checkpoint "$CKPT" \
  --synthetic-size 256 \
  --beta 0.05 \
  2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e
echo "v241_p0_preflight_done rc=$rc $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [ "$rc" -eq 0 ]; then echo V241_P0_STAGE0_PREFLIGHT_OK; else echo V241_P0_STAGE0_PREFLIGHT_FAILED; fi
exit "$rc"
