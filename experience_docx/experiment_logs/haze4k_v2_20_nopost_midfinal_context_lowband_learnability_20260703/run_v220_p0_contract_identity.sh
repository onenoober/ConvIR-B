#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v2-20-nopost-midfinal-context-lowband-learnability
EVID="$REMOTE_ROOT/experience_docx/experiment_logs/haze4k_v2_20_nopost_midfinal_context_lowband_learnability_20260703"
PY=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
DATA=/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K
CKPT=/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl
STATUS="$EVID/status.txt"
LOG="$EVID/v220_p0_contract_identity.log"

mkdir -p "$EVID"
cd "$REMOTE_ROOT"

{
  echo "v220_p0_state=PREFLIGHT_RUNNING"
  echo "v220_p0_start $(date --iso-8601=seconds)"
  echo "branch=$(git branch --show-current)"
  echo "commit=$(git rev-parse --short HEAD)"
  echo "remote_root=$REMOTE_ROOT"
  echo "python=$PY"
  echo "data=$DATA"
  echo "checkpoint=$CKPT"
  echo "locked_test_touched=false"
  echo "training_launched=false"
} | tee -a "$STATUS"

test -x "$PY"
test -d "$DATA/train"
test -f "$CKPT"

set +e
"$PY" experience_docx/tools/nopost_lowband_v220_contract_identity.py \
  --data-dir "$DATA" \
  --checkpoint "$CKPT" \
  --out-dir "$EVID" \
  --max-images 8 \
  2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e

{
  echo "v220_p0_done rc=$rc $(date --iso-8601=seconds)"
  if [ "$rc" -eq 0 ]; then
    echo "v220_p0_state=COMPLETED_GATE_PASS"
    echo "V220_P0_OK"
  else
    echo "v220_p0_state=PREFLIGHT_FAILED_ENGINEERING"
    echo "V220_P0_FAILED"
  fi
} | tee -a "$STATUS"

exit "$rc"
