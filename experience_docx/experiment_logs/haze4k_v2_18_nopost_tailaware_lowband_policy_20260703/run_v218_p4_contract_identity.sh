#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v2-18-nopost-tailaware-lowband-policy
EVID="$REMOTE_ROOT/experience_docx/experiment_logs/haze4k_v2_18_nopost_tailaware_lowband_policy_20260703"
PY=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
DATA=/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K
CKPT=/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl
STATUS="$EVID/status.txt"
LOG="$EVID/v218_p4_contract_identity.log"

mkdir -p "$EVID"
cd "$REMOTE_ROOT"

{
  echo "v218_p4_state=RUNNING_AUDIT"
  echo "v218_p4_start $(date --iso-8601=seconds)"
  echo "locked_test_touched=false"
} | tee -a "$STATUS"

test -x "$PY"
test -d "$DATA/train"
test -f "$CKPT"

set +e
"$PY" experience_docx/tools/nopost_lowband_v218_contract_identity.py \
  --data-dir "$DATA" \
  --checkpoint "$CKPT" \
  --out-dir "$EVID" \
  --max-images 8 \
  2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e

{
  echo "v218_p4_done rc=$rc $(date --iso-8601=seconds)"
  if [ "$rc" -eq 0 ]; then
    echo "v218_p4_state=COMPLETED_AUDIT"
    echo "V218_P4_OK"
  else
    echo "v218_p4_state=FAILED_COMMAND"
    echo "V218_P4_FAILED"
  fi
} | tee -a "$STATUS"

exit "$rc"
