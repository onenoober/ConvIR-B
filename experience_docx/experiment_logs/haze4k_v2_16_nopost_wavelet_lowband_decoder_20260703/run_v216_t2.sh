#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v2-16-nopost-wavelet-lowband-decoder
EVID="$REMOTE_ROOT/experience_docx/experiment_logs/haze4k_v2_16_nopost_wavelet_lowband_decoder_20260703"
PY=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
DATA=/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K
CKPT=/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl
STATUS="$EVID/status.txt"
LOG="$EVID/v216_t2_contract_identity.log"

mkdir -p "$EVID"
cd "$REMOTE_ROOT"

{
  echo "t2_state=RUNNING_AUDIT"
  echo "locked_test_touched=false"
  echo "training_launched=false"
  echo "t2_run_start $(date --iso-8601=seconds)"
} | tee -a "$STATUS"

test -x "$PY"
test -d "$DATA/train"
test -f "$CKPT"
test -f "$EVID/v216_t1_decision.md"
grep -q 'T1_LOWBAND_HEADROOM_PASS_ALLOW_T2' "$EVID/v216_t1_decision.md"

set +e
"$PY" experience_docx/tools/nopost_wldb_t2_contract_identity.py \
  --data-dir "$DATA" \
  --checkpoint "$CKPT" \
  --out-dir "$EVID" \
  --max-images 32 \
  2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e

{
  echo "t2_run_done rc=$rc $(date --iso-8601=seconds)"
  if [ "$rc" -eq 0 ]; then
    echo "t2_state=COMPLETED_T2_AUDIT"
    echo "V216_T2_RUN_OK"
  else
    echo "t2_state=FAILED_COMMAND_OR_AUDIT"
    echo "V216_T2_RUN_FAILED"
  fi
} | tee -a "$STATUS"

exit "$rc"
