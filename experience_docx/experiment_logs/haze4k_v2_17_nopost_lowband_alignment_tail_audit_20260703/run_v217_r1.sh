#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v2-17-nopost-lowband-alignment-tail-audit
EVID="$REMOTE_ROOT/experience_docx/experiment_logs/haze4k_v2_17_nopost_lowband_alignment_tail_audit_20260703"
V216="$REMOTE_ROOT/experience_docx/experiment_logs/haze4k_v2_16_nopost_wavelet_lowband_decoder_20260703"
PY=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
DATA=/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K
CKPT=/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl
STATUS="$EVID/status.txt"
LOG="$EVID/v217_r1_postmortem.log"

mkdir -p "$EVID"
cd "$REMOTE_ROOT"

{
  echo "v217_r1_state=RUNNING_AUDIT"
  echo "v217_r1_start $(date --iso-8601=seconds)"
  echo "locked_test_touched=false"
} | tee -a "$STATUS"

test -x "$PY"
test -d "$DATA/train"
test -f "$CKPT"
test -f "$V216/v216_t1_per_image_band_deltas.csv"
test -f "$V216/wldb_a_seed3407/v216_wldb_a_eval_model_5_per_image.csv"
test -f "$V216/wldb_a_seed3407/checkpoints/model_5.pkl"

set +e
"$PY" experience_docx/tools/nopost_lowband_v217_r1_postmortem.py \
  --v216-dir "$V216" \
  --out-dir "$EVID" \
  --data-dir "$DATA" \
  --official-checkpoint "$CKPT" \
  --fold 0 \
  2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e

{
  echo "v217_r1_done rc=$rc $(date --iso-8601=seconds)"
  if [ "$rc" -eq 0 ]; then
    echo "v217_r1_state=COMPLETED_AUDIT"
    echo "V217_R1_OK"
  else
    echo "v217_r1_state=FAILED_COMMAND"
    echo "V217_R1_FAILED"
  fi
} | tee -a "$STATUS"

exit "$rc"
