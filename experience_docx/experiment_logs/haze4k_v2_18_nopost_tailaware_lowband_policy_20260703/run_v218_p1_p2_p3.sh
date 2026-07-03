#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v2-18-nopost-tailaware-lowband-policy
EVID="$REMOTE_ROOT/experience_docx/experiment_logs/haze4k_v2_18_nopost_tailaware_lowband_policy_20260703"
V217=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v2-17-nopost-lowband-alignment-tail-audit/experience_docx/experiment_logs/haze4k_v2_17_nopost_lowband_alignment_tail_audit_20260703
V216=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v2-17-nopost-lowband-alignment-tail-audit/experience_docx/experiment_logs/haze4k_v2_16_nopost_wavelet_lowband_decoder_20260703
PY=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
DATA=/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K
CKPT=/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl
STATUS="$EVID/status.txt"
LOG="$EVID/v218_p1_p2_p3_policy_objective_audit.log"

mkdir -p "$EVID"
cd "$REMOTE_ROOT"

{
  echo "v218_p1_p2_p3_state=RUNNING_AUDIT"
  echo "v218_p1_p2_p3_start $(date --iso-8601=seconds)"
  echo "locked_test_touched=false"
} | tee -a "$STATUS"

test -x "$PY"
test -d "$DATA/train"
test -f "$CKPT"
test -f "$V216/v216_t1_per_image_band_deltas.csv"
test -f "$V217/v217_r1_tail_case_manifest.csv"
test -f "$V217/v217_r1_action_norm_stats.csv"
test -f "$V217/v217_r3_per_image_loss_terms.csv"

set +e
"$PY" experience_docx/tools/nopost_lowband_v218_policy_objective_audit.py \
  --data-dir "$DATA" \
  --checkpoint "$CKPT" \
  --split-csv "$V216/v216_t1_per_image_band_deltas.csv" \
  --v216-dir "$V216" \
  --v217-dir "$V217" \
  --out-dir "$EVID" \
  --steps-o1 25 \
  --lr 0.08 \
  --delta-scale 0.50 \
  --mlp-epochs 320 \
  --mlp-hidden 64 \
  --ridge-lambda 1.0 \
  --seed 218 \
  --print-freq 25 \
  2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e

{
  echo "v218_p1_p2_p3_done rc=$rc $(date --iso-8601=seconds)"
  if [ "$rc" -eq 0 ]; then
    echo "v218_p1_p2_p3_state=COMPLETED_AUDIT"
    echo "V218_P1_P2_P3_OK"
  else
    echo "v218_p1_p2_p3_state=FAILED_COMMAND"
    echo "V218_P1_P2_P3_FAILED"
  fi
} | tee -a "$STATUS"

exit "$rc"
