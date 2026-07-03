#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v2-16-nopost-wavelet-lowband-decoder
EVID="$REMOTE_ROOT/experience_docx/experiment_logs/haze4k_v2_16_nopost_wavelet_lowband_decoder_20260703"
PY=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
DATA=/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K
CKPT=/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl
V213_TABLE=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v2-13-nopost-feature-gated-adapter/experience_docx/experiment_logs/haze4k_v2_13_nopost_feature_gated_adapter_20260702/v213_n1_feature_rows_cloud_only.csv
V215_EVID=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v2-15-nopost-spatial-internal-risk-audit/experience_docx/experiment_logs/haze4k_v2_15_nopost_spatial_internal_risk_audit_20260703
STATUS="$EVID/status.txt"
LOG="$EVID/v216_t0_t1_audit.log"

mkdir -p "$EVID"
cd "$REMOTE_ROOT"

{
  echo "route=haze4k-v2-16-nopost-wavelet-lowband-decoder"
  echo "state=RUNNING_AUDIT"
  echo "locked_test_touched=false"
  echo "training_launched=false"
  echo "branch=$(git branch --show-current)"
  echo "commit=$(git rev-parse --short HEAD)"
  echo "python=$PY"
  echo "data=$DATA"
  echo "checkpoint=$CKPT"
  echo "run_start $(date --iso-8601=seconds)"
} | tee "$STATUS"

test -x "$PY"
test -d "$DATA/train"
test -f "$CKPT"
test -f "$V213_TABLE"
test -f "$V215_EVID/v215_s4_oof_predictions.csv"
test -f "$V215_EVID/v215_s1_top100_hazy_vs_all_runtime.csv"
test -f "$V215_EVID/v215_s1_lost_severe_cases.csv"
test -f "$V215_EVID/v215_s1_gained_false_positive_cases.csv"

set +e
"$PY" experience_docx/tools/nopost_wldb_t0_t1_audit.py \
  --data-dir "$DATA" \
  --checkpoint "$CKPT" \
  --feature-table "$V213_TABLE" \
  --v215-predictions "$V215_EVID/v215_s4_oof_predictions.csv" \
  --v215-top100 "$V215_EVID/v215_s1_top100_hazy_vs_all_runtime.csv" \
  --v215-lost-severe "$V215_EVID/v215_s1_lost_severe_cases.csv" \
  --v215-gained-fp "$V215_EVID/v215_s1_gained_false_positive_cases.csv" \
  --out-dir "$EVID" \
  --print-freq 50 \
  2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e

{
  echo "run_done rc=$rc $(date --iso-8601=seconds)"
  if [ "$rc" -eq 0 ]; then
    echo "state=COMPLETED_T0_T1_AUDIT"
    echo "V216_T0_T1_RUN_OK"
  else
    echo "state=FAILED_COMMAND_OR_AUDIT"
    echo "V216_T0_T1_RUN_FAILED"
  fi
} | tee -a "$STATUS"

exit "$rc"
