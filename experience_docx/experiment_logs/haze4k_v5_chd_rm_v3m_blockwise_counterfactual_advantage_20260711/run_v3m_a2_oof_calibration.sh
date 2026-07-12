#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v3m-blockwise-counterfactual-advantage-20260711
EVID="$REMOTE_ROOT/experience_docx/experiment_logs/haze4k_v5_chd_rm_v3m_blockwise_counterfactual_advantage_20260711"
PY=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
STATUS="$EVID/status.txt"
LOG="$EVID/run_v3m_a2_oof_calibration.log"

mkdir -p "$EVID"
echo "run_start v3m_a2 $(date --iso-8601=seconds)" | tee -a "$STATUS"
set +e
"$PY" "$REMOTE_ROOT/experience_docx/tools/chd_rm_v3m_a2_oof_calibration_audit.py" \
  --a1-block-rows "$EVID/v3m_a1_block_rows_cloud_only.csv" \
  --a1-summary "$EVID/v3m_a1_summary.json" \
  --a1-signal-summary "$EVID/v3m_a1_signal_summary.csv" \
  --a1-source-manifest "$EVID/v3m_a1_source_manifest.json" \
  --output-dir "$EVID" \
  --run-tag v3m_a2 \
  --expected-a1-block-rows-sha256 e29c8ca2f1759e4025637924e5a826cd0fdcd86cfcdea2b38fd2f8682782aa39 \
  --expected-a1-block-rows-line-count 2177351 \
  --expected-a1-summary-sha256 6387ee7819460366b7606b73aeb8e1d64ab7d80fb774a507a30968d56c7392e1 \
  --expected-a1-signal-summary-sha256 449de6183ba1104bb0618ddf0ecc8c509739abdf1a96aaea79b4127454643e69 \
  --expected-a1-source-manifest-sha256 304512574ed23142f06fc5d17d04ef1635d32e28cbd50fdd1c199c0dc726d2b1 \
  2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e
echo "run_done rc=$rc v3m_a2 $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [ "$rc" -eq 0 ]; then
  echo "V3M_A2_OOF_CALIBRATION_OK"
else
  echo "V3M_A2_OOF_CALIBRATION_FAILED"
fi
exit "$rc"
