#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v2-15-nopost-spatial-internal-risk-audit
EVID="$REMOTE_ROOT/experience_docx/experiment_logs/haze4k_v2_15_nopost_spatial_internal_risk_audit_20260703"
PY=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
DATA_DIR=/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K
CHECKPOINT=/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl
V213_TABLE=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v2-13-nopost-feature-gated-adapter/experience_docx/experiment_logs/haze4k_v2_13_nopost_feature_gated_adapter_20260702/v213_n1_feature_rows_cloud_only.csv
V214_PRED=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v2-14-nopost-runtime-evidence-audit/experience_docx/experiment_logs/haze4k_v2_14_nopost_runtime_evidence_audit_20260703/v214_n1r_oof_predictions.csv
STATUS="$EVID/status.txt"
S2_LOG="$EVID/v215_s2_s3_feature_build.log"
S4_LOG="$EVID/v215_s1_s4_oof_probe.log"

mkdir -p "$EVID"
{
  printf 'run_start v215_n1s %s\n' "$(date --iso-8601=seconds)"
  printf 'remote_root=%s\n' "$REMOTE_ROOT"
  printf 'python=%s\n' "$PY"
  printf 'locked_test_touched=false\n'
  printf 'training_launched=false\n'
} | tee -a "$STATUS"

cd "$REMOTE_ROOT"
{
  printf 'branch='
  git branch --show-current
  printf 'head='
  git rev-parse --short HEAD
  printf 'status_short_begin\n'
  git status --short
  printf 'status_short_end\n'
  "$PY" --version
  test -f "$V213_TABLE" && printf 'v213_feature_table=FOUND\n' || printf 'v213_feature_table=MISSING\n'
  test -f "$V214_PRED" && printf 'v214_predictions=FOUND\n' || printf 'v214_predictions=MISSING\n'
} | tee -a "$STATUS"

printf 's2_s3_feature_build_start %s\n' "$(date --iso-8601=seconds)" | tee -a "$STATUS"
set +e
"$PY" experience_docx/tools/build_nopost_spatial_risk_features.py \
  --feature-table "$V213_TABLE" \
  --data-dir "$DATA_DIR" \
  --checkpoint "$CHECKPOINT" \
  --out-dir "$EVID" \
  --print-freq 50 \
  2>&1 | tee "$S2_LOG"
rc_s2=${PIPESTATUS[0]}
set -e
printf 's2_s3_feature_build_done rc=%s %s\n' "$rc_s2" "$(date --iso-8601=seconds)" | tee -a "$STATUS"
if [ "$rc_s2" -ne 0 ]; then
  printf 'V215_S2_S3_FAILED\n' | tee -a "$STATUS"
  exit "$rc_s2"
fi

printf 's1_s4_oof_probe_start %s\n' "$(date --iso-8601=seconds)" | tee -a "$STATUS"
set +e
"$PY" experience_docx/tools/oof_probe_nopost_spatial_risk_v215.py \
  --v214-feature-table "$V213_TABLE" \
  --v214-predictions "$V214_PRED" \
  --spatial-table "$EVID/v215_s2_spatial_feature_rows.csv" \
  --fam-table "$EVID/v215_s3_fam_response_features.csv" \
  --skip-table "$EVID/v215_s3_skip_merge_disagreement.csv" \
  --jitter-table "$EVID/v215_s3_feature_jitter_consistency.csv" \
  --out-dir "$EVID" \
  --seeds 3407,3411,2026 \
  --bootstrap-iterations 1000 \
  2>&1 | tee "$S4_LOG"
rc_s4=${PIPESTATUS[0]}
set -e
printf 's1_s4_oof_probe_done rc=%s %s\n' "$rc_s4" "$(date --iso-8601=seconds)" | tee -a "$STATUS"
if [ "$rc_s4" -eq 0 ]; then
  printf 'V215_N1S_SCRIPT_OK\n' | tee -a "$STATUS"
else
  printf 'V215_N1S_SCRIPT_FAILED\n' | tee -a "$STATUS"
fi
printf 'run_done rc=%s v215_n1s %s\n' "$rc_s4" "$(date --iso-8601=seconds)" | tee -a "$STATUS"
exit "$rc_s4"
