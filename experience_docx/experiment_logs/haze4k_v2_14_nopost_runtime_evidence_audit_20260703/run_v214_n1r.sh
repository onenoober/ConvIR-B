#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v2-14-nopost-runtime-evidence-audit
PREV_ROOT=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v2-13-nopost-feature-gated-adapter
EVID="$REMOTE_ROOT/experience_docx/experiment_logs/haze4k_v2_14_nopost_runtime_evidence_audit_20260703"
PY=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
DATA_DIR=/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K
CHECKPOINT=/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl
SPLIT_MANIFEST=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v24-c12-wd0375-distill/experience_docx/experiment_logs/haze4k_v2_4_c12_wd0375_distill_20260615/v24_c12_split_manifest.json
TEACHER_CACHE=/sda/home/wangyuxin/ConvIR-B/runtime_cache/v24_c12_wd0375_teacher
FEATURE_TABLE="$PREV_ROOT/experience_docx/experiment_logs/haze4k_v2_13_nopost_feature_gated_adapter_20260702/v213_n1_feature_rows_cloud_only.csv"
STATUS="$EVID/status.txt"
LOG="$EVID/v214_n1r_runtime_probe.log"

mkdir -p "$EVID"
{
  printf 'run_start v214_n1r %s\n' "$(date --iso-8601=seconds)"
  printf 'remote_root=%s\n' "$REMOTE_ROOT"
  printf 'python=%s\n' "$PY"
  printf 'locked_test_touched=false\n'
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
} | tee -a "$STATUS"

if [ ! -f "$FEATURE_TABLE" ]; then
  printf 'feature_table_missing_rebuild_start %s\n' "$(date --iso-8601=seconds)" | tee -a "$STATUS"
  REBUILD_DIR="$EVID/raw_feature_table_rebuild_cloud_only"
  mkdir -p "$REBUILD_DIR"
  "$PY" experience_docx/tools/build_nopost_feature_table.py \
    --data-dir "$DATA_DIR" \
    --checkpoint "$CHECKPOINT" \
    --split-manifest "$SPLIT_MANIFEST" \
    --teacher-cache-dir "$TEACHER_CACHE" \
    --out-dir "$REBUILD_DIR" \
    --scope train_core \
    --print-freq 100 \
    2>&1 | tee "$EVID/v214_n1r_feature_table_rebuild.log"
  FEATURE_TABLE="$REBUILD_DIR/v213_n1_feature_rows_cloud_only.csv"
  printf 'feature_table_missing_rebuild_done %s\n' "$(date --iso-8601=seconds)" | tee -a "$STATUS"
fi

printf 'feature_table=%s\n' "$FEATURE_TABLE" | tee -a "$STATUS"

set +e
"$PY" experience_docx/tools/oof_probe_gain_risk_n1r.py \
  --feature-table "$FEATURE_TABLE" \
  --out-dir "$EVID" \
  --benefit-threshold 0.05 \
  --risk-threshold -0.20 \
  --bootstrap-iterations 1000 \
  --steps 1200 \
  2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e

printf 'run_done rc=%s v214_n1r %s\n' "$rc" "$(date --iso-8601=seconds)" | tee -a "$STATUS"
if [ "$rc" -eq 0 ]; then
  printf 'V214_N1R_SCRIPT_OK\n' | tee -a "$STATUS"
else
  printf 'V214_N1R_SCRIPT_FAILED\n' | tee -a "$STATUS"
fi
exit "$rc"
