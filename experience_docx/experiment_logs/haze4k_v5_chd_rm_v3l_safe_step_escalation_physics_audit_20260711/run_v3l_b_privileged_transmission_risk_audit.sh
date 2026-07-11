#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT=${REMOTE_ROOT:-/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v3l-safe-step-escalation-physics-audit-20260711}
BASE=${BASE:-/sda/home/wangyuxin/ConvIR-B}
PY=${PY:-/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python}
ROUTE_ID=haze4k_v5_chd_rm_v3l_safe_step_escalation_physics_audit_20260711
BRANCH=codex/haze4k-v5-v3l-safe-step-escalation-physics-audit
EVID="$REMOTE_ROOT/experience_docx/experiment_logs/$ROUTE_ID"
STATUS="$EVID/status.txt"
STAMP=$(date +%Y%m%dT%H%M%S)
LOG="$EVID/v3l_b_privileged_transmission_risk_audit_${STAMP}.log"
A1_SUMMARY="$EVID/v3l_a1_oracle_granularity_summary.json"
A1_POLICY_ROWS="$EVID/v3l_a1_oracle_policy_rows_cloud_only.csv"
A1_DIRECTION_ROWS="$EVID/v3l_a1_direct_direction_rows_cloud_only.csv"

mkdir -p "$EVID"

echo "v3l_b_start $ROUTE_ID $(date --iso-8601=seconds)" | tee -a "$STATUS"
echo "v3l_b_log $LOG" | tee -a "$STATUS"

cd "$REMOTE_ROOT"
if [ "$(git branch --show-current)" != "$BRANCH" ]; then
  echo "V3L_B_PREFLIGHT_FAILED wrong_branch=$(git branch --show-current)" | tee -a "$STATUS"
  exit 2
fi

"$PY" - <<PY
import json
from pathlib import Path

required = [
    Path("$BASE/datasets/Haze4K/Haze4K/train/trans"),
    Path("$A1_SUMMARY"),
    Path("$A1_POLICY_ROWS"),
    Path("$A1_DIRECTION_ROWS"),
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit("missing required assets: " + ", ".join(missing))
summary = json.loads(Path("$A1_SUMMARY").read_text(encoding="utf-8"))
expected = "V3L_A1_ORACLE_GRANULARITY_PASS_AUTHORIZE_B_PHYSICS_RISK_AUDIT_ONLY"
if summary.get("decision") != expected:
    raise SystemExit(f"A1 did not authorize B: {summary.get('decision')}")
if summary.get("canary_authorized") or summary.get("locked_test_touched"):
    raise SystemExit("A1 boundary violation")
print("V3L_B_PREFLIGHT_OK")
PY

heartbeat() {
  while true; do
    sleep 300
    echo "v3l_b_heartbeat $ROUTE_ID $(date --iso-8601=seconds)" >> "$STATUS"
  done
}
heartbeat &
HB_PID=$!
cleanup() {
  kill "$HB_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT

set +e
PYTHONUNBUFFERED=1 "$PY" experience_docx/tools/chd_rm_v3l_b_privileged_transmission_risk_audit.py \
  --data_dir "$BASE/datasets/Haze4K/Haze4K" \
  --a1_summary "$A1_SUMMARY" \
  --a1_policy_rows "$A1_POLICY_ROWS" \
  --a1_direction_rows "$A1_DIRECTION_ROWS" \
  --output_dir "$EVID" \
  --source_split train \
  --include_confirm_audit \
  --min_direct_severe_auc 0.65 \
  --min_low_alpha_auc 0.60 \
  --min_wrong_harmful_auc 0.58 \
  --bin_count 5 \
  --allow_overwrite \
  > "$LOG" 2>&1
rc=$?
set -e
echo "v3l_b_done rc=$rc $ROUTE_ID $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [ "$rc" -eq 0 ]; then
  echo "V3L_B_PRIVILEGED_TRANSMISSION_RISK_AUDIT_OK" | tee -a "$STATUS"
else
  echo "V3L_B_PRIVILEGED_TRANSMISSION_RISK_AUDIT_FAILED" | tee -a "$STATUS"
fi
exit "$rc"
