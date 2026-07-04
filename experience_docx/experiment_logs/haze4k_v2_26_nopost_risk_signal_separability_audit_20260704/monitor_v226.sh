#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v2-26-nopost-risk-signal-separability-audit
EVID=$REMOTE_ROOT/experience_docx/experiment_logs/haze4k_v2_26_nopost_risk_signal_separability_audit_20260704
PY=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python

printf 'remote_time=%s\n' "$(date -Is)"
for s in v226_p0_p2 v226_p3 v226_p4; do
  if tmux has-session -t "$s" 2>/dev/null; then
    printf '%s=ACTIVE\n' "$s"
  else
    printf '%s=NOT_ACTIVE\n' "$s"
  fi
done
if [ -f "$EVID/status.txt" ]; then
  printf '%s\n' '--- status tail ---'
  tail -n 30 "$EVID/status.txt"
else
  printf 'status_missing=%s\n' "$EVID/status.txt"
fi
printf '%s\n' '--- recent evidence ---'
find "$EVID" -maxdepth 2 -type f -printf '%TY-%Tm-%Td %TH:%TM %p\n' 2>/dev/null | sort | tail -n 30 || true
if [ -f "$EVID/v226_closeout.json" ]; then
  "$PY" - "$EVID/v226_closeout.json" <<'PY'
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
print("overall_decision=" + str(data.get("overall_decision")))
for key in ("p0", "p1", "p2", "p3", "p4"):
    if key in data:
        print(f"{key}_decision={data[key].get('decision')}")
PY
fi
printf 'REMOTE_V226_MONITOR_OK\n'
