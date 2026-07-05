#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT=${REMOTE_ROOT:-/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v2-31-nopost-action-value-identifiability-audit}
EVID="$REMOTE_ROOT/experience_docx/experiment_logs/haze4k_v2_31_nopost_action_value_identifiability_audit_20260705"
SESSION=${SESSION:-v231_p2a}
PY=${PY:-/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python}

date --iso-8601=seconds
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "v231_p2a=ACTIVE"
else
  echo "v231_p2a=NOT_ACTIVE"
fi
if [ -f "$EVID/status.txt" ]; then
  tail -60 "$EVID/status.txt"
else
  echo "status.txt=MISSING"
fi
if [ -f "$EVID/v231_p2a_closeout.json" ]; then
  "$PY" -c 'import json,sys; d=json.load(open(sys.argv[1])); print("decision="+str(d.get("decision"))); print("locked_test_touched="+str(d.get("locked_test_touched"))); print("training_launched="+str(d.get("training_launched"))); print("p2b_selector_probe_launched="+str(d.get("p2b_selector_probe_launched"))); print("primary="+str(d.get("p2a",{}).get("primary_diagnosis",{})))' "$EVID/v231_p2a_closeout.json"
fi
find "$EVID" -maxdepth 1 -type f -printf "%TY-%Tm-%Td %TH:%TM %s %f\n" | sort | tail -40
echo "V231_MONITOR_OK"
