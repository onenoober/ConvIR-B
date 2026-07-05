#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT=${REMOTE_ROOT:-/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v2-30-nopost-ilfrb-acs-compatibility-gated-oof-table-policy}
EVID="$REMOTE_ROOT/experience_docx/experiment_logs/haze4k_v2_30_nopost_ilfrb_acs_compatibility_gated_oof_table_policy_20260705"
SESSION=${SESSION:-v230_p2a}
PY=${PY:-/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python}

date --iso-8601=seconds
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "v230_p2a=ACTIVE"
else
  echo "v230_p2a=NOT_ACTIVE"
fi
if [ -f "$EVID/status.txt" ]; then
  tail -40 "$EVID/status.txt"
else
  echo "status.txt=MISSING"
fi
if [ -f "$EVID/v230_p2a_closeout.json" ]; then
  "$PY" -c 'import json,sys; d=json.load(open(sys.argv[1])); print("decision="+str(d.get("decision"))); print("locked_test_touched="+str(d.get("locked_test_touched"))); print("training_launched="+str(d.get("training_launched"))); print("p2b_selector_probe_launched="+str(d.get("p2b_selector_probe_launched")))' "$EVID/v230_p2a_closeout.json"
fi
find "$EVID" -maxdepth 1 -type f -printf "%TY-%Tm-%Td %TH:%TM %s %f\n" | sort | tail -30
echo "V230_MONITOR_OK"
