#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v2-27-nopost-ilfrb-action-conditioned-selective-distill
EVID=$REMOTE_ROOT/experience_docx/experiment_logs/haze4k_v2_27_nopost_ilfrb_action_conditioned_selective_distill_20260705
PY=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
STATUS=$EVID/status.txt

printf 'remote_time=%s\n' "$(date --iso-8601=seconds)"
if [ -d "$REMOTE_ROOT/.git" ]; then
  cd "$REMOTE_ROOT"
  printf 'branch=%s\n' "$(git branch --show-current)"
  printf 'commit=%s\n' "$(git rev-parse --short HEAD)"
  printf 'workspace_status_lines=%s\n' "$(git status --short | wc -l)"
fi

for session in v227_p0 v227_p1_p5; do
  if tmux has-session -t "$session" 2>/dev/null; then
    printf '%s=ACTIVE\n' "$session"
  else
    printf '%s=NOT_ACTIVE\n' "$session"
  fi
done

if command -v nvidia-smi >/dev/null 2>&1; then
  printf '%s\n' '--- gpu ---'
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
fi

if [ -f "$STATUS" ]; then
  printf '%s\n' '--- status tail ---'
  tail -n 80 "$STATUS"
else
  printf 'status_file=MISSING\n'
fi

printf '%s\n' '--- recent evidence ---'
find "$EVID" -maxdepth 1 -type f -printf '%TY-%Tm-%Td %TH:%TM %s %f\n' 2>/dev/null | sort | tail -60 || true

if [ -f "$EVID/v227_closeout.json" ]; then
  "$PY" - "$EVID/v227_closeout.json" <<'PY'
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
print("closeout_decision=" + str(data.get("decision")))
for key in ("p0", "p1", "p2", "p3", "p4", "p5"):
    if key in data:
        val = data[key]
        if isinstance(val, dict):
            print(f"{key}_decision={val.get('decision')}")
PY
fi

printf 'REMOTE_V227_MONITOR_OK\n'
