#!/usr/bin/env bash
set -euo pipefail

SESSION='convir-haze4k_v5_r3_propo-r3_a2_acv_-r3-a2-oof--d61263795c00'
OUTPUT='/sda/home/wangyuxin/ConvIR-B/runs/haze4k_v5_r3_proposal_first_acv_20260717/r3-a2-oof-r1'
HEARTBEAT="$OUTPUT/heartbeat.json"
STATUS="$OUTPUT/status.txt"
PY='/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python'

if tmux has-session -t "$SESSION" 2>/dev/null; then
  printf 'R3_A2_TERMINAL_CLEANUP_BLOCKED session_active=true\n'
  exit 92
fi
test -f "$STATUS"

if test -e "$HEARTBEAT"; then
  "$PY" - "$HEARTBEAT" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
value = json.loads(path.read_text(encoding="utf-8"))
if value.get("route_id") != "haze4k_v5_r3_proposal_first_acv_20260717":
    raise SystemExit("R3_A2_TERMINAL_CLEANUP_IDENTITY_MISMATCH route_id")
if value.get("run_id") != "r3-a2-oof-r1":
    raise SystemExit("R3_A2_TERMINAL_CLEANUP_IDENTITY_MISMATCH run_id")
PY
  rm -f -- "$HEARTBEAT"
fi

test ! -e "$HEARTBEAT"
printf 'R3_A2_TERMINAL_CLEANUP_OK session_active=false heartbeat_present=false status_present=true\n'
