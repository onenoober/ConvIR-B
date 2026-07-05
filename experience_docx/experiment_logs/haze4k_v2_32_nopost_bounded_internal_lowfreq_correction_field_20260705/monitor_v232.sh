#!/usr/bin/env bash
set -euo pipefail
BASE=/sda/home/wangyuxin/ConvIR-B
WORK=$BASE/repos/ConvIR-B-haze4k-v2-32-nopost-bounded-internal-lowfreq-correction-field
EVID=$WORK/experience_docx/experiment_logs/haze4k_v2_32_nopost_bounded_internal_lowfreq_correction_field_20260705
PY=$BASE/envs/convir-cu121/bin/python

printf 'remote_time=%s\n' "$(date --iso-8601=seconds)"
printf 'work=%s\n' "$WORK"
for session in v232_p0_preflight v232_p1_sanity v232_p2_canary32 v232_p2_canary80_oof; do
  if tmux has-session -t "$session" 2>/dev/null; then
    printf '%s=ACTIVE\n' "$session"
  else
    printf '%s=NOT_ACTIVE\n' "$session"
  fi
done
if [[ -f "$EVID/status.txt" ]]; then
  printf '%s\n' '--- status_tail ---'
  tail -n 80 "$EVID/status.txt"
else
  printf 'status=MISSING\n'
fi
printf '%s\n' '--- recent_evidence ---'
find "$EVID" -maxdepth 1 -type f -printf '%TY-%Tm-%Td %TH:%TM %f\n' | sort | tail -n 40
"$PY" - <<'PY'
import json
from pathlib import Path
evid = Path('/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-haze4k-v2-32-nopost-bounded-internal-lowfreq-correction-field/experience_docx/experiment_logs/haze4k_v2_32_nopost_bounded_internal_lowfreq_correction_field_20260705')
for name in [
    'v232_p0_identity_zero_init_report.json',
    'v232_p1_field_sanity_closeout.json',
    'v232_p2_canary32_closeout.json',
    'v232_p2_canary80_oof_closeout.json',
    'v232_closeout.json',
]:
    path = evid / name
    if path.exists():
        data = json.loads(path.read_text(encoding='utf-8'))
        print(f'{name}: decision={data.get("decision")} pass={data.get("pass", data.get("gate_pass"))}')
PY
printf 'REMOTE_MONITOR_OK\n'
