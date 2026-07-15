#!/usr/bin/env bash
set -euo pipefail
route_id="haze4k_v5_chd_rm_v4a_a1x_exact_half_deployable_accessibility_20260715"
mode="${1:-}"
authorization="${2:-}"
if [ "$mode" != "s0" ]; then echo "A1X runner refused: formal mode is not enabled" >&2; exit 2; fi
if [ -z "$authorization" ] || [ ! -f "$authorization" ]; then echo "A1X runner refused: missing authorization" >&2; exit 2; fi
python3 - "$authorization" "$route_id" <<'PY'
import json, sys
p=json.load(open(sys.argv[1], encoding="utf-8")); expected={"route_id":sys.argv[2],"state":"PLANNED","decision":"V4A_A1X_S0_AUTHORIZED_INITIAL_ONLY","authorizes":"A1X_S0_ONLY","source_commit":"3b4da35440c8c26a7d1bcaf1daf342e11d9a3898"}
if any(p.get(k) != v for k,v in expected.items()): raise SystemExit("A1X runner refused: authorization mismatch")
PY
exec python3 experience_docx/tools/chd_rm_v4a_a1x_exact_half_accessibility.py --stage s0 --authorization-json "$authorization"
