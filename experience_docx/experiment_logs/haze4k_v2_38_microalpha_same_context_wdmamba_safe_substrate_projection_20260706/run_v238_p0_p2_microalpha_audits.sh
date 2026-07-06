#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT="/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-haze4k-v2-38-microalpha-same-context-wdmamba-safe-substrate-projection"
EVID="$REMOTE_ROOT/experience_docx/experiment_logs/haze4k_v2_38_microalpha_same_context_wdmamba_safe_substrate_projection_20260706"
PY="/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python"
SOURCE_P0="/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-haze4k-v2-37-tail-safe-same-context-wdmamba-eligibility-preservation/experience_docx/experiment_logs/haze4k_v2_37_tail_safe_same_context_wdmamba_eligibility_preservation_20260706/v237_p0_alpha_safety_sweep_per_image.csv"
STATUS="$EVID/status.txt"
mkdir -p "$EVID/runtime_logs"
cd "$REMOTE_ROOT"

echo "v238_p0_p2_preflight $(date --iso-8601=seconds) branch=$(git branch --show-current) head=$(git rev-parse --short HEAD) locked=untouched bridge=blocked generator=blocked" | tee -a "$STATUS"
if [ ! -f "$SOURCE_P0" ]; then
  echo "v238_blocked_missing_source_p0 $SOURCE_P0 $(date --iso-8601=seconds)" | tee -a "$STATUS"
  echo V238_P0_P2_COMMAND_FAILED
  exit 2
fi
"$PY" -m py_compile experience_docx/tools/run_haze4k_v238_microalpha_audit.py

echo "v238_p0_start $(date --iso-8601=seconds)" | tee -a "$STATUS"
set +e
"$PY" -u experience_docx/tools/run_haze4k_v238_microalpha_audit.py p0 \
  --out-dir "$EVID" \
  --source-p0-csv "$SOURCE_P0" 2>&1 | tee "$EVID/runtime_logs/v238_p0_microalpha_safety_sweep.log"
rc=${PIPESTATUS[0]}
set -e
echo "v238_p0_done rc=$rc $(date --iso-8601=seconds)" | tee -a "$STATUS"
[ "$rc" -eq 0 ] || exit "$rc"

P0_PASS=$(EVID="$EVID" "$PY" - <<'PY'
import json, os
print(str(bool(json.load(open(os.path.join(os.environ["EVID"], "v238_p0_closeout.json"))).get("gate_pass"))).lower())
PY
)
if [ "$P0_PASS" != "true" ]; then
  "$PY" -u experience_docx/tools/run_haze4k_v238_microalpha_audit.py closeout --out-dir "$EVID"
  echo "v238_stop_after_p0_fail $(date --iso-8601=seconds)" | tee -a "$STATUS"
  echo V238_P0_P2_COMMAND_OK
  exit 0
fi

echo "v238_p1_start $(date --iso-8601=seconds)" | tee -a "$STATUS"
set +e
"$PY" -u experience_docx/tools/run_haze4k_v238_microalpha_audit.py p1 \
  --out-dir "$EVID" \
  --p0-csv "$EVID/v238_p0_microalpha_safety_sweep_per_image.csv" 2>&1 | tee "$EVID/runtime_logs/v238_p1_oof_microalpha_selection.log"
rc=${PIPESTATUS[0]}
set -e
echo "v238_p1_done rc=$rc $(date --iso-8601=seconds)" | tee -a "$STATUS"
[ "$rc" -eq 0 ] || exit "$rc"

P1_PASS=$(EVID="$EVID" "$PY" - <<'PY'
import json, os
print(str(bool(json.load(open(os.path.join(os.environ["EVID"], "v238_p1_closeout.json"))).get("gate_pass"))).lower())
PY
)
if [ "$P1_PASS" != "true" ]; then
  "$PY" -u experience_docx/tools/run_haze4k_v238_microalpha_audit.py closeout --out-dir "$EVID"
  echo "v238_stop_after_p1_fail $(date --iso-8601=seconds)" | tee -a "$STATUS"
  echo V238_P0_P2_COMMAND_OK
  exit 0
fi

echo "v238_p2_start $(date --iso-8601=seconds)" | tee -a "$STATUS"
set +e
"$PY" -u experience_docx/tools/run_haze4k_v238_microalpha_audit.py p2 \
  --out-dir "$EVID" \
  --p0-csv "$EVID/v238_p0_microalpha_safety_sweep_per_image.csv" \
  --p1-summary "$EVID/v238_p1_oof_microalpha_selection_summary.json" \
  --p1-per-image-csv "$EVID/v238_p1_oof_microalpha_selection_per_image.csv" 2>&1 | tee "$EVID/runtime_logs/v238_p2_critical_alpha_margin.log"
rc=${PIPESTATUS[0]}
set -e
echo "v238_p2_done rc=$rc $(date --iso-8601=seconds)" | tee -a "$STATUS"
[ "$rc" -eq 0 ] || exit "$rc"

"$PY" -u experience_docx/tools/run_haze4k_v238_microalpha_audit.py closeout --out-dir "$EVID"
P2_PASS=$(EVID="$EVID" "$PY" - <<'PY'
import json, os
print(str(bool(json.load(open(os.path.join(os.environ["EVID"], "v238_p2_closeout.json"))).get("gate_pass"))).lower())
PY
)
if [ "$P2_PASS" = "true" ]; then
  echo "v238_p3_authorized_not_launched $(date --iso-8601=seconds)" | tee -a "$STATUS"
else
  echo "v238_stop_after_p2_fail $(date --iso-8601=seconds)" | tee -a "$STATUS"
fi
echo V238_P0_P2_COMMAND_OK
