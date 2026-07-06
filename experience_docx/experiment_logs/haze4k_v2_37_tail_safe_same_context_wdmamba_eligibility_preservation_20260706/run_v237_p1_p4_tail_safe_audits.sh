#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT="/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-haze4k-v2-37-tail-safe-same-context-wdmamba-eligibility-preservation"
EVID="$REMOTE_ROOT/experience_docx/experiment_logs/haze4k_v2_37_tail_safe_same_context_wdmamba_eligibility_preservation_20260706"
PY="/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python"
STATUS="$EVID/status.txt"

mkdir -p "$EVID/runtime_logs"
cd "$REMOTE_ROOT"

P0_CLOSEOUT="$EVID/v237_p0_closeout.json"
if [ ! -f "$P0_CLOSEOUT" ]; then
  echo "v237_p1_p4_blocked missing_p0_closeout $(date --iso-8601=seconds)" | tee -a "$STATUS"
  exit 2
fi

P0_PASS=$(P0_CLOSEOUT="$P0_CLOSEOUT" "$PY" - <<'PY'
import json
import os
print(str(bool(json.load(open(os.environ["P0_CLOSEOUT"])).get("gate_pass"))).lower())
PY
)
if [ "$P0_PASS" = "true" ]; then
  echo "v237_p1_p4_not_authorized p0_unmasked_alpha_pass $(date --iso-8601=seconds)" | tee -a "$STATUS"
  "$PY" -u experience_docx/tools/run_haze4k_v237_tail_safe_wdmamba_audit.py closeout --out-dir "$EVID"
  echo V237_P1_P4_NOT_AUTHORIZED_P0_PASS
  exit 0
fi

echo "v237_p1_start $(date --iso-8601=seconds)" | tee -a "$STATUS"
set +e
"$PY" -u experience_docx/tools/run_haze4k_v237_tail_safe_wdmamba_audit.py p1-tail-atlas \
  --out-dir "$EVID" \
  --p0-csv "$EVID/v237_p0_alpha_safety_sweep_per_image.csv" \
  2>&1 | tee "$EVID/runtime_logs/v237_p1_tail_failure_atlas.log"
rc=${PIPESTATUS[0]}
set -e
echo "v237_p1_done rc=$rc $(date --iso-8601=seconds)" | tee -a "$STATUS"
[ "$rc" -eq 0 ] || exit "$rc"

echo "v237_p2_start $(date --iso-8601=seconds)" | tee -a "$STATUS"
set +e
"$PY" -u experience_docx/tools/run_haze4k_v237_tail_safe_wdmamba_audit.py p2-mask-sweep \
  --out-dir "$EVID" \
  --p0-csv "$EVID/v237_p0_alpha_safety_sweep_per_image.csv" \
  2>&1 | tee "$EVID/runtime_logs/v237_p2_mask_preservation_sweep.log"
rc=${PIPESTATUS[0]}
set -e
echo "v237_p2_done rc=$rc $(date --iso-8601=seconds)" | tee -a "$STATUS"
[ "$rc" -eq 0 ] || exit "$rc"

P2_PASS=$(EVID="$EVID" "$PY" - <<'PY'
import json
import os
path = os.path.join(os.environ["EVID"], "v237_p2_closeout.json")
print(str(bool(json.load(open(path)).get("gate_pass"))).lower())
PY
)
if [ "$P2_PASS" != "true" ]; then
  echo "v237_p3_blocked p2_failed $(date --iso-8601=seconds)" | tee -a "$STATUS"
  "$PY" -u experience_docx/tools/run_haze4k_v237_tail_safe_wdmamba_audit.py closeout --out-dir "$EVID"
  echo V237_STOP_AFTER_P2_FAIL
  exit 0
fi

echo "v237_p3_start $(date --iso-8601=seconds)" | tee -a "$STATUS"
set +e
"$PY" -u experience_docx/tools/run_haze4k_v237_tail_safe_wdmamba_audit.py p3-oof-mask \
  --out-dir "$EVID" \
  --p2-csv "$EVID/v237_p2_mask_preservation_sweep_per_image.csv" \
  2>&1 | tee "$EVID/runtime_logs/v237_p3_oof_mask_selection.log"
rc=${PIPESTATUS[0]}
set -e
echo "v237_p3_done rc=$rc $(date --iso-8601=seconds)" | tee -a "$STATUS"
[ "$rc" -eq 0 ] || exit "$rc"

P3_PASS=$(EVID="$EVID" "$PY" - <<'PY'
import json
import os
path = os.path.join(os.environ["EVID"], "v237_p3_closeout.json")
print(str(bool(json.load(open(path)).get("gate_pass"))).lower())
PY
)
if [ "$P3_PASS" != "true" ]; then
  echo "v237_p4_blocked p3_failed $(date --iso-8601=seconds)" | tee -a "$STATUS"
  "$PY" -u experience_docx/tools/run_haze4k_v237_tail_safe_wdmamba_audit.py closeout --out-dir "$EVID"
  echo V237_STOP_AFTER_P3_FAIL
  exit 0
fi

echo "v237_p4_start $(date --iso-8601=seconds)" | tee -a "$STATUS"
set +e
"$PY" -u experience_docx/tools/run_haze4k_v237_tail_safe_wdmamba_audit.py p4-target-only \
  --out-dir "$EVID" \
  --p1-atlas-csv "$EVID/v237_p1_tail_failure_atlas.csv" \
  --p3-csv "$EVID/v237_p3_oof_mask_selection_per_image.csv" \
  2>&1 | tee "$EVID/runtime_logs/v237_p4_target_only_eligibility.log"
rc=${PIPESTATUS[0]}
set -e
echo "v237_p4_done rc=$rc $(date --iso-8601=seconds)" | tee -a "$STATUS"
[ "$rc" -eq 0 ] || exit "$rc"

"$PY" -u experience_docx/tools/run_haze4k_v237_tail_safe_wdmamba_audit.py closeout --out-dir "$EVID"
echo "v237_p1_p4_done $(date --iso-8601=seconds)" | tee -a "$STATUS"
echo V237_P1_P4_COMMAND_OK
