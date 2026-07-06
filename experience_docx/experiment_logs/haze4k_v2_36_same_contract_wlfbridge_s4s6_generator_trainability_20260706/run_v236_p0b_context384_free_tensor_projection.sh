#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

REMOTE_ROOT="/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-haze4k-v2-36-same-contract-wlfbridge-s4s6-generator-trainability"
EVID="$REMOTE_ROOT/experience_docx/experiment_logs/haze4k_v2_36_same_contract_wlfbridge_s4s6_generator_trainability_20260706"
PY="/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python"
P0C="/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-haze4k-v2-34-nopost-teacher-delta-projection-and-multistage-bridge-audit/experience_docx/experiment_logs/haze4k_v2_34_nopost_teacher_delta_projection_and_multistage_bridge_audit_20260706/v234_p0c_metric_contract_diagnostic.csv"
LOG="$EVID/runtime_logs/v236_p0b_context384_free_tensor_projection.log"
STATUS="$EVID/status.txt"

mkdir -p "$EVID/runtime_logs"
cd "$REMOTE_ROOT"
echo "v236_p0b_start $(date --iso-8601=seconds) locked=untouched cuda_visible=${CUDA_VISIBLE_DEVICES}" | tee -a "$STATUS"

"$PY" - <<'PY'
import json
from pathlib import Path
closeout = Path("experience_docx/experiment_logs/haze4k_v2_36_same_contract_wlfbridge_s4s6_generator_trainability_20260706/v236_p0_closeout.json")
payload = json.loads(closeout.read_text(encoding="utf-8"))
if not payload.get("gate_pass"):
    raise SystemExit("P0 gate is not pass; P0B is not authorized")
print("V236_P0B_AUTHORIZED_BY_P0")
PY

set +e
"$PY" -u experience_docx/tools/run_haze4k_v236_wlfbridge_audit.py p0b-projection \
  --out-dir "$EVID" \
  --p0c-csv "$P0C" \
  --context-size 384 \
  --alpha 0.5 \
  --device cuda \
  --learning-rate 0.003 \
  --grad-clip-norm 0.5 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e
echo "v236_p0b_done rc=$rc $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [ "$rc" -eq 0 ]; then echo V236_P0B_COMMAND_OK; else echo V236_P0B_COMMAND_FAILED; fi
exit "$rc"
