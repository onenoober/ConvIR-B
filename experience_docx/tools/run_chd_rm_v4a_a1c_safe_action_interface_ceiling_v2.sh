#!/usr/bin/env bash
# Schema-v2 runner. It is cloud-only and accepts only manifest-sealed S0/formal modes.
set -euo pipefail

ROUTE_ID=haze4k_v5_chd_rm_v4a_a1c_safe_action_interface_ceiling_20260715
REMOTE_REPO=${REMOTE_REPO:?set by the authorized operation}
RUN_ROOT=${RUN_ROOT:?set by the authorized operation}
RUN_ID=${RUN_ID:?set by the authorized operation}
MODE=${MODE:?set by the authorized operation}
EXPECTED_ROUTE_COMMIT=${EXPECTED_ROUTE_COMMIT:?set by the authorized operation}
RUNNER_SHA256=${RUNNER_SHA256:?set by the authorized operation}
GPU=${GPU:?set by the authorized operation}
PY=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
BASE=/sda/home/wangyuxin/ConvIR-B
RUNNER=$REMOTE_REPO/experience_docx/tools/chd_rm_v4a_a1c_safe_action_interface_ceiling_v2.py
CARD=$REMOTE_REPO/experience_docx/experiment_cards/2026-07-15-haze4k-v5-v4a-a1c-safe-action-interface-ceiling.md
AUTH0=$REMOTE_REPO/experience_docx/experiment_logs/$ROUTE_ID/initial_authorization.json
S0_CLOSEOUT=$REMOTE_REPO/experience_docx/experiment_logs/$ROUTE_ID/v4a_a1c_s0_closeout.json
OUT=$RUN_ROOT/$RUN_ID
STATUS=$RUN_ROOT/status.txt
EVID=$REMOTE_REPO/experience_docx/experiment_logs/$ROUTE_ID
V3Z_ROOT=$BASE/repos/ConvIR-B-v3z-source-3caddcc5265732e5be77e3404119a28cb28c11e6
A1F_MODULE=$BASE/repos/ConvIR-B-v4a-a1f-deltau-action-feasibility-20260714/experience_docx/tools/chd_rm_v4a_a1f_action_feasibility.py
A0R_TRACE=$BASE/runs/haze4k_v5_chd_rm_v4a_conditional_safety_audit_20260714/a0r_r2/r1/trace

case "$MODE" in s0|formal) ;; *) echo "invalid sealed MODE=$MODE" >&2; exit 2;; esac
test "$(git -C "$REMOTE_REPO" branch --show-current)" = codex/haze4k-v5-v4a-a1c-safe-action-interface-ceiling-20260715
test "$(git -C "$REMOTE_REPO" rev-parse HEAD)" = "$EXPECTED_ROUTE_COMMIT"
test -z "$(git -C "$REMOTE_REPO" status --porcelain)"
test -x "$PY"; test -f "$RUNNER" -a -f "$CARD" -a -f "$AUTH0"
test "$(sha256sum "$RUNNER" | awk '{print $1}')" = "$RUNNER_SHA256"
test ! -e "$OUT"
if [ "$MODE" = s0 ]; then
  "$PY" - "$AUTH0" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); assert (x['state'],x['decision'],x['authorizes']) == ('PLANNED','V4A_A1C_S0_AUTHORIZED_INITIAL_ONLY','A1C_S0_ONLY')
PY
else
  test -f "$S0_CLOSEOUT"
  "$PY" - "$S0_CLOSEOUT" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); assert (x['state'],x['decision'],x['authorizes']) == ('COMPLETED_GATE_PASS','V4A_A1C_S0_ALIGNMENT_PASS_AUTHORIZE_FORMAL_ONLY','A1C_FORMAL_INTERFACE_CEILING_ONLY')
PY
fi
GPU_STATUS=$(nvidia-smi -i "$GPU" --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits)
GPU_FREE=$(printf '%s\n' "$GPU_STATUS" | awk -F, '{gsub(/ /,"",$1);print $1}')
GPU_UTIL=$(printf '%s\n' "$GPU_STATUS" | awk -F, '{gsub(/ /,"",$2);print $2}')
test "$GPU_FREE" -ge 12000; test "$GPU_UTIL" -le 10
mkdir -p "$RUN_ROOT" "$EVID"
printf 'heartbeat route=%s mode=%s phase=preflight gpu=%s free_mib=%s util=%s\n' "$ROUTE_ID" "$MODE" "$GPU" "$GPU_FREE" "$GPU_UTIL" | tee -a "$STATUS"
LOG=$RUN_ROOT/v4a_a1c_${MODE}_$(date +%Y%m%dT%H%M%S).log
set +e
CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 "$PY" "$RUNNER" audit --mode "$MODE" --v3z-root "$V3Z_ROOT" --a1f-module "$A1F_MODULE" --a0r-trace-dir "$A0R_TRACE" --expected-route-commit "$EXPECTED_ROUTE_COMMIT" --expected-route-card-sha256 "$(sha256sum "$CARD" | awk '{print $1}')" --runner-sha256 "$RUNNER_SHA256" --status-file "$STATUS" "${A1C_FROZEN_ARGS[@]}" 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}; set -e
printf 'stage_done route=%s mode=%s run=%s rc=%s\n' "$ROUTE_ID" "$MODE" "$RUN_ID" "$rc" | tee -a "$STATUS"
if [ "$rc" -ne 0 ]; then echo "V4A_A1C_${MODE^^}_FAILED_RUNTIME_REQUIRES_CLASSIFICATION" | tee -a "$STATUS"; exit "$rc"; fi
if [ "$MODE" = s0 ]; then DEST=$S0_CLOSEOUT; else DEST=$EVID/v4a_a1c_formal_closeout.json; fi
cp "$OUT/v4a_a1c_closeout.json" "$DEST"
cp "$OUT/v4a_a1c_source_manifest.json" "$EVID/v4a_a1c_${MODE}_source_manifest.json"
cp "$OUT/v4a_a1c_bootstrap_summary.json" "$EVID/v4a_a1c_${MODE}_bootstrap_summary.json"
echo "V4A_A1C_${MODE^^}_OK" | tee -a "$STATUS"
