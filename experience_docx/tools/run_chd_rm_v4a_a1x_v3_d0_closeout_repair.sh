#!/usr/bin/env bash
set -euo pipefail
ROUTE_ID=haze4k_v5_chd_rm_v4a_a1x_accessibility_v3_20260716
PY=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
ENTRY=$REMOTE_REPO/experience_docx/tools/chd_rm_v4a_a1x_v3_d0_closeout_repair.py
EVID=$REMOTE_REPO/experience_docx/experiment_logs/$ROUTE_ID
export OUTPUT_PATH RUN_ID EXPECTED_ROUTE_COMMIT RUNNER_SHA256
mkdir -p "$OUTPUT_PATH" "$EVID"
test "$MODE" = d0_closeout_repair
test "$(git -C "$REMOTE_REPO" rev-parse HEAD)" = "$EXPECTED_ROUTE_COMMIT"
test "$(sha256sum "$REMOTE_REPO/experience_docx/tools/run_chd_rm_v4a_a1x_v3_d0_closeout_repair.sh" | awk '{print $1}')" = "$RUNNER_SHA256"
"$PY" "$ENTRY" >"$OUTPUT_PATH/runtime.log" 2>&1
cp "$OUTPUT_PATH/a1x_v3_d0_summary.json" "$EVID/a1x_v3_d0_summary.json"
cp "$OUTPUT_PATH/a1x_v3_d0_repaired_closeout.json" "$EVID/a1x_v3_d0_repaired_closeout.json.tmp"
mv "$EVID/a1x_v3_d0_repaired_closeout.json.tmp" "$EVID/a1x_v3_d0_repaired_closeout.json"
echo A1X_V3_D0_CLOSEOUT_REPAIR_OK
