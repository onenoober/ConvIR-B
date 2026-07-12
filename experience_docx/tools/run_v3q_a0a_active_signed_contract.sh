#!/usr/bin/env bash
set -euo pipefail

REMOTE_REPO=${REMOTE_REPO:-/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v3q-active-signed-value-20260712}
RUN_ROOT=${RUN_ROOT:-/sda/home/wangyuxin/ConvIR-B/runs/haze4k_v5_chd_rm_v3q_active_signed_value_20260712}
PY=${PY:-/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python}
V3P_RUN_ROOT=${V3P_RUN_ROOT:-/sda/home/wangyuxin/ConvIR-B/runs/haze4k_v5_chd_rm_v3p_canonical_signed_gain_20260712}
MODE=${MODE:-smoke}
EXPECTED_ROUTE_COMMIT=${EXPECTED_ROUTE_COMMIT:?set the exact v3q route commit before launch}

ROUTE_ID=haze4k_v5_chd_rm_v3q_active_signed_value_20260712
BRANCH=codex/haze4k-v5-v3q-active-signed-value-20260712
EVID_STAGE=$REMOTE_REPO/experience_docx/experiment_logs/$ROUTE_ID
STATUS=$RUN_ROOT/status.txt
CANONICAL_BLOCKS=$V3P_RUN_ROOT/a0_formal/v3p_a0_block_candidate_losses_cloud_only.csv
CANONICAL_IMAGES=$V3P_RUN_ROOT/a0_formal/v3p_a0_image_candidate_replay_cloud_only.csv
CANONICAL_SHA256=52e6cd8829d37750cfb1e9e2fec39e6ac5cead2e324dbc353df93e5263e89765
STAMP=$(date +%Y%m%dT%H%M%S)

case "$MODE" in
  smoke)
    STAGE=v3q-A0a-smoke
    OUT=$RUN_ROOT/a0a_smoke32
    TAG=v3q_a0a_smoke32
    MAX_IMAGES=32
    ;;
  formal)
    STAGE=v3q-A0a-formal
    OUT=$RUN_ROOT/a0a_formal
    TAG=v3q_a0a
    MAX_IMAGES=0
    test -s "$EVID_STAGE/v3q_a0a_smoke32_closeout.json"
    "$PY" - "$EVID_STAGE/v3q_a0a_smoke32_closeout.json" <<'PY'
import json
import sys

value = json.load(open(sys.argv[1], encoding="utf-8"))
if value["decision"] != "V3Q_A0A_SMOKE_PASS_AUTHORIZE_FORMAL_ONLY":
    raise SystemExit("v3q A0a formal requires a passing smoke closeout")
PY
    ;;
  *)
    echo "V3Q_A0A_INVALID_MODE mode=$MODE"
    exit 2
    ;;
esac

test "$(git -C "$REMOTE_REPO" branch --show-current)" = "$BRANCH"
test "$(git -C "$REMOTE_REPO" rev-parse HEAD)" = "$EXPECTED_ROUTE_COMMIT"
test -z "$(git -C "$REMOTE_REPO" status --porcelain)"
test -x "$PY"
test -s "$CANONICAL_BLOCKS"
test -s "$CANONICAL_IMAGES"
test ! -e "$OUT"

mkdir -p "$RUN_ROOT" "$EVID_STAGE"
LOG=$RUN_ROOT/${TAG}_${STAMP}.log
echo "stage_start route=$ROUTE_ID stage=$STAGE run=$TAG time=$(date --iso-8601=seconds)" | tee -a "$STATUS"
echo "stage_paths repo=$REMOTE_REPO run_root=$RUN_ROOT evid_stage=$EVID_STAGE log=$LOG" | tee -a "$STATUS"

set +e
PYTHONUNBUFFERED=1 "$PY" "$REMOTE_REPO/experience_docx/tools/chd_rm_v3q_a0a_active_signed_contract.py" \
  --canonical-blocks "$CANONICAL_BLOCKS" \
  --canonical-images "$CANONICAL_IMAGES" \
  --output-dir "$OUT" \
  --run-tag "$TAG" \
  --run-mode "$MODE" \
  --max-images "$MAX_IMAGES" \
  --expected-canonical-blocks-sha256 "$CANONICAL_SHA256" \
  --expected-route-commit "$EXPECTED_ROUTE_COMMIT" > "$LOG" 2>&1 &
pid=$!
(
  while kill -0 "$pid" 2>/dev/null; do
    echo "stage_heartbeat route=$ROUTE_ID stage=$STAGE pid=$pid time=$(date --iso-8601=seconds)" >> "$STATUS"
    sleep 60
  done
) &
heartbeat_pid=$!
wait "$pid"
rc=$?
kill "$heartbeat_pid" 2>/dev/null
wait "$heartbeat_pid" 2>/dev/null
set -e

echo "stage_done route=$ROUTE_ID stage=$STAGE rc=$rc time=$(date --iso-8601=seconds)" | tee -a "$STATUS"
if [ "$rc" -ne 0 ]; then
  echo "V3Q_A0A_${MODE^^}_FAILED" | tee -a "$STATUS"
  exit "$rc"
fi

cp "$OUT/${TAG}_source_manifest.json" "$EVID_STAGE/${TAG}_source_manifest.json"
cp "$OUT/${TAG}_summary.json" "$EVID_STAGE/${TAG}_summary.json"
cp "$OUT/${TAG}_closeout.json" "$EVID_STAGE/${TAG}_closeout.json"
cp "$OUT/${TAG}_by_operator.csv" "$EVID_STAGE/${TAG}_by_operator.csv"
echo "V3Q_A0A_${MODE^^}_OK" | tee -a "$STATUS"
