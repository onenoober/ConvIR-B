#!/usr/bin/env bash
set -euo pipefail

REMOTE_REPO=${REMOTE_REPO:-/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v3q-active-signed-value-20260712}
RUN_ROOT=${RUN_ROOT:-/sda/home/wangyuxin/ConvIR-B/runs/haze4k_v5_chd_rm_v3q_active_signed_value_20260712}
PY=${PY:-/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python}
MODE=${MODE:-smoke}
GPU=${GPU:-1}
EXPECTED_ROUTE_COMMIT=${EXPECTED_ROUTE_COMMIT:?set exact route commit}
ROUTE_ID=haze4k_v5_chd_rm_v3q_active_signed_value_20260712
EVID=$REMOTE_REPO/experience_docx/experiment_logs/$ROUTE_ID
STATUS=$RUN_ROOT/status.txt

case "$MODE" in
  smoke) TAG=v3q_a1_smoke32; FEATURES=$RUN_ROOT/a0b_smoke32/v3q_a0b_smoke32_active_features_cloud_only.csv; FEATURE_SHA=bd04ea09f1d325a2425ce7cadaa76a5a060e598feca9e256fcf06588be4576ab; OUT=$RUN_ROOT/a1_smoke32 ;;
  formal) TAG=v3q_a1; FEATURES=$RUN_ROOT/a0b_formal/v3q_a0b_active_features_cloud_only.csv; FEATURE_SHA=2cda824bf6ca7b11ed0a50d56d785790200a6680fcfb9a4b34e154f40f111b82; OUT=$RUN_ROOT/a1_formal; test -s "$EVID/v3q_a1_smoke32_closeout.json"; "$PY" - "$EVID/v3q_a1_smoke32_closeout.json" <<'PY'
import json
import sys

if json.load(open(sys.argv[1], encoding="utf-8")).get("decision") != "V3Q_A1_SMOKE_PASS_AUTHORIZE_FORMAL_ONLY":
    raise SystemExit("A1 formal requires a passing A1 smoke closeout")
PY
  ;;
  *) echo V3Q_A1_INVALID_MODE; exit 2 ;;
esac
test "$(git -C "$REMOTE_REPO" rev-parse HEAD)" = "$EXPECTED_ROUTE_COMMIT"
test -z "$(git -C "$REMOTE_REPO" status --porcelain)"
test -x "$PY"
test -s "$FEATURES"
test -s "$EVID/v3q_a0b_closeout.json"
test -s "$EVID/v3q_a0b_schema.json"
test ! -e "$OUT"
mkdir -p "$RUN_ROOT" "$EVID"
LOG=$RUN_ROOT/${TAG}_$(date +%Y%m%dT%H%M%S).log
echo "stage_start route=$ROUTE_ID stage=v3q-A1-$MODE run=$TAG time=$(date --iso-8601=seconds)" | tee -a "$STATUS"
set +e
CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 "$PY" "$REMOTE_REPO/experience_docx/tools/chd_rm_v3q_a1_signed_linear_probe.py" --features "$FEATURES" --expected-features-sha256 "$FEATURE_SHA" --a0b-closeout "$EVID/v3q_a0b_closeout.json" --expected-a0b-closeout-sha256 8ad2b7235b5ccb7447338a5abedbce1854e7e8d201b5ba4db05e1965476fc202 --schema "$EVID/v3q_a0b_schema.json" --expected-schema-sha256 3934b44811e802216735b8b46da24fab656dee22adc55f90c253bfbdc83e40dd --output-dir "$OUT" --run-tag "$TAG" --run-mode "$MODE" --route-commit "$EXPECTED_ROUTE_COMMIT" --fold-count 5 --epochs 60 --learning-rate 0.08 --l2 1e-4 --seed 3407 --device cuda:0 > "$LOG" 2>&1 &
pid=$!
( while kill -0 "$pid" 2>/dev/null; do echo "stage_heartbeat route=$ROUTE_ID stage=v3q-A1-$MODE pid=$pid time=$(date --iso-8601=seconds)" >> "$STATUS"; sleep 60; done ) &
heartbeat=$!
wait "$pid"; rc=$?
kill "$heartbeat" 2>/dev/null; wait "$heartbeat" 2>/dev/null
set -e
echo "stage_done route=$ROUTE_ID stage=v3q-A1-$MODE rc=$rc time=$(date --iso-8601=seconds)" | tee -a "$STATUS"
test "$rc" -eq 0 || exit "$rc"
for name in summary.csv by_fold.csv summary.json source_manifest.json closeout.json; do cp "$OUT/${TAG}_${name}" "$EVID/${TAG}_${name}"; done
echo "V3Q_A1_${MODE^^}_OK" | tee -a "$STATUS"
