#!/usr/bin/env bash
set -euo pipefail

REMOTE_REPO=${REMOTE_REPO:-/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v3q-active-signed-value-20260712}
RUN_ROOT=${RUN_ROOT:-/sda/home/wangyuxin/ConvIR-B/runs/haze4k_v5_chd_rm_v3q_active_signed_value_20260712}
PY=${PY:-/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python}
V3P_REPO=${V3P_REPO:-/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v3p-canonical-signed-gain-20260712}
V3P_RUN_ROOT=${V3P_RUN_ROOT:-/sda/home/wangyuxin/ConvIR-B/runs/haze4k_v5_chd_rm_v3p_canonical_signed_gain_20260712}
BASE=${BASE:-/sda/home/wangyuxin/ConvIR-B}
V3M_ROOT=${V3M_ROOT:-/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v3m-blockwise-counterfactual-advantage-20260711}
V3L_ROOT=${V3L_ROOT:-/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v3l-safe-step-escalation-physics-audit-20260711}
GPU=${GPU:-1}
MODE=${MODE:-smoke}
EXPECTED_ROUTE_COMMIT=${EXPECTED_ROUTE_COMMIT:?set the exact v3q route commit before launch}

ROUTE_ID=haze4k_v5_chd_rm_v3q_active_signed_value_20260712
BRANCH=codex/haze4k-v5-v3q-active-signed-value-20260712
EVID_STAGE=$REMOTE_REPO/experience_docx/experiment_logs/$ROUTE_ID
STATUS=$RUN_ROOT/status.txt
V3M_EVID=$V3M_ROOT/experience_docx/experiment_logs/haze4k_v5_chd_rm_v3m_blockwise_counterfactual_advantage_20260711
V3L_EVID=$V3L_ROOT/experience_docx/experiment_logs/haze4k_v5_chd_rm_v3l_safe_step_escalation_physics_audit_20260711
V3J_EVID=$V3M_ROOT/experience_docx/experiment_logs/haze4k_v5_chd_rm_v3j_bounded_safe_correction_audit_20260711
CANONICAL_BLOCKS=$V3P_RUN_ROOT/a0_formal/v3p_a0_block_candidate_losses_cloud_only.csv
NUMERICAL_CONTRACT=$V3P_REPO/experience_docx/experiment_logs/haze4k_v5_chd_rm_v3p_canonical_signed_gain_20260712/v3p_a0_numerical_reference_contract.json
STAMP=$(date +%Y%m%dT%H%M%S)

case "$MODE" in
  smoke)
    STAGE=v3q-A0b-smoke
    OUT=$RUN_ROOT/a0b_smoke32
    TAG=v3q_a0b_smoke32
    ;;
  formal)
    STAGE=v3q-A0b-formal
    OUT=$RUN_ROOT/a0b_formal
    TAG=v3q_a0b
    test -s "$EVID_STAGE/v3q_a0b_smoke32_closeout.json"
    "$PY" - "$EVID_STAGE/v3q_a0b_smoke32_closeout.json" <<'PY'
import json
import sys

value = json.load(open(sys.argv[1], encoding="utf-8"))
if value["decision"] != "V3Q_A0B_SMOKE_PASS_AUTHORIZE_FORMAL_ONLY":
    raise SystemExit("A0b formal requires a passing A0b smoke closeout")
PY
    ;;
  *)
    echo "V3Q_A0B_INVALID_MODE mode=$MODE"
    exit 2
    ;;
esac

test "$(git -C "$REMOTE_REPO" branch --show-current)" = "$BRANCH"
test "$(git -C "$REMOTE_REPO" rev-parse HEAD)" = "$EXPECTED_ROUTE_COMMIT"
test -z "$(git -C "$REMOTE_REPO" status --porcelain)"
test "$(git -C "$V3P_REPO" rev-parse HEAD)" = 555fd008e29f02128564f2fad41d0095ee44f5ea
test -x "$PY"
test -s "$CANONICAL_BLOCKS"
test -s "$NUMERICAL_CONTRACT"
test ! -e "$OUT"
nvidia-smi -i "$GPU" --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader

mkdir -p "$RUN_ROOT" "$EVID_STAGE"
LOG=$RUN_ROOT/${TAG}_${STAMP}.log
echo "stage_start route=$ROUTE_ID stage=$STAGE run=$TAG time=$(date --iso-8601=seconds)" | tee -a "$STATUS"
echo "stage_paths repo=$REMOTE_REPO run_root=$RUN_ROOT evid_stage=$EVID_STAGE gpu=$GPU log=$LOG" | tee -a "$STATUS"
set +e
CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 "$PY" "$REMOTE_REPO/experience_docx/tools/chd_rm_v3q_a0b_candidate_features.py" \
  --v3p-repo "$V3P_REPO" \
  --expected-v3p-source-commit 555fd008e29f02128564f2fad41d0095ee44f5ea \
  --canonical-blocks "$CANONICAL_BLOCKS" \
  --expected-canonical-blocks-sha256 52e6cd8829d37750cfb1e9e2fec39e6ac5cead2e324dbc353df93e5263e89765 \
  --output-dir "$OUT" --run-tag "$TAG" --run-mode "$MODE" \
  --a0-checkpoint "$BASE/checkpoints/official/Haze4K/haze4k-base.pkl" \
  --control-checkpoint "$BASE/repos/ConvIR-B-v3d-rarm-adapter-only-preflight/Dehazing/ITS/results/ConvIR-Haze4K-v3d-fam2modres-control-e5frome1-seed3407-20260710/Training-Results/Final.pkl" \
  --data-dir "$BASE/datasets/Haze4K/Haze4K" \
  --fresh-split-manifest "$V3J_EVID/fresh_route_confirm_split_manifest.json" \
  --v3j-a-bounds "$V3J_EVID/bounded_action_space_bounds.json" \
  --a0-closeout "$V3L_EVID/v3l_a0_canonical_operator_closeout.json" \
  --operator-artifact-manifest "$V3L_EVID/v3l_a0_canonical_operator_artifact_manifest.json" \
  --density-artifact "$BASE/repos/ConvIR-B-haze4k-v5-v2-chd-rm-density-need-calibration/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2_density_need_calibration_20260708/artifacts/d3_density_only_head.pt" \
  --d7c-artifact "$BASE/repos/ConvIR-B-haze4k-v5-v2d-chd-rm-need-spatial-hard-negative/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2d_need_spatial_hard_negative_20260709/d7c_full/artifacts/d7c_mc_topk_hn_ordinal_head.pt" \
  --reference-oof-rows "$V3M_EVID/cloud_only_raw_common_action/v3l_a1_oracle_policy_oof_rows_cloud_only.csv" \
  --v3m-a0-source-manifest "$V3M_EVID/v3m_a0_source_manifest.json" \
  --numerical-contract "$NUMERICAL_CONTRACT" \
  --expected-fresh-split-manifest-sha256 c8c00fefc965ded3389b6311fc67ea521e1f3174f27793688544abe09dc420e7 \
  --expected-parent-a0-closeout-sha256 2ca39ec1e17f4be794121603e3392a4e042e4d93b0e823454f7cf539f172d05d \
  --expected-parent-operator-manifest-sha256 1d2ffa499128ad08a272d67c5439583900afe8ef87fb3256193ad5fe21c3af84 \
  --expected-reference-oof-rows-sha256 b4a10184fab77b0045440dc88530d000a892acf2105a6295d5ad8a488c67ecb1 \
  --expected-v3m-a0-source-manifest-sha256 8966996c9c93f6f2f3fbdda536b69ea6aa03e1bf5432f127de47ca8ea95dd8a5 \
  --expected-density-artifact-sha256 1ffce13dccb41d96a47c2b5275f87bf2fdb73c226a190cfa240e5c71c1ec326f \
  --expected-d7c-artifact-sha256 09f449232024395cf64db15a2a0efa0f12d3e0e049e1da3d67229a3dc5729361 \
  --expected-a0-checkpoint-sha256 6f42037d57a4e3de3a10ac0ab909d66a3415864a19433c29204a975f4efa4088 \
  --expected-control-checkpoint-sha256 08207119a5cf9e5c439dd2cb81b99029ade1861f2739d31e75f2f9f78d57c0f2 \
  --block-size 16 --fold-count 5 --proj-channels 24 --operator-labels D_ref D_rep \
  --common-alphas 0 0.125 0.25 0.5 1 --replay-tolerance-db 1e-6 \
  --seed 3407 --progress-every 25 --device cuda:0 --route-commit "$EXPECTED_ROUTE_COMMIT" > "$LOG" 2>&1 &
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
  echo "V3Q_A0B_${MODE^^}_FAILED" | tee -a "$STATUS"
  exit "$rc"
fi
cp "$OUT/${TAG}_schema.json" "$EVID_STAGE/${TAG}_schema.json"
cp "$OUT/${TAG}_source_manifest.json" "$EVID_STAGE/${TAG}_source_manifest.json"
cp "$OUT/${TAG}_summary.json" "$EVID_STAGE/${TAG}_summary.json"
cp "$OUT/${TAG}_closeout.json" "$EVID_STAGE/${TAG}_closeout.json"
cp "$OUT/${TAG}_by_operator.csv" "$EVID_STAGE/${TAG}_by_operator.csv"
echo "V3Q_A0B_${MODE^^}_OK" | tee -a "$STATUS"
