#!/usr/bin/env bash
set -euo pipefail

REMOTE_REPO=${REMOTE_REPO:-/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v3r-signed-margin-operator-repair-20260712}
RUN_ROOT=${RUN_ROOT:-/sda/home/wangyuxin/ConvIR-B/runs/haze4k_v5_chd_rm_v3r_signed_margin_operator_repair_20260712}
PY=${PY:-/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python}
BASE=${BASE:-/sda/home/wangyuxin/ConvIR-B}
V3P_REPO=${V3P_REPO:-/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v3p-canonical-signed-gain-20260712}
V3P_RUN=${V3P_RUN:-/sda/home/wangyuxin/ConvIR-B/runs/haze4k_v5_chd_rm_v3p_canonical_signed_gain_20260712}
V3M_ROOT=${V3M_ROOT:-/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v3m-blockwise-counterfactual-advantage-20260711}
V3L_ROOT=${V3L_ROOT:-/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v3l-safe-step-escalation-physics-audit-20260711}
GPU=${GPU:-1}
MODE=${MODE:-smoke}
EXPECTED_ROUTE_COMMIT=${EXPECTED_ROUTE_COMMIT:?set the exact v3r route commit before launch}
ROUTE_ID=haze4k_v5_chd_rm_v3r_signed_margin_operator_repair_20260712
BRANCH=codex/haze4k-v5-v3r-signed-margin-operator-repair-20260712
V3P_SOURCE_COMMIT=555fd008e29f02128564f2fad41d0095ee44f5ea
V3P_A0_SCRIPT_SHA256=de6a8adaf266e47895a231b485d9d6185a315a42ab025cdaa361af1b1874007f
CANONICAL_SHA256=52e6cd8829d37750cfb1e9e2fec39e6ac5cead2e324dbc353df93e5263e89765
EVID_STAGE=$REMOTE_REPO/experience_docx/experiment_logs/$ROUTE_ID
V3P_EVID=$V3P_REPO/experience_docx/experiment_logs/haze4k_v5_chd_rm_v3p_canonical_signed_gain_20260712
V3M_EVID=$V3M_ROOT/experience_docx/experiment_logs/haze4k_v5_chd_rm_v3m_blockwise_counterfactual_advantage_20260711
V3L_EVID=$V3L_ROOT/experience_docx/experiment_logs/haze4k_v5_chd_rm_v3l_safe_step_escalation_physics_audit_20260711
V3J_EVID=$V3M_ROOT/experience_docx/experiment_logs/haze4k_v5_chd_rm_v3j_bounded_safe_correction_audit_20260711
CANONICAL_BLOCKS=$V3P_RUN/a0_formal/v3p_a0_block_candidate_losses_cloud_only.csv
NUMERICAL_CONTRACT=$V3P_EVID/v3p_a0_numerical_reference_contract.json
STATUS=$RUN_ROOT/status.txt
STAMP=$(date +%Y%m%dT%H%M%S)

case "$MODE" in
  smoke)
    STAGE=v3r-A0-smoke
    OUT=$RUN_ROOT/a0_smoke32
    TAG=v3r_a0_smoke32
    MAX_TRAIN=32
    SOURCE_ROWS=40000
    PROGRESS=8
    ;;
  smoke_r1)
    STAGE=v3r-A0-smoke-r1
    OUT=$RUN_ROOT/a0_smoke32_r1
    TAG=v3r_a0_smoke32_r1
    MAX_TRAIN=32
    SOURCE_ROWS=40000
    PROGRESS=8
    ;;
  smoke_r2)
    STAGE=v3r-A0-smoke-r2
    OUT=$RUN_ROOT/a0_smoke32_r2
    TAG=v3r_a0_smoke32_r2
    MAX_TRAIN=32
    SOURCE_ROWS=40000
    PROGRESS=8
    ;;
  formal)
    STAGE=v3r-A0-formal
    OUT=$RUN_ROOT/a0_formal
    TAG=v3r_a0
    MAX_TRAIN=1200
    SOURCE_ROWS=2177350
    PROGRESS=25
    ;;
  *)
    echo "V3R_A0_INVALID_MODE mode=$MODE"
    exit 2
    ;;
esac

test "$(git -C "$REMOTE_REPO" branch --show-current)" = "$BRANCH"
test "$(git -C "$REMOTE_REPO" rev-parse HEAD)" = "$EXPECTED_ROUTE_COMMIT"
test -z "$(git -C "$REMOTE_REPO" status --porcelain)"
test "$(git -C "$V3P_REPO" rev-parse HEAD)" = "$V3P_SOURCE_COMMIT"
test -z "$(git -C "$V3P_REPO" status --porcelain)"
test -x "$PY"
test -s "$CANONICAL_BLOCKS"
test -s "$NUMERICAL_CONTRACT"
test ! -e "$OUT"
test "$(sha256sum "$CANONICAL_BLOCKS" | awk '{print $1}')" = "$CANONICAL_SHA256"
test "$(sha256sum "$V3P_REPO/experience_docx/tools/chd_rm_v3p_a0_canonical_signed_gain.py" | awk '{print $1}')" = "$V3P_A0_SCRIPT_SHA256"
test -s "$BASE/checkpoints/official/Haze4K/haze4k-base.pkl"
test -s "$BASE/repos/ConvIR-B-v3d-rarm-adapter-only-preflight/Dehazing/ITS/results/ConvIR-Haze4K-v3d-fam2modres-control-e5frome1-seed3407-20260710/Training-Results/Final.pkl"
test -s "$V3J_EVID/fresh_route_confirm_split_manifest.json"
test -s "$V3J_EVID/bounded_action_space_bounds.json"
test -s "$V3L_EVID/v3l_a0_canonical_operator_closeout.json"
test -s "$V3L_EVID/v3l_a0_canonical_operator_artifact_manifest.json"
test -s "$V3M_EVID/cloud_only_raw_common_action/v3l_a1_oracle_policy_oof_rows_cloud_only.csv"
test -s "$V3M_EVID/v3m_a0_source_manifest.json"
nvidia-smi -i "$GPU" --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader

if [ "$MODE" = formal ]; then
  test -s "$EVID_STAGE/v3r_a0_smoke32_r2_closeout.json"
  "$PY" - "$EVID_STAGE/v3r_a0_smoke32_r2_closeout.json" <<'PY'
import json
import sys
closeout = json.load(open(sys.argv[1], encoding="utf-8"))
if closeout.get("decision") != "V3R_A0_SMOKE_PASS_AUTHORIZE_FORMAL_ONLY":
    raise SystemExit("v3r A0 formal requires the exact smoke authorization")
if closeout.get("locked_test_touched") or closeout.get("canary_touched") or closeout.get("training_occurred"):
    raise SystemExit("v3r A0 smoke closeout has forbidden runtime flags")
PY
fi

mkdir -p "$RUN_ROOT" "$EVID_STAGE"
LOG=$RUN_ROOT/${TAG}_${STAMP}.log
echo "stage_start route=$ROUTE_ID stage=$STAGE run=$TAG time=$(date --iso-8601=seconds)" | tee -a "$STATUS"
echo "stage_paths repo=$REMOTE_REPO run_root=$RUN_ROOT evid_stage=$EVID_STAGE gpu=$GPU log=$LOG" | tee -a "$STATUS"
set +e
CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 "$PY" "$REMOTE_REPO/experience_docx/tools/chd_rm_v3r_a0_privileged_repair_geometry.py" \
  --v3p_source_root "$V3P_REPO" \
  --canonical_blocks "$CANONICAL_BLOCKS" \
  --v3p_numerical_contract "$NUMERICAL_CONTRACT" \
  --a0_checkpoint "$BASE/checkpoints/official/Haze4K/haze4k-base.pkl" \
  --control_checkpoint "$BASE/repos/ConvIR-B-v3d-rarm-adapter-only-preflight/Dehazing/ITS/results/ConvIR-Haze4K-v3d-fam2modres-control-e5frome1-seed3407-20260710/Training-Results/Final.pkl" \
  --data_dir "$BASE/datasets/Haze4K/Haze4K" \
  --fresh_split_manifest "$V3J_EVID/fresh_route_confirm_split_manifest.json" \
  --v3j_a_bounds "$V3J_EVID/bounded_action_space_bounds.json" \
  --a0_closeout "$V3L_EVID/v3l_a0_canonical_operator_closeout.json" \
  --operator_artifact_manifest "$V3L_EVID/v3l_a0_canonical_operator_artifact_manifest.json" \
  --density_artifact "$BASE/repos/ConvIR-B-haze4k-v5-v2-chd-rm-density-need-calibration/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2_density_need_calibration_20260708/artifacts/d3_density_only_head.pt" \
  --d7c_artifact "$BASE/repos/ConvIR-B-haze4k-v5-v2d-chd-rm-need-spatial-hard-negative/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2d_need_spatial_hard_negative_20260709/d7c_full/artifacts/d7c_mc_topk_hn_ordinal_head.pt" \
  --reference_oof_rows "$V3M_EVID/cloud_only_raw_common_action/v3l_a1_oracle_policy_oof_rows_cloud_only.csv" \
  --v3m_a0_source_manifest "$V3M_EVID/v3m_a0_source_manifest.json" \
  --output_dir "$OUT" --run_tag "$TAG" --run_mode "$MODE" --max_train_samples "$MAX_TRAIN" \
  --expected_canonical_blocks_sha256 "$CANONICAL_SHA256" \
  --expected_v3p_a0_script_sha256 "$V3P_A0_SCRIPT_SHA256" \
  --expected_v3p_source_commit "$V3P_SOURCE_COMMIT" \
  --expected_source_rows "$SOURCE_ROWS" \
  --expected_fresh_split_manifest_sha256 c8c00fefc965ded3389b6311fc67ea521e1f3174f27793688544abe09dc420e7 \
  --expected_parent_a0_closeout_sha256 2ca39ec1e17f4be794121603e3392a4e042e4d93b0e823454f7cf539f172d05d \
  --expected_parent_operator_manifest_sha256 1d2ffa499128ad08a272d67c5439583900afe8ef87fb3256193ad5fe21c3af84 \
  --expected_reference_oof_rows_sha256 b4a10184fab77b0045440dc88530d000a892acf2105a6295d5ad8a488c67ecb1 \
  --expected_v3m_a0_source_manifest_sha256 8966996c9c93f6f2f3fbdda536b69ea6aa03e1bf5432f127de47ca8ea95dd8a5 \
  --expected_density_artifact_sha256 1ffce13dccb41d96a47c2b5275f87bf2fdb73c226a190cfa240e5c71c1ec326f \
  --expected_d7c_artifact_sha256 09f449232024395cf64db15a2a0efa0f12d3e0e049e1da3d67229a3dc5729361 \
  --expected_a0_checkpoint_sha256 6f42037d57a4e3de3a10ac0ab909d66a3415864a19433c29204a975f4efa4088 \
  --expected_control_checkpoint_sha256 08207119a5cf9e5c439dd2cb81b99029ade1861f2739d31e75f2f9f78d57c0f2 \
  --source_split train --train_key v3j_controller_train --confirm_key v3j_route_confirm \
  --operator_labels D_ref D_rep --common_alphas 0 0.125 0.25 0.5 1 \
  --block_size 16 --grid_steps 64 --fold_count 5 --proj_channels 24 \
  --d7c_threshold 0.5773006677627563 --replay_tolerance_db 1e-6 \
  --seed 3407 --progress_every "$PROGRESS" --device cuda \
  --parent_evidence_main_commit f999591b8d6d09beef3079d51db6eac53d1ae302 \
  --runnable_source_commit "$EXPECTED_ROUTE_COMMIT" > "$LOG" 2>&1 &
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
  echo "V3R_A0_${MODE^^}_FAILED" | tee -a "$STATUS"
  exit "$rc"
fi

cp "$OUT/${TAG}_source_manifest.json" "$EVID_STAGE/${TAG}_source_manifest.json"
cp "$OUT/${TAG}_repair_type_summary.csv" "$EVID_STAGE/${TAG}_repair_type_summary.csv"
cp "$OUT/${TAG}_category_summary.csv" "$EVID_STAGE/${TAG}_category_summary.csv"
cp "$OUT/${TAG}_dual_operator_decision.json" "$EVID_STAGE/${TAG}_dual_operator_decision.json"
cp "$OUT/${TAG}_closeout.json" "$EVID_STAGE/${TAG}_closeout.json"
echo "V3R_A0_${MODE^^}_OK" | tee -a "$STATUS"
