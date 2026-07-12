#!/usr/bin/env bash
set -euo pipefail

REMOTE_REPO=${REMOTE_REPO:-/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v3s-delta-u-direction-repair-20260713}
RUN_ROOT=${RUN_ROOT:-/sda/home/wangyuxin/ConvIR-B/runs/haze4k_v5_chd_rm_v3s_delta_u_direction_repair_20260713}
PY=${PY:-/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python}
BASE=${BASE:-/sda/home/wangyuxin/ConvIR-B}
V3P_ROOT=${V3P_ROOT:-/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v3p-canonical-signed-gain-20260712}
V3M_ROOT=${V3M_ROOT:-/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v3m-blockwise-counterfactual-advantage-20260711}
V3L_ROOT=${V3L_ROOT:-/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v3l-safe-step-escalation-physics-audit-20260711}
GPU=${GPU:-1}
MODE=${MODE:?set MODE=noop_smoke|scout_train|formal_train|formal_eval}
EXPECTED_ROUTE_COMMIT=${EXPECTED_ROUTE_COMMIT:?set the exact v3s route commit before launch}

ROUTE_ID=haze4k_v5_chd_rm_v3s_delta_u_direction_repair_20260713
BRANCH=codex/haze4k-v5-v3s-delta-u-direction-repair-20260713
EXPECTED_LEGACY_COMMIT=555fd008e29f02128564f2fad41d0095ee44f5ea
EVID_STAGE=$REMOTE_REPO/experience_docx/experiment_logs/$ROUTE_ID
V3J_EVID=$V3M_ROOT/experience_docx/experiment_logs/haze4k_v5_chd_rm_v3j_bounded_safe_correction_audit_20260711
V3L_EVID=$V3L_ROOT/experience_docx/experiment_logs/haze4k_v5_chd_rm_v3l_safe_step_escalation_physics_audit_20260711
V3M_EVID=$V3M_ROOT/experience_docx/experiment_logs/haze4k_v5_chd_rm_v3m_blockwise_counterfactual_advantage_20260711
STATUS=$RUN_ROOT/status.txt
STAMP=$(date +%Y%m%dT%H%M%S)

case "$MODE" in
  noop_smoke)
    STAGE=v3s-S0-exact-noop-smoke
    RUN_TAG=v3s_s0_noop32
    OUT=$RUN_ROOT/s0_noop32
    ;;
  scout_train)
    STAGE=v3s-S1-fixed32-trainability-scout
    RUN_TAG=v3s_s1_scout32
    OUT=$RUN_ROOT/s1_scout32
    ;;
  formal_train)
    STAGE=v3s-S2-five-fold-train
    RUN_TAG=v3s_s2_train5fold
    OUT=$RUN_ROOT/s2_train5fold
    ;;
  formal_eval)
    STAGE=v3s-S3-canonical-float64-OOF-evaluation
    RUN_TAG=v3s_s3_eval
    OUT=$RUN_ROOT/s3_eval
    ;;
  *)
    echo "V3S_INVALID_MODE mode=$MODE"
    exit 2
    ;;
esac

test "$(git -C "$REMOTE_REPO" branch --show-current)" = "$BRANCH"
test "$(git -C "$REMOTE_REPO" rev-parse HEAD)" = "$EXPECTED_ROUTE_COMMIT"
test -z "$(git -C "$REMOTE_REPO" status --porcelain)"
test "$(git -C "$V3P_ROOT" rev-parse HEAD)" = "$EXPECTED_LEGACY_COMMIT"
test -z "$(git -C "$V3P_ROOT" status --porcelain)"
test -x "$PY"
test -d "$BASE/datasets/Haze4K/Haze4K/train/haze"
test -d "$BASE/datasets/Haze4K/Haze4K/train/gt"
test -s "$BASE/checkpoints/official/Haze4K/haze4k-base.pkl"
test -s "$BASE/repos/ConvIR-B-v3d-rarm-adapter-only-preflight/Dehazing/ITS/results/ConvIR-Haze4K-v3d-fam2modres-control-e5frome1-seed3407-20260710/Training-Results/Final.pkl"
test -s "$V3J_EVID/fresh_route_confirm_split_manifest.json"
test -s "$V3J_EVID/bounded_action_space_bounds.json"
test -s "$V3L_EVID/v3l_a0_canonical_operator_artifact_manifest.json"
test -s "$V3M_EVID/cloud_only_raw_common_action/v3l_a1_oracle_policy_oof_rows_cloud_only.csv"
test -s "$BASE/repos/ConvIR-B-haze4k-v5-v2-chd-rm-density-need-calibration/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2_density_need_calibration_20260708/artifacts/d3_density_only_head.pt"
test -s "$BASE/repos/ConvIR-B-haze4k-v5-v2d-chd-rm-need-spatial-hard-negative/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2d_need_spatial_hard_negative_20260709/d7c_full/artifacts/d7c_mc_topk_hn_ordinal_head.pt"
test ! -e "$OUT"
nvidia-smi -i "$GPU" --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader

case "$MODE" in
  scout_train)
    test -s "$EVID_STAGE/v3s_s0_noop32_closeout.json"
    "$PY" - "$EVID_STAGE/v3s_s0_noop32_closeout.json" <<'PY'
import json
import sys
closeout = json.load(open(sys.argv[1], encoding="utf-8"))
if closeout.get("decision") != "V3S_S0_EXACT_NOOP_PASS_AUTHORIZE_SCOUT_ONLY":
    raise SystemExit("v3s scout requires S0 exact-noop pass")
PY
    ;;
  formal_train)
    test -s "$EVID_STAGE/v3s_s1_scout32_closeout.json"
    "$PY" - "$EVID_STAGE/v3s_s1_scout32_closeout.json" <<'PY'
import json
import sys
closeout = json.load(open(sys.argv[1], encoding="utf-8"))
if closeout.get("decision") != "V3S_S1_TRAINABILITY_PASS_AUTHORIZE_FORMAL_TRAIN_ONLY":
    raise SystemExit("v3s formal train requires S1 scout pass")
PY
    ;;
  formal_eval)
    test -s "$EVID_STAGE/v3s_s2_train5fold_closeout.json"
    "$PY" - "$EVID_STAGE/v3s_s2_train5fold_closeout.json" <<'PY'
import json
import sys
closeout = json.load(open(sys.argv[1], encoding="utf-8"))
if closeout.get("decision") != "V3S_S2_TRAIN_COMPLETE_AUTHORIZE_FORMAL_OOF_EVAL_ONLY":
    raise SystemExit("v3s formal evaluation requires S2 training pass")
PY
    ;;
esac

mkdir -p "$RUN_ROOT" "$EVID_STAGE"
LOG=$RUN_ROOT/${RUN_TAG}_${STAMP}.log
echo "stage_start route=$ROUTE_ID stage=$STAGE run=$RUN_TAG time=$(date --iso-8601=seconds)" | tee -a "$STATUS"
echo "stage_paths repo=$REMOTE_REPO run_root=$RUN_ROOT evid_stage=$EVID_STAGE output=$OUT log=$LOG" | tee -a "$STATUS"

ARGS=(
  --mode "$MODE"
  --legacy_repo "$V3P_ROOT"
  --a0_checkpoint "$BASE/checkpoints/official/Haze4K/haze4k-base.pkl"
  --control_checkpoint "$BASE/repos/ConvIR-B-v3d-rarm-adapter-only-preflight/Dehazing/ITS/results/ConvIR-Haze4K-v3d-fam2modres-control-e5frome1-seed3407-20260710/Training-Results/Final.pkl"
  --data_dir "$BASE/datasets/Haze4K/Haze4K"
  --fresh_split_manifest "$V3J_EVID/fresh_route_confirm_split_manifest.json"
  --v3j_a_bounds "$V3J_EVID/bounded_action_space_bounds.json"
  --operator_artifact_manifest "$V3L_EVID/v3l_a0_canonical_operator_artifact_manifest.json"
  --density_artifact "$BASE/repos/ConvIR-B-haze4k-v5-v2-chd-rm-density-need-calibration/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2_density_need_calibration_20260708/artifacts/d3_density_only_head.pt"
  --d7c_artifact "$BASE/repos/ConvIR-B-haze4k-v5-v2d-chd-rm-need-spatial-hard-negative/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2d_need_spatial_hard_negative_20260709/d7c_full/artifacts/d7c_mc_topk_hn_ordinal_head.pt"
  --reference_oof_rows "$V3M_EVID/cloud_only_raw_common_action/v3l_a1_oracle_policy_oof_rows_cloud_only.csv"
  --expected_a0_checkpoint_sha256 6f42037d57a4e3de3a10ac0ab909d66a3415864a19433c29204a975f4efa4088
  --expected_control_checkpoint_sha256 08207119a5cf9e5c439dd2cb81b99029ade1861f2739d31e75f2f9f78d57c0f2
  --expected_density_artifact_sha256 1ffce13dccb41d96a47c2b5275f87bf2fdb73c226a190cfa240e5c71c1ec326f
  --expected_d7c_artifact_sha256 09f449232024395cf64db15a2a0efa0f12d3e0e049e1da3d67229a3dc5729361
  --expected_fresh_split_manifest_sha256 c8c00fefc965ded3389b6311fc67ea521e1f3174f27793688544abe09dc420e7
  --expected_operator_manifest_sha256 1d2ffa499128ad08a272d67c5439583900afe8ef87fb3256193ad5fe21c3af84
  --expected_reference_oof_rows_sha256 b4a10184fab77b0045440dc88530d000a892acf2105a6295d5ad8a488c67ecb1
  --expected_v3j_a_bounds_sha256 485ea12ff14c33b87105a50b6d118a9937c7e7f1b113062fe03d91eef3c9cc21
  --output_dir "$OUT"
  --run_tag "$RUN_TAG"
  --seed 3407
  --device cuda
  --expected_legacy_commit "$EXPECTED_LEGACY_COMMIT"
)

if [ "$MODE" = formal_eval ]; then
  ARGS+=(--train_closeout "$EVID_STAGE/v3s_s2_train5fold_closeout.json")
fi

set +e
CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 "$PY" \
  "$REMOTE_REPO/experience_docx/tools/chd_rm_v3s_delta_u_direction_repair.py" \
  "${ARGS[@]}" 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e
echo "stage_done route=$ROUTE_ID stage=$STAGE rc=$rc time=$(date --iso-8601=seconds)" | tee -a "$STATUS"
if [ "$rc" -ne 0 ]; then
  echo "V3S_${MODE^^}_FAILED" | tee -a "$STATUS"
  exit "$rc"
fi

cp "$OUT/${RUN_TAG}_source_manifest.json" "$EVID_STAGE/${RUN_TAG}_source_manifest.json"
cp "$OUT/${RUN_TAG}_closeout.json" "$EVID_STAGE/${RUN_TAG}_closeout.json"
case "$MODE" in
  scout_train|formal_train)
    cp "$OUT/${RUN_TAG}_history.csv" "$EVID_STAGE/${RUN_TAG}_history.csv"
    ;;
  formal_eval)
    cp "$OUT/${RUN_TAG}_operator_summary.csv" "$EVID_STAGE/${RUN_TAG}_operator_summary.csv"
    cp "$OUT/${RUN_TAG}_canonical_g1_contract.json" "$EVID_STAGE/${RUN_TAG}_canonical_g1_contract.json"
    ;;
esac
echo "V3S_${MODE^^}_OK" | tee -a "$STATUS"
