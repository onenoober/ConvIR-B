#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT=${REMOTE_ROOT:-/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v3o-signed-adjacent-advantage-20260712}
V3M_ROOT=${V3M_ROOT:-/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v3m-blockwise-counterfactual-advantage-20260711}
V3L_ROOT=${V3L_ROOT:-/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v3l-safe-step-escalation-physics-audit-20260711}
BASE=${BASE:-/sda/home/wangyuxin/ConvIR-B}
PY=${PY:-/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python}
GPU=${GPU:-0}
MODE=${MODE:-smoke}
ROUTE_ID=haze4k_v5_chd_rm_v3o_signed_adjacent_advantage_identifiability_20260712
V3M_ROUTE_ID=haze4k_v5_chd_rm_v3m_blockwise_counterfactual_advantage_20260711
V3L_ROUTE_ID=haze4k_v5_chd_rm_v3l_safe_step_escalation_physics_audit_20260711
V3J_ROUTE_ID=haze4k_v5_chd_rm_v3j_bounded_safe_correction_audit_20260711
BRANCH=codex/haze4k-v5-v3o-signed-adjacent-advantage-identifiability
EVID="$REMOTE_ROOT/experience_docx/experiment_logs/$ROUTE_ID"
V3M_EVID="$V3M_ROOT/experience_docx/experiment_logs/$V3M_ROUTE_ID"
V3L_EVID="$V3L_ROOT/experience_docx/experiment_logs/$V3L_ROUTE_ID"
V3J_EVID="$V3M_ROOT/experience_docx/experiment_logs/$V3J_ROUTE_ID"
STATUS="$EVID/status.txt"
STAMP=$(date +%Y%m%dT%H%M%S)

case "$MODE" in
  smoke)
    OUT="$EVID/a0_smoke32"
    TAG=v3o_a0_smoke32
    MAX_TRAIN=32
    PROGRESS=8
    LOG="$EVID/v3o_a0_smoke32_${STAMP}.log"
    ;;
  formal)
    OUT="$EVID"
    TAG=v3o_a0
    MAX_TRAIN=1200
    PROGRESS=25
    LOG="$EVID/v3o_a0_formal_${STAMP}.log"
    ;;
  *)
    echo "V3O_A0_INVALID_MODE $MODE" | tee -a "$STATUS"
    exit 2
    ;;
esac

echo "v3o_a0_${MODE}_start $ROUTE_ID $(date --iso-8601=seconds)" | tee -a "$STATUS"
echo "v3o_a0_${MODE}_log $LOG gpu=$GPU" | tee -a "$STATUS"
cd "$REMOTE_ROOT"
test "$(git branch --show-current)" = "$BRANCH"
test "$(git rev-parse HEAD)" = "$(git rev-parse --verify HEAD)"
test -x "$PY"
test -s "$V3J_EVID/fresh_route_confirm_split_manifest.json"
test -s "$V3J_EVID/bounded_action_space_bounds.json"
test -s "$V3L_EVID/v3l_a0_canonical_operator_closeout.json"
test -s "$V3L_EVID/v3l_a0_canonical_operator_artifact_manifest.json"
test -s "$V3M_EVID/cloud_only_raw_common_action/v3l_a1_oracle_policy_oof_rows_cloud_only.csv"
test -s "$V3M_EVID/v3m_a0_source_manifest.json"
test -s "$BASE/checkpoints/official/Haze4K/haze4k-base.pkl"
test -s "$BASE/repos/ConvIR-B-v3d-rarm-adapter-only-preflight/Dehazing/ITS/results/ConvIR-Haze4K-v3d-fam2modres-control-e5frome1-seed3407-20260710/Training-Results/Final.pkl"
test ! -e "$OUT/${TAG}_block_candidate_losses_cloud_only.csv"
if [ "$MODE" = formal ]; then
  SMOKE="$EVID/a0_smoke32/v3o_a0_smoke32_replay_integrity.json"
  test -s "$SMOKE"
  "$PY" - "$SMOKE" <<'PY'
import json
import sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
if value["decision"] != "V3O_A0_SMOKE_REPLAY_INTEGRITY_PASS_AUTHORIZE_FORMAL_OOF_ONLY":
    raise SystemExit("formal A0 requires a passing v3o-A0 smoke integrity gate")
PY
fi
nvidia-smi -i "$GPU" --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader

set +e
CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 "$PY" experience_docx/tools/chd_rm_v3o_a0_candidate_gain_audit.py \
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
  --expected_fresh_split_manifest_sha256 c8c00fefc965ded3389b6311fc67ea521e1f3174f27793688544abe09dc420e7 \
  --expected_parent_a0_closeout_sha256 2ca39ec1e17f4be794121603e3392a4e042e4d93b0e823454f7cf539f172d05d \
  --expected_parent_operator_manifest_sha256 1d2ffa499128ad08a272d67c5439583900afe8ef87fb3256193ad5fe21c3af84 \
  --expected_reference_oof_rows_sha256 b4a10184fab77b0045440dc88530d000a892acf2105a6295d5ad8a488c67ecb1 \
  --expected_v3m_a0_source_manifest_sha256 8966996c9c93f6f2f3fbdda536b69ea6aa03e1bf5432f127de47ca8ea95dd8a5 \
  --expected_density_artifact_sha256 1ffce13dccb41d96a47c2b5275f87bf2fdb73c226a190cfa240e5c71c1ec326f \
  --expected_d7c_artifact_sha256 09f449232024395cf64db15a2a0efa0f12d3e0e049e1da3d67229a3dc5729361 \
  --expected_a0_checkpoint_sha256 6f42037d57a4e3de3a10ac0ab909d66a3415864a19433c29204a975f4efa4088 \
  --expected_control_checkpoint_sha256 08207119a5cf9e5c439dd2cb81b99029ade1861f2739d31e75f2f9f78d57c0f2 \
  --block_size 16 --fold_count 5 --proj_channels 24 --operator_labels D_ref D_rep \
  --common_alphas 0 0.125 0.25 0.5 1 --replay_tolerance_db 1e-6 --mse_aggregation_tolerance 1e-10 \
  --seed 3407 --progress_every "$PROGRESS" --parent_evidence_main_commit 4d097680160e4f6b433b5e6fb62d31df46c28415 \
  --runnable_source_commit 2c9cb511627895981c4c489cacd990326185ced6 \
  > "$LOG" 2>&1
rc=$?
set -e
echo "v3o_a0_${MODE}_done rc=$rc $ROUTE_ID $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [ "$rc" -eq 0 ]; then
  echo "V3O_A0_${MODE^^}_OK" | tee -a "$STATUS"
else
  echo "V3O_A0_${MODE^^}_FAILED" | tee -a "$STATUS"
fi
exit "$rc"
