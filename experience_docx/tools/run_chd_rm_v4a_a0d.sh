#!/usr/bin/env bash
set -euo pipefail

ROUTE_ID=haze4k_v5_chd_rm_v4a_conditional_safety_audit_20260714
REMOTE_REPO=${REMOTE_REPO:-/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v4a-conditional-safety-audit-20260714}
RUN_ROOT=${RUN_ROOT:-/sda/home/wangyuxin/ConvIR-B/runs/$ROUTE_ID}
PY=${PY:-/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python}
BASE=${BASE:-/sda/home/wangyuxin/ConvIR-B}
V3Z_ROOT=${V3Z_ROOT:-$BASE/repos/ConvIR-B-v3z-source-3caddcc5265732e5be77e3404119a28cb28c11e6}
V3S_ROOT=${V3S_ROOT:-$BASE/repos/ConvIR-B-v3s-source-2860f580bb25cc75ec9ade56378af6d77f5c8d8b}
V3P_ROOT=${V3P_ROOT:-$BASE/repos/ConvIR-B-v3p-source-555fd008e29f02128564f2fad41d0095ee44f5ea}
V3M_ROOT=${V3M_ROOT:-$BASE/repos/ConvIR-B-v3m-blockwise-counterfactual-advantage-20260711}
V3L_ROOT=${V3L_ROOT:-$BASE/repos/ConvIR-B-v3l-safe-step-escalation-physics-audit-20260711}
GPU=${GPU:-1}
RUN_ID=${RUN_ID:-a0d_r1}
EXPECTED_ROUTE_COMMIT=${EXPECTED_ROUTE_COMMIT:?set exact amended v4a route commit}
EXPECTED_V3Z_COMMIT=3caddcc5265732e5be77e3404119a28cb28c11e6
EXPECTED_V3S_COMMIT=2860f580bb25cc75ec9ade56378af6d77f5c8d8b
EXPECTED_V3P_COMMIT=555fd008e29f02128564f2fad41d0095ee44f5ea

OUT=$RUN_ROOT/$RUN_ID
STATUS=$RUN_ROOT/status.txt
EVID_STAGE=$REMOTE_REPO/experience_docx/experiment_logs/$ROUTE_ID
RUNNER=$REMOTE_REPO/experience_docx/tools/chd_rm_v4a_a0d_audit.py
A0R_ROOT=$RUN_ROOT/a0r_r2
A0R_CLOSEOUT=$A0R_ROOT/v4a_a0r_closeout.json
A0R_FINAL_STATE=$A0R_ROOT/r1/trace/states/epoch16_update512_final.pt
V3J_EVID=$V3M_ROOT/experience_docx/experiment_logs/haze4k_v5_chd_rm_v3j_bounded_safe_correction_audit_20260711
V3L_EVID=$V3L_ROOT/experience_docx/experiment_logs/haze4k_v5_chd_rm_v3l_safe_step_escalation_physics_audit_20260711
V3M_EVID=$V3M_ROOT/experience_docx/experiment_logs/haze4k_v5_chd_rm_v3m_blockwise_counterfactual_advantage_20260711

test "$(git -C "$REMOTE_REPO" branch --show-current)" = "codex/haze4k-v5-v4a-conditional-safety-audit-20260714"
test "$(git -C "$REMOTE_REPO" rev-parse HEAD)" = "$EXPECTED_ROUTE_COMMIT"
test -z "$(git -C "$REMOTE_REPO" status --porcelain)"
test "$(git -C "$V3Z_ROOT" rev-parse HEAD)" = "$EXPECTED_V3Z_COMMIT"
test -z "$(git -C "$V3Z_ROOT" status --porcelain)"
test "$(git -C "$V3S_ROOT" rev-parse HEAD)" = "$EXPECTED_V3S_COMMIT"
test -z "$(git -C "$V3S_ROOT" status --porcelain)"
test "$(git -C "$V3P_ROOT" rev-parse HEAD)" = "$EXPECTED_V3P_COMMIT"
test -z "$(git -C "$V3P_ROOT" status --porcelain)"
test -x "$PY"
test -f "$RUNNER"
test -f "$A0R_CLOSEOUT"
test -f "$A0R_FINAL_STATE"
test ! -e "$OUT"
test -d "$BASE/datasets/Haze4K/Haze4K/train/haze"
test -d "$BASE/datasets/Haze4K/Haze4K/train/gt"
test -s "$BASE/checkpoints/official/Haze4K/haze4k-base.pkl"
test -s "$V3J_EVID/fresh_route_confirm_split_manifest.json"
test -s "$V3J_EVID/bounded_action_space_bounds.json"
test -s "$V3L_EVID/v3l_a0_canonical_operator_artifact_manifest.json"
test -s "$V3M_EVID/cloud_only_raw_common_action/v3l_a1_oracle_policy_oof_rows_cloud_only.csv"
test -s "$BASE/repos/ConvIR-B-v3d-rarm-adapter-only-preflight/Dehazing/ITS/results/ConvIR-Haze4K-v3d-fam2modres-control-e5frome1-seed3407-20260710/Training-Results/Final.pkl"
test -s "$BASE/repos/ConvIR-B-haze4k-v5-v2-chd-rm-density-need-calibration/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2_density_need_calibration_20260708/artifacts/d3_density_only_head.pt"
test -s "$BASE/repos/ConvIR-B-haze4k-v5-v2d-chd-rm-need-spatial-hard-negative/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2d_need_spatial_hard_negative_20260709/d7c_full/artifacts/d7c_mc_topk_hn_ordinal_head.pt"
nvidia-smi -i "$GPU" --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader

mkdir -p "$RUN_ROOT" "$EVID_STAGE"
STAMP=$(date +%Y%m%dT%H%M%S)
LOG=$RUN_ROOT/v4a_a0d_${STAMP}.log
echo "stage_start route=$ROUTE_ID stage=v4a-A0D-descriptive-risk-decomposition run=$RUN_ID time=$(date --iso-8601=seconds)" | tee -a "$STATUS"
echo "stage_paths repo=$REMOTE_REPO run_root=$RUN_ROOT evid_stage=$EVID_STAGE output=$OUT log=$LOG" | tee -a "$STATUS"

COMMON_ARGS=(
  --v3s_root "$V3S_ROOT" --expected_v3s_commit "$EXPECTED_V3S_COMMIT"
  --v3p_root "$V3P_ROOT" --expected_v3p_commit "$EXPECTED_V3P_COMMIT"
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
  --sample_count 128 --seed 3407 --device cuda
)

set +e
CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 "$PY" "$RUNNER" audit \
  --v3z-root "$V3Z_ROOT" --a0r-closeout "$A0R_CLOSEOUT" --a0r-final-state "$A0R_FINAL_STATE" \
  --mode projected --output_dir "$OUT" --run_tag v4a_a0d "${COMMON_ARGS[@]}" 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
set -e

echo "stage_done route=$ROUTE_ID stage=v4a-A0D-descriptive-risk-decomposition run=$RUN_ID rc=$rc time=$(date --iso-8601=seconds)" | tee -a "$STATUS"
if [ "$rc" -ne 0 ]; then
  echo V4A_A0D_FAILED_COMMAND_OR_INFRA | tee -a "$STATUS"
  exit "$rc"
fi

cp "$OUT/v4a_a0d_group_tail_summary.csv" "$EVID_STAGE/"
cp "$OUT/v4a_a0d_closeout.json" "$EVID_STAGE/"
if grep -q '"state": "COMPLETED_GATE_PASS"' "$OUT/v4a_a0d_closeout.json"; then
  echo V4A_A0D_OK | tee -a "$STATUS"
else
  echo V4A_A0D_GATE_FAIL | tee -a "$STATUS"
fi
