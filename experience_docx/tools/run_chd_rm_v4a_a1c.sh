#!/usr/bin/env bash
set -euo pipefail

ROUTE_ID=haze4k_v5_chd_rm_v4a_a1c_safe_action_interface_ceiling_20260715
REMOTE_REPO=${REMOTE_REPO:?set by convir-ops}
RUN_ROOT=${RUN_ROOT:?set by convir-ops}
RUN_ID=${RUN_ID:?set by convir-ops}
MODE=${MODE:?set by convir-ops}
EXPECTED_ROUTE_COMMIT=${EXPECTED_ROUTE_COMMIT:?set by convir-ops}
RUNNER_SHA256=${RUNNER_SHA256:?set by convir-ops}
GPU=${GPU:?set by convir-ops}
PY=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
BASE=/sda/home/wangyuxin/ConvIR-B
V3Z_ROOT=$BASE/repos/ConvIR-B-v3z-source-3caddcc5265732e5be77e3404119a28cb28c11e6
V3S_ROOT=$BASE/repos/ConvIR-B-v3s-source-2860f580bb25cc75ec9ade56378af6d77f5c8d8b
V3P_ROOT=$BASE/repos/ConvIR-B-v3p-source-555fd008e29f02128564f2fad41d0095ee44f5ea
V3M_ROOT=$BASE/repos/ConvIR-B-v3m-blockwise-counterfactual-advantage-20260711
V3L_ROOT=$BASE/repos/ConvIR-B-v3l-safe-step-escalation-physics-audit-20260711
A1F_ROOT=$BASE/repos/ConvIR-B-v4a-a1f-deltau-action-feasibility-20260714
A0R_TRACE=$BASE/runs/haze4k_v5_chd_rm_v4a_conditional_safety_audit_20260714/a0r_r2/r1/trace
RUNNER=$REMOTE_REPO/experience_docx/tools/chd_rm_v4a_a1c_safe_action_interface_ceiling.py
PARENT=$A1F_ROOT/experience_docx/tools/chd_rm_v4a_a1f_action_feasibility.py
EVID_STAGE=$REMOTE_REPO/experience_docx/experiment_logs/$ROUTE_ID
STATUS=$RUN_ROOT/status.txt
OUT=$RUN_ROOT/$RUN_ID
CARD=$REMOTE_REPO/experience_docx/experiment_cards/2026-07-15-haze4k-v5-v4a-a1c-safe-action-interface-ceiling.md
A1R_REVIEW=$REMOTE_REPO/experience_docx/experiment_logs/haze4k_v5_chd_rm_v4a_a1r_representation_sufficiency_20260714/v4a_a1r_r3_review.json
V3J_EVID=$V3M_ROOT/experience_docx/experiment_logs/haze4k_v5_chd_rm_v3j_bounded_safe_correction_audit_20260711
V3L_EVID=$V3L_ROOT/experience_docx/experiment_logs/haze4k_v5_chd_rm_v3l_safe_step_escalation_physics_audit_20260711

case "$MODE" in
  s0) STAGE=smoke; INTERFACE=exact_half ;;
  exact-half) STAGE=formal; INTERFACE=exact_half ;;
  half-aa) STAGE=formal; INTERFACE=half_aa ;;
  full) STAGE=formal; INTERFACE=full ;;
  *) echo "invalid MODE=$MODE" >&2; exit 2 ;;
esac

test "$(git -C "$REMOTE_REPO" branch --show-current)" = "codex/haze4k-v5-v4a-a1c-safe-action-interface-ceiling-20260715"
test "$(git -C "$REMOTE_REPO" rev-parse HEAD)" = "$EXPECTED_ROUTE_COMMIT"
test -z "$(git -C "$REMOTE_REPO" status --porcelain)"
test "$(git -C "$V3Z_ROOT" rev-parse HEAD)" = 3caddcc5265732e5be77e3404119a28cb28c11e6
test "$(git -C "$V3S_ROOT" rev-parse HEAD)" = 2860f580bb25cc75ec9ade56378af6d77f5c8d8b
test "$(git -C "$V3P_ROOT" rev-parse HEAD)" = 555fd008e29f02128564f2fad41d0095ee44f5ea
test "$(git -C "$A1F_ROOT" rev-parse HEAD)" = d4f8d0936869c822ae19b5c21172efc2eb973dd8
test -z "$(git -C "$V3Z_ROOT" status --porcelain)"
test -z "$(git -C "$V3S_ROOT" status --porcelain)"
test -z "$(git -C "$V3P_ROOT" status --porcelain)"
test -z "$(git -C "$A1F_ROOT" status --porcelain)"
test -x "$PY"
test -f "$RUNNER" -a -f "$PARENT" -a -f "$CARD" -a -f "$A1R_REVIEW"
test "$(sha256sum "$RUNNER" | awk '{print $1}')" = "$RUNNER_SHA256"
test "$(sha256sum "$PARENT" | awk '{print $1}')" = c7d382ee9519e307e8207af725ffd4a760bb8018819d6d1dc08eafe4dda3dce4
test "$(sha256sum "$A1R_REVIEW" | awk '{print $1}')" = 923ec693b089d01a0fa1e9950b56d30647ef9da70f147287e7ed6dcb040c8994
test -f "$A0R_TRACE/trace_manifest.json"
test ! -e "$OUT"
test -d "$BASE/datasets/Haze4K/Haze4K/train/haze" -a -d "$BASE/datasets/Haze4K/Haze4K/train/gt"
test -s "$BASE/checkpoints/official/Haze4K/haze4k-base.pkl"
test -s "$V3J_EVID/fresh_route_confirm_split_manifest.json"
test -s "$V3J_EVID/bounded_action_space_bounds.json"
test -s "$V3L_EVID/v3l_a0_canonical_operator_artifact_manifest.json"

CARD_SHA=$(sha256sum "$CARD" | awk '{print $1}')
GPU_STATUS=$(nvidia-smi -i "$GPU" --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits)
GPU_FREE_MIB=$(printf '%s\n' "$GPU_STATUS" | awk -F, '{gsub(/ /,"",$1); print $1}')
GPU_UTIL=$(printf '%s\n' "$GPU_STATUS" | awk -F, '{gsub(/ /,"",$2); print $2}')
test "$GPU_FREE_MIB" -ge 12000
test "$GPU_UTIL" -le 10
mkdir -p "$RUN_ROOT" "$EVID_STAGE"
LOG=$RUN_ROOT/v4a_a1c_${MODE}_$(date +%Y%m%dT%H%M%S).log
echo "stage_start route=$ROUTE_ID stage=$STAGE mode=$MODE run=$RUN_ID time=$(date --iso-8601=seconds)" | tee -a "$STATUS"
echo "heartbeat route=$ROUTE_ID phase=preflight gpu=$GPU free_mib=$GPU_FREE_MIB util=$GPU_UTIL" | tee -a "$STATUS"

COMMON_ARGS=(
  --v3s_root "$V3S_ROOT" --expected_v3s_commit 2860f580bb25cc75ec9ade56378af6d77f5c8d8b
  --v3p_root "$V3P_ROOT" --expected_v3p_commit 555fd008e29f02128564f2fad41d0095ee44f5ea
  --a0_checkpoint "$BASE/checkpoints/official/Haze4K/haze4k-base.pkl"
  --control_checkpoint "$BASE/repos/ConvIR-B-v3d-rarm-adapter-only-preflight/Dehazing/ITS/results/ConvIR-Haze4K-v3d-fam2modres-control-e5frome1-seed3407-20260710/Training-Results/Final.pkl"
  --data_dir "$BASE/datasets/Haze4K/Haze4K" --fresh_split_manifest "$V3J_EVID/fresh_route_confirm_split_manifest.json"
  --v3j_a_bounds "$V3J_EVID/bounded_action_space_bounds.json" --operator_artifact_manifest "$V3L_EVID/v3l_a0_canonical_operator_artifact_manifest.json"
  --density_artifact "$BASE/repos/ConvIR-B-haze4k-v5-v2-chd-rm-density-need-calibration/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2_density_need_calibration_20260708/artifacts/d3_density_only_head.pt"
  --d7c_artifact "$BASE/repos/ConvIR-B-haze4k-v5-v2d-chd-rm-need-spatial-hard-negative/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2d_need_spatial_hard_negative_20260709/d7c_full/artifacts/d7c_mc_topk_hn_ordinal_head.pt"
  --reference_oof_rows "$V3M_ROOT/experience_docx/experiment_logs/haze4k_v5_chd_rm_v3m_blockwise_counterfactual_advantage_20260711/cloud_only_raw_common_action/v3l_a1_oracle_policy_oof_rows_cloud_only.csv"
  --expected_a0_checkpoint_sha256 6f42037d57a4e3de3a10ac0ab909d66a3415864a19433c29204a975f4efa4088
  --expected_control_checkpoint_sha256 08207119a5cf9e5c439dd2cb81b99029ade1861f2739d31e75f2f9f78d57c0f2
  --expected_density_artifact_sha256 1ffce13dccb41d96a47c2b5275f87bf2fdb73c226a190cfa240e5c71c1ec326f
  --expected_d7c_artifact_sha256 09f449232024395cf64db15a2a0efa0f12d3e0e049e1da3d67229a3dc5729361
  --expected_fresh_split_manifest_sha256 c8c00fefc965ded3389b6311fc67ea521e1f3174f27793688544abe09dc420e7
  --expected_operator_manifest_sha256 1d2ffa499128ad08a272d67c5439583900afe8ef87fb3256193ad5fe21c3af84
  --expected_reference_oof_rows_sha256 b4a10184fab77b0045440dc88530d000a892acf2105a6295d5ad8a488c67ecb1
  --expected_v3j_a_bounds_sha256 485ea12ff14c33b87105a50b6d118a9937c7e7f1b113062fe03d91eef3c9cc21
  --sample_count 128 --epochs 16 --risk_window 4 --warmup_epochs 8 --learning_rate 0.0005 --weight_decay 0.00001 --grad_clip_norm 0.1 --cvar_fraction 0.25 --seed 3407 --device cuda
)

set +e
CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 "$PY" "$RUNNER" audit --v3z-root "$V3Z_ROOT" --a1f-module "$PARENT" --a0r-trace-dir "$A0R_TRACE" --a1r-review "$A1R_REVIEW" --expected-a1r-review-sha256 923ec693b089d01a0fa1e9950b56d30647ef9da70f147287e7ed6dcb040c8994 --expected-route-commit "$EXPECTED_ROUTE_COMMIT" --expected-route-card-sha256 "$CARD_SHA" --runner-sha256 "$RUNNER_SHA256" --status-file "$STATUS" --a1c-stage "$STAGE" --interface "$INTERFACE" --mode projected --output_dir "$OUT" --run_tag "$RUN_ID" "${COMMON_ARGS[@]}" 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
set -e
echo "stage_done route=$ROUTE_ID stage=$STAGE mode=$MODE run=$RUN_ID rc=$rc time=$(date --iso-8601=seconds)" | tee -a "$STATUS"
if [ "$rc" -ne 0 ]; then echo "V4A_A1C_${MODE^^}_FAILED_RUNTIME_REQUIRES_CLASSIFICATION" | tee -a "$STATUS"; exit "$rc"; fi
cp "$OUT/v4a_a1c_closeout.json" "$EVID_STAGE/v4a_a1c_s0_closeout.json"
cp "$OUT/v4a_a1c_source_manifest.json" "$EVID_STAGE/v4a_a1c_s0_source_manifest.json"
cp "$OUT/v4a_a1c_operator_summary.csv" "$EVID_STAGE/v4a_a1c_s0_operator_summary.csv"
cp "$OUT/v4a_a1c_bootstrap_summary.json" "$EVID_STAGE/v4a_a1c_s0_bootstrap_summary.json"
echo "state=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["state"])' "$OUT/v4a_a1c_closeout.json") decision=$("$PY" -c 'import json,sys; x=json.load(open(sys.argv[1])); print(x["decision"])' "$OUT/v4a_a1c_closeout.json") authorizes=$("$PY" -c 'import json,sys; x=json.load(open(sys.argv[1])); print(x["authorizes"])' "$OUT/v4a_a1c_closeout.json")" | tee -a "$STATUS"
echo "V4A_A1C_${MODE^^}_OK" | tee -a "$STATUS"
