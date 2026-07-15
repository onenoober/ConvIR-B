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
SHELL_RUNNER=$0
EVALUATOR=$REMOTE_REPO/experience_docx/tools/chd_rm_v4a_a1c_safe_action_interface_ceiling_v2.py
CARD=$REMOTE_REPO/experience_docx/experiment_cards/2026-07-15-haze4k-v5-v4a-a1c-safe-action-interface-ceiling.md
AUTH0=$REMOTE_REPO/experience_docx/experiment_logs/$ROUTE_ID/initial_authorization.json
S0_CLOSEOUT=$REMOTE_REPO/experience_docx/experiment_logs/$ROUTE_ID/v4a_a1c_s0_closeout.json
OUT=$RUN_ROOT/$RUN_ID
STATUS=$RUN_ROOT/status.txt
EVID=$REMOTE_REPO/experience_docx/experiment_logs/$ROUTE_ID
V3Z_ROOT=$BASE/repos/ConvIR-B-v3z-source-3caddcc5265732e5be77e3404119a28cb28c11e6
V3S_ROOT=$BASE/repos/ConvIR-B-v3s-source-2860f580bb25cc75ec9ade56378af6d77f5c8d8b
V3P_ROOT=$BASE/repos/ConvIR-B-v3p-source-555fd008e29f02128564f2fad41d0095ee44f5ea
V3M_ROOT=$BASE/repos/ConvIR-B-v3m-blockwise-counterfactual-advantage-20260711
V3L_ROOT=$BASE/repos/ConvIR-B-v3l-safe-step-escalation-physics-audit-20260711
A1F_MODULE=$BASE/repos/ConvIR-B-v4a-a1f-deltau-action-feasibility-20260714/experience_docx/tools/chd_rm_v4a_a1f_action_feasibility.py
A0R_TRACE=$BASE/runs/haze4k_v5_chd_rm_v4a_conditional_safety_audit_20260714/a0r_r2/r1/trace
V3J_EVID=$V3M_ROOT/experience_docx/experiment_logs/haze4k_v5_chd_rm_v3j_bounded_safe_correction_audit_20260711
V3L_EVID=$V3L_ROOT/experience_docx/experiment_logs/haze4k_v5_chd_rm_v3l_safe_step_escalation_physics_audit_20260711

mkdir -p "$RUN_ROOT" "$EVID"
LOG=$RUN_ROOT/v4a_a1c_${MODE}_$(date +%Y%m%dT%H%M%S).log
exec > >(tee -a "$LOG") 2>&1

status() {
  printf '%s\n' "$1" >> "$STATUS"
  printf '%s\n' "$1"
}

preflight_error() {
  local rc=$?
  status "command_engineering_failure route=$ROUTE_ID mode=$MODE phase=preflight rc=$rc command=$BASH_COMMAND"
  status "V4A_A1C_${MODE^^}_COMMAND_ENGINEERING_FAILURE_REQUIRES_CLASSIFICATION"
  exit "$rc"
}

trap preflight_error ERR
status "heartbeat route=$ROUTE_ID mode=$MODE phase=preflight_setup"

case "$MODE" in s0|formal) ;; *) echo "invalid sealed MODE=$MODE" >&2; exit 2;; esac
test "$(git -C "$REMOTE_REPO" branch --show-current)" = codex/haze4k-v5-v4a-a1c-safe-action-interface-ceiling-20260715
test "$(git -C "$REMOTE_REPO" rev-parse HEAD)" = "$EXPECTED_ROUTE_COMMIT"
test -z "$(git -C "$REMOTE_REPO" status --porcelain)"
test -x "$PY"; test -f "$SHELL_RUNNER" -a -f "$EVALUATOR" -a -f "$CARD" -a -f "$AUTH0"
test "$(sha256sum "$SHELL_RUNNER" | awk '{print $1}')" = "$RUNNER_SHA256"
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
status "heartbeat route=$ROUTE_ID mode=$MODE phase=preflight gpu=$GPU free_mib=$GPU_FREE util=$GPU_UTIL"
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
  --mode projected --output_dir "$OUT" --run_tag "$RUN_ID"
)
trap - ERR
set +e
CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 "$PY" "$EVALUATOR" audit --a1c-stage "$MODE" --v3z-root "$V3Z_ROOT" --a1f-module "$A1F_MODULE" --a0r-trace-dir "$A0R_TRACE" --expected-route-commit "$EXPECTED_ROUTE_COMMIT" --expected-route-card-sha256 "$(sha256sum "$CARD" | awk '{print $1}')" --runner-sha256 "$RUNNER_SHA256" --status-file "$STATUS" "${COMMON_ARGS[@]}" 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}; set -e
printf 'stage_done route=%s mode=%s run=%s rc=%s\n' "$ROUTE_ID" "$MODE" "$RUN_ID" "$rc" | tee -a "$STATUS"
if [ "$rc" -ne 0 ]; then echo "V4A_A1C_${MODE^^}_FAILED_RUNTIME_REQUIRES_CLASSIFICATION" | tee -a "$STATUS"; exit "$rc"; fi
if [ "$MODE" = s0 ]; then DEST=$S0_CLOSEOUT; else DEST=$EVID/v4a_a1c_formal_closeout.json; fi
cp "$OUT/v4a_a1c_closeout.json" "$DEST"
cp "$OUT/v4a_a1c_source_manifest.json" "$EVID/v4a_a1c_${MODE}_source_manifest.json"
cp "$OUT/v4a_a1c_bootstrap_summary.json" "$EVID/v4a_a1c_${MODE}_bootstrap_summary.json"
echo "V4A_A1C_${MODE^^}_OK" | tee -a "$STATUS"
