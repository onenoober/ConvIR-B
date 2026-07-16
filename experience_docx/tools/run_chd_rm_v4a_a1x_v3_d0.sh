#!/usr/bin/env bash
set -euo pipefail
ROUTE_ID=haze4k_v5_chd_rm_v4a_a1x_accessibility_v3_20260716
BASE=/sda/home/wangyuxin/ConvIR-B
PY=$BASE/envs/convir-cu121/bin/python
ENTRY=$REMOTE_REPO/experience_docx/tools/chd_rm_v4a_a1x_v3_d0.py
A1R=$BASE/repos/ConvIR-B-v4a-a1r-representation-sufficiency-20260714
A1F=$BASE/repos/ConvIR-B-v4a-a1f-deltau-action-feasibility-20260714
V3Z=$BASE/repos/ConvIR-B-v3z-source-3caddcc5265732e5be77e3404119a28cb28c11e6
V3S=$BASE/repos/ConvIR-B-v3s-source-2860f580bb25cc75ec9ade56378af6d77f5c8d8b
V3P=$BASE/repos/ConvIR-B-v3p-source-555fd008e29f02128564f2fad41d0095ee44f5ea
V3M=$BASE/repos/ConvIR-B-v3m-blockwise-counterfactual-advantage-20260711
V3L=$BASE/repos/ConvIR-B-v3l-safe-step-escalation-physics-audit-20260711
EVID=$REMOTE_REPO/experience_docx/experiment_logs/$ROUTE_ID
WORK=$OUTPUT_PATH/workload
STATUS=$OUTPUT_PATH/status.txt
LOG=$OUTPUT_PATH/runtime.log
CLOSEOUT_PATH=$OUTPUT_PATH/a1x_v3_d0_closeout.json
export REMOTE_REPO RUN_ID RUNNER_SHA256 CLOSEOUT_PATH
mkdir -p "$OUTPUT_PATH" "$EVID"
on_exit() {
  rc=$?
  trap - EXIT
  if [[ $rc -ne 0 && ! -f "$CLOSEOUT_PATH" ]]; then
    printf '{"authorizes":"NONE","canary_touched":false,"confirmation_images_targets_outcomes_touched":false,"decision":null,"evidence_role":"development_screening","locked_test_touched":false,"returncode":%s,"route_commit":"%s","route_id":"%s","run_id":"%s","runner_sha256":"%s","schema_version":1,"stage":"d0","state":"FAILED_ENGINEERING"}\n' "$rc" "$EXPECTED_ROUTE_COMMIT" "$ROUTE_ID" "$RUN_ID" "$RUNNER_SHA256" >"$CLOSEOUT_PATH"
    cp "$CLOSEOUT_PATH" "$EVID/a1x_v3_d0_closeout.json"
  fi
  exit "$rc"
}
trap on_exit EXIT
test "$MODE" = d0
test "$(git -C "$REMOTE_REPO" rev-parse HEAD)" = "$EXPECTED_ROUTE_COMMIT"
test "$(sha256sum "$REMOTE_REPO/experience_docx/tools/run_chd_rm_v4a_a1x_v3_d0.sh" | awk '{print $1}')" = "$RUNNER_SHA256"
test ! -e "$WORK"
A1FE=$A1F/experience_docx/experiment_logs/haze4k_v5_chd_rm_v4a_a1f_deltau_action_feasibility_20260714
A0R=$BASE/runs/haze4k_v5_chd_rm_v4a_conditional_safety_audit_20260714/a0r_r2/r1/trace
V3JE=$V3M/experience_docx/experiment_logs/haze4k_v5_chd_rm_v3j_bounded_safe_correction_audit_20260711
V3LE=$V3L/experience_docx/experiment_logs/haze4k_v5_chd_rm_v3l_safe_step_escalation_physics_audit_20260711
V3ME=$V3M/experience_docx/experiment_logs/haze4k_v5_chd_rm_v3m_blockwise_counterfactual_advantage_20260711
set +e
timeout --signal=TERM --kill-after=5m 8h env CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 "$PY" "$ENTRY" audit \
 --a1r-root "$A1R" --expected-a1r-commit 7053405a5d6b862ab6edfce12acd491099ad58f6 \
 --a1f-root "$A1F" --expected-a1f-root-commit d4f8d0936869c822ae19b5c21172efc2eb973dd8 \
 --v3z-root "$V3Z" --a1f-r3-review "$A1FE/v4a_a1f_r3_review.json" --expected-a1f-r3-review-sha256 a8b9064308710ac5fc890b9de0158c1faddb4d51f7d298d4991e9ddfb3616e1d \
 --a0r-trace-dir "$A0R" --a1r-stage formal --expected-route-commit "$EXPECTED_ROUTE_COMMIT" --expected-route-card-sha256 A1X_V3_D0 --status-file "$STATUS" \
 --fresh-start-index 256 --fresh-count 512 --outer-folds 4 --probe-epochs 8 --probe-batch-size 8 --probe-width 24 --probe-learning-rate 0.0005 --probe-weight-decay 0.00001 --probe-grad-clip-norm 0.1 --probe-seed 3407 \
 --mode projected --output_dir "$WORK" --run_tag "$RUN_ID" \
 --v3s_root "$V3S" --expected_v3s_commit 2860f580bb25cc75ec9ade56378af6d77f5c8d8b --v3p_root "$V3P" --expected_v3p_commit 555fd008e29f02128564f2fad41d0095ee44f5ea \
 --a0_checkpoint "$BASE/checkpoints/official/Haze4K/haze4k-base.pkl" --control_checkpoint "$BASE/repos/ConvIR-B-v3d-rarm-adapter-only-preflight/Dehazing/ITS/results/ConvIR-Haze4K-v3d-fam2modres-control-e5frome1-seed3407-20260710/Training-Results/Final.pkl" \
 --data_dir "$BASE/datasets/Haze4K/Haze4K" --fresh_split_manifest "$V3JE/fresh_route_confirm_split_manifest.json" --v3j_a_bounds "$V3JE/bounded_action_space_bounds.json" --operator_artifact_manifest "$V3LE/v3l_a0_canonical_operator_artifact_manifest.json" \
 --density_artifact "$BASE/repos/ConvIR-B-haze4k-v5-v2-chd-rm-density-need-calibration/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2_density_need_calibration_20260708/artifacts/d3_density_only_head.pt" --d7c_artifact "$BASE/repos/ConvIR-B-haze4k-v5-v2d-chd-rm-need-spatial-hard-negative/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2d_need_spatial_hard_negative_20260709/d7c_full/artifacts/d7c_mc_topk_hn_ordinal_head.pt" \
 --reference_oof_rows "$V3ME/cloud_only_raw_common_action/v3l_a1_oracle_policy_oof_rows_cloud_only.csv" \
 --expected_a0_checkpoint_sha256 6f42037d57a4e3de3a10ac0ab909d66a3415864a19433c29204a975f4efa4088 --expected_control_checkpoint_sha256 08207119a5cf9e5c439dd2cb81b99029ade1861f2739d31e75f2f9f78d57c0f2 --expected_density_artifact_sha256 1ffce13dccb41d96a47c2b5275f87bf2fdb73c226a190cfa240e5c71c1ec326f --expected_d7c_artifact_sha256 09f449232024395cf64db15a2a0efa0f12d3e0e049e1da3d67229a3dc5729361 --expected_fresh_split_manifest_sha256 c8c00fefc965ded3389b6311fc67ea521e1f3174f27793688544abe09dc420e7 --expected_operator_manifest_sha256 1d2ffa499128ad08a272d67c5439583900afe8ef87fb3256193ad5fe21c3af84 --expected_reference_oof_rows_sha256 b4a10184fab77b0045440dc88530d000a892acf2105a6295d5ad8a488c67ecb1 --expected_v3j_a_bounds_sha256 485ea12ff14c33b87105a50b6d118a9937c7e7f1b113062fe03d91eef3c9cc21 \
 --sample_count 128 --epochs 16 --risk_window 4 --warmup_epochs 8 --learning_rate 0.0005 --weight_decay 0.00001 --grad_clip_norm 0.1 --cvar_fraction 0.25 --seed 3407 --device cuda 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e
if [[ $rc -eq 0 && -f "$CLOSEOUT_PATH" ]]; then cp "$CLOSEOUT_PATH" "$EVID/a1x_v3_d0_closeout.json"; cp "$WORK/a1x_v3_d0_summary.json" "$EVID/a1x_v3_d0_summary.json"; echo A1X_V3_D0_OK; exit 0; fi
echo A1X_V3_D0_FAILED >&2
exit "${rc:-2}"
