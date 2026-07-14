#!/usr/bin/env bash
set -euo pipefail

ROUTE_ID=haze4k_v5_chd_rm_v4a_a1f_deltau_action_feasibility_20260714
REMOTE_REPO=${REMOTE_REPO:-/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v4a-a1f-deltau-action-feasibility-20260714}
RUN_ROOT=${RUN_ROOT:-/sda/home/wangyuxin/ConvIR-B/runs/$ROUTE_ID}
PY=${PY:-/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python}
BASE=${BASE:-/sda/home/wangyuxin/ConvIR-B}
V3Z_ROOT=${V3Z_ROOT:-$BASE/repos/ConvIR-B-v3z-source-3caddcc5265732e5be77e3404119a28cb28c11e6}
V3S_ROOT=${V3S_ROOT:-$BASE/repos/ConvIR-B-v3s-source-2860f580bb25cc75ec9ade56378af6d77f5c8d8b}
V3P_ROOT=${V3P_ROOT:-$BASE/repos/ConvIR-B-v3p-source-555fd008e29f02128564f2fad41d0095ee44f5ea}
V3M_ROOT=${V3M_ROOT:-$BASE/repos/ConvIR-B-v3m-blockwise-counterfactual-advantage-20260711}
V3L_ROOT=${V3L_ROOT:-$BASE/repos/ConvIR-B-v3l-safe-step-escalation-physics-audit-20260711}
GPU=${GPU:?set the dynamically selected GPU index}
STAGE=${STAGE:?set smoke or formal}
RUN_ID=${RUN_ID:?set a fresh run id}
EXPECTED_ROUTE_COMMIT=${EXPECTED_ROUTE_COMMIT:?set the exact A1F route commit}
EXPECTED_ROUTE_CARD_SHA256=${EXPECTED_ROUTE_CARD_SHA256:?set the validated A1F route-card SHA-256}
EXPECTED_V3Z_COMMIT=3caddcc5265732e5be77e3404119a28cb28c11e6
EXPECTED_V3S_COMMIT=2860f580bb25cc75ec9ade56378af6d77f5c8d8b
EXPECTED_V3P_COMMIT=555fd008e29f02128564f2fad41d0095ee44f5ea
EXPECTED_R3_REVIEW_SHA256=cddb543c67c5c1e167686bb3bed08dc25ca2c504a0835e899b123973e6b62a99
EXPECTED_A0D_ROWS_SHA256=045fecdb6701b0d7cad06772e15d7d2f2b5330db075f8c57cf9043d101b0dd05
EXPECTED_FINAL_STATE_SHA256=ed0832f220996af3fd8e617b7d04d643dc6ca052a3603adee99d59e78fd1e125

case "$STAGE" in
  smoke|formal) ;;
  *) echo "invalid STAGE=$STAGE" >&2; exit 2 ;;
esac

OUT=$RUN_ROOT/$RUN_ID
STATUS=$RUN_ROOT/status.txt
EVID_STAGE=$REMOTE_REPO/experience_docx/experiment_logs/$ROUTE_ID
RUNNER=$REMOTE_REPO/experience_docx/tools/chd_rm_v4a_a1f_action_feasibility.py
PARENT_EVID=$REMOTE_REPO/experience_docx/experiment_logs/haze4k_v5_chd_rm_v4a_conditional_safety_audit_20260714
R3_REVIEW=$PARENT_EVID/v4a_a0p_r3_review.json
A0R_TRACE=$BASE/runs/haze4k_v5_chd_rm_v4a_conditional_safety_audit_20260714/a0r_r2/r1/trace
FINAL_STATE=$A0R_TRACE/states/epoch16_update512_final.pt
A0D_ROWS=$BASE/runs/haze4k_v5_chd_rm_v4a_conditional_safety_audit_20260714/a0d_r2/v4a_a0d_a0d_rows_cloud_only.csv
V3J_EVID=$V3M_ROOT/experience_docx/experiment_logs/haze4k_v5_chd_rm_v3j_bounded_safe_correction_audit_20260711
V3L_EVID=$V3L_ROOT/experience_docx/experiment_logs/haze4k_v5_chd_rm_v3l_safe_step_escalation_physics_audit_20260711
V3M_EVID=$V3M_ROOT/experience_docx/experiment_logs/haze4k_v5_chd_rm_v3m_blockwise_counterfactual_advantage_20260711

test "$(git -C "$REMOTE_REPO" branch --show-current)" = "codex/haze4k-v5-v4a-a1f-deltau-action-feasibility-20260714"
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
test -f "$R3_REVIEW"
test "$(sha256sum "$R3_REVIEW" | awk '{print $1}')" = "$EXPECTED_R3_REVIEW_SHA256"
test -f "$A0R_TRACE/trace_manifest.json"
test -f "$FINAL_STATE"
test "$(sha256sum "$FINAL_STATE" | awk '{print $1}')" = "$EXPECTED_FINAL_STATE_SHA256"
test -f "$A0D_ROWS"
test "$(sha256sum "$A0D_ROWS" | awk '{print $1}')" = "$EXPECTED_A0D_ROWS_SHA256"
test ! -e "$OUT"
test -d "$BASE/datasets/Haze4K/Haze4K/train/haze"
test -d "$BASE/datasets/Haze4K/Haze4K/train/gt"
test -s "$BASE/checkpoints/official/Haze4K/haze4k-base.pkl"
test -s "$V3J_EVID/fresh_route_confirm_split_manifest.json"
test -s "$V3J_EVID/bounded_action_space_bounds.json"
test -s "$V3L_EVID/v3l_a0_canonical_operator_artifact_manifest.json"
test -s "$V3M_EVID/cloud_only_raw_common_action/v3l_a1_oracle_policy_oof_rows_cloud_only.csv"

"$PY" - "$R3_REVIEW" <<'PY'
import json
import sys
from pathlib import Path

value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = {
    "route_id": "haze4k_v5_chd_rm_v4a_conditional_safety_audit_20260714",
    "state": "COMPLETED_R3_REVIEW",
    "decision": "V4A_A0P_NO_LOCAL_CORRECTION_AUTHORIZE_A1F_METRIC_ALIGNED_FEASIBILITY_ONLY",
    "authorizes": "A1F_ROUTE_DESIGN_AND_IMPLEMENTATION_ONLY",
}
for key, item in expected.items():
    if value.get(key) != item:
        raise SystemExit(f"R3 authorization mismatch: {key}={value.get(key)!r}")
print("V4A_A1F_R3_AUTHORIZATION_OK")
PY

if [ "$STAGE" = formal ]; then
  SMOKE_CLOSEOUT=${SMOKE_CLOSEOUT:?set the passed smoke closeout}
  test -f "$SMOKE_CLOSEOUT"
  "$PY" - "$SMOKE_CLOSEOUT" <<'PY'
import json
import sys
from pathlib import Path

value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = {
    "route_id": "haze4k_v5_chd_rm_v4a_a1f_deltau_action_feasibility_20260714",
    "state": "COMPLETED_GATE_PASS",
    "decision": "V4A_A1F_S0_ALIGNMENT_PASS_AUTHORIZE_FORMAL_ONLY",
    "authorizes": "A1F_FORMAL_ONLY",
}
for key, item in expected.items():
    if value.get(key) != item:
        raise SystemExit(f"smoke authorization mismatch: {key}={value.get(key)!r}")
print("V4A_A1F_SMOKE_AUTHORIZATION_OK")
PY
fi

GPU_STATUS=$(nvidia-smi -i "$GPU" --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits)
GPU_FREE_MIB=$(printf '%s\n' "$GPU_STATUS" | awk -F, '{gsub(/ /,"",$1); print $1}')
GPU_UTIL=$(printf '%s\n' "$GPU_STATUS" | awk -F, '{gsub(/ /,"",$2); print $2}')
test "$GPU_FREE_MIB" -ge 18000
test "$GPU_UTIL" -le 10
printf 'GPU_PREFLIGHT index=%s free_mib=%s utilization=%s\n' "$GPU" "$GPU_FREE_MIB" "$GPU_UTIL"
mkdir -p "$RUN_ROOT" "$EVID_STAGE"
STAMP=$(date +%Y%m%dT%H%M%S)
LOG=$RUN_ROOT/v4a_a1f_${STAGE}_${STAMP}.log
echo "stage_start route=$ROUTE_ID stage=v4a-A1F-$STAGE run=$RUN_ID time=$(date --iso-8601=seconds)" | tee -a "$STATUS"
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
  --sample_count 128 --epochs 16 --risk_window 4 --warmup_epochs 8
  --learning_rate 0.0005 --weight_decay 0.00001 --grad_clip_norm 0.1 --cvar_fraction 0.25
  --seed 3407 --device cuda
)

set +e
CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 "$PY" "$RUNNER" audit \
  --v3z-root "$V3Z_ROOT" --a0r-trace-dir "$A0R_TRACE" \
  --a0d-rows "$A0D_ROWS" --expected-a0d-rows-sha256 "$EXPECTED_A0D_ROWS_SHA256" \
  --r3-review "$R3_REVIEW" --expected-r3-review-sha256 "$EXPECTED_R3_REVIEW_SHA256" \
  --expected-route-commit "$EXPECTED_ROUTE_COMMIT" \
  --expected-route-card-sha256 "$EXPECTED_ROUTE_CARD_SHA256" \
  --status-file "$STATUS" \
  --a1f-stage "$STAGE" --mode projected --output_dir "$OUT" \
  --run_tag "$RUN_ID" "${COMMON_ARGS[@]}" 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
set -e

echo "stage_done route=$ROUTE_ID stage=v4a-A1F-$STAGE run=$RUN_ID rc=$rc time=$(date --iso-8601=seconds)" | tee -a "$STATUS"
if [ "$rc" -ne 0 ]; then
  echo "V4A_A1F_${STAGE^^}_FAILED_RUNTIME_REQUIRES_CLASSIFICATION" | tee -a "$STATUS"
  exit "$rc"
fi

cp "$OUT/v4a_a1f_closeout.json" "$EVID_STAGE/v4a_a1f_${STAGE}_closeout.json"
cp "$OUT/v4a_a1f_source_manifest.json" "$EVID_STAGE/v4a_a1f_${STAGE}_source_manifest.json"
cp "$OUT/v4a_a1f_operator_summary.csv" "$EVID_STAGE/v4a_a1f_${STAGE}_operator_summary.csv"
cp "$OUT/v4a_a1f_bootstrap_summary.json" "$EVID_STAGE/v4a_a1f_${STAGE}_bootstrap_summary.json"
if grep -q '"state": "COMPLETED_GATE_PASS"' "$OUT/v4a_a1f_closeout.json"; then
  echo "V4A_A1F_${STAGE^^}_OK" | tee -a "$STATUS"
else
  echo "V4A_A1F_${STAGE^^}_GATE_FAIL" | tee -a "$STATUS"
fi
