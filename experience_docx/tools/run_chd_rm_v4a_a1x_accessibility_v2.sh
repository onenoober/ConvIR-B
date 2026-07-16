#!/usr/bin/env bash
set -euo pipefail

ROUTE_ID=haze4k_v5_chd_rm_v4a_a1x_accessibility_v2_20260716
PY=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
BASE=/sda/home/wangyuxin/ConvIR-B
ENTRYPOINT="$REMOTE_REPO/experience_docx/tools/chd_rm_v4a_a1x_accessibility_v2.py"
ASSET_MANIFEST="$REMOTE_REPO/experience_docx/tools/a1x_v2_asset_manifest.json"
EVID_STAGE="$REMOTE_REPO/experience_docx/experiment_logs/$ROUTE_ID"
STATUS_PATH="$OUTPUT_PATH/status.jsonl"
HEARTBEAT_PATH="$OUTPUT_PATH/heartbeat.json"
RUNTIME_LOG_PATH="$OUTPUT_PATH/runtime.log"

: "${MODE:?MODE is required}"
: "${REMOTE_REPO:?REMOTE_REPO is required}"
: "${RUN_ROOT:?RUN_ROOT is required}"
: "${OUTPUT_PATH:?OUTPUT_PATH is required}"
: "${RUN_ID:?RUN_ID is required}"
: "${GPU:?GPU is required}"
: "${EXPECTED_ROUTE_COMMIT:?EXPECTED_ROUTE_COMMIT is required}"
: "${RUNNER_SHA256:?RUNNER_SHA256 is required}"

case "$MODE" in
  s0) A1X_STAGE=s0; LEGACY_STAGE=smoke; LIMIT=30m; CLOSEOUT_FILENAME=a1x_v2_s0_closeout.json ;;
  formal|formal_resume) A1X_STAGE=formal; LEGACY_STAGE=formal; LIMIT=10h; CLOSEOUT_FILENAME=a1x_v2_formal_closeout.json ;;
  *) echo "A1X_V2_INVALID_MODE mode=$MODE" >&2; exit 2 ;;
esac

CLOSEOUT_PATH="$OUTPUT_PATH/$CLOSEOUT_FILENAME"
SEALED_CLOSEOUT_PATH="$EVID_STAGE/$CLOSEOUT_FILENAME"
mkdir -p "$OUTPUT_PATH"
test -x "$PY"
test -f "$ENTRYPOINT"
test -f "$ASSET_MANIFEST"
test "$(git -C "$REMOTE_REPO" rev-parse HEAD)" = "$EXPECTED_ROUTE_COMMIT"
test "$(sha256sum "$REMOTE_REPO/experience_docx/tools/run_chd_rm_v4a_a1x_accessibility_v2.sh" | awk '{print $1}')" = "$RUNNER_SHA256"

export A1X_STAGE CLOSEOUT_PATH EVID_STAGE EXPECTED_ROUTE_COMMIT HEARTBEAT_PATH
export REMOTE_REPO RUN_ID RUNNER_SHA256 STATUS_PATH

"$PY" - "$ASSET_MANIFEST" <<'PY'
import hashlib
import json
import subprocess
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert manifest["schema_version"] == 1
assert manifest["route_id"] == "haze4k_v5_chd_rm_v4a_a1x_accessibility_v2_20260716"
for value in manifest["source_checkouts"].values():
    root = Path(value["path"])
    assert root.is_dir()
    head = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    assert head == value["commit"] and not dirty
for value in manifest["files"].values():
    path = Path(value["path"])
    assert path.is_file() and path.is_absolute() and not path.is_symlink()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    assert digest.hexdigest() == value["sha256"]
data = Path(manifest["data"]["path"])
assert (data / "train" / "haze").is_dir()
assert (data / "train" / "gt").is_dir()
assert manifest["data"]["a1x_indices"] == [768, 1200]
print("A1X_V2_ASSET_PREFLIGHT_OK")
PY

A1R_ROOT=$BASE/repos/ConvIR-B-v4a-a1r-representation-sufficiency-20260714
A1F_ROOT=$BASE/repos/ConvIR-B-v4a-a1f-deltau-action-feasibility-20260714
V3Z_ROOT=$BASE/repos/ConvIR-B-v3z-source-3caddcc5265732e5be77e3404119a28cb28c11e6
V3S_ROOT=$BASE/repos/ConvIR-B-v3s-source-2860f580bb25cc75ec9ade56378af6d77f5c8d8b
V3P_ROOT=$BASE/repos/ConvIR-B-v3p-source-555fd008e29f02128564f2fad41d0095ee44f5ea
V3M_ROOT=$BASE/repos/ConvIR-B-v3m-blockwise-counterfactual-advantage-20260711
V3L_ROOT=$BASE/repos/ConvIR-B-v3l-safe-step-escalation-physics-audit-20260711
A1F_EVID=$A1F_ROOT/experience_docx/experiment_logs/haze4k_v5_chd_rm_v4a_a1f_deltau_action_feasibility_20260714
A0R_TRACE=$BASE/runs/haze4k_v5_chd_rm_v4a_conditional_safety_audit_20260714/a0r_r2/r1/trace
V3J_EVID=$V3M_ROOT/experience_docx/experiment_logs/haze4k_v5_chd_rm_v3j_bounded_safe_correction_audit_20260711
V3L_EVID=$V3L_ROOT/experience_docx/experiment_logs/haze4k_v5_chd_rm_v3l_safe_step_escalation_physics_audit_20260711
V3M_EVID=$V3M_ROOT/experience_docx/experiment_logs/haze4k_v5_chd_rm_v3m_blockwise_counterfactual_advantage_20260711

COMMON_ARGS=(
  --v3s_root "$V3S_ROOT" --expected_v3s_commit 2860f580bb25cc75ec9ade56378af6d77f5c8d8b
  --v3p_root "$V3P_ROOT" --expected_v3p_commit 555fd008e29f02128564f2fad41d0095ee44f5ea
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

echo "stage_start route=$ROUTE_ID stage=$A1X_STAGE run=$RUN_ID time=$(date --iso-8601=seconds)" | tee -a "$STATUS_PATH"
set +e
timeout --signal=TERM --kill-after=5m "$LIMIT" env CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 "$PY" "$ENTRYPOINT" audit \
  --a1r-root "$A1R_ROOT" --expected-a1r-commit 7053405a5d6b862ab6edfce12acd491099ad58f6 \
  --a1f-root "$A1F_ROOT" --expected-a1f-root-commit d4f8d0936869c822ae19b5c21172efc2eb973dd8 \
  --v3z-root "$V3Z_ROOT" --a1f-r3-review "$A1F_EVID/v4a_a1f_r3_review.json" \
  --expected-a1f-r3-review-sha256 a8b9064308710ac5fc890b9de0158c1faddb4d51f7d298d4991e9ddfb3616e1d \
  --a0r-trace-dir "$A0R_TRACE" --a1r-stage "$LEGACY_STAGE" \
  --expected-route-commit "$EXPECTED_ROUTE_COMMIT" --expected-route-card-sha256 unused-by-a1x-v2 \
  --status-file "$STATUS_PATH" --fresh-start-index 768 --fresh-count 432 --outer-folds 4 \
  --probe-epochs 8 --probe-batch-size 8 --probe-width 24 \
  --probe-learning-rate 0.0005 --probe-weight-decay 0.00001 \
  --probe-grad-clip-norm 0.1 --probe-seed 3407 \
  --mode projected --output_dir "$OUTPUT_PATH" --run_tag "$RUN_ID" "${COMMON_ARGS[@]}" \
  2>&1 | tee -a "$RUNTIME_LOG_PATH"
rc=${PIPESTATUS[0]}
set -e
echo "stage_done route=$ROUTE_ID stage=$A1X_STAGE run=$RUN_ID rc=$rc time=$(date --iso-8601=seconds)" | tee -a "$STATUS_PATH"

if [[ $rc -eq 0 && -f $CLOSEOUT_PATH ]]; then
  mkdir -p "$EVID_STAGE"
  if [[ $A1X_STAGE == s0 ]]; then
    cp "$OUTPUT_PATH/a1x_v2_s0_summary.json" "$EVID_STAGE/a1x_v2_s0_summary.json"
  else
    cp "$OUTPUT_PATH/a1x_v2_formal_summary.json" "$EVID_STAGE/a1x_v2_formal_summary.json"
    cp "$OUTPUT_PATH/a1x_v2_unit_manifest.json" "$EVID_STAGE/a1x_v2_unit_manifest.json"
  fi
  cp "$CLOSEOUT_PATH" "$SEALED_CLOSEOUT_PATH.tmp"
  mv "$SEALED_CLOSEOUT_PATH.tmp" "$SEALED_CLOSEOUT_PATH"
  echo "A1X_V2_${A1X_STAGE^^}_OK"
  exit 0
fi
if [[ $rc -eq 0 ]]; then rc=2; fi
echo "A1X_V2_${A1X_STAGE^^}_FAILED" >&2
exit "$rc"
