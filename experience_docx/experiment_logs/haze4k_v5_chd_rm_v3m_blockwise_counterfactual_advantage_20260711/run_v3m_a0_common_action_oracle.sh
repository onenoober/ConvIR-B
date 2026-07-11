#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT=${REMOTE_ROOT:-/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v3m-blockwise-counterfactual-advantage-20260711}
BASE=${BASE:-/sda/home/wangyuxin/ConvIR-B}
PARENT_ROOT=${PARENT_ROOT:-/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v3l-safe-step-escalation-physics-audit-20260711}
PY=${PY:-/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python}
GPU_ID=${GPU_ID:-1}
ROUTE_ID=haze4k_v5_chd_rm_v3m_blockwise_counterfactual_advantage_20260711
V3J_ROUTE_ID=haze4k_v5_chd_rm_v3j_bounded_safe_correction_audit_20260711
BRANCH=codex/haze4k-v5-v3m-blockwise-counterfactual-advantage
EVID="$REMOTE_ROOT/experience_docx/experiment_logs/$ROUTE_ID"
V3J_EVID="$REMOTE_ROOT/experience_docx/experiment_logs/$V3J_ROUTE_ID"
PARENT_EVID="$PARENT_ROOT/experience_docx/experiment_logs/haze4k_v5_chd_rm_v3l_safe_step_escalation_physics_audit_20260711"
STATUS="$EVID/status.txt"
STAMP=$(date +%Y%m%dT%H%M%S)
LOG="$EVID/v3m_a0_common_action_oracle_${STAMP}.log"

mkdir -p "$EVID"
echo "v3m_a0_start $ROUTE_ID $(date --iso-8601=seconds)" | tee -a "$STATUS"
echo "v3m_a0_log $LOG" | tee -a "$STATUS"

cd "$REMOTE_ROOT"
if [ "$(git branch --show-current)" != "$BRANCH" ]; then
  echo "V3M_A0_PREFLIGHT_FAILED wrong_branch=$(git branch --show-current)" | tee -a "$STATUS"
  exit 2
fi
if [ -e "$EVID/v3m_a0_common_action_summary.json" ] || [ -e "$EVID/cloud_only_raw_common_action" ]; then
  echo "V3M_A0_PREFLIGHT_FAILED existing_output" | tee -a "$STATUS"
  exit 2
fi

"$PY" - <<PY
import hashlib
import json
from pathlib import Path

required = [
    Path("$BASE/checkpoints/official/Haze4K/haze4k-base.pkl"),
    Path("$BASE/repos/ConvIR-B-v3d-rarm-adapter-only-preflight/Dehazing/ITS/results/ConvIR-Haze4K-v3d-fam2modres-control-e5frome1-seed3407-20260710/Training-Results/Final.pkl"),
    Path("$BASE/datasets/Haze4K/Haze4K"),
    Path("$V3J_EVID/fresh_route_confirm_split_manifest.json"),
    Path("$V3J_EVID/bounded_action_space_bounds.json"),
    Path("$PARENT_EVID/v3l_a0_canonical_operator_closeout.json"),
    Path("$PARENT_EVID/v3l_a0_canonical_operator_artifact_manifest.json"),
    Path("$BASE/repos/ConvIR-B-haze4k-v5-v2-chd-rm-density-need-calibration/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2_density_need_calibration_20260708/artifacts/d3_density_only_head.pt"),
    Path("$BASE/repos/ConvIR-B-haze4k-v5-v2d-chd-rm-need-spatial-hard-negative/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2d_need_spatial_hard_negative_20260709/d7c_full/artifacts/d7c_mc_topk_hn_ordinal_head.pt"),
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit("missing required assets: " + ", ".join(missing))
closeout = json.loads(Path("$PARENT_EVID/v3l_a0_canonical_operator_closeout.json").read_text(encoding="utf-8"))
expected = "V3L_A0_CANONICAL_OPERATOR_REPLAY_PASS_AUTHORIZE_A1_ORACLE_GRANULARITY_AUDIT"
if closeout.get("decision") != expected:
    raise SystemExit("v3l A0 authorization mismatch: " + str(closeout.get("decision")))
for result in closeout.get("results", []):
    path = Path(result["artifact_path"])
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != result.get("artifact_sha256"):
        raise SystemExit("operator hash mismatch: " + str(path))
print("V3M_A0_PREFLIGHT_OK")
PY

heartbeat() {
  while true; do
    sleep 300
    echo "v3m_a0_heartbeat $ROUTE_ID $(date --iso-8601=seconds)" >> "$STATUS"
  done
}
heartbeat &
HB_PID=$!
cleanup() {
  kill "$HB_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
set +e
PYTHONUNBUFFERED=1 "$PY" experience_docx/tools/chd_rm_v3m_a0_common_action_oracle.py \
  --a0_checkpoint "$BASE/checkpoints/official/Haze4K/haze4k-base.pkl" \
  --control_checkpoint "$BASE/repos/ConvIR-B-v3d-rarm-adapter-only-preflight/Dehazing/ITS/results/ConvIR-Haze4K-v3d-fam2modres-control-e5frome1-seed3407-20260710/Training-Results/Final.pkl" \
  --data_dir "$BASE/datasets/Haze4K/Haze4K" \
  --fresh_split_manifest "$V3J_EVID/fresh_route_confirm_split_manifest.json" \
  --v3j_a_bounds "$V3J_EVID/bounded_action_space_bounds.json" \
  --a0_closeout "$PARENT_EVID/v3l_a0_canonical_operator_closeout.json" \
  --operator_artifact_manifest "$PARENT_EVID/v3l_a0_canonical_operator_artifact_manifest.json" \
  --density_artifact "$BASE/repos/ConvIR-B-haze4k-v5-v2-chd-rm-density-need-calibration/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2_density_need_calibration_20260708/artifacts/d3_density_only_head.pt" \
  --d7c_artifact "$BASE/repos/ConvIR-B-haze4k-v5-v2d-chd-rm-need-spatial-hard-negative/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2d_need_spatial_hard_negative_20260709/d7c_full/artifacts/d7c_mc_topk_hn_ordinal_head.pt" \
  --output_dir "$EVID" \
  --parent_evidence_commit 5acaaa54d7aca7c9764dc3dd757ff58cdf6d96fa \
  --parent_cloud_worktree "$PARENT_ROOT" \
  --source_branch "$BRANCH" \
  --source_commit "$(git rev-parse HEAD)" \
  --max_train_samples 1200 \
  --max_confirm_samples 600 \
  --fold_count 5 \
  --operator_labels D_ref D_rep \
  --common_alphas 0 0.125 0.25 0.5 1 \
  --block_sizes 16 32 \
  --oracle_block_sizes 16 32 \
  --block16_retention_min 0.80 \
  --bootstrap_draws 4000 \
  --progress_every 25 \
  --include_confirm_audit \
  > "$LOG" 2>&1
rc=$?
set -e
echo "v3m_a0_done rc=$rc $ROUTE_ID $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [ "$rc" -eq 0 ]; then
  echo "V3M_A0_COMMON_ACTION_ORACLE_OK" | tee -a "$STATUS"
else
  echo "V3M_A0_COMMON_ACTION_ORACLE_FAILED" | tee -a "$STATUS"
fi
exit "$rc"
