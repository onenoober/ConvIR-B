#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT=${REMOTE_ROOT:-/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v3l-safe-step-escalation-physics-audit-20260711}
BASE=${BASE:-/sda/home/wangyuxin/ConvIR-B}
PY=${PY:-/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python}
GPU_ID=${GPU_ID:-0}
ROUTE_ID=haze4k_v5_chd_rm_v3l_safe_step_escalation_physics_audit_20260711
V3J_ROUTE_ID=haze4k_v5_chd_rm_v3j_bounded_safe_correction_audit_20260711
BRANCH=codex/haze4k-v5-v3l-safe-step-escalation-physics-audit
EVID="$REMOTE_ROOT/experience_docx/experiment_logs/$ROUTE_ID"
V3J_EVID="$REMOTE_ROOT/experience_docx/experiment_logs/$V3J_ROUTE_ID"
STATUS="$EVID/status.txt"
STAMP=$(date +%Y%m%dT%H%M%S)
LOG="$EVID/v3l_a1_oracle_granularity_audit_${STAMP}.log"
A0_CLOSEOUT="$EVID/v3l_a0_canonical_operator_closeout.json"

mkdir -p "$EVID"

echo "v3l_a1_start $ROUTE_ID $(date --iso-8601=seconds)" | tee -a "$STATUS"
echo "v3l_a1_log $LOG" | tee -a "$STATUS"

cd "$REMOTE_ROOT"
if [ "$(git branch --show-current)" != "$BRANCH" ]; then
  echo "V3L_A1_PREFLIGHT_FAILED wrong_branch=$(git branch --show-current)" | tee -a "$STATUS"
  exit 2
fi

"$PY" - <<PY
import json
from pathlib import Path

evid = Path("$EVID")
required = [
    Path("$BASE/checkpoints/official/Haze4K/haze4k-base.pkl"),
    Path("$BASE/repos/ConvIR-B-v3d-rarm-adapter-only-preflight/Dehazing/ITS/results/ConvIR-Haze4K-v3d-fam2modres-control-e5frome1-seed3407-20260710/Training-Results/Final.pkl"),
    Path("$BASE/datasets/Haze4K/Haze4K"),
    Path("$V3J_EVID/fresh_route_confirm_split_manifest.json"),
    Path("$V3J_EVID/bounded_action_space_bounds.json"),
    Path("$BASE/repos/ConvIR-B-haze4k-v5-v2-chd-rm-density-need-calibration/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2_density_need_calibration_20260708/artifacts/d3_density_only_head.pt"),
    Path("$BASE/repos/ConvIR-B-haze4k-v5-v2d-chd-rm-need-spatial-hard-negative/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2d_need_spatial_hard_negative_20260709/d7c_full/artifacts/d7c_mc_topk_hn_ordinal_head.pt"),
    Path("$A0_CLOSEOUT"),
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit("missing required assets: " + ", ".join(missing))
closeout = json.loads(Path("$A0_CLOSEOUT").read_text(encoding="utf-8"))
expected = "V3L_A0_CANONICAL_OPERATOR_REPLAY_PASS_AUTHORIZE_A1_ORACLE_GRANULARITY_AUDIT"
if closeout.get("decision") != expected:
    raise SystemExit(f"A0 did not authorize A1: {closeout.get('decision')}")
for result in closeout.get("results", []):
    path = Path(result["artifact_path"])
    if not path.exists():
        raise SystemExit(f"A0 artifact missing: {path}")
print("V3L_A1_PREFLIGHT_OK")
PY

heartbeat() {
  while true; do
    sleep 300
    echo "v3l_a1_heartbeat $ROUTE_ID $(date --iso-8601=seconds)" >> "$STATUS"
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
PYTHONUNBUFFERED=1 "$PY" experience_docx/tools/chd_rm_v3l_a1_oracle_granularity_audit.py \
  --a0_checkpoint "$BASE/checkpoints/official/Haze4K/haze4k-base.pkl" \
  --control_checkpoint "$BASE/repos/ConvIR-B-v3d-rarm-adapter-only-preflight/Dehazing/ITS/results/ConvIR-Haze4K-v3d-fam2modres-control-e5frome1-seed3407-20260710/Training-Results/Final.pkl" \
  --data_dir "$BASE/datasets/Haze4K/Haze4K" \
  --fresh_split_manifest "$V3J_EVID/fresh_route_confirm_split_manifest.json" \
  --v3j_a_bounds "$V3J_EVID/bounded_action_space_bounds.json" \
  --a0_closeout "$A0_CLOSEOUT" \
  --density_artifact "$BASE/repos/ConvIR-B-haze4k-v5-v2-chd-rm-density-need-calibration/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2_density_need_calibration_20260708/artifacts/d3_density_only_head.pt" \
  --d7c_artifact "$BASE/repos/ConvIR-B-haze4k-v5-v2d-chd-rm-need-spatial-hard-negative/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2d_need_spatial_hard_negative_20260709/d7c_full/artifacts/d7c_mc_topk_hn_ordinal_head.pt" \
  --output_dir "$EVID" \
  --max_train_samples 1200 \
  --max_confirm_samples 600 \
  --fold_count 5 \
  --operator_labels D_ref D_rep \
  --fixed_alphas 0 0.125 0.25 0.375 0.5 0.75 1 \
  --oracle_alphas 0 0.03125 0.0625 0.09375 0.125 0.15625 0.1875 0.21875 0.25 0.28125 0.3125 0.34375 0.375 0.40625 0.4375 0.46875 0.5 0.53125 0.5625 0.59375 0.625 0.65625 0.6875 0.71875 0.75 0.78125 0.8125 0.84375 0.875 0.90625 0.9375 0.96875 1 \
  --block_sizes 16 32 \
  --oracle_block_sizes 16 32 \
  --min_oracle_mean_lift_db 0.02 \
  --bootstrap_draws 2000 \
  --progress_every 25 \
  --include_confirm_audit \
  --allow_overwrite \
  > "$LOG" 2>&1
rc=$?
set -e
echo "v3l_a1_done rc=$rc $ROUTE_ID $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [ "$rc" -eq 0 ]; then
  echo "V3L_A1_ORACLE_GRANULARITY_AUDIT_OK" | tee -a "$STATUS"
else
  echo "V3L_A1_ORACLE_GRANULARITY_AUDIT_FAILED" | tee -a "$STATUS"
fi
exit "$rc"
