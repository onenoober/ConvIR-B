#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT=${REMOTE_ROOT:-/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v3k-tail-risk-observability-20260711}
BASE=${BASE:-/sda/home/wangyuxin/ConvIR-B}
PY=${PY:-/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python}
GPU_ID=${GPU_ID:-0}
SEED=${SEED:-3408}
ROUTE_ID=haze4k_v5_chd_rm_v3k_tail_risk_observability_20260711
V3J_ROUTE_ID=haze4k_v5_chd_rm_v3j_bounded_safe_correction_audit_20260711
BRANCH=codex/haze4k-v5-v3k-tail-risk-observability
EVID="$REMOTE_ROOT/experience_docx/experiment_logs/$ROUTE_ID"
SUBEVID="$EVID/v3k_d_context_seed${SEED}_open_holdout_stability"
V3J_EVID="$REMOTE_ROOT/experience_docx/experiment_logs/$V3J_ROUTE_ID"
HOLDOUT_MANIFEST="$EVID/v3k_c_val_inner_open_holdout_manifest.json"
STATUS="$EVID/status.txt"
STAMP=$(date +%Y%m%dT%H%M%S)
LOG="$SUBEVID/v3k_d_context_seed${SEED}_open_holdout_stability_${STAMP}.log"
SUMMARY="$SUBEVID/v3k_c_open_holdout_fixed_alpha_summary.json"

mkdir -p "$SUBEVID"
cat > "$SUBEVID/README.md" <<'MD'
# v3k-D Context Seed Stability

This is a lightweight robustness check for the v3k-C open-holdout result.
It retrains only the context diagnostic head with a different seed and evaluates
the OOF-preselected alpha values 0.125 and 0.25 on historical `val_inner`.

This remains open-holdout diagnostic evidence only. It is not a new sealed split
and cannot authorize canary or promotion.
MD

echo "v3k_d_start seed=$SEED $ROUTE_ID $(date --iso-8601=seconds)" | tee -a "$STATUS"
echo "v3k_d_log $LOG" | tee -a "$STATUS"

cd "$REMOTE_ROOT"
if [ "$(git branch --show-current)" != "$BRANCH" ]; then
  echo "V3K_D_PREFLIGHT_FAILED wrong_branch=$(git branch --show-current)" | tee -a "$STATUS"
  exit 2
fi
if [ -e "$SUMMARY" ] && [ "${ALLOW_OVERWRITE:-0}" != "1" ]; then
  echo "V3K_D_PREFLIGHT_FAILED existing_summary=$SUMMARY" | tee -a "$STATUS"
  exit 3
fi

"$PY" - <<PY
import json
from pathlib import Path

required = [
    Path("$BASE/checkpoints/official/Haze4K/haze4k-base.pkl"),
    Path("$BASE/repos/ConvIR-B-v3d-rarm-adapter-only-preflight/Dehazing/ITS/results/ConvIR-Haze4K-v3d-fam2modres-control-e5frome1-seed3407-20260710/Training-Results/Final.pkl"),
    Path("$BASE/datasets/Haze4K/Haze4K"),
    Path("$V3J_EVID/fresh_route_confirm_split_manifest.json"),
    Path("$V3J_EVID/bounded_action_space_bounds.json"),
    Path("$V3J_EVID/v3j_a_bounded_action_audit_summary.json"),
    Path("$V3J_EVID/direct_correction_probe_summary.json"),
    Path("$HOLDOUT_MANIFEST"),
    Path("$BASE/repos/ConvIR-B-haze4k-v5-v2-chd-rm-density-need-calibration/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2_density_need_calibration_20260708/artifacts/d3_density_only_head.pt"),
    Path("$BASE/repos/ConvIR-B-haze4k-v5-v2d-chd-rm-need-spatial-hard-negative/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2d_need_spatial_hard_negative_20260709/d7c_full/artifacts/d7c_mc_topk_hn_ordinal_head.pt"),
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit("missing required assets: " + ", ".join(missing))
manifest = json.loads(Path("$HOLDOUT_MANIFEST").read_text(encoding="utf-8"))
if manifest.get("counts", {}).get("train_holdout_overlap") != 0:
    raise SystemExit("train_holdout_overlap is not zero")
print("V3K_D_PREFLIGHT_OK")
PY

heartbeat() {
  while true; do
    sleep 300
    echo "v3k_d_heartbeat seed=$SEED $ROUTE_ID $(date --iso-8601=seconds)" >> "$STATUS"
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
PYTHONUNBUFFERED=1 "$PY" experience_docx/tools/chd_rm_v3k_open_holdout_fixed_alpha.py \
  --a0_checkpoint "$BASE/checkpoints/official/Haze4K/haze4k-base.pkl" \
  --control_checkpoint "$BASE/repos/ConvIR-B-v3d-rarm-adapter-only-preflight/Dehazing/ITS/results/ConvIR-Haze4K-v3d-fam2modres-control-e5frome1-seed3407-20260710/Training-Results/Final.pkl" \
  --data_dir "$BASE/datasets/Haze4K/Haze4K" \
  --holdout_manifest "$HOLDOUT_MANIFEST" \
  --fresh_split_manifest "$V3J_EVID/fresh_route_confirm_split_manifest.json" \
  --v3j_a_summary "$V3J_EVID/v3j_a_bounded_action_audit_summary.json" \
  --v3j_a_bounds "$V3J_EVID/bounded_action_space_bounds.json" \
  --v3j_b_summary "$V3J_EVID/direct_correction_probe_summary.json" \
  --density_artifact "$BASE/repos/ConvIR-B-haze4k-v5-v2-chd-rm-density-need-calibration/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2_density_need_calibration_20260708/artifacts/d3_density_only_head.pt" \
  --d7c_artifact "$BASE/repos/ConvIR-B-haze4k-v5-v2d-chd-rm-need-spatial-hard-negative/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2d_need_spatial_hard_negative_20260709/d7c_full/artifacts/d7c_mc_topk_hn_ordinal_head.pt" \
  --output_dir "$SUBEVID" \
  --train_key v3j_controller_train \
  --holdout_key v3k_val_inner_open_holdout \
  --max_train_samples 1200 \
  --max_holdout_samples 600 \
  --active_sample_per_image 96 \
  --inactive_sample_per_image 32 \
  --context_steps 480 \
  --bootstrap_draws 2000 \
  --heads context \
  --fixed_alphas 0.125 0.25 \
  --seed "$SEED" \
  --progress_every 25 \
  > "$LOG" 2>&1
rc=$?
set -e
echo "v3k_d_done rc=$rc seed=$SEED $ROUTE_ID $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [ "$rc" -eq 0 ]; then
  echo "V3K_D_CONTEXT_SEED_HOLDOUT_STABILITY_OK" | tee -a "$STATUS"
else
  echo "V3K_D_CONTEXT_SEED_HOLDOUT_STABILITY_FAILED" | tee -a "$STATUS"
fi
exit "$rc"
