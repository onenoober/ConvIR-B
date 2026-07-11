#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT=${REMOTE_ROOT:-/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v3k-tail-risk-observability-20260711}
BASE=${BASE:-/sda/home/wangyuxin/ConvIR-B}
PY=${PY:-/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python}
GPU_ID=${GPU_ID:-0}
ROUTE_ID=haze4k_v5_chd_rm_v3k_tail_risk_observability_20260711
V3J_ROUTE_ID=haze4k_v5_chd_rm_v3j_bounded_safe_correction_audit_20260711
BRANCH=codex/haze4k-v5-v3k-tail-risk-observability
EVID="$REMOTE_ROOT/experience_docx/experiment_logs/$ROUTE_ID"
V3J_EVID="$REMOTE_ROOT/experience_docx/experiment_logs/$V3J_ROUTE_ID"
STATUS="$EVID/status.txt"
STAMP=$(date +%Y%m%dT%H%M%S)
LOG="$EVID/v3k_a0_a1_a2_new_replicate_diagnostic_${STAMP}.log"
SUMMARY="$EVID/v3k_tail_risk_observability_summary.json"

mkdir -p "$EVID"
cat > "$EVID/v3k_new_replicate_label.md" <<'MD'
# v3k New Replicate Label

Attempt1 strict reconstruction failed: row identity and row count matched v3j-B,
but direct-head PSNR deltas and severe sets did not exactly match archived replay.

Therefore subsequent v3k outputs in this evidence directory are labeled as a
new diagnostic replicate. They may support provisional mechanism diagnosis, but
must not be used to select alpha, clip, energy bound, risk thresholds, canary,
or promotion without a new sealed train-derived validation split.
MD

echo "v3k_new_replicate_start $ROUTE_ID $(date --iso-8601=seconds)" | tee -a "$STATUS"
echo "v3k_new_replicate_log $LOG" | tee -a "$STATUS"

cd "$REMOTE_ROOT"
if [ "$(git branch --show-current)" != "$BRANCH" ]; then
  echo "V3K_NEW_REPLICATE_PREFLIGHT_FAILED wrong_branch=$(git branch --show-current)" | tee -a "$STATUS"
  exit 2
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
    Path("$V3J_EVID/direct_correction_oof_policy_replay.csv"),
    Path("$V3J_EVID/direct_correction_route_confirm_policy_replay.csv"),
    Path("$BASE/repos/ConvIR-B-haze4k-v5-v2-chd-rm-density-need-calibration/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2_density_need_calibration_20260708/artifacts/d3_density_only_head.pt"),
    Path("$BASE/repos/ConvIR-B-haze4k-v5-v2d-chd-rm-need-spatial-hard-negative/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2d_need_spatial_hard_negative_20260709/d7c_full/artifacts/d7c_mc_topk_hn_ordinal_head.pt"),
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit("missing required assets: " + ", ".join(missing))
summary = json.loads(Path("$V3J_EVID/direct_correction_probe_summary.json").read_text(encoding="utf-8"))
expected = "V3J_DIRECT_SAFE_CORRECTION_OOF_FAIL_REQUIRE_NEW_INFORMATION_NO_INTERNAL_ROUTER"
if summary.get("decision") != expected:
    raise SystemExit(f"v3j-B decision mismatch: {summary.get('decision')}")
print("V3K_NEW_REPLICATE_PREFLIGHT_OK")
PY

heartbeat() {
  while true; do
    sleep 300
    echo "v3k_new_replicate_heartbeat $ROUTE_ID $(date --iso-8601=seconds)" >> "$STATUS"
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
PYTHONUNBUFFERED=1 "$PY" experience_docx/tools/chd_rm_v3k_tail_risk_observability.py \
  --a0_checkpoint "$BASE/checkpoints/official/Haze4K/haze4k-base.pkl" \
  --control_checkpoint "$BASE/repos/ConvIR-B-v3d-rarm-adapter-only-preflight/Dehazing/ITS/results/ConvIR-Haze4K-v3d-fam2modres-control-e5frome1-seed3407-20260710/Training-Results/Final.pkl" \
  --data_dir "$BASE/datasets/Haze4K/Haze4K" \
  --fresh_split_manifest "$V3J_EVID/fresh_route_confirm_split_manifest.json" \
  --v3j_a_summary "$V3J_EVID/v3j_a_bounded_action_audit_summary.json" \
  --v3j_a_bounds "$V3J_EVID/bounded_action_space_bounds.json" \
  --v3j_b_summary "$V3J_EVID/direct_correction_probe_summary.json" \
  --reference_oof_replay "$V3J_EVID/direct_correction_oof_policy_replay.csv" \
  --reference_confirm_replay "$V3J_EVID/direct_correction_route_confirm_policy_replay.csv" \
  --density_artifact "$BASE/repos/ConvIR-B-haze4k-v5-v2-chd-rm-density-need-calibration/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2_density_need_calibration_20260708/artifacts/d3_density_only_head.pt" \
  --d7c_artifact "$BASE/repos/ConvIR-B-haze4k-v5-v2d-chd-rm-need-spatial-hard-negative/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2d_need_spatial_hard_negative_20260709/d7c_full/artifacts/d7c_mc_topk_hn_ordinal_head.pt" \
  --output_dir "$EVID" \
  --max_train_samples 1200 \
  --max_confirm_samples 600 \
  --fold_count 5 \
  --active_sample_per_image 96 \
  --inactive_sample_per_image 32 \
  --linear_steps 360 \
  --context_steps 480 \
  --bootstrap_draws 2000 \
  --block_sizes 16 32 \
  --fixed_alphas 0 0.125 0.25 0.375 0.5 0.75 1.0 \
  --allow_overwrite \
  --progress_every 25 \
  > "$LOG" 2>&1
rc=$?
set -e
echo "v3k_new_replicate_done rc=$rc $ROUTE_ID $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [ "$rc" -eq 0 ]; then
  echo "V3K_NEW_REPLICATE_DIAGNOSTIC_OK" | tee -a "$STATUS"
else
  echo "V3K_NEW_REPLICATE_DIAGNOSTIC_FAILED" | tee -a "$STATUS"
fi
exit "$rc"
