#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT=${REMOTE_ROOT:-/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v3m-blockwise-counterfactual-advantage-20260711}
PARENT_ROOT=${PARENT_ROOT:-/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v3l-safe-step-escalation-physics-audit-20260711}
PY=${PY:-/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python}
ROUTE_ID=haze4k_v5_chd_rm_v3m_blockwise_counterfactual_advantage_20260711
PARENT_ROUTE_ID=haze4k_v5_chd_rm_v3l_safe_step_escalation_physics_audit_20260711
BRANCH=codex/haze4k-v5-v3m-blockwise-counterfactual-advantage
EVID="$REMOTE_ROOT/experience_docx/experiment_logs/$ROUTE_ID"
PARENT_EVID="$PARENT_ROOT/experience_docx/experiment_logs/$PARENT_ROUTE_ID"
STATUS="$EVID/status.txt"
STAMP=$(date +%Y%m%dT%H%M%S)
LOG="$EVID/v3m_a0b_dense_continuous_cross_audit_${STAMP}.log"

echo "v3m_a0b_cross_audit_start $ROUTE_ID $(date --iso-8601=seconds)" | tee -a "$STATUS"
echo "v3m_a0b_cross_audit_log $LOG" | tee -a "$STATUS"

cd "$REMOTE_ROOT"
test "$(git branch --show-current)" = "$BRANCH"
for f in \
  "$PARENT_EVID/v3l_a0_canonical_operator_artifact_manifest.json" \
  "$PARENT_EVID/v3l_a0_canonical_operator_closeout.json" \
  "$PARENT_EVID/v3l_a1_oracle_policy_oof_rows_cloud_only.csv" \
  "$PARENT_EVID/v3l_a1_oracle_policy_summary.csv" \
  "$PARENT_EVID/v3l_a1_oracle_granularity_summary.json" \
  "$EVID/cloud_only_raw_common_action/v3l_a1_oracle_policy_oof_rows_cloud_only.csv" \
  "$EVID/v3m_a0_common_action_oracle_summary.csv" \
  "$EVID/v3m_a0_source_manifest.json"
do
  test -s "$f"
done
for f in v3m_a0b_quantization_gap_summary.csv v3m_a0b_cross_audit.json v3m_a0b_source_manifest.json; do
  test ! -e "$EVID/$f"
done

set +e
PYTHONUNBUFFERED=1 "$PY" experience_docx/tools/chd_rm_v3m_a0b_dense_continuous_cross_audit.py \
  --parent_operator_manifest "$PARENT_EVID/v3l_a0_canonical_operator_artifact_manifest.json" \
  --parent_a0_closeout "$PARENT_EVID/v3l_a0_canonical_operator_closeout.json" \
  --parent_oof_rows "$PARENT_EVID/v3l_a1_oracle_policy_oof_rows_cloud_only.csv" \
  --parent_policy_summary "$PARENT_EVID/v3l_a1_oracle_policy_summary.csv" \
  --parent_granularity_summary "$PARENT_EVID/v3l_a1_oracle_granularity_summary.json" \
  --v3m_oof_rows "$EVID/cloud_only_raw_common_action/v3l_a1_oracle_policy_oof_rows_cloud_only.csv" \
  --v3m_policy_summary "$EVID/v3m_a0_common_action_oracle_summary.csv" \
  --v3m_source_manifest "$EVID/v3m_a0_source_manifest.json" \
  --output_dir "$EVID" \
  --expected_parent_operator_manifest_sha256 1d2ffa499128ad08a272d67c5439583900afe8ef87fb3256193ad5fe21c3af84 \
  --expected_parent_a0_closeout_sha256 2ca39ec1e17f4be794121603e3392a4e042e4d93b0e823454f7cf539f172d05d \
  --expected_parent_oof_rows_sha256 2a1b3a45cbeab6e646da4c45d17d7a8ad8c45f4ba477d06dbf5d3ab630e284cc \
  --expected_parent_policy_summary_sha256 7b538152bdac38526d50b500148c961819a78c6c0f9219be4626b962ca795d78 \
  --expected_parent_granularity_summary_sha256 fca8e73dcf86e58cd2b60cfb8fc74967167a6f91d70ca2b3b3cb4d9c959964db \
  --expected_v3m_oof_rows_sha256 b4a10184fab77b0045440dc88530d000a892acf2105a6295d5ad8a488c67ecb1 \
  --expected_v3m_policy_summary_sha256 925da8410154d16a20ab54ba0e3996dd99224fd94cad6d655c092dc745b940ec \
  --expected_v3m_source_manifest_sha256 8966996c9c93f6f2f3fbdda536b69ea6aa03e1bf5432f127de47ca8ea95dd8a5 \
  --max_mean_gap_db 0.005 \
  --fixed_replay_tolerance_db 1e-12 \
  --monotonic_tolerance_db 1e-9 \
  --bootstrap_draws 4000 \
  --seed 3407 \
  > "$LOG" 2>&1
rc=$?
set -e
echo "v3m_a0b_cross_audit_done rc=$rc $ROUTE_ID $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [ "$rc" -eq 0 ]; then
  echo "V3M_A0B_DENSE_CONTINUOUS_CROSS_AUDIT_OK" | tee -a "$STATUS"
else
  echo "V3M_A0B_DENSE_CONTINUOUS_CROSS_AUDIT_FAILED" | tee -a "$STATUS"
fi
exit "$rc"
