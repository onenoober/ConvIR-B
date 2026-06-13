#!/usr/bin/env bash
set -euo pipefail

WORK_DIR="${DTA_V37_WORK_DIR:-$(pwd)}"
PY="${DTA_V37_PY:-/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python}"
EVID="$WORK_DIR/experience_docx/experiment_logs/haze4k_dta_v3_7_u_tqs_mix_20260613"
STATUS="$EVID/status_phase_d7_fixed_outputdiff_confirm.txt"
LOG="$EVID/phase_d7_fixed_outputdiff_confirm.log"

mkdir -p "$EVID"
{
  printf 'dta_v3_7_phase_d7_fixed_outputdiff_start %s\n' "$(date -Is)"
  printf 'state=RUNNING_D7_FIXED_OUTPUTDIFF_CONFIRM\n'
  printf 'work=%s\n' "$WORK_DIR"
  printf 'python=%s\n' "$PY"
  printf 'locked_test_touched=false\n'
  printf 'raw_d1_full_5x3_run=false\n'
  git -C "$WORK_DIR" branch --show-current || true
  git -C "$WORK_DIR" rev-parse --short HEAD || true
  git -C "$WORK_DIR" status --short || true
} > "$STATUS"

required=(
  "$EVID/v37_tau_real_blend_single_actions_all.csv"
  "$EVID/v37_tau_oof_per_image_action_table.csv"
  "$EVID/v37_d6_outputdiff_features_all.csv"
  "$EVID/v37_d6_outputdiff_policy_aggregate.csv"
)
for path in "${required[@]}"; do
  if [[ ! -f "$path" ]]; then
    {
      printf 'state=PREFLIGHT_FAILED_ENGINEERING\n'
      printf 'missing=%s\n' "$path"
      printf 'dta_v3_7_phase_d7_fixed_outputdiff_done rc=2 %s\n' "$(date -Is)"
    } >> "$STATUS"
    exit 2
  fi
done

set +e
"$PY" "$WORK_DIR/experience_docx/tools/confirm_haze4k_dta_v37_d7_fixed_outputdiff_policy.py" \
  --single_actions_csv "$EVID/v37_tau_real_blend_single_actions_all.csv" \
  --feature_action_table_csv "$EVID/v37_tau_oof_per_image_action_table.csv" \
  --outputdiff_features_csv "$EVID/v37_d6_outputdiff_features_all.csv" \
  --d6_aggregate_csv "$EVID/v37_d6_outputdiff_policy_aggregate.csv" \
  --output_dir "$EVID" \
  --output_prefix v37_d7_fixed_outputdiff \
  2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e

{
  printf 'dta_v3_7_phase_d7_fixed_outputdiff_done rc=%s %s\n' "$rc" "$(date -Is)"
  if [[ "$rc" -eq 0 ]]; then
    printf 'DTA_V3_7_PHASE_D7_FIXED_OUTPUTDIFF_CONFIRM_OK %s\n' "$(date -Is)"
  else
    printf 'DTA_V3_7_PHASE_D7_FIXED_OUTPUTDIFF_CONFIRM_FAILED %s\n' "$(date -Is)"
  fi
} >> "$STATUS"

exit "$rc"
