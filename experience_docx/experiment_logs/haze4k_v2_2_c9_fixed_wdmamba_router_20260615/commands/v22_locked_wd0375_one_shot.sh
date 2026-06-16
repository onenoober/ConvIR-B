#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PY="/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python"
EVID="$ROOT/experience_docx/experiment_logs/haze4k_v2_2_c9_fixed_wdmamba_router_20260615"
STATUS="$EVID/status_locked_wd0375_one_shot.txt"
LOG="$EVID/v22_locked_wd0375_one_shot.log"
SUMMARY="$EVID/v22_locked_wd0375_one_shot_summary.json"

if [ -e "$STATUS" ] || [ -e "$SUMMARY" ]; then
  echo "V22_LOCKED_WD0375_REFUSE_EXISTING_OUTPUT status=$STATUS summary=$SUMMARY" | tee -a "$LOG"
  exit 3
fi

mkdir -p "$EVID"
{
  echo "v22_locked_wd0375_one_shot_start $(date -Is)"
  echo "remote_root=$ROOT"
  echo "fixed_profile=WD0375"
  echo "alpha=0.375"
  echo "locked_test_authorized_by=C10_FORMAL_5X3_WD0375_PASS_AUTHORIZE_LOCKED_ONE_SHOT_REVIEW"
  echo "locked_test_touched=true"
  echo "one_shot=true"
  echo "no_tuning_from_locked=true"
  git -C "$ROOT" rev-parse HEAD | sed 's/^/source_commit=/'
} | tee "$STATUS"

cd "$ROOT"
set +e
"$PY" experience_docx/tools/audit_haze4k_v22_c10_wd0375_locked_one_shot.py \
  --convir-its-dir "$ROOT/Dehazing/ITS" \
  --data-dir "/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K" \
  --data-split test \
  --a0-checkpoint "/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl" \
  --wdmamba-repo "/sda/home/wangyuxin/ConvIR-B/repos/external_experts/WDMamba" \
  --wdmamba-checkpoint "/sda/home/wangyuxin/ConvIR-B/checkpoints/WDMamba_ckpts/haze4k_35.88.pth" \
  --c10-decision "$EVID/v22_c10_formal_5x3_decision.md" \
  --out-dir "$EVID" \
  --prefix v22_locked_wd0375_one_shot \
  --print-freq 50 \
  2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e
echo "v22_locked_wd0375_one_shot_done rc=$rc $(date -Is)" | tee -a "$STATUS"
if [ "$rc" -eq 0 ]; then
  echo "V22_LOCKED_WD0375_ONE_SHOT_OK $(date -Is)" | tee -a "$STATUS"
else
  echo "V22_LOCKED_WD0375_ONE_SHOT_FAILED rc=$rc $(date -Is)" | tee -a "$STATUS"
fi
exit "$rc"
