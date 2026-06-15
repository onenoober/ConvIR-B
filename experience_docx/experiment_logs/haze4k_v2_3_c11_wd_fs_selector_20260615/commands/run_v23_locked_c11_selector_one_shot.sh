#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PY="/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python"
EVID="$ROOT/experience_docx/experiment_logs/haze4k_v2_3_c11_wd_fs_selector_20260615"
STATUS="$EVID/status_locked_c11_selector_one_shot.txt"
LOG="$EVID/v23_locked_c11_selector_one_shot.log"
SUMMARY="$EVID/v23_locked_c11_selector_one_shot_summary.json"
PREFIX=v23_locked_c11_selector_one_shot

if [ -e "$STATUS" ] || [ -e "$SUMMARY" ]; then
  echo "V23_LOCKED_C11_SELECTOR_REFUSE_EXISTING_OUTPUT status=$STATUS summary=$SUMMARY" | tee -a "$LOG"
  exit 3
fi

mkdir -p "$EVID"
{
  echo "v23_locked_c11_selector_one_shot_start $(date -Is)"
  echo "remote_root=$ROOT"
  echo "fixed_selector=v23_c11e_sealed_selector.json"
  echo "locked_test_authorized_by=C11E_SEALED_SELECTOR_PASS_READY_FOR_LOCKED_ONE_SHOT_REVIEW"
  echo "locked_test_touched=true"
  echo "one_shot=true"
  echo "no_tuning_from_locked=true"
  git -C "$ROOT" rev-parse HEAD | sed 's/^/source_commit=/'
} | tee "$STATUS"

cd "$ROOT"
set +e
"$PY" experience_docx/tools/audit_haze4k_v23_c11_locked_selector_one_shot.py \
  --repo-root "$ROOT" \
  --convir-its-dir "$ROOT/Dehazing/ITS" \
  --udp-repo "/sda/home/wangyuxin/ConvIR-B/repos/UDPNet" \
  --data-dir "/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K" \
  --data-split test \
  --depth-cache-dir "/sda/home/wangyuxin/ConvIR-B/depth_cache/depth_anything_v2_small_hf" \
  --a0-checkpoint "/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl" \
  --fulludp-checkpoint "/sda/home/wangyuxin/ConvIR-B/checkpoints/udpnet/ConvIR_UDPNet_haze4k.ckpt" \
  --fsudp-checkpoint "/sda/home/wangyuxin/ConvIR-B/checkpoints/udpnet/FSNet_UDPNet_haze4k.ckpt" \
  --wdmamba-repo "/sda/home/wangyuxin/ConvIR-B/repos/external_experts/WDMamba" \
  --wdmamba-checkpoint "/sda/home/wangyuxin/ConvIR-B/checkpoints/WDMamba_ckpts/haze4k_35.88.pth" \
  --sealed-selector "$EVID/v23_c11e_sealed_selector.json" \
  --sealed-decision "$EVID/v23_c11e_sealed_selector_decision.md" \
  --out-dir "$EVID" \
  --prefix "$PREFIX" \
  --print-freq 25 \
  2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e
echo "v23_locked_c11_selector_one_shot_done rc=$rc $(date -Is)" | tee -a "$STATUS"
if [ "$rc" -eq 0 ]; then
  echo "V23_LOCKED_C11_SELECTOR_ONE_SHOT_OK $(date -Is)" | tee -a "$STATUS"
else
  echo "V23_LOCKED_C11_SELECTOR_ONE_SHOT_FAILED rc=$rc $(date -Is)" | tee -a "$STATUS"
fi
exit "$rc"
