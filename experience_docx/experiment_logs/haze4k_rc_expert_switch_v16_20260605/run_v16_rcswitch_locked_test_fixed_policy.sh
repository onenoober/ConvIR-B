#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-/root/miniconda3/envs/convir-cu128/bin/python}
WORK=${WORK:-/root/autodl-tmp/workspace/ConvIR-B-v1-6-rcswitch-runtime}
CONVIR_ITS=${CONVIR_ITS:-$WORK/Dehazing/ITS}
UDP_REPO=${UDP_REPO:-/root/autodl-tmp/workspace/UDPNet}
DATA=${DATA:-/root/autodl-tmp/workspace/Dehaze-Net/dataset/HAZE4K}
DEPTH=${DEPTH:-/root/autodl-tmp/workspace/Dehaze-Net/experiment/HAZE4K/depth_cache/depth_anything_v2_small_hf}
A0=${A0:-/root/autodl-tmp/workspace/ConvIR-B/Dehazing/pretrained_models/haze4k-base.pkl}
OFFICIAL_CKPT=${OFFICIAL_CKPT:-/root/autodl-tmp/workspace/UDPNet_official_download/ConvIR_UDPNet_haze4k.ckpt}

FEATURE=udp_a0_luma_shift_mean
DIRECTION=low
THRESHOLD=-0.003969017509371043

EVID=$WORK/experience_docx/experiment_logs/haze4k_rc_expert_switch_v16_20260605
OUT=$EVID/locked_test_fixed_policy
STATUS=$EVID/status.txt
LOG=$EVID/v16_rcswitch_locked_test_fixed_policy.log

mkdir -p "$OUT"
{
  echo "v16_locked_test_fixed_policy_start $(date --iso-8601=seconds)"
  echo "state=RUNNING_EVAL"
  echo "locked_test_touched=YES"
  echo "feature=$FEATURE"
  echo "direction=$DIRECTION"
  echo "threshold=$THRESHOLD"
  echo "work=$WORK"
  echo "python=$PY"
  echo "data=$DATA"
  echo "depth=$DEPTH"
  echo "a0=$A0"
  echo "official_ckpt=$OFFICIAL_CKPT"
  if [ -f "$OFFICIAL_CKPT" ]; then
    sha256sum "$OFFICIAL_CKPT" | sed 's/^/official_ckpt_sha256=/'
  fi
} | tee -a "$STATUS"

cd "$WORK"
set +e
PYTHONUNBUFFERED=1 "$PY" experience_docx/tools/eval_haze4k_rcswitch_locked_test.py \
  --convir_its_dir "$CONVIR_ITS" \
  --udp_repo "$UDP_REPO" \
  --data_dir "$DATA" \
  --depth_cache_dir "$DEPTH" \
  --a0_checkpoint "$A0" \
  --official_checkpoint "$OFFICIAL_CKPT" \
  --depth_split test \
  --feature "$FEATURE" \
  --direction "$DIRECTION" \
  --threshold "$THRESHOLD" \
  --output_dir "$OUT" \
  2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e
echo "v16_locked_test_fixed_policy_done rc=$rc output=$OUT $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [ "$rc" -eq 0 ]; then
  echo "state=LOCKED_TEST_COMPLETE_PENDING_DOC_SYNC $(date --iso-8601=seconds)" | tee -a "$STATUS"
  echo "V16_RCSWITCH_LOCKED_TEST_FIXED_POLICY_OK"
  exit 0
fi
echo "state=FAILED_COMMAND locked_test_rc=$rc $(date --iso-8601=seconds)" | tee -a "$STATUS"
echo "V16_RCSWITCH_LOCKED_TEST_FIXED_POLICY_FAILED rc=$rc"
exit "$rc"
