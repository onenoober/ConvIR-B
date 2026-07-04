#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v2-27-nopost-ilfrb-action-conditioned-selective-distill
EVID=$REMOTE_ROOT/experience_docx/experiment_logs/haze4k_v2_27_nopost_ilfrb_action_conditioned_selective_distill_20260705
PY=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
DATA=/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K
CKPT=/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl
SPLIT=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v2-17-nopost-lowband-alignment-tail-audit/experience_docx/experiment_logs/haze4k_v2_16_nopost_wavelet_lowband_decoder_20260703/v216_t1_per_image_band_deltas.csv
STATUS=$EVID/status.txt
LOG=$EVID/v227_p1_p5_diagnostics.log
MAX_IMAGES=${MAX_IMAGES:-80}
ORACLE_STEPS=${ORACLE_STEPS:-10}

mkdir -p "$EVID"
cd "$REMOTE_ROOT"

test -f "$EVID/v227_p0_decision.md"
grep -q 'P0_PASS_ILFRB_ACS_CONTRACT_IDENTITY_SOURCE_CLEAN' "$EVID/v227_p0_decision.md"

if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_ID=${GPU_ID:-$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F, '{gsub(/ /,"",$1); gsub(/ /,"",$2); print $2" "$1}' | sort -n | awk 'NR==1{print $2}')}
  export CUDA_VISIBLE_DEVICES=$GPU_ID
else
  GPU_ID=cpu
fi

{
  echo "v227_p1_p5_state=RUNNING_AUDIT"
  echo "v227_p1_p5_start $(date --iso-8601=seconds) gpu=$GPU_ID max_images=$MAX_IMAGES oracle_steps=$ORACLE_STEPS"
  echo "branch=$(git branch --show-current)"
  echo "commit=$(git rev-parse --short HEAD)"
  echo "remote_root=$REMOTE_ROOT"
  echo "python=$PY"
  echo "data=$DATA"
  echo "checkpoint=$CKPT"
  echo "split_csv=$SPLIT"
  echo "locked_test_touched=false"
  echo "training_launched=false"
} | tee -a "$STATUS"

test -x "$PY"
test -d "$DATA/train"
test -f "$CKPT"
test -f "$SPLIT"

set +e
"$PY" experience_docx/tools/nopost_v227_ilfrb_acs_diagnostics.py \
  --phases p1_p5 \
  --data-dir "$DATA" \
  --checkpoint "$CKPT" \
  --split-csv "$SPLIT" \
  --out-dir "$EVID" \
  --max-images "$MAX_IMAGES" \
  --oracle-steps "$ORACLE_STEPS" \
  --oracle-lr 0.08 \
  --oracle-delta-scale 0.50 \
  --probe-epochs 260 \
  --canary-epochs 360 \
  --print-freq 10 \
  2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e

decision=UNKNOWN
if [ -f "$EVID/v227_closeout.json" ]; then
  decision=$("$PY" - "$EVID/v227_closeout.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8")).get("decision", "UNKNOWN"))
PY
)
fi

state=FAILED_COMMAND
marker=V227_P1_P5_FAILED
if [ "$rc" -eq 0 ]; then
  state=COMPLETED_GATE_FAIL
  marker=V227_NORMAL_GATE_PAUSE
  case "$decision" in
    P5_PASS_RISK_COVERAGE_REPLAY_REVIEW_P6_MICROFIT_CARD)
      state=COMPLETED_GATE_PASS
      marker=V227_P1_P5_OK
      ;;
  esac
fi

{
  echo "v227_p1_p5_done rc=$rc decision=$decision $(date --iso-8601=seconds)"
  echo "v227_p1_p5_state=$state"
  echo "$marker"
} | tee -a "$STATUS"

exit "$rc"
