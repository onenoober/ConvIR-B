#!/usr/bin/env bash
set -euo pipefail

STAGE=${STAGE:-scout5full}
SEED=${SEED:-3407}
FOLD=${FOLD:-0}
BASE=${BASE:-/sda/home/wangyuxin/ConvIR-B}
WORK=${WORK:-$BASE/repos/ConvIR-B-dta-v3-dapc-finetune-v32}
EVID=$WORK/experience_docx/experiment_logs/haze4k_dta_v3_dapc_20260611
STATUS=$EVID/status.txt
SCRIPT=$EVID/run_dta_v3_2_safemix_scout_convir4090.sh

mkdir -p "$EVID"
echo "dta_v3_2_safemix_launcher_start stage=$STAGE seed=$SEED fold=$FOLD $(date --iso-8601=seconds)" | tee -a "$STATUS"
cd "$WORK"
{
  echo "work=$WORK"
  git branch --show-current
  git rev-parse --short HEAD
  git status --short
  echo "locked_test_touched=false"
} | tee -a "$STATUS"

if [[ ! -f "$SCRIPT" ]]; then
  echo "DTA_V3_2_SAFEMIX_MISSING_SCRIPT $SCRIPT" | tee -a "$STATUS"
  exit 3
fi

launch_one() {
  local variant=$1
  local gpu=$2
  local session=dta_v32_${variant}_s${SEED}_f${FOLD}_${STAGE}
  local log=$EVID/${session}_tmux.log
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "DTA_V3_2_SAFEMIX_SESSION_ACTIVE session=$session" | tee -a "$STATUS"
    return 2
  fi
  echo "dta_v3_2_safemix_launch variant=$variant gpu=$gpu session=$session $(date --iso-8601=seconds)" | tee -a "$STATUS"
  tmux new-session -d -s "$session" \
    "cd '$WORK' && VARIANT='$variant' STAGE='$STAGE' SEED='$SEED' FOLD='$FOLD' CUDA_VISIBLE_DEVICES='$gpu' bash '$SCRIPT' 2>&1 | tee '$log'"
}

launch_one c1_gate "${GPU_C1:-2}"
launch_one c3_full "${GPU_C3:-3}"

tmux ls | tee -a "$STATUS"
echo "DTA_V3_2_SAFEMIX_LAUNCHER_OK $(date --iso-8601=seconds)" | tee -a "$STATUS"
