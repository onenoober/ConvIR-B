#!/usr/bin/env bash
set -euo pipefail

TARGET=${TARGET:-convir-4090}
BASE=${BASE:-/sda/home/wangyuxin/ConvIR-B}
REPO_URL=${REPO_URL:-https://github.com/onenoober/ConvIR-B.git}
BRANCH=${BRANCH:-codex/haze4k-dta-v2-calibrated}
REMOTE_ROOT=${REMOTE_ROOT:-$BASE/repos/ConvIR-B-dta-v2-calibrated}
ENV_ROOT=${ENV_ROOT:-$BASE/envs/convir-cu121}
DATA=$BASE/datasets/Haze4K/Haze4K
A0=$BASE/checkpoints/official/Haze4K/haze4k-base.pkl
DEPTH=$BASE/depth_cache/depth_anything_v2_small_hf
EVID=$REMOTE_ROOT/experience_docx/experiment_logs/haze4k_dta_v2_calibrated_20260611
STATUS=$EVID/convir4090_setup_status.txt

echo "setup_start dta_v2 convir4090 $(date --iso-8601=seconds)"
ssh -o BatchMode=yes "$TARGET" 'printf "TARGET_SSH_OK host=%s user=%s home=%s\n" "$(hostname)" "$(whoami)" "$HOME"'
ssh "$TARGET" "bash -s" <<REMOTE
set -euo pipefail
mkdir -p "$BASE"/{repos,envs,checkpoints/official/Haze4K,depth_cache,datasets/Haze4K,logs,tmp}
if [ -d "$REMOTE_ROOT/.git" ]; then
  cd "$REMOTE_ROOT"
  if [ -n "\$(git status --short -- . ':!experience_docx/experiment_logs/haze4k_dta_v2_calibrated_20260611')" ]; then
    echo "REMOTE_REPO_DIRTY $REMOTE_ROOT" >&2
    git status --short -- . ':!experience_docx/experiment_logs/haze4k_dta_v2_calibrated_20260611' >&2
    exit 3
  fi
  git fetch origin "$BRANCH"
  git switch "$BRANCH" || git switch -c "$BRANCH" "origin/$BRANCH"
  git pull --ff-only origin "$BRANCH"
else
  git clone --branch "$BRANCH" "$REPO_URL" "$REMOTE_ROOT"
fi
mkdir -p "$EVID"
{
  echo "setup_remote_repo_ok \$(date --iso-8601=seconds)"
  echo "base=$BASE"
  echo "remote_root=$REMOTE_ROOT"
  echo "env_root=$ENV_ROOT"
  echo "data=$DATA"
  echo "checkpoint=$A0"
  echo "depth=$DEPTH"
  cd "$REMOTE_ROOT"
  echo "branch=\$(git branch --show-current)"
  echo "commit=\$(git rev-parse --short HEAD)"
  echo "status=\$(git status --short | wc -l) dirty_lines"
} | tee -a "$STATUS"
if [ ! -x "$ENV_ROOT/bin/python" ]; then
  echo "FAILED_INFRA_MISSING_PYTHON $ENV_ROOT/bin/python" | tee -a "$STATUS"
  exit 2
fi
if [ ! -f "$A0" ]; then
  echo "FAILED_INFRA_MISSING_A0 $A0" | tee -a "$STATUS"
  exit 2
fi
if [ ! -d "$DEPTH/train" ] || [ ! -d "$DEPTH/test" ]; then
  echo "FAILED_INFRA_MISSING_DEPTH_CACHE $DEPTH" | tee -a "$STATUS"
  exit 2
fi
if [ ! -d "$DATA/train" ] || [ ! -d "$DATA/test" ]; then
  echo "FAILED_INFRA_MISSING_DATA $DATA" | tee -a "$STATUS"
  exit 2
fi
cd "$REMOTE_ROOT"
"$ENV_ROOT/bin/python" -m py_compile \
  Dehazing/ITS/models/ConvIR.py \
  Dehazing/ITS/main.py \
  Dehazing/ITS/train.py \
  Dehazing/ITS/valid.py \
  Dehazing/ITS/eval.py \
  Dehazing/ITS/data/data_load.py \
  Dehazing/ITS/data/data_augment.py \
  experience_docx/tools/check_haze4k_dta_preflight.py \
  experience_docx/tools/eval_haze4k_checkpoint_compare.py \
  experience_docx/tools/audit_haze4k_depth_transmission.py \
  experience_docx/tools/make_haze4k_dta_oof_splits.py
find "$DATA/train/haze" -maxdepth 1 -type f | wc -l | awk '{print "train_haze_count="\$1}' | tee -a "$STATUS"
find "$DATA/train/gt" -maxdepth 1 -type f | wc -l | awk '{print "train_gt_count="\$1}' | tee -a "$STATUS"
find "$DATA/train/trans" -maxdepth 1 -type f | wc -l | awk '{print "train_trans_count="\$1}' | tee -a "$STATUS"
find "$DATA/test/haze" -maxdepth 1 -type f | wc -l | awk '{print "test_haze_count="\$1}' | tee -a "$STATUS"
echo "CONVIR4090_DTA_V2_SETUP_OK" | tee -a "$STATUS"
REMOTE
ssh "$TARGET" "cat '$STATUS'; echo CONVIR4090_DTA_V2_SETUP_SCRIPT_OK"
echo "setup_done dta_v2 convir4090 $(date --iso-8601=seconds)"
