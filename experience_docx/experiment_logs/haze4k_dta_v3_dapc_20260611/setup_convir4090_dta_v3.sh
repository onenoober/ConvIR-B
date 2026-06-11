#!/usr/bin/env bash
set -euo pipefail
BASE=${BASE:-/sda/home/wangyuxin/ConvIR-B}
WORK=${WORK:-$BASE/repos/ConvIR-B-dta-v3-dapc-finetune}
BRANCH=${BRANCH:-codex/haze4k-dta-v3-dapc-finetune}
PY=${PY:-$BASE/envs/convir-cu121/bin/python}
DATA=${DATA:-$BASE/datasets/Haze4K/Haze4K}
A0=${A0:-$BASE/checkpoints/official/Haze4K/haze4k-base.pkl}
EVID=$WORK/experience_docx/experiment_logs/haze4k_dta_v3_dapc_20260611
STATUS=$EVID/status.txt
mkdir -p "$BASE/repos"
if [[ ! -d "$WORK/.git" ]]; then
  git clone git@github.com:onenoober/ConvIR-B.git "$WORK"
fi
cd "$WORK"
git fetch github '+refs/heads/*:refs/remotes/github/*'
git switch "$BRANCH" 2>/dev/null || git switch -c "$BRANCH" "github/$BRANCH"
git pull --ff-only github "$BRANCH"
mkdir -p "$EVID"
{
  echo "setup_start dta_v3 $(date --iso-8601=seconds)"
  echo "work=$WORK"
  echo "branch=$(git branch --show-current)"
  echo "commit=$(git rev-parse --short HEAD)"
  echo "python=$PY"
  echo "data=$DATA"
  echo "a0=$A0"
} | tee -a "$STATUS"

DEPTH_CANDIDATES=(
  "$BASE/depth_cache/depth_anything_v2_small_hf"
  "$BASE/depth_cache/depth_anything_v2_small"
  "$BASE/datasets/Haze4K/depth_anything_v2_small_hf"
)
DEPTH_FOUND=""
for path in "${DEPTH_CANDIDATES[@]}"; do
  if [[ -d "$path" ]]; then
    DEPTH_FOUND="$path"
    break
  fi
done
if [[ -z "$DEPTH_FOUND" ]]; then
  echo "MISSING_DEPTH_CACHE candidates=${DEPTH_CANDIDATES[*]}" | tee -a "$STATUS"
  exit 2
fi
{
  echo "depth=$DEPTH_FOUND"
  test -x "$PY" && echo "python_exists=1" || echo "python_exists=0"
  test -d "$DATA" && echo "data_exists=1" || echo "data_exists=0"
  test -f "$A0" && echo "a0_exists=1" || echo "a0_exists=0"
} | tee -a "$STATUS"

cd "$WORK"
"$PY" -m py_compile \
  Dehazing/ITS/models/ConvIR.py \
  Dehazing/ITS/main.py \
  Dehazing/ITS/train.py \
  Dehazing/ITS/eval.py \
  Dehazing/ITS/valid.py \
  experience_docx/tools/check_haze4k_dta_preflight.py \
  experience_docx/tools/eval_haze4k_checkpoint_compare.py \
  experience_docx/tools/audit_haze4k_dta_v2_checkpoint.py \
  experience_docx/tools/audit_haze4k_dta_depth_pairing.py \
  experience_docx/tools/aggregate_haze4k_dta_v3_controls.py \
  experience_docx/tools/make_haze4k_dta_contact_sheet.py

echo "setup_done rc=0 dta_v3 $(date --iso-8601=seconds)" | tee -a "$STATUS"
echo "DTA_V3_SETUP_OK depth=$DEPTH_FOUND" | tee -a "$STATUS"
