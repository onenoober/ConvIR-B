#!/usr/bin/env bash
set -euo pipefail
ROOT=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v28-nhhaze-official-weights
EVID=$ROOT/experience_docx/experiment_logs/haze4k_v2_9_nhhaze_official_test_alpha_grid_20260616
PY=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
TOOL=$ROOT/experience_docx/tools/eval_nhhaze_v28_official_weights.py
DATA=/sda/home/wangyuxin/ConvIR-B/datasets/NH-HAZE-official-test-51-55
A0=/sda/home/wangyuxin/ConvIR-B/checkpoints/official/nhhaze-base.pkl
WDM=/sda/home/wangyuxin/ConvIR-B/checkpoints/WDMamba_ckpts/NH_20.83.pth
STATUS=$EVID/status_v29_nhhaze_official_test_alpha_grid.txt
LOGDIR=$EVID/runtime_logs
LOG=$LOGDIR/v29_nhhaze_official_test_alpha_grid.log
PREFIX=v29_nhhaze_official_test_alpha_grid
mkdir -p "$EVID" "$LOGDIR"
PAIR_COUNT=$(find -L "$DATA" -maxdepth 1 -type f -name '*_hazy.png' | wc -l)
GT_COUNT=$(find -L "$DATA" -maxdepth 1 -type f -name '*_GT.png' | wc -l)
IDS=$(find -L "$DATA" -maxdepth 1 -type f -name '*_hazy.png' -printf '%f\n' | sort | sed 's/_hazy.png//' | tr '\n' ' ' | sed 's/ $//')
{
  echo "v29_start $(date --iso-8601=seconds)"
  echo "host=$(hostname)"
  echo "root=$ROOT"
  echo "data=$DATA"
  echo "official_split=test_51_55_only"
  echo "stage_pair_count=$PAIR_COUNT stage_gt_count=$GT_COUNT stage_ids=$IDS"
  echo "a0=$A0"
  echo "wdmamba=$WDM"
  echo "a0_sha256=$(sha256sum \"$A0\" | awk '{print $1}')"
  echo "wdmamba_sha256=$(sha256sum \"$WDM\" | awk '{print $1}')"
  if git -C "$ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "branch=$(git -C \"$ROOT\" branch --show-current || true)"
    echo "commit=$(git -C \"$ROOT\" rev-parse HEAD || true)"
    echo 'status_short_begin'
    git -C "$ROOT" status --short || true
    echo 'status_short_end'
  else
    echo 'branch=RSYNC_SNAPSHOT_NO_REMOTE_GIT'
    echo "commit=$(cat \"$ROOT/.source_commit\" 2>/dev/null || echo UNKNOWN)"
    echo 'status_short_begin'
    echo 'remote_git_metadata=unavailable_snapshot_rsync'
    echo 'status_short_end'
  fi
} | tee "$STATUS"
if [ "$PAIR_COUNT" -ne 5 ] || [ "$GT_COUNT" -ne 5 ] || [ "$IDS" != '51 52 53 54 55' ]; then
  echo 'FAILED_STAGE_SPLIT_INVALID_AT_RUNTIME' | tee -a "$STATUS"
  exit 3
fi
"$PY" -m py_compile "$TOOL" | tee -a "$STATUS"
echo "py_compile_ok $(date --iso-8601=seconds)" | tee -a "$STATUS"
set +e
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} "$PY" "$TOOL" \
  --mode all \
  --data-dir "$DATA" \
  --out-dir "$EVID" \
  --prefix "$PREFIX" \
  --convir-its-dir "$ROOT/Dehazing/ITS" \
  --a0-data NHR \
  --a0-checkpoint "$A0" \
  --wdmamba-checkpoint "$WDM" \
  --wdmamba-repo /sda/home/wangyuxin/ConvIR-B/repos/external_experts/WDMamba \
  --wdmamba-de-blocks 4 \
  --alphas 0 0.125 0.25 0.375 0.50 0.75 1.0 \
  --print-freq 1 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e
echo "v29_done rc=$rc $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [ "$rc" -eq 0 ]; then
  echo 'V29_NHHAZE_OFFICIAL_TEST_ALPHA_GRID_OK' | tee -a "$STATUS"
else
  echo 'V29_NHHAZE_OFFICIAL_TEST_ALPHA_GRID_FAILED' | tee -a "$STATUS"
fi
exit "$rc"
