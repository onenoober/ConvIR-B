#!/usr/bin/env bash
set -euo pipefail
ROOT=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v24-c12-wd0375-distill
PY=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
EVID=$ROOT/experience_docx/experiment_logs/haze4k_v2_4_c12_wd0375_distill_20260615
DATA=/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K
CACHE=/sda/home/wangyuxin/ConvIR-B/runtime_cache/v24_c12_wd0375_teacher/train_core
STATUS=$EVID/status_teacher_cache.txt
mkdir -p "$EVID/runtime_logs" "$CACHE"
{
  echo "state=RUNNING_AUDIT"
  echo "run_id=v24_c12_teacher_cache_shards"
  echo "start_time=$(date --iso-8601=seconds)"
  echo "locked_test_touched=false"
  echo "locked_per_image_read=false"
  echo "source_commit=$(git -C "$ROOT" rev-parse HEAD)"
  echo "num_shards=7"
} > "$STATUS"
pids=()
for shard in 0 1 2 3 4 5 6; do
  gpu=$shard
  log="$EVID/runtime_logs/v24_c12_teacher_cache_shard${shard}.log"
  (
    export CUDA_VISIBLE_DEVICES=$gpu
    "$PY" "$ROOT/experience_docx/tools/cache_haze4k_v24_c12_wd0375_teacher.py" \
      --repo-root "$ROOT" \
      --convir-dir "$ROOT/Dehazing/ITS" \
      --data-dir "$DATA" \
      --split-manifest "$EVID/v24_c12_split_manifest.json" \
      --scope train_core \
      --a0-checkpoint /sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl \
      --wdmamba-repo /sda/home/wangyuxin/ConvIR-B/repos/external_experts/WDMamba \
      --wdmamba-checkpoint /sda/home/wangyuxin/ConvIR-B/checkpoints/WDMamba_ckpts/haze4k_35.88.pth \
      --cache-dir "$CACHE" \
      --out-dir "$EVID" \
      --shard-index "$shard" \
      --num-shards 7 \
      --print-freq 25
  ) > "$log" 2>&1 &
  pids+=("$!")
  echo "launched_shard=$shard gpu=$gpu pid=${pids[-1]}" | tee -a "$STATUS"
done
rc=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then rc=1; fi
done
count=$(find "$CACHE" -maxdepth 1 -type f -name '*.png' | wc -l)
{
  echo "cache_png_count=$count"
  echo "finish_time=$(date --iso-8601=seconds)"
  if [ "$rc" -eq 0 ]; then
    echo "state=COMPLETED_TEACHER_CACHE"
    echo "C12_TEACHER_CACHE_SHARDS_OK"
  else
    echo "state=FAILED_COMMAND"
    echo "C12_TEACHER_CACHE_SHARDS_FAILED"
  fi
} >> "$STATUS"
exit "$rc"
