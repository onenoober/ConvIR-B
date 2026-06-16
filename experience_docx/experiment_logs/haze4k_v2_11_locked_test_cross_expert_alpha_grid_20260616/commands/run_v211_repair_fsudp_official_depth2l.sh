#!/usr/bin/env bash
set -euo pipefail

ROOT=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v22-c9-fixed-wdmamba-router-locked
EVID=$ROOT/experience_docx/experiment_logs/haze4k_v2_11_locked_test_cross_expert_alpha_grid_20260616
PY=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
TOOL=$EVID/commands/eval_v211_locked_cross_expert_alpha_grid.py
DEPTH_TOOL=$EVID/commands/make_v211_haze4k_depth2l_from_cache.py
DATA=/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K
DEPTH=/sda/home/wangyuxin/ConvIR-B/depth_cache/depth_anything_v2_small_hf/test
DEPTH2L=/sda/home/wangyuxin/ConvIR-B/depth_cache/depth_anything_v2_small_hf_depth2l/test
A0=/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl
FSUDP=/sda/home/wangyuxin/ConvIR-B/checkpoints/udpnet/FSNet_UDPNet_haze4k.ckpt
MBT=/sda/home/wangyuxin/ConvIR-B/checkpoints/mb-taylorformerv2/HAZE4K-L.pth
UDP_REPO=/sda/home/wangyuxin/ConvIR-B/repos/UDPNet
MBT_REPO=/sda/home/wangyuxin/ConvIR-B/repos/external_experts/MB-TaylorFormerV2
STATUS=$EVID/status_v211_repair_fsudp_official_depth2l.txt
MAIN_LOG=$EVID/runtime_logs/v211_repair_fsudp_official_depth2l_main.log
PREFIX=v211_haze4k_locked_cross_expert_alpha_grid
ALPHAS=(0 0.125 0.25 0.375 0.50 0.75 1.0)
REPAIR_ID=official_depth2l_pad8_$(date +%Y%m%d_%H%M%S)

mkdir -p "$EVID/runtime_logs" "$EVID/shards/fsudp" "$DEPTH2L"

{
  echo "run_start v211_repair_fsudp_official_depth2l $REPAIR_ID $(date --iso-8601=seconds)"
  echo "state=RUNNING_EVAL"
  echo "reason=invalidates previous FSNet+UDP preliminary run that used raw npy depth cache and fsudp pad factor 32"
  echo "locked_policy=diagnostic_grid_only_no_alpha_selection"
  echo "root=$ROOT"
  cd "$ROOT"
  echo "branch=$(git branch --show-current)"
  echo "commit=$(git rev-parse --short HEAD)"
  echo "status_count=$(git status --short | wc -l)"
  echo "python=$PY"
  echo "tool_sha256=$(sha256sum "$TOOL" | awk '{print $1}')"
  echo "depth_tool_sha256=$(sha256sum "$DEPTH_TOOL" | awk '{print $1}')"
  echo "depth_raw_count=$(find "$DEPTH" -maxdepth 1 -type f -name '*.npy' | wc -l)"
} | tee -a "$STATUS" | tee -a "$MAIN_LOG"

"$PY" "$DEPTH_TOOL" \
  --haze-dir "$DATA/test/haze" \
  --depth-cache-dir "$DEPTH" \
  --out-dir "$DEPTH2L" \
  --summary-json "$EVID/v211_depth2l_generation_summary.json" \
  --overwrite 2>&1 | tee -a "$MAIN_LOG"
echo "depth2l_count=$(find "$DEPTH2L" -maxdepth 1 -type f -name '*.png' | wc -l)" | tee -a "$STATUS" "$MAIN_LOG"

"$PY" "$TOOL" \
  --mode preflight \
  --data-dir "$DATA" \
  --data-split test \
  --depth-cache-dir "$DEPTH" \
  --depth2l-dir "$DEPTH2L" \
  --out-dir "$EVID" \
  --prefix "$PREFIX" \
  --a0-checkpoint "$A0" \
  --fsudp-checkpoint "$FSUDP" \
  --mbtaylor-checkpoint "$MBT" \
  --expert-checkpoint "$FSUDP" \
  --udp-repo "$UDP_REPO" \
  --mbtaylor-repo "$MBT_REPO" \
  --expected-count 1000 2>&1 | tee -a "$MAIN_LOG"
echo "preflight_depth2l_done $(date --iso-8601=seconds)" | tee -a "$STATUS" "$MAIN_LOG"

INVALID=$EVID/invalidated_fsudp_raw_npy_pad32_$REPAIR_ID
mkdir -p "$INVALID"
if test -f "$EVID/${PREFIX}_fsudp_per_image.csv"; then
  mv "$EVID/${PREFIX}_fsudp_per_image.csv" "$INVALID/"
fi
if test -f "$EVID/${PREFIX}_fsudp_alpha_grid_absolute_metrics.csv"; then
  mv "$EVID/${PREFIX}_fsudp_alpha_grid_absolute_metrics.csv" "$INVALID/"
fi
if test -f "$EVID/${PREFIX}_fsudp_alpha_grid_compact_metrics.csv"; then
  mv "$EVID/${PREFIX}_fsudp_alpha_grid_compact_metrics.csv" "$INVALID/"
fi
if test -d "$EVID/shards/fsudp"; then
  mv "$EVID/shards/fsudp" "$INVALID/shards_fsudp_raw_npy_pad32"
fi
mkdir -p "$EVID/shards/fsudp"
echo "invalidated_old_fsudp_dir=$INVALID" | tee -a "$STATUS" "$MAIN_LOG"

mapfile -t GPUS < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F, '{g=$1+0; mem=$2+0; util=$3+0; if (mem < 14000 && util < 85) print g}')
if [ "${#GPUS[@]}" -eq 0 ]; then
  echo "V211_REPAIR_NO_FREE_GPU" | tee -a "$STATUS" "$MAIN_LOG"
  exit 1
fi
SHARDS=${#GPUS[@]}
if [ "$SHARDS" -gt 4 ]; then
  SHARDS=4
fi
printf 'selected_gpus=%s\n' "${GPUS[*]:0:$SHARDS}" | tee -a "$STATUS" "$MAIN_LOG"
printf 'fsudp_repair_shards=%s\n' "$SHARDS" | tee -a "$STATUS" "$MAIN_LOG"

declare -a PIDS=()
declare -a FS_CSVS=()
for shard in $(seq 0 $((SHARDS-1))); do
  gpu=${GPUS[$shard]}
  OUT=$EVID/shards/fsudp/shard_${shard}
  LOG=$EVID/runtime_logs/fsudp_repair_depth2l_shard_${shard}_gpu_${gpu}.log
  mkdir -p "$OUT"
  FS_CSVS+=("$OUT/${PREFIX}_fsudp_shard${shard}_per_image.csv")
  echo "launch_repair_shard expert=fsudp shard=$shard shard_count=$SHARDS gpu=$gpu $(date --iso-8601=seconds)" | tee -a "$STATUS" "$MAIN_LOG"
  (
    set -euo pipefail
    CUDA_VISIBLE_DEVICES=$gpu "$PY" "$TOOL" \
      --mode evaluate \
      --expert fsudp \
      --data-dir "$DATA" \
      --data-split test \
      --depth-cache-dir "$DEPTH" \
      --depth2l-dir "$DEPTH2L" \
      --fsudp-pad-factor 8 \
      --out-dir "$OUT" \
      --prefix "${PREFIX}_fsudp_shard${shard}" \
      --convir-its-dir "$ROOT/Dehazing/ITS" \
      --a0-checkpoint "$A0" \
      --fsudp-checkpoint "$FSUDP" \
      --mbtaylor-checkpoint "$MBT" \
      --expert-checkpoint "$FSUDP" \
      --udp-repo "$UDP_REPO" \
      --mbtaylor-repo "$MBT_REPO" \
      --alphas "${ALPHAS[@]}" \
      --shard-index "$shard" \
      --shard-count "$SHARDS" \
      --print-freq 10
  ) > "$LOG" 2>&1 &
  PIDS+=("$!")
done

fail=0
for idx in "${!PIDS[@]}"; do
  pid=${PIDS[$idx]}
  if wait "$pid"; then
    echo "repair_shard_done shard=$idx rc=0 $(date --iso-8601=seconds)" | tee -a "$STATUS" "$MAIN_LOG"
  else
    rc=$?
    echo "repair_shard_done shard=$idx rc=$rc $(date --iso-8601=seconds)" | tee -a "$STATUS" "$MAIN_LOG"
    fail=1
  fi
done
if [ "$fail" -ne 0 ]; then
  echo "V211_REPAIR_FSUDP_SHARD_FAILED" | tee -a "$STATUS" "$MAIN_LOG"
  exit 1
fi

mapfile -t MB_CSVS < <(find "$EVID/shards/mbtaylor" -maxdepth 2 -type f -name "${PREFIX}_mbtaylor_shard*_per_image.csv" | sort)
if [ "${#MB_CSVS[@]}" -eq 0 ]; then
  echo "V211_REPAIR_MBTAYLOR_CSVS_MISSING" | tee -a "$STATUS" "$MAIN_LOG"
  exit 1
fi
UDP_HEAD=$(cd "$UDP_REPO" && git rev-parse --short HEAD 2>/dev/null || true)
MBT_HEAD=$(cd "$MBT_REPO" && git rev-parse --short HEAD 2>/dev/null || true)
"$PY" "$TOOL" \
  --mode aggregate \
  --out-dir "$EVID" \
  --prefix "$PREFIX" \
  --a0-checkpoint "$A0" \
  --fsudp-checkpoint "$FSUDP" \
  --mbtaylor-checkpoint "$MBT" \
  --expert-checkpoint "$FSUDP" \
  --udp-repo "$UDP_REPO" \
  --mbtaylor-repo "$MBT_REPO" \
  --udp-repo-head "$UDP_HEAD" \
  --mbtaylor-repo-head "$MBT_HEAD" \
  --alphas "${ALPHAS[@]}" \
  --fsudp-input-csvs "${FS_CSVS[@]}" \
  --mbtaylor-input-csvs "${MB_CSVS[@]}" \
  --expected-count 1000 2>&1 | tee -a "$MAIN_LOG"

{
  echo "repair_postprocess_done $(date --iso-8601=seconds)"
  echo "V211_REPAIR_FSUDP_OFFICIAL_DEPTH2L_OK"
  echo "state=COMPLETED_GATE_PASS"
} | tee -a "$STATUS" "$MAIN_LOG"
