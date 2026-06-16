#!/usr/bin/env bash
set -euo pipefail

ROOT=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v22-c9-fixed-wdmamba-router-locked
EVID=$ROOT/experience_docx/experiment_logs/haze4k_v2_11_locked_test_cross_expert_alpha_grid_20260616
PY=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
TOOL=$EVID/commands/eval_v211_locked_cross_expert_alpha_grid.py
DATA=/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K
DEPTH=/sda/home/wangyuxin/ConvIR-B/depth_cache/depth_anything_v2_small_hf/test
DEPTH2L=/sda/home/wangyuxin/ConvIR-B/depth_cache/depth_anything_v2_small_hf_depth2l/test
A0=/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl
FSUDP=/sda/home/wangyuxin/ConvIR-B/checkpoints/udpnet/FSNet_UDPNet_haze4k.ckpt
MBT=/sda/home/wangyuxin/ConvIR-B/checkpoints/mb-taylorformerv2/HAZE4K-L.pth
UDP_REPO=/sda/home/wangyuxin/ConvIR-B/repos/UDPNet
MBT_REPO=/sda/home/wangyuxin/ConvIR-B/repos/external_experts/MB-TaylorFormerV2
STATUS=$EVID/status_v211_locked_cross_expert_alpha_grid.txt
MAIN_LOG=$EVID/runtime_logs/v211_locked_cross_expert_alpha_grid_main.log
PREFIX=v211_haze4k_locked_cross_expert_alpha_grid
ALPHAS=(0 0.125 0.25 0.375 0.50 0.75 1.0)

mkdir -p "$EVID/runtime_logs" "$EVID/shards/fsudp" "$EVID/shards/mbtaylor"

{
  echo "run_start v211_locked_cross_expert_alpha_grid $(date --iso-8601=seconds)"
  echo "state=RUNNING_EVAL"
  echo "locked_test_touched=true"
  echo "locked_policy=diagnostic_grid_only_no_alpha_selection"
  echo "metric_protocol=v2.2_locked_one_shot_compatible_alpha_candidates_plus_endpoint_repro"
  echo "root=$ROOT"
  cd "$ROOT"
  if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "root_is_git=1"
    echo "branch=$(git branch --show-current)"
    echo "commit=$(git rev-parse --short HEAD)"
    echo "status_count=$(git status --short | wc -l)"
  else
    echo "root_is_git=0"
  fi
  echo "python=$PY"
  "$PY" - <<'PYINFO'
import sys, torch
print('python_version=' + sys.version.split()[0])
print('torch_version=' + torch.__version__)
print('cuda_available=' + str(torch.cuda.is_available()))
print('cuda_device_count=' + str(torch.cuda.device_count()))
PYINFO
  echo "tool=$TOOL"
  echo "tool_sha256=$(sha256sum "$TOOL" | awk '{print $1}')"
  echo "data=$DATA"
  echo "test_haze_count=$(find "$DATA/test/haze" -maxdepth 1 -type f -iname '*.png' | wc -l)"
  echo "test_gt_count=$(find "$DATA/test/gt" -maxdepth 1 -type f -iname '*.png' | wc -l)"
  echo "depth_count=$(find "$DEPTH" -maxdepth 1 -type f -name '*.npy' | wc -l)"
  echo "depth2l_count=$(find "$DEPTH2L" -maxdepth 1 -type f -name '*.png' 2>/dev/null | wc -l)"
  echo "a0=$A0"
  echo "a0_sha256=$(sha256sum "$A0" | awk '{print $1}')"
  echo "fsudp=$FSUDP"
  echo "fsudp_sha256=$(sha256sum "$FSUDP" | awk '{print $1}')"
  echo "mbtaylor=$MBT"
  echo "mbtaylor_sha256=$(sha256sum "$MBT" | awk '{print $1}')"
  echo "udp_repo=$UDP_REPO"
  echo "udp_repo_head=$(cd "$UDP_REPO" && git rev-parse --short HEAD 2>/dev/null || true)"
  echo "mbtaylor_repo=$MBT_REPO"
  echo "mbtaylor_repo_head=$(cd "$MBT_REPO" && git rev-parse --short HEAD 2>/dev/null || true)"
} | tee -a "$STATUS" | tee -a "$MAIN_LOG"

UDP_HEAD=$(cd "$UDP_REPO" && git rev-parse --short HEAD 2>/dev/null || true)
MBT_HEAD=$(cd "$MBT_REPO" && git rev-parse --short HEAD 2>/dev/null || true)

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
echo "preflight_done $(date --iso-8601=seconds)" | tee -a "$STATUS" "$MAIN_LOG"

mapfile -t GPUS < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F, '{g=$1+0; mem=$2+0; util=$3+0; if (mem < 14000 && util < 85) print g}')
if [ "${#GPUS[@]}" -lt 2 ]; then
  echo "V211_NOT_ENOUGH_FREE_GPU count=${#GPUS[@]}" | tee -a "$STATUS" "$MAIN_LOG"
  exit 1
fi

printf 'selected_gpus=%s\n' "${GPUS[*]}" | tee -a "$STATUS" "$MAIN_LOG"
TOTAL=${#GPUS[@]}
FS_SHARDS=$((TOTAL / 2))
MB_SHARDS=$((TOTAL - FS_SHARDS))
if [ "$FS_SHARDS" -lt 1 ] || [ "$MB_SHARDS" -lt 1 ]; then
  echo "V211_INVALID_SHARD_SPLIT fs=$FS_SHARDS mb=$MB_SHARDS" | tee -a "$STATUS" "$MAIN_LOG"
  exit 1
fi
printf 'fsudp_shards=%s\n' "$FS_SHARDS" | tee -a "$STATUS" "$MAIN_LOG"
printf 'mbtaylor_shards=%s\n' "$MB_SHARDS" | tee -a "$STATUS" "$MAIN_LOG"

declare -a PIDS=()
declare -a LABELS=()
declare -a FS_CSVS=()
declare -a MB_CSVS=()

for shard in $(seq 0 $((FS_SHARDS-1))); do
  gpu=${GPUS[$shard]}
  OUT=$EVID/shards/fsudp/shard_${shard}
  LOG=$EVID/runtime_logs/fsudp_shard_${shard}_gpu_${gpu}.log
  mkdir -p "$OUT"
  FS_CSVS+=("$OUT/${PREFIX}_fsudp_shard${shard}_per_image.csv")
  echo "launch_shard expert=fsudp shard=$shard shard_count=$FS_SHARDS gpu=$gpu $(date --iso-8601=seconds)" | tee -a "$STATUS" "$MAIN_LOG"
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
      --shard-count "$FS_SHARDS" \
      --print-freq 10
  ) > "$LOG" 2>&1 &
  PIDS+=("$!")
  LABELS+=("fsudp:$shard")
done

for shard in $(seq 0 $((MB_SHARDS-1))); do
  gpu=${GPUS[$((FS_SHARDS + shard))]}
  OUT=$EVID/shards/mbtaylor/shard_${shard}
  LOG=$EVID/runtime_logs/mbtaylor_shard_${shard}_gpu_${gpu}.log
  mkdir -p "$OUT"
  MB_CSVS+=("$OUT/${PREFIX}_mbtaylor_shard${shard}_per_image.csv")
  echo "launch_shard expert=mbtaylor shard=$shard shard_count=$MB_SHARDS gpu=$gpu $(date --iso-8601=seconds)" | tee -a "$STATUS" "$MAIN_LOG"
  (
    set -euo pipefail
    CUDA_VISIBLE_DEVICES=$gpu "$PY" "$TOOL" \
      --mode evaluate \
      --expert mbtaylor \
      --data-dir "$DATA" \
      --data-split test \
      --depth-cache-dir "$DEPTH" \
      --out-dir "$OUT" \
      --prefix "${PREFIX}_mbtaylor_shard${shard}" \
      --convir-its-dir "$ROOT/Dehazing/ITS" \
      --a0-checkpoint "$A0" \
      --fsudp-checkpoint "$FSUDP" \
      --mbtaylor-checkpoint "$MBT" \
      --expert-checkpoint "$MBT" \
      --udp-repo "$UDP_REPO" \
      --mbtaylor-repo "$MBT_REPO" \
      --alphas "${ALPHAS[@]}" \
      --shard-index "$shard" \
      --shard-count "$MB_SHARDS" \
      --print-freq 10
  ) > "$LOG" 2>&1 &
  PIDS+=("$!")
  LABELS+=("mbtaylor:$shard")
done

fail=0
for idx in "${!PIDS[@]}"; do
  pid=${PIDS[$idx]}
  label=${LABELS[$idx]}
  if wait "$pid"; then
    echo "shard_done label=$label rc=0 $(date --iso-8601=seconds)" | tee -a "$STATUS" "$MAIN_LOG"
  else
    rc=$?
    echo "shard_done label=$label rc=$rc $(date --iso-8601=seconds)" | tee -a "$STATUS" "$MAIN_LOG"
    fail=1
  fi
done
if [ "$fail" -ne 0 ]; then
  echo "V211_SHARD_FAILED" | tee -a "$STATUS" "$MAIN_LOG"
  exit 1
fi

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

echo "postprocess_done $(date --iso-8601=seconds)" | tee -a "$STATUS" "$MAIN_LOG"
echo "V211_LOCKED_CROSS_EXPERT_ALPHA_GRID_OK" | tee -a "$STATUS" "$MAIN_LOG"
echo "state=COMPLETED_GATE_PASS" | tee -a "$STATUS" "$MAIN_LOG"
