#!/usr/bin/env bash
set -euo pipefail

ROOT="/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v27-nhhaze-transfer"
EVID="$ROOT/experience_docx/experiment_logs/haze4k_v2_7_nhhaze_transfer_20260616"
PY="/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python"
TOOL="$ROOT/experience_docx/tools/eval_nhhaze_v27_wdmamba_transfer.py"
STATUS="$EVID/status_v27_nhhaze_transfer.txt"
LOG_DIR="$EVID/runtime_logs"
DATA="/sda/home/wangyuxin/ConvIR-B/datasets/NH-HAZE"
A0="/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl"
WDMAMBA="/sda/home/wangyuxin/ConvIR-B/checkpoints/WDMamba_ckpts/haze4k_35.88.pth"

mkdir -p "$EVID/commands" "$LOG_DIR" "$EVID/shards"
export PYTHONPATH="$ROOT/experience_docx/tools:${PYTHONPATH:-}"

echo "v27_nhhaze_transfer_start $(date -Is) locked_haze4k=untouched nhhaze_tuning=false" | tee "$STATUS"
cd "$ROOT"
{
  printf 'branch=%s\n' "$(git branch --show-current)"
  printf 'commit=%s\n' "$(git rev-parse HEAD)"
  printf 'status_short_begin\n'
  git status --short
  printf 'status_short_end\n'
  printf 'dataset=%s\n' "$DATA"
  printf 'a0=%s\n' "$A0"
  printf 'wdmamba=%s\n' "$WDMAMBA"
} | tee -a "$STATUS"

"$PY" -m py_compile "$TOOL"
echo "cloud_py_compile_ok $(date -Is)" | tee -a "$STATUS"

PAIR_COUNT=$(find "$DATA" -maxdepth 1 -type f -name '*_hazy.png' | wc -l)
echo "nhhaze_pair_count=${PAIR_COUNT}" | tee -a "$STATUS"
if [ "$PAIR_COUNT" -ne 55 ]; then
  echo "FAILED_ENGINEERING unexpected_nhhaze_pair_count=${PAIR_COUNT}" | tee -a "$STATUS"
  exit 2
fi

mapfile -t GPUS < <(
  nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits |
  awk -F, '{gsub(/ /,"",$1); gsub(/ /,"",$2); gsub(/ /,"",$3); if ($2 >= 12000) print $1","$2","$3}' |
  sort -t, -k2,2nr |
  head -n 3 |
  cut -d, -f1
)

if [ "${#GPUS[@]}" -lt 1 ]; then
  echo "FAILED_INFRA no_gpu_with_12gb_free" | tee -a "$STATUS"
  exit 2
fi

SHARDS="${#GPUS[@]}"
echo "selected_gpus=${GPUS[*]} shard_count=${SHARDS}" | tee -a "$STATUS"

run_shard() {
  local shard="$1"
  local gpu="$2"
  local prefix="v27_nhhaze_wdmamba_transfer_shard${shard}"
  local status_file="$EVID/shards/status_${prefix}.txt"
  local log="$LOG_DIR/${prefix}.log"
  echo "${prefix}_start $(date -Is) gpu=${gpu}" | tee "$status_file"
  set +e
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$TOOL" \
    --mode evaluate \
    --data-dir "$DATA" \
    --out-dir "$EVID/shards" \
    --prefix "$prefix" \
    --convir-its-dir "$ROOT/Dehazing/ITS" \
    --a0-checkpoint "$A0" \
    --wdmamba-checkpoint "$WDMAMBA" \
    --wdmamba-repo "/sda/home/wangyuxin/ConvIR-B/repos/external_experts/WDMamba" \
    --alphas 0 0.125 0.25 0.375 0.50 0.75 1.0 \
    --shard-index "$shard" \
    --shard-count "$SHARDS" \
    --print-freq 2 2>&1 | tee "$log"
  local rc=${PIPESTATUS[0]}
  set -e
  echo "${prefix}_done rc=${rc} $(date -Is)" | tee -a "$status_file"
  return "$rc"
}

pids=()
for idx in "${!GPUS[@]}"; do
  run_shard "$idx" "${GPUS[$idx]}" &
  pids+=("$!")
done

rc=0
for pid in "${pids[@]}"; do
  wait "$pid" || rc=1
done

if [ "$rc" -ne 0 ]; then
  echo "V27_NHHAZE_TRANSFER_SHARDS_FAILED rc=${rc} $(date -Is)" | tee -a "$STATUS"
  exit "$rc"
fi

mapfile -t CSVS < <(find "$EVID/shards" -maxdepth 1 -type f -name 'v27_nhhaze_wdmamba_transfer_shard*_per_image.csv' | sort)
if [ "${#CSVS[@]}" -ne "$SHARDS" ]; then
  echo "FAILED_ENGINEERING expected_${SHARDS}_shard_csvs_got_${#CSVS[@]}" | tee -a "$STATUS"
  exit 3
fi

"$PY" "$TOOL" \
  --mode aggregate \
  --data-dir "$DATA" \
  --out-dir "$EVID" \
  --prefix "v27_nhhaze_wdmamba_transfer" \
  --alphas 0 0.125 0.25 0.375 0.50 0.75 1.0 \
  --input-csvs "${CSVS[@]}" 2>&1 | tee "$LOG_DIR/v27_nhhaze_aggregate.log"

echo "V27_NHHAZE_TRANSFER_OK $(date -Is)" | tee -a "$STATUS"
echo "V27_NHHAZE_TRANSFER_OK"
