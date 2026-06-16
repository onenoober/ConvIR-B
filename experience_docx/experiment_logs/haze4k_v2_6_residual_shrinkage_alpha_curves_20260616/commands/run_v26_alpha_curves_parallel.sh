#!/usr/bin/env bash
set -euo pipefail

ROOT="/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v26-residual-shrinkage-alpha-curves"
EVID="$ROOT/experience_docx/experiment_logs/haze4k_v2_6_residual_shrinkage_alpha_curves_20260616"
PY="/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python"
TOOL="$ROOT/experience_docx/tools/audit_haze4k_v26_residual_shrinkage_alpha_curves.py"
SUMMARY_TOOL="$ROOT/experience_docx/tools/summarize_haze4k_v26_alpha_curves.py"
STATUS="$EVID/status_v26_alpha_curves.txt"
LOG_DIR="$EVID/runtime_logs"
SPLIT_JSON="$ROOT/experience_docx/experiment_logs/haze4k_dpga_v13_hsdf_20260604/internal_val/haze4k_dpga_v13_regular_hard_seed3407.json"

mkdir -p "$EVID/commands" "$LOG_DIR"
export PYTHONPATH="$ROOT/experience_docx/tools:${PYTHONPATH:-}"

echo "v26_alpha_curves_start $(date -Is) locked=untouched" | tee "$STATUS"
cd "$ROOT"
{
  printf 'branch=%s\n' "$(git branch --show-current)"
  printf 'commit=%s\n' "$(git rev-parse HEAD)"
  printf 'status_short_begin\n'
  git status --short
  printf 'status_short_end\n'
} | tee -a "$STATUS"

"$PY" -m py_compile "$TOOL" "$SUMMARY_TOOL"
echo "cloud_py_compile_ok $(date -Is)" | tee -a "$STATUS"

mapfile -t GPUS < <(
  nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits |
  awk -F, '{gsub(/ /,"",$1); gsub(/ /,"",$2); gsub(/ /,"",$3); print $1","$2","$3}' |
  sort -t, -k2,2nr |
  head -n 3 |
  cut -d, -f1
)

if [ "${#GPUS[@]}" -lt 3 ]; then
  echo "FAILED_INFRA only_${#GPUS[@]}_gpus_detected_for_three_parallel_jobs" | tee -a "$STATUS"
  exit 2
fi

echo "selected_gpus=${GPUS[*]}" | tee -a "$STATUS"

run_one() {
  local expert="$1"
  local prefix="$2"
  local checkpoint="$3"
  local gpu="$4"
  local status_file="$EVID/status_${prefix}.txt"
  local log="$LOG_DIR/${prefix}.log"
  echo "${prefix}_start $(date -Is) gpu=${gpu} locked=untouched" | tee "$status_file"
  set +e
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$TOOL" \
    --expert "$expert" \
    --prefix "$prefix" \
    --out-dir "$EVID" \
    --convir-its-dir "$ROOT/Dehazing/ITS" \
    --udp-repo "/sda/home/wangyuxin/ConvIR-B/repos/UDPNet" \
    --data-dir "/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K" \
    --depth-cache-dir "/sda/home/wangyuxin/ConvIR-B/depth_cache/depth_anything_v2_small_hf" \
    --a0-checkpoint "/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl" \
    --fulludp-checkpoint "/sda/home/wangyuxin/ConvIR-B/checkpoints/udpnet/ConvIR_UDPNet_haze4k.ckpt" \
    --expert-checkpoint "$checkpoint" \
    --split-json "$SPLIT_JSON" \
    --splits val_regular val_hard \
    --alphas 0 0.125 0.25 0.375 0.50 0.75 1.0 \
    --print-freq 25 2>&1 | tee "$log"
  local rc=${PIPESTATUS[0]}
  set -e
  echo "${prefix}_done rc=${rc} $(date -Is)" | tee -a "$status_file"
  return "$rc"
}

run_one "wdmamba" "v26_wdmamba_alpha_curve" "/sda/home/wangyuxin/ConvIR-B/checkpoints/WDMamba_ckpts/haze4k_35.88.pth" "${GPUS[0]}" &
pid_wd=$!
run_one "fsudp" "v26_fsudp_alpha_curve" "/sda/home/wangyuxin/ConvIR-B/checkpoints/udpnet/FSNet_UDPNet_haze4k.ckpt" "${GPUS[1]}" &
pid_fs=$!
run_one "mbtaylor" "v26_mbtaylor_alpha_curve" "/sda/home/wangyuxin/ConvIR-B/checkpoints/mb-taylorformerv2/HAZE4K-L.pth" "${GPUS[2]}" &
pid_mb=$!

rc=0
wait "$pid_wd" || rc=1
wait "$pid_fs" || rc=1
wait "$pid_mb" || rc=1

if [ "$rc" -ne 0 ]; then
  echo "V26_ALPHA_CURVES_FAILED rc=${rc} $(date -Is)" | tee -a "$STATUS"
  exit "$rc"
fi

"$PY" "$SUMMARY_TOOL" --evidence-dir "$EVID" --prefix v26 2>&1 | tee "$LOG_DIR/v26_summary.log"

echo "V26_ALPHA_CURVES_OK $(date -Is)" | tee -a "$STATUS"
echo "V26_ALPHA_CURVES_OK"
