#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/workspace/ConvIR-B-saferhfd-v2-stage-scale}"
PFD_ROOT="${PFD_ROOT:-/root/autodl-tmp/workspace/ConvIR-B-pfd-mainline}"
PY="${PY:-/root/miniconda3/envs/convir-cu128/bin/python}"
DATA_DIR="${DATA_DIR:-/root/autodl-tmp/workspace/Dehaze-Net/dataset/HAZE4K}"

A0="${A0:-/root/autodl-tmp/workspace/ConvIR-B/Dehazing/pretrained_models/haze4k-base.pkl}"
B1="${B1:-$PFD_ROOT/Dehazing/ITS/results/ConvIR-Haze4K-PFD-B1-rhfd-stop20-seed3407-20260602/Training-Results/Best.pkl}"
OUT="${OUT:-$ROOT/experience_docx/experiment_logs/haze4k_saferhfd_v2_stage_scale_20260602}"
STATUS="$OUT/status.txt"

mkdir -p "$OUT"
cd "$ROOT"

mark_status() {
  local status="$1"
  {
    echo "status=$status"
    echo "timestamp=$(date -Is)"
    echo "root=$ROOT"
    echo "pfd_root=$PFD_ROOT"
    echo "a0=$A0"
    echo "b1=$B1"
    echo "data_dir=$DATA_DIR"
    echo "pairs=${PAIRS[*]:-}"
  } > "$STATUS"
}

declare -a PAIRS=(
  "0.70 0.70"
  "0.70 0.00"
  "0.00 0.70"
  "0.70 0.50"
  "0.50 0.70"
  "1.00 0.70"
  "0.70 1.00"
  "1.00 0.50"
  "0.50 1.00"
  "0.80 0.60"
  "0.60 0.80"
)

trap 'mark_status failed' ERR
mark_status running

for PAIR in "${PAIRS[@]}"; do
  RHFD2="$(awk '{print $1}' <<< "$PAIR")"
  RHFD1="$(awk '{print $2}' <<< "$PAIR")"
  TAG="saferhfd_v2_rhfd2_${RHFD2}_rhfd1_${RHFD1}_vs_a0"
  CKPT="$ROOT/Dehazing/ITS/results/PFD-SafeRHFD-v2-rhfd2-${RHFD2}-rhfd1-${RHFD1}/Training-Results/Best.pkl"

  echo "=== $TAG start $(date -Is) ==="
  "$PY" experience_docx/tools/make_pfd_rhfd_stage_surgery_checkpoint.py \
    --a0_checkpoint "$A0" \
    --b1_checkpoint "$B1" \
    --rhfd2_scale "$RHFD2" \
    --rhfd1_scale "$RHFD1" \
    --output "$CKPT"

  "$PY" experience_docx/tools/eval_haze4k_checkpoint_compare.py \
    --data_dir "$DATA_DIR" \
    --original_checkpoint "$A0" \
    --original_arch convir \
    --original_mode original \
    --original_name a0 \
    --candidate_checkpoint "$CKPT" \
    --candidate_arch pfd \
    --candidate_mode original \
    --candidate_name "$TAG" \
    --candidate_pfd_rhfd 1 \
    --candidate_pfd_hscm 0 \
    --candidate_pfd_pffb 0 \
    --candidate_pfd_pffb_high 0 \
    --candidate_pfd_teacher 0 \
    --output_dir "$OUT" \
    --tag "$TAG"

  "$PY" experience_docx/tools/analyze_haze4k_delta_buckets.py \
    --csv "$OUT/scout_eval_per_image_${TAG}.csv" \
    --candidate_name "$TAG" \
    --output "$OUT/scout_eval_bucket_analysis_${TAG}.json"
  echo "=== $TAG end $(date -Is) ==="
done

"$PY" experience_docx/tools/summarize_haze4k_saferhfd_stage_scale.py \
  --output_dir "$OUT" \
  --output_json "$OUT/stage_scale_summary.json" \
  --output_csv "$OUT/stage_scale_summary.csv"

mark_status complete
