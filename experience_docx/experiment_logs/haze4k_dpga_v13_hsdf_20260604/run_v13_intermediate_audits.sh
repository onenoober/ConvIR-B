#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-/root/miniconda3/envs/convir-cu128/bin/python}
WORK=${WORK:-/root/autodl-tmp/workspace/ConvIR-B-v1-3-hsdf}
DATA=${DATA:-/root/autodl-tmp/workspace/Dehaze-Net/dataset/HAZE4K}
A0=${A0:-/root/autodl-tmp/workspace/ConvIR-B/Dehazing/pretrained_models/haze4k-base.pkl}
LOG_DIR=$WORK/experience_docx/experiment_logs/haze4k_dpga_v13_hsdf_20260604
SPLIT_JSON=${SPLIT_JSON:-$LOG_DIR/intenal_val/haze4k_dpga_v13_regular_hard_seed3407.json}
OUT=$LOG_DIR/intermediates
STATUS=$LOG_DIR/status.txt

if [[ ! -f "$SPLIT_JSON" ]]; then
  echo "missing split json: $SPLIT_JSON" >&2
  exit 2
fi

mkdir -p "$OUT"
echo "v13_intermediate_audit_start $(date --iso-8601=seconds)" | tee -a "$STATUS"
cd "$WORK"
PYTHONUNBUFFERED=1 "$PY" experience_docx/tools/audit_haze4k_dpga_v13_intermediates.py \
  --its_dir Dehazing/ITS \
  --data_dir "$DATA" \
  --split_json "$SPLIT_JSON" \
  --a0_checkpoint "$A0" \
  --output_dir "$OUT" \
  > "$LOG_DIR/audit_v13_intermediates.log" 2>&1
echo "v13_intermediate_audit_done rc=$? output=$OUT $(date --iso-8601=seconds)" | tee -a "$STATUS"
