#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-/root/miniconda3/envs/convir-cu128/bin/python}
WORK=${WORK:-/root/autodl-tmp/workspace/ConvIR-B-v1-5-full-udpnet}
UDP_REPO=${UDP_REPO:-/root/autodl-tmp/workspace/UDPNet}
DATA=${DATA:-/root/autodl-tmp/workspace/Dehaze-Net/dataset/HAZE4K}
DEPTH=${DEPTH:-/root/autodl-tmp/workspace/Dehaze-Net/experiment/HAZE4K/depth_cache/depth_anything_v2_small_hf}
A0=${A0:-/root/autodl-tmp/workspace/ConvIR-B/Dehazing/pretrained_models/haze4k-base.pkl}
OFFICIAL_CKPT=${OFFICIAL_CKPT:-/root/autodl-tmp/workspace/UDPNet_checkpoints/ConvIR_UDPNet_haze4k.ckpt}
BAIDUPCS=${BAIDUPCS:-/root/autodl-tmp/workspace/tools/bin/BaiduPCS-Go}
LOG_DIR=$WORK/experience_docx/experiment_logs/haze4k_fulludp_v15_phase0_repro_20260605
OUT=$LOG_DIR/phase0_repro_audit
STATUS=$LOG_DIR/status.txt
LOG=$OUT/v15_phase0_repro_audit.log

mkdir -p "$OUT"
{
  echo "v15_phase0_repro_audit_start $(date --iso-8601=seconds)"
  echo "work=$WORK"
  echo "udp_repo=$UDP_REPO"
  echo "data=$DATA"
  echo "depth=$DEPTH"
  echo "a0=$A0"
  echo "official_ckpt=$OFFICIAL_CKPT"
} | tee -a "$STATUS"

cd "$WORK"
set +e
PYTHONUNBUFFERED=1 "$PY" experience_docx/tools/audit_udpnet_v15_phase0_repro.py \
  --udp_repo "$UDP_REPO" \
  --data_dir "$DATA" \
  --depth_cache_dir "$DEPTH" \
  --a0_checkpoint "$A0" \
  --official_checkpoint "$OFFICIAL_CKPT" \
  --baidupcs_bin "$BAIDUPCS" \
  --output_dir "$OUT" \
  2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e
echo "v15_phase0_repro_audit_done rc=$rc output=$OUT $(date --iso-8601=seconds)" | tee -a "$STATUS"
exit "$rc"
