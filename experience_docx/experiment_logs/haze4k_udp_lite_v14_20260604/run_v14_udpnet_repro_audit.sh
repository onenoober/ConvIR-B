#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-/root/miniconda3/envs/convir-cu128/bin/python}
WORK=${WORK:-/root/autodl-tmp/workspace/ConvIR-B-v1-4-udp-lite}
UDP_REPO=${UDP_REPO:-/root/autodl-tmp/workspace/UDPNet}
LOG_DIR=$WORK/experience_docx/experiment_logs/haze4k_udp_lite_v14_20260604
OUT=$LOG_DIR/udpnet_repro_audit
STATUS=$LOG_DIR/status.txt

mkdir -p "$OUT"
cd "$WORK"
PYTHONUNBUFFERED=1 "$PY" experience_docx/tools/audit_udpnet_convir_repro.py \
  --repo_dir "$UDP_REPO" \
  --clone_if_missing \
  --output_dir "$OUT" \
  > "$OUT/udpnet_convir_repro_audit.log" 2>&1

cp "$OUT/v14_udpnet_repro_audit.md" "$LOG_DIR/v14_udpnet_repro_audit.md"
echo "v14_udpnet_repro_audit_done output=$OUT $(date --iso-8601=seconds)" | tee -a "$STATUS"
