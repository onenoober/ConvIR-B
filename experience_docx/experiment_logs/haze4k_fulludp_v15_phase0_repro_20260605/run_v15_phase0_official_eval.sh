#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-/root/miniconda3/envs/convir-cu128/bin/python}
WORK=${WORK:-/root/autodl-tmp/workspace/ConvIR-B-v1-5-fulludp-runtime}
CONVIR_ITS=${CONVIR_ITS:-$WORK/Dehazing/ITS}
UDP_REPO=${UDP_REPO:-/root/autodl-tmp/workspace/UDPNet}
DATA=${DATA:-/root/autodl-tmp/workspace/Dehaze-Net/dataset/HAZE4K}
DEPTH=${DEPTH:-/root/autodl-tmp/workspace/Dehaze-Net/experiment/HAZE4K/depth_cache/depth_anything_v2_small_hf}
A0=${A0:-/root/autodl-tmp/workspace/ConvIR-B/Dehazing/pretrained_models/haze4k-base.pkl}
OFFICIAL_CKPT=${OFFICIAL_CKPT:-/root/autodl-tmp/workspace/UDPNet_official_download/ConvIR_UDPNet_haze4k.ckpt}
SPLIT_JSON=${SPLIT_JSON:-$WORK/experience_docx/experiment_logs/haze4k_dpga_v13_hsdf_20260604/internal_val/haze4k_dpga_v13_regular_hard_seed3407.json}
LOG_DIR=$WORK/experience_docx/experiment_logs/haze4k_fulludp_v15_phase0_repro_20260605
OUT=$LOG_DIR/phase0_official_eval
STATUS=$LOG_DIR/status.txt
LOG=$OUT/v15_phase0_official_eval.log

mkdir -p "$OUT"
{
  echo "v15_phase0_official_eval_start $(date --iso-8601=seconds)"
  echo "work=$WORK"
  echo "convir_its=$CONVIR_ITS"
  echo "udp_repo=$UDP_REPO"
  echo "data=$DATA"
  echo "depth=$DEPTH"
  echo "a0=$A0"
  echo "official_ckpt=$OFFICIAL_CKPT"
  echo "split_json=$SPLIT_JSON"
  echo "splits=val_regular,val_hard"
  echo "locked_test_touched=NO"
  echo "python=$PY"
  git -C "$WORK" branch --show-current 2>/dev/null | sed 's/^/branch=/'
  git -C "$WORK" rev-parse --short HEAD 2>/dev/null | sed 's/^/commit=/'
  sha256sum "$OFFICIAL_CKPT" | sed 's/^/official_ckpt_sha256=/'
} | tee -a "$STATUS"

cd "$WORK"
set +e
PYTHONUNBUFFERED=1 "$PY" experience_docx/tools/eval_udpnet_v15_phase0_repro.py \
  --convir_its_dir "$CONVIR_ITS" \
  --udp_repo "$UDP_REPO" \
  --data_dir "$DATA" \
  --depth_cache_dir "$DEPTH" \
  --a0_checkpoint "$A0" \
  --official_checkpoint "$OFFICIAL_CKPT" \
  --split_json "$SPLIT_JSON" \
  --splits val_regular val_hard \
  --output_dir "$OUT" \
  2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e
echo "v15_phase0_official_eval_done rc=$rc output=$OUT $(date --iso-8601=seconds)" | tee -a "$STATUS"
exit "$rc"
