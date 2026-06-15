#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT=${REMOTE_ROOT:-/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v21-segmix-multialpha-local}
EVID=${EVID:-$REMOTE_ROOT/experience_docx/experiment_logs/haze4k_v2_1_segmix_multialpha_local_20260615}
PY=${PY:-/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python}
DATA=${DATA:-/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K}
DEPTH_CACHE=${DEPTH_CACHE:-/sda/home/wangyuxin/ConvIR-B/depth_cache/depth_anything_v2_small_hf}
SPLIT_JSON=${SPLIT_JSON:-$REMOTE_ROOT/experience_docx/experiment_logs/haze4k_dpga_v13_hsdf_20260604/internal_val/haze4k_dpga_v13_regular_hard_seed3407.json}
A0_CKPT=${A0_CKPT:-/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl}
UDPNET_REPO=${UDPNET_REPO:-/sda/home/wangyuxin/ConvIR-B/repos/UDPNet}
UDPNET_CKPT=${UDPNET_CKPT:-/sda/home/wangyuxin/ConvIR-B/checkpoints/udpnet/ConvIR_UDPNet_haze4k.ckpt}
PATCH_ROWS=${PATCH_ROWS:-$EVID/v21_c7b_patch_feature_rows.csv}
IMAGE_ROWS=${IMAGE_ROWS:-$EVID/v21_c7b_image_rows.csv}
STATUS=$EVID/status_c7c.txt
LOG=$EVID/v21_c7c_risk_tighten.log

mkdir -p "$EVID"
echo "===== v21_c7c_risk_tighten_start $(date --iso-8601=seconds) =====" >> "$LOG"
{
  echo "v21_c7c_risk_tighten_start $(date --iso-8601=seconds)"
  echo "remote_root=$REMOTE_ROOT"
  echo "python=$PY"
  echo "patch_rows=$PATCH_ROWS"
  echo "image_rows=$IMAGE_ROWS"
  echo "locked_test_touched=false"
  if [ -f "$REMOTE_ROOT/.codex_source_branch" ]; then sed 's/^/source_branch=/' "$REMOTE_ROOT/.codex_source_branch"; fi
  if [ -f "$REMOTE_ROOT/.codex_source_commit" ]; then sed 's/^/source_commit=/' "$REMOTE_ROOT/.codex_source_commit"; fi
  if [ -f "$REMOTE_ROOT/.codex_source_copy_time" ]; then sed 's/^/source_copy_time=/' "$REMOTE_ROOT/.codex_source_copy_time"; fi
  test -x "$PY" && echo "python_exists=true"
  test -f "$PATCH_ROWS" && echo "patch_rows_exists=true"
  test -f "$IMAGE_ROWS" && echo "image_rows_exists=true"
  test -d "$DATA" && echo "data_exists=true"
  test -d "$DEPTH_CACHE" && echo "depth_cache_exists=true"
  test -f "$SPLIT_JSON" && echo "split_json_exists=true"
  test -f "$A0_CKPT" && sha256sum "$A0_CKPT" | sed 's/^/a0_sha256=/'
  test -f "$UDPNET_CKPT" && sha256sum "$UDPNET_CKPT" | sed 's/^/udpnet_sha256=/'
} | tee -a "$STATUS"

cd "$REMOTE_ROOT"
set +e
"$PY" experience_docx/tools/audit_haze4k_v21_c7c_local_alpha_risk_tighten.py \
  --patch_rows "$PATCH_ROWS" \
  --image_rows "$IMAGE_ROWS" \
  --convir_its_dir "$REMOTE_ROOT/Dehazing/ITS" \
  --udp_repo "$UDPNET_REPO" \
  --data_dir "$DATA" \
  --depth_cache_dir "$DEPTH_CACHE" \
  --a0_checkpoint "$A0_CKPT" \
  --official_checkpoint "$UDPNET_CKPT" \
  --split_json "$SPLIT_JSON" \
  --splits val_regular val_hard \
  --depth_split train \
  --pad_factor 32 \
  --patch_size 128 \
  --print_freq 50 \
  --top_k 900 \
  --low_pool_limit 80 \
  --high_pool_limit 120 \
  --out_dir "$EVID" \
  2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
set -e
echo "v21_c7c_risk_tighten_done rc=$rc $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [ "$rc" -eq 0 ]; then
  echo "V21_C7C_RISK_TIGHTEN_OK $(date --iso-8601=seconds)" | tee -a "$STATUS"
else
  echo "V21_C7C_RISK_TIGHTEN_FAILED rc=$rc $(date --iso-8601=seconds)" | tee -a "$STATUS"
fi
exit "$rc"
