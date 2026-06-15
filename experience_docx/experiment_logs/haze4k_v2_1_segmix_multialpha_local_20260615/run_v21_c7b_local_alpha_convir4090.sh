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
STATUS=$EVID/status_c7b.txt
LOG=$EVID/v21_c7b_local_alpha.log

mkdir -p "$EVID"
echo "===== v21_c7b_local_alpha_start $(date --iso-8601=seconds) =====" >> "$LOG"
{
  echo "v21_c7b_local_alpha_start $(date --iso-8601=seconds)"
  echo "remote_root=$REMOTE_ROOT"
  echo "python=$PY"
  echo "data=$DATA"
  echo "depth_cache=$DEPTH_CACHE"
  echo "split_json=$SPLIT_JSON"
  echo "a0_ckpt=$A0_CKPT"
  echo "udpnet_repo=$UDPNET_REPO"
  echo "udpnet_ckpt=$UDPNET_CKPT"
  echo "patch_size=128"
  echo "locked_test_touched=false"
  if [ -f "$REMOTE_ROOT/.codex_source_branch" ]; then sed 's/^/source_branch=/' "$REMOTE_ROOT/.codex_source_branch"; fi
  if [ -f "$REMOTE_ROOT/.codex_source_commit" ]; then sed 's/^/source_commit=/' "$REMOTE_ROOT/.codex_source_commit"; fi
  if [ -f "$REMOTE_ROOT/.codex_source_copy_time" ]; then sed 's/^/source_copy_time=/' "$REMOTE_ROOT/.codex_source_copy_time"; fi
  if [ -d "$REMOTE_ROOT/.git" ]; then
    git -C "$REMOTE_ROOT" branch --show-current | sed 's/^/branch=/'
    git -C "$REMOTE_ROOT" rev-parse --short HEAD | sed 's/^/commit=/'
    git -C "$REMOTE_ROOT" status --short | sed -n '1,240p' | sed 's/^/git_status=/'
  fi
  test -x "$PY" && echo "python_exists=true"
  test -d "$DATA" && echo "data_exists=true"
  test -d "$DEPTH_CACHE" && echo "depth_cache_exists=true"
  test -f "$SPLIT_JSON" && echo "split_json_exists=true"
  test -f "$A0_CKPT" && sha256sum "$A0_CKPT" | sed 's/^/a0_sha256=/'
  test -f "$UDPNET_CKPT" && sha256sum "$UDPNET_CKPT" | sed 's/^/udpnet_sha256=/'
} | tee -a "$STATUS"

cd "$REMOTE_ROOT"
set +e
"$PY" experience_docx/tools/audit_haze4k_v21_c7b_local_alpha_prototype.py \
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
echo "v21_c7b_local_alpha_done rc=$rc $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [ "$rc" -eq 0 ]; then
  echo "V21_C7B_LOCAL_ALPHA_OK $(date --iso-8601=seconds)" | tee -a "$STATUS"
else
  echo "V21_C7B_LOCAL_ALPHA_FAILED rc=$rc $(date --iso-8601=seconds)" | tee -a "$STATUS"
fi
exit "$rc"
