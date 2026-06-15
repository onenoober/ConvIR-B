#!/usr/bin/env bash
set -euo pipefail
ROOT="/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v22-c8-mini-expert-oracle"
EVID="/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v22-c8-mini-expert-oracle/experience_docx/experiment_logs/haze4k_v2_2_c8_mini_expert_oracle_20260615"
PY="/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python"
TOOL="/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v22-c8-mini-expert-oracle/experience_docx/tools/audit_haze4k_v22_c8_render_oracle.py"
STATUS="$EVID/v22_c8_3_mbtaylor_smoke_status.txt"
LOG="$EVID/runtime_logs/v22_c8_3_mbtaylor_smoke.log"
export CUDA_VISIBLE_DEVICES=3
export PYTHONPATH="$ROOT/experience_docx/tools:${PYTHONPATH:-}"
echo "v22_c8_3_mbtaylor_smoke_start $(date -Is) gpu=3" | tee "$STATUS"
"$PY" "$TOOL"   --expert "mbtaylor"   --prefix "v22_c8_3_mbtaylor_smoke"   --out-dir "$EVID"   --convir-its-dir "$ROOT/Dehazing/ITS"   --udp-repo "/sda/home/wangyuxin/ConvIR-B/repos/UDPNet"   --data-dir "/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K"   --depth-cache-dir "/sda/home/wangyuxin/ConvIR-B/depth_cache/depth_anything_v2_small_hf"   --a0-checkpoint "/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl"   --fulludp-checkpoint "/sda/home/wangyuxin/ConvIR-B/checkpoints/udpnet/ConvIR_UDPNet_haze4k.ckpt"   --expert-checkpoint "/sda/home/wangyuxin/ConvIR-B/checkpoints/mb-taylorformerv2/HAZE4K-L.pth"   --split-json "/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v22-c8-mini-expert-oracle/experience_docx/experiment_logs/haze4k_dpga_v13_hsdf_20260604/internal_val/haze4k_dpga_v13_regular_hard_seed3407.json"   --splits val_regular val_hard   --max-images 2   --print-freq 1 2>&1 | tee "$LOG"
echo "v22_c8_3_mbtaylor_smoke_done $(date -Is)" | tee -a "$STATUS"
echo "v22_c8_3_mbtaylor_smoke_OK"
