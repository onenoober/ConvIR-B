#!/usr/bin/env bash
set -euo pipefail
PY=/root/miniconda3/envs/convir-cu128/bin/python
DP=/root/autodl-tmp/workspace/ConvIR-B-dpga-lite-826caaf
DATA=/root/autodl-tmp/workspace/Dehaze-Net/dataset/HAZE4K
DEPTH=/root/autodl-tmp/workspace/Dehaze-Net/experiment/HAZE4K/depth_cache/depth_anything_v2_small_hf
SN=/root/.cache/huggingface/hub/models--depth-anything--Depth-Anything-V2-Small-hf/snapshots/5426e4f0f36572d16453bbda7a8389317b1bef99
LOG=$DP/experience_docx/experiment_logs/haze4k_dpga_lite_20260604/cache_test_depth.log
{
  echo "cache_test_depth_start $(date --iso-8601=seconds)"
  "$PY" "$DP/experience_docx/tools/cache_depth_anything_v2_haze4k.py" \
    --data_dir "$DATA" \
    --output_dir "$DEPTH" \
    --model_path "$SN" \
    --splits test \
    --device cuda \
    --local_files_only \
    --progress_freq 50
  echo "cache_test_depth_done $(date --iso-8601=seconds)"
} 2>&1 | tee -a "$LOG"
