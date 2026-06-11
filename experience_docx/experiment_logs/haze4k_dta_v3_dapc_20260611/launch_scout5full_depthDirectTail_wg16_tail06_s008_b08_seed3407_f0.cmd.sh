#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=5
export BASE="/sda/home/wangyuxin/ConvIR-B"
export WORK="/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-v3-dapc-finetune-tailguard"
bash "/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-v3-dapc-finetune-tailguard/experience_docx/experiment_logs/haze4k_dta_v3_dapc_20260611/run_dta_v3_depth_direct_tailguard_scout_convir4090.sh" "scout5full" "wg16_tail06_s008_b08" "3407" "0"
