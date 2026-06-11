#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=4
export BASE="/sda/home/wangyuxin/ConvIR-B"
export WORK="/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-v3-dapc-finetune-taillite"
bash "/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-v3-dapc-finetune-taillite/experience_docx/experiment_logs/haze4k_dta_v3_dapc_20260611/run_dta_v3_depth_direct_tailguard_scout_convir4090.sh" "scout5full" "wg16_lite_s006_b12" "3407" "0"
