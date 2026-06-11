#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=4
export BASE="/sda/home/wangyuxin/ConvIR-B"
export WORK="/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-v3-dapc-finetune-v31"
bash "/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-v3-dapc-finetune-v31/experience_docx/experiment_logs/haze4k_dta_v3_dapc_20260611/run_dta_v3_1_wg18_light_hinge_scout_convir4090.sh" "scout5full" "3407" "0"
