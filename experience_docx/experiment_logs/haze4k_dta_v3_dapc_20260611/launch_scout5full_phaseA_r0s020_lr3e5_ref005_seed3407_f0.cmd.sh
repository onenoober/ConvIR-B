#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=4
export BASE="/sda/home/wangyuxin/ConvIR-B"
export WORK="/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-v3-dapc-finetune-r0variants"
export MAKE_CONTACTSHEETS="1"
bash "/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-v3-dapc-finetune-r0variants/experience_docx/experiment_logs/haze4k_dta_v3_dapc_20260611/run_dta_v3_phase_a_r0_variant_convir4090.sh" "scout5full" "r0s020_lr3e5_ref005" "3407" "0"
