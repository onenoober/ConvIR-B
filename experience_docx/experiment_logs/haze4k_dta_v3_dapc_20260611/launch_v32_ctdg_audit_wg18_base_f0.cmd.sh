#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=3
export BASE="/sda/home/wangyuxin/ConvIR-B"
export WORK="/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-v3-dapc-finetune-v32"
export SOURCE_RUN_ID="v31_wg18_base_s008_b14_seed3407_f0"
export RUN_ID="v32_ctdg_diag_wg18_base_s008_b14_seed3407_f0"
export CANDIDATE_NAME="wg18_base_s008_b14"
export CANDIDATE="/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-v3-dapc-finetune-taillite/Dehazing/ITS/results/ConvIR-Haze4K-DTA-v3-DAPC-DepthDirectTail-wg18_base_s008_b14-seed3407-f0-scout5full/Training-Results/Final.pkl"
bash "/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-v3-dapc-finetune-v32/experience_docx/experiment_logs/haze4k_dta_v3_dapc_20260611/run_dta_v3_2_ctdg_safemix_audits_convir4090.sh"
