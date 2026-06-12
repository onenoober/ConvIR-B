#!/usr/bin/env bash
set -euo pipefail
cd '/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-v3-dapc-finetune-v33-routerfusion'
export CUDA_VISIBLE_DEVICES='1'
echo worker=1 gpu=1 command_index=1 start $(date --iso-8601=seconds) | tee -a '/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-v3-dapc-finetune-v33-routerfusion/experience_docx/experiment_logs/haze4k_dta_v3_dapc_20260611/status.txt'
VARIANT='d1_loss' STAGE='triage5full' SEED='3407' FOLD='1' RUN_DIAG='0' bash '/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-v3-dapc-finetune-v33-routerfusion/experience_docx/experiment_logs/haze4k_dta_v3_dapc_20260611/run_dta_v3_3_routerfusion_scout_convir4090.sh'
echo worker=1 gpu=1 command_index=1 done $(date --iso-8601=seconds) | tee -a '/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-v3-dapc-finetune-v33-routerfusion/experience_docx/experiment_logs/haze4k_dta_v3_dapc_20260611/status.txt'
echo worker=1 gpu=1 command_index=5 start $(date --iso-8601=seconds) | tee -a '/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-v3-dapc-finetune-v33-routerfusion/experience_docx/experiment_logs/haze4k_dta_v3_dapc_20260611/status.txt'
VARIANT='d2_lowphys' STAGE='triage5full' SEED='3407' FOLD='1' RUN_DIAG='0' bash '/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-v3-dapc-finetune-v33-routerfusion/experience_docx/experiment_logs/haze4k_dta_v3_dapc_20260611/run_dta_v3_3_routerfusion_scout_convir4090.sh'
echo worker=1 gpu=1 command_index=5 done $(date --iso-8601=seconds) | tee -a '/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-v3-dapc-finetune-v33-routerfusion/experience_docx/experiment_logs/haze4k_dta_v3_dapc_20260611/status.txt'
echo worker=1 gpu=1 command_index=9 start $(date --iso-8601=seconds) | tee -a '/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-v3-dapc-finetune-v33-routerfusion/experience_docx/experiment_logs/haze4k_dta_v3_dapc_20260611/status.txt'
VARIANT='d3_router' STAGE='triage5full' SEED='3407' FOLD='1' RUN_DIAG='0' bash '/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-v3-dapc-finetune-v33-routerfusion/experience_docx/experiment_logs/haze4k_dta_v3_dapc_20260611/run_dta_v3_3_routerfusion_scout_convir4090.sh'
echo worker=1 gpu=1 command_index=9 done $(date --iso-8601=seconds) | tee -a '/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-v3-dapc-finetune-v33-routerfusion/experience_docx/experiment_logs/haze4k_dta_v3_dapc_20260611/status.txt'
echo DTA_V3_3_ROUTERFUSION_WORKER_OK worker=1 gpu=1 $(date --iso-8601=seconds) | tee -a '/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-v3-dapc-finetune-v33-routerfusion/experience_docx/experiment_logs/haze4k_dta_v3_dapc_20260611/status.txt'
