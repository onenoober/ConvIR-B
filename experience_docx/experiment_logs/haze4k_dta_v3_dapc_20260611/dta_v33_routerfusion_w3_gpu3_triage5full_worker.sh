#!/usr/bin/env bash
set -euo pipefail
cd '/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-v3-dapc-finetune-v33-routerfusion'
export CUDA_VISIBLE_DEVICES='3'
echo worker=3 gpu=3 command_index=3 start $(date --iso-8601=seconds) | tee -a '/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-v3-dapc-finetune-v33-routerfusion/experience_docx/experiment_logs/haze4k_dta_v3_dapc_20260611/status.txt'
VARIANT='d1_loss' STAGE='triage5full' SEED='3411' FOLD='1' RUN_DIAG='0' bash '/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-v3-dapc-finetune-v33-routerfusion/experience_docx/experiment_logs/haze4k_dta_v3_dapc_20260611/run_dta_v3_3_routerfusion_scout_convir4090.sh'
echo worker=3 gpu=3 command_index=3 done $(date --iso-8601=seconds) | tee -a '/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-v3-dapc-finetune-v33-routerfusion/experience_docx/experiment_logs/haze4k_dta_v3_dapc_20260611/status.txt'
echo worker=3 gpu=3 command_index=7 start $(date --iso-8601=seconds) | tee -a '/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-v3-dapc-finetune-v33-routerfusion/experience_docx/experiment_logs/haze4k_dta_v3_dapc_20260611/status.txt'
VARIANT='d2_lowphys' STAGE='triage5full' SEED='3411' FOLD='1' RUN_DIAG='0' bash '/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-v3-dapc-finetune-v33-routerfusion/experience_docx/experiment_logs/haze4k_dta_v3_dapc_20260611/run_dta_v3_3_routerfusion_scout_convir4090.sh'
echo worker=3 gpu=3 command_index=7 done $(date --iso-8601=seconds) | tee -a '/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-v3-dapc-finetune-v33-routerfusion/experience_docx/experiment_logs/haze4k_dta_v3_dapc_20260611/status.txt'
echo worker=3 gpu=3 command_index=11 start $(date --iso-8601=seconds) | tee -a '/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-v3-dapc-finetune-v33-routerfusion/experience_docx/experiment_logs/haze4k_dta_v3_dapc_20260611/status.txt'
VARIANT='d3_router' STAGE='triage5full' SEED='3411' FOLD='1' RUN_DIAG='0' bash '/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-v3-dapc-finetune-v33-routerfusion/experience_docx/experiment_logs/haze4k_dta_v3_dapc_20260611/run_dta_v3_3_routerfusion_scout_convir4090.sh'
echo worker=3 gpu=3 command_index=11 done $(date --iso-8601=seconds) | tee -a '/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-v3-dapc-finetune-v33-routerfusion/experience_docx/experiment_logs/haze4k_dta_v3_dapc_20260611/status.txt'
echo DTA_V3_3_ROUTERFUSION_WORKER_OK worker=3 gpu=3 $(date --iso-8601=seconds) | tee -a '/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-v3-dapc-finetune-v33-routerfusion/experience_docx/experiment_logs/haze4k_dta_v3_dapc_20260611/status.txt'
