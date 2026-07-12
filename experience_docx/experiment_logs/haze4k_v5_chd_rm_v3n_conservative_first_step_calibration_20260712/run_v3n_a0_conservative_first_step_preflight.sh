#!/usr/bin/env bash
set -euo pipefail
cd /sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v3n-conservative-first-step-calibration-20260712/experience_docx/experiment_logs/haze4k_v5_chd_rm_v3n_conservative_first_step_calibration_20260712
echo v3n_a0_start 2026-07-12T0900+08:00 code_sha=dd73be9084014f746d289f29c981159a380a97f6 | tee -a status.txt
/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python /sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v3n-conservative-first-step-calibration-20260712/experience_docx/tools/chd_rm_v3n_conservative_first_step_preflight.py --block_rows /sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v3m-blockwise-counterfactual-advantage-20260711/experience_docx/experiment_logs/haze4k_v5_chd_rm_v3m_blockwise_counterfactual_advantage_20260711/v3m_a1_block_rows_cloud_only.csv --evid /sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v3n-conservative-first-step-calibration-20260712/experience_docx/experiment_logs/haze4k_v5_chd_rm_v3n_conservative_first_step_calibration_20260712 2>&1 | tee v3n_a0_conservative_first_step_preflight_20260712T0900.log
echo v3n_a0_done 2026-07-12T0900+08:00 | tee -a status.txt
echo V3N_A0_COMMAND_OK
