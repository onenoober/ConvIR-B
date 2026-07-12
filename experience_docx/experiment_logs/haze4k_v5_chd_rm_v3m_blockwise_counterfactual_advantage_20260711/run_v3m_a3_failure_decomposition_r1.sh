#!/usr/bin/env bash
set -euo pipefail
echo run_start_v3m_a3_failure_decomposition_r1
date --iso-8601=seconds
/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python /sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v3m-a3-failure-decomp-20260712/experience_docx/tools/chd_rm_v3m_a3_failure_decomposition.py --evid /sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v3m-blockwise-counterfactual-advantage-20260711/experience_docx/experiment_logs/haze4k_v5_chd_rm_v3m_blockwise_counterfactual_advantage_20260711 2>&1 | tee /sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v3m-blockwise-counterfactual-advantage-20260711/experience_docx/experiment_logs/haze4k_v5_chd_rm_v3m_blockwise_counterfactual_advantage_20260711/v3m_a3_failure_decomposition_r1_20260712T0825.log
echo run_done_v3m_a3_failure_decomposition_r1
date --iso-8601=seconds
echo V3M_A3_FAILURE_DECOMP_R1_COMMAND_OK
