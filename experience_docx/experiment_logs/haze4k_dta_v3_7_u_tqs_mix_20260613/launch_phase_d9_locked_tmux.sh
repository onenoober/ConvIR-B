#!/usr/bin/env bash
set -euo pipefail
cd /sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-v3-7-u-tqs-mix-d8-formal-5541ca9
export D9_RUNNER_SOURCE_COMMIT=8bf4030
export OUTER_GROUPS=0:3407,0:3411,1:3407,1:3411
export GPU_LIST=1,2,3,4
export FEATURE_MAX_SIDE=384
export MAX_IMAGES=0
export INCLUDE_RUN_SUBSTRING=d8formal
bash run_dta_v3_7_phase_d9_locked_confirm_convir4090.sh
