#!/usr/bin/env bash
set -euo pipefail

LOG=/sda/home/wangyuxin/ConvIR-B/runs/nhhaze_cdprm_s1_region_target_identifiability_20260720/cdprm-s1-region-target-identifiability-r1/runtime.log
test -f "$LOG"
/usr/bin/tail -n 160 "$LOG"
echo CDPRM_S1_R1_CONTRACT_INSPECTION_OK
