#!/usr/bin/env bash
set -euo pipefail
EVID="/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v22-c8-mini-expert-oracle/experience_docx/experiment_logs/haze4k_v2_2_c8_mini_expert_oracle_20260615"
PY="/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python"
STATUS="$EVID/v22_c8_wdmamba_deps_nodeps_status.txt"
LOG="$EVID/runtime_logs/v22_c8_wdmamba_deps_nodeps.log"
export MAX_JOBS=4
echo "wdmamba_deps_nodeps_start $(date -Is)" | tee "$STATUS"
{
  "$PY" -m pip install --no-cache-dir --no-deps ninja pytorch_wavelets
  "$PY" -m pip install --no-cache-dir --no-build-isolation --no-deps causal-conv1d==1.5.0.post8 mamba-ssm==2.2.4
  "$PY" - <<'PY'
import importlib.util, torch
print('torch_version', torch.__version__)
for m in ['ninja','pytorch_wavelets','causal_conv1d','mamba_ssm']:
    print(m, bool(importlib.util.find_spec(m)))
PY
} 2>&1 | tee "$LOG"
echo "wdmamba_deps_nodeps_done $(date -Is)" | tee -a "$STATUS"
echo WDMAMBA_DEPS_NODEPS_OK
