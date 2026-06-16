#!/usr/bin/env bash
set -euo pipefail
EVID="/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v22-c8-mini-expert-oracle/experience_docx/experiment_logs/haze4k_v2_2_c8_mini_expert_oracle_20260615"
PY="/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python"
WD="$EVID/wheels"
STATUS="$EVID/v22_c8_wdmamba_wheel_direct_status.txt"
LOG="$EVID/runtime_logs/v22_c8_wdmamba_wheel_direct.log"
mkdir -p "$WD"
echo "wdmamba_wheel_direct_start $(date -Is)" | tee "$STATUS"
{
  set -x
  causal_url='https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.0.post8/causal_conv1d-1.5.0.post8+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl'
  mamba_url='https://github.com/state-spaces/mamba/releases/download/v2.2.4/mamba_ssm-2.2.4+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl'
  causal_whl="$WD/causal_conv1d-1.5.0.post8+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
  mamba_whl="$WD/mamba_ssm-2.2.4+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
  for spec in "$causal_url|$causal_whl" "$mamba_url|$mamba_whl"; do
    url="${spec%%|*}"; out="${spec#*|}"
    if [ ! -s "$out" ]; then
      curl -L --fail --retry 8 --retry-all-errors --retry-delay 8 --connect-timeout 40 --max-time 900 -o "$out.part" "$url"
      mv "$out.part" "$out"
    fi
    ls -lh "$out"
  done
  "$PY" -m pip install --no-deps "$causal_whl" "$mamba_whl"
  "$PY" - <<'PY'
import importlib.util, torch
print('torch_version', torch.__version__)
for m in ['pytorch_wavelets','causal_conv1d','mamba_ssm']:
    print(m, bool(importlib.util.find_spec(m)))
PY
} 2>&1 | tee "$LOG"
echo "wdmamba_wheel_direct_done $(date -Is)" | tee -a "$STATUS"
echo WDMAMBA_WHEEL_DIRECT_OK
