#!/usr/bin/env bash
set -euo pipefail

output_root=/sda/home/wangyuxin/ConvIR-B/runs/sots-ots-demand-specific-overshoot-v1/sots-ots-demand-specific-overshoot-v1-r1
for relative in contract/contract_result.json contract/demand_specificity_contract_details.json runtime.log status.txt; do
  target="${output_root}/${relative}"
  if [[ -f "${target}" ]]; then
    echo "FILE ${relative}"
    /usr/bin/head -c 16384 "${target}"
    echo
  else
    echo "MISSING ${relative}"
  fi
done
