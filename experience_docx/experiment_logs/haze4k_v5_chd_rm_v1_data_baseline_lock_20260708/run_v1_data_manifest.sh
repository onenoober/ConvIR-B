#!/usr/bin/env bash
set -euo pipefail
ROOT=/sda/home/wangyuxin/ConvIR-B
WS=$ROOT/repos/ConvIR-B-haze4k-v5-v1-chd-rm-data-baseline-lock
PY=$ROOT/envs/convir-cu121/bin/python
DATA=$ROOT/datasets/Haze4K/Haze4K
EVID=$WS/experience_docx/experiment_logs/haze4k_v5_chd_rm_v1_data_baseline_lock_20260708
STATUS=$EVID/status.txt
LOG=$EVID/v1_data_manifest.log
mkdir -p "$EVID"
{
  echo "v1_data_manifest_start $(date --iso-8601=seconds)"
  echo "workspace=$WS"
  echo "data=$DATA"
  "$PY" "$EVID/build_v1_data_manifest.py" --data_dir "$DATA" --output_dir "$EVID" --seed 3407
  echo "v1_data_manifest_done $(date --iso-8601=seconds)"
  echo "CHDRM_V1_DATA_MANIFEST_OK"
} 2>&1 | tee "$LOG"
echo "CHDRM_V1_DATA_MANIFEST_OK $(date --iso-8601=seconds)" >> "$STATUS"
