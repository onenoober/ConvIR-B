#!/usr/bin/env bash
set -euo pipefail
ROOT=/sda/home/wangyuxin/ConvIR-B
WS=$ROOT/repos/ConvIR-B-haze4k-v5-v1-chd-rm-data-baseline-lock
PY=$ROOT/envs/convir-cu121/bin/python
DATA=$ROOT/datasets/Haze4K/Haze4K
CKPT=$ROOT/checkpoints/official/Haze4K/haze4k-base.pkl
EVID=$WS/experience_docx/experiment_logs/haze4k_v5_chd_rm_v1_data_baseline_lock_20260708
SPLIT_JSON=$EVID/haze4k_internal_split_2400_600.json
OUT=$EVID/a0_val600_eval
LOG=$EVID/v1_a0_val600_eval.log
STATUS=$EVID/status.txt
mkdir -p "$OUT"
{
  echo "v1_a0_val600_eval_start $(date --iso-8601=seconds)"
  echo "workspace=$WS"
  echo "data=$DATA"
  echo "checkpoint=$CKPT"
  echo "split_json=$SPLIT_JSON"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
  "$PY" "$EVID/eval_v1_a0_val600.py" \
    --data_dir "$DATA" \
    --checkpoint "$CKPT" \
    --split_json "$SPLIT_JSON" \
    --split_name val_inner \
    --output_dir "$OUT"
  cp "$OUT/a0_val600_global_metrics.csv" "$EVID/a0_val600_global_metrics.csv"
  cp "$OUT/a0_val600_per_image_metrics.csv" "$EVID/a0_val600_per_image_metrics.csv"
  cp "$OUT/metric_repro_audit.json" "$EVID/metric_repro_audit.json"
  cp "$OUT/a0_efficiency_metrics.json" "$EVID/a0_efficiency_metrics.json"
  echo "v1_a0_val600_eval_done $(date --iso-8601=seconds)"
  echo "CHDRM_V1_A0_VAL600_OK"
} 2>&1 | tee "$LOG"
echo "CHDRM_V1_A0_VAL600_OK $(date --iso-8601=seconds)" >> "$STATUS"
