#!/usr/bin/env bash
set -euo pipefail

BASE=/sda/home/wangyuxin/ConvIR-B
WORK=$BASE/repos/ConvIR-B-haze4k-v5-v2i-fam2-noop-arch-equivalence
EVID=$WORK/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2i_fam2_noop_arch_equivalence_20260710
PY=$BASE/envs/convir-cu121/bin/python
DATA=$BASE/datasets/Haze4K/Haze4K
A0=$BASE/checkpoints/official/Haze4K/haze4k-base.pkl
SPLIT=$WORK/experience_docx/experiment_logs/haze4k_v5_chd_rm_v1_data_baseline_lock_20260708/haze4k_internal_split_2400_600.json
STATUS=$EVID/status.txt
LOG=$EVID/v2i_fam2_noop_equivalence.log

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

mkdir -p "$EVID"
{
  echo "v2i_start $(date --iso-8601=seconds)"
  echo "work=$WORK"
  echo "python=$PY"
  echo "data=$DATA"
  echo "split=$SPLIT"
  echo "checkpoint=$A0"
  echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
  echo "locked_haze4k_test_usage=none"
  echo "training=none RARM=not_connected_or_trained D7c_gate=not_connected"
} | tee -a "$STATUS"

cd "$WORK"
set +e
PYTHONUNBUFFERED=1 "$PY" experience_docx/tools/run_chd_rm_v2i_fam2_noop_equivalence.py \
  --data_dir "$DATA" \
  --checkpoint "$A0" \
  --split_json "$SPLIT" \
  --source_split train \
  --real_batch_split val_inner \
  --full_val_split val_inner \
  --output_dir "$EVID" \
  --seed 3407 \
  --real_batch_size 2 \
  --num_worker 0 \
  --expected_val_samples 600 \
  --max_abs_threshold 1e-7 \
  --metric_delta_threshold 1e-10 \
  > "$LOG" 2>&1
rc=$?
set -e

echo "v2i_done rc=$rc $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [[ "$rc" -eq 0 ]]; then
  echo "V2I_FAM2_NOOP_EQUIVALENCE_OK" | tee -a "$STATUS"
else
  echo "V2I_FAM2_NOOP_EQUIVALENCE_FAILED" | tee -a "$STATUS"
fi
exit "$rc"
