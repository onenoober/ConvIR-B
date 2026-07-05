#!/usr/bin/env bash
set -euo pipefail
BASE=/sda/home/wangyuxin/ConvIR-B
WORK=$BASE/repos/ConvIR-B-haze4k-v2-32-nopost-bounded-internal-lowfreq-correction-field
EVID=$WORK/experience_docx/experiment_logs/haze4k_v2_32_nopost_bounded_internal_lowfreq_correction_field_20260705
PY=$BASE/envs/convir-cu121/bin/python
DATA=$BASE/datasets/Haze4K/Haze4K
A0=$BASE/checkpoints/official/Haze4K/haze4k-base.pkl
STATUS=$EVID/status.txt
LOG=$EVID/v232_p2_canary80_oof.log
export CUDA_VISIBLE_DEVICES=0 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

mkdir -p "$EVID"
echo "p2_canary80_oof_start $(date --iso-8601=seconds)" | tee -a "$STATUS"
cd "$WORK"
set +e
PYTHONUNBUFFERED=1 "$PY" experience_docx/tools/run_haze4k_v232_bilfcf_canary.py \
  --phase canary80_oof \
  --data_dir "$DATA" \
  --checkpoint "$A0" \
  --output_dir "$EVID" \
  --sample_count 80 \
  --epochs 3 \
  --batch_size 4 \
  --loss_variant C \
  > "$LOG" 2>&1
rc=$?
set -e
echo "p2_canary80_oof_done rc=$rc $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [[ "$rc" -eq 0 ]]; then
  echo "V232_P2_CANARY80_OOF_OK" | tee -a "$STATUS"
elif [[ "$rc" -eq 2 ]]; then
  echo "V232_P2_CANARY80_OOF_GATE_FAIL_NORMAL_PAUSE" | tee -a "$STATUS"
else
  echo "V232_P2_CANARY80_OOF_FAILED_ENGINEERING" | tee -a "$STATUS"
fi
exit "$rc"
