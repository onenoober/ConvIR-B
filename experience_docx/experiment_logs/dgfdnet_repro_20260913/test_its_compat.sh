#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT=/sda/home/wangyuxin/ConvIR-B/repos/dgfdnet-repro-20260913
EVID=/sda/home/wangyuxin/ConvIR-B/experience_docx/experiment_logs/dgfdnet_repro_20260913
PY=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
DATA=/sda/home/wangyuxin/ConvIR-B/datasets/RESIDE/official/ITS/
CHECKPOINT=${1:?checkpoint path required}
MODEL_NAME=${2:-dgfdnet_its_compat_center_crop_20260913}
STATUS=$EVID/status.txt
LOG=$EVID/test_its_compat_${MODEL_NAME}.log
CSV=$EVID/${MODEL_NAME}_per_image.csv

mkdir -p "$EVID"
printf '%s\n' "RUNNING_EVAL_COMPAT $(date --iso-8601=seconds)" | tee -a "$STATUS"
printf '%s\n' "CHECKPOINT=$CHECKPOINT" | tee -a "$STATUS"
printf '%s\n' "ALIGNMENT=center_crop_gt_to_model_output" | tee -a "$STATUS"
if [[ ! -f "$CHECKPOINT" ]]; then
  printf '%s\n' "CHECKPOINT_MISSING=$CHECKPOINT" | tee -a "$STATUS"
  printf '%s\n' "PREFLIGHT_FAILED_ENGINEERING $(date --iso-8601=seconds)" | tee -a "$STATUS"
  exit 2
fi

cd "$REMOTE_ROOT/Dehazing/ITS"
set +e
"$PY" "$EVID/eval_its_center_crop.py" \
  --data_dir "$DATA" \
  --test_model "$CHECKPOINT" \
  --csv_path "$CSV" \
  2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e
printf '%s\n' "run_done_compat rc=$rc $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [[ "$rc" -eq 0 ]]; then
  printf '%s\n' "DGFD_ITS_COMPAT_TEST_OK" | tee -a "$STATUS"
else
  printf '%s\n' "DGFD_ITS_COMPAT_TEST_FAILED rc=$rc" | tee -a "$STATUS"
fi
exit "$rc"
