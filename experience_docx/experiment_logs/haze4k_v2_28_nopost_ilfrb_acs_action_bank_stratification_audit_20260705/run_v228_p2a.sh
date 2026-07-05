#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v2-28-nopost-ilfrb-acs-action-bank-stratification-audit
EVID=$REMOTE_ROOT/experience_docx/experiment_logs/haze4k_v2_28_nopost_ilfrb_acs_action_bank_stratification_audit_20260705
PY=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
DATA=/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K
CHECKPOINT=/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl
SPLIT_CSV=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v2-17-nopost-lowband-alignment-tail-audit/experience_docx/experiment_logs/haze4k_v2_16_nopost_wavelet_lowband_decoder_20260703/v216_t1_per_image_band_deltas.csv
STATUS=$EVID/status.txt
LOG=$EVID/v228_p2a_diagnostics.log
GPU=${GPU:-0}
MAX_IMAGES=${MAX_IMAGES:-80}
ORACLE_STEPS=${ORACLE_STEPS:-10}
STAGE_SETS=${STAGE_SETS:-S6_early_mid_final,S5_bottleneck_mid,S4_final_decoder}

mkdir -p "$EVID"
cd "$REMOTE_ROOT"

{
  echo "v228_p2a_start $(date --iso-8601=seconds)"
  echo "branch=$(git branch --show-current)"
  echo "commit=$(git rev-parse --short HEAD)"
  echo "remote_root=$REMOTE_ROOT"
  echo "python=$PY"
  echo "data=$DATA"
  echo "checkpoint=$CHECKPOINT"
  echo "split_csv=$SPLIT_CSV"
  echo "gpu=$GPU"
  echo "max_images=$MAX_IMAGES"
  echo "oracle_steps=$ORACLE_STEPS"
  echo "stage_sets=$STAGE_SETS"
  echo "locked_test_touched=false"
  echo "training_launched=false"
} | tee -a "$STATUS"

set +e
CUDA_VISIBLE_DEVICES="$GPU" "$PY" \
  experience_docx/tools/nopost_v228_action_bank_stratification_audit.py \
  --data-dir "$DATA" \
  --checkpoint "$CHECKPOINT" \
  --split-csv "$SPLIT_CSV" \
  --out-dir "$EVID" \
  --max-images "$MAX_IMAGES" \
  --oracle-steps "$ORACLE_STEPS" \
  --stage-sets "$STAGE_SETS" \
  2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e

decision="UNKNOWN"
if [ -f "$EVID/v228_closeout.json" ]; then
  decision=$("$PY" - <<'PY'
import json
from pathlib import Path
p = Path("/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v2-28-nopost-ilfrb-acs-action-bank-stratification-audit/experience_docx/experiment_logs/haze4k_v2_28_nopost_ilfrb_acs_action_bank_stratification_audit_20260705/v228_closeout.json")
print(json.loads(p.read_text()).get("decision", "UNKNOWN"))
PY
)
fi

{
  echo "v228_p2a_done rc=$rc decision=$decision $(date --iso-8601=seconds)"
  if [ "$rc" -eq 0 ]; then
    echo "V228_P2A_COMMAND_OK"
  else
    echo "V228_P2A_COMMAND_FAILED"
  fi
} | tee -a "$STATUS"

exit "$rc"
