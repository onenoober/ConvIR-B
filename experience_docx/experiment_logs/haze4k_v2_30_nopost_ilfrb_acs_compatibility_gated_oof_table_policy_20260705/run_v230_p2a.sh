#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT=${REMOTE_ROOT:-/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v2-30-nopost-ilfrb-acs-compatibility-gated-oof-table-policy}
EVID="$REMOTE_ROOT/experience_docx/experiment_logs/haze4k_v2_30_nopost_ilfrb_acs_compatibility_gated_oof_table_policy_20260705"
PY=${PY:-/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python}
DATA=${DATA:-/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K}
CHECKPOINT=${CHECKPOINT:-/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl}
SPLIT_CSV=${SPLIT_CSV:-/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v2-17-nopost-lowband-alignment-tail-audit/experience_docx/experiment_logs/haze4k_v2_16_nopost_wavelet_lowband_decoder_20260703/v216_t1_per_image_band_deltas.csv}
GPU=${GPU:-0}
MAX_IMAGES=${MAX_IMAGES:-80}
ORACLE_STEPS=${ORACLE_STEPS:-10}
STATUS="$EVID/status.txt"
LOG="$EVID/v230_p2a_diagnostics.log"

mkdir -p "$EVID"
{
  echo "v230_p2a_start $(date --iso-8601=seconds)"
  echo "branch=$(git -C "$REMOTE_ROOT" branch --show-current)"
  echo "commit=$(git -C "$REMOTE_ROOT" rev-parse --short HEAD)"
  echo "remote_root=$REMOTE_ROOT"
  echo "python=$PY"
  echo "data=$DATA"
  echo "checkpoint=$CHECKPOINT"
  echo "split_csv=$SPLIT_CSV"
  echo "gpu=$GPU"
  echo "max_images=$MAX_IMAGES"
  echo "oracle_steps=$ORACLE_STEPS"
  echo "locked_test_touched=false"
  echo "training_launched=false"
  echo "p2b_selector_probe_launched=false"
} | tee -a "$STATUS"

cd "$REMOTE_ROOT"
export CUDA_VISIBLE_DEVICES="$GPU"
set +e
"$PY" experience_docx/tools/nopost_v230_compatibility_gated_oof_table_policy.py \
  --data-root "$DATA" \
  --checkpoint "$CHECKPOINT" \
  --split-csv "$SPLIT_CSV" \
  --out-dir "$EVID" \
  --max-images "$MAX_IMAGES" \
  --oracle-steps "$ORACLE_STEPS" \
  --parent-commit 936e3e0 \
  2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e

decision="UNKNOWN"
if [ -f "$EVID/v230_p2a_closeout.json" ]; then
  decision=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("decision","UNKNOWN"))' "$EVID/v230_p2a_closeout.json")
fi
echo "v230_p2a_done rc=$rc decision=$decision $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [ "$rc" -eq 0 ]; then
  echo "V230_P2A_COMMAND_OK" | tee -a "$STATUS"
else
  echo "V230_P2A_COMMAND_FAILED" | tee -a "$STATUS"
fi
exit "$rc"
