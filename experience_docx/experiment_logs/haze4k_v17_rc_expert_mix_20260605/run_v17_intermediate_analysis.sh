#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-/root/miniconda3/envs/convir-cu128/bin/python}
WORK=${WORK:-/root/autodl-tmp/workspace/ConvIR-B-v1-7-rcmix-runtime}
CONVIR_ITS=${CONVIR_ITS:-$WORK/Dehazing/ITS}
UDP_REPO=${UDP_REPO:-/root/autodl-tmp/workspace/UDPNet}
DATA=${DATA:-/root/autodl-tmp/workspace/Dehaze-Net/dataset/HAZE4K}
DEPTH=${DEPTH:-/root/autodl-tmp/workspace/Dehaze-Net/experiment/HAZE4K/depth_cache/depth_anything_v2_small_hf}
A0=${A0:-/root/autodl-tmp/workspace/ConvIR-B/Dehazing/pretrained_models/haze4k-base.pkl}
OFFICIAL_CKPT=${OFFICIAL_CKPT:-/root/autodl-tmp/workspace/UDPNet_official_download/ConvIR_UDPNet_haze4k.ckpt}
SPLIT_JSON=${SPLIT_JSON:-$WORK/experience_docx/experiment_logs/haze4k_dpga_v13_hsdf_20260604/internal_val/haze4k_dpga_v13_regular_hard_seed3407.json}
MAX_IMAGES=${MAX_IMAGES:-0}

EVID=$WORK/experience_docx/experiment_logs/haze4k_v17_rc_expert_mix_20260605
STATUS=$EVID/status.txt
FEATURE_OUT=$EVID/v17_fulltrain_features
ANALYSIS_OUT=$EVID/v17_mix_analysis
FEATURE_LOG=$EVID/v17_fulltrain_feature_extraction.log
ANALYSIS_LOG=$EVID/v17_mix_analysis.log

mkdir -p "$EVID" "$FEATURE_OUT" "$ANALYSIS_OUT"
{
  echo "v17_rcmix_start $(date --iso-8601=seconds)"
  echo "state=RUNNING_AUDIT"
  echo "work=$WORK"
  echo "python=$PY"
  echo "convir_its=$CONVIR_ITS"
  echo "udp_repo=$UDP_REPO"
  echo "data=$DATA"
  echo "depth=$DEPTH"
  echo "a0=$A0"
  echo "official_ckpt=$OFFICIAL_CKPT"
  echo "split_json=$SPLIT_JSON"
  echo "splits=train_inner,val_regular,val_hard"
  echo "max_images=$MAX_IMAGES"
  echo "locked_test_touched=NO"
  if [ -d "$WORK/.git" ]; then
    git -C "$WORK" branch --show-current 2>/dev/null | sed 's/^/branch=/'
    git -C "$WORK" rev-parse --short HEAD 2>/dev/null | sed 's/^/commit=/'
    git -C "$WORK" status --short 2>/dev/null | sed 's/^/git_status=/'
  fi
  if [ -f "$OFFICIAL_CKPT" ]; then
    sha256sum "$OFFICIAL_CKPT" | sed 's/^/official_ckpt_sha256=/'
  else
    echo "official_ckpt_missing=$OFFICIAL_CKPT"
  fi
} | tee -a "$STATUS"

cd "$WORK"

feature_args=()
if [ "$MAX_IMAGES" -gt 0 ]; then
  feature_args+=(--max_images "$MAX_IMAGES")
  echo "v17_feature_preflight_mode max_images=$MAX_IMAGES $(date --iso-8601=seconds)" | tee -a "$STATUS"
fi

set +e
PYTHONUNBUFFERED=1 "$PY" experience_docx/tools/extract_haze4k_v17_fulltrain_a0_udp_features.py \
  --convir_its_dir "$CONVIR_ITS" \
  --udp_repo "$UDP_REPO" \
  --data_dir "$DATA" \
  --depth_cache_dir "$DEPTH" \
  --a0_checkpoint "$A0" \
  --official_checkpoint "$OFFICIAL_CKPT" \
  --split_json "$SPLIT_JSON" \
  --splits train_inner val_regular val_hard \
  --depth_split train \
  --output_dir "$FEATURE_OUT" \
  "${feature_args[@]}" \
  2>&1 | tee "$FEATURE_LOG"
feature_rc=${PIPESTATUS[0]}
set -e
echo "v17_fulltrain_feature_extraction_done rc=$feature_rc output=$FEATURE_OUT $(date --iso-8601=seconds)" | tee -a "$STATUS"

analysis_rc=0
if [ "$feature_rc" -eq 0 ] && [ "$MAX_IMAGES" -eq 0 ]; then
  set +e
  PYTHONUNBUFFERED=1 "$PY" experience_docx/tools/analyze_haze4k_v17_risk_controlled_mix.py \
    --feature_csv "$FEATURE_OUT/v17_fulltrain_a0_udp_feature_table.csv" \
    --output_dir "$ANALYSIS_OUT" \
    --train_splits train_inner \
    --holdout_splits val_regular val_hard \
    2>&1 | tee "$ANALYSIS_LOG"
  analysis_rc=${PIPESTATUS[0]}
  set -e
  echo "v17_mix_analysis_done rc=$analysis_rc output=$ANALYSIS_OUT $(date --iso-8601=seconds)" | tee -a "$STATUS"
else
  echo "v17_mix_analysis_skipped feature_rc=$feature_rc max_images=$MAX_IMAGES $(date --iso-8601=seconds)" | tee -a "$STATUS"
fi

if [ "$feature_rc" -eq 0 ] && [ "$analysis_rc" -eq 0 ]; then
  if [ "$MAX_IMAGES" -eq 0 ]; then
    echo "state=COMPLETED_GATE_PENDING_DOC_SYNC $(date --iso-8601=seconds)" | tee -a "$STATUS"
    echo "V17_RCMIX_INTERMEDIATE_ANALYSIS_OK"
  else
    echo "state=PREFLIGHT_COMPLETE $(date --iso-8601=seconds)" | tee -a "$STATUS"
    echo "V17_RCMIX_PREFLIGHT_OK"
  fi
  exit 0
fi

echo "state=FAILED_COMMAND feature_rc=$feature_rc analysis_rc=$analysis_rc $(date --iso-8601=seconds)" | tee -a "$STATUS"
echo "V17_RCMIX_INTERMEDIATE_ANALYSIS_FAILED feature_rc=$feature_rc analysis_rc=$analysis_rc"
exit 1
