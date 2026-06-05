#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-/root/miniconda3/envs/convir-cu128/bin/python}
WORK=${WORK:-/root/autodl-tmp/workspace/ConvIR-B-v1-6-rcswitch-runtime}
CONVIR_ITS=${CONVIR_ITS:-$WORK/Dehazing/ITS}
UDP_REPO=${UDP_REPO:-/root/autodl-tmp/workspace/UDPNet}
DATA=${DATA:-/root/autodl-tmp/workspace/Dehaze-Net/dataset/HAZE4K}
DEPTH=${DEPTH:-/root/autodl-tmp/workspace/Dehaze-Net/experiment/HAZE4K/depth_cache/depth_anything_v2_small_hf}
A0=${A0:-/root/autodl-tmp/workspace/ConvIR-B/Dehazing/pretrained_models/haze4k-base.pkl}
OFFICIAL_CKPT=${OFFICIAL_CKPT:-/root/autodl-tmp/workspace/UDPNet_official_download/ConvIR_UDPNet_haze4k.ckpt}
SPLIT_JSON=${SPLIT_JSON:-$WORK/experience_docx/experiment_logs/haze4k_dpga_v13_hsdf_20260604/internal_val/haze4k_dpga_v13_regular_hard_seed3407.json}

EVID=$WORK/experience_docx/experiment_logs/haze4k_rc_expert_switch_v16_20260605
STATUS=$EVID/status.txt
OFFLINE_OUT=$EVID/offline_intermediate_analysis
FEATURE_OUT=$EVID/udp_switch_features
OFFLINE_LOG=$EVID/v16_rcswitch_offline_intermediate_analysis.log
FEATURE_LOG=$EVID/v16_rcswitch_udp_switch_feature_extraction.log
POST_FEATURE_LOG=$EVID/v16_rcswitch_post_feature_router_analysis.log

mkdir -p "$EVID" "$OFFLINE_OUT" "$FEATURE_OUT"
{
  echo "v16_rcswitch_start $(date --iso-8601=seconds)"
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
  echo "splits=val_regular,val_hard"
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

set +e
PYTHONUNBUFFERED=1 "$PY" experience_docx/tools/analyze_haze4k_rc_expert_switch.py \
  --repo_root "$WORK" \
  --output_dir "$OFFLINE_OUT" \
  2>&1 | tee "$OFFLINE_LOG"
offline_rc=${PIPESTATUS[0]}
set -e
echo "v16_offline_intermediate_analysis_done rc=$offline_rc output=$OFFLINE_OUT $(date --iso-8601=seconds)" | tee -a "$STATUS"

set +e
PYTHONUNBUFFERED=1 "$PY" experience_docx/tools/extract_haze4k_udp_switch_features.py \
  --convir_its_dir "$CONVIR_ITS" \
  --udp_repo "$UDP_REPO" \
  --data_dir "$DATA" \
  --depth_cache_dir "$DEPTH" \
  --a0_checkpoint "$A0" \
  --official_checkpoint "$OFFICIAL_CKPT" \
  --split_json "$SPLIT_JSON" \
  --splits val_regular val_hard \
  --output_dir "$FEATURE_OUT" \
  2>&1 | tee "$FEATURE_LOG"
feature_rc=${PIPESTATUS[0]}
set -e
echo "v16_udp_switch_feature_extraction_done rc=$feature_rc output=$FEATURE_OUT $(date --iso-8601=seconds)" | tee -a "$STATUS"

post_feature_rc=0
if [ "$feature_rc" -eq 0 ]; then
  set +e
  PYTHONUNBUFFERED=1 "$PY" experience_docx/tools/analyze_haze4k_rc_expert_switch.py \
    --repo_root "$WORK" \
    --output_dir "$OFFLINE_OUT" \
    --udp_feature_csv "$FEATURE_OUT/udp_switch_feature_table.csv" \
    2>&1 | tee "$POST_FEATURE_LOG"
  post_feature_rc=${PIPESTATUS[0]}
  set -e
  echo "v16_post_feature_router_analysis_done rc=$post_feature_rc output=$OFFLINE_OUT $(date --iso-8601=seconds)" | tee -a "$STATUS"
else
  echo "v16_post_feature_router_analysis_skipped feature_rc=$feature_rc $(date --iso-8601=seconds)" | tee -a "$STATUS"
fi

if [ "$offline_rc" -eq 0 ] && [ "$feature_rc" -eq 0 ] && [ "$post_feature_rc" -eq 0 ]; then
  echo "state=COMPLETED_GATE_PENDING_DOC_SYNC $(date --iso-8601=seconds)" | tee -a "$STATUS"
  echo "V16_RCSWITCH_INTERMEDIATE_ANALYSIS_OK"
  exit 0
fi

echo "state=FAILED_COMMAND offline_rc=$offline_rc feature_rc=$feature_rc post_feature_rc=$post_feature_rc $(date --iso-8601=seconds)" | tee -a "$STATUS"
echo "V16_RCSWITCH_INTERMEDIATE_ANALYSIS_FAILED offline_rc=$offline_rc feature_rc=$feature_rc post_feature_rc=$post_feature_rc"
exit 1
