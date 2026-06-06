#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-/root/miniconda3/envs/convir-cu128/bin/python}
WORK=${WORK:-/root/autodl-tmp/workspace/ConvIR-B-v1-8-execution-queue}
EVID=$WORK/experience_docx/experiment_logs/haze4k_v18_execution_queue_20260606
STATUS=$EVID/status.txt
V17_FEATURE_CSV=${V17_FEATURE_CSV:-$WORK/experience_docx/experiment_logs/haze4k_v17_rc_expert_mix_20260605/v17_fulltrain_features/v17_fulltrain_a0_udp_feature_table.csv}
DOMAIN_CSV=${DOMAIN_CSV:-$EVID/v18_domain_data_preflight/v18_domain_data_preflight_per_image.csv}
OUT=$EVID/v18_domain_adaptation_q5
LOG=$OUT/v18_domain_adaptation_q5.log

mkdir -p "$OUT"

log_status() {
  printf '%s %s\n' "$*" "$(date --iso-8601=seconds)" | tee -a "$STATUS"
}

log_status "step_start name=v18_domain_adaptation_q5 log=$LOG"
set +e
(
  cd "$WORK"
  PYTHONUNBUFFERED=1 "$PY" experience_docx/tools/analyze_haze4k_v18_domain_adaptation.py \
    --feature_csv "$V17_FEATURE_CSV" \
    --domain_csv "$DOMAIN_CSV" \
    --output_dir "$OUT" \
    --real_data_candidates \
      "$WORK/Dehazing/ITS/datasets/real_haze" \
      "$WORK/dataset/real_haze" \
      "$WORK/datasets/real_haze" \
      /root/autodl-tmp/workspace/Dehaze-Net/dataset/real_haze \
      /root/autodl-tmp/workspace/dataset/real_haze \
      /root/autodl-tmp/workspace/datasets/real_haze \
      /root/autodl-tmp/dataset/real_haze \
      /root/autodl-tmp/datasets/real_haze
) 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e
log_status "step_done name=v18_domain_adaptation_q5 rc=$rc log=$LOG"
if [ "$rc" -eq 0 ]; then
  printf 'V18_DOMAIN_ADAPTATION_Q5_OK\n'
else
  printf 'V18_DOMAIN_ADAPTATION_Q5_FAILED rc=%s\n' "$rc"
fi
exit "$rc"
