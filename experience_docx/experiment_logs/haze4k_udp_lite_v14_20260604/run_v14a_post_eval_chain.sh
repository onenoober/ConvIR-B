#!/usr/bin/env bash
set -euo pipefail

WORK=${WORK:-/root/autodl-tmp/workspace/ConvIR-B-v1-4-udp-lite}
LOG_DIR=$WORK/experience_docx/experiment_logs/haze4k_udp_lite_v14_20260604
MODEL_NAME=${MODEL_NAME:-ConvIR-Haze4K-v1.4A-UDP-Lite-DPFM123-adapter-only-seed3407-20260604}
FINAL=$WORK/Dehazing/ITS/results/$MODEL_NAME/Training-Results/Final.pkl
STATUS=$LOG_DIR/status.txt

cd "$WORK"
echo "v14a_post_wait_start $(date --iso-8601=seconds)" | tee -a "$STATUS"
while tmux has-session -t v14a_udp_lite_train 2>/dev/null; do
  sleep 60
done
if [[ ! -f "$FINAL" ]]; then
  echo "v14a_post_no_final final=$FINAL $(date --iso-8601=seconds)" | tee -a "$STATUS"
  exit 3
fi

echo "v14a_post_eval_start $(date --iso-8601=seconds)" | tee -a "$STATUS"
bash "$LOG_DIR/run_v14a_eval_regular_hard.sh"
echo "v14a_post_intermediate_start $(date --iso-8601=seconds)" | tee -a "$STATUS"
bash "$LOG_DIR/run_v14_intermediate_audits.sh"
echo "v14a_post_done $(date --iso-8601=seconds)" | tee -a "$STATUS"
