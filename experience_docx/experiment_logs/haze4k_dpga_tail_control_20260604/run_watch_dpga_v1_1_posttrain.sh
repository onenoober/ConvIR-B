#!/usr/bin/env bash
set -euo pipefail

WORK=${WORK:-/root/autodl-tmp/workspace/ConvIR-B-dpga-tail-control}
LOG_DIR=$WORK/experience_docx/experiment_logs/haze4k_dpga_tail_control_20260604
DECISION_JSON=$LOG_DIR/v1_1_decision/dpga_v1_1_training_decision.json
WATCH_LOG=$LOG_DIR/watch_dpga_v1_1_posttrain.log
STATUS=$LOG_DIR/status.txt
SLEEP_SECONDS=${SLEEP_SECONDS:-120}
MAX_WAIT_SECONDS=${MAX_WAIT_SECONDS:-43200}

eval "$(/root/miniconda3/envs/convir-cu128/bin/python - "$DECISION_JSON" <<'PY'
import json
import shlex
import sys

decision = json.load(open(sys.argv[1], "r", encoding="utf-8"))
model_name = decision["training_args"]["model_name"]
print(f"MODEL_NAME={shlex.quote(model_name)}")
PY
)"

FINAL=$WORK/Dehazing/ITS/results/$MODEL_NAME/Training-Results/Final.pkl
start_ts=$(date +%s)

{
  echo "watch_dpga_v1_1_posttrain_start $(date --iso-8601=seconds)"
  echo "model_name=$MODEL_NAME"
  echo "final=$FINAL"
  while true; do
    now=$(date +%s)
    elapsed=$((now - start_ts))
    if [[ -s "$FINAL" ]]; then
      echo "v1_1_final_ready elapsed=${elapsed}s $(date --iso-8601=seconds)"
      break
    fi
    if ! pgrep -af "main.py --model_name $MODEL_NAME" >/dev/null 2>&1; then
      echo "v1_1_train_process_missing_before_final elapsed=${elapsed}s" >&2
      exit 5
    fi
    if [[ $elapsed -ge $MAX_WAIT_SECONDS ]]; then
      echo "v1_1_posttrain_wait_timeout elapsed=${elapsed}s" >&2
      exit 6
    fi
    echo "waiting_for_v1_1_final elapsed=${elapsed}s $(date --iso-8601=seconds)"
    sleep "$SLEEP_SECONDS"
  done
  bash "$LOG_DIR/run_eval_dpga_v1_1_val_inner.sh"
  echo "watch_dpga_v1_1_posttrain_done $(date --iso-8601=seconds)"
} 2>&1 | tee "$WATCH_LOG"

echo "watch_dpga_v1_1_posttrain_log=$WATCH_LOG" | tee -a "$STATUS"
