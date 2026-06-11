#!/usr/bin/env bash
set -euo pipefail
ROOT=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-v2-calibrated
EVID=$ROOT/experience_docx/experiment_logs/haze4k_dta_v2_calibrated_20260611
SCRIPT=$EVID/run_dta_v2_train_eval_convir4090.sh
SPLIT_JSON=$EVID/dta_v2_haze4k_oof_splits_seed3407.json
STATUS=$EVID/status.txt
# GPU0 is intentionally skipped because non-DTA user processes occupy it.
GPUS=(1 2 3 4 5 6 7)
SEEDS=(3411 3413)
FOLDS=(0 1 2 3 4)
MODES=(invert normal shuffle zero)
JOB_FILE=$EVID/dta_v2_oof20_adapter_only_multiseed_3411_3413_jobs.tsv
: > "$JOB_FILE"
idx=0
for seed in "${SEEDS[@]}"; do
  for fold in "${FOLDS[@]}"; do
    for mode in "${MODES[@]}"; do
      gpu=${GPUS[$((idx % ${#GPUS[@]}))]}
      printf '%s\t%s\t%s\t%s\n' "$seed" "$fold" "$mode" "$gpu" >> "$JOB_FILE"
      idx=$((idx + 1))
    done
  done
done
for gpu in "${GPUS[@]}"; do
  worker_script=$EVID/dta_v2_multiseed_worker_gpu${gpu}.sh
  cat > "$worker_script" <<WORKER
#!/usr/bin/env bash
set -euo pipefail
ROOT=$ROOT
EVID=$EVID
SCRIPT=$SCRIPT
SPLIT_JSON=$SPLIT_JSON
STATUS=$STATUS
GPU=$gpu
JOB_FILE=$JOB_FILE
while IFS=\$'\t' read -r seed fold mode job_gpu; do
  [[ "\$job_gpu" == "\$GPU" ]] || continue
  run_id="oof20_adapter_only_\${mode}_seed\${seed}_f\${fold}"
  echo "multiseed_job_start run_id=\$run_id gpu=\$GPU \\$(date --iso-8601=seconds)" | tee -a "\$STATUS"
  set +e
  CUDA_VISIBLE_DEVICES="\$GPU" FOLD_TAG="f\${fold}" SPLIT_JSON="\$SPLIT_JSON" TRAIN_SPLIT="fold\${fold}_train" EVAL_SPLIT="fold\${fold}_val" TRAIN_DEPTH_SPLIT=train EVAL_DEPTH_SPLIT=train bash "\$SCRIPT" oof20 adapter_only "\$mode" "\$seed"
  rc=\$?
  set -e
  echo "multiseed_job_done rc=\$rc run_id=\$run_id gpu=\$GPU \\$(date --iso-8601=seconds)" | tee -a "\$STATUS"
  if [[ "\$rc" -ne 0 ]]; then
    echo "DTA_V2_MULTI_SEED_JOB_FAILED run_id=\$run_id gpu=\$GPU" | tee -a "\$STATUS"
    exit "\$rc"
  fi
done < "\$JOB_FILE"
echo "DTA_V2_MULTI_SEED_WORKER_OK gpu=\$GPU \\$(date --iso-8601=seconds)" | tee -a "\$STATUS"
WORKER
  chmod +x "$worker_script"
  session="dta_v2_multiseed_3411_3413_gpu${gpu}"
  echo "multiseed_worker_launch session=$session gpu=$gpu $(date --iso-8601=seconds)" | tee -a "$STATUS"
  tmux new-session -d -s "$session" "bash '$worker_script'"
  sleep 1
  tmux has-session -t "$session"
  echo "multiseed_worker_launch_ok session=$session gpu=$gpu $(date --iso-8601=seconds)" | tee -a "$STATUS"
done
echo "DTA_V2_OOF20_ADAPTER_ONLY_MULTI_SEED_3411_3413_LAUNCH_OK jobs=$idx gpus=${GPUS[*]} $(date --iso-8601=seconds)" | tee -a "$STATUS"
