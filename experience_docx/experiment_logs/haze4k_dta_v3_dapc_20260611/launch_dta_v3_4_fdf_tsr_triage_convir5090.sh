#!/usr/bin/env bash
set -euo pipefail

BASE=${BASE:-/home/caozhiyang/ConvIR-B}
WORK=${WORK:-$BASE/repos/ConvIR-B-dta-v3-4-fdf-tsr-finetune}
EVID=$WORK/experience_docx/experiment_logs/haze4k_dta_v3_dapc_20260611
STATUS=$EVID/status.txt
RUN_SCRIPT=$EVID/run_dta_v3_4_fdf_tsr_convir5090.sh
TASKS=$EVID/dta_v3_4_fdf_tsr_triage_tasks.tsv
WORKER_COUNT=${WORKER_COUNT:-5}
GPUS=${GPUS:-0,1,2,3,4}
STAGE=${STAGE:-quick5full}
SEEDS=${SEEDS:-3407,3411}
FOLDS=${FOLDS:-0,1}
VARIANTS=${VARIANTS:-e1_feature_only,e2_tiny_residual,e3_tsr_full,e4_plus_film}
FORCE=${FORCE:-0}
mkdir -p "$EVID"

IFS=',' read -r -a gpu_array <<< "$GPUS"
IFS=',' read -r -a seed_array <<< "$SEEDS"
IFS=',' read -r -a fold_array <<< "$FOLDS"
IFS=',' read -r -a variant_array <<< "$VARIANTS"
: > "$TASKS"
for variant in "${variant_array[@]}"; do
  for seed in "${seed_array[@]}"; do
    for fold in "${fold_array[@]}"; do
      printf '%s\t%s\t%s\n' "$variant" "$seed" "$fold" >> "$TASKS"
    done
  done
done

echo "dta_v3_4_triage_launch_start tasks=$(wc -l < "$TASKS") workers=$WORKER_COUNT gpus=$GPUS stage=$STAGE $(date --iso-8601=seconds)" | tee -a "$STATUS"
cd "$WORK"
printf 'branch=%s head=%s\n' "$(git branch --show-current)" "$(git rev-parse --short HEAD)" | tee -a "$STATUS"

for worker in $(seq 0 $((WORKER_COUNT - 1))); do
  gpu=${gpu_array[$((worker % ${#gpu_array[@]}))]}
  worker_script=$EVID/dta_v3_4_fdf_tsr_triage_worker_${worker}.sh
  worker_log=$EVID/dta_v3_4_fdf_tsr_triage_worker_${worker}.log
  worker_pid=$EVID/dta_v3_4_fdf_tsr_triage_worker_${worker}.pid
  cat > "$worker_script" <<EOS
#!/usr/bin/env bash
set -euo pipefail
EVID="$EVID"
TASKS="$TASKS"
RUN_SCRIPT="$RUN_SCRIPT"
STATUS="$STATUS"
WORKER=$worker
WORKER_COUNT=$WORKER_COUNT
GPU=$gpu
STAGE="$STAGE"
FORCE="$FORCE"
idx=0
while IFS=\$'\t' read -r variant seed fold; do
  if (( idx % WORKER_COUNT == WORKER )); then
    echo "dta_v3_4_triage_task_start worker=\$WORKER gpu=\$GPU idx=\$idx variant=\$variant seed=\$seed fold=\$fold \$(date --iso-8601=seconds)" | tee -a "\$STATUS"
    CUDA_VISIBLE_DEVICES="\$GPU" VARIANT="\$variant" SEED="\$seed" FOLD="\$fold" STAGE="\$STAGE" USE_SPLIT=1 RUN_TRAIN_CONTROLS=1 RUN_TEST=0 FORCE="\$FORCE" bash "\$RUN_SCRIPT"
    echo "dta_v3_4_triage_task_done worker=\$WORKER gpu=\$GPU idx=\$idx variant=\$variant seed=\$seed fold=\$fold \$(date --iso-8601=seconds)" | tee -a "\$STATUS"
  fi
  idx=\$((idx + 1))
done < "\$TASKS"
echo "DTA_V3_4_TRIAGE_WORKER_OK worker=\$WORKER gpu=\$GPU \$(date --iso-8601=seconds)" | tee -a "\$STATUS"
EOS
  chmod +x "$worker_script"
  if [[ -f "$worker_pid" ]] && kill -0 "$(cat "$worker_pid")" 2>/dev/null; then
    echo "dta_v3_4_triage_worker_active worker=$worker pid=$(cat "$worker_pid")" | tee -a "$STATUS"
  else
    nohup bash "$worker_script" >> "$worker_log" 2>&1 &
    pid=$!
    echo "$pid" > "$worker_pid"
    echo "dta_v3_4_triage_worker_launched worker=$worker gpu=$gpu pid=$pid" | tee -a "$STATUS"
  fi
done

echo "DTA_V3_4_TRIAGE_LAUNCH_OK $(date --iso-8601=seconds)" | tee -a "$STATUS"
