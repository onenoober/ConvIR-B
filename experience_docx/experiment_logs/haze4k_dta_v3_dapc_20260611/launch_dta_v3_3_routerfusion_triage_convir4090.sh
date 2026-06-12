#!/usr/bin/env bash
set -euo pipefail

STAGE=${STAGE:-triage5full}
BASE=${BASE:-/sda/home/wangyuxin/ConvIR-B}
WORK=${WORK:-$BASE/repos/ConvIR-B-dta-v3-dapc-finetune-v33-routerfusion}
EVID=$WORK/experience_docx/experiment_logs/haze4k_dta_v3_dapc_20260611
STATUS=$EVID/status.txt
SCRIPT=$EVID/run_dta_v3_3_routerfusion_scout_convir4090.sh
VARIANTS=(${VARIANTS:-d1_loss d2_lowphys d3_router})
SEEDS=(${SEEDS:-3407 3411})
FOLDS=(${FOLDS:-0 1})
GPUS=(${GPUS:-0 1 2 3})

mkdir -p "$EVID"
echo "dta_v3_3_routerfusion_launcher_start stage=$STAGE variants=${VARIANTS[*]} seeds=${SEEDS[*]} folds=${FOLDS[*]} gpus=${GPUS[*]} $(date --iso-8601=seconds)" | tee -a "$STATUS"
cd "$WORK"
{
  echo "work=$WORK"
  git branch --show-current
  git rev-parse --short HEAD
  git status --short
  echo "locked_test_touched=false"
} | tee -a "$STATUS"

if [[ ! -f "$SCRIPT" ]]; then
  echo "DTA_V3_3_ROUTERFUSION_MISSING_SCRIPT $SCRIPT" | tee -a "$STATUS"
  exit 3
fi

declare -a COMMANDS=()
for variant in "${VARIANTS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for fold in "${FOLDS[@]}"; do
      run_diag=0
      if [[ "$variant" == "d3_router" && "$seed" == "3407" && "$fold" == "0" ]]; then
        run_diag=1
      fi
      COMMANDS+=("VARIANT='$variant' STAGE='$STAGE' SEED='$seed' FOLD='$fold' RUN_DIAG='$run_diag' bash '$SCRIPT'")
    done
  done
done

for wi in "${!GPUS[@]}"; do
  gpu=${GPUS[$wi]}
  session=dta_v33_routerfusion_w${wi}_gpu${gpu}_${STAGE}
  log=$EVID/${session}_tmux.log
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "DTA_V3_3_ROUTERFUSION_SESSION_ACTIVE session=$session" | tee -a "$STATUS"
    continue
  fi
  worker_script=$EVID/${session}_worker.sh
  {
    echo '#!/usr/bin/env bash'
    echo 'set -euo pipefail'
    echo "cd '$WORK'"
    echo "export CUDA_VISIBLE_DEVICES='$gpu'"
    for ci in "${!COMMANDS[@]}"; do
      if (( ci % ${#GPUS[@]} == wi )); then
        echo "echo worker=$wi gpu=$gpu command_index=$ci start \$(date --iso-8601=seconds) | tee -a '$STATUS'"
        echo "${COMMANDS[$ci]}"
        echo "echo worker=$wi gpu=$gpu command_index=$ci done \$(date --iso-8601=seconds) | tee -a '$STATUS'"
      fi
    done
    echo "echo DTA_V3_3_ROUTERFUSION_WORKER_OK worker=$wi gpu=$gpu \$(date --iso-8601=seconds) | tee -a '$STATUS'"
  } > "$worker_script"
  chmod +x "$worker_script"
  echo "dta_v3_3_routerfusion_launch_worker worker=$wi gpu=$gpu session=$session script=$worker_script $(date --iso-8601=seconds)" | tee -a "$STATUS"
  tmux new-session -d -s "$session" "bash '$worker_script' 2>&1 | tee '$log'"
done

tmux ls | tee -a "$STATUS"
echo "DTA_V3_3_ROUTERFUSION_LAUNCHER_OK commands=${#COMMANDS[@]} workers=${#GPUS[@]} $(date --iso-8601=seconds)" | tee -a "$STATUS"
