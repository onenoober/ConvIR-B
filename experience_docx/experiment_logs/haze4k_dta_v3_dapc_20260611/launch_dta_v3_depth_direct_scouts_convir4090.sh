#!/usr/bin/env bash
set -euo pipefail
STAGE=${1:-scout5full}
SEED=${2:-3407}
FOLD=${3:-0}
BASE=${BASE:-/sda/home/wangyuxin/ConvIR-B}
WORK=${WORK:-$BASE/repos/ConvIR-B-dta-v3-dapc-finetune}
EVID=$WORK/experience_docx/experiment_logs/haze4k_dta_v3_dapc_20260611
STATUS=$EVID/status.txt
mkdir -p "$EVID"
MODES=(${MODES:-invert normal zero shuffle})
GPUS=(${GPUS:-2 3 4 5})
if [[ "${#GPUS[@]}" -lt "${#MODES[@]}" ]]; then echo "Need GPUS >= MODES" >&2; exit 66; fi
{
  echo "depth_direct_scout_launch_start stage=$STAGE seed=$SEED fold=$FOLD work=$WORK $(date --iso-8601=seconds)"
  echo "modes=${MODES[*]}"
  echo "gpus=${GPUS[*]}"
} | tee -a "$STATUS"
for idx in "${!MODES[@]}"; do
  mode=${MODES[$idx]}
  gpu=${GPUS[$idx]}
  session=dta_v3_${STAGE}_depthDirect_${mode}_f${FOLD}
  run_id=${STAGE}_depthDirect_${mode}_seed${SEED}_f${FOLD}
  cmd_script=$EVID/launch_${run_id}.cmd.sh
  tmux_log=$EVID/tmux_${session}.out
  model_name=ConvIR-Haze4K-DTA-v3-DAPC-DepthDirect-${mode}-seed${SEED}-f${FOLD}-${STAGE}
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "depth_direct_scout_skip_active session=$session run_id=$run_id" | tee -a "$STATUS"
    continue
  fi
  if [[ "${FORCE:-0}" != "1" && -f "$WORK/Dehazing/ITS/results/$model_name/Training-Results/Final.pkl" ]]; then
    echo "depth_direct_scout_skip_existing session=$session run_id=$run_id model=$model_name" | tee -a "$STATUS"
    continue
  fi
  cat > "$cmd_script" <<CMD
#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=$gpu
export BASE="$BASE"
export WORK="$WORK"
bash "$EVID/run_dta_v3_depth_direct_scout_convir4090.sh" "$STAGE" "$mode" "$SEED" "$FOLD"
CMD
  chmod +x "$cmd_script"
  tmux new-session -d -s "$session" "bash '$cmd_script' 2>&1 | tee '$tmux_log'"
  echo "depth_direct_scout_launched session=$session gpu=$gpu run_id=$run_id cmd=$cmd_script tmux_log=$tmux_log $(date --iso-8601=seconds)" | tee -a "$STATUS"
done
echo "DTA_V3_DEPTH_DIRECT_SCOUTS_LAUNCHED stage=$STAGE" | tee -a "$STATUS"
