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
VARIANTS=(${VARIANTS:-r0s005_lr3e5_ref005 r0s010_lr3e5_ref005 r0s020_lr3e5_ref005 r0s010_lr1e5_ref010})
GPUS=(${GPUS:-2 3 4 5})
if [[ "${#GPUS[@]}" -lt "${#VARIANTS[@]}" ]]; then
  echo "Need at least as many GPUS as VARIANTS: gpucount=${#GPUS[@]} variants=${#VARIANTS[@]}" >&2
  exit 66
fi
{
  echo "phase_a_variant_scout_launch_start stage=$STAGE seed=$SEED fold=$FOLD work=$WORK $(date --iso-8601=seconds)"
  echo "variants=${VARIANTS[*]}"
  echo "gpus=${GPUS[*]}"
} | tee -a "$STATUS"
for idx in "${!VARIANTS[@]}"; do
  variant=${VARIANTS[$idx]}
  gpu=${GPUS[$idx]}
  session=dta_v3_${STAGE}_${variant}_f${FOLD}
  run_id=${STAGE}_phaseA_${variant}_seed${SEED}_f${FOLD}
  cmd_script=$EVID/launch_${run_id}.cmd.sh
  tmux_log=$EVID/tmux_${session}.out
  model_name=ConvIR-Haze4K-DTA-v3-DAPC-PhaseA-${variant}-seed${SEED}-f${FOLD}-${STAGE}
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "phase_a_variant_scout_skip_active session=$session run_id=$run_id" | tee -a "$STATUS"
    continue
  fi
  if [[ "${FORCE:-0}" != "1" && -f "$WORK/Dehazing/ITS/results/$model_name/Training-Results/Final.pkl" ]]; then
    echo "phase_a_variant_scout_skip_existing session=$session run_id=$run_id model=$model_name" | tee -a "$STATUS"
    continue
  fi
  cat > "$cmd_script" <<CMD
#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=$gpu
export BASE="$BASE"
export WORK="$WORK"
export MAKE_CONTACTSHEETS="${MAKE_CONTACTSHEETS:-1}"
bash "$EVID/run_dta_v3_phase_a_r0_variant_convir4090.sh" "$STAGE" "$variant" "$SEED" "$FOLD"
CMD
  chmod +x "$cmd_script"
  tmux new-session -d -s "$session" "bash '$cmd_script' 2>&1 | tee '$tmux_log'"
  echo "phase_a_variant_scout_launched session=$session gpu=$gpu run_id=$run_id cmd=$cmd_script tmux_log=$tmux_log $(date --iso-8601=seconds)" | tee -a "$STATUS"
done
echo "DTA_V3_PHASE_A_VARIANT_SCOUTS_LAUNCHED stage=$STAGE" | tee -a "$STATUS"
