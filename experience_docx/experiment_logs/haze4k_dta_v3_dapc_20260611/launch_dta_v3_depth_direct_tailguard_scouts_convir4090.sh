#!/usr/bin/env bash
set -euo pipefail
STAGE=${1:-scout5full}
SEED=${2:-3407}
FOLD=${3:-0}
BASE=${BASE:-/sda/home/wangyuxin/ConvIR-B}
WORK=${WORK:-$BASE/repos/ConvIR-B-dta-v3-dapc-finetune-depthdirect}
EVID=$WORK/experience_docx/experiment_logs/haze4k_dta_v3_dapc_20260611
STATUS=$EVID/status.txt
mkdir -p "$EVID"
VARIANTS=(${VARIANTS:-wg16_tail08_s005_b10 wg18_tail10_s006_b12 wg20_tail12_s006_b10 wg16_tail06_s008_b08})
GPUS=(${GPUS:-2 3 4 5})
if [[ "${#GPUS[@]}" -lt "${#VARIANTS[@]}" ]]; then echo "Need GPUS >= VARIANTS" >&2; exit 66; fi
{
  echo "depth_direct_tailguard_scout_launch_start stage=$STAGE seed=$SEED fold=$FOLD work=$WORK $(date --iso-8601=seconds)"
  echo "variants=${VARIANTS[*]}"
  echo "gpus=${GPUS[*]}"
} | tee -a "$STATUS"
for idx in "${!VARIANTS[@]}"; do
  variant=${VARIANTS[$idx]}
  gpu=${GPUS[$idx]}
  session=dta_v3_${STAGE}_depthDirectTail_${variant}_f${FOLD}
  run_id=${STAGE}_depthDirectTail_${variant}_seed${SEED}_f${FOLD}
  cmd_script=$EVID/launch_${run_id}.cmd.sh
  tmux_log=$EVID/tmux_${session}.out
  model_name=ConvIR-Haze4K-DTA-v3-DAPC-DepthDirectTail-${variant}-seed${SEED}-f${FOLD}-${STAGE}
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "depth_direct_tailguard_scout_skip_active session=$session run_id=$run_id" | tee -a "$STATUS"
    continue
  fi
  if [[ "${FORCE:-0}" != "1" && -f "$WORK/Dehazing/ITS/results/$model_name/Training-Results/Final.pkl" ]]; then
    echo "depth_direct_tailguard_scout_skip_existing session=$session run_id=$run_id model=$model_name" | tee -a "$STATUS"
    continue
  fi
  cat > "$cmd_script" <<CMD
#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=$gpu
export BASE="$BASE"
export WORK="$WORK"
bash "$EVID/run_dta_v3_depth_direct_tailguard_scout_convir4090.sh" "$STAGE" "$variant" "$SEED" "$FOLD"
CMD
  chmod +x "$cmd_script"
  tmux new-session -d -s "$session" "bash '$cmd_script' 2>&1 | tee '$tmux_log'"
  echo "depth_direct_tailguard_scout_launched session=$session gpu=$gpu run_id=$run_id cmd=$cmd_script tmux_log=$tmux_log $(date --iso-8601=seconds)" | tee -a "$STATUS"
done
echo "DTA_V3_DEPTH_DIRECT_TAILGUARD_SCOUTS_LAUNCHED stage=$STAGE" | tee -a "$STATUS"
