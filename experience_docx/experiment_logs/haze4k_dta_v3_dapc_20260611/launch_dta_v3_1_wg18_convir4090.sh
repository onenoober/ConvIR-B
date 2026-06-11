#!/usr/bin/env bash
set -euo pipefail

BASE=${BASE:-/sda/home/wangyuxin/ConvIR-B}
WORK=${WORK:-$BASE/repos/ConvIR-B-dta-v3-dapc-finetune-v31}
EVID=$WORK/experience_docx/experiment_logs/haze4k_dta_v3_dapc_20260611
STATUS=$EVID/status.txt
STAGE=${STAGE:-scout5full}
SEED=${SEED:-3407}
FOLD=${FOLD:-0}
AUDIT_GPU=${AUDIT_GPU:-3}
HINGE_GPU=${HINGE_GPU:-4}
mkdir -p "$EVID"

{
  echo "dta_v3_1_launch_start stage=$STAGE seed=$SEED fold=$FOLD work=$WORK $(date --iso-8601=seconds)"
  echo "audit_gpu=$AUDIT_GPU hinge_gpu=$HINGE_GPU"
} | tee -a "$STATUS"

audit_session=dta_v31_wg18_audit_f${FOLD}
hinge_session=dta_v31_wg18_hinge_${STAGE}_f${FOLD}

if tmux has-session -t "$audit_session" 2>/dev/null; then
  echo "dta_v3_1_launch_skip_active session=$audit_session" | tee -a "$STATUS"
else
  audit_cmd=$EVID/launch_v31_wg18_audit_f${FOLD}.cmd.sh
  cat > "$audit_cmd" <<CMD
#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=$AUDIT_GPU
export BASE="$BASE"
export WORK="$WORK"
bash "$EVID/run_dta_v3_1_wg18_riskselect_audit_convir4090.sh"
CMD
  chmod +x "$audit_cmd"
  tmux new-session -d -s "$audit_session" "bash '$audit_cmd' 2>&1 | tee '$EVID/tmux_${audit_session}.out'"
  echo "dta_v3_1_launch_audit session=$audit_session cmd=$audit_cmd $(date --iso-8601=seconds)" | tee -a "$STATUS"
fi

if tmux has-session -t "$hinge_session" 2>/dev/null; then
  echo "dta_v3_1_launch_skip_active session=$hinge_session" | tee -a "$STATUS"
else
  hinge_cmd=$EVID/launch_v31_wg18_hinge_${STAGE}_f${FOLD}.cmd.sh
  cat > "$hinge_cmd" <<CMD
#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=$HINGE_GPU
export BASE="$BASE"
export WORK="$WORK"
bash "$EVID/run_dta_v3_1_wg18_light_hinge_scout_convir4090.sh" "$STAGE" "$SEED" "$FOLD"
CMD
  chmod +x "$hinge_cmd"
  tmux new-session -d -s "$hinge_session" "bash '$hinge_cmd' 2>&1 | tee '$EVID/tmux_${hinge_session}.out'"
  echo "dta_v3_1_launch_hinge session=$hinge_session cmd=$hinge_cmd $(date --iso-8601=seconds)" | tee -a "$STATUS"
fi

echo "DTA_V3_1_WG18_LAUNCH_OK stage=$STAGE fold=$FOLD" | tee -a "$STATUS"
