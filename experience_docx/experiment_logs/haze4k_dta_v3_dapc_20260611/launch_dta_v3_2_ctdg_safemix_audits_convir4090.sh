#!/usr/bin/env bash
set -euo pipefail

BASE=${BASE:-/sda/home/wangyuxin/ConvIR-B}
WORK=${WORK:-$BASE/repos/ConvIR-B-dta-v3-dapc-finetune-v32}
EVID=$WORK/experience_docx/experiment_logs/haze4k_dta_v3_dapc_20260611
STATUS=$EVID/status.txt
BASE_GPU=${BASE_GPU:-3}
HINGE_GPU=${HINGE_GPU:-4}
mkdir -p "$EVID"

{
  echo "dta_v3_2_launch_start work=$WORK $(date --iso-8601=seconds)"
  echo "base_gpu=$BASE_GPU hinge_gpu=$HINGE_GPU"
  echo "locked_test_touched=false"
} | tee -a "$STATUS"

base_session=dta_v32_ctdg_audit_wg18_base_f0
hinge_session=dta_v32_ctdg_audit_wg18_hinge_f0

if tmux has-session -t "$base_session" 2>/dev/null; then
  echo "dta_v3_2_launch_skip_active session=$base_session" | tee -a "$STATUS"
else
  base_cmd=$EVID/launch_v32_ctdg_audit_wg18_base_f0.cmd.sh
  cat > "$base_cmd" <<CMD
#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=$BASE_GPU
export BASE="$BASE"
export WORK="$WORK"
export SOURCE_RUN_ID="v31_wg18_base_s008_b14_seed3407_f0"
export RUN_ID="v32_ctdg_diag_wg18_base_s008_b14_seed3407_f0"
export CANDIDATE_NAME="wg18_base_s008_b14"
export CANDIDATE="$BASE/repos/ConvIR-B-dta-v3-dapc-finetune-taillite/Dehazing/ITS/results/ConvIR-Haze4K-DTA-v3-DAPC-DepthDirectTail-wg18_base_s008_b14-seed3407-f0-scout5full/Training-Results/Final.pkl"
bash "$EVID/run_dta_v3_2_ctdg_safemix_audits_convir4090.sh"
CMD
  chmod +x "$base_cmd"
  tmux new-session -d -s "$base_session" "bash '$base_cmd' 2>&1 | tee '$EVID/tmux_${base_session}.out'"
  echo "dta_v3_2_launch_base session=$base_session cmd=$base_cmd $(date --iso-8601=seconds)" | tee -a "$STATUS"
fi

if tmux has-session -t "$hinge_session" 2>/dev/null; then
  echo "dta_v3_2_launch_skip_active session=$hinge_session" | tee -a "$STATUS"
else
  hinge_cmd=$EVID/launch_v32_ctdg_audit_wg18_hinge_f0.cmd.sh
  cat > "$hinge_cmd" <<CMD
#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=$HINGE_GPU
export BASE="$BASE"
export WORK="$WORK"
export SOURCE_RUN_ID="v31_wg18_light_hinge_seed3407_f0_scout5full_post"
export RUN_ID="v32_ctdg_diag_wg18_light_hinge_seed3407_f0"
export CANDIDATE_NAME="wg18_light_hinge"
export CANDIDATE="$WORK/Dehazing/ITS/results/ConvIR-Haze4K-DTA-v3-1-WG18LightHinge-seed3407-f0-scout5full/Training-Results/Final.pkl"
bash "$EVID/run_dta_v3_2_ctdg_safemix_audits_convir4090.sh"
CMD
  chmod +x "$hinge_cmd"
  tmux new-session -d -s "$hinge_session" "bash '$hinge_cmd' 2>&1 | tee '$EVID/tmux_${hinge_session}.out'"
  echo "dta_v3_2_launch_hinge session=$hinge_session cmd=$hinge_cmd $(date --iso-8601=seconds)" | tee -a "$STATUS"
fi

echo "DTA_V3_2_CTDG_SAFEMIX_AUDIT_LAUNCH_OK" | tee -a "$STATUS"
